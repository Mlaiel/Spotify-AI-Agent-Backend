"""
Scaler intelligent tenant-aware pour l'autoscaling multi-tenant
Gestion de l'isolation des ressources et priorisation par tenant
Développé par l'équipe d'experts dirigée par Fahed Mlaiel
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from collections import defaultdict

from .config_manager import AutoscalingConfigManager, TenantTier, ScalingMode
from .hpa_controller import HorizontalPodAutoscaler, HPADecision
from .vpa_controller import VerticalPodAutoscaler, ResourceRecommendation
from .metrics_collector import MetricsCollector
from .scaling_policies import ScalingPolicyEngine, PolicyEvaluationResult, ActionType

logger = logging.getLogger(__name__)

@dataclass
class TenantScalingState:
    """État de scaling d'un tenant"""
    tenant_id: str
    total_replicas: int
    total_cpu_requests: float
    total_memory_requests: float  # en GB
    active_services: List[str]
    last_scaling_action: Optional[datetime]
    scaling_debt: float  # Dette de scaling (ressources demandées vs allouées)
    priority_score: float

@dataclass
class ScalingDecision:
    """Décision de scaling multi-tenant"""
    tenant_id: str
    service_name: str
    action_type: str  # scale_up, scale_down, maintain, reject
    current_replicas: int
    target_replicas: int
    reasoning: str
    priority: int
    estimated_cost_impact: float
    resource_requirements: Dict[str, float]

class TenantAwareScaler:
    """Scaler intelligent avec conscience des tenants"""
    
    def __init__(self, config_manager: AutoscalingConfigManager,
                 hpa_controller: HorizontalPodAutoscaler,
                 vpa_controller: VerticalPodAutoscaler,
                 metrics_collector: MetricsCollector,
                 policy_engine: ScalingPolicyEngine):
        self.config_manager = config_manager
        self.hpa_controller = hpa_controller
        self.vpa_controller = vpa_controller
        self.metrics_collector = metrics_collector
        self.policy_engine = policy_engine
        
        # État du système
        self.tenant_states: Dict[str, TenantScalingState] = {}
        self.global_resource_pool = GlobalResourcePool()
        self.scaling_queue = ScalingQueue()
        self.fairness_controller = FairnessController()
        
        # Configuration
        self.max_concurrent_scalings = 5
        self.global_cpu_limit = 1000.0  # Cores
        self.global_memory_limit = 2000.0  # GB
        self.fairness_window_minutes = 30
    
    async def evaluate_tenant_scaling_needs(self, tenant_id: str) -> List[ScalingDecision]:
        """Évalue les besoins de scaling pour un tenant spécifique"""
        
        # Mise à jour de l'état du tenant
        await self._update_tenant_state(tenant_id)
        
        tenant_config = self.config_manager.get_tenant_config(tenant_id)
        if not tenant_config:
            logger.warning(f"No configuration found for tenant {tenant_id}")
            return []
        
        # Récupération des services du tenant
        services = await self._get_tenant_services(tenant_id)
        scaling_decisions = []
        
        for service_name in services:
            try:
                # Collecte des métriques
                current_metrics = await self.metrics_collector.get_service_metrics(
                    tenant_id, service_name
                )
                
                # Évaluation HPA
                hpa_decision = await self.hpa_controller.evaluate_scaling_decision(
                    tenant_id, service_name
                )
                
                # Évaluation des politiques
                policy_results = await self.policy_engine.evaluate_policies(
                    tenant_id, service_name, current_metrics
                )
                
                # Création de la décision de scaling tenant-aware
                decision = await self._create_tenant_aware_decision(
                    tenant_id, service_name, hpa_decision, policy_results, current_metrics
                )
                
                if decision:
                    scaling_decisions.append(decision)
                    
            except Exception as e:
                logger.error(f"Error evaluating scaling for {tenant_id}:{service_name}: {e}")
        
        # Priorisation et filtrage des décisions
        prioritized_decisions = self._prioritize_scaling_decisions(tenant_id, scaling_decisions)
        
        return prioritized_decisions
    
    async def _update_tenant_state(self, tenant_id: str):
        """Met à jour l'état d'un tenant"""
        
        services = await self._get_tenant_services(tenant_id)
        total_replicas = 0
        total_cpu = 0.0
        total_memory = 0.0
        
        for service in services:
            try:
                # Récupération des ressources actuelles
                current_replicas = await self._get_service_replicas(service)
                service_resources = await self._get_service_resources(service)
                
                total_replicas += current_replicas
                total_cpu += service_resources.get('cpu', 0.0) * current_replicas
                total_memory += service_resources.get('memory_gb', 0.0) * current_replicas
                
            except Exception as e:
                logger.warning(f"Could not get resources for service {service}: {e}")
        
        # Calcul du score de priorité
        tenant_config = self.config_manager.get_tenant_config(tenant_id)
        priority_score = self._calculate_tenant_priority_score(tenant_config)
        
        # Calcul de la dette de scaling
        scaling_debt = self._calculate_scaling_debt(tenant_id)
        
        self.tenant_states[tenant_id] = TenantScalingState(
            tenant_id=tenant_id,
            total_replicas=total_replicas,
            total_cpu_requests=total_cpu,
            total_memory_requests=total_memory,
            active_services=services,
            last_scaling_action=self._get_last_scaling_action(tenant_id),
            scaling_debt=scaling_debt,
            priority_score=priority_score
        )
    
    def _calculate_tenant_priority_score(self, tenant_config) -> float:
        """Calcule le score de priorité d'un tenant"""
        if not tenant_config:
            return 1.0
        
        # Base score selon le tier
        tier_scores = {
            TenantTier.ENTERPRISE: 10.0,
            TenantTier.PREMIUM: 5.0,
            TenantTier.BASIC: 1.0,
            TenantTier.FREE: 0.5
        }
        
        base_score = tier_scores.get(tenant_config.tier, 1.0)
        
        # Bonus pour les SLA stricts
        sla_multiplier = 1.0
        if hasattr(tenant_config, 'sla_requirements'):
            availability = tenant_config.sla_requirements.get('availability', 99.0)
            if availability >= 99.99:
                sla_multiplier = 2.0
            elif availability >= 99.9:
                sla_multiplier = 1.5
        
        # Facteur de coût - plus le budget est élevé, plus la priorité augmente
        cost_factor = min(2.0, tenant_config.cost_budget / 1000.0)
        
        return base_score * sla_multiplier * cost_factor
    
    def _calculate_scaling_debt(self, tenant_id: str) -> float:
        """Calcule la dette de scaling d'un tenant"""
        # La dette représente les ressources demandées mais non allouées
        # Pour l'instant, implémentation simple
        return 0.0
    
    async def _create_tenant_aware_decision(self, tenant_id: str, service_name: str,
                                          hpa_decision: HPADecision,
                                          policy_results: List[PolicyEvaluationResult],
                                          metrics: Dict[str, float]) -> Optional[ScalingDecision]:
        """Crée une décision de scaling tenant-aware"""
        
        tenant_config = self.config_manager.get_tenant_config(tenant_id)
        tenant_state = self.tenant_states.get(tenant_id)
        
        if not tenant_config or not tenant_state:
            return None
        
        # Analyse des résultats de politiques
        policy_actions = []
        for result in policy_results:
            if result.triggered:
                policy_actions.extend(result.actions)
        
        # Détermination de l'action principale
        if hpa_decision.desired_replicas > hpa_decision.current_replicas:
            base_action = "scale_up"
            target_replicas = hpa_decision.desired_replicas
        elif hpa_decision.desired_replicas < hpa_decision.current_replicas:
            base_action = "scale_down"
            target_replicas = hpa_decision.desired_replicas
        else:
            base_action = "maintain"
            target_replicas = hpa_decision.current_replicas
        
        # Application des contraintes tenant
        final_action, final_target = self._apply_tenant_constraints(
            tenant_id, service_name, base_action, target_replicas, tenant_config
        )
        
        # Vérification des ressources globales
        if final_action == "scale_up":
            resource_check = await self._check_global_resource_availability(
                tenant_id, service_name, final_target - hpa_decision.current_replicas
            )
            if not resource_check.allowed:
                final_action = "reject"
                final_target = hpa_decision.current_replicas
        
        # Calcul de l'impact coût
        cost_impact = self._estimate_cost_impact(
            service_name, hpa_decision.current_replicas, final_target
        )
        
        # Calcul des ressources requises
        service_resources = await self._get_service_resources(service_name)
        resource_delta = (final_target - hpa_decision.current_replicas)
        resource_requirements = {
            'cpu_cores': service_resources.get('cpu', 0.0) * resource_delta,
            'memory_gb': service_resources.get('memory_gb', 0.0) * resource_delta,
            'storage_gb': service_resources.get('storage_gb', 0.0) * resource_delta
        }
        
        # Génération du raisonnement
        reasoning_parts = [hpa_decision.scaling_reason]
        if policy_actions:
            policy_reasons = [f"Policy: {action.action_type.value}" for action in policy_actions]
            reasoning_parts.extend(policy_reasons)
        
        if final_action != base_action:
            reasoning_parts.append(f"Modified by tenant constraints ({base_action} -> {final_action})")
        
        reasoning = "; ".join(reasoning_parts)
        
        return ScalingDecision(
            tenant_id=tenant_id,
            service_name=service_name,
            action_type=final_action,
            current_replicas=hpa_decision.current_replicas,
            target_replicas=final_target,
            reasoning=reasoning,
            priority=int(tenant_state.priority_score),
            estimated_cost_impact=cost_impact,
            resource_requirements=resource_requirements
        )
    
    def _apply_tenant_constraints(self, tenant_id: str, service_name: str,
                                action: str, target_replicas: int, tenant_config) -> Tuple[str, int]:
        """Applique les contraintes spécifiques au tenant"""
        
        resource_limits = tenant_config.resource_limits
        
        # Application des limites min/max replicas
        constrained_target = max(
            resource_limits.min_replicas,
            min(target_replicas, resource_limits.max_replicas)
        )
        
        # Vérification du budget
        if not self.config_manager.is_scaling_allowed(tenant_id, 0):  # TODO: coût réel
            if action == "scale_up":
                return "reject", target_replicas
        
        # Limitations selon le tier
        if tenant_config.tier == TenantTier.BASIC:
            # Scaling plus conservateur pour les tenants Basic
            if action == "scale_up" and constrained_target > target_replicas * 1.5:
                constrained_target = int(target_replicas * 1.5)
        
        elif tenant_config.tier == TenantTier.FREE:
            # Très restrictif pour les tenants gratuits
            if action == "scale_up" and constrained_target > 2:
                constrained_target = 2
        
        # Ajustement de l'action si nécessaire
        if constrained_target != target_replicas:
            if constrained_target > target_replicas:
                final_action = "scale_up"
            elif constrained_target < target_replicas:
                final_action = "scale_down"
            else:
                final_action = "maintain"
        else:
            final_action = action
        
        return final_action, constrained_target
    
    async def _check_global_resource_availability(self, tenant_id: str, service_name: str,
                                                replica_delta: int) -> 'ResourceCheckResult':
        """Vérifie la disponibilité des ressources globales"""
        
        if replica_delta <= 0:
            return ResourceCheckResult(allowed=True, reason="No additional resources needed")
        
        # Calcul des ressources requises
        service_resources = await self._get_service_resources(service_name)
        required_cpu = service_resources.get('cpu', 0.0) * replica_delta
        required_memory = service_resources.get('memory_gb', 0.0) * replica_delta
        
        # Vérification des limites globales
        current_total_cpu = sum(state.total_cpu_requests for state in self.tenant_states.values())
        current_total_memory = sum(state.total_memory_requests for state in self.tenant_states.values())
        
        if current_total_cpu + required_cpu > self.global_cpu_limit:
            return ResourceCheckResult(
                allowed=False,
                reason=f"Global CPU limit exceeded ({current_total_cpu + required_cpu:.1f} > {self.global_cpu_limit})"
            )
        
        if current_total_memory + required_memory > self.global_memory_limit:
            return ResourceCheckResult(
                allowed=False,
                reason=f"Global memory limit exceeded ({current_total_memory + required_memory:.1f} > {self.global_memory_limit})"
            )
        
        return ResourceCheckResult(allowed=True, reason="Resources available")
    
    def _prioritize_scaling_decisions(self, tenant_id: str, decisions: List[ScalingDecision]) -> List[ScalingDecision]:
        """Priorise les décisions de scaling"""
        
        tenant_state = self.tenant_states.get(tenant_id)
        if not tenant_state:
            return decisions
        
        # Tri par priorité décroissante, puis par impact coût croissant
        prioritized = sorted(decisions, key=lambda d: (-d.priority, d.estimated_cost_impact))
        
        # Application de la fairness
        fairness_adjusted = self.fairness_controller.adjust_priorities(tenant_id, prioritized)
        
        return fairness_adjusted
    
    async def execute_scaling_decisions(self, decisions: List[ScalingDecision]) -> Dict[str, Any]:
        """Exécute les décisions de scaling"""
        
        execution_results = {
            'successful': [],
            'failed': [],
            'skipped': [],
            'total_decisions': len(decisions)
        }
        
        # Limitation du nombre de scalings concurrents
        semaphore = asyncio.Semaphore(self.max_concurrent_scalings)
        
        async def execute_single_decision(decision):
            async with semaphore:
                try:
                    result = await self._execute_single_scaling_decision(decision)
                    if result:
                        execution_results['successful'].append({
                            'tenant_id': decision.tenant_id,
                            'service': decision.service_name,
                            'action': decision.action_type,
                            'replicas': f"{decision.current_replicas} -> {decision.target_replicas}"
                        })
                    else:
                        execution_results['failed'].append({
                            'tenant_id': decision.tenant_id,
                            'service': decision.service_name,
                            'error': 'Execution failed'
                        })
                except Exception as e:
                    execution_results['failed'].append({
                        'tenant_id': decision.tenant_id,
                        'service': decision.service_name,
                        'error': str(e)
                    })
        
        # Exécution en parallèle avec limitation
        tasks = [execute_single_decision(decision) for decision in decisions if decision.action_type != 'reject']
        
        # Ajout des décisions rejetées
        for decision in decisions:
            if decision.action_type == 'reject':
                execution_results['skipped'].append({
                    'tenant_id': decision.tenant_id,
                    'service': decision.service_name,
                    'reason': decision.reasoning
                })
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return execution_results
    
    async def _execute_single_scaling_decision(self, decision: ScalingDecision) -> bool:
        """Exécute une décision de scaling individuelle"""
        
        if decision.action_type in ['scale_up', 'scale_down']:
            # Création d'une décision HPA compatible
            hpa_decision = HPADecision(
                current_replicas=decision.current_replicas,
                desired_replicas=decision.target_replicas,
                scaling_reason=decision.reasoning,
                confidence=0.8,  # Confiance par défaut
                metrics_snapshot={},
                timestamp=datetime.utcnow()
            )
            
            # Application du scaling
            success = await self.hpa_controller.apply_scaling_decision(
                hpa_decision, decision.service_name
            )
            
            if success:
                # Mise à jour de l'état du tenant
                await self._update_tenant_state(decision.tenant_id)
                
                # Enregistrement de l'action
                self._record_scaling_action(decision)
            
            return success
        
        elif decision.action_type == 'maintain':
            # Pas d'action requise
            return True
        
        else:
            logger.warning(f"Unknown action type: {decision.action_type}")
            return False
    
    def _record_scaling_action(self, decision: ScalingDecision):
        """Enregistre une action de scaling pour le suivi"""
        # TODO: Implémentation du tracking des actions de scaling
        pass
    
    async def _get_tenant_services(self, tenant_id: str) -> List[str]:
        """Récupère la liste des services d'un tenant"""
        # TODO: Implémentation réelle basée sur les labels Kubernetes ou la DB
        return ["api-service", "ml-service", "audio-processor"]
    
    async def _get_service_replicas(self, service_name: str) -> int:
        """Récupère le nombre actuel de replicas d'un service"""
        # TODO: Implémentation réelle via l'API Kubernetes
        return 1
    
    async def _get_service_resources(self, service_name: str) -> Dict[str, float]:
        """Récupère les ressources d'un service"""
        # TODO: Implémentation réelle via l'API Kubernetes
        return {
            'cpu': 0.5,        # cores
            'memory_gb': 1.0,  # GB
            'storage_gb': 10.0 # GB
        }
    
    def _get_last_scaling_action(self, tenant_id: str) -> Optional[datetime]:
        """Récupère la date de la dernière action de scaling"""
        # TODO: Implémentation du tracking
        return None
    
    def _estimate_cost_impact(self, service_name: str, current_replicas: int, target_replicas: int) -> float:
        """Estime l'impact coût d'un changement de scaling"""
        # Coûts estimés par replica ($/heure)
        cost_per_replica_hour = {
            'api-service': 0.10,
            'ml-service': 0.25,
            'audio-processor': 0.15,
        }
        
        cost_per_hour = cost_per_replica_hour.get(service_name, 0.10)
        replica_delta = target_replicas - current_replicas
        
        # Impact mensuel (approximation)
        monthly_impact = replica_delta * cost_per_hour * 24 * 30
        
        return monthly_impact
    
    def get_tenant_scaling_summary(self, tenant_id: str) -> Dict[str, Any]:
        """Génère un résumé du scaling pour un tenant"""
        
        tenant_state = self.tenant_states.get(tenant_id)
        if not tenant_state:
            return {'error': 'Tenant not found'}
        
        tenant_config = self.config_manager.get_tenant_config(tenant_id)
        
        return {
            'tenant_id': tenant_id,
            'tier': tenant_config.tier.value if tenant_config else 'unknown',
            'priority_score': tenant_state.priority_score,
            'current_resources': {
                'total_replicas': tenant_state.total_replicas,
                'total_cpu_cores': tenant_state.total_cpu_requests,
                'total_memory_gb': tenant_state.total_memory_requests
            },
            'resource_limits': {
                'max_replicas': tenant_config.resource_limits.max_replicas if tenant_config else 'N/A',
                'max_cpu': tenant_config.resource_limits.max_cpu if tenant_config else 'N/A',
                'max_memory': tenant_config.resource_limits.max_memory if tenant_config else 'N/A'
            },
            'active_services': tenant_state.active_services,
            'last_scaling_action': tenant_state.last_scaling_action.isoformat() if tenant_state.last_scaling_action else None,
            'scaling_debt': tenant_state.scaling_debt
        }


@dataclass
class ResourceCheckResult:
    """Résultat de vérification des ressources"""
    allowed: bool
    reason: str


class GlobalResourcePool:
    """Gestionnaire du pool global de ressources"""
    
    def __init__(self):
        self.total_cpu_capacity = 1000.0
        self.total_memory_capacity = 2000.0
        self.reserved_cpu = 0.0
        self.reserved_memory = 0.0
    
    def reserve_resources(self, cpu: float, memory: float) -> bool:
        """Réserve des ressources"""
        if (self.reserved_cpu + cpu <= self.total_cpu_capacity and 
            self.reserved_memory + memory <= self.total_memory_capacity):
            self.reserved_cpu += cpu
            self.reserved_memory += memory
            return True
        return False
    
    def release_resources(self, cpu: float, memory: float):
        """Libère des ressources"""
        self.reserved_cpu = max(0, self.reserved_cpu - cpu)
        self.reserved_memory = max(0, self.reserved_memory - memory)


class ScalingQueue:
    """Queue de scaling pour la gestion des priorités"""
    
    def __init__(self):
        self.queue = []
        self.processing = False
    
    def add_decision(self, decision: ScalingDecision):
        """Ajoute une décision à la queue"""
        self.queue.append(decision)
        self.queue.sort(key=lambda d: -d.priority)
    
    def get_next_batch(self, max_size: int = 5) -> List[ScalingDecision]:
        """Récupère le prochain batch de décisions"""
        batch = self.queue[:max_size]
        self.queue = self.queue[max_size:]
        return batch


class FairnessController:
    """Contrôleur de fairness pour l'allocation des ressources"""
    
    def __init__(self):
        self.tenant_usage_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self.fairness_window = timedelta(minutes=30)
    
    def adjust_priorities(self, tenant_id: str, decisions: List[ScalingDecision]) -> List[ScalingDecision]:
        """Ajuste les priorités selon la fairness"""
        
        # Calcul de l'utilisation récente du tenant
        recent_usage = self._get_recent_usage(tenant_id)
        
        # Ajustement des priorités
        for decision in decisions:
            if recent_usage > 0.8:  # Tenant ayant beaucoup utilisé récemment
                decision.priority = max(1, decision.priority - 2)
            elif recent_usage < 0.2:  # Tenant ayant peu utilisé
                decision.priority = min(10, decision.priority + 1)
        
        return decisions
    
    def _get_recent_usage(self, tenant_id: str) -> float:
        """Calcule l'utilisation récente d'un tenant"""
        cutoff_time = datetime.utcnow() - self.fairness_window
        recent_usage = [
            usage for timestamp, usage in self.tenant_usage_history[tenant_id]
            if timestamp > cutoff_time
        ]
        
        return sum(recent_usage) / len(recent_usage) if recent_usage else 0.0
    
    def record_usage(self, tenant_id: str, usage_factor: float):
        """Enregistre l'utilisation d'un tenant"""
        self.tenant_usage_history[tenant_id].append((datetime.utcnow(), usage_factor))
        
        # Nettoyage de l'historique
        cutoff_time = datetime.utcnow() - self.fairness_window * 2
        self.tenant_usage_history[tenant_id] = [
            (timestamp, usage) for timestamp, usage in self.tenant_usage_history[tenant_id]
            if timestamp > cutoff_time
        ]
