"""
Moteur de politiques de scaling intelligent
Système de règles avancées pour l'autoscaling multi-tenant
Développé par l'équipe d'experts dirigée par Fahed Mlaiel
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import yaml
from pathlib import Path

from .config_manager import AutoscalingConfigManager, ScalingMode, TenantTier

logger = logging.getLogger(__name__)

class PolicyType(Enum):
    REACTIVE = "reactive"           # Réaction aux métriques actuelles
    PREDICTIVE = "predictive"       # Basé sur les prédictions
    SCHEDULED = "scheduled"         # Basé sur les horaires
    EVENT_DRIVEN = "event_driven"   # Basé sur les événements
    COST_AWARE = "cost_aware"      # Optimisation des coûts

class ActionType(Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"
    ALERT = "alert"
    OPTIMIZE = "optimize"

@dataclass
class PolicyCondition:
    """Condition d'une politique de scaling"""
    metric_name: str
    operator: str  # >, <, >=, <=, ==, !=
    threshold: float
    duration_seconds: int = 60  # Durée minimum de la condition
    weight: float = 1.0         # Poids dans la décision

@dataclass
class PolicyAction:
    """Action à exécuter suite à une politique"""
    action_type: ActionType
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    cooldown_seconds: int = 300

@dataclass
class ScalingPolicy:
    """Politique de scaling complète"""
    policy_id: str
    name: str
    description: str
    policy_type: PolicyType
    tenant_filters: List[str] = field(default_factory=list)  # Tenants concernés
    service_filters: List[str] = field(default_factory=list)  # Services concernés
    conditions: List[PolicyCondition] = field(default_factory=list)
    actions: List[PolicyAction] = field(default_factory=list)
    enabled: bool = True
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    schedule_cron: Optional[str] = None  # Pour les politiques programmées

class PolicyEvaluationResult:
    """Résultat d'évaluation d'une politique"""
    
    def __init__(self, policy_id: str, triggered: bool, confidence: float, 
                 actions: List[PolicyAction], reasoning: str):
        self.policy_id = policy_id
        self.triggered = triggered
        self.confidence = confidence
        self.actions = actions
        self.reasoning = reasoning
        self.timestamp = datetime.utcnow()

class ScalingPolicyEngine:
    """Moteur de politiques de scaling intelligent"""
    
    def __init__(self, config_manager: AutoscalingConfigManager):
        self.config_manager = config_manager
        self.policies: Dict[str, ScalingPolicy] = {}
        self.policy_history: Dict[str, List[PolicyEvaluationResult]] = {}
        self.last_evaluations: Dict[str, datetime] = {}
        self.custom_functions: Dict[str, Callable] = {}
        self._load_default_policies()
    
    def _load_default_policies(self):
        """Charge les politiques par défaut"""
        
        # Politique de scaling réactif CPU
        cpu_reactive_policy = ScalingPolicy(
            policy_id="cpu_reactive_high",
            name="CPU High Utilization Reactive",
            description="Scale up when CPU utilization is consistently high",
            policy_type=PolicyType.REACTIVE,
            conditions=[
                PolicyCondition(
                    metric_name="cpu_utilization_percentage",
                    operator=">=",
                    threshold=80.0,
                    duration_seconds=180,
                    weight=1.0
                )
            ],
            actions=[
                PolicyAction(
                    action_type=ActionType.SCALE_UP,
                    parameters={"factor": 1.5, "max_increment": 5},
                    cooldown_seconds=300
                )
            ]
        )
        self.add_policy(cpu_reactive_policy)
        
        # Politique de scaling réactif mémoire
        memory_reactive_policy = ScalingPolicy(
            policy_id="memory_reactive_high",
            name="Memory High Utilization Reactive", 
            description="Scale up when memory utilization is consistently high",
            policy_type=PolicyType.REACTIVE,
            conditions=[
                PolicyCondition(
                    metric_name="memory_utilization_percentage",
                    operator=">=",
                    threshold=85.0,
                    duration_seconds=120,
                    weight=1.2  # Mémoire plus critique
                )
            ],
            actions=[
                PolicyAction(
                    action_type=ActionType.SCALE_UP,
                    parameters={"factor": 1.3, "max_increment": 3},
                    cooldown_seconds=600  # Plus long pour mémoire
                )
            ]
        )
        self.add_policy(memory_reactive_policy)
        
        # Politique de scaling down conservateur
        conservative_scale_down = ScalingPolicy(
            policy_id="conservative_scale_down",
            name="Conservative Scale Down",
            description="Scale down gradually when resources are underutilized",
            policy_type=PolicyType.REACTIVE,
            conditions=[
                PolicyCondition(
                    metric_name="cpu_utilization_percentage",
                    operator="<=",
                    threshold=30.0,
                    duration_seconds=600,  # 10 minutes
                    weight=0.8
                ),
                PolicyCondition(
                    metric_name="memory_utilization_percentage", 
                    operator="<=",
                    threshold=40.0,
                    duration_seconds=600,
                    weight=0.7
                )
            ],
            actions=[
                PolicyAction(
                    action_type=ActionType.SCALE_DOWN,
                    parameters={"factor": 0.8, "min_replicas": 1},
                    cooldown_seconds=900  # 15 minutes
                )
            ]
        )
        self.add_policy(conservative_scale_down)
        
        # Politique de coût pour les tenants basiques
        cost_optimization_basic = ScalingPolicy(
            policy_id="cost_optimization_basic",
            name="Cost Optimization for Basic Tenants",
            description="Aggressive cost optimization for basic tier tenants",
            policy_type=PolicyType.COST_AWARE,
            tenant_filters=["basic_*"],  # Tous les tenants basic
            conditions=[
                PolicyCondition(
                    metric_name="cost_utilization_percentage",
                    operator=">=",
                    threshold=80.0,
                    duration_seconds=300,
                    weight=1.5
                )
            ],
            actions=[
                PolicyAction(
                    action_type=ActionType.OPTIMIZE,
                    parameters={"use_spot_instances": True, "reduce_resources": True},
                    cooldown_seconds=1800  # 30 minutes
                )
            ]
        )
        self.add_policy(cost_optimization_basic)
        
        # Politique d'urgence pour erreurs critiques
        emergency_scale_up = ScalingPolicy(
            policy_id="emergency_scale_up",
            name="Emergency Scale Up on High Error Rate",
            description="Emergency scaling when error rate is critical",
            policy_type=PolicyType.REACTIVE,
            conditions=[
                PolicyCondition(
                    metric_name="error_rate_percentage",
                    operator=">=",
                    threshold=10.0,
                    duration_seconds=60,  # Réaction rapide
                    weight=2.0  # Poids très élevé
                )
            ],
            actions=[
                PolicyAction(
                    action_type=ActionType.SCALE_UP,
                    parameters={"factor": 2.0, "emergency": True},
                    priority=10,  # Priorité maximale
                    cooldown_seconds=60
                ),
                PolicyAction(
                    action_type=ActionType.ALERT,
                    parameters={"severity": "critical", "escalate": True},
                    priority=10
                )
            ]
        )
        self.add_policy(emergency_scale_up)
    
    def add_policy(self, policy: ScalingPolicy):
        """Ajoute une politique au moteur"""
        self.policies[policy.policy_id] = policy
        logger.info(f"Policy added: {policy.policy_id} - {policy.name}")
    
    def remove_policy(self, policy_id: str) -> bool:
        """Supprime une politique"""
        if policy_id in self.policies:
            del self.policies[policy_id]
            logger.info(f"Policy removed: {policy_id}")
            return True
        return False
    
    def enable_policy(self, policy_id: str) -> bool:
        """Active une politique"""
        if policy_id in self.policies:
            self.policies[policy_id].enabled = True
            return True
        return False
    
    def disable_policy(self, policy_id: str) -> bool:
        """Désactive une politique"""
        if policy_id in self.policies:
            self.policies[policy_id].enabled = False
            return True
        return False
    
    async def evaluate_policies(self, tenant_id: str, service_name: str,
                              current_metrics: Dict[str, float]) -> List[PolicyEvaluationResult]:
        """Évalue toutes les politiques applicables"""
        
        results = []
        applicable_policies = self._get_applicable_policies(tenant_id, service_name)
        
        for policy in applicable_policies:
            try:
                result = await self._evaluate_single_policy(
                    policy, tenant_id, service_name, current_metrics
                )
                results.append(result)
                
                # Stockage dans l'historique
                policy_key = f"{tenant_id}:{service_name}:{policy.policy_id}"
                if policy_key not in self.policy_history:
                    self.policy_history[policy_key] = []
                self.policy_history[policy_key].append(result)
                
                # Nettoyage de l'historique (garde 24h)
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                self.policy_history[policy_key] = [
                    r for r in self.policy_history[policy_key]
                    if r.timestamp > cutoff_time
                ]
                
            except Exception as e:
                logger.error(f"Error evaluating policy {policy.policy_id}: {e}")
        
        return results
    
    def _get_applicable_policies(self, tenant_id: str, service_name: str) -> List[ScalingPolicy]:
        """Récupère les politiques applicables à un tenant/service"""
        
        applicable = []
        current_time = datetime.utcnow()
        
        for policy in self.policies.values():
            # Vérifications de base
            if not policy.enabled:
                continue
            
            # Vérification de la validité temporelle
            if policy.valid_from and current_time < policy.valid_from:
                continue
            if policy.valid_until and current_time > policy.valid_until:
                continue
            
            # Vérification des filtres tenant
            if policy.tenant_filters:
                matches = False
                for filter_pattern in policy.tenant_filters:
                    if self._matches_pattern(tenant_id, filter_pattern):
                        matches = True
                        break
                if not matches:
                    continue
            
            # Vérification des filtres service
            if policy.service_filters:
                matches = False
                for filter_pattern in policy.service_filters:
                    if self._matches_pattern(service_name, filter_pattern):
                        matches = True
                        break
                if not matches:
                    continue
            
            # Vérification du cooldown
            policy_key = f"{tenant_id}:{service_name}:{policy.policy_id}"
            if policy_key in self.last_evaluations:
                last_eval = self.last_evaluations[policy_key]
                min_interval = min(action.cooldown_seconds for action in policy.actions)
                if (current_time - last_eval).total_seconds() < min_interval:
                    continue
            
            applicable.append(policy)
        
        return applicable
    
    def _matches_pattern(self, value: str, pattern: str) -> bool:
        """Vérifie si une valeur correspond à un pattern (avec wildcards)"""
        if '*' not in pattern:
            return value == pattern
        
        # Conversion du pattern en regex simple
        import re
        regex_pattern = pattern.replace('*', '.*')
        return re.match(f"^{regex_pattern}$", value) is not None
    
    async def _evaluate_single_policy(self, policy: ScalingPolicy, tenant_id: str,
                                    service_name: str, metrics: Dict[str, float]) -> PolicyEvaluationResult:
        """Évalue une politique individuelle"""
        
        if not policy.conditions:
            return PolicyEvaluationResult(
                policy_id=policy.policy_id,
                triggered=False,
                confidence=0.0,
                actions=[],
                reasoning="No conditions defined"
            )
        
        # Évaluation des conditions
        conditions_met = 0
        total_weight = 0
        condition_details = []
        
        for condition in policy.conditions:
            is_met, confidence = self._evaluate_condition(condition, metrics, tenant_id, service_name)
            
            if is_met:
                conditions_met += condition.weight
                condition_details.append(f"✓ {condition.metric_name} {condition.operator} {condition.threshold}")
            else:
                condition_details.append(f"✗ {condition.metric_name} {condition.operator} {condition.threshold}")
            
            total_weight += condition.weight
        
        # Calcul de la confiance globale
        if total_weight > 0:
            overall_confidence = conditions_met / total_weight
        else:
            overall_confidence = 0.0
        
        # Déclenchement de la politique (seuil : 70% des conditions pondérées)
        triggered = overall_confidence >= 0.7
        
        # Actions à exécuter
        actions_to_execute = policy.actions if triggered else []
        
        # Génération du raisonnement
        reasoning = f"Policy '{policy.name}': {len(condition_details)} conditions evaluated. "
        reasoning += f"Confidence: {overall_confidence:.2f}. "
        reasoning += "Conditions: " + "; ".join(condition_details)
        
        # Mise à jour du dernier temps d'évaluation
        policy_key = f"{tenant_id}:{service_name}:{policy.policy_id}"
        self.last_evaluations[policy_key] = datetime.utcnow()
        
        return PolicyEvaluationResult(
            policy_id=policy.policy_id,
            triggered=triggered,
            confidence=overall_confidence,
            actions=actions_to_execute,
            reasoning=reasoning
        )
    
    def _evaluate_condition(self, condition: PolicyCondition, metrics: Dict[str, float],
                          tenant_id: str, service_name: str) -> tuple[bool, float]:
        """Évalue une condition individuelle"""
        
        # Récupération de la métrique
        metric_value = metrics.get(condition.metric_name)
        if metric_value is None:
            return False, 0.0
        
        # Évaluation de l'opérateur
        operators = {
            '>': lambda x, y: x > y,
            '<': lambda x, y: x < y,
            '>=': lambda x, y: x >= y,
            '<=': lambda x, y: x <= y,
            '==': lambda x, y: abs(x - y) < 0.001,  # Égalité avec tolérance
            '!=': lambda x, y: abs(x - y) >= 0.001
        }
        
        if condition.operator not in operators:
            logger.warning(f"Unknown operator: {condition.operator}")
            return False, 0.0
        
        is_met = operators[condition.operator](metric_value, condition.threshold)
        
        # Calcul de la confiance basée sur la distance au seuil
        if condition.operator in ['>', '>=']:
            distance = (metric_value - condition.threshold) / condition.threshold
            confidence = min(1.0, max(0.0, distance + 1.0))
        elif condition.operator in ['<', '<=']:
            distance = (condition.threshold - metric_value) / condition.threshold
            confidence = min(1.0, max(0.0, distance + 1.0))
        else:  # == ou !=
            distance = abs(metric_value - condition.threshold) / condition.threshold
            confidence = max(0.0, 1.0 - distance)
        
        return is_met, confidence
    
    def get_policy_recommendations(self, tenant_id: str) -> List[Dict[str, Any]]:
        """Génère des recommandations de politiques pour un tenant"""
        
        recommendations = []
        tenant_config = self.config_manager.get_tenant_config(tenant_id)
        
        if not tenant_config:
            return recommendations
        
        # Recommandations basées sur le tier
        if tenant_config.tier == TenantTier.ENTERPRISE:
            recommendations.append({
                'type': 'policy_suggestion',
                'title': 'Politique de scaling prédictif',
                'description': 'Activez le scaling prédictif pour une meilleure performance',
                'priority': 'high',
                'policy_template': 'predictive_scaling_enterprise'
            })
        
        elif tenant_config.tier == TenantTier.BASIC:
            recommendations.append({
                'type': 'policy_suggestion', 
                'title': 'Optimisation des coûts',
                'description': 'Politique de réduction des coûts adaptée au niveau Basic',
                'priority': 'medium',
                'policy_template': 'cost_optimization_basic'
            })
        
        # Recommandations basées sur le mode de scaling
        if tenant_config.scaling_mode == ScalingMode.CONSERVATIVE:
            recommendations.append({
                'type': 'policy_optimization',
                'title': 'Réduction des temps de réaction',
                'description': 'Vos politiques sont très conservatives, considérez un mode équilibré',
                'priority': 'low'
            })
        
        return recommendations
    
    def export_policies(self, format_type: str = "yaml") -> str:
        """Exporte toutes les politiques dans le format spécifié"""
        
        policies_data = {}
        for policy_id, policy in self.policies.items():
            policies_data[policy_id] = {
                'name': policy.name,
                'description': policy.description,
                'type': policy.policy_type.value,
                'enabled': policy.enabled,
                'tenant_filters': policy.tenant_filters,
                'service_filters': policy.service_filters,
                'conditions': [
                    {
                        'metric_name': c.metric_name,
                        'operator': c.operator,
                        'threshold': c.threshold,
                        'duration_seconds': c.duration_seconds,
                        'weight': c.weight
                    }
                    for c in policy.conditions
                ],
                'actions': [
                    {
                        'action_type': a.action_type.value,
                        'parameters': a.parameters,
                        'priority': a.priority,
                        'cooldown_seconds': a.cooldown_seconds
                    }
                    for a in policy.actions
                ]
            }
        
        if format_type.lower() == "json":
            return json.dumps(policies_data, indent=2)
        else:  # yaml par défaut
            return yaml.dump(policies_data, default_flow_style=False)
    
    def import_policies(self, data: str, format_type: str = "yaml"):
        """Importe des politiques depuis une chaîne de données"""
        
        try:
            if format_type.lower() == "json":
                policies_data = json.loads(data)
            else:  # yaml par défaut
                policies_data = yaml.safe_load(data)
            
            for policy_id, policy_data in policies_data.items():
                # Reconstruction des conditions
                conditions = []
                for cond_data in policy_data.get('conditions', []):
                    conditions.append(PolicyCondition(**cond_data))
                
                # Reconstruction des actions
                actions = []
                for action_data in policy_data.get('actions', []):
                    action_data['action_type'] = ActionType(action_data['action_type'])
                    actions.append(PolicyAction(**action_data))
                
                # Création de la politique
                policy = ScalingPolicy(
                    policy_id=policy_id,
                    name=policy_data['name'],
                    description=policy_data['description'],
                    policy_type=PolicyType(policy_data['type']),
                    tenant_filters=policy_data.get('tenant_filters', []),
                    service_filters=policy_data.get('service_filters', []),
                    conditions=conditions,
                    actions=actions,
                    enabled=policy_data.get('enabled', True)
                )
                
                self.add_policy(policy)
                
            logger.info(f"Imported {len(policies_data)} policies")
            
        except Exception as e:
            logger.error(f"Error importing policies: {e}")
            raise
    
    def get_policy_statistics(self) -> Dict[str, Any]:
        """Génère des statistiques sur les politiques"""
        
        total_policies = len(self.policies)
        enabled_policies = sum(1 for p in self.policies.values() if p.enabled)
        
        # Statistiques par type
        type_stats = {}
        for policy_type in PolicyType:
            count = sum(1 for p in self.policies.values() if p.policy_type == policy_type)
            type_stats[policy_type.value] = count
        
        # Statistiques d'évaluation récentes
        recent_evaluations = 0
        recent_triggers = 0
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        
        for history in self.policy_history.values():
            recent_evals = [r for r in history if r.timestamp > cutoff_time]
            recent_evaluations += len(recent_evals)
            recent_triggers += sum(1 for r in recent_evals if r.triggered)
        
        return {
            'total_policies': total_policies,
            'enabled_policies': enabled_policies,
            'disabled_policies': total_policies - enabled_policies,
            'policies_by_type': type_stats,
            'recent_evaluations_1h': recent_evaluations,
            'recent_triggers_1h': recent_triggers,
            'trigger_rate_1h': recent_triggers / recent_evaluations if recent_evaluations > 0 else 0
        }
    
    def register_custom_function(self, name: str, func: Callable):
        """Enregistre une fonction custom pour les politiques avancées"""
        self.custom_functions[name] = func
        logger.info(f"Custom function registered: {name}")
    
    def validate_policy(self, policy: ScalingPolicy) -> List[str]:
        """Valide une politique et retourne les erreurs trouvées"""
        
        errors = []
        
        # Validation de base
        if not policy.policy_id:
            errors.append("Policy ID is required")
        
        if not policy.name:
            errors.append("Policy name is required")
        
        # Validation des conditions
        if not policy.conditions:
            errors.append("At least one condition is required")
        
        for i, condition in enumerate(policy.conditions):
            if condition.threshold < 0:
                errors.append(f"Condition {i}: threshold cannot be negative")
            
            if condition.operator not in ['>', '<', '>=', '<=', '==', '!=']:
                errors.append(f"Condition {i}: invalid operator '{condition.operator}'")
            
            if condition.duration_seconds < 0:
                errors.append(f"Condition {i}: duration cannot be negative")
        
        # Validation des actions
        if not policy.actions:
            errors.append("At least one action is required")
        
        for i, action in enumerate(policy.actions):
            if action.cooldown_seconds < 0:
                errors.append(f"Action {i}: cooldown cannot be negative")
            
            if action.priority < 1 or action.priority > 10:
                errors.append(f"Action {i}: priority must be between 1 and 10")
        
        return errors
