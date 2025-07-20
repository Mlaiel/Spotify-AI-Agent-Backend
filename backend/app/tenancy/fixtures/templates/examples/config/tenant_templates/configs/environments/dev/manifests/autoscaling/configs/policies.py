"""
Advanced Policy Management System
================================

Système de gestion des politiques d'autoscaling avancé.
Intègre intelligence artificielle, apprentissage automatique et gouvernance.

Fonctionnalités:
- Politiques dynamiques adaptatives
- Machine Learning pour optimisation
- Gouvernance automatisée
- Conformité réglementaire
"""

import asyncio
import json
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import structlog
import uuid
from abc import ABC, abstractmethod

# Configuration logging
logger = structlog.get_logger(__name__)


class PolicyType(Enum):
    """Types de politiques."""
    SCALING = "scaling"
    RESOURCE = "resource"
    SECURITY = "security"
    PERFORMANCE = "performance"
    COST = "cost"
    COMPLIANCE = "compliance"
    GOVERNANCE = "governance"
    CUSTOM = "custom"


class PolicyScope(Enum):
    """Portée des politiques."""
    GLOBAL = "global"
    TENANT = "tenant"
    APPLICATION = "application"
    SERVICE = "service"
    RESOURCE = "resource"


class PolicyPriority(Enum):
    """Priorité des politiques."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    INFO = 5


class PolicyAction(Enum):
    """Actions de politique."""
    ALLOW = "allow"
    DENY = "deny"
    WARN = "warn"
    AUDIT = "audit"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    OPTIMIZE = "optimize"
    QUARANTINE = "quarantine"


class PolicyStatus(Enum):
    """États des politiques."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DRAFT = "draft"
    DEPRECATED = "deprecated"
    TESTING = "testing"


@dataclass
class PolicyCondition:
    """Condition de politique."""
    field: str
    operator: str  # eq, ne, gt, lt, gte, lte, in, not_in, contains, regex
    value: Any
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyRule:
    """Règle de politique."""
    rule_id: str
    name: str
    description: str
    conditions: List[PolicyCondition]
    actions: List[PolicyAction]
    priority: PolicyPriority = PolicyPriority.MEDIUM
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Policy:
    """Politique complète."""
    policy_id: str
    name: str
    description: str
    policy_type: PolicyType
    scope: PolicyScope
    rules: List[PolicyRule]
    status: PolicyStatus = PolicyStatus.DRAFT
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyEvaluation:
    """Résultat d'évaluation de politique."""
    evaluation_id: str
    policy_id: str
    rule_id: str
    context: Dict[str, Any]
    result: bool
    action: PolicyAction
    confidence: float
    reason: str
    evaluated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PolicyEngine:
    """
    Moteur de politiques avancé avec intelligence artificielle.
    
    Fonctionnalités:
    - Évaluation de politiques en temps réel
    - Apprentissage automatique pour optimisation
    - Gestion dynamique des règles
    - Audit et traçabilité complète
    """
    
    def __init__(self):
        self.policies: Dict[str, Policy] = {}
        self.policy_cache: Dict[str, Any] = {}
        self.evaluation_history: List[PolicyEvaluation] = []
        self.ml_model = None
        
        # Configuration
        self.cache_ttl = 300  # 5 minutes
        self.max_history = 10000
        self.ml_enabled = True
        self.learning_enabled = True
        
        # Métriques
        self.evaluation_count = 0
        self.cache_hits = 0
        self.ml_predictions = 0
        
        logger.info("PolicyEngine initialized")
    
    async def initialize(self):
        """Initialise le moteur de politiques."""
        try:
            # Charger les politiques par défaut
            await self._load_default_policies()
            
            # Initialiser le modèle ML
            if self.ml_enabled:
                await self._initialize_ml_model()
            
            # Démarrer les tâches de maintenance
            asyncio.create_task(self._cache_cleanup_loop())
            asyncio.create_task(self._ml_training_loop())
            asyncio.create_task(self._policy_optimization_loop())
            
            logger.info("PolicyEngine fully initialized")
            
        except Exception as e:
            logger.error("Failed to initialize PolicyEngine", error=str(e))
            raise
    
    async def create_policy(
        self,
        name: str,
        description: str,
        policy_type: PolicyType,
        scope: PolicyScope,
        rules: List[PolicyRule],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Policy:
        """Crée une nouvelle politique."""
        try:
            policy_id = str(uuid.uuid4())
            
            policy = Policy(
                policy_id=policy_id,
                name=name,
                description=description,
                policy_type=policy_type,
                scope=scope,
                rules=rules,
                metadata=metadata or {}
            )
            
            # Valider la politique
            await self._validate_policy(policy)
            
            # Enregistrer la politique
            self.policies[policy_id] = policy
            
            # Invalider le cache
            await self._invalidate_cache()
            
            logger.info(
                "Policy created",
                policy_id=policy_id,
                name=name,
                type=policy_type.value,
                scope=scope.value
            )
            
            return policy
            
        except Exception as e:
            logger.error(
                "Failed to create policy",
                name=name,
                type=policy_type.value,
                error=str(e)
            )
            raise
    
    async def evaluate_policies(
        self,
        context: Dict[str, Any],
        policy_type: Optional[PolicyType] = None,
        scope: Optional[PolicyScope] = None
    ) -> List[PolicyEvaluation]:
        """Évalue les politiques pour un contexte donné."""
        try:
            evaluations = []
            
            # Filtrer les politiques applicables
            applicable_policies = await self._get_applicable_policies(
                context, policy_type, scope
            )
            
            # Évaluer chaque politique
            for policy in applicable_policies:
                policy_evaluations = await self._evaluate_policy(policy, context)
                evaluations.extend(policy_evaluations)
            
            # Trier par priorité
            evaluations.sort(key=lambda e: self.policies[e.policy_id].rules[0].priority.value)
            
            # Enregistrer dans l'historique
            self.evaluation_history.extend(evaluations)
            self._trim_history()
            
            # Mettre à jour les métriques
            self.evaluation_count += len(evaluations)
            
            # Apprentissage ML
            if self.learning_enabled and self.ml_model:
                await self._update_ml_model(context, evaluations)
            
            logger.debug(
                "Policies evaluated",
                context_keys=list(context.keys()),
                evaluations_count=len(evaluations),
                applicable_policies=len(applicable_policies)
            )
            
            return evaluations
            
        except Exception as e:
            logger.error(
                "Failed to evaluate policies",
                context_keys=list(context.keys()) if context else [],
                error=str(e)
            )
            return []
    
    async def update_policy(
        self,
        policy_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Met à jour une politique."""
        try:
            if policy_id not in self.policies:
                logger.error("Policy not found", policy_id=policy_id)
                return False
            
            policy = self.policies[policy_id]
            
            # Appliquer les mises à jour
            for key, value in updates.items():
                if hasattr(policy, key):
                    setattr(policy, key, value)
            
            policy.updated_at = datetime.utcnow()
            
            # Valider la politique mise à jour
            await self._validate_policy(policy)
            
            # Invalider le cache
            await self._invalidate_cache()
            
            logger.info("Policy updated", policy_id=policy_id, updates=list(updates.keys()))
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to update policy",
                policy_id=policy_id,
                error=str(e)
            )
            return False
    
    async def delete_policy(self, policy_id: str) -> bool:
        """Supprime une politique."""
        try:
            if policy_id not in self.policies:
                logger.error("Policy not found", policy_id=policy_id)
                return False
            
            # Marquer comme dépréciée au lieu de supprimer
            policy = self.policies[policy_id]
            policy.status = PolicyStatus.DEPRECATED
            policy.updated_at = datetime.utcnow()
            
            # Invalider le cache
            await self._invalidate_cache()
            
            logger.info("Policy deprecated", policy_id=policy_id)
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to delete policy",
                policy_id=policy_id,
                error=str(e)
            )
            return False
    
    async def get_policy_recommendations(
        self,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Génère des recommandations de politiques basées sur l'IA."""
        try:
            recommendations = []
            
            if not self.ml_model:
                return recommendations
            
            # Analyser le contexte avec le modèle ML
            analysis = await self._analyze_context_with_ml(context)
            
            # Générer des recommandations
            if analysis.get("scaling_recommendation"):
                recommendations.append({
                    "type": "scaling",
                    "action": analysis["scaling_recommendation"],
                    "confidence": analysis.get("confidence", 0.5),
                    "reason": analysis.get("reason", "ML prediction")
                })
            
            if analysis.get("resource_optimization"):
                recommendations.append({
                    "type": "resource",
                    "action": "optimize",
                    "details": analysis["resource_optimization"],
                    "confidence": analysis.get("confidence", 0.5)
                })
            
            if analysis.get("security_alert"):
                recommendations.append({
                    "type": "security",
                    "action": "investigate",
                    "alert": analysis["security_alert"],
                    "priority": "high"
                })
            
            self.ml_predictions += len(recommendations)
            
            logger.info(
                "Policy recommendations generated",
                recommendations_count=len(recommendations),
                context_keys=list(context.keys())
            )
            
            return recommendations
            
        except Exception as e:
            logger.error(
                "Failed to generate policy recommendations",
                error=str(e)
            )
            return []
    
    async def get_policies(
        self,
        policy_type: Optional[PolicyType] = None,
        scope: Optional[PolicyScope] = None,
        status: Optional[PolicyStatus] = None
    ) -> List[Policy]:
        """Récupère les politiques selon les critères."""
        try:
            policies = list(self.policies.values())
            
            # Filtrer par type
            if policy_type:
                policies = [p for p in policies if p.policy_type == policy_type]
            
            # Filtrer par portée
            if scope:
                policies = [p for p in policies if p.scope == scope]
            
            # Filtrer par statut
            if status:
                policies = [p for p in policies if p.status == status]
            
            return policies
            
        except Exception as e:
            logger.error("Failed to get policies", error=str(e))
            return []
    
    async def get_policy_metrics(self) -> Dict[str, Any]:
        """Récupère les métriques des politiques."""
        try:
            active_policies = len([p for p in self.policies.values() if p.status == PolicyStatus.ACTIVE])
            
            # Statistiques d'évaluation
            recent_evaluations = [
                e for e in self.evaluation_history
                if e.evaluated_at > datetime.utcnow() - timedelta(hours=24)
            ]
            
            success_rate = 0
            if recent_evaluations:
                successful = len([e for e in recent_evaluations if e.result])
                success_rate = successful / len(recent_evaluations) * 100
            
            # Métriques par type de politique
            type_stats = {}
            for policy_type in PolicyType:
                count = len([p for p in self.policies.values() if p.policy_type == policy_type])
                type_stats[policy_type.value] = count
            
            return {
                "total_policies": len(self.policies),
                "active_policies": active_policies,
                "evaluation_count": self.evaluation_count,
                "cache_hits": self.cache_hits,
                "ml_predictions": self.ml_predictions,
                "success_rate": success_rate,
                "recent_evaluations": len(recent_evaluations),
                "type_distribution": type_stats,
                "cache_size": len(self.policy_cache)
            }
            
        except Exception as e:
            logger.error("Failed to get policy metrics", error=str(e))
            return {}
    
    # Méthodes privées
    
    async def _load_default_policies(self):
        """Charge les politiques par défaut."""
        try:
            # Politique de scaling par défaut
            scaling_rules = [
                PolicyRule(
                    rule_id="cpu_scale_up",
                    name="CPU Scale Up",
                    description="Scale up when CPU usage is high",
                    conditions=[
                        PolicyCondition("cpu_usage", "gt", 80),
                        PolicyCondition("duration", "gt", 300)  # 5 minutes
                    ],
                    actions=[PolicyAction.SCALE_UP],
                    priority=PolicyPriority.HIGH
                ),
                PolicyRule(
                    rule_id="cpu_scale_down",
                    name="CPU Scale Down",
                    description="Scale down when CPU usage is low",
                    conditions=[
                        PolicyCondition("cpu_usage", "lt", 20),
                        PolicyCondition("duration", "gt", 600)  # 10 minutes
                    ],
                    actions=[PolicyAction.SCALE_DOWN],
                    priority=PolicyPriority.MEDIUM
                )
            ]
            
            scaling_policy = Policy(
                policy_id="default_scaling",
                name="Default Scaling Policy",
                description="Default autoscaling policy for CPU-based scaling",
                policy_type=PolicyType.SCALING,
                scope=PolicyScope.GLOBAL,
                rules=scaling_rules,
                status=PolicyStatus.ACTIVE
            )
            
            self.policies[scaling_policy.policy_id] = scaling_policy
            
            # Politique de sécurité par défaut
            security_rules = [
                PolicyRule(
                    rule_id="suspicious_activity",
                    name="Suspicious Activity Detection",
                    description="Detect and quarantine suspicious activities",
                    conditions=[
                        PolicyCondition("failed_requests", "gt", 100),
                        PolicyCondition("time_window", "eq", 60)
                    ],
                    actions=[PolicyAction.QUARANTINE, PolicyAction.AUDIT],
                    priority=PolicyPriority.CRITICAL
                )
            ]
            
            security_policy = Policy(
                policy_id="default_security",
                name="Default Security Policy",
                description="Default security policy for threat detection",
                policy_type=PolicyType.SECURITY,
                scope=PolicyScope.GLOBAL,
                rules=security_rules,
                status=PolicyStatus.ACTIVE
            )
            
            self.policies[security_policy.policy_id] = security_policy
            
            logger.info("Default policies loaded", count=len(self.policies))
            
        except Exception as e:
            logger.error("Failed to load default policies", error=str(e))
            raise
    
    async def _initialize_ml_model(self):
        """Initialise le modèle ML pour les prédictions."""
        try:
            # Simuler l'initialisation d'un modèle ML
            # En production, ceci chargerait un vrai modèle TensorFlow/PyTorch
            self.ml_model = {
                "type": "policy_predictor",
                "version": "1.0",
                "initialized": True,
                "features": ["cpu_usage", "memory_usage", "request_rate", "error_rate"],
                "accuracy": 0.85
            }
            
            logger.info("ML model initialized for policy prediction")
            
        except Exception as e:
            logger.error("Failed to initialize ML model", error=str(e))
            # Continuer sans ML si l'initialisation échoue
            self.ml_enabled = False
    
    async def _validate_policy(self, policy: Policy):
        """Valide une politique."""
        if not policy.name:
            raise ValueError("Policy name is required")
        
        if not policy.rules:
            raise ValueError("Policy must have at least one rule")
        
        for rule in policy.rules:
            if not rule.conditions:
                raise ValueError(f"Rule {rule.rule_id} must have at least one condition")
            
            if not rule.actions:
                raise ValueError(f"Rule {rule.rule_id} must have at least one action")
    
    async def _get_applicable_policies(
        self,
        context: Dict[str, Any],
        policy_type: Optional[PolicyType] = None,
        scope: Optional[PolicyScope] = None
    ) -> List[Policy]:
        """Récupère les politiques applicables pour un contexte."""
        applicable = []
        
        for policy in self.policies.values():
            if policy.status != PolicyStatus.ACTIVE:
                continue
            
            if policy_type and policy.policy_type != policy_type:
                continue
            
            if scope and policy.scope != scope:
                continue
            
            # Vérifier si la politique s'applique au contexte
            if await self._policy_applies_to_context(policy, context):
                applicable.append(policy)
        
        return applicable
    
    async def _policy_applies_to_context(
        self,
        policy: Policy,
        context: Dict[str, Any]
    ) -> bool:
        """Vérifie si une politique s'applique à un contexte."""
        # Logique simple de correspondance
        # En production, ceci serait plus sophistiqué
        
        if policy.scope == PolicyScope.GLOBAL:
            return True
        
        if policy.scope == PolicyScope.TENANT:
            return "tenant_id" in context
        
        if policy.scope == PolicyScope.APPLICATION:
            return "application_id" in context
        
        return True
    
    async def _evaluate_policy(
        self,
        policy: Policy,
        context: Dict[str, Any]
    ) -> List[PolicyEvaluation]:
        """Évalue une politique spécifique."""
        evaluations = []
        
        for rule in policy.rules:
            if not rule.enabled:
                continue
            
            evaluation = await self._evaluate_rule(policy, rule, context)
            evaluations.append(evaluation)
        
        return evaluations
    
    async def _evaluate_rule(
        self,
        policy: Policy,
        rule: PolicyRule,
        context: Dict[str, Any]
    ) -> PolicyEvaluation:
        """Évalue une règle spécifique."""
        try:
            result = True
            confidence = 1.0
            reasons = []
            
            # Évaluer chaque condition
            for condition in rule.conditions:
                condition_result, condition_confidence = await self._evaluate_condition(
                    condition, context
                )
                
                if not condition_result:
                    result = False
                    reasons.append(f"Condition {condition.field} {condition.operator} {condition.value} failed")
                
                confidence *= condition_confidence * condition.weight
            
            # Déterminer l'action
            action = rule.actions[0] if rule.actions else PolicyAction.AUDIT
            
            evaluation = PolicyEvaluation(
                evaluation_id=str(uuid.uuid4()),
                policy_id=policy.policy_id,
                rule_id=rule.rule_id,
                context=context.copy(),
                result=result,
                action=action,
                confidence=confidence,
                reason="; ".join(reasons) if reasons else "All conditions met"
            )
            
            return evaluation
            
        except Exception as e:
            logger.error(
                "Failed to evaluate rule",
                policy_id=policy.policy_id,
                rule_id=rule.rule_id,
                error=str(e)
            )
            
            # Retourner une évaluation d'échec
            return PolicyEvaluation(
                evaluation_id=str(uuid.uuid4()),
                policy_id=policy.policy_id,
                rule_id=rule.rule_id,
                context=context.copy(),
                result=False,
                action=PolicyAction.AUDIT,
                confidence=0.0,
                reason=f"Evaluation error: {str(e)}"
            )
    
    async def _evaluate_condition(
        self,
        condition: PolicyCondition,
        context: Dict[str, Any]
    ) -> tuple[bool, float]:
        """Évalue une condition."""
        try:
            field_value = context.get(condition.field)
            
            if field_value is None:
                return False, 0.0
            
            # Évaluer selon l'opérateur
            if condition.operator == "eq":
                result = field_value == condition.value
            elif condition.operator == "ne":
                result = field_value != condition.value
            elif condition.operator == "gt":
                result = field_value > condition.value
            elif condition.operator == "lt":
                result = field_value < condition.value
            elif condition.operator == "gte":
                result = field_value >= condition.value
            elif condition.operator == "lte":
                result = field_value <= condition.value
            elif condition.operator == "in":
                result = field_value in condition.value
            elif condition.operator == "not_in":
                result = field_value not in condition.value
            elif condition.operator == "contains":
                result = condition.value in str(field_value)
            else:
                result = False
            
            confidence = 1.0 if result else 0.0
            
            return result, confidence
            
        except Exception as e:
            logger.error(
                "Failed to evaluate condition",
                field=condition.field,
                operator=condition.operator,
                error=str(e)
            )
            return False, 0.0
    
    async def _analyze_context_with_ml(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse le contexte avec le modèle ML."""
        try:
            if not self.ml_model:
                return {}
            
            # Simuler l'analyse ML
            # En production, ceci utiliserait un vrai modèle
            
            cpu_usage = context.get("cpu_usage", 0)
            memory_usage = context.get("memory_usage", 0)
            request_rate = context.get("request_rate", 0)
            
            analysis = {}
            
            # Prédiction de scaling
            if cpu_usage > 75 or memory_usage > 80:
                analysis["scaling_recommendation"] = "scale_up"
                analysis["confidence"] = 0.8
                analysis["reason"] = "High resource usage detected"
            elif cpu_usage < 25 and memory_usage < 30:
                analysis["scaling_recommendation"] = "scale_down"
                analysis["confidence"] = 0.7
                analysis["reason"] = "Low resource usage detected"
            
            # Optimisation des ressources
            if request_rate > 1000 and cpu_usage < 50:
                analysis["resource_optimization"] = {
                    "suggestion": "increase_cpu_efficiency",
                    "reason": "High request rate with low CPU usage"
                }
            
            # Alerte sécurité
            failed_requests = context.get("failed_requests", 0)
            if failed_requests > 50:
                analysis["security_alert"] = {
                    "type": "suspicious_activity",
                    "severity": "medium" if failed_requests < 100 else "high"
                }
            
            return analysis
            
        except Exception as e:
            logger.error("ML analysis failed", error=str(e))
            return {}
    
    async def _update_ml_model(
        self,
        context: Dict[str, Any],
        evaluations: List[PolicyEvaluation]
    ):
        """Met à jour le modèle ML avec les nouvelles données."""
        try:
            if not self.ml_model or not self.learning_enabled:
                return
            
            # Simuler la mise à jour du modèle
            # En production, ceci entrainerait le modèle avec les nouvelles données
            
            # Extraire les features
            features = {
                "cpu_usage": context.get("cpu_usage", 0),
                "memory_usage": context.get("memory_usage", 0),
                "request_rate": context.get("request_rate", 0),
                "error_rate": context.get("error_rate", 0)
            }
            
            # Extraire les labels (résultats des évaluations)
            labels = {
                "scaling_needed": any(e.action in [PolicyAction.SCALE_UP, PolicyAction.SCALE_DOWN] for e in evaluations),
                "optimization_needed": any(e.action == PolicyAction.OPTIMIZE for e in evaluations),
                "security_alert": any(e.action == PolicyAction.QUARANTINE for e in evaluations)
            }
            
            # Simuler l'apprentissage
            logger.debug("ML model updated with new training data", features=features, labels=labels)
            
        except Exception as e:
            logger.error("Failed to update ML model", error=str(e))
    
    async def _invalidate_cache(self):
        """Invalide le cache des politiques."""
        self.policy_cache.clear()
    
    def _trim_history(self):
        """Taille l'historique des évaluations."""
        if len(self.evaluation_history) > self.max_history:
            self.evaluation_history = self.evaluation_history[-self.max_history:]
    
    # Boucles de maintenance
    
    async def _cache_cleanup_loop(self):
        """Boucle de nettoyage du cache."""
        while True:
            try:
                # Nettoyer les entrées expirées du cache
                current_time = datetime.utcnow()
                expired_keys = [
                    key for key, value in self.policy_cache.items()
                    if value.get("expires_at", current_time) <= current_time
                ]
                
                for key in expired_keys:
                    del self.policy_cache[key]
                
                if expired_keys:
                    logger.debug("Cache cleanup completed", expired_entries=len(expired_keys))
                
                await asyncio.sleep(60)  # Nettoyer toutes les minutes
                
            except Exception as e:
                logger.error("Error in cache cleanup loop", error=str(e))
                await asyncio.sleep(300)
    
    async def _ml_training_loop(self):
        """Boucle d'entrainement ML."""
        while True:
            try:
                if self.ml_model and self.learning_enabled:
                    # Réentrainer le modèle périodiquement
                    logger.debug("ML model retraining scheduled")
                
                await asyncio.sleep(3600)  # Entrainer toutes les heures
                
            except Exception as e:
                logger.error("Error in ML training loop", error=str(e))
                await asyncio.sleep(7200)
    
    async def _policy_optimization_loop(self):
        """Boucle d'optimisation des politiques."""
        while True:
            try:
                # Analyser les performances des politiques
                # Suggérer des optimisations
                
                await asyncio.sleep(1800)  # Optimiser toutes les 30 minutes
                
            except Exception as e:
                logger.error("Error in policy optimization loop", error=str(e))
                await asyncio.sleep(3600)
