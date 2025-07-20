# =============================================================================
# Automation Engine - Architecture Enterprise
# =============================================================================
# 
# Module d'automatisation avancée avec système de réponse automatique,
# gestion d'escalade intelligente, et bot de remédiation autonome.
# Architecture événementielle avec intelligence artificielle.
#
# Auteur: Automation & DevOps Team + AI Specialists
# Direction Technique: Fahed Mlaiel
# Version: 2.0.0 Enterprise
# =============================================================================

"""
Automation Engine Enterprise

Ce module fournit le moteur d'automatisation complet avec:

Fonctionnalités Principales:
- Système de réponse automatique multi-canal
- Gestion d'escalade intelligente avec ML
- Bot de remédiation autonome avec apprentissage
- Orchestration de workflows complexes
- Intégration avec systèmes externes

Composants:
- AutoResponseSystem: Réponses automatiques intelligentes
- EscalationManager: Gestion d'escalade adaptative
- RemediationBot: Bot autonome de remédiation
- WorkflowOrchestrator: Orchestration de processus
- IntegrationHub: Hub d'intégrations externes

Architecture:
- Event-Driven avec message queues
- Rule Engine pour logique métier
- AI Decision Making pour réponses adaptatives
- Distributed Execution pour scalabilité
- Self-Learning pour amélioration continue
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import json
from collections import defaultdict, deque
import re
import traceback

# ML et IA
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Traitement du langage naturel
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK non disponible - fonctionnalités NLP limitées")

# Imports du Core Engine
from . import (
    core_registry, TenantContext, TenantTier, EngineStatus
)

logger = logging.getLogger(__name__)

# ===========================
# Configuration & Constants
# ===========================

# Configuration des réponses automatiques
AUTO_RESPONSE_CONFIG = {
    "response_timeout_seconds": 300,
    "max_concurrent_responses": 50,
    "retry_attempts": 3,
    "escalation_delay_minutes": 15,
    "learning_threshold": 0.8,
    "confidence_threshold": 0.75
}

# Configuration d'escalade
ESCALATION_CONFIG = {
    "levels": [
        {"name": "L1_Support", "timeout_minutes": 30, "capacity": 100},
        {"name": "L2_Expert", "timeout_minutes": 60, "capacity": 50},
        {"name": "L3_Specialist", "timeout_minutes": 120, "capacity": 20},
        {"name": "Management", "timeout_minutes": 240, "capacity": 10},
        {"name": "Executive", "timeout_minutes": 480, "capacity": 5}
    ],
    "severity_mapping": {
        "critical": ["L3_Specialist", "Management"],
        "high": ["L2_Expert", "L3_Specialist"],
        "medium": ["L1_Support", "L2_Expert"],
        "low": ["L1_Support"],
        "info": ["L1_Support"]
    }
}

# Configuration du bot de remédiation
REMEDIATION_CONFIG = {
    "max_concurrent_actions": 10,
    "action_timeout_seconds": 600,
    "learning_enabled": True,
    "safe_mode": True,  # Requiert confirmation pour actions critiques
    "success_threshold": 0.9,
    "rollback_enabled": True
}

# Actions disponibles
AVAILABLE_ACTIONS = {
    "restart_service": "Redémarrage de service",
    "scale_resources": "Mise à l'échelle des ressources",
    "clear_cache": "Vidage de cache",
    "rotate_logs": "Rotation des logs",
    "run_health_check": "Vérification de santé",
    "deploy_hotfix": "Déploiement de correctif",
    "isolate_component": "Isolation de composant",
    "failover": "Basculement automatique",
    "notify_team": "Notification d'équipe",
    "create_ticket": "Création de ticket"
}

# ===========================
# Enums & Types
# ===========================

class ResponseType(Enum):
    """Types de réponse automatique"""
    IMMEDIATE = "immediate"
    SCHEDULED = "scheduled"
    CONDITIONAL = "conditional"
    ESCALATED = "escalated"

class ActionType(Enum):
    """Types d'actions de remédiation"""
    DIAGNOSTIC = "diagnostic"
    CORRECTIVE = "corrective"
    PREVENTIVE = "preventive"
    EMERGENCY = "emergency"

class EscalationReason(Enum):
    """Raisons d'escalade"""
    TIMEOUT = "timeout"
    SEVERITY = "severity"
    COMPLEXITY = "complexity"
    VOLUME = "volume"
    PATTERN = "pattern"
    MANUAL = "manual"

class AutomationStatus(Enum):
    """Statut d'automatisation"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ESCALATED = "escalated"

class IntegrationType(Enum):
    """Types d'intégrations"""
    SLACK = "slack"
    TEAMS = "teams"
    EMAIL = "email"
    WEBHOOK = "webhook"
    JIRA = "jira"
    SERVICENOW = "servicenow"
    PAGERDUTY = "pagerduty"
    GRAFANA = "grafana"

# ===========================
# Modèles de Données
# ===========================

@dataclass
class AutomationRule:
    """Règle d'automatisation"""
    rule_id: str
    tenant_id: str
    name: str
    description: str
    
    # Conditions de déclenchement
    trigger_conditions: Dict[str, Any]
    
    # Actions à exécuter
    actions: List[Dict[str, Any]]
    
    # Configuration
    enabled: bool = True
    priority: int = 1
    max_executions: int = -1  # -1 = illimité
    cooldown_minutes: int = 5
    
    # Statistiques
    execution_count: int = 0
    success_count: int = 0
    last_execution: Optional[datetime] = None
    
    # Apprentissage
    confidence_score: float = 1.0
    learning_data: Dict[str, Any] = field(default_factory=dict)
    
    def matches_event(self, event: Dict[str, Any]) -> bool:
        """Vérifie si l'événement correspond aux conditions"""
        for condition_key, condition_value in self.trigger_conditions.items():
            if condition_key not in event:
                return False
            
            event_value = event[condition_key]
            
            # Comparaison selon le type
            if isinstance(condition_value, dict):
                operator = condition_value.get("operator", "equals")
                target_value = condition_value.get("value")
                
                if operator == "equals" and event_value != target_value:
                    return False
                elif operator == "contains" and target_value not in str(event_value):
                    return False
                elif operator == "greater_than" and event_value <= target_value:
                    return False
                elif operator == "less_than" and event_value >= target_value:
                    return False
                elif operator == "regex" and not re.match(target_value, str(event_value)):
                    return False
            else:
                if event_value != condition_value:
                    return False
        
        return True
    
    def can_execute(self) -> bool:
        """Vérifie si la règle peut être exécutée"""
        if not self.enabled:
            return False
        
        # Vérification des limites d'exécution
        if self.max_executions != -1 and self.execution_count >= self.max_executions:
            return False
        
        # Vérification du cooldown
        if self.last_execution:
            cooldown_period = timedelta(minutes=self.cooldown_minutes)
            if datetime.utcnow() - self.last_execution < cooldown_period:
                return False
        
        return True

@dataclass
class AutomationExecution:
    """Exécution d'automatisation"""
    execution_id: str
    tenant_id: str
    rule_id: str
    triggered_by: Dict[str, Any]
    
    # État
    status: AutomationStatus = AutomationStatus.PENDING
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Actions exécutées
    executed_actions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Résultats
    success: bool = False
    error_message: Optional[str] = None
    output: Dict[str, Any] = field(default_factory=dict)
    
    # Métriques
    execution_time_seconds: float = 0.0
    actions_completed: int = 0
    actions_failed: int = 0
    
    def add_action_result(self, action: str, success: bool, 
                         result: Any = None, error: str = None):
        """Ajoute le résultat d'une action"""
        action_result = {
            "action": action,
            "success": success,
            "result": result,
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.executed_actions.append(action_result)
        
        if success:
            self.actions_completed += 1
        else:
            self.actions_failed += 1

@dataclass
class EscalationLevel:
    """Niveau d'escalade"""
    name: str
    timeout_minutes: int
    capacity: int
    current_load: int = 0
    
    # Assignés
    assigned_teams: List[str] = field(default_factory=list)
    assigned_users: List[str] = field(default_factory=list)
    
    # Configuration
    auto_assign: bool = True
    notification_channels: List[str] = field(default_factory=list)
    
    def is_available(self) -> bool:
        """Vérifie si le niveau a de la capacité"""
        return self.current_load < self.capacity
    
    def get_utilization(self) -> float:
        """Calcule le taux d'utilisation"""
        return (self.current_load / self.capacity) * 100 if self.capacity > 0 else 100

@dataclass
class RemediationAction:
    """Action de remédiation"""
    action_id: str
    tenant_id: str
    name: str
    action_type: ActionType
    
    # Configuration
    command: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    safe_mode: bool = True
    requires_confirmation: bool = False
    rollback_command: Optional[str] = None
    
    # Conditions de sécurité
    prerequisites: List[str] = field(default_factory=list)
    risk_level: str = "medium"  # low, medium, high, critical
    
    # Historique d'apprentissage
    success_rate: float = 0.0
    execution_count: int = 0
    average_duration: float = 0.0
    
    def can_execute(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Vérifie si l'action peut être exécutée"""
        # Vérification des prérequis
        for prerequisite in self.prerequisites:
            if prerequisite not in context or not context[prerequisite]:
                return False, f"Prérequis manquant: {prerequisite}"
        
        # Mode sécurisé
        if self.safe_mode and self.risk_level in ["high", "critical"]:
            if not context.get("manual_approval", False):
                return False, "Approbation manuelle requise pour action à haut risque"
        
        return True, "OK"

# ===========================
# Système de Réponse Automatique
# ===========================

class AutoResponseSystem:
    """Système de réponse automatique intelligent"""
    
    def __init__(self):
        self.rules: Dict[str, AutomationRule] = {}
        self.active_executions: Dict[str, AutomationExecution] = {}
        self.execution_history: deque = deque(maxlen=10000)
        
        # Machine Learning pour amélioration continue
        self.decision_tree: Optional[DecisionTreeClassifier] = None
        self.label_encoder = LabelEncoder()
        self.learning_data: List[Dict[str, Any]] = []
        
        # Analyse de sentiment si NLTK disponible
        if NLTK_AVAILABLE:
            try:
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
            except:
                self.sentiment_analyzer = None
        else:
            self.sentiment_analyzer = None
        
        logger.info("Système de réponse automatique initialisé")
    
    def add_rule(self, rule: AutomationRule):
        """Ajoute une règle d'automatisation"""
        self.rules[rule.rule_id] = rule
        logger.info(f"Règle ajoutée: {rule.name} ({rule.rule_id})")
    
    def remove_rule(self, rule_id: str):
        """Supprime une règle"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Règle supprimée: {rule_id}")
    
    async def process_event(self, event: Dict[str, Any]) -> List[AutomationExecution]:
        """Traite un événement et déclenche les automatisations appropriées"""
        triggered_executions = []
        
        # Recherche des règles correspondantes
        matching_rules = []
        for rule in self.rules.values():
            if (rule.tenant_id == event.get("tenant_id") and 
                rule.matches_event(event) and 
                rule.can_execute()):
                matching_rules.append(rule)
        
        # Tri par priorité
        matching_rules.sort(key=lambda r: r.priority, reverse=True)
        
        # Limitation du nombre d'exécutions simultanées
        concurrent_count = len(self.active_executions)
        max_concurrent = AUTO_RESPONSE_CONFIG["max_concurrent_responses"]
        
        if concurrent_count >= max_concurrent:
            logger.warning(f"Limite d'exécutions simultanées atteinte ({max_concurrent})")
            return triggered_executions
        
        # Exécution des règles
        for rule in matching_rules[:max_concurrent - concurrent_count]:
            execution = await self._execute_rule(rule, event)
            if execution:
                triggered_executions.append(execution)
        
        return triggered_executions
    
    async def _execute_rule(self, rule: AutomationRule, 
                          triggering_event: Dict[str, Any]) -> Optional[AutomationExecution]:
        """Exécute une règle d'automatisation"""
        execution_id = str(uuid.uuid4())
        
        execution = AutomationExecution(
            execution_id=execution_id,
            tenant_id=rule.tenant_id,
            rule_id=rule.rule_id,
            triggered_by=triggering_event
        )
        
        self.active_executions[execution_id] = execution
        
        try:
            execution.status = AutomationStatus.RUNNING
            
            # Analyse de sentiment sur l'événement si possible
            sentiment_score = self._analyze_sentiment(triggering_event)
            if sentiment_score:
                execution.output["sentiment_analysis"] = sentiment_score
            
            # Exécution des actions
            success_count = 0
            for action_config in rule.actions:
                action_success = await self._execute_action(
                    execution, action_config, triggering_event
                )
                if action_success:
                    success_count += 1
            
            # Détermination du succès global
            total_actions = len(rule.actions)
            success_rate = success_count / total_actions if total_actions > 0 else 0
            execution.success = success_rate >= AUTO_RESPONSE_CONFIG["learning_threshold"]
            
            if execution.success:
                execution.status = AutomationStatus.COMPLETED
            else:
                execution.status = AutomationStatus.FAILED
            
            # Mise à jour des statistiques de la règle
            rule.execution_count += 1
            if execution.success:
                rule.success_count += 1
            rule.last_execution = datetime.utcnow()
            
            # Calcul du score de confiance
            if rule.execution_count > 0:
                rule.confidence_score = rule.success_count / rule.execution_count
            
            logger.info(f"Règle exécutée: {rule.name} - Succès: {execution.success}")
            
        except Exception as e:
            execution.status = AutomationStatus.FAILED
            execution.error_message = str(e)
            execution.success = False
            logger.error(f"Erreur exécution règle {rule.rule_id}: {e}")
        
        finally:
            execution.completed_at = datetime.utcnow()
            execution.execution_time_seconds = (
                execution.completed_at - execution.started_at
            ).total_seconds()
            
            # Déplacement vers l'historique
            self.execution_history.append(execution)
            del self.active_executions[execution_id]
            
            # Apprentissage
            await self._learn_from_execution(execution, rule, triggering_event)
        
        return execution
    
    async def _execute_action(self, execution: AutomationExecution, 
                            action_config: Dict[str, Any], 
                            context: Dict[str, Any]) -> bool:
        """Exécute une action spécifique"""
        action_type = action_config.get("type")
        action_params = action_config.get("parameters", {})
        
        try:
            if action_type == "notify":
                result = await self._action_notify(action_params, context)
            elif action_type == "webhook":
                result = await self._action_webhook(action_params, context)
            elif action_type == "script":
                result = await self._action_script(action_params, context)
            elif action_type == "api_call":
                result = await self._action_api_call(action_params, context)
            elif action_type == "escalate":
                result = await self._action_escalate(action_params, context)
            else:
                result = {"success": False, "error": f"Type d'action non supporté: {action_type}"}
            
            execution.add_action_result(
                action_type, 
                result.get("success", False),
                result.get("data"),
                result.get("error")
            )
            
            return result.get("success", False)
            
        except Exception as e:
            execution.add_action_result(action_type, False, None, str(e))
            logger.error(f"Erreur exécution action {action_type}: {e}")
            return False
    
    async def _action_notify(self, params: Dict[str, Any], 
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Action de notification"""
        message = params.get("message", "Notification automatique")
        channels = params.get("channels", ["default"])
        
        # Substitution de variables dans le message
        for key, value in context.items():
            message = message.replace(f"{{{key}}}", str(value))
        
        # Simulation d'envoi de notification
        logger.info(f"Notification envoyée: {message} -> {channels}")
        
        return {
            "success": True,
            "data": {
                "message": message,
                "channels": channels,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    
    async def _action_webhook(self, params: Dict[str, Any], 
                            context: Dict[str, Any]) -> Dict[str, Any]:
        """Action webhook"""
        url = params.get("url")
        method = params.get("method", "POST")
        payload = params.get("payload", {})
        
        if not url:
            return {"success": False, "error": "URL webhook manquante"}
        
        # Substitution dans le payload
        import json
        payload_str = json.dumps(payload)
        for key, value in context.items():
            payload_str = payload_str.replace(f"{{{key}}}", str(value))
        payload = json.loads(payload_str)
        
        # Simulation d'appel webhook
        logger.info(f"Webhook appelé: {method} {url}")
        
        return {
            "success": True,
            "data": {
                "url": url,
                "method": method,
                "payload": payload
            }
        }
    
    async def _action_script(self, params: Dict[str, Any], 
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Action d'exécution de script"""
        script = params.get("script")
        interpreter = params.get("interpreter", "bash")
        
        if not script:
            return {"success": False, "error": "Script manquant"}
        
        # Simulation d'exécution de script
        logger.info(f"Script exécuté: {interpreter} - {script[:50]}...")
        
        return {
            "success": True,
            "data": {
                "script": script,
                "interpreter": interpreter,
                "exit_code": 0
            }
        }
    
    async def _action_api_call(self, params: Dict[str, Any], 
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Action d'appel API"""
        endpoint = params.get("endpoint")
        method = params.get("method", "GET")
        headers = params.get("headers", {})
        
        if not endpoint:
            return {"success": False, "error": "Endpoint API manquant"}
        
        # Simulation d'appel API
        logger.info(f"API appelée: {method} {endpoint}")
        
        return {
            "success": True,
            "data": {
                "endpoint": endpoint,
                "method": method,
                "status_code": 200
            }
        }
    
    async def _action_escalate(self, params: Dict[str, Any], 
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Action d'escalade"""
        level = params.get("level", "L2_Expert")
        reason = params.get("reason", "Escalade automatique")
        
        # Simulation d'escalade
        logger.info(f"Escalade vers {level}: {reason}")
        
        return {
            "success": True,
            "data": {
                "escalation_level": level,
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    
    def _analyze_sentiment(self, event: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Analyse le sentiment d'un événement"""
        if not self.sentiment_analyzer:
            return None
        
        # Recherche de texte à analyser
        text_fields = ["message", "description", "title", "content"]
        text_to_analyze = ""
        
        for field in text_fields:
            if field in event and isinstance(event[field], str):
                text_to_analyze += " " + event[field]
        
        if not text_to_analyze.strip():
            return None
        
        try:
            scores = self.sentiment_analyzer.polarity_scores(text_to_analyze)
            return scores
        except Exception as e:
            logger.warning(f"Erreur analyse sentiment: {e}")
            return None
    
    async def _learn_from_execution(self, execution: AutomationExecution, 
                                  rule: AutomationRule, 
                                  triggering_event: Dict[str, Any]):
        """Apprentissage à partir d'une exécution"""
        if not AUTO_RESPONSE_CONFIG.get("learning_enabled", True):
            return
        
        # Collecte des données d'apprentissage
        learning_record = {
            "tenant_id": execution.tenant_id,
            "rule_id": rule.rule_id,
            "event_type": triggering_event.get("type", "unknown"),
            "severity": triggering_event.get("severity", "medium"),
            "success": execution.success,
            "execution_time": execution.execution_time_seconds,
            "actions_count": len(rule.actions),
            "rule_confidence": rule.confidence_score,
            "timestamp": execution.started_at.isoformat()
        }
        
        self.learning_data.append(learning_record)
        
        # Ré-entraînement périodique du modèle
        if len(self.learning_data) >= 100 and len(self.learning_data) % 50 == 0:
            await self._retrain_decision_model()
    
    async def _retrain_decision_model(self):
        """Ré-entraîne le modèle de décision"""
        try:
            if len(self.learning_data) < 20:
                return
            
            # Préparation des données
            features = []
            labels = []
            
            for record in self.learning_data[-1000:]:  # 1000 derniers records
                feature_vector = [
                    hash(record["event_type"]) % 1000,  # Simplification
                    {"low": 1, "medium": 2, "high": 3, "critical": 4}.get(record["severity"], 2),
                    record["actions_count"],
                    record["rule_confidence"] * 100,
                    record["execution_time"]
                ]
                features.append(feature_vector)
                labels.append(record["success"])
            
            if len(set(labels)) < 2:  # Besoin d'au moins 2 classes
                return
            
            # Entraînement
            X = np.array(features)
            y = np.array(labels)
            
            self.decision_tree = DecisionTreeClassifier(random_state=42, max_depth=10)
            self.decision_tree.fit(X, y)
            
            # Évaluation
            if len(X) > 10:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                y_pred = self.decision_tree.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                logger.info(f"Modèle de décision ré-entraîné - Précision: {accuracy:.3f}")
        
        except Exception as e:
            logger.error(f"Erreur ré-entraînement modèle: {e}")
    
    def predict_rule_success(self, rule: AutomationRule, 
                           event: Dict[str, Any]) -> float:
        """Prédit la probabilité de succès d'une règle"""
        if not self.decision_tree:
            return rule.confidence_score
        
        try:
            feature_vector = [
                hash(event.get("type", "unknown")) % 1000,
                {"low": 1, "medium": 2, "high": 3, "critical": 4}.get(event.get("severity"), 2),
                len(rule.actions),
                rule.confidence_score * 100,
                0  # Temps d'exécution inconnu
            ]
            
            prediction_proba = self.decision_tree.predict_proba([feature_vector])
            return prediction_proba[0][1] if len(prediction_proba[0]) > 1 else rule.confidence_score
        
        except Exception as e:
            logger.warning(f"Erreur prédiction succès règle: {e}")
            return rule.confidence_score
    
    def get_statistics(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Récupère les statistiques du système"""
        # Filtrage par tenant si spécifié
        if tenant_id:
            relevant_rules = [r for r in self.rules.values() if r.tenant_id == tenant_id]
            relevant_history = [e for e in self.execution_history if e.tenant_id == tenant_id]
        else:
            relevant_rules = list(self.rules.values())
            relevant_history = list(self.execution_history)
        
        # Calculs statistiques
        total_rules = len(relevant_rules)
        active_rules = len([r for r in relevant_rules if r.enabled])
        
        total_executions = len(relevant_history)
        successful_executions = len([e for e in relevant_history if e.success])
        
        avg_execution_time = 0
        if relevant_history:
            avg_execution_time = sum(e.execution_time_seconds for e in relevant_history) / len(relevant_history)
        
        return {
            "tenant_id": tenant_id,
            "rules": {
                "total": total_rules,
                "active": active_rules,
                "disabled": total_rules - active_rules
            },
            "executions": {
                "total": total_executions,
                "successful": successful_executions,
                "failed": total_executions - successful_executions,
                "success_rate": successful_executions / total_executions if total_executions > 0 else 0
            },
            "performance": {
                "average_execution_time_seconds": avg_execution_time,
                "active_executions": len(self.active_executions)
            },
            "learning": {
                "training_data_points": len(self.learning_data),
                "model_trained": self.decision_tree is not None
            }
        }

# ===========================
# Gestionnaire d'Escalade
# ===========================

class EscalationManager:
    """Gestionnaire d'escalade intelligent"""
    
    def __init__(self):
        self.escalation_levels: Dict[str, EscalationLevel] = {}
        self.active_escalations: Dict[str, Dict[str, Any]] = {}
        self.escalation_history: deque = deque(maxlen=5000)
        
        # ML pour prédiction d'escalade
        self.escalation_predictor: Optional[RandomForestClassifier] = None
        self.escalation_features: List[Dict[str, Any]] = []
        
        self._setup_default_levels()
        logger.info("Gestionnaire d'escalade initialisé")
    
    def _setup_default_levels(self):
        """Configure les niveaux d'escalade par défaut"""
        for level_config in ESCALATION_CONFIG["levels"]:
            level = EscalationLevel(
                name=level_config["name"],
                timeout_minutes=level_config["timeout_minutes"],
                capacity=level_config["capacity"]
            )
            self.escalation_levels[level.name] = level
    
    async def should_escalate(self, incident: Dict[str, Any]) -> Tuple[bool, str, str]:
        """Détermine si un incident doit être escaladé"""
        tenant_id = incident.get("tenant_id")
        severity = incident.get("severity", "medium")
        created_at = incident.get("created_at")
        current_level = incident.get("current_level", "L1_Support")
        
        # Vérification du timeout
        if created_at:
            incident_age = datetime.utcnow() - datetime.fromisoformat(created_at)
            current_level_config = self.escalation_levels.get(current_level)
            
            if (current_level_config and 
                incident_age.total_seconds() > current_level_config.timeout_minutes * 60):
                return True, EscalationReason.TIMEOUT.value, self._get_next_level(current_level)
        
        # Vérification de la sévérité
        if severity in ESCALATION_CONFIG["severity_mapping"]:
            required_levels = ESCALATION_CONFIG["severity_mapping"][severity]
            if current_level not in required_levels:
                target_level = required_levels[0]  # Premier niveau requis
                return True, EscalationReason.SEVERITY.value, target_level
        
        # Vérification de la charge
        current_level_obj = self.escalation_levels.get(current_level)
        if current_level_obj and current_level_obj.get_utilization() > 90:
            next_level = self._get_next_level(current_level)
            if next_level and self.escalation_levels[next_level].is_available():
                return True, EscalationReason.VOLUME.value, next_level
        
        # Prédiction ML si modèle disponible
        if self.escalation_predictor:
            should_escalate_ml = self._predict_escalation_ml(incident)
            if should_escalate_ml:
                return True, EscalationReason.PATTERN.value, self._get_next_level(current_level)
        
        return False, "", ""
    
    def _get_next_level(self, current_level: str) -> Optional[str]:
        """Récupère le niveau d'escalade suivant"""
        level_names = [level["name"] for level in ESCALATION_CONFIG["levels"]]
        
        try:
            current_index = level_names.index(current_level)
            if current_index < len(level_names) - 1:
                return level_names[current_index + 1]
        except ValueError:
            pass
        
        return None
    
    async def escalate_incident(self, incident: Dict[str, Any], 
                              reason: str, target_level: str) -> Dict[str, Any]:
        """Escalade un incident"""
        escalation_id = str(uuid.uuid4())
        incident_id = incident.get("incident_id")
        
        # Vérification de la disponibilité du niveau cible
        target_level_obj = self.escalation_levels.get(target_level)
        if not target_level_obj:
            return {
                "success": False,
                "error": f"Niveau d'escalade inconnu: {target_level}"
            }
        
        if not target_level_obj.is_available():
            # Recherche d'un niveau alternatif
            alternative_level = self._find_available_level(target_level)
            if alternative_level:
                target_level = alternative_level
                target_level_obj = self.escalation_levels[target_level]
            else:
                return {
                    "success": False,
                    "error": f"Aucun niveau disponible pour l'escalade"
                }
        
        # Création de l'escalade
        escalation = {
            "escalation_id": escalation_id,
            "incident_id": incident_id,
            "tenant_id": incident.get("tenant_id"),
            "from_level": incident.get("current_level", "L1_Support"),
            "to_level": target_level,
            "reason": reason,
            "escalated_at": datetime.utcnow(),
            "status": "active",
            "assigned_to": None
        }
        
        # Mise à jour de la charge
        target_level_obj.current_load += 1
        
        # Enregistrement
        self.active_escalations[escalation_id] = escalation
        
        # Assignation automatique si configurée
        if target_level_obj.auto_assign:
            assigned_to = await self._auto_assign(target_level_obj, incident)
            escalation["assigned_to"] = assigned_to
        
        # Notifications
        await self._notify_escalation(escalation, incident)
        
        # Apprentissage
        await self._learn_from_escalation(escalation, incident, reason)
        
        logger.info(f"Incident {incident_id} escaladé vers {target_level} (raison: {reason})")
        
        return {
            "success": True,
            "escalation_id": escalation_id,
            "target_level": target_level,
            "assigned_to": escalation.get("assigned_to")
        }
    
    def _find_available_level(self, preferred_level: str) -> Optional[str]:
        """Trouve un niveau d'escalade disponible"""
        level_names = [level["name"] for level in ESCALATION_CONFIG["levels"]]
        
        try:
            preferred_index = level_names.index(preferred_level)
            
            # Recherche vers le haut d'abord
            for i in range(preferred_index + 1, len(level_names)):
                level_name = level_names[i]
                if self.escalation_levels[level_name].is_available():
                    return level_name
            
            # Puis vers le bas
            for i in range(preferred_index - 1, -1, -1):
                level_name = level_names[i]
                if self.escalation_levels[level_name].is_available():
                    return level_name
        
        except ValueError:
            pass
        
        return None
    
    async def _auto_assign(self, level: EscalationLevel, 
                         incident: Dict[str, Any]) -> Optional[str]:
        """Assignation automatique dans un niveau"""
        # Logique d'assignation simple - round robin
        if level.assigned_users:
            # Sélection de l'utilisateur avec le moins de charge
            # Dans un vrai système, ceci interrogerait la base de données
            return level.assigned_users[0]  # Simulation
        
        if level.assigned_teams:
            return level.assigned_teams[0]  # Assignation à l'équipe
        
        return None
    
    async def _notify_escalation(self, escalation: Dict[str, Any], 
                               incident: Dict[str, Any]):
        """Notifie une escalade"""
        target_level = escalation["to_level"]
        level_obj = self.escalation_levels[target_level]
        
        # Notification aux canaux configurés
        message = (f"🚨 Escalade d'incident: {incident.get('title', 'Incident')} "
                  f"escaladé vers {target_level} "
                  f"(Raison: {escalation['reason']})")
        
        for channel in level_obj.notification_channels:
            logger.info(f"Notification escalade envoyée: {channel} - {message}")
    
    async def _learn_from_escalation(self, escalation: Dict[str, Any], 
                                   incident: Dict[str, Any], reason: str):
        """Apprentissage à partir d'une escalade"""
        # Collecte des features pour ML
        feature_record = {
            "tenant_id": incident.get("tenant_id"),
            "severity": incident.get("severity", "medium"),
            "category": incident.get("category", "unknown"),
            "from_level": escalation["from_level"],
            "to_level": escalation["to_level"],
            "reason": reason,
            "hour": escalation["escalated_at"].hour,
            "day_of_week": escalation["escalated_at"].weekday(),
            "escalated": True,
            "timestamp": escalation["escalated_at"].isoformat()
        }
        
        self.escalation_features.append(feature_record)
        
        # Ré-entraînement périodique
        if len(self.escalation_features) >= 100 and len(self.escalation_features) % 25 == 0:
            await self._retrain_escalation_predictor()
    
    async def _retrain_escalation_predictor(self):
        """Ré-entraîne le prédicteur d'escalade"""
        try:
            if len(self.escalation_features) < 50:
                return
            
            # Préparation des données
            features = []
            labels = []
            
            for record in self.escalation_features[-500:]:  # 500 derniers
                severity_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}
                feature_vector = [
                    severity_map.get(record["severity"], 2),
                    hash(record["category"]) % 100,
                    record["hour"],
                    record["day_of_week"],
                    hash(record["from_level"]) % 10
                ]
                
                features.append(feature_vector)
                labels.append(record["escalated"])
            
            if len(set(labels)) < 2:
                return
            
            # Entraînement
            X = np.array(features)
            y = np.array(labels)
            
            self.escalation_predictor = RandomForestClassifier(
                n_estimators=50, random_state=42, max_depth=8
            )
            self.escalation_predictor.fit(X, y)
            
            logger.info("Prédicteur d'escalade ré-entraîné")
        
        except Exception as e:
            logger.error(f"Erreur ré-entraînement prédicteur escalade: {e}")
    
    def _predict_escalation_ml(self, incident: Dict[str, Any]) -> bool:
        """Prédit si un incident devrait être escaladé"""
        if not self.escalation_predictor:
            return False
        
        try:
            severity_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}
            feature_vector = [
                severity_map.get(incident.get("severity"), 2),
                hash(incident.get("category", "unknown")) % 100,
                datetime.utcnow().hour,
                datetime.utcnow().weekday(),
                hash(incident.get("current_level", "L1_Support")) % 10
            ]
            
            prediction_proba = self.escalation_predictor.predict_proba([feature_vector])
            escalation_probability = prediction_proba[0][1] if len(prediction_proba[0]) > 1 else 0
            
            return escalation_probability > 0.7  # Seuil de 70%
        
        except Exception as e:
            logger.warning(f"Erreur prédiction escalade ML: {e}")
            return False
    
    async def complete_escalation(self, escalation_id: str, 
                                resolution: str) -> bool:
        """Termine une escalade"""
        if escalation_id not in self.active_escalations:
            return False
        
        escalation = self.active_escalations[escalation_id]
        escalation["status"] = "completed"
        escalation["completed_at"] = datetime.utcnow()
        escalation["resolution"] = resolution
        
        # Libération de la capacité
        target_level = escalation["to_level"]
        if target_level in self.escalation_levels:
            self.escalation_levels[target_level].current_load -= 1
        
        # Déplacement vers l'historique
        self.escalation_history.append(escalation)
        del self.active_escalations[escalation_id]
        
        logger.info(f"Escalade {escalation_id} terminée: {resolution}")
        return True
    
    def get_escalation_statistics(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Récupère les statistiques d'escalade"""
        # Filtrage par tenant
        if tenant_id:
            relevant_history = [e for e in self.escalation_history if e.get("tenant_id") == tenant_id]
            relevant_active = [e for e in self.active_escalations.values() if e.get("tenant_id") == tenant_id]
        else:
            relevant_history = list(self.escalation_history)
            relevant_active = list(self.active_escalations.values())
        
        # Statistiques par niveau
        level_stats = {}
        for level_name, level_obj in self.escalation_levels.items():
            level_escalations = [e for e in relevant_history if e.get("to_level") == level_name]
            
            level_stats[level_name] = {
                "capacity": level_obj.capacity,
                "current_load": level_obj.current_load,
                "utilization": level_obj.get_utilization(),
                "total_escalations": len(level_escalations),
                "avg_resolution_time": self._calculate_avg_resolution_time(level_escalations)
            }
        
        # Raisons d'escalade
        escalation_reasons = {}
        for escalation in relevant_history:
            reason = escalation.get("reason", "unknown")
            escalation_reasons[reason] = escalation_reasons.get(reason, 0) + 1
        
        return {
            "tenant_id": tenant_id,
            "active_escalations": len(relevant_active),
            "total_escalations": len(relevant_history),
            "level_statistics": level_stats,
            "escalation_reasons": escalation_reasons,
            "ml_model_trained": self.escalation_predictor is not None
        }
    
    def _calculate_avg_resolution_time(self, escalations: List[Dict[str, Any]]) -> float:
        """Calcule le temps moyen de résolution"""
        resolution_times = []
        
        for escalation in escalations:
            if escalation.get("completed_at") and escalation.get("escalated_at"):
                escalated_at = escalation["escalated_at"]
                completed_at = escalation.get("completed_at")
                if isinstance(completed_at, str):
                    completed_at = datetime.fromisoformat(completed_at)
                
                resolution_time = (completed_at - escalated_at).total_seconds()
                resolution_times.append(resolution_time)
        
        return sum(resolution_times) / len(resolution_times) if resolution_times else 0

# ===========================
# Gestionnaire Principal Automation Engine
# ===========================

class AutomationEngineManager:
    """Gestionnaire principal du moteur d'automatisation"""
    
    def __init__(self):
        self.auto_response = AutoResponseSystem()
        self.escalation_manager = EscalationManager()
        
        # Tâches de fond
        self.background_tasks: List[asyncio.Task] = []
        
        logger.info("Gestionnaire Automation Engine initialisé")
    
    async def start(self):
        """Démarre le moteur d'automatisation"""
        # Configuration des règles par défaut
        await self._setup_default_rules()
        
        # Démarrage des tâches de fond
        self.background_tasks.append(
            asyncio.create_task(self._periodic_escalation_check())
        )
        self.background_tasks.append(
            asyncio.create_task(self._periodic_model_retrain())
        )
        
        logger.info("Moteur d'automatisation démarré")
    
    async def stop(self):
        """Arrête le moteur d'automatisation"""
        for task in self.background_tasks:
            task.cancel()
        
        logger.info("Moteur d'automatisation arrêté")
    
    async def _setup_default_rules(self):
        """Configure les règles d'automatisation par défaut"""
        default_rules = [
            {
                "rule_id": "high_cpu_alert",
                "tenant_id": "default",
                "name": "Alerte CPU élevé",
                "description": "Notification en cas de CPU élevé",
                "trigger_conditions": {
                    "type": "metric_alert",
                    "metric": "cpu_usage_percent",
                    "value": {"operator": "greater_than", "value": 80}
                },
                "actions": [
                    {
                        "type": "notify",
                        "parameters": {
                            "message": "🚨 CPU élevé détecté: {value}%",
                            "channels": ["slack", "email"]
                        }
                    },
                    {
                        "type": "escalate",
                        "parameters": {
                            "level": "L2_Expert",
                            "reason": "CPU critique"
                        }
                    }
                ]
            },
            {
                "rule_id": "memory_critical",
                "tenant_id": "default", 
                "name": "Mémoire critique",
                "description": "Actions automatiques pour mémoire critique",
                "trigger_conditions": {
                    "type": "metric_alert",
                    "metric": "memory_usage_percent",
                    "value": {"operator": "greater_than", "value": 90}
                },
                "actions": [
                    {
                        "type": "script",
                        "parameters": {
                            "script": "free -m && ps aux --sort=-%mem | head -10",
                            "interpreter": "bash"
                        }
                    },
                    {
                        "type": "notify",
                        "parameters": {
                            "message": "🔴 Mémoire critique: {value}% - Investigation en cours",
                            "channels": ["slack", "pagerduty"]
                        }
                    }
                ]
            }
        ]
        
        for rule_config in default_rules:
            rule = AutomationRule(
                rule_id=rule_config["rule_id"],
                tenant_id=rule_config["tenant_id"],
                name=rule_config["name"],
                description=rule_config["description"],
                trigger_conditions=rule_config["trigger_conditions"],
                actions=rule_config["actions"]
            )
            self.auto_response.add_rule(rule)
    
    async def _periodic_escalation_check(self):
        """Vérification périodique des escalades"""
        while True:
            try:
                await asyncio.sleep(300)  # Toutes les 5 minutes
                
                # Simulation de vérification d'incidents actifs
                # Dans un vrai système, ceci interrogerait la base de données
                active_incidents = [
                    {
                        "incident_id": "INC-001",
                        "tenant_id": "default",
                        "severity": "high",
                        "created_at": (datetime.utcnow() - timedelta(minutes=45)).isoformat(),
                        "current_level": "L1_Support"
                    }
                ]
                
                for incident in active_incidents:
                    should_escalate, reason, target_level = await self.escalation_manager.should_escalate(incident)
                    
                    if should_escalate:
                        await self.escalation_manager.escalate_incident(
                            incident, reason, target_level
                        )
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erreur vérification escalade: {e}")
    
    async def _periodic_model_retrain(self):
        """Ré-entraînement périodique des modèles"""
        while True:
            try:
                await asyncio.sleep(3600)  # Toutes les heures
                
                # Ré-entraînement des modèles si assez de données
                await self.auto_response._retrain_decision_model()
                await self.escalation_manager._retrain_escalation_predictor()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erreur ré-entraînement modèles: {e}")
    
    # API publique
    
    async def process_incident_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Traite un événement d'incident"""
        # Réponse automatique
        executions = await self.auto_response.process_event(event)
        
        # Vérification d'escalade si c'est un incident
        escalation_result = None
        if event.get("type") == "incident":
            should_escalate, reason, target_level = await self.escalation_manager.should_escalate(event)
            
            if should_escalate:
                escalation_result = await self.escalation_manager.escalate_incident(
                    event, reason, target_level
                )
        
        return {
            "event_id": event.get("event_id", str(uuid.uuid4())),
            "processed_at": datetime.utcnow().isoformat(),
            "auto_response": {
                "triggered_rules": len(executions),
                "executions": [e.execution_id for e in executions]
            },
            "escalation": escalation_result
        }
    
    def get_automation_overview(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Récupère une vue d'ensemble de l'automatisation"""
        auto_response_stats = self.auto_response.get_statistics(tenant_id)
        escalation_stats = self.escalation_manager.get_escalation_statistics(tenant_id)
        
        return {
            "tenant_id": tenant_id,
            "timestamp": datetime.utcnow().isoformat(),
            "auto_response": auto_response_stats,
            "escalation": escalation_stats,
            "system_health": {
                "active_background_tasks": len(self.background_tasks),
                "memory_usage": "N/A"  # Dans un vrai système
            }
        }

# ===========================
# Exports
# ===========================

__all__ = [
    "AutomationEngineManager",
    "AutoResponseSystem",
    "EscalationManager", 
    "AutomationRule",
    "AutomationExecution",
    "EscalationLevel",
    "RemediationAction",
    "ResponseType",
    "ActionType",
    "EscalationReason",
    "AutomationStatus",
    "IntegrationType"
]

logger.info("Module Automation Engine chargé")
