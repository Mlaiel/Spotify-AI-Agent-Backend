"""
Module d'automatisation avancé pour Alertmanager Receivers

Ce module implémente l'intelligence artificielle et l'automatisation pour
la gestion dynamique des receivers, l'auto-healing et l'optimisation continue.

Author: Spotify AI Agent Team  
Maintainer: Fahed Mlaiel - Lead Dev + Architecte IA
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import aiofiles
import pickle

logger = logging.getLogger(__name__)

class AutomationLevel(Enum):
    """Niveaux d'automatisation disponibles"""
    MANUAL = "manual"
    SEMI_AUTO = "semi_auto"
    FULL_AUTO = "full_auto"
    AI_DRIVEN = "ai_driven"

class ActionType(Enum):
    """Types d'actions automatiques"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    RESTART_SERVICE = "restart_service"
    SWITCH_RECEIVER = "switch_receiver"
    UPDATE_CONFIG = "update_config"
    SEND_ALERT = "send_alert"
    EXECUTE_RUNBOOK = "execute_runbook"
    FAILOVER = "failover"

class TriggerCondition(Enum):
    """Conditions de déclenchement"""
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    ERROR_RATE_HIGH = "error_rate_high"
    LATENCY_HIGH = "latency_high"
    CAPACITY_LOW = "capacity_low"
    PATTERN_DETECTED = "pattern_detected"
    ANOMALY_DETECTED = "anomaly_detected"

@dataclass
class AutomationRule:
    """Règle d'automatisation"""
    name: str
    description: str
    tenant: str
    enabled: bool = True
    priority: int = 1
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    cooldown_minutes: int = 15
    max_executions_per_hour: int = 5
    requires_approval: bool = False
    success_rate_threshold: float = 0.9
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_executed: Optional[datetime] = None
    execution_count: int = 0

@dataclass
class AutomationMetrics:
    """Métriques d'automatisation"""
    total_rules: int = 0
    active_rules: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_execution_time: float = 0.0
    success_rate: float = 0.0
    cost_savings: float = 0.0
    incidents_prevented: int = 0

class MLPredictor:
    """Prédicteur basé sur l'apprentissage automatique"""
    
    def __init__(self):
        self.models = {}
        self.training_data = {}
        self.predictions_cache = {}
        
    async def train_anomaly_detection(self, tenant: str, historical_data: List[Dict]):
        """Entraîne un modèle de détection d'anomalies"""
        try:
            logger.info(f"Training anomaly detection model for {tenant}")
            
            # Préparation des données
            features = self._extract_features(historical_data)
            
            # Entraînement du modèle (simulation - en prod utiliser scikit-learn, TensorFlow)
            model = self._train_isolation_forest(features)
            self.models[f"{tenant}_anomaly"] = model
            
            logger.info(f"Anomaly detection model trained for {tenant}")
            
        except Exception as e:
            logger.error(f"Failed to train model for {tenant}: {e}")
    
    def _extract_features(self, data: List[Dict]) -> np.ndarray:
        """Extrait les features des données historiques"""
        features = []
        for record in data:
            feature_vector = [
                record.get('cpu_usage', 0),
                record.get('memory_usage', 0),
                record.get('request_rate', 0),
                record.get('error_rate', 0),
                record.get('response_time', 0)
            ]
            features.append(feature_vector)
        return np.array(features)
    
    def _train_isolation_forest(self, features: np.ndarray):
        """Entraîne un modèle Isolation Forest (simulation)"""
        # En production: utiliser sklearn.ensemble.IsolationForest
        return {"type": "isolation_forest", "features_shape": features.shape}
    
    async def predict_anomaly(self, tenant: str, current_metrics: Dict) -> Tuple[bool, float]:
        """Prédit si les métriques actuelles sont anormales"""
        model_key = f"{tenant}_anomaly"
        if model_key not in self.models:
            return False, 0.0
        
        # Simulation de prédiction
        feature_vector = [
            current_metrics.get('cpu_usage', 0),
            current_metrics.get('memory_usage', 0),
            current_metrics.get('request_rate', 0),
            current_metrics.get('error_rate', 0),
            current_metrics.get('response_time', 0)
        ]
        
        # Score d'anomalie simulé
        anomaly_score = np.random.random()
        is_anomaly = anomaly_score > 0.7
        
        return is_anomaly, anomaly_score
    
    async def predict_capacity_needs(self, tenant: str, time_horizon: int = 24) -> Dict:
        """Prédit les besoins en capacité"""
        # Simulation de prédiction de capacité
        return {
            "predicted_cpu_usage": 75.5,
            "predicted_memory_usage": 68.2,
            "predicted_request_rate": 1250,
            "confidence": 0.85,
            "time_horizon_hours": time_horizon
        }

class AutomationConfigManager:
    """Gestionnaire principal de l'automatisation"""
    
    def __init__(self):
        self.rules: Dict[str, AutomationRule] = {}
        self.ml_predictor = MLPredictor()
        self.metrics = AutomationMetrics()
        self.execution_history: List[Dict] = []
        self.active_automations: Dict[str, asyncio.Task] = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    async def initialize_automation(self) -> bool:
        """Initialise le système d'automatisation"""
        try:
            logger.info("Initializing automation configuration manager")
            
            # Chargement des règles par défaut
            await self._load_default_rules()
            
            # Initialisation des modèles ML
            await self._initialize_ml_models()
            
            # Démarrage des tâches de monitoring
            await self._start_monitoring_tasks()
            
            logger.info("Automation system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize automation: {e}")
            return False
    
    async def _load_default_rules(self):
        """Charge les règles d'automatisation par défaut"""
        
        # Règle d'auto-scaling pour Premium
        premium_scaling_rule = AutomationRule(
            name="premium-auto-scaling",
            description="Auto-scaling intelligent pour les services Premium",
            tenant="spotify-premium",
            conditions=[
                {
                    "type": TriggerCondition.THRESHOLD_EXCEEDED.value,
                    "metric": "cpu_usage",
                    "threshold": 80,
                    "duration_minutes": 5
                },
                {
                    "type": TriggerCondition.THRESHOLD_EXCEEDED.value,
                    "metric": "memory_usage", 
                    "threshold": 85,
                    "duration_minutes": 3
                }
            ],
            actions=[
                {
                    "type": ActionType.SCALE_UP.value,
                    "target": "alertmanager_receivers",
                    "scale_factor": 1.5,
                    "max_instances": 10
                },
                {
                    "type": ActionType.SEND_ALERT.value,
                    "channel": "slack",
                    "message": "Auto-scaling triggered for Premium services"
                }
            ],
            cooldown_minutes=10,
            max_executions_per_hour=6
        )
        
        # Règle de détection d'anomalies
        anomaly_detection_rule = AutomationRule(
            name="ml-anomaly-detection",
            description="Détection d'anomalies basée sur l'IA",
            tenant="all",
            conditions=[
                {
                    "type": TriggerCondition.ANOMALY_DETECTED.value,
                    "model": "isolation_forest",
                    "confidence_threshold": 0.8
                }
            ],
            actions=[
                {
                    "type": ActionType.EXECUTE_RUNBOOK.value,
                    "runbook": "investigate_anomaly",
                    "auto_execute": False
                },
                {
                    "type": ActionType.SEND_ALERT.value,
                    "channel": "pagerduty",
                    "severity": "high"
                }
            ],
            requires_approval=True,
            priority=1
        )
        
        # Règle de failover automatique
        failover_rule = AutomationRule(
            name="automatic-failover",
            description="Failover automatique en cas de panne critique",
            tenant="spotify-premium",
            conditions=[
                {
                    "type": TriggerCondition.ERROR_RATE_HIGH.value,
                    "threshold": 10,  # 10% d'erreurs
                    "duration_minutes": 2
                }
            ],
            actions=[
                {
                    "type": ActionType.FAILOVER.value,
                    "primary_region": "us-east-1",
                    "backup_region": "us-west-2",
                    "traffic_percentage": 100
                }
            ],
            cooldown_minutes=30,
            max_executions_per_hour=2,
            priority=1
        )
        
        self.rules = {
            premium_scaling_rule.name: premium_scaling_rule,
            anomaly_detection_rule.name: anomaly_detection_rule,
            failover_rule.name: failover_rule
        }
    
    async def _initialize_ml_models(self):
        """Initialise les modèles d'apprentissage automatique"""
        try:
            # Simulation de données historiques
            historical_data = [
                {"cpu_usage": 45, "memory_usage": 60, "request_rate": 1000, "error_rate": 1, "response_time": 150},
                {"cpu_usage": 65, "memory_usage": 70, "request_rate": 1500, "error_rate": 2, "response_time": 200},
                {"cpu_usage": 55, "memory_usage": 65, "request_rate": 1200, "error_rate": 1.5, "response_time": 175}
            ]
            
            # Entraînement pour chaque tenant
            for tenant in ["spotify-premium", "spotify-free", "spotify-family"]:
                await self.ml_predictor.train_anomaly_detection(tenant, historical_data)
            
        except Exception as e:
            logger.error(f"Failed to initialize ML models: {e}")
    
    async def _start_monitoring_tasks(self):
        """Démarre les tâches de monitoring continu"""
        # Tâche de monitoring des métriques
        self.active_automations["metrics_monitor"] = asyncio.create_task(
            self._monitor_metrics_continuously()
        )
        
        # Tâche de nettoyage des données
        self.active_automations["cleanup_task"] = asyncio.create_task(
            self._cleanup_old_data()
        )
        
        # Tâche d'optimisation des règles
        self.active_automations["rule_optimizer"] = asyncio.create_task(
            self._optimize_rules_continuously()
        )
    
    async def _monitor_metrics_continuously(self):
        """Monitore les métriques en continu"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Récupération des métriques actuelles (simulation)
                current_metrics = await self._get_current_metrics()
                
                # Évaluation des règles
                await self._evaluate_rules(current_metrics)
                
            except Exception as e:
                logger.error(f"Error in metrics monitoring: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _get_current_metrics(self) -> Dict:
        """Récupère les métriques actuelles (simulation)"""
        return {
            "cpu_usage": np.random.uniform(30, 90),
            "memory_usage": np.random.uniform(40, 85),
            "request_rate": np.random.uniform(800, 2000),
            "error_rate": np.random.uniform(0.1, 5),
            "response_time": np.random.uniform(100, 500)
        }
    
    async def _evaluate_rules(self, metrics: Dict):
        """Évalue toutes les règles actives"""
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
                
            try:
                # Vérification des conditions
                conditions_met = await self._check_conditions(rule, metrics)
                
                if conditions_met:
                    # Vérification du cooldown
                    if await self._check_cooldown(rule):
                        await self._execute_rule(rule, metrics)
                
            except Exception as e:
                logger.error(f"Error evaluating rule {rule_name}: {e}")
    
    async def _check_conditions(self, rule: AutomationRule, metrics: Dict) -> bool:
        """Vérifie si les conditions d'une règle sont remplies"""
        for condition in rule.conditions:
            condition_type = TriggerCondition(condition["type"])
            
            if condition_type == TriggerCondition.THRESHOLD_EXCEEDED:
                metric_name = condition["metric"]
                threshold = condition["threshold"]
                current_value = metrics.get(metric_name, 0)
                
                if current_value < threshold:
                    return False
                    
            elif condition_type == TriggerCondition.ANOMALY_DETECTED:
                is_anomaly, score = await self.ml_predictor.predict_anomaly(
                    rule.tenant, metrics
                )
                confidence_threshold = condition.get("confidence_threshold", 0.8)
                
                if not is_anomaly or score < confidence_threshold:
                    return False
        
        return True
    
    async def _check_cooldown(self, rule: AutomationRule) -> bool:
        """Vérifie si le cooldown est respecté"""
        if rule.last_executed is None:
            return True
        
        cooldown_period = timedelta(minutes=rule.cooldown_minutes)
        return datetime.utcnow() - rule.last_executed > cooldown_period
    
    async def _execute_rule(self, rule: AutomationRule, metrics: Dict):
        """Exécute les actions d'une règle"""
        try:
            logger.info(f"Executing automation rule: {rule.name}")
            
            # Vérification du nombre max d'exécutions
            if rule.execution_count >= rule.max_executions_per_hour:
                logger.warning(f"Rule {rule.name} reached max executions per hour")
                return
            
            # Exécution des actions
            for action in rule.actions:
                await self._execute_action(action, rule.tenant, metrics)
            
            # Mise à jour des métriques de la règle
            rule.last_executed = datetime.utcnow()
            rule.execution_count += 1
            
            # Enregistrement dans l'historique
            execution_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "rule_name": rule.name,
                "tenant": rule.tenant,
                "metrics": metrics,
                "actions_executed": len(rule.actions),
                "success": True
            }
            self.execution_history.append(execution_record)
            
            # Mise à jour des métriques globales
            self.metrics.successful_executions += 1
            
        except Exception as e:
            logger.error(f"Failed to execute rule {rule.name}: {e}")
            self.metrics.failed_executions += 1
    
    async def _execute_action(self, action: Dict, tenant: str, metrics: Dict):
        """Exécute une action spécifique"""
        action_type = ActionType(action["type"])
        
        if action_type == ActionType.SCALE_UP:
            await self._scale_up_service(action, tenant)
        elif action_type == ActionType.SEND_ALERT:
            await self._send_automated_alert(action, tenant, metrics)
        elif action_type == ActionType.FAILOVER:
            await self._execute_failover(action, tenant)
        elif action_type == ActionType.EXECUTE_RUNBOOK:
            await self._execute_runbook(action, tenant)
        # Ajouter d'autres types d'actions...
    
    async def _scale_up_service(self, action: Dict, tenant: str):
        """Effectue le scaling d'un service"""
        logger.info(f"Scaling up service {action.get('target')} for {tenant}")
        # Implementation du scaling (K8s, Docker Swarm, etc.)
    
    async def _send_automated_alert(self, action: Dict, tenant: str, metrics: Dict):
        """Envoie une alerte automatique"""
        channel = action.get("channel", "slack")
        message = action.get("message", f"Automation triggered for {tenant}")
        
        logger.info(f"Sending automated alert to {channel}: {message}")
        # Implementation de l'envoi d'alerte
    
    async def _execute_failover(self, action: Dict, tenant: str):
        """Exécute un failover automatique"""
        primary = action.get("primary_region")
        backup = action.get("backup_region")
        
        logger.info(f"Executing failover from {primary} to {backup} for {tenant}")
        # Implementation du failover
    
    async def _execute_runbook(self, action: Dict, tenant: str):
        """Exécute un runbook automatiquement"""
        runbook = action.get("runbook")
        auto_execute = action.get("auto_execute", False)
        
        if auto_execute:
            logger.info(f"Auto-executing runbook {runbook} for {tenant}")
        else:
            logger.info(f"Runbook {runbook} requires manual approval for {tenant}")
    
    async def _cleanup_old_data(self):
        """Nettoie les anciennes données"""
        while True:
            try:
                await asyncio.sleep(3600)  # Cleanup every hour
                
                # Nettoyage de l'historique (garder 7 jours)
                cutoff_date = datetime.utcnow() - timedelta(days=7)
                self.execution_history = [
                    record for record in self.execution_history
                    if datetime.fromisoformat(record["timestamp"]) > cutoff_date
                ]
                
                logger.info("Completed cleanup of old automation data")
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
    
    async def _optimize_rules_continuously(self):
        """Optimise les règles en continu"""
        while True:
            try:
                await asyncio.sleep(86400)  # Optimize daily
                
                # Analyse des performances des règles
                await self._analyze_rule_performance()
                
                # Ajustement automatique des seuils
                await self._auto_tune_thresholds()
                
                logger.info("Completed rule optimization cycle")
                
            except Exception as e:
                logger.error(f"Error in rule optimization: {e}")
    
    async def _analyze_rule_performance(self):
        """Analyse les performances des règles"""
        for rule_name, rule in self.rules.items():
            # Calcul du taux de succès
            recent_executions = [
                record for record in self.execution_history
                if record["rule_name"] == rule_name
            ]
            
            if recent_executions:
                success_rate = sum(1 for r in recent_executions if r["success"]) / len(recent_executions)
                logger.info(f"Rule {rule_name} success rate: {success_rate:.2%}")
    
    async def _auto_tune_thresholds(self):
        """Ajuste automatiquement les seuils basés sur l'historique"""
        # Implementation de l'auto-tuning basé sur l'historique des métriques
        logger.info("Auto-tuning rule thresholds based on historical data")
    
    def add_automation_rule(self, rule: AutomationRule) -> bool:
        """Ajoute une nouvelle règle d'automatisation"""
        try:
            self.rules[rule.name] = rule
            self.metrics.total_rules += 1
            if rule.enabled:
                self.metrics.active_rules += 1
            
            logger.info(f"Added automation rule: {rule.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add rule {rule.name}: {e}")
            return False
    
    def remove_automation_rule(self, rule_name: str) -> bool:
        """Supprime une règle d'automatisation"""
        if rule_name in self.rules:
            if self.rules[rule_name].enabled:
                self.metrics.active_rules -= 1
            del self.rules[rule_name]
            self.metrics.total_rules -= 1
            return True
        return False
    
    def get_automation_metrics(self) -> AutomationMetrics:
        """Récupère les métriques d'automatisation"""
        # Calcul du taux de succès
        total_executions = self.metrics.successful_executions + self.metrics.failed_executions
        if total_executions > 0:
            self.metrics.success_rate = self.metrics.successful_executions / total_executions
        
        return self.metrics
    
    async def shutdown(self):
        """Arrête proprement le système d'automatisation"""
        logger.info("Shutting down automation system")
        
        # Arrêt des tâches actives
        for task_name, task in self.active_automations.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Fermeture de l'executor
        self.executor.shutdown(wait=True)
        
        logger.info("Automation system shutdown completed")

# Instance singleton
automation_manager = AutomationConfigManager()
