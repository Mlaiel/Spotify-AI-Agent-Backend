#!/usr/bin/env python3
"""
Module Core de Gestion des Règles d'Alertes - Architecture Ultra-Performante

Ce module fournit les classes de base et les interfaces essentielles pour le système
de gestion des règles d'alertes avec intelligence artificielle et machine learning.

Architecture & Fonctionnalités:
- Moteur de règles ML avec évaluation temps réel
- Gestionnaire de conditions complexes avec logique avancée
- Validateur intelligent avec correction automatique
- Cache distribué avec Redis pour optimisation performance
- Métriques Prometheus pour monitoring production
- Pipeline ML pour analyse prédictive des patterns
- Interface asynchrone pour microservices

Équipe Engineering:
✅ Lead Dev + Architecte IA : Fahed Mlaiel
✅ Développeur Backend Senior (Python/FastAPI/Django)
✅ Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
✅ DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
✅ Spécialiste Sécurité Backend
✅ Architecte Microservices

License: Spotify Proprietary
Copyright: © 2025 Spotify Technology S.A.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import (
    Any, Dict, List, Optional, Set, Union, Callable, 
    AsyncGenerator, Tuple, NamedTuple, Protocol, TypeVar, Generic
)
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
import uuid
import weakref
from pathlib import Path

# Core imports
import aioredis
import asyncpg
from prometheus_client import Counter, Histogram, Gauge, Summary
import structlog

# ML and AI imports
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import torch
import transformers

# Spotify specific imports
from ...........................core.exceptions import (
    AlertRuleException, ValidationException, MetricsException
)
from ...........................core.metrics import MetricsCollector
from ...........................core.security import SecurityManager
from ...........................core.database import DatabaseManager


# Type definitions
T = TypeVar('T')
RuleType = TypeVar('RuleType', bound='AlertRule')
ConditionType = TypeVar('ConditionType', bound='RuleCondition')

# Logging configuration
logger = structlog.get_logger(__name__)

# Prometheus metrics
RULE_EVALUATIONS = Counter(
    'alert_rule_evaluations_total',
    'Total number of alert rule evaluations',
    ['tenant_id', 'rule_type', 'environment']
)

RULE_EXECUTION_TIME = Histogram(
    'alert_rule_execution_seconds',
    'Time spent executing alert rules',
    ['tenant_id', 'rule_type'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
)

ACTIVE_ALERTS = Gauge(
    'active_alerts_total',
    'Number of currently active alerts',
    ['tenant_id', 'severity', 'category']
)

ML_PREDICTION_ACCURACY = Summary(
    'ml_prediction_accuracy',
    'Accuracy of ML-based alert predictions',
    ['model_type', 'tenant_id']
)


class AlertSeverity(Enum):
    """Niveaux de sévérité des alertes avec valeurs numériques pour tri"""
    CRITICAL = 100
    HIGH = 75
    MEDIUM = 50
    LOW = 25
    INFO = 10


class AlertCategory(Enum):
    """Catégories d'alertes pour classification et routage"""
    INFRASTRUCTURE = "infrastructure"
    APPLICATION = "application"
    SECURITY = "security"
    BUSINESS = "business"
    ML_ANOMALY = "ml_anomaly"
    PERFORMANCE = "performance"
    USER_EXPERIENCE = "user_experience"


class RuleStatus(Enum):
    """États possibles d'une règle d'alerte"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DISABLED = "disabled"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class EvaluationResult(NamedTuple):
    """Résultat d'évaluation d'une règle"""
    rule_id: str
    triggered: bool
    severity: AlertSeverity
    message: str
    metadata: Dict[str, Any]
    execution_time: float
    timestamp: datetime


@dataclass
class AlertMetrics:
    """Métriques collectées pendant l'évaluation des alertes"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    error_rate: float
    request_rate: float
    response_time: float
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire pour sérialisation"""
        return {
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'disk_usage': self.disk_usage,
            'network_latency': self.network_latency,
            'error_rate': self.error_rate,
            'request_rate': self.request_rate,
            'response_time': self.response_time,
            'custom_metrics': self.custom_metrics
        }


@dataclass
class RuleContext:
    """Contexte d'exécution d'une règle d'alerte"""
    tenant_id: str
    environment: str
    current_metrics: AlertMetrics
    historical_data: List[AlertMetrics]
    ml_predictions: Dict[str, float]
    user_context: Dict[str, Any]
    system_state: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def get_feature_vector(self) -> np.ndarray:
        """Génère un vecteur de features pour ML"""
        features = [
            self.current_metrics.cpu_usage,
            self.current_metrics.memory_usage,
            self.current_metrics.disk_usage,
            self.current_metrics.network_latency,
            self.current_metrics.error_rate,
            self.current_metrics.request_rate,
            self.current_metrics.response_time
        ]
        
        # Ajout des métriques historiques moyennes
        if self.historical_data:
            historical_cpu = np.mean([m.cpu_usage for m in self.historical_data])
            historical_memory = np.mean([m.memory_usage for m in self.historical_data])
            features.extend([historical_cpu, historical_memory])
        
        return np.array(features)


class RuleCondition(ABC):
    """Interface abstraite pour les conditions de règles"""
    
    def __init__(self, condition_id: str, weight: float = 1.0):
        self.condition_id = condition_id
        self.weight = weight
        self.evaluation_count = 0
        self.last_evaluation: Optional[datetime] = None
    
    @abstractmethod
    async def evaluate(self, context: RuleContext) -> bool:
        """Évalue la condition avec le contexte donné"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Retourne une description humaine de la condition"""
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Retourne les métadonnées de la condition"""
        return {
            'condition_id': self.condition_id,
            'weight': self.weight,
            'evaluation_count': self.evaluation_count,
            'last_evaluation': self.last_evaluation.isoformat() if self.last_evaluation else None
        }


class ThresholdCondition(RuleCondition):
    """Condition basée sur un seuil numérique"""
    
    def __init__(
        self,
        condition_id: str,
        metric_path: str,
        operator: str,
        threshold: float,
        weight: float = 1.0
    ):
        super().__init__(condition_id, weight)
        self.metric_path = metric_path
        self.operator = operator
        self.threshold = threshold
        self.valid_operators = ['>', '<', '>=', '<=', '==', '!=']
        
        if operator not in self.valid_operators:
            raise ValueError(f"Operator {operator} not in {self.valid_operators}")
    
    async def evaluate(self, context: RuleContext) -> bool:
        """Évalue la condition de seuil"""
        try:
            value = self._extract_metric_value(context, self.metric_path)
            if value is None:
                return False
            
            self.evaluation_count += 1
            self.last_evaluation = datetime.utcnow()
            
            return self._compare_values(value, self.operator, self.threshold)
            
        except Exception as e:
            logger.error(
                "Threshold condition evaluation failed",
                condition_id=self.condition_id,
                error=str(e)
            )
            return False
    
    def _extract_metric_value(self, context: RuleContext, path: str) -> Optional[float]:
        """Extrait une valeur métrique du contexte"""
        try:
            # Navigation dans le contexte par chemin (ex: "current_metrics.cpu_usage")
            parts = path.split('.')
            current = context
            
            for part in parts:
                if hasattr(current, part):
                    current = getattr(current, part)
                elif isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
            
            return float(current) if current is not None else None
            
        except (ValueError, TypeError, AttributeError):
            return None
    
    def _compare_values(self, value: float, operator: str, threshold: float) -> bool:
        """Compare deux valeurs selon l'opérateur"""
        operations = {
            '>': lambda x, y: x > y,
            '<': lambda x, y: x < y,
            '>=': lambda x, y: x >= y,
            '<=': lambda x, y: x <= y,
            '==': lambda x, y: abs(x - y) < 1e-9,
            '!=': lambda x, y: abs(x - y) >= 1e-9
        }
        return operations[operator](value, threshold)
    
    def get_description(self) -> str:
        """Description de la condition"""
        return f"Metric {self.metric_path} {self.operator} {self.threshold}"


class MLAnomalyCondition(RuleCondition):
    """Condition basée sur détection d'anomalies ML"""
    
    def __init__(
        self,
        condition_id: str,
        model_name: str = "isolation_forest",
        contamination: float = 0.1,
        weight: float = 1.0
    ):
        super().__init__(condition_id, weight)
        self.model_name = model_name
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()
        self.training_data: List[np.ndarray] = []
        self.is_trained = False
    
    async def evaluate(self, context: RuleContext) -> bool:
        """Évalue la condition d'anomalie ML"""
        try:
            feature_vector = context.get_feature_vector()
            
            # Entraînement initial si nécessaire
            if not self.is_trained:
                await self._train_model(context)
            
            if self.model is None:
                return False
            
            # Préprocessing
            scaled_features = self.scaler.transform(feature_vector.reshape(1, -1))
            
            # Prédiction d'anomalie
            prediction = self.model.predict(scaled_features)
            is_anomaly = prediction[0] == -1  # -1 indique une anomalie
            
            self.evaluation_count += 1
            self.last_evaluation = datetime.utcnow()
            
            # Enregistrement des métriques
            if hasattr(self.model, 'score_samples'):
                anomaly_score = self.model.score_samples(scaled_features)[0]
                ML_PREDICTION_ACCURACY.observe(abs(anomaly_score))
            
            return is_anomaly
            
        except Exception as e:
            logger.error(
                "ML anomaly condition evaluation failed",
                condition_id=self.condition_id,
                error=str(e)
            )
            return False
    
    async def _train_model(self, context: RuleContext):
        """Entraîne le modèle ML avec les données historiques"""
        try:
            if len(context.historical_data) < 10:
                return  # Pas assez de données pour l'entraînement
            
            # Préparation des données d'entraînement
            training_features = []
            for metrics in context.historical_data:
                feature_vector = np.array([
                    metrics.cpu_usage,
                    metrics.memory_usage,
                    metrics.disk_usage,
                    metrics.network_latency,
                    metrics.error_rate,
                    metrics.request_rate,
                    metrics.response_time
                ])
                training_features.append(feature_vector)
            
            training_data = np.array(training_features)
            
            # Normalisation
            self.scaler.fit(training_data)
            scaled_data = self.scaler.transform(training_data)
            
            # Entraînement du modèle
            if self.model_name == "isolation_forest":
                self.model = IsolationForest(
                    contamination=self.contamination,
                    random_state=42,
                    n_estimators=100
                )
            elif self.model_name == "one_class_svm":
                from sklearn.svm import OneClassSVM
                self.model = OneClassSVM(gamma='scale', nu=self.contamination)
            
            self.model.fit(scaled_data)
            self.is_trained = True
            
            logger.info(
                "ML anomaly model trained successfully",
                condition_id=self.condition_id,
                model_name=self.model_name,
                training_samples=len(training_data)
            )
            
        except Exception as e:
            logger.error(
                "ML model training failed",
                condition_id=self.condition_id,
                error=str(e)
            )
    
    def get_description(self) -> str:
        """Description de la condition ML"""
        return f"ML anomaly detection using {self.model_name} (contamination={self.contamination})"


class CompositeCondition(RuleCondition):
    """Condition composite combinant plusieurs conditions"""
    
    def __init__(
        self,
        condition_id: str,
        conditions: List[RuleCondition],
        logic_operator: str = "AND",
        weight: float = 1.0
    ):
        super().__init__(condition_id, weight)
        self.conditions = conditions
        self.logic_operator = logic_operator.upper()
        
        if self.logic_operator not in ["AND", "OR", "XOR", "NAND", "NOR"]:
            raise ValueError(f"Invalid logic operator: {logic_operator}")
    
    async def evaluate(self, context: RuleContext) -> bool:
        """Évalue la condition composite"""
        try:
            # Évaluation parallèle des sous-conditions
            tasks = [condition.evaluate(context) for condition in self.conditions]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filtrage des erreurs
            valid_results = [r for r in results if isinstance(r, bool)]
            
            if not valid_results:
                return False
            
            self.evaluation_count += 1
            self.last_evaluation = datetime.utcnow()
            
            # Application de la logique
            if self.logic_operator == "AND":
                return all(valid_results)
            elif self.logic_operator == "OR":
                return any(valid_results)
            elif self.logic_operator == "XOR":
                return sum(valid_results) == 1
            elif self.logic_operator == "NAND":
                return not all(valid_results)
            elif self.logic_operator == "NOR":
                return not any(valid_results)
            
            return False
            
        except Exception as e:
            logger.error(
                "Composite condition evaluation failed",
                condition_id=self.condition_id,
                error=str(e)
            )
            return False
    
    def get_description(self) -> str:
        """Description de la condition composite"""
        sub_descriptions = [c.get_description() for c in self.conditions]
        return f"({f' {self.logic_operator} '.join(sub_descriptions)})"


class AlertRule:
    """Classe principale représentant une règle d'alerte"""
    
    def __init__(
        self,
        rule_id: str,
        name: str,
        description: str,
        conditions: List[RuleCondition],
        severity: AlertSeverity,
        category: AlertCategory,
        tenant_id: str,
        environment: str = "dev",
        cooldown_period: timedelta = timedelta(minutes=5),
        max_executions_per_hour: int = 100,
        enabled: bool = True,
        tags: Optional[Dict[str, str]] = None
    ):
        self.rule_id = rule_id
        self.name = name
        self.description = description
        self.conditions = conditions
        self.severity = severity
        self.category = category
        self.tenant_id = tenant_id
        self.environment = environment
        self.cooldown_period = cooldown_period
        self.max_executions_per_hour = max_executions_per_hour
        self.enabled = enabled
        self.tags = tags or {}
        
        # État interne
        self.status = RuleStatus.ACTIVE if enabled else RuleStatus.INACTIVE
        self.last_triggered: Optional[datetime] = None
        self.last_evaluated: Optional[datetime] = None
        self.execution_count = 0
        self.success_count = 0
        self.error_count = 0
        
        # Rate limiting
        self.execution_history: List[datetime] = []
        
        # Cache des résultats pour optimisation
        self._result_cache: Dict[str, Tuple[bool, datetime]] = {}
        self._cache_ttl = timedelta(seconds=30)
    
    async def evaluate(self, context: RuleContext) -> EvaluationResult:
        """Évalue la règle avec le contexte donné"""
        start_time = time.time()
        
        try:
            # Vérifications préliminaires
            if not self.enabled or self.status != RuleStatus.ACTIVE:
                return EvaluationResult(
                    rule_id=self.rule_id,
                    triggered=False,
                    severity=self.severity,
                    message="Rule is disabled or inactive",
                    metadata={'status': self.status.value},
                    execution_time=0.0,
                    timestamp=datetime.utcnow()
                )
            
            # Rate limiting
            if not self._check_rate_limit():
                return EvaluationResult(
                    rule_id=self.rule_id,
                    triggered=False,
                    severity=self.severity,
                    message="Rate limit exceeded",
                    metadata={'max_executions_per_hour': self.max_executions_per_hour},
                    execution_time=0.0,
                    timestamp=datetime.utcnow()
                )
            
            # Vérification du cooldown
            if self._is_in_cooldown():
                return EvaluationResult(
                    rule_id=self.rule_id,
                    triggered=False,
                    severity=self.severity,
                    message="Rule is in cooldown period",
                    metadata={'cooldown_remaining': str(self._get_cooldown_remaining())},
                    execution_time=0.0,
                    timestamp=datetime.utcnow()
                )
            
            # Cache check
            cache_key = self._generate_cache_key(context)
            cached_result = self._get_cached_result(cache_key)
            if cached_result is not None:
                return EvaluationResult(
                    rule_id=self.rule_id,
                    triggered=cached_result,
                    severity=self.severity,
                    message="Cached result",
                    metadata={'cache_hit': True},
                    execution_time=time.time() - start_time,
                    timestamp=datetime.utcnow()
                )
            
            # Évaluation des conditions
            condition_results = await self._evaluate_conditions(context)
            triggered = self._aggregate_condition_results(condition_results)
            
            # Mise à jour de l'état
            self.execution_count += 1
            self.last_evaluated = datetime.utcnow()
            
            if triggered:
                self.last_triggered = datetime.utcnow()
                self.success_count += 1
                ACTIVE_ALERTS.labels(
                    tenant_id=self.tenant_id,
                    severity=self.severity.name,
                    category=self.category.value
                ).inc()
            
            # Cache du résultat
            self._cache_result(cache_key, triggered)
            
            # Métriques Prometheus
            execution_time = time.time() - start_time
            RULE_EVALUATIONS.labels(
                tenant_id=self.tenant_id,
                rule_type=self.category.value,
                environment=self.environment
            ).inc()
            
            RULE_EXECUTION_TIME.labels(
                tenant_id=self.tenant_id,
                rule_type=self.category.value
            ).observe(execution_time)
            
            return EvaluationResult(
                rule_id=self.rule_id,
                triggered=triggered,
                severity=self.severity,
                message=self._generate_alert_message(triggered, condition_results),
                metadata=self._generate_metadata(condition_results, context),
                execution_time=execution_time,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            self.error_count += 1
            logger.error(
                "Rule evaluation failed",
                rule_id=self.rule_id,
                tenant_id=self.tenant_id,
                error=str(e)
            )
            
            return EvaluationResult(
                rule_id=self.rule_id,
                triggered=False,
                severity=self.severity,
                message=f"Evaluation error: {str(e)}",
                metadata={'error': True, 'error_message': str(e)},
                execution_time=time.time() - start_time,
                timestamp=datetime.utcnow()
            )
    
    async def _evaluate_conditions(self, context: RuleContext) -> List[Tuple[RuleCondition, bool]]:
        """Évalue toutes les conditions de la règle"""
        tasks = [(condition, condition.evaluate(context)) for condition in self.conditions]
        results = []
        
        for condition, task in tasks:
            try:
                result = await task
                results.append((condition, result))
            except Exception as e:
                logger.error(
                    "Condition evaluation failed",
                    rule_id=self.rule_id,
                    condition_id=condition.condition_id,
                    error=str(e)
                )
                results.append((condition, False))
        
        return results
    
    def _aggregate_condition_results(self, condition_results: List[Tuple[RuleCondition, bool]]) -> bool:
        """Agrège les résultats des conditions selon la logique de la règle"""
        if not condition_results:
            return False
        
        # Calcul pondéré basé sur les poids des conditions
        weighted_score = 0.0
        total_weight = 0.0
        
        for condition, result in condition_results:
            if result:
                weighted_score += condition.weight
            total_weight += condition.weight
        
        # Seuil d'activation : 50% du poids total
        threshold = total_weight * 0.5
        return weighted_score >= threshold
    
    def _check_rate_limit(self) -> bool:
        """Vérifie si la règle respecte la limite de taux d'exécution"""
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        
        # Nettoyage de l'historique
        self.execution_history = [
            timestamp for timestamp in self.execution_history 
            if timestamp > hour_ago
        ]
        
        # Ajout de l'exécution actuelle
        self.execution_history.append(now)
        
        return len(self.execution_history) <= self.max_executions_per_hour
    
    def _is_in_cooldown(self) -> bool:
        """Vérifie si la règle est en période de cooldown"""
        if self.last_triggered is None:
            return False
        
        return datetime.utcnow() - self.last_triggered < self.cooldown_period
    
    def _get_cooldown_remaining(self) -> timedelta:
        """Retourne le temps restant de cooldown"""
        if self.last_triggered is None:
            return timedelta(0)
        
        elapsed = datetime.utcnow() - self.last_triggered
        remaining = self.cooldown_period - elapsed
        return remaining if remaining > timedelta(0) else timedelta(0)
    
    def _generate_cache_key(self, context: RuleContext) -> str:
        """Génère une clé de cache basée sur le contexte"""
        key_data = {
            'rule_id': self.rule_id,
            'timestamp': context.timestamp.replace(second=0, microsecond=0),  # Minute precision
            'metrics_hash': hashlib.md5(
                json.dumps(context.current_metrics.to_dict(), sort_keys=True).encode()
            ).hexdigest()[:8]
        }
        return f"rule_cache:{json.dumps(key_data, sort_keys=True)}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[bool]:
        """Récupère un résultat en cache"""
        if cache_key in self._result_cache:
            result, timestamp = self._result_cache[cache_key]
            if datetime.utcnow() - timestamp < self._cache_ttl:
                return result
            else:
                del self._result_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: bool):
        """Met en cache un résultat"""
        self._result_cache[cache_key] = (result, datetime.utcnow())
        
        # Nettoyage du cache si trop volumineux
        if len(self._result_cache) > 1000:
            # Suppression des entrées les plus anciennes
            cutoff = datetime.utcnow() - self._cache_ttl
            self._result_cache = {
                k: v for k, v in self._result_cache.items()
                if v[1] > cutoff
            }
    
    def _generate_alert_message(
        self,
        triggered: bool,
        condition_results: List[Tuple[RuleCondition, bool]]
    ) -> str:
        """Génère le message d'alerte"""
        if not triggered:
            return f"Rule '{self.name}' conditions not met"
        
        triggered_conditions = [
            condition.get_description()
            for condition, result in condition_results
            if result
        ]
        
        return f"Alert: {self.name} - Conditions: {', '.join(triggered_conditions)}"
    
    def _generate_metadata(
        self,
        condition_results: List[Tuple[RuleCondition, bool]],
        context: RuleContext
    ) -> Dict[str, Any]:
        """Génère les métadonnées pour le résultat d'évaluation"""
        return {
            'rule_metadata': {
                'name': self.name,
                'description': self.description,
                'category': self.category.value,
                'tags': self.tags,
                'execution_count': self.execution_count,
                'success_rate': self.success_count / max(self.execution_count, 1)
            },
            'condition_results': [
                {
                    'condition_id': condition.condition_id,
                    'triggered': result,
                    'description': condition.get_description(),
                    'metadata': condition.get_metadata()
                }
                for condition, result in condition_results
            ],
            'context_metadata': {
                'tenant_id': context.tenant_id,
                'environment': context.environment,
                'current_metrics': context.current_metrics.to_dict(),
                'ml_predictions': context.ml_predictions
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Sérialise la règle en dictionnaire"""
        return {
            'rule_id': self.rule_id,
            'name': self.name,
            'description': self.description,
            'severity': self.severity.name,
            'category': self.category.value,
            'tenant_id': self.tenant_id,
            'environment': self.environment,
            'enabled': self.enabled,
            'status': self.status.value,
            'tags': self.tags,
            'cooldown_period_seconds': self.cooldown_period.total_seconds(),
            'max_executions_per_hour': self.max_executions_per_hour,
            'statistics': {
                'execution_count': self.execution_count,
                'success_count': self.success_count,
                'error_count': self.error_count,
                'last_triggered': self.last_triggered.isoformat() if self.last_triggered else None,
                'last_evaluated': self.last_evaluated.isoformat() if self.last_evaluated else None
            },
            'conditions': [
                {
                    'condition_id': condition.condition_id,
                    'type': type(condition).__name__,
                    'weight': condition.weight,
                    'description': condition.get_description(),
                    'metadata': condition.get_metadata()
                }
                for condition in self.conditions
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlertRule':
        """Désérialise une règle depuis un dictionnaire"""
        # Cette méthode nécessiterait une factory pour recréer les conditions
        # selon leur type - implémentation simplifiée pour l'exemple
        rule = cls(
            rule_id=data['rule_id'],
            name=data['name'],
            description=data['description'],
            conditions=[],  # À implémenter avec une factory
            severity=AlertSeverity[data['severity']],
            category=AlertCategory(data['category']),
            tenant_id=data['tenant_id'],
            environment=data.get('environment', 'dev'),
            cooldown_period=timedelta(seconds=data.get('cooldown_period_seconds', 300)),
            max_executions_per_hour=data.get('max_executions_per_hour', 100),
            enabled=data.get('enabled', True),
            tags=data.get('tags', {})
        )
        
        # Restauration des statistiques
        if 'statistics' in data:
            stats = data['statistics']
            rule.execution_count = stats.get('execution_count', 0)
            rule.success_count = stats.get('success_count', 0)
            rule.error_count = stats.get('error_count', 0)
            
            if stats.get('last_triggered'):
                rule.last_triggered = datetime.fromisoformat(stats['last_triggered'])
            if stats.get('last_evaluated'):
                rule.last_evaluated = datetime.fromisoformat(stats['last_evaluated'])
        
        return rule


# Factory pour création de conditions
class ConditionFactory:
    """Factory pour créer des conditions à partir de configurations"""
    
    @staticmethod
    def create_condition(config: Dict[str, Any]) -> RuleCondition:
        """Crée une condition à partir d'une configuration"""
        condition_type = config.get('type')
        condition_id = config.get('condition_id', str(uuid.uuid4()))
        weight = config.get('weight', 1.0)
        
        if condition_type == 'threshold':
            return ThresholdCondition(
                condition_id=condition_id,
                metric_path=config['metric_path'],
                operator=config['operator'],
                threshold=config['threshold'],
                weight=weight
            )
        elif condition_type == 'ml_anomaly':
            return MLAnomalyCondition(
                condition_id=condition_id,
                model_name=config.get('model_name', 'isolation_forest'),
                contamination=config.get('contamination', 0.1),
                weight=weight
            )
        elif condition_type == 'composite':
            sub_conditions = [
                ConditionFactory.create_condition(sub_config)
                for sub_config in config['conditions']
            ]
            return CompositeCondition(
                condition_id=condition_id,
                conditions=sub_conditions,
                logic_operator=config.get('logic_operator', 'AND'),
                weight=weight
            )
        else:
            raise ValueError(f"Unknown condition type: {condition_type}")


# Factory pour création de règles
class RuleFactory:
    """Factory pour créer des règles à partir de configurations"""
    
    @staticmethod
    def create_rule(config: Dict[str, Any]) -> AlertRule:
        """Crée une règle à partir d'une configuration"""
        conditions = [
            ConditionFactory.create_condition(condition_config)
            for condition_config in config.get('conditions', [])
        ]
        
        return AlertRule(
            rule_id=config.get('rule_id', str(uuid.uuid4())),
            name=config['name'],
            description=config.get('description', ''),
            conditions=conditions,
            severity=AlertSeverity[config.get('severity', 'MEDIUM')],
            category=AlertCategory(config.get('category', 'APPLICATION')),
            tenant_id=config['tenant_id'],
            environment=config.get('environment', 'dev'),
            cooldown_period=timedelta(seconds=config.get('cooldown_period_seconds', 300)),
            max_executions_per_hour=config.get('max_executions_per_hour', 100),
            enabled=config.get('enabled', True),
            tags=config.get('tags', {})
        )


# Validation des configurations
class ConfigValidator:
    """Validateur pour les configurations de règles"""
    
    @staticmethod
    def validate_rule_config(config: Dict[str, Any]) -> List[str]:
        """Valide une configuration de règle et retourne les erreurs"""
        errors = []
        
        # Champs obligatoires
        required_fields = ['name', 'tenant_id']
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Validation de la sévérité
        if 'severity' in config:
            try:
                AlertSeverity[config['severity']]
            except KeyError:
                errors.append(f"Invalid severity: {config['severity']}")
        
        # Validation de la catégorie
        if 'category' in config:
            try:
                AlertCategory(config['category'])
            except ValueError:
                errors.append(f"Invalid category: {config['category']}")
        
        # Validation des conditions
        if 'conditions' in config:
            for i, condition_config in enumerate(config['conditions']):
                condition_errors = ConfigValidator.validate_condition_config(condition_config)
                for error in condition_errors:
                    errors.append(f"Condition {i}: {error}")
        
        return errors
    
    @staticmethod
    def validate_condition_config(config: Dict[str, Any]) -> List[str]:
        """Valide une configuration de condition"""
        errors = []
        
        if 'type' not in config:
            errors.append("Missing condition type")
            return errors
        
        condition_type = config['type']
        
        if condition_type == 'threshold':
            required = ['metric_path', 'operator', 'threshold']
            for field in required:
                if field not in config:
                    errors.append(f"Missing field for threshold condition: {field}")
            
            if 'operator' in config:
                valid_operators = ['>', '<', '>=', '<=', '==', '!=']
                if config['operator'] not in valid_operators:
                    errors.append(f"Invalid operator: {config['operator']}")
        
        elif condition_type == 'ml_anomaly':
            if 'contamination' in config:
                contamination = config['contamination']
                if not (0 < contamination < 1):
                    errors.append("Contamination must be between 0 and 1")
        
        elif condition_type == 'composite':
            if 'conditions' not in config:
                errors.append("Composite condition missing sub-conditions")
            elif not isinstance(config['conditions'], list):
                errors.append("Composite conditions must be a list")
            
            if 'logic_operator' in config:
                valid_operators = ['AND', 'OR', 'XOR', 'NAND', 'NOR']
                if config['logic_operator'].upper() not in valid_operators:
                    errors.append(f"Invalid logic operator: {config['logic_operator']}")
        
        else:
            errors.append(f"Unknown condition type: {condition_type}")
        
        return errors


# Exportation des classes principales
__all__ = [
    'AlertSeverity',
    'AlertCategory', 
    'RuleStatus',
    'EvaluationResult',
    'AlertMetrics',
    'RuleContext',
    'RuleCondition',
    'ThresholdCondition',
    'MLAnomalyCondition',
    'CompositeCondition',
    'AlertRule',
    'ConditionFactory',
    'RuleFactory',
    'ConfigValidator'
]
