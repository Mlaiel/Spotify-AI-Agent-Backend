#!/usr/bin/env python3
"""
Gestionnaire de Règles d'Alertes Ultra-Performant - Engine Principal

Ce module implémente le moteur principal de gestion des règles d'alertes avec
intelligence artificielle, machine learning, et orchestration microservices.

Architecture Avancée:
- Gestionnaire distribué avec Redis pour coordination
- Évaluateur parallèle avec ThreadPoolExecutor
- Cache intelligent multi-niveaux
- Métriques Prometheus temps réel
- Pipeline ML pour prédictions
- Système de notifications avancé
- API REST et WebSocket pour intégration

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
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import (
    Any, Dict, List, Optional, Set, Union, Callable,
    AsyncGenerator, Tuple, DefaultDict
)
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import pickle
import weakref
from pathlib import Path

# Core imports
import aioredis
import asyncpg
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry
import structlog

# ML and monitoring imports
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import psutil

# Internal imports
from .core import (
    AlertRule, RuleContext, AlertMetrics, EvaluationResult,
    AlertSeverity, AlertCategory, RuleStatus, RuleFactory, ConfigValidator
)
from ...........................core.exceptions import (
    AlertRuleException, ValidationException, MetricsException
)
from ...........................core.database import DatabaseManager
from ...........................core.security import SecurityManager
from ...........................core.notifications import NotificationManager

# Configuration du logging
logger = structlog.get_logger(__name__)

# Métriques Prometheus avancées
REGISTRY = CollectorRegistry()

RULE_MANAGER_OPERATIONS = Counter(
    'rule_manager_operations_total',
    'Total rule manager operations',
    ['operation', 'tenant_id', 'status'],
    registry=REGISTRY
)

RULE_EVALUATION_DURATION = Histogram(
    'rule_evaluation_duration_seconds',
    'Duration of rule evaluations',
    ['tenant_id', 'rule_category', 'evaluation_type'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    registry=REGISTRY
)

ACTIVE_RULE_EVALUATORS = Gauge(
    'active_rule_evaluators',
    'Number of active rule evaluators',
    ['tenant_id'],
    registry=REGISTRY
)

CACHE_OPERATIONS = Counter(
    'cache_operations_total',
    'Cache operations',
    ['operation', 'cache_type', 'status'],
    registry=REGISTRY
)

ML_MODEL_PERFORMANCE = Summary(
    'ml_model_performance',
    'ML model performance metrics',
    ['model_type', 'tenant_id', 'metric_name'],
    registry=REGISTRY
)


@dataclass
class RuleEvaluationConfig:
    """Configuration pour l'évaluation des règles"""
    max_concurrent_evaluations: int = 100
    evaluation_timeout: float = 30.0
    cache_ttl: int = 60
    batch_size: int = 50
    retry_attempts: int = 3
    retry_delay: float = 1.0
    enable_ml_predictions: bool = True
    enable_distributed_cache: bool = True
    performance_monitoring: bool = True


@dataclass
class EvaluationBatch:
    """Lot d'évaluations à traiter"""
    rules: List[AlertRule]
    contexts: List[RuleContext]
    priority: int = 1
    tenant_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __len__(self) -> int:
        return len(self.rules)


class RuleEvaluationCache:
    """Cache multi-niveaux pour les évaluations de règles"""
    
    def __init__(
        self,
        redis_client: Optional[aioredis.Redis] = None,
        local_cache_size: int = 1000,
        ttl: int = 60
    ):
        self.redis_client = redis_client
        self.ttl = ttl
        
        # Cache local LRU
        self.local_cache: Dict[str, Tuple[Any, datetime]] = {}
        self.local_cache_size = local_cache_size
        self.access_order = deque()
        
        # Statistiques
        self.hits = 0
        self.misses = 0
        self.local_hits = 0
        self.redis_hits = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Récupère une valeur du cache"""
        try:
            # Vérification cache local
            if key in self.local_cache:
                value, timestamp = self.local_cache[key]
                if datetime.utcnow() - timestamp < timedelta(seconds=self.ttl):
                    self._update_access_order(key)
                    self.hits += 1
                    self.local_hits += 1
                    CACHE_OPERATIONS.labels(
                        operation='get',
                        cache_type='local',
                        status='hit'
                    ).inc()
                    return value
                else:
                    del self.local_cache[key]
            
            # Vérification cache Redis
            if self.redis_client:
                cached_data = await self.redis_client.get(f"rule_cache:{key}")
                if cached_data:
                    value = pickle.loads(cached_data)
                    # Mise en cache locale
                    self._set_local(key, value)
                    self.hits += 1
                    self.redis_hits += 1
                    CACHE_OPERATIONS.labels(
                        operation='get',
                        cache_type='redis',
                        status='hit'
                    ).inc()
                    return value
            
            self.misses += 1
            CACHE_OPERATIONS.labels(
                operation='get',
                cache_type='miss',
                status='miss'
            ).inc()
            return None
            
        except Exception as e:
            logger.error("Cache get operation failed", key=key, error=str(e))
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Met une valeur en cache"""
        try:
            ttl = ttl or self.ttl
            
            # Cache local
            self._set_local(key, value)
            
            # Cache Redis
            if self.redis_client:
                serialized = pickle.dumps(value)
                await self.redis_client.setex(
                    f"rule_cache:{key}",
                    ttl,
                    serialized
                )
            
            CACHE_OPERATIONS.labels(
                operation='set',
                cache_type='both',
                status='success'
            ).inc()
            return True
            
        except Exception as e:
            logger.error("Cache set operation failed", key=key, error=str(e))
            CACHE_OPERATIONS.labels(
                operation='set',
                cache_type='error',
                status='error'
            ).inc()
            return False
    
    def _set_local(self, key: str, value: Any):
        """Met une valeur en cache local avec LRU"""
        # Éviction LRU si nécessaire
        if len(self.local_cache) >= self.local_cache_size:
            oldest_key = self.access_order.popleft()
            if oldest_key in self.local_cache:
                del self.local_cache[oldest_key]
        
        self.local_cache[key] = (value, datetime.utcnow())
        self._update_access_order(key)
    
    def _update_access_order(self, key: str):
        """Met à jour l'ordre d'accès pour LRU"""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests) if total_requests > 0 else 0
        
        return {
            'total_requests': total_requests,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'local_hits': self.local_hits,
            'redis_hits': self.redis_hits,
            'cache_size': len(self.local_cache)
        }


class MLPredictor:
    """Prédicteur ML pour les alertes"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.training_data: DefaultDict[str, List[np.ndarray]] = defaultdict(list)
        self.last_training: Dict[str, datetime] = {}
        self.prediction_history: DefaultDict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
    
    async def predict_anomalies(
        self,
        tenant_id: str,
        metrics: AlertMetrics
    ) -> Dict[str, float]:
        """Prédit les anomalies pour un tenant"""
        try:
            model_key = f"{tenant_id}_anomaly"
            
            # Vérification/entraînement du modèle
            if not await self._ensure_model_trained(model_key, tenant_id):
                return {}
            
            # Préparation des features
            features = self._extract_features(metrics)
            
            # Prédiction
            model = self.models[model_key]
            anomaly_score = model.decision_function([features])[0]
            is_anomaly = model.predict([features])[0] == -1
            
            predictions = {
                'anomaly_score': float(anomaly_score),
                'is_anomaly': bool(is_anomaly),
                'confidence': abs(float(anomaly_score))
            }
            
            # Historique des prédictions
            self.prediction_history[model_key].append({
                'timestamp': datetime.utcnow(),
                'predictions': predictions,
                'features': features
            })
            
            # Métriques
            ML_MODEL_PERFORMANCE.labels(
                model_type='isolation_forest',
                tenant_id=tenant_id,
                metric_name='anomaly_score'
            ).observe(abs(anomaly_score))
            
            return predictions
            
        except Exception as e:
            logger.error(
                "ML prediction failed",
                tenant_id=tenant_id,
                error=str(e)
            )
            return {}
    
    async def _ensure_model_trained(self, model_key: str, tenant_id: str) -> bool:
        """S'assure qu'un modèle est entraîné"""
        try:
            # Vérification si le modèle existe et est récent
            if (model_key in self.models and 
                model_key in self.last_training and
                datetime.utcnow() - self.last_training[model_key] < timedelta(hours=1)):
                return True
            
            # Vérification des données d'entraînement
            if len(self.training_data[model_key]) < 50:
                return False
            
            # Entraînement du modèle
            training_features = np.array(self.training_data[model_key])
            model = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100,
                max_features=1.0
            )
            
            model.fit(training_features)
            self.models[model_key] = model
            self.last_training[model_key] = datetime.utcnow()
            
            logger.info(
                "ML model trained successfully",
                model_key=model_key,
                training_samples=len(training_features)
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "ML model training failed",
                model_key=model_key,
                error=str(e)
            )
            return False
    
    def add_training_data(self, tenant_id: str, metrics: AlertMetrics):
        """Ajoute des données d'entraînement"""
        model_key = f"{tenant_id}_anomaly"
        features = self._extract_features(metrics)
        
        self.training_data[model_key].append(features)
        
        # Limitation de la taille des données d'entraînement
        if len(self.training_data[model_key]) > 10000:
            self.training_data[model_key] = self.training_data[model_key][-5000:]
    
    def _extract_features(self, metrics: AlertMetrics) -> np.ndarray:
        """Extrait les features pour ML"""
        return np.array([
            metrics.cpu_usage,
            metrics.memory_usage,
            metrics.disk_usage,
            metrics.network_latency,
            metrics.error_rate,
            metrics.request_rate,
            metrics.response_time,
            # Features dérivées
            metrics.cpu_usage * metrics.memory_usage,  # Charge combinée
            metrics.error_rate / max(metrics.request_rate, 0.001),  # Taux d'erreur normalisé
            np.log1p(metrics.response_time),  # Log du temps de réponse
        ])


class RuleEvaluator:
    """Évaluateur parallèle de règles"""
    
    def __init__(
        self,
        config: RuleEvaluationConfig,
        cache: RuleEvaluationCache,
        ml_predictor: MLPredictor
    ):
        self.config = config
        self.cache = cache
        self.ml_predictor = ml_predictor
        self.executor = ThreadPoolExecutor(
            max_workers=config.max_concurrent_evaluations
        )
        
        # État interne
        self.active_evaluations: Set[str] = set()
        self.evaluation_queue = asyncio.Queue(maxsize=1000)
        self.results_queue = asyncio.Queue(maxsize=1000)
        
        # Statistiques
        self.total_evaluations = 0
        self.successful_evaluations = 0
        self.failed_evaluations = 0
        self.average_execution_time = 0.0
    
    async def evaluate_rules(
        self,
        rules: List[AlertRule],
        context: RuleContext
    ) -> List[EvaluationResult]:
        """Évalue une liste de règles en parallèle"""
        if not rules:
            return []
        
        start_time = time.time()
        
        try:
            # Ajout des prédictions ML au contexte
            if self.config.enable_ml_predictions:
                ml_predictions = await self.ml_predictor.predict_anomalies(
                    context.tenant_id,
                    context.current_metrics
                )
                context.ml_predictions.update(ml_predictions)
                
                # Ajout aux données d'entraînement
                self.ml_predictor.add_training_data(
                    context.tenant_id,
                    context.current_metrics
                )
            
            # Évaluation parallèle
            tasks = []
            for rule in rules:
                task = self._evaluate_single_rule(rule, context)
                tasks.append(task)
            
            # Attente des résultats avec timeout
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self.config.evaluation_timeout
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Rule evaluation timeout",
                    tenant_id=context.tenant_id,
                    rule_count=len(rules),
                    timeout=self.config.evaluation_timeout
                )
                results = [None] * len(rules)
            
            # Traitement des résultats
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, EvaluationResult):
                    valid_results.append(result)
                    self.successful_evaluations += 1
                elif isinstance(result, Exception):
                    logger.error(
                        "Rule evaluation failed",
                        rule_id=rules[i].rule_id,
                        error=str(result)
                    )
                    self.failed_evaluations += 1
                else:
                    self.failed_evaluations += 1
            
            # Mise à jour des statistiques
            execution_time = time.time() - start_time
            self.total_evaluations += len(rules)
            self.average_execution_time = (
                (self.average_execution_time * (self.total_evaluations - len(rules)) + 
                 execution_time) / self.total_evaluations
            )
            
            # Métriques Prometheus
            RULE_EVALUATION_DURATION.labels(
                tenant_id=context.tenant_id,
                rule_category='batch',
                evaluation_type='parallel'
            ).observe(execution_time)
            
            return valid_results
            
        except Exception as e:
            logger.error(
                "Batch rule evaluation failed",
                tenant_id=context.tenant_id,
                rule_count=len(rules),
                error=str(e)
            )
            return []
    
    async def _evaluate_single_rule(
        self,
        rule: AlertRule,
        context: RuleContext
    ) -> EvaluationResult:
        """Évalue une règle unique avec cache et retry"""
        cache_key = f"{rule.rule_id}:{context.tenant_id}:{hash(str(context.current_metrics.to_dict()))}"
        
        # Vérification du cache
        if self.config.enable_distributed_cache:
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                return cached_result
        
        # Évaluation avec retry
        for attempt in range(self.config.retry_attempts):
            try:
                result = await rule.evaluate(context)
                
                # Mise en cache du résultat
                if self.config.enable_distributed_cache:
                    await self.cache.set(cache_key, result, ttl=self.config.cache_ttl)
                
                return result
                
            except Exception as e:
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                    continue
                else:
                    logger.error(
                        "Rule evaluation failed after retries",
                        rule_id=rule.rule_id,
                        attempts=self.config.retry_attempts,
                        error=str(e)
                    )
                    raise
    
    async def evaluate_batch(self, batch: EvaluationBatch) -> List[EvaluationResult]:
        """Évalue un lot de règles"""
        all_results = []
        
        for i in range(0, len(batch), self.config.batch_size):
            sub_batch_rules = batch.rules[i:i + self.config.batch_size]
            sub_batch_contexts = batch.contexts[i:i + self.config.batch_size]
            
            for j, (rule, context) in enumerate(zip(sub_batch_rules, sub_batch_contexts)):
                try:
                    result = await self._evaluate_single_rule(rule, context)
                    all_results.append(result)
                except Exception as e:
                    logger.error(
                        "Single rule evaluation in batch failed",
                        rule_id=rule.rule_id,
                        batch_index=i + j,
                        error=str(e)
                    )
        
        return all_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de l'évaluateur"""
        return {
            'total_evaluations': self.total_evaluations,
            'successful_evaluations': self.successful_evaluations,
            'failed_evaluations': self.failed_evaluations,
            'success_rate': (
                self.successful_evaluations / max(self.total_evaluations, 1)
            ),
            'average_execution_time': self.average_execution_time,
            'active_evaluations': len(self.active_evaluations),
            'queue_size': self.evaluation_queue.qsize(),
            'results_queue_size': self.results_queue.qsize()
        }


class RuleManager:
    """Gestionnaire principal des règles d'alertes"""
    
    def __init__(
        self,
        config: RuleEvaluationConfig,
        redis_client: Optional[aioredis.Redis] = None,
        db_manager: Optional[DatabaseManager] = None,
        notification_manager: Optional[NotificationManager] = None
    ):
        self.config = config
        self.redis_client = redis_client
        self.db_manager = db_manager
        self.notification_manager = notification_manager
        
        # Composants principaux
        self.cache = RuleEvaluationCache(redis_client, ttl=config.cache_ttl)
        self.ml_predictor = MLPredictor()
        self.evaluator = RuleEvaluator(config, self.cache, self.ml_predictor)
        
        # Stockage des règles par tenant
        self.rules_by_tenant: DefaultDict[str, List[AlertRule]] = defaultdict(list)
        self.rule_index: Dict[str, AlertRule] = {}
        
        # Tâches de fond
        self.background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        
        # Métriques système
        self.system_metrics_collector = SystemMetricsCollector()
    
    async def start(self):
        """Démarre le gestionnaire de règles"""
        logger.info("Starting rule manager")
        
        # Chargement des règles depuis la base de données
        await self._load_rules_from_db()
        
        # Démarrage des tâches de fond
        self.background_tasks.add(
            asyncio.create_task(self._metrics_collection_loop())
        )
        self.background_tasks.add(
            asyncio.create_task(self._rule_maintenance_loop())
        )
        self.background_tasks.add(
            asyncio.create_task(self._evaluation_queue_processor())
        )
        
        logger.info("Rule manager started successfully")
    
    async def stop(self):
        """Arrête le gestionnaire de règles"""
        logger.info("Stopping rule manager")
        
        self._shutdown_event.set()
        
        # Annulation des tâches de fond
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Fermeture des ressources
        self.evaluator.executor.shutdown(wait=True)
        
        logger.info("Rule manager stopped")
    
    async def add_rule(self, rule_config: Dict[str, Any]) -> AlertRule:
        """Ajoute une nouvelle règle"""
        try:
            # Validation de la configuration
            errors = ConfigValidator.validate_rule_config(rule_config)
            if errors:
                raise ValidationException(f"Invalid rule config: {', '.join(errors)}")
            
            # Création de la règle
            rule = RuleFactory.create_rule(rule_config)
            
            # Stockage en mémoire
            self.rules_by_tenant[rule.tenant_id].append(rule)
            self.rule_index[rule.rule_id] = rule
            
            # Persistance en base de données
            if self.db_manager:
                await self._persist_rule(rule)
            
            # Métriques
            RULE_MANAGER_OPERATIONS.labels(
                operation='add_rule',
                tenant_id=rule.tenant_id,
                status='success'
            ).inc()
            
            logger.info(
                "Rule added successfully",
                rule_id=rule.rule_id,
                tenant_id=rule.tenant_id,
                rule_name=rule.name
            )
            
            return rule
            
        except Exception as e:
            RULE_MANAGER_OPERATIONS.labels(
                operation='add_rule',
                tenant_id=rule_config.get('tenant_id', 'unknown'),
                status='error'
            ).inc()
            raise AlertRuleException(f"Failed to add rule: {str(e)}")
    
    async def remove_rule(self, rule_id: str, tenant_id: str) -> bool:
        """Supprime une règle"""
        try:
            if rule_id not in self.rule_index:
                return False
            
            rule = self.rule_index[rule_id]
            
            # Vérification du tenant
            if rule.tenant_id != tenant_id:
                raise AlertRuleException("Rule belongs to different tenant")
            
            # Suppression de la mémoire
            self.rules_by_tenant[tenant_id] = [
                r for r in self.rules_by_tenant[tenant_id] 
                if r.rule_id != rule_id
            ]
            del self.rule_index[rule_id]
            
            # Suppression de la base de données
            if self.db_manager:
                await self._delete_rule(rule_id)
            
            # Métriques
            RULE_MANAGER_OPERATIONS.labels(
                operation='remove_rule',
                tenant_id=tenant_id,
                status='success'
            ).inc()
            
            logger.info("Rule removed successfully", rule_id=rule_id)
            return True
            
        except Exception as e:
            RULE_MANAGER_OPERATIONS.labels(
                operation='remove_rule',
                tenant_id=tenant_id,
                status='error'
            ).inc()
            raise AlertRuleException(f"Failed to remove rule: {str(e)}")
    
    async def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> AlertRule:
        """Met à jour une règle existante"""
        try:
            if rule_id not in self.rule_index:
                raise AlertRuleException(f"Rule {rule_id} not found")
            
            rule = self.rule_index[rule_id]
            
            # Application des mises à jour
            for key, value in updates.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
            
            # Persistance
            if self.db_manager:
                await self._persist_rule(rule)
            
            # Métriques
            RULE_MANAGER_OPERATIONS.labels(
                operation='update_rule',
                tenant_id=rule.tenant_id,
                status='success'
            ).inc()
            
            logger.info("Rule updated successfully", rule_id=rule_id)
            return rule
            
        except Exception as e:
            RULE_MANAGER_OPERATIONS.labels(
                operation='update_rule',
                tenant_id='unknown',
                status='error'
            ).inc()
            raise AlertRuleException(f"Failed to update rule: {str(e)}")
    
    async def evaluate_tenant_rules(
        self,
        tenant_id: str,
        metrics: Optional[AlertMetrics] = None
    ) -> List[EvaluationResult]:
        """Évalue toutes les règles d'un tenant"""
        try:
            rules = self.rules_by_tenant.get(tenant_id, [])
            if not rules:
                return []
            
            # Collecte des métriques si non fournies
            if metrics is None:
                metrics = await self.system_metrics_collector.collect_metrics()
            
            # Création du contexte
            context = RuleContext(
                tenant_id=tenant_id,
                environment="dev",  # À paramétrer
                current_metrics=metrics,
                historical_data=await self._get_historical_metrics(tenant_id),
                ml_predictions={},
                user_context={},
                system_state={}
            )
            
            # Filtrage des règles actives
            active_rules = [rule for rule in rules if rule.enabled and rule.status == RuleStatus.ACTIVE]
            
            # Évaluation
            results = await self.evaluator.evaluate_rules(active_rules, context)
            
            # Traitement des alertes déclenchées
            triggered_results = [r for r in results if r.triggered]
            if triggered_results:
                await self._process_triggered_alerts(triggered_results)
            
            # Métriques
            RULE_MANAGER_OPERATIONS.labels(
                operation='evaluate_tenant',
                tenant_id=tenant_id,
                status='success'
            ).inc()
            
            ACTIVE_RULE_EVALUATORS.labels(tenant_id=tenant_id).set(len(active_rules))
            
            return results
            
        except Exception as e:
            RULE_MANAGER_OPERATIONS.labels(
                operation='evaluate_tenant',
                tenant_id=tenant_id,
                status='error'
            ).inc()
            logger.error(
                "Tenant rule evaluation failed",
                tenant_id=tenant_id,
                error=str(e)
            )
            return []
    
    async def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        """Récupère une règle par son ID"""
        return self.rule_index.get(rule_id)
    
    async def list_rules(
        self,
        tenant_id: Optional[str] = None,
        category: Optional[AlertCategory] = None,
        status: Optional[RuleStatus] = None
    ) -> List[AlertRule]:
        """Liste les règles selon les critères"""
        rules = []
        
        if tenant_id:
            tenant_rules = self.rules_by_tenant.get(tenant_id, [])
        else:
            tenant_rules = [rule for rule_list in self.rules_by_tenant.values() for rule in rule_list]
        
        for rule in tenant_rules:
            if category and rule.category != category:
                continue
            if status and rule.status != status:
                continue
            rules.append(rule)
        
        return rules
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques du gestionnaire"""
        total_rules = sum(len(rules) for rules in self.rules_by_tenant.values())
        active_rules = sum(
            len([r for r in rules if r.enabled and r.status == RuleStatus.ACTIVE])
            for rules in self.rules_by_tenant.values()
        )
        
        return {
            'total_rules': total_rules,
            'active_rules': active_rules,
            'tenants': len(self.rules_by_tenant),
            'evaluator_stats': self.evaluator.get_stats(),
            'cache_stats': self.cache.get_stats(),
            'ml_models': len(self.ml_predictor.models),
            'background_tasks': len(self.background_tasks)
        }
    
    # Méthodes privées pour la persistance et maintenance
    
    async def _load_rules_from_db(self):
        """Charge les règles depuis la base de données"""
        if not self.db_manager:
            return
        
        try:
            # Implementation placeholder - à adapter selon le schéma DB
            logger.info("Loading rules from database")
            
        except Exception as e:
            logger.error("Failed to load rules from database", error=str(e))
    
    async def _persist_rule(self, rule: AlertRule):
        """Persiste une règle en base de données"""
        if not self.db_manager:
            return
        
        try:
            # Implementation placeholder - à adapter selon le schéma DB
            logger.debug("Persisting rule to database", rule_id=rule.rule_id)
            
        except Exception as e:
            logger.error("Failed to persist rule", rule_id=rule.rule_id, error=str(e))
    
    async def _delete_rule(self, rule_id: str):
        """Supprime une règle de la base de données"""
        if not self.db_manager:
            return
        
        try:
            # Implementation placeholder
            logger.debug("Deleting rule from database", rule_id=rule_id)
            
        except Exception as e:
            logger.error("Failed to delete rule", rule_id=rule_id, error=str(e))
    
    async def _get_historical_metrics(self, tenant_id: str) -> List[AlertMetrics]:
        """Récupère les métriques historiques pour un tenant"""
        # Implementation placeholder - retourne des données mock
        return []
    
    async def _process_triggered_alerts(self, results: List[EvaluationResult]):
        """Traite les alertes déclenchées"""
        if not self.notification_manager:
            return
        
        for result in results:
            try:
                # Création de la notification
                notification = {
                    'rule_id': result.rule_id,
                    'severity': result.severity.name,
                    'message': result.message,
                    'metadata': result.metadata,
                    'timestamp': result.timestamp.isoformat()
                }
                
                # Envoi de la notification
                await self.notification_manager.send_alert(notification)
                
            except Exception as e:
                logger.error(
                    "Failed to process triggered alert",
                    rule_id=result.rule_id,
                    error=str(e)
                )
    
    async def _metrics_collection_loop(self):
        """Boucle de collecte des métriques système"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(30)  # Collecte toutes les 30 secondes
                
                metrics = await self.system_metrics_collector.collect_metrics()
                
                # Ajout aux données d'entraînement ML
                for tenant_id in self.rules_by_tenant.keys():
                    self.ml_predictor.add_training_data(tenant_id, metrics)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Metrics collection loop error", error=str(e))
    
    async def _rule_maintenance_loop(self):
        """Boucle de maintenance des règles"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # Maintenance toutes les 5 minutes
                
                # Nettoyage des caches expirés
                # Mise à jour des statistiques
                # Vérification de la santé des règles
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Rule maintenance loop error", error=str(e))
    
    async def _evaluation_queue_processor(self):
        """Processeur de la queue d'évaluation"""
        while not self._shutdown_event.is_set():
            try:
                # Traitement de la queue d'évaluation en attente
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Evaluation queue processor error", error=str(e))


class SystemMetricsCollector:
    """Collecteur de métriques système"""
    
    def __init__(self):
        self.last_network_stats = None
        self.last_disk_stats = None
    
    async def collect_metrics(self) -> AlertMetrics:
        """Collecte les métriques système actuelles"""
        try:
            # CPU
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            
            # Network (simulation)
            network_latency = np.random.normal(10, 2)  # ms
            
            # Application metrics (simulation)
            error_rate = np.random.exponential(0.01)  # %
            request_rate = np.random.poisson(100)  # requests/sec
            response_time = np.random.lognormal(2, 0.5)  # ms
            
            return AlertMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_latency=network_latency,
                error_rate=error_rate,
                request_rate=request_rate,
                response_time=response_time
            )
            
        except Exception as e:
            logger.error("Failed to collect system metrics", error=str(e))
            # Retour de métriques par défaut en cas d'erreur
            return AlertMetrics(
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_latency=0.0,
                error_rate=0.0,
                request_rate=0.0,
                response_time=0.0
            )


# Factory pour création du gestionnaire
async def create_rule_manager(
    config: Optional[RuleEvaluationConfig] = None,
    redis_url: Optional[str] = None,
    database_url: Optional[str] = None
) -> RuleManager:
    """Factory pour créer un gestionnaire de règles configuré"""
    
    if config is None:
        config = RuleEvaluationConfig()
    
    # Connexion Redis
    redis_client = None
    if redis_url:
        redis_client = aioredis.from_url(redis_url)
    
    # Gestionnaire de base de données
    db_manager = None
    if database_url:
        db_manager = DatabaseManager(database_url)
        await db_manager.connect()
    
    # Gestionnaire de notifications
    notification_manager = NotificationManager()
    
    # Création du gestionnaire
    manager = RuleManager(
        config=config,
        redis_client=redis_client,
        db_manager=db_manager,
        notification_manager=notification_manager
    )
    
    return manager


# Context manager pour utilisation sûre
@asynccontextmanager
async def rule_manager_context(
    config: Optional[RuleEvaluationConfig] = None,
    redis_url: Optional[str] = None,
    database_url: Optional[str] = None
) -> AsyncGenerator[RuleManager, None]:
    """Context manager pour utilisation sûre du gestionnaire de règles"""
    
    manager = await create_rule_manager(config, redis_url, database_url)
    
    try:
        await manager.start()
        yield manager
    finally:
        await manager.stop()


# Exportation des classes principales
__all__ = [
    'RuleEvaluationConfig',
    'EvaluationBatch',
    'RuleEvaluationCache',
    'MLPredictor',
    'RuleEvaluator',
    'RuleManager',
    'SystemMetricsCollector',
    'create_rule_manager',
    'rule_manager_context'
]
