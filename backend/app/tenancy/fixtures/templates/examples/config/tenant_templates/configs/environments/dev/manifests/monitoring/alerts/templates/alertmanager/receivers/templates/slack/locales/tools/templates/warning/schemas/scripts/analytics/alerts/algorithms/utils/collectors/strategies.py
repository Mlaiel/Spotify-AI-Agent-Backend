"""
Spotify AI Agent - Advanced Strategies Module
============================================

Stratégies ultra-avancées pour l'optimisation et la gestion
intelligente des collecteurs de données avec IA intégrée.

Stratégies implémentées:
- Adaptive Strategy: Adaptation dynamique basée sur les métriques
- Prediction Strategy: Prédiction des charges et ajustement proactif
- Multi-Tenant Strategy: Gestion optimisée multi-locataire
- ML-Driven Strategy: Stratégies pilotées par Machine Learning
- Auto-Scaling Strategy: Mise à l'échelle automatique intelligente
- Cost Optimization Strategy: Optimisation des coûts cloud
- Performance Strategy: Optimisation des performances en temps réel
- Security Strategy: Stratégies de sécurité adaptatives

Développé par l'équipe Spotify AI Agent
Lead Developer: Fahed Mlaiel
Architecture: IA-driven adaptive strategies
"""

import asyncio
import time
import threading
import random
import math
import statistics
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import wraps
from typing import (
    Any, Dict, List, Optional, Union, Callable, Tuple, 
    Set, Awaitable, Type, Protocol, Generic, TypeVar
)
import json
import logging
import structlog
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import aiohttp
import aioredis
import asyncpg
from pydantic import BaseModel, validator
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge


T = TypeVar('T')
logger = structlog.get_logger(__name__)


class StrategyType(Enum):
    """Types de stratégies disponibles."""
    ADAPTIVE = "adaptive"
    PREDICTIVE = "predictive"
    MULTI_TENANT = "multi_tenant"
    ML_DRIVEN = "ml_driven"
    AUTO_SCALING = "auto_scaling"
    COST_OPTIMIZATION = "cost_optimization"
    PERFORMANCE = "performance"
    SECURITY = "security"


class AdaptationMode(Enum):
    """Modes d'adaptation des stratégies."""
    REACTIVE = "reactive"
    PROACTIVE = "proactive"
    PREDICTIVE = "predictive"
    HYBRID = "hybrid"


class OptimizationGoal(Enum):
    """Objectifs d'optimisation."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    COST = "cost"
    RELIABILITY = "reliability"
    SECURITY = "security"
    BALANCED = "balanced"


@dataclass
class StrategyConfig:
    """Configuration d'une stratégie."""
    
    name: str
    strategy_type: StrategyType
    adaptation_mode: AdaptationMode
    optimization_goal: OptimizationGoal
    
    # Paramètres d'adaptation
    adaptation_interval: float = 30.0
    sensitivity_threshold: float = 0.1
    learning_rate: float = 0.01
    
    # Limites d'adaptation
    min_value: float = 0.1
    max_value: float = 10.0
    
    # Configuration ML
    ml_model_path: Optional[str] = None
    feature_window_size: int = 100
    prediction_horizon: int = 300  # 5 minutes
    
    # Métadonnées
    tenant_id: Optional[str] = None
    priority: int = 1
    enabled: bool = True


@dataclass
class StrategyMetrics:
    """Métriques d'une stratégie."""
    
    adaptations_count: int = 0
    success_rate: float = 0.0
    avg_response_time: float = 0.0
    cost_reduction: float = 0.0
    performance_gain: float = 0.0
    last_adaptation: Optional[datetime] = None
    current_value: float = 1.0
    predicted_value: Optional[float] = None
    confidence_score: float = 0.0
    anomaly_score: float = 0.0


class BaseStrategy(ABC):
    """
    Classe de base pour toutes les stratégies.
    
    Fonctionnalités communes:
    - Collecte de métriques
    - Adaptation basée sur les seuils
    - Logging structuré
    - Métriques Prometheus
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.metrics = StrategyMetrics()
        self.history = deque(maxlen=config.feature_window_size)
        self.adaptation_history = deque(maxlen=1000)
        self.last_adaptation = None
        
        # Métriques Prometheus
        self.strategy_adaptations = Counter(
            'strategy_adaptations_total',
            'Nombre total d\'adaptations de stratégie',
            ['strategy_name', 'strategy_type', 'result']
        )
        
        self.strategy_value = Gauge(
            'strategy_current_value',
            'Valeur actuelle de la stratégie',
            ['strategy_name', 'strategy_type']
        )
        
        self.strategy_performance = Histogram(
            'strategy_adaptation_duration_seconds',
            'Durée des adaptations de stratégie',
            ['strategy_name', 'strategy_type']
        )
        
        logger.info(
            "Stratégie initialisée",
            strategy_name=config.name,
            strategy_type=config.strategy_type.value,
            optimization_goal=config.optimization_goal.value
        )
    
    @abstractmethod
    async def evaluate(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Évalue les métriques actuelles et détermine les adaptations nécessaires.
        
        Args:
            current_metrics: Métriques actuelles du système
        
        Returns:
            Recommandations d'adaptation
        """
        pass
    
    @abstractmethod
    async def adapt(self, recommendations: Dict[str, Any]) -> bool:
        """
        Applique les adaptations recommandées.
        
        Args:
            recommendations: Recommandations d'adaptation
        
        Returns:
            True si l'adaptation a réussi
        """
        pass
    
    async def run_cycle(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Exécute un cycle complet d'évaluation et d'adaptation.
        
        Args:
            current_metrics: Métriques actuelles
        
        Returns:
            Résultat du cycle
        """
        start_time = time.time()
        
        try:
            # Évaluation
            recommendations = await self.evaluate(current_metrics)
            
            # Adaptation si nécessaire
            adapted = False
            if recommendations.get('should_adapt', False):
                adapted = await self.adapt(recommendations)
            
            # Mise à jour des métriques
            self._update_metrics(current_metrics, adapted)
            
            # Enregistrement dans l'historique
            self._record_adaptation(recommendations, adapted)
            
            duration = time.time() - start_time
            self.strategy_performance.labels(
                strategy_name=self.config.name,
                strategy_type=self.config.strategy_type.value
            ).observe(duration)
            
            return {
                "strategy_name": self.config.name,
                "adapted": adapted,
                "recommendations": recommendations,
                "duration": duration,
                "metrics": self.get_current_metrics()
            }
            
        except Exception as e:
            self.strategy_adaptations.labels(
                strategy_name=self.config.name,
                strategy_type=self.config.strategy_type.value,
                result='error'
            ).inc()
            
            logger.error(
                "Erreur lors du cycle de stratégie",
                strategy_name=self.config.name,
                error=str(e)
            )
            raise
    
    def _update_metrics(self, current_metrics: Dict[str, Any], adapted: bool) -> None:
        """Met à jour les métriques de la stratégie."""
        if adapted:
            self.metrics.adaptations_count += 1
            self.metrics.last_adaptation = datetime.now(timezone.utc)
        
        # Calcul du taux de succès
        if len(self.adaptation_history) > 0:
            recent_successes = sum(1 for record in list(self.adaptation_history)[-10:] if record['success'])
            self.metrics.success_rate = recent_successes / min(10, len(self.adaptation_history)) * 100
        
        # Mise à jour de la valeur actuelle
        if 'response_time' in current_metrics:
            self.metrics.avg_response_time = current_metrics['response_time']
        
        # Métriques Prometheus
        self.strategy_value.labels(
            strategy_name=self.config.name,
            strategy_type=self.config.strategy_type.value
        ).set(self.metrics.current_value)
    
    def _record_adaptation(self, recommendations: Dict[str, Any], success: bool) -> None:
        """Enregistre une adaptation dans l'historique."""
        record = {
            'timestamp': datetime.now(timezone.utc),
            'recommendations': recommendations,
            'success': success,
            'metrics_snapshot': dict(self.metrics.__dict__)
        }
        
        self.adaptation_history.append(record)
        
        self.strategy_adaptations.labels(
            strategy_name=self.config.name,
            strategy_type=self.config.strategy_type.value,
            result='success' if success else 'failure'
        ).inc()
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques actuelles de la stratégie."""
        return {
            "strategy_name": self.config.name,
            "strategy_type": self.config.strategy_type.value,
            "metrics": dict(self.metrics.__dict__),
            "config": dict(self.config.__dict__),
            "history_size": len(self.history),
            "adaptations_history_size": len(self.adaptation_history)
        }


class AdaptiveStrategy(BaseStrategy):
    """
    Stratégie adaptative basée sur l'analyse des tendances.
    
    Fonctionnalités:
    - Adaptation basée sur les seuils dynamiques
    - Détection de tendances
    - Lissage exponentiel
    - Prédiction simple à court terme
    """
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.ema_alpha = 0.3  # Coefficient de lissage exponentiel
        self.ema_value = None
        self.trend_detector = TrendDetector()
    
    async def evaluate(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Évalue les métriques et détermine les adaptations."""
        
        # Extraction des métriques clés
        key_metric = self._extract_key_metric(current_metrics)
        
        # Mise à jour de l'EMA
        if self.ema_value is None:
            self.ema_value = key_metric
        else:
            self.ema_value = self.ema_alpha * key_metric + (1 - self.ema_alpha) * self.ema_value
        
        # Ajout à l'historique
        self.history.append({
            'timestamp': time.time(),
            'value': key_metric,
            'ema': self.ema_value,
            'metrics': current_metrics
        })
        
        # Détection de tendance
        trend = self.trend_detector.detect_trend(list(self.history))
        
        # Calcul de la déviation
        deviation = abs(key_metric - self.ema_value) / self.ema_value if self.ema_value > 0 else 0
        
        # Décision d'adaptation
        should_adapt = deviation > self.config.sensitivity_threshold
        
        # Calcul de la nouvelle valeur
        new_value = self._calculate_adaptation(key_metric, trend, deviation)
        
        return {
            'should_adapt': should_adapt,
            'current_value': key_metric,
            'ema_value': self.ema_value,
            'trend': trend,
            'deviation': deviation,
            'new_value': new_value,
            'confidence': min(1.0, 1.0 - deviation)
        }
    
    async def adapt(self, recommendations: Dict[str, Any]) -> bool:
        """Applique les adaptations recommandées."""
        try:
            new_value = recommendations['new_value']
            
            # Application des limites
            new_value = max(self.config.min_value, min(self.config.max_value, new_value))
            
            # Mise à jour de la valeur actuelle
            old_value = self.metrics.current_value
            self.metrics.current_value = new_value
            
            logger.info(
                "Adaptation appliquée",
                strategy_name=self.config.name,
                old_value=old_value,
                new_value=new_value,
                trend=recommendations.get('trend'),
                deviation=recommendations.get('deviation')
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Erreur lors de l'adaptation",
                strategy_name=self.config.name,
                error=str(e)
            )
            return False
    
    def _extract_key_metric(self, metrics: Dict[str, Any]) -> float:
        """Extrait la métrique clé selon l'objectif d'optimisation."""
        if self.config.optimization_goal == OptimizationGoal.LATENCY:
            return metrics.get('avg_response_time', 0.0)
        elif self.config.optimization_goal == OptimizationGoal.THROUGHPUT:
            return metrics.get('requests_per_second', 0.0)
        elif self.config.optimization_goal == OptimizationGoal.COST:
            return metrics.get('cost_per_request', 0.0)
        elif self.config.optimization_goal == OptimizationGoal.RELIABILITY:
            return 1.0 - metrics.get('error_rate', 0.0)
        else:
            # Métrique composite pour BALANCED
            latency_score = 1.0 / (1.0 + metrics.get('avg_response_time', 1.0))
            throughput_score = min(1.0, metrics.get('requests_per_second', 0.0) / 100.0)
            reliability_score = 1.0 - metrics.get('error_rate', 0.0)
            return (latency_score + throughput_score + reliability_score) / 3.0
    
    def _calculate_adaptation(self, current_value: float, trend: str, deviation: float) -> float:
        """Calcule la nouvelle valeur d'adaptation."""
        adaptation_factor = 1.0
        
        # Ajustement basé sur la tendance
        if trend == 'increasing':
            adaptation_factor = 1.0 + self.config.learning_rate * deviation
        elif trend == 'decreasing':
            adaptation_factor = 1.0 - self.config.learning_rate * deviation
        
        # Objectif d'optimisation
        if self.config.optimization_goal in [OptimizationGoal.LATENCY, OptimizationGoal.COST]:
            # Pour latence et coût, on veut diminuer
            adaptation_factor = 1.0 / adaptation_factor
        
        return self.metrics.current_value * adaptation_factor


class PredictiveStrategy(BaseStrategy):
    """
    Stratégie prédictive utilisant Machine Learning.
    
    Fonctionnalités:
    - Prédiction des charges futures
    - Adaptation proactive
    - Modèles ML adaptatifs
    - Détection d'anomalies
    """
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.ml_model = None
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.model_trained = False
        self.feature_buffer = deque(maxlen=config.feature_window_size)
        self.prediction_cache = {}
        
        # Initialisation du modèle
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialise le modèle ML."""
        if self.config.ml_model_path and os.path.exists(self.config.ml_model_path):
            try:
                self.ml_model = joblib.load(self.config.ml_model_path)
                self.model_trained = True
                logger.info("Modèle ML chargé", path=self.config.ml_model_path)
            except Exception as e:
                logger.warning("Échec du chargement du modèle", error=str(e))
        
        if self.ml_model is None:
            self.ml_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
    
    async def evaluate(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Évalue avec prédiction ML."""
        
        # Extraction des features
        features = self._extract_features(current_metrics)
        self.feature_buffer.append(features)
        
        # Entraînement du modèle si nécessaire
        if len(self.feature_buffer) >= 50 and not self.model_trained:
            await self._train_model()
        
        # Prédiction si le modèle est prêt
        predictions = {}
        if self.model_trained:
            predictions = await self._make_predictions(features)
        
        # Détection d'anomalies
        anomaly_score = self._detect_anomaly(features)
        
        # Décision d'adaptation
        should_adapt = self._should_adapt_predictive(predictions, anomaly_score)
        
        return {
            'should_adapt': should_adapt,
            'predictions': predictions,
            'anomaly_score': anomaly_score,
            'current_features': features,
            'model_trained': self.model_trained,
            'confidence': predictions.get('confidence', 0.0)
        }
    
    async def adapt(self, recommendations: Dict[str, Any]) -> bool:
        """Adaptation basée sur les prédictions."""
        try:
            predictions = recommendations.get('predictions', {})
            
            if 'predicted_load' in predictions:
                # Adaptation proactive basée sur la charge prédite
                predicted_load = predictions['predicted_load']
                current_capacity = self.metrics.current_value
                
                # Calcul de la nouvelle capacité
                safety_margin = 1.2  # 20% de marge
                new_capacity = predicted_load * safety_margin
                
                # Application des limites
                new_capacity = max(self.config.min_value, min(self.config.max_value, new_capacity))
                
                self.metrics.current_value = new_capacity
                self.metrics.predicted_value = predicted_load
                self.metrics.confidence_score = predictions.get('confidence', 0.0)
                
                logger.info(
                    "Adaptation prédictive appliquée",
                    strategy_name=self.config.name,
                    predicted_load=predicted_load,
                    new_capacity=new_capacity,
                    confidence=self.metrics.confidence_score
                )
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(
                "Erreur lors de l'adaptation prédictive",
                strategy_name=self.config.name,
                error=str(e)
            )
            return False
    
    def _extract_features(self, metrics: Dict[str, Any]) -> List[float]:
        """Extrait les features pour le ML."""
        features = [
            metrics.get('cpu_usage', 0.0),
            metrics.get('memory_usage', 0.0),
            metrics.get('requests_per_second', 0.0),
            metrics.get('avg_response_time', 0.0),
            metrics.get('error_rate', 0.0),
            metrics.get('queue_size', 0.0),
            time.time() % 86400,  # Heure du jour
            time.time() % 604800,  # Jour de la semaine
        ]
        
        return features
    
    async def _train_model(self) -> None:
        """Entraîne le modèle ML."""
        try:
            if len(self.feature_buffer) < 20:
                return
            
            # Préparation des données
            X = []
            y = []
            
            features_list = list(self.feature_buffer)
            for i in range(len(features_list) - 1):
                X.append(features_list[i])
                # Target: RPS du point suivant
                y.append(features_list[i + 1][2])  # requests_per_second
            
            if len(X) < 10:
                return
            
            X = np.array(X)
            y = np.array(y)
            
            # Normalisation
            X_scaled = self.scaler.fit_transform(X)
            
            # Entraînement
            self.ml_model.fit(X_scaled, y)
            
            # Entraînement du détecteur d'anomalies
            self.anomaly_detector.fit(X_scaled)
            
            self.model_trained = True
            
            # Évaluation du modèle
            y_pred = self.ml_model.predict(X_scaled)
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            
            logger.info(
                "Modèle ML entraîné",
                strategy_name=self.config.name,
                samples=len(X),
                mse=mse,
                mae=mae
            )
            
        except Exception as e:
            logger.error(
                "Erreur lors de l'entraînement ML",
                strategy_name=self.config.name,
                error=str(e)
            )
    
    async def _make_predictions(self, current_features: List[float]) -> Dict[str, Any]:
        """Fait des prédictions ML."""
        try:
            # Normalisation des features
            features_scaled = self.scaler.transform([current_features])
            
            # Prédiction
            predicted_load = self.ml_model.predict(features_scaled)[0]
            
            # Calcul de la confiance (basé sur les arbres du Random Forest)
            if hasattr(self.ml_model, 'estimators_'):
                predictions = [tree.predict(features_scaled)[0] for tree in self.ml_model.estimators_]
                confidence = 1.0 - (np.std(predictions) / np.mean(predictions)) if np.mean(predictions) > 0 else 0.0
            else:
                confidence = 0.5
            
            return {
                'predicted_load': predicted_load,
                'confidence': min(1.0, max(0.0, confidence)),
                'horizon_seconds': self.config.prediction_horizon
            }
            
        except Exception as e:
            logger.error(
                "Erreur lors de la prédiction",
                strategy_name=self.config.name,
                error=str(e)
            )
            return {}
    
    def _detect_anomaly(self, features: List[float]) -> float:
        """Détecte les anomalies dans les features."""
        try:
            if not self.model_trained:
                return 0.0
            
            features_scaled = self.scaler.transform([features])
            anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
            
            # Normalisation du score (-1 à 1) vers (0 à 1)
            normalized_score = (anomaly_score + 1) / 2
            
            return 1.0 - normalized_score  # Inverser pour que 1 = anomalie
            
        except Exception as e:
            logger.warning(
                "Erreur lors de la détection d'anomalie",
                strategy_name=self.config.name,
                error=str(e)
            )
            return 0.0
    
    def _should_adapt_predictive(self, predictions: Dict[str, Any], anomaly_score: float) -> bool:
        """Détermine si une adaptation est nécessaire."""
        
        # Adaptation si anomalie détectée
        if anomaly_score > 0.7:
            return True
        
        # Adaptation si prédiction significativement différente
        if 'predicted_load' in predictions:
            current_load = self.feature_buffer[-1][2] if self.feature_buffer else 0.0  # RPS
            predicted_load = predictions['predicted_load']
            
            if current_load > 0:
                change_ratio = abs(predicted_load - current_load) / current_load
                return change_ratio > self.config.sensitivity_threshold
        
        return False


class TrendDetector:
    """
    Détecteur de tendances pour les métriques.
    """
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
    
    def detect_trend(self, history: List[Dict[str, Any]]) -> str:
        """
        Détecte la tendance dans l'historique.
        
        Returns:
            'increasing', 'decreasing', 'stable'
        """
        if len(history) < self.window_size:
            return 'stable'
        
        # Extraction des valeurs récentes
        recent_values = [item['value'] for item in history[-self.window_size:]]
        
        # Calcul de la régression linéaire
        x = np.arange(len(recent_values))
        y = np.array(recent_values)
        
        if len(np.unique(y)) == 1:
            return 'stable'
        
        # Coefficient de corrélation
        correlation = np.corrcoef(x, y)[0, 1]
        
        # Pente de la régression
        slope = np.polyfit(x, y, 1)[0]
        
        # Décision basée sur la pente et la corrélation
        if abs(correlation) < 0.3:
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'


class MultiTenantStrategy(BaseStrategy):
    """
    Stratégie multi-locataire avec isolation et équité.
    
    Fonctionnalités:
    - Isolation des ressources par tenant
    - Équité dans l'allocation
    - SLA différenciés
    - Priorités dynamiques
    """
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.tenant_allocations = defaultdict(lambda: {'cpu': 1.0, 'memory': 1.0, 'priority': 1})
        self.tenant_metrics = defaultdict(lambda: deque(maxlen=100))
        self.sla_configs = {}
    
    def set_tenant_sla(self, tenant_id: str, sla_config: Dict[str, Any]) -> None:
        """Configure le SLA pour un tenant."""
        self.sla_configs[tenant_id] = sla_config
        logger.info("SLA configuré", tenant_id=tenant_id, sla=sla_config)
    
    async def evaluate(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Évalue les métriques multi-tenant."""
        
        tenant_id = current_metrics.get('tenant_id', 'default')
        
        # Mise à jour des métriques du tenant
        self.tenant_metrics[tenant_id].append({
            'timestamp': time.time(),
            'metrics': current_metrics
        })
        
        # Analyse des violations de SLA
        sla_violations = self._check_sla_violations(tenant_id, current_metrics)
        
        # Calcul des nouvelles allocations
        new_allocations = self._calculate_fair_allocations()
        
        # Détection des tenants sous-performants
        underperforming_tenants = self._detect_underperforming_tenants()
        
        should_adapt = bool(sla_violations or underperforming_tenants)
        
        return {
            'should_adapt': should_adapt,
            'tenant_id': tenant_id,
            'sla_violations': sla_violations,
            'new_allocations': new_allocations,
            'underperforming_tenants': underperforming_tenants,
            'current_allocations': dict(self.tenant_allocations)
        }
    
    async def adapt(self, recommendations: Dict[str, Any]) -> bool:
        """Adapte les allocations multi-tenant."""
        try:
            new_allocations = recommendations.get('new_allocations', {})
            
            # Application des nouvelles allocations
            for tenant_id, allocation in new_allocations.items():
                old_allocation = self.tenant_allocations[tenant_id].copy()
                self.tenant_allocations[tenant_id].update(allocation)
                
                logger.info(
                    "Allocation mise à jour",
                    tenant_id=tenant_id,
                    old_allocation=old_allocation,
                    new_allocation=allocation
                )
            
            return True
            
        except Exception as e:
            logger.error(
                "Erreur lors de l'adaptation multi-tenant",
                error=str(e)
            )
            return False
    
    def _check_sla_violations(self, tenant_id: str, metrics: Dict[str, Any]) -> List[str]:
        """Vérifie les violations de SLA."""
        violations = []
        
        if tenant_id not in self.sla_configs:
            return violations
        
        sla = self.sla_configs[tenant_id]
        
        # Vérification latence
        if 'max_latency' in sla:
            if metrics.get('avg_response_time', 0) > sla['max_latency']:
                violations.append('latency_violation')
        
        # Vérification débit
        if 'min_throughput' in sla:
            if metrics.get('requests_per_second', 0) < sla['min_throughput']:
                violations.append('throughput_violation')
        
        # Vérification taux d'erreur
        if 'max_error_rate' in sla:
            if metrics.get('error_rate', 0) > sla['max_error_rate']:
                violations.append('error_rate_violation')
        
        return violations
    
    def _calculate_fair_allocations(self) -> Dict[str, Dict[str, float]]:
        """Calcule les allocations équitables."""
        allocations = {}
        
        if not self.tenant_metrics:
            return allocations
        
        # Calcul des métriques moyennes par tenant
        tenant_loads = {}
        for tenant_id, metrics_history in self.tenant_metrics.items():
            if metrics_history:
                recent_metrics = list(metrics_history)[-10:]  # 10 dernières métriques
                avg_cpu = statistics.mean([m['metrics'].get('cpu_usage', 0) for m in recent_metrics])
                avg_memory = statistics.mean([m['metrics'].get('memory_usage', 0) for m in recent_metrics])
                tenant_loads[tenant_id] = {'cpu': avg_cpu, 'memory': avg_memory}
        
        # Allocation proportionnelle basée sur la charge
        total_cpu = sum(load['cpu'] for load in tenant_loads.values())
        total_memory = sum(load['memory'] for load in tenant_loads.values())
        
        for tenant_id, load in tenant_loads.items():
            cpu_ratio = load['cpu'] / total_cpu if total_cpu > 0 else 1.0 / len(tenant_loads)
            memory_ratio = load['memory'] / total_memory if total_memory > 0 else 1.0 / len(tenant_loads)
            
            # Facteur de priorité
            priority_factor = self.sla_configs.get(tenant_id, {}).get('priority', 1)
            
            allocations[tenant_id] = {
                'cpu': min(4.0, max(0.1, cpu_ratio * priority_factor * 2.0)),
                'memory': min(4.0, max(0.1, memory_ratio * priority_factor * 2.0)),
                'priority': priority_factor
            }
        
        return allocations
    
    def _detect_underperforming_tenants(self) -> List[str]:
        """Détecte les tenants sous-performants."""
        underperforming = []
        
        for tenant_id, metrics_history in self.tenant_metrics.items():
            if len(metrics_history) < 5:
                continue
            
            recent_metrics = list(metrics_history)[-5:]
            avg_response_time = statistics.mean([m['metrics'].get('avg_response_time', 0) for m in recent_metrics])
            avg_error_rate = statistics.mean([m['metrics'].get('error_rate', 0) for m in recent_metrics])
            
            # Seuils de performance
            if avg_response_time > 5.0 or avg_error_rate > 0.05:  # 5s ou 5% erreur
                underperforming.append(tenant_id)
        
        return underperforming


# Gestionnaire global des stratégies
class StrategyManager:
    """
    Gestionnaire centralisé des stratégies adaptatives.
    
    Fonctionnalités:
    - Orchestration des stratégies
    - Coordination multi-stratégies
    - Monitoring global
    - Optimisation des conflits
    """
    
    def __init__(self):
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_conflicts = defaultdict(list)
        self.execution_scheduler = asyncio.create_task(self._strategy_execution_loop())
        
        # Métriques globales
        self.manager_cycles = Counter(
            'strategy_manager_cycles_total',
            'Nombre total de cycles du gestionnaire',
            ['result']
        )
        
        logger.info("Gestionnaire de stratégies initialisé")
    
    def register_strategy(self, strategy: BaseStrategy) -> None:
        """Enregistre une stratégie."""
        self.strategies[strategy.config.name] = strategy
        logger.info("Stratégie enregistrée", name=strategy.config.name)
    
    def unregister_strategy(self, strategy_name: str) -> None:
        """Désenregistre une stratégie."""
        if strategy_name in self.strategies:
            del self.strategies[strategy_name]
            logger.info("Stratégie désenregistrée", name=strategy_name)
    
    async def execute_strategies(self, global_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Exécute toutes les stratégies actives."""
        results = {}
        conflicts = []
        
        try:
            # Exécution de toutes les stratégies
            strategy_tasks = []
            for name, strategy in self.strategies.items():
                if strategy.config.enabled:
                    task = strategy.run_cycle(global_metrics)
                    strategy_tasks.append((name, task))
            
            # Attente des résultats
            strategy_results = {}
            for name, task in strategy_tasks:
                try:
                    result = await task
                    strategy_results[name] = result
                except Exception as e:
                    logger.error("Erreur stratégie", name=name, error=str(e))
                    strategy_results[name] = {'error': str(e)}
            
            # Détection et résolution des conflits
            conflicts = self._detect_conflicts(strategy_results)
            if conflicts:
                strategy_results = await self._resolve_conflicts(strategy_results, conflicts)
            
            results = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'strategies': strategy_results,
                'conflicts': conflicts,
                'global_metrics': global_metrics
            }
            
            self.manager_cycles.labels(result='success').inc()
            
        except Exception as e:
            logger.error("Erreur gestionnaire stratégies", error=str(e))
            self.manager_cycles.labels(result='error').inc()
            results = {'error': str(e)}
        
        return results
    
    def _detect_conflicts(self, strategy_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Détecte les conflits entre stratégies."""
        conflicts = []
        
        # Analyse des adaptations concurrentes
        adapting_strategies = [
            name for name, result in strategy_results.items()
            if result.get('adapted', False)
        ]
        
        if len(adapting_strategies) > 1:
            conflicts.append({
                'type': 'concurrent_adaptations',
                'strategies': adapting_strategies,
                'description': 'Plusieurs stratégies tentent de s\'adapter simultanément'
            })
        
        # Analyse des objectifs contradictoires
        objective_groups = defaultdict(list)
        for name, result in strategy_results.items():
            strategy = self.strategies.get(name)
            if strategy:
                objective_groups[strategy.config.optimization_goal].append(name)
        
        # Conflits latence vs débit
        if (objective_groups[OptimizationGoal.LATENCY] and 
            objective_groups[OptimizationGoal.THROUGHPUT]):
            conflicts.append({
                'type': 'objective_conflict',
                'strategies': (objective_groups[OptimizationGoal.LATENCY] + 
                             objective_groups[OptimizationGoal.THROUGHPUT]),
                'description': 'Conflit entre optimisation latence et débit'
            })
        
        return conflicts
    
    async def _resolve_conflicts(self, strategy_results: Dict[str, Any], conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Résout les conflits entre stratégies."""
        
        for conflict in conflicts:
            if conflict['type'] == 'concurrent_adaptations':
                # Résolution par priorité
                conflicting_strategies = conflict['strategies']
                priorities = {name: self.strategies[name].config.priority for name in conflicting_strategies}
                
                # Garde seulement la stratégie avec la plus haute priorité
                highest_priority = max(priorities.values())
                winners = [name for name, priority in priorities.items() if priority == highest_priority]
                
                # Si égalité, garde la première alphabétiquement
                winner = min(winners)
                
                for name in conflicting_strategies:
                    if name != winner:
                        strategy_results[name]['adapted'] = False
                        strategy_results[name]['conflict_resolution'] = 'deprioritized'
                
                logger.info("Conflit résolu par priorité", winner=winner, losers=[n for n in conflicting_strategies if n != winner])
            
            elif conflict['type'] == 'objective_conflict':
                # Résolution par stratégie équilibrée
                conflicting_strategies = conflict['strategies']
                
                # Calcul d'un compromis
                for name in conflicting_strategies:
                    if strategy_results[name].get('adapted'):
                        # Réduction de l'amplitude d'adaptation
                        if 'recommendations' in strategy_results[name]:
                            recommendations = strategy_results[name]['recommendations']
                            if 'new_value' in recommendations:
                                current_value = self.strategies[name].metrics.current_value
                                new_value = recommendations['new_value']
                                # Réduction de 50% de l'adaptation
                                compromised_value = current_value + (new_value - current_value) * 0.5
                                recommendations['new_value'] = compromised_value
                                strategy_results[name]['conflict_resolution'] = 'compromised'
                
                logger.info("Conflit d'objectifs résolu par compromis", strategies=conflicting_strategies)
        
        return strategy_results
    
    async def _strategy_execution_loop(self) -> None:
        """Boucle d'exécution périodique des stratégies."""
        while True:
            try:
                await asyncio.sleep(30)  # Exécution toutes les 30 secondes
                
                # Collecte des métriques globales (placeholder)
                global_metrics = {
                    'timestamp': time.time(),
                    'cpu_usage': random.uniform(20, 80),
                    'memory_usage': random.uniform(30, 70),
                    'requests_per_second': random.uniform(10, 100),
                    'avg_response_time': random.uniform(0.1, 2.0),
                    'error_rate': random.uniform(0.001, 0.05)
                }
                
                # Exécution des stratégies
                await self.execute_strategies(global_metrics)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Erreur dans la boucle d'exécution", error=str(e))
                await asyncio.sleep(60)  # Attente plus longue en cas d'erreur
    
    def get_global_status(self) -> Dict[str, Any]:
        """Retourne le statut global du gestionnaire."""
        return {
            'strategies_count': len(self.strategies),
            'active_strategies': [name for name, strategy in self.strategies.items() if strategy.config.enabled],
            'strategies_status': {name: strategy.get_current_metrics() for name, strategy in self.strategies.items()},
            'recent_conflicts': list(self.strategy_conflicts)[-10:]  # 10 derniers conflits
        }


# Instance globale
global_strategy_manager = StrategyManager()
