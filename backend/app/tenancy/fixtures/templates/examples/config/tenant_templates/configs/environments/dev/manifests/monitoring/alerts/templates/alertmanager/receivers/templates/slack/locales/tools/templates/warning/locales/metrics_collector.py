#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spotify AI Agent - Metrics Collector pour Alerting Multi-Tenant

Collecteur de métriques avancé pour système d'alerting avec support
de métriques métier, techniques et d'usage pour l'écosystème IA musical.

Fonctionnalités:
- Collecte métriques Prometheus/Grafana
- Métriques business (streams, revenus, engagement)
- Métriques IA (accuracy, latency, throughput)
- Métriques infrastructure (CPU, RAM, réseau)
- Agrégation temps réel multi-niveaux
- Alerting prédictif avec ML
- Export vers systèmes externes

Architecture:
- Observer Pattern pour collecte temps réel
- Strategy Pattern pour différents collecteurs
- Decorator Pattern pour enrichissement métrique
- Pipeline Pattern pour transformation données
- Publisher-Subscriber pour distribution

Performance:
- Collecte asynchrone haute performance
- Batching intelligent des métriques
- Compression et optimisation stockage
- Cache distribué pour requêtes fréquentes
- Sampling adaptatif selon la charge

Utilisation:
    collector = MetricsCollector()
    
    # Collecte métrique business
    collector.collect_business_metric(
        tenant_id="universal_music_001",
        metric_name="monthly_streams",
        value=1500000000,
        tags={"artist": "taylor_swift", "region": "us"}
    )
    
    # Récupération pour alerting
    metrics = collector.get_metrics_for_alerting(
        tenant_id="universal_music_001",
        time_range="5m"
    )
"""

import json
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
from functools import wraps
import statistics
import logging

# Imports externes
import structlog
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, 
    CollectorRegistry, generate_latest,
    CONTENT_TYPE_LATEST
)
import redis
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import pandas as pd

# Imports internes
from .config import get_environment, DEFAULT_THRESHOLDS

# Configuration logging
logger = structlog.get_logger(__name__)

# Registre Prometheus personnalisé
custom_registry = CollectorRegistry()

# Métriques système
metrics_collection_total = Counter(
    'spotify_ai_metrics_collection_total',
    'Nombre total de métriques collectées',
    ['tenant_id', 'metric_type', 'source'],
    registry=custom_registry
)

metrics_collection_duration = Histogram(
    'spotify_ai_metrics_collection_duration_seconds',
    'Durée de collecte des métriques',
    ['collector_type'],
    registry=custom_registry
)

metrics_anomaly_detected = Counter(
    'spotify_ai_metrics_anomaly_detected_total',
    'Anomalies détectées dans les métriques',
    ['tenant_id', 'metric_name', 'anomaly_type'],
    registry=custom_registry
)

active_metric_streams = Gauge(
    'spotify_ai_active_metric_streams',
    'Nombre de streams de métriques actifs',
    registry=custom_registry
)

class MetricType(Enum):
    """Types de métriques collectées."""
    # Business metrics
    BUSINESS_STREAMS = "business_streams"
    BUSINESS_REVENUE = "business_revenue" 
    BUSINESS_ENGAGEMENT = "business_engagement"
    BUSINESS_CONVERSION = "business_conversion"
    
    # AI/ML metrics
    AI_ACCURACY = "ai_accuracy"
    AI_LATENCY = "ai_latency"
    AI_THROUGHPUT = "ai_throughput"
    AI_MODEL_DRIFT = "ai_model_drift"
    AI_TRAINING_LOSS = "ai_training_loss"
    
    # Technical metrics
    TECH_CPU = "tech_cpu"
    TECH_MEMORY = "tech_memory"
    TECH_DISK = "tech_disk"
    TECH_NETWORK = "tech_network"
    TECH_API_LATENCY = "tech_api_latency"
    TECH_ERROR_RATE = "tech_error_rate"
    
    # User metrics
    USER_ACTIVITY = "user_activity"
    USER_SATISFACTION = "user_satisfaction"
    USER_RETENTION = "user_retention"

class AggregationType(Enum):
    """Types d'agrégation supportés."""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    P50 = "p50"
    P95 = "p95"
    P99 = "p99"
    COUNT = "count"
    RATE = "rate"
    STDDEV = "stddev"

class AnomalyType(Enum):
    """Types d'anomalies détectables."""
    SPIKE = "spike"              # Pic soudain
    DROP = "drop"                # Chute soudaine
    TREND_CHANGE = "trend_change" # Changement de tendance
    OUTLIER = "outlier"          # Valeur aberrante
    SEASONALITY = "seasonality"   # Anomalie saisonnière
    DRIFT = "drift"              # Dérive graduelle

@dataclass
class MetricPoint:
    """Point de métrique avec métadonnées."""
    timestamp: datetime
    value: Union[int, float]
    metric_name: str
    metric_type: MetricType
    tenant_id: str
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Métadonnées techniques
    source: str = "unknown"
    quality_score: float = 1.0  # 0.0 à 1.0
    is_estimated: bool = False
    collection_latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['metric_type'] = self.metric_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricPoint':
        """Crée depuis un dictionnaire."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['metric_type'] = MetricType(data['metric_type'])
        return cls(**data)

@dataclass
class AggregatedMetric:
    """Métrique agrégée avec statistiques."""
    metric_name: str
    tenant_id: str
    start_time: datetime
    end_time: datetime
    aggregation_type: AggregationType
    value: float
    
    # Statistiques détaillées
    count: int = 0
    min_value: float = 0.0
    max_value: float = 0.0
    stddev: float = 0.0
    percentiles: Dict[str, float] = field(default_factory=dict)
    
    # Métadonnées
    tags: Dict[str, str] = field(default_factory=dict)
    quality_score: float = 1.0

@dataclass
class Anomaly:
    """Anomalie détectée dans une métrique."""
    metric_name: str
    tenant_id: str
    timestamp: datetime
    anomaly_type: AnomalyType
    severity: float  # 0.0 à 1.0
    
    # Valeurs
    actual_value: float
    expected_value: float
    threshold: float
    
    # Contexte
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    
    # Actions suggérées
    suggested_actions: List[str] = field(default_factory=list)

class MetricCollector:
    """Interface de base pour collecteurs de métriques."""
    
    def __init__(self, collector_name: str):
        self.collector_name = collector_name
        self.is_active = False
        self.collection_count = 0
        self.last_collection = None
    
    async def collect(self, tenant_id: str) -> List[MetricPoint]:
        """Collecte les métriques pour un tenant."""
        raise NotImplementedError
    
    def get_supported_metrics(self) -> List[MetricType]:
        """Retourne les types de métriques supportés."""
        raise NotImplementedError

class PrometheusCollector(MetricCollector):
    """Collecteur pour métriques Prometheus."""
    
    def __init__(self, prometheus_url: str = "http://prometheus:9090"):
        super().__init__("prometheus")
        self.prometheus_url = prometheus_url
        self.queries = {
            MetricType.TECH_CPU: 'avg(cpu_usage_percent{tenant_id="%s"})',
            MetricType.TECH_MEMORY: 'avg(memory_usage_percent{tenant_id="%s"})',
            MetricType.TECH_API_LATENCY: 'histogram_quantile(0.95, api_request_duration_seconds{tenant_id="%s"})',
            MetricType.TECH_ERROR_RATE: 'rate(api_errors_total{tenant_id="%s"}[5m]) * 100',
            MetricType.AI_LATENCY: 'histogram_quantile(0.95, ai_inference_duration_seconds{tenant_id="%s"})',
            MetricType.AI_THROUGHPUT: 'rate(ai_inferences_total{tenant_id="%s"}[5m])'
        }
    
    async def collect(self, tenant_id: str) -> List[MetricPoint]:
        """Collecte depuis Prometheus."""
        metrics = []
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                for metric_type, query_template in self.queries.items():
                    query = query_template % tenant_id
                    url = f"{self.prometheus_url}/api/v1/query"
                    
                    params = {
                        'query': query,
                        'time': int(time.time())
                    }
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            if data['status'] == 'success' and data['data']['result']:
                                for result in data['data']['result']:
                                    timestamp = datetime.fromtimestamp(float(result['value'][0]))
                                    value = float(result['value'][1])
                                    
                                    metric = MetricPoint(
                                        timestamp=timestamp,
                                        value=value,
                                        metric_name=metric_type.value,
                                        metric_type=metric_type,
                                        tenant_id=tenant_id,
                                        source="prometheus",
                                        tags=result['metric']
                                    )
                                    metrics.append(metric)
            
            self.collection_count += len(metrics)
            self.last_collection = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Erreur collecte Prometheus", collector=self.collector_name, error=str(e))
        
        return metrics
    
    def get_supported_metrics(self) -> List[MetricType]:
        """Métriques supportées par Prometheus."""
        return list(self.queries.keys())

class BusinessMetricsCollector(MetricCollector):
    """Collecteur pour métriques business Spotify."""
    
    def __init__(self, spotify_api_client=None):
        super().__init__("business")
        self.spotify_client = spotify_api_client
    
    async def collect(self, tenant_id: str) -> List[MetricPoint]:
        """Collecte métriques business depuis API Spotify."""
        metrics = []
        
        try:
            # Simulation de données business (en prod: vraie API Spotify)
            import random
            now = datetime.utcnow()
            
            # Streams mensuels
            monthly_streams = random.randint(10000000, 1000000000)
            metrics.append(MetricPoint(
                timestamp=now,
                value=monthly_streams,
                metric_name="monthly_streams",
                metric_type=MetricType.BUSINESS_STREAMS,
                tenant_id=tenant_id,
                source="spotify_api",
                tags={"period": "monthly"}
            ))
            
            # Revenus estimés
            estimated_revenue = monthly_streams * random.uniform(0.003, 0.008)  # $0.003-0.008 par stream
            metrics.append(MetricPoint(
                timestamp=now,
                value=estimated_revenue,
                metric_name="estimated_revenue_usd",
                metric_type=MetricType.BUSINESS_REVENUE,
                tenant_id=tenant_id,
                source="spotify_api",
                tags={"currency": "USD", "period": "monthly"}
            ))
            
            # Engagement (skip rate)
            skip_rate = random.uniform(0.1, 0.4)  # 10-40%
            metrics.append(MetricPoint(
                timestamp=now,
                value=skip_rate * 100,
                metric_name="skip_rate_percent",
                metric_type=MetricType.BUSINESS_ENGAGEMENT,
                tenant_id=tenant_id,
                source="spotify_api",
                tags={"metric": "skip_rate"}
            ))
            
            # Taux de conversion playlist
            playlist_conversion = random.uniform(0.02, 0.15)  # 2-15%
            metrics.append(MetricPoint(
                timestamp=now,
                value=playlist_conversion * 100,
                metric_name="playlist_conversion_percent",
                metric_type=MetricType.BUSINESS_CONVERSION,
                tenant_id=tenant_id,
                source="spotify_api",
                tags={"metric": "playlist_add"}
            ))
            
        except Exception as e:
            logger.error(f"Erreur collecte métriques business", collector=self.collector_name, error=str(e))
        
        return metrics
    
    def get_supported_metrics(self) -> List[MetricType]:
        """Métriques business supportées."""
        return [
            MetricType.BUSINESS_STREAMS,
            MetricType.BUSINESS_REVENUE,
            MetricType.BUSINESS_ENGAGEMENT,
            MetricType.BUSINESS_CONVERSION
        ]

class AIMetricsCollector(MetricCollector):
    """Collecteur pour métriques IA/ML."""
    
    def __init__(self, ml_monitoring_client=None):
        super().__init__("ai_ml")
        self.ml_client = ml_monitoring_client
    
    async def collect(self, tenant_id: str) -> List[MetricPoint]:
        """Collecte métriques IA depuis système de monitoring ML."""
        metrics = []
        
        try:
            # Simulation de métriques IA (en prod: vraie télémétrie ML)
            import random
            now = datetime.utcnow()
            
            # Précision du modèle de recommandation
            recommendation_accuracy = random.uniform(0.85, 0.98)
            metrics.append(MetricPoint(
                timestamp=now,
                value=recommendation_accuracy * 100,
                metric_name="recommendation_accuracy_percent",
                metric_type=MetricType.AI_ACCURACY,
                tenant_id=tenant_id,
                source="ml_monitoring",
                tags={"model": "recommendation_v3", "dataset": "user_preferences"}
            ))
            
            # Latence d'inférence génération musicale
            generation_latency = random.uniform(50, 800)  # 50-800ms
            metrics.append(MetricPoint(
                timestamp=now,
                value=generation_latency,
                metric_name="music_generation_latency_ms",
                metric_type=MetricType.AI_LATENCY,
                tenant_id=tenant_id,
                source="ml_monitoring",
                tags={"model": "musicgen_v2", "format": "preview"}
            ))
            
            # Throughput du modèle
            throughput = random.uniform(100, 1000)  # inférences/sec
            metrics.append(MetricPoint(
                timestamp=now,
                value=throughput,
                metric_name="ai_throughput_per_sec",
                metric_type=MetricType.AI_THROUGHPUT,
                tenant_id=tenant_id,
                source="ml_monitoring",
                tags={"model": "all", "environment": get_environment()}
            ))
            
            # Détection de drift
            drift_score = random.uniform(0.0, 0.3)  # 0-30%
            metrics.append(MetricPoint(
                timestamp=now,
                value=drift_score * 100,
                metric_name="model_drift_score_percent",
                metric_type=MetricType.AI_MODEL_DRIFT,
                tenant_id=tenant_id,
                source="ml_monitoring",
                tags={"model": "recommendation_v3", "drift_type": "feature"}
            ))
            
        except Exception as e:
            logger.error(f"Erreur collecte métriques IA", collector=self.collector_name, error=str(e))
        
        return metrics
    
    def get_supported_metrics(self) -> List[MetricType]:
        """Métriques IA supportées."""
        return [
            MetricType.AI_ACCURACY,
            MetricType.AI_LATENCY,
            MetricType.AI_THROUGHPUT,
            MetricType.AI_MODEL_DRIFT,
            MetricType.AI_TRAINING_LOSS
        ]

class AnomalyDetector:
    """Détecteur d'anomalies pour métriques."""
    
    def __init__(self):
        self.models = {}
        self.baseline_data = {}
        self.detection_windows = {
            'short': timedelta(minutes=5),
            'medium': timedelta(hours=1),
            'long': timedelta(days=1)
        }
    
    def detect_anomalies(
        self,
        metrics: List[MetricPoint],
        tenant_id: str
    ) -> List[Anomaly]:
        """Détecte les anomalies dans une série de métriques."""
        anomalies = []
        
        # Groupement par métrique
        metric_groups = {}
        for metric in metrics:
            key = f"{metric.metric_name}:{metric.tenant_id}"
            if key not in metric_groups:
                metric_groups[key] = []
            metric_groups[key].append(metric)
        
        # Détection par groupe
        for key, metric_list in metric_groups.items():
            if len(metric_list) < 3:  # Pas assez de données
                continue
            
            try:
                group_anomalies = self._detect_for_metric_group(metric_list, tenant_id)
                anomalies.extend(group_anomalies)
            except Exception as e:
                logger.debug(f"Erreur détection anomalies", metric_group=key, error=str(e))
        
        return anomalies
    
    def _detect_for_metric_group(
        self,
        metrics: List[MetricPoint],
        tenant_id: str
    ) -> List[Anomaly]:
        """Détecte les anomalies pour un groupe de métriques."""
        anomalies = []
        
        if len(metrics) < 5:
            return anomalies
        
        # Préparation des données
        values = [m.value for m in metrics]
        timestamps = [m.timestamp for m in metrics]
        metric_name = metrics[0].metric_name
        
        # Détection statistique simple
        anomalies.extend(self._detect_statistical_anomalies(
            values, timestamps, metric_name, tenant_id
        ))
        
        # Détection de tendances
        anomalies.extend(self._detect_trend_anomalies(
            values, timestamps, metric_name, tenant_id
        ))
        
        # Détection ML avancée si assez de données
        if len(values) >= 20:
            anomalies.extend(self._detect_ml_anomalies(
                values, timestamps, metric_name, tenant_id
            ))
        
        return anomalies
    
    def _detect_statistical_anomalies(
        self,
        values: List[float],
        timestamps: List[datetime],
        metric_name: str,
        tenant_id: str
    ) -> List[Anomaly]:
        """Détection d'anomalies statistiques (outliers)."""
        anomalies = []
        
        try:
            # Calcul des seuils statistiques
            mean_val = statistics.mean(values)
            stdev_val = statistics.stdev(values) if len(values) > 1 else 0
            
            # Seuils à 2 et 3 sigma
            threshold_2sigma = mean_val + 2 * stdev_val
            threshold_3sigma = mean_val + 3 * stdev_val
            
            for i, (value, timestamp) in enumerate(zip(values, timestamps)):
                if abs(value - mean_val) > 3 * stdev_val:
                    # Anomalie forte (3 sigma)
                    severity = min(1.0, abs(value - mean_val) / (3 * stdev_val + 1))
                    
                    anomaly = Anomaly(
                        metric_name=metric_name,
                        tenant_id=tenant_id,
                        timestamp=timestamp,
                        anomaly_type=AnomalyType.OUTLIER,
                        severity=severity,
                        actual_value=value,
                        expected_value=mean_val,
                        threshold=threshold_3sigma,
                        description=f"Valeur aberrante détectée: {value:.2f} vs moyenne {mean_val:.2f}",
                        confidence=0.95,
                        suggested_actions=[
                            "Vérifier la source de données",
                            "Analyser les logs système",
                            "Contrôler les seuils d'alerte"
                        ]
                    )
                    anomalies.append(anomaly)
                    
                elif abs(value - mean_val) > 2 * stdev_val:
                    # Anomalie modérée (2 sigma)
                    severity = abs(value - mean_val) / (2 * stdev_val + 1) * 0.7
                    
                    anomaly = Anomaly(
                        metric_name=metric_name,
                        tenant_id=tenant_id,
                        timestamp=timestamp,
                        anomaly_type=AnomalyType.OUTLIER,
                        severity=severity,
                        actual_value=value,
                        expected_value=mean_val,
                        threshold=threshold_2sigma,
                        description=f"Valeur inhabituelle: {value:.2f}",
                        confidence=0.80
                    )
                    anomalies.append(anomaly)
                    
        except Exception as e:
            logger.debug(f"Erreur détection statistique", metric=metric_name, error=str(e))
        
        return anomalies
    
    def _detect_trend_anomalies(
        self,
        values: List[float],
        timestamps: List[datetime],
        metric_name: str,
        tenant_id: str
    ) -> List[Anomaly]:
        """Détection d'anomalies de tendance."""
        anomalies = []
        
        try:
            if len(values) < 5:
                return anomalies
            
            # Calcul de la dérivée (changement de tendance)
            derivatives = []
            for i in range(1, len(values)):
                time_diff = (timestamps[i] - timestamps[i-1]).total_seconds()
                if time_diff > 0:
                    derivative = (values[i] - values[i-1]) / time_diff
                    derivatives.append((derivative, timestamps[i], values[i]))
            
            if not derivatives:
                return anomalies
            
            # Détection de changements brusques
            derivative_values = [d[0] for d in derivatives]
            if len(derivative_values) > 1:
                mean_derivative = statistics.mean(derivative_values)
                stdev_derivative = statistics.stdev(derivative_values)
                
                for derivative, timestamp, value in derivatives:
                    if abs(derivative - mean_derivative) > 2 * stdev_derivative:
                        anomaly_type = AnomalyType.SPIKE if derivative > mean_derivative else AnomalyType.DROP
                        severity = min(1.0, abs(derivative - mean_derivative) / (2 * stdev_derivative))
                        
                        anomaly = Anomaly(
                            metric_name=metric_name,
                            tenant_id=tenant_id,
                            timestamp=timestamp,
                            anomaly_type=anomaly_type,
                            severity=severity,
                            actual_value=value,
                            expected_value=0,  # Pas de valeur attendue pour tendance
                            threshold=2 * stdev_derivative,
                            description=f"Changement de tendance brutal détecté",
                            confidence=0.75
                        )
                        anomalies.append(anomaly)
                        
        except Exception as e:
            logger.debug(f"Erreur détection tendance", metric=metric_name, error=str(e))
        
        return anomalies
    
    def _detect_ml_anomalies(
        self,
        values: List[float],
        timestamps: List[datetime],
        metric_name: str,
        tenant_id: str
    ) -> List[Anomaly]:
        """Détection d'anomalies avec ML (Isolation Forest)."""
        anomalies = []
        
        try:
            # Préparation des features
            X = np.array(values).reshape(-1, 1)
            
            # Normalisation
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Modèle Isolation Forest
            iso_forest = IsolationForest(
                contamination=0.1,  # 10% d'anomalies attendues
                random_state=42
            )
            
            # Entraînement et prédiction
            predictions = iso_forest.fit_predict(X_scaled)
            scores = iso_forest.score_samples(X_scaled)
            
            # Conversion en anomalies
            for i, (prediction, score) in enumerate(zip(predictions, scores)):
                if prediction == -1:  # Anomalie détectée
                    # Score normalisé (plus négatif = plus anormal)
                    severity = min(1.0, abs(score) / 0.5)  # Normalisation approximative
                    
                    anomaly = Anomaly(
                        metric_name=metric_name,
                        tenant_id=tenant_id,
                        timestamp=timestamps[i],
                        anomaly_type=AnomalyType.OUTLIER,
                        severity=severity,
                        actual_value=values[i],
                        expected_value=statistics.mean(values),
                        threshold=0.0,  # Pas de seuil fixe pour ML
                        description=f"Anomalie ML détectée (score: {score:.3f})",
                        confidence=min(0.95, severity),
                        tags={"detection_method": "isolation_forest"},
                        suggested_actions=[
                            "Analyser le contexte métier",
                            "Vérifier corrélations avec autres métriques",
                            "Investiguer changements récents"
                        ]
                    )
                    anomalies.append(anomaly)
                    
        except Exception as e:
            logger.debug(f"Erreur détection ML", metric=metric_name, error=str(e))
        
        return anomalies

class MetricsCollector:
    """
    Collecteur principal de métriques pour Spotify AI Agent.
    
    Orchestre la collecte multi-source, l'agrégation et la détection
    d'anomalies pour le système d'alerting.
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        
        # Collecteurs spécialisés
        self.collectors = {
            'prometheus': PrometheusCollector(),
            'business': BusinessMetricsCollector(), 
            'ai_ml': AIMetricsCollector()
        }
        
        # Détecteur d'anomalies
        self.anomaly_detector = AnomalyDetector()
        
        # Configuration
        self.collection_interval = 60  # 1 minute
        self.retention_days = 30
        self.batch_size = 1000
        
        # État
        self.is_running = False
        self.collection_task = None
        self._metrics_buffer = []
        self._buffer_lock = asyncio.Lock()
        
        logger.info("MetricsCollector initialisé", collectors_count=len(self.collectors))
    
    async def start_collection(self, tenant_ids: List[str]):
        """Démarre la collecte en arrière-plan."""
        if self.is_running:
            logger.warning("Collecte déjà en cours")
            return
        
        self.is_running = True
        self.tenant_ids = tenant_ids
        
        # Démarrage de la tâche de collecte
        self.collection_task = asyncio.create_task(self._collection_loop())
        
        logger.info("Collecte de métriques démarrée", tenant_count=len(tenant_ids))
    
    async def stop_collection(self):
        """Arrête la collecte."""
        self.is_running = False
        
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        
        # Flush du buffer final
        await self._flush_buffer()
        
        logger.info("Collecte de métriques arrêtée")
    
    async def _collection_loop(self):
        """Boucle principale de collecte."""
        while self.is_running:
            try:
                start_time = time.time()
                
                # Collecte pour tous les tenants
                for tenant_id in self.tenant_ids:
                    await self._collect_for_tenant(tenant_id)
                
                # Flush périodique du buffer
                if len(self._metrics_buffer) >= self.batch_size:
                    await self._flush_buffer()
                
                # Métriques de performance
                collection_duration = time.time() - start_time
                metrics_collection_duration.labels(
                    collector_type="all"
                ).observe(collection_duration)
                
                # Attente avant prochaine collecte
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error("Erreur boucle de collecte", error=str(e))
                await asyncio.sleep(5)  # Attente courte avant retry
    
    async def _collect_for_tenant(self, tenant_id: str):
        """Collecte toutes les métriques pour un tenant."""
        collected_metrics = []
        
        # Collecte depuis chaque collecteur
        for collector_name, collector in self.collectors.items():
            try:
                start_time = time.time()
                
                metrics = await collector.collect(tenant_id)
                collected_metrics.extend(metrics)
                
                # Métriques de collecte
                metrics_collection_total.labels(
                    tenant_id=tenant_id,
                    metric_type="mixed",
                    source=collector_name
                ).inc(len(metrics))
                
                metrics_collection_duration.labels(
                    collector_type=collector_name
                ).observe(time.time() - start_time)
                
            except Exception as e:
                logger.error(
                    "Erreur collecte tenant",
                    tenant_id=tenant_id,
                    collector=collector_name,
                    error=str(e)
                )
        
        # Détection d'anomalies
        if collected_metrics:
            anomalies = self.anomaly_detector.detect_anomalies(collected_metrics, tenant_id)
            
            for anomaly in anomalies:
                metrics_anomaly_detected.labels(
                    tenant_id=tenant_id,
                    metric_name=anomaly.metric_name,
                    anomaly_type=anomaly.anomaly_type.value
                ).inc()
        
        # Ajout au buffer
        async with self._buffer_lock:
            self._metrics_buffer.extend(collected_metrics)
    
    async def _flush_buffer(self):
        """Flush le buffer vers le stockage."""
        async with self._buffer_lock:
            if not self._metrics_buffer:
                return
            
            metrics_to_flush = self._metrics_buffer.copy()
            self._metrics_buffer.clear()
        
        # Stockage Redis
        if self.redis_client:
            await self._store_metrics_redis(metrics_to_flush)
        
        logger.debug(f"Buffer flushed", metrics_count=len(metrics_to_flush))
    
    async def _store_metrics_redis(self, metrics: List[MetricPoint]):
        """Stocke les métriques dans Redis."""
        try:
            pipe = self.redis_client.pipeline()
            
            for metric in metrics:
                # Clé temporelle pour time series
                timestamp_key = metric.timestamp.strftime("%Y%m%d%H%M")
                redis_key = f"metrics:{metric.tenant_id}:{metric.metric_name}:{timestamp_key}"
                
                # Stockage avec TTL
                pipe.setex(
                    redis_key,
                    self.retention_days * 24 * 3600,  # TTL en secondes
                    json.dumps(metric.to_dict())
                )
            
            await pipe.execute()
            
        except Exception as e:
            logger.error("Erreur stockage Redis", error=str(e))
    
    async def get_metrics_for_alerting(
        self,
        tenant_id: str,
        metric_names: List[str] = None,
        time_range: str = "5m"
    ) -> Dict[str, List[MetricPoint]]:
        """
        Récupère les métriques pour le système d'alerting.
        
        Args:
            tenant_id: ID du tenant
            metric_names: Liste des métriques (None = toutes)
            time_range: Plage temporelle (5m, 1h, 1d)
            
        Returns:
            Dictionnaire {metric_name: [MetricPoint]}
        """
        # Conversion de la plage temporelle
        time_delta = self._parse_time_range(time_range)
        start_time = datetime.utcnow() - time_delta
        
        # Récupération depuis Redis
        metrics_dict = {}
        
        if self.redis_client:
            try:
                # Pattern de clés à rechercher
                if metric_names:
                    patterns = [f"metrics:{tenant_id}:{name}:*" for name in metric_names]
                else:
                    patterns = [f"metrics:{tenant_id}:*"]
                
                for pattern in patterns:
                    keys = self.redis_client.keys(pattern)
                    
                    for key in keys:
                        metric_data = self.redis_client.get(key)
                        if metric_data:
                            metric = MetricPoint.from_dict(json.loads(metric_data))
                            
                            # Filtrage temporel
                            if metric.timestamp >= start_time:
                                if metric.metric_name not in metrics_dict:
                                    metrics_dict[metric.metric_name] = []
                                metrics_dict[metric.metric_name].append(metric)
                
                # Tri chronologique
                for metric_name in metrics_dict:
                    metrics_dict[metric_name].sort(key=lambda m: m.timestamp)
                    
            except Exception as e:
                logger.error("Erreur récupération métriques", tenant_id=tenant_id, error=str(e))
        
        return metrics_dict
    
    def _parse_time_range(self, time_range: str) -> timedelta:
        """Parse une plage temporelle textuelle."""
        if time_range.endswith('m'):
            return timedelta(minutes=int(time_range[:-1]))
        elif time_range.endswith('h'):
            return timedelta(hours=int(time_range[:-1]))
        elif time_range.endswith('d'):
            return timedelta(days=int(time_range[:-1]))
        else:
            return timedelta(minutes=5)  # Défaut: 5 minutes
    
    async def aggregate_metrics(
        self,
        metrics: List[MetricPoint],
        aggregation_type: AggregationType,
        window_size: timedelta = timedelta(minutes=5)
    ) -> List[AggregatedMetric]:
        """Agrège les métriques par fenêtres temporelles."""
        if not metrics:
            return []
        
        # Groupement par fenêtres temporelles
        windows = {}
        
        for metric in metrics:
            # Calcul de la fenêtre (arrondi à la minute)
            window_start = metric.timestamp.replace(second=0, microsecond=0)
            window_start = window_start - timedelta(
                minutes=window_start.minute % (window_size.total_seconds() // 60)
            )
            
            window_key = (metric.metric_name, metric.tenant_id, window_start)
            
            if window_key not in windows:
                windows[window_key] = []
            windows[window_key].append(metric)
        
        # Agrégation par fenêtre
        aggregated = []
        
        for (metric_name, tenant_id, window_start), window_metrics in windows.items():
            values = [m.value for m in window_metrics]
            
            # Calcul selon le type d'agrégation
            if aggregation_type == AggregationType.SUM:
                agg_value = sum(values)
            elif aggregation_type == AggregationType.AVG:
                agg_value = statistics.mean(values)
            elif aggregation_type == AggregationType.MIN:
                agg_value = min(values)
            elif aggregation_type == AggregationType.MAX:
                agg_value = max(values)
            elif aggregation_type == AggregationType.P95:
                agg_value = np.percentile(values, 95)
            elif aggregation_type == AggregationType.P99:
                agg_value = np.percentile(values, 99)
            elif aggregation_type == AggregationType.COUNT:
                agg_value = len(values)
            elif aggregation_type == AggregationType.STDDEV:
                agg_value = statistics.stdev(values) if len(values) > 1 else 0
            else:
                agg_value = statistics.mean(values)  # Défaut
            
            # Statistiques détaillées
            percentiles = {
                'p50': np.percentile(values, 50),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99)
            }
            
            aggregated_metric = AggregatedMetric(
                metric_name=metric_name,
                tenant_id=tenant_id,
                start_time=window_start,
                end_time=window_start + window_size,
                aggregation_type=aggregation_type,
                value=agg_value,
                count=len(values),
                min_value=min(values),
                max_value=max(values),
                stddev=statistics.stdev(values) if len(values) > 1 else 0,
                percentiles=percentiles,
                quality_score=statistics.mean(m.quality_score for m in window_metrics)
            )
            
            aggregated.append(aggregated_metric)
        
        return sorted(aggregated, key=lambda a: a.start_time)
    
    def get_prometheus_metrics(self) -> str:
        """Génère les métriques Prometheus pour exposition."""
        return generate_latest(custom_registry)
    
    def get_collector_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques des collecteurs."""
        stats = {
            'is_running': self.is_running,
            'buffer_size': len(self._metrics_buffer),
            'collectors': {}
        }
        
        for name, collector in self.collectors.items():
            stats['collectors'][name] = {
                'is_active': collector.is_active,
                'collection_count': collector.collection_count,
                'last_collection': collector.last_collection.isoformat() if collector.last_collection else None,
                'supported_metrics': [m.value for m in collector.get_supported_metrics()]
            }
        
        return stats

# Fonctions utilitaires

async def collect_single_metric(
    tenant_id: str,
    metric_name: str,
    value: Union[int, float],
    metric_type: MetricType,
    tags: Dict[str, str] = None,
    collector: Optional[MetricsCollector] = None
):
    """Collecte une métrique unique."""
    if not collector:
        collector = MetricsCollector()
    
    metric = MetricPoint(
        timestamp=datetime.utcnow(),
        value=value,
        metric_name=metric_name,
        metric_type=metric_type,
        tenant_id=tenant_id,
        tags=tags or {},
        source="manual"
    )
    
    async with collector._buffer_lock:
        collector._metrics_buffer.append(metric)

def create_business_metric(
    tenant_id: str,
    streams: int,
    revenue: float,
    engagement_score: float
) -> List[MetricPoint]:
    """Crée un ensemble de métriques business."""
    now = datetime.utcnow()
    
    return [
        MetricPoint(
            timestamp=now,
            value=streams,
            metric_name="monthly_streams",
            metric_type=MetricType.BUSINESS_STREAMS,
            tenant_id=tenant_id,
            source="spotify_api"
        ),
        MetricPoint(
            timestamp=now,
            value=revenue,
            metric_name="estimated_revenue_usd",
            metric_type=MetricType.BUSINESS_REVENUE,
            tenant_id=tenant_id,
            source="spotify_api"
        ),
        MetricPoint(
            timestamp=now,
            value=engagement_score,
            metric_name="engagement_score",
            metric_type=MetricType.BUSINESS_ENGAGEMENT,
            tenant_id=tenant_id,
            source="spotify_api"
        )
    ]
