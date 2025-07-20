"""
Collecteur de Métriques Avancé - Spotify AI Agent
=================================================

Système de collecte de métriques intelligent avec capacités d'agrégation,
prédiction et analyse en temps réel pour l'architecture multi-tenant.

Fonctionnalités:
- Collecte de métriques multi-dimensionnelles
- Agrégation intelligente par tenant/service
- Détection d'anomalies en temps réel
- Prédiction de tendances avec ML
- Export vers Prometheus/InfluxDB/TimescaleDB
- Cache adaptatif et optimisation des performances
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import json
import time
import statistics
import numpy as np
from scipy import signal
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import redis.asyncio as redis
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, 
    CollectorRegistry, push_to_gateway
)
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor


class MetricType(Enum):
    """Types de métriques supportées"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    BUSINESS = "business"
    SECURITY = "security"
    PERFORMANCE = "performance"


class AggregationType(Enum):
    """Types d'agrégation pour les métriques"""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    P95 = "p95"
    P99 = "p99"
    COUNT = "count"
    RATE = "rate"


@dataclass
class MetricPoint:
    """Point de métrique avec métadonnées complètes"""
    name: str
    value: Union[float, int]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = field(default_factory=dict)
    tenant_id: str = ""
    service: str = ""
    environment: str = ""
    metric_type: MetricType = MetricType.GAUGE
    unit: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class AggregatedMetric:
    """Métrique agrégée avec statistiques"""
    name: str
    aggregation_type: AggregationType
    value: float
    timestamp: datetime
    window_size: timedelta
    sample_count: int
    labels: Dict[str, str] = field(default_factory=dict)
    percentiles: Dict[str, float] = field(default_factory=dict)
    trend: str = ""  # "rising", "falling", "stable"
    anomaly_score: float = 0.0


class MetricsCollector:
    """Collecteur avancé de métriques avec ML et analytics"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.collection_interval = config.get('collection_interval', 30)
        self.retention_period = timedelta(days=config.get('retention_days', 30))
        self.batch_size = config.get('batch_size', 1000)
        self.max_memory_usage = config.get('max_memory_mb', 512) * 1024 * 1024
        
        # Stockage des métriques
        self.metrics_buffer: deque = deque(maxlen=100000)
        self.aggregated_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.metric_schemas: Dict[str, Dict] = {}
        
        # Cache et index
        self.redis_client = None
        self.last_collection_time = {}
        self.collection_stats = defaultdict(int)
        
        # Détection d'anomalies
        self.anomaly_detectors: Dict[str, IsolationForest] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.baseline_metrics: Dict[str, List[float]] = defaultdict(list)
        
        # Prometheus
        self.prometheus_registry = CollectorRegistry()
        self.prometheus_metrics: Dict[str, Any] = {}
        
        # Collecteurs système
        self.system_collectors = {}
        self.custom_collectors: List[Callable] = []
        
        # Threading pour les tâches intensives
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Configuration des fenêtres d'agrégation
        self.aggregation_windows = [
            timedelta(minutes=1),
            timedelta(minutes=5),
            timedelta(minutes=15),
            timedelta(hours=1),
            timedelta(hours=6),
            timedelta(days=1)
        ]
        
    async def initialize(self):
        """Initialisation asynchrone du collecteur"""
        try:
            # Connexion Redis
            self.redis_client = redis.from_url(
                self.config.get('redis_url', 'redis://localhost:6379'),
                decode_responses=True
            )
            
            # Initialisation des collecteurs système
            await self._initialize_system_collectors()
            
            # Chargement des schémas de métriques
            await self._load_metric_schemas()
            
            # Initialisation des détecteurs d'anomalies
            await self._initialize_anomaly_detectors()
            
            # Démarrage des tâches de collecte
            asyncio.create_task(self._continuous_collection())
            asyncio.create_task(self._periodic_aggregation())
            asyncio.create_task(self._anomaly_detection_task())
            asyncio.create_task(self._cleanup_task())
            asyncio.create_task(self._prometheus_export_task())
            
            self.logger.info("MetricsCollector initialisé avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation: {e}")
            raise
    
    async def collect_metric(self, metric: MetricPoint) -> bool:
        """Collection d'une métrique individuelle"""
        try:
            # Validation du schéma
            if not await self._validate_metric(metric):
                return False
            
            # Enrichissement automatique
            await self._enrich_metric(metric)
            
            # Stockage dans le buffer
            self.metrics_buffer.append(metric)
            
            # Mise à jour des statistiques
            self.collection_stats[f"{metric.tenant_id}:{metric.name}"] += 1
            
            # Détection d'anomalie immédiate si critique
            if metric.metric_type in [MetricType.SECURITY, MetricType.PERFORMANCE]:
                await self._check_immediate_anomaly(metric)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la collecte de métrique: {e}")
            return False
    
    async def collect_batch(self, metrics: List[MetricPoint]) -> int:
        """Collection en lot de métriques"""
        successful = 0
        
        for metric in metrics:
            if await self.collect_metric(metric):
                successful += 1
        
        return successful
    
    async def get_metrics(self, 
                         tenant_id: str = None,
                         service: str = None,
                         metric_names: List[str] = None,
                         start_time: datetime = None,
                         end_time: datetime = None,
                         aggregation: AggregationType = None) -> List[Dict[str, Any]]:
        """Récupération de métriques avec filtrage avancé"""
        
        try:
            # Construction de la requête
            query_params = {
                'tenant_id': tenant_id,
                'service': service,
                'metric_names': metric_names,
                'start_time': start_time or datetime.utcnow() - timedelta(hours=1),
                'end_time': end_time or datetime.utcnow(),
                'aggregation': aggregation
            }
            
            # Récupération depuis le cache ou le stockage
            if aggregation:
                return await self._get_aggregated_metrics(query_params)
            else:
                return await self._get_raw_metrics(query_params)
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des métriques: {e}")
            return []
    
    async def create_custom_metric(self, 
                                  name: str,
                                  description: str,
                                  metric_type: MetricType,
                                  labels: List[str] = None,
                                  unit: str = "",
                                  tenant_id: str = "") -> bool:
        """Création d'une métrique personnalisée"""
        
        try:
            schema = {
                'name': name,
                'description': description,
                'type': metric_type.value,
                'labels': labels or [],
                'unit': unit,
                'tenant_id': tenant_id,
                'created_at': datetime.utcnow().isoformat()
            }
            
            self.metric_schemas[name] = schema
            
            # Persistance du schéma
            await self.redis_client.hset(
                'metric_schemas',
                name,
                json.dumps(schema)
            )
            
            # Création de la métrique Prometheus correspondante
            await self._create_prometheus_metric(schema)
            
            self.logger.info(f"Métrique personnalisée créée: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la création de métrique: {e}")
            return False
    
    def register_custom_collector(self, collector_func: Callable):
        """Enregistrement d'un collecteur personnalisé"""
        self.custom_collectors.append(collector_func)
        self.logger.info(f"Collecteur personnalisé enregistré: {collector_func.__name__}")
    
    async def get_tenant_metrics_summary(self, tenant_id: str) -> Dict[str, Any]:
        """Résumé des métriques pour un tenant"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=24)
            
            # Récupération des métriques récentes
            recent_metrics = await self.get_metrics(
                tenant_id=tenant_id,
                start_time=start_time,
                end_time=end_time
            )
            
            # Calcul des statistiques
            summary = {
                'tenant_id': tenant_id,
                'period': '24h',
                'total_metrics': len(recent_metrics),
                'unique_services': len(set(m.get('service', '') for m in recent_metrics)),
                'metric_types': self._count_metric_types(recent_metrics),
                'top_metrics': self._get_top_metrics(recent_metrics),
                'anomalies_detected': await self._get_tenant_anomalies(tenant_id),
                'performance_indicators': await self._calculate_performance_indicators(tenant_id),
                'trends': await self._calculate_metric_trends(tenant_id),
                'health_score': await self._calculate_health_score(tenant_id)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul du résumé: {e}")
            return {}
    
    async def detect_anomalies(self, metric_name: str, tenant_id: str = None) -> List[Dict[str, Any]]:
        """Détection d'anomalies pour une métrique spécifique"""
        try:
            key = f"{tenant_id}:{metric_name}" if tenant_id else metric_name
            
            if key not in self.anomaly_detectors:
                await self._train_anomaly_detector(key)
            
            # Récupération des données récentes
            recent_data = await self._get_recent_metric_values(metric_name, tenant_id)
            
            if len(recent_data) < 10:
                return []
            
            # Détection
            detector = self.anomaly_detectors[key]
            scaler = self.scalers[key]
            
            scaled_data = scaler.transform(np.array(recent_data).reshape(-1, 1))
            anomaly_scores = detector.decision_function(scaled_data)
            predictions = detector.predict(scaled_data)
            
            anomalies = []
            for i, (score, prediction) in enumerate(zip(anomaly_scores, predictions)):
                if prediction == -1:  # Anomalie détectée
                    anomalies.append({
                        'timestamp': datetime.utcnow() - timedelta(minutes=i),
                        'value': recent_data[-(i+1)],
                        'anomaly_score': float(score),
                        'severity': self._classify_anomaly_severity(score)
                    })
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la détection d'anomalies: {e}")
            return []
    
    async def _continuous_collection(self):
        """Collecte continue des métriques système"""
        while True:
            try:
                start_time = time.time()
                
                # Collecte des métriques système
                await self._collect_system_metrics()
                
                # Collecte des métriques personnalisées
                await self._collect_custom_metrics()
                
                # Flush du buffer si nécessaire
                if len(self.metrics_buffer) > self.batch_size:
                    await self._flush_metrics_buffer()
                
                # Contrôle de la mémoire
                await self._manage_memory_usage()
                
                collection_time = time.time() - start_time
                sleep_time = max(0, self.collection_interval - collection_time)
                
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Erreur dans la collecte continue: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _periodic_aggregation(self):
        """Agrégation périodique des métriques"""
        while True:
            try:
                await asyncio.sleep(300)  # Agrégation toutes les 5 minutes
                
                for window in self.aggregation_windows:
                    await self._aggregate_metrics_for_window(window)
                
            except Exception as e:
                self.logger.error(f"Erreur dans l'agrégation périodique: {e}")
    
    async def _collect_system_metrics(self):
        """Collection des métriques système"""
        timestamp = datetime.utcnow()
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=None)
        await self.collect_metric(MetricPoint(
            name="system_cpu_percent",
            value=cpu_percent,
            timestamp=timestamp,
            metric_type=MetricType.PERFORMANCE,
            unit="percent",
            description="CPU utilization percentage"
        ))
        
        # Mémoire
        memory = psutil.virtual_memory()
        await self.collect_metric(MetricPoint(
            name="system_memory_percent",
            value=memory.percent,
            timestamp=timestamp,
            metric_type=MetricType.PERFORMANCE,
            unit="percent",
            description="Memory utilization percentage"
        ))
        
        # Disque
        disk = psutil.disk_usage('/')
        await self.collect_metric(MetricPoint(
            name="system_disk_percent",
            value=disk.percent,
            timestamp=timestamp,
            metric_type=MetricType.PERFORMANCE,
            unit="percent",
            description="Disk utilization percentage"
        ))
        
        # Réseau (si disponible)
        try:
            network = psutil.net_io_counters()
            await self.collect_metric(MetricPoint(
                name="system_network_bytes_sent",
                value=network.bytes_sent,
                timestamp=timestamp,
                metric_type=MetricType.COUNTER,
                unit="bytes",
                description="Total bytes sent over network"
            ))
        except:
            pass  # Net IO stats might not be available
    
    async def _anomaly_detection_task(self):
        """Tâche de détection d'anomalies en arrière-plan"""
        while True:
            try:
                await asyncio.sleep(600)  # Vérification toutes les 10 minutes
                
                # Mise à jour des détecteurs d'anomalies
                for key in list(self.anomaly_detectors.keys()):
                    await self._update_anomaly_detector(key)
                
            except Exception as e:
                self.logger.error(f"Erreur dans la détection d'anomalies: {e}")
    
    async def _validate_metric(self, metric: MetricPoint) -> bool:
        """Validation d'une métrique selon son schéma"""
        if metric.name in self.metric_schemas:
            schema = self.metric_schemas[metric.name]
            
            # Vérification du type
            expected_type = MetricType(schema['type'])
            if metric.metric_type != expected_type:
                return False
            
            # Vérification des labels requis
            required_labels = schema.get('labels', [])
            for label in required_labels:
                if label not in metric.labels:
                    return False
        
        # Validation générale
        return (metric.name and 
                isinstance(metric.value, (int, float)) and
                not np.isnan(metric.value) and
                not np.isinf(metric.value))
    
    async def _enrich_metric(self, metric: MetricPoint):
        """Enrichissement automatique d'une métrique"""
        # Ajout de métadonnées système
        if not metric.labels.get('hostname'):
            metric.labels['hostname'] = self.config.get('hostname', 'unknown')
        
        if not metric.labels.get('environment'):
            metric.labels['environment'] = self.config.get('environment', 'dev')
        
        # Ajout de tags automatiques
        if metric.metric_type == MetricType.PERFORMANCE:
            metric.tags.append('performance')
        if metric.metric_type == MetricType.SECURITY:
            metric.tags.append('security')
    
    def _count_metric_types(self, metrics: List[Dict[str, Any]]) -> Dict[str, int]:
        """Comptage des types de métriques"""
        type_counts = defaultdict(int)
        for metric in metrics:
            metric_type = metric.get('metric_type', 'unknown')
            type_counts[metric_type] += 1
        return dict(type_counts)
    
    def _classify_anomaly_severity(self, score: float) -> str:
        """Classification de la sévérité d'une anomalie"""
        if score < -0.5:
            return "critical"
        elif score < -0.3:
            return "high"
        elif score < -0.1:
            return "medium"
        else:
            return "low"
