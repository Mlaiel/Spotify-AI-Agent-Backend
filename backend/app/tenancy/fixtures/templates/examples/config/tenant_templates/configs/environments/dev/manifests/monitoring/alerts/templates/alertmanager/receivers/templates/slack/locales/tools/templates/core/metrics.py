"""
Gestionnaire de métriques avancé pour le système de tenancy
Auteur: Fahed Mlaiel - Lead Dev & Architecte IA
"""

import time
import asyncio
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import psutil
import aioredis
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import structlog

logger = structlog.get_logger(__name__)

class MetricType(Enum):
    """Types de métriques supportées"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class MetricStatus(Enum):
    """Statuts des métriques"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    WARNING = "warning"

@dataclass
class MetricDefinition:
    """Définition d'une métrique"""
    name: str
    metric_type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None
    unit: str = "count"
    namespace: str = "spotify_ai"
    subsystem: str = "tenancy"

@dataclass
class MetricValue:
    """Valeur d'une métrique avec métadonnées"""
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tenant_id: Optional[str] = None

class MetricsCollector:
    """Collecteur de métriques avancé"""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        self.metrics: Dict[str, Any] = {}
        self.collectors: Dict[str, Callable] = {}
        self.thresholds: Dict[str, Dict[str, float]] = {}
        
    def register_metric(self, definition: MetricDefinition) -> None:
        """Enregistre une nouvelle métrique"""
        try:
            metric_name = f"{definition.namespace}_{definition.subsystem}_{definition.name}"
            
            if definition.metric_type == MetricType.COUNTER:
                metric = Counter(
                    metric_name,
                    definition.description,
                    definition.labels,
                    registry=self.registry
                )
            elif definition.metric_type == MetricType.GAUGE:
                metric = Gauge(
                    metric_name,
                    definition.description,
                    definition.labels,
                    registry=self.registry
                )
            elif definition.metric_type == MetricType.HISTOGRAM:
                metric = Histogram(
                    metric_name,
                    definition.description,
                    definition.labels,
                    buckets=definition.buckets,
                    registry=self.registry
                )
            
            self.metrics[definition.name] = metric
            logger.info("Métrique enregistrée", metric_name=metric_name)
            
        except Exception as e:
            logger.error("Erreur lors de l'enregistrement de la métrique", error=str(e))

    def record_metric(self, name: str, value: MetricValue) -> None:
        """Enregistre une valeur de métrique"""
        try:
            if name not in self.metrics:
                logger.warning("Métrique non trouvée", metric_name=name)
                return
                
            metric = self.metrics[name]
            
            if hasattr(metric, 'labels'):
                labeled_metric = metric.labels(**value.labels)
                if hasattr(labeled_metric, 'inc'):
                    labeled_metric.inc(value.value)
                elif hasattr(labeled_metric, 'set'):
                    labeled_metric.set(value.value)
                elif hasattr(labeled_metric, 'observe'):
                    labeled_metric.observe(value.value)
            
            self._check_thresholds(name, value)
            
        except Exception as e:
            logger.error("Erreur lors de l'enregistrement de la métrique", error=str(e))

    def _check_thresholds(self, name: str, value: MetricValue) -> None:
        """Vérifie les seuils d'alerte"""
        if name in self.thresholds:
            thresholds = self.thresholds[name]
            
            if "critical" in thresholds and value.value >= thresholds["critical"]:
                logger.critical("Seuil critique dépassé", 
                              metric=name, value=value.value, threshold=thresholds["critical"])
            elif "warning" in thresholds and value.value >= thresholds["warning"]:
                logger.warning("Seuil d'avertissement dépassé", 
                             metric=name, value=value.value, threshold=thresholds["warning"])

class TenancyMetrics:
    """Métriques spécifiques au système de tenancy"""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self._register_default_metrics()
        
    def _register_default_metrics(self):
        """Enregistre les métriques par défaut"""
        metrics_definitions = [
            MetricDefinition(
                name="tenant_requests_total",
                metric_type=MetricType.COUNTER,
                description="Nombre total de requêtes par tenant",
                labels=["tenant_id", "method", "endpoint", "status"]
            ),
            MetricDefinition(
                name="tenant_response_time_seconds",
                metric_type=MetricType.HISTOGRAM,
                description="Temps de réponse des requêtes par tenant",
                labels=["tenant_id", "endpoint"],
                buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
            ),
            MetricDefinition(
                name="tenant_active_connections",
                metric_type=MetricType.GAUGE,
                description="Nombre de connexions actives par tenant",
                labels=["tenant_id"]
            ),
            MetricDefinition(
                name="tenant_storage_usage_bytes",
                metric_type=MetricType.GAUGE,
                description="Utilisation du stockage par tenant",
                labels=["tenant_id", "storage_type"]
            ),
            MetricDefinition(
                name="tenant_api_quota_usage",
                metric_type=MetricType.GAUGE,
                description="Utilisation du quota API par tenant",
                labels=["tenant_id", "quota_type"]
            )
        ]
        
        for definition in metrics_definitions:
            self.collector.register_metric(definition)

    def record_request(self, tenant_id: str, method: str, endpoint: str, 
                      status: int, response_time: float):
        """Enregistre une requête"""
        # Compteur de requêtes
        self.collector.record_metric(
            "tenant_requests_total",
            MetricValue(
                value=1,
                labels={
                    "tenant_id": tenant_id,
                    "method": method,
                    "endpoint": endpoint,
                    "status": str(status)
                },
                tenant_id=tenant_id
            )
        )
        
        # Temps de réponse
        self.collector.record_metric(
            "tenant_response_time_seconds",
            MetricValue(
                value=response_time,
                labels={
                    "tenant_id": tenant_id,
                    "endpoint": endpoint
                },
                tenant_id=tenant_id
            )
        )

    def update_connections(self, tenant_id: str, connections: int):
        """Met à jour le nombre de connexions actives"""
        self.collector.record_metric(
            "tenant_active_connections",
            MetricValue(
                value=connections,
                labels={"tenant_id": tenant_id},
                tenant_id=tenant_id
            )
        )

    def update_storage_usage(self, tenant_id: str, storage_type: str, usage_bytes: int):
        """Met à jour l'utilisation du stockage"""
        self.collector.record_metric(
            "tenant_storage_usage_bytes",
            MetricValue(
                value=usage_bytes,
                labels={
                    "tenant_id": tenant_id,
                    "storage_type": storage_type
                },
                tenant_id=tenant_id
            )
        )

    def update_quota_usage(self, tenant_id: str, quota_type: str, usage_percent: float):
        """Met à jour l'utilisation des quotas"""
        self.collector.record_metric(
            "tenant_api_quota_usage",
            MetricValue(
                value=usage_percent,
                labels={
                    "tenant_id": tenant_id,
                    "quota_type": quota_type
                },
                tenant_id=tenant_id
            )
        )

class SystemMetrics:
    """Métriques système avancées"""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self._register_system_metrics()
        
    def _register_system_metrics(self):
        """Enregistre les métriques système"""
        metrics_definitions = [
            MetricDefinition(
                name="system_cpu_usage_percent",
                metric_type=MetricType.GAUGE,
                description="Utilisation CPU du système",
                labels=["core"]
            ),
            MetricDefinition(
                name="system_memory_usage_bytes",
                metric_type=MetricType.GAUGE,
                description="Utilisation mémoire du système",
                labels=["type"]
            ),
            MetricDefinition(
                name="system_disk_usage_percent",
                metric_type=MetricType.GAUGE,
                description="Utilisation disque du système",
                labels=["mountpoint"]
            ),
            MetricDefinition(
                name="system_network_bytes_total",
                metric_type=MetricType.COUNTER,
                description="Trafic réseau total",
                labels=["interface", "direction"]
            )
        ]
        
        for definition in metrics_definitions:
            self.collector.register_metric(definition)

    async def collect_system_metrics(self):
        """Collecte les métriques système"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
            for i, percent in enumerate(cpu_percent):
                self.collector.record_metric(
                    "system_cpu_usage_percent",
                    MetricValue(value=percent, labels={"core": str(i)})
                )
            
            # Mémoire
            memory = psutil.virtual_memory()
            self.collector.record_metric(
                "system_memory_usage_bytes",
                MetricValue(value=memory.used, labels={"type": "used"})
            )
            self.collector.record_metric(
                "system_memory_usage_bytes",
                MetricValue(value=memory.available, labels={"type": "available"})
            )
            
            # Disque
            disk_usage = psutil.disk_usage('/')
            usage_percent = (disk_usage.used / disk_usage.total) * 100
            self.collector.record_metric(
                "system_disk_usage_percent",
                MetricValue(value=usage_percent, labels={"mountpoint": "/"})
            )
            
            # Réseau
            network = psutil.net_io_counters(pernic=True)
            for interface, stats in network.items():
                self.collector.record_metric(
                    "system_network_bytes_total",
                    MetricValue(value=stats.bytes_sent, 
                              labels={"interface": interface, "direction": "sent"})
                )
                self.collector.record_metric(
                    "system_network_bytes_total",
                    MetricValue(value=stats.bytes_recv, 
                              labels={"interface": interface, "direction": "recv"})
                )
                
        except Exception as e:
            logger.error("Erreur lors de la collecte des métriques système", error=str(e))

class MetricsAggregator:
    """Agrégateur de métriques avec calculs avancés"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        
    async def aggregate_metrics(self, tenant_id: str, 
                              time_window: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """Agrège les métriques pour un tenant"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - time_window
            
            # Récupération des métriques depuis Redis
            metrics_key = f"tenant_metrics:{tenant_id}"
            raw_metrics = await self.redis.zrangebyscore(
                metrics_key,
                start_time.timestamp(),
                end_time.timestamp(),
                withscores=True
            )
            
            # Calculs d'agrégation
            aggregated = {
                "tenant_id": tenant_id,
                "time_window": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "duration_seconds": time_window.total_seconds()
                },
                "requests": {
                    "total": 0,
                    "success_rate": 0.0,
                    "avg_response_time": 0.0,
                    "peak_rps": 0.0
                },
                "resources": {
                    "avg_connections": 0.0,
                    "peak_connections": 0,
                    "storage_usage": 0,
                    "quota_usage": {}
                },
                "performance": {
                    "p50_response_time": 0.0,
                    "p95_response_time": 0.0,
                    "p99_response_time": 0.0
                }
            }
            
            # Traitement des métriques brutes
            response_times = []
            success_count = 0
            total_requests = 0
            
            for metric_data, timestamp in raw_metrics:
                metric = eval(metric_data.decode())  # En production, utiliser json.loads
                
                if metric.get("type") == "request":
                    total_requests += 1
                    if metric.get("status", 0) < 400:
                        success_count += 1
                    response_times.append(metric.get("response_time", 0))
            
            if total_requests > 0:
                aggregated["requests"]["total"] = total_requests
                aggregated["requests"]["success_rate"] = success_count / total_requests
                aggregated["requests"]["avg_response_time"] = sum(response_times) / len(response_times)
                
                # Calcul des percentiles
                response_times.sort()
                if response_times:
                    aggregated["performance"]["p50_response_time"] = self._percentile(response_times, 50)
                    aggregated["performance"]["p95_response_time"] = self._percentile(response_times, 95)
                    aggregated["performance"]["p99_response_time"] = self._percentile(response_times, 99)
            
            return aggregated
            
        except Exception as e:
            logger.error("Erreur lors de l'agrégation des métriques", error=str(e))
            return {}

    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calcule un percentile"""
        if not data:
            return 0.0
        index = int((percentile / 100) * len(data))
        return data[min(index, len(data) - 1)]

class MetricsExporter:
    """Exportateur de métriques vers différents backends"""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        
    async def export_to_prometheus(self) -> str:
        """Exporte les métriques au format Prometheus"""
        try:
            from prometheus_client import generate_latest
            return generate_latest(self.collector.registry)
        except Exception as e:
            logger.error("Erreur lors de l'export Prometheus", error=str(e))
            return ""

    async def export_to_json(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Exporte les métriques au format JSON"""
        try:
            export_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": {},
                "metadata": {
                    "exporter": "TenancyMetricsExporter",
                    "version": "1.0.0"
                }
            }
            
            if tenant_id:
                export_data["tenant_id"] = tenant_id
                
            # Collecte des métriques actuelles
            for name, metric in self.collector.metrics.items():
                export_data["metrics"][name] = {
                    "type": type(metric).__name__,
                    "description": getattr(metric, '_documentation', ''),
                    "current_value": self._extract_metric_value(metric)
                }
            
            return export_data
            
        except Exception as e:
            logger.error("Erreur lors de l'export JSON", error=str(e))
            return {}

    def _extract_metric_value(self, metric) -> Any:
        """Extrait la valeur actuelle d'une métrique"""
        try:
            if hasattr(metric, '_value'):
                return metric._value._value
            elif hasattr(metric, '_upper_bounds'):
                return dict(metric._buckets)
            return "N/A"
        except:
            return "N/A"

# Instance globale du système de métriques
metrics_collector = MetricsCollector()
tenancy_metrics = TenancyMetrics(metrics_collector)
system_metrics = SystemMetrics(metrics_collector)

# Configuration des seuils par défaut
metrics_collector.thresholds = {
    "tenant_response_time_seconds": {
        "warning": 2.0,
        "critical": 5.0
    },
    "tenant_active_connections": {
        "warning": 1000,
        "critical": 2000
    },
    "system_cpu_usage_percent": {
        "warning": 80.0,
        "critical": 95.0
    },
    "system_memory_usage_bytes": {
        "warning": 85.0,
        "critical": 95.0
    }
}
