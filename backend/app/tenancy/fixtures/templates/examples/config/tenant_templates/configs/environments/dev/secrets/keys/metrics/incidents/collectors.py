"""
Collecteurs de Métriques Avancés - Architecture de Monitoring Enterprise
========================================================================

Système de collecte de métriques multi-dimensionnel avec:
- Collecte en temps réel haute performance
- Agrégation intelligente et sampling adaptatif
- Détection d'anomalies par ML
- Corrélation cross-metrics
- Export vers multiple backends (Prometheus, InfluxDB, etc.)

Collecteurs spécialisés:
    - RealTimeMetricsCollector: Métriques temps réel
    - BusinessMetricsCollector: Métriques métier
    - SecurityMetricsCollector: Métriques de sécurité
    - PerformanceMetricsCollector: Métriques de performance
    - UserExperienceMetricsCollector: Métriques UX
"""

import asyncio
import json
import time
import statistics
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import structlog
import aioredis
from prometheus_client import Counter, Histogram, Gauge, Summary
import numpy as np
from scipy import stats
import pandas as pd

logger = structlog.get_logger(__name__)

class MetricType(Enum):
    """Types de métriques supportées"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    RATE = "rate"
    PERCENTAGE = "percentage"

class MetricSeverity(Enum):
    """Niveaux de criticité des métriques"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"

@dataclass
class MetricPoint:
    """Point de métrique individuel"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    tenant_id: str = ""
    source: str = ""
    type: MetricType = MetricType.GAUGE
    
    def to_prometheus_format(self) -> str:
        """Conversion au format Prometheus"""
        labels_str = ",".join([f'{k}="{v}"' for k, v in {**self.tags, **self.labels}.items()])
        return f"{self.name}{{{labels_str}}} {self.value} {int(self.timestamp.timestamp() * 1000)}"
    
    def to_influx_format(self) -> str:
        """Conversion au format InfluxDB"""
        tags_str = ",".join([f"{k}={v}" for k, v in self.tags.items()])
        fields_str = f"value={self.value}"
        timestamp_ns = int(self.timestamp.timestamp() * 1_000_000_000)
        return f"{self.name},{tags_str} {fields_str} {timestamp_ns}"

@dataclass
class MetricThreshold:
    """Seuil de métrique avec actions"""
    name: str
    threshold_value: float
    operator: str  # >, <, >=, <=, ==, !=
    severity: MetricSeverity
    action: Optional[Callable] = None
    cooldown_minutes: int = 5
    last_triggered: Optional[datetime] = None

@dataclass
class AnomalyResult:
    """Résultat de détection d'anomalie"""
    metric_name: str
    is_anomaly: bool
    confidence_score: float
    expected_value: float
    actual_value: float
    deviation_score: float
    anomaly_type: str  # "spike", "drop", "trend", "outlier"
    timestamp: datetime

class BaseMetricsCollector(ABC):
    """Classe de base pour tous les collecteurs de métriques"""
    
    def __init__(self, collector_name: str, collection_interval: int = 60):
        self.collector_name = collector_name
        self.collection_interval = collection_interval
        self.metrics_buffer = deque(maxlen=10000)
        self.thresholds = {}
        self.exporters = []
        self.is_running = False
        self.anomaly_detector = None
        self.last_collection = None
        
        # Métriques Prometheus pour ce collecteur
        self.collection_duration = Histogram(
            f'{collector_name}_collection_duration_seconds',
            'Time spent collecting metrics'
        )
        self.metrics_collected = Counter(
            f'{collector_name}_metrics_collected_total',
            'Total number of metrics collected'
        )
        self.collection_errors = Counter(
            f'{collector_name}_collection_errors_total',
            'Total number of collection errors'
        )
    
    @abstractmethod
    async def collect_metrics(self) -> List[MetricPoint]:
        """Collecte des métriques spécifiques au collecteur"""
        pass
    
    async def start_collection(self):
        """Démarrage de la collecte en arrière-plan"""
        self.is_running = True
        asyncio.create_task(self._collection_loop())
        logger.info(f"Collecteur {self.collector_name} démarré")
    
    async def stop_collection(self):
        """Arrêt de la collecte"""
        self.is_running = False
        logger.info(f"Collecteur {self.collector_name} arrêté")
    
    async def _collection_loop(self):
        """Boucle principale de collecte"""
        while self.is_running:
            try:
                start_time = time.time()
                
                # Collecte des métriques
                metrics = await self.collect_metrics()
                
                # Ajout au buffer
                self.metrics_buffer.extend(metrics)
                
                # Vérification des seuils
                await self._check_thresholds(metrics)
                
                # Détection d'anomalies
                if self.anomaly_detector:
                    await self._detect_anomalies(metrics)
                
                # Export des métriques
                await self._export_metrics(metrics)
                
                # Mise à jour des métriques de performance
                collection_time = time.time() - start_time
                self.collection_duration.observe(collection_time)
                self.metrics_collected.inc(len(metrics))
                
                self.last_collection = datetime.utcnow()
                
                # Attente avant la prochaine collecte
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                self.collection_errors.inc()
                logger.error(f"Erreur lors de la collecte {self.collector_name}: {e}")
                await asyncio.sleep(10)  # Pause en cas d'erreur
    
    async def _check_thresholds(self, metrics: List[MetricPoint]):
        """Vérification des seuils configurés"""
        for metric in metrics:
            threshold = self.thresholds.get(metric.name)
            if threshold and self._should_check_threshold(threshold):
                if self._evaluate_threshold(metric.value, threshold):
                    await self._trigger_threshold_action(metric, threshold)
    
    def _should_check_threshold(self, threshold: MetricThreshold) -> bool:
        """Vérification du cooldown des seuils"""
        if threshold.last_triggered is None:
            return True
        
        time_since_last = datetime.utcnow() - threshold.last_triggered
        return time_since_last.total_seconds() >= threshold.cooldown_minutes * 60
    
    def _evaluate_threshold(self, value: float, threshold: MetricThreshold) -> bool:
        """Évaluation d'un seuil"""
        operators = {
            ">": lambda v, t: v > t,
            "<": lambda v, t: v < t,
            ">=": lambda v, t: v >= t,
            "<=": lambda v, t: v <= t,
            "==": lambda v, t: v == t,
            "!=": lambda v, t: v != t
        }
        
        operator_func = operators.get(threshold.operator)
        if operator_func:
            return operator_func(value, threshold.threshold_value)
        return False
    
    async def _trigger_threshold_action(self, metric: MetricPoint, threshold: MetricThreshold):
        """Déclenchement de l'action associée à un seuil"""
        threshold.last_triggered = datetime.utcnow()
        
        logger.warning(
            f"Seuil dépassé pour {metric.name}",
            metric_value=metric.value,
            threshold_value=threshold.threshold_value,
            severity=threshold.severity.value
        )
        
        if threshold.action:
            try:
                await threshold.action(metric, threshold)
            except Exception as e:
                logger.error(f"Erreur lors de l'exécution de l'action de seuil: {e}")

class RealTimeMetricsCollector(BaseMetricsCollector):
    """Collecteur de métriques en temps réel haute performance"""
    
    def __init__(self, redis_client: aioredis.Redis):
        super().__init__("realtime", collection_interval=5)  # 5 secondes
        self.redis = redis_client
        self.websocket_connections = set()
        self.stream_buffer = deque(maxlen=1000)
        
        # Métriques système critiques
        self.system_metrics = [
            "cpu_usage_percent",
            "memory_usage_percent", 
            "disk_usage_percent",
            "network_io_bytes",
            "active_connections",
            "response_time_ms",
            "error_rate_percent",
            "throughput_rps"
        ]
    
    async def collect_metrics(self) -> List[MetricPoint]:
        """Collecte des métriques système en temps réel"""
        metrics = []
        timestamp = datetime.utcnow()
        
        # Métriques CPU
        cpu_usage = await self._get_cpu_usage()
        metrics.append(MetricPoint(
            name="system_cpu_usage_percent",
            value=cpu_usage,
            timestamp=timestamp,
            tags={"collector": "realtime", "type": "system"}
        ))
        
        # Métriques mémoire
        memory_usage = await self._get_memory_usage()
        metrics.append(MetricPoint(
            name="system_memory_usage_percent",
            value=memory_usage,
            timestamp=timestamp,
            tags={"collector": "realtime", "type": "system"}
        ))
        
        # Métriques réseau
        network_stats = await self._get_network_stats()
        for stat_name, value in network_stats.items():
            metrics.append(MetricPoint(
                name=f"system_network_{stat_name}",
                value=value,
                timestamp=timestamp,
                tags={"collector": "realtime", "type": "network"}
            ))
        
        # Métriques application
        app_metrics = await self._get_application_metrics()
        metrics.extend(app_metrics)
        
        # Diffusion en temps réel
        await self._stream_metrics(metrics)
        
        return metrics
    
    async def _get_cpu_usage(self) -> float:
        """Récupération de l'utilisation CPU"""
        # Simulation - à remplacer par psutil ou API système
        import random
        return random.uniform(10, 90)
    
    async def _get_memory_usage(self) -> float:
        """Récupération de l'utilisation mémoire"""
        # Simulation - à remplacer par psutil
        import random
        return random.uniform(30, 80)
    
    async def _get_network_stats(self) -> Dict[str, float]:
        """Récupération des statistiques réseau"""
        # Simulation - à remplacer par API système
        import random
        return {
            "bytes_sent": random.uniform(1000000, 10000000),
            "bytes_recv": random.uniform(1000000, 10000000),
            "packets_sent": random.uniform(1000, 10000),
            "packets_recv": random.uniform(1000, 10000)
        }
    
    async def _get_application_metrics(self) -> List[MetricPoint]:
        """Collecte des métriques applicatives"""
        metrics = []
        timestamp = datetime.utcnow()
        
        # Métriques Redis
        redis_info = await self.redis.info()
        metrics.append(MetricPoint(
            name="redis_connected_clients",
            value=float(redis_info.get("connected_clients", 0)),
            timestamp=timestamp,
            tags={"service": "redis", "type": "connection"}
        ))
        
        # Métriques de performance des requêtes
        # Ces métriques seraient normalement collectées via des middleware
        import random
        metrics.extend([
            MetricPoint(
                name="http_request_duration_ms",
                value=random.uniform(10, 500),
                timestamp=timestamp,
                tags={"endpoint": "/api/v1/metrics", "method": "GET"}
            ),
            MetricPoint(
                name="http_requests_total",
                value=random.randint(100, 1000),
                timestamp=timestamp,
                tags={"status": "200", "method": "GET"},
                type=MetricType.COUNTER
            )
        ])
        
        return metrics
    
    async def _stream_metrics(self, metrics: List[MetricPoint]):
        """Diffusion des métriques en temps réel via WebSocket"""
        if not self.websocket_connections:
            return
            
        # Préparation des données pour streaming
        stream_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "tags": m.tags
                } for m in metrics
            ]
        }
        
        # Diffusion vers tous les clients connectés
        disconnected = set()
        for ws in self.websocket_connections:
            try:
                await ws.send_text(json.dumps(stream_data))
            except Exception:
                disconnected.add(ws)
        
        # Nettoyage des connexions fermées
        self.websocket_connections -= disconnected

class BusinessMetricsCollector(BaseMetricsCollector):
    """Collecteur de métriques métier spécialisé"""
    
    def __init__(self, db_session):
        super().__init__("business", collection_interval=300)  # 5 minutes
        self.db = db_session
        self.business_kpis = {}
        self.revenue_metrics = {}
        self.user_metrics = {}
        
    async def collect_metrics(self) -> List[MetricPoint]:
        """Collecte des métriques métier"""
        metrics = []
        timestamp = datetime.utcnow()
        
        # Métriques de revenus
        revenue_metrics = await self._collect_revenue_metrics(timestamp)
        metrics.extend(revenue_metrics)
        
        # Métriques utilisateurs
        user_metrics = await self._collect_user_metrics(timestamp)
        metrics.extend(user_metrics)
        
        # Métriques de conversion
        conversion_metrics = await self._collect_conversion_metrics(timestamp)
        metrics.extend(conversion_metrics)
        
        # KPIs métier personnalisés
        custom_kpis = await self._collect_custom_kpis(timestamp)
        metrics.extend(custom_kpis)
        
        return metrics
    
    async def _collect_revenue_metrics(self, timestamp: datetime) -> List[MetricPoint]:
        """Collecte des métriques de revenus"""
        metrics = []
        
        # Simulation des requêtes de revenus
        # En réalité, ces données viendraient de la base de données
        import random
        
        metrics.extend([
            MetricPoint(
                name="business_revenue_total",
                value=random.uniform(10000, 50000),
                timestamp=timestamp,
                tags={"period": "daily", "currency": "EUR"}
            ),
            MetricPoint(
                name="business_revenue_per_user",
                value=random.uniform(5, 25),
                timestamp=timestamp,
                tags={"period": "daily", "currency": "EUR"}
            ),
            MetricPoint(
                name="business_subscription_count",
                value=random.randint(1000, 5000),
                timestamp=timestamp,
                tags={"type": "active", "tier": "premium"}
            )
        ])
        
        return metrics
    
    async def _collect_user_metrics(self, timestamp: datetime) -> List[MetricPoint]:
        """Collecte des métriques utilisateurs"""
        metrics = []
        import random
        
        metrics.extend([
            MetricPoint(
                name="business_active_users",
                value=random.randint(10000, 50000),
                timestamp=timestamp,
                tags={"period": "daily", "type": "unique"}
            ),
            MetricPoint(
                name="business_new_registrations",
                value=random.randint(100, 500),
                timestamp=timestamp,
                tags={"period": "daily", "source": "organic"}
            ),
            MetricPoint(
                name="business_user_retention_rate",
                value=random.uniform(0.7, 0.95),
                timestamp=timestamp,
                tags={"period": "weekly", "cohort": "new_users"}
            )
        ])
        
        return metrics

class SecurityMetricsCollector(BaseMetricsCollector):
    """Collecteur de métriques de sécurité"""
    
    def __init__(self):
        super().__init__("security", collection_interval=60)  # 1 minute
        self.threat_feeds = []
        self.security_events = deque(maxlen=1000)
        self.blocked_ips = set()
        
    async def collect_metrics(self) -> List[MetricPoint]:
        """Collecte des métriques de sécurité"""
        metrics = []
        timestamp = datetime.utcnow()
        
        # Métriques d'authentification
        auth_metrics = await self._collect_auth_metrics(timestamp)
        metrics.extend(auth_metrics)
        
        # Métriques de menaces
        threat_metrics = await self._collect_threat_metrics(timestamp)
        metrics.extend(threat_metrics)
        
        # Métriques de conformité
        compliance_metrics = await self._collect_compliance_metrics(timestamp)
        metrics.extend(compliance_metrics)
        
        return metrics
    
    async def _collect_auth_metrics(self, timestamp: datetime) -> List[MetricPoint]:
        """Métriques d'authentification et accès"""
        import random
        
        return [
            MetricPoint(
                name="security_failed_logins",
                value=random.randint(0, 50),
                timestamp=timestamp,
                tags={"period": "last_hour", "type": "failed"},
                type=MetricType.COUNTER
            ),
            MetricPoint(
                name="security_suspicious_activities",
                value=random.randint(0, 10),
                timestamp=timestamp,
                tags={"severity": "medium", "type": "anomaly"}
            ),
            MetricPoint(
                name="security_blocked_ips_count",
                value=len(self.blocked_ips),
                timestamp=timestamp,
                tags={"type": "active_blocks"}
            )
        ]

class AnomalyDetectionEngine:
    """Moteur de détection d'anomalies par ML"""
    
    def __init__(self):
        self.models = {}
        self.baseline_data = defaultdict(list)
        self.detection_sensitivity = 0.95
        self.min_data_points = 50
        
    async def train_baseline(self, metric_name: str, historical_data: List[float]):
        """Entraînement de la baseline pour une métrique"""
        if len(historical_data) >= self.min_data_points:
            # Calcul des statistiques de base
            mean = statistics.mean(historical_data)
            std_dev = statistics.stdev(historical_data)
            
            # Détection des outliers avec IQR
            q1 = np.percentile(historical_data, 25)
            q3 = np.percentile(historical_data, 75)
            iqr = q3 - q1
            
            self.models[metric_name] = {
                "mean": mean,
                "std_dev": std_dev,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "min": min(historical_data),
                "max": max(historical_data),
                "training_size": len(historical_data)
            }
            
            logger.info(f"Baseline entraînée pour {metric_name} avec {len(historical_data)} points")
    
    async def detect_anomaly(self, metric_name: str, value: float, timestamp: datetime) -> AnomalyResult:
        """Détection d'anomalie pour une métrique"""
        model = self.models.get(metric_name)
        
        if not model:
            # Pas assez de données historiques
            return AnomalyResult(
                metric_name=metric_name,
                is_anomaly=False,
                confidence_score=0.0,
                expected_value=value,
                actual_value=value,
                deviation_score=0.0,
                anomaly_type="insufficient_data",
                timestamp=timestamp
            )
        
        # Calcul du z-score
        z_score = abs((value - model["mean"]) / model["std_dev"]) if model["std_dev"] > 0 else 0
        
        # Détection d'outlier par IQR
        is_outlier_iqr = value < (model["q1"] - 1.5 * model["iqr"]) or value > (model["q3"] + 1.5 * model["iqr"])
        
        # Détection par z-score (généralement > 3 est considéré comme anomalie)
        is_outlier_zscore = z_score > 3
        
        # Classification du type d'anomalie
        anomaly_type = "normal"
        if is_outlier_zscore or is_outlier_iqr:
            if value > model["mean"]:
                anomaly_type = "spike"
            else:
                anomaly_type = "drop"
        
        # Calcul de la confiance
        confidence_score = min(z_score / 3.0, 1.0) if z_score > 3 else 0.0
        
        return AnomalyResult(
            metric_name=metric_name,
            is_anomaly=is_outlier_zscore or is_outlier_iqr,
            confidence_score=confidence_score,
            expected_value=model["mean"],
            actual_value=value,
            deviation_score=z_score,
            anomaly_type=anomaly_type,
            timestamp=timestamp
        )

class MetricsAggregator:
    """Agrégateur de métriques avec optimisations de performance"""
    
    def __init__(self):
        self.aggregation_rules = {}
        self.time_windows = [60, 300, 900, 3600]  # 1m, 5m, 15m, 1h
        self.aggregated_data = defaultdict(lambda: defaultdict(dict))
        
    async def aggregate_metrics(self, metrics: List[MetricPoint]) -> Dict[str, Any]:
        """Agrégation des métriques par fenêtres temporelles"""
        aggregated = {}
        
        # Groupement par nom de métrique
        metrics_by_name = defaultdict(list)
        for metric in metrics:
            metrics_by_name[metric.name].append(metric)
        
        # Agrégation pour chaque métrique
        for metric_name, metric_points in metrics_by_name.items():
            aggregated[metric_name] = await self._aggregate_metric_points(metric_points)
        
        return aggregated
    
    async def _aggregate_metric_points(self, points: List[MetricPoint]) -> Dict[str, Any]:
        """Agrégation des points d'une métrique"""
        if not points:
            return {}
        
        values = [p.value for p in points]
        
        return {
            "count": len(values),
            "sum": sum(values),
            "avg": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
            "percentiles": {
                "p50": np.percentile(values, 50),
                "p90": np.percentile(values, 90),
                "p95": np.percentile(values, 95),
                "p99": np.percentile(values, 99)
            },
            "first_timestamp": min(p.timestamp for p in points),
            "last_timestamp": max(p.timestamp for p in points)
        }

class MetricsExporter:
    """Exporteur de métriques vers différents backends"""
    
    def __init__(self):
        self.exporters = {}
        self.export_formats = ["prometheus", "influxdb", "json", "csv"]
    
    def register_exporter(self, name: str, exporter_func: Callable):
        """Enregistrement d'un exporteur"""
        self.exporters[name] = exporter_func
    
    async def export_metrics(self, metrics: List[MetricPoint], format_name: str) -> str:
        """Export des métriques dans le format spécifié"""
        if format_name == "prometheus":
            return self._export_prometheus(metrics)
        elif format_name == "influxdb":
            return self._export_influxdb(metrics)
        elif format_name == "json":
            return self._export_json(metrics)
        elif format_name == "csv":
            return self._export_csv(metrics)
        else:
            raise ValueError(f"Format d'export non supporté: {format_name}")
    
    def _export_prometheus(self, metrics: List[MetricPoint]) -> str:
        """Export au format Prometheus"""
        lines = []
        for metric in metrics:
            lines.append(metric.to_prometheus_format())
        return "\n".join(lines)
    
    def _export_influxdb(self, metrics: List[MetricPoint]) -> str:
        """Export au format InfluxDB Line Protocol"""
        lines = []
        for metric in metrics:
            lines.append(metric.to_influx_format())
        return "\n".join(lines)
    
    def _export_json(self, metrics: List[MetricPoint]) -> str:
        """Export au format JSON"""
        data = []
        for metric in metrics:
            data.append({
                "name": metric.name,
                "value": metric.value,
                "timestamp": metric.timestamp.isoformat(),
                "tags": metric.tags,
                "labels": metric.labels,
                "tenant_id": metric.tenant_id,
                "source": metric.source,
                "type": metric.type.value
            })
        return json.dumps(data, indent=2)
    
    def _export_csv(self, metrics: List[MetricPoint]) -> str:
        """Export au format CSV"""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # En-têtes
        writer.writerow(["name", "value", "timestamp", "tags", "tenant_id", "source", "type"])
        
        # Données
        for metric in metrics:
            writer.writerow([
                metric.name,
                metric.value,
                metric.timestamp.isoformat(),
                json.dumps(metric.tags),
                metric.tenant_id,
                metric.source,
                metric.type.value
            ])
        
        return output.getvalue()

# Instance globale du gestionnaire de métriques
class MetricsManager:
    """Gestionnaire principal des métriques"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.collectors = {}
        self.aggregator = MetricsAggregator()
        self.exporter = MetricsExporter()
        self.anomaly_detector = AnomalyDetectionEngine()
        self.is_running = False
        
    async def start(self):
        """Démarrage du système de métriques"""
        # Initialisation des collecteurs
        await self._initialize_collectors()
        
        # Démarrage de tous les collecteurs
        for collector in self.collectors.values():
            await collector.start_collection()
        
        self.is_running = True
        logger.info("Système de métriques démarré")
    
    async def stop(self):
        """Arrêt du système de métriques"""
        for collector in self.collectors.values():
            await collector.stop_collection()
        
        self.is_running = False
        logger.info("Système de métriques arrêté")
    
    async def _initialize_collectors(self):
        """Initialisation des collecteurs par défaut"""
        self.collectors["realtime"] = RealTimeMetricsCollector(self.redis)
        self.collectors["business"] = BusinessMetricsCollector(None)  # DB session à injecter
        self.collectors["security"] = SecurityMetricsCollector()
        
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Récupération d'un résumé des métriques"""
        summary = {
            "collectors_status": {
                name: collector.is_running 
                for name, collector in self.collectors.items()
            },
            "last_collection_times": {
                name: collector.last_collection 
                for name, collector in self.collectors.items()
            },
            "total_metrics_collected": sum(
                len(collector.metrics_buffer) 
                for collector in self.collectors.values()
            )
        }
        
        return summary
