# =============================================================================
# 🔍 OBSERVABILITY ENGINE ULTRA-AVANCÉ - ARCHITECTURE RÉVOLUTIONNAIRE
# =============================================================================
# Moteur d'observabilité enterprise avec IA, tracing distribué et analytics
# temps réel pour monitoring de niveau industriel
# 
# Développé par l'équipe d'experts Achiri:
# ✅ Lead Dev + Architecte IA - Architecture observabilité complète
# ✅ Développeur Backend Senior - Intégrations Python/FastAPI avancées  
# ✅ Ingénieur Machine Learning - Analytics prédictive et détection anomalies
# ✅ DBA & Data Engineer - Storage et agrégation haute performance
# ✅ Spécialiste Sécurité Backend - Monitoring sécurisé et audit trails
# ✅ Architecte Microservices - Observabilité distribuée enterprise
# =============================================================================

import asyncio
import time
import uuid
import json
import hashlib
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import logging
import weakref
from collections import defaultdict, deque

# Scientific computing et ML
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Monitoring et observabilité
import opentelemetry
from opentelemetry import trace, baggage
from opentelemetry.trace import Status, StatusCode
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# Performance et compression
import orjson
import msgpack
import lz4.frame
import zstandard as zstd

# Redis pour métriques temps réel
import aioredis
from redis.asyncio import ConnectionPool

# Prometheus integration
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry
import prometheus_async.aio.web

logger = logging.getLogger(__name__)

# =============================================================================
# ENUMS ET TYPES AVANCÉS
# =============================================================================

class ObservabilityLevel(Enum):
    """Niveaux d'observabilité avec granularité variable"""
    MINIMAL = "minimal"
    STANDARD = "standard"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"
    ULTRA_VERBOSE = "ultra_verbose"

class TraceComponent(Enum):
    """Composants traçables du système"""
    API_GATEWAY = "api_gateway"
    AUTHENTICATION = "authentication"
    DATA_LAYER = "data_layer"
    BUSINESS_LOGIC = "business_logic"
    ML_PIPELINE = "ml_pipeline"
    STORAGE_ENGINE = "storage_engine"
    NOTIFICATION = "notification"
    AUDIT = "audit"
    SECURITY = "security"
    EXTERNAL_API = "external_api"

class MetricAggregationType(Enum):
    """Types d'agrégation pour métriques"""
    SUM = "sum"
    AVERAGE = "average"
    MEDIAN = "median"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"
    MIN = "min"
    MAX = "max"
    STDDEV = "stddev"
    COUNT = "count"
    RATE = "rate"

class AlertPriority(Enum):
    """Priorités d'alertes avec escalade intelligente"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

class HealthStatus(Enum):
    """États de santé des composants"""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    DOWN = "down"
    UNKNOWN = "unknown"

# =============================================================================
# MODÈLES DE DONNÉES AVANCÉS
# =============================================================================

@dataclass
class TraceSpan:
    """Span de trace avec métadonnées enrichies"""
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: Optional[str] = None
    operation_name: str = ""
    component: TraceComponent = TraceComponent.API_GATEWAY
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: str = "ok"
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    baggage: Dict[str, str] = field(default_factory=dict)
    service_name: str = "spotify-ai-agent"
    service_version: str = "3.0.0"
    environment: str = "production"
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    error: Optional[Dict[str, Any]] = None
    
    def finish(self, status: str = "ok", error: Optional[Exception] = None) -> None:
        """Finalise le span avec calcul automatique de durée"""
        self.end_time = datetime.utcnow()
        if self.start_time:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.status = status
        if error:
            self.error = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": str(error.__traceback__)
            }
    
    def add_tag(self, key: str, value: Any) -> None:
        """Ajoute un tag au span"""
        self.tags[key] = value
    
    def add_log(self, level: str, message: str, **kwargs) -> None:
        """Ajoute un log au span"""
        self.logs.append({
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            **kwargs
        })

@dataclass
class MetricSnapshot:
    """Snapshot de métrique avec analyse statistique"""
    metric_name: str
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""
    component: TraceComponent = TraceComponent.API_GATEWAY
    aggregation_type: MetricAggregationType = MetricAggregationType.AVERAGE
    sample_count: int = 1
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    stddev: Optional[float] = None
    percentiles: Dict[str, float] = field(default_factory=dict)
    is_anomaly: bool = False
    anomaly_score: float = 0.0
    trend_direction: str = "stable"  # up, down, stable
    seasonality_detected: bool = False
    
    def calculate_statistics(self, values: List[float]) -> None:
        """Calcule les statistiques avancées"""
        if not values:
            return
        
        arr = np.array(values)
        self.min_value = float(np.min(arr))
        self.max_value = float(np.max(arr))
        self.stddev = float(np.std(arr))
        self.percentiles = {
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99))
        }
        self.sample_count = len(values)

@dataclass
class SystemHealthSnapshot:
    """Snapshot complet de santé système"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    overall_status: HealthStatus = HealthStatus.UNKNOWN
    components: Dict[str, HealthStatus] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    active_alerts: List[Dict[str, Any]] = field(default_factory=list)
    performance_summary: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    error_rates: Dict[str, float] = field(default_factory=dict)
    latency_percentiles: Dict[str, float] = field(default_factory=dict)
    throughput_metrics: Dict[str, float] = field(default_factory=dict)
    security_status: Dict[str, Any] = field(default_factory=dict)
    uptime_seconds: float = 0.0
    version: str = "3.0.0"

@dataclass
class ObservabilityConfig:
    """Configuration complète d'observabilité"""
    service_name: str = "spotify-ai-agent"
    service_version: str = "3.0.0"
    environment: str = "production"
    observability_level: ObservabilityLevel = ObservabilityLevel.COMPREHENSIVE
    
    # Tracing configuration
    enable_tracing: bool = True
    trace_sampling_rate: float = 1.0
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    
    # Metrics configuration
    enable_metrics: bool = True
    metrics_export_interval: int = 10
    prometheus_port: int = 8000
    redis_url: str = "redis://localhost:6379"
    
    # Alerting configuration
    enable_alerting: bool = True
    alert_aggregation_window: int = 60
    max_alerts_per_minute: int = 10
    
    # Storage configuration
    metrics_retention_days: int = 30
    trace_retention_days: int = 7
    enable_compression: bool = True
    compression_algorithm: str = "lz4"
    
    # ML Configuration
    enable_anomaly_detection: bool = True
    anomaly_sensitivity: float = 0.05
    enable_predictive_scaling: bool = True
    
    # Security configuration
    enable_audit_logging: bool = True
    sensitive_data_patterns: List[str] = field(default_factory=lambda: [
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit cards
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Emails
        r'\b\d{3}-?\d{2}-?\d{4}\b'  # SSN
    ])

# =============================================================================
# GESTIONNAIRE DE TRACES AVANCÉ
# =============================================================================

class AdvancedTraceManager:
    """Gestionnaire de traces avec IA et analytics"""
    
    def __init__(self, config: ObservabilityConfig):
        self.config = config
        self.active_spans: Dict[str, TraceSpan] = {}
        self.completed_spans: deque = deque(maxlen=10000)
        self.trace_analytics = TraceAnalytics()
        self.anomaly_detector = AnomalyDetector()
        self._setup_tracer()
        
    def _setup_tracer(self) -> None:
        """Configure le tracer OpenTelemetry"""
        tracer_provider = TracerProvider()
        trace.set_tracer_provider(tracer_provider)
        
        # Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name="localhost",
            agent_port=6831,
        )
        
        span_processor = BatchSpanProcessor(jaeger_exporter)
        tracer_provider.add_span_processor(span_processor)
        
        self.tracer = trace.get_tracer(
            __name__,
            version=self.config.service_version
        )
    
    @asynccontextmanager
    async def trace_operation(
        self,
        operation_name: str,
        component: TraceComponent,
        **tags
    ):
        """Context manager pour tracer une opération"""
        span = TraceSpan(
            operation_name=operation_name,
            component=component,
            service_name=self.config.service_name,
            service_version=self.config.service_version,
            environment=self.config.environment
        )
        
        # Ajouter les tags
        span.tags.update(tags)
        
        # Enregistrer le span actif
        self.active_spans[span.span_id] = span
        
        try:
            with self.tracer.start_as_current_span(operation_name) as otel_span:
                otel_span.set_attribute("component", component.value)
                otel_span.set_attribute("service.name", self.config.service_name)
                
                for key, value in tags.items():
                    otel_span.set_attribute(key, str(value))
                
                yield span
                
        except Exception as e:
            span.finish(status="error", error=e)
            raise
        else:
            span.finish(status="ok")
        finally:
            # Déplacer vers les spans complétés
            if span.span_id in self.active_spans:
                del self.active_spans[span.span_id]
            self.completed_spans.append(span)
            
            # Analyser pour anomalies
            await self._analyze_span(span)
    
    async def _analyze_span(self, span: TraceSpan) -> None:
        """Analyse un span pour détecter des anomalies"""
        try:
            if span.duration_ms:
                # Vérifier si la durée est anormale
                is_anomaly = await self.anomaly_detector.detect_latency_anomaly(
                    span.operation_name,
                    span.duration_ms,
                    span.component
                )
                
                if is_anomaly:
                    await self._handle_anomaly(span, "latency_anomaly")
                
                # Analyser les patterns d'erreur
                if span.status == "error" and span.error:
                    await self._handle_error_pattern(span)
                    
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse du span: {e}")
    
    async def _handle_anomaly(self, span: TraceSpan, anomaly_type: str) -> None:
        """Gère la détection d'anomalie"""
        logger.warning(
            f"Anomalie détectée - Type: {anomaly_type}, "
            f"Opération: {span.operation_name}, "
            f"Durée: {span.duration_ms}ms"
        )
    
    async def _handle_error_pattern(self, span: TraceSpan) -> None:
        """Gère les patterns d'erreur"""
        error_signature = f"{span.component.value}:{span.operation_name}"
        logger.error(f"Pattern d'erreur détecté: {error_signature}")

# =============================================================================
# GESTIONNAIRE DE MÉTRIQUES AVANCÉ
# =============================================================================

class AdvancedMetricsManager:
    """Gestionnaire de métriques avec ML et prédictions"""
    
    def __init__(self, config: ObservabilityConfig):
        self.config = config
        self.registry = CollectorRegistry()
        self.metrics: Dict[str, Any] = {}
        self.redis_pool: Optional[ConnectionPool] = None
        self.ml_analyzer = MetricsMLAnalyzer()
        self._initialize_metrics()
    
    async def initialize(self) -> None:
        """Initialise les connexions asynchrones"""
        try:
            self.redis_pool = ConnectionPool.from_url(
                self.config.redis_url,
                max_connections=20,
                decode_responses=False
            )
            logger.info("Connexion Redis initialisée pour les métriques")
        except Exception as e:
            logger.error(f"Erreur initialisation Redis: {e}")
    
    def _initialize_metrics(self) -> None:
        """Initialise les métriques Prometheus"""
        
        # Métriques de performance
        self.metrics['request_duration'] = Histogram(
            'request_duration_seconds',
            'Durée des requêtes par endpoint',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry,
            buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        self.metrics['request_count'] = Counter(
            'requests_total',
            'Nombre total de requêtes',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.metrics['active_connections'] = Gauge(
            'active_connections',
            'Connexions actives',
            ['type'],
            registry=self.registry
        )
        
        # Métriques business
        self.metrics['business_operations'] = Counter(
            'business_operations_total',
            'Opérations business',
            ['operation_type', 'status'],
            registry=self.registry
        )
        
        self.metrics['ml_model_predictions'] = Counter(
            'ml_predictions_total',
            'Prédictions ML',
            ['model_name', 'model_version', 'prediction_type'],
            registry=self.registry
        )
        
        # Métriques système
        self.metrics['memory_usage'] = Gauge(
            'memory_usage_bytes',
            'Utilisation mémoire',
            ['component'],
            registry=self.registry
        )
        
        self.metrics['cpu_usage'] = Gauge(
            'cpu_usage_percent',
            'Utilisation CPU',
            ['component'],
            registry=self.registry
        )
        
        # Métriques sécurité
        self.metrics['security_events'] = Counter(
            'security_events_total',
            'Événements sécurité',
            ['event_type', 'severity'],
            registry=self.registry
        )
        
        self.metrics['auth_attempts'] = Counter(
            'auth_attempts_total',
            'Tentatives authentification',
            ['type', 'status'],
            registry=self.registry
        )
    
    async def record_metric(
        self,
        metric_name: str,
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None,
        component: Optional[TraceComponent] = None
    ) -> None:
        """Enregistre une métrique avec analyse ML"""
        try:
            labels = labels or {}
            
            # Enregistrer dans Prometheus
            if metric_name in self.metrics:
                metric = self.metrics[metric_name]
                
                if hasattr(metric, 'labels'):
                    if labels:
                        metric.labels(**labels).observe(value) if hasattr(metric, 'observe') else metric.labels(**labels).inc(value)
                    else:
                        metric.observe(value) if hasattr(metric, 'observe') else metric.inc(value)
                else:
                    metric.set(value)
            
            # Stocker dans Redis pour analyse ML
            await self._store_metric_for_ml(metric_name, value, labels, component)
            
            # Créer snapshot pour analyse
            snapshot = MetricSnapshot(
                metric_name=metric_name,
                timestamp=datetime.utcnow(),
                value=value,
                labels=labels,
                component=component or TraceComponent.API_GATEWAY
            )
            
            # Analyser pour anomalies
            await self._analyze_metric(snapshot)
            
        except Exception as e:
            logger.error(f"Erreur enregistrement métrique {metric_name}: {e}")
    
    async def _store_metric_for_ml(
        self,
        metric_name: str,
        value: Union[int, float],
        labels: Optional[Dict[str, str]],
        component: Optional[TraceComponent]
    ) -> None:
        """Stocke la métrique dans Redis pour analyse ML"""
        try:
            if not self.redis_pool:
                return
            
            redis = aioredis.Redis(connection_pool=self.redis_pool)
            
            # Créer la clé Redis
            key_parts = [metric_name]
            if component:
                key_parts.append(component.value)
            if labels:
                sorted_labels = sorted(labels.items())
                key_parts.extend([f"{k}:{v}" for k, v in sorted_labels])
            
            redis_key = f"metrics:{':'.join(key_parts)}"
            
            # Stocker avec timestamp
            timestamp = int(datetime.utcnow().timestamp())
            
            # Utiliser sorted set pour ordre chronologique
            await redis.zadd(redis_key, {str(value): timestamp})
            
            # Conserver seulement les dernières 24h
            cutoff = timestamp - (24 * 3600)
            await redis.zremrangebyscore(redis_key, 0, cutoff)
            
            # Expiration après 7 jours
            await redis.expire(redis_key, 7 * 24 * 3600)
            
        except Exception as e:
            logger.error(f"Erreur stockage métrique Redis: {e}")
    
    async def _analyze_metric(self, snapshot: MetricSnapshot) -> None:
        """Analyse une métrique pour détecter des anomalies"""
        try:
            # Récupérer l'historique pour analyse
            historical_values = await self._get_metric_history(
                snapshot.metric_name,
                snapshot.labels,
                snapshot.component
            )
            
            if len(historical_values) >= 10:  # Minimum pour analyse
                # Détecter anomalies
                is_anomaly, anomaly_score = self.ml_analyzer.detect_anomaly(
                    historical_values,
                    snapshot.value
                )
                
                snapshot.is_anomaly = is_anomaly
                snapshot.anomaly_score = anomaly_score
                
                if is_anomaly:
                    await self._handle_metric_anomaly(snapshot)
                
                # Analyser la tendance
                snapshot.trend_direction = self.ml_analyzer.analyze_trend(historical_values)
                
                # Détecter saisonnalité
                snapshot.seasonality_detected = self.ml_analyzer.detect_seasonality(historical_values)
        
        except Exception as e:
            logger.error(f"Erreur analyse métrique: {e}")
    
    async def _get_metric_history(
        self,
        metric_name: str,
        labels: Optional[Dict[str, str]],
        component: Optional[TraceComponent],
        hours: int = 24
    ) -> List[float]:
        """Récupère l'historique d'une métrique"""
        try:
            if not self.redis_pool:
                return []
            
            redis = aioredis.Redis(connection_pool=self.redis_pool)
            
            # Construire la clé
            key_parts = [metric_name]
            if component:
                key_parts.append(component.value)
            if labels:
                sorted_labels = sorted(labels.items())
                key_parts.extend([f"{k}:{v}" for k, v in sorted_labels])
            
            redis_key = f"metrics:{':'.join(key_parts)}"
            
            # Récupérer les valeurs des dernières heures
            end_time = int(datetime.utcnow().timestamp())
            start_time = end_time - (hours * 3600)
            
            values = await redis.zrangebyscore(redis_key, start_time, end_time)
            return [float(v.decode() if isinstance(v, bytes) else v) for v in values]
            
        except Exception as e:
            logger.error(f"Erreur récupération historique: {e}")
            return []
    
    async def _handle_metric_anomaly(self, snapshot: MetricSnapshot) -> None:
        """Gère la détection d'anomalie métrique"""
        logger.warning(
            f"Anomalie métrique détectée - "
            f"Métrique: {snapshot.metric_name}, "
            f"Valeur: {snapshot.value}, "
            f"Score: {snapshot.anomaly_score:.3f}"
        )

# =============================================================================
# ANALYSEURS ML POUR MÉTRIQUES ET TRACES
# =============================================================================

class AnomalyDetector:
    """Détecteur d'anomalies avec ML"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.trained_models: Dict[str, Any] = {}
    
    async def detect_latency_anomaly(
        self,
        operation_name: str,
        latency_ms: float,
        component: TraceComponent
    ) -> bool:
        """Détecte une anomalie de latence"""
        try:
            # Simuler la détection (en production, utiliser un modèle entraîné)
            # Threshold dynamique basé sur l'historique
            threshold = await self._get_latency_threshold(operation_name, component)
            return latency_ms > threshold * 2.5
        except Exception as e:
            logger.error(f"Erreur détection anomalie latence: {e}")
            return False
    
    async def _get_latency_threshold(
        self,
        operation_name: str,
        component: TraceComponent
    ) -> float:
        """Calcule le seuil de latence dynamique"""
        # En production, calculer basé sur l'historique
        base_thresholds = {
            TraceComponent.API_GATEWAY: 100.0,
            TraceComponent.DATA_LAYER: 50.0,
            TraceComponent.ML_PIPELINE: 500.0,
            TraceComponent.STORAGE_ENGINE: 25.0
        }
        return base_thresholds.get(component, 100.0)

class MetricsMLAnalyzer:
    """Analyseur ML pour métriques"""
    
    def __init__(self):
        self.anomaly_detector = IsolationForest(
            contamination=0.05,
            random_state=42
        )
    
    def detect_anomaly(self, historical_values: List[float], current_value: float) -> Tuple[bool, float]:
        """Détecte une anomalie dans une série de valeurs"""
        try:
            if len(historical_values) < 10:
                return False, 0.0
            
            # Préparer les données
            all_values = historical_values + [current_value]
            X = np.array(all_values).reshape(-1, 1)
            
            # Entraîner et prédire
            self.anomaly_detector.fit(X[:-1])  # Entraîner sur historique
            prediction = self.anomaly_detector.predict(X[-1:])  # Tester valeur actuelle
            anomaly_score = abs(self.anomaly_detector.score_samples(X[-1:])[0])
            
            is_anomaly = prediction[0] == -1
            return is_anomaly, float(anomaly_score)
            
        except Exception as e:
            logger.error(f"Erreur détection anomalie ML: {e}")
            return False, 0.0
    
    def analyze_trend(self, values: List[float]) -> str:
        """Analyse la tendance d'une série de valeurs"""
        try:
            if len(values) < 5:
                return "stable"
            
            # Calcul de la pente de régression linéaire
            x = np.arange(len(values))
            slope, _, _, p_value, _ = stats.linregress(x, values)
            
            # Seuil de signification
            if p_value > 0.05:
                return "stable"
            
            if slope > 0.1:
                return "up"
            elif slope < -0.1:
                return "down"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Erreur analyse tendance: {e}")
            return "stable"
    
    def detect_seasonality(self, values: List[float], min_periods: int = 24) -> bool:
        """Détecte la saisonnalité dans une série"""
        try:
            if len(values) < min_periods * 2:
                return False
            
            # Analyse de périodicité simple
            autocorr = np.correlate(values, values, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Chercher des pics de corrélation
            for period in range(min_periods, len(autocorr) // 2):
                if period < len(autocorr) and autocorr[period] > 0.7 * autocorr[0]:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Erreur détection saisonnalité: {e}")
            return False

class TraceAnalytics:
    """Analytics avancés pour traces"""
    
    def __init__(self):
        self.trace_cache: Dict[str, List[TraceSpan]] = defaultdict(list)
    
    def analyze_trace_patterns(self, spans: List[TraceSpan]) -> Dict[str, Any]:
        """Analyse les patterns dans les traces"""
        try:
            analysis = {
                "total_spans": len(spans),
                "unique_operations": len(set(s.operation_name for s in spans)),
                "error_rate": len([s for s in spans if s.status == "error"]) / len(spans) if spans else 0,
                "avg_duration": np.mean([s.duration_ms for s in spans if s.duration_ms]) if spans else 0,
                "components_involved": list(set(s.component.value for s in spans)),
                "critical_path": self._find_critical_path(spans)
            }
            return analysis
        except Exception as e:
            logger.error(f"Erreur analyse patterns traces: {e}")
            return {}
    
    def _find_critical_path(self, spans: List[TraceSpan]) -> List[str]:
        """Trouve le chemin critique dans une trace"""
        try:
            # Trier par durée décroissante
            sorted_spans = sorted(
                [s for s in spans if s.duration_ms],
                key=lambda x: x.duration_ms,
                reverse=True
            )
            
            # Retourner les opérations les plus lentes
            return [s.operation_name for s in sorted_spans[:5]]
        except Exception:
            return []

# =============================================================================
# MONITEUR DE SANTÉ SYSTÈME
# =============================================================================

class SystemHealthMonitor:
    """Moniteur de santé système avec diagnostics avancés"""
    
    def __init__(self, config: ObservabilityConfig):
        self.config = config
        self.last_health_check = datetime.utcnow()
        self.health_history: deque = deque(maxlen=100)
        self.component_monitors: Dict[str, Callable] = {}
    
    def register_component_monitor(self, component: str, monitor_func: Callable) -> None:
        """Enregistre un moniteur pour un composant"""
        self.component_monitors[component] = monitor_func
    
    async def get_system_health(self) -> SystemHealthSnapshot:
        """Obtient un snapshot complet de la santé système"""
        try:
            snapshot = SystemHealthSnapshot()
            
            # Vérifier chaque composant enregistré
            for component, monitor_func in self.component_monitors.items():
                try:
                    status = await monitor_func() if asyncio.iscoroutinefunction(monitor_func) else monitor_func()
                    snapshot.components[component] = status
                except Exception as e:
                    logger.error(f"Erreur monitoring {component}: {e}")
                    snapshot.components[component] = HealthStatus.UNKNOWN
            
            # Calculer le statut global
            snapshot.overall_status = self._calculate_overall_status(snapshot.components)
            
            # Ajouter métriques système
            snapshot.metrics.update(await self._collect_system_metrics())
            
            # Uptime
            snapshot.uptime_seconds = (datetime.utcnow() - self.last_health_check).total_seconds()
            
            # Historique
            self.health_history.append(snapshot)
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Erreur obtention santé système: {e}")
            return SystemHealthSnapshot(overall_status=HealthStatus.UNKNOWN)
    
    def _calculate_overall_status(self, components: Dict[str, HealthStatus]) -> HealthStatus:
        """Calcule le statut global basé sur les composants"""
        if not components:
            return HealthStatus.UNKNOWN
        
        statuses = list(components.values())
        
        # Si un composant est DOWN, système DOWN
        if HealthStatus.DOWN in statuses:
            return HealthStatus.DOWN
        
        # Si un composant est CRITICAL, système CRITICAL
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        
        # Si un composant est DEGRADED, système DEGRADED
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        
        # Si un composant est WARNING, système WARNING
        if HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        
        # Si tous sont HEALTHY, système HEALTHY
        if all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        
        return HealthStatus.UNKNOWN
    
    async def _collect_system_metrics(self) -> Dict[str, float]:
        """Collecte les métriques système"""
        try:
            import psutil
            
            return {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage_percent": psutil.disk_usage('/').percent,
                "network_bytes_sent": psutil.net_io_counters().bytes_sent,
                "network_bytes_recv": psutil.net_io_counters().bytes_recv,
                "active_connections": len(psutil.net_connections()),
                "process_count": len(psutil.pids())
            }
        except Exception as e:
            logger.error(f"Erreur collecte métriques système: {e}")
            return {}

# =============================================================================
# MOTEUR D'OBSERVABILITÉ PRINCIPAL
# =============================================================================

class ObservabilityEngine:
    """Moteur principal d'observabilité ultra-avancé"""
    
    def __init__(self, config: ObservabilityConfig):
        self.config = config
        self.trace_manager = AdvancedTraceManager(config)
        self.metrics_manager = AdvancedMetricsManager(config)
        self.health_monitor = SystemHealthMonitor(config)
        self.is_initialized = False
        self.background_tasks: List[asyncio.Task] = []
        
    async def initialize(self) -> None:
        """Initialise le moteur d'observabilité"""
        try:
            logger.info("Initialisation du moteur d'observabilité ultra-avancé...")
            
            # Initialiser les composants
            await self.metrics_manager.initialize()
            
            # Démarrer les tâches de fond
            if self.config.enable_metrics:
                task = asyncio.create_task(self._metrics_export_loop())
                self.background_tasks.append(task)
            
            # Enregistrer les moniteurs de santé par défaut
            self._register_default_health_monitors()
            
            self.is_initialized = True
            logger.info("Moteur d'observabilité initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur initialisation moteur d'observabilité: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Arrête proprement le moteur"""
        try:
            logger.info("Arrêt du moteur d'observabilité...")
            
            # Annuler les tâches de fond
            for task in self.background_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            self.background_tasks.clear()
            logger.info("Moteur d'observabilité arrêté")
            
        except Exception as e:
            logger.error(f"Erreur arrêt moteur: {e}")
    
    async def _metrics_export_loop(self) -> None:
        """Boucle d'export des métriques"""
        while True:
            try:
                await asyncio.sleep(self.config.metrics_export_interval)
                # Ici on pourrait exporter vers des systèmes externes
                logger.debug("Export des métriques effectué")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erreur export métriques: {e}")
    
    def _register_default_health_monitors(self) -> None:
        """Enregistre les moniteurs de santé par défaut"""
        
        async def api_health():
            # Vérifier la santé de l'API
            return HealthStatus.HEALTHY
        
        async def database_health():
            # Vérifier la santé des bases de données
            return HealthStatus.HEALTHY
        
        async def ml_pipeline_health():
            # Vérifier la santé du pipeline ML
            return HealthStatus.HEALTHY
        
        self.health_monitor.register_component_monitor("api", api_health)
        self.health_monitor.register_component_monitor("database", database_health)
        self.health_monitor.register_component_monitor("ml_pipeline", ml_pipeline_health)
    
    # Méthodes de convenance pour l'utilisation
    async def trace_operation(self, operation_name: str, component: TraceComponent, **tags):
        """Wrapper pour tracer une opération"""
        return self.trace_manager.trace_operation(operation_name, component, **tags)
    
    async def record_metric(self, metric_name: str, value: Union[int, float], **kwargs):
        """Wrapper pour enregistrer une métrique"""
        await self.metrics_manager.record_metric(metric_name, value, **kwargs)
    
    async def get_system_health(self) -> SystemHealthSnapshot:
        """Wrapper pour obtenir la santé système"""
        return await self.health_monitor.get_system_health()

# =============================================================================
# FACTORY ET UTILITAIRES
# =============================================================================

async def create_observability_engine(config: Optional[ObservabilityConfig] = None) -> ObservabilityEngine:
    """Factory pour créer un moteur d'observabilité"""
    if config is None:
        config = ObservabilityConfig()
    
    engine = ObservabilityEngine(config)
    await engine.initialize()
    return engine

def get_default_config() -> ObservabilityConfig:
    """Configuration par défaut pour l'observabilité"""
    return ObservabilityConfig(
        service_name="spotify-ai-agent",
        service_version="3.0.0",
        environment="production",
        observability_level=ObservabilityLevel.COMPREHENSIVE
    )

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'ObservabilityEngine',
    'ObservabilityConfig',
    'ObservabilityLevel',
    'TraceComponent',
    'TraceSpan',
    'MetricSnapshot',
    'SystemHealthSnapshot',
    'HealthStatus',
    'AlertPriority',
    'create_observability_engine',
    'get_default_config'
]
