"""
🎵 Spotify AI Agent - Advanced Monitoring & Observability Middleware
=================================================================

Middleware de monitoring avancé avec métriques, tracing, profiling et alerting.
Solution complète d'observabilité pour environnements de production.

Architecture:
- Métriques temps réel (Prometheus/Grafana)
- Distributed Tracing (Jaeger/Zipkin)
- Application Performance Monitoring (APM)
- Health Checks & Circuit Breakers
- Real-time Alerting
- Custom Business Metrics
- Resource Usage Monitoring
- Database Query Performance
- AI Model Performance Tracking

Enterprise Features:
- Multi-tenant monitoring
- Anomaly detection
- Predictive alerting
- Custom dashboards
- SLA monitoring
- Compliance reporting
"""

import asyncio
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Union
from enum import Enum
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import json
import uuid
from contextlib import asynccontextmanager

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary, Info
import structlog
import opentelemetry.trace as trace
from opentelemetry import baggage
from opentelemetry.trace import Status, StatusCode
from opentelemetry.propagate import extract, inject
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from ..core.config import get_settings
from ..core.logging import get_logger
from ..utils.metrics import MetricsCollector
from ..utils.performance import PerformanceTracker
from ..utils.alerts import AlertManager


class MetricType(str, Enum):
    """Types de métriques"""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    SUMMARY = "summary"
    INFO = "info"


class AlertSeverity(str, Enum):
    """Niveaux de sévérité des alertes"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class HealthStatus(str, Enum):
    """Statuts de santé"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class MetricDefinition:
    """Définition d'une métrique"""
    name: str
    type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: List[float] = None  # Pour les histogrammes
    quantiles: List[float] = None  # Pour les summaries
    
    def __post_init__(self):
        if self.buckets is None and self.type == MetricType.HISTOGRAM:
            self.buckets = [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
        if self.quantiles is None and self.type == MetricType.SUMMARY:
            self.quantiles = [0.5, 0.9, 0.95, 0.99]


@dataclass
class AlertRule:
    """Règle d'alerte"""
    name: str
    metric_name: str
    condition: str  # >threshold, <threshold, ==value, etc.
    threshold: float
    duration: int  # seconds
    severity: AlertSeverity
    description: str
    labels: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    
    def evaluate(self, current_value: float, duration_seconds: int) -> bool:
        """Évalue si l'alerte doit être déclenchée"""
        if not self.enabled:
            return False
        
        if duration_seconds < self.duration:
            return False
        
        if self.condition.startswith(">"):
            return current_value > self.threshold
        elif self.condition.startswith("<"):
            return current_value < self.threshold
        elif self.condition.startswith("=="):
            return abs(current_value - self.threshold) < 0.001
        elif self.condition.startswith("!="):
            return abs(current_value - self.threshold) > 0.001
        elif self.condition.startswith(">="):
            return current_value >= self.threshold
        elif self.condition.startswith("<="):
            return current_value <= self.threshold
        
        return False


@dataclass
class TraceSpan:
    """Span de trace distribué"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: str = "UNSET"
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    
    def finish(self, status: str = "OK"):
        """Termine le span"""
        self.end_time = datetime.utcnow()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.status = status
    
    def add_tag(self, key: str, value: Any):
        """Ajoute un tag au span"""
        self.tags[key] = value
    
    def log(self, message: str, level: str = "INFO", **kwargs):
        """Ajoute un log au span"""
        self.logs.append({
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            **kwargs
        })


@dataclass
class PerformanceMetrics:
    """Métriques de performance système"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_percent: float
    network_sent_mb: float
    network_recv_mb: float
    active_connections: int
    process_count: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RequestMetrics:
    """Métriques de requête"""
    request_id: str
    method: str
    path: str
    status_code: int
    duration_ms: float
    request_size_bytes: int
    response_size_bytes: int
    user_agent: str
    ip_address: str
    timestamp: datetime
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    
    def to_labels(self) -> Dict[str, str]:
        """Convertit en labels pour les métriques"""
        return {
            "method": self.method,
            "path": self._normalize_path(),
            "status_code": str(self.status_code),
            "status_class": f"{self.status_code // 100}xx"
        }
    
    def _normalize_path(self) -> str:
        """Normalise le chemin pour les métriques"""
        # Remplacer les IDs par des placeholders
        import re
        path = self.path
        
        # UUID patterns
        path = re.sub(r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '/{uuid}', path)
        
        # Numeric IDs
        path = re.sub(r'/\d+', '/{id}', path)
        
        # Spotify IDs
        path = re.sub(r'/[0-9A-Za-z]{22}', '/{spotify_id}', path)
        
        return path


class MetricsRegistry:
    """Registre des métriques Prometheus"""
    
    def __init__(self):
        self.registry = prometheus_client.REGISTRY
        self.metrics: Dict[str, Any] = {}
        self._setup_default_metrics()
    
    def _setup_default_metrics(self):
        """Configure les métriques par défaut"""
        # Métriques HTTP
        self.http_requests_total = Counter(
            'spotify_ai_http_requests_total',
            'Total HTTP requests',
            ['method', 'path', 'status_code', 'status_class'],
            registry=self.registry
        )
        
        self.http_request_duration = Histogram(
            'spotify_ai_http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'path', 'status_code'],
            buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0],
            registry=self.registry
        )
        
        self.http_request_size = Histogram(
            'spotify_ai_http_request_size_bytes',
            'HTTP request size',
            ['method', 'path'],
            registry=self.registry
        )
        
        self.http_response_size = Histogram(
            'spotify_ai_http_response_size_bytes',
            'HTTP response size',
            ['method', 'path', 'status_code'],
            registry=self.registry
        )
        
        # Métriques système
        self.system_cpu_percent = Gauge(
            'spotify_ai_system_cpu_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_percent = Gauge(
            'spotify_ai_system_memory_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        
        self.system_disk_percent = Gauge(
            'spotify_ai_system_disk_percent',
            'System disk usage percentage',
            registry=self.registry
        )
        
        self.active_connections = Gauge(
            'spotify_ai_active_connections',
            'Number of active connections',
            registry=self.registry
        )
        
        # Métriques applicatives
        self.cache_hits_total = Counter(
            'spotify_ai_cache_hits_total',
            'Total cache hits',
            ['cache_type', 'cache_level'],
            registry=self.registry
        )
        
        self.cache_misses_total = Counter(
            'spotify_ai_cache_misses_total',
            'Total cache misses',
            ['cache_type', 'cache_level'],
            registry=self.registry
        )
        
        self.database_queries_total = Counter(
            'spotify_ai_database_queries_total',
            'Total database queries',
            ['query_type', 'table', 'status'],
            registry=self.registry
        )
        
        self.database_query_duration = Histogram(
            'spotify_ai_database_query_duration_seconds',
            'Database query duration',
            ['query_type', 'table'],
            registry=self.registry
        )
        
        # Métriques Spotify API
        self.spotify_api_requests_total = Counter(
            'spotify_ai_spotify_api_requests_total',
            'Total Spotify API requests',
            ['endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.spotify_api_rate_limit = Gauge(
            'spotify_ai_spotify_api_rate_limit_remaining',
            'Spotify API rate limit remaining',
            ['endpoint'],
            registry=self.registry
        )
        
        # Métriques IA
        self.ai_model_predictions_total = Counter(
            'spotify_ai_model_predictions_total',
            'Total AI model predictions',
            ['model_name', 'model_version', 'status'],
            registry=self.registry
        )
        
        self.ai_model_prediction_duration = Histogram(
            'spotify_ai_model_prediction_duration_seconds',
            'AI model prediction duration',
            ['model_name', 'model_version'],
            registry=self.registry
        )
        
        self.ai_model_accuracy = Gauge(
            'spotify_ai_model_accuracy',
            'AI model accuracy score',
            ['model_name', 'model_version'],
            registry=self.registry
        )
        
        # Métriques business
        self.user_actions_total = Counter(
            'spotify_ai_user_actions_total',
            'Total user actions',
            ['action_type', 'user_tier'],
            registry=self.registry
        )
        
        self.revenue_total = Counter(
            'spotify_ai_revenue_total_cents',
            'Total revenue in cents',
            ['subscription_tier', 'region'],
            registry=self.registry
        )
        
        self.active_users = Gauge(
            'spotify_ai_active_users',
            'Number of active users',
            ['time_window'],
            registry=self.registry
        )
    
    def register_custom_metric(self, definition: MetricDefinition) -> Any:
        """Enregistre une métrique personnalisée"""
        if definition.name in self.metrics:
            return self.metrics[definition.name]
        
        if definition.type == MetricType.COUNTER:
            metric = Counter(
                definition.name,
                definition.description,
                definition.labels,
                registry=self.registry
            )
        elif definition.type == MetricType.HISTOGRAM:
            metric = Histogram(
                definition.name,
                definition.description,
                definition.labels,
                buckets=definition.buckets,
                registry=self.registry
            )
        elif definition.type == MetricType.GAUGE:
            metric = Gauge(
                definition.name,
                definition.description,
                definition.labels,
                registry=self.registry
            )
        elif definition.type == MetricType.SUMMARY:
            metric = Summary(
                definition.name,
                definition.description,
                definition.labels,
                registry=self.registry
            )
        elif definition.type == MetricType.INFO:
            metric = Info(
                definition.name,
                definition.description,
                registry=self.registry
            )
        else:
            raise ValueError(f"Type de métrique non supporté: {definition.type}")
        
        self.metrics[definition.name] = metric
        return metric
    
    def get_metric(self, name: str) -> Optional[Any]:
        """Récupère une métrique par nom"""
        return self.metrics.get(name)


class DistributedTracer:
    """Tracer distribué pour le suivi des requêtes"""
    
    def __init__(self):
        self.active_spans: Dict[str, TraceSpan] = {}
        self.completed_spans: deque = deque(maxlen=10000)
        self.logger = get_logger("distributed_tracer")
    
    def start_span(
        self,
        operation_name: str,
        parent_span_id: Optional[str] = None,
        trace_id: Optional[str] = None
    ) -> TraceSpan:
        """Démarre un nouveau span"""
        span_id = str(uuid.uuid4())
        if trace_id is None:
            trace_id = str(uuid.uuid4())
        
        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=datetime.utcnow()
        )
        
        self.active_spans[span_id] = span
        return span
    
    def finish_span(self, span_id: str, status: str = "OK"):
        """Termine un span"""
        if span_id in self.active_spans:
            span = self.active_spans[span_id]
            span.finish(status)
            self.completed_spans.append(span)
            del self.active_spans[span_id]
    
    def get_span(self, span_id: str) -> Optional[TraceSpan]:
        """Récupère un span actif"""
        return self.active_spans.get(span_id)
    
    def inject_headers(self, span: TraceSpan) -> Dict[str, str]:
        """Injecte les headers de trace"""
        return {
            "x-trace-id": span.trace_id,
            "x-span-id": span.span_id,
            "x-parent-span-id": span.parent_span_id or ""
        }
    
    def extract_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Extrait les headers de trace"""
        return {
            "trace_id": headers.get("x-trace-id"),
            "span_id": headers.get("x-span-id"),
            "parent_span_id": headers.get("x-parent-span-id") or None
        }


class SystemMonitor:
    """Moniteur de ressources système"""
    
    def __init__(self, update_interval: int = 30):
        self.update_interval = update_interval
        self.logger = get_logger("system_monitor")
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self.current_metrics = PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0)
        
        # Métriques réseau de base
        self._last_network_stats = psutil.net_io_counters()
        self._last_update = time.time()
    
    async def start(self):
        """Démarre le monitoring système"""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        self.logger.info("Monitoring système démarré")
    
    async def stop(self):
        """Arrête le monitoring système"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.logger.info("Monitoring système arrêté")
    
    async def _monitor_loop(self):
        """Boucle de monitoring"""
        while self._running:
            try:
                await self._collect_metrics()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                self.logger.error(f"Erreur monitoring système: {e}")
                await asyncio.sleep(5)
    
    async def _collect_metrics(self):
        """Collecte les métriques système"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Mémoire
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            
            # Disque
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Réseau
            current_network = psutil.net_io_counters()
            current_time = time.time()
            
            time_delta = current_time - self._last_update
            sent_delta = current_network.bytes_sent - self._last_network_stats.bytes_sent
            recv_delta = current_network.bytes_recv - self._last_network_stats.bytes_recv
            
            network_sent_mb = (sent_delta / time_delta) / (1024 * 1024) if time_delta > 0 else 0
            network_recv_mb = (recv_delta / time_delta) / (1024 * 1024) if time_delta > 0 else 0
            
            # Connexions et processus
            active_connections = len(psutil.net_connections())
            process_count = len(psutil.pids())
            
            # Mettre à jour les métriques
            self.current_metrics = PerformanceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                disk_percent=disk_percent,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                active_connections=active_connections,
                process_count=process_count
            )
            
            # Sauvegarder pour le prochain calcul
            self._last_network_stats = current_network
            self._last_update = current_time
            
        except Exception as e:
            self.logger.error(f"Erreur collecte métriques: {e}")
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Retourne les métriques actuelles"""
        return self.current_metrics


class AlertManager:
    """Gestionnaire d'alertes"""
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.logger = get_logger("alert_manager")
        
        # Callbacks pour les alertes
        self.alert_callbacks: List[Callable] = []
        
        # État des métriques pour l'évaluation
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
    
    def add_rule(self, rule: AlertRule):
        """Ajoute une règle d'alerte"""
        self.rules[rule.name] = rule
        self.logger.info(f"Règle d'alerte ajoutée: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Supprime une règle d'alerte"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            if rule_name in self.active_alerts:
                del self.active_alerts[rule_name]
            self.logger.info(f"Règle d'alerte supprimée: {rule_name}")
    
    def add_alert_callback(self, callback: Callable):
        """Ajoute un callback pour les alertes"""
        self.alert_callbacks.append(callback)
    
    def record_metric_value(self, metric_name: str, value: float):
        """Enregistre une valeur de métrique pour l'évaluation"""
        self.metric_history[metric_name].append({
            "value": value,
            "timestamp": datetime.utcnow()
        })
        
        # Évaluer les règles d'alerte pour cette métrique
        asyncio.create_task(self._evaluate_rules_for_metric(metric_name))
    
    async def _evaluate_rules_for_metric(self, metric_name: str):
        """Évalue les règles d'alerte pour une métrique"""
        try:
            for rule_name, rule in self.rules.items():
                if rule.metric_name == metric_name:
                    await self._evaluate_rule(rule)
        except Exception as e:
            self.logger.error(f"Erreur évaluation règles: {e}")
    
    async def _evaluate_rule(self, rule: AlertRule):
        """Évalue une règle d'alerte"""
        if not rule.enabled:
            return
        
        metric_data = self.metric_history.get(rule.metric_name, deque())
        if not metric_data:
            return
        
        current_value = metric_data[-1]["value"]
        
        # Calculer la durée de l'état actuel
        duration_seconds = 0
        if len(metric_data) > 1:
            start_time = metric_data[-1]["timestamp"]
            for data_point in reversed(metric_data):
                if not rule.evaluate(data_point["value"], 0):
                    break
                duration_seconds = (start_time - data_point["timestamp"]).total_seconds()
        
        # Vérifier si l'alerte doit être déclenchée
        if rule.evaluate(current_value, duration_seconds):
            await self._trigger_alert(rule, current_value)
        else:
            await self._resolve_alert(rule.name)
    
    async def _trigger_alert(self, rule: AlertRule, current_value: float):
        """Déclenche une alerte"""
        alert_id = f"{rule.name}_{int(time.time())}"
        
        alert_data = {
            "id": alert_id,
            "rule_name": rule.name,
            "metric_name": rule.metric_name,
            "current_value": current_value,
            "threshold": rule.threshold,
            "severity": rule.severity,
            "description": rule.description,
            "labels": rule.labels,
            "timestamp": datetime.utcnow(),
            "status": "FIRING"
        }
        
        # Vérifier si l'alerte est déjà active
        if rule.name not in self.active_alerts:
            self.active_alerts[rule.name] = alert_data
            self.alert_history.append(alert_data.copy())
            
            self.logger.warning(
                f"Alerte déclenchée: {rule.name}",
                extra={
                    "alert_id": alert_id,
                    "metric": rule.metric_name,
                    "value": current_value,
                    "threshold": rule.threshold,
                    "severity": rule.severity.value
                }
            )
            
            # Exécuter les callbacks
            for callback in self.alert_callbacks:
                try:
                    await callback(alert_data)
                except Exception as e:
                    self.logger.error(f"Erreur callback alerte: {e}")
    
    async def _resolve_alert(self, rule_name: str):
        """Résout une alerte"""
        if rule_name in self.active_alerts:
            alert_data = self.active_alerts[rule_name]
            alert_data["status"] = "RESOLVED"
            alert_data["resolved_at"] = datetime.utcnow()
            
            del self.active_alerts[rule_name]
            self.alert_history.append(alert_data.copy())
            
            self.logger.info(f"Alerte résolue: {rule_name}")
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Retourne les alertes actives"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retourne l'historique des alertes"""
        return list(self.alert_history)[-limit:]


class AdvancedMonitoringMiddleware:
    """Middleware de monitoring et observabilité avancé"""
    
    def __init__(self, 
                 enable_metrics: bool = True,
                 enable_tracing: bool = True,
                 enable_system_monitoring: bool = True,
                 enable_alerting: bool = True):
        
        self.enable_metrics = enable_metrics
        self.enable_tracing = enable_tracing
        self.enable_system_monitoring = enable_system_monitoring
        self.enable_alerting = enable_alerting
        
        self.logger = get_logger("monitoring_middleware")
        
        # Composants de monitoring
        self.metrics_registry = MetricsRegistry() if enable_metrics else None
        self.tracer = DistributedTracer() if enable_tracing else None
        self.system_monitor = SystemMonitor() if enable_system_monitoring else None
        self.alert_manager = AlertManager() if enable_alerting else None
        
        # État d'initialisation
        self._initialized = False
        
        # Métriques de santé des services
        self.service_health = {}
        
        # Configuration des alertes par défaut
        if self.alert_manager:
            self._setup_default_alerts()
    
    async def initialize(self):
        """Initialise le middleware"""
        if self._initialized:
            return
        
        try:
            # Démarrer le monitoring système
            if self.system_monitor:
                await self.system_monitor.start()
            
            # Configurer les callbacks d'alerte
            if self.alert_manager:
                self.alert_manager.add_alert_callback(self._handle_alert)
            
            self._initialized = True
            self.logger.info("Middleware de monitoring initialisé")
            
        except Exception as e:
            self.logger.error(f"Erreur initialisation monitoring: {e}")
            raise
    
    async def __call__(self, request: Request, call_next):
        """Traite la requête avec monitoring"""
        # Initialiser si nécessaire
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Extraire les informations de trace
        trace_context = {}
        if self.tracer:
            trace_context = self.tracer.extract_headers(dict(request.headers))
        
        # Démarrer un span de trace
        span = None
        if self.tracer:
            span = self.tracer.start_span(
                f"{request.method} {request.url.path}",
                trace_id=trace_context.get("trace_id"),
                parent_span_id=trace_context.get("span_id")
            )
            span.add_tag("http.method", request.method)
            span.add_tag("http.url", str(request.url))
            span.add_tag("http.user_agent", request.headers.get("user-agent", ""))
            span.add_tag("request.id", request_id)
        
        # Ajouter les informations au contexte de la requête
        request.state.request_id = request_id
        request.state.trace_id = trace_context.get("trace_id") or str(uuid.uuid4())
        request.state.span_id = span.span_id if span else None
        request.state.start_time = start_time
        
        try:
            # Calculer la taille de la requête
            request_size = 0
            if hasattr(request, "body"):
                body = await request.body()
                request_size = len(body)
            
            # Exécuter la requête
            response = await call_next(request)
            
            # Calculer la durée
            duration = time.time() - start_time
            
            # Calculer la taille de la réponse
            response_size = 0
            if hasattr(response, "body"):
                # Note: En production, éviter de lire tout le body
                response_size = getattr(response, "content_length", 0) or 0
            
            # Créer les métriques de requête
            request_metrics = RequestMetrics(
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration * 1000,
                request_size_bytes=request_size,
                response_size_bytes=response_size,
                user_agent=request.headers.get("user-agent", ""),
                ip_address=request.client.host if request.client else "",
                timestamp=datetime.utcnow(),
                trace_id=request.state.trace_id,
                span_id=request.state.span_id
            )
            
            # Enregistrer les métriques Prometheus
            if self.metrics_registry:
                await self._record_prometheus_metrics(request_metrics)
            
            # Terminer le span de trace
            if span:
                span.add_tag("http.status_code", response.status_code)
                span.add_tag("http.response_size", response_size)
                status = "OK" if response.status_code < 400 else "ERROR"
                self.tracer.finish_span(span.span_id, status)
            
            # Enregistrer les métriques système
            if self.system_monitor and self.metrics_registry:
                await self._record_system_metrics()
            
            # Ajouter les headers de monitoring
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            if span:
                response.headers["X-Trace-ID"] = span.trace_id
                response.headers["X-Span-ID"] = span.span_id
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Enregistrer l'erreur dans les métriques
            if self.metrics_registry:
                error_metrics = RequestMetrics(
                    request_id=request_id,
                    method=request.method,
                    path=request.url.path,
                    status_code=500,
                    duration_ms=duration * 1000,
                    request_size_bytes=0,
                    response_size_bytes=0,
                    user_agent=request.headers.get("user-agent", ""),
                    ip_address=request.client.host if request.client else "",
                    timestamp=datetime.utcnow(),
                    trace_id=request.state.trace_id,
                    span_id=request.state.span_id
                )
                await self._record_prometheus_metrics(error_metrics)
            
            # Terminer le span avec erreur
            if span:
                span.add_tag("error", True)
                span.add_tag("error.message", str(e))
                span.log(f"Exception: {e}", level="ERROR")
                self.tracer.finish_span(span.span_id, "ERROR")
            
            raise
    
    async def _record_prometheus_metrics(self, metrics: RequestMetrics):
        """Enregistre les métriques Prometheus"""
        try:
            labels = metrics.to_labels()
            
            # Métriques HTTP
            self.metrics_registry.http_requests_total.labels(**labels).inc()
            
            self.metrics_registry.http_request_duration.labels(
                method=labels["method"],
                path=labels["path"],
                status_code=labels["status_code"]
            ).observe(metrics.duration_ms / 1000)
            
            self.metrics_registry.http_request_size.labels(
                method=labels["method"],
                path=labels["path"]
            ).observe(metrics.request_size_bytes)
            
            self.metrics_registry.http_response_size.labels(
                method=labels["method"],
                path=labels["path"],
                status_code=labels["status_code"]
            ).observe(metrics.response_size_bytes)
            
            # Enregistrer pour les alertes
            if self.alert_manager:
                self.alert_manager.record_metric_value(
                    "http_request_duration_seconds",
                    metrics.duration_ms / 1000
                )
                self.alert_manager.record_metric_value(
                    "http_requests_total",
                    1
                )
            
        except Exception as e:
            self.logger.error(f"Erreur enregistrement métriques: {e}")
    
    async def _record_system_metrics(self):
        """Enregistre les métriques système"""
        try:
            current_metrics = self.system_monitor.get_current_metrics()
            
            # Mettre à jour les gauges Prometheus
            self.metrics_registry.system_cpu_percent.set(current_metrics.cpu_percent)
            self.metrics_registry.system_memory_percent.set(current_metrics.memory_percent)
            self.metrics_registry.system_disk_percent.set(current_metrics.disk_percent)
            self.metrics_registry.active_connections.set(current_metrics.active_connections)
            
            # Enregistrer pour les alertes
            if self.alert_manager:
                self.alert_manager.record_metric_value("system_cpu_percent", current_metrics.cpu_percent)
                self.alert_manager.record_metric_value("system_memory_percent", current_metrics.memory_percent)
                self.alert_manager.record_metric_value("system_disk_percent", current_metrics.disk_percent)
            
        except Exception as e:
            self.logger.error(f"Erreur enregistrement métriques système: {e}")
    
    def _setup_default_alerts(self):
        """Configure les alertes par défaut"""
        default_rules = [
            AlertRule(
                name="high_cpu_usage",
                metric_name="system_cpu_percent",
                condition=">threshold",
                threshold=80.0,
                duration=300,  # 5 minutes
                severity=AlertSeverity.HIGH,
                description="CPU usage is above 80% for 5 minutes"
            ),
            AlertRule(
                name="high_memory_usage",
                metric_name="system_memory_percent",
                condition=">threshold",
                threshold=85.0,
                duration=300,
                severity=AlertSeverity.HIGH,
                description="Memory usage is above 85% for 5 minutes"
            ),
            AlertRule(
                name="high_response_time",
                metric_name="http_request_duration_seconds",
                condition=">threshold",
                threshold=2.0,
                duration=60,  # 1 minute
                severity=AlertSeverity.MEDIUM,
                description="Average response time is above 2 seconds"
            ),
            AlertRule(
                name="high_error_rate",
                metric_name="http_error_rate",
                condition=">threshold",
                threshold=0.05,  # 5%
                duration=300,
                severity=AlertSeverity.CRITICAL,
                description="Error rate is above 5% for 5 minutes"
            )
        ]
        
        for rule in default_rules:
            self.alert_manager.add_rule(rule)
    
    async def _handle_alert(self, alert_data: Dict[str, Any]):
        """Gère les alertes déclenchées"""
        try:
            self.logger.warning(
                f"🚨 ALERTE: {alert_data['rule_name']}",
                extra={
                    "alert_data": alert_data,
                    "severity": alert_data["severity"],
                    "metric": alert_data["metric_name"],
                    "value": alert_data["current_value"],
                    "threshold": alert_data["threshold"]
                }
            )
            
            # Ici on pourrait envoyer des notifications
            # - Slack/Discord webhook
            # - Email
            # - SMS
            # - PagerDuty
            # - etc.
            
        except Exception as e:
            self.logger.error(f"Erreur gestion alerte: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Retourne le statut de santé global"""
        try:
            health_data = {
                "status": HealthStatus.HEALTHY,
                "timestamp": datetime.utcnow().isoformat(),
                "services": {},
                "metrics": {},
                "alerts": {
                    "active_count": len(self.alert_manager.get_active_alerts()) if self.alert_manager else 0,
                    "active_alerts": self.alert_manager.get_active_alerts() if self.alert_manager else []
                }
            }
            
            # Vérifier les métriques système
            if self.system_monitor:
                current_metrics = self.system_monitor.get_current_metrics()
                health_data["metrics"]["system"] = asdict(current_metrics)
                
                # Déterminer le statut basé sur les métriques
                if current_metrics.cpu_percent > 90 or current_metrics.memory_percent > 95:
                    health_data["status"] = HealthStatus.UNHEALTHY
                elif current_metrics.cpu_percent > 75 or current_metrics.memory_percent > 85:
                    health_data["status"] = HealthStatus.DEGRADED
            
            # Vérifier les alertes critiques
            if self.alert_manager:
                active_alerts = self.alert_manager.get_active_alerts()
                critical_alerts = [a for a in active_alerts if a["severity"] == AlertSeverity.CRITICAL]
                
                if critical_alerts:
                    health_data["status"] = HealthStatus.UNHEALTHY
                elif active_alerts:
                    health_data["status"] = HealthStatus.DEGRADED
            
            return health_data
            
        except Exception as e:
            self.logger.error(f"Erreur status santé: {e}")
            return {
                "status": HealthStatus.UNKNOWN,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des métriques"""
        try:
            if not self.metrics_registry:
                return {"error": "Métriques non activées"}
            
            # Ici on pourrait implémenter une logique pour 
            # récupérer les valeurs actuelles des métriques Prometheus
            # En production, cela se ferait via l'API Prometheus
            
            return {
                "metrics_available": True,
                "registry_size": len(self.metrics_registry.metrics),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Erreur résumé métriques: {e}")
            return {"error": str(e)}
    
    async def shutdown(self):
        """Arrête proprement le middleware"""
        try:
            if self.system_monitor:
                await self.system_monitor.stop()
            
            self.logger.info("Middleware de monitoring arrêté")
            
        except Exception as e:
            self.logger.error(f"Erreur arrêt monitoring: {e}")


# Factory functions

def create_monitoring_middleware(
    enable_metrics: bool = True,
    enable_tracing: bool = True,
    enable_system_monitoring: bool = True,
    enable_alerting: bool = True
) -> AdvancedMonitoringMiddleware:
    """Crée un middleware de monitoring configuré"""
    return AdvancedMonitoringMiddleware(
        enable_metrics=enable_metrics,
        enable_tracing=enable_tracing,
        enable_system_monitoring=enable_system_monitoring,
        enable_alerting=enable_alerting
    )


def create_production_monitoring() -> AdvancedMonitoringMiddleware:
    """Configuration de monitoring pour la production"""
    return AdvancedMonitoringMiddleware(
        enable_metrics=True,
        enable_tracing=True,
        enable_system_monitoring=True,
        enable_alerting=True
    )


def create_development_monitoring() -> AdvancedMonitoringMiddleware:
    """Configuration de monitoring pour le développement"""
    return AdvancedMonitoringMiddleware(
        enable_metrics=True,
        enable_tracing=False,
        enable_system_monitoring=False,
        enable_alerting=False
    )


# Export des classes principales
__all__ = [
    "MetricType",
    "AlertSeverity", 
    "HealthStatus",
    "MetricDefinition",
    "AlertRule",
    "TraceSpan",
    "PerformanceMetrics",
    "RequestMetrics",
    "MetricsRegistry",
    "DistributedTracer",
    "SystemMonitor",
    "AlertManager",
    "AdvancedMonitoringMiddleware",
    "create_monitoring_middleware",
    "create_production_monitoring",
    "create_development_monitoring"
]
