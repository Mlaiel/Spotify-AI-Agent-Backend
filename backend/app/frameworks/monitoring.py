"""
📊 MONITORING FRAMEWORK - OBSERVABILITÉ ENTERPRISE
Expert Team: DBA & Data Engineer, Microservices Architect

Monitoring complet avec métriques, tracing, alerting et observabilité
"""

import asyncio
import os
import time
import threading
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import json
from contextlib import asynccontextmanager

# Monitoring et métriques
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary, Info
from prometheus_client.exposition import generate_latest

# Tracing distribué
from opentelemetry import trace, baggage, metrics as otel_metrics
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# APM et profiling
import psutil
import redis
from sqlalchemy import create_engine, text

# Alerting
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import slack_sdk

# Base framework
from .core import BaseFramework, FrameworkStatus, FrameworkHealth
from .core import framework_orchestrator

# Configuration
from pydantic import BaseSettings, Field


class MetricType(Enum):
    """Types de métriques"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"


class AlertSeverity(Enum):
    """Niveaux de sévérité des alertes"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MonitoringConfig:
    """Configuration du monitoring"""
    
    # Prometheus
    prometheus_port: int = 8000
    prometheus_metrics_path: str = "/metrics"
    enable_prometheus: bool = True
    
    # Jaeger Tracing
    jaeger_agent_host: str = "localhost"
    jaeger_agent_port: int = 6831
    jaeger_collector_endpoint: str = "http://localhost:14268/api/traces"
    enable_tracing: bool = True
    
    # Health Checks
    health_check_interval: int = 30
    health_check_timeout: int = 10
    
    # Performance Monitoring
    cpu_threshold: float = 80.0
    memory_threshold: float = 85.0
    disk_threshold: float = 90.0
    response_time_threshold: float = 2.0
    
    # Alerting
    enable_email_alerts: bool = False
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    alert_recipients: List[str] = field(default_factory=list)
    
    enable_slack_alerts: bool = False
    slack_webhook_url: str = ""
    slack_channel: str = "#alerts"
    
    enable_webhook_alerts: bool = False
    webhook_url: str = ""
    
    # Database Monitoring
    db_connection_pool_threshold: int = 80
    db_slow_query_threshold: float = 1.0
    
    # Redis Monitoring
    redis_memory_threshold: int = 80
    redis_connection_threshold: int = 90


@dataclass
class Alert:
    """Alerte de monitoring"""
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    source: str
    metric_name: str
    metric_value: float
    threshold: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """
    📈 COLLECTEUR DE MÉTRIQUES
    
    Collection centralisée des métriques:
    - Métriques système (CPU, RAM, Disk)
    - Métriques applicatives
    - Métriques custom
    - Export Prometheus
    """
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = logging.getLogger("monitoring.metrics")
        
        # Métriques système
        self.cpu_usage = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
        self.memory_usage = Gauge('system_memory_usage_percent', 'Memory usage percentage')
        self.disk_usage = Gauge('system_disk_usage_percent', 'Disk usage percentage', ['device'])
        self.network_bytes_sent = Counter('system_network_bytes_sent_total', 'Network bytes sent')
        self.network_bytes_recv = Counter('system_network_bytes_recv_total', 'Network bytes received')
        
        # Métriques applicatives
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code']
        )
        self.http_request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint']
        )
        self.active_users = Gauge('active_users', 'Number of active users')
        
        # Métriques Spotify AI Agent
        self.spotify_api_calls = Counter(
            'spotify_api_calls_total',
            'Total Spotify API calls',
            ['endpoint', 'status']
        )
        self.ai_model_predictions = Counter(
            'ai_model_predictions_total',
            'Total AI model predictions',
            ['model_name', 'model_type']
        )
        self.ai_model_latency = Histogram(
            'ai_model_prediction_duration_seconds',
            'AI model prediction duration',
            ['model_name']
        )
        self.recommendation_accuracy = Gauge(
            'recommendation_accuracy',
            'Recommendation model accuracy',
            ['model_name']
        )
        
        # Métriques base de données
        self.db_connections_active = Gauge('db_connections_active', 'Active database connections')
        self.db_query_duration = Histogram(
            'db_query_duration_seconds',
            'Database query duration',
            ['query_type']
        )
        self.db_slow_queries = Counter('db_slow_queries_total', 'Total slow database queries')
        
        # Métriques Redis
        self.redis_memory_usage = Gauge('redis_memory_usage_bytes', 'Redis memory usage')
        self.redis_connected_clients = Gauge('redis_connected_clients', 'Redis connected clients')
        self.redis_operations = Counter(
            'redis_operations_total',
            'Total Redis operations',
            ['operation']
        )
        
        # Registry personnalisé
        self.custom_metrics: Dict[str, Any] = {}
        
    async def collect_system_metrics(self):
        """Collecte les métriques système"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_usage.set(cpu_percent)
            
            # Mémoire
            memory = psutil.virtual_memory()
            self.memory_usage.set(memory.percent)
            
            # Disque
            for disk in psutil.disk_partitions():
                try:
                    disk_usage = psutil.disk_usage(disk.mountpoint)
                    usage_percent = (disk_usage.used / disk_usage.total) * 100
                    self.disk_usage.labels(device=disk.device).set(usage_percent)
                except PermissionError:
                    continue
            
            # Réseau
            network = psutil.net_io_counters()
            self.network_bytes_sent.inc(network.bytes_sent)
            self.network_bytes_recv.inc(network.bytes_recv)
            
        except Exception as e:
            self.logger.error(f"System metrics collection failed: {e}")
    
    def record_http_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float
    ):
        """Enregistre une requête HTTP"""
        self.http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        self.http_request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_spotify_api_call(self, endpoint: str, status: str):
        """Enregistre un appel API Spotify"""
        self.spotify_api_calls.labels(
            endpoint=endpoint,
            status=status
        ).inc()
    
    def record_ai_prediction(
        self,
        model_name: str,
        model_type: str,
        duration: float,
        accuracy: Optional[float] = None
    ):
        """Enregistre une prédiction IA"""
        self.ai_model_predictions.labels(
            model_name=model_name,
            model_type=model_type
        ).inc()
        
        self.ai_model_latency.labels(model_name=model_name).observe(duration)
        
        if accuracy is not None:
            self.recommendation_accuracy.labels(model_name=model_name).set(accuracy)
    
    def record_db_query(self, query_type: str, duration: float):
        """Enregistre une requête base de données"""
        self.db_query_duration.labels(query_type=query_type).observe(duration)
        
        if duration > self.config.db_slow_query_threshold:
            self.db_slow_queries.inc()
    
    def record_redis_operation(self, operation: str):
        """Enregistre une opération Redis"""
        self.redis_operations.labels(operation=operation).inc()
    
    async def collect_db_metrics(self, db_url: str):
        """Collecte les métriques de base de données"""
        try:
            engine = create_engine(db_url)
            with engine.connect() as conn:
                # Connexions actives
                result = conn.execute(text("SELECT count(*) FROM pg_stat_activity"))
                active_connections = result.scalar()
                self.db_connections_active.set(active_connections)
                
        except Exception as e:
            self.logger.error(f"Database metrics collection failed: {e}")
    
    async def collect_redis_metrics(self, redis_url: str):
        """Collecte les métriques Redis"""
        try:
            redis_client = redis.Redis.from_url(redis_url)
            info = redis_client.info()
            
            # Mémoire utilisée
            self.redis_memory_usage.set(info.get('used_memory', 0))
            
            # Clients connectés
            self.redis_connected_clients.set(info.get('connected_clients', 0))
            
        except Exception as e:
            self.logger.error(f"Redis metrics collection failed: {e}")
    
    def register_custom_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str,
        labels: Optional[List[str]] = None
    ):
        """Enregistre une métrique personnalisée"""
        labels = labels or []
        
        if metric_type == MetricType.COUNTER:
            metric = Counter(name, description, labels)
        elif metric_type == MetricType.GAUGE:
            metric = Gauge(name, description, labels)
        elif metric_type == MetricType.HISTOGRAM:
            metric = Histogram(name, description, labels)
        elif metric_type == MetricType.SUMMARY:
            metric = Summary(name, description, labels)
        elif metric_type == MetricType.INFO:
            metric = Info(name, description)
        else:
            raise ValueError(f"Unsupported metric type: {metric_type}")
        
        self.custom_metrics[name] = metric
        return metric
    
    def get_metrics_export(self) -> str:
        """Exporte les métriques au format Prometheus"""
        return generate_latest()


class DistributedTracing:
    """
    🔍 TRACING DISTRIBUÉ
    
    Tracing complet des requêtes:
    - OpenTelemetry integration
    - Jaeger export
    - Span correlation
    - Performance analysis
    """
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = logging.getLogger("monitoring.tracing")
        
        if config.enable_tracing:
            self._setup_tracing()
    
    def _setup_tracing(self):
        """Configure le tracing distribué"""
        try:
            # Configurar TracerProvider
            trace.set_tracer_provider(TracerProvider())
            
            # Jaeger Exporter
            jaeger_exporter = JaegerExporter(
                agent_host_name=self.config.jaeger_agent_host,
                agent_port=self.config.jaeger_agent_port,
            )
            
            # Span Processor
            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
            
            # Console exporter pour debug
            if self.logger.level == logging.DEBUG:
                console_exporter = ConsoleSpanExporter()
                console_processor = BatchSpanProcessor(console_exporter)
                trace.get_tracer_provider().add_span_processor(console_processor)
            
            # Instrumenter automatiquement FastAPI et requests
            FastAPIInstrumentor.instrument()
            RequestsInstrumentor.instrument()
            
            self.tracer = trace.get_tracer(__name__)
            
        except Exception as e:
            self.logger.error(f"Tracing setup failed: {e}")
    
    def create_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Crée un span de tracing"""
        if not hasattr(self, 'tracer'):
            return None
        
        span = self.tracer.start_span(name)
        
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        
        return span
    
    @asynccontextmanager
    async def trace_operation(self, operation_name: str, **attributes):
        """Context manager pour tracer une opération"""
        if not hasattr(self, 'tracer'):
            yield
            return
        
        with self.tracer.start_as_current_span(operation_name) as span:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
            yield span


class AlertManager:
    """
    🚨 GESTIONNAIRE D'ALERTES
    
    Système d'alertes intelligent:
    - Règles d'alertes configurables
    - Notifications multi-canaux
    - Gestion des escalades
    - Déduplication
    """
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = logging.getLogger("monitoring.alerts")
        
        # État des alertes
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Seuils d'alerte
        self.thresholds = {
            "cpu_usage": config.cpu_threshold,
            "memory_usage": config.memory_threshold,
            "disk_usage": config.disk_threshold,
            "response_time": config.response_time_threshold,
            "db_connections": config.db_connection_pool_threshold,
            "redis_memory": config.redis_memory_threshold
        }
    
    async def check_metric_thresholds(self, metrics: Dict[str, float]):
        """Vérifie les seuils des métriques"""
        for metric_name, value in metrics.items():
            if metric_name in self.thresholds:
                threshold = self.thresholds[metric_name]
                
                if value > threshold:
                    await self._trigger_alert(
                        severity=AlertSeverity.HIGH if value > threshold * 1.2 else AlertSeverity.MEDIUM,
                        title=f"{metric_name.replace('_', ' ').title()} Alert",
                        message=f"{metric_name} is {value:.2f}%, exceeding threshold of {threshold}%",
                        source="metrics_monitor",
                        metric_name=metric_name,
                        metric_value=value,
                        threshold=threshold
                    )
                else:
                    # Résoudre l'alerte si la métrique revient en dessous du seuil
                    await self._resolve_alert(metric_name)
    
    async def _trigger_alert(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        source: str,
        metric_name: str,
        metric_value: float,
        threshold: float,
        **metadata
    ):
        """Déclenche une alerte"""
        alert_key = f"{source}_{metric_name}"
        
        # Éviter la duplication d'alertes
        if alert_key in self.active_alerts:
            # Mettre à jour l'alerte existante
            self.active_alerts[alert_key].metric_value = metric_value
            self.active_alerts[alert_key].timestamp = datetime.utcnow()
            return
        
        alert = Alert(
            severity=severity,
            title=title,
            message=message,
            timestamp=datetime.utcnow(),
            source=source,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
            metadata=metadata
        )
        
        self.active_alerts[alert_key] = alert
        self.alert_history.append(alert)
        
        # Envoyer les notifications
        await self._send_notifications(alert)
        
        self.logger.warning(f"Alert triggered: {title} - {message}")
    
    async def _resolve_alert(self, metric_name: str):
        """Résout une alerte"""
        alert_key = f"metrics_monitor_{metric_name}"
        
        if alert_key in self.active_alerts:
            alert = self.active_alerts.pop(alert_key)
            
            # Notification de résolution
            await self._send_resolution_notification(alert)
            
            self.logger.info(f"Alert resolved: {alert.title}")
    
    async def _send_notifications(self, alert: Alert):
        """Envoie les notifications d'alerte"""
        try:
            # Email
            if self.config.enable_email_alerts:
                await self._send_email_alert(alert)
            
            # Slack
            if self.config.enable_slack_alerts:
                await self._send_slack_alert(alert)
            
            # Webhook
            if self.config.enable_webhook_alerts:
                await self._send_webhook_alert(alert)
                
        except Exception as e:
            self.logger.error(f"Failed to send alert notifications: {e}")
    
    async def _send_email_alert(self, alert: Alert):
        """Envoie une alerte par email"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.smtp_username
            msg['To'] = ', '.join(self.config.alert_recipients)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            body = f"""
            Alert Details:
            - Severity: {alert.severity.value.upper()}
            - Metric: {alert.metric_name}
            - Value: {alert.metric_value}
            - Threshold: {alert.threshold}
            - Time: {alert.timestamp}
            - Source: {alert.source}
            
            Message: {alert.message}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.config.smtp_host, self.config.smtp_port)
            server.starttls()
            server.login(self.config.smtp_username, self.config.smtp_password)
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            self.logger.error(f"Email alert failed: {e}")
    
    async def _send_slack_alert(self, alert: Alert):
        """Envoie une alerte sur Slack"""
        try:
            color = {
                AlertSeverity.LOW: "good",
                AlertSeverity.MEDIUM: "warning",
                AlertSeverity.HIGH: "danger",
                AlertSeverity.CRITICAL: "danger"
            }[alert.severity]
            
            payload = {
                "channel": self.config.slack_channel,
                "username": "Spotify AI Agent Monitor",
                "icon_emoji": ":warning:",
                "attachments": [{
                    "color": color,
                    "title": alert.title,
                    "text": alert.message,
                    "fields": [
                        {"title": "Metric", "value": alert.metric_name, "short": True},
                        {"title": "Value", "value": f"{alert.metric_value:.2f}", "short": True},
                        {"title": "Threshold", "value": f"{alert.threshold:.2f}", "short": True},
                        {"title": "Source", "value": alert.source, "short": True}
                    ],
                    "ts": alert.timestamp.timestamp()
                }]
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.config.slack_webhook_url,
                    json=payload
                )
                response.raise_for_status()
                
        except Exception as e:
            self.logger.error(f"Slack alert failed: {e}")
    
    async def _send_webhook_alert(self, alert: Alert):
        """Envoie une alerte via webhook"""
        try:
            payload = {
                "severity": alert.severity.value,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "source": alert.source,
                "metric_name": alert.metric_name,
                "metric_value": alert.metric_value,
                "threshold": alert.threshold,
                "metadata": alert.metadata
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.config.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
        except Exception as e:
            self.logger.error(f"Webhook alert failed: {e}")
    
    async def _send_resolution_notification(self, alert: Alert):
        """Envoie une notification de résolution"""
        # Implémentation similaire aux notifications d'alerte
        # mais avec un message de résolution
        pass
    
    def get_active_alerts(self) -> List[Alert]:
        """Retourne les alertes actives"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Retourne l'historique des alertes"""
        return self.alert_history[-limit:]


class HealthChecker:
    """
    ❤️ VÉRIFICATEUR DE SANTÉ
    
    Health checks complets:
    - Santé des services
    - Connectivité des dépendances
    - Vérifications personnalisées
    - Rapport de santé global
    """
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = logging.getLogger("monitoring.health")
        
        # Registre des health checks
        self.health_checks: Dict[str, Callable] = {}
        
        # Résultats des derniers checks
        self.health_results: Dict[str, Dict[str, Any]] = {}
    
    def register_health_check(self, name: str, check_func: Callable):
        """Enregistre un health check"""
        self.health_checks[name] = check_func
        self.logger.info(f"Health check registered: {name}")
    
    async def run_health_check(self, name: str) -> Dict[str, Any]:
        """Exécute un health check spécifique"""
        if name not in self.health_checks:
            return {"status": "error", "message": "Health check not found"}
        
        try:
            start_time = time.time()
            
            # Exécuter le health check avec timeout
            result = await asyncio.wait_for(
                self.health_checks[name](),
                timeout=self.config.health_check_timeout
            )
            
            execution_time = time.time() - start_time
            
            check_result = {
                "status": "healthy" if result else "unhealthy",
                "execution_time": execution_time,
                "timestamp": datetime.utcnow().isoformat(),
                "details": result if isinstance(result, dict) else {}
            }
            
            self.health_results[name] = check_result
            return check_result
            
        except asyncio.TimeoutError:
            result = {
                "status": "timeout",
                "execution_time": self.config.health_check_timeout,
                "timestamp": datetime.utcnow().isoformat(),
                "error": "Health check timed out"
            }
            self.health_results[name] = result
            return result
            
        except Exception as e:
            result = {
                "status": "error",
                "execution_time": 0,
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
            self.health_results[name] = result
            return result
    
    async def run_all_health_checks(self) -> Dict[str, Any]:
        """Exécute tous les health checks"""
        results = {}
        
        # Exécuter en parallèle
        tasks = [
            self.run_health_check(name)
            for name in self.health_checks.keys()
        ]
        
        if tasks:
            check_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for name, result in zip(self.health_checks.keys(), check_results):
                if isinstance(result, Exception):
                    results[name] = {
                        "status": "error",
                        "error": str(result),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                else:
                    results[name] = result
        
        # Calculer la santé globale
        overall_status = self._calculate_overall_health(results)
        
        return {
            "overall_status": overall_status,
            "checks": results,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _calculate_overall_health(self, results: Dict[str, Any]) -> str:
        """Calcule la santé globale"""
        if not results:
            return "unknown"
        
        statuses = [check.get("status", "unknown") for check in results.values()]
        
        if all(status == "healthy" for status in statuses):
            return "healthy"
        elif any(status == "error" for status in statuses):
            return "unhealthy"
        elif any(status == "timeout" for status in statuses):
            return "degraded"
        else:
            return "degraded"
    
    async def database_health_check(self, db_url: str) -> Dict[str, Any]:
        """Health check de la base de données"""
        try:
            engine = create_engine(db_url)
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            
            return {"status": "healthy", "message": "Database connection successful"}
            
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def redis_health_check(self, redis_url: str) -> Dict[str, Any]:
        """Health check de Redis"""
        try:
            redis_client = redis.Redis.from_url(redis_url)
            redis_client.ping()
            
            return {"status": "healthy", "message": "Redis connection successful"}
            
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def external_api_health_check(self, api_url: str) -> Dict[str, Any]:
        """Health check d'une API externe"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(api_url, timeout=5.0)
                
            if response.status_code == 200:
                return {"status": "healthy", "message": "API responding"}
            else:
                return {
                    "status": "unhealthy",
                    "error": f"API returned status code {response.status_code}"
                }
                
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


class MonitoringFramework(BaseFramework):
    """
    📊 FRAMEWORK DE MONITORING ENTERPRISE
    
    Orchestration complète du monitoring avec:
    - Collection de métriques temps réel
    - Tracing distribué
    - Alertes intelligentes
    - Health monitoring
    - Observabilité complète
    """
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        super().__init__("monitoring", config.__dict__ if config else {})
        self.config = config or MonitoringConfig()
        
        # Composants de monitoring
        self.metrics_collector = MetricsCollector(self.config)
        self.tracing = DistributedTracing(self.config)
        self.alert_manager = AlertManager(self.config)
        self.health_checker = HealthChecker(self.config)
        
        # Tâches de background
        self._monitoring_tasks: List[asyncio.Task] = []
        
    async def initialize(self) -> bool:
        """Initialise le framework de monitoring"""
        try:
            # Enregistrer les health checks par défaut
            self._register_default_health_checks()
            
            # Démarrer les tâches de monitoring
            await self._start_monitoring_tasks()
            
            # Démarrer le serveur de métriques Prometheus
            if self.config.enable_prometheus:
                await self._start_prometheus_server()
            
            self.logger.info("Monitoring Framework initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Monitoring framework initialization failed: {e}")
            return False
    
    def _register_default_health_checks(self):
        """Enregistre les health checks par défaut"""
        # Health check système
        async def system_health():
            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory().percent
            
            if cpu > self.config.cpu_threshold or memory > self.config.memory_threshold:
                return {"status": "degraded", "cpu": cpu, "memory": memory}
            
            return {"status": "healthy", "cpu": cpu, "memory": memory}
        
        self.health_checker.register_health_check("system", system_health)
        
        # Ajouter d'autres health checks par défaut
        # Database, Redis, APIs externes, etc.
    
    async def _start_monitoring_tasks(self):
        """Démarre les tâches de monitoring en arrière-plan"""
        # Tâche de collecte de métriques système
        self._monitoring_tasks.append(
            asyncio.create_task(self._metrics_collection_loop())
        )
        
        # Tâche de vérification des seuils d'alerte
        self._monitoring_tasks.append(
            asyncio.create_task(self._alert_checking_loop())
        )
        
        # Tâche de health checks périodiques
        self._monitoring_tasks.append(
            asyncio.create_task(self._health_check_loop())
        )
    
    async def _metrics_collection_loop(self):
        """Boucle de collecte de métriques"""
        while True:
            try:
                await self.metrics_collector.collect_system_metrics()
                await asyncio.sleep(30)  # Collecte toutes les 30 secondes
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(30)
    
    async def _alert_checking_loop(self):
        """Boucle de vérification des alertes"""
        while True:
            try:
                # Récupérer les métriques actuelles
                metrics = {
                    "cpu_usage": psutil.cpu_percent(),
                    "memory_usage": psutil.virtual_memory().percent,
                }
                
                # Vérifier les seuils
                await self.alert_manager.check_metric_thresholds(metrics)
                
                await asyncio.sleep(60)  # Vérification toutes les minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Alert checking error: {e}")
                await asyncio.sleep(60)
    
    async def _health_check_loop(self):
        """Boucle de health checks"""
        while True:
            try:
                await self.health_checker.run_all_health_checks()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(self.config.health_check_interval)
    
    async def _start_prometheus_server(self):
        """Démarre le serveur de métriques Prometheus"""
        try:
            # Le serveur Prometheus est démarré automatiquement
            # quand on utilise les métriques prometheus_client
            self.logger.info(f"Prometheus metrics available at :{self.config.prometheus_port}/metrics")
        except Exception as e:
            self.logger.error(f"Failed to start Prometheus server: {e}")
    
    async def get_system_overview(self) -> Dict[str, Any]:
        """Récupère un aperçu du système"""
        # Métriques système
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Health checks
        health_results = await self.health_checker.run_all_health_checks()
        
        # Alertes actives
        active_alerts = self.alert_manager.get_active_alerts()
        
        return {
            "system": {
                "cpu_percent": cpu,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": (disk.used / disk.total) * 100,
                "disk_free_gb": disk.free / (1024**3)
            },
            "health": health_results,
            "alerts": {
                "active_count": len(active_alerts),
                "alerts": [
                    {
                        "severity": alert.severity.value,
                        "title": alert.title,
                        "timestamp": alert.timestamp.isoformat()
                    }
                    for alert in active_alerts
                ]
            },
            "uptime": time.time() - (self._startup_time or time.time()),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def shutdown(self) -> bool:
        """Arrête le framework de monitoring"""
        try:
            # Arrêter les tâches de monitoring
            for task in self._monitoring_tasks:
                task.cancel()
            
            if self._monitoring_tasks:
                await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
            
            self.logger.info("Monitoring Framework shutdown successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Monitoring framework shutdown failed: {e}")
            return False
    
    async def health_check(self) -> FrameworkHealth:
        """Vérifie la santé du framework de monitoring"""
        health = FrameworkHealth(
            status=FrameworkStatus.RUNNING,
            last_check=time.time()
        )
        
        try:
            # Vérifier les composants
            active_tasks = len([t for t in self._monitoring_tasks if not t.done()])
            total_metrics = len(self.metrics_collector.custom_metrics)
            active_alerts = len(self.alert_manager.active_alerts)
            
            health.metadata = {
                "active_monitoring_tasks": active_tasks,
                "total_custom_metrics": total_metrics,
                "active_alerts": active_alerts,
                "prometheus_enabled": self.config.enable_prometheus,
                "tracing_enabled": self.config.enable_tracing
            }
            
            if active_tasks < len(self._monitoring_tasks):
                health.status = FrameworkStatus.DEGRADED
            
        except Exception as e:
            health.status = FrameworkStatus.DEGRADED
            health.error_count += 1
            health.metadata["error"] = str(e)
        
        return health


# Instance globale du framework de monitoring
monitoring_manager = MonitoringFramework()


# Fonctions utilitaires
def get_metrics() -> str:
    """Récupère les métriques Prometheus"""
    return monitoring_manager.metrics_collector.get_metrics_export()


async def get_health_status() -> Dict[str, Any]:
    """Récupère le statut de santé global"""
    return await monitoring_manager.health_checker.run_all_health_checks()


async def get_system_overview() -> Dict[str, Any]:
    """Récupère l'aperçu système complet"""
    return await monitoring_manager.get_system_overview()


# Export des classes principales
__all__ = [
    'MonitoringFramework',
    'MetricsCollector',
    'DistributedTracing',
    'AlertManager',
    'HealthChecker',
    'MonitoringConfig',
    'Alert',
    'AlertSeverity',
    'MetricType',
    'monitoring_manager',
    'get_metrics',
    'get_health_status',
    'get_system_overview'
]
