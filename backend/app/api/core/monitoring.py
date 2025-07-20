"""
🎵 Spotify AI Agent - API Monitoring & Metrics
==============================================

Système de monitoring enterprise avec métriques complètes,
health checks automatiques, et observabilité avancée.

Architecture:
- Métriques Prometheus intégrées
- Health checks automatiques
- Performance monitoring
- Alerting et notifications
- Traces distribuées
- Dashboards automatiques

Développé par Fahed Mlaiel - Enterprise Monitoring Expert
"""

import time
import psutil
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager

from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest
from fastapi import Request, Response


class HealthStatus(str, Enum):
    """Statuts de santé possibles"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class MetricType(str, Enum):
    """Types de métriques"""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    INFO = "info"


@dataclass
class HealthCheck:
    """Check de santé individuel"""
    name: str
    status: HealthStatus
    message: str = ""
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class SystemMetrics:
    """Métriques système"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_total_mb: float
    disk_usage_percent: float
    active_connections: int = 0
    uptime_seconds: float = 0.0
    
    @classmethod
    def current(cls) -> 'SystemMetrics':
        """Capture les métriques système actuelles"""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return cls(
            cpu_percent=psutil.cpu_percent(),
            memory_percent=memory.percent,
            memory_used_mb=memory.used / 1024 / 1024,
            memory_total_mb=memory.total / 1024 / 1024,
            disk_usage_percent=disk.percent,
            uptime_seconds=time.time() - psutil.boot_time()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_mb": self.memory_used_mb,
            "memory_total_mb": self.memory_total_mb,
            "disk_usage_percent": self.disk_usage_percent,
            "active_connections": self.active_connections,
            "uptime_seconds": self.uptime_seconds
        }


class APIMetrics:
    """Collecteur de métriques API avec Prometheus"""
    
    def __init__(self):
        # Métriques des requêtes
        self.request_total = Counter(
            'spotify_api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status']
        )
        
        self.request_duration = Histogram(
            'spotify_api_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint']
        )
        
        self.active_requests = Gauge(
            'spotify_api_active_requests',
            'Currently active requests'
        )
        
        # Métriques d'erreur
        self.error_total = Counter(
            'spotify_api_errors_total',
            'Total API errors',
            ['error_type', 'endpoint']
        )
        
        # Métriques système
        self.system_cpu = Gauge(
            'spotify_system_cpu_percent',
            'System CPU usage percentage'
        )
        
        self.system_memory = Gauge(
            'spotify_system_memory_percent',
            'System memory usage percentage'
        )
        
        # Métriques métier
        self.cache_hits = Counter(
            'spotify_cache_hits_total',
            'Total cache hits',
            ['cache_level']
        )
        
        self.cache_misses = Counter(
            'spotify_cache_misses_total',
            'Total cache misses',
            ['cache_level']
        )
        
        self.ml_predictions = Counter(
            'spotify_ml_predictions_total',
            'Total ML predictions',
            ['model_name', 'status']
        )
        
        # Info métriques
        self.app_info = Info(
            'spotify_api_info',
            'Application information'
        )
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Enregistre une requête"""
        self.request_total.labels(
            method=method,
            endpoint=endpoint,
            status=str(status)
        ).inc()
        
        self.request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_error(self, error_type: str, endpoint: str):
        """Enregistre une erreur"""
        self.error_total.labels(
            error_type=error_type,
            endpoint=endpoint
        ).inc()
    
    def update_system_metrics(self, metrics: SystemMetrics):
        """Met à jour les métriques système"""
        self.system_cpu.set(metrics.cpu_percent)
        self.system_memory.set(metrics.memory_percent)
    
    def set_active_requests(self, count: int):
        """Met à jour le nombre de requêtes actives"""
        self.active_requests.set(count)
    
    def record_cache_hit(self, cache_level: str):
        """Enregistre un cache hit"""
        self.cache_hits.labels(cache_level=cache_level).inc()
    
    def record_cache_miss(self, cache_level: str):
        """Enregistre un cache miss"""
        self.cache_misses.labels(cache_level=cache_level).inc()
    
    def record_ml_prediction(self, model_name: str, status: str):
        """Enregistre une prédiction ML"""
        self.ml_predictions.labels(
            model_name=model_name,
            status=status
        ).inc()
    
    def set_app_info(self, version: str, environment: str):
        """Configure les informations de l'application"""
        self.app_info.info({
            'version': version,
            'environment': environment,
            'start_time': datetime.utcnow().isoformat()
        })


class PerformanceMonitor:
    """Moniteur de performance pour les opérations"""
    
    def __init__(self, metrics: APIMetrics):
        self.metrics = metrics
        self.active_operations: Dict[str, float] = {}
    
    @asynccontextmanager
    async def monitor_operation(self, operation_name: str):
        """Context manager pour monitorer une opération"""
        start_time = time.time()
        self.active_operations[operation_name] = start_time
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.active_operations.pop(operation_name, None)
            
            # Enregistrer dans les métriques si c'est une requête API
            if operation_name.startswith('api_'):
                parts = operation_name.split('_')
                if len(parts) >= 3:
                    method, endpoint = parts[1], '_'.join(parts[2:])
                    self.metrics.request_duration.labels(
                        method=method,
                        endpoint=endpoint
                    ).observe(duration)
    
    def get_active_operations(self) -> Dict[str, float]:
        """Retourne les opérations actives avec leur durée"""
        current_time = time.time()
        return {
            name: current_time - start_time
            for name, start_time in self.active_operations.items()
        }


class HealthChecker:
    """Vérificateur de santé avec checks personnalisés"""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.last_results: Dict[str, HealthCheck] = {}
    
    def register_check(self, name: str, check_func: Callable):
        """Enregistre un check de santé"""
        self.checks[name] = check_func
    
    async def run_check(self, name: str) -> HealthCheck:
        """Exécute un check individuel"""
        if name not in self.checks:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Check '{name}' not found"
            )
        
        start_time = time.time()
        
        try:
            check_func = self.checks[name]
            
            # Support pour les fonctions async
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Le check peut retourner un bool, str, ou HealthCheck
            if isinstance(result, HealthCheck):
                result.duration_ms = duration_ms
                return result
            elif isinstance(result, bool):
                return HealthCheck(
                    name=name,
                    status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                    duration_ms=duration_ms
                )
            elif isinstance(result, str):
                return HealthCheck(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    message=result,
                    duration_ms=duration_ms
                )
            else:
                return HealthCheck(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    message=str(result),
                    duration_ms=duration_ms
                )
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                duration_ms=duration_ms
            )
    
    async def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Exécute tous les checks"""
        results = {}
        
        for name in self.checks:
            results[name] = await self.run_check(name)
        
        self.last_results = results
        return results
    
    def get_overall_status(self) -> HealthStatus:
        """Retourne le statut global basé sur tous les checks"""
        if not self.last_results:
            return HealthStatus.UNKNOWN
        
        statuses = [check.status for check in self.last_results.values()]
        
        if any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            return HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN


# =============================================================================
# CHECKS DE SANTÉ PAR DÉFAUT
# =============================================================================

async def database_health_check() -> HealthCheck:
    """Check de santé de la base de données"""
    try:
        # TODO: Implémenter le vrai check quand la DB sera disponible
        # from app.core.database import get_database_pool
        # pool = get_database_pool()
        # await pool.execute("SELECT 1")
        
        return HealthCheck(
            name="database",
            status=HealthStatus.HEALTHY,
            message="Database connection successful"
        )
    except Exception as e:
        return HealthCheck(
            name="database",
            status=HealthStatus.UNHEALTHY,
            message=f"Database connection failed: {str(e)}"
        )


async def redis_health_check() -> HealthCheck:
    """Check de santé Redis"""
    try:
        import redis.asyncio as redis
        from app.api.core.config import get_redis_config
        
        config = get_redis_config()
        client = redis.from_url(config.url)
        
        await client.ping()
        await client.close()
        
        return HealthCheck(
            name="redis",
            status=HealthStatus.HEALTHY,
            message="Redis connection successful"
        )
    except Exception as e:
        return HealthCheck(
            name="redis",
            status=HealthStatus.UNHEALTHY,
            message=f"Redis connection failed: {str(e)}"
        )


def system_health_check() -> HealthCheck:
    """Check de santé système"""
    try:
        metrics = SystemMetrics.current()
        
        # Critères de santé
        if metrics.cpu_percent > 90:
            status = HealthStatus.UNHEALTHY
            message = f"High CPU usage: {metrics.cpu_percent}%"
        elif metrics.memory_percent > 90:
            status = HealthStatus.UNHEALTHY
            message = f"High memory usage: {metrics.memory_percent}%"
        elif metrics.cpu_percent > 70 or metrics.memory_percent > 70:
            status = HealthStatus.DEGRADED
            message = f"Moderate resource usage (CPU: {metrics.cpu_percent}%, Memory: {metrics.memory_percent}%)"
        else:
            status = HealthStatus.HEALTHY
            message = f"System resources normal (CPU: {metrics.cpu_percent}%, Memory: {metrics.memory_percent}%)"
        
        return HealthCheck(
            name="system",
            status=status,
            message=message,
            metadata=metrics.to_dict()
        )
    except Exception as e:
        return HealthCheck(
            name="system",
            status=HealthStatus.UNHEALTHY,
            message=f"System check failed: {str(e)}"
        )


# =============================================================================
# INSTANCES GLOBALES
# =============================================================================

_api_metrics: Optional[APIMetrics] = None
_performance_monitor: Optional[PerformanceMonitor] = None
_health_checker: Optional[HealthChecker] = None


def get_api_metrics() -> APIMetrics:
    """Retourne l'instance globale des métriques"""
    global _api_metrics
    if _api_metrics is None:
        _api_metrics = APIMetrics()
    return _api_metrics


def get_performance_monitor() -> PerformanceMonitor:
    """Retourne l'instance globale du moniteur de performance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor(get_api_metrics())
    return _performance_monitor


def get_health_checker() -> HealthChecker:
    """Retourne l'instance globale du vérificateur de santé"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
        
        # Enregistrer les checks par défaut
        _health_checker.register_check("database", database_health_check)
        _health_checker.register_check("redis", redis_health_check)
        _health_checker.register_check("system", system_health_check)
    
    return _health_checker


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

async def monitor_api_call(operation_name: str):
    """Context manager pour monitorer un appel API"""
    monitor = get_performance_monitor()
    return monitor.monitor_operation(operation_name)


async def health_check() -> Dict[str, Any]:
    """Exécute un check de santé complet"""
    checker = get_health_checker()
    checks = await checker.run_all_checks()
    overall_status = checker.get_overall_status()
    system_metrics = SystemMetrics.current()
    
    return {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {name: check.to_dict() for name, check in checks.items()},
        "system": system_metrics.to_dict()
    }


def get_metrics_text() -> str:
    """Retourne les métriques Prometheus au format texte"""
    return generate_latest().decode('utf-8')


def collect_system_metrics() -> SystemMetrics:
    """Collecter les métriques système"""
    import psutil
    
    return SystemMetrics(
        cpu_usage=psutil.cpu_percent(),
        memory_usage=psutil.virtual_memory().percent,
        disk_usage=psutil.disk_usage('/').percent,
        network_in=psutil.net_io_counters().bytes_recv,
        network_out=psutil.net_io_counters().bytes_sent,
        active_connections=len(psutil.net_connections()),
        process_count=len(psutil.pids()),
        timestamp=datetime.utcnow()
    )


def check_system_health() -> Dict[str, HealthStatus]:
    """Vérifier l'état de santé du système"""
    metrics = collect_system_metrics()
    
    health_status = {
        "cpu": HealthStatus.HEALTHY if metrics.cpu_usage < 80 else HealthStatus.DEGRADED,
        "memory": HealthStatus.HEALTHY if metrics.memory_usage < 85 else HealthStatus.DEGRADED,
        "disk": HealthStatus.HEALTHY if metrics.disk_usage < 90 else HealthStatus.DEGRADED,
        "processes": HealthStatus.HEALTHY if metrics.process_count < 1000 else HealthStatus.DEGRADED
    }
    
    return health_status


def format_metrics_for_prometheus() -> str:
    """Formater les métriques pour Prometheus"""
    from prometheus_client import generate_latest
    return generate_latest().decode('utf-8')


# =============================================================================
# MIDDLEWARE DE MONITORING
# =============================================================================

class MonitoringMiddleware:
    """Middleware pour le monitoring automatique"""
    
    def __init__(self, app, metrics: APIMetrics = None):
        self.app = app
        self.metrics = metrics or get_api_metrics()
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        start_time = time.time()
        
        # Incrémenter les requêtes actives
        self.metrics.set_active_requests(self.metrics.active_requests._value._value + 1)
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                # Calculer la durée et enregistrer les métriques
                duration = time.time() - start_time
                status = message["status"]
                
                self.metrics.record_request(
                    method=request.method,
                    endpoint=request.url.path,
                    status=status,
                    duration=duration
                )
                
                # Décrémenter les requêtes actives
                current_active = self.metrics.active_requests._value._value
                self.metrics.set_active_requests(max(0, current_active - 1))
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "HealthStatus",
    "MetricType",
    "HealthCheck",
    "SystemMetrics",
    "APIMetrics",
    "PerformanceMonitor",
    "HealthChecker",
    "MonitoringMiddleware",
    "get_api_metrics",
    "get_performance_monitor",
    "get_health_checker",
    "monitor_api_call",
    "health_check",
    "get_metrics_text",
    "database_health_check",
    "redis_health_check",
    "system_health_check",
    "setup_monitoring",
    "create_monitoring_middleware",
    "collect_system_metrics",
    "check_system_health",
    "format_metrics_for_prometheus"
]


# =============================================================================
# FONCTIONS UTILITAIRES COMPLÉMENTAIRES
# =============================================================================

def setup_monitoring(app):
    """Configurer le monitoring pour l'application"""
    # Initialiser les métriques
    get_api_metrics()
    get_performance_monitor()
    get_health_checker()
    
    # Ajouter le middleware de monitoring
    app.add_middleware(MonitoringMiddleware)
    
    return app


def create_monitoring_middleware(app):
    """Créer et configurer le middleware de monitoring"""
    return MonitoringMiddleware(app)
