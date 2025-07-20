# 🎵 ML Analytics Monitoring
# ==========================
# 
# Système de monitoring enterprise pour ML Analytics
# Surveillance temps réel et alertes intelligentes
#
# 🎖️ Expert: Architecte Microservices + DBA & Data Engineer

"""
🔍 ML Analytics Monitoring System
==================================

Comprehensive monitoring and alerting system:
- Real-time performance monitoring
- Model drift detection
- Data quality monitoring
- Alert management and notifications
- Health checks and diagnostics
- Metrics collection and aggregation
"""

import asyncio
import logging
import time
import json
import threading
from typing import Dict, Any, List, Optional, Callable, Union, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from pathlib import Path
import psutil
import aioredis
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import schedule
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Niveaux de sévérité des alertes"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types de métriques"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Alert:
    """Alerte système"""
    id: str
    severity: AlertSeverity
    title: str
    message: str
    source: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def resolve(self):
        """Résolution de l'alerte"""
        self.resolved = True
        self.resolved_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['resolved_at'] = self.resolved_at.isoformat() if self.resolved_at else None
        data['severity'] = self.severity.value
        return data


@dataclass
class HealthCheck:
    """Contrôle de santé"""
    name: str
    status: str
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class MetricCollector:
    """Collecteur de métriques"""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        self.counters: Dict[str, Counter] = {}
        self.gauges: Dict[str, Gauge] = {}
        self.histograms: Dict[str, Histogram] = {}
        self._lock = threading.Lock()
    
    def create_counter(self, name: str, description: str, labels: Optional[List[str]] = None) -> Counter:
        """Création d'un compteur"""
        with self._lock:
            if name not in self.counters:
                self.counters[name] = Counter(
                    name, description, labels or []
                )
            return self.counters[name]
    
    def create_gauge(self, name: str, description: str, labels: Optional[List[str]] = None) -> Gauge:
        """Création d'une jauge"""
        with self._lock:
            if name not in self.gauges:
                self.gauges[name] = Gauge(
                    name, description, labels or []
                )
            return self.gauges[name]
    
    def create_histogram(self, name: str, description: str, labels: Optional[List[str]] = None) -> Histogram:
        """Création d'un histogramme"""
        with self._lock:
            if name not in self.histograms:
                self.histograms[name] = Histogram(
                    name, description, labels or []
                )
            return self.histograms[name]
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Incrémentation d'un compteur"""
        if name in self.counters:
            if labels:
                self.counters[name].labels(**labels).inc(value)
            else:
                self.counters[name].inc(value)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Définition d'une valeur de jauge"""
        if name in self.gauges:
            if labels:
                self.gauges[name].labels(**labels).set(value)
            else:
                self.gauges[name].set(value)
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observation d'une valeur d'histogramme"""
        if name in self.histograms:
            if labels:
                self.histograms[name].labels(**labels).observe(value)
            else:
                self.histograms[name].observe(value)


class ModelDriftDetector:
    """Détecteur de dérive de modèle"""
    
    def __init__(self, reference_data: Optional[np.ndarray] = None):
        self.reference_data = reference_data
        self.reference_stats = self._calculate_stats(reference_data) if reference_data is not None else None
        self.drift_threshold = 0.1
        self.samples_window = 1000
        self.recent_samples = deque(maxlen=self.samples_window)
    
    def _calculate_stats(self, data: np.ndarray) -> Dict[str, float]:
        """Calcul des statistiques de référence"""
        return {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0),
            'median': np.median(data, axis=0)
        }
    
    def add_sample(self, sample: np.ndarray):
        """Ajout d'un échantillon"""
        self.recent_samples.append(sample)
    
    def detect_drift(self) -> Dict[str, Any]:
        """Détection de la dérive"""
        if not self.recent_samples or self.reference_stats is None:
            return {'drift_detected': False, 'reason': 'Insufficient data'}
        
        current_data = np.array(list(self.recent_samples))
        current_stats = self._calculate_stats(current_data)
        
        # Calcul de la dérive basée sur la divergence KL
        drift_score = self._calculate_kl_divergence(
            self.reference_stats, current_stats
        )
        
        drift_detected = drift_score > self.drift_threshold
        
        return {
            'drift_detected': drift_detected,
            'drift_score': drift_score,
            'threshold': self.drift_threshold,
            'samples_analyzed': len(self.recent_samples),
            'reference_stats': self.reference_stats,
            'current_stats': current_stats
        }
    
    def _calculate_kl_divergence(self, ref_stats: Dict, current_stats: Dict) -> float:
        """Calcul de la divergence KL simplifiée"""
        # Approximation basée sur les moyennes et écarts-types
        mean_diff = np.mean(np.abs(current_stats['mean'] - ref_stats['mean']))
        std_ratio = np.mean(current_stats['std'] / (ref_stats['std'] + 1e-8))
        
        # Score de dérive combiné
        drift_score = mean_diff + abs(1 - std_ratio)
        return drift_score


class DataQualityMonitor:
    """Moniteur de qualité des données"""
    
    def __init__(self):
        self.quality_rules: List[Callable] = []
        self.quality_scores: Dict[str, float] = {}
    
    def add_rule(self, name: str, rule_func: Callable[[Any], bool]):
        """Ajout d'une règle de qualité"""
        self.quality_rules.append((name, rule_func))
    
    def check_data_quality(self, data: Any) -> Dict[str, Any]:
        """Vérification de la qualité des données"""
        results = {}
        total_score = 0
        
        for name, rule_func in self.quality_rules:
            try:
                passed = rule_func(data)
                results[name] = {'passed': passed, 'score': 1.0 if passed else 0.0}
                total_score += 1.0 if passed else 0.0
            except Exception as e:
                results[name] = {'passed': False, 'score': 0.0, 'error': str(e)}
        
        overall_score = total_score / len(self.quality_rules) if self.quality_rules else 0.0
        
        return {
            'overall_score': overall_score,
            'rule_results': results,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def add_standard_rules(self):
        """Ajout des règles de qualité standard"""
        # Règle: pas de valeurs nulles
        self.add_rule(
            'no_null_values',
            lambda data: not pd.DataFrame(data).isnull().any().any() if hasattr(data, '__iter__') else True
        )
        
        # Règle: plage de valeurs raisonnable
        self.add_rule(
            'reasonable_range',
            lambda data: np.all(np.isfinite(np.array(data))) if hasattr(data, '__iter__') else True
        )
        
        # Règle: pas de doublons
        self.add_rule(
            'no_duplicates',
            lambda data: not pd.DataFrame(data).duplicated().any() if hasattr(data, '__iter__') else True
        )


class AlertManager:
    """Gestionnaire d'alertes"""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_handlers: Dict[AlertSeverity, List[Callable]] = defaultdict(list)
        self.rate_limits: Dict[str, datetime] = {}
        self.rate_limit_window = timedelta(minutes=5)
        self._lock = threading.Lock()
    
    def add_handler(self, severity: AlertSeverity, handler: Callable[[Alert], None]):
        """Ajout d'un gestionnaire d'alerte"""
        self.alert_handlers[severity].append(handler)
    
    def create_alert(
        self,
        alert_id: str,
        severity: AlertSeverity,
        title: str,
        message: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """Création d'une nouvelle alerte"""
        with self._lock:
            # Vérification de la limitation de taux
            if self._is_rate_limited(alert_id):
                return None
            
            alert = Alert(
                id=alert_id,
                severity=severity,
                title=title,
                message=message,
                source=source,
                metadata=metadata or {}
            )
            
            self.alerts[alert_id] = alert
            self._update_rate_limit(alert_id)
            
            # Déclenchement des handlers
            for handler in self.alert_handlers[severity]:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Erreur dans le handler d'alerte: {e}")
            
            return alert
    
    def resolve_alert(self, alert_id: str):
        """Résolution d'une alerte"""
        with self._lock:
            if alert_id in self.alerts:
                self.alerts[alert_id].resolve()
    
    def get_active_alerts(self) -> List[Alert]:
        """Récupération des alertes actives"""
        with self._lock:
            return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Résumé des alertes"""
        active_alerts = self.get_active_alerts()
        
        summary = {
            'total_active': len(active_alerts),
            'by_severity': defaultdict(int),
            'by_source': defaultdict(int),
            'oldest_alert': None,
            'newest_alert': None
        }
        
        if active_alerts:
            for alert in active_alerts:
                summary['by_severity'][alert.severity.value] += 1
                summary['by_source'][alert.source] += 1
            
            # Plus ancienne et plus récente
            sorted_alerts = sorted(active_alerts, key=lambda a: a.timestamp)
            summary['oldest_alert'] = sorted_alerts[0].to_dict()
            summary['newest_alert'] = sorted_alerts[-1].to_dict()
        
        return dict(summary)
    
    def _is_rate_limited(self, alert_id: str) -> bool:
        """Vérification de la limitation de taux"""
        if alert_id in self.rate_limits:
            return datetime.utcnow() - self.rate_limits[alert_id] < self.rate_limit_window
        return False
    
    def _update_rate_limit(self, alert_id: str):
        """Mise à jour de la limitation de taux"""
        self.rate_limits[alert_id] = datetime.utcnow()


class HealthMonitor:
    """Moniteur de santé du système"""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable] = {}
        self.last_results: Dict[str, HealthCheck] = {}
    
    def register_check(self, name: str, check_func: Callable[[], HealthCheck]):
        """Enregistrement d'un contrôle de santé"""
        self.health_checks[name] = check_func
    
    async def run_check(self, name: str) -> HealthCheck:
        """Exécution d'un contrôle de santé"""
        if name not in self.health_checks:
            return HealthCheck(
                name=name,
                status="error",
                message="Check not found"
            )
        
        start_time = time.time()
        try:
            result = self.health_checks[name]()
            if asyncio.iscoroutine(result):
                result = await result
            result.duration_ms = (time.time() - start_time) * 1000
            self.last_results[name] = result
            return result
        except Exception as e:
            result = HealthCheck(
                name=name,
                status="error",
                message=str(e),
                duration_ms=(time.time() - start_time) * 1000
            )
            self.last_results[name] = result
            return result
    
    async def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Exécution de tous les contrôles"""
        results = {}
        for name in self.health_checks:
            results[name] = await self.run_check(name)
        return results
    
    def get_system_health(self) -> Dict[str, Any]:
        """État de santé global du système"""
        all_healthy = all(
            result.status == "healthy"
            for result in self.last_results.values()
        )
        
        return {
            'overall_status': 'healthy' if all_healthy else 'unhealthy',
            'checks': {
                name: result.to_dict()
                for name, result in self.last_results.items()
            },
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def add_standard_checks(self):
        """Ajout des contrôles standard"""
        # Contrôle de la mémoire
        def memory_check() -> HealthCheck:
            memory = psutil.virtual_memory()
            status = "healthy" if memory.percent < 85 else "unhealthy"
            return HealthCheck(
                name="memory",
                status=status,
                message=f"Memory usage: {memory.percent:.1f}%",
                metadata={'percent': memory.percent, 'available_gb': memory.available / 1024**3}
            )
        
        # Contrôle du CPU
        def cpu_check() -> HealthCheck:
            cpu_percent = psutil.cpu_percent(interval=1)
            status = "healthy" if cpu_percent < 80 else "unhealthy"
            return HealthCheck(
                name="cpu",
                status=status,
                message=f"CPU usage: {cpu_percent:.1f}%",
                metadata={'percent': cpu_percent}
            )
        
        # Contrôle de l'espace disque
        def disk_check() -> HealthCheck:
            disk = psutil.disk_usage('/')
            percent_used = (disk.used / disk.total) * 100
            status = "healthy" if percent_used < 85 else "unhealthy"
            return HealthCheck(
                name="disk",
                status=status,
                message=f"Disk usage: {percent_used:.1f}%",
                metadata={'percent': percent_used, 'free_gb': disk.free / 1024**3}
            )
        
        self.register_check("memory", memory_check)
        self.register_check("cpu", cpu_check)
        self.register_check("disk", disk_check)


class MLAnalyticsMonitor:
    """Moniteur principal ML Analytics"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metric_collector = MetricCollector()
        self.alert_manager = AlertManager()
        self.health_monitor = HealthMonitor()
        self.drift_detector = ModelDriftDetector()
        self.quality_monitor = DataQualityMonitor()
        
        self.running = False
        self.monitoring_tasks: Set[asyncio.Task] = set()
        
        # Configuration des métriques Prometheus
        self._setup_prometheus_metrics()
        
        # Configuration des contrôles de santé standard
        self.health_monitor.add_standard_checks()
        self.quality_monitor.add_standard_rules()
        
        # Configuration des handlers d'alertes
        self._setup_alert_handlers()
    
    def _setup_prometheus_metrics(self):
        """Configuration des métriques Prometheus"""
        # Métriques de base
        self.metric_collector.create_counter(
            'ml_analytics_requests_total',
            'Total number of ML analytics requests',
            ['endpoint', 'status']
        )
        
        self.metric_collector.create_histogram(
            'ml_analytics_request_duration_seconds',
            'ML analytics request duration in seconds',
            ['endpoint']
        )
        
        self.metric_collector.create_gauge(
            'ml_analytics_model_accuracy',
            'Current model accuracy',
            ['model_id']
        )
        
        self.metric_collector.create_gauge(
            'ml_analytics_active_alerts',
            'Number of active alerts',
            ['severity']
        )
    
    def _setup_alert_handlers(self):
        """Configuration des handlers d'alertes"""
        def log_alert(alert: Alert):
            level = {
                AlertSeverity.INFO: logging.INFO,
                AlertSeverity.WARNING: logging.WARNING,
                AlertSeverity.ERROR: logging.ERROR,
                AlertSeverity.CRITICAL: logging.CRITICAL
            }[alert.severity]
            
            logger.log(level, f"[ALERT] {alert.title}: {alert.message}")
        
        # Handler de logging pour toutes les sévérités
        for severity in AlertSeverity:
            self.alert_manager.add_handler(severity, log_alert)
    
    async def start_monitoring(self):
        """Démarrage du monitoring"""
        if self.running:
            return
        
        self.running = True
        
        # Démarrage du serveur Prometheus
        prometheus_port = self.config.get('prometheus_port', 8000)
        start_http_server(prometheus_port)
        logger.info(f"Serveur Prometheus démarré sur le port {prometheus_port}")
        
        # Tâches de monitoring périodique
        self.monitoring_tasks.add(
            asyncio.create_task(self._periodic_health_check())
        )
        self.monitoring_tasks.add(
            asyncio.create_task(self._periodic_metric_collection())
        )
        self.monitoring_tasks.add(
            asyncio.create_task(self._periodic_alert_cleanup())
        )
        
        logger.info("Monitoring ML Analytics démarré")
    
    async def stop_monitoring(self):
        """Arrêt du monitoring"""
        if not self.running:
            return
        
        self.running = False
        
        # Annulation des tâches
        for task in self.monitoring_tasks:
            task.cancel()
        
        # Attente de l'arrêt des tâches
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        self.monitoring_tasks.clear()
        
        logger.info("Monitoring ML Analytics arrêté")
    
    async def _periodic_health_check(self):
        """Contrôles de santé périodiques"""
        while self.running:
            try:
                results = await self.health_monitor.run_all_checks()
                
                # Création d'alertes pour les problèmes de santé
                for name, result in results.items():
                    if result.status == "unhealthy":
                        self.alert_manager.create_alert(
                            alert_id=f"health_{name}",
                            severity=AlertSeverity.WARNING,
                            title=f"Health check failed: {name}",
                            message=result.message,
                            source="health_monitor",
                            metadata=result.metadata
                        )
                    else:
                        # Résolution des alertes si le service est de nouveau sain
                        self.alert_manager.resolve_alert(f"health_{name}")
                
                await asyncio.sleep(60)  # Vérification toutes les minutes
            except Exception as e:
                logger.error(f"Erreur lors du contrôle de santé: {e}")
                await asyncio.sleep(60)
    
    async def _periodic_metric_collection(self):
        """Collection périodique de métriques"""
        while self.running:
            try:
                # Mise à jour des métriques d'alertes
                alert_summary = self.alert_manager.get_alert_summary()
                for severity, count in alert_summary['by_severity'].items():
                    self.metric_collector.set_gauge(
                        'ml_analytics_active_alerts',
                        count,
                        {'severity': severity}
                    )
                
                await asyncio.sleep(30)  # Collection toutes les 30 secondes
            except Exception as e:
                logger.error(f"Erreur lors de la collection de métriques: {e}")
                await asyncio.sleep(30)
    
    async def _periodic_alert_cleanup(self):
        """Nettoyage périodique des alertes"""
        while self.running:
            try:
                # Nettoyage des alertes anciennes résolues
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                to_remove = []
                
                for alert_id, alert in self.alert_manager.alerts.items():
                    if (alert.resolved and 
                        alert.resolved_at and 
                        alert.resolved_at < cutoff_time):
                        to_remove.append(alert_id)
                
                for alert_id in to_remove:
                    del self.alert_manager.alerts[alert_id]
                
                await asyncio.sleep(3600)  # Nettoyage toutes les heures
            except Exception as e:
                logger.error(f"Erreur lors du nettoyage des alertes: {e}")
                await asyncio.sleep(3600)
    
    @asynccontextmanager
    async def monitor_operation(self, operation_name: str):
        """Context manager pour le monitoring d'opérations"""
        start_time = time.time()
        
        try:
            yield
            
            # Métrique de succès
            duration = time.time() - start_time
            self.metric_collector.observe_histogram(
                'ml_analytics_request_duration_seconds',
                duration,
                {'endpoint': operation_name}
            )
            self.metric_collector.increment_counter(
                'ml_analytics_requests_total',
                labels={'endpoint': operation_name, 'status': 'success'}
            )
            
        except Exception as e:
            # Métrique d'erreur
            duration = time.time() - start_time
            self.metric_collector.observe_histogram(
                'ml_analytics_request_duration_seconds',
                duration,
                {'endpoint': operation_name}
            )
            self.metric_collector.increment_counter(
                'ml_analytics_requests_total',
                labels={'endpoint': operation_name, 'status': 'error'}
            )
            
            # Alerte d'erreur
            self.alert_manager.create_alert(
                alert_id=f"operation_error_{operation_name}_{int(time.time())}",
                severity=AlertSeverity.ERROR,
                title=f"Operation failed: {operation_name}",
                message=str(e),
                source="operation_monitor"
            )
            
            raise
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """État du monitoring"""
        return {
            'running': self.running,
            'health': self.health_monitor.get_system_health(),
            'alerts': self.alert_manager.get_alert_summary(),
            'metrics_count': {
                'counters': len(self.metric_collector.counters),
                'gauges': len(self.metric_collector.gauges),
                'histograms': len(self.metric_collector.histograms)
            }
        }


# Instance globale du moniteur
ml_monitor = MLAnalyticsMonitor()

# Exports publics
__all__ = [
    'Alert',
    'AlertSeverity',
    'HealthCheck',
    'MetricCollector',
    'ModelDriftDetector',
    'DataQualityMonitor',
    'AlertManager',
    'HealthMonitor',
    'MLAnalyticsMonitor',
    'ml_monitor'
]
