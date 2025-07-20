"""
Monitoring de Locales Avancé pour Spotify AI Agent
Système de monitoring et métriques en temps réel pour les locales
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import threading
import json
import weakref
from enum import Enum

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types de métriques"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertLevel(Enum):
    """Niveaux d'alerte"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """Point de métrique"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alerte de monitoring"""
    id: str
    level: AlertLevel
    title: str
    description: str
    timestamp: datetime
    source: str
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class MonitoringConfig:
    """Configuration du monitoring"""
    enable_metrics: bool = True
    enable_alerts: bool = True
    enable_tracing: bool = True
    metrics_retention: int = 86400  # 24 heures
    alerts_retention: int = 604800  # 7 jours
    collection_interval: int = 10  # 10 secondes
    alert_cooldown: int = 300  # 5 minutes
    performance_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'response_time_p95': 1000.0,  # ms
        'error_rate': 0.05,  # 5%
        'cache_hit_rate': 0.8,  # 80%
        'memory_usage': 0.8  # 80%
    })


class MetricsCollector(ABC):
    """Interface pour les collecteurs de métriques"""
    
    @abstractmethod
    async def collect_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        labels: Dict[str, str] = None
    ):
        """Collecte une métrique"""
        pass
    
    @abstractmethod
    async def get_metrics(
        self,
        name_pattern: str = "*",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, List[MetricPoint]]:
        """Récupère les métriques"""
        pass


class InMemoryMetricsCollector(MetricsCollector):
    """Collecteur de métriques en mémoire"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self._metrics = defaultdict(lambda: deque(maxlen=10000))
        self._lock = threading.RLock()
        self._cleanup_task = None
        self._running = False
    
    async def start(self):
        """Démarre le collecteur"""
        if not self._running:
            self._running = True
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop(self):
        """Arrête le collecteur"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def collect_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        labels: Dict[str, str] = None
    ):
        """Collecte une métrique"""
        try:
            point = MetricPoint(
                timestamp=datetime.now(),
                value=value,
                labels=labels or {},
                metadata={'type': metric_type.value}
            )
            
            with self._lock:
                self._metrics[name].append(point)
                
        except Exception as e:
            logger.error(f"Metric collection error: {e}")
    
    async def get_metrics(
        self,
        name_pattern: str = "*",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, List[MetricPoint]]:
        """Récupère les métriques"""
        try:
            with self._lock:
                result = {}
                
                for name, points in self._metrics.items():
                    if self._matches_pattern(name, name_pattern):
                        filtered_points = []
                        
                        for point in points:
                            if start_time and point.timestamp < start_time:
                                continue
                            if end_time and point.timestamp > end_time:
                                continue
                            filtered_points.append(point)
                        
                        if filtered_points:
                            result[name] = filtered_points
                
                return result
                
        except Exception as e:
            logger.error(f"Metrics retrieval error: {e}")
            return {}
    
    def _matches_pattern(self, name: str, pattern: str) -> bool:
        """Vérifie si le nom correspond au pattern"""
        if pattern == "*":
            return True
        
        import fnmatch
        return fnmatch.fnmatch(name, pattern)
    
    async def _cleanup_loop(self):
        """Boucle de nettoyage des métriques anciennes"""
        while self._running:
            try:
                await asyncio.sleep(300)  # Toutes les 5 minutes
                await self._cleanup_old_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def _cleanup_old_metrics(self):
        """Nettoie les métriques anciennes"""
        try:
            cutoff_time = datetime.now() - timedelta(seconds=self.config.metrics_retention)
            
            with self._lock:
                for name, points in self._metrics.items():
                    # Filtrer les points trop anciens
                    while points and points[0].timestamp < cutoff_time:
                        points.popleft()
                        
        except Exception as e:
            logger.error(f"Metrics cleanup error: {e}")


class PrometheusMetricsCollector(MetricsCollector):
    """Collecteur de métriques compatible Prometheus"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self._metrics = {}
        self._lock = threading.RLock()
    
    async def collect_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        labels: Dict[str, str] = None
    ):
        """Collecte une métrique au format Prometheus"""
        try:
            # Format Prometheus
            metric_name = self._sanitize_name(name)
            label_str = self._format_labels(labels or {})
            
            metric_line = f"{metric_name}{label_str} {value} {int(time.time() * 1000)}"
            
            with self._lock:
                if metric_name not in self._metrics:
                    self._metrics[metric_name] = {
                        'type': metric_type.value,
                        'help': f"Locale monitoring metric: {name}",
                        'samples': []
                    }
                
                self._metrics[metric_name]['samples'].append(metric_line)
                
                # Limiter le nombre d'échantillons
                if len(self._metrics[metric_name]['samples']) > 1000:
                    self._metrics[metric_name]['samples'] = self._metrics[metric_name]['samples'][-500:]
                    
        except Exception as e:
            logger.error(f"Prometheus metric collection error: {e}")
    
    async def get_metrics(
        self,
        name_pattern: str = "*",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, List[MetricPoint]]:
        """Récupère les métriques (format adapté)"""
        # Implémentation adaptée pour Prometheus
        return {}
    
    def get_prometheus_format(self) -> str:
        """Retourne les métriques au format Prometheus"""
        try:
            with self._lock:
                lines = []
                
                for name, metric_data in self._metrics.items():
                    lines.append(f"# HELP {name} {metric_data['help']}")
                    lines.append(f"# TYPE {name} {metric_data['type']}")
                    
                    for sample in metric_data['samples']:
                        lines.append(sample)
                    
                    lines.append("")
                
                return "\n".join(lines)
                
        except Exception as e:
            logger.error(f"Prometheus format error: {e}")
            return ""
    
    def _sanitize_name(self, name: str) -> str:
        """Nettoie le nom pour Prometheus"""
        import re
        return re.sub(r'[^a-zA-Z0-9_:]', '_', name)
    
    def _format_labels(self, labels: Dict[str, str]) -> str:
        """Formate les labels pour Prometheus"""
        if not labels:
            return ""
        
        label_pairs = [f'{k}="{v}"' for k, v in labels.items()]
        return "{" + ",".join(label_pairs) + "}"


class AlertManager:
    """Gestionnaire d'alertes"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self._alerts = {}
        self._alert_rules = []
        self._subscribers = weakref.WeakSet()
        self._lock = threading.RLock()
        self._stats = defaultdict(int)
    
    async def add_alert_rule(
        self,
        name: str,
        condition: Callable[[Dict[str, Any]], bool],
        level: AlertLevel,
        title: str,
        description: str,
        cooldown: Optional[int] = None
    ):
        """Ajoute une règle d'alerte"""
        rule = {
            'name': name,
            'condition': condition,
            'level': level,
            'title': title,
            'description': description,
            'cooldown': cooldown or self.config.alert_cooldown,
            'last_triggered': None
        }
        
        with self._lock:
            self._alert_rules.append(rule)
    
    async def evaluate_alerts(self, metrics: Dict[str, Any]):
        """Évalue les règles d'alerte"""
        try:
            current_time = datetime.now()
            
            for rule in self._alert_rules:
                try:
                    # Vérifier le cooldown
                    if (rule['last_triggered'] and 
                        current_time - rule['last_triggered'] < timedelta(seconds=rule['cooldown'])):
                        continue
                    
                    # Évaluer la condition
                    if rule['condition'](metrics):
                        alert = Alert(
                            id=f"{rule['name']}_{int(current_time.timestamp())}",
                            level=rule['level'],
                            title=rule['title'],
                            description=rule['description'],
                            timestamp=current_time,
                            source='locale_monitoring',
                            labels={'rule': rule['name']}
                        )
                        
                        await self._trigger_alert(alert)
                        rule['last_triggered'] = current_time
                        
                except Exception as e:
                    logger.error(f"Alert evaluation error for rule {rule['name']}: {e}")
                    
        except Exception as e:
            logger.error(f"Alert evaluation error: {e}")
    
    async def trigger_manual_alert(
        self,
        level: AlertLevel,
        title: str,
        description: str,
        source: str,
        labels: Dict[str, str] = None,
        metadata: Dict[str, Any] = None
    ):
        """Déclenche une alerte manuellement"""
        alert = Alert(
            id=f"manual_{int(datetime.now().timestamp())}",
            level=level,
            title=title,
            description=description,
            timestamp=datetime.now(),
            source=source,
            labels=labels or {},
            metadata=metadata or {}
        )
        
        await self._trigger_alert(alert)
    
    async def resolve_alert(self, alert_id: str):
        """Résout une alerte"""
        try:
            with self._lock:
                if alert_id in self._alerts:
                    alert = self._alerts[alert_id]
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    
                    await self._notify_subscribers('alert_resolved', alert)
                    self._stats['alerts_resolved'] += 1
                    
        except Exception as e:
            logger.error(f"Alert resolution error: {e}")
    
    async def get_active_alerts(self) -> List[Alert]:
        """Retourne les alertes actives"""
        with self._lock:
            return [alert for alert in self._alerts.values() if not alert.resolved]
    
    async def get_alert_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques d'alertes"""
        with self._lock:
            active_alerts = [a for a in self._alerts.values() if not a.resolved]
            
            by_level = defaultdict(int)
            for alert in active_alerts:
                by_level[alert.level.value] += 1
            
            return {
                'total_alerts': len(self._alerts),
                'active_alerts': len(active_alerts),
                'resolved_alerts': len([a for a in self._alerts.values() if a.resolved]),
                'alerts_by_level': dict(by_level),
                'stats': dict(self._stats),
                'rules_count': len(self._alert_rules)
            }
    
    def subscribe(self, callback: Callable):
        """S'abonne aux notifications d'alertes"""
        self._subscribers.add(callback)
    
    def unsubscribe(self, callback: Callable):
        """Se désabonne des notifications"""
        self._subscribers.discard(callback)
    
    async def _trigger_alert(self, alert: Alert):
        """Déclenche une alerte"""
        try:
            with self._lock:
                self._alerts[alert.id] = alert
                self._stats['alerts_triggered'] += 1
                self._stats[f'alerts_{alert.level.value}'] += 1
            
            await self._notify_subscribers('alert_triggered', alert)
            
            logger.warning(f"Alert triggered: {alert.title} ({alert.level.value})")
            
        except Exception as e:
            logger.error(f"Alert trigger error: {e}")
    
    async def _notify_subscribers(self, event_type: str, alert: Alert):
        """Notifie les abonnés"""
        for subscriber in list(self._subscribers):
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(event_type, alert)
                else:
                    subscriber(event_type, alert)
            except Exception as e:
                logger.warning(f"Subscriber notification error: {e}")


class LocaleMonitoring:
    """Système de monitoring principal pour les locales"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self._collectors = []
        self._alert_manager = AlertManager(config)
        self._traces = deque(maxlen=10000)
        self._performance_stats = defaultdict(list)
        self._lock = threading.RLock()
        self._monitoring_task = None
        self._running = False
        
        # Initialiser les collecteurs
        if config.enable_metrics:
            self._collectors.append(InMemoryMetricsCollector(config))
            # Ajouter Prometheus si disponible
            # self._collectors.append(PrometheusMetricsCollector(config))
    
    async def start(self):
        """Démarre le monitoring"""
        if not self._running:
            self._running = True
            
            # Démarrer les collecteurs
            for collector in self._collectors:
                if hasattr(collector, 'start'):
                    await collector.start()
            
            # Démarrer les règles d'alerte par défaut
            await self._setup_default_alert_rules()
            
            # Démarrer la boucle de monitoring
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            logger.info("Locale monitoring started")
    
    async def stop(self):
        """Arrête le monitoring"""
        self._running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Arrêter les collecteurs
        for collector in self._collectors:
            if hasattr(collector, 'stop'):
                await collector.stop()
        
        logger.info("Locale monitoring stopped")
    
    async def record_performance(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        tenant_id: Optional[str] = None,
        locale_code: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ):
        """Enregistre une métrique de performance"""
        try:
            labels = {
                'operation': operation,
                'status': 'success' if success else 'error'
            }
            
            if tenant_id:
                labels['tenant_id'] = tenant_id
            if locale_code:
                labels['locale_code'] = locale_code
            
            # Enregistrer la durée
            await self._record_metric(
                'locale_operation_duration_ms',
                duration_ms,
                MetricType.HISTOGRAM,
                labels
            )
            
            # Enregistrer le taux de succès
            await self._record_metric(
                'locale_operation_total',
                1.0,
                MetricType.COUNTER,
                labels
            )
            
            # Stocker pour l'analyse
            with self._lock:
                self._performance_stats[operation].append({
                    'timestamp': datetime.now(),
                    'duration_ms': duration_ms,
                    'success': success,
                    'tenant_id': tenant_id,
                    'locale_code': locale_code,
                    'metadata': metadata or {}
                })
                
                # Limiter la taille
                if len(self._performance_stats[operation]) > 1000:
                    self._performance_stats[operation] = self._performance_stats[operation][-500:]
            
        except Exception as e:
            logger.error(f"Performance recording error: {e}")
    
    async def record_cache_event(
        self,
        event_type: str,  # hit, miss, set, delete
        cache_level: str,  # memory, redis
        tenant_id: Optional[str] = None,
        locale_code: Optional[str] = None
    ):
        """Enregistre un événement de cache"""
        try:
            labels = {
                'event_type': event_type,
                'cache_level': cache_level
            }
            
            if tenant_id:
                labels['tenant_id'] = tenant_id
            if locale_code:
                labels['locale_code'] = locale_code
            
            await self._record_metric(
                'locale_cache_events_total',
                1.0,
                MetricType.COUNTER,
                labels
            )
            
        except Exception as e:
            logger.error(f"Cache event recording error: {e}")
    
    async def record_error(
        self,
        error_type: str,
        error_message: str,
        tenant_id: Optional[str] = None,
        locale_code: Optional[str] = None,
        severity: str = "error"
    ):
        """Enregistre une erreur"""
        try:
            labels = {
                'error_type': error_type,
                'severity': severity
            }
            
            if tenant_id:
                labels['tenant_id'] = tenant_id
            if locale_code:
                labels['locale_code'] = locale_code
            
            await self._record_metric(
                'locale_errors_total',
                1.0,
                MetricType.COUNTER,
                labels
            )
            
            # Déclencher une alerte si critique
            if severity == "critical":
                await self._alert_manager.trigger_manual_alert(
                    AlertLevel.CRITICAL,
                    f"Erreur critique: {error_type}",
                    error_message,
                    'locale_monitoring',
                    labels
                )
            
        except Exception as e:
            logger.error(f"Error recording error: {e}")
    
    async def add_trace(
        self,
        operation: str,
        start_time: datetime,
        end_time: datetime,
        success: bool,
        tenant_id: Optional[str] = None,
        locale_code: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ):
        """Ajoute une trace d'exécution"""
        if not self.config.enable_tracing:
            return
        
        try:
            trace = {
                'operation': operation,
                'start_time': start_time,
                'end_time': end_time,
                'duration_ms': (end_time - start_time).total_seconds() * 1000,
                'success': success,
                'tenant_id': tenant_id,
                'locale_code': locale_code,
                'metadata': metadata or {}
            }
            
            with self._lock:
                self._traces.append(trace)
            
        except Exception as e:
            logger.error(f"Trace recording error: {e}")
    
    async def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Retourne les données du tableau de bord"""
        try:
            # Collecter les métriques récentes
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=1)
            
            metrics = {}
            for collector in self._collectors:
                collector_metrics = await collector.get_metrics(
                    "*",
                    start_time,
                    end_time
                )
                metrics.update(collector_metrics)
            
            # Calculer les statistiques de performance
            perf_stats = await self._calculate_performance_stats()
            
            # Obtenir les alertes actives
            active_alerts = await self._alert_manager.get_active_alerts()
            alert_stats = await self._alert_manager.get_alert_stats()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'time_range': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat()
                },
                'performance': perf_stats,
                'alerts': {
                    'active': [
                        {
                            'id': alert.id,
                            'level': alert.level.value,
                            'title': alert.title,
                            'timestamp': alert.timestamp.isoformat()
                        }
                        for alert in active_alerts
                    ],
                    'stats': alert_stats
                },
                'metrics_summary': await self._summarize_metrics(metrics),
                'health_status': await self._calculate_health_status(perf_stats)
            }
            
        except Exception as e:
            logger.error(f"Dashboard generation error: {e}")
            return {}
    
    async def get_performance_report(
        self,
        operation: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Génère un rapport de performance"""
        try:
            if not start_time:
                start_time = datetime.now() - timedelta(hours=24)
            if not end_time:
                end_time = datetime.now()
            
            report = {
                'period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat()
                },
                'operations': {}
            }
            
            with self._lock:
                operations = [operation] if operation else self._performance_stats.keys()
                
                for op in operations:
                    if op not in self._performance_stats:
                        continue
                    
                    # Filtrer par période
                    filtered_stats = [
                        stat for stat in self._performance_stats[op]
                        if start_time <= stat['timestamp'] <= end_time
                    ]
                    
                    if not filtered_stats:
                        continue
                    
                    # Calculer les statistiques
                    durations = [s['duration_ms'] for s in filtered_stats]
                    success_count = len([s for s in filtered_stats if s['success']])
                    
                    report['operations'][op] = {
                        'total_requests': len(filtered_stats),
                        'success_rate': success_count / len(filtered_stats),
                        'avg_duration_ms': sum(durations) / len(durations),
                        'min_duration_ms': min(durations),
                        'max_duration_ms': max(durations),
                        'p95_duration_ms': self._percentile(durations, 95),
                        'p99_duration_ms': self._percentile(durations, 99)
                    }
            
            return report
            
        except Exception as e:
            logger.error(f"Performance report error: {e}")
            return {}
    
    async def _record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        labels: Dict[str, str] = None
    ):
        """Enregistre une métrique dans tous les collecteurs"""
        for collector in self._collectors:
            try:
                await collector.collect_metric(name, value, metric_type, labels)
            except Exception as e:
                logger.warning(f"Collector error: {e}")
    
    async def _monitoring_loop(self):
        """Boucle principale de monitoring"""
        while self._running:
            try:
                await asyncio.sleep(self.config.collection_interval)
                
                # Collecter les métriques système
                await self._collect_system_metrics()
                
                # Évaluer les alertes
                if self.config.enable_alerts:
                    metrics = await self._get_current_metrics()
                    await self._alert_manager.evaluate_alerts(metrics)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
    
    async def _setup_default_alert_rules(self):
        """Configure les règles d'alerte par défaut"""
        try:
            # Alerte de taux d'erreur élevé
            await self._alert_manager.add_alert_rule(
                'high_error_rate',
                lambda metrics: metrics.get('error_rate', 0) > self.config.performance_thresholds.get('error_rate', 0.05),
                AlertLevel.WARNING,
                'Taux d\'erreur élevé',
                'Le taux d\'erreur des locales dépasse le seuil acceptable'
            )
            
            # Alerte de temps de réponse élevé
            await self._alert_manager.add_alert_rule(
                'high_response_time',
                lambda metrics: metrics.get('avg_response_time', 0) > self.config.performance_thresholds.get('response_time_p95', 1000),
                AlertLevel.WARNING,
                'Temps de réponse élevé',
                'Le temps de réponse moyen des locales est trop élevé'
            )
            
            # Alerte de taux de cache faible
            await self._alert_manager.add_alert_rule(
                'low_cache_hit_rate',
                lambda metrics: metrics.get('cache_hit_rate', 1) < self.config.performance_thresholds.get('cache_hit_rate', 0.8),
                AlertLevel.INFO,
                'Taux de cache faible',
                'Le taux de succès du cache est inférieur aux attentes'
            )
            
        except Exception as e:
            logger.error(f"Alert rules setup error: {e}")
    
    async def _collect_system_metrics(self):
        """Collecte les métriques système"""
        try:
            # Métrique de santé générale
            await self._record_metric(
                'locale_monitoring_health',
                1.0,
                MetricType.GAUGE,
                {'status': 'healthy'}
            )
            
            # Nombre de traces actives
            with self._lock:
                trace_count = len(self._traces)
            
            await self._record_metric(
                'locale_traces_total',
                float(trace_count),
                MetricType.GAUGE
            )
            
        except Exception as e:
            logger.error(f"System metrics collection error: {e}")
    
    async def _get_current_metrics(self) -> Dict[str, Any]:
        """Récupère les métriques actuelles pour l'évaluation des alertes"""
        try:
            # Calculer les métriques en temps réel
            now = datetime.now()
            recent_time = now - timedelta(minutes=5)
            
            # Analyser les performances récentes
            recent_stats = []
            with self._lock:
                for operation_stats in self._performance_stats.values():
                    recent_stats.extend([
                        s for s in operation_stats
                        if s['timestamp'] >= recent_time
                    ])
            
            if not recent_stats:
                return {}
            
            # Calculer les métriques
            durations = [s['duration_ms'] for s in recent_stats]
            successes = [s for s in recent_stats if s['success']]
            
            return {
                'total_requests': len(recent_stats),
                'success_count': len(successes),
                'error_rate': (len(recent_stats) - len(successes)) / len(recent_stats),
                'avg_response_time': sum(durations) / len(durations),
                'p95_response_time': self._percentile(durations, 95),
                'cache_hit_rate': 0.85  # Placeholder - à calculer depuis les vraies métriques
            }
            
        except Exception as e:
            logger.error(f"Current metrics calculation error: {e}")
            return {}
    
    async def _calculate_performance_stats(self) -> Dict[str, Any]:
        """Calcule les statistiques de performance"""
        try:
            with self._lock:
                all_stats = []
                for operation_stats in self._performance_stats.values():
                    all_stats.extend(operation_stats)
                
                if not all_stats:
                    return {}
                
                # Filtrer les statistiques récentes (dernière heure)
                recent_time = datetime.now() - timedelta(hours=1)
                recent_stats = [s for s in all_stats if s['timestamp'] >= recent_time]
                
                if not recent_stats:
                    return {}
                
                durations = [s['duration_ms'] for s in recent_stats]
                success_count = len([s for s in recent_stats if s['success']])
                
                return {
                    'total_requests': len(recent_stats),
                    'success_rate': success_count / len(recent_stats),
                    'avg_duration_ms': sum(durations) / len(durations),
                    'p50_duration_ms': self._percentile(durations, 50),
                    'p95_duration_ms': self._percentile(durations, 95),
                    'p99_duration_ms': self._percentile(durations, 99),
                    'error_rate': (len(recent_stats) - success_count) / len(recent_stats)
                }
                
        except Exception as e:
            logger.error(f"Performance stats calculation error: {e}")
            return {}
    
    async def _summarize_metrics(self, metrics: Dict[str, List[MetricPoint]]) -> Dict[str, Any]:
        """Résume les métriques collectées"""
        try:
            summary = {
                'metrics_count': len(metrics),
                'total_data_points': sum(len(points) for points in metrics.values()),
                'metric_names': list(metrics.keys())
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Metrics summary error: {e}")
            return {}
    
    async def _calculate_health_status(self, perf_stats: Dict[str, Any]) -> str:
        """Calcule le statut de santé global"""
        try:
            if not perf_stats:
                return "unknown"
            
            error_rate = perf_stats.get('error_rate', 0)
            avg_duration = perf_stats.get('avg_duration_ms', 0)
            
            # Déterminer le statut selon les seuils
            if error_rate > 0.1 or avg_duration > 2000:
                return "critical"
            elif error_rate > 0.05 or avg_duration > 1000:
                return "warning"
            else:
                return "healthy"
                
        except Exception as e:
            logger.error(f"Health status calculation error: {e}")
            return "unknown"
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calcule un percentile"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]


class MetricsCollector:
    """Collecteur de métriques simplifié"""
    
    def __init__(self, monitoring: LocaleMonitoring):
        self.monitoring = monitoring
    
    async def increment_counter(
        self,
        name: str,
        labels: Dict[str, str] = None,
        value: float = 1.0
    ):
        """Incrémente un compteur"""
        await self.monitoring._record_metric(
            name,
            value,
            MetricType.COUNTER,
            labels
        )
    
    async def set_gauge(
        self,
        name: str,
        value: float,
        labels: Dict[str, str] = None
    ):
        """Définit une jauge"""
        await self.monitoring._record_metric(
            name,
            value,
            MetricType.GAUGE,
            labels
        )
    
    async def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Dict[str, str] = None
    ):
        """Observe une valeur d'histogramme"""
        await self.monitoring._record_metric(
            name,
            value,
            MetricType.HISTOGRAM,
            labels
        )
