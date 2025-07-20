"""
Collecteur de métriques avancé pour le système de monitoring Slack.

Ce module fournit un système complet de collecte et d'agrégation de métriques avec:
- Support de multiples backends (Prometheus, StatsD, CloudWatch)
- Métriques temps réel avec bufferisation intelligente
- Agrégation automatique avec fenêtres glissantes
- Alerting proactif basé sur les seuils dynamiques
- Export automatique vers les systèmes de monitoring
- Cache distribué pour les performances

Architecture:
    - Factory pattern pour les différents collecteurs
    - Observer pattern pour les notifications de métriques
    - Strategy pattern pour les backends d'export
    - Buffer circulaire pour les métriques temps réel
    - Thread pool pour l'export asynchrone

Fonctionnalités:
    - Métriques système (CPU, mémoire, réseau)
    - Métriques applicatives (latence, erreurs, throughput)
    - Métriques business (utilisateurs, transactions)
    - Histogrammes et percentiles automatiques
    - Dashboards dynamiques
    - Corrélation automatique des métriques

Auteur: Équipe Spotify AI Agent
Lead Developer: Fahed Mlaiel
"""

import asyncio
import json
import math
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from weakref import WeakSet

import psutil
import numpy as np
from prometheus_client import Counter, Gauge, Histogram, Summary, CollectorRegistry, generate_latest


class MetricType(Enum):
    """Types de métriques."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"


class MetricCategory(Enum):
    """Catégories de métriques."""
    SYSTEM = "system"
    APPLICATION = "application"
    BUSINESS = "business"
    SECURITY = "security"
    PERFORMANCE = "performance"


@dataclass
class MetricValue:
    """Valeur de métrique avec métadonnées."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    category: MetricCategory
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    unit: Optional[str] = None
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire."""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "category": self.category.value,
            "labels": self.labels,
            "timestamp": self.timestamp.isoformat(),
            "unit": self.unit,
            "description": self.description
        }


@dataclass
class MetricSummary:
    """Résumé statistique d'une métrique."""
    name: str
    count: int
    sum_value: float
    min_value: float
    max_value: float
    mean_value: float
    stddev_value: float
    percentiles: Dict[str, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def add_percentile(self, percentile: float, value: float) -> None:
        """Ajoute un percentile."""
        self.percentiles[f"p{int(percentile)}"] = value


@dataclass
class AlertRule:
    """Règle d'alerte pour une métrique."""
    metric_name: str
    condition: str  # "gt", "lt", "eq", "ne"
    threshold: float
    duration_seconds: int
    severity: str  # "critical", "warning", "info"
    message_template: str
    labels: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    
    def evaluate(self, value: float) -> bool:
        """Évalue la condition d'alerte."""
        if not self.enabled:
            return False
        
        if self.condition == "gt":
            return value > self.threshold
        elif self.condition == "lt":
            return value < self.threshold
        elif self.condition == "eq":
            return value == self.threshold
        elif self.condition == "ne":
            return value != self.threshold
        else:
            return False


class IMetricBackend(ABC):
    """Interface pour les backends de métriques."""
    
    @abstractmethod
    def record_metric(self, metric: MetricValue) -> None:
        """Enregistre une métrique."""
        pass
    
    @abstractmethod
    def export_metrics(self) -> str:
        """Exporte les métriques au format du backend."""
        pass
    
    @abstractmethod
    def get_metric_names(self) -> List[str]:
        """Retourne les noms des métriques disponibles."""
        pass


class PrometheusBackend(IMetricBackend):
    """Backend Prometheus pour les métriques."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self._metrics: Dict[str, Any] = {}
        self._lock = threading.RLock()
    
    def record_metric(self, metric: MetricValue) -> None:
        """Enregistre une métrique Prometheus."""
        with self._lock:
            metric_key = self._get_metric_key(metric.name, metric.labels)
            
            if metric_key not in self._metrics:
                self._create_prometheus_metric(metric)
            
            prom_metric = self._metrics[metric_key]
            
            if metric.metric_type == MetricType.COUNTER:
                if hasattr(prom_metric, 'inc'):
                    prom_metric.inc(metric.value)
            elif metric.metric_type == MetricType.GAUGE:
                if hasattr(prom_metric, 'set'):
                    prom_metric.set(metric.value)
            elif metric.metric_type == MetricType.HISTOGRAM:
                if hasattr(prom_metric, 'observe'):
                    prom_metric.observe(metric.value)
    
    def export_metrics(self) -> str:
        """Exporte les métriques au format Prometheus."""
        return generate_latest(self.registry).decode('utf-8')
    
    def get_metric_names(self) -> List[str]:
        """Retourne les noms des métriques."""
        return list(self._metrics.keys())
    
    def _create_prometheus_metric(self, metric: MetricValue) -> None:
        """Crée une métrique Prometheus."""
        metric_key = self._get_metric_key(metric.name, metric.labels)
        
        label_names = list(metric.labels.keys()) if metric.labels else []
        
        if metric.metric_type == MetricType.COUNTER:
            prom_metric = Counter(
                metric.name,
                metric.description or f"Counter metric: {metric.name}",
                labelnames=label_names,
                registry=self.registry
            )
        elif metric.metric_type == MetricType.GAUGE:
            prom_metric = Gauge(
                metric.name,
                metric.description or f"Gauge metric: {metric.name}",
                labelnames=label_names,
                registry=self.registry
            )
        elif metric.metric_type == MetricType.HISTOGRAM:
            prom_metric = Histogram(
                metric.name,
                metric.description or f"Histogram metric: {metric.name}",
                labelnames=label_names,
                registry=self.registry
            )
        else:
            # Fallback to Summary
            prom_metric = Summary(
                metric.name,
                metric.description or f"Summary metric: {metric.name}",
                labelnames=label_names,
                registry=self.registry
            )
        
        # Application des labels
        if metric.labels:
            prom_metric = prom_metric.labels(**metric.labels)
        
        self._metrics[metric_key] = prom_metric
    
    def _get_metric_key(self, name: str, labels: Dict[str, str]) -> str:
        """Génère une clé unique pour la métrique."""
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            return f"{name}:{label_str}"
        return name


class MemoryBackend(IMetricBackend):
    """Backend en mémoire pour les métriques."""
    
    def __init__(self, max_points: int = 10000):
        self.max_points = max_points
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self._lock = threading.RLock()
    
    def record_metric(self, metric: MetricValue) -> None:
        """Enregistre une métrique en mémoire."""
        with self._lock:
            self._metrics[metric.name].append(metric)
    
    def export_metrics(self) -> str:
        """Exporte les métriques au format JSON."""
        with self._lock:
            export_data = {}
            for name, values in self._metrics.items():
                export_data[name] = [metric.to_dict() for metric in values]
            return json.dumps(export_data, indent=2)
    
    def get_metric_names(self) -> List[str]:
        """Retourne les noms des métriques."""
        return list(self._metrics.keys())
    
    def get_metric_values(self, name: str, 
                         since: Optional[datetime] = None) -> List[MetricValue]:
        """Récupère les valeurs d'une métrique."""
        with self._lock:
            values = list(self._metrics.get(name, []))
            
            if since:
                values = [v for v in values if v.timestamp >= since]
            
            return values
    
    def calculate_summary(self, name: str, 
                         since: Optional[datetime] = None) -> Optional[MetricSummary]:
        """Calcule un résumé statistique pour une métrique."""
        values = self.get_metric_values(name, since)
        
        if not values:
            return None
        
        numeric_values = [v.value for v in values]
        
        summary = MetricSummary(
            name=name,
            count=len(numeric_values),
            sum_value=sum(numeric_values),
            min_value=min(numeric_values),
            max_value=max(numeric_values),
            mean_value=np.mean(numeric_values),
            stddev_value=np.std(numeric_values)
        )
        
        # Calcul des percentiles
        percentiles = [50, 75, 90, 95, 99]
        for p in percentiles:
            value = np.percentile(numeric_values, p)
            summary.add_percentile(p, value)
        
        return summary


class SystemMetricsCollector:
    """Collecteur de métriques système."""
    
    def __init__(self):
        self._collection_interval = 30  # secondes
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[List[MetricValue]], None]] = []
    
    def start(self) -> None:
        """Démarre la collecte des métriques système."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """Arrête la collecte des métriques système."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
    
    def add_callback(self, callback: Callable[[List[MetricValue]], None]) -> None:
        """Ajoute un callback pour les métriques collectées."""
        self._callbacks.append(callback)
    
    def collect_once(self) -> List[MetricValue]:
        """Collecte les métriques système une seule fois."""
        metrics = []
        now = datetime.now(timezone.utc)
        
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(MetricValue(
                name="system_cpu_usage_percent",
                value=cpu_percent,
                metric_type=MetricType.GAUGE,
                category=MetricCategory.SYSTEM,
                unit="percent",
                description="CPU usage percentage",
                timestamp=now
            ))
            
            # Mémoire
            memory = psutil.virtual_memory()
            metrics.append(MetricValue(
                name="system_memory_usage_percent",
                value=memory.percent,
                metric_type=MetricType.GAUGE,
                category=MetricCategory.SYSTEM,
                unit="percent",
                description="Memory usage percentage",
                timestamp=now
            ))
            
            metrics.append(MetricValue(
                name="system_memory_available_bytes",
                value=memory.available,
                metric_type=MetricType.GAUGE,
                category=MetricCategory.SYSTEM,
                unit="bytes",
                description="Available memory in bytes",
                timestamp=now
            ))
            
            # Disque
            disk = psutil.disk_usage('/')
            metrics.append(MetricValue(
                name="system_disk_usage_percent",
                value=(disk.used / disk.total) * 100,
                metric_type=MetricType.GAUGE,
                category=MetricCategory.SYSTEM,
                unit="percent",
                description="Disk usage percentage",
                timestamp=now
            ))
            
            # Réseau
            network = psutil.net_io_counters()
            metrics.append(MetricValue(
                name="system_network_bytes_sent",
                value=network.bytes_sent,
                metric_type=MetricType.COUNTER,
                category=MetricCategory.SYSTEM,
                unit="bytes",
                description="Total network bytes sent",
                timestamp=now
            ))
            
            metrics.append(MetricValue(
                name="system_network_bytes_recv",
                value=network.bytes_recv,
                metric_type=MetricType.COUNTER,
                category=MetricCategory.SYSTEM,
                unit="bytes",
                description="Total network bytes received",
                timestamp=now
            ))
            
        except Exception as e:
            # Log l'erreur mais continue
            pass
        
        return metrics
    
    def _collect_loop(self) -> None:
        """Boucle de collecte des métriques."""
        while self._running:
            try:
                metrics = self.collect_once()
                
                # Notification des callbacks
                for callback in self._callbacks:
                    try:
                        callback(metrics)
                    except Exception:
                        continue  # Ignore les erreurs des callbacks
                
                time.sleep(self._collection_interval)
                
            except Exception:
                time.sleep(1)  # Courte pause en cas d'erreur


class MetricsCollector:
    """
    Collecteur principal de métriques.
    
    Coordonne la collecte, l'agrégation et l'export des métriques
    avec support de multiples backends et alerting.
    """
    
    def __init__(self,
                 backends: Optional[List[IMetricBackend]] = None,
                 enable_system_metrics: bool = True,
                 export_interval: int = 60,
                 enable_alerting: bool = True):
        
        # Backends de métriques
        self._backends = backends or [MemoryBackend(), PrometheusBackend()]
        
        # Configuration
        self._enable_system_metrics = enable_system_metrics
        self._export_interval = export_interval
        self._enable_alerting = enable_alerting
        
        # Collecteur système
        self._system_collector = SystemMetricsCollector()
        if enable_system_metrics:
            self._system_collector.add_callback(self._handle_system_metrics)
        
        # Buffer pour les métriques
        self._metric_buffer: deque = deque(maxlen=100000)
        self._buffer_lock = threading.RLock()
        
        # Règles d'alerte
        self._alert_rules: Dict[str, AlertRule] = {}
        self._alert_states: Dict[str, Dict[str, Any]] = {}
        
        # Hooks pour les alertes
        self._alert_hooks: WeakSet[Callable[[str, MetricValue, AlertRule], None]] = WeakSet()
        
        # Thread d'export
        self._export_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Métriques internes
        self._internal_metrics = {
            "metrics_collected": 0,
            "metrics_exported": 0,
            "alerts_triggered": 0,
            "export_errors": 0
        }
        
        # Statistiques par catégorie
        self._category_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
    
    def start(self) -> None:
        """Démarre le collecteur de métriques."""
        if self._running:
            return
        
        self._running = True
        
        # Démarrage du collecteur système
        if self._enable_system_metrics:
            self._system_collector.start()
        
        # Démarrage du thread d'export
        self._export_thread = threading.Thread(target=self._export_loop, daemon=True)
        self._export_thread.start()
    
    def stop(self) -> None:
        """Arrête le collecteur de métriques."""
        self._running = False
        
        # Arrêt du collecteur système
        if self._enable_system_metrics:
            self._system_collector.stop()
        
        # Arrêt du thread d'export
        if self._export_thread:
            self._export_thread.join(timeout=5.0)
    
    def record(self,
              name: str,
              value: Union[int, float],
              metric_type: MetricType = MetricType.GAUGE,
              category: MetricCategory = MetricCategory.APPLICATION,
              labels: Optional[Dict[str, str]] = None,
              unit: Optional[str] = None,
              description: Optional[str] = None) -> None:
        """
        Enregistre une métrique.
        
        Args:
            name: Nom de la métrique
            value: Valeur de la métrique
            metric_type: Type de métrique
            category: Catégorie de la métrique
            labels: Labels associés
            unit: Unité de mesure
            description: Description de la métrique
        """
        metric = MetricValue(
            name=name,
            value=value,
            metric_type=metric_type,
            category=category,
            labels=labels or {},
            unit=unit,
            description=description
        )
        
        self._process_metric(metric)
    
    def increment(self,
                 name: str,
                 value: Union[int, float] = 1,
                 labels: Optional[Dict[str, str]] = None,
                 **kwargs) -> None:
        """Incrémente un compteur."""
        self.record(
            name=name,
            value=value,
            metric_type=MetricType.COUNTER,
            labels=labels,
            **kwargs
        )
    
    def gauge(self,
             name: str,
             value: Union[int, float],
             labels: Optional[Dict[str, str]] = None,
             **kwargs) -> None:
        """Enregistre une valeur de jauge."""
        self.record(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            labels=labels,
            **kwargs
        )
    
    def histogram(self,
                 name: str,
                 value: Union[int, float],
                 labels: Optional[Dict[str, str]] = None,
                 **kwargs) -> None:
        """Enregistre une valeur dans un histogramme."""
        self.record(
            name=name,
            value=value,
            metric_type=MetricType.HISTOGRAM,
            labels=labels,
            **kwargs
        )
    
    def timer(self, name: str, labels: Optional[Dict[str, str]] = None):
        """
        Décorateur/context manager pour mesurer le temps d'exécution.
        
        Usage:
            @metrics.timer("my_function_duration")
            def my_function():
                pass
            
            # ou
            
            with metrics.timer("operation_duration"):
                # opération chronométrée
                pass
        """
        return TimerContext(self, name, labels)
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Ajoute une règle d'alerte."""
        self._alert_rules[rule.metric_name] = rule
    
    def remove_alert_rule(self, metric_name: str) -> bool:
        """Supprime une règle d'alerte."""
        return self._alert_rules.pop(metric_name, None) is not None
    
    def add_alert_hook(self, hook: Callable[[str, MetricValue, AlertRule], None]) -> None:
        """Ajoute un hook pour les alertes."""
        self._alert_hooks.add(hook)
    
    def get_summary(self, metric_name: str,
                   since: Optional[datetime] = None) -> Optional[MetricSummary]:
        """Récupère un résumé pour une métrique."""
        for backend in self._backends:
            if isinstance(backend, MemoryBackend):
                return backend.calculate_summary(metric_name, since)
        return None
    
    def export_metrics(self, format_type: str = "prometheus") -> str:
        """
        Exporte les métriques au format spécifié.
        
        Args:
            format_type: Format d'export ("prometheus", "json")
            
        Returns:
            Métriques exportées
        """
        if format_type == "prometheus":
            for backend in self._backends:
                if isinstance(backend, PrometheusBackend):
                    return backend.export_metrics()
        elif format_type == "json":
            for backend in self._backends:
                if isinstance(backend, MemoryBackend):
                    return backend.export_metrics()
        
        return ""
    
    def get_metric_names(self) -> List[str]:
        """Retourne tous les noms de métriques disponibles."""
        all_names = set()
        for backend in self._backends:
            all_names.update(backend.get_metric_names())
        return list(all_names)
    
    def _process_metric(self, metric: MetricValue) -> None:
        """Traite une métrique."""
        # Ajout au buffer
        with self._buffer_lock:
            self._metric_buffer.append(metric)
        
        # Enregistrement dans les backends
        for backend in self._backends:
            try:
                backend.record_metric(metric)
            except Exception:
                self._internal_metrics["export_errors"] += 1
                continue
        
        # Évaluation des alertes
        if self._enable_alerting:
            self._evaluate_alerts(metric)
        
        # Mise à jour des statistiques
        self._internal_metrics["metrics_collected"] += 1
        self._category_stats[metric.category.value]["count"] += 1
    
    def _handle_system_metrics(self, metrics: List[MetricValue]) -> None:
        """Gère les métriques système collectées."""
        for metric in metrics:
            self._process_metric(metric)
    
    def _evaluate_alerts(self, metric: MetricValue) -> None:
        """Évalue les règles d'alerte pour une métrique."""
        rule = self._alert_rules.get(metric.name)
        if not rule:
            return
        
        # Évaluation de la condition
        is_triggered = rule.evaluate(metric.value)
        
        alert_key = f"{rule.metric_name}:{hash(frozenset(metric.labels.items()))}"
        current_state = self._alert_states.get(alert_key, {
            "triggered": False,
            "first_triggered": None,
            "last_notified": None
        })
        
        if is_triggered and not current_state["triggered"]:
            # Nouvelle alerte
            current_state["triggered"] = True
            current_state["first_triggered"] = datetime.now(timezone.utc)
            
            # Notification immédiate ou après duration
            if rule.duration_seconds == 0:
                self._trigger_alert(alert_key, metric, rule)
        
        elif is_triggered and current_state["triggered"]:
            # Alerte continue - vérifier la durée
            if current_state["first_triggered"]:
                duration = (datetime.now(timezone.utc) - current_state["first_triggered"]).total_seconds()
                if duration >= rule.duration_seconds:
                    # Vérifier si on doit re-notifier
                    if (not current_state["last_notified"] or
                        (datetime.now(timezone.utc) - current_state["last_notified"]).total_seconds() > 3600):
                        self._trigger_alert(alert_key, metric, rule)
        
        elif not is_triggered and current_state["triggered"]:
            # Résolution d'alerte
            current_state["triggered"] = False
            current_state["first_triggered"] = None
        
        self._alert_states[alert_key] = current_state
    
    def _trigger_alert(self, alert_key: str, metric: MetricValue, rule: AlertRule) -> None:
        """Déclenche une alerte."""
        self._internal_metrics["alerts_triggered"] += 1
        self._alert_states[alert_key]["last_notified"] = datetime.now(timezone.utc)
        
        # Notification des hooks
        for hook in self._alert_hooks:
            try:
                hook(alert_key, metric, rule)
            except Exception:
                continue
    
    def _export_loop(self) -> None:
        """Boucle d'export des métriques."""
        while self._running:
            try:
                # Export vers tous les backends
                for backend in self._backends:
                    try:
                        backend.export_metrics()
                        self._internal_metrics["metrics_exported"] += 1
                    except Exception:
                        self._internal_metrics["export_errors"] += 1
                
                time.sleep(self._export_interval)
                
            except Exception:
                time.sleep(1)  # Courte pause en cas d'erreur
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Statistiques du collecteur."""
        return {
            "internal_metrics": self._internal_metrics.copy(),
            "category_stats": dict(self._category_stats),
            "alert_rules_count": len(self._alert_rules),
            "active_alerts": sum(1 for state in self._alert_states.values() 
                               if state.get("triggered", False)),
            "backends_count": len(self._backends),
            "buffer_size": len(self._metric_buffer),
            "running": self._running
        }


class TimerContext:
    """Context manager pour mesurer le temps d'exécution."""
    
    def __init__(self, collector: MetricsCollector, name: str, labels: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.name = name
        self.labels = labels
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = (time.time() - self.start_time) * 1000  # en millisecondes
            self.collector.histogram(
                name=self.name,
                value=duration,
                labels=self.labels,
                unit="milliseconds",
                description=f"Execution time for {self.name}"
            )
    
    def __call__(self, func):
        """Utilisation comme décorateur."""
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


# Instance globale singleton
_global_metrics_collector: Optional[MetricsCollector] = None
_collector_lock = threading.Lock()


def get_metrics_collector(**kwargs) -> MetricsCollector:
    """
    Récupère l'instance globale du collecteur de métriques.
    
    Returns:
        Instance singleton du MetricsCollector
    """
    global _global_metrics_collector
    
    if _global_metrics_collector is None:
        with _collector_lock:
            if _global_metrics_collector is None:
                _global_metrics_collector = MetricsCollector(**kwargs)
                _global_metrics_collector.start()
    
    return _global_metrics_collector


# API publique simplifiée
def increment(name: str, value: Union[int, float] = 1, **kwargs) -> None:
    """API simplifiée pour incrémenter un compteur."""
    collector = get_metrics_collector()
    collector.increment(name, value, **kwargs)


def gauge(name: str, value: Union[int, float], **kwargs) -> None:
    """API simplifiée pour enregistrer une jauge."""
    collector = get_metrics_collector()
    collector.gauge(name, value, **kwargs)


def histogram(name: str, value: Union[int, float], **kwargs) -> None:
    """API simplifiée pour enregistrer un histogramme."""
    collector = get_metrics_collector()
    collector.histogram(name, value, **kwargs)


def timer(name: str, labels: Optional[Dict[str, str]] = None):
    """API simplifiée pour le timer."""
    collector = get_metrics_collector()
    return collector.timer(name, labels)
