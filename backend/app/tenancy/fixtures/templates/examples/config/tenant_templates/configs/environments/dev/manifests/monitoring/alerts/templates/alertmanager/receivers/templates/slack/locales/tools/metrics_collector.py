#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collecteur de Métriques Avancé pour Outils Slack

Ce module fournit un système de collecte de métriques complet avec:
- Métriques Prometheus détaillées
- Monitoring de performance en temps réel
- Alertes sur les seuils critiques
- Tableaux de bord automatisés
- Export vers différents backends de monitoring
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import structlog
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
import psutil
import threading

logger = structlog.get_logger(__name__)

class MetricType(Enum):
    """Types de métriques supportés."""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    SUMMARY = "summary"
    INFO = "info"

@dataclass
class MetricDefinition:
    """Définition d'une métrique."""
    name: str
    description: str
    metric_type: MetricType
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None
    namespace: str = "slack_tools"

@dataclass
class PerformanceMetrics:
    """Métriques de performance système."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    thread_count: int
    timestamp: datetime = field(default_factory=datetime.utcnow)

class MetricsCollector:
    """
    Collecteur de métriques avancé avec monitoring système.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger.bind(component="metrics_collector")
        
        # Configuration
        self.enabled = config.get("enabled", True)
        self.collection_interval = config.get("collection_interval", 30)
        self.namespace = config.get("namespace", "slack_tools")
        self.export_enabled = config.get("export_enabled", True)
        self.system_metrics_enabled = config.get("system_metrics_enabled", True)
        
        # Registry Prometheus personnalisé
        self.registry = CollectorRegistry()
        
        # Métriques principales
        self._metrics: Dict[str, Any] = {}
        self._custom_metrics: Dict[str, Any] = {}
        
        # Historique des performances
        self._performance_history: List[PerformanceMetrics] = []
        self._max_history_size = config.get("max_history_size", 1000)
        
        # Callbacks d'alerte
        self._alert_callbacks: List[Callable] = []
        
        # Tâches de collecte
        self._collection_task = None
        self._system_metrics_task = None
        
        # Lock pour thread-safety
        self._lock = threading.Lock()
        
        # Initialisation
        if self.enabled:
            asyncio.create_task(self._initialize())
    
    async def _initialize(self):
        """Initialise le collecteur de métriques."""
        try:
            await self._setup_core_metrics()
            await self._start_collection_tasks()
            
            self.logger.info(
                "Collecteur de métriques initialisé",
                namespace=self.namespace,
                collection_interval=self.collection_interval
            )
            
        except Exception as e:
            self.logger.error("Erreur lors de l'initialisation", error=str(e))
            raise
    
    async def _setup_core_metrics(self):
        """Configure les métriques principales."""
        
        # Métriques des notifications Slack
        self._metrics["notifications_total"] = Counter(
            "slack_notifications_total",
            "Total des notifications Slack envoyées",
            ["tenant", "channel", "status", "severity"],
            registry=self.registry
        )
        
        self._metrics["notification_duration"] = Histogram(
            "slack_notification_duration_seconds",
            "Durée de traitement des notifications",
            ["tenant", "type"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry
        )
        
        self._metrics["template_renders"] = Counter(
            "slack_template_renders_total",
            "Total des rendus de templates",
            ["template", "tenant", "status"],
            registry=self.registry
        )
        
        self._metrics["template_render_duration"] = Histogram(
            "slack_template_render_duration_seconds",
            "Durée de rendu des templates",
            ["template", "complexity"],
            registry=self.registry
        )
        
        # Métriques de cache
        self._metrics["cache_operations"] = Counter(
            "slack_cache_operations_total",
            "Opérations de cache",
            ["operation", "cache_type", "result"],
            registry=self.registry
        )
        
        self._metrics["cache_hit_ratio"] = Gauge(
            "slack_cache_hit_ratio",
            "Ratio de hits du cache",
            ["cache_type"],
            registry=self.registry
        )
        
        self._metrics["cache_size"] = Gauge(
            "slack_cache_size_bytes",
            "Taille du cache en bytes",
            ["cache_type"],
            registry=self.registry
        )
        
        # Métriques de localisation
        self._metrics["localization_requests"] = Counter(
            "slack_localization_requests_total",
            "Requêtes de localisation",
            ["locale", "tenant", "status"],
            registry=self.registry
        )
        
        self._metrics["active_locales"] = Gauge(
            "slack_localization_active_locales",
            "Nombre de locales actives",
            registry=self.registry
        )
        
        # Métriques de tenant
        self._metrics["tenant_contexts"] = Gauge(
            "slack_tenant_contexts_active",
            "Contextes tenant actifs",
            registry=self.registry
        )
        
        self._metrics["tenant_operations"] = Counter(
            "slack_tenant_operations_total",
            "Opérations tenant",
            ["operation", "tenant", "result"],
            registry=self.registry
        )
        
        # Métriques d'alertes
        self._metrics["alert_processing"] = Counter(
            "slack_alert_processing_total",
            "Alertes traitées",
            ["tenant", "severity", "status", "result"],
            registry=self.registry
        )
        
        self._metrics["alert_aggregations"] = Counter(
            "slack_alert_aggregations_total",
            "Agrégations d'alertes",
            ["tenant", "type"],
            registry=self.registry
        )
        
        # Métriques système
        if self.system_metrics_enabled:
            self._metrics["system_cpu_usage"] = Gauge(
                "slack_system_cpu_usage_percent",
                "Utilisation CPU système",
                registry=self.registry
            )
            
            self._metrics["system_memory_usage"] = Gauge(
                "slack_system_memory_usage_bytes",
                "Utilisation mémoire système",
                registry=self.registry
            )
            
            self._metrics["system_disk_usage"] = Gauge(
                "slack_system_disk_usage_percent",
                "Utilisation disque système",
                registry=self.registry
            )
            
            self._metrics["system_network_bytes"] = Counter(
                "slack_system_network_bytes_total",
                "Bytes réseau transférés",
                ["direction"],
                registry=self.registry
            )
        
        # Métriques de santé
        self._metrics["health_checks"] = Counter(
            "slack_health_checks_total",
            "Vérifications de santé",
            ["component", "status"],
            registry=self.registry
        )
        
        self._metrics["errors"] = Counter(
            "slack_errors_total",
            "Erreurs système",
            ["component", "error_type"],
            registry=self.registry
        )
        
        # Informations système
        self._metrics["build_info"] = Info(
            "slack_build_info",
            "Informations de build",
            registry=self.registry
        )
        
        # Initialiser les informations de build
        self._metrics["build_info"].info({
            "version": "2.1.0",
            "build_date": datetime.utcnow().isoformat(),
            "python_version": "3.9+",
            "component": "slack_localization_tools"
        })
    
    async def _start_collection_tasks(self):
        """Démarre les tâches de collecte périodique."""
        if self.system_metrics_enabled:
            self._system_metrics_task = asyncio.create_task(self._collect_system_metrics())
        
        self._collection_task = asyncio.create_task(self._performance_monitoring())
    
    async def _collect_system_metrics(self):
        """Collecte périodique des métriques système."""
        last_network_stats = psutil.net_io_counters()
        
        while True:
            try:
                await asyncio.sleep(self.collection_interval)
                
                # CPU
                cpu_percent = psutil.cpu_percent(interval=1)
                self._metrics["system_cpu_usage"].set(cpu_percent)
                
                # Mémoire
                memory = psutil.virtual_memory()
                self._metrics["system_memory_usage"].set(memory.used)
                
                # Disque
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                self._metrics["system_disk_usage"].set(disk_percent)
                
                # Réseau
                current_network_stats = psutil.net_io_counters()
                bytes_sent_delta = current_network_stats.bytes_sent - last_network_stats.bytes_sent
                bytes_recv_delta = current_network_stats.bytes_recv - last_network_stats.bytes_recv
                
                self._metrics["system_network_bytes"].labels(direction="sent").inc(bytes_sent_delta)
                self._metrics["system_network_bytes"].labels(direction="received").inc(bytes_recv_delta)
                
                last_network_stats = current_network_stats
                
                # Stocker l'historique
                perf_metrics = PerformanceMetrics(
                    cpu_usage=cpu_percent,
                    memory_usage=memory.percent,
                    disk_usage=disk_percent,
                    network_bytes_sent=current_network_stats.bytes_sent,
                    network_bytes_recv=current_network_stats.bytes_recv,
                    process_count=len(psutil.pids()),
                    thread_count=threading.active_count()
                )
                
                with self._lock:
                    self._performance_history.append(perf_metrics)
                    if len(self._performance_history) > self._max_history_size:
                        self._performance_history.pop(0)
                
                # Vérifier les seuils d'alerte
                await self._check_alert_thresholds(perf_metrics)
                
            except Exception as e:
                self.logger.error("Erreur collecte métriques système", error=str(e))
                self.inc_error("system_metrics", "collection_error")
    
    async def _performance_monitoring(self):
        """Monitoring de performance continu."""
        while True:
            try:
                await asyncio.sleep(60)  # Toutes les minutes
                
                # Analyser les tendances de performance
                await self._analyze_performance_trends()
                
                # Nettoyer l'historique ancien
                await self._cleanup_old_metrics()
                
            except Exception as e:
                self.logger.error("Erreur monitoring performance", error=str(e))
    
    async def _analyze_performance_trends(self):
        """Analyse les tendances de performance."""
        if len(self._performance_history) < 10:
            return
        
        with self._lock:
            recent_metrics = self._performance_history[-10:]
        
        # Calculer les moyennes
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        
        # Détecter les anomalies
        if avg_cpu > 80:
            self.logger.warning("Utilisation CPU élevée détectée", avg_cpu=avg_cpu)
            await self._trigger_alert("high_cpu", {"cpu_usage": avg_cpu})
        
        if avg_memory > 85:
            self.logger.warning("Utilisation mémoire élevée détectée", avg_memory=avg_memory)
            await self._trigger_alert("high_memory", {"memory_usage": avg_memory})
    
    async def _check_alert_thresholds(self, metrics: PerformanceMetrics):
        """Vérifie les seuils d'alerte."""
        thresholds = self.config.get("alert_thresholds", {})
        
        # CPU
        cpu_threshold = thresholds.get("cpu_critical", 90)
        if metrics.cpu_usage > cpu_threshold:
            await self._trigger_alert("cpu_critical", {"value": metrics.cpu_usage})
        
        # Mémoire
        memory_threshold = thresholds.get("memory_critical", 90)
        if metrics.memory_usage > memory_threshold:
            await self._trigger_alert("memory_critical", {"value": metrics.memory_usage})
        
        # Disque
        disk_threshold = thresholds.get("disk_critical", 95)
        if metrics.disk_usage > disk_threshold:
            await self._trigger_alert("disk_critical", {"value": metrics.disk_usage})
    
    async def _trigger_alert(self, alert_type: str, data: Dict[str, Any]):
        """Déclenche une alerte."""
        alert_data = {
            "type": alert_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
            "component": "slack_tools_metrics"
        }
        
        # Appeler les callbacks d'alerte
        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert_data)
                else:
                    callback(alert_data)
            except Exception as e:
                self.logger.error("Erreur callback alerte", error=str(e))
    
    async def _cleanup_old_metrics(self):
        """Nettoie les métriques anciennes."""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        with self._lock:
            self._performance_history = [
                m for m in self._performance_history
                if m.timestamp > cutoff_time
            ]
    
    def register_custom_metric(self, definition: MetricDefinition) -> Any:
        """Enregistre une métrique personnalisée."""
        metric_name = f"{self.namespace}_{definition.name}"
        
        if definition.metric_type == MetricType.COUNTER:
            metric = Counter(
                metric_name,
                definition.description,
                definition.labels,
                registry=self.registry
            )
        elif definition.metric_type == MetricType.HISTOGRAM:
            buckets = definition.buckets or [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
            metric = Histogram(
                metric_name,
                definition.description,
                definition.labels,
                buckets=buckets,
                registry=self.registry
            )
        elif definition.metric_type == MetricType.GAUGE:
            metric = Gauge(
                metric_name,
                definition.description,
                definition.labels,
                registry=self.registry
            )
        elif definition.metric_type == MetricType.SUMMARY:
            metric = Summary(
                metric_name,
                definition.description,
                definition.labels,
                registry=self.registry
            )
        elif definition.metric_type == MetricType.INFO:
            metric = Info(
                metric_name,
                definition.description,
                registry=self.registry
            )
        else:
            raise ValueError(f"Type de métrique non supporté: {definition.metric_type}")
        
        self._custom_metrics[definition.name] = metric
        
        self.logger.info(
            "Métrique personnalisée enregistrée",
            name=definition.name,
            type=definition.metric_type.value
        )
        
        return metric
    
    def get_metric(self, name: str) -> Optional[Any]:
        """Récupère une métrique par son nom."""
        return self._metrics.get(name) or self._custom_metrics.get(name)
    
    def inc_counter(self, metric_name: str, labels: Optional[Dict[str, str]] = None, value: float = 1):
        """Incrémente un compteur."""
        metric = self.get_metric(metric_name)
        if metric and hasattr(metric, 'inc'):
            if labels:
                metric.labels(**labels).inc(value)
            else:
                metric.inc(value)
    
    def set_gauge(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Définit la valeur d'une gauge."""
        metric = self.get_metric(metric_name)
        if metric and hasattr(metric, 'set'):
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)
    
    def observe_histogram(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe une valeur dans un histogramme."""
        metric = self.get_metric(metric_name)
        if metric and hasattr(metric, 'observe'):
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)
    
    def inc_notification(self, tenant: str, channel: str, status: str, severity: str = "info"):
        """Incrémente le compteur de notifications."""
        self.inc_counter("notifications_total", {
            "tenant": tenant,
            "channel": channel,
            "status": status,
            "severity": severity
        })
    
    def observe_notification_duration(self, tenant: str, notification_type: str, duration: float):
        """Observe la durée d'une notification."""
        self.observe_histogram("notification_duration", duration, {
            "tenant": tenant,
            "type": notification_type
        })
    
    def inc_template_render(self, template: str, tenant: str, status: str):
        """Incrémente le compteur de rendu de templates."""
        self.inc_counter("template_renders", {
            "template": template,
            "tenant": tenant,
            "status": status
        })
    
    def inc_cache_operation(self, operation: str, cache_type: str, result: str):
        """Incrémente le compteur d'opérations de cache."""
        self.inc_counter("cache_operations", {
            "operation": operation,
            "cache_type": cache_type,
            "result": result
        })
    
    def set_cache_hit_ratio(self, cache_type: str, ratio: float):
        """Définit le ratio de hits du cache."""
        self.set_gauge("cache_hit_ratio", ratio, {"cache_type": cache_type})
    
    def inc_localization_request(self, locale: str, tenant: str, status: str):
        """Incrémente le compteur de requêtes de localisation."""
        self.inc_counter("localization_requests", {
            "locale": locale,
            "tenant": tenant,
            "status": status
        })
    
    def inc_tenant_operation(self, operation: str, tenant: str, result: str):
        """Incrémente le compteur d'opérations tenant."""
        self.inc_counter("tenant_operations", {
            "operation": operation,
            "tenant": tenant,
            "result": result
        })
    
    def inc_alert_processing(self, tenant: str, severity: str, status: str, result: str):
        """Incrémente le compteur de traitement d'alertes."""
        self.inc_counter("alert_processing", {
            "tenant": tenant,
            "severity": severity,
            "status": status,
            "result": result
        })
    
    def inc_health_check(self, component: str, status: str):
        """Incrémente le compteur de vérifications de santé."""
        self.inc_counter("health_checks", {
            "component": component,
            "status": status
        })
    
    def inc_error(self, component: str, error_type: str):
        """Incrémente le compteur d'erreurs."""
        self.inc_counter("errors", {
            "component": component,
            "error_type": error_type
        })
    
    def add_alert_callback(self, callback: Callable):
        """Ajoute un callback d'alerte."""
        self._alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable):
        """Supprime un callback d'alerte."""
        if callback in self._alert_callbacks:
            self._alert_callbacks.remove(callback)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des performances."""
        if not self._performance_history:
            return {"status": "no_data"}
        
        with self._lock:
            recent_metrics = self._performance_history[-10:] if len(self._performance_history) >= 10 else self._performance_history
        
        if not recent_metrics:
            return {"status": "no_data"}
        
        return {
            "status": "ok",
            "timespan_minutes": (recent_metrics[-1].timestamp - recent_metrics[0].timestamp).total_seconds() / 60,
            "cpu": {
                "current": recent_metrics[-1].cpu_usage,
                "average": sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
                "max": max(m.cpu_usage for m in recent_metrics)
            },
            "memory": {
                "current": recent_metrics[-1].memory_usage,
                "average": sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
                "max": max(m.memory_usage for m in recent_metrics)
            },
            "disk": {
                "current": recent_metrics[-1].disk_usage,
                "average": sum(m.disk_usage for m in recent_metrics) / len(recent_metrics),
                "max": max(m.disk_usage for m in recent_metrics)
            },
            "processes": recent_metrics[-1].process_count,
            "threads": recent_metrics[-1].thread_count
        }
    
    def export_metrics(self) -> str:
        """Exporte les métriques au format Prometheus."""
        if not self.export_enabled:
            return ""
        
        return generate_latest(self.registry).decode('utf-8')
    
    def get_metrics_content_type(self) -> str:
        """Retourne le content-type pour les métriques."""
        return CONTENT_TYPE_LATEST
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérifie la santé du collecteur de métriques."""
        try:
            return {
                "status": "healthy",
                "enabled": self.enabled,
                "metrics_count": len(self._metrics) + len(self._custom_metrics),
                "collection_interval": self.collection_interval,
                "system_metrics_enabled": self.system_metrics_enabled,
                "performance_history_size": len(self._performance_history),
                "alert_callbacks_count": len(self._alert_callbacks),
                "last_check": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.utcnow().isoformat()
            }
    
    async def close(self):
        """Ferme le collecteur et arrête les tâches."""
        try:
            if self._collection_task:
                self._collection_task.cancel()
            if self._system_metrics_task:
                self._system_metrics_task.cancel()
            
            self.logger.info("Collecteur de métriques fermé")
            
        except Exception as e:
            self.logger.error("Erreur fermeture collecteur", error=str(e))

# Factory function
def create_metrics_collector(config: Dict[str, Any]) -> MetricsCollector:
    """Crée une instance du collecteur de métriques."""
    return MetricsCollector(config)
