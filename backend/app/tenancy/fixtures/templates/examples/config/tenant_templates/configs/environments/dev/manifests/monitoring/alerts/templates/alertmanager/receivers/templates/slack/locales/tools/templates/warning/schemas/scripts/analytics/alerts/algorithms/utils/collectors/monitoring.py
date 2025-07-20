"""
Spotify AI Agent - Advanced Monitoring Module
============================================

Système de monitoring ultra-avancé pour les collecteurs de données
avec observabilité complète, alerting intelligent, et analytics en temps réel.

Fonctionnalités de monitoring:
- Métriques Prometheus avec labels dynamiques
- Alerting basé sur des seuils adaptatifs
- Dashboards Grafana automatisés
- Tracing distribué avec OpenTelemetry
- Health checks intelligents
- Analytics de performance en temps réel
- Détection d'anomalies avec ML
- Corrélation d'événements
- Monitoring business et technique

Développé par l'équipe Spotify AI Agent
Lead Developer: Fahed Mlaiel
Architecture: Enterprise-grade multi-tenant monitoring system
"""

import asyncio
import time
import threading
import weakref
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Union, Callable, Tuple, 
    Set, Awaitable, Type, Protocol, NamedTuple
)
import json
import logging
import structlog
import psutil
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import aiofiles
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import prometheus_client
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info, Enum as PrometheusEnum,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
from opentelemetry import trace, metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
import redis.asyncio as aioredis
from pydantic import BaseModel, validator
import asyncpg


logger = structlog.get_logger(__name__)


class MetricType(Enum):
    """Types de métriques supportées."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"
    ENUM = "enum"


class AlertSeverity(Enum):
    """Niveaux de sévérité des alertes."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


class HealthStatus(Enum):
    """États de santé des composants."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class MetricDefinition:
    """Définition d'une métrique."""
    
    name: str
    metric_type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # Pour histogrammes
    quantiles: Optional[List[float]] = None  # Pour summaries
    unit: Optional[str] = None
    namespace: str = "spotify_ai_agent"
    subsystem: str = "collectors"


@dataclass
class AlertRule:
    """Règle d'alerte."""
    
    name: str
    metric_name: str
    condition: str  # ex: ">" , "<", "==", "!=", ">=", "<="
    threshold: float
    duration: int  # Durée en secondes
    severity: AlertSeverity
    description: str
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    cooldown: int = 300  # Période de cooldown en secondes


@dataclass
class HealthCheck:
    """Vérification de santé d'un composant."""
    
    name: str
    check_function: Callable[[], Awaitable[Tuple[HealthStatus, str]]]
    interval: int = 30  # Intervalle en secondes
    timeout: int = 10  # Timeout en secondes
    retries: int = 3
    enabled: bool = True
    critical: bool = False  # Si True, l'échec affecte la santé globale


class MetricsCollector:
    """
    Collecteur de métriques avancé avec support Prometheus et OpenTelemetry.
    
    Fonctionnalités:
    - Métriques personnalisées avec labels dynamiques
    - Export Prometheus automatique
    - Intégration OpenTelemetry
    - Agrégation en temps réel
    - Persistence des métriques
    """
    
    def __init__(
        self,
        registry: Optional[CollectorRegistry] = None,
        enable_opentelemetry: bool = True,
        export_port: int = 9090,
        namespace: str = "spotify_ai_agent"
    ):
        self.registry = registry or CollectorRegistry()
        self.enable_opentelemetry = enable_opentelemetry
        self.export_port = export_port
        self.namespace = namespace
        
        # Stockage des métriques
        self._metrics: Dict[str, Any] = {}
        self._metric_definitions: Dict[str, MetricDefinition] = {}
        
        # OpenTelemetry setup
        if enable_opentelemetry:
            self._setup_opentelemetry()
        
        # Cache pour l'agrégation
        self._aggregation_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._last_aggregation = time.time()
        
        # Métriques système automatiques
        self._system_metrics_enabled = True
        self._system_collection_task = None
        
        # Démarrage des tâches de collecte
        asyncio.create_task(self._start_background_tasks())
    
    def _setup_opentelemetry(self) -> None:
        """Configure OpenTelemetry pour les métriques et traces."""
        
        # Configuration du provider de métriques
        metric_reader = PrometheusMetricReader()
        meter_provider = MeterProvider(metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)
        self.meter = metrics.get_meter(__name__)
        
        # Configuration du tracing
        trace.set_tracer_provider(TracerProvider())
        
        # Export vers Jaeger (si configuré)
        jaeger_exporter = JaegerExporter(
            agent_host_name="localhost",
            agent_port=14268,
        )
        
        span_processor = BatchSpanProcessor(jaeger_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        
        self.tracer = trace.get_tracer(__name__)
    
    async def _start_background_tasks(self) -> None:
        """Démarre les tâches de collecte en arrière-plan."""
        if self._system_metrics_enabled:
            self._system_collection_task = asyncio.create_task(self._collect_system_metrics())
    
    def register_metric(self, definition: MetricDefinition) -> None:
        """Enregistre une nouvelle métrique."""
        
        full_name = f"{definition.namespace}_{definition.subsystem}_{definition.name}"
        
        if definition.metric_type == MetricType.COUNTER:
            metric = Counter(
                full_name,
                definition.description,
                definition.labels,
                registry=self.registry
            )
        elif definition.metric_type == MetricType.GAUGE:
            metric = Gauge(
                full_name,
                definition.description,
                definition.labels,
                registry=self.registry
            )
        elif definition.metric_type == MetricType.HISTOGRAM:
            metric = Histogram(
                full_name,
                definition.description,
                definition.labels,
                buckets=definition.buckets,
                registry=self.registry
            )
        elif definition.metric_type == MetricType.SUMMARY:
            metric = Summary(
                full_name,
                definition.description,
                definition.labels,
                registry=self.registry
            )
        elif definition.metric_type == MetricType.INFO:
            metric = Info(
                full_name,
                definition.description,
                registry=self.registry
            )
        elif definition.metric_type == MetricType.ENUM:
            # Enum nécessite des états prédéfinis
            metric = PrometheusEnum(
                full_name,
                definition.description,
                definition.labels,
                states=['unknown'],  # État par défaut
                registry=self.registry
            )
        else:
            raise ValueError(f"Type de métrique non supporté: {definition.metric_type}")
        
        self._metrics[definition.name] = metric
        self._metric_definitions[definition.name] = definition
        
        logger.info(
            "Métrique enregistrée",
            name=definition.name,
            type=definition.metric_type.value,
            labels=definition.labels
        )
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Incrémente un counter."""
        if name not in self._metrics:
            logger.warning("Métrique non trouvée", name=name)
            return
        
        metric = self._metrics[name]
        if not isinstance(metric, Counter):
            logger.error("Type de métrique incorrect", name=name, expected="Counter")
            return
        
        if labels:
            metric.labels(**labels).inc(value)
        else:
            metric.inc(value)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Définit la valeur d'une gauge."""
        if name not in self._metrics:
            logger.warning("Métrique non trouvée", name=name)
            return
        
        metric = self._metrics[name]
        if not isinstance(metric, Gauge):
            logger.error("Type de métrique incorrect", name=name, expected="Gauge")
            return
        
        if labels:
            metric.labels(**labels).set(value)
        else:
            metric.set(value)
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Ajoute une observation à un histogramme."""
        if name not in self._metrics:
            logger.warning("Métrique non trouvée", name=name)
            return
        
        metric = self._metrics[name]
        if not isinstance(metric, Histogram):
            logger.error("Type de métrique incorrect", name=name, expected="Histogram")
            return
        
        if labels:
            metric.labels(**labels).observe(value)
        else:
            metric.observe(value)
    
    def observe_summary(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Ajoute une observation à un summary."""
        if name not in self._metrics:
            logger.warning("Métrique non trouvée", name=name)
            return
        
        metric = self._metrics[name]
        if not isinstance(metric, Summary):
            logger.error("Type de métrique incorrect", name=name, expected="Summary")
            return
        
        if labels:
            metric.labels(**labels).observe(value)
        else:
            metric.observe(value)
    
    async def _collect_system_metrics(self) -> None:
        """Collecte automatique des métriques système."""
        
        # Enregistrement des métriques système si pas déjà fait
        system_metrics = [
            MetricDefinition("cpu_usage_percent", MetricType.GAUGE, "Utilisation CPU en pourcentage"),
            MetricDefinition("memory_usage_bytes", MetricType.GAUGE, "Utilisation mémoire en bytes"),
            MetricDefinition("memory_usage_percent", MetricType.GAUGE, "Utilisation mémoire en pourcentage"),
            MetricDefinition("disk_usage_bytes", MetricType.GAUGE, "Utilisation disque en bytes", ["path"]),
            MetricDefinition("disk_usage_percent", MetricType.GAUGE, "Utilisation disque en pourcentage", ["path"]),
            MetricDefinition("network_bytes_sent", MetricType.COUNTER, "Bytes réseau envoyés", ["interface"]),
            MetricDefinition("network_bytes_recv", MetricType.COUNTER, "Bytes réseau reçus", ["interface"]),
            MetricDefinition("process_count", MetricType.GAUGE, "Nombre de processus"),
            MetricDefinition("load_average", MetricType.GAUGE, "Charge moyenne du système", ["period"]),
        ]
        
        for metric_def in system_metrics:
            if metric_def.name not in self._metrics:
                self.register_metric(metric_def)
        
        while True:
            try:
                # CPU
                cpu_percent = psutil.cpu_percent(interval=1)
                self.set_gauge("cpu_usage_percent", cpu_percent)
                
                # Mémoire
                memory = psutil.virtual_memory()
                self.set_gauge("memory_usage_bytes", memory.used)
                self.set_gauge("memory_usage_percent", memory.percent)
                
                # Disque
                for partition in psutil.disk_partitions():
                    try:
                        disk_usage = psutil.disk_usage(partition.mountpoint)
                        self.set_gauge("disk_usage_bytes", disk_usage.used, {"path": partition.mountpoint})
                        self.set_gauge("disk_usage_percent", 
                                     (disk_usage.used / disk_usage.total) * 100,
                                     {"path": partition.mountpoint})
                    except PermissionError:
                        continue
                
                # Réseau
                network_io = psutil.net_io_counters(pernic=True)
                for interface, stats in network_io.items():
                    self.set_gauge("network_bytes_sent", stats.bytes_sent, {"interface": interface})
                    self.set_gauge("network_bytes_recv", stats.bytes_recv, {"interface": interface})
                
                # Processus
                self.set_gauge("process_count", len(psutil.pids()))
                
                # Charge système (Linux uniquement)
                try:
                    load_avg = psutil.getloadavg()
                    self.set_gauge("load_average", load_avg[0], {"period": "1min"})
                    self.set_gauge("load_average", load_avg[1], {"period": "5min"})
                    self.set_gauge("load_average", load_avg[2], {"period": "15min"})
                except AttributeError:
                    pass  # getloadavg n'est pas disponible sur tous les OS
                
                await asyncio.sleep(30)  # Collecte toutes les 30 secondes
                
            except Exception as e:
                logger.error("Erreur lors de la collecte des métriques système", error=str(e))
                await asyncio.sleep(60)
    
    def get_prometheus_metrics(self) -> str:
        """Retourne les métriques au format Prometheus."""
        return generate_latest(self.registry).decode('utf-8')
    
    def get_metric_families(self) -> List[Any]:
        """Retourne les familles de métriques."""
        return list(self.registry.collect())


class AlertManager:
    """
    Gestionnaire d'alertes avancé avec seuils adaptatifs et ML.
    
    Fonctionnalités:
    - Règles d'alertes dynamiques
    - Seuils adaptatifs basés sur l'historique
    - Détection d'anomalies avec Machine Learning
    - Corrélation d'événements
    - Escalation automatique
    - Intégration Slack/Email/PagerDuty
    """
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        enable_ml_detection: bool = True,
        webhook_urls: Optional[Dict[str, str]] = None
    ):
        self.metrics_collector = metrics_collector
        self.enable_ml_detection = enable_ml_detection
        self.webhook_urls = webhook_urls or {}
        
        # Stockage des règles et états
        self._alert_rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, Dict[str, Any]] = {}
        self._alert_history: deque = deque(maxlen=10000)
        
        # Détection d'anomalies ML
        if enable_ml_detection:
            self._setup_ml_detection()
        
        # Cache pour les données historiques
        self._historical_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        
        # Tâche de vérification des alertes
        self._alert_check_task = None
        
        # Démarrage du monitoring
        asyncio.create_task(self._start_alert_monitoring())
    
    def _setup_ml_detection(self) -> None:
        """Configure la détection d'anomalies par Machine Learning."""
        self.anomaly_detector = IsolationForest(
            contamination=0.1,  # 10% d'anomalies attendues
            random_state=42
        )
        self.scaler = StandardScaler()
        self._ml_training_data: Dict[str, List[float]] = defaultdict(list)
        self._ml_models_trained = False
    
    async def _start_alert_monitoring(self) -> None:
        """Démarre le monitoring des alertes."""
        self._alert_check_task = asyncio.create_task(self._alert_monitoring_loop())
    
    def register_alert_rule(self, rule: AlertRule) -> None:
        """Enregistre une nouvelle règle d'alerte."""
        self._alert_rules[rule.name] = rule
        
        logger.info(
            "Règle d'alerte enregistrée",
            name=rule.name,
            metric=rule.metric_name,
            condition=rule.condition,
            threshold=rule.threshold,
            severity=rule.severity.value
        )
    
    async def _alert_monitoring_loop(self) -> None:
        """Boucle principale de monitoring des alertes."""
        while True:
            try:
                await self._check_all_alerts()
                await self._update_ml_models()
                await asyncio.sleep(10)  # Vérification toutes les 10 secondes
            except Exception as e:
                logger.error("Erreur dans la boucle de monitoring", error=str(e))
                await asyncio.sleep(30)
    
    async def _check_all_alerts(self) -> None:
        """Vérifie toutes les règles d'alertes."""
        
        # Récupération des métriques actuelles
        current_metrics = await self._get_current_metrics()
        
        for rule_name, rule in self._alert_rules.items():
            if not rule.enabled:
                continue
            
            try:
                await self._check_alert_rule(rule, current_metrics)
            except Exception as e:
                logger.error(
                    "Erreur lors de la vérification d'une règle",
                    rule_name=rule_name,
                    error=str(e)
                )
    
    async def _get_current_metrics(self) -> Dict[str, float]:
        """Récupère les valeurs actuelles des métriques."""
        
        current_metrics = {}
        
        # Extraction des valeurs depuis le registry Prometheus
        for family in self.metrics_collector.get_metric_families():
            for sample in family.samples:
                metric_name = sample.name
                value = sample.value
                
                # Stockage de la valeur (on prend la dernière pour les métriques avec labels)
                current_metrics[metric_name] = value
                
                # Ajout à l'historique pour ML
                if self.enable_ml_detection:
                    self._ml_training_data[metric_name].append(value)
                    if len(self._ml_training_data[metric_name]) > 1000:
                        self._ml_training_data[metric_name].pop(0)
        
        return current_metrics
    
    async def _check_alert_rule(self, rule: AlertRule, current_metrics: Dict[str, float]) -> None:
        """Vérifie une règle d'alerte spécifique."""
        
        if rule.metric_name not in current_metrics:
            return
        
        current_value = current_metrics[rule.metric_name]
        threshold = rule.threshold
        
        # Adaptation du seuil si demandé
        if rule.metric_name in self._historical_data:
            threshold = await self._calculate_adaptive_threshold(rule)
        
        # Vérification de la condition
        alert_triggered = self._evaluate_condition(current_value, rule.condition, threshold)
        
        alert_key = f"{rule.name}_{rule.metric_name}"
        
        if alert_triggered:
            if alert_key not in self._active_alerts:
                # Nouvelle alerte
                alert_data = {
                    "rule": rule,
                    "triggered_at": time.time(),
                    "current_value": current_value,
                    "threshold": threshold,
                    "count": 1
                }
                self._active_alerts[alert_key] = alert_data
                
                # Vérification de la durée avant déclenchement
                if rule.duration == 0:
                    await self._fire_alert(alert_data)
            else:
                # Alerte existante - mise à jour
                alert_data = self._active_alerts[alert_key]
                alert_data["current_value"] = current_value
                alert_data["count"] += 1
                
                # Vérification si la durée est atteinte
                duration_passed = time.time() - alert_data["triggered_at"]
                if duration_passed >= rule.duration:
                    await self._fire_alert(alert_data)
        else:
            # Condition non remplie - résolution de l'alerte si active
            if alert_key in self._active_alerts:
                await self._resolve_alert(alert_key)
    
    def _evaluate_condition(self, current_value: float, condition: str, threshold: float) -> bool:
        """Évalue une condition d'alerte."""
        
        if condition == ">":
            return current_value > threshold
        elif condition == "<":
            return current_value < threshold
        elif condition == ">=":
            return current_value >= threshold
        elif condition == "<=":
            return current_value <= threshold
        elif condition == "==":
            return abs(current_value - threshold) < 0.001  # Tolérance pour float
        elif condition == "!=":
            return abs(current_value - threshold) >= 0.001
        else:
            logger.error("Condition d'alerte non supportée", condition=condition)
            return False
    
    async def _calculate_adaptive_threshold(self, rule: AlertRule) -> float:
        """Calcule un seuil adaptatif basé sur l'historique."""
        
        if rule.metric_name not in self._historical_data:
            return rule.threshold
        
        historical_values = list(self._historical_data[rule.metric_name])
        
        if len(historical_values) < 10:
            return rule.threshold
        
        # Statistiques de base
        mean_value = np.mean(historical_values)
        std_value = np.std(historical_values)
        
        # Ajustement du seuil basé sur l'écart type
        if rule.condition in [">", ">="]:
            # Pour les alertes de dépassement, seuil = moyenne + N * écart-type
            adaptive_threshold = mean_value + (2 * std_value)
        else:
            # Pour les alertes de sous-performance, seuil = moyenne - N * écart-type
            adaptive_threshold = mean_value - (2 * std_value)
        
        # Hybridation avec le seuil fixe (70% adaptatif, 30% fixe)
        final_threshold = (adaptive_threshold * 0.7) + (rule.threshold * 0.3)
        
        return final_threshold
    
    async def _fire_alert(self, alert_data: Dict[str, Any]) -> None:
        """Déclenche une alerte."""
        
        rule = alert_data["rule"]
        
        # Vérification du cooldown
        if await self._is_in_cooldown(rule.name):
            return
        
        # Création du message d'alerte
        alert_message = {
            "alert_name": rule.name,
            "metric_name": rule.metric_name,
            "severity": rule.severity.value,
            "description": rule.description,
            "current_value": alert_data["current_value"],
            "threshold": alert_data["threshold"],
            "condition": rule.condition,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "labels": rule.labels,
            "annotations": rule.annotations
        }
        
        # Ajout à l'historique
        self._alert_history.append(alert_message)
        
        # Envoi des notifications
        await self._send_alert_notifications(alert_message)
        
        # Mise à jour du cooldown
        self._active_alerts[f"{rule.name}_{rule.metric_name}"]["last_fired"] = time.time()
        
        logger.error(
            "Alerte déclenchée",
            **{k: v for k, v in alert_message.items() if k not in ['labels', 'annotations']}
        )
    
    async def _resolve_alert(self, alert_key: str) -> None:
        """Résout une alerte active."""
        
        if alert_key in self._active_alerts:
            alert_data = self._active_alerts[alert_key]
            rule = alert_data["rule"]
            
            resolution_message = {
                "alert_name": rule.name,
                "metric_name": rule.metric_name,
                "severity": "resolved",
                "description": f"Alerte résolue: {rule.description}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration": time.time() - alert_data["triggered_at"]
            }
            
            # Envoi de la notification de résolution
            await self._send_resolution_notifications(resolution_message)
            
            # Suppression de l'alerte active
            del self._active_alerts[alert_key]
            
            logger.info(
                "Alerte résolue",
                alert_name=rule.name,
                metric_name=rule.metric_name,
                duration=resolution_message["duration"]
            )
    
    async def _is_in_cooldown(self, rule_name: str) -> bool:
        """Vérifie si une règle est en période de cooldown."""
        
        for alert_key, alert_data in self._active_alerts.items():
            if alert_data["rule"].name == rule_name:
                last_fired = alert_data.get("last_fired", 0)
                cooldown_period = alert_data["rule"].cooldown
                
                if time.time() - last_fired < cooldown_period:
                    return True
        
        return False
    
    async def _send_alert_notifications(self, alert_message: Dict[str, Any]) -> None:
        """Envoie les notifications d'alerte."""
        
        # Slack
        if "slack" in self.webhook_urls:
            await self._send_slack_notification(alert_message)
        
        # Email (à implémenter)
        # if "email" in self.webhook_urls:
        #     await self._send_email_notification(alert_message)
        
        # PagerDuty (à implémenter)
        # if "pagerduty" in self.webhook_urls:
        #     await self._send_pagerduty_notification(alert_message)
    
    async def _send_resolution_notifications(self, resolution_message: Dict[str, Any]) -> None:
        """Envoie les notifications de résolution."""
        
        if "slack" in self.webhook_urls:
            await self._send_slack_resolution(resolution_message)
    
    async def _send_slack_notification(self, alert_message: Dict[str, Any]) -> None:
        """Envoie une notification Slack."""
        
        try:
            webhook_url = self.webhook_urls["slack"]
            
            # Couleur selon la sévérité
            color_map = {
                "info": "#36a64f",
                "warning": "#ff9900", 
                "critical": "#ff0000",
                "fatal": "#800000"
            }
            
            color = color_map.get(alert_message["severity"], "#ff0000")
            
            slack_payload = {
                "attachments": [
                    {
                        "color": color,
                        "title": f"🚨 Alerte: {alert_message['alert_name']}",
                        "text": alert_message["description"],
                        "fields": [
                            {
                                "title": "Métrique",
                                "value": alert_message["metric_name"],
                                "short": True
                            },
                            {
                                "title": "Valeur actuelle",
                                "value": f"{alert_message['current_value']:.2f}",
                                "short": True
                            },
                            {
                                "title": "Seuil",
                                "value": f"{alert_message['condition']} {alert_message['threshold']:.2f}",
                                "short": True
                            },
                            {
                                "title": "Sévérité",
                                "value": alert_message["severity"].upper(),
                                "short": True
                            }
                        ],
                        "timestamp": int(time.time())
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=slack_payload) as response:
                    if response.status != 200:
                        logger.error(
                            "Erreur lors de l'envoi Slack",
                            status=response.status,
                            response=await response.text()
                        )
        
        except Exception as e:
            logger.error("Erreur lors de l'envoi de notification Slack", error=str(e))
    
    async def _send_slack_resolution(self, resolution_message: Dict[str, Any]) -> None:
        """Envoie une notification de résolution Slack."""
        
        try:
            webhook_url = self.webhook_urls["slack"]
            
            slack_payload = {
                "attachments": [
                    {
                        "color": "#36a64f",
                        "title": f"✅ Alerte résolue: {resolution_message['alert_name']}",
                        "text": resolution_message["description"],
                        "fields": [
                            {
                                "title": "Durée",
                                "value": f"{resolution_message['duration']:.0f} secondes",
                                "short": True
                            }
                        ],
                        "timestamp": int(time.time())
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=slack_payload) as response:
                    if response.status != 200:
                        logger.error(
                            "Erreur lors de l'envoi Slack résolution",
                            status=response.status
                        )
        
        except Exception as e:
            logger.error("Erreur lors de l'envoi de résolution Slack", error=str(e))
    
    async def _update_ml_models(self) -> None:
        """Met à jour les modèles de détection d'anomalies."""
        
        if not self.enable_ml_detection:
            return
        
        # Mise à jour périodique (toutes les heures)
        current_time = time.time()
        if not hasattr(self, '_last_ml_update'):
            self._last_ml_update = current_time
        
        if current_time - self._last_ml_update < 3600:  # 1 heure
            return
        
        # Entraînement des modèles avec les nouvelles données
        for metric_name, data in self._ml_training_data.items():
            if len(data) >= 100:  # Minimum de données pour l'entraînement
                try:
                    # Préparation des données
                    X = np.array(data).reshape(-1, 1)
                    X_scaled = self.scaler.fit_transform(X)
                    
                    # Entraînement du modèle
                    self.anomaly_detector.fit(X_scaled)
                    
                    # Détection d'anomalies sur les dernières valeurs
                    recent_data = X_scaled[-10:]  # 10 dernières valeurs
                    anomalies = self.anomaly_detector.predict(recent_data)
                    
                    # Génération d'alertes pour les anomalies détectées
                    for i, is_anomaly in enumerate(anomalies):
                        if is_anomaly == -1:  # Anomalie détectée
                            await self._generate_ml_alert(metric_name, data[-(10-i)])
                
                except Exception as e:
                    logger.error(
                        "Erreur lors de l'entraînement ML",
                        metric_name=metric_name,
                        error=str(e)
                    )
        
        self._last_ml_update = current_time
        self._ml_models_trained = True
    
    async def _generate_ml_alert(self, metric_name: str, anomalous_value: float) -> None:
        """Génère une alerte basée sur la détection ML d'anomalie."""
        
        alert_message = {
            "alert_name": f"ml_anomaly_{metric_name}",
            "metric_name": metric_name,
            "severity": "warning",
            "description": f"Anomalie détectée par ML sur la métrique {metric_name}",
            "current_value": anomalous_value,
            "threshold": "ML_DETECTED",
            "condition": "anomaly",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "labels": {"type": "ml_anomaly"},
            "annotations": {"algorithm": "isolation_forest"}
        }
        
        # Ajout à l'historique
        self._alert_history.append(alert_message)
        
        # Envoi des notifications si configurées
        await self._send_alert_notifications(alert_message)
        
        logger.warning(
            "Anomalie ML détectée",
            metric_name=metric_name,
            value=anomalous_value
        )
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Retourne la liste des alertes actives."""
        
        active_alerts = []
        for alert_key, alert_data in self._active_alerts.items():
            rule = alert_data["rule"]
            active_alerts.append({
                "name": rule.name,
                "metric_name": rule.metric_name,
                "severity": rule.severity.value,
                "triggered_at": alert_data["triggered_at"],
                "current_value": alert_data["current_value"],
                "threshold": alert_data.get("threshold", rule.threshold),
                "count": alert_data["count"]
            })
        
        return active_alerts
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retourne l'historique des alertes."""
        return list(self._alert_history)[-limit:]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques des alertes."""
        
        total_alerts = len(self._alert_history)
        active_count = len(self._active_alerts)
        
        # Comptage par sévérité
        severity_counts = defaultdict(int)
        for alert in self._alert_history:
            severity_counts[alert["severity"]] += 1
        
        # Métriques les plus problématiques
        metric_counts = defaultdict(int)
        for alert in self._alert_history:
            metric_counts[alert["metric_name"]] += 1
        
        top_metrics = sorted(metric_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_alerts": total_alerts,
            "active_alerts": active_count,
            "severity_breakdown": dict(severity_counts),
            "top_problematic_metrics": top_metrics,
            "ml_models_trained": getattr(self, '_ml_models_trained', False)
        }


class HealthMonitor:
    """
    Moniteur de santé des composants avec checks intelligents.
    
    Fonctionnalités:
    - Health checks périodiques
    - Détection de dégradation progressive
    - Corrélation de santé entre composants
    - Auto-healing basique
    - Rapports de santé détaillés
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        
        # Stockage des checks et résultats
        self._health_checks: Dict[str, HealthCheck] = {}
        self._health_status: Dict[str, HealthStatus] = {}
        self._health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # État global
        self._global_health = HealthStatus.UNKNOWN
        
        # Tâche de monitoring
        self._health_monitoring_task = None
        
        # Métriques de santé
        self._register_health_metrics()
        
        # Démarrage du monitoring
        asyncio.create_task(self._start_health_monitoring())
    
    def _register_health_metrics(self) -> None:
        """Enregistre les métriques de santé."""
        
        health_metrics = [
            MetricDefinition(
                "component_health_status",
                MetricType.GAUGE,
                "Statut de santé des composants (0=unhealthy, 1=degraded, 2=healthy)",
                ["component"]
            ),
            MetricDefinition(
                "health_check_duration_seconds",
                MetricType.HISTOGRAM,
                "Durée des health checks",
                ["component"],
                buckets=[0.001, 0.01, 0.1, 1.0, 5.0, 10.0]
            ),
            MetricDefinition(
                "health_check_failures_total",
                MetricType.COUNTER,
                "Nombre total d'échecs de health checks",
                ["component"]
            )
        ]
        
        for metric_def in health_metrics:
            if metric_def.name not in self.metrics_collector._metrics:
                self.metrics_collector.register_metric(metric_def)
    
    async def _start_health_monitoring(self) -> None:
        """Démarre le monitoring de santé."""
        self._health_monitoring_task = asyncio.create_task(self._health_monitoring_loop())
    
    def register_health_check(self, health_check: HealthCheck) -> None:
        """Enregistre un nouveau health check."""
        
        self._health_checks[health_check.name] = health_check
        self._health_status[health_check.name] = HealthStatus.UNKNOWN
        
        logger.info(
            "Health check enregistré",
            name=health_check.name,
            interval=health_check.interval,
            critical=health_check.critical
        )
    
    async def _health_monitoring_loop(self) -> None:
        """Boucle principale de monitoring de santé."""
        
        # Stockage des dernières vérifications
        last_checks: Dict[str, float] = {}
        
        while True:
            try:
                current_time = time.time()
                
                for check_name, health_check in self._health_checks.items():
                    if not health_check.enabled:
                        continue
                    
                    # Vérification de l'intervalle
                    last_check = last_checks.get(check_name, 0)
                    if current_time - last_check < health_check.interval:
                        continue
                    
                    # Exécution du check
                    await self._execute_health_check(health_check)
                    last_checks[check_name] = current_time
                
                # Mise à jour de la santé globale
                self._update_global_health()
                
                await asyncio.sleep(5)  # Vérification toutes les 5 secondes
                
            except Exception as e:
                logger.error("Erreur dans la boucle de health monitoring", error=str(e))
                await asyncio.sleep(30)
    
    async def _execute_health_check(self, health_check: HealthCheck) -> None:
        """Exécute un health check spécifique."""
        
        start_time = time.time()
        
        try:
            # Exécution avec timeout et retries
            for attempt in range(health_check.retries + 1):
                try:
                    status, message = await asyncio.wait_for(
                        health_check.check_function(),
                        timeout=health_check.timeout
                    )
                    break
                except asyncio.TimeoutError:
                    if attempt == health_check.retries:
                        status = HealthStatus.UNHEALTHY
                        message = f"Timeout après {health_check.timeout}s"
                    else:
                        await asyncio.sleep(1)  # Attente avant retry
                        continue
                except Exception as e:
                    if attempt == health_check.retries:
                        status = HealthStatus.UNHEALTHY
                        message = f"Erreur: {str(e)}"
                    else:
                        await asyncio.sleep(1)
                        continue
            
            # Mise à jour du statut
            previous_status = self._health_status.get(health_check.name)
            self._health_status[health_check.name] = status
            
            # Ajout à l'historique
            health_record = {
                "timestamp": time.time(),
                "status": status,
                "message": message,
                "duration": time.time() - start_time
            }
            self._health_history[health_check.name].append(health_record)
            
            # Métriques
            status_value = {"healthy": 2, "degraded": 1, "unhealthy": 0, "unknown": -1}[status.value]
            self.metrics_collector.set_gauge(
                "component_health_status",
                status_value,
                {"component": health_check.name}
            )
            
            self.metrics_collector.observe_histogram(
                "health_check_duration_seconds",
                time.time() - start_time,
                {"component": health_check.name}
            )
            
            if status == HealthStatus.UNHEALTHY:
                self.metrics_collector.increment_counter(
                    "health_check_failures_total",
                    labels={"component": health_check.name}
                )
            
            # Log des changements de statut
            if previous_status and previous_status != status:
                logger.info(
                    "Changement de statut de santé",
                    component=health_check.name,
                    previous_status=previous_status.value,
                    new_status=status.value,
                    message=message
                )
        
        except Exception as e:
            logger.error(
                "Erreur lors de l'exécution du health check",
                component=health_check.name,
                error=str(e)
            )
            
            # Statut d'erreur
            self._health_status[health_check.name] = HealthStatus.UNKNOWN
    
    def _update_global_health(self) -> None:
        """Met à jour la santé globale du système."""
        
        if not self._health_status:
            self._global_health = HealthStatus.UNKNOWN
            return
        
        # Vérification des composants critiques
        critical_components = [
            name for name, check in self._health_checks.items()
            if check.critical and check.enabled
        ]
        
        critical_unhealthy = any(
            self._health_status.get(name) == HealthStatus.UNHEALTHY
            for name in critical_components
        )
        
        if critical_unhealthy:
            self._global_health = HealthStatus.UNHEALTHY
            return
        
        # Vérification de la dégradation
        degraded_count = sum(
            1 for status in self._health_status.values()
            if status == HealthStatus.DEGRADED
        )
        
        unhealthy_count = sum(
            1 for status in self._health_status.values()
            if status == HealthStatus.UNHEALTHY
        )
        
        total_components = len(self._health_status)
        
        if unhealthy_count > 0 or degraded_count > (total_components * 0.3):
            self._global_health = HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY for status in self._health_status.values()):
            self._global_health = HealthStatus.HEALTHY
        else:
            self._global_health = HealthStatus.DEGRADED
    
    def get_health_status(self, component_name: Optional[str] = None) -> Union[HealthStatus, Dict[str, HealthStatus]]:
        """Retourne le statut de santé d'un composant ou global."""
        
        if component_name:
            return self._health_status.get(component_name, HealthStatus.UNKNOWN)
        else:
            return {
                "global": self._global_health,
                **self._health_status
            }
    
    def get_health_report(self) -> Dict[str, Any]:
        """Génère un rapport de santé détaillé."""
        
        # Statistiques globales
        total_components = len(self._health_checks)
        healthy_count = sum(1 for s in self._health_status.values() if s == HealthStatus.HEALTHY)
        degraded_count = sum(1 for s in self._health_status.values() if s == HealthStatus.DEGRADED)
        unhealthy_count = sum(1 for s in self._health_status.values() if s == HealthStatus.UNHEALTHY)
        
        # Composants problématiques
        problematic_components = []
        for name, status in self._health_status.items():
            if status in [HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]:
                recent_history = list(self._health_history[name])[-5:]
                problematic_components.append({
                    "name": name,
                    "status": status.value,
                    "critical": self._health_checks[name].critical,
                    "recent_history": [
                        {
                            "timestamp": record["timestamp"],
                            "status": record["status"].value,
                            "message": record["message"]
                        }
                        for record in recent_history
                    ]
                })
        
        return {
            "global_health": self._global_health.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_components": total_components,
                "healthy": healthy_count,
                "degraded": degraded_count,
                "unhealthy": unhealthy_count,
                "health_percentage": (healthy_count / total_components * 100) if total_components > 0 else 0
            },
            "problematic_components": problematic_components,
            "recommendations": self._generate_health_recommendations()
        }
    
    def _generate_health_recommendations(self) -> List[str]:
        """Génère des recommandations basées sur l'état de santé."""
        
        recommendations = []
        
        # Analyse des patterns de défaillance
        for name, history in self._health_history.items():
            recent_failures = [
                record for record in list(history)[-10:]
                if record["status"] == HealthStatus.UNHEALTHY
            ]
            
            if len(recent_failures) >= 3:
                recommendations.append(
                    f"Composant {name}: Échecs fréquents détectés, "
                    "vérifier la configuration et les dépendances"
                )
        
        # Vérifications de performance
        for name, history in self._health_history.items():
            recent_records = list(history)[-5:]
            if recent_records:
                avg_duration = np.mean([r["duration"] for r in recent_records])
                if avg_duration > 5.0:  # Plus de 5 secondes
                    recommendations.append(
                        f"Composant {name}: Health checks lents (avg: {avg_duration:.2f}s), "
                        "optimiser les vérifications"
                    )
        
        # État global
        if self._global_health == HealthStatus.DEGRADED:
            recommendations.append(
                "Système en état dégradé: Examiner les composants problématiques "
                "et leurs interdépendances"
            )
        elif self._global_health == HealthStatus.UNHEALTHY:
            recommendations.append(
                "Système critique: Intervention immédiate requise sur les "
                "composants critiques défaillants"
            )
        
        return recommendations


# Fonctions d'aide pour la création de health checks
async def database_health_check(connection_string: str) -> Tuple[HealthStatus, str]:
    """Health check pour base de données PostgreSQL."""
    try:
        conn = await asyncpg.connect(connection_string)
        await conn.execute("SELECT 1")
        await conn.close()
        return HealthStatus.HEALTHY, "Base de données accessible"
    except Exception as e:
        return HealthStatus.UNHEALTHY, f"Erreur DB: {str(e)}"


async def redis_health_check(redis_url: str) -> Tuple[HealthStatus, str]:
    """Health check pour Redis."""
    try:
        redis = aioredis.from_url(redis_url)
        await redis.ping()
        await redis.close()
        return HealthStatus.HEALTHY, "Redis accessible"
    except Exception as e:
        return HealthStatus.UNHEALTHY, f"Erreur Redis: {str(e)}"


async def http_endpoint_health_check(url: str, timeout: int = 5) -> Tuple[HealthStatus, str]:
    """Health check pour endpoint HTTP."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=timeout) as response:
                if response.status == 200:
                    return HealthStatus.HEALTHY, f"Endpoint accessible (status: {response.status})"
                else:
                    return HealthStatus.DEGRADED, f"Status non optimal: {response.status}"
    except Exception as e:
        return HealthStatus.UNHEALTHY, f"Erreur endpoint: {str(e)}"


async def disk_space_health_check(path: str, warning_threshold: float = 80.0, critical_threshold: float = 90.0) -> Tuple[HealthStatus, str]:
    """Health check pour l'espace disque."""
    try:
        disk_usage = psutil.disk_usage(path)
        usage_percent = (disk_usage.used / disk_usage.total) * 100
        
        if usage_percent >= critical_threshold:
            return HealthStatus.UNHEALTHY, f"Espace disque critique: {usage_percent:.1f}%"
        elif usage_percent >= warning_threshold:
            return HealthStatus.DEGRADED, f"Espace disque faible: {usage_percent:.1f}%"
        else:
            return HealthStatus.HEALTHY, f"Espace disque OK: {usage_percent:.1f}%"
    except Exception as e:
        return HealthStatus.UNKNOWN, f"Erreur vérification disque: {str(e)}"


async def memory_health_check(warning_threshold: float = 80.0, critical_threshold: float = 90.0) -> Tuple[HealthStatus, str]:
    """Health check pour l'utilisation mémoire."""
    try:
        memory = psutil.virtual_memory()
        usage_percent = memory.percent
        
        if usage_percent >= critical_threshold:
            return HealthStatus.UNHEALTHY, f"Mémoire critique: {usage_percent:.1f}%"
        elif usage_percent >= warning_threshold:
            return HealthStatus.DEGRADED, f"Mémoire élevée: {usage_percent:.1f}%"
        else:
            return HealthStatus.HEALTHY, f"Mémoire OK: {usage_percent:.1f}%"
    except Exception as e:
        return HealthStatus.UNKNOWN, f"Erreur vérification mémoire: {str(e)}"


# Instances globales
global_metrics_collector = MetricsCollector()
global_alert_manager = AlertManager(global_metrics_collector)
global_health_monitor = HealthMonitor(global_metrics_collector)


# Configuration par défaut
def setup_default_monitoring():
    """Configure le monitoring par défaut."""
    
    # Enregistrement des métriques business
    business_metrics = [
        MetricDefinition(
            "data_collection_operations_total",
            MetricType.COUNTER,
            "Nombre total d'opérations de collecte",
            ["collector_type", "tenant_id", "status"]
        ),
        MetricDefinition(
            "data_collection_duration_seconds",
            MetricType.HISTOGRAM,
            "Durée des opérations de collecte",
            ["collector_type", "tenant_id"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        ),
        MetricDefinition(
            "data_records_processed_total",
            MetricType.COUNTER,
            "Nombre total d'enregistrements traités",
            ["collector_type", "tenant_id"]
        ),
        MetricDefinition(
            "data_size_bytes_total",
            MetricType.COUNTER,
            "Taille totale des données collectées en bytes",
            ["collector_type", "tenant_id"]
        ),
        MetricDefinition(
            "active_collectors",
            MetricType.GAUGE,
            "Nombre de collecteurs actifs",
            ["type"]
        )
    ]
    
    for metric_def in business_metrics:
        global_metrics_collector.register_metric(metric_def)
    
    # Alertes par défaut
    default_alerts = [
        AlertRule(
            name="high_cpu_usage",
            metric_name="spotify_ai_agent_collectors_cpu_usage_percent",
            condition=">",
            threshold=80.0,
            duration=300,  # 5 minutes
            severity=AlertSeverity.WARNING,
            description="Utilisation CPU élevée détectée"
        ),
        AlertRule(
            name="critical_cpu_usage",
            metric_name="spotify_ai_agent_collectors_cpu_usage_percent",
            condition=">",
            threshold=95.0,
            duration=60,  # 1 minute
            severity=AlertSeverity.CRITICAL,
            description="Utilisation CPU critique détectée"
        ),
        AlertRule(
            name="high_memory_usage",
            metric_name="spotify_ai_agent_collectors_memory_usage_percent",
            condition=">",
            threshold=85.0,
            duration=300,
            severity=AlertSeverity.WARNING,
            description="Utilisation mémoire élevée détectée"
        ),
        AlertRule(
            name="data_collection_failures",
            metric_name="spotify_ai_agent_collectors_data_collection_operations_total",
            condition=">",
            threshold=10.0,
            duration=300,
            severity=AlertSeverity.WARNING,
            description="Taux d'échec de collecte élevé détecté"
        )
    ]
    
    for alert_rule in default_alerts:
        global_alert_manager.register_alert_rule(alert_rule)
    
    # Health checks par défaut
    default_health_checks = [
        HealthCheck(
            name="memory_check",
            check_function=memory_health_check,
            interval=60,
            critical=True
        ),
        HealthCheck(
            name="disk_check",
            check_function=lambda: disk_space_health_check("/"),
            interval=300,  # 5 minutes
            critical=True
        )
    ]
    
    for health_check in default_health_checks:
        global_health_monitor.register_health_check(health_check)


# Initialisation automatique
setup_default_monitoring()
