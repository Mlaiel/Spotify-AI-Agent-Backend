# =============================================================================
# Monitoring & Observability - Prometheus Metrics Enterprise
# =============================================================================
# 
# Système de monitoring enterprise avec métriques Prometheus avancées,
# alerting intelligent et observabilité complète.
#
# Développé par l'équipe d'experts techniques:
# - Lead Developer + AI Architect (Architecture monitoring et métriques)
# - Backend Senior Developer (Performance et optimisation)
# - ML Engineer (Analytics et métriques ML)
# - Microservices Architect (Observabilité distribuée)
#
# Direction Technique: Fahed Mlaiel
# =============================================================================

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
import time
import statistics
from pathlib import Path
import yaml

# Imports Prometheus et monitoring
import prometheus_client
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info, Enum as PrometheusEnum,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST as OPENMETRICS_CONTENT_TYPE

# Imports pour alerting
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp
import aioredis

# Imports pour intégrations
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine
import psutil
import socket
import platform

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# MODÈLES DE MONITORING
# =============================================================================

class MetricType(Enum):
    """Types de métriques Prometheus"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"
    ENUM = "enum"

class AlertSeverity(Enum):
    """Niveaux de sévérité des alertes"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertStatus(Enum):
    """Statuts des alertes"""
    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

@dataclass
class MetricDefinition:
    """Définition d'une métrique Prometheus"""
    name: str
    description: str
    metric_type: MetricType
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # Pour histogrammes
    quantiles: Optional[Dict[float, float]] = None  # Pour summaries
    namespace: str = "incidents"
    subsystem: str = ""

@dataclass
class AlertRule:
    """Règle d'alerte Prometheus"""
    name: str
    query: str  # PromQL query
    for_duration: str = "5m"
    severity: AlertSeverity = AlertSeverity.WARNING
    description: str = ""
    summary: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True

@dataclass
class Alert:
    """Instance d'alerte"""
    id: str
    rule_name: str
    status: AlertStatus
    severity: AlertSeverity
    started_at: datetime
    resolved_at: Optional[datetime] = None
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    value: Optional[float] = None
    tenant_id: str = ""

@dataclass
class NotificationChannel:
    """Canal de notification"""
    name: str
    type: str  # email, slack, webhook, pagerduty
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    severity_filter: List[AlertSeverity] = field(default_factory=list)

# =============================================================================
# REGISTRY ENTERPRISE DE MÉTRIQUES
# =============================================================================

class EnterpriseMetricsRegistry:
    """
    Registry enterprise pour métriques Prometheus avec gestion avancée,
    auto-découverte et métriques ML.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.registry = CollectorRegistry()
        self.metrics: Dict[str, Any] = {}
        self.custom_collectors: List[Callable] = []
        
        # Métriques système de base
        self.system_metrics = {}
        
        # Métriques business
        self.business_metrics = {}
        
        # Métriques ML/AI
        self.ml_metrics = {}
        
        # Configuration
        self.namespace = config.get('namespace', 'incidents')
        self.enable_system_metrics = config.get('enable_system_metrics', True)
        self.enable_ml_metrics = config.get('enable_ml_metrics', True)
        
        logger.info("EnterpriseMetricsRegistry initialisé")

    async def initialize(self):
        """Initialisation du registry de métriques"""
        try:
            # Création des métriques de base
            await self.create_base_metrics()
            
            # Métriques système si activées
            if self.enable_system_metrics:
                await self.create_system_metrics()
            
            # Métriques ML si activées
            if self.enable_ml_metrics:
                await self.create_ml_metrics()
            
            # Métriques business
            await self.create_business_metrics()
            
            # Enregistrement des collectors personnalisés
            await self.register_custom_collectors()
            
            logger.info("Registry de métriques initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur initialisation registry: {e}")
            raise

    async def create_base_metrics(self):
        """Création des métriques de base du système"""
        
        # Métriques d'incidents
        self.metrics['incidents_total'] = Counter(
            'incidents_total',
            'Total des incidents créés',
            ['tenant_id', 'severity', 'category', 'source'],
            registry=self.registry
        )
        
        self.metrics['incidents_duration'] = Histogram(
            'incidents_duration_seconds',
            'Durée de résolution des incidents',
            ['tenant_id', 'severity', 'category'],
            buckets=[30, 60, 300, 900, 1800, 3600, 7200, float('inf')],
            registry=self.registry
        )
        
        self.metrics['incidents_active'] = Gauge(
            'incidents_active',
            'Nombre d\'incidents actifs',
            ['tenant_id', 'severity'],
            registry=self.registry
        )
        
        # Métriques de performance
        self.metrics['api_requests_total'] = Counter(
            'api_requests_total',
            'Total des requêtes API',
            ['method', 'endpoint', 'status_code', 'tenant_id'],
            registry=self.registry
        )
        
        self.metrics['api_request_duration'] = Histogram(
            'api_request_duration_seconds',
            'Durée des requêtes API',
            ['method', 'endpoint', 'tenant_id'],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf')],
            registry=self.registry
        )
        
        # Métriques de métriques (meta-métriques)
        self.metrics['metrics_collected_total'] = Counter(
            'metrics_collected_total',
            'Total des métriques collectées',
            ['tenant_id', 'metric_type', 'source'],
            registry=self.registry
        )
        
        self.metrics['metrics_processing_duration'] = Histogram(
            'metrics_processing_duration_seconds',
            'Durée de traitement des métriques',
            ['operation', 'tenant_id'],
            registry=self.registry
        )

    async def create_system_metrics(self):
        """Création des métriques système"""
        
        # Métriques CPU
        self.system_metrics['cpu_usage_percent'] = Gauge(
            'system_cpu_usage_percent',
            'Utilisation CPU du système',
            ['cpu'],
            registry=self.registry
        )
        
        # Métriques mémoire
        self.system_metrics['memory_usage_bytes'] = Gauge(
            'system_memory_usage_bytes',
            'Utilisation mémoire en bytes',
            ['type'],  # used, available, total, cached
            registry=self.registry
        )
        
        # Métriques disque
        self.system_metrics['disk_usage_bytes'] = Gauge(
            'system_disk_usage_bytes',
            'Utilisation disque en bytes',
            ['device', 'type'],  # type: used, free, total
            registry=self.registry
        )
        
        # Métriques réseau
        self.system_metrics['network_bytes_total'] = Counter(
            'system_network_bytes_total',
            'Total bytes réseau',
            ['interface', 'direction'],  # direction: sent, received
            registry=self.registry
        )
        
        # Métriques base de données
        self.system_metrics['db_connections_active'] = Gauge(
            'database_connections_active',
            'Connexions base de données actives',
            ['database', 'tenant_id'],
            registry=self.registry
        )
        
        self.system_metrics['db_query_duration'] = Histogram(
            'database_query_duration_seconds',
            'Durée des requêtes base de données',
            ['query_type', 'tenant_id'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, float('inf')],
            registry=self.registry
        )

    async def create_ml_metrics(self):
        """Création des métriques ML/AI"""
        
        # Métriques de modèles ML
        self.ml_metrics['ml_model_predictions_total'] = Counter(
            'ml_model_predictions_total',
            'Total des prédictions ML',
            ['model_name', 'model_version', 'tenant_id', 'result'],
            registry=self.registry
        )
        
        self.ml_metrics['ml_model_inference_duration'] = Histogram(
            'ml_model_inference_duration_seconds',
            'Durée d\'inférence des modèles ML',
            ['model_name', 'model_version'],
            buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 5.0, float('inf')],
            registry=self.registry
        )
        
        self.ml_metrics['ml_model_accuracy'] = Gauge(
            'ml_model_accuracy',
            'Précision des modèles ML',
            ['model_name', 'model_version', 'dataset'],
            registry=self.registry
        )
        
        # Métriques de détection d'anomalies
        self.ml_metrics['anomalies_detected_total'] = Counter(
            'anomalies_detected_total',
            'Total des anomalies détectées',
            ['detector_type', 'severity', 'tenant_id'],
            registry=self.registry
        )
        
        self.ml_metrics['anomaly_score'] = Histogram(
            'anomaly_score',
            'Score d\'anomalie',
            ['detector_type', 'tenant_id'],
            buckets=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
            registry=self.registry
        )

    async def create_business_metrics(self):
        """Création des métriques business"""
        
        # Métriques utilisateurs
        self.business_metrics['users_active'] = Gauge(
            'users_active',
            'Utilisateurs actifs',
            ['tenant_id', 'time_range'],  # time_range: 1h, 24h, 7d
            registry=self.registry
        )
        
        # Métriques de coût
        self.business_metrics['cost_total'] = Gauge(
            'cost_total_usd',
            'Coût total en USD',
            ['tenant_id', 'cost_type'],  # compute, storage, network
            registry=self.registry
        )
        
        # Métriques SLA
        self.business_metrics['sla_uptime_percent'] = Gauge(
            'sla_uptime_percent',
            'Pourcentage de disponibilité SLA',
            ['service', 'tenant_id'],
            registry=self.registry
        )
        
        # Métriques d'automation
        self.business_metrics['automation_executions_total'] = Counter(
            'automation_executions_total',
            'Total des exécutions d\'automation',
            ['rule_id', 'status', 'tenant_id'],
            registry=self.registry
        )

    async def register_custom_collectors(self):
        """Enregistrement des collectors personnalisés"""
        
        # Collector pour métriques système
        if self.enable_system_metrics:
            system_collector = SystemMetricsCollector(self.system_metrics)
            self.registry.register(system_collector)
            self.custom_collectors.append(system_collector)
        
        logger.info(f"Enregistré {len(self.custom_collectors)} collectors personnalisés")

    def get_metric(self, name: str) -> Optional[Any]:
        """Récupération d'une métrique par nom"""
        return self.metrics.get(name) or self.system_metrics.get(name) or self.ml_metrics.get(name) or self.business_metrics.get(name)

    def record_incident(self, tenant_id: str, severity: str, category: str, source: str = "manual"):
        """Enregistrement d'un incident"""
        self.metrics['incidents_total'].labels(
            tenant_id=tenant_id,
            severity=severity,
            category=category,
            source=source
        ).inc()

    def record_incident_duration(self, tenant_id: str, severity: str, category: str, duration_seconds: float):
        """Enregistrement de la durée d'un incident"""
        self.metrics['incidents_duration'].labels(
            tenant_id=tenant_id,
            severity=severity,
            category=category
        ).observe(duration_seconds)

    def set_active_incidents(self, tenant_id: str, severity: str, count: int):
        """Mise à jour du nombre d'incidents actifs"""
        self.metrics['incidents_active'].labels(
            tenant_id=tenant_id,
            severity=severity
        ).set(count)

    def record_api_request(self, method: str, endpoint: str, status_code: int, tenant_id: str, duration: float):
        """Enregistrement d'une requête API"""
        self.metrics['api_requests_total'].labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code),
            tenant_id=tenant_id
        ).inc()
        
        self.metrics['api_request_duration'].labels(
            method=method,
            endpoint=endpoint,
            tenant_id=tenant_id
        ).observe(duration)

    def record_ml_prediction(self, model_name: str, model_version: str, tenant_id: str, result: str, duration: float):
        """Enregistrement d'une prédiction ML"""
        self.ml_metrics['ml_model_predictions_total'].labels(
            model_name=model_name,
            model_version=model_version,
            tenant_id=tenant_id,
            result=result
        ).inc()
        
        self.ml_metrics['ml_model_inference_duration'].labels(
            model_name=model_name,
            model_version=model_version
        ).observe(duration)

    def record_anomaly(self, detector_type: str, severity: str, tenant_id: str, score: float):
        """Enregistrement d'une anomalie détectée"""
        self.ml_metrics['anomalies_detected_total'].labels(
            detector_type=detector_type,
            severity=severity,
            tenant_id=tenant_id
        ).inc()
        
        self.ml_metrics['anomaly_score'].labels(
            detector_type=detector_type,
            tenant_id=tenant_id
        ).observe(score)

    def export_metrics(self, content_type: str = CONTENT_TYPE_LATEST) -> tuple[str, str]:
        """Export des métriques au format Prometheus"""
        try:
            metrics_output = generate_latest(self.registry)
            return metrics_output.decode('utf-8'), content_type
        except Exception as e:
            logger.error(f"Erreur export métriques: {e}")
            return "", content_type

# =============================================================================
# COLLECTOR SYSTÈME PERSONNALISÉ
# =============================================================================

class SystemMetricsCollector:
    """Collector personnalisé pour métriques système"""
    
    def __init__(self, system_metrics: Dict[str, Any]):
        self.system_metrics = system_metrics
        self._last_network_stats = {}
    
    def collect(self):
        """Collection des métriques système"""
        try:
            # Métriques CPU
            cpu_percentages = psutil.cpu_percent(percpu=True)
            for i, cpu_percent in enumerate(cpu_percentages):
                self.system_metrics['cpu_usage_percent'].labels(cpu=f"cpu{i}").set(cpu_percent)
            
            # Métriques mémoire
            memory = psutil.virtual_memory()
            self.system_metrics['memory_usage_bytes'].labels(type="used").set(memory.used)
            self.system_metrics['memory_usage_bytes'].labels(type="available").set(memory.available)
            self.system_metrics['memory_usage_bytes'].labels(type="total").set(memory.total)
            self.system_metrics['memory_usage_bytes'].labels(type="cached").set(getattr(memory, 'cached', 0))
            
            # Métriques disque
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    device = partition.device.replace('/', '_')
                    
                    self.system_metrics['disk_usage_bytes'].labels(device=device, type="used").set(usage.used)
                    self.system_metrics['disk_usage_bytes'].labels(device=device, type="free").set(usage.free)
                    self.system_metrics['disk_usage_bytes'].labels(device=device, type="total").set(usage.total)
                except (PermissionError, OSError):
                    continue
            
            # Métriques réseau
            network_stats = psutil.net_io_counters(pernic=True)
            for interface, stats in network_stats.items():
                # Calcul des deltas depuis la dernière collecte
                if interface in self._last_network_stats:
                    last_stats = self._last_network_stats[interface]
                    
                    bytes_sent_delta = stats.bytes_sent - last_stats.bytes_sent
                    bytes_recv_delta = stats.bytes_recv - last_stats.bytes_recv
                    
                    if bytes_sent_delta >= 0:  # Éviter les valeurs négatives lors des resets
                        self.system_metrics['network_bytes_total'].labels(
                            interface=interface, direction="sent"
                        )._value._value = stats.bytes_sent
                    
                    if bytes_recv_delta >= 0:
                        self.system_metrics['network_bytes_total'].labels(
                            interface=interface, direction="received"
                        )._value._value = stats.bytes_recv
                
                self._last_network_stats[interface] = stats
            
        except Exception as e:
            logger.error(f"Erreur collection métriques système: {e}")
        
        return []  # Les métriques sont déjà mises à jour dans le registry

# =============================================================================
# GESTIONNAIRE D'ALERTES ENTERPRISE
# =============================================================================

class EnterpriseAlertManager:
    """
    Gestionnaire d'alertes enterprise avec évaluation de règles,
    notification multi-canaux et gestion des escalades.
    """
    
    def __init__(self, config: Dict[str, Any], metrics_registry: EnterpriseMetricsRegistry):
        self.config = config
        self.metrics_registry = metrics_registry
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.notification_channels: Dict[str, NotificationChannel] = {}
        
        # Connexions
        self.redis_client: Optional[aioredis.Redis] = None
        self.db_engine = None
        
        # État d'évaluation
        self.evaluation_interval = config.get('evaluation_interval', 30)  # secondes
        self.running = False
        
        logger.info("EnterpriseAlertManager initialisé")

    async def initialize(self):
        """Initialisation du gestionnaire d'alertes"""
        try:
            # Connexions
            self.redis_client = aioredis.from_url(
                self.config['redis_url'],
                encoding='utf-8',
                decode_responses=True
            )
            
            self.db_engine = create_async_engine(self.config['database_url'])
            
            # Chargement des règles et canaux
            await self.load_alert_rules()
            await self.load_notification_channels()
            
            # Démarrage de l'évaluation périodique
            await self.start_evaluation_loop()
            
            logger.info("Gestionnaire d'alertes initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur initialisation AlertManager: {e}")
            raise

    async def load_alert_rules(self):
        """Chargement des règles d'alerte"""
        # Règles par défaut
        default_rules = [
            AlertRule(
                name="high_cpu_usage",
                query="system_cpu_usage_percent > 85",
                for_duration="5m",
                severity=AlertSeverity.WARNING,
                description="Utilisation CPU élevée",
                summary="CPU usage is above 85% for more than 5 minutes",
                labels={"team": "infrastructure"},
                annotations={"runbook": "https://runbooks.company.com/high-cpu"}
            ),
            AlertRule(
                name="incident_spike",
                query="rate(incidents_total[5m]) > 0.1",
                for_duration="2m",
                severity=AlertSeverity.CRITICAL,
                description="Pic d'incidents détecté",
                summary="Incident rate is above 0.1/sec for more than 2 minutes",
                labels={"team": "oncall"}
            ),
            AlertRule(
                name="api_high_latency",
                query="histogram_quantile(0.95, api_request_duration_seconds) > 2",
                for_duration="3m",
                severity=AlertSeverity.WARNING,
                description="Latence API élevée",
                summary="95th percentile API latency is above 2 seconds"
            ),
            AlertRule(
                name="anomaly_detection",
                query="anomaly_score > 0.9",
                for_duration="1m",
                severity=AlertSeverity.CRITICAL,
                description="Anomalie détectée par ML",
                summary="ML anomaly detector found anomaly with score > 0.9"
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.name] = rule
        
        logger.info(f"Chargé {len(self.alert_rules)} règles d'alerte")

    async def load_notification_channels(self):
        """Chargement des canaux de notification"""
        # Canaux par défaut
        default_channels = [
            NotificationChannel(
                name="email_ops",
                type="email",
                config={
                    "smtp_server": self.config.get("smtp_server", "localhost"),
                    "smtp_port": self.config.get("smtp_port", 587),
                    "username": self.config.get("smtp_username", ""),
                    "password": self.config.get("smtp_password", ""),
                    "recipients": ["ops@company.com", "oncall@company.com"]
                },
                severity_filter=[AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
            ),
            NotificationChannel(
                name="slack_alerts",
                type="slack",
                config={
                    "webhook_url": self.config.get("slack_webhook_url", ""),
                    "channel": "#alerts"
                },
                severity_filter=[AlertSeverity.WARNING, AlertSeverity.CRITICAL]
            ),
            NotificationChannel(
                name="webhook_general",
                type="webhook",
                config={
                    "url": self.config.get("webhook_url", ""),
                    "method": "POST",
                    "headers": {"Content-Type": "application/json"}
                }
            )
        ]
        
        for channel in default_channels:
            self.notification_channels[channel.name] = channel
        
        logger.info(f"Chargé {len(self.notification_channels)} canaux de notification")

    async def start_evaluation_loop(self):
        """Démarrage de la boucle d'évaluation des alertes"""
        self.running = True
        asyncio.create_task(self.evaluation_loop())

    async def evaluation_loop(self):
        """Boucle d'évaluation périodique des alertes"""
        while self.running:
            try:
                await self.evaluate_all_rules()
                await asyncio.sleep(self.evaluation_interval)
            except Exception as e:
                logger.error(f"Erreur dans la boucle d'évaluation: {e}")
                await asyncio.sleep(5)  # Pause courte en cas d'erreur

    async def evaluate_all_rules(self):
        """Évaluation de toutes les règles d'alerte"""
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            try:
                await self.evaluate_rule(rule)
            except Exception as e:
                logger.error(f"Erreur évaluation règle {rule_name}: {e}")

    async def evaluate_rule(self, rule: AlertRule):
        """Évaluation d'une règle d'alerte spécifique"""
        # Simulation d'évaluation PromQL (à remplacer par un vrai client Prometheus)
        is_firing = await self.simulate_promql_evaluation(rule.query)
        
        alert_id = f"alert_{rule.name}_{int(time.time())}"
        
        if is_firing:
            # Vérifier si l'alerte existe déjà
            existing_alert = None
            for alert in self.active_alerts.values():
                if alert.rule_name == rule.name and alert.status == AlertStatus.FIRING:
                    existing_alert = alert
                    break
            
            if not existing_alert:
                # Nouvelle alerte
                alert = Alert(
                    id=alert_id,
                    rule_name=rule.name,
                    status=AlertStatus.FIRING,
                    severity=rule.severity,
                    started_at=datetime.utcnow(),
                    labels=rule.labels.copy(),
                    annotations=rule.annotations.copy()
                )
                
                self.active_alerts[alert_id] = alert
                await self.send_alert_notification(alert, "firing")
                
                logger.warning(f"Alerte déclenchée: {rule.name}")
        
        else:
            # Résoudre les alertes actives pour cette règle
            alerts_to_resolve = [
                alert for alert in self.active_alerts.values()
                if alert.rule_name == rule.name and alert.status == AlertStatus.FIRING
            ]
            
            for alert in alerts_to_resolve:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.utcnow()
                await self.send_alert_notification(alert, "resolved")
                
                logger.info(f"Alerte résolue: {rule.name}")

    async def simulate_promql_evaluation(self, query: str) -> bool:
        """Simulation d'évaluation PromQL (remplacer par un vrai client)"""
        # Simulation basique - à remplacer par une vraie évaluation Prometheus
        import random
        
        # Simulation de conditions aléatoires pour la démo
        if "cpu_usage_percent" in query:
            return random.random() < 0.1  # 10% de chance d'être en alerte
        elif "incidents_total" in query:
            return random.random() < 0.05  # 5% de chance
        elif "api_request_duration" in query:
            return random.random() < 0.15  # 15% de chance
        elif "anomaly_score" in query:
            return random.random() < 0.02  # 2% de chance
        
        return False

    async def send_alert_notification(self, alert: Alert, action: str):
        """Envoi de notification d'alerte"""
        for channel_name, channel in self.notification_channels.items():
            if not channel.enabled:
                continue
            
            # Filtrage par sévérité
            if channel.severity_filter and alert.severity not in channel.severity_filter:
                continue
            
            try:
                await self.send_notification_to_channel(alert, action, channel)
            except Exception as e:
                logger.error(f"Erreur envoi notification {channel_name}: {e}")

    async def send_notification_to_channel(self, alert: Alert, action: str, channel: NotificationChannel):
        """Envoi de notification vers un canal spécifique"""
        if channel.type == "email":
            await self.send_email_notification(alert, action, channel)
        elif channel.type == "slack":
            await self.send_slack_notification(alert, action, channel)
        elif channel.type == "webhook":
            await self.send_webhook_notification(alert, action, channel)
        else:
            logger.warning(f"Type de canal non supporté: {channel.type}")

    async def send_email_notification(self, alert: Alert, action: str, channel: NotificationChannel):
        """Envoi de notification par email"""
        try:
            config = channel.config
            
            # Préparation du message
            subject = f"[{alert.severity.value.upper()}] Alert {action}: {alert.rule_name}"
            
            body = f"""
            Alert: {alert.rule_name}
            Status: {action}
            Severity: {alert.severity.value}
            Started: {alert.started_at}
            {"Resolved: " + str(alert.resolved_at) if alert.resolved_at else ""}
            
            Labels: {json.dumps(alert.labels, indent=2)}
            Annotations: {json.dumps(alert.annotations, indent=2)}
            """
            
            # Envoi du mail (simulation)
            logger.info(f"Email envoyé: {subject} vers {config.get('recipients', [])}")
            
        except Exception as e:
            logger.error(f"Erreur envoi email: {e}")

    async def send_slack_notification(self, alert: Alert, action: str, channel: NotificationChannel):
        """Envoi de notification Slack"""
        try:
            config = channel.config
            webhook_url = config.get('webhook_url')
            
            if not webhook_url:
                logger.warning("URL webhook Slack manquante")
                return
            
            # Préparation du message Slack
            color = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning", 
                AlertSeverity.CRITICAL: "danger",
                AlertSeverity.EMERGENCY: "danger"
            }.get(alert.severity, "warning")
            
            payload = {
                "channel": config.get('channel', '#alerts'),
                "username": "AlertManager",
                "icon_emoji": ":warning:",
                "attachments": [{
                    "color": color,
                    "title": f"Alert {action}: {alert.rule_name}",
                    "fields": [
                        {"title": "Severity", "value": alert.severity.value, "short": True},
                        {"title": "Status", "value": action, "short": True},
                        {"title": "Started", "value": str(alert.started_at), "short": True}
                    ]
                }]
            }
            
            # Envoi via webhook (simulation)
            logger.info(f"Slack notification envoyée: {alert.rule_name}")
            
        except Exception as e:
            logger.error(f"Erreur envoi Slack: {e}")

    async def send_webhook_notification(self, alert: Alert, action: str, channel: NotificationChannel):
        """Envoi de notification webhook"""
        try:
            config = channel.config
            url = config.get('url')
            
            if not url:
                logger.warning("URL webhook manquante")
                return
            
            # Préparation du payload
            payload = {
                "alert_id": alert.id,
                "rule_name": alert.rule_name,
                "action": action,
                "severity": alert.severity.value,
                "started_at": alert.started_at.isoformat(),
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
                "labels": alert.labels,
                "annotations": alert.annotations
            }
            
            # Envoi du webhook (simulation)
            logger.info(f"Webhook envoyé: {alert.rule_name} vers {url}")
            
        except Exception as e:
            logger.error(f"Erreur envoi webhook: {e}")

    async def add_alert_rule(self, rule: AlertRule):
        """Ajout d'une nouvelle règle d'alerte"""
        self.alert_rules[rule.name] = rule
        logger.info(f"Règle d'alerte ajoutée: {rule.name}")

    async def remove_alert_rule(self, rule_name: str):
        """Suppression d'une règle d'alerte"""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.info(f"Règle d'alerte supprimée: {rule_name}")

    async def get_active_alerts(self, tenant_id: Optional[str] = None) -> List[Alert]:
        """Récupération des alertes actives"""
        active_alerts = [
            alert for alert in self.active_alerts.values()
            if alert.status == AlertStatus.FIRING
        ]
        
        if tenant_id:
            active_alerts = [
                alert for alert in active_alerts
                if alert.tenant_id == tenant_id
            ]
        
        return active_alerts

    async def shutdown(self):
        """Arrêt propre du gestionnaire d'alertes"""
        self.running = False
        
        if self.redis_client:
            await self.redis_client.close()
        
        if self.db_engine:
            await self.db_engine.dispose()
        
        logger.info("Gestionnaire d'alertes arrêté")

# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================

async def main():
    """Exemple d'utilisation du système de monitoring"""
    
    config = {
        'namespace': 'incidents',
        'enable_system_metrics': True,
        'enable_ml_metrics': True,
        'redis_url': 'redis://localhost:6379/0',
        'database_url': 'postgresql+asyncpg://user:pass@localhost/monitoring',
        'evaluation_interval': 10,  # 10 secondes pour la démo
        'smtp_server': 'smtp.company.com',
        'slack_webhook_url': 'https://hooks.slack.com/services/...'
    }
    
    # Initialisation du registry de métriques
    metrics_registry = EnterpriseMetricsRegistry(config)
    await metrics_registry.initialize()
    
    # Initialisation du gestionnaire d'alertes
    alert_manager = EnterpriseAlertManager(config, metrics_registry)
    await alert_manager.initialize()
    
    try:
        # Simulation d'utilisation
        print("=== Démonstration du système de monitoring ===")
        
        # Enregistrement de quelques métriques
        metrics_registry.record_incident("tenant_123", "critical", "database", "monitoring")
        metrics_registry.record_api_request("GET", "/api/incidents", 200, "tenant_123", 0.5)
        metrics_registry.record_ml_prediction("anomaly_detector", "v1.0", "tenant_123", "normal", 0.15)
        
        # Export des métriques
        metrics_output, content_type = metrics_registry.export_metrics()
        print(f"Métriques exportées ({len(metrics_output)} caractères)")
        
        # Affichage des alertes actives
        active_alerts = await alert_manager.get_active_alerts()
        print(f"Alertes actives: {len(active_alerts)}")
        
        # Attendre pour voir l'évaluation des alertes
        print("Attente de l'évaluation des alertes...")
        await asyncio.sleep(15)
        
        # Vérification des nouvelles alertes
        active_alerts = await alert_manager.get_active_alerts()
        print(f"Alertes actives après évaluation: {len(active_alerts)}")
        
        for alert in active_alerts:
            print(f"- {alert.rule_name}: {alert.severity.value} (depuis {alert.started_at})")
        
    finally:
        await alert_manager.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
