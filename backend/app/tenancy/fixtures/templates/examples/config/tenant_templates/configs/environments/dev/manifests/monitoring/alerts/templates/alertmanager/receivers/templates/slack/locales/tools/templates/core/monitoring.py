"""
Advanced Monitoring System
==========================

Système de monitoring avancé pour le multi-tenancy avec métriques en temps réel,
alertes automatiques et tableau de bord intégré.

Auteur: Fahed Mlaiel
"""

import asyncio
import logging
import time
import psutil
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import aioredis
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import weakref

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types de métriques"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Niveaux de sévérité des alertes"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricDefinition:
    """Définition d'une métrique"""
    name: str
    type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # Pour les histogrammes


@dataclass
class AlertRule:
    """Règle d'alerte"""
    name: str
    metric: str
    condition: str  # ex: "> 0.8", "< 100"
    severity: AlertSeverity
    duration: int = 300  # secondes avant déclenchement
    cooldown: int = 600  # secondes entre alertes
    message: str = ""
    enabled: bool = True


@dataclass
class MonitoringEvent:
    """Événement de monitoring"""
    timestamp: datetime
    tenant_id: Optional[str]
    metric_name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MonitoringSystem:
    """
    Système de monitoring avancé pour le multi-tenancy
    
    Fonctionnalités:
    - Métriques Prometheus
    - Alertes en temps réel
    - Monitoring des ressources système
    - Suivi des performances par tenant
    - Health checks automatiques
    - Exports vers différents backends
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le système de monitoring
        
        Args:
            config: Configuration du monitoring
        """
        self.config = config
        self.is_initialized = False
        self.is_running = False
        
        # Métriques Prometheus
        self.prometheus_metrics: Dict[str, Any] = {}
        self.custom_metrics: Dict[str, MetricDefinition] = {}
        
        # Règles d'alertes
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alert_states: Dict[str, Dict[str, Any]] = {}
        
        # Cache Redis pour les métriques
        self.redis_client: Optional[aioredis.Redis] = None
        
        # Callbacks d'alertes
        self.alert_callbacks: List[Callable] = []
        
        # Historique des événements
        self.event_history: List[MonitoringEvent] = []
        self.max_history_size = 10000
        
        # Références faibles vers les composants monitorés
        self.monitored_components = weakref.WeakSet()
        
        logger.info("MonitoringSystem initialisé")
    
    async def initialize(self) -> None:
        """Initialise le système de monitoring"""
        if self.is_initialized:
            return
        
        logger.info("Initialisation du MonitoringSystem...")
        
        try:
            # Configuration Redis pour le cache des métriques
            await self._init_redis()
            
            # Initialisation des métriques Prometheus
            await self._init_prometheus_metrics()
            
            # Démarrage du serveur de métriques
            if self.config.get("export_prometheus", True):
                await self._start_prometheus_server()
            
            # Chargement des règles d'alertes
            await self._load_alert_rules()
            
            # Démarrage des tâches de monitoring
            await self._start_monitoring_tasks()
            
            self.is_initialized = True
            self.is_running = True
            
            logger.info("MonitoringSystem initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du MonitoringSystem: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Arrêt propre du système de monitoring"""
        if not self.is_initialized:
            return
        
        logger.info("Arrêt du MonitoringSystem...")
        
        try:
            self.is_running = False
            
            # Fermeture de la connexion Redis
            if self.redis_client:
                await self.redis_client.close()
            
            self.is_initialized = False
            logger.info("MonitoringSystem arrêté avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'arrêt: {e}")
    
    async def _init_redis(self) -> None:
        """Initialise la connexion Redis"""
        try:
            self.redis_client = aioredis.from_url(
                f"redis://{self.config.get('redis_host', 'localhost')}:"
                f"{self.config.get('redis_port', 6379)}/"
                f"{self.config.get('redis_db', 1)}",
                password=self.config.get('redis_password'),
                decode_responses=True
            )
            
            # Test de connexion
            await self.redis_client.ping()
            logger.info("Connexion Redis établie pour le monitoring")
            
        except Exception as e:
            logger.warning(f"Impossible de se connecter à Redis: {e}")
            self.redis_client = None
    
    async def _init_prometheus_metrics(self) -> None:
        """Initialise les métriques Prometheus"""
        # Métriques système de base
        self.prometheus_metrics.update({
            'tenant_requests_total': Counter(
                'tenant_requests_total',
                'Total requests per tenant',
                ['tenant_id', 'method', 'endpoint']
            ),
            'tenant_request_duration': Histogram(
                'tenant_request_duration_seconds',
                'Request duration per tenant',
                ['tenant_id', 'method', 'endpoint'],
                buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
            ),
            'tenant_active_users': Gauge(
                'tenant_active_users',
                'Active users per tenant',
                ['tenant_id']
            ),
            'tenant_storage_usage': Gauge(
                'tenant_storage_usage_bytes',
                'Storage usage per tenant',
                ['tenant_id']
            ),
            'tenant_db_connections': Gauge(
                'tenant_db_connections_active',
                'Active database connections per tenant',
                ['tenant_id']
            ),
            'system_cpu_usage': Gauge(
                'system_cpu_usage_percent',
                'System CPU usage percentage'
            ),
            'system_memory_usage': Gauge(
                'system_memory_usage_percent',
                'System memory usage percentage'
            ),
            'system_disk_usage': Gauge(
                'system_disk_usage_percent',
                'System disk usage percentage'
            )
        })
        
        # Métriques personnalisées définies par configuration
        custom_metrics_config = self.config.get("custom_metrics", [])
        for metric_config in custom_metrics_config:
            await self._register_custom_metric(metric_config)
    
    async def _register_custom_metric(self, metric_config: Dict[str, Any]) -> None:
        """Enregistre une métrique personnalisée"""
        try:
            metric_def = MetricDefinition(
                name=metric_config["name"],
                type=MetricType(metric_config["type"]),
                description=metric_config["description"],
                labels=metric_config.get("labels", []),
                buckets=metric_config.get("buckets")
            )
            
            if metric_def.type == MetricType.COUNTER:
                prometheus_metric = Counter(
                    metric_def.name,
                    metric_def.description,
                    metric_def.labels
                )
            elif metric_def.type == MetricType.GAUGE:
                prometheus_metric = Gauge(
                    metric_def.name,
                    metric_def.description,
                    metric_def.labels
                )
            elif metric_def.type == MetricType.HISTOGRAM:
                prometheus_metric = Histogram(
                    metric_def.name,
                    metric_def.description,
                    metric_def.labels,
                    buckets=metric_def.buckets
                )
            else:
                logger.warning(f"Type de métrique non supporté: {metric_def.type}")
                return
            
            self.custom_metrics[metric_def.name] = metric_def
            self.prometheus_metrics[metric_def.name] = prometheus_metric
            
            logger.info(f"Métrique personnalisée enregistrée: {metric_def.name}")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement de la métrique {metric_config}: {e}")
    
    async def _start_prometheus_server(self) -> None:
        """Démarre le serveur de métriques Prometheus"""
        try:
            port = self.config.get("metrics_port", 9090)
            start_http_server(port)
            logger.info(f"Serveur de métriques Prometheus démarré sur le port {port}")
            
        except Exception as e:
            logger.error(f"Erreur lors du démarrage du serveur Prometheus: {e}")
    
    async def _load_alert_rules(self) -> None:
        """Charge les règles d'alertes"""
        default_rules = [
            AlertRule(
                name="high_cpu_usage",
                metric="system_cpu_usage",
                condition="> 80",
                severity=AlertSeverity.WARNING,
                message="Utilisation CPU élevée: {value}%"
            ),
            AlertRule(
                name="critical_cpu_usage",
                metric="system_cpu_usage",
                condition="> 95",
                severity=AlertSeverity.CRITICAL,
                message="Utilisation CPU critique: {value}%"
            ),
            AlertRule(
                name="high_memory_usage",
                metric="system_memory_usage",
                condition="> 85",
                severity=AlertSeverity.WARNING,
                message="Utilisation mémoire élevée: {value}%"
            ),
            AlertRule(
                name="tenant_quota_exceeded",
                metric="tenant_storage_usage",
                condition="> tenant_storage_limit",
                severity=AlertSeverity.ERROR,
                message="Quota de stockage dépassé pour le tenant {tenant_id}"
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.name] = rule
            self.alert_states[rule.name] = {
                "last_triggered": None,
                "last_alert_sent": None,
                "consecutive_violations": 0
            }
        
        # Chargement des règles personnalisées depuis la configuration
        custom_rules = self.config.get("alert_rules", [])
        for rule_config in custom_rules:
            rule = AlertRule(**rule_config)
            self.alert_rules[rule.name] = rule
            self.alert_states[rule.name] = {
                "last_triggered": None,
                "last_alert_sent": None,
                "consecutive_violations": 0
            }
        
        logger.info(f"Chargées {len(self.alert_rules)} règles d'alertes")
    
    async def _start_monitoring_tasks(self) -> None:
        """Démarre les tâches de monitoring en arrière-plan"""
        # Monitoring des ressources système
        asyncio.create_task(self._monitor_system_resources())
        
        # Évaluation des règles d'alertes
        asyncio.create_task(self._evaluate_alert_rules())
        
        # Nettoyage de l'historique
        asyncio.create_task(self._cleanup_history())
        
        # Export des métriques vers Redis
        if self.redis_client:
            asyncio.create_task(self._export_metrics_to_redis())
        
        logger.info("Tâches de monitoring démarrées")
    
    async def _monitor_system_resources(self) -> None:
        """Surveille les ressources système"""
        while self.is_running:
            try:
                # CPU
                cpu_percent = psutil.cpu_percent(interval=1)
                self.prometheus_metrics['system_cpu_usage'].set(cpu_percent)
                await self._record_event("system_cpu_usage", cpu_percent)
                
                # Mémoire
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                self.prometheus_metrics['system_memory_usage'].set(memory_percent)
                await self._record_event("system_memory_usage", memory_percent)
                
                # Disque
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                self.prometheus_metrics['system_disk_usage'].set(disk_percent)
                await self._record_event("system_disk_usage", disk_percent)
                
                await asyncio.sleep(self.config.get("system_monitor_interval", 30))
                
            except Exception as e:
                logger.error(f"Erreur dans le monitoring système: {e}")
                await asyncio.sleep(60)
    
    async def _evaluate_alert_rules(self) -> None:
        """Évalue les règles d'alertes"""
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                
                for rule_name, rule in self.alert_rules.items():
                    if not rule.enabled:
                        continue
                    
                    # Récupération de la valeur actuelle de la métrique
                    metric_value = await self._get_current_metric_value(rule.metric)
                    
                    if metric_value is not None:
                        # Évaluation de la condition
                        violation = await self._evaluate_condition(rule.condition, metric_value)
                        
                        if violation:
                            await self._handle_rule_violation(rule, metric_value, current_time)
                        else:
                            await self._handle_rule_recovery(rule, current_time)
                
                await asyncio.sleep(self.config.get("alert_evaluation_interval", 60))
                
            except Exception as e:
                logger.error(f"Erreur dans l'évaluation des alertes: {e}")
                await asyncio.sleep(120)
    
    async def _get_current_metric_value(self, metric_name: str) -> Optional[float]:
        """Récupère la valeur actuelle d'une métrique"""
        try:
            if metric_name in self.prometheus_metrics:
                metric = self.prometheus_metrics[metric_name]
                
                # Pour les Gauges, on peut récupérer la valeur directement
                if hasattr(metric, '_value'):
                    return float(metric._value.get())
                
                # Pour les autres types, on cherche dans l'historique récent
                recent_events = [
                    e for e in self.event_history[-100:]
                    if e.metric_name == metric_name
                ]
                
                if recent_events:
                    return recent_events[-1].value
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la métrique {metric_name}: {e}")
            return None
    
    async def _evaluate_condition(self, condition: str, value: float) -> bool:
        """Évalue une condition d'alerte"""
        try:
            # Nettoyage et validation de la condition
            condition = condition.strip()
            
            # Conditions simples supportées: >, <, >=, <=, ==, !=
            if condition.startswith('>'):
                if condition.startswith('>='):
                    threshold = float(condition[2:].strip())
                    return value >= threshold
                else:
                    threshold = float(condition[1:].strip())
                    return value > threshold
            elif condition.startswith('<'):
                if condition.startswith('<='):
                    threshold = float(condition[2:].strip())
                    return value <= threshold
                else:
                    threshold = float(condition[1:].strip())
                    return value < threshold
            elif condition.startswith('=='):
                threshold = float(condition[2:].strip())
                return value == threshold
            elif condition.startswith('!='):
                threshold = float(condition[2:].strip())
                return value != threshold
            
            return False
            
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation de la condition '{condition}': {e}")
            return False
    
    async def _handle_rule_violation(self, rule: AlertRule, value: float, current_time: datetime) -> None:
        """Gère une violation de règle d'alerte"""
        state = self.alert_states[rule.name]
        
        # Première violation ou nouvelle violation après récupération
        if state["last_triggered"] is None:
            state["last_triggered"] = current_time
            state["consecutive_violations"] = 1
        else:
            state["consecutive_violations"] += 1
        
        # Vérification de la durée avant déclenchement
        violation_duration = (current_time - state["last_triggered"]).total_seconds()
        
        if violation_duration >= rule.duration:
            # Vérification du cooldown
            if (state["last_alert_sent"] is None or 
                (current_time - state["last_alert_sent"]).total_seconds() >= rule.cooldown):
                
                await self._send_alert(rule, value, current_time)
                state["last_alert_sent"] = current_time
    
    async def _handle_rule_recovery(self, rule: AlertRule, current_time: datetime) -> None:
        """Gère la récupération d'une règle d'alerte"""
        state = self.alert_states[rule.name]
        
        if state["last_triggered"] is not None:
            # Réinitialisation de l'état
            state["last_triggered"] = None
            state["consecutive_violations"] = 0
            
            # Notification de récupération si une alerte avait été envoyée
            if state["last_alert_sent"] is not None:
                await self._send_recovery_notification(rule, current_time)
    
    async def _send_alert(self, rule: AlertRule, value: float, timestamp: datetime) -> None:
        """Envoie une alerte"""
        alert_data = {
            "rule_name": rule.name,
            "metric": rule.metric,
            "value": value,
            "condition": rule.condition,
            "severity": rule.severity.value,
            "message": rule.message.format(value=value),
            "timestamp": timestamp.isoformat()
        }
        
        logger.warning(f"ALERTE {rule.severity.value.upper()}: {alert_data['message']}")
        
        # Appel des callbacks d'alertes
        for callback in self.alert_callbacks:
            try:
                await callback(alert_data)
            except Exception as e:
                logger.error(f"Erreur dans le callback d'alerte: {e}")
        
        # Enregistrement dans l'historique
        await self._record_event(
            f"alert_{rule.name}",
            value,
            metadata=alert_data
        )
    
    async def _send_recovery_notification(self, rule: AlertRule, timestamp: datetime) -> None:
        """Envoie une notification de récupération"""
        recovery_data = {
            "rule_name": rule.name,
            "metric": rule.metric,
            "message": f"Récupération de l'alerte: {rule.name}",
            "timestamp": timestamp.isoformat()
        }
        
        logger.info(f"RÉCUPÉRATION: {recovery_data['message']}")
        
        # Appel des callbacks avec le type recovery
        for callback in self.alert_callbacks:
            try:
                await callback(recovery_data, alert_type="recovery")
            except Exception as e:
                logger.error(f"Erreur dans le callback de récupération: {e}")
    
    async def _record_event(
        self,
        metric_name: str,
        value: float,
        tenant_id: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Enregistre un événement de monitoring"""
        event = MonitoringEvent(
            timestamp=datetime.utcnow(),
            tenant_id=tenant_id,
            metric_name=metric_name,
            value=value,
            labels=labels or {},
            metadata=metadata or {}
        )
        
        # Ajout à l'historique
        self.event_history.append(event)
        
        # Limitation de la taille de l'historique
        if len(self.event_history) > self.max_history_size:
            self.event_history = self.event_history[-self.max_history_size:]
    
    async def _cleanup_history(self) -> None:
        """Nettoie l'historique des événements"""
        while self.is_running:
            try:
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                
                # Suppression des événements anciens
                self.event_history = [
                    e for e in self.event_history
                    if e.timestamp > cutoff_time
                ]
                
                await asyncio.sleep(3600)  # Nettoyage toutes les heures
                
            except Exception as e:
                logger.error(f"Erreur lors du nettoyage de l'historique: {e}")
                await asyncio.sleep(1800)
    
    async def _export_metrics_to_redis(self) -> None:
        """Exporte les métriques vers Redis"""
        while self.is_running and self.redis_client:
            try:
                current_time = datetime.utcnow()
                
                # Export des métriques récentes
                recent_events = [
                    e for e in self.event_history
                    if (current_time - e.timestamp).total_seconds() < 300
                ]
                
                for event in recent_events:
                    key = f"metrics:{event.metric_name}:{event.tenant_id or 'system'}"
                    data = {
                        "value": event.value,
                        "timestamp": event.timestamp.isoformat(),
                        "labels": json.dumps(event.labels),
                        "metadata": json.dumps(event.metadata)
                    }
                    
                    await self.redis_client.hset(key, mapping=data)
                    await self.redis_client.expire(key, 3600)  # TTL 1 heure
                
                await asyncio.sleep(60)  # Export toutes les minutes
                
            except Exception as e:
                logger.error(f"Erreur lors de l'export vers Redis: {e}")
                await asyncio.sleep(300)
    
    # API publique
    
    async def record_tenant_metric(
        self,
        tenant_id: str,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Enregistre une métrique pour un tenant
        
        Args:
            tenant_id: ID du tenant
            metric_name: Nom de la métrique
            value: Valeur de la métrique
            labels: Labels optionnels
        """
        if not self.is_initialized:
            return
        
        # Mise à jour de la métrique Prometheus si elle existe
        if metric_name in self.prometheus_metrics:
            metric = self.prometheus_metrics[metric_name]
            
            if hasattr(metric, 'labels'):
                # Métrique avec labels
                label_values = [tenant_id] + [labels.get(label, "") for label in metric._labelnames[1:]]
                labeled_metric = metric.labels(*label_values)
                
                if hasattr(labeled_metric, 'set'):
                    labeled_metric.set(value)
                elif hasattr(labeled_metric, 'inc'):
                    labeled_metric.inc(value)
            else:
                # Métrique simple
                if hasattr(metric, 'set'):
                    metric.set(value)
                elif hasattr(metric, 'inc'):
                    metric.inc(value)
        
        # Enregistrement de l'événement
        await self._record_event(metric_name, value, tenant_id, labels)
    
    async def increment_tenant_counter(
        self,
        tenant_id: str,
        counter_name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Incrémente un compteur pour un tenant
        
        Args:
            tenant_id: ID du tenant
            counter_name: Nom du compteur
            value: Valeur d'incrémentation
            labels: Labels optionnels
        """
        await self.record_tenant_metric(tenant_id, counter_name, value, labels)
    
    async def observe_tenant_histogram(
        self,
        tenant_id: str,
        histogram_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Observe une valeur dans un histogramme pour un tenant
        
        Args:
            tenant_id: ID du tenant
            histogram_name: Nom de l'histogramme
            value: Valeur observée
            labels: Labels optionnels
        """
        if histogram_name in self.prometheus_metrics:
            histogram = self.prometheus_metrics[histogram_name]
            
            if hasattr(histogram, 'labels'):
                label_values = [tenant_id] + [labels.get(label, "") for label in histogram._labelnames[1:]]
                histogram.labels(*label_values).observe(value)
            else:
                histogram.observe(value)
        
        await self._record_event(histogram_name, value, tenant_id, labels)
    
    def add_alert_callback(self, callback: Callable) -> None:
        """
        Ajoute un callback pour les alertes
        
        Args:
            callback: Fonction appelée lors des alertes
        """
        self.alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable) -> None:
        """
        Supprime un callback d'alerte
        
        Args:
            callback: Fonction à supprimer
        """
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
    
    async def get_tenant_metrics(
        self,
        tenant_id: str,
        metric_names: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[MonitoringEvent]:
        """
        Récupère les métriques d'un tenant
        
        Args:
            tenant_id: ID du tenant
            metric_names: Noms des métriques (toutes si None)
            start_time: Début de la période
            end_time: Fin de la période
            
        Returns:
            Liste des événements de monitoring
        """
        events = [
            e for e in self.event_history
            if e.tenant_id == tenant_id
        ]
        
        if metric_names:
            events = [e for e in events if e.metric_name in metric_names]
        
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        return sorted(events, key=lambda x: x.timestamp)
    
    async def get_system_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[MonitoringEvent]:
        """
        Récupère les métriques système
        
        Args:
            start_time: Début de la période
            end_time: Fin de la période
            
        Returns:
            Liste des événements de monitoring
        """
        events = [
            e for e in self.event_history
            if e.tenant_id is None
        ]
        
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        return sorted(events, key=lambda x: x.timestamp)
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Vérification de l'état de santé du monitoring
        
        Returns:
            Rapport d'état
        """
        try:
            return {
                "status": "healthy",
                "is_running": self.is_running,
                "prometheus_metrics_count": len(self.prometheus_metrics),
                "alert_rules_count": len(self.alert_rules),
                "events_in_history": len(self.event_history),
                "redis_connected": self.redis_client is not None,
                "monitored_components": len(self.monitored_components)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "is_running": self.is_running
            }
