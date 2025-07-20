"""
Advanced Alert Manager
======================

Gestionnaire d'alertes avancé avec support multi-canal (Slack, Email, Webhook),
escalade automatique, et intégration avec les systèmes de monitoring.

Auteur: Fahed Mlaiel
"""

import asyncio
import logging
import json
import aiohttp
import smtplib
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiofiles
import jinja2

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Niveaux de sévérité des alertes"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Statuts d'alerte"""
    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"
    SILENCED = "silenced"


class NotificationChannel(Enum):
    """Canaux de notification"""
    SLACK = "slack"
    EMAIL = "email"
    WEBHOOK = "webhook"
    SMS = "sms"
    TEAMS = "teams"


@dataclass
class AlertDefinition:
    """Définition d'une alerte"""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    condition: str
    metric: str
    threshold: float
    duration: int = 300  # secondes
    cooldown: int = 600  # secondes
    enabled: bool = True
    channels: List[NotificationChannel] = field(default_factory=list)
    escalation_rules: List[Dict[str, Any]] = field(default_factory=list)
    template_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Instance d'alerte"""
    id: str
    definition: AlertDefinition
    status: AlertStatus
    severity: AlertSeverity
    message: str
    value: float
    tenant_id: Optional[str] = None
    fired_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    last_notification: Optional[datetime] = None
    notification_count: int = 0
    escalation_level: int = 0
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)


@dataclass
class NotificationConfig:
    """Configuration des notifications"""
    channel: NotificationChannel
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    template: Optional[str] = None
    retry_count: int = 3
    retry_delay: int = 30
    timeout: int = 30


class AlertManager:
    """
    Gestionnaire d'alertes avancé
    
    Fonctionnalités:
    - Gestion des alertes multi-niveaux
    - Notifications multi-canaux
    - Escalade automatique
    - Templates personnalisables
    - Silencing et grouping
    - Historique et métriques
    - Intégration avec monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le gestionnaire d'alertes
        
        Args:
            config: Configuration du gestionnaire d'alertes
        """
        self.config = config
        self.is_initialized = False
        self.is_running = False
        
        # Définitions et instances d'alertes
        self.alert_definitions: Dict[str, AlertDefinition] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.resolved_alerts: List[Alert] = []
        
        # Configuration des notifications
        self.notification_configs: Dict[NotificationChannel, NotificationConfig] = {}
        
        # Templates Jinja2
        self.template_env: Optional[jinja2.Environment] = None
        
        # Silencing et grouping
        self.silenced_alerts: Dict[str, datetime] = {}
        self.alert_groups: Dict[str, List[str]] = {}
        
        # Métriques
        self.metrics = {
            "alerts_fired": 0,
            "alerts_resolved": 0,
            "notifications_sent": 0,
            "notifications_failed": 0
        }
        
        # Callbacks
        self.alert_callbacks: List[Callable] = []
        
        logger.info("AlertManager initialisé")
    
    async def initialize(self) -> None:
        """Initialise le gestionnaire d'alertes"""
        if self.is_initialized:
            return
        
        logger.info("Initialisation de l'AlertManager...")
        
        try:
            # Configuration des templates
            await self._init_templates()
            
            # Configuration des canaux de notification
            await self._init_notification_channels()
            
            # Chargement des définitions d'alertes
            await self._load_alert_definitions()
            
            # Démarrage des tâches de traitement
            await self._start_alert_processing_tasks()
            
            self.is_initialized = True
            self.is_running = True
            
            logger.info("AlertManager initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de l'AlertManager: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Arrêt propre du gestionnaire d'alertes"""
        if not self.is_initialized:
            return
        
        logger.info("Arrêt de l'AlertManager...")
        
        try:
            self.is_running = False
            
            # Résolution de toutes les alertes actives
            for alert in self.active_alerts.values():
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.utcnow()
            
            self.is_initialized = False
            logger.info("AlertManager arrêté avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'arrêt: {e}")
    
    async def _init_templates(self) -> None:
        """Initialise le système de templates"""
        template_dir = self.config.get("template_dir", "templates/alerts")
        
        # Configuration Jinja2
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Ajout de filtres personnalisés
        self.template_env.filters['datetime'] = self._format_datetime
        self.template_env.filters['severity_color'] = self._get_severity_color
        self.template_env.filters['duration'] = self._format_duration
        
        logger.info(f"Templates initialisés depuis: {template_dir}")
    
    def _format_datetime(self, dt: datetime) -> str:
        """Formate une datetime pour les templates"""
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    
    def _get_severity_color(self, severity: str) -> str:
        """Retourne la couleur associée à une sévérité"""
        colors = {
            "info": "#36a64f",      # Vert
            "warning": "#ff9900",   # Orange
            "error": "#ff0000",     # Rouge
            "critical": "#8b0000"   # Rouge foncé
        }
        return colors.get(severity.lower(), "#808080")
    
    def _format_duration(self, seconds: int) -> str:
        """Formate une durée en secondes"""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds // 60}m {seconds % 60}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"
    
    async def _init_notification_channels(self) -> None:
        """Initialise les canaux de notification"""
        channels_config = self.config.get("notification_channels", {})
        
        # Slack
        if "slack" in channels_config:
            slack_config = channels_config["slack"]
            self.notification_configs[NotificationChannel.SLACK] = NotificationConfig(
                channel=NotificationChannel.SLACK,
                enabled=slack_config.get("enabled", True),
                config={
                    "token": slack_config.get("token"),
                    "channel": slack_config.get("channel", "#alerts"),
                    "username": slack_config.get("username", "AlertManager"),
                    "icon_emoji": slack_config.get("icon_emoji", ":warning:")
                },
                template=slack_config.get("template", "slack_alert.j2")
            )
        
        # Email
        if "email" in channels_config:
            email_config = channels_config["email"]
            self.notification_configs[NotificationChannel.EMAIL] = NotificationConfig(
                channel=NotificationChannel.EMAIL,
                enabled=email_config.get("enabled", True),
                config={
                    "smtp_host": email_config.get("smtp_host", "localhost"),
                    "smtp_port": email_config.get("smtp_port", 587),
                    "smtp_user": email_config.get("smtp_user"),
                    "smtp_password": email_config.get("smtp_password"),
                    "from_email": email_config.get("from_email", "alerts@example.com"),
                    "to_emails": email_config.get("to_emails", []),
                    "use_tls": email_config.get("use_tls", True)
                },
                template=email_config.get("template", "email_alert.j2")
            )
        
        # Webhook
        if "webhook" in channels_config:
            webhook_config = channels_config["webhook"]
            self.notification_configs[NotificationChannel.WEBHOOK] = NotificationConfig(
                channel=NotificationChannel.WEBHOOK,
                enabled=webhook_config.get("enabled", True),
                config={
                    "url": webhook_config.get("url"),
                    "method": webhook_config.get("method", "POST"),
                    "headers": webhook_config.get("headers", {}),
                    "auth": webhook_config.get("auth")
                },
                template=webhook_config.get("template", "webhook_alert.j2")
            )
        
        logger.info(f"Configurés {len(self.notification_configs)} canaux de notification")
    
    async def _load_alert_definitions(self) -> None:
        """Charge les définitions d'alertes"""
        # Définitions par défaut
        default_definitions = [
            AlertDefinition(
                id="high_error_rate",
                name="Taux d'erreur élevé",
                description="Le taux d'erreur dépasse le seuil acceptable",
                severity=AlertSeverity.ERROR,
                condition="error_rate > 0.05",
                metric="error_rate",
                threshold=0.05,
                channels=[NotificationChannel.SLACK, NotificationChannel.EMAIL]
            ),
            AlertDefinition(
                id="system_overload",
                name="Surcharge système",
                description="Le système est en surcharge",
                severity=AlertSeverity.CRITICAL,
                condition="cpu_usage > 0.95 AND memory_usage > 0.90",
                metric="system_load",
                threshold=0.95,
                channels=[NotificationChannel.SLACK, NotificationChannel.EMAIL, NotificationChannel.WEBHOOK]
            ),
            AlertDefinition(
                id="tenant_quota_exceeded",
                name="Quota tenant dépassé",
                description="Un tenant a dépassé son quota",
                severity=AlertSeverity.WARNING,
                condition="tenant_usage > tenant_quota",
                metric="tenant_usage_ratio",
                threshold=1.0,
                channels=[NotificationChannel.SLACK]
            )
        ]
        
        for definition in default_definitions:
            self.alert_definitions[definition.id] = definition
        
        # Chargement des définitions personnalisées
        custom_definitions = self.config.get("alert_definitions", [])
        for def_config in custom_definitions:
            definition = AlertDefinition(**def_config)
            self.alert_definitions[definition.id] = definition
        
        logger.info(f"Chargées {len(self.alert_definitions)} définitions d'alertes")
    
    async def _start_alert_processing_tasks(self) -> None:
        """Démarre les tâches de traitement des alertes"""
        # Processus de résolution automatique
        asyncio.create_task(self._auto_resolve_alerts())
        
        # Nettoyage des alertes résolues
        asyncio.create_task(self._cleanup_resolved_alerts())
        
        # Escalade automatique
        asyncio.create_task(self._handle_escalation())
        
        # Gestion du silencing
        asyncio.create_task(self._manage_silencing())
        
        logger.info("Tâches de traitement des alertes démarrées")
    
    async def _auto_resolve_alerts(self) -> None:
        """Résout automatiquement les alertes qui ne sont plus valides"""
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                
                for alert_id, alert in list(self.active_alerts.items()):
                    # Vérification si l'alerte doit être résolue automatiquement
                    if await self._should_resolve_alert(alert, current_time):
                        await self.resolve_alert(alert_id, "Auto-résolution")
                
                await asyncio.sleep(60)  # Vérification toutes les minutes
                
            except Exception as e:
                logger.error(f"Erreur dans la résolution automatique: {e}")
                await asyncio.sleep(120)
    
    async def _should_resolve_alert(self, alert: Alert, current_time: datetime) -> bool:
        """Détermine si une alerte doit être résolue automatiquement"""
        # Auto-résolution après un certain temps sans nouvelle occurrence
        if alert.fired_at:
            time_since_fired = (current_time - alert.fired_at).total_seconds()
            auto_resolve_timeout = alert.definition.metadata.get("auto_resolve_timeout", 3600)
            
            if time_since_fired > auto_resolve_timeout:
                return True
        
        return False
    
    async def _cleanup_resolved_alerts(self) -> None:
        """Nettoie les alertes résolues anciennes"""
        while self.is_running:
            try:
                cutoff_time = datetime.utcnow() - timedelta(days=7)
                
                # Suppression des alertes résolues anciennes
                self.resolved_alerts = [
                    alert for alert in self.resolved_alerts
                    if alert.resolved_at and alert.resolved_at > cutoff_time
                ]
                
                await asyncio.sleep(3600)  # Nettoyage toutes les heures
                
            except Exception as e:
                logger.error(f"Erreur lors du nettoyage: {e}")
                await asyncio.sleep(1800)
    
    async def _handle_escalation(self) -> None:
        """Gère l'escalade des alertes"""
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                
                for alert in self.active_alerts.values():
                    if alert.definition.escalation_rules:
                        await self._check_escalation(alert, current_time)
                
                await asyncio.sleep(300)  # Vérification toutes les 5 minutes
                
            except Exception as e:
                logger.error(f"Erreur dans l'escalade: {e}")
                await asyncio.sleep(600)
    
    async def _check_escalation(self, alert: Alert, current_time: datetime) -> None:
        """Vérifie si une alerte doit être escaladée"""
        if not alert.fired_at:
            return
        
        time_since_fired = (current_time - alert.fired_at).total_seconds()
        
        for rule in alert.definition.escalation_rules:
            escalation_time = rule.get("after_seconds", 1800)  # 30 minutes par défaut
            escalation_level = rule.get("level", 1)
            
            if (time_since_fired >= escalation_time and 
                alert.escalation_level < escalation_level):
                
                await self._escalate_alert(alert, rule)
    
    async def _escalate_alert(self, alert: Alert, escalation_rule: Dict[str, Any]) -> None:
        """Escalade une alerte"""
        alert.escalation_level = escalation_rule.get("level", 1)
        alert.severity = AlertSeverity(escalation_rule.get("severity", alert.severity.value))
        
        # Ajout de canaux d'escalade
        escalation_channels = escalation_rule.get("channels", [])
        for channel_name in escalation_channels:
            try:
                channel = NotificationChannel(channel_name)
                if channel not in alert.definition.channels:
                    alert.definition.channels.append(channel)
            except ValueError:
                logger.warning(f"Canal d'escalade inconnu: {channel_name}")
        
        # Notification d'escalade
        escalation_message = f"Alerte escaladée au niveau {alert.escalation_level}: {alert.message}"
        
        await self._send_notifications(alert, escalation_message, alert_type="escalation")
        
        logger.warning(f"Alerte {alert.id} escaladée au niveau {alert.escalation_level}")
    
    async def _manage_silencing(self) -> None:
        """Gère le silencing des alertes"""
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                
                # Suppression des silences expirés
                expired_silences = [
                    alert_id for alert_id, expiry in self.silenced_alerts.items()
                    if expiry <= current_time
                ]
                
                for alert_id in expired_silences:
                    del self.silenced_alerts[alert_id]
                    logger.info(f"Silence expiré pour l'alerte: {alert_id}")
                
                await asyncio.sleep(60)  # Vérification toutes les minutes
                
            except Exception as e:
                logger.error(f"Erreur dans la gestion du silencing: {e}")
                await asyncio.sleep(120)
    
    # API publique
    
    async def fire_alert(
        self,
        definition_id: str,
        value: float,
        message: Optional[str] = None,
        tenant_id: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        annotations: Optional[Dict[str, str]] = None
    ) -> Optional[Alert]:
        """
        Déclenche une alerte
        
        Args:
            definition_id: ID de la définition d'alerte
            value: Valeur qui a déclenché l'alerte
            message: Message personnalisé
            tenant_id: ID du tenant concerné
            labels: Labels additionnels
            annotations: Annotations additionnelles
            
        Returns:
            Instance d'alerte créée ou None si erreur
        """
        if not self.is_initialized:
            logger.warning("AlertManager non initialisé")
            return None
        
        definition = self.alert_definitions.get(definition_id)
        if not definition:
            logger.error(f"Définition d'alerte inconnue: {definition_id}")
            return None
        
        # Vérification du silencing
        if definition_id in self.silenced_alerts:
            logger.debug(f"Alerte silencée: {definition_id}")
            return None
        
        try:
            # Création de l'ID d'alerte unique
            alert_id = f"{definition_id}_{tenant_id or 'system'}_{int(datetime.utcnow().timestamp())}"
            
            # Création de l'alerte
            alert = Alert(
                id=alert_id,
                definition=definition,
                status=AlertStatus.FIRING,
                severity=definition.severity,
                message=message or f"Alerte {definition.name}: valeur {value}",
                value=value,
                tenant_id=tenant_id,
                fired_at=datetime.utcnow(),
                labels=labels or {},
                annotations=annotations or {}
            )
            
            # Ajout aux alertes actives
            self.active_alerts[alert_id] = alert
            
            # Mise à jour des métriques
            self.metrics["alerts_fired"] += 1
            
            # Notification
            await self._send_notifications(alert, alert.message)
            
            # Appel des callbacks
            for callback in self.alert_callbacks:
                try:
                    await callback(alert, "fired")
                except Exception as e:
                    logger.error(f"Erreur dans le callback d'alerte: {e}")
            
            logger.info(f"Alerte déclenchée: {alert_id} ({definition.name})")
            return alert
            
        except Exception as e:
            logger.error(f"Erreur lors du déclenchement de l'alerte {definition_id}: {e}")
            return None
    
    async def resolve_alert(self, alert_id: str, resolution_message: str = "") -> bool:
        """
        Résout une alerte
        
        Args:
            alert_id: ID de l'alerte
            resolution_message: Message de résolution
            
        Returns:
            True si résolue avec succès
        """
        if alert_id not in self.active_alerts:
            logger.warning(f"Alerte active non trouvée: {alert_id}")
            return False
        
        try:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()
            
            # Déplacement vers les alertes résolues
            self.resolved_alerts.append(alert)
            del self.active_alerts[alert_id]
            
            # Mise à jour des métriques
            self.metrics["alerts_resolved"] += 1
            
            # Notification de résolution
            resolve_message = f"Alerte résolue: {alert.message}"
            if resolution_message:
                resolve_message += f" - {resolution_message}"
            
            await self._send_notifications(alert, resolve_message, alert_type="resolved")
            
            # Appel des callbacks
            for callback in self.alert_callbacks:
                try:
                    await callback(alert, "resolved")
                except Exception as e:
                    logger.error(f"Erreur dans le callback de résolution: {e}")
            
            logger.info(f"Alerte résolue: {alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la résolution de l'alerte {alert_id}: {e}")
            return False
    
    async def silence_alert(self, alert_pattern: str, duration_seconds: int, reason: str = "") -> bool:
        """
        Silence une alerte ou un pattern d'alertes
        
        Args:
            alert_pattern: Pattern d'alerte à silencer
            duration_seconds: Durée du silence en secondes
            reason: Raison du silence
            
        Returns:
            True si silencé avec succès
        """
        try:
            expiry_time = datetime.utcnow() + timedelta(seconds=duration_seconds)
            self.silenced_alerts[alert_pattern] = expiry_time
            
            logger.info(f"Alerte silencée: {alert_pattern} jusqu'à {expiry_time} - {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du silencing: {e}")
            return False
    
    async def unsilence_alert(self, alert_pattern: str) -> bool:
        """
        Retire le silence d'une alerte
        
        Args:
            alert_pattern: Pattern d'alerte à dé-silencer
            
        Returns:
            True si dé-silencé avec succès
        """
        if alert_pattern in self.silenced_alerts:
            del self.silenced_alerts[alert_pattern]
            logger.info(f"Silence retiré pour: {alert_pattern}")
            return True
        return False
    
    async def _send_notifications(self, alert: Alert, message: str, alert_type: str = "alert") -> None:
        """Envoie les notifications pour une alerte"""
        for channel in alert.definition.channels:
            if channel in self.notification_configs:
                config = self.notification_configs[channel]
                
                if config.enabled:
                    try:
                        await self._send_notification(channel, config, alert, message, alert_type)
                        self.metrics["notifications_sent"] += 1
                        
                    except Exception as e:
                        logger.error(f"Erreur lors de l'envoi vers {channel.value}: {e}")
                        self.metrics["notifications_failed"] += 1
    
    async def _send_notification(
        self,
        channel: NotificationChannel,
        config: NotificationConfig,
        alert: Alert,
        message: str,
        alert_type: str
    ) -> None:
        """Envoie une notification sur un canal spécifique"""
        if channel == NotificationChannel.SLACK:
            await self._send_slack_notification(config, alert, message, alert_type)
        elif channel == NotificationChannel.EMAIL:
            await self._send_email_notification(config, alert, message, alert_type)
        elif channel == NotificationChannel.WEBHOOK:
            await self._send_webhook_notification(config, alert, message, alert_type)
    
    async def _send_slack_notification(
        self,
        config: NotificationConfig,
        alert: Alert,
        message: str,
        alert_type: str
    ) -> None:
        """Envoie une notification Slack"""
        token = config.config.get("token")
        if not token:
            raise ValueError("Token Slack manquant")
        
        # Préparation du payload
        payload = {
            "channel": config.config.get("channel", "#alerts"),
            "username": config.config.get("username", "AlertManager"),
            "icon_emoji": config.config.get("icon_emoji", ":warning:"),
            "attachments": [
                {
                    "color": self._get_severity_color(alert.severity.value),
                    "title": f"[{alert.severity.value.upper()}] {alert.definition.name}",
                    "text": message,
                    "fields": [
                        {
                            "title": "Tenant",
                            "value": alert.tenant_id or "Système",
                            "short": True
                        },
                        {
                            "title": "Valeur",
                            "value": str(alert.value),
                            "short": True
                        },
                        {
                            "title": "Déclenché",
                            "value": self._format_datetime(alert.fired_at) if alert.fired_at else "N/A",
                            "short": True
                        },
                        {
                            "title": "Statut",
                            "value": alert.status.value,
                            "short": True
                        }
                    ],
                    "footer": "Spotify AI Agent - AlertManager",
                    "ts": int(alert.fired_at.timestamp()) if alert.fired_at else int(datetime.utcnow().timestamp())
                }
            ]
        }
        
        # Envoi via API Slack
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            async with session.post(
                "https://slack.com/api/chat.postMessage",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=config.timeout)
            ) as response:
                if response.status != 200:
                    response_text = await response.text()
                    raise Exception(f"Erreur Slack API: {response.status} - {response_text}")
    
    async def _send_email_notification(
        self,
        config: NotificationConfig,
        alert: Alert,
        message: str,
        alert_type: str
    ) -> None:
        """Envoie une notification par email"""
        # Préparation du template
        if config.template and self.template_env:
            try:
                template = self.template_env.get_template(config.template)
                html_content = template.render(
                    alert=alert,
                    message=message,
                    alert_type=alert_type
                )
            except Exception as e:
                logger.warning(f"Erreur template email: {e}")
                html_content = message
        else:
            html_content = message
        
        # Création du message email
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.definition.name}"
        msg['From'] = config.config.get("from_email")
        msg['To'] = ", ".join(config.config.get("to_emails", []))
        
        # Partie texte et HTML
        text_part = MIMEText(message, 'plain')
        html_part = MIMEText(html_content, 'html')
        
        msg.attach(text_part)
        msg.attach(html_part)
        
        # Envoi SMTP
        smtp_config = config.config
        server = smtplib.SMTP(smtp_config.get("smtp_host"), smtp_config.get("smtp_port"))
        
        if smtp_config.get("use_tls", True):
            server.starttls()
        
        if smtp_config.get("smtp_user") and smtp_config.get("smtp_password"):
            server.login(smtp_config.get("smtp_user"), smtp_config.get("smtp_password"))
        
        server.send_message(msg)
        server.quit()
    
    async def _send_webhook_notification(
        self,
        config: NotificationConfig,
        alert: Alert,
        message: str,
        alert_type: str
    ) -> None:
        """Envoie une notification webhook"""
        url = config.config.get("url")
        if not url:
            raise ValueError("URL webhook manquante")
        
        # Préparation du payload
        payload = {
            "alert_id": alert.id,
            "alert_type": alert_type,
            "definition_id": alert.definition.id,
            "name": alert.definition.name,
            "severity": alert.severity.value,
            "status": alert.status.value,
            "message": message,
            "value": alert.value,
            "tenant_id": alert.tenant_id,
            "fired_at": alert.fired_at.isoformat() if alert.fired_at else None,
            "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
            "labels": alert.labels,
            "annotations": alert.annotations
        }
        
        # Configuration de la requête
        method = config.config.get("method", "POST").upper()
        headers = config.config.get("headers", {})
        headers.setdefault("Content-Type", "application/json")
        
        # Authentification
        auth = config.config.get("auth")
        auth_header = None
        if auth:
            if auth.get("type") == "bearer":
                auth_header = f"Bearer {auth.get('token')}"
            elif auth.get("type") == "basic":
                import base64
                credentials = f"{auth.get('username')}:{auth.get('password')}"
                encoded = base64.b64encode(credentials.encode()).decode()
                auth_header = f"Basic {encoded}"
        
        if auth_header:
            headers["Authorization"] = auth_header
        
        # Envoi de la requête
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method,
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=config.timeout)
            ) as response:
                if response.status >= 400:
                    response_text = await response.text()
                    raise Exception(f"Erreur webhook: {response.status} - {response_text}")
    
    def add_alert_callback(self, callback: Callable) -> None:
        """Ajoute un callback pour les alertes"""
        self.alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable) -> None:
        """Supprime un callback d'alerte"""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
    
    async def get_active_alerts(
        self,
        tenant_id: Optional[str] = None,
        severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """
        Récupère les alertes actives
        
        Args:
            tenant_id: Filtrer par tenant
            severity: Filtrer par sévérité
            
        Returns:
            Liste des alertes actives
        """
        alerts = list(self.active_alerts.values())
        
        if tenant_id:
            alerts = [a for a in alerts if a.tenant_id == tenant_id]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda x: x.fired_at or datetime.min, reverse=True)
    
    async def get_metrics(self) -> Dict[str, Any]:
        """
        Récupère les métriques du gestionnaire d'alertes
        
        Returns:
            Métriques
        """
        return {
            **self.metrics,
            "active_alerts": len(self.active_alerts),
            "resolved_alerts": len(self.resolved_alerts),
            "silenced_patterns": len(self.silenced_alerts),
            "alert_definitions": len(self.alert_definitions),
            "notification_channels": len(self.notification_configs)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Vérification de l'état de santé du gestionnaire d'alertes
        
        Returns:
            Rapport d'état
        """
        try:
            return {
                "status": "healthy",
                "is_running": self.is_running,
                "active_alerts": len(self.active_alerts),
                "alert_definitions": len(self.alert_definitions),
                "notification_channels": len(self.notification_configs),
                "templates_loaded": self.template_env is not None
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "is_running": self.is_running
            }
