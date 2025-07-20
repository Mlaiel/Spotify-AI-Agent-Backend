"""
Distributeur de Notifications Intelligent - Spotify AI Agent
===========================================================

Système avancé de distribution de notifications multi-canal avec
intelligence artificielle, déduplication et personnalisation.

Fonctionnalités:
- Distribution multi-canal (Slack, Teams, Email, SMS, Webhook)
- Déduplication intelligente des notifications
- Personnalisation basée sur les préférences utilisateur
- Escalade automatique avec rotation d'équipe
- Rate limiting et gestion de la charge
- Templates dynamiques avec variables contextuelles
- Analytics et tracking des notifications
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import jinja2
from collections import defaultdict, deque
import aiohttp
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import redis.asyncio as redis
from prometheus_client import Counter, Histogram


class NotificationChannel(Enum):
    """Canaux de notification supportés"""
    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"
    SMS = "sms"
    WEBHOOK = "webhook"
    PUSH = "push"
    PAGERDUTY = "pagerduty"


class NotificationPriority(Enum):
    """Priorités des notifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class DeliveryStatus(Enum):
    """États de livraison des notifications"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    RATE_LIMITED = "rate_limited"


@dataclass
class NotificationTemplate:
    """Template de notification avec variables dynamiques"""
    id: str
    name: str
    channel: NotificationChannel
    subject_template: str
    body_template: str
    priority: NotificationPriority = NotificationPriority.MEDIUM
    tenant_id: str = ""
    variables: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class NotificationRecipient:
    """Destinataire avec préférences"""
    id: str
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    slack_user_id: Optional[str] = None
    teams_user_id: Optional[str] = None
    webhook_url: Optional[str] = None
    tenant_id: str = ""
    preferred_channels: List[NotificationChannel] = field(default_factory=list)
    quiet_hours: Dict[str, Any] = field(default_factory=dict)
    escalation_delay: timedelta = field(default=timedelta(minutes=30))
    active: bool = True


@dataclass
class Notification:
    """Notification avec métadonnées complètes"""
    id: str
    title: str
    message: str
    channel: NotificationChannel
    priority: NotificationPriority
    recipient: NotificationRecipient
    tenant_id: str = ""
    service: str = ""
    alert_id: Optional[str] = None
    correlation_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    scheduled_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    status: DeliveryStatus = DeliveryStatus.PENDING
    attempts: int = 0
    max_attempts: int = 3
    template_id: Optional[str] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    attachments: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class NotificationDispatcher:
    """Distributeur intelligent de notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration des canaux
        self.channel_configs = config.get('channels', {})
        self.rate_limits = config.get('rate_limits', {})
        self.retry_config = config.get('retry', {})
        
        # Stockage
        self.redis_client = None
        self.notification_queue: deque = deque()
        self.pending_notifications: Dict[str, Notification] = {}
        self.delivery_history: deque = deque(maxlen=50000)
        
        # Templates et destinataires
        self.templates: Dict[str, NotificationTemplate] = {}
        self.recipients: Dict[str, NotificationRecipient] = {}
        self.escalation_chains: Dict[str, List[str]] = {}
        
        # Rate limiting
        self.rate_limiters: Dict[str, deque] = defaultdict(lambda: deque())
        
        # Déduplication
        self.deduplication_window = timedelta(minutes=5)
        self.recent_notifications: Dict[str, datetime] = {}
        
        # Templates Jinja2
        self.jinja_env = jinja2.Environment(
            loader=jinja2.DictLoader({}),
            autoescape=True
        )
        
        # HTTP Session pour webhooks
        self.http_session = None
        
        # Métriques
        self.metrics = {
            'notifications_sent': Counter('notifications_sent_total', 'Total notifications sent', 
                                        ['channel', 'priority', 'tenant_id']),
            'notifications_failed': Counter('notifications_failed_total', 'Failed notifications',
                                           ['channel', 'reason', 'tenant_id']),
            'delivery_duration': Histogram('notification_delivery_duration_seconds', 
                                         'Notification delivery duration', ['channel'])
        }
        
    async def initialize(self):
        """Initialisation asynchrone du dispatcher"""
        try:
            # Connexion Redis
            self.redis_client = redis.from_url(
                self.config.get('redis_url', 'redis://localhost:6379'),
                decode_responses=True
            )
            
            # Session HTTP
            self.http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            # Chargement des templates et destinataires
            await self._load_templates()
            await self._load_recipients()
            await self._load_escalation_chains()
            
            # Démarrage des tâches de fond
            asyncio.create_task(self._notification_processor())
            asyncio.create_task(self._retry_failed_notifications())
            asyncio.create_task(self._cleanup_old_data())
            asyncio.create_task(self._rate_limit_cleanup())
            
            self.logger.info("NotificationDispatcher initialisé avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation: {e}")
            raise
    
    async def dispatch_alert(self, alert: Any) -> List[str]:
        """Distribution des notifications pour une alerte"""
        try:
            notification_ids = []
            
            # Recherche des destinataires pour cette alerte
            recipients = await self._find_alert_recipients(alert)
            
            for recipient in recipients:
                # Vérification de la déduplication
                if await self._is_duplicate_notification(alert, recipient):
                    continue
                
                # Sélection du canal approprié
                channel = self._select_best_channel(alert, recipient)
                
                # Sélection du template
                template = await self._select_template(alert, channel, recipient)
                
                # Création de la notification
                notification = await self._create_notification_from_alert(
                    alert, recipient, channel, template
                )
                
                # Ajout à la queue
                notification_id = await self.queue_notification(notification)
                notification_ids.append(notification_id)
            
            self.logger.info(f"Alerte {alert.id} - {len(notification_ids)} notifications créées")
            return notification_ids
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la distribution d'alerte: {e}")
            return []
    
    async def queue_notification(self, notification: Notification) -> str:
        """Ajout d'une notification à la queue de traitement"""
        try:
            # Génération d'un ID unique
            notification.id = f"notif_{int(datetime.utcnow().timestamp())}_{len(self.notification_queue)}"
            
            # Vérification des quiet hours
            if self._is_in_quiet_hours(notification):
                notification.scheduled_at = self._calculate_next_send_time(notification)
            
            # Ajout à la queue
            self.notification_queue.append(notification)
            self.pending_notifications[notification.id] = notification
            
            # Persistance
            await self._persist_notification(notification)
            
            self.logger.debug(f"Notification mise en queue: {notification.id}")
            return notification.id
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la mise en queue: {e}")
            return ""
    
    async def send_immediate_notification(self, notification: Notification) -> bool:
        """Envoi immédiat d'une notification (bypass de la queue)"""
        try:
            # Vérification du rate limiting
            if not await self._check_rate_limit(notification):
                notification.status = DeliveryStatus.RATE_LIMITED
                return False
            
            # Envoi selon le canal
            success = await self._send_notification(notification)
            
            if success:
                notification.status = DeliveryStatus.SENT
                notification.sent_at = datetime.utcnow()
                
                # Mise à jour des métriques
                self.metrics['notifications_sent'].labels(
                    channel=notification.channel.value,
                    priority=notification.priority.value,
                    tenant_id=notification.tenant_id
                ).inc()
            else:
                notification.status = DeliveryStatus.FAILED
                notification.attempts += 1
                
                self.metrics['notifications_failed'].labels(
                    channel=notification.channel.value,
                    reason='send_failed',
                    tenant_id=notification.tenant_id
                ).inc()
            
            # Sauvegarde
            await self._persist_notification(notification)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'envoi immédiat: {e}")
            return False
    
    async def create_template(self, template_data: Dict[str, Any]) -> NotificationTemplate:
        """Création d'un template de notification"""
        try:
            template = NotificationTemplate(
                id=template_data.get('id', f"template_{len(self.templates)}"),
                name=template_data['name'],
                channel=NotificationChannel(template_data['channel']),
                subject_template=template_data['subject_template'],
                body_template=template_data['body_template'],
                priority=NotificationPriority(template_data.get('priority', 'medium')),
                tenant_id=template_data.get('tenant_id', ''),
                variables=template_data.get('variables', []),
                conditions=template_data.get('conditions', {})
            )
            
            self.templates[template.id] = template
            
            # Ajout au loader Jinja2
            self.jinja_env.loader.mapping[f"{template.id}_subject"] = template.subject_template
            self.jinja_env.loader.mapping[f"{template.id}_body"] = template.body_template
            
            # Persistance
            await self.redis_client.hset(
                'notification_templates',
                template.id,
                json.dumps({
                    'name': template.name,
                    'channel': template.channel.value,
                    'subject_template': template.subject_template,
                    'body_template': template.body_template,
                    'priority': template.priority.value,
                    'tenant_id': template.tenant_id,
                    'variables': template.variables,
                    'conditions': template.conditions,
                    'created_at': template.created_at.isoformat()
                })
            )
            
            self.logger.info(f"Template créé: {template.id} - {template.name}")
            return template
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la création de template: {e}")
            raise
    
    async def add_recipient(self, recipient_data: Dict[str, Any]) -> NotificationRecipient:
        """Ajout d'un destinataire"""
        try:
            recipient = NotificationRecipient(
                id=recipient_data.get('id', f"recipient_{len(self.recipients)}"),
                name=recipient_data['name'],
                email=recipient_data.get('email'),
                phone=recipient_data.get('phone'),
                slack_user_id=recipient_data.get('slack_user_id'),
                teams_user_id=recipient_data.get('teams_user_id'),
                webhook_url=recipient_data.get('webhook_url'),
                tenant_id=recipient_data.get('tenant_id', ''),
                preferred_channels=[
                    NotificationChannel(ch) for ch in recipient_data.get('preferred_channels', [])
                ],
                quiet_hours=recipient_data.get('quiet_hours', {}),
                escalation_delay=timedelta(
                    minutes=recipient_data.get('escalation_delay_minutes', 30)
                ),
                active=recipient_data.get('active', True)
            )
            
            self.recipients[recipient.id] = recipient
            
            # Persistance
            await self.redis_client.hset(
                'notification_recipients',
                recipient.id,
                json.dumps({
                    'name': recipient.name,
                    'email': recipient.email,
                    'phone': recipient.phone,
                    'slack_user_id': recipient.slack_user_id,
                    'teams_user_id': recipient.teams_user_id,
                    'webhook_url': recipient.webhook_url,
                    'tenant_id': recipient.tenant_id,
                    'preferred_channels': [ch.value for ch in recipient.preferred_channels],
                    'quiet_hours': recipient.quiet_hours,
                    'escalation_delay_minutes': int(recipient.escalation_delay.total_seconds() / 60),
                    'active': recipient.active
                })
            )
            
            self.logger.info(f"Destinataire ajouté: {recipient.id} - {recipient.name}")
            return recipient
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'ajout de destinataire: {e}")
            raise
    
    async def _notification_processor(self):
        """Processeur principal des notifications en queue"""
        while True:
            try:
                if not self.notification_queue:
                    await asyncio.sleep(1)
                    continue
                
                notification = self.notification_queue.popleft()
                
                # Vérification si c'est le moment d'envoyer
                if (notification.scheduled_at and 
                    datetime.utcnow() < notification.scheduled_at):
                    # Remettre en queue pour plus tard
                    self.notification_queue.append(notification)
                    await asyncio.sleep(10)
                    continue
                
                # Tentative d'envoi
                success = await self.send_immediate_notification(notification)
                
                # Gestion des échecs
                if not success and notification.attempts < notification.max_attempts:
                    # Programmer un retry
                    retry_delay = self._calculate_retry_delay(notification.attempts)
                    notification.scheduled_at = datetime.utcnow() + retry_delay
                    notification.status = DeliveryStatus.RETRYING
                    self.notification_queue.append(notification)
                
                # Nettoyage des notifications traitées
                if notification.id in self.pending_notifications:
                    if notification.status in [DeliveryStatus.SENT, DeliveryStatus.DELIVERED]:
                        del self.pending_notifications[notification.id]
                        self.delivery_history.append(notification)
                
                await asyncio.sleep(0.1)  # Éviter la surcharge CPU
                
            except Exception as e:
                self.logger.error(f"Erreur dans le processeur de notifications: {e}")
                await asyncio.sleep(5)
    
    async def _send_notification(self, notification: Notification) -> bool:
        """Envoi d'une notification selon son canal"""
        try:
            start_time = datetime.utcnow()
            
            success = False
            
            if notification.channel == NotificationChannel.EMAIL:
                success = await self._send_email(notification)
            elif notification.channel == NotificationChannel.SLACK:
                success = await self._send_slack(notification)
            elif notification.channel == NotificationChannel.TEAMS:
                success = await self._send_teams(notification)
            elif notification.channel == NotificationChannel.SMS:
                success = await self._send_sms(notification)
            elif notification.channel == NotificationChannel.WEBHOOK:
                success = await self._send_webhook(notification)
            elif notification.channel == NotificationChannel.PAGERDUTY:
                success = await self._send_pagerduty(notification)
            
            # Métriques de durée
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.metrics['delivery_duration'].labels(
                channel=notification.channel.value
            ).observe(duration)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'envoi via {notification.channel.value}: {e}")
            return False
    
    async def _send_email(self, notification: Notification) -> bool:
        """Envoi d'email"""
        try:
            if not notification.recipient.email:
                return False
            
            email_config = self.channel_configs.get('email', {})
            
            msg = MimeMultipart()
            msg['From'] = email_config.get('from_address', 'noreply@spotify-ai-agent.com')
            msg['To'] = notification.recipient.email
            msg['Subject'] = notification.title
            
            body = MimeText(notification.message, 'html' if '<' in notification.message else 'plain')
            msg.attach(body)
            
            # Configuration SMTP
            smtp_server = email_config.get('smtp_server', 'localhost')
            smtp_port = email_config.get('smtp_port', 587)
            username = email_config.get('username')
            password = email_config.get('password')
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            if email_config.get('use_tls', True):
                server.starttls()
            
            if username and password:
                server.login(username, password)
            
            server.send_message(msg)
            server.quit()
            
            self.logger.debug(f"Email envoyé à {notification.recipient.email}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'envoi d'email: {e}")
            return False
    
    async def _send_slack(self, notification: Notification) -> bool:
        """Envoi de notification Slack"""
        try:
            slack_config = self.channel_configs.get('slack', {})
            webhook_url = slack_config.get('webhook_url')
            
            if not webhook_url:
                return False
            
            # Construction du payload Slack
            payload = {
                'text': notification.title,
                'attachments': [{
                    'color': self._get_color_for_priority(notification.priority),
                    'fields': [
                        {
                            'title': 'Service',
                            'value': notification.service,
                            'short': True
                        },
                        {
                            'title': 'Tenant',
                            'value': notification.tenant_id,
                            'short': True
                        }
                    ],
                    'text': notification.message,
                    'ts': int(notification.created_at.timestamp())
                }]
            }
            
            # Mention d'utilisateur si configuré
            if notification.recipient.slack_user_id:
                payload['text'] = f"<@{notification.recipient.slack_user_id}> {payload['text']}"
            
            async with self.http_session.post(webhook_url, json=payload) as response:
                return response.status == 200
                
        except Exception as e:
            self.logger.error(f"Erreur lors de l'envoi Slack: {e}")
            return False
    
    def _get_color_for_priority(self, priority: NotificationPriority) -> str:
        """Couleur selon la priorité"""
        colors = {
            NotificationPriority.LOW: '#36a64f',      # Vert
            NotificationPriority.MEDIUM: '#ffb74d',   # Orange
            NotificationPriority.HIGH: '#f44336',     # Rouge
            NotificationPriority.CRITICAL: '#d32f2f', # Rouge foncé
            NotificationPriority.EMERGENCY: '#b71c1c' # Rouge très foncé
        }
        return colors.get(priority, '#757575')
    
    def _select_best_channel(self, alert: Any, recipient: NotificationRecipient) -> NotificationChannel:
        """Sélection du meilleur canal pour un destinataire"""
        # Priorité basée sur la sévérité de l'alerte
        severity_priorities = {
            'critical': [NotificationChannel.PAGERDUTY, NotificationChannel.SMS, NotificationChannel.SLACK],
            'high': [NotificationChannel.SLACK, NotificationChannel.EMAIL, NotificationChannel.TEAMS],
            'medium': [NotificationChannel.EMAIL, NotificationChannel.SLACK],
            'low': [NotificationChannel.EMAIL]
        }
        
        preferred_channels = severity_priorities.get(alert.severity.value, [NotificationChannel.EMAIL])
        
        # Intersection avec les préférences du destinataire
        for channel in preferred_channels:
            if channel in recipient.preferred_channels:
                return channel
        
        # Fallback sur le premier canal disponible
        for channel in preferred_channels:
            if self._is_channel_available(channel, recipient):
                return channel
        
        return NotificationChannel.EMAIL
    
    def _is_channel_available(self, channel: NotificationChannel, recipient: NotificationRecipient) -> bool:
        """Vérification de la disponibilité d'un canal"""
        if channel == NotificationChannel.EMAIL:
            return recipient.email is not None
        elif channel == NotificationChannel.SLACK:
            return recipient.slack_user_id is not None
        elif channel == NotificationChannel.SMS:
            return recipient.phone is not None
        elif channel == NotificationChannel.WEBHOOK:
            return recipient.webhook_url is not None
        
        return False
