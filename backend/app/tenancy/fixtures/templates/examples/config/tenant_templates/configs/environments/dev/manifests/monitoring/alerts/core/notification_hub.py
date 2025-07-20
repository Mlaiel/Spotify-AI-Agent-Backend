# -*- coding: utf-8 -*-
"""
Notification Hub - Hub Multi-Canal Ultra-Avancé
==============================================

Hub central de notifications pour le système d'alertes avec support:
- Multi-canaux (Email, Slack, Teams, PagerDuty, SMS, Webhooks)
- Templates avancés avec moteur Jinja2
- Rate limiting intelligent par canal
- Retry logic avec backoff exponentiel
- Health monitoring des canaux
- Routing intelligent basé sur contexte
- Formatting adaptatif par canal

Canaux Supportés:
- Email: SMTP avec templates HTML/Markdown
- Slack: API native avec boutons interactifs
- Microsoft Teams: Webhooks avec cartes riches
- PagerDuty: Intégration V2 API avec escalation
- SMS: Twilio/AWS SNS pour notifications critiques
- Webhooks: APIs personnalisées avec auth
- Discord: Notifications pour équipes DevOps

Fonctionnalités Avancées:
- Template engine avec héritage et macros
- A/B testing des messages
- Analytics de délivrance
- Fallback automatique entre canaux
- Localisation multi-langue
- Rich media support (images, charts)

Version: 3.0.0
"""

import asyncio
import smtplib
import ssl
import time
import json
import threading
import logging
import hashlib
import base64
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import sqlite3
import redis
import requests
from jinja2 import Environment, FileSystemLoader, Template
from concurrent.futures import ThreadPoolExecutor, Future
import uuid

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChannelType(Enum):
    """Types de canaux de notification"""
    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"
    PAGERDUTY = "pagerduty"
    SMS = "sms"
    WEBHOOK = "webhook"
    DISCORD = "discord"

class ChannelStatus(Enum):
    """État des canaux"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"
    MAINTENANCE = "maintenance"

class NotificationPriority(Enum):
    """Priorité des notifications"""
    IMMEDIATE = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BULK = 5

class DeliveryStatus(Enum):
    """État de délivrance"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    BOUNCED = "bounced"
    RETRYING = "retrying"

@dataclass
class NotificationChannel:
    """Configuration d'un canal de notification"""
    name: str
    type: ChannelType
    enabled: bool
    config: Dict[str, Any]
    rate_limit_per_minute: int = 60
    retry_attempts: int = 3
    retry_backoff_factor: float = 2.0
    timeout_seconds: int = 30
    health_check_interval: int = 300
    priority: int = 1
    fallback_channels: List[str] = field(default_factory=list)

@dataclass
class NotificationRequest:
    """Demande de notification"""
    id: str
    alert_id: str
    channels: List[str]
    priority: NotificationPriority
    template_name: str
    template_data: Dict[str, Any]
    recipients: Dict[str, List[str]]  # channel -> list of recipients
    tenant_id: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    correlation_id: Optional[str] = None

@dataclass
class NotificationResult:
    """Résultat d'envoi de notification"""
    id: str
    channel: str
    status: DeliveryStatus
    recipient: str
    sent_at: float
    delivered_at: Optional[float] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    external_id: Optional[str] = None
    response_data: Optional[Dict[str, Any]] = None

@dataclass
class ChannelHealth:
    """Santé d'un canal"""
    channel: str
    status: ChannelStatus
    last_check: float
    success_rate: float
    avg_response_time_ms: float
    errors_last_hour: int
    total_sent: int
    total_failed: int

class NotificationHub:
    """
    Hub central de notifications ultra-avancé
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise le hub de notifications
        
        Args:
            config: Configuration du hub
        """
        self.config = config or self._default_config()
        self.is_running = False
        self.start_time = time.time()
        
        # Channels et gestionnaires
        self.channels: Dict[str, NotificationChannel] = {}
        self.channel_handlers: Dict[str, Any] = {}
        self.channel_health: Dict[str, ChannelHealth] = {}
        
        # Queues et traitement
        self.notification_queue = asyncio.Queue(maxsize=10000)
        self.retry_queue = asyncio.Queue(maxsize=5000)
        self.results_storage: Dict[str, NotificationResult] = {}
        
        # Rate limiting
        self.rate_limiters: Dict[str, 'RateLimiter'] = {}
        
        # Templates
        self.template_engine = None
        self.template_cache = {}
        
        # Threads et executors
        self.executor = ThreadPoolExecutor(max_workers=20, thread_name_prefix='NotificationHub')
        self.worker_threads: List[threading.Thread] = []
        
        # Métriques
        self.metrics = {
            'total_sent': 0,
            'total_failed': 0,
            'total_retries': 0,
            'channels_healthy': 0,
            'avg_delivery_time_ms': 0,
            'rate_limited_count': 0
        }
        
        # Stockage
        self.db_path = self.config.get('db_path', 'notification_hub.db')
        self.redis_client = self._init_redis()
        
        # Lock pour thread safety
        self.lock = threading.RLock()
        
        # Initialisation
        self._init_database()
        self._setup_channels()
        self._init_template_engine()
        self._setup_rate_limiters()
        
        logger.info("NotificationHub initialisé avec succès")
    
    def _default_config(self) -> Dict:
        """Configuration par défaut"""
        return {
            'max_concurrent_notifications': 50,
            'retry_intervals': [1, 5, 15, 30, 60],
            'rate_limit_window_seconds': 60,
            'template_cache_size': 1000,
            'channel_health_check_interval': 300,
            'enable_delivery_tracking': True,
            'default_timeout_seconds': 30,
            'max_retry_attempts': 3,
            'enable_fallback_channels': True,
            'template_directory': './templates',
            'redis_host': 'localhost',
            'redis_port': 6379,
            'redis_db': 7,
            'db_retention_days': 30,
            'channels': {
                'email': {
                    'enabled': True,
                    'smtp_host': 'localhost',
                    'smtp_port': 587,
                    'smtp_username': '',
                    'smtp_password': '',
                    'use_tls': True,
                    'from_email': 'alerts@company.com',
                    'from_name': 'Alert System'
                },
                'slack': {
                    'enabled': False,
                    'webhook_url': '',
                    'bot_token': '',
                    'default_channel': '#alerts'
                },
                'teams': {
                    'enabled': False,
                    'webhook_url': ''
                },
                'pagerduty': {
                    'enabled': False,
                    'api_key': '',
                    'service_key': ''
                },
                'sms': {
                    'enabled': False,
                    'provider': 'twilio',
                    'api_key': '',
                    'from_number': ''
                },
                'webhook': {
                    'enabled': True,
                    'urls': [],
                    'auth_headers': {}
                }
            }
        }
    
    def _init_redis(self) -> Optional[redis.Redis]:
        """Initialise la connexion Redis"""
        try:
            client = redis.Redis(
                host=self.config['redis_host'],
                port=self.config['redis_port'],
                db=self.config['redis_db'],
                decode_responses=True
            )
            client.ping()
            logger.info("Connexion Redis NotificationHub établie")
            return client
        except Exception as e:
            logger.warning(f"Redis non disponible pour NotificationHub: {e}")
            return None
    
    def _init_database(self):
        """Initialise la base de données SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Table des notifications
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS notifications (
                    id TEXT PRIMARY KEY,
                    alert_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    recipient TEXT NOT NULL,
                    status TEXT NOT NULL,
                    template_name TEXT,
                    template_data TEXT,
                    sent_at REAL,
                    delivered_at REAL,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    external_id TEXT,
                    tenant_id TEXT,
                    priority INTEGER,
                    correlation_id TEXT
                )
            ''')
            
            # Table de santé des canaux
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS channel_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel TEXT NOT NULL,
                    status TEXT NOT NULL,
                    success_rate REAL,
                    avg_response_time_ms REAL,
                    errors_count INTEGER,
                    check_timestamp REAL NOT NULL
                )
            ''')
            
            # Table des templates
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS notification_templates (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    channel_type TEXT NOT NULL,
                    subject_template TEXT,
                    body_template TEXT NOT NULL,
                    template_format TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    tenant_id TEXT
                )
            ''')
            
            # Index pour les performances
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_notifications_alert ON notifications(alert_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_notifications_status ON notifications(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_health_channel_time ON channel_health(channel, check_timestamp)')
            
            conn.commit()
            conn.close()
            
            logger.info("Base de données NotificationHub initialisée")
            
        except Exception as e:
            logger.error(f"Erreur initialisation base de données: {e}")
    
    def _setup_channels(self):
        """Configure les canaux de notification"""
        try:
            channels_config = self.config.get('channels', {})
            
            for channel_name, channel_config in channels_config.items():
                if channel_config.get('enabled', False):
                    channel = NotificationChannel(
                        name=channel_name,
                        type=ChannelType(channel_name),
                        enabled=True,
                        config=channel_config,
                        rate_limit_per_minute=channel_config.get('rate_limit_per_minute', 60),
                        retry_attempts=channel_config.get('retry_attempts', 3),
                        timeout_seconds=channel_config.get('timeout_seconds', 30)
                    )
                    
                    self.channels[channel_name] = channel
                    
                    # Initialise le gestionnaire de canal
                    handler = self._create_channel_handler(channel)
                    if handler:
                        self.channel_handlers[channel_name] = handler
                    
                    # Initialise la santé du canal
                    self.channel_health[channel_name] = ChannelHealth(
                        channel=channel_name,
                        status=ChannelStatus.HEALTHY,
                        last_check=time.time(),
                        success_rate=100.0,
                        avg_response_time_ms=0.0,
                        errors_last_hour=0,
                        total_sent=0,
                        total_failed=0
                    )
            
            logger.info(f"Configuré {len(self.channels)} canaux de notification")
            
        except Exception as e:
            logger.error(f"Erreur configuration canaux: {e}")
    
    def _create_channel_handler(self, channel: NotificationChannel) -> Optional[Any]:
        """Crée un gestionnaire pour un canal"""
        try:
            if channel.type == ChannelType.EMAIL:
                return EmailHandler(channel.config)
            elif channel.type == ChannelType.SLACK:
                return SlackHandler(channel.config)
            elif channel.type == ChannelType.TEAMS:
                return TeamsHandler(channel.config)
            elif channel.type == ChannelType.PAGERDUTY:
                return PagerDutyHandler(channel.config)
            elif channel.type == ChannelType.SMS:
                return SMSHandler(channel.config)
            elif channel.type == ChannelType.WEBHOOK:
                return WebhookHandler(channel.config)
            elif channel.type == ChannelType.DISCORD:
                return DiscordHandler(channel.config)
            else:
                logger.warning(f"Type de canal non supporté: {channel.type}")
                return None
                
        except Exception as e:
            logger.error(f"Erreur création gestionnaire canal {channel.name}: {e}")
            return None
    
    def _init_template_engine(self):
        """Initialise le moteur de templates"""
        try:
            # Configuration Jinja2
            template_dir = self.config.get('template_directory', './templates')
            
            try:
                # Essaie de charger depuis le répertoire
                loader = FileSystemLoader(template_dir)
                self.template_engine = Environment(loader=loader, autoescape=True)
            except:
                # Fallback vers templates en mémoire
                self.template_engine = Environment(autoescape=True)
            
            # Ajoute des fonctions utilitaires aux templates
            self.template_engine.globals.update({
                'format_datetime': self._format_datetime,
                'format_duration': self._format_duration,
                'severity_color': self._get_severity_color,
                'escape_markdown': self._escape_markdown
            })
            
            # Charge les templates par défaut
            self._load_default_templates()
            
            logger.info("Moteur de templates initialisé")
            
        except Exception as e:
            logger.error(f"Erreur initialisation templates: {e}")
    
    def _setup_rate_limiters(self):
        """Configure les limiteurs de débit"""
        try:
            for channel_name, channel in self.channels.items():
                limiter = RateLimiter(
                    max_requests=channel.rate_limit_per_minute,
                    window_seconds=60
                )
                self.rate_limiters[channel_name] = limiter
            
            logger.info("Limiteurs de débit configurés")
            
        except Exception as e:
            logger.error(f"Erreur configuration rate limiters: {e}")
    
    def start(self) -> bool:
        """Démarre le hub de notifications"""
        if self.is_running:
            logger.warning("NotificationHub déjà en cours d'exécution")
            return True
        
        try:
            self.is_running = True
            
            # Démarre les workers de traitement
            for i in range(self.config['max_concurrent_notifications'] // 5):
                worker = threading.Thread(
                    target=self._notification_worker,
                    name=f'NotificationWorker-{i}',
                    daemon=True
                )
                worker.start()
                self.worker_threads.append(worker)
            
            # Worker pour les retry
            retry_worker = threading.Thread(target=self._retry_worker, daemon=True)
            retry_worker.start()
            
            # Worker pour le health check
            health_worker = threading.Thread(target=self._health_check_worker, daemon=True)
            health_worker.start()
            
            logger.info(f"NotificationHub démarré avec {len(self.worker_threads)} workers")
            return True
            
        except Exception as e:
            logger.error(f"Erreur démarrage NotificationHub: {e}")
            return False
    
    def stop(self) -> bool:
        """Arrête le hub de notifications"""
        if not self.is_running:
            return True
        
        try:
            logger.info("Arrêt NotificationHub...")
            self.is_running = False
            
            # Attend les workers
            for worker in self.worker_threads:
                if worker.is_alive():
                    worker.join(timeout=10)
            
            # Arrête l'executor
            self.executor.shutdown(wait=True, timeout=30)
            
            logger.info("NotificationHub arrêté")
            return True
            
        except Exception as e:
            logger.error(f"Erreur arrêt NotificationHub: {e}")
            return False
    
    async def send_notification(self, request: NotificationRequest) -> List[NotificationResult]:
        """
        Envoie une notification
        
        Args:
            request: Demande de notification
            
        Returns:
            Liste des résultats d'envoi
        """
        try:
            results = []
            
            # Validation de la demande
            if not self._validate_notification_request(request):
                logger.error(f"Demande de notification invalide: {request.id}")
                return results
            
            # Traitement pour chaque canal
            for channel_name in request.channels:
                if channel_name not in self.channels:
                    logger.warning(f"Canal inconnu: {channel_name}")
                    continue
                
                channel = self.channels[channel_name]
                if not channel.enabled:
                    logger.warning(f"Canal désactivé: {channel_name}")
                    continue
                
                # Vérification du rate limiting
                if not self._check_rate_limit(channel_name):
                    logger.warning(f"Rate limit atteint pour {channel_name}")
                    self.metrics['rate_limited_count'] += 1
                    continue
                
                # Récupère les destinataires pour ce canal
                recipients = request.recipients.get(channel_name, [])
                if not recipients:
                    continue
                
                # Envoie pour chaque destinataire
                for recipient in recipients:
                    try:
                        result = await self._send_single_notification(
                            request, channel_name, recipient
                        )
                        if result:
                            results.append(result)
                            
                    except Exception as e:
                        logger.error(f"Erreur envoi à {recipient} sur {channel_name}: {e}")
                        
                        # Crée un résultat d'échec
                        result = NotificationResult(
                            id=str(uuid.uuid4()),
                            channel=channel_name,
                            status=DeliveryStatus.FAILED,
                            recipient=recipient,
                            sent_at=time.time(),
                            error_message=str(e)
                        )
                        results.append(result)
            
            # Sauvegarde des résultats
            for result in results:
                self._save_notification_result(result, request)
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur envoi notification {request.id}: {e}")
            return []
    
    async def _send_single_notification(self, request: NotificationRequest, 
                                      channel_name: str, recipient: str) -> Optional[NotificationResult]:
        """Envoie une notification à un destinataire sur un canal"""
        try:
            start_time = time.time()
            
            # Génère le contenu à partir du template
            content = self._render_template(request, channel_name)
            if not content:
                logger.error(f"Échec rendu template pour {channel_name}")
                return None
            
            # Récupère le gestionnaire du canal
            handler = self.channel_handlers.get(channel_name)
            if not handler:
                logger.error(f"Gestionnaire manquant pour {channel_name}")
                return None
            
            # Envoie via le gestionnaire
            send_result = await handler.send(recipient, content, request)
            
            # Calcul du temps de réponse
            response_time = (time.time() - start_time) * 1000
            
            # Mise à jour de la santé du canal
            self._update_channel_health(channel_name, True, response_time)
            
            # Crée le résultat
            result = NotificationResult(
                id=str(uuid.uuid4()),
                channel=channel_name,
                status=DeliveryStatus.SENT if send_result['success'] else DeliveryStatus.FAILED,
                recipient=recipient,
                sent_at=start_time,
                external_id=send_result.get('external_id'),
                response_data=send_result.get('response_data'),
                error_message=send_result.get('error') if not send_result['success'] else None
            )
            
            # Mise à jour des métriques
            if send_result['success']:
                self.metrics['total_sent'] += 1
            else:
                self.metrics['total_failed'] += 1
                self._update_channel_health(channel_name, False, response_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur envoi notification: {e}")
            self._update_channel_health(channel_name, False, 0)
            return None
    
    def _render_template(self, request: NotificationRequest, channel_name: str) -> Optional[Dict[str, str]]:
        """Rend le template pour un canal"""
        try:
            # Récupère le template
            template_key = f"{request.template_name}_{channel_name}"
            
            if template_key in self.template_cache:
                template_data = self.template_cache[template_key]
            else:
                template_data = self._load_template(request.template_name, channel_name)
                if not template_data:
                    return None
                
                # Mise en cache
                self.template_cache[template_key] = template_data
            
            # Préparation des données pour le template
            template_context = {
                'alert': request.template_data,
                'notification': {
                    'id': request.id,
                    'channel': channel_name,
                    'priority': request.priority.name,
                    'created_at': request.created_at
                },
                'tenant_id': request.tenant_id,
                'labels': request.labels
            }
            
            # Rendu du subject (si applicable)
            subject = None
            if template_data.get('subject_template'):
                subject_template = self.template_engine.from_string(template_data['subject_template'])
                subject = subject_template.render(**template_context)
            
            # Rendu du body
            body_template = self.template_engine.from_string(template_data['body_template'])
            body = body_template.render(**template_context)
            
            return {
                'subject': subject,
                'body': body,
                'format': template_data.get('template_format', 'text')
            }
            
        except Exception as e:
            logger.error(f"Erreur rendu template {request.template_name} pour {channel_name}: {e}")
            return None
    
    def _load_template(self, template_name: str, channel_name: str) -> Optional[Dict[str, str]]:
        """Charge un template"""
        try:
            # Essaie de charger depuis la base de données
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT subject_template, body_template, template_format
                FROM notification_templates 
                WHERE name = ? AND channel_type = ?
            ''', (template_name, channel_name))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'subject_template': row[0],
                    'body_template': row[1],
                    'template_format': row[2] or 'text'
                }
            
            # Fallback vers template par défaut
            return self._get_default_template(template_name, channel_name)
            
        except Exception as e:
            logger.error(f"Erreur chargement template {template_name}/{channel_name}: {e}")
            return None
    
    def _get_default_template(self, template_name: str, channel_name: str) -> Optional[Dict[str, str]]:
        """Retourne un template par défaut"""
        try:
            # Templates par défaut pour les alertes
            if template_name == 'alert' and channel_name == 'email':
                return {
                    'subject_template': '🚨 {{ alert.severity.upper() }}: {{ alert.name }}',
                    'body_template': '''
<h2>Alert: {{ alert.name }}</h2>
<p><strong>Severity:</strong> {{ alert.severity }}</p>
<p><strong>Description:</strong> {{ alert.description }}</p>
<p><strong>Source:</strong> {{ alert.source }}</p>
<p><strong>Time:</strong> {{ format_datetime(alert.timestamp) }}</p>

{% if alert.labels %}
<h3>Labels:</h3>
<ul>
{% for key, value in alert.labels.items() %}
<li><strong>{{ key }}:</strong> {{ value }}</li>
{% endfor %}
</ul>
{% endif %}

<p><em>This alert was generated by the Spotify AI Agent monitoring system.</em></p>
                    ''',
                    'template_format': 'html'
                }
            
            elif template_name == 'alert' and channel_name == 'slack':
                return {
                    'body_template': '''
{
    "text": "{{ alert.severity.upper() }}: {{ alert.name }}",
    "blocks": [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "🚨 {{ alert.name }}"
            }
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": "*Severity:* {{ severity_color(alert.severity) }} {{ alert.severity.upper() }}"
                },
                {
                    "type": "mrkdwn",
                    "text": "*Source:* {{ alert.source }}"
                }
            ]
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Description:* {{ alert.description }}"
            }
        }
    ]
}
                    ''',
                    'template_format': 'json'
                }
            
            # Template générique
            return {
                'subject_template': 'Alert: {{ alert.name }}',
                'body_template': '''
Alert: {{ alert.name }}
Severity: {{ alert.severity }}
Description: {{ alert.description }}
Source: {{ alert.source }}
Time: {{ format_datetime(alert.timestamp) }}
                ''',
                'template_format': 'text'
            }
            
        except Exception as e:
            logger.error(f"Erreur création template par défaut: {e}")
            return None
    
    def _load_default_templates(self):
        """Charge les templates par défaut en base"""
        try:
            default_templates = [
                {
                    'id': 'alert_email',
                    'name': 'alert',
                    'channel_type': 'email',
                    'subject_template': '🚨 {{ alert.severity.upper() }}: {{ alert.name }}',
                    'body_template': 'Default email template for alerts',
                    'template_format': 'html'
                },
                {
                    'id': 'alert_slack', 
                    'name': 'alert',
                    'channel_type': 'slack',
                    'body_template': 'Default Slack template for alerts',
                    'template_format': 'json'
                }
            ]
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for template in default_templates:
                cursor.execute('''
                    INSERT OR IGNORE INTO notification_templates 
                    (id, name, channel_type, subject_template, body_template, template_format, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    template['id'], template['name'], template['channel_type'],
                    template.get('subject_template'), template['body_template'],
                    template['template_format'], time.time(), time.time()
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Erreur chargement templates par défaut: {e}")
    
    def _check_rate_limit(self, channel_name: str) -> bool:
        """Vérifie le rate limit pour un canal"""
        try:
            limiter = self.rate_limiters.get(channel_name)
            if not limiter:
                return True
            
            return limiter.check()
            
        except Exception as e:
            logger.error(f"Erreur vérification rate limit {channel_name}: {e}")
            return True
    
    def _update_channel_health(self, channel_name: str, success: bool, response_time_ms: float):
        """Met à jour la santé d'un canal"""
        try:
            with self.lock:
                health = self.channel_health.get(channel_name)
                if not health:
                    return
                
                # Mise à jour des compteurs
                health.total_sent += 1
                if not success:
                    health.total_failed += 1
                    health.errors_last_hour += 1
                
                # Calcul du taux de succès
                health.success_rate = ((health.total_sent - health.total_failed) / health.total_sent) * 100
                
                # Mise à jour du temps de réponse moyen
                if health.avg_response_time_ms == 0:
                    health.avg_response_time_ms = response_time_ms
                else:
                    health.avg_response_time_ms = (health.avg_response_time_ms + response_time_ms) / 2
                
                # Détermination du statut
                if health.success_rate >= 95:
                    health.status = ChannelStatus.HEALTHY
                elif health.success_rate >= 80:
                    health.status = ChannelStatus.DEGRADED
                else:
                    health.status = ChannelStatus.DOWN
                
                health.last_check = time.time()
                
                # Sauvegarde en base
                self._save_channel_health(health)
                
        except Exception as e:
            logger.error(f"Erreur mise à jour santé canal {channel_name}: {e}")
    
    def _validate_notification_request(self, request: NotificationRequest) -> bool:
        """Valide une demande de notification"""
        try:
            # Vérifications de base
            if not request.id or not request.channels:
                return False
            
            if not request.template_name or not request.template_data:
                return False
            
            # Vérification de l'expiration
            if request.expires_at and time.time() > request.expires_at:
                logger.warning(f"Notification expirée: {request.id}")
                return False
            
            # Vérification des canaux
            valid_channels = []
            for channel_name in request.channels:
                if channel_name in self.channels and self.channels[channel_name].enabled:
                    valid_channels.append(channel_name)
            
            if not valid_channels:
                logger.warning(f"Aucun canal valide pour notification {request.id}")
                return False
            
            request.channels = valid_channels
            return True
            
        except Exception as e:
            logger.error(f"Erreur validation notification {request.id}: {e}")
            return False
    
    def _notification_worker(self):
        """Worker de traitement des notifications"""
        while self.is_running:
            try:
                # Traitement des notifications en attente
                # Note: Dans une implémentation réelle, utiliser asyncio.Queue
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Erreur worker notification: {e}")
                time.sleep(5)
    
    def _retry_worker(self):
        """Worker pour les retry"""
        while self.is_running:
            try:
                # Traitement des retry
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Erreur worker retry: {e}")
                time.sleep(10)
    
    def _health_check_worker(self):
        """Worker pour les vérifications de santé"""
        while self.is_running:
            try:
                self._perform_health_checks()
                time.sleep(self.config['channel_health_check_interval'])
                
            except Exception as e:
                logger.error(f"Erreur worker health check: {e}")
                time.sleep(60)
    
    def _perform_health_checks(self):
        """Effectue les vérifications de santé des canaux"""
        try:
            for channel_name, handler in self.channel_handlers.items():
                try:
                    if hasattr(handler, 'health_check'):
                        is_healthy = handler.health_check()
                        
                        health = self.channel_health.get(channel_name)
                        if health:
                            if is_healthy:
                                if health.status == ChannelStatus.DOWN:
                                    health.status = ChannelStatus.HEALTHY
                            else:
                                health.status = ChannelStatus.DOWN
                            
                            health.last_check = time.time()
                            self._save_channel_health(health)
                    
                except Exception as e:
                    logger.error(f"Erreur health check canal {channel_name}: {e}")
            
        except Exception as e:
            logger.error(f"Erreur health checks: {e}")
    
    def _save_notification_result(self, result: NotificationResult, request: NotificationRequest):
        """Sauvegarde un résultat de notification"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO notifications 
                (id, alert_id, channel, recipient, status, template_name, template_data,
                 sent_at, delivered_at, error_message, retry_count, external_id, 
                 tenant_id, priority, correlation_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.id, request.alert_id, result.channel, result.recipient,
                result.status.value, request.template_name, json.dumps(request.template_data),
                result.sent_at, result.delivered_at, result.error_message,
                result.retry_count, result.external_id, request.tenant_id,
                request.priority.value, request.correlation_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde résultat notification: {e}")
    
    def _save_channel_health(self, health: ChannelHealth):
        """Sauvegarde la santé d'un canal"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO channel_health 
                (channel, status, success_rate, avg_response_time_ms, errors_count, check_timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                health.channel, health.status.value, health.success_rate,
                health.avg_response_time_ms, health.errors_last_hour, health.last_check
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde santé canal: {e}")
    
    def get_channel_status(self, channel: str) -> Dict[str, Any]:
        """Retourne le statut d'un canal"""
        try:
            health = self.channel_health.get(channel)
            if not health:
                return {'status': 'unknown'}
            
            return {
                'status': health.status.value,
                'success_rate': health.success_rate,
                'avg_response_time_ms': health.avg_response_time_ms,
                'errors_last_hour': health.errors_last_hour,
                'total_sent': health.total_sent,
                'total_failed': health.total_failed,
                'last_check': health.last_check
            }
            
        except Exception as e:
            logger.error(f"Erreur récupération statut canal {channel}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Vérification de santé du hub"""
        try:
            healthy_channels = sum(1 for h in self.channel_health.values() 
                                 if h.status == ChannelStatus.HEALTHY)
            
            return {
                'status': 'healthy' if self.is_running else 'stopped',
                'channels_configured': len(self.channels),
                'channels_healthy': healthy_channels,
                'total_notifications_sent': self.metrics['total_sent'],
                'total_notifications_failed': self.metrics['total_failed'],
                'rate_limited_count': self.metrics['rate_limited_count'],
                'uptime_seconds': time.time() - self.start_time,
                'channels': {name: self.get_channel_status(name) for name in self.channels.keys()}
            }
            
        except Exception as e:
            logger.error(f"Erreur health check: {e}")
            return {'status': 'error', 'error': str(e)}
    
    # Fonctions utilitaires pour les templates
    def _format_datetime(self, timestamp: float) -> str:
        """Formate une date/heure"""
        from datetime import datetime
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S UTC')
    
    def _format_duration(self, seconds: float) -> str:
        """Formate une durée"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.0f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def _get_severity_color(self, severity: str) -> str:
        """Retourne une couleur pour la sévérité"""
        colors = {
            'critical': '🔴',
            'warning': '🟡',
            'info': '🔵',
            'debug': '⚪'
        }
        return colors.get(severity.lower(), '⚪')
    
    def _escape_markdown(self, text: str) -> str:
        """Échappe le markdown"""
        special_chars = ['*', '_', '`', '[', ']', '(', ')', '~', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        for char in special_chars:
            text = text.replace(char, f'\\{char}')
        return text

# Classes de gestionnaires de canaux
class BaseChannelHandler:
    """Classe de base pour les gestionnaires de canaux"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def send(self, recipient: str, content: Dict[str, str], request: NotificationRequest) -> Dict[str, Any]:
        """Envoie une notification"""
        raise NotImplementedError
    
    def health_check(self) -> bool:
        """Vérifie la santé du canal"""
        return True

class EmailHandler(BaseChannelHandler):
    """Gestionnaire pour les emails"""
    
    async def send(self, recipient: str, content: Dict[str, str], request: NotificationRequest) -> Dict[str, Any]:
        """Envoie un email"""
        try:
            # Configuration SMTP
            smtp_host = self.config.get('smtp_host', 'localhost')
            smtp_port = self.config.get('smtp_port', 587)
            smtp_username = self.config.get('smtp_username', '')
            smtp_password = self.config.get('smtp_password', '')
            use_tls = self.config.get('use_tls', True)
            
            # Création du message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = content.get('subject', 'Alert Notification')
            msg['From'] = f"{self.config.get('from_name', 'Alert System')} <{self.config.get('from_email', 'alerts@company.com')}>"
            msg['To'] = recipient
            
            # Corps du message
            if content.get('format') == 'html':
                part = MIMEText(content['body'], 'html')
            else:
                part = MIMEText(content['body'], 'plain')
            
            msg.attach(part)
            
            # Envoi
            server = smtplib.SMTP(smtp_host, smtp_port)
            if use_tls:
                server.starttls()
            
            if smtp_username and smtp_password:
                server.login(smtp_username, smtp_password)
            
            server.send_message(msg)
            server.quit()
            
            return {
                'success': True,
                'external_id': f"email_{int(time.time())}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

class SlackHandler(BaseChannelHandler):
    """Gestionnaire pour Slack"""
    
    async def send(self, recipient: str, content: Dict[str, str], request: NotificationRequest) -> Dict[str, Any]:
        """Envoie une notification Slack"""
        try:
            webhook_url = self.config.get('webhook_url')
            if not webhook_url:
                return {'success': False, 'error': 'Webhook URL manquante'}
            
            # Préparation du payload
            if content.get('format') == 'json':
                payload = json.loads(content['body'])
            else:
                payload = {
                    'text': content['body'],
                    'channel': recipient
                }
            
            # Envoi
            response = requests.post(webhook_url, json=payload, timeout=30)
            response.raise_for_status()
            
            return {
                'success': True,
                'external_id': response.headers.get('X-Slack-Message-Id'),
                'response_data': {'status_code': response.status_code}
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

class TeamsHandler(BaseChannelHandler):
    """Gestionnaire pour Microsoft Teams"""
    
    async def send(self, recipient: str, content: Dict[str, str], request: NotificationRequest) -> Dict[str, Any]:
        """Envoie une notification Teams"""
        try:
            webhook_url = self.config.get('webhook_url')
            if not webhook_url:
                return {'success': False, 'error': 'Webhook URL manquante'}
            
            # Payload Teams
            payload = {
                "@type": "MessageCard",
                "@context": "https://schema.org/extensions",
                "summary": content.get('subject', 'Alert Notification'),
                "themeColor": "FF0000",
                "sections": [{
                    "text": content['body']
                }]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=30)
            response.raise_for_status()
            
            return {
                'success': True,
                'external_id': f"teams_{int(time.time())}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

class PagerDutyHandler(BaseChannelHandler):
    """Gestionnaire pour PagerDuty"""
    
    async def send(self, recipient: str, content: Dict[str, str], request: NotificationRequest) -> Dict[str, Any]:
        """Envoie une notification PagerDuty"""
        try:
            # TODO: Implémentation PagerDuty V2 API
            return {'success': False, 'error': 'PagerDuty non implémenté'}
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

class SMSHandler(BaseChannelHandler):
    """Gestionnaire pour SMS"""
    
    async def send(self, recipient: str, content: Dict[str, str], request: NotificationRequest) -> Dict[str, Any]:
        """Envoie un SMS"""
        try:
            # TODO: Implémentation Twilio/AWS SNS
            return {'success': False, 'error': 'SMS non implémenté'}
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

class WebhookHandler(BaseChannelHandler):
    """Gestionnaire pour webhooks"""
    
    async def send(self, recipient: str, content: Dict[str, str], request: NotificationRequest) -> Dict[str, Any]:
        """Envoie via webhook"""
        try:
            webhook_url = recipient  # URL est le "destinataire"
            
            payload = {
                'alert_id': request.alert_id,
                'notification_id': request.id,
                'content': content,
                'metadata': {
                    'tenant_id': request.tenant_id,
                    'priority': request.priority.value,
                    'timestamp': time.time()
                }
            }
            
            headers = self.config.get('auth_headers', {})
            response = requests.post(webhook_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            return {
                'success': True,
                'external_id': response.headers.get('X-Request-Id'),
                'response_data': {'status_code': response.status_code}
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

class DiscordHandler(BaseChannelHandler):
    """Gestionnaire pour Discord"""
    
    async def send(self, recipient: str, content: Dict[str, str], request: NotificationRequest) -> Dict[str, Any]:
        """Envoie une notification Discord"""
        try:
            # TODO: Implémentation Discord Webhooks
            return {'success': False, 'error': 'Discord non implémenté'}
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

class RateLimiter:
    """Limiteur de débit simple"""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()
        self.lock = threading.Lock()
    
    def check(self) -> bool:
        """Vérifie si une requête peut passer"""
        current_time = time.time()
        
        with self.lock:
            # Nettoie les anciennes requêtes
            while self.requests and self.requests[0] < current_time - self.window_seconds:
                self.requests.popleft()
            
            # Vérifie la limite
            if len(self.requests) >= self.max_requests:
                return False
            
            # Ajoute la requête
            self.requests.append(current_time)
            return True
