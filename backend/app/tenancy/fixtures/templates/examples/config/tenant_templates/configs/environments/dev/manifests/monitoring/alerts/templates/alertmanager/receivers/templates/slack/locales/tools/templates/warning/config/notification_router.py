"""
Routeur de Notifications Ultra-Avancé - Spotify AI Agent
=======================================================

Système de routage intelligent pour les notifications d'alertes Warning
avec support multi-canal, load balancing et résilience.

Auteur: Équipe d'experts dirigée par Fahed Mlaiel
"""

import os
import json
import logging
import asyncio
import aiohttp
import smtplib
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import requests
import hashlib
from collections import defaultdict, deque
import random
import ssl

# Configuration du logging
logger = logging.getLogger(__name__)

class NotificationChannel(Enum):
    """Canaux de notification disponibles."""
    SLACK = "slack"
    EMAIL = "email"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGERDUTY = "pagerduty"
    TEAMS = "teams"
    DISCORD = "discord"

class DeliveryStatus(Enum):
    """Statuts de livraison des notifications."""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"

class Priority(Enum):
    """Niveaux de priorité des notifications."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

@dataclass
class NotificationConfig:
    """Configuration d'un canal de notification."""
    channel: NotificationChannel
    endpoint: str
    credentials: Dict[str, str]
    retry_config: Dict[str, Any]
    rate_limits: Dict[str, int]
    timeout_seconds: int
    enabled: bool
    fallback_channels: List[NotificationChannel]
    templates: Dict[str, str]
    metadata: Dict[str, Any]

@dataclass
class NotificationRequest:
    """Requête de notification."""
    request_id: str
    tenant_id: str
    alert_id: str
    channels: List[NotificationChannel]
    priority: Priority
    message: str
    template_data: Dict[str, Any]
    delivery_time: Optional[datetime]
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class DeliveryResult:
    """Résultat de livraison d'une notification."""
    request_id: str
    channel: NotificationChannel
    status: DeliveryStatus
    response_data: Dict[str, Any]
    error_message: Optional[str]
    delivery_time: datetime
    response_time_ms: int
    retry_count: int

class RateLimiter:
    """Limiteur de débit par canal."""
    
    def __init__(self):
        self.limits = {}
        self.windows = {}
        self.lock = threading.RLock()
    
    def set_limit(self, channel: NotificationChannel, requests_per_minute: int):
        """Définit une limite de débit pour un canal."""
        with self.lock:
            self.limits[channel] = requests_per_minute
            self.windows[channel] = deque()
    
    def can_send(self, channel: NotificationChannel) -> bool:
        """Vérifie si l'envoi est autorisé selon le rate limiting."""
        with self.lock:
            if channel not in self.limits:
                return True
            
            now = datetime.now()
            window = self.windows[channel]
            limit = self.limits[channel]
            
            # Nettoyage de la fenêtre glissante (dernière minute)
            cutoff = now - timedelta(minutes=1)
            while window and window[0] < cutoff:
                window.popleft()
            
            # Vérification de la limite
            if len(window) >= limit:
                return False
            
            # Enregistrement de la requête
            window.append(now)
            return True
    
    def get_wait_time(self, channel: NotificationChannel) -> int:
        """Retourne le temps d'attente en secondes avant le prochain envoi."""
        with self.lock:
            if channel not in self.limits:
                return 0
            
            window = self.windows[channel]
            if not window:
                return 0
            
            # Temps jusqu'à expiration de la plus ancienne requête
            oldest = window[0]
            wait_until = oldest + timedelta(minutes=1)
            wait_seconds = max(0, (wait_until - datetime.now()).total_seconds())
            
            return int(wait_seconds)

class HealthChecker:
    """Vérificateur de santé des canaux de notification."""
    
    def __init__(self):
        self.health_status = {}
        self.last_checks = {}
        self.check_interval = 300  # 5 minutes
        self.lock = threading.RLock()
    
    async def check_channel_health(self, channel: NotificationChannel, 
                                 config: NotificationConfig) -> bool:
        """Vérifie la santé d'un canal de notification."""
        
        try:
            if channel == NotificationChannel.SLACK:
                return await self._check_slack_health(config)
            elif channel == NotificationChannel.EMAIL:
                return await self._check_email_health(config)
            elif channel == NotificationChannel.WEBHOOK:
                return await self._check_webhook_health(config)
            else:
                return True  # Par défaut, considéré comme sain
                
        except Exception as e:
            logger.error(f"Erreur vérification santé {channel.value}: {e}")
            return False
    
    async def _check_slack_health(self, config: NotificationConfig) -> bool:
        """Vérifie la santé de Slack."""
        try:
            webhook_url = config.endpoint
            test_payload = {
                "text": "Health check - Spotify AI Agent",
                "username": "Health Check Bot"
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.post(webhook_url, json=test_payload) as response:
                    return response.status == 200
        except:
            return False
    
    async def _check_email_health(self, config: NotificationConfig) -> bool:
        """Vérifie la santé du serveur email."""
        try:
            smtp_server = config.credentials.get('smtp_server')
            smtp_port = int(config.credentials.get('smtp_port', 587))
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.quit()
            return True
        except:
            return False
    
    async def _check_webhook_health(self, config: NotificationConfig) -> bool:
        """Vérifie la santé d'un webhook."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(config.endpoint) as response:
                    return response.status < 500
        except:
            return False
    
    def is_healthy(self, channel: NotificationChannel) -> bool:
        """Retourne l'état de santé d'un canal."""
        with self.lock:
            return self.health_status.get(channel, True)
    
    def set_health_status(self, channel: NotificationChannel, healthy: bool):
        """Définit l'état de santé d'un canal."""
        with self.lock:
            self.health_status[channel] = healthy
            self.last_checks[channel] = datetime.now()

class SlackNotifier:
    """Notificateur Slack avec fonctionnalités avancées."""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
        self.webhook_url = config.endpoint
        self.default_username = config.credentials.get('username', 'Spotify AI Agent')
        self.default_icon = config.credentials.get('icon_emoji', ':robot_face:')
    
    async def send_notification(self, request: NotificationRequest) -> DeliveryResult:
        """Envoie une notification Slack."""
        start_time = time.time()
        
        try:
            payload = self._build_slack_payload(request)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                ) as response:
                    response_time = int((time.time() - start_time) * 1000)
                    response_data = await response.text()
                    
                    if response.status == 200:
                        return DeliveryResult(
                            request_id=request.request_id,
                            channel=NotificationChannel.SLACK,
                            status=DeliveryStatus.SENT,
                            response_data={"status_code": response.status, "response": response_data},
                            error_message=None,
                            delivery_time=datetime.now(),
                            response_time_ms=response_time,
                            retry_count=request.retry_count
                        )
                    else:
                        return DeliveryResult(
                            request_id=request.request_id,
                            channel=NotificationChannel.SLACK,
                            status=DeliveryStatus.FAILED,
                            response_data={"status_code": response.status, "response": response_data},
                            error_message=f"HTTP {response.status}",
                            delivery_time=datetime.now(),
                            response_time_ms=response_time,
                            retry_count=request.retry_count
                        )
        
        except asyncio.TimeoutError:
            return DeliveryResult(
                request_id=request.request_id,
                channel=NotificationChannel.SLACK,
                status=DeliveryStatus.FAILED,
                response_data={},
                error_message="Timeout",
                delivery_time=datetime.now(),
                response_time_ms=int((time.time() - start_time) * 1000),
                retry_count=request.retry_count
            )
        except Exception as e:
            return DeliveryResult(
                request_id=request.request_id,
                channel=NotificationChannel.SLACK,
                status=DeliveryStatus.FAILED,
                response_data={},
                error_message=str(e),
                delivery_time=datetime.now(),
                response_time_ms=int((time.time() - start_time) * 1000),
                retry_count=request.retry_count
            )
    
    def _build_slack_payload(self, request: NotificationRequest) -> Dict[str, Any]:
        """Construit le payload Slack."""
        
        template_data = request.template_data
        priority_colors = {
            Priority.LOW: "#808080",
            Priority.NORMAL: "#00CED1",
            Priority.HIGH: "#FFD700",
            Priority.URGENT: "#FF8C00",
            Priority.CRITICAL: "#FF0000"
        }
        
        # Construction du message principal
        payload = {
            "username": self.default_username,
            "icon_emoji": self.default_icon,
            "text": request.message[:500],  # Limitation Slack
        }
        
        # Ajout des attachments pour un formatage riche
        if template_data:
            attachment = {
                "color": priority_colors.get(request.priority, "#00CED1"),
                "fields": [],
                "footer": "Spotify AI Agent",
                "ts": int(datetime.now().timestamp())
            }
            
            # Ajout des champs d'information
            if "alert_id" in template_data:
                attachment["fields"].append({
                    "title": "Alert ID",
                    "value": template_data["alert_id"],
                    "short": True
                })
            
            if "tenant_id" in template_data:
                attachment["fields"].append({
                    "title": "Tenant",
                    "value": template_data["tenant_id"],
                    "short": True
                })
            
            if "source" in template_data:
                attachment["fields"].append({
                    "title": "Source",
                    "value": template_data["source"],
                    "short": True
                })
            
            if "escalation_count" in template_data and template_data["escalation_count"] > 0:
                attachment["fields"].append({
                    "title": "Escalation",
                    "value": f"Level {template_data['escalation_count']}",
                    "short": True
                })
            
            # Actions (boutons)
            if "alert_url" in template_data:
                attachment["actions"] = [{
                    "type": "button",
                    "text": "View Alert",
                    "url": template_data["alert_url"],
                    "style": "primary"
                }]
            
            payload["attachments"] = [attachment]
        
        return payload

class EmailNotifier:
    """Notificateur Email avec support HTML et pièces jointes."""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
        self.smtp_server = config.credentials.get('smtp_server')
        self.smtp_port = int(config.credentials.get('smtp_port', 587))
        self.username = config.credentials.get('username')
        self.password = config.credentials.get('password')
        self.from_email = config.credentials.get('from_email')
        self.from_name = config.credentials.get('from_name', 'Spotify AI Agent')
    
    async def send_notification(self, request: NotificationRequest) -> DeliveryResult:
        """Envoie une notification par email."""
        start_time = time.time()
        
        try:
            # Construction du message email
            msg = self._build_email_message(request)
            
            # Connexion SMTP
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls(context=ssl.create_default_context())
            server.login(self.username, self.password)
            
            # Envoi
            to_emails = self._get_recipients(request)
            server.send_message(msg, to_addrs=to_emails)
            server.quit()
            
            response_time = int((time.time() - start_time) * 1000)
            
            return DeliveryResult(
                request_id=request.request_id,
                channel=NotificationChannel.EMAIL,
                status=DeliveryStatus.SENT,
                response_data={"recipients": to_emails},
                error_message=None,
                delivery_time=datetime.now(),
                response_time_ms=response_time,
                retry_count=request.retry_count
            )
            
        except Exception as e:
            return DeliveryResult(
                request_id=request.request_id,
                channel=NotificationChannel.EMAIL,
                status=DeliveryStatus.FAILED,
                response_data={},
                error_message=str(e),
                delivery_time=datetime.now(),
                response_time_ms=int((time.time() - start_time) * 1000),
                retry_count=request.retry_count
            )
    
    def _build_email_message(self, request: NotificationRequest) -> MIMEMultipart:
        """Construit le message email."""
        
        template_data = request.template_data
        
        # Message multipart
        msg = MIMEMultipart('alternative')
        msg['Subject'] = self._generate_subject(request)
        msg['From'] = f"{self.from_name} <{self.from_email}>"
        msg['To'] = ", ".join(self._get_recipients(request))
        
        # Version texte
        text_content = self._generate_text_content(request)
        text_part = MIMEText(text_content, 'plain', 'utf-8')
        msg.attach(text_part)
        
        # Version HTML
        html_content = self._generate_html_content(request)
        html_part = MIMEText(html_content, 'html', 'utf-8')
        msg.attach(html_part)
        
        return msg
    
    def _generate_subject(self, request: NotificationRequest) -> str:
        """Génère le sujet de l'email."""
        template_data = request.template_data
        level = template_data.get('level', 'WARNING')
        message = template_data.get('message', request.message)
        
        subject = f"[{level}] Spotify AI Agent Alert"
        if message:
            subject += f" - {message[:50]}"
            if len(message) > 50:
                subject += "..."
        
        return subject
    
    def _generate_text_content(self, request: NotificationRequest) -> str:
        """Génère le contenu texte de l'email."""
        template_data = request.template_data
        
        content = f"""
Spotify AI Agent - Alert Notification

Alert Details:
==============
Level: {template_data.get('level', 'WARNING')}
Message: {request.message}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
Tenant: {template_data.get('tenant_id', 'Unknown')}
Source: {template_data.get('source', 'Unknown')}
Alert ID: {template_data.get('alert_id', 'Unknown')}

{self._format_metadata_text(template_data.get('metadata', {}))}

This is an automated message from Spotify AI Agent.
        """.strip()
        
        return content
    
    def _generate_html_content(self, request: NotificationRequest) -> str:
        """Génère le contenu HTML de l'email."""
        template_data = request.template_data
        
        priority_colors = {
            Priority.LOW: "#808080",
            Priority.NORMAL: "#00CED1", 
            Priority.HIGH: "#FFD700",
            Priority.URGENT: "#FF8C00",
            Priority.CRITICAL: "#FF0000"
        }
        
        color = priority_colors.get(request.priority, "#00CED1")
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Spotify AI Agent Alert</title>
        </head>
        <body style="font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5;">
            <div style="max-width: 600px; margin: 0 auto; background-color: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                
                <!-- Header -->
                <div style="background-color: {color}; color: white; padding: 20px; text-align: center;">
                    <h1 style="margin: 0; font-size: 24px;">🚨 Alert Notification</h1>
                    <p style="margin: 5px 0 0 0; opacity: 0.9;">Spotify AI Agent</p>
                </div>
                
                <!-- Content -->
                <div style="padding: 30px;">
                    <h2 style="color: #333; margin-top: 0;">{template_data.get('level', 'WARNING')} Alert</h2>
                    
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                        <p style="margin: 0; font-size: 16px; line-height: 1.5;">{request.message}</p>
                    </div>
                    
                    <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
                        <tr>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee; font-weight: bold; width: 30%;">Alert ID:</td>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee;">{template_data.get('alert_id', 'Unknown')}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee; font-weight: bold;">Tenant:</td>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee;">{template_data.get('tenant_id', 'Unknown')}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee; font-weight: bold;">Source:</td>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee;">{template_data.get('source', 'Unknown')}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee; font-weight: bold;">Time:</td>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</td>
                        </tr>
                    </table>
                    
                    {self._format_metadata_html(template_data.get('metadata', {}))}
                    
                    <div style="text-align: center; margin-top: 30px;">
                        <a href="{template_data.get('alert_url', '#')}" style="background-color: {color}; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; font-weight: bold;">View Alert Details</a>
                    </div>
                </div>
                
                <!-- Footer -->
                <div style="background-color: #f8f9fa; padding: 20px; text-align: center; font-size: 12px; color: #666;">
                    <p style="margin: 0;">This is an automated message from Spotify AI Agent</p>
                    <p style="margin: 5px 0 0 0;">© 2025 Spotify AI Agent. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _format_metadata_text(self, metadata: Dict[str, Any]) -> str:
        """Formate les métadonnées en texte."""
        if not metadata:
            return ""
        
        lines = ["Metadata:"]
        for key, value in metadata.items():
            lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)
    
    def _format_metadata_html(self, metadata: Dict[str, Any]) -> str:
        """Formate les métadonnées en HTML."""
        if not metadata:
            return ""
        
        html = "<h3>Metadata</h3><ul>"
        for key, value in metadata.items():
            html += f"<li><strong>{key}:</strong> {value}</li>"
        html += "</ul>"
        
        return html
    
    def _get_recipients(self, request: NotificationRequest) -> List[str]:
        """Récupère la liste des destinataires."""
        # Configuration par défaut ou récupération depuis la configuration tenant
        default_recipients = self.config.metadata.get('default_recipients', ['admin@spotify-ai-agent.com'])
        tenant_recipients = request.template_data.get('recipients', [])
        
        recipients = list(set(default_recipients + tenant_recipients))
        return recipients

class WebhookNotifier:
    """Notificateur Webhook avec retry et fallback."""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
        self.endpoint = config.endpoint
        self.headers = config.credentials.get('headers', {})
        self.auth_token = config.credentials.get('auth_token')
    
    async def send_notification(self, request: NotificationRequest) -> DeliveryResult:
        """Envoie une notification via webhook."""
        start_time = time.time()
        
        try:
            payload = self._build_webhook_payload(request)
            headers = self._build_headers()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                ) as response:
                    response_time = int((time.time() - start_time) * 1000)
                    response_data = await response.json() if response.headers.get('content-type', '').startswith('application/json') else await response.text()
                    
                    if 200 <= response.status < 300:
                        return DeliveryResult(
                            request_id=request.request_id,
                            channel=NotificationChannel.WEBHOOK,
                            status=DeliveryStatus.SENT,
                            response_data={"status_code": response.status, "response": response_data},
                            error_message=None,
                            delivery_time=datetime.now(),
                            response_time_ms=response_time,
                            retry_count=request.retry_count
                        )
                    else:
                        return DeliveryResult(
                            request_id=request.request_id,
                            channel=NotificationChannel.WEBHOOK,
                            status=DeliveryStatus.FAILED,
                            response_data={"status_code": response.status, "response": response_data},
                            error_message=f"HTTP {response.status}",
                            delivery_time=datetime.now(),
                            response_time_ms=response_time,
                            retry_count=request.retry_count
                        )
                        
        except Exception as e:
            return DeliveryResult(
                request_id=request.request_id,
                channel=NotificationChannel.WEBHOOK,
                status=DeliveryStatus.FAILED,
                response_data={},
                error_message=str(e),
                delivery_time=datetime.now(),
                response_time_ms=int((time.time() - start_time) * 1000),
                retry_count=request.retry_count
            )
    
    def _build_webhook_payload(self, request: NotificationRequest) -> Dict[str, Any]:
        """Construit le payload webhook."""
        return {
            "request_id": request.request_id,
            "tenant_id": request.tenant_id,
            "alert_id": request.alert_id,
            "priority": request.priority.name,
            "message": request.message,
            "timestamp": datetime.now().isoformat(),
            "data": request.template_data
        }
    
    def _build_headers(self) -> Dict[str, str]:
        """Construit les headers HTTP."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Spotify-AI-Agent/1.0"
        }
        
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        headers.update(self.headers)
        return headers

class NotificationRouter:
    """
    Routeur de notifications ultra-avancé.
    
    Fonctionnalités:
    - Routage intelligent multi-canal
    - Load balancing et failover automatique
    - Rate limiting adaptatif
    - Retry avec backoff exponentiel
    - Health checking des canaux
    - Métriques et monitoring en temps réel
    """
    
    def __init__(self):
        """Initialise le routeur de notifications."""
        self.channel_configs = {}
        self.notifiers = {}
        self.rate_limiter = RateLimiter()
        self.health_checker = HealthChecker()
        self.executor = ThreadPoolExecutor(max_workers=20)
        
        # Métriques
        self.metrics = {
            'notifications_sent': 0,
            'notifications_failed': 0,
            'total_response_time_ms': 0,
            'rate_limit_hits': 0,
            'retries_attempted': 0,
            'fallback_used': 0
        }
        self.delivery_history = deque(maxlen=10000)
        self.metrics_lock = threading.RLock()
        
        # Configuration par défaut
        self._load_default_configs()
        
        logger.info("NotificationRouter initialisé avec succès")
    
    def _load_default_configs(self):
        """Charge les configurations par défaut."""
        
        # Configuration Slack
        slack_config = NotificationConfig(
            channel=NotificationChannel.SLACK,
            endpoint=os.getenv('SLACK_WEBHOOK_DEFAULT', ''),
            credentials={
                'username': os.getenv('SLACK_USERNAME', 'Spotify AI Agent'),
                'icon_emoji': os.getenv('SLACK_ICON_EMOJI', ':robot_face:')
            },
            retry_config={
                'max_retries': 3,
                'backoff_factor': 2,
                'initial_delay': 1
            },
            rate_limits={'requests_per_minute': 60},
            timeout_seconds=30,
            enabled=os.getenv('SLACK_ENABLED', 'true').lower() == 'true',
            fallback_channels=[NotificationChannel.EMAIL],
            templates={},
            metadata={}
        )
        
        # Configuration Email
        email_config = NotificationConfig(
            channel=NotificationChannel.EMAIL,
            endpoint="",  # Non utilisé pour email
            credentials={
                'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
                'smtp_port': os.getenv('SMTP_PORT', '587'),
                'username': os.getenv('SMTP_USERNAME', ''),
                'password': os.getenv('SMTP_PASSWORD', ''),
                'from_email': os.getenv('SMTP_FROM_EMAIL', 'noreply@spotify-ai-agent.com'),
                'from_name': os.getenv('SMTP_FROM_NAME', 'Spotify AI Agent')
            },
            retry_config={
                'max_retries': 2,
                'backoff_factor': 3,
                'initial_delay': 5
            },
            rate_limits={'requests_per_minute': 30},
            timeout_seconds=60,
            enabled=os.getenv('EMAIL_ENABLED', 'true').lower() == 'true',
            fallback_channels=[],
            templates={},
            metadata={
                'default_recipients': ['admin@spotify-ai-agent.com']
            }
        )
        
        # Configuration Webhook
        webhook_config = NotificationConfig(
            channel=NotificationChannel.WEBHOOK,
            endpoint=os.getenv('WEBHOOK_URL', ''),
            credentials={
                'auth_token': os.getenv('WEBHOOK_AUTH_TOKEN', ''),
                'headers': {}
            },
            retry_config={
                'max_retries': 3,
                'backoff_factor': 2,
                'initial_delay': 2
            },
            rate_limits={'requests_per_minute': 100},
            timeout_seconds=30,
            enabled=os.getenv('WEBHOOK_ENABLED', 'false').lower() == 'true',
            fallback_channels=[NotificationChannel.EMAIL],
            templates={},
            metadata={}
        )
        
        # Enregistrement des configurations
        self.register_channel(slack_config)
        self.register_channel(email_config)
        self.register_channel(webhook_config)
    
    def register_channel(self, config: NotificationConfig):
        """Enregistre un canal de notification."""
        
        if not config.enabled:
            logger.info(f"Canal {config.channel.value} désactivé")
            return
        
        self.channel_configs[config.channel] = config
        
        # Initialisation du notificateur
        if config.channel == NotificationChannel.SLACK:
            self.notifiers[config.channel] = SlackNotifier(config)
        elif config.channel == NotificationChannel.EMAIL:
            self.notifiers[config.channel] = EmailNotifier(config)
        elif config.channel == NotificationChannel.WEBHOOK:
            self.notifiers[config.channel] = WebhookNotifier(config)
        
        # Configuration du rate limiting
        rate_limit = config.rate_limits.get('requests_per_minute', 60)
        self.rate_limiter.set_limit(config.channel, rate_limit)
        
        logger.info(f"Canal {config.channel.value} enregistré avec succès")
    
    async def send_notification(self, request: NotificationRequest) -> List[DeliveryResult]:
        """Envoie une notification sur tous les canaux spécifiés."""
        
        results = []
        tasks = []
        
        for channel in request.channels:
            if channel not in self.channel_configs:
                logger.warning(f"Canal {channel.value} non configuré")
                continue
            
            if not self.channel_configs[channel].enabled:
                logger.info(f"Canal {channel.value} désactivé")
                continue
            
            # Vérification du rate limiting
            if not self.rate_limiter.can_send(channel):
                self._increment_metric('rate_limit_hits')
                logger.warning(f"Rate limit atteint pour {channel.value}")
                
                # Tentative avec les canaux de fallback
                fallback_channels = self.channel_configs[channel].fallback_channels
                for fallback in fallback_channels:
                    if fallback in self.channel_configs and self.rate_limiter.can_send(fallback):
                        tasks.append(self._send_to_channel(request, fallback))
                        self._increment_metric('fallback_used')
                        break
                continue
            
            # Vérification de la santé du canal
            if not self.health_checker.is_healthy(channel):
                logger.warning(f"Canal {channel.value} en mauvaise santé")
                
                # Tentative avec les canaux de fallback
                fallback_channels = self.channel_configs[channel].fallback_channels
                for fallback in fallback_channels:
                    if fallback in self.channel_configs and self.health_checker.is_healthy(fallback):
                        tasks.append(self._send_to_channel(request, fallback))
                        self._increment_metric('fallback_used')
                        break
                continue
            
            # Envoi normal
            tasks.append(self._send_to_channel(request, channel))
        
        # Exécution asynchrone de tous les envois
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filtrage des exceptions
            valid_results = []
            for result in results:
                if isinstance(result, DeliveryResult):
                    valid_results.append(result)
                    self._record_delivery_result(result)
                elif isinstance(result, Exception):
                    logger.error(f"Erreur envoi notification: {result}")
            
            results = valid_results
        
        return results
    
    async def _send_to_channel(self, request: NotificationRequest, 
                             channel: NotificationChannel) -> DeliveryResult:
        """Envoie une notification à un canal spécifique avec retry."""
        
        notifier = self.notifiers.get(channel)
        if not notifier:
            return DeliveryResult(
                request_id=request.request_id,
                channel=channel,
                status=DeliveryStatus.FAILED,
                response_data={},
                error_message="Notificateur non disponible",
                delivery_time=datetime.now(),
                response_time_ms=0,
                retry_count=0
            )
        
        config = self.channel_configs[channel]
        retry_config = config.retry_config
        max_retries = retry_config.get('max_retries', 3)
        backoff_factor = retry_config.get('backoff_factor', 2)
        initial_delay = retry_config.get('initial_delay', 1)
        
        last_result = None
        
        for attempt in range(max_retries + 1):
            try:
                request.retry_count = attempt
                result = await notifier.send_notification(request)
                
                if result.status == DeliveryStatus.SENT:
                    self._increment_metric('notifications_sent')
                    return result
                else:
                    last_result = result
                    self._increment_metric('notifications_failed')
                    
                    if attempt < max_retries:
                        self._increment_metric('retries_attempted')
                        delay = initial_delay * (backoff_factor ** attempt)
                        await asyncio.sleep(delay)
                        logger.info(f"Retry {attempt + 1}/{max_retries} pour {request.request_id} sur {channel.value}")
                
            except Exception as e:
                last_result = DeliveryResult(
                    request_id=request.request_id,
                    channel=channel,
                    status=DeliveryStatus.FAILED,
                    response_data={},
                    error_message=str(e),
                    delivery_time=datetime.now(),
                    response_time_ms=0,
                    retry_count=attempt
                )
                
                if attempt < max_retries:
                    delay = initial_delay * (backoff_factor ** attempt)
                    await asyncio.sleep(delay)
        
        return last_result or DeliveryResult(
            request_id=request.request_id,
            channel=channel,
            status=DeliveryStatus.FAILED,
            response_data={},
            error_message="Échec après tous les retries",
            delivery_time=datetime.now(),
            response_time_ms=0,
            retry_count=max_retries
        )
    
    def create_notification_request(self, tenant_id: str, alert_id: str,
                                  message: str, channels: List[NotificationChannel],
                                  priority: Priority = Priority.NORMAL,
                                  template_data: Dict[str, Any] = None) -> NotificationRequest:
        """Crée une requête de notification."""
        
        request_id = self._generate_request_id(tenant_id, alert_id)
        
        return NotificationRequest(
            request_id=request_id,
            tenant_id=tenant_id,
            alert_id=alert_id,
            channels=channels,
            priority=priority,
            message=message,
            template_data=template_data or {},
            delivery_time=None
        )
    
    def _generate_request_id(self, tenant_id: str, alert_id: str) -> str:
        """Génère un ID unique pour la requête."""
        timestamp = int(time.time() * 1000)
        data = f"{tenant_id}:{alert_id}:{timestamp}"
        hash_obj = hashlib.md5(data.encode())
        return f"notif_{hash_obj.hexdigest()[:12]}"
    
    def _record_delivery_result(self, result: DeliveryResult):
        """Enregistre le résultat de livraison."""
        with self.metrics_lock:
            self.delivery_history.append({
                'timestamp': result.delivery_time,
                'channel': result.channel.value,
                'status': result.status.value,
                'response_time_ms': result.response_time_ms,
                'retry_count': result.retry_count
            })
            
            self.metrics['total_response_time_ms'] += result.response_time_ms
    
    def _increment_metric(self, metric_name: str):
        """Incrémente une métrique."""
        with self.metrics_lock:
            self.metrics[metric_name] = self.metrics.get(metric_name, 0) + 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques du routeur."""
        with self.metrics_lock:
            metrics = self.metrics.copy()
            
            # Calcul des métriques dérivées
            total_notifications = metrics['notifications_sent'] + metrics['notifications_failed']
            if total_notifications > 0:
                metrics['success_rate'] = metrics['notifications_sent'] / total_notifications
                metrics['avg_response_time_ms'] = metrics['total_response_time_ms'] / total_notifications
            else:
                metrics['success_rate'] = 0
                metrics['avg_response_time_ms'] = 0
            
            metrics['active_channels'] = len([c for c in self.channel_configs.values() if c.enabled])
            metrics['delivery_history_size'] = len(self.delivery_history)
            
            return metrics
    
    def get_channel_health(self) -> Dict[str, bool]:
        """Retourne l'état de santé de tous les canaux."""
        health_status = {}
        for channel in self.channel_configs:
            health_status[channel.value] = self.health_checker.is_healthy(channel)
        return health_status
    
    async def health_check(self) -> Dict[str, Any]:
        """Effectue une vérification complète de santé."""
        health_results = {}
        
        for channel, config in self.channel_configs.items():
            if config.enabled:
                is_healthy = await self.health_checker.check_channel_health(channel, config)
                self.health_checker.set_health_status(channel, is_healthy)
                health_results[channel.value] = is_healthy
        
        overall_health = all(health_results.values()) if health_results else False
        
        return {
            "overall_healthy": overall_health,
            "channels": health_results,
            "timestamp": datetime.now().isoformat()
        }
    
    def cleanup(self):
        """Nettoie les ressources."""
        self.executor.shutdown(wait=True)
        logger.info("NotificationRouter nettoyé avec succès")

# Factory function
def create_notification_router() -> NotificationRouter:
    """Factory function pour créer un routeur de notifications."""
    return NotificationRouter()
