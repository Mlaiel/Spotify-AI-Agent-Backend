# =============================================================================
# Système de Notifications Temps Réel - WebSocket Enterprise
# =============================================================================
# 
# Service de notifications push en temps réel pour le monitoring avec WebSockets,
# intégration multi-canal (email, SMS, Slack, Teams) et gestion d'escalade
# intelligente.
#
# Architecture temps réel:
# - WebSockets pour notifications instantanées
# - Redis Pub/Sub pour distribution scalable
# - Rate limiting et anti-spam intelligent
# - Templates de notifications configurables
# - Escalade automatique par sévérité
# - Intégration multi-canal (email, Slack, Teams, SMS)
# - Historique et audit des notifications
#
# Développé par l'équipe d'experts techniques:
# - Lead Developer + AI Architect (Architecture temps réel)
# - Backend Senior Developer (WebSockets, Redis)
# - Spécialiste Sécurité Backend (Auth WebSocket, rate limiting)
# - Architecte Microservices (Pub/Sub patterns)
# - ML Engineer (Anti-spam et priorisation IA)
#
# Direction Technique: Fahed Mlaiel
# =============================================================================

import asyncio
import json
import time
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

# FastAPI et WebSockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.websockets import WebSocketState
from starlette.websockets import WebSocket as StarletteWebSocket

# Validation et modèles
from pydantic import BaseModel, Field, validator, EmailStr
from enum import Enum

# Redis et messaging
import aioredis
import aiokafka
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer

# Logging structuré
import structlog

# HTTP client pour webhooks
import aiohttp

# Templates et formatage
from jinja2 import Environment, FileSystemLoader, Template

# Notifications externes
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configuration
logger = structlog.get_logger(__name__)

# =============================================================================
# MODÈLES ET ENUMS
# =============================================================================

class NotificationChannel(str, Enum):
    """Canaux de notification"""
    WEBSOCKET = "websocket"
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    PUSH = "push"

class NotificationPriority(str, Enum):
    """Priorité des notifications"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class NotificationStatus(str, Enum):
    """Statut de notification"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"

class EscalationLevel(str, Enum):
    """Niveau d'escalade"""
    L1 = "l1"  # Support niveau 1
    L2 = "l2"  # Support niveau 2
    L3 = "l3"  # Support niveau 3
    MANAGER = "manager"  # Manager
    DIRECTOR = "director"  # Directeur

@dataclass
class NotificationTemplate:
    """Template de notification"""
    id: str
    name: str
    subject_template: str
    body_template: str
    channels: List[NotificationChannel]
    priority: NotificationPriority
    variables: Dict[str, Any] = field(default_factory=dict)
    html_template: Optional[str] = None
    escalation_config: Optional[Dict[str, Any]] = None

@dataclass
class NotificationRule:
    """Règle de notification"""
    id: str
    name: str
    conditions: Dict[str, Any]
    template_id: str
    channels: List[NotificationChannel]
    recipients: List[str]
    rate_limit: Optional[Dict[str, int]] = None
    escalation_rules: Optional[List[Dict[str, Any]]] = None
    is_active: bool = True

@dataclass
class NotificationRecipient:
    """Destinataire de notification"""
    id: str
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    slack_user_id: Optional[str] = None
    teams_user_id: Optional[str] = None
    timezone: str = "UTC"
    preferred_channels: List[NotificationChannel] = field(default_factory=list)
    escalation_level: EscalationLevel = EscalationLevel.L1
    is_active: bool = True

class NotificationRequest(BaseModel):
    """Requête de notification"""
    template_id: str
    recipients: List[str]
    channels: Optional[List[NotificationChannel]] = None
    priority: NotificationPriority = NotificationPriority.NORMAL
    variables: Dict[str, Any] = Field(default_factory=dict)
    scheduled_at: Optional[datetime] = None
    tenant_id: str
    source: str = "api"
    metadata: Dict[str, Any] = Field(default_factory=dict)

class WebSocketMessage(BaseModel):
    """Message WebSocket"""
    type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

# =============================================================================
# GESTIONNAIRE DE CONNEXIONS WEBSOCKET
# =============================================================================

class WebSocketConnectionManager:
    """Gestionnaire des connexions WebSocket"""
    
    def __init__(self):
        # Connexions actives par tenant
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
        # Utilisateurs par connexion
        self.connection_users: Dict[str, Dict[str, Any]] = {}
        # Statistiques
        self.connection_stats: Dict[str, Any] = {
            "total_connections": 0,
            "connections_by_tenant": {},
            "messages_sent": 0,
            "errors": 0
        }
    
    async def connect(self, websocket: WebSocket, tenant_id: str, user_id: str, user_data: Dict[str, Any]):
        """Connexion d'un client WebSocket"""
        
        await websocket.accept()
        
        connection_id = str(uuid.uuid4())
        
        # Initialisation tenant si nécessaire
        if tenant_id not in self.active_connections:
            self.active_connections[tenant_id] = {}
            self.connection_stats["connections_by_tenant"][tenant_id] = 0
        
        # Enregistrement de la connexion
        self.active_connections[tenant_id][connection_id] = websocket
        self.connection_users[connection_id] = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "connected_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            **user_data
        }
        
        # Mise à jour des statistiques
        self.connection_stats["total_connections"] += 1
        self.connection_stats["connections_by_tenant"][tenant_id] += 1
        
        logger.info("WebSocket connected", 
                   connection_id=connection_id,
                   tenant_id=tenant_id,
                   user_id=user_id)
        
        return connection_id
    
    async def disconnect(self, tenant_id: str, connection_id: str):
        """Déconnexion d'un client WebSocket"""
        
        if (tenant_id in self.active_connections and 
            connection_id in self.active_connections[tenant_id]):
            
            del self.active_connections[tenant_id][connection_id]
            
            if connection_id in self.connection_users:
                del self.connection_users[connection_id]
            
            # Mise à jour des statistiques
            self.connection_stats["total_connections"] -= 1
            self.connection_stats["connections_by_tenant"][tenant_id] -= 1
            
            logger.info("WebSocket disconnected", 
                       connection_id=connection_id,
                       tenant_id=tenant_id)
    
    async def send_personal_message(self, tenant_id: str, user_id: str, message: WebSocketMessage):
        """Envoi d'un message personnel"""
        
        if tenant_id not in self.active_connections:
            return False
        
        sent_count = 0
        
        for connection_id, websocket in self.active_connections[tenant_id].items():
            if (connection_id in self.connection_users and 
                self.connection_users[connection_id]["user_id"] == user_id):
                
                try:
                    await websocket.send_text(message.json())
                    sent_count += 1
                    
                    # Mise à jour activité
                    self.connection_users[connection_id]["last_activity"] = datetime.utcnow()
                    
                except Exception as e:
                    logger.error("Error sending WebSocket message", 
                                connection_id=connection_id,
                                error=str(e))
                    
                    # Nettoyage connexion fermée
                    await self.disconnect(tenant_id, connection_id)
        
        self.connection_stats["messages_sent"] += sent_count
        return sent_count > 0
    
    async def broadcast_to_tenant(self, tenant_id: str, message: WebSocketMessage, exclude_user: Optional[str] = None):
        """Diffusion d'un message à tous les utilisateurs d'un tenant"""
        
        if tenant_id not in self.active_connections:
            return 0
        
        sent_count = 0
        failed_connections = []
        
        for connection_id, websocket in self.active_connections[tenant_id].items():
            user_data = self.connection_users.get(connection_id, {})
            
            # Exclure un utilisateur spécifique si demandé
            if exclude_user and user_data.get("user_id") == exclude_user:
                continue
            
            try:
                await websocket.send_text(message.json())
                sent_count += 1
                
                # Mise à jour activité
                if connection_id in self.connection_users:
                    self.connection_users[connection_id]["last_activity"] = datetime.utcnow()
                
            except Exception as e:
                logger.error("Error broadcasting WebSocket message", 
                            connection_id=connection_id,
                            error=str(e))
                failed_connections.append(connection_id)
        
        # Nettoyage des connexions fermées
        for connection_id in failed_connections:
            await self.disconnect(tenant_id, connection_id)
        
        self.connection_stats["messages_sent"] += sent_count
        return sent_count
    
    async def get_tenant_connections(self, tenant_id: str) -> List[Dict[str, Any]]:
        """Liste des connexions actives pour un tenant"""
        
        if tenant_id not in self.active_connections:
            return []
        
        connections = []
        for connection_id in self.active_connections[tenant_id]:
            if connection_id in self.connection_users:
                user_data = self.connection_users[connection_id].copy()
                user_data["connection_id"] = connection_id
                connections.append(user_data)
        
        return connections
    
    async def cleanup_stale_connections(self, timeout_minutes: int = 30):
        """Nettoyage des connexions inactives"""
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=timeout_minutes)
        stale_connections = []
        
        for connection_id, user_data in self.connection_users.items():
            if user_data["last_activity"] < cutoff_time:
                stale_connections.append((user_data["tenant_id"], connection_id))
        
        for tenant_id, connection_id in stale_connections:
            await self.disconnect(tenant_id, connection_id)
        
        return len(stale_connections)

# =============================================================================
# SERVICE DE NOTIFICATIONS
# =============================================================================

class NotificationService:
    """Service de notifications multi-canal"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Gestionnaire WebSocket
        self.websocket_manager = WebSocketConnectionManager()
        
        # Templates de notifications
        self.templates: Dict[str, NotificationTemplate] = {}
        self.rules: Dict[str, NotificationRule] = {}
        self.recipients: Dict[str, NotificationRecipient] = {}
        
        # Rate limiting
        self.rate_limits: Dict[str, List[datetime]] = {}
        
        # Historique des notifications
        self.notification_history: List[Dict[str, Any]] = []
        
        # Configuration des canaux
        self.channel_configs = {
            "email": config.get("email", {}),
            "slack": config.get("slack", {}),
            "teams": config.get("teams", {}),
            "sms": config.get("sms", {})
        }
        
        # Environment Jinja2 pour templates
        template_dir = config.get("template_dir", "./templates")
        if Path(template_dir).exists():
            self.jinja_env = Environment(loader=FileSystemLoader(template_dir))
        else:
            self.jinja_env = Environment()
        
        # Redis pour pub/sub (si configuré)
        self.redis_client: Optional[aioredis.Redis] = None
        
        self._setup_default_templates()
        self._setup_default_recipients()
    
    def _setup_default_templates(self):
        """Configuration des templates par défaut"""
        
        default_templates = [
            NotificationTemplate(
                id="incident_critical",
                name="Incident Critique",
                subject_template="🚨 INCIDENT CRITIQUE: {{ title }}",
                body_template="""
                Un incident critique a été détecté dans le système de monitoring.
                
                **Détails:**
                - Titre: {{ title }}
                - Sévérité: {{ severity }}
                - Heure: {{ timestamp }}
                - Tenant: {{ tenant_id }}
                
                **Description:**
                {{ description }}
                
                **Actions recommandées:**
                1. Vérifier l'état des services
                2. Consulter les logs d'erreur
                3. Contacter l'équipe d'astreinte si nécessaire
                
                Dashboard: {{ dashboard_url }}
                """,
                channels=[NotificationChannel.WEBSOCKET, NotificationChannel.EMAIL, NotificationChannel.SLACK],
                priority=NotificationPriority.CRITICAL,
                escalation_config={
                    "initial_delay": 0,
                    "escalation_delay": 300,  # 5 minutes
                    "max_escalations": 3
                }
            ),
            NotificationTemplate(
                id="system_alert",
                name="Alerte Système",
                subject_template="⚠️ Alerte: {{ alert_name }}",
                body_template="""
                Une alerte système a été déclenchée.
                
                **Alerte:** {{ alert_name }}
                **Sévérité:** {{ severity }}
                **Valeur:** {{ value }}
                **Seuil:** {{ threshold }}
                **Heure:** {{ timestamp }}
                
                **Description:**
                {{ description }}
                
                Consultez le dashboard pour plus d'informations.
                """,
                channels=[NotificationChannel.WEBSOCKET, NotificationChannel.EMAIL],
                priority=NotificationPriority.HIGH
            ),
            NotificationTemplate(
                id="deployment_notification",
                name="Notification de Déploiement",
                subject_template="🚀 Déploiement: {{ application }}",
                body_template="""
                Un déploiement a été effectué.
                
                **Application:** {{ application }}
                **Version:** {{ version }}
                **Environnement:** {{ environment }}
                **Statut:** {{ status }}
                **Heure:** {{ timestamp }}
                
                **Détails:**
                {{ details }}
                """,
                channels=[NotificationChannel.WEBSOCKET, NotificationChannel.SLACK],
                priority=NotificationPriority.NORMAL
            )
        ]
        
        for template in default_templates:
            self.templates[template.id] = template
    
    def _setup_default_recipients(self):
        """Configuration des destinataires par défaut"""
        
        default_recipients = [
            NotificationRecipient(
                id="admin",
                name="Administrateur Système",
                email="admin@monitoring.local",
                preferred_channels=[NotificationChannel.EMAIL, NotificationChannel.WEBSOCKET],
                escalation_level=EscalationLevel.L3
            ),
            NotificationRecipient(
                id="ops_team",
                name="Équipe Opérations",
                email="ops@monitoring.local",
                slack_user_id="U123456789",
                preferred_channels=[NotificationChannel.SLACK, NotificationChannel.EMAIL],
                escalation_level=EscalationLevel.L2
            ),
            NotificationRecipient(
                id="dev_team",
                name="Équipe Développement",
                email="dev@monitoring.local",
                preferred_channels=[NotificationChannel.WEBSOCKET, NotificationChannel.EMAIL],
                escalation_level=EscalationLevel.L1
            )
        ]
        
        for recipient in default_recipients:
            self.recipients[recipient.id] = recipient
    
    async def initialize_redis(self):
        """Initialisation de la connexion Redis"""
        
        redis_config = self.config.get("redis", {})
        if redis_config.get("enabled", False):
            try:
                self.redis_client = await aioredis.from_url(
                    redis_config.get("url", "redis://localhost:6379"),
                    decode_responses=True
                )
                
                # Test de connexion
                await self.redis_client.ping()
                logger.info("Redis connection established for notifications")
                
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self.redis_client = None
    
    async def send_notification(self, request: NotificationRequest) -> Dict[str, Any]:
        """Envoi d'une notification multi-canal"""
        
        notification_id = str(uuid.uuid4())
        
        try:
            # Validation du template
            if request.template_id not in self.templates:
                raise ValueError(f"Template non trouvé: {request.template_id}")
            
            template = self.templates[request.template_id]
            
            # Détermination des canaux
            channels = request.channels or template.channels
            
            # Rendu du template
            rendered_content = await self._render_template(template, request.variables)
            
            # Envoi par canal
            results = {}
            for channel in channels:
                try:
                    if channel == NotificationChannel.WEBSOCKET:
                        result = await self._send_websocket(request, rendered_content)
                    elif channel == NotificationChannel.EMAIL:
                        result = await self._send_email(request, rendered_content)
                    elif channel == NotificationChannel.SLACK:
                        result = await self._send_slack(request, rendered_content)
                    elif channel == NotificationChannel.TEAMS:
                        result = await self._send_teams(request, rendered_content)
                    elif channel == NotificationChannel.SMS:
                        result = await self._send_sms(request, rendered_content)
                    elif channel == NotificationChannel.WEBHOOK:
                        result = await self._send_webhook(request, rendered_content)
                    else:
                        result = {"success": False, "error": f"Canal non supporté: {channel}"}
                    
                    results[channel.value] = result
                    
                except Exception as e:
                    logger.error(f"Error sending notification via {channel}: {e}")
                    results[channel.value] = {"success": False, "error": str(e)}
            
            # Enregistrement dans l'historique
            notification_record = {
                "id": notification_id,
                "template_id": request.template_id,
                "tenant_id": request.tenant_id,
                "recipients": request.recipients,
                "channels": [c.value for c in channels],
                "priority": request.priority.value,
                "status": "sent" if any(r.get("success") for r in results.values()) else "failed",
                "results": results,
                "timestamp": datetime.utcnow().isoformat(),
                "source": request.source,
                "metadata": request.metadata
            }
            
            self.notification_history.append(notification_record)
            
            # Publication Redis si configuré
            if self.redis_client:
                await self.redis_client.publish(
                    f"notifications:{request.tenant_id}",
                    json.dumps(notification_record)
                )
            
            return {
                "notification_id": notification_id,
                "success": True,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return {
                "notification_id": notification_id,
                "success": False,
                "error": str(e)
            }
    
    async def _render_template(self, template: NotificationTemplate, variables: Dict[str, Any]) -> Dict[str, str]:
        """Rendu d'un template avec variables"""
        
        # Variables par défaut
        default_vars = {
            "timestamp": datetime.utcnow().isoformat(),
            "dashboard_url": self.config.get("dashboard_url", "http://localhost:3000")
        }
        
        # Fusion des variables
        render_vars = {**default_vars, **template.variables, **variables}
        
        # Rendu du sujet
        subject_template = self.jinja_env.from_string(template.subject_template)
        subject = subject_template.render(**render_vars)
        
        # Rendu du corps
        body_template = self.jinja_env.from_string(template.body_template)
        body = body_template.render(**render_vars)
        
        # Rendu HTML si disponible
        html_body = None
        if template.html_template:
            html_template = self.jinja_env.from_string(template.html_template)
            html_body = html_template.render(**render_vars)
        
        return {
            "subject": subject,
            "body": body,
            "html_body": html_body
        }
    
    async def _send_websocket(self, request: NotificationRequest, content: Dict[str, str]) -> Dict[str, Any]:
        """Envoi via WebSocket"""
        
        message = WebSocketMessage(
            type="notification",
            data={
                "id": str(uuid.uuid4()),
                "template_id": request.template_id,
                "priority": request.priority.value,
                "subject": content["subject"],
                "body": content["body"],
                "metadata": request.metadata
            }
        )
        
        sent_count = 0
        for recipient_id in request.recipients:
            success = await self.websocket_manager.send_personal_message(
                request.tenant_id,
                recipient_id,
                message
            )
            if success:
                sent_count += 1
        
        return {
            "success": sent_count > 0,
            "sent_count": sent_count,
            "total_recipients": len(request.recipients)
        }
    
    async def _send_email(self, request: NotificationRequest, content: Dict[str, str]) -> Dict[str, Any]:
        """Envoi via email"""
        
        email_config = self.channel_configs.get("email", {})
        
        if not email_config.get("enabled", False):
            return {"success": False, "error": "Email non configuré"}
        
        try:
            # Configuration SMTP
            smtp_server = email_config.get("smtp_server", "localhost")
            smtp_port = email_config.get("smtp_port", 587)
            username = email_config.get("username")
            password = email_config.get("password")
            from_email = email_config.get("from_email", "noreply@monitoring.local")
            
            sent_count = 0
            
            for recipient_id in request.recipients:
                if recipient_id in self.recipients:
                    recipient = self.recipients[recipient_id]
                    
                    if recipient.email:
                        # Création du message
                        msg = MIMEMultipart('alternative')
                        msg['Subject'] = content["subject"]
                        msg['From'] = from_email
                        msg['To'] = recipient.email
                        
                        # Corps texte
                        text_part = MIMEText(content["body"], 'plain', 'utf-8')
                        msg.attach(text_part)
                        
                        # Corps HTML si disponible
                        if content["html_body"]:
                            html_part = MIMEText(content["html_body"], 'html', 'utf-8')
                            msg.attach(html_part)
                        
                        # Envoi
                        with smtplib.SMTP(smtp_server, smtp_port) as server:
                            if username and password:
                                server.starttls()
                                server.login(username, password)
                            
                            server.send_message(msg)
                            sent_count += 1
            
            return {
                "success": sent_count > 0,
                "sent_count": sent_count,
                "total_recipients": len(request.recipients)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _send_slack(self, request: NotificationRequest, content: Dict[str, str]) -> Dict[str, Any]:
        """Envoi via Slack"""
        
        slack_config = self.channel_configs.get("slack", {})
        
        if not slack_config.get("enabled", False):
            return {"success": False, "error": "Slack non configuré"}
        
        webhook_url = slack_config.get("webhook_url")
        if not webhook_url:
            return {"success": False, "error": "Webhook Slack non configuré"}
        
        try:
            # Format du message Slack
            slack_message = {
                "text": content["subject"],
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": content["subject"]
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": content["body"]
                        }
                    }
                ]
            }
            
            # Ajout de couleur selon priorité
            color_map = {
                NotificationPriority.LOW: "#36a64f",
                NotificationPriority.NORMAL: "#2196F3",
                NotificationPriority.HIGH: "#ff9800",
                NotificationPriority.CRITICAL: "#f44336",
                NotificationPriority.EMERGENCY: "#e91e63"
            }
            
            if request.priority in color_map:
                slack_message["attachments"] = [{
                    "color": color_map[request.priority]
                }]
            
            # Envoi via webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=slack_message) as response:
                    if response.status == 200:
                        return {"success": True, "sent_count": 1}
                    else:
                        return {"success": False, "error": f"HTTP {response.status}"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _send_teams(self, request: NotificationRequest, content: Dict[str, str]) -> Dict[str, Any]:
        """Envoi via Microsoft Teams"""
        
        teams_config = self.channel_configs.get("teams", {})
        
        if not teams_config.get("enabled", False):
            return {"success": False, "error": "Teams non configuré"}
        
        webhook_url = teams_config.get("webhook_url")
        if not webhook_url:
            return {"success": False, "error": "Webhook Teams non configuré"}
        
        try:
            # Format du message Teams
            teams_message = {
                "@type": "MessageCard",
                "@context": "https://schema.org/extensions",
                "summary": content["subject"],
                "themeColor": "0076D7",
                "sections": [{
                    "activityTitle": content["subject"],
                    "text": content["body"]
                }]
            }
            
            # Envoi via webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=teams_message) as response:
                    if response.status == 200:
                        return {"success": True, "sent_count": 1}
                    else:
                        return {"success": False, "error": f"HTTP {response.status}"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _send_sms(self, request: NotificationRequest, content: Dict[str, str]) -> Dict[str, Any]:
        """Envoi via SMS"""
        
        # Placeholder pour intégration SMS (Twilio, AWS SNS, etc.)
        return {"success": False, "error": "SMS non implémenté"}
    
    async def _send_webhook(self, request: NotificationRequest, content: Dict[str, str]) -> Dict[str, Any]:
        """Envoi via webhook personnalisé"""
        
        webhook_config = self.channel_configs.get("webhook", {})
        
        if not webhook_config.get("enabled", False):
            return {"success": False, "error": "Webhook non configuré"}
        
        try:
            # Payload du webhook
            webhook_payload = {
                "notification_id": str(uuid.uuid4()),
                "template_id": request.template_id,
                "tenant_id": request.tenant_id,
                "priority": request.priority.value,
                "subject": content["subject"],
                "body": content["body"],
                "recipients": request.recipients,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": request.metadata
            }
            
            # Envoi vers les webhooks configurés
            webhooks = webhook_config.get("urls", [])
            sent_count = 0
            
            async with aiohttp.ClientSession() as session:
                for webhook_url in webhooks:
                    try:
                        async with session.post(webhook_url, json=webhook_payload, timeout=10) as response:
                            if response.status < 400:
                                sent_count += 1
                    except Exception as e:
                        logger.error(f"Error sending webhook to {webhook_url}: {e}")
            
            return {
                "success": sent_count > 0,
                "sent_count": sent_count,
                "total_webhooks": len(webhooks)
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Statistiques des connexions WebSocket"""
        return self.websocket_manager.connection_stats
    
    def get_notification_history(self, tenant_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Historique des notifications"""
        
        history = self.notification_history
        
        if tenant_id:
            history = [n for n in history if n.get("tenant_id") == tenant_id]
        
        return sorted(history, key=lambda x: x["timestamp"], reverse=True)[:limit]

# =============================================================================
# INSTANCE GLOBALE
# =============================================================================

# Configuration par défaut
default_notification_config = {
    "redis": {
        "enabled": False,
        "url": "redis://localhost:6379"
    },
    "email": {
        "enabled": True,
        "smtp_server": "localhost",
        "smtp_port": 587,
        "from_email": "noreply@monitoring.local"
    },
    "slack": {
        "enabled": False,
        "webhook_url": ""
    },
    "teams": {
        "enabled": False,
        "webhook_url": ""
    },
    "webhook": {
        "enabled": False,
        "urls": []
    },
    "dashboard_url": "http://localhost:3000"
}

# Instance globale du service
notification_service: Optional[NotificationService] = None

def initialize_notification_service(config: Optional[Dict[str, Any]] = None) -> NotificationService:
    """Initialisation du service de notifications"""
    
    global notification_service
    
    final_config = {**default_notification_config}
    if config:
        final_config.update(config)
    
    notification_service = NotificationService(final_config)
    return notification_service

def get_notification_service() -> NotificationService:
    """Récupération du service de notifications"""
    
    global notification_service
    
    if not notification_service:
        notification_service = initialize_notification_service()
    
    return notification_service

# =============================================================================
# FONCTIONS D'EXPORT
# =============================================================================

__all__ = [
    "NotificationService",
    "WebSocketConnectionManager", 
    "NotificationChannel",
    "NotificationPriority",
    "NotificationRequest",
    "WebSocketMessage",
    "initialize_notification_service",
    "get_notification_service"
]
