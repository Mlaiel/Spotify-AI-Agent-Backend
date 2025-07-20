"""
Module de Notifications Avancées - Spotify AI Agent
=====================================================

Module ultra-avancé pour la gestion complète des notifications multi-canal,
alertes en temps réel, et monitoring intelligent.

Fonctionnalités clés:
- Notifications multi-canal (Slack, Email, SMS, Push, Discord, Teams)
- Alertes intelligentes avec escalade automatique
- Templates personnalisables et localisés
- Métriques et analytics en temps réel
- Rate limiting et throttling intelligent
- Retry automatique avec backoff exponentiel
- Circuit breaker pour la résilience
- Audit trail complet
- Intégration ML pour priorisation intelligente

Architecture:
- Patterns: Factory, Observer, Strategy, Template Method
- Event-driven avec bus d'événements
- Queue management avec Redis/RabbitMQ
- Monitoring avec Prometheus/Grafana
- Logging structuré avec OpenTelemetry

Auteurs:
- Lead Dev & Architecte IA: Fahed Mlaiel
- Développeur Backend Senior: Fahed Mlaiel
- Ingénieur ML: Fahed Mlaiel
- Architecte Microservices: Fahed Mlaiel

Version: 1.0.0
Dernière mise à jour: 2025-07-19
"""

# Core imports
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union, Set, Callable, Protocol
from uuid import UUID, uuid4
from enum import Enum, IntEnum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio
import json
import logging
from pathlib import Path

# External imports
from pydantic import BaseModel, Field, validator, root_validator
from sqlalchemy import Column, String, DateTime, JSON, Boolean, Integer, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge

# Internal imports
from .models import *
from .schemas import *
from .services import *
from .channels import *
from .templates import *
from .analytics import *
from .validators import *
from .processors import *
from .middleware import *
from .config import *

from pydantic import BaseModel, Field, validator, root_validator, computed_field
from pydantic.types import EmailStr, HttpUrl, StrictStr

from ..base import BaseSchema, TimestampMixin, TenantMixin
from ..base.enums import NotificationChannel, AlertLevel, Priority


class NotificationStatus(str, Enum):
    """Statuts de notification"""
    PENDING = "pending"
    QUEUED = "queued"
    SENDING = "sending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"
    RATE_LIMITED = "rate_limited"
    
    @property
    def is_final(self) -> bool:
        """Indique si le statut est final"""
        return self in {self.SENT, self.DELIVERED, self.FAILED, self.CANCELLED}
    
    @property
    def is_error(self) -> bool:
        """Indique si le statut représente une erreur"""
        return self in {self.FAILED, self.RATE_LIMITED}


class NotificationTemplate(BaseSchema):
    """Template de notification personnalisable"""
    name: StrictStr = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    channel: NotificationChannel = Field(...)
    
    # Template content
    subject_template: Optional[str] = Field(None, max_length=255)
    body_template: str = Field(..., min_length=1, max_length=10000)
    html_template: Optional[str] = Field(None, max_length=50000)
    
    # Formatage spécifique au canal
    slack_template: Optional[Dict[str, Any]] = Field(None)
    email_template: Optional[Dict[str, Any]] = Field(None)
    teams_template: Optional[Dict[str, Any]] = Field(None)
    
    # Variables disponibles
    available_variables: List[str] = Field(default_factory=list)
    required_variables: List[str] = Field(default_factory=list)
    
    # Configuration
    is_active: bool = Field(True)
    priority: Priority = Field(Priority.NORMAL)
    locale: str = Field("en_US", regex=r"^[a-z]{2}_[A-Z]{2}$")
    
    # Validation et prévisualisation
    sample_data: Dict[str, Any] = Field(default_factory=dict)
    rendered_preview: Optional[str] = Field(None)
    
    @validator('body_template')
    def validate_body_template(cls, v):
        """Valide le template de body"""
        if not v or v.isspace():
            raise ValueError('Body template cannot be empty')
        return v.strip()
    
    @validator('available_variables', 'required_variables')
    def validate_variables(cls, v):
        """Valide les variables"""
        if v:
            for var in v:
                if not var.isidentifier():
                    raise ValueError(f'Invalid variable name: {var}')
        return v
    
    def render(self, variables: Dict[str, Any]) -> Dict[str, str]:
        """Rend le template avec les variables"""
        import jinja2
        
        # Vérification des variables requises
        missing_vars = set(self.required_variables) - set(variables.keys())
        if missing_vars:
            raise ValueError(f'Missing required variables: {missing_vars}')
        
        env = jinja2.Environment(
            loader=jinja2.DictLoader({
                'subject': self.subject_template or '',
                'body': self.body_template,
                'html': self.html_template or ''
            }),
            autoescape=True
        )
        
        rendered = {}
        
        if self.subject_template:
            rendered['subject'] = env.get_template('subject').render(**variables)
        
        rendered['body'] = env.get_template('body').render(**variables)
        
        if self.html_template:
            rendered['html'] = env.get_template('html').render(**variables)
        
        return rendered


class NotificationMessage(BaseSchema):
    """Message de notification"""
    message_id: UUID = Field(default_factory=uuid4)
    template_id: Optional[UUID] = Field(None)
    
    # Configuration
    channel: NotificationChannel = Field(...)
    priority: Priority = Field(Priority.NORMAL)
    
    # Destinataires
    recipients: List[str] = Field(..., min_items=1, max_items=100)
    cc_recipients: List[str] = Field(default_factory=list, max_items=50)
    bcc_recipients: List[str] = Field(default_factory=list, max_items=50)
    
    # Contenu
    subject: Optional[str] = Field(None, max_length=255)
    body: str = Field(..., min_length=1, max_length=10000)
    html_body: Optional[str] = Field(None, max_length=50000)
    attachments: List[Dict[str, Any]] = Field(default_factory=list, max_items=10)
    
    # Métadonnées
    correlation_id: Optional[UUID] = Field(None)
    alert_id: Optional[UUID] = Field(None)
    campaign_id: Optional[UUID] = Field(None)
    
    # Programmation
    scheduled_at: Optional[datetime] = Field(None)
    expires_at: Optional[datetime] = Field(None)
    
    # État
    status: NotificationStatus = Field(NotificationStatus.PENDING)
    retry_count: int = Field(0, ge=0, le=10)
    max_retries: int = Field(3, ge=0, le=10)
    
    # Tracking
    sent_at: Optional[datetime] = Field(None)
    delivered_at: Optional[datetime] = Field(None)
    read_at: Optional[datetime] = Field(None)
    clicked_at: Optional[datetime] = Field(None)
    
    # Configuration spécifique au canal
    channel_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Erreurs
    last_error: Optional[str] = Field(None)
    error_details: List[Dict[str, Any]] = Field(default_factory=list)
    
    @validator('recipients', 'cc_recipients', 'bcc_recipients')
    def validate_recipients(cls, v):
        """Valide les destinataires selon le canal"""
        # Cette validation sera étendue selon le canal
        return v
    
    @root_validator
    def validate_scheduling(cls, values):
        """Valide la programmation"""
        scheduled_at = values.get('scheduled_at')
        expires_at = values.get('expires_at')
        
        if scheduled_at and expires_at and scheduled_at >= expires_at:
            raise ValueError('scheduled_at must be before expires_at')
        
        return values
    
    @computed_field
    @property
    def is_expired(self) -> bool:
        """Indique si le message a expiré"""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    @computed_field
    @property
    def can_retry(self) -> bool:
        """Indique si une nouvelle tentative est possible"""
        return (
            self.retry_count < self.max_retries and
            not self.status.is_final and
            not self.is_expired
        )
    
    def add_error(self, error_message: str, error_details: Optional[Dict[str, Any]] = None):
        """Ajoute une erreur au message"""
        self.last_error = error_message
        error_record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': error_message,
            'details': error_details or {},
            'retry_count': self.retry_count
        }
        self.error_details.append(error_record)
    
    def mark_sent(self):
        """Marque le message comme envoyé"""
        self.status = NotificationStatus.SENT
        self.sent_at = datetime.now(timezone.utc)
    
    def mark_delivered(self):
        """Marque le message comme livré"""
        self.status = NotificationStatus.DELIVERED
        self.delivered_at = datetime.now(timezone.utc)
    
    def mark_failed(self, error_message: str):
        """Marque le message comme échoué"""
        self.status = NotificationStatus.FAILED
        self.add_error(error_message)


class SlackNotificationConfig(BaseModel):
    """Configuration spécifique Slack"""
    webhook_url: HttpUrl = Field(...)
    channel: Optional[str] = Field(None, regex=r"^#?[a-zA-Z0-9_-]+$")
    username: Optional[str] = Field(None, max_length=100)
    icon_emoji: Optional[str] = Field(None, regex=r"^:[a-zA-Z0-9_+-]+:$")
    icon_url: Optional[HttpUrl] = Field(None)
    
    # Configuration avancée
    thread_ts: Optional[str] = Field(None)  # Pour répondre dans un thread
    unfurl_links: bool = Field(True)
    unfurl_media: bool = Field(True)
    link_names: bool = Field(True)
    
    # Formatage
    mrkdwn: bool = Field(True)
    color: Optional[str] = Field(None, regex=r"^(good|warning|danger|#[0-9A-Fa-f]{6})$")
    
    class Config:
        schema_extra = {
            "example": {
                "webhook_url": "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX",
                "channel": "#alerts",
                "username": "Spotify AI Bot",
                "icon_emoji": ":robot_face:",
                "color": "danger"
            }
        }


class EmailNotificationConfig(BaseModel):
    """Configuration spécifique Email"""
    smtp_server: str = Field(..., min_length=1)
    smtp_port: int = Field(587, ge=1, le=65535)
    smtp_username: str = Field(...)
    smtp_password: str = Field(...)
    use_tls: bool = Field(True)
    use_ssl: bool = Field(False)
    
    # Expéditeur
    from_email: EmailStr = Field(...)
    from_name: Optional[str] = Field(None, max_length=100)
    reply_to: Optional[EmailStr] = Field(None)
    
    # Configuration avancée
    timeout_seconds: int = Field(30, ge=1, le=300)
    max_connections: int = Field(10, ge=1, le=100)
    
    # Headers personnalisés
    custom_headers: Dict[str, str] = Field(default_factory=dict)
    
    @root_validator
    def validate_ssl_tls(cls, values):
        """Valide la configuration SSL/TLS"""
        use_tls = values.get('use_tls', False)
        use_ssl = values.get('use_ssl', False)
        
        if use_tls and use_ssl:
            raise ValueError('Cannot use both TLS and SSL simultaneously')
        
        return values


class WebhookNotificationConfig(BaseModel):
    """Configuration spécifique Webhook"""
    url: HttpUrl = Field(...)
    method: str = Field("POST", regex=r"^(GET|POST|PUT|PATCH)$")
    timeout_seconds: int = Field(30, ge=1, le=300)
    
    # Authentification
    auth_type: Optional[str] = Field(None, regex=r"^(basic|bearer|api_key|custom)$")
    auth_username: Optional[str] = Field(None)
    auth_password: Optional[str] = Field(None)
    auth_token: Optional[str] = Field(None)
    api_key_header: Optional[str] = Field(None)
    api_key_value: Optional[str] = Field(None)
    
    # Headers et payload
    headers: Dict[str, str] = Field(default_factory=dict)
    payload_template: Optional[str] = Field(None)
    content_type: str = Field("application/json")
    
    # Retry et validation
    verify_ssl: bool = Field(True)
    follow_redirects: bool = Field(True)
    expected_status_codes: List[int] = Field(default_factory=lambda: [200, 201, 202])
    
    class Config:
        schema_extra = {
            "example": {
                "url": "https://api.example.com/webhooks/alerts",
                "method": "POST",
                "auth_type": "bearer",
                "auth_token": "your-token-here",
                "headers": {"X-Source": "Spotify-AI-Agent"},
                "content_type": "application/json"
            }
        }


class NotificationChannel(BaseSchema):
    """Canal de notification configuré"""
    name: StrictStr = Field(..., min_length=1, max_length=255)
    channel_type: NotificationChannel = Field(...)
    description: Optional[str] = Field(None, max_length=500)
    
    # Configuration générale
    is_active: bool = Field(True)
    priority: Priority = Field(Priority.NORMAL)
    
    # Rate limiting
    rate_limit_per_minute: Optional[int] = Field(None, ge=1, le=1000)
    rate_limit_per_hour: Optional[int] = Field(None, ge=1, le=10000)
    
    # Configuration spécifique
    slack_config: Optional[SlackNotificationConfig] = Field(None)
    email_config: Optional[EmailNotificationConfig] = Field(None)
    webhook_config: Optional[WebhookNotificationConfig] = Field(None)
    
    # Filtres et règles
    alert_level_filters: List[AlertLevel] = Field(default_factory=list)
    category_filters: List[str] = Field(default_factory=list)
    tag_filters: Dict[str, str] = Field(default_factory=dict)
    
    # Heures actives
    active_hours_start: Optional[str] = Field(None, regex=r"^([01]?[0-9]|2[0-3]):[0-5][0-9]$")
    active_hours_end: Optional[str] = Field(None, regex=r"^([01]?[0-9]|2[0-3]):[0-5][0-9]$")
    active_days: List[str] = Field(
        default_factory=lambda: ["monday", "tuesday", "wednesday", "thursday", "friday"],
        description="Jours actifs"
    )
    
    # Statistiques
    messages_sent_total: int = Field(0, ge=0)
    messages_failed_total: int = Field(0, ge=0)
    last_used_at: Optional[datetime] = Field(None)
    
    @root_validator
    def validate_config_presence(cls, values):
        """Valide la présence de la configuration appropriée"""
        channel_type = values.get('channel_type')
        
        if channel_type == NotificationChannel.SLACK and not values.get('slack_config'):
            raise ValueError('slack_config is required for Slack channels')
        
        if channel_type == NotificationChannel.EMAIL and not values.get('email_config'):
            raise ValueError('email_config is required for Email channels')
        
        if channel_type == NotificationChannel.WEBHOOK and not values.get('webhook_config'):
            raise ValueError('webhook_config is required for Webhook channels')
        
        return values
    
    @computed_field
    @property
    def success_rate(self) -> float:
        """Taux de succès des notifications"""
        total = self.messages_sent_total + self.messages_failed_total
        if total == 0:
            return 1.0
        return self.messages_sent_total / total
    
    def is_active_now(self) -> bool:
        """Vérifie si le canal est actif maintenant"""
        if not self.is_active:
            return False
        
        now = datetime.now(timezone.utc)
        current_day = now.strftime("%A").lower()
        current_time = now.strftime("%H:%M")
        
        # Vérification du jour
        if self.active_days and current_day not in self.active_days:
            return False
        
        # Vérification des heures
        if self.active_hours_start and self.active_hours_end:
            if self.active_hours_start <= self.active_hours_end:
                # Même jour
                return self.active_hours_start <= current_time <= self.active_hours_end
            else:
                # Traverse minuit
                return current_time >= self.active_hours_start or current_time <= self.active_hours_end
        
        return True
    
    def can_send_alert(self, alert_level: AlertLevel, category: str, tags: Dict[str, str]) -> bool:
        """Vérifie si une alerte peut être envoyée sur ce canal"""
        if not self.is_active_now():
            return False
        
        # Filtrage par niveau
        if self.alert_level_filters and alert_level not in self.alert_level_filters:
            return False
        
        # Filtrage par catégorie
        if self.category_filters and category not in self.category_filters:
            return False
        
        # Filtrage par tags
        if self.tag_filters:
            for key, value in self.tag_filters.items():
                if key not in tags or tags[key] != value:
                    return False
        
        return True


class NotificationBatch(BaseSchema):
    """Lot de notifications pour traitement en masse"""
    batch_id: UUID = Field(default_factory=uuid4)
    batch_name: Optional[str] = Field(None, max_length=255)
    
    # Messages
    message_ids: List[UUID] = Field(..., min_items=1, max_items=1000)
    total_messages: int = Field(...)
    
    # État du lot
    status: str = Field("pending", regex=r"^(pending|processing|completed|failed|partial)$")
    processed_count: int = Field(0, ge=0)
    success_count: int = Field(0, ge=0)
    failed_count: int = Field(0, ge=0)
    
    # Timing
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    estimated_completion: Optional[datetime] = Field(None)
    
    # Configuration
    max_concurrent: int = Field(10, ge=1, le=100)
    retry_failed: bool = Field(True)
    
    @computed_field
    @property
    def progress_percentage(self) -> float:
        """Pourcentage de progression"""
        if self.total_messages == 0:
            return 100.0
        return (self.processed_count / self.total_messages) * 100
    
    @computed_field
    @property
    def success_rate(self) -> float:
        """Taux de succès du lot"""
        if self.processed_count == 0:
            return 0.0
        return (self.success_count / self.processed_count) * 100
