"""
Notification Schema Module
=========================

Ce module définit les schémas pour le système de notifications multi-canal
avec support de templates dynamiques, routage intelligent et analytics
avancés pour l'optimisation des communications dans un environnement multi-tenant.
"""

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.networks import HttpUrl, EmailStr


class NotificationChannel(str, Enum):
    """Canaux de notification supportés."""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    TEAMS = "teams"
    DISCORD = "discord"
    WEBHOOK = "webhook"
    PUSH = "push"
    IN_APP = "in_app"
    PHONE = "phone"
    PAGERDUTY = "pagerduty"
    TELEGRAM = "telegram"
    WHATSAPP = "whatsapp"


class NotificationPriority(str, Enum):
    """Niveaux de priorité des notifications."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    SILENT = "silent"


class NotificationStatus(str, Enum):
    """États des notifications."""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"
    RETRY = "retry"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class NotificationType(str, Enum):
    """Types de notifications."""
    ALERT = "alert"
    WARNING = "warning"
    INFO = "info"
    SYSTEM = "system"
    SECURITY = "security"
    BILLING = "billing"
    MAINTENANCE = "maintenance"
    MARKETING = "marketing"
    REMINDER = "reminder"
    UPDATE = "update"


class DeliveryMode(str, Enum):
    """Modes de livraison."""
    IMMEDIATE = "immediate"
    SCHEDULED = "scheduled"
    BATCH = "batch"
    DIGEST = "digest"
    RATE_LIMITED = "rate_limited"


class NotificationRecipient(BaseModel):
    """Destinataire d'une notification."""
    recipient_id: str = Field(..., description="ID du destinataire")
    recipient_type: str = Field(..., regex="^(user|group|role|tenant|external)$")
    contact_info: Dict[str, str] = Field(..., description="Informations de contact")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="Préférences")
    timezone: str = Field("UTC", description="Fuseau horaire")
    locale: str = Field("en", description="Locale")
    active: bool = Field(True, description="Destinataire actif")
    
    class Config:
        schema_extra = {
            "example": {
                "recipient_id": "user_123",
                "recipient_type": "user",
                "contact_info": {
                    "email": "user@example.com",
                    "phone": "+1234567890",
                    "slack_user_id": "U1234567890"
                },
                "preferences": {
                    "email_enabled": True,
                    "sms_enabled": False,
                    "quiet_hours": "22:00-08:00"
                },
                "timezone": "America/New_York",
                "locale": "en_US"
            }
        }


class NotificationTemplate(BaseModel):
    """Template de notification avec support multi-canal."""
    template_id: str = Field(..., description="ID du template")
    name: str = Field(..., description="Nom du template")
    description: Optional[str] = Field(None, description="Description")
    channel: NotificationChannel = Field(..., description="Canal cible")
    subject_template: Optional[str] = Field(None, description="Template sujet")
    body_template: str = Field(..., description="Template corps")
    html_template: Optional[str] = Field(None, description="Template HTML")
    variables: List[str] = Field(default_factory=list, description="Variables disponibles")
    locales: Dict[str, Dict[str, str]] = Field(default_factory=dict, description="Traductions")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Métadonnées")
    
    class Config:
        schema_extra = {
            "example": {
                "template_id": "alert_email_template",
                "name": "Alert Email Template",
                "channel": "email",
                "subject_template": "🚨 {severity} Alert: {title}",
                "body_template": "Alert {title} detected at {timestamp}. Details: {description}",
                "variables": ["severity", "title", "timestamp", "description"],
                "locales": {
                    "fr": {
                        "subject_template": "🚨 Alerte {severity}: {title}",
                        "body_template": "Alerte {title} détectée à {timestamp}. Détails: {description}"
                    }
                }
            }
        }


class NotificationRule(BaseModel):
    """Règle de routage et filtrage des notifications."""
    rule_id: str = Field(..., description="ID de la règle")
    name: str = Field(..., description="Nom de la règle")
    tenant_id: Optional[str] = Field(None, description="ID tenant (null = global)")
    conditions: Dict[str, Any] = Field(..., description="Conditions de déclenchement")
    actions: List[Dict[str, Any]] = Field(..., description="Actions à exécuter")
    priority_override: Optional[NotificationPriority] = Field(None, description="Override priorité")
    channel_preferences: List[NotificationChannel] = Field(default_factory=list)
    escalation_rules: List[Dict[str, Any]] = Field(default_factory=list)
    rate_limiting: Dict[str, Any] = Field(default_factory=dict, description="Rate limiting")
    quiet_hours: Dict[str, str] = Field(default_factory=dict, description="Heures silencieuses")
    enabled: bool = Field(True, description="Règle activée")
    
    class Config:
        schema_extra = {
            "example": {
                "rule_id": "critical_alerts_rule",
                "name": "Critical Alerts Routing",
                "conditions": {
                    "severity": ["critical"],
                    "category": ["security", "system"]
                },
                "actions": [
                    {
                        "type": "notify",
                        "channels": ["email", "sms", "slack"],
                        "template": "critical_alert_template"
                    }
                ],
                "rate_limiting": {
                    "max_per_hour": 10,
                    "burst_limit": 3
                }
            }
        }


class NotificationDelivery(BaseModel):
    """Détails de livraison d'une notification."""
    delivery_id: str = Field(default_factory=lambda: str(uuid4()))
    channel: NotificationChannel = Field(..., description="Canal utilisé")
    recipient: NotificationRecipient = Field(..., description="Destinataire")
    status: NotificationStatus = Field(NotificationStatus.PENDING)
    attempts: int = Field(0, ge=0, description="Nombre de tentatives")
    max_attempts: int = Field(3, ge=1, le=10, description="Tentatives max")
    
    # Timing
    scheduled_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    
    # Résultats et métriques
    delivery_time_ms: Optional[int] = Field(None, ge=0, description="Temps de livraison (ms)")
    error_message: Optional[str] = Field(None, description="Message d'erreur")
    external_id: Optional[str] = Field(None, description="ID externe (provider)")
    cost: Optional[float] = Field(None, ge=0, description="Coût de livraison")
    
    # Métadonnées
    metadata: Dict[str, Any] = Field(default_factory=dict)
    provider_response: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "channel": "email",
                "recipient": {
                    "recipient_id": "user_123",
                    "recipient_type": "user",
                    "contact_info": {"email": "user@example.com"}
                },
                "status": "delivered",
                "attempts": 1,
                "delivery_time_ms": 1250,
                "cost": 0.001
            }
        }


class NotificationAnalytics(BaseModel):
    """Analytics et métriques des notifications."""
    open_rate: Optional[float] = Field(None, ge=0, le=100, description="Taux d'ouverture %")
    click_rate: Optional[float] = Field(None, ge=0, le=100, description="Taux de clic %")
    bounce_rate: Optional[float] = Field(None, ge=0, le=100, description="Taux de rebond %")
    unsubscribe_rate: Optional[float] = Field(None, ge=0, le=100, description="Taux de désabonnement %")
    response_time_avg_ms: Optional[float] = Field(None, ge=0, description="Temps de réponse moyen")
    delivery_success_rate: Optional[float] = Field(None, ge=0, le=100, description="Taux de succès livraison %")
    user_engagement_score: Optional[float] = Field(None, ge=0, le=10, description="Score d'engagement")
    conversion_rate: Optional[float] = Field(None, ge=0, le=100, description="Taux de conversion %")
    
    # Métriques par canal
    channel_performance: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    
    # Tendances temporelles
    hourly_distribution: Dict[str, int] = Field(default_factory=dict)
    daily_distribution: Dict[str, int] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "open_rate": 85.2,
                "click_rate": 12.8,
                "delivery_success_rate": 99.1,
                "user_engagement_score": 7.8,
                "channel_performance": {
                    "email": {"delivery_rate": 99.5, "open_rate": 85.2},
                    "slack": {"delivery_rate": 100.0, "read_rate": 95.3}
                }
            }
        }


class NotificationSchema(BaseModel):
    """
    Schéma principal de notification avec support multi-canal,
    templates dynamiques et analytics avancés pour optimisation.
    """
    # Identifiants et métadonnées
    notification_id: str = Field(default_factory=lambda: str(uuid4()))
    tenant_id: str = Field(..., description="ID du tenant")
    correlation_id: Optional[str] = Field(None, description="ID de corrélation")
    parent_notification_id: Optional[str] = Field(None, description="Notification parente")
    campaign_id: Optional[str] = Field(None, description="ID de campagne")
    
    # Classification et contenu
    notification_type: NotificationType = Field(..., description="Type de notification")
    priority: NotificationPriority = Field(NotificationPriority.NORMAL)
    title: str = Field(..., min_length=1, max_length=200, description="Titre")
    message: str = Field(..., min_length=1, max_length=5000, description="Message principal")
    html_content: Optional[str] = Field(None, description="Contenu HTML")
    summary: Optional[str] = Field(None, max_length=500, description="Résumé court")
    
    # Configuration de livraison
    delivery_mode: DeliveryMode = Field(DeliveryMode.IMMEDIATE)
    preferred_channels: List[NotificationChannel] = Field(..., min_items=1)
    recipients: List[NotificationRecipient] = Field(..., min_items=1)
    deliveries: List[NotificationDelivery] = Field(default_factory=list)
    
    # Template et personnalisation
    template_id: Optional[str] = Field(None, description="ID du template utilisé")
    template_variables: Dict[str, Any] = Field(default_factory=dict)
    localization: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    
    # Timing et planification
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None
    scheduled_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    send_after: Optional[datetime] = None
    send_before: Optional[datetime] = None
    
    # Règles et configuration
    rate_limit_key: Optional[str] = Field(None, description="Clé pour rate limiting")
    max_delivery_attempts: int = Field(3, ge=1, le=10)
    retry_delay_minutes: int = Field(5, ge=1, le=60)
    respect_quiet_hours: bool = Field(True, description="Respecter heures silencieuses")
    
    # Tracking et analytics
    tracking_enabled: bool = Field(True, description="Tracking activé")
    analytics: NotificationAnalytics = Field(default_factory=NotificationAnalytics)
    
    # Actions et interactivité
    actions: List[Dict[str, Any]] = Field(default_factory=list, description="Actions disponibles")
    callback_url: Optional[HttpUrl] = Field(None, description="URL de callback")
    deep_link: Optional[str] = Field(None, description="Deep link mobile")
    
    # Métadonnées et contexte
    source_system: Optional[str] = Field(None, description="Système source")
    source_event_id: Optional[str] = Field(None, description="ID événement source")
    business_context: Dict[str, Any] = Field(default_factory=dict)
    technical_context: Dict[str, Any] = Field(default_factory=dict)
    
    # Tags et classification
    tags: Dict[str, str] = Field(default_factory=dict)
    categories: List[str] = Field(default_factory=list)
    
    # Compliance et sécurité
    pii_data: bool = Field(False, description="Contient des données PII")
    encryption_required: bool = Field(False, description="Chiffrement requis")
    retention_days: int = Field(90, ge=1, le=2555, description="Rétention (jours)")
    
    class Config:
        use_enum_values = True
        validate_assignment = True
        extra = "forbid"
        schema_extra = {
            "example": {
                "tenant_id": "enterprise_001",
                "notification_type": "alert",
                "priority": "high",
                "title": "Critical System Alert",
                "message": "Database connection pool exhausted. Immediate attention required.",
                "preferred_channels": ["email", "slack", "sms"],
                "recipients": [
                    {
                        "recipient_id": "user_123",
                        "recipient_type": "user",
                        "contact_info": {"email": "admin@company.com"},
                        "timezone": "America/New_York"
                    }
                ],
                "template_variables": {
                    "severity": "critical",
                    "service": "database",
                    "timestamp": "2024-01-15T10:30:00Z"
                },
                "respect_quiet_hours": False
            }
        }
    
    @validator('updated_at', always=True)
    def set_updated_at(cls, v):
        """Met à jour automatiquement le timestamp."""
        return v or datetime.now(timezone.utc)
    
    @validator('expires_at')
    def validate_expiry(cls, v, values):
        """Valide la date d'expiration."""
        if v and 'created_at' in values:
            if v <= values['created_at']:
                raise ValueError("expires_at must be after created_at")
        return v
    
    @validator('send_before')
    def validate_send_window(cls, v, values):
        """Valide la fenêtre d'envoi."""
        send_after = values.get('send_after')
        if v and send_after and v <= send_after:
            raise ValueError("send_before must be after send_after")
        return v
    
    @root_validator
    def validate_priority_channels(cls, values):
        """Valide la cohérence entre priorité et canaux."""
        priority = values.get('priority')
        channels = values.get('preferred_channels', [])
        
        # Les notifications critiques doivent avoir des canaux immédiats
        if priority == NotificationPriority.CRITICAL:
            immediate_channels = [NotificationChannel.SMS, NotificationChannel.PHONE, NotificationChannel.PAGERDUTY]
            if not any(ch in channels for ch in immediate_channels):
                raise ValueError("Critical notifications must include immediate channels (SMS/Phone/PagerDuty)")
        
        return values
    
    @root_validator
    def validate_scheduled_delivery(cls, values):
        """Valide la livraison planifiée."""
        delivery_mode = values.get('delivery_mode')
        scheduled_at = values.get('scheduled_at')
        
        if delivery_mode == DeliveryMode.SCHEDULED and not scheduled_at:
            raise ValueError("Scheduled delivery mode requires scheduled_at")
        
        return values
    
    def get_total_recipients(self) -> int:
        """Retourne le nombre total de destinataires."""
        return len(self.recipients)
    
    def get_delivery_status_summary(self) -> Dict[str, int]:
        """Retourne un résumé des statuts de livraison."""
        status_counts = {}
        for delivery in self.deliveries:
            status = delivery.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        return status_counts
    
    def calculate_success_rate(self) -> float:
        """Calcule le taux de succès de livraison."""
        if not self.deliveries:
            return 0.0
        
        successful = sum(1 for d in self.deliveries 
                        if d.status in [NotificationStatus.DELIVERED, NotificationStatus.READ])
        return (successful / len(self.deliveries)) * 100
    
    def get_failed_deliveries(self) -> List[NotificationDelivery]:
        """Retourne les livraisons échouées."""
        return [d for d in self.deliveries if d.status == NotificationStatus.FAILED]
    
    def should_retry(self) -> bool:
        """Détermine si la notification doit être retentée."""
        failed_deliveries = self.get_failed_deliveries()
        for delivery in failed_deliveries:
            if delivery.attempts < delivery.max_attempts:
                return True
        return False
    
    def estimate_cost(self) -> float:
        """Estime le coût total de la notification."""
        # Coûts par canal (exemples en USD)
        channel_costs = {
            NotificationChannel.EMAIL: 0.0001,
            NotificationChannel.SMS: 0.01,
            NotificationChannel.PHONE: 0.05,
            NotificationChannel.SLACK: 0.0,
            NotificationChannel.PUSH: 0.0001,
            NotificationChannel.WEBHOOK: 0.0001
        }
        
        total_cost = 0.0
        for delivery in self.deliveries:
            channel_cost = channel_costs.get(delivery.channel, 0.001)
            total_cost += channel_cost * delivery.attempts
        
        return total_cost
    
    def get_preferred_locale(self, recipient_id: str) -> str:
        """Retourne la locale préférée d'un destinataire."""
        for recipient in self.recipients:
            if recipient.recipient_id == recipient_id:
                return recipient.locale
        return "en"  # Locale par défaut


class NotificationBatchSchema(BaseModel):
    """Schéma pour l'envoi en lot de notifications."""
    batch_id: str = Field(default_factory=lambda: str(uuid4()))
    tenant_id: str = Field(..., description="ID du tenant")
    name: str = Field(..., description="Nom du lot")
    description: Optional[str] = Field(None, description="Description")
    notifications: List[NotificationSchema] = Field(..., min_items=1, max_items=1000)
    
    # Configuration du lot
    batch_size: int = Field(100, ge=1, le=1000, description="Taille du lot")
    delay_between_batches_seconds: int = Field(1, ge=0, le=3600)
    max_concurrent_deliveries: int = Field(10, ge=1, le=100)
    
    # Timing
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Métriques
    total_notifications: int = Field(0, ge=0)
    successful_deliveries: int = Field(0, ge=0)
    failed_deliveries: int = Field(0, ge=0)
    
    class Config:
        schema_extra = {
            "example": {
                "tenant_id": "enterprise_001",
                "name": "Weekly System Report",
                "description": "Weekly system health report for all users",
                "batch_size": 50,
                "delay_between_batches_seconds": 5,
                "total_notifications": 1000
            }
        }
