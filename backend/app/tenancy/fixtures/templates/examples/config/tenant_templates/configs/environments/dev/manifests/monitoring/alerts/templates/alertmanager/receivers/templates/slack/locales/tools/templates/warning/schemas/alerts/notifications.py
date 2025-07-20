"""
Schémas de notifications - Spotify AI Agent
Gestion complète des canaux et templates de notification
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Union, Literal
from uuid import UUID, uuid4
from enum import Enum
import json
import re
from urllib.parse import urlparse

from pydantic import BaseModel, Field, validator, computed_field, ConfigDict, EmailStr, HttpUrl

from . import (
    BaseSchema, TimestampMixin, TenantMixin, MetadataMixin,
    AlertLevel, AlertStatus, WarningCategory, NotificationChannel,
    Priority, Environment
)


class DeliveryStatus(str, Enum):
    """Statuts de livraison des notifications"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    BOUNCED = "bounced"
    RETRY = "retry"
    SUPPRESSED = "suppressed"


class ChannelType(str, Enum):
    """Types de canaux de notification étendus"""
    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"
    DISCORD = "discord"
    SMS = "sms"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    OPSGENIE = "opsgenie"
    SERVICENOW = "servicenow"
    JIRA = "jira"
    PUSH_NOTIFICATION = "push_notification"
    TELEGRAM = "telegram"
    WHATSAPP = "whatsapp"


class NotificationPriority(str, Enum):
    """Priorités de notification"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    IMMEDIATE = "immediate"


class TemplateFormat(str, Enum):
    """Formats de template"""
    PLAIN_TEXT = "plain_text"
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    RICH_TEXT = "rich_text"


class ThrottleMode(str, Enum):
    """Modes de limitation de débit"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    EXPONENTIAL_BACKOFF = "exponential_backoff"


class NotificationChannel(BaseSchema, TimestampMixin, TenantMixin, MetadataMixin):
    """Canal de notification avancé"""
    
    # Informations de base
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    channel_type: ChannelType = Field(...)
    environment: Environment = Field(...)
    
    # Configuration du canal
    configuration: Dict[str, Any] = Field(...)
    credentials: Optional[Dict[str, str]] = Field(None)  # Chiffré en DB
    
    # Paramètres de livraison
    retry_attempts: int = Field(3, ge=0, le=10)
    retry_delay_seconds: int = Field(30, ge=1, le=3600)
    timeout_seconds: int = Field(30, ge=1, le=300)
    
    # Limitation de débit
    rate_limit_enabled: bool = Field(True)
    max_notifications_per_minute: int = Field(10, ge=1, le=1000)
    max_notifications_per_hour: int = Field(100, ge=1, le=10000)
    throttle_mode: ThrottleMode = Field(ThrottleMode.SLIDING_WINDOW)
    
    # Filtrage et conditions
    alert_level_filter: List[AlertLevel] = Field(default_factory=list)
    category_filter: List[WarningCategory] = Field(default_factory=list)
    priority_filter: List[Priority] = Field(default_factory=list)
    environment_filter: List[Environment] = Field(default_factory=list)
    
    # Horaires et disponibilité
    business_hours_only: bool = Field(False)
    timezone: str = Field("UTC")
    quiet_hours_start: Optional[str] = Field(None)  # Format: "HH:MM"
    quiet_hours_end: Optional[str] = Field(None)    # Format: "HH:MM"
    weekend_enabled: bool = Field(True)
    
    # État et santé
    enabled: bool = Field(True)
    health_check_enabled: bool = Field(True)
    last_health_check: Optional[datetime] = Field(None)
    health_status: str = Field("unknown")  # healthy, degraded, unhealthy
    
    # Métriques
    total_sent: int = Field(0, ge=0)
    total_delivered: int = Field(0, ge=0)
    total_failed: int = Field(0, ge=0)
    last_sent: Optional[datetime] = Field(None)
    
    # Tags et labels
    tags: Set[str] = Field(default_factory=set)
    labels: Dict[str, str] = Field(default_factory=dict)
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )

    @validator('configuration')
    def validate_configuration(cls, v, values):
        """Valide la configuration selon le type de canal"""
        channel_type = values.get('channel_type')
        
        if channel_type == ChannelType.EMAIL:
            required_fields = ['smtp_host', 'smtp_port', 'from_address']
        elif channel_type == ChannelType.SLACK:
            required_fields = ['webhook_url']
        elif channel_type == ChannelType.WEBHOOK:
            required_fields = ['url']
        elif channel_type == ChannelType.SMS:
            required_fields = ['provider', 'api_key']
        else:
            return v
        
        missing_fields = [field for field in required_fields if field not in v]
        if missing_fields:
            raise ValueError(f'Missing required configuration fields: {missing_fields}')
        
        return v

    @validator('quiet_hours_start', 'quiet_hours_end')
    def validate_time_format(cls, v):
        """Valide le format des heures"""
        if v and not re.match(r'^([01]?[0-9]|2[0-3]):[0-5][0-9]$', v):
            raise ValueError('Time must be in HH:MM format')
        return v

    @computed_field
    @property
    def delivery_rate(self) -> float:
        """Taux de livraison du canal"""
        if self.total_sent == 0:
            return 0.0
        return (self.total_delivered / self.total_sent) * 100

    @computed_field
    @property
    def is_healthy(self) -> bool:
        """Indique si le canal est en bonne santé"""
        return self.health_status == "healthy" and self.enabled

    def can_send_now(self) -> bool:
        """Vérifie si une notification peut être envoyée maintenant"""
        if not self.enabled:
            return False
        
        now = datetime.now(timezone.utc)
        
        # Vérifier les heures de silence
        if self.quiet_hours_start and self.quiet_hours_end:
            current_time = now.strftime("%H:%M")
            if self.quiet_hours_start <= current_time <= self.quiet_hours_end:
                return False
        
        # Vérifier les heures ouvrables
        if self.business_hours_only:
            # Logique simplifiée - peut être étendue
            if now.weekday() >= 5 and not self.weekend_enabled:  # Weekend
                return False
        
        return True


class NotificationTemplate(BaseSchema, TimestampMixin, TenantMixin, MetadataMixin):
    """Template de notification personnalisable"""
    
    # Informations de base
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    version: str = Field("1.0.0")
    
    # Configuration du template
    format: TemplateFormat = Field(...)
    channel_types: List[ChannelType] = Field(..., min_items=1)
    
    # Contenu du template
    subject_template: Optional[str] = Field(None, max_length=500)
    body_template: str = Field(..., min_length=1)
    
    # Templates spécialisés par canal
    channel_templates: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    
    # Variables et paramètres
    variables: Dict[str, Any] = Field(default_factory=dict)
    required_variables: Set[str] = Field(default_factory=set)
    
    # Localisation
    locale: str = Field("en_US")
    localized_templates: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    
    # Catégories et filtres
    alert_levels: List[AlertLevel] = Field(default_factory=list)
    categories: List[WarningCategory] = Field(default_factory=list)
    
    # État et validation
    enabled: bool = Field(True)
    validated: bool = Field(False)
    validation_errors: List[str] = Field(default_factory=list)
    
    # Usage et métriques
    usage_count: int = Field(0, ge=0)
    last_used: Optional[datetime] = Field(None)
    
    # Tags et organisation
    tags: Set[str] = Field(default_factory=set)
    
    @validator('body_template')
    def validate_template_syntax(cls, v):
        """Valide la syntaxe du template"""
        try:
            # Test de base pour vérifier les variables Jinja2
            import jinja2
            jinja2.Template(v)
        except Exception as e:
            raise ValueError(f'Invalid template syntax: {str(e)}')
        return v

    def render(self, context: Dict[str, Any], channel_type: ChannelType, 
               locale: Optional[str] = None) -> Dict[str, str]:
        """Rend le template avec le contexte donné"""
        import jinja2
        
        # Sélectionner le template approprié
        template_data = self.channel_templates.get(channel_type.value, {
            'subject': self.subject_template or '',
            'body': self.body_template
        })
        
        # Localisation si disponible
        if locale and locale in self.localized_templates:
            localized = self.localized_templates[locale]
            template_data.update(localized)
        
        # Variables par défaut
        default_vars = {
            'tenant_id': str(context.get('tenant_id', '')),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'environment': context.get('environment', 'unknown'),
            **self.variables
        }
        
        # Fusionner avec le contexte
        render_context = {**default_vars, **context}
        
        # Rendre les templates
        result = {}
        for key, template_str in template_data.items():
            if template_str:
                template = jinja2.Template(template_str)
                result[key] = template.render(**render_context)
        
        return result


class NotificationRule(BaseSchema, TimestampMixin, TenantMixin):
    """Règle de routage des notifications"""
    
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    priority: int = Field(100, ge=1, le=1000)  # Plus bas = plus prioritaire
    
    # Conditions de routage
    conditions: Dict[str, Any] = Field(...)
    
    # Canaux cibles
    channel_ids: List[UUID] = Field(..., min_items=1)
    template_id: Optional[UUID] = Field(None)
    
    # Configuration
    enabled: bool = Field(True)
    stop_on_match: bool = Field(False)
    
    # Délais et timing
    delay_seconds: int = Field(0, ge=0, le=3600)
    batch_notifications: bool = Field(False)
    batch_window_seconds: int = Field(300, ge=1, le=3600)
    
    def matches(self, alert_data: Dict[str, Any]) -> bool:
        """Vérifie si l'alerte correspond aux conditions"""
        # Implémentation simplifiée - peut être étendue
        for key, expected_value in self.conditions.items():
            if key not in alert_data:
                return False
            
            actual_value = alert_data[key]
            
            if isinstance(expected_value, list):
                if actual_value not in expected_value:
                    return False
            elif isinstance(expected_value, dict):
                # Conditions avancées (regex, range, etc.)
                operator = expected_value.get('operator', 'eq')
                value = expected_value.get('value')
                
                if operator == 'eq' and actual_value != value:
                    return False
                elif operator == 'regex' and not re.match(value, str(actual_value)):
                    return False
                # Ajouter d'autres opérateurs selon les besoins
            else:
                if actual_value != expected_value:
                    return False
        
        return True


class NotificationBatch(BaseSchema, TimestampMixin, TenantMixin):
    """Lot de notifications groupées"""
    
    batch_id: UUID = Field(default_factory=uuid4)
    channel_id: UUID = Field(...)
    template_id: Optional[UUID] = Field(None)
    
    # Alertes du lot
    alert_ids: List[UUID] = Field(..., min_items=1)
    total_alerts: int = Field(..., ge=1)
    
    # État du lot
    status: DeliveryStatus = Field(DeliveryStatus.PENDING)
    scheduled_at: datetime = Field(...)
    sent_at: Optional[datetime] = Field(None)
    delivered_at: Optional[datetime] = Field(None)
    
    # Résultats de livraison
    delivery_attempts: int = Field(0, ge=0)
    last_error: Optional[str] = Field(None)
    
    @computed_field
    @property
    def is_ready_to_send(self) -> bool:
        """Indique si le lot est prêt à être envoyé"""
        return (
            self.status == DeliveryStatus.PENDING and
            datetime.now(timezone.utc) >= self.scheduled_at
        )


class NotificationHistory(BaseSchema, TimestampMixin, TenantMixin):
    """Historique des notifications envoyées"""
    
    notification_id: UUID = Field(default_factory=uuid4)
    alert_id: UUID = Field(...)
    channel_id: UUID = Field(...)
    template_id: Optional[UUID] = Field(None)
    
    # Contenu
    subject: Optional[str] = Field(None)
    body: str = Field(...)
    
    # Livraison
    status: DeliveryStatus = Field(...)
    sent_at: Optional[datetime] = Field(None)
    delivered_at: Optional[datetime] = Field(None)
    
    # Détails techniques
    delivery_attempts: int = Field(1, ge=1)
    response_code: Optional[int] = Field(None)
    response_message: Optional[str] = Field(None)
    delivery_duration_ms: Optional[float] = Field(None, ge=0)
    
    # Métadonnées
    user_agent: Optional[str] = Field(None)
    ip_address: Optional[str] = Field(None)
    external_id: Optional[str] = Field(None)  # ID du service externe
    
    # Traçabilité
    correlation_id: Optional[UUID] = Field(None)
    batch_id: Optional[UUID] = Field(None)


__all__ = [
    'DeliveryStatus', 'ChannelType', 'NotificationPriority', 'TemplateFormat', 'ThrottleMode',
    'NotificationChannel', 'NotificationTemplate', 'NotificationRule', 
    'NotificationBatch', 'NotificationHistory'
]
