"""
Schémas Pydantic pour validation et sérialisation des notifications
===================================================================

Schémas ultra-avancés avec validation complète, transformation des données,
et support des API REST/GraphQL.
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union, Set, Literal
from uuid import UUID
from enum import Enum
import re

from pydantic import (
    BaseModel, Field, validator, root_validator, EmailStr,
    HttpUrl, constr, conint, confloat, create_model
)
from pydantic.color import Color


class NotificationPriorityEnum(str, Enum):
    """Énumération des priorités de notification"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class NotificationStatusEnum(str, Enum):
    """Énumération des statuts de notification"""
    PENDING = "pending"
    PROCESSING = "processing"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"
    RETRYING = "retrying"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ChannelTypeEnum(str, Enum):
    """Types de canaux supportés"""
    SLACK = "slack"
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    DISCORD = "discord"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    DATABASE = "database"
    TELEGRAM = "telegram"
    WHATSAPP = "whatsapp"


class BaseNotificationSchema(BaseModel):
    """Schéma de base pour toutes les notifications"""
    
    class Config:
        use_enum_values = True
        validate_assignment = True
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class RecipientSchema(BaseNotificationSchema):
    """Schéma pour les destinataires"""
    id: str = Field(..., description="Identifiant unique du destinataire")
    type: Literal["user", "group", "role", "webhook", "external"] = Field(
        ..., description="Type de destinataire"
    )
    display_name: Optional[str] = Field(None, description="Nom d'affichage")
    email: Optional[EmailStr] = Field(None, description="Adresse email")
    phone: Optional[str] = Field(None, regex=r'^\+?[1-9]\d{1,14}$', description="Numéro de téléphone")
    slack_user_id: Optional[str] = Field(None, description="ID utilisateur Slack")
    discord_user_id: Optional[str] = Field(None, description="ID utilisateur Discord")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Métadonnées additionnelles")
    
    @validator('phone')
    def validate_phone(cls, v):
        if v and not re.match(r'^\+?[1-9]\d{1,14}$', v):
            raise ValueError('Format de téléphone invalide')
        return v


class ChannelConfigSchema(BaseNotificationSchema):
    """Configuration spécifique à chaque canal"""
    type: ChannelTypeEnum = Field(..., description="Type de canal")
    enabled: bool = Field(True, description="Canal activé")
    priority_threshold: NotificationPriorityEnum = Field(
        NotificationPriorityEnum.LOW, 
        description="Priorité minimum pour ce canal"
    )
    
    # Slack specific
    slack_channel: Optional[str] = Field(None, description="Canal Slack (#channel ou @user)")
    slack_thread_ts: Optional[str] = Field(None, description="Timestamp du thread Slack")
    slack_blocks: Optional[List[Dict]] = Field(None, description="Blocs Slack avancés")
    
    # Email specific
    email_template: Optional[str] = Field(None, description="Template email spécifique")
    email_attachments: Optional[List[str]] = Field(None, description="Pièces jointes")
    email_reply_to: Optional[EmailStr] = Field(None, description="Adresse de réponse")
    
    # SMS specific
    sms_short_url: bool = Field(False, description="Utiliser des URLs courtes")
    sms_sender_id: Optional[str] = Field(None, description="ID expéditeur SMS")
    
    # Push specific
    push_badge_count: Optional[int] = Field(None, ge=0, description="Nombre de badges")
    push_sound: Optional[str] = Field(None, description="Son de notification")
    push_category: Optional[str] = Field(None, description="Catégorie de notification push")
    push_action_buttons: Optional[List[Dict]] = Field(None, description="Boutons d'action")
    
    # Webhook specific
    webhook_url: Optional[HttpUrl] = Field(None, description="URL du webhook")
    webhook_method: Optional[Literal["GET", "POST", "PUT", "PATCH"]] = Field("POST")
    webhook_headers: Dict[str, str] = Field(default_factory=dict)
    webhook_auth: Optional[Dict[str, str]] = Field(None, description="Authentification webhook")
    
    # Rate limiting
    rate_limit_per_minute: Optional[int] = Field(None, ge=1, le=1000)
    rate_limit_per_hour: Optional[int] = Field(None, ge=1, le=10000)
    
    # Retry config
    max_retries: int = Field(3, ge=0, le=10, description="Nombre maximum de tentatives")
    retry_backoff: float = Field(1.5, ge=1.0, le=10.0, description="Facteur de backoff exponentiel")
    retry_jitter: bool = Field(True, description="Ajouter du jitter aux retries")


class NotificationTemplateSchema(BaseNotificationSchema):
    """Schéma pour les templates de notification"""
    id: Optional[UUID] = Field(None, description="ID du template")
    name: str = Field(..., min_length=1, max_length=255, description="Nom du template")
    channel_type: ChannelTypeEnum = Field(..., description="Type de canal cible")
    version: str = Field("1.0.0", regex=r'^\d+\.\d+\.\d+$', description="Version sémantique")
    
    # Content templates
    subject_template: Optional[str] = Field(None, description="Template du sujet")
    body_template: str = Field(..., min_length=1, description="Template du corps")
    html_template: Optional[str] = Field(None, description="Template HTML")
    
    # Localization
    locale: str = Field("en", regex=r'^[a-z]{2}(-[A-Z]{2})?$', description="Locale principal")
    fallback_locale: str = Field("en", description="Locale de fallback")
    
    # Metadata
    metadata_template: Dict[str, Any] = Field(default_factory=dict)
    tags: Set[str] = Field(default_factory=set, description="Tags pour organisation")
    category: Optional[str] = Field(None, description="Catégorie du template")
    
    # A/B Testing
    ab_test_group: Optional[str] = Field(None, description="Groupe de test A/B")
    ab_test_weight: float = Field(1.0, ge=0.0, le=1.0, description="Poids pour A/B testing")
    
    # Validation rules
    required_variables: Set[str] = Field(default_factory=set, description="Variables obligatoires")
    optional_variables: Set[str] = Field(default_factory=set, description="Variables optionnelles")
    
    # Status
    is_active: bool = Field(True, description="Template actif")
    usage_count: int = Field(0, ge=0, description="Nombre d'utilisations")
    success_rate: float = Field(0.0, ge=0.0, le=1.0, description="Taux de succès")
    
    # Audit
    created_at: Optional[datetime] = Field(None, description="Date de création")
    updated_at: Optional[datetime] = Field(None, description="Date de modification")
    created_by: Optional[str] = Field(None, description="Créateur")
    tenant_id: str = Field(..., description="ID du tenant")
    
    @validator('required_variables', 'optional_variables')
    def validate_variables(cls, v):
        # Valider que les noms de variables sont valides
        for var in v:
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', var):
                raise ValueError(f'Nom de variable invalide: {var}')
        return v
    
    @root_validator
    def validate_template_consistency(cls, values):
        body = values.get('body_template', '')
        required_vars = values.get('required_variables', set())
        
        # Extraire les variables du template
        template_vars = set(re.findall(r'\{\{(\w+)\}\}', body))
        
        # Vérifier que toutes les variables requises sont dans le template
        missing_vars = required_vars - template_vars
        if missing_vars:
            raise ValueError(f'Variables requises manquantes dans le template: {missing_vars}')
        
        return values


class NotificationRuleSchema(BaseNotificationSchema):
    """Schéma pour les règles de notification"""
    id: Optional[UUID] = Field(None, description="ID de la règle")
    name: str = Field(..., min_length=1, max_length=255, description="Nom de la règle")
    description: Optional[str] = Field(None, description="Description détaillée")
    
    # Conditions de déclenchement
    conditions: Dict[str, Any] = Field(..., description="Conditions JSON pour déclencher la règle")
    priority_threshold: NotificationPriorityEnum = Field(
        NotificationPriorityEnum.LOW,
        description="Priorité minimum pour déclencher"
    )
    
    # Configuration des canaux
    channels: List[ChannelConfigSchema] = Field(..., min_items=1, description="Canaux à utiliser")
    
    # Escalade
    escalation_rules: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Règles d'escalade automatique"
    )
    
    # Rate limiting
    rate_limit_per_minute: int = Field(100, ge=1, le=10000)
    rate_limit_per_hour: int = Field(1000, ge=1, le=100000)
    burst_allowance: int = Field(10, ge=1, le=1000)
    
    # Planification
    active_hours: Optional[Dict[str, List[str]]] = Field(
        None,
        description="Heures actives par jour de la semaine"
    )
    timezone: str = Field("UTC", description="Fuseau horaire")
    
    # Status
    is_active: bool = Field(True, description="Règle active")
    execution_count: int = Field(0, ge=0, description="Nombre d'exécutions")
    last_executed: Optional[datetime] = Field(None, description="Dernière exécution")
    
    # Audit
    created_at: Optional[datetime] = Field(None)
    updated_at: Optional[datetime] = Field(None)
    tenant_id: str = Field(..., description="ID du tenant")


class NotificationCreateSchema(BaseNotificationSchema):
    """Schéma pour créer une nouvelle notification"""
    title: str = Field(..., min_length=1, max_length=500, description="Titre de la notification")
    message: str = Field(..., min_length=1, description="Message principal")
    priority: NotificationPriorityEnum = Field(
        NotificationPriorityEnum.NORMAL,
        description="Priorité de la notification"
    )
    
    # Destinataires
    recipients: List[RecipientSchema] = Field(..., min_items=1, description="Liste des destinataires")
    
    # Canal de communication
    channels: List[ChannelConfigSchema] = Field(..., min_items=1, description="Canaux à utiliser")
    
    # Template (optionnel)
    template_id: Optional[UUID] = Field(None, description="ID du template à utiliser")
    template_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Variables pour le template"
    )
    
    # Planification
    scheduled_at: Optional[datetime] = Field(None, description="Date d'envoi programmée")
    expires_at: Optional[datetime] = Field(None, description="Date d'expiration")
    
    # Métadonnées
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Métadonnées libres")
    tags: List[str] = Field(default_factory=list, description="Tags pour classification")
    source_system: Optional[str] = Field(None, description="Système source")
    source_event: Optional[str] = Field(None, description="Événement source")
    correlation_id: Optional[str] = Field(None, description="ID de corrélation")
    
    # Configuration avancée
    retry_config: Optional[Dict[str, Any]] = Field(None, description="Configuration des tentatives")
    batching_key: Optional[str] = Field(None, description="Clé pour le regroupement")
    
    @validator('expires_at')
    def validate_expiration(cls, v, values):
        scheduled = values.get('scheduled_at')
        if v and scheduled and v <= scheduled:
            raise ValueError('La date d\'expiration doit être postérieure à la date programmée')
        if v and v <= datetime.now(timezone.utc):
            raise ValueError('La date d\'expiration ne peut pas être dans le passé')
        return v


class NotificationResponseSchema(BaseNotificationSchema):
    """Schéma de réponse pour les notifications"""
    id: UUID = Field(..., description="ID unique de la notification")
    title: str = Field(..., description="Titre")
    message: str = Field(..., description="Message")
    priority: NotificationPriorityEnum = Field(..., description="Priorité")
    status: NotificationStatusEnum = Field(..., description="Statut actuel")
    
    # Destinataire
    recipient_id: str = Field(..., description="ID du destinataire")
    recipient_type: str = Field(..., description="Type de destinataire")
    
    # Canal
    channel_type: ChannelTypeEnum = Field(..., description="Type de canal")
    
    # Timing
    created_at: datetime = Field(..., description="Date de création")
    scheduled_at: Optional[datetime] = Field(None, description="Date programmée")
    sent_at: Optional[datetime] = Field(None, description="Date d'envoi")
    delivered_at: Optional[datetime] = Field(None, description="Date de livraison")
    read_at: Optional[datetime] = Field(None, description="Date de lecture")
    expires_at: Optional[datetime] = Field(None, description="Date d'expiration")
    
    # Retry info
    retry_count: int = Field(0, description="Nombre de tentatives")
    max_retries: int = Field(3, description="Maximum de tentatives")
    next_retry_at: Optional[datetime] = Field(None, description="Prochaine tentative")
    
    # Métadonnées
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    correlation_id: Optional[str] = Field(None)
    
    # Analytics
    delivery_time_ms: Optional[int] = Field(None, description="Temps de livraison en ms")
    read_time_ms: Optional[int] = Field(None, description="Temps de lecture en ms")
    interaction_count: int = Field(0, description="Nombre d'interactions")
    
    # Relations
    template_id: Optional[UUID] = Field(None, description="ID du template utilisé")
    parent_notification_id: Optional[UUID] = Field(None, description="Notification parente")
    
    # Audit
    tenant_id: str = Field(..., description="ID du tenant")


class NotificationUpdateSchema(BaseNotificationSchema):
    """Schéma pour mettre à jour une notification"""
    status: Optional[NotificationStatusEnum] = Field(None, description="Nouveau statut")
    priority: Optional[NotificationPriorityEnum] = Field(None, description="Nouvelle priorité")
    scheduled_at: Optional[datetime] = Field(None, description="Nouvelle date programmée")
    expires_at: Optional[datetime] = Field(None, description="Nouvelle date d'expiration")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Nouvelles métadonnées")
    tags: Optional[List[str]] = Field(None, description="Nouveaux tags")


class NotificationBatchSchema(BaseNotificationSchema):
    """Schéma pour l'envoi en lot de notifications"""
    notifications: List[NotificationCreateSchema] = Field(
        ..., 
        min_items=1, 
        max_items=1000,
        description="Notifications à envoyer"
    )
    batch_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration du lot"
    )
    priority_boost: Optional[NotificationPriorityEnum] = Field(
        None,
        description="Augmenter la priorité de toutes les notifications"
    )
    
    @validator('notifications')
    def validate_batch_consistency(cls, v):
        if len(v) > 1000:
            raise ValueError('Maximum 1000 notifications par lot')
        return v


class NotificationMetricsSchema(BaseNotificationSchema):
    """Schéma pour les métriques de notification"""
    time_bucket: datetime = Field(..., description="Période de temps")
    tenant_id: str = Field(..., description="ID du tenant")
    channel_type: ChannelTypeEnum = Field(..., description="Type de canal")
    priority: NotificationPriorityEnum = Field(..., description="Priorité")
    
    # Compteurs
    total_sent: int = Field(0, ge=0, description="Nombre total envoyé")
    total_delivered: int = Field(0, ge=0, description="Nombre total livré")
    total_failed: int = Field(0, ge=0, description="Nombre total échoué")
    total_read: int = Field(0, ge=0, description="Nombre total lu")
    
    # Temps de réponse
    avg_delivery_time_ms: float = Field(0.0, ge=0.0, description="Temps moyen de livraison")
    avg_read_time_ms: float = Field(0.0, ge=0.0, description="Temps moyen de lecture")
    p95_delivery_time_ms: float = Field(0.0, ge=0.0, description="P95 temps de livraison")
    p99_delivery_time_ms: float = Field(0.0, ge=0.0, description="P99 temps de livraison")
    
    # Taux
    retry_rate: float = Field(0.0, ge=0.0, le=1.0, description="Taux de retry")
    success_rate: float = Field(0.0, ge=0.0, le=1.0, description="Taux de succès")
    read_rate: float = Field(0.0, ge=0.0, le=1.0, description="Taux de lecture")
    
    # Détail des erreurs
    error_breakdown: Dict[str, int] = Field(default_factory=dict, description="Répartition des erreurs")


class NotificationPreferencesSchema(BaseNotificationSchema):
    """Schéma pour les préférences de notification"""
    user_id: str = Field(..., description="ID de l'utilisateur")
    tenant_id: str = Field(..., description="ID du tenant")
    
    # Canaux
    enabled_channels: List[ChannelTypeEnum] = Field(
        default_factory=list,
        description="Canaux activés"
    )
    disabled_channels: List[ChannelTypeEnum] = Field(
        default_factory=list,
        description="Canaux désactivés"
    )
    
    # Priorités
    min_priority: NotificationPriorityEnum = Field(
        NotificationPriorityEnum.LOW,
        description="Priorité minimum"
    )
    priority_overrides: Dict[str, NotificationPriorityEnum] = Field(
        default_factory=dict,
        description="Surcharges de priorité par catégorie"
    )
    
    # Heures de silence
    quiet_hours_start: Optional[constr(regex=r'^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$')] = Field(
        None, description="Début des heures de silence (HH:MM)"
    )
    quiet_hours_end: Optional[constr(regex=r'^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$')] = Field(
        None, description="Fin des heures de silence (HH:MM)"
    )
    quiet_hours_timezone: str = Field("UTC", description="Fuseau horaire")
    
    # Limites de fréquence
    max_notifications_per_hour: int = Field(50, ge=1, le=1000)
    max_notifications_per_day: int = Field(200, ge=1, le=10000)
    
    # Préférences de contenu
    language: str = Field("en", regex=r'^[a-z]{2}(-[A-Z]{2})?$')
    template_style: str = Field("default", description="Style de template préféré")
    include_attachments: bool = Field(True, description="Inclure les pièces jointes")
    
    # Fonctionnalités intelligentes
    enable_ai_filtering: bool = Field(False, description="Filtrage IA des notifications")
    enable_smart_batching: bool = Field(True, description="Regroupement intelligent")
    enable_sentiment_analysis: bool = Field(False, description="Analyse de sentiment")
    
    # Audit
    created_at: Optional[datetime] = Field(None)
    updated_at: Optional[datetime] = Field(None)
