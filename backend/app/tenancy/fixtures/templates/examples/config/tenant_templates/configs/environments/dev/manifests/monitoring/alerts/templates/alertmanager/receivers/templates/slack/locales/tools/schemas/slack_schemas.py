"""
Schémas Pydantic avancés pour l'intégration Slack.

Ce module définit tous les schémas pour la configuration Slack, messages,
templates, webhooks et éléments d'interface.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator, HttpUrl
import re


class SlackMessageType(str, Enum):
    """Types de messages Slack."""
    TEXT = "text"
    RICH_TEXT = "rich_text"
    ATTACHMENT = "attachment"
    BLOCK_KIT = "block_kit"


class SlackColorScheme(str, Enum):
    """Schémas de couleurs Slack."""
    GOOD = "good"
    WARNING = "warning"
    DANGER = "danger"
    PRIMARY = "#36a64f"
    SECONDARY = "#ddd"


class SlackBlockType(str, Enum):
    """Types de blocs Slack."""
    SECTION = "section"
    DIVIDER = "divider"
    IMAGE = "image"
    ACTIONS = "actions"
    CONTEXT = "context"
    INPUT = "input"
    HEADER = "header"


class SlackElementType(str, Enum):
    """Types d'éléments Slack."""
    BUTTON = "button"
    SELECT = "static_select"
    MULTI_SELECT = "multi_static_select"
    OVERFLOW = "overflow"
    DATE_PICKER = "datepicker"
    TIME_PICKER = "timepicker"
    PLAIN_TEXT_INPUT = "plain_text_input"
    RADIO_BUTTONS = "radio_buttons"
    CHECKBOXES = "checkboxes"


class SlackFieldSchema(BaseModel):
    """Schéma pour les champs d'attachement."""
    title: str = Field(..., description="Titre du champ")
    value: str = Field(..., description="Valeur du champ")
    short: bool = Field(False, description="Affichage court")


class SlackActionSchema(BaseModel):
    """Schéma pour les actions Slack."""
    type: str = Field(..., description="Type d'action")
    text: str = Field(..., description="Texte de l'action")
    url: Optional[HttpUrl] = Field(None, description="URL de l'action")
    style: Optional[str] = Field(None, description="Style de l'action")
    confirm: Optional[Dict[str, str]] = Field(None, description="Confirmation d'action")
    value: Optional[str] = Field(None, description="Valeur de l'action")


class SlackButtonSchema(BaseModel):
    """Schéma pour les boutons Slack."""
    type: str = Field("button", description="Type d'élément")
    text: Dict[str, str] = Field(..., description="Texte du bouton")
    action_id: str = Field(..., description="ID de l'action")
    url: Optional[HttpUrl] = Field(None, description="URL du bouton")
    value: Optional[str] = Field(None, description="Valeur du bouton")
    style: Optional[str] = Field(None, description="Style du bouton")
    confirm: Optional[Dict[str, Any]] = Field(None, description="Dialogue de confirmation")


class SlackSelectSchema(BaseModel):
    """Schéma pour les menus de sélection Slack."""
    type: str = Field("static_select", description="Type de sélection")
    placeholder: Dict[str, str] = Field(..., description="Texte placeholder")
    action_id: str = Field(..., description="ID de l'action")
    options: List[Dict[str, Any]] = Field(..., description="Options disponibles")
    initial_option: Optional[Dict[str, Any]] = Field(None, description="Option initiale")
    confirm: Optional[Dict[str, Any]] = Field(None, description="Dialogue de confirmation")


class SlackBlockSchema(BaseModel):
    """Schéma pour les blocs Slack."""
    type: SlackBlockType = Field(..., description="Type de bloc")
    block_id: Optional[str] = Field(None, description="ID du bloc")
    text: Optional[Dict[str, str]] = Field(None, description="Texte du bloc")
    fields: Optional[List[Dict[str, str]]] = Field(None, description="Champs du bloc")
    accessory: Optional[Dict[str, Any]] = Field(None, description="Accessoire du bloc")
    elements: Optional[List[Dict[str, Any]]] = Field(None, description="Éléments du bloc")
    image_url: Optional[HttpUrl] = Field(None, description="URL de l'image")
    alt_text: Optional[str] = Field(None, description="Texte alternatif")
    
    class Config:
        use_enum_values = True


class SlackElementSchema(BaseModel):
    """Schéma pour les éléments Slack."""
    type: SlackElementType = Field(..., description="Type d'élément")
    action_id: Optional[str] = Field(None, description="ID de l'action")
    placeholder: Optional[Dict[str, str]] = Field(None, description="Placeholder")
    initial_value: Optional[str] = Field(None, description="Valeur initiale")
    multiline: Optional[bool] = Field(None, description="Multiline pour input")
    options: Optional[List[Dict[str, Any]]] = Field(None, description="Options pour select")
    
    class Config:
        use_enum_values = True


class SlackAttachmentSchema(BaseModel):
    """Schéma pour les attachements Slack."""
    color: Optional[str] = Field(None, description="Couleur de l'attachement")
    fallback: Optional[str] = Field(None, description="Texte de fallback")
    author_name: Optional[str] = Field(None, description="Nom de l'auteur")
    author_link: Optional[HttpUrl] = Field(None, description="Lien de l'auteur")
    author_icon: Optional[HttpUrl] = Field(None, description="Icône de l'auteur")
    title: Optional[str] = Field(None, description="Titre de l'attachement")
    title_link: Optional[HttpUrl] = Field(None, description="Lien du titre")
    text: Optional[str] = Field(None, description="Texte de l'attachement")
    fields: List[SlackFieldSchema] = Field(default_factory=list, description="Champs de l'attachement")
    image_url: Optional[HttpUrl] = Field(None, description="URL de l'image")
    thumb_url: Optional[HttpUrl] = Field(None, description="URL de la miniature")
    footer: Optional[str] = Field(None, description="Pied de page")
    footer_icon: Optional[HttpUrl] = Field(None, description="Icône du pied de page")
    ts: Optional[int] = Field(None, description="Timestamp")
    actions: List[SlackActionSchema] = Field(default_factory=list, description="Actions disponibles")
    callback_id: Optional[str] = Field(None, description="ID de callback")
    attachment_type: Optional[str] = Field(None, description="Type d'attachement")


class SlackModalSchema(BaseModel):
    """Schéma pour les modales Slack."""
    type: str = Field("modal", description="Type de vue")
    title: Dict[str, str] = Field(..., description="Titre de la modale")
    blocks: List[SlackBlockSchema] = Field(..., description="Blocs de la modale")
    close: Optional[Dict[str, str]] = Field(None, description="Bouton fermer")
    submit: Optional[Dict[str, str]] = Field(None, description="Bouton soumettre")
    private_metadata: Optional[str] = Field(None, description="Métadonnées privées")
    callback_id: Optional[str] = Field(None, description="ID de callback")
    clear_on_close: bool = Field(False, description="Effacer à la fermeture")
    notify_on_close: bool = Field(False, description="Notifier à la fermeture")
    external_id: Optional[str] = Field(None, description="ID externe")


class SlackMessageSchema(BaseModel):
    """Schéma principal pour les messages Slack."""
    channel: str = Field(..., description="Canal de destination")
    text: Optional[str] = Field(None, description="Texte du message")
    username: Optional[str] = Field(None, description="Nom d'utilisateur du bot")
    icon_emoji: Optional[str] = Field(None, description="Emoji d'icône")
    icon_url: Optional[HttpUrl] = Field(None, description="URL d'icône")
    link_names: bool = Field(True, description="Lier les noms d'utilisateur")
    unfurl_links: bool = Field(True, description="Déplier les liens")
    unfurl_media: bool = Field(True, description="Déplier les médias")
    parse: Optional[str] = Field(None, description="Mode de parsing")
    reply_broadcast: bool = Field(False, description="Diffuser la réponse")
    thread_ts: Optional[str] = Field(None, description="Timestamp du thread")
    mrkdwn: bool = Field(True, description="Activer le markdown")
    attachments: List[SlackAttachmentSchema] = Field(default_factory=list, description="Attachements")
    blocks: List[SlackBlockSchema] = Field(default_factory=list, description="Blocs du message")
    
    @validator('channel')
    def validate_channel(cls, v):
        """Valide le format du canal."""
        if not v.startswith('#') and not v.startswith('@') and not re.match(r'^[A-Z0-9]+$', v):
            raise ValueError('Format de canal invalide')
        return v


class SlackTemplateSchema(BaseModel):
    """Schéma pour les templates de messages Slack."""
    name: str = Field(..., description="Nom du template")
    description: Optional[str] = Field(None, description="Description du template")
    template_type: SlackMessageType = Field(..., description="Type de template")
    variables: List[str] = Field(default_factory=list, description="Variables du template")
    default_values: Dict[str, str] = Field(default_factory=dict, description="Valeurs par défaut")
    message_template: SlackMessageSchema = Field(..., description="Template de message")
    conditions: List[Dict[str, Any]] = Field(default_factory=list, description="Conditions d'application")
    priority: int = Field(0, description="Priorité du template")
    
    class Config:
        use_enum_values = True


class SlackWebhookSchema(BaseModel):
    """Schéma pour les webhooks Slack."""
    url: HttpUrl = Field(..., description="URL du webhook")
    channel: Optional[str] = Field(None, description="Canal par défaut")
    username: Optional[str] = Field(None, description="Nom d'utilisateur par défaut")
    icon_emoji: Optional[str] = Field(None, description="Emoji d'icône par défaut")
    icon_url: Optional[HttpUrl] = Field(None, description="URL d'icône par défaut")
    link_names: bool = Field(True, description="Lier les noms par défaut")
    timeout: int = Field(30, description="Timeout en secondes")
    retry_count: int = Field(3, description="Nombre de tentatives")
    retry_delay: int = Field(1, description="Délai entre tentatives")


class SlackChannelSchema(BaseModel):
    """Schéma pour les canaux Slack."""
    id: str = Field(..., description="ID du canal")
    name: str = Field(..., description="Nom du canal")
    is_private: bool = Field(False, description="Canal privé")
    is_archived: bool = Field(False, description="Canal archivé")
    is_general: bool = Field(False, description="Canal général")
    members: List[str] = Field(default_factory=list, description="Membres du canal")
    topic: Optional[str] = Field(None, description="Sujet du canal")
    purpose: Optional[str] = Field(None, description="But du canal")
    created: Optional[datetime] = Field(None, description="Date de création")
    creator: Optional[str] = Field(None, description="Créateur du canal")


class SlackUserGroupSchema(BaseModel):
    """Schéma pour les groupes d'utilisateurs Slack."""
    id: str = Field(..., description="ID du groupe")
    name: str = Field(..., description="Nom du groupe")
    handle: str = Field(..., description="Handle du groupe")
    description: Optional[str] = Field(None, description="Description du groupe")
    is_external: bool = Field(False, description="Groupe externe")
    date_create: Optional[datetime] = Field(None, description="Date de création")
    date_update: Optional[datetime] = Field(None, description="Date de mise à jour")
    date_delete: Optional[datetime] = Field(None, description="Date de suppression")
    auto_type: Optional[str] = Field(None, description="Type automatique")
    created_by: Optional[str] = Field(None, description="Créé par")
    updated_by: Optional[str] = Field(None, description="Mis à jour par")
    deleted_by: Optional[str] = Field(None, description="Supprimé par")
    prefs: Dict[str, Any] = Field(default_factory=dict, description="Préférences")
    users: List[str] = Field(default_factory=list, description="Utilisateurs du groupe")


class SlackConfigSchema(BaseModel):
    """Configuration complète de l'intégration Slack."""
    api_token: str = Field(..., description="Token d'API Slack")
    bot_token: Optional[str] = Field(None, description="Token du bot")
    app_token: Optional[str] = Field(None, description="Token de l'application")
    signing_secret: str = Field(..., description="Secret de signature")
    verification_token: Optional[str] = Field(None, description="Token de vérification")
    
    default_channel: str = Field(..., description="Canal par défaut")
    default_username: str = Field("Alert Bot", description="Nom d'utilisateur par défaut")
    default_icon_emoji: str = Field(":warning:", description="Emoji d'icône par défaut")
    
    webhooks: List[SlackWebhookSchema] = Field(default_factory=list, description="Webhooks configurés")
    channels: List[SlackChannelSchema] = Field(default_factory=list, description="Canaux disponibles")
    user_groups: List[SlackUserGroupSchema] = Field(default_factory=list, description="Groupes d'utilisateurs")
    templates: List[SlackTemplateSchema] = Field(default_factory=list, description="Templates de messages")
    
    rate_limit: Dict[str, int] = Field(default_factory=lambda: {"messages_per_minute": 60}, description="Limites de débit")
    retry_config: Dict[str, int] = Field(default_factory=lambda: {"max_retries": 3, "backoff_factor": 2}, description="Configuration de retry")
    timeout_config: Dict[str, int] = Field(default_factory=lambda: {"connect": 10, "read": 30}, description="Configuration des timeouts")
    
    features: Dict[str, bool] = Field(
        default_factory=lambda: {
            "interactive_messages": True,
            "slash_commands": True,
            "event_subscriptions": True,
            "bot_events": True,
            "user_events": False
        },
        description="Fonctionnalités activées"
    )
    
    security: Dict[str, Any] = Field(
        default_factory=lambda: {
            "verify_requests": True,
            "encrypt_tokens": True,
            "audit_logs": True,
            "ip_whitelist": []
        },
        description="Configuration de sécurité"
    )
    
    monitoring: Dict[str, Any] = Field(
        default_factory=lambda: {
            "track_message_delivery": True,
            "track_user_interactions": True,
            "log_api_calls": True,
            "alert_on_failures": True
        },
        description="Configuration de monitoring"
    )
    
    @validator('api_token', 'bot_token', 'app_token')
    def validate_tokens(cls, v):
        """Valide le format des tokens Slack."""
        if v and not v.startswith('xoxb-') and not v.startswith('xoxp-') and not v.startswith('xapp-'):
            raise ValueError('Format de token Slack invalide')
        return v
    
    @validator('default_channel')
    def validate_default_channel(cls, v):
        """Valide le canal par défaut."""
        if not v.startswith('#') and not v.startswith('@') and not re.match(r'^[A-Z0-9]+$', v):
            raise ValueError('Format de canal par défaut invalide')
        return v
