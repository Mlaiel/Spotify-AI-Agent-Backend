"""
Énumérations avancées pour la configuration Alertmanager Receivers

Ce module définit toutes les énumérations utilisées dans le système
avec une organisation claire et une documentation complète.

Author: Spotify AI Agent Team
Maintainer: Fahed Mlaiel - Lead Dev + Architecte IA
"""

from enum import Enum, IntEnum, Flag, auto
from typing import Dict, List, Any

# ============================================================================
# ÉNUMÉRATIONS DE BASE
# ============================================================================

class TenantTier(Enum):
    """Niveaux de service par tenant avec métadonnées"""
    FREE = "free"
    PREMIUM = "premium"
    FAMILY = "family"
    STUDENT = "student"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"
    
    @property
    def priority_level(self) -> int:
        """Retourne le niveau de priorité (plus bas = plus prioritaire)"""
        priority_map = {
            self.ENTERPRISE: 1,
            self.BUSINESS: 2,
            self.PREMIUM: 3,
            self.FAMILY: 4,
            self.STUDENT: 5,
            self.FREE: 6
        }
        return priority_map[self]
    
    @property
    def max_alerts_per_hour(self) -> int:
        """Nombre maximum d'alertes par heure"""
        limits = {
            self.ENTERPRISE: 10000,
            self.BUSINESS: 5000,
            self.PREMIUM: 1000,
            self.FAMILY: 500,
            self.STUDENT: 200,
            self.FREE: 100
        }
        return limits[self]
    
    @property
    def sla_availability(self) -> float:
        """SLA de disponibilité en pourcentage"""
        sla_map = {
            self.ENTERPRISE: 99.99,
            self.BUSINESS: 99.9,
            self.PREMIUM: 99.9,
            self.FAMILY: 99.8,
            self.STUDENT: 99.5,
            self.FREE: 99.5
        }
        return sla_map[self]

class AlertSeverity(IntEnum):
    """Niveaux de sévérité avec valeurs numériques pour comparaison"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    INFO = 5
    DEBUG = 6
    
    @property
    def emoji(self) -> str:
        """Emoji représentant la sévérité"""
        emoji_map = {
            self.CRITICAL: "🚨",
            self.HIGH: "⚠️",
            self.MEDIUM: "🟡",
            self.LOW: "🔵",
            self.INFO: "ℹ️",
            self.DEBUG: "🔧"
        }
        return emoji_map[self]
    
    @property
    def color_hex(self) -> str:
        """Couleur hexadécimale pour l'affichage"""
        color_map = {
            self.CRITICAL: "#FF0000",
            self.HIGH: "#FF6600",
            self.MEDIUM: "#FFAA00",
            self.LOW: "#0066FF",
            self.INFO: "#00AA00",
            self.DEBUG: "#888888"
        }
        return color_map[self]
    
    @property
    def escalation_delay_minutes(self) -> int:
        """Délai d'escalade en minutes"""
        delay_map = {
            self.CRITICAL: 5,
            self.HIGH: 15,
            self.MEDIUM: 30,
            self.LOW: 60,
            self.INFO: 0,  # Pas d'escalade
            self.DEBUG: 0  # Pas d'escalade
        }
        return delay_map[self]

class NotificationChannel(Enum):
    """Canaux de notification avec capacités"""
    # Messaging platforms
    SLACK = "slack"
    TEAMS = "teams"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    WHATSAPP = "whatsapp"
    
    # Traditional communication
    EMAIL = "email"
    SMS = "sms"
    VOICE = "voice"
    MOBILE_PUSH = "mobile_push"
    
    # Incident management
    PAGERDUTY = "pagerduty"
    OPSGENIE = "opsgenie"
    XMATTERS = "xmatters"
    VICTORIOPS = "victoriops"
    
    # Ticketing systems
    JIRA = "jira"
    SERVICENOW = "servicenow"
    ZENDESK = "zendesk"
    FRESHDESK = "freshdesk"
    
    # Monitoring & Observability
    DATADOG = "datadog"
    NEW_RELIC = "new_relic"
    SPLUNK = "splunk"
    ELASTIC = "elastic"
    GRAFANA = "grafana"
    
    # Generic integrations
    WEBHOOK = "webhook"
    HTTP_POST = "http_post"
    CUSTOM = "custom"
    
    @property
    def supports_rich_formatting(self) -> bool:
        """Supporte le formatage riche (markdown, HTML, etc.)"""
        rich_channels = {
            self.SLACK, self.TEAMS, self.DISCORD, self.EMAIL,
            self.JIRA, self.SERVICENOW, self.GRAFANA, self.WEBHOOK
        }
        return self in rich_channels
    
    @property
    def supports_attachments(self) -> bool:
        """Supporte les pièces jointes"""
        attachment_channels = {
            self.EMAIL, self.SLACK, self.TEAMS, self.JIRA,
            self.SERVICENOW, self.ZENDESK, self.WEBHOOK
        }
        return self in attachment_channels
    
    @property
    def typical_delivery_time_seconds(self) -> int:
        """Temps de livraison typique en secondes"""
        delivery_times = {
            self.SLACK: 2,
            self.TEAMS: 3,
            self.DISCORD: 2,
            self.EMAIL: 30,
            self.SMS: 10,
            self.VOICE: 5,
            self.MOBILE_PUSH: 5,
            self.PAGERDUTY: 5,
            self.OPSGENIE: 8,
            self.JIRA: 15,
            self.SERVICENOW: 20,
            self.WEBHOOK: 3,
            self.HTTP_POST: 2
        }
        return delivery_times.get(self, 10)
    
    @property
    def reliability_score(self) -> float:
        """Score de fiabilité de 0 à 1"""
        reliability_scores = {
            self.PAGERDUTY: 0.999,
            self.OPSGENIE: 0.998,
            self.SLACK: 0.99,
            self.TEAMS: 0.98,
            self.EMAIL: 0.95,
            self.SMS: 0.92,
            self.VOICE: 0.90,
            self.JIRA: 0.97,
            self.SERVICENOW: 0.96,
            self.WEBHOOK: 0.90,
            self.HTTP_POST: 0.88
        }
        return reliability_scores.get(self, 0.85)

class EscalationLevel(IntEnum):
    """Niveaux d'escalade avec ordre hiérarchique"""
    LEVEL_1 = 1  # Équipe de garde
    LEVEL_2 = 2  # Lead technique
    LEVEL_3 = 3  # Manager d'équipe
    LEVEL_4 = 4  # Direction technique
    LEVEL_5 = 5  # C-Level / Executive
    
    @property
    def role_description(self) -> str:
        """Description du rôle pour ce niveau"""
        roles = {
            self.LEVEL_1: "On-Call Engineer / First Responder",
            self.LEVEL_2: "Technical Lead / Senior Engineer",
            self.LEVEL_3: "Engineering Manager / Team Lead",
            self.LEVEL_4: "Engineering Director / VP Engineering",
            self.LEVEL_5: "CTO / C-Level Executive"
        }
        return roles[self]
    
    @property
    def expected_response_time_minutes(self) -> int:
        """Temps de réponse attendu en minutes"""
        response_times = {
            self.LEVEL_1: 5,
            self.LEVEL_2: 10,
            self.LEVEL_3: 15,
            self.LEVEL_4: 30,
            self.LEVEL_5: 60
        }
        return response_times[self]

class SecurityLevel(Enum):
    """Niveaux de classification de sécurité"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"
    
    @property
    def clearance_required(self) -> bool:
        """Nécessite une habilitation de sécurité"""
        return self in {self.RESTRICTED, self.TOP_SECRET}
    
    @property
    def encryption_required(self) -> bool:
        """Nécessite un chiffrement"""
        return self in {self.CONFIDENTIAL, self.RESTRICTED, self.TOP_SECRET}
    
    @property
    def audit_level(self) -> str:
        """Niveau d'audit requis"""
        audit_levels = {
            self.PUBLIC: "basic",
            self.INTERNAL: "standard",
            self.CONFIDENTIAL: "detailed",
            self.RESTRICTED: "comprehensive",
            self.TOP_SECRET: "maximum"
        }
        return audit_levels[self]

# ============================================================================
# ÉNUMÉRATIONS D'ÉTAT
# ============================================================================

class AlertStatus(Enum):
    """Statuts possibles d'une alerte"""
    FIRING = "firing"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SILENCED = "silenced"
    ESCALATED = "escalated"
    EXPIRED = "expired"
    SUPPRESSED = "suppressed"
    PENDING = "pending"
    
    @property
    def is_active(self) -> bool:
        """L'alerte est-elle active"""
        active_states = {self.FIRING, self.ESCALATED, self.PENDING}
        return self in active_states
    
    @property
    def requires_action(self) -> bool:
        """Nécessite-t-elle une action"""
        action_required = {self.FIRING, self.ESCALATED}
        return self in action_required

class IntegrationStatus(Enum):
    """Statuts d'intégration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    DEGRADED = "degraded"
    INITIALIZING = "initializing"
    UNKNOWN = "unknown"
    
    @property
    def is_operational(self) -> bool:
        """L'intégration est-elle opérationnelle"""
        operational_states = {self.ACTIVE, self.DEGRADED}
        return self in operational_states
    
    @property
    def health_score(self) -> float:
        """Score de santé de 0 à 1"""
        health_scores = {
            self.ACTIVE: 1.0,
            self.DEGRADED: 0.7,
            self.MAINTENANCE: 0.5,
            self.INITIALIZING: 0.3,
            self.ERROR: 0.1,
            self.INACTIVE: 0.0,
            self.UNKNOWN: 0.0
        }
        return health_scores[self]

class ConfigValidationStatus(Enum):
    """Statuts de validation de configuration"""
    VALID = "valid"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    
    @property
    def is_blocking(self) -> bool:
        """Le statut bloque-t-il l'exécution"""
        blocking_states = {self.ERROR, self.CRITICAL}
        return self in blocking_states

# ============================================================================
# ÉNUMÉRATIONS TECHNIQUES
# ============================================================================

class EncryptionAlgorithm(Enum):
    """Algorithmes de chiffrement supportés"""
    AES_256_GCM = "aes-256-gcm"
    AES_256_CBC = "aes-256-cbc"
    CHACHA20_POLY1305 = "chacha20-poly1305"
    AES_128_GCM = "aes-128-gcm"
    
    @property
    def key_size_bits(self) -> int:
        """Taille de clé en bits"""
        key_sizes = {
            self.AES_256_GCM: 256,
            self.AES_256_CBC: 256,
            self.CHACHA20_POLY1305: 256,
            self.AES_128_GCM: 128
        }
        return key_sizes[self]
    
    @property
    def is_aead(self) -> bool:
        """Est un algorithme AEAD (Authenticated Encryption with Associated Data)"""
        aead_algorithms = {self.AES_256_GCM, self.CHACHA20_POLY1305, self.AES_128_GCM}
        return self in aead_algorithms

class AuthenticationMethod(Enum):
    """Méthodes d'authentification"""
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    OAUTH2 = "oauth2"
    BASIC_AUTH = "basic_auth"
    BEARER_TOKEN = "bearer_token"
    MUTUAL_TLS = "mutual_tls"
    SAML = "saml"
    OIDC = "oidc"
    CERTIFICATE = "certificate"
    
    @property
    def security_level(self) -> int:
        """Niveau de sécurité de 1 (faible) à 5 (élevé)"""
        security_levels = {
            self.BASIC_AUTH: 2,
            self.API_KEY: 3,
            self.BEARER_TOKEN: 3,
            self.JWT_TOKEN: 4,
            self.OAUTH2: 4,
            self.OIDC: 4,
            self.SAML: 4,
            self.CERTIFICATE: 5,
            self.MUTUAL_TLS: 5
        }
        return security_levels[self]
    
    @property
    def supports_expiration(self) -> bool:
        """Supporte l'expiration automatique"""
        expiring_methods = {
            self.JWT_TOKEN, self.OAUTH2, self.BEARER_TOKEN, self.SAML, self.OIDC
        }
        return self in expiring_methods

class MessageFormat(Enum):
    """Formats de message supportés"""
    PLAIN_TEXT = "plain_text"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    XML = "xml"
    SLACK_BLOCKS = "slack_blocks"
    TEAMS_ADAPTIVE_CARDS = "teams_adaptive_cards"
    
    @property
    def supports_rich_content(self) -> bool:
        """Supporte le contenu riche"""
        rich_formats = {
            self.MARKDOWN, self.HTML, self.SLACK_BLOCKS, 
            self.TEAMS_ADAPTIVE_CARDS
        }
        return self in rich_formats

# ============================================================================
# FLAGS POUR PERMISSIONS
# ============================================================================

class Permission(Flag):
    """Permissions système avec support des opérations bitwise"""
    READ = auto()
    WRITE = auto()
    DELETE = auto()
    ADMIN = auto()
    AUDIT = auto()
    ESCALATE = auto()
    CONFIGURE = auto()
    MONITOR = auto()
    
    # Combinaisons courantes
    READ_WRITE = READ | WRITE
    FULL_ACCESS = READ | WRITE | DELETE | ADMIN
    OPERATOR = READ | WRITE | MONITOR
    ADMIN_FULL = READ | WRITE | DELETE | ADMIN | CONFIGURE | AUDIT

class FeatureFlag(Flag):
    """Flags de fonctionnalités"""
    BASIC_ALERTS = auto()
    RICH_FORMATTING = auto()
    ESCALATION = auto()
    AUTOMATION = auto()
    ML_INSIGHTS = auto()
    ADVANCED_ROUTING = auto()
    CUSTOM_INTEGRATIONS = auto()
    AUDIT_LOGGING = auto()
    ENCRYPTION = auto()
    MULTI_REGION = auto()
    
    # Packages de fonctionnalités
    FREE_TIER = BASIC_ALERTS
    PREMIUM_TIER = BASIC_ALERTS | RICH_FORMATTING | ESCALATION
    ENTERPRISE_TIER = (BASIC_ALERTS | RICH_FORMATTING | ESCALATION | 
                      AUTOMATION | ML_INSIGHTS | ADVANCED_ROUTING |
                      CUSTOM_INTEGRATIONS | AUDIT_LOGGING | ENCRYPTION |
                      MULTI_REGION)

# ============================================================================
# ÉNUMÉRATIONS MÉTIER SPOTIFY
# ============================================================================

class SpotifyService(Enum):
    """Services Spotify principaux"""
    # Core services
    MUSIC_STREAMING = "music_streaming"
    SEARCH = "search"
    RECOMMENDATIONS = "recommendations"
    USER_AUTH = "user_auth"
    
    # Premium features
    HIGH_QUALITY_AUDIO = "high_quality_audio"
    OFFLINE_SYNC = "offline_sync"
    AD_FREE = "ad_free"
    
    # Social features
    PLAYLISTS = "playlists"
    SHARING = "sharing"
    SOCIAL_FEATURES = "social_features"
    
    # Analytics & ML
    USAGE_TRACKING = "usage_tracking"
    RECOMMENDATIONS_ML = "recommendations_ml"
    CONTENT_ANALYSIS = "content_analysis"
    
    # Infrastructure
    CDN = "cdn"
    STORAGE = "storage"
    COMPUTE = "compute"
    NETWORKING = "networking"
    
    # Business
    BILLING = "billing"
    SUBSCRIPTIONS = "subscriptions"
    PAYMENTS = "payments"
    ADVERTISING = "advertising"
    
    @property
    def criticality_level(self) -> int:
        """Niveau de criticité de 1 (critique) à 5 (non-critique)"""
        criticality_map = {
            # Services critiques
            self.MUSIC_STREAMING: 1,
            self.USER_AUTH: 1,
            self.SEARCH: 1,
            
            # Services importants
            self.RECOMMENDATIONS: 2,
            self.HIGH_QUALITY_AUDIO: 2,
            self.PLAYLISTS: 2,
            self.BILLING: 2,
            
            # Services standards
            self.OFFLINE_SYNC: 3,
            self.SHARING: 3,
            self.SUBSCRIPTIONS: 3,
            
            # Services secondaires
            self.SOCIAL_FEATURES: 4,
            self.CONTENT_ANALYSIS: 4,
            self.ADVERTISING: 4,
            
            # Services non-critiques
            self.USAGE_TRACKING: 5
        }
        return criticality_map.get(self, 3)
    
    @property
    def requires_premium(self) -> bool:
        """Nécessite un abonnement Premium"""
        premium_services = {
            self.HIGH_QUALITY_AUDIO, self.OFFLINE_SYNC, self.AD_FREE
        }
        return self in premium_services

class SpotifyRegion(Enum):
    """Régions géographiques Spotify"""
    NORTH_AMERICA = "north_america"
    EUROPE = "europe"
    ASIA_PACIFIC = "asia_pacific"
    SOUTH_AMERICA = "south_america"
    AFRICA = "africa"
    OCEANIA = "oceania"
    
    @property
    def aws_regions(self) -> List[str]:
        """Régions AWS correspondantes"""
        region_map = {
            self.NORTH_AMERICA: ["us-east-1", "us-west-2", "ca-central-1"],
            self.EUROPE: ["eu-west-1", "eu-central-1", "eu-north-1"],
            self.ASIA_PACIFIC: ["ap-southeast-1", "ap-northeast-1", "ap-south-1"],
            self.SOUTH_AMERICA: ["sa-east-1"],
            self.AFRICA: ["af-south-1"],
            self.OCEANIA: ["ap-southeast-2"]
        }
        return region_map[self]
    
    @property
    def timezone_offset_hours(self) -> int:
        """Décalage horaire principal en heures par rapport à UTC"""
        timezone_map = {
            self.NORTH_AMERICA: -5,  # EST
            self.EUROPE: 1,          # CET
            self.ASIA_PACIFIC: 8,    # CST
            self.SOUTH_AMERICA: -3,  # BRT
            self.AFRICA: 2,          # SAST
            self.OCEANIA: 10         # AEST
        }
        return timezone_map[self]

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Énumérations principales
    "TenantTier",
    "AlertSeverity",
    "NotificationChannel", 
    "EscalationLevel",
    "SecurityLevel",
    
    # États
    "AlertStatus",
    "IntegrationStatus",
    "ConfigValidationStatus",
    
    # Techniques
    "EncryptionAlgorithm",
    "AuthenticationMethod",
    "MessageFormat",
    
    # Permissions et flags
    "Permission",
    "FeatureFlag",
    
    # Métier Spotify
    "SpotifyService",
    "SpotifyRegion"
]
