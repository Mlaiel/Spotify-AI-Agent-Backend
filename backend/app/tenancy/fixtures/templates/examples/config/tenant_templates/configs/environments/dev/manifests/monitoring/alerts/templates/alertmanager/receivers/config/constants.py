"""
Constantes et énumérations pour la configuration Alertmanager Receivers

Ce module centralise toutes les constantes, énumérations et configurations
par défaut utilisées dans le système de receivers.

Author: Spotify AI Agent Team
Maintainer: Fahed Mlaiel - Lead Dev + Architecte IA
"""

from enum import Enum
from typing import Dict, List, Any

# ============================================================================
# ÉNUMÉRATIONS PRINCIPALES
# ============================================================================

class TenantTier(Enum):
    """Niveaux de service par tenant"""
    FREE = "free"
    PREMIUM = "premium"
    FAMILY = "family"
    STUDENT = "student"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"

class AlertSeverity(Enum):
    """Niveaux de sévérité des alertes"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"
    DEBUG = "debug"

class NotificationChannel(Enum):
    """Canaux de notification disponibles"""
    SLACK = "slack"
    EMAIL = "email"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    TEAMS = "teams"
    DISCORD = "discord"
    JIRA = "jira"
    SERVICENOW = "servicenow"
    OPSGENIE = "opsgenie"
    DATADOG = "datadog"
    SMS = "sms"
    VOICE = "voice"
    MOBILE_PUSH = "mobile_push"
    TELEGRAM = "telegram"
    WHATSAPP = "whatsapp"

class EscalationLevel(Enum):
    """Niveaux d'escalade"""
    LEVEL_1 = "level_1"  # Équipe de garde
    LEVEL_2 = "level_2"  # Lead technique
    LEVEL_3 = "level_3"  # Manager d'équipe
    LEVEL_4 = "level_4"  # Direction technique
    LEVEL_5 = "level_5"  # C-Level

class SecurityLevel(Enum):
    """Niveaux de sécurité"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"

class IntegrationStatus(Enum):
    """Statuts d'intégration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    DEGRADED = "degraded"

class AlertStatus(Enum):
    """Statuts des alertes"""
    FIRING = "firing"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SILENCED = "silenced"
    ESCALATED = "escalated"
    EXPIRED = "expired"

# ============================================================================
# TIMEOUTS ET DÉLAIS
# ============================================================================

DEFAULT_TIMEOUTS = {
    # Timeouts de base (en secondes)
    "default_timeout": 30,
    "short_timeout": 10,
    "medium_timeout": 60,
    "long_timeout": 300,
    "critical_timeout": 5,
    
    # Timeouts par intégration
    "slack_timeout": 15,
    "email_timeout": 30,
    "pagerduty_timeout": 20,
    "webhook_timeout": 25,
    "jira_timeout": 45,
    "datadog_timeout": 20,
    
    # Timeouts de résolution d'alertes
    "resolve_timeout": 300,  # 5 minutes
    "escalation_timeout": 900,  # 15 minutes
    "auto_resolve_timeout": 3600,  # 1 heure
    
    # Délais de retry
    "retry_delay_initial": 2,
    "retry_delay_max": 60,
    "retry_backoff_multiplier": 2,
    
    # Cooldowns
    "escalation_cooldown": 600,  # 10 minutes
    "notification_cooldown": 180,  # 3 minutes
    "health_check_interval": 60,  # 1 minute
}

# ============================================================================
# LIMITES ET QUOTAS
# ============================================================================

RATE_LIMITS = {
    # Limites par minute
    "default_per_minute": 100,
    "premium_per_minute": 1000,
    "enterprise_per_minute": 5000,
    
    # Limites par heure
    "default_per_hour": 1000,
    "premium_per_hour": 10000,
    "enterprise_per_hour": 50000,
    
    # Limites par intégration
    "slack_per_minute": 60,
    "pagerduty_per_minute": 120,
    "email_per_minute": 200,
    "webhook_per_minute": 300,
    
    # Limites d'escalade
    "max_escalations_per_hour": 10,
    "max_escalation_levels": 5,
    
    # Limites de retry
    "max_retry_attempts": 5,
    "max_parallel_notifications": 20,
}

# ============================================================================
# CONFIGURATIONS PAR SÉVÉRITÉ
# ============================================================================

SEVERITY_LEVELS = {
    AlertSeverity.CRITICAL: {
        "priority": 1,
        "escalation_delay": 300,  # 5 minutes
        "max_escalation_levels": 5,
        "required_channels": [NotificationChannel.PAGERDUTY, NotificationChannel.SLACK],
        "auto_escalate": True,
        "business_hours_only": False,
        "sla_response_time": 300,  # 5 minutes
        "color": "#FF0000",
        "emoji": "🚨",
        "sound": "critical_alarm"
    },
    AlertSeverity.HIGH: {
        "priority": 2,
        "escalation_delay": 600,  # 10 minutes
        "max_escalation_levels": 3,
        "required_channels": [NotificationChannel.SLACK, NotificationChannel.EMAIL],
        "auto_escalate": True,
        "business_hours_only": False,
        "sla_response_time": 900,  # 15 minutes
        "color": "#FF6600",
        "emoji": "⚠️",
        "sound": "high_alert"
    },
    AlertSeverity.MEDIUM: {
        "priority": 3,
        "escalation_delay": 1800,  # 30 minutes
        "max_escalation_levels": 2,
        "required_channels": [NotificationChannel.SLACK],
        "auto_escalate": False,
        "business_hours_only": True,
        "sla_response_time": 3600,  # 1 hour
        "color": "#FFAA00",
        "emoji": "🟡",
        "sound": "medium_beep"
    },
    AlertSeverity.LOW: {
        "priority": 4,
        "escalation_delay": 3600,  # 1 hour
        "max_escalation_levels": 1,
        "required_channels": [NotificationChannel.EMAIL],
        "auto_escalate": False,
        "business_hours_only": True,
        "sla_response_time": 7200,  # 2 hours
        "color": "#0066FF",
        "emoji": "🔵",
        "sound": "low_ping"
    },
    AlertSeverity.INFO: {
        "priority": 5,
        "escalation_delay": 0,  # Pas d'escalade
        "max_escalation_levels": 0,
        "required_channels": [],
        "auto_escalate": False,
        "business_hours_only": True,
        "sla_response_time": 86400,  # 24 hours
        "color": "#00AA00",
        "emoji": "ℹ️",
        "sound": "info_chime"
    }
}

# ============================================================================
# CANAUX DE NOTIFICATION
# ============================================================================

NOTIFICATION_CHANNELS = {
    NotificationChannel.SLACK: {
        "name": "Slack",
        "type": "messaging",
        "supports_rich_formatting": True,
        "supports_threading": True,
        "supports_reactions": True,
        "max_message_length": 40000,
        "supports_attachments": True,
        "rate_limit_per_minute": 60,
        "typical_delivery_time": 2,  # secondes
        "reliability_score": 0.99
    },
    NotificationChannel.EMAIL: {
        "name": "Email",
        "type": "messaging", 
        "supports_rich_formatting": True,
        "supports_threading": False,
        "supports_reactions": False,
        "max_message_length": 1000000,
        "supports_attachments": True,
        "rate_limit_per_minute": 200,
        "typical_delivery_time": 30,
        "reliability_score": 0.95
    },
    NotificationChannel.PAGERDUTY: {
        "name": "PagerDuty",
        "type": "incident_management",
        "supports_rich_formatting": False,
        "supports_threading": False,
        "supports_reactions": False,
        "max_message_length": 1024,
        "supports_attachments": False,
        "rate_limit_per_minute": 120,
        "typical_delivery_time": 5,
        "reliability_score": 0.999
    },
    NotificationChannel.WEBHOOK: {
        "name": "Generic Webhook",
        "type": "integration",
        "supports_rich_formatting": True,
        "supports_threading": False,
        "supports_reactions": False,
        "max_message_length": 65536,
        "supports_attachments": False,
        "rate_limit_per_minute": 300,
        "typical_delivery_time": 3,
        "reliability_score": 0.90
    },
    NotificationChannel.JIRA: {
        "name": "Jira",
        "type": "ticketing",
        "supports_rich_formatting": True,
        "supports_threading": True,
        "supports_reactions": False,
        "max_message_length": 32768,
        "supports_attachments": True,
        "rate_limit_per_minute": 100,
        "typical_delivery_time": 10,
        "reliability_score": 0.97
    }
}

# ============================================================================
# POLITIQUES D'ESCALADE
# ============================================================================

ESCALATION_POLICIES = {
    "default": {
        "name": "Politique par défaut",
        "levels": [
            {
                "level": 1,
                "delay_minutes": 15,
                "targets": ["on_call_engineer"],
                "channels": [NotificationChannel.PAGERDUTY, NotificationChannel.SLACK]
            },
            {
                "level": 2,
                "delay_minutes": 30,
                "targets": ["team_lead"],
                "channels": [NotificationChannel.PAGERDUTY, NotificationChannel.EMAIL]
            },
            {
                "level": 3,
                "delay_minutes": 60,
                "targets": ["engineering_manager"],
                "channels": [NotificationChannel.EMAIL, NotificationChannel.SMS]
            }
        ]
    },
    "premium": {
        "name": "Politique Premium",
        "levels": [
            {
                "level": 1,
                "delay_minutes": 5,
                "targets": ["premium_on_call"],
                "channels": [NotificationChannel.PAGERDUTY, NotificationChannel.SLACK, NotificationChannel.SMS]
            },
            {
                "level": 2,
                "delay_minutes": 10,
                "targets": ["premium_lead", "backup_engineer"],
                "channels": [NotificationChannel.PAGERDUTY, NotificationChannel.VOICE]
            },
            {
                "level": 3,
                "delay_minutes": 15,
                "targets": ["premium_manager", "site_reliability_lead"],
                "channels": [NotificationChannel.EMAIL, NotificationChannel.SMS]
            },
            {
                "level": 4,
                "delay_minutes": 30,
                "targets": ["vp_engineering"],
                "channels": [NotificationChannel.EMAIL, NotificationChannel.VOICE]
            }
        ]
    },
    "critical_service": {
        "name": "Services Critiques",
        "levels": [
            {
                "level": 1,
                "delay_minutes": 0,  # Immédiat
                "targets": ["critical_response_team"],
                "channels": [NotificationChannel.PAGERDUTY, NotificationChannel.SLACK, 
                           NotificationChannel.SMS, NotificationChannel.VOICE]
            },
            {
                "level": 2,
                "delay_minutes": 5,
                "targets": ["senior_sre", "service_owner", "backup_team"],
                "channels": [NotificationChannel.PAGERDUTY, NotificationChannel.VOICE]
            },
            {
                "level": 3,
                "delay_minutes": 10,
                "targets": ["engineering_director", "cto"],
                "channels": [NotificationChannel.EMAIL, NotificationChannel.SMS, 
                           NotificationChannel.VOICE]
            }
        ]
    }
}

# ============================================================================
# POLITIQUES DE SÉCURITÉ
# ============================================================================

SECURITY_POLICIES = {
    "encryption": {
        "algorithms": ["AES-256-GCM", "ChaCha20-Poly1305"],
        "key_rotation_days": 30,
        "min_key_length": 256,
        "require_tls": True,
        "tls_min_version": "1.2"
    },
    "authentication": {
        "methods": ["JWT", "API_KEY", "OAUTH2", "MUTUAL_TLS"],
        "token_lifetime_seconds": 3600,
        "require_2fa_for_admin": True,
        "password_policy": {
            "min_length": 12,
            "require_uppercase": True,
            "require_lowercase": True,
            "require_numbers": True,
            "require_symbols": True
        }
    },
    "access_control": {
        "default_permissions": ["read"],
        "admin_permissions": ["read", "write", "delete", "admin"],
        "tenant_isolation": True,
        "rbac_enabled": True,
        "audit_all_actions": True
    },
    "data_protection": {
        "encrypt_at_rest": True,
        "encrypt_in_transit": True,
        "data_retention_days": 90,
        "pii_anonymization": True,
        "gdpr_compliance": True
    }
}

# ============================================================================
# CONFIGURATIONS MÉTIER SPOTIFY
# ============================================================================

SPOTIFY_BUSINESS_RULES = {
    "service_categories": {
        "core": ["music_streaming", "search", "recommendations", "user_auth"],
        "premium": ["high_quality_audio", "offline_sync", "ad_free"],
        "social": ["playlists", "sharing", "social_features"],
        "analytics": ["usage_tracking", "recommendations_ml", "content_analysis"],
        "infrastructure": ["cdn", "storage", "compute", "networking"],
        "payments": ["billing", "subscriptions", "payments_processing"]
    },
    "sla_requirements": {
        TenantTier.FREE: {
            "availability": "99.5%",
            "response_time_p95": "2000ms",
            "error_rate_max": "1%"
        },
        TenantTier.PREMIUM: {
            "availability": "99.9%", 
            "response_time_p95": "500ms",
            "error_rate_max": "0.1%"
        },
        TenantTier.FAMILY: {
            "availability": "99.8%",
            "response_time_p95": "800ms", 
            "error_rate_max": "0.5%"
        },
        TenantTier.ENTERPRISE: {
            "availability": "99.99%",
            "response_time_p95": "200ms",
            "error_rate_max": "0.01%"
        }
    },
    "geographic_regions": {
        "north_america": ["us-east-1", "us-west-2", "ca-central-1"],
        "europe": ["eu-west-1", "eu-central-1", "eu-north-1"],
        "asia_pacific": ["ap-southeast-1", "ap-northeast-1", "ap-south-1"],
        "south_america": ["sa-east-1"],
        "africa": ["af-south-1"]
    }
}

# ============================================================================
# TEMPLATES DE MESSAGES
# ============================================================================

MESSAGE_TEMPLATES = {
    "alert_summary": {
        "short": "🚨 {severity} Alert: {service} on {tenant}",
        "medium": "🚨 {severity} Alert for {service}\nTenant: {tenant}\nTime: {timestamp}",
        "detailed": """
🚨 **{severity_emoji} {severity} ALERT**

**Service:** {service}
**Tenant:** {tenant} ({tier})
**Time:** {timestamp}
**Description:** {description}

**Metrics:**
{metrics}

**Actions:**
- 🔍 [Investigate]({monitoring_url})
- 📖 [Runbook]({runbook_url})
- 🛠️ [Dashboard]({dashboard_url})
"""
    },
    "escalation_notice": {
        "template": """
⚡ **ESCALATION NOTICE** - Level {level}

The following alert has been escalated due to no response:

**Alert:** {alert_summary}
**Original Time:** {original_timestamp}
**Escalation Time:** {escalation_timestamp}
**Previous Assignee:** {previous_assignee}
**New Assignee:** {new_assignee}

**Required Action:** Immediate response required within {sla_time} minutes.
"""
    },
    "resolution_notice": {
        "template": """
✅ **ALERT RESOLVED**

**Alert:** {alert_summary}
**Resolved By:** {resolved_by}
**Resolution Time:** {resolution_timestamp}
**Total Duration:** {total_duration}
**Root Cause:** {root_cause}

Thank you for your swift response!
"""
    }
}

# ============================================================================
# MÉTRIQUES ET KPI
# ============================================================================

METRICS_CONFIG = {
    "collection_interval_seconds": 30,
    "retention_days": 90,
    "aggregation_intervals": ["1m", "5m", "15m", "1h", "1d"],
    "key_metrics": [
        "alert_volume",
        "response_time", 
        "escalation_rate",
        "resolution_time",
        "false_positive_rate",
        "integration_success_rate",
        "system_availability"
    ],
    "sla_thresholds": {
        "critical_response_time": 300,  # 5 minutes
        "high_response_time": 900,      # 15 minutes
        "medium_response_time": 3600,   # 1 hour
        "escalation_rate_max": 0.1,     # 10%
        "false_positive_rate_max": 0.05  # 5%
    }
}

# ============================================================================
# CONFIGURATION DE DÉVELOPPEMENT
# ============================================================================

DEVELOPMENT_SETTINGS = {
    "debug_mode": True,
    "log_level": "DEBUG",
    "mock_integrations": True,
    "test_tenant": "spotify-test",
    "development_webhooks": {
        "slack": "https://hooks.slack.com/services/TEST/TEST/TEST",
        "general": "https://webhook.site/test-endpoint"
    },
    "bypass_rate_limits": True,
    "enable_detailed_logging": True
}

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "TenantTier",
    "AlertSeverity", 
    "NotificationChannel",
    "EscalationLevel",
    "SecurityLevel",
    "IntegrationStatus",
    "AlertStatus",
    
    # Constantes
    "DEFAULT_TIMEOUTS",
    "RATE_LIMITS",
    "SEVERITY_LEVELS",
    "NOTIFICATION_CHANNELS",
    "ESCALATION_POLICIES",
    "SECURITY_POLICIES",
    "SPOTIFY_BUSINESS_RULES",
    "MESSAGE_TEMPLATES",
    "METRICS_CONFIG",
    "DEVELOPMENT_SETTINGS"
]
