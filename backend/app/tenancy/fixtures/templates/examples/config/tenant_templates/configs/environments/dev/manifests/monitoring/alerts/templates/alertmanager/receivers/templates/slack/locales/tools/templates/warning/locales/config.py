#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spotify AI Agent - Configuration centrale pour Alerting Multi-Tenant

Configuration centralisée pour le système d'alerting avec support
des différents environnements, tenants et canaux de notification.

Contient:
- Configurations des locales supportées
- Templates d'alertes par défaut
- Mappings des canaux Slack
- Seuils et règles d'alerting
- Configurations environnement-spécifiques
- Intégrations tierces (Prometheus, PagerDuty, etc.)

Architecture:
- Configuration hiérarchique (global -> env -> tenant)
- Support hot-reload via signaux système
- Validation automatique des configurations
- Secrets management intégré
- Override par variables d'environnement

Utilisation:
    from config import (
        SUPPORTED_LOCALES,
        get_alert_config,
        get_slack_webhook
    )
    
    config = get_alert_config('production', 'universal_music')
    webhook = get_slack_webhook('critical_alerts')
"""

import os
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import yaml
import json
from pathlib import Path

# Locales supportées avec métadonnées
SUPPORTED_LOCALES = {
    "fr": {
        "name": "Français",
        "native_name": "Français",
        "iso_code": "fr-FR",
        "direction": "ltr",
        "fallback": "en",
        "enabled": True,
        "priority": 10,
        "date_format": "%d/%m/%Y %H:%M:%S",
        "number_format": "{:,.2f}",
        "currency": "EUR",
        "currency_symbol": "€",
        "timezone": "Europe/Paris",
        "decimal_separator": ",",
        "thousands_separator": " "
    },
    "en": {
        "name": "English",
        "native_name": "English", 
        "iso_code": "en-US",
        "direction": "ltr",
        "fallback": None,  # Langue par défaut
        "enabled": True,
        "priority": 100,  # Priorité la plus haute
        "date_format": "%Y-%m-%d %H:%M:%S",
        "number_format": "{:,.2f}",
        "currency": "USD",
        "currency_symbol": "$",
        "timezone": "UTC",
        "decimal_separator": ".",
        "thousands_separator": ","
    },
    "de": {
        "name": "German",
        "native_name": "Deutsch",
        "iso_code": "de-DE", 
        "direction": "ltr",
        "fallback": "en",
        "enabled": True,
        "priority": 20,
        "date_format": "%d.%m.%Y %H:%M:%S",
        "number_format": "{:,.2f}",
        "currency": "EUR",
        "currency_symbol": "€",
        "timezone": "Europe/Berlin",
        "decimal_separator": ",",
        "thousands_separator": "."
    },
    "es": {
        "name": "Spanish",
        "native_name": "Español",
        "iso_code": "es-ES",
        "direction": "ltr", 
        "fallback": "en",
        "enabled": True,
        "priority": 15,
        "date_format": "%d/%m/%Y %H:%M:%S",
        "number_format": "{:,.2f}",
        "currency": "EUR",
        "currency_symbol": "€",
        "timezone": "Europe/Madrid",
        "decimal_separator": ",",
        "thousands_separator": "."
    },
    "it": {
        "name": "Italian",
        "native_name": "Italiano",
        "iso_code": "it-IT",
        "direction": "ltr",
        "fallback": "en",
        "enabled": True,
        "priority": 12,
        "date_format": "%d/%m/%Y %H:%M:%S", 
        "number_format": "{:,.2f}",
        "currency": "EUR",
        "currency_symbol": "€",
        "timezone": "Europe/Rome",
        "decimal_separator": ",",
        "thousands_separator": "."
    }
}

# Locale par défaut
DEFAULT_LOCALE = "en"

# Niveaux de sévérité avec métadonnées
ALERT_SEVERITY_LEVELS = {
    "info": {
        "priority": 0,
        "color": "#36a64f",
        "emoji": "ℹ️",
        "slack_color": "good",
        "pagerduty_severity": "info",
        "escalation_delay": 0,  # Pas d'escalation
        "auto_resolve": True,
        "auto_resolve_delay": 3600  # 1 heure
    },
    "warning": {
        "priority": 1,
        "color": "#ff9500",
        "emoji": "⚠️",
        "slack_color": "warning", 
        "pagerduty_severity": "warning",
        "escalation_delay": 900,  # 15 minutes
        "auto_resolve": True,
        "auto_resolve_delay": 1800  # 30 minutes
    },
    "critical": {
        "priority": 2,
        "color": "#ff0000",
        "emoji": "🚨",
        "slack_color": "danger",
        "pagerduty_severity": "error",
        "escalation_delay": 300,  # 5 minutes
        "auto_resolve": False,
        "requires_ack": True
    },
    "emergency": {
        "priority": 3,
        "color": "#8B0000",
        "emoji": "🔥",
        "slack_color": "danger",
        "pagerduty_severity": "critical",
        "escalation_delay": 60,  # 1 minute
        "auto_resolve": False,
        "requires_ack": True,
        "immediate_notification": True
    }
}

# Types d'alertes avec configuration
ALERT_TYPES = {
    # IA & Machine Learning
    "ai_model_performance": {
        "category": "ml",
        "description": "Performance degradation of AI models",
        "default_severity": "warning",
        "thresholds": {
            "warning": {"accuracy_drop": 5.0, "latency_increase": 50.0},
            "critical": {"accuracy_drop": 10.0, "latency_increase": 100.0},
            "emergency": {"accuracy_drop": 20.0, "latency_increase": 200.0}
        },
        "metrics": ["accuracy", "latency", "throughput", "error_rate"],
        "escalation_policy": "ml_team",
        "runbook_url": "https://docs.spotify-ai-agent.com/runbooks/ai-model-performance"
    },
    "ai_model_training": {
        "category": "ml",
        "description": "Issues with model training processes",
        "default_severity": "info",
        "thresholds": {
            "warning": {"training_time_increase": 25.0},
            "critical": {"training_failure_rate": 5.0}
        },
        "metrics": ["training_time", "loss", "validation_accuracy"],
        "escalation_policy": "ml_team"
    },
    "ai_inference_latency": {
        "category": "ml", 
        "description": "High latency in AI inference",
        "default_severity": "warning",
        "thresholds": {
            "warning": {"p95_latency_ms": 500},
            "critical": {"p95_latency_ms": 1000},
            "emergency": {"p95_latency_ms": 2000}
        },
        "metrics": ["p50_latency", "p95_latency", "p99_latency"],
        "escalation_policy": "platform_team"
    },
    
    # API & Services
    "api_latency": {
        "category": "api",
        "description": "High API response latency",
        "default_severity": "warning",
        "thresholds": {
            "warning": {"p95_latency_ms": 200},
            "critical": {"p95_latency_ms": 500},
            "emergency": {"p95_latency_ms": 1000}
        },
        "metrics": ["response_time", "throughput", "error_rate"],
        "escalation_policy": "api_team"
    },
    "api_error_rate": {
        "category": "api",
        "description": "High API error rate",
        "default_severity": "critical",
        "thresholds": {
            "warning": {"error_rate_percent": 1.0},
            "critical": {"error_rate_percent": 5.0},
            "emergency": {"error_rate_percent": 10.0}
        },
        "metrics": ["error_rate", "5xx_errors", "4xx_errors"],
        "escalation_policy": "api_team"
    },
    
    # Infrastructure  
    "resource_usage": {
        "category": "infrastructure",
        "description": "High resource utilization",
        "default_severity": "warning",
        "thresholds": {
            "warning": {"cpu_percent": 70, "memory_percent": 80},
            "critical": {"cpu_percent": 85, "memory_percent": 90},
            "emergency": {"cpu_percent": 95, "memory_percent": 95}
        },
        "metrics": ["cpu_usage", "memory_usage", "disk_usage"],
        "escalation_policy": "sre_team"
    },
    
    # Sécurité
    "security_breach": {
        "category": "security",
        "description": "Security incident detected",
        "default_severity": "emergency",
        "thresholds": {
            "critical": {"suspicious_requests": 10},
            "emergency": {"data_access_anomaly": 1}
        },
        "metrics": ["failed_logins", "suspicious_ips", "data_access"],
        "escalation_policy": "security_team",
        "immediate_notification": True
    },
    
    # Business Logic
    "music_generation_failed": {
        "category": "business",
        "description": "Music generation failures",
        "default_severity": "warning",
        "thresholds": {
            "warning": {"failure_rate_percent": 2.0},
            "critical": {"failure_rate_percent": 5.0},
            "emergency": {"failure_rate_percent": 10.0}
        },
        "metrics": ["generation_success_rate", "generation_time"],
        "escalation_policy": "product_team"
    },
    "recommendation_degraded": {
        "category": "business",
        "description": "Music recommendation quality degraded",
        "default_severity": "warning", 
        "thresholds": {
            "warning": {"click_through_rate_drop": 10.0},
            "critical": {"click_through_rate_drop": 25.0}
        },
        "metrics": ["ctr", "user_satisfaction", "recommendation_diversity"],
        "escalation_policy": "product_team"
    }
}

# Types de tenants
TENANT_TYPES = {
    "artist": {
        "description": "Individual artists",
        "icon": "🎤",
        "default_channels": ["slack", "email"],
        "escalation_policy": "artist_support",
        "billing_tier": "premium"
    },
    "label": {
        "description": "Record labels",
        "icon": "🏢", 
        "default_channels": ["slack", "email", "webhook"],
        "escalation_policy": "enterprise_support",
        "billing_tier": "enterprise"
    },
    "studio": {
        "description": "Recording studios",
        "icon": "🎚️",
        "default_channels": ["slack", "email"],
        "escalation_policy": "studio_support", 
        "billing_tier": "professional"
    },
    "platform": {
        "description": "Platform integrations",
        "icon": "🌐",
        "default_channels": ["webhook", "api"],
        "escalation_policy": "platform_support",
        "billing_tier": "enterprise"
    }
}

# Templates d'alertes par défaut
DEFAULT_TEMPLATES = {
    "ai_model_performance": {
        "title_template": "{{ severity_emoji }} Dégradation Performance IA - {{ model_name }}",
        "message_template": """
Le modèle {{ model_name }} présente une dégradation de performance:
• Précision actuelle: {{ accuracy_score }}%
• Baisse de {{ change_percent }}% par rapport à la baseline
• {{ affected_users }} utilisateurs impactés
• Environnement: {{ environment }}
        """.strip(),
        "slack_template": {
            "pretext": "Alerte Performance IA",
            "fields": [
                {"title": "Modèle", "value": "{{ model_name }}", "short": True},
                {"title": "Précision", "value": "{{ accuracy_score }}%", "short": True},
                {"title": "Utilisateurs", "value": "{{ affected_users }}", "short": True},
                {"title": "Environnement", "value": "{{ environment }}", "short": True}
            ]
        }
    },
    "api_latency": {
        "title_template": "{{ severity_emoji }} Latence API Élevée - {{ service_name }}",
        "message_template": """
Latence API élevée détectée sur {{ service_name }}:
• Latence P95: {{ latency_ms }}ms
• Seuil configuré: {{ threshold }}ms
• Augmentation de {{ change_percent }}%
        """.strip()
    },
    "security_breach": {
        "title_template": "🔥 INCIDENT SÉCURITÉ - {{ tenant_name }}",
        "message_template": """
ALERTE SÉCURITÉ CRITIQUE:
• Tenant: {{ tenant_name }}
• Type: {{ alert_type }}
• Détails: {{ security_details }}
• Action immédiate requise!
        """.strip(),
        "requires_immediate_action": True
    }
}

# Configuration des canaux Slack
SLACK_CHANNEL_MAPPING = {
    "critical_alerts": {
        "webhook_url": os.getenv("SLACK_WEBHOOK_CRITICAL"),
        "channel": "#alerts-critical",
        "severities": ["critical", "emergency"],
        "mention_users": ["@here"],
        "thread_alerts": False,
        "rate_limit": 5  # Max 5 messages par minute
    },
    "warning_alerts": {
        "webhook_url": os.getenv("SLACK_WEBHOOK_WARNING"),
        "channel": "#alerts-warning", 
        "severities": ["warning"],
        "mention_users": [],
        "thread_alerts": True,
        "rate_limit": 10
    },
    "info_alerts": {
        "webhook_url": os.getenv("SLACK_WEBHOOK_INFO"),
        "channel": "#alerts-info",
        "severities": ["info"],
        "mention_users": [],
        "thread_alerts": True,
        "rate_limit": 20
    },
    "ml_alerts": {
        "webhook_url": os.getenv("SLACK_WEBHOOK_ML"),
        "channel": "#ml-alerts",
        "categories": ["ml"],
        "mention_users": ["@ml-team"],
        "thread_alerts": True
    },
    "security_alerts": {
        "webhook_url": os.getenv("SLACK_WEBHOOK_SECURITY"),
        "channel": "#security-alerts",
        "categories": ["security"],
        "mention_users": ["@security-team", "@oncall"],
        "thread_alerts": False,
        "immediate_delivery": True
    }
}

# Règles de formatage Slack
SLACK_FORMATTING_RULES = {
    "mention_patterns": {
        "critical": "@here",
        "emergency": "@channel", 
        "ml_team": "@ml-team",
        "security_team": "@security-team"
    },
    "color_mapping": {
        "info": "good",
        "warning": "warning",
        "critical": "danger", 
        "emergency": "danger"
    },
    "emoji_mapping": {
        "info": "ℹ️",
        "warning": "⚠️",
        "critical": "🚨",
        "emergency": "🔥"
    },
    "field_limits": {
        "max_fields": 10,
        "max_field_length": 100,
        "max_title_length": 50
    },
    "message_limits": {
        "max_text_length": 3000,
        "max_attachments": 20,
        "max_blocks": 50
    }
}

# Configuration environnements
ENVIRONMENT_CONFIGS = {
    "development": {
        "alert_enabled": True,
        "rate_limits": {
            "alerts_per_minute": 100,
            "alerts_per_hour": 1000
        },
        "default_channels": ["console", "slack"],
        "auto_resolve_enabled": True,
        "escalation_enabled": False,
        "notification_delay": 0
    },
    "staging": {
        "alert_enabled": True,
        "rate_limits": {
            "alerts_per_minute": 50,
            "alerts_per_hour": 500
        },
        "default_channels": ["slack", "email"],
        "auto_resolve_enabled": True,
        "escalation_enabled": True,
        "notification_delay": 60  # 1 minute
    },
    "production": {
        "alert_enabled": True,
        "rate_limits": {
            "alerts_per_minute": 20,
            "alerts_per_hour": 200
        },
        "default_channels": ["slack", "email", "pagerduty"],
        "auto_resolve_enabled": False,
        "escalation_enabled": True,
        "notification_delay": 0,
        "require_ack": True
    }
}

# Intégrations tierces
INTEGRATIONS = {
    "prometheus": {
        "enabled": True,
        "url": os.getenv("PROMETHEUS_URL", "http://prometheus:9090"),
        "query_timeout": 30,
        "metrics_prefix": "spotify_ai_"
    },
    "alertmanager": {
        "enabled": True,
        "url": os.getenv("ALERTMANAGER_URL", "http://alertmanager:9093"),
        "webhook_path": "/api/v1/alerts",
        "auth_token": os.getenv("ALERTMANAGER_TOKEN")
    },
    "grafana": {
        "enabled": True,
        "url": os.getenv("GRAFANA_URL", "http://grafana:3000"),
        "api_key": os.getenv("GRAFANA_API_KEY"),
        "dashboard_base_url": "/d/"
    },
    "pagerduty": {
        "enabled": os.getenv("PAGERDUTY_ENABLED", "false").lower() == "true",
        "integration_key": os.getenv("PAGERDUTY_INTEGRATION_KEY"),
        "service_key": os.getenv("PAGERDUTY_SERVICE_KEY"),
        "api_url": "https://events.pagerduty.com/v2/enqueue"
    },
    "datadog": {
        "enabled": os.getenv("DATADOG_ENABLED", "false").lower() == "true",
        "api_key": os.getenv("DATADOG_API_KEY"),
        "app_key": os.getenv("DATADOG_APP_KEY"),
        "site": os.getenv("DATADOG_SITE", "datadoghq.com")
    }
}

# Configuration Redis pour cache
REDIS_CONFIG = {
    "enabled": os.getenv("REDIS_ENABLED", "true").lower() == "true",
    "url": os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    "key_prefix": "spotify_ai_alerts:",
    "ttl_default": 3600,  # 1 heure
    "max_connections": 20,
    "socket_timeout": 5,
    "socket_connect_timeout": 5,
    "retry_on_timeout": True,
    "health_check_interval": 30
}

# Configuration logging
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "json",  # json ou text
    "output": ["console", "file"],
    "file_path": "/var/log/spotify-ai-agent/alerts.log",
    "rotation": {
        "max_size": "100MB",
        "backup_count": 10
    },
    "structured_logging": True,
    "include_trace_id": True
}

# Seuils par défaut
DEFAULT_THRESHOLDS = {
    "api_latency_ms": {
        "warning": 200,
        "critical": 500,
        "emergency": 1000
    },
    "error_rate_percent": {
        "warning": 1.0,
        "critical": 5.0,
        "emergency": 10.0
    },
    "cpu_usage_percent": {
        "warning": 70,
        "critical": 85,
        "emergency": 95
    },
    "memory_usage_percent": {
        "warning": 80,
        "critical": 90,
        "emergency": 95
    },
    "disk_usage_percent": {
        "warning": 80,
        "critical": 90,
        "emergency": 95
    },
    "ai_accuracy_drop_percent": {
        "warning": 5.0,
        "critical": 10.0,
        "emergency": 20.0
    }
}

# Fonctions utilitaires

def get_environment() -> str:
    """Retourne l'environnement actuel."""
    return os.getenv("ENVIRONMENT", "development")

def get_alert_config(environment: str = None, tenant_type: str = None) -> Dict[str, Any]:
    """Récupère la configuration d'alerte pour un environnement et type de tenant."""
    env = environment or get_environment()
    
    # Configuration de base
    config = ENVIRONMENT_CONFIGS.get(env, ENVIRONMENT_CONFIGS["development"]).copy()
    
    # Override par type de tenant
    if tenant_type and tenant_type in TENANT_TYPES:
        tenant_config = TENANT_TYPES[tenant_type]
        if "default_channels" in tenant_config:
            config["default_channels"] = tenant_config["default_channels"]
        if "escalation_policy" in tenant_config:
            config["escalation_policy"] = tenant_config["escalation_policy"]
    
    return config

def get_slack_webhook(channel_name: str) -> Optional[str]:
    """Récupère l'URL du webhook Slack pour un canal."""
    channel_config = SLACK_CHANNEL_MAPPING.get(channel_name)
    return channel_config.get("webhook_url") if channel_config else None

def get_severity_config(severity: str) -> Dict[str, Any]:
    """Récupère la configuration pour un niveau de sévérité."""
    return ALERT_SEVERITY_LEVELS.get(severity, ALERT_SEVERITY_LEVELS["info"])

def get_alert_type_config(alert_type: str) -> Dict[str, Any]:
    """Récupère la configuration pour un type d'alerte."""
    return ALERT_TYPES.get(alert_type, {})

def validate_locale(locale: str) -> bool:
    """Valide qu'une locale est supportée."""
    return locale in SUPPORTED_LOCALES and SUPPORTED_LOCALES[locale].get("enabled", False)

def get_fallback_locale(locale: str) -> str:
    """Récupère la locale de fallback."""
    if locale in SUPPORTED_LOCALES:
        fallback = SUPPORTED_LOCALES[locale].get("fallback")
        if fallback and validate_locale(fallback):
            return fallback
    return DEFAULT_LOCALE

def load_custom_config(config_path: str) -> Dict[str, Any]:
    """Charge une configuration personnalisée depuis un fichier."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Fichier de configuration introuvable: {config_path}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            if config_file.suffix.lower() == '.yaml' or config_file.suffix.lower() == '.yml':
                return yaml.safe_load(f)
            elif config_file.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Format de fichier non supporté: {config_file.suffix}")
    except Exception as e:
        raise RuntimeError(f"Erreur lecture configuration: {e}")

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Fusionne deux configurations de manière récursive."""
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result

def get_runtime_config() -> Dict[str, Any]:
    """Récupère la configuration runtime complète."""
    base_config = {
        "locales": SUPPORTED_LOCALES,
        "default_locale": DEFAULT_LOCALE,
        "severity_levels": ALERT_SEVERITY_LEVELS,
        "alert_types": ALERT_TYPES,
        "tenant_types": TENANT_TYPES,
        "slack_channels": SLACK_CHANNEL_MAPPING,
        "environment": get_environment(),
        "integrations": INTEGRATIONS,
        "redis": REDIS_CONFIG,
        "logging": LOGGING_CONFIG
    }
    
    # Merge avec configuration d'environnement
    env_config = get_alert_config()
    base_config["environment_config"] = env_config
    
    return base_config

# Validation de la configuration au démarrage
def validate_config():
    """Valide la configuration au démarrage."""
    errors = []
    
    # Validation des locales
    if not SUPPORTED_LOCALES:
        errors.append("Aucune locale configurée")
    
    if DEFAULT_LOCALE not in SUPPORTED_LOCALES:
        errors.append(f"Locale par défaut '{DEFAULT_LOCALE}' non trouvée dans les locales supportées")
    
    # Validation des webhooks Slack critiques
    critical_webhook = get_slack_webhook("critical_alerts")
    if not critical_webhook:
        errors.append("Webhook Slack critique non configuré")
    
    # Validation des intégrations
    if INTEGRATIONS["prometheus"]["enabled"] and not INTEGRATIONS["prometheus"]["url"]:
        errors.append("URL Prometheus manquante")
    
    if errors:
        raise RuntimeError(f"Erreurs de configuration: {'; '.join(errors)}")

# Auto-validation au import
try:
    validate_config()
except RuntimeError as e:
    print(f"ATTENTION: {e}")  # Log warning mais n'empêche pas l'import
