"""
Configuration et Templates par Défaut pour Alertes Slack
========================================================

Ce module contient les configurations par défaut et les templates de base
pour chaque type d'alerte et locale supportée dans le système multi-tenant.

Structure:
- Templates par locale (fr, en, de, es, it)
- Configurations par type d'alerte
- Fallbacks et templates de base
- Validation des configurations

Auteur: Fahed Mlaiel - Lead Developer Achiri
Version: 2.5.0
"""

import json
from typing import Dict, Any, List
from datetime import datetime
from enum import Enum

# Configuration globale des templates par défaut
DEFAULT_TEMPLATES = {
    "fr": {
        "system_critical": {
            "channel": "#alerts-critical",
            "username": "Achiri Alert Bot",
            "icon_emoji": ":rotating_light:",
            "attachments": [{
                "color": "danger",
                "title": "🚨 Alerte Système Critique",
                "text": "{{ alert.summary }}",
                "fields": [
                    {
                        "title": "Sévérité",
                        "value": "{{ alert.severity | upper }}",
                        "short": True
                    },
                    {
                        "title": "Tenant",
                        "value": "{{ tenant.name }}",
                        "short": True
                    },
                    {
                        "title": "Environnement",
                        "value": "{{ environment | upper }}",
                        "short": True
                    },
                    {
                        "title": "Timestamp",
                        "value": "{{ timestamp }}",
                        "short": True
                    },
                    {
                        "title": "Description",
                        "value": "{{ alert.description }}",
                        "short": False
                    }
                ],
                "footer": "Système de Monitoring Achiri",
                "footer_icon": "https://achiri.com/favicon.ico",
                "ts": "{{ alert.timestamp | int }}"
            }]
        },
        "system_warning": {
            "channel": "#alerts-warning",
            "username": "Achiri Alert Bot",
            "icon_emoji": ":warning:",
            "attachments": [{
                "color": "warning",
                "title": "⚠️ Alerte Système - Attention",
                "text": "{{ alert.summary }}",
                "fields": [
                    {
                        "title": "Sévérité",
                        "value": "{{ alert.severity | upper }}",
                        "short": True
                    },
                    {
                        "title": "Tenant",
                        "value": "{{ tenant.name }}",
                        "short": True
                    },
                    {
                        "title": "Métrique",
                        "value": "{{ alert.metric_name }}",
                        "short": True
                    },
                    {
                        "title": "Valeur",
                        "value": "{{ alert.metric_value }}",
                        "short": True
                    },
                    {
                        "title": "Seuil",
                        "value": "{{ alert.threshold }}",
                        "short": True
                    },
                    {
                        "title": "Recommandation",
                        "value": "{{ alert.recommendation | default('Vérifier les métriques système') }}",
                        "short": False
                    }
                ],
                "footer": "Monitoring Achiri - {{ environment }}",
                "ts": "{{ alert.timestamp | int }}"
            }]
        },
        "application_error": {
            "channel": "#alerts-app",
            "username": "Achiri Alert Bot",
            "icon_emoji": ":exclamation:",
            "attachments": [{
                "color": "#ff6b6b",
                "title": "🐛 Erreur Application",
                "text": "{{ alert.summary }}",
                "fields": [
                    {
                        "title": "Application",
                        "value": "{{ alert.application }}",
                        "short": True
                    },
                    {
                        "title": "Service",
                        "value": "{{ alert.service }}",
                        "short": True
                    },
                    {
                        "title": "Environnement",
                        "value": "{{ environment }}",
                        "short": True
                    },
                    {
                        "title": "Erreur",
                        "value": "```{{ alert.error_message }}```",
                        "short": False
                    },
                    {
                        "title": "Stack Trace",
                        "value": "```{{ alert.stack_trace | truncate(500) }}```",
                        "short": False
                    }
                ],
                "actions": [
                    {
                        "type": "button",
                        "text": "Voir les logs",
                        "url": "{{ alert.logs_url }}"
                    },
                    {
                        "type": "button",
                        "text": "Dashboard",
                        "url": "{{ alert.dashboard_url }}"
                    }
                ],
                "footer": "Achiri Error Tracking",
                "ts": "{{ alert.timestamp | int }}"
            }]
        },
        "security_alert": {
            "channel": "#security-alerts",
            "username": "Achiri Security Bot",
            "icon_emoji": ":shield:",
            "attachments": [{
                "color": "#8b0000",
                "title": "🔒 Alerte de Sécurité",
                "text": "{{ alert.summary }}",
                "fields": [
                    {
                        "title": "Type de Menace",
                        "value": "{{ alert.threat_type }}",
                        "short": True
                    },
                    {
                        "title": "Niveau de Risque",
                        "value": "{{ alert.risk_level }}",
                        "short": True
                    },
                    {
                        "title": "Source IP",
                        "value": "{{ alert.source_ip }}",
                        "short": True
                    },
                    {
                        "title": "Géolocalisation",
                        "value": "{{ alert.geolocation }}",
                        "short": True
                    },
                    {
                        "title": "Action Entreprise",
                        "value": "{{ alert.action_taken }}",
                        "short": False
                    }
                ],
                "footer": "Achiri Security Center",
                "ts": "{{ alert.timestamp | int }}"
            }]
        },
        "performance_degradation": {
            "channel": "#performance",
            "username": "Achiri Performance Bot",
            "icon_emoji": ":chart_with_downwards_trend:",
            "attachments": [{
                "color": "#ffa500",
                "title": "📊 Dégradation des Performances",
                "text": "{{ alert.summary }}",
                "fields": [
                    {
                        "title": "Service",
                        "value": "{{ alert.service }}",
                        "short": True
                    },
                    {
                        "title": "Métrique",
                        "value": "{{ alert.metric_name }}",
                        "short": True
                    },
                    {
                        "title": "Valeur Actuelle",
                        "value": "{{ alert.current_value }}",
                        "short": True
                    },
                    {
                        "title": "Valeur Normale",
                        "value": "{{ alert.baseline_value }}",
                        "short": True
                    },
                    {
                        "title": "Dégradation",
                        "value": "{{ alert.degradation_percentage }}%",
                        "short": True
                    },
                    {
                        "title": "Impact Utilisateur",
                        "value": "{{ alert.user_impact }}",
                        "short": True
                    }
                ],
                "actions": [
                    {
                        "type": "button",
                        "text": "Voir Grafana",
                        "url": "{{ alert.grafana_url }}"
                    },
                    {
                        "type": "button",
                        "text": "Analyser",
                        "url": "{{ alert.analysis_url }}"
                    }
                ],
                "footer": "Achiri Performance Monitoring",
                "ts": "{{ alert.timestamp | int }}"
            }]
        },
        "deployment_notification": {
            "channel": "#deployments",
            "username": "Achiri Deploy Bot",
            "icon_emoji": ":rocket:",
            "attachments": [{
                "color": "good",
                "title": "🚀 Notification de Déploiement",
                "text": "{{ alert.summary }}",
                "fields": [
                    {
                        "title": "Application",
                        "value": "{{ alert.application }}",
                        "short": True
                    },
                    {
                        "title": "Version",
                        "value": "{{ alert.version }}",
                        "short": True
                    },
                    {
                        "title": "Environnement",
                        "value": "{{ environment }}",
                        "short": True
                    },
                    {
                        "title": "Statut",
                        "value": "{{ alert.status }}",
                        "short": True
                    },
                    {
                        "title": "Déployé par",
                        "value": "{{ alert.deployed_by }}",
                        "short": True
                    },
                    {
                        "title": "Durée",
                        "value": "{{ alert.duration }}",
                        "short": True
                    }
                ],
                "actions": [
                    {
                        "type": "button",
                        "text": "Logs de déploiement",
                        "url": "{{ alert.deployment_logs_url }}"
                    },
                    {
                        "type": "button",
                        "text": "Release Notes",
                        "url": "{{ alert.release_notes_url }}"
                    }
                ],
                "footer": "Achiri CI/CD Pipeline",
                "ts": "{{ alert.timestamp | int }}"
            }]
        }
    },
    "en": {
        "system_critical": {
            "channel": "#alerts-critical",
            "username": "Achiri Alert Bot",
            "icon_emoji": ":rotating_light:",
            "attachments": [{
                "color": "danger",
                "title": "🚨 Critical System Alert",
                "text": "{{ alert.summary }}",
                "fields": [
                    {
                        "title": "Severity",
                        "value": "{{ alert.severity | upper }}",
                        "short": True
                    },
                    {
                        "title": "Tenant",
                        "value": "{{ tenant.name }}",
                        "short": True
                    },
                    {
                        "title": "Environment",
                        "value": "{{ environment | upper }}",
                        "short": True
                    },
                    {
                        "title": "Timestamp",
                        "value": "{{ timestamp }}",
                        "short": True
                    },
                    {
                        "title": "Description",
                        "value": "{{ alert.description }}",
                        "short": False
                    }
                ],
                "footer": "Achiri Monitoring System",
                "footer_icon": "https://achiri.com/favicon.ico",
                "ts": "{{ alert.timestamp | int }}"
            }]
        },
        "system_warning": {
            "channel": "#alerts-warning",
            "username": "Achiri Alert Bot",
            "icon_emoji": ":warning:",
            "attachments": [{
                "color": "warning",
                "title": "⚠️ System Warning Alert",
                "text": "{{ alert.summary }}",
                "fields": [
                    {
                        "title": "Severity",
                        "value": "{{ alert.severity | upper }}",
                        "short": True
                    },
                    {
                        "title": "Tenant",
                        "value": "{{ tenant.name }}",
                        "short": True
                    },
                    {
                        "title": "Metric",
                        "value": "{{ alert.metric_name }}",
                        "short": True
                    },
                    {
                        "title": "Value",
                        "value": "{{ alert.metric_value }}",
                        "short": True
                    },
                    {
                        "title": "Threshold",
                        "value": "{{ alert.threshold }}",
                        "short": True
                    },
                    {
                        "title": "Recommendation",
                        "value": "{{ alert.recommendation | default('Check system metrics') }}",
                        "short": False
                    }
                ],
                "footer": "Achiri Monitoring - {{ environment }}",
                "ts": "{{ alert.timestamp | int }}"
            }]
        }
    },
    "de": {
        "system_critical": {
            "channel": "#alerts-critical",
            "username": "Achiri Alert Bot",
            "icon_emoji": ":rotating_light:",
            "attachments": [{
                "color": "danger",
                "title": "🚨 Kritischer System-Alert",
                "text": "{{ alert.summary }}",
                "fields": [
                    {
                        "title": "Schweregrad",
                        "value": "{{ alert.severity | upper }}",
                        "short": True
                    },
                    {
                        "title": "Tenant",
                        "value": "{{ tenant.name }}",
                        "short": True
                    },
                    {
                        "title": "Umgebung",
                        "value": "{{ environment | upper }}",
                        "short": True
                    },
                    {
                        "title": "Zeitstempel",
                        "value": "{{ timestamp }}",
                        "short": True
                    },
                    {
                        "title": "Beschreibung",
                        "value": "{{ alert.description }}",
                        "short": False
                    }
                ],
                "footer": "Achiri Monitoring System",
                "footer_icon": "https://achiri.com/favicon.ico",
                "ts": "{{ alert.timestamp | int }}"
            }]
        }
    },
    "es": {
        "system_critical": {
            "channel": "#alerts-critical",
            "username": "Achiri Alert Bot",
            "icon_emoji": ":rotating_light:",
            "attachments": [{
                "color": "danger",
                "title": "🚨 Alerta Crítica del Sistema",
                "text": "{{ alert.summary }}",
                "fields": [
                    {
                        "title": "Severidad",
                        "value": "{{ alert.severity | upper }}",
                        "short": True
                    },
                    {
                        "title": "Tenant",
                        "value": "{{ tenant.name }}",
                        "short": True
                    },
                    {
                        "title": "Entorno",
                        "value": "{{ environment | upper }}",
                        "short": True
                    },
                    {
                        "title": "Marca de tiempo",
                        "value": "{{ timestamp }}",
                        "short": True
                    },
                    {
                        "title": "Descripción",
                        "value": "{{ alert.description }}",
                        "short": False
                    }
                ],
                "footer": "Sistema de Monitoreo Achiri",
                "footer_icon": "https://achiri.com/favicon.ico",
                "ts": "{{ alert.timestamp | int }}"
            }]
        }
    },
    "it": {
        "system_critical": {
            "channel": "#alerts-critical",
            "username": "Achiri Alert Bot",
            "icon_emoji": ":rotating_light:",
            "attachments": [{
                "color": "danger",
                "title": "🚨 Allerta Sistema Critica",
                "text": "{{ alert.summary }}",
                "fields": [
                    {
                        "title": "Gravità",
                        "value": "{{ alert.severity | upper }}",
                        "short": True
                    },
                    {
                        "title": "Tenant",
                        "value": "{{ tenant.name }}",
                        "short": True
                    },
                    {
                        "title": "Ambiente",
                        "value": "{{ environment | upper }}",
                        "short": True
                    },
                    {
                        "title": "Timestamp",
                        "value": "{{ timestamp }}",
                        "short": True
                    },
                    {
                        "title": "Descrizione",
                        "value": "{{ alert.description }}",
                        "short": False
                    }
                ],
                "footer": "Sistema di Monitoraggio Achiri",
                "footer_icon": "https://achiri.com/favicon.ico",
                "ts": "{{ alert.timestamp | int }}"
            }]
        }
    }
}

# Configuration des canaux par défaut
DEFAULT_CHANNELS = {
    "critical": "#alerts-critical",
    "warning": "#alerts-warning",
    "info": "#alerts-info",
    "low": "#alerts-low",
    "security": "#security-alerts",
    "performance": "#performance",
    "deployment": "#deployments",
    "application": "#alerts-app"
}

# Configuration des couleurs par sévérité
SEVERITY_COLORS = {
    "critical": "danger",      # Rouge
    "warning": "warning",      # Orange/Jaune
    "info": "good",           # Vert
    "low": "#36a64f",         # Vert clair
    "security": "#8b0000",    # Rouge foncé
    "performance": "#ffa500",  # Orange
    "deployment": "good"       # Vert
}

# Configuration des emojis par type d'alerte
ALERT_EMOJIS = {
    "system_critical": ":rotating_light:",
    "system_warning": ":warning:",
    "application_error": ":exclamation:",
    "security_alert": ":shield:",
    "performance_degradation": ":chart_with_downwards_trend:",
    "deployment_notification": ":rocket:",
    "database_alert": ":file_cabinet:",
    "network_alert": ":globe_with_meridians:",
    "storage_alert": ":floppy_disk:",
    "backup_alert": ":package:"
}

# Métadonnées des templates
TEMPLATE_METADATA = {
    "version": "2.5.0",
    "last_updated": datetime.now().isoformat(),
    "supported_locales": ["fr", "en", "de", "es", "it"],
    "supported_alert_types": [
        "system_critical",
        "system_warning",
        "application_error",
        "security_alert",
        "performance_degradation",
        "deployment_notification",
        "database_alert",
        "network_alert",
        "storage_alert",
        "backup_alert"
    ],
    "author": "Fahed Mlaiel - Lead Developer Achiri",
    "description": "Templates par défaut pour alertes Slack multi-tenant"
}

# Configuration des retry et timeouts
RETRY_CONFIG = {
    "max_retries": 3,
    "initial_delay": 1.0,
    "max_delay": 60.0,
    "exponential_base": 2.0,
    "jitter": True
}

# Configuration du cache
CACHE_CONFIG = {
    "default_ttl": 3600,  # 1 heure
    "max_size": 1000,
    "cleanup_interval": 300  # 5 minutes
}

# Fonctions utilitaires pour les templates

def get_template_by_locale_and_type(locale: str, alert_type: str) -> Dict[str, Any]:
    """
    Récupère un template par locale et type d'alerte.
    
    Args:
        locale: Code de locale (fr, en, de, es, it)
        alert_type: Type d'alerte
        
    Returns:
        Template correspondant ou template par défaut
    """
    if locale in DEFAULT_TEMPLATES and alert_type in DEFAULT_TEMPLATES[locale]:
        return DEFAULT_TEMPLATES[locale][alert_type]
    
    # Fallback vers anglais
    if alert_type in DEFAULT_TEMPLATES.get("en", {}):
        return DEFAULT_TEMPLATES["en"][alert_type]
    
    # Template de base si rien trouvé
    return {
        "channel": "#alerts-general",
        "username": "Achiri Alert Bot",
        "icon_emoji": ":warning:",
        "text": "{{ alert.summary }}",
        "attachments": [{
            "color": "warning",
            "title": f"Alert: {alert_type}",
            "text": "{{ alert.description }}",
            "footer": "Achiri Monitoring System"
        }]
    }

def get_channel_for_alert_type(alert_type: str) -> str:
    """
    Détermine le canal approprié pour un type d'alerte.
    
    Args:
        alert_type: Type d'alerte
        
    Returns:
        Nom du canal Slack
    """
    if "critical" in alert_type:
        return DEFAULT_CHANNELS["critical"]
    elif "warning" in alert_type:
        return DEFAULT_CHANNELS["warning"]
    elif "security" in alert_type:
        return DEFAULT_CHANNELS["security"]
    elif "performance" in alert_type:
        return DEFAULT_CHANNELS["performance"]
    elif "deployment" in alert_type:
        return DEFAULT_CHANNELS["deployment"]
    elif "application" in alert_type or "error" in alert_type:
        return DEFAULT_CHANNELS["application"]
    else:
        return "#alerts-general"

def get_color_for_severity(severity: str) -> str:
    """
    Retourne la couleur appropriée pour une sévérité.
    
    Args:
        severity: Niveau de sévérité
        
    Returns:
        Code couleur Slack
    """
    return SEVERITY_COLORS.get(severity.lower(), "warning")

def get_emoji_for_alert_type(alert_type: str) -> str:
    """
    Retourne l'emoji approprié pour un type d'alerte.
    
    Args:
        alert_type: Type d'alerte
        
    Returns:
        Code emoji Slack
    """
    return ALERT_EMOJIS.get(alert_type, ":warning:")

def validate_template(template: Dict[str, Any]) -> List[str]:
    """
    Valide la structure d'un template Slack.
    
    Args:
        template: Template à valider
        
    Returns:
        Liste des erreurs de validation
    """
    errors = []
    
    # Vérifications de base
    if not isinstance(template, dict):
        errors.append("Le template doit être un dictionnaire")
        return errors
    
    # Channel obligatoire
    if "channel" not in template:
        errors.append("Le champ 'channel' est obligatoire")
    elif not template["channel"].startswith("#"):
        errors.append("Le canal doit commencer par '#'")
    
    # Validation des attachments
    if "attachments" in template:
        if not isinstance(template["attachments"], list):
            errors.append("Les attachments doivent être une liste")
        else:
            for i, attachment in enumerate(template["attachments"]):
                if not isinstance(attachment, dict):
                    errors.append(f"L'attachment {i} doit être un dictionnaire")
                elif "color" not in attachment:
                    errors.append(f"L'attachment {i} doit avoir une couleur")
    
    return errors

# Export des configurations principales
__all__ = [
    "DEFAULT_TEMPLATES",
    "DEFAULT_CHANNELS", 
    "SEVERITY_COLORS",
    "ALERT_EMOJIS",
    "TEMPLATE_METADATA",
    "RETRY_CONFIG",
    "CACHE_CONFIG",
    "get_template_by_locale_and_type",
    "get_channel_for_alert_type",
    "get_color_for_severity",
    "get_emoji_for_alert_type",
    "validate_template"
]
