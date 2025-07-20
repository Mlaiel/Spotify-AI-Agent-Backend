#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spotify AI Agent - Multi-Tenant Warning Alert Locales Module

Ce module gère la localisation des alertes d'avertissement pour le système multi-tenant
du Spotify AI Agent, intégré avec Alertmanager et Slack.

Fonctionnalités:
- Localisation multi-langues (fr, en, de, es, it)
- Templates d'alertes contextuelles pour Spotify AI
- Intégration avec Prometheus/Alertmanager
- Support multi-tenant avec isolation complète
- Messages d'alerte enrichis pour l'IA musicale
- Formatage Slack adaptatif selon la criticité

Architecture:
- Patterns Observer pour notifications temps réel
- Factory Pattern pour génération de messages
- Strategy Pattern pour formatage multi-canaux
- Template Method pour pipelines d'alerte

Modules principaux:
- LocaleManager: Gestion centralisée des langues
- AlertFormatter: Formatage contextualisé des alertes
- SlackTemplateEngine: Rendu Slack avancé
- TenantContextProvider: Isolation par tenant
- MetricsCollector: Métriques d'alerting

Utilisation:
    from locales import LocaleManager, AlertFormatter
    
    manager = LocaleManager()
    formatter = AlertFormatter(manager)
    
    alert = formatter.format_warning(
        tenant_id="spotify_artist_123",
        alert_type="ai_model_performance",
        severity="medium",
        locale="fr"
    )

Auteur: Équipe DevOps/MLOps - Fahed Mlaiel (Lead)
Licence: Propriétaire Spotify AI Agent
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "Spotify AI Agent DevOps Team"
__email__ = "devops@spotify-ai-agent.com"
__status__ = "Production"

# Imports optimisés
from .locale_manager import LocaleManager
from .alert_formatter import AlertFormatter
from .slack_template_engine import SlackTemplateEngine
from .tenant_context_provider import TenantContextProvider
from .metrics_collector import MetricsCollector
from .config import (
    SUPPORTED_LOCALES,
    DEFAULT_LOCALE,
    ALERT_TEMPLATES,
    SLACK_CHANNEL_MAPPING
)

# Public API
__all__ = [
    'LocaleManager',
    'AlertFormatter', 
    'SlackTemplateEngine',
    'TenantContextProvider',
    'MetricsCollector',
    'SUPPORTED_LOCALES',
    'DEFAULT_LOCALE',
    'get_localized_alert',
    'format_slack_message',
    'validate_tenant_context'
]

# Factory functions pour utilisation simplifiée
def get_localized_alert(tenant_id: str, alert_data: dict, locale: str = None) -> dict:
    """Factory function pour générer une alerte localisée complète."""
    manager = LocaleManager()
    formatter = AlertFormatter(manager)
    return formatter.format_complete_alert(tenant_id, alert_data, locale)

def format_slack_message(alert_data: dict, channel_config: dict) -> dict:
    """Factory function pour formatage Slack optimisé."""
    engine = SlackTemplateEngine()
    return engine.render_alert_message(alert_data, channel_config)

def validate_tenant_context(tenant_id: str, user_context: dict) -> bool:
    """Validation du contexte tenant pour sécurité."""
    provider = TenantContextProvider()
    return provider.validate_context(tenant_id, user_context)

# Configuration logging module
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Handler pour logs structurés (JSON)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
        '"module": "%(name)s", "message": "%(message)s", '
        '"tenant_id": "%(tenant_id)s"}'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Métriques d'initialisation
from prometheus_client import Counter, Histogram, Gauge

# Compteurs Prometheus
alert_generation_total = Counter(
    'spotify_ai_alert_generation_total',
    'Nombre total d\'alertes générées',
    ['tenant_id', 'alert_type', 'locale', 'severity']
)

alert_processing_duration = Histogram(
    'spotify_ai_alert_processing_seconds',
    'Durée de traitement des alertes',
    ['tenant_id', 'alert_type']
)

active_tenants = Gauge(
    'spotify_ai_active_tenants',
    'Nombre de tenants actifs'
)

logger.info("Module de localisation d'alertes initialisé avec succès")
