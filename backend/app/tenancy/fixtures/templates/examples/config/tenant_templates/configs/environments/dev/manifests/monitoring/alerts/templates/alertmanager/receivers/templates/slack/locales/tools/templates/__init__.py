"""
Spotify AI Agent - Templates Slack pour Alertmanager
Module de templates avancés pour notifications Slack multi-tenant

Ce module fournit des templates industrialisés pour les notifications
Slack d'Alertmanager dans un environnement multi-tenant.
"""

__version__ = "1.0.0"
__author__ = "Spotify AI Agent Team"
__email__ = "dev@spotify-ai-agent.com"

from .template_engine import SlackTemplateEngine
from .template_manager import SlackTemplateManager
from .template_validator import SlackTemplateValidator
from .locale_manager import LocaleManager

__all__ = [
    "SlackTemplateEngine",
    "SlackTemplateManager", 
    "SlackTemplateValidator",
    "LocaleManager"
]
