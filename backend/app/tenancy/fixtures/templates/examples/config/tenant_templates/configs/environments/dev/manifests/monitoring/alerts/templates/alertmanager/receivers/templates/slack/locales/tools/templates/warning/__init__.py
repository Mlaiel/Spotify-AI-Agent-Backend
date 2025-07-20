"""
Module de templates d'alertes et de monitoring pour Spotify AI Agent
Module d'alerting centralisé pour la gestion des warnings multi-tenant
"""

from .alert_manager import AlertManager
from .slack_notifier import SlackNotifier
from .warning_processor import WarningProcessor
from .locale_manager import LocaleManager
from .template_engine import TemplateEngine

__all__ = [
    'AlertManager',
    'SlackNotifier', 
    'WarningProcessor',
    'LocaleManager',
    'TemplateEngine'
]

__version__ = "1.0.0"
__author__ = "Spotify AI Agent Team"
