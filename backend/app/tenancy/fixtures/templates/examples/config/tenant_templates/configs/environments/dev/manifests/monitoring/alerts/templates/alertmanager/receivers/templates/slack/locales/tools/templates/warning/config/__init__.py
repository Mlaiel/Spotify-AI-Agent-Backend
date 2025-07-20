"""
Module de Configuration d'Alertes Warning - Spotify AI Agent
===========================================================

Module ultra-avancé pour la gestion des configurations d'alertes de type Warning
avec support multi-tenant, escalade automatique et intégration Slack.

Architecte Principal: Fahed Mlaiel
Équipe de Développement:
- Lead Dev + Architecte IA
- Développeur Backend Senior (Python/FastAPI/Django)  
- Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Spécialiste Sécurité Backend
- Architecte Microservices

Version: 1.0.0
Licence: Propriétaire - Spotify AI Agent
Copyright: 2025 - Tous droits réservés
"""

__version__ = "1.0.0"
__author__ = "Fahed Mlaiel"
__email__ = "fahed.mlaiel@spotify-ai-agent.com"
__description__ = "Configuration avancée pour les alertes Warning du système Spotify AI Agent"

# Imports principaux
from .config_manager import WarningConfigManager
from .template_engine import AlertTemplateEngine
from .escalation_engine import EscalationEngine
from .notification_router import NotificationRouter
from .security_validator import SecurityValidator
from .performance_monitor import PerformanceMonitor

# Configuration par défaut
DEFAULT_CONFIG = {
    "alert_level": "WARNING",
    "escalation_enabled": True,
    "notification_channels": ["slack"],
    "rate_limit": 100,
    "cache_ttl": 3600
}

# Exports publics
__all__ = [
    "WarningConfigManager",
    "AlertTemplateEngine", 
    "EscalationEngine",
    "NotificationRouter",
    "SecurityValidator",
    "PerformanceMonitor",
    "DEFAULT_CONFIG"
]
