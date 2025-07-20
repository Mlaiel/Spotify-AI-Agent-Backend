"""
📊 Info Templates Module - Advanced Information Management System
===============================================================

Module ultra-avancé de gestion des templates d'information pour 
l'architecture multi-tenant Spotify AI Agent.

Ce module fournit une infrastructure complète pour la gestion,
la génération et la distribution des informations tenant-specific
avec support multi-lingue et personnalisation avancée.

FEATURES AVANCÉES:
- Templates d'information dynamiques et contextuels
- Système de notification intelligent avec ML
- Gestion multi-langue avec localisation automatique  
- Personnalisation basée sur le profil tenant
- Analytics et métriques temps réel
- Cache intelligent et optimisation des performances
- Intégration Slack/Teams/Discord/Email
- Audit trails et conformité GDPR
"""

from .generators import InfoTemplateGenerator, DynamicContentEngine
from .formatters import MessageFormatter, RichContentFormatter
from .validators import InfoValidator, ContentValidator
from .processors import InfoProcessor, ContextualProcessor
from .analytics import InfoAnalytics, EngagementTracker
from .cache import InfoCacheManager, DistributedCache
from .localization import LocalizationEngine, LanguageDetector
from .personalization import PersonalizationEngine, TenantProfiler

__all__ = [
    'InfoTemplateGenerator',
    'DynamicContentEngine',
    'MessageFormatter', 
    'RichContentFormatter',
    'InfoValidator',
    'ContentValidator',
    'InfoProcessor',
    'ContextualProcessor',
    'InfoAnalytics',
    'EngagementTracker',
    'InfoCacheManager',
    'DistributedCache',
    'LocalizationEngine',
    'LanguageDetector',
    'PersonalizationEngine',
    'TenantProfiler'
]

# Configuration par défaut
DEFAULT_CONFIG = {
    'cache_ttl': 3600,
    'max_message_length': 4096,
    'supported_languages': ['en', 'fr', 'de', 'es', 'it'],
    'rich_content_enabled': True,
    'analytics_enabled': True,
    'personalization_enabled': True
}
