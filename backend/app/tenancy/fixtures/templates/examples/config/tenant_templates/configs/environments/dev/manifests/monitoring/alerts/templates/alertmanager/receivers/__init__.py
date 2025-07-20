"""
Alertmanager Receivers Module for Spotify AI Agent

Module industrialisé pour la gestion des receivers d'alertes multi-tenant.
Architecture avancée pour la notification et l'escalade d'alertes.

Fonctionnalités:
- Gestion multi-tenant des receivers
- Templates dynamiques d'alertes
- Escalade automatique intelligente
- Intégration multi-canaux (Slack, Email, PagerDuty, Teams, Discord)
- Métriques et monitoring des notifications
- Rate limiting et circuit breaker
- Retry logic avancé avec backoff exponentiel
"""

from .core import AlertReceiverManager, ReceiverTemplate
from .models import (
    ReceiverConfig,
    NotificationChannel,
    EscalationPolicy,
    AlertContext
)
from .factories import (
    SlackReceiverFactory,
    EmailReceiverFactory,
    PagerDutyReceiverFactory,
    WebhookReceiverFactory,
    TeamsReceiverFactory,
    DiscordReceiverFactory
)
from .utils import (
    ReceiverValidator,
    TemplateRenderer,
    NotificationThrottler,
    MetricsCollector
)
from .exceptions import (
    ReceiverConfigError,
    NotificationError,
    TemplateRenderError,
    EscalationError
)

__all__ = [
    # Core components
    'AlertReceiverManager',
    'ReceiverTemplate',
    
    # Models
    'ReceiverConfig',
    'NotificationChannel',
    'EscalationPolicy',
    'AlertContext',
    
    # Factories
    'SlackReceiverFactory',
    'EmailReceiverFactory',
    'PagerDutyReceiverFactory',
    'WebhookReceiverFactory',
    'TeamsReceiverFactory',
    'DiscordReceiverFactory',
    
    # Utilities
    'ReceiverValidator',
    'TemplateRenderer',
    'NotificationThrottler',
    'MetricsCollector',
    
    # Exceptions
    'ReceiverConfigError',
    'NotificationError',
    'TemplateRenderError',
    'EscalationError'
]

# Version et métadonnées
__version__ = "2.1.0"
__author__ = "Spotify AI Agent Team"
__maintainer__ = "DevOps & SRE Team"
__status__ = "Production"

# Configuration par défaut
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_TIMEOUT = 30
DEFAULT_RATE_LIMIT = 100  # notifications par minute
DEFAULT_ESCALATION_DELAY = 300  # 5 minutes

# Types de receivers supportés
SUPPORTED_RECEIVER_TYPES = {
    'slack',
    'email',
    'pagerduty',
    'webhook',
    'teams',
    'discord',
    'opsgenie',
    'victorops',
    'pushover',
    'telegram'
}

# Niveaux de criticité
CRITICALITY_LEVELS = {
    'critical': 1,
    'high': 2,
    'medium': 3,
    'low': 4,
    'info': 5
}
