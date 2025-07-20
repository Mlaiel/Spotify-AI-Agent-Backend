"""
Spotify AI Agent - Module de Configuration Slack pour AlertManager
==================================================================

Module ultra-avancé pour la configuration et gestion des notifications Slack
dans le système de monitoring AlertManager du Spotify AI Agent.

Ce module fournit:
- Configuration dynamique des webhooks Slack
- Templates de messages personnalisés
- Gestion des canaux par criticité
- Formatage intelligent des alertes
- Routage automatique par tenant
- Escalade automatique des alertes critiques

Architecture:
    config/
    ├── __init__.py              # Point d'entrée principal
    ├── slack_config.py          # Configuration Slack centralisée
    ├── webhook_manager.py       # Gestionnaire webhooks
    ├── template_manager.py      # Gestionnaire templates
    ├── channel_router.py        # Routeur de canaux
    ├── formatter.py             # Formatage des messages
    ├── escalation.py            # Système d'escalade
    ├── validation.py            # Validation configuration
    └── utils.py                 # Utilitaires communs

Utilisation:
    from .slack_config import SlackNotificationConfig
    from .webhook_manager import SlackWebhookManager
    from .template_manager import SlackTemplateManager
    
    # Initialisation
    config = SlackNotificationConfig()
    webhook_manager = SlackWebhookManager(config)
    template_manager = SlackTemplateManager(config)

Classes principales:
    - SlackNotificationConfig: Configuration centralisée
    - SlackWebhookManager: Gestion des webhooks
    - SlackTemplateManager: Gestion des templates
    - SlackChannelRouter: Routage intelligent
    - SlackMessageFormatter: Formatage avancé
    - SlackEscalationManager: Gestion escalades

Sécurité:
    - Chiffrement des tokens Slack
    - Validation stricte des webhooks
    - Rate limiting intégré
    - Audit trail complet
    - Gestion des erreurs robuste

Performance:
    - Cache Redis pour templates
    - Pool de connexions HTTP
    - Retry automatique
    - Métriques détaillées
    - Monitoring temps réel
"""

from typing import Dict, List, Optional, Any, Union
import logging
from dataclasses import dataclass, field
from enum import Enum, auto

# Version du module
__version__ = "2.1.0"
__author__ = "Spotify AI Agent Team"
__maintainer__ = "Backend Team"

# Configuration logging
logger = logging.getLogger(__name__)

class SlackSeverity(Enum):
    """Niveaux de sévérité pour les notifications Slack."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class SlackChannelType(Enum):
    """Types de canaux Slack supportés."""
    ALERTS = "alerts"
    MONITORING = "monitoring"
    INCIDENTS = "incidents"
    ESCALATION = "escalation"
    AUDIT = "audit"
    SYSTEM = "system"

class SlackNotificationStatus(Enum):
    """États des notifications Slack."""
    PENDING = auto()
    SENT = auto()
    FAILED = auto()
    RETRY = auto()
    ESCALATED = auto()

@dataclass
class SlackConfig:
    """Configuration de base pour Slack."""
    enabled: bool = True
    default_channel: str = "#alerts"
    bot_token: Optional[str] = None
    signing_secret: Optional[str] = None
    app_token: Optional[str] = None
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: int = 5
    rate_limit: int = 100
    
    # Channels par sévérité
    severity_channels: Dict[str, str] = field(default_factory=lambda: {
        "critical": "#critical-alerts",
        "high": "#high-alerts", 
        "medium": "#medium-alerts",
        "low": "#low-alerts",
        "info": "#info-alerts"
    })
    
    # Configuration escalade
    escalation_enabled: bool = True
    escalation_timeout: int = 300  # 5 minutes
    escalation_channels: List[str] = field(default_factory=lambda: [
        "#escalation",
        "#management"
    ])

# Export des classes principales
from .slack_config import SlackNotificationConfig
from .webhook_manager import SlackWebhookManager
from .template_manager import SlackTemplateManager
from .channel_router import SlackChannelRouter
from .formatter import SlackMessageFormatter
from .escalation import SlackEscalationManager
from .validation import SlackConfigValidator
from .utils import SlackUtils

__all__ = [
    # Enums
    "SlackSeverity",
    "SlackChannelType", 
    "SlackNotificationStatus",
    
    # Dataclasses
    "SlackConfig",
    
    # Classes principales
    "SlackNotificationConfig",
    "SlackWebhookManager",
    "SlackTemplateManager",
    "SlackChannelRouter",
    "SlackMessageFormatter",
    "SlackEscalationManager",
    "SlackConfigValidator",
    "SlackUtils",
    
    # Constantes
    "__version__",
    "__author__",
    "__maintainer__"
]

# Configuration par défaut
DEFAULT_SLACK_CONFIG = SlackConfig()

# Messages de log
logger.info(f"Module Slack Configuration v{__version__} initialisé")
logger.debug("Configuration Slack par défaut chargée")
