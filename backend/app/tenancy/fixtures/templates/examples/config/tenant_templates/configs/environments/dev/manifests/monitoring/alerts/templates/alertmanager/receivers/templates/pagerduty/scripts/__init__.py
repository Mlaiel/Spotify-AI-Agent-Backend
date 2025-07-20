#!/usr/bin/env python3
"""
PagerDuty Scripts Module - Spotify AI Agent

Module principal pour les scripts d'intégration et de gestion PagerDuty.
Fournit des utilitaires complets pour le déploiement, la configuration,
et la maintenance des intégrations PagerDuty dans un environnement multi-tenant.

Architecture:
- Scripts de déploiement automatisés
- Gestionnaires de configuration
- Utilitaires de maintenance
- Scripts de monitoring et alerting
- Outils de backup et recovery

Compatibilité:
- Python 3.8+
- AsyncIO support
- Multi-tenant architecture
- Environment-aware configuration
- High availability ready

Version: 1.0.0
Dernière mise à jour: 2025-07-18
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

# Version information
__version__ = "1.0.0"
__author__ = "Spotify AI Agent Development Team"
__license__ = "MIT"
__maintainer__ = "Backend Development Team"
__status__ = "Production"

# Module constants
DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_TIMEOUT = 300

# Supported environments
SUPPORTED_ENVIRONMENTS = [
    "development",
    "staging", 
    "production",
    "testing"
]

# PagerDuty service levels
SERVICE_LEVELS = {
    "critical": {"priority": 1, "escalation_delay": 0},
    "high": {"priority": 2, "escalation_delay": 300},
    "medium": {"priority": 3, "escalation_delay": 600},
    "low": {"priority": 4, "escalation_delay": 1800}
}

def get_version() -> str:
    """Retourne la version du module."""
    return __version__

def get_supported_environments() -> List[str]:
    """Retourne la liste des environnements supportés."""
    return SUPPORTED_ENVIRONMENTS.copy()

def validate_environment(environment: str) -> bool:
    """Valide si l'environnement est supporté."""
    return environment in SUPPORTED_ENVIRONMENTS

def setup_logging(level: str = DEFAULT_LOG_LEVEL) -> logging.Logger:
    """Configure le logging pour le module."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('pagerduty_scripts.log')
        ]
    )
    return logging.getLogger(__name__)

# Initialize module logger
logger = setup_logging()
logger.info(f"PagerDuty Scripts Module v{__version__} initialized")

# Export public API
__all__ = [
    "__version__",
    "get_version", 
    "get_supported_environments",
    "validate_environment",
    "setup_logging",
    "SERVICE_LEVELS",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_TIMEOUT"
]
