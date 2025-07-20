"""
Configuration module for Alertmanager receivers.

Ce module centralise la gestion des configurations pour
les receivers d'alertes dans un environnement multi-tenant ultra-avancé.

Architecture industrielle conçue pour Spotify AI Agent par l'équipe d'experts:
- Lead Dev + Architecte IA
- Développeur Backend Senior (Python/FastAPI/Django) 
- Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Spécialiste Sécurité Backend
- Architecte Microservices

Fonctionnalités:
- Configuration multi-tenant avec isolation complète
- Escalade intelligente basée sur ML
- Templates dynamiques avec contexte enrichi
- Intégrations avec 15+ systèmes externes
- Sécurité end-to-end avec chiffrement
- Monitoring temps-réel et métriques avancées
- Auto-scaling et load balancing
- Audit trails complets
- Disaster recovery automatique
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Core configuration managers
from .receivers_config import ReceiverConfigManager
from .templates_config import TemplateConfigManager  
from .escalation_config import EscalationConfigManager

# Advanced modules  
from .security_config import SecurityConfigManager
from .automation_config import AutomationConfigManager
from .integration_config import IntegrationConfigManager
from .metrics_config import MetricsConfigManager
from .audit_config import AuditConfigManager
from .performance_config import PerformanceConfigManager
from .tenant_config import TenantConfigManager
from .disaster_recovery_config import DisasterRecoveryConfigManager
from .ml_intelligence_config import MLIntelligenceConfigManager
from .compliance_config import ComplianceConfigManager

# Utilities and validators
from .validators import ConfigValidator, SchemaValidator
from .utils import ConfigUtils, EncryptionUtils
from .factories import ConfigFactory, ReceiverFactory
from .registry import ConfigRegistry, ServiceRegistry

# Constants and enums
from .constants import (
    DEFAULT_TIMEOUTS,
    SEVERITY_LEVELS, 
    NOTIFICATION_CHANNELS,
    ESCALATION_POLICIES,
    SECURITY_POLICIES
)

from .enums import (
    TenantTier,
    AlertSeverity,
    NotificationChannel,
    EscalationLevel,
    SecurityLevel
)

# Configure module logger
logger = logging.getLogger(__name__)

class ConfigManagerState(Enum):
    """États possibles du gestionnaire de configuration"""
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    MAINTENANCE = "maintenance"

@dataclass
class ModuleInfo:
    """Informations sur le module de configuration"""
    name: str = "Alertmanager Receivers Config"
    version: str = "2.1.0"
    description: str = "Module ultra-avancé de gestion des receivers Alertmanager"
    author: str = "Spotify AI Agent Team"
    maintainer: str = "SRE Team"
    last_updated: str = "2025-07-18"
    min_python_version: str = "3.11"
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = [
                "pydantic>=2.0.0",
                "aiofiles>=0.8.0", 
                "cryptography>=3.4.8",
                "jinja2>=3.1.0",
                "prometheus-client>=0.14.0",
                "structlog>=22.1.0"
            ]

# Module info instance
MODULE_INFO = ModuleInfo()

# Main exports
__all__ = [
    # Core managers
    'ReceiverConfigManager',
    'TemplateConfigManager', 
    'EscalationConfigManager',
    
    # Advanced managers
    'SecurityConfigManager',
    'AutomationConfigManager',
    'IntegrationConfigManager', 
    'MetricsConfigManager',
    'AuditConfigManager',
    'PerformanceConfigManager',
    'TenantConfigManager',
    'DisasterRecoveryConfigManager',
    'MLIntelligenceConfigManager',
    'ComplianceConfigManager',
    
    # Utilities
    'ConfigValidator',
    'SchemaValidator',
    'ConfigUtils',
    'EncryptionUtils',
    'ConfigFactory',
    'ReceiverFactory',
    'ConfigRegistry',
    'ServiceRegistry',
    
    # Constants and enums
    'DEFAULT_TIMEOUTS',
    'SEVERITY_LEVELS',
    'NOTIFICATION_CHANNELS', 
    'ESCALATION_POLICIES',
    'SECURITY_POLICIES',
    'TenantTier',
    'AlertSeverity',
    'NotificationChannel',
    'EscalationLevel',
    'SecurityLevel',
    
    # Module metadata
    'MODULE_INFO',
    'ConfigManagerState'
]

# Module initialization
def initialize_module():
    """Initialise le module avec toutes les dépendances"""
    logger.info(f"Initializing {MODULE_INFO.name} v{MODULE_INFO.version}")
    
    # Validation des dépendances
    ConfigValidator.validate_dependencies()
    
    # Chargement des configurations
    ConfigRegistry.load_all_configs()
    
    # Initialisation des services
    ServiceRegistry.initialize_services()
    
    logger.info("Module initialization completed successfully")

# Auto-initialize when imported
try:
    initialize_module()
except Exception as e:
    logger.error(f"Failed to initialize module: {e}")
    raise
