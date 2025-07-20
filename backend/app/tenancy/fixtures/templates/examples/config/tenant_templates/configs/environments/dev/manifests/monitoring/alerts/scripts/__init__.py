"""
Module de scripts d'alertes de monitoring ultra-avancé pour Spotify AI Agent
Architecture Microservices Multi-Tenant avec Intelligence Artificielle

Ce module fournit une infrastructure complète de gestion des alertes
de monitoring en temps réel avec:
- Détection automatique d'anomalies par IA
- Escalade intelligente des incidents
- Corrélation des métriques multi-tenant
- Auto-remédiation basée sur des règles ML
- Intégration native avec l'écosystème Spotify
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

__version__ = "2.1.0"
__author__ = "Spotify AI Agent Team"
__license__ = "MIT"

class AlertSeverity(Enum):
    """Niveaux de sévérité des alertes avec priorité IA"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class AlertCategory(Enum):
    """Catégories d'alertes spécialisées pour l'audio streaming"""
    PERFORMANCE = "performance"
    SECURITY = "security"
    AVAILABILITY = "availability"
    AUDIO_QUALITY = "audio_quality"
    ML_MODEL = "ml_model"
    TENANT_ISOLATION = "tenant_isolation"
    COST_OPTIMIZATION = "cost_optimization"

class ScriptType(Enum):
    """Types de scripts d'alertes automatisées"""
    DETECTION = "detection"
    NOTIFICATION = "notification"
    REMEDIATION = "remediation"
    ESCALATION = "escalation"
    ANALYTICS = "analytics"

@dataclass
class AlertConfig:
    """Configuration avancée d'une alerte de monitoring"""
    name: str
    category: AlertCategory
    severity: AlertSeverity
    script_type: ScriptType
    enabled: bool = True
    tenant_id: Optional[str] = None
    thresholds: Dict[str, Any] = field(default_factory=dict)
    conditions: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    ml_enabled: bool = False
    auto_remediation: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

# Gestionnaire global des configurations d'alertes
_alert_registry: Dict[str, AlertConfig] = {}

def register_alert(config: AlertConfig) -> None:
    """Enregistre une nouvelle configuration d'alerte"""
    _alert_registry[config.name] = config

def get_alert_config(name: str) -> Optional[AlertConfig]:
    """Récupère une configuration d'alerte par nom"""
    return _alert_registry.get(name)

def list_alerts() -> List[AlertConfig]:
    """Liste toutes les configurations d'alertes enregistrées"""
    return list(_alert_registry.values())

# Importation automatique des modules de scripts
from .detection_scripts import *
from .notification_scripts import *
from .remediation_scripts import *
from .escalation_scripts import *
from .analytics_scripts import *
from .performance_monitors import *
from .security_monitors import *
from .ml_anomaly_detectors import *
from .cost_optimizers import *
from .tenant_isolators import *

__all__ = [
    "AlertSeverity",
    "AlertCategory", 
    "ScriptType",
    "AlertConfig",
    "register_alert",
    "get_alert_config",
    "list_alerts",
]

# Logger configuré pour le module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialisation automatique du registre d'alertes
async def _initialize_alert_registry():
    """Initialise automatiquement le registre des alertes au démarrage"""
    try:
        # Auto-découverte des scripts d'alertes
        script_path = Path(__file__).parent
        for script_file in script_path.glob("*.py"):
            if script_file.name != "__init__.py":
                module_name = script_file.stem
                logger.info(f"Initialisation du module d'alertes: {module_name}")
        
        logger.info(f"Registre d'alertes initialisé avec {len(_alert_registry)} configurations")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du registre d'alertes: {e}")

# Démarrage automatique de l'initialisation
asyncio.create_task(_initialize_alert_registry())
