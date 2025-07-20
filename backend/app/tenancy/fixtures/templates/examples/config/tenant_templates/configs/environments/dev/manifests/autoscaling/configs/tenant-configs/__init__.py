"""
Module de Configuration Tenant Avancé - Autoscaling Industriel
==============================================================

Ce module fournit un système de configuration tenant ultra-avancé pour l'autoscaling
des ressources dans un environnement multi-tenant de production Spotify AI Agent.

Auteur: Système d'Architecture Multi-Expert
Version: 2.0.0
Date: 2025-07-17

Composants principaux:
- Gestionnaire de configuration dynamique
- Moteur d'autoscaling adaptatif
- Système de monitoring en temps réel
- Gestionnaire de ressources cloud-native
- Interface de gouvernance et compliance
"""

from .core import TenantConfigManager, AutoscalingEngine
from .monitoring import TenantMetricsCollector, PerformanceAnalyzer
from .resources import ResourceManager, CloudProviderAdapter
from .security import TenantSecurityManager, ComplianceValidator
from .governance import GovernanceEngine, PolicyManager
from .analytics import TenantAnalytics, PredictiveScaler
from .automation import WorkflowManager, DeploymentOrchestrator

__version__ = "2.0.0"
__author__ = "Multi-Expert Architecture Team"

# Configuration de logging spécialisé
import logging
import structlog

logger = structlog.get_logger(__name__)

# Export des classes principales
__all__ = [
    # Configuration Core
    "TenantConfigManager",
    "AutoscalingEngine",
    
    # Monitoring & Analytics
    "TenantMetricsCollector",
    "PerformanceAnalyzer",
    "TenantAnalytics",
    "PredictiveScaler",
    
    # Gestion des ressources
    "ResourceManager",
    "CloudProviderAdapter",
    
    # Sécurité & Gouvernance
    "TenantSecurityManager",
    "ComplianceValidator",
    "GovernanceEngine",
    "PolicyManager",
    
    # Automation
    "WorkflowManager",
    "DeploymentOrchestrator",
]

# Configuration par défaut du module
DEFAULT_CONFIG = {
    "autoscaling": {
        "enabled": True,
        "min_replicas": 2,
        "max_replicas": 50,
        "target_cpu_utilization": 70,
        "target_memory_utilization": 80,
        "scale_up_cooldown": 300,
        "scale_down_cooldown": 600,
    },
    "monitoring": {
        "enabled": True,
        "metrics_interval": 30,
        "alert_threshold": 90,
        "health_check_interval": 60,
    },
    "governance": {
        "policy_validation": True,
        "compliance_checks": True,
        "audit_logging": True,
    }
}

# Initialisation du système
def initialize_tenant_config_system():
    """
    Initialise le système de configuration tenant avec tous les composants.
    """
    logger.info("Initializing advanced tenant configuration system")
    
    # Configuration des composants
    config_manager = TenantConfigManager()
    autoscaling_engine = AutoscalingEngine()
    governance_engine = GovernanceEngine()
    
    logger.info("Tenant configuration system initialized successfully")
    
    return {
        "config_manager": config_manager,
        "autoscaling_engine": autoscaling_engine,
        "governance_engine": governance_engine,
    }
