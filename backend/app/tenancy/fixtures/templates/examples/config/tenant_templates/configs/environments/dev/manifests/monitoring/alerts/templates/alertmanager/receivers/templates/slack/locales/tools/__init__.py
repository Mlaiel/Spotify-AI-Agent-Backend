#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spotify AI Agent - Module d'Outils de Localisation Slack pour Alertes Multi-Tenant

Ce module fournit des outils avancés pour la gestion des notifications Slack
localisées dans un environnement multi-tenant avec surveillance intelligente.

Architecture:
- Système de localisation dynamique multi-langue
- Formatage contextuel des alertes par tenant
- Intégration avec AlertManager et Prometheus
- Support des templates Jinja2 avancés
- Cache Redis pour performances optimisées
- Monitoring et métriques en temps réel

Modules disponibles:
- localization_engine: Moteur de localisation principal
- template_renderer: Rendu de templates Slack
- alert_formatter: Formatage des alertes par contexte
- tenant_context: Gestionnaire de contexte tenant
- metrics_collector: Collecte de métriques
- cache_manager: Gestionnaire de cache Redis
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

# Configuration des logs
logger = logging.getLogger(__name__)

# Métadonnées du module
__version__ = "2.1.0"
__author__ = "Spotify AI Agent Team"
__email__ = "ai-team@spotify-agent.com"
__status__ = "Production"
__license__ = "MIT"

# Chemins et configurations
MODULE_ROOT = Path(__file__).parent
TEMPLATES_DIR = MODULE_ROOT / "templates"
LOCALES_DIR = MODULE_ROOT / "locales"
CONFIGS_DIR = MODULE_ROOT / "configs"
CACHE_DIR = MODULE_ROOT / "cache"

# Configuration par défaut
DEFAULT_CONFIG = {
    "slack": {
        "default_locale": "fr_FR",
        "supported_locales": ["fr_FR", "en_US", "de_DE", "es_ES"],
        "timeout": 30,
        "retry_attempts": 3,
        "rate_limit": {
            "requests_per_minute": 60,
            "burst_limit": 10
        }
    },
    "alertmanager": {
        "template_engine": "jinja2",
        "cache_ttl": 300,
        "batch_size": 50,
        "compression": True
    },
    "tenant": {
        "isolation_level": "strict",
        "context_cache_ttl": 600,
        "max_context_size": 1024
    },
    "monitoring": {
        "metrics_enabled": True,
        "health_check_interval": 60,
        "performance_tracking": True
    }
}

# Imports des modules principaux
try:
    from .localization_engine import LocalizationEngine
    from .template_renderer import SlackTemplateRenderer
    from .alert_formatter import AlertFormatter
    from .tenant_context import TenantContextManager
    from .metrics_collector import MetricsCollector
    from .cache_manager import CacheManager
    
    # Initialisation des services principaux
    _localization_engine = None
    _template_renderer = None
    _alert_formatter = None
    _tenant_context = None
    _metrics_collector = None
    _cache_manager = None
    
except ImportError as e:
    logger.warning(f"Certains modules ne sont pas disponibles: {e}")

def initialize_services(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Initialise tous les services du module avec configuration.
    
    Args:
        config: Configuration personnalisée (optionnel)
        
    Returns:
        Dict contenant les instances des services initialisés
    """
    global _localization_engine, _template_renderer, _alert_formatter
    global _tenant_context, _metrics_collector, _cache_manager
    
    try:
        # Fusion de la configuration
        final_config = {**DEFAULT_CONFIG}
        if config:
            final_config.update(config)
        
        # Initialisation du cache manager
        _cache_manager = CacheManager(final_config.get("cache", {}))
        
        # Initialisation du collecteur de métriques
        _metrics_collector = MetricsCollector(final_config.get("monitoring", {}))
        
        # Initialisation du gestionnaire de contexte tenant
        _tenant_context = TenantContextManager(
            final_config.get("tenant", {}),
            cache_manager=_cache_manager
        )
        
        # Initialisation du moteur de localisation
        _localization_engine = LocalizationEngine(
            final_config.get("localization", {}),
            cache_manager=_cache_manager
        )
        
        # Initialisation du renderer de templates
        _template_renderer = SlackTemplateRenderer(
            final_config.get("slack", {}),
            localization_engine=_localization_engine
        )
        
        # Initialisation du formateur d'alertes
        _alert_formatter = AlertFormatter(
            final_config.get("alertmanager", {}),
            template_renderer=_template_renderer,
            tenant_context=_tenant_context
        )
        
        logger.info("Services du module d'outils Slack initialisés avec succès")
        
        return {
            "localization_engine": _localization_engine,
            "template_renderer": _template_renderer,
            "alert_formatter": _alert_formatter,
            "tenant_context": _tenant_context,
            "metrics_collector": _metrics_collector,
            "cache_manager": _cache_manager
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation des services: {e}")
        raise

def get_service(service_name: str) -> Any:
    """
    Récupère une instance de service par son nom.
    
    Args:
        service_name: Nom du service à récupérer
        
    Returns:
        Instance du service demandé
    """
    services = {
        "localization_engine": _localization_engine,
        "template_renderer": _template_renderer,
        "alert_formatter": _alert_formatter,
        "tenant_context": _tenant_context,
        "metrics_collector": _metrics_collector,
        "cache_manager": _cache_manager
    }
    
    service = services.get(service_name)
    if service is None:
        raise ValueError(f"Service '{service_name}' non trouvé ou non initialisé")
    
    return service

# Export des classes principales
__all__ = [
    "LocalizationEngine",
    "SlackTemplateRenderer", 
    "AlertFormatter",
    "TenantContextManager",
    "MetricsCollector",
    "CacheManager",
    "initialize_services",
    "get_service",
    "DEFAULT_CONFIG",
    "MODULE_ROOT"
]
