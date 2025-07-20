"""
Module de Configuration des Outils de Monitoring et d'Alertes Slack Localisés.

Ce module fournit une configuration avancée pour le système de monitoring et d'alertes
Slack avec support multilingue, multi-tenant, et haute disponibilité.

Architecture:
    - Configuration par environnement (dev, staging, production)
    - Support multi-tenant avec isolation stricte
    - Localisation complète en 5 langues
    - Métriques et monitoring avancés
    - Sécurité et audit intégrés
    - Circuit breakers et résilience
    - Cache multi-niveaux
    - Validation stricte des données

Modules:
    - config_loader: Chargement dynamique des configurations
    - validator: Validation des configurations et données
    - localization: Gestion des langues et traductions
    - metrics: Collecte et export des métriques
    - security: Sécurité et audit
    - tenant_manager: Gestion multi-tenant
    - cache_manager: Gestion du cache multi-niveaux

Auteur: Équipe Spotify AI Agent
Lead Developer: Fahed Mlaiel
Architecture: Microservices avec Event Sourcing
Version: 2.0.0
"""

from typing import Dict, Any, Optional, List
import os
import yaml
from pathlib import Path

__version__ = "2.0.0"
__author__ = "Spotify AI Agent Team - Lead by Fahed Mlaiel"

# Constantes du module
DEFAULT_ENVIRONMENT = "dev"
SUPPORTED_ENVIRONMENTS = ["dev", "staging", "production"]
SUPPORTED_LOCALES = ["fr_FR", "en_US", "de_DE", "es_ES", "it_IT"]

# Configuration par défaut
DEFAULT_CONFIG = {
    "environment": DEFAULT_ENVIRONMENT,
    "debug": False,
    "locale": "fr_FR",
    "tenant_id": "default",
    "cache_enabled": True,
    "monitoring_enabled": True,
    "security_enabled": True,
}

class ConfigurationError(Exception):
    """Exception levée en cas d'erreur de configuration."""
    pass

class ValidationError(Exception):
    """Exception levée en cas d'erreur de validation."""
    pass

def get_config_path(environment: str = None) -> Path:
    """
    Retourne le chemin vers le fichier de configuration pour l'environnement donné.
    
    Args:
        environment: Environnement cible (dev, staging, production)
        
    Returns:
        Path vers le fichier de configuration
        
    Raises:
        ConfigurationError: Si l'environnement n'est pas supporté
    """
    env = environment or os.getenv("ENVIRONMENT", DEFAULT_ENVIRONMENT)
    
    if env not in SUPPORTED_ENVIRONMENTS:
        raise ConfigurationError(f"Environnement non supporté: {env}")
    
    return Path(__file__).parent / f"{env}.yaml"

def load_config(environment: str = None) -> Dict[str, Any]:
    """
    Charge la configuration pour l'environnement spécifié.
    
    Args:
        environment: Environnement cible
        
    Returns:
        Configuration chargée
        
    Raises:
        ConfigurationError: Si le fichier n'existe pas ou est invalide
    """
    config_path = get_config_path(environment)
    
    if not config_path.exists():
        raise ConfigurationError(f"Fichier de configuration introuvable: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Fusion avec la configuration par défaut
        merged_config = {**DEFAULT_CONFIG, **config}
        
        return merged_config
        
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Erreur lors du parsing YAML: {e}")
    except Exception as e:
        raise ConfigurationError(f"Erreur lors du chargement de la configuration: {e}")

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Valide la configuration chargée.
    
    Args:
        config: Configuration à valider
        
    Returns:
        True si la configuration est valide
        
    Raises:
        ValidationError: Si la configuration est invalide
    """
    required_keys = ["environment", "redis", "slack", "alertmanager"]
    
    for key in required_keys:
        if key not in config:
            raise ValidationError(f"Clé de configuration manquante: {key}")
    
    # Validation de l'environnement
    if config["environment"] not in SUPPORTED_ENVIRONMENTS:
        raise ValidationError(f"Environnement invalide: {config['environment']}")
    
    # Validation des locales
    if "localization" in config:
        locales = config["localization"].get("supported_locales", [])
        for locale_info in locales:
            if locale_info["code"] not in SUPPORTED_LOCALES:
                raise ValidationError(f"Locale non supportée: {locale_info['code']}")
    
    return True

# Export des éléments principaux
__all__ = [
    "ConfigurationError",
    "ValidationError", 
    "get_config_path",
    "load_config",
    "validate_config",
    "DEFAULT_CONFIG",
    "SUPPORTED_ENVIRONMENTS",
    "SUPPORTED_LOCALES",
]
