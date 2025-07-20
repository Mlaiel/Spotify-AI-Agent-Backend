"""
Enterprise Tenant Configuration Environments Module

Ce module fournit la gestion avancée des configurations d'environnement
pour le système multi-tenant de l'agent IA Spotify.

Architecture:
- Configuration par environnement (dev, staging, prod)
- Validation des configurations
- Chargement dynamique des paramètres
- Gestion des secrets et variables d'environnement
- Support des overrides par tenant

Developed by Expert Team:
- Lead Dev + AI Architect
- Senior Backend Developer (Python/FastAPI/Django)
- Machine Learning Engineer (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Backend Security Specialist
- Microservices Architect
"""

from typing import Dict, Any, Optional, Union
import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EnvironmentType(str, Enum):
    """Types d'environnements supportés."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"
    LOCAL = "local"


@dataclass
class EnvironmentConfig:
    """Configuration d'environnement pour un tenant."""
    
    name: str
    type: EnvironmentType
    config_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialisation avec validation."""
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Valide la configuration d'environnement."""
        if not self.name:
            raise ValueError("Environment name is required")
        
        if not isinstance(self.type, EnvironmentType):
            raise ValueError(f"Invalid environment type: {self.type}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Récupère une valeur de configuration."""
        keys = key.split(".")
        current = self.config_data
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def set(self, key: str, value: Any) -> None:
        """Définit une valeur de configuration."""
        keys = key.split(".")
        current = self.config_data
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def merge(self, other_config: Dict[str, Any]) -> None:
        """Fusionne avec une autre configuration."""
        self._deep_merge(self.config_data, other_config)
    
    def _deep_merge(self, base: Dict, update: Dict) -> None:
        """Fusion profonde de dictionnaires."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value


class EnvironmentConfigManager:
    """Gestionnaire des configurations d'environnement."""
    
    def __init__(self, config_root: str = None):
        """Initialise le gestionnaire."""
        self.config_root = Path(config_root) if config_root else Path(__file__).parent
        self._configs: Dict[str, EnvironmentConfig] = {}
        self._load_all_configs()
    
    def _load_all_configs(self) -> None:
        """Charge toutes les configurations d'environnement."""
        for env_type in EnvironmentType:
            env_dir = self.config_root / env_type.value
            if env_dir.exists() and env_dir.is_dir():
                config_file = env_dir / f"{env_type.value}.yml"
                if config_file.exists():
                    self._load_config(env_type, config_file)
    
    def _load_config(self, env_type: EnvironmentType, config_file: Path) -> None:
        """Charge une configuration spécifique."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            env_config = EnvironmentConfig(
                name=env_type.value,
                type=env_type,
                config_data=config_data or {},
                metadata={
                    "config_file": str(config_file),
                    "loaded_at": "now"
                }
            )
            
            self._configs[env_type.value] = env_config
            logger.info(f"Loaded environment config: {env_type.value}")
            
        except Exception as e:
            logger.error(f"Failed to load config for {env_type.value}: {e}")
            raise
    
    def get_config(self, environment: str) -> Optional[EnvironmentConfig]:
        """Récupère la configuration pour un environnement."""
        return self._configs.get(environment)
    
    def get_all_configs(self) -> Dict[str, EnvironmentConfig]:
        """Récupère toutes les configurations."""
        return self._configs.copy()
    
    def validate_all_configs(self) -> Dict[str, bool]:
        """Valide toutes les configurations."""
        results = {}
        for env_name, config in self._configs.items():
            try:
                config._validate_config()
                results[env_name] = True
            except Exception as e:
                logger.error(f"Validation failed for {env_name}: {e}")
                results[env_name] = False
        
        return results


# Instance globale du gestionnaire
config_manager = EnvironmentConfigManager()


def get_environment_config(environment: str = None) -> EnvironmentConfig:
    """
    Récupère la configuration pour l'environnement spécifié.
    
    Args:
        environment: Nom de l'environnement (par défaut: depuis ENV var)
    
    Returns:
        Configuration de l'environnement
    """
    if not environment:
        environment = os.getenv("ENVIRONMENT", "development")
    
    config = config_manager.get_config(environment)
    if not config:
        raise ValueError(f"Configuration not found for environment: {environment}")
    
    return config


def load_environment_variables(config: EnvironmentConfig) -> None:
    """
    Charge les variables d'environnement à partir de la configuration.
    
    Args:
        config: Configuration d'environnement
    """
    env_vars = config.get("environment_variables", {})
    
    for section_name, section_vars in env_vars.items():
        if isinstance(section_vars, dict):
            for var_name, var_value in section_vars.items():
                if isinstance(var_value, str):
                    os.environ[var_name] = var_value


__all__ = [
    "EnvironmentType",
    "EnvironmentConfig", 
    "EnvironmentConfigManager",
    "config_manager",
    "get_environment_config",
    "load_environment_variables"
]
