"""
Advanced Configuration Loader for Enterprise Tenant Environments

Ce module fournit un système de chargement avancé des configurations
avec support pour le cache, les overrides, et l'interpolation de variables.

Développé par l'équipe d'experts dirigée par Fahed Mlaiel
"""

import os
import yaml
import json
import re
from typing import Dict, Any, List, Optional, Union, Set
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import threading
import hashlib
from functools import lru_cache
import asyncio
import aiofiles

logger = logging.getLogger(__name__)


@dataclass
class ConfigCache:
    """Cache entry pour les configurations."""
    
    config: Dict[str, Any]
    loaded_at: datetime
    file_hash: str
    ttl: timedelta
    
    def is_expired(self) -> bool:
        """Vérifie si l'entrée de cache a expiré."""
        return datetime.now() - self.loaded_at > self.ttl
    
    def is_valid_for_file(self, file_path: Path) -> bool:
        """Vérifie si le cache est valide pour un fichier."""
        if not file_path.exists():
            return False
        
        current_hash = self._calculate_file_hash(file_path)
        return current_hash == self.file_hash and not self.is_expired()
    
    @staticmethod
    def _calculate_file_hash(file_path: Path) -> str:
        """Calcule le hash MD5 d'un fichier."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()


class VariableInterpolator:
    """Interpolateur de variables pour les configurations."""
    
    def __init__(self):
        """Initialise l'interpolateur."""
        self.variable_pattern = re.compile(r'\$\{([^}]+)\}')
        self.env_pattern = re.compile(r'\$\{([^}:]+):?([^}]*)\}')
    
    def interpolate(self, config: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Interpole les variables dans une configuration.
        
        Args:
            config: Configuration à interpoler
            context: Variables contextuelles additionnelles
            
        Returns:
            Configuration avec variables interpolées
        """
        if context is None:
            context = {}
        
        # Fusion des variables d'environnement et du contexte
        variables = {**os.environ, **context}
        
        return self._interpolate_recursive(config, variables)
    
    def _interpolate_recursive(self, obj: Any, variables: Dict[str, str]) -> Any:
        """Interpolation récursive."""
        if isinstance(obj, dict):
            return {key: self._interpolate_recursive(value, variables) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._interpolate_recursive(item, variables) for item in obj]
        elif isinstance(obj, str):
            return self._interpolate_string(obj, variables)
        else:
            return obj
    
    def _interpolate_string(self, text: str, variables: Dict[str, str]) -> str:
        """Interpole les variables dans une chaîne."""
        def replace_var(match):
            var_expr = match.group(1)
            
            # Support pour les valeurs par défaut: ${VAR:default}
            if ':' in var_expr:
                var_name, default_value = var_expr.split(':', 1)
                return variables.get(var_name.strip(), default_value)
            else:
                var_name = var_expr.strip()
                if var_name in variables:
                    return variables[var_name]
                else:
                    logger.warning(f"Variable non trouvée: {var_name}")
                    return match.group(0)  # Retourne la variable non interpolée
        
        return self.variable_pattern.sub(replace_var, text)
    
    def extract_variables(self, config: Dict[str, Any]) -> Set[str]:
        """Extrait toutes les variables utilisées dans une configuration."""
        variables = set()
        self._extract_variables_recursive(config, variables)
        return variables
    
    def _extract_variables_recursive(self, obj: Any, variables: Set[str]) -> None:
        """Extraction récursive des variables."""
        if isinstance(obj, dict):
            for value in obj.values():
                self._extract_variables_recursive(value, variables)
        elif isinstance(obj, list):
            for item in obj:
                self._extract_variables_recursive(item, variables)
        elif isinstance(obj, str):
            matches = self.variable_pattern.findall(obj)
            for match in matches:
                var_name = match.split(':')[0].strip()
                variables.add(var_name)


class ConfigMerger:
    """Fusionneur de configurations avec support des overrides."""
    
    def merge_configs(self, base_config: Dict[str, Any], *override_configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fusionne plusieurs configurations.
        
        Args:
            base_config: Configuration de base
            *override_configs: Configurations de surcharge
            
        Returns:
            Configuration fusionnée
        """
        result = self._deep_copy(base_config)
        
        for override_config in override_configs:
            self._deep_merge(result, override_config)
        
        return result
    
    def _deep_copy(self, obj: Any) -> Any:
        """Copie profonde d'un objet."""
        if isinstance(obj, dict):
            return {key: self._deep_copy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy(item) for item in obj]
        else:
            return obj
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Fusion profonde de deux dictionnaires."""
        for key, value in override.items():
            if key in base:
                if isinstance(base[key], dict) and isinstance(value, dict):
                    self._deep_merge(base[key], value)
                elif isinstance(base[key], list) and isinstance(value, list):
                    # Pour les listes, on peut choisir différentes stratégies
                    base[key] = self._merge_lists(base[key], value)
                else:
                    base[key] = value
            else:
                base[key] = value
    
    def _merge_lists(self, base_list: List[Any], override_list: List[Any]) -> List[Any]:
        """Fusionne deux listes selon la stratégie configurée."""
        # Par défaut, on remplace complètement
        return override_list


class ConfigLoader:
    """Chargeur avancé de configurations avec cache et optimisations."""
    
    def __init__(self, 
                 cache_ttl: timedelta = timedelta(minutes=15),
                 enable_cache: bool = True,
                 enable_watch: bool = False):
        """
        Initialise le chargeur.
        
        Args:
            cache_ttl: Durée de vie du cache
            enable_cache: Active le cache
            enable_watch: Active la surveillance des fichiers
        """
        self.cache_ttl = cache_ttl
        self.enable_cache = enable_cache
        self.enable_watch = enable_watch
        
        self._cache: Dict[str, ConfigCache] = {}
        self._lock = threading.RLock()
        self._interpolator = VariableInterpolator()
        self._merger = ConfigMerger()
        self._watchers: Dict[str, Any] = {}
    
    def load_config(self, 
                   config_path: Union[str, Path],
                   overrides: List[Union[str, Path]] = None,
                   context: Dict[str, Any] = None,
                   interpolate: bool = True) -> Dict[str, Any]:
        """
        Charge une configuration avec support des overrides.
        
        Args:
            config_path: Chemin vers la configuration principale
            overrides: Liste des configurations de surcharge
            context: Variables contextuelles
            interpolate: Active l'interpolation des variables
            
        Returns:
            Configuration chargée et fusionnée
        """
        config_path = Path(config_path)
        cache_key = self._generate_cache_key(config_path, overrides or [])
        
        # Vérification du cache
        if self.enable_cache and cache_key in self._cache:
            cached_config = self._cache[cache_key]
            if cached_config.is_valid_for_file(config_path):
                logger.debug(f"Configuration chargée depuis le cache: {config_path}")
                return cached_config.config
        
        with self._lock:
            # Double-check locking pattern
            if self.enable_cache and cache_key in self._cache:
                cached_config = self._cache[cache_key]
                if cached_config.is_valid_for_file(config_path):
                    return cached_config.config
            
            # Chargement de la configuration
            config = self._load_single_config(config_path)
            
            # Chargement et fusion des overrides
            if overrides:
                override_configs = []
                for override_path in overrides:
                    override_path = Path(override_path)
                    if override_path.exists():
                        override_config = self._load_single_config(override_path)
                        override_configs.append(override_config)
                    else:
                        logger.warning(f"Override non trouvé: {override_path}")
                
                if override_configs:
                    config = self._merger.merge_configs(config, *override_configs)
            
            # Interpolation des variables
            if interpolate:
                config = self._interpolator.interpolate(config, context)
            
            # Mise en cache
            if self.enable_cache:
                file_hash = ConfigCache._calculate_file_hash(config_path)
                cache_entry = ConfigCache(
                    config=config,
                    loaded_at=datetime.now(),
                    file_hash=file_hash,
                    ttl=self.cache_ttl
                )
                self._cache[cache_key] = cache_entry
            
            # Configuration de la surveillance
            if self.enable_watch and config_path not in self._watchers:
                self._setup_file_watcher(config_path)
            
            return config
    
    def _load_single_config(self, file_path: Path) -> Dict[str, Any]:
        """Charge une configuration depuis un fichier."""
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier de configuration non trouvé: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yml', '.yaml']:
                    return yaml.safe_load(f) or {}
                elif file_path.suffix.lower() == '.json':
                    return json.load(f) or {}
                else:
                    raise ValueError(f"Format de fichier non supporté: {file_path.suffix}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de {file_path}: {e}")
            raise
    
    async def load_config_async(self,
                               config_path: Union[str, Path],
                               overrides: List[Union[str, Path]] = None,
                               context: Dict[str, Any] = None,
                               interpolate: bool = True) -> Dict[str, Any]:
        """Version asynchrone du chargement de configuration."""
        config_path = Path(config_path)
        
        # Chargement asynchrone du fichier principal
        config = await self._load_single_config_async(config_path)
        
        # Chargement asynchrone des overrides
        if overrides:
            override_configs = []
            for override_path in overrides:
                override_path = Path(override_path)
                if override_path.exists():
                    override_config = await self._load_single_config_async(override_path)
                    override_configs.append(override_config)
            
            if override_configs:
                config = self._merger.merge_configs(config, *override_configs)
        
        # Interpolation des variables
        if interpolate:
            config = self._interpolator.interpolate(config, context)
        
        return config
    
    async def _load_single_config_async(self, file_path: Path) -> Dict[str, Any]:
        """Charge une configuration de manière asynchrone."""
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier de configuration non trouvé: {file_path}")
        
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                
                if file_path.suffix.lower() in ['.yml', '.yaml']:
                    return yaml.safe_load(content) or {}
                elif file_path.suffix.lower() == '.json':
                    return json.loads(content) or {}
                else:
                    raise ValueError(f"Format de fichier non supporté: {file_path.suffix}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement asynchrone de {file_path}: {e}")
            raise
    
    def _generate_cache_key(self, config_path: Path, overrides: List[Union[str, Path]]) -> str:
        """Génère une clé de cache unique."""
        key_parts = [str(config_path)]
        key_parts.extend(str(Path(override)) for override in overrides)
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _setup_file_watcher(self, file_path: Path) -> None:
        """Configure la surveillance d'un fichier (à implémenter avec watchdog)."""
        # Implémentation future avec watchdog
        pass
    
    def clear_cache(self, config_path: Union[str, Path] = None) -> None:
        """Vide le cache pour un fichier spécifique ou complètement."""
        with self._lock:
            if config_path:
                # Supprimer les entrées de cache pour ce fichier
                keys_to_remove = [
                    key for key in self._cache.keys() 
                    if str(config_path) in key
                ]
                for key in keys_to_remove:
                    del self._cache[key]
            else:
                # Vider tout le cache
                self._cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache."""
        with self._lock:
            total_entries = len(self._cache)
            expired_entries = sum(1 for cache_entry in self._cache.values() if cache_entry.is_expired())
            
            return {
                "total_entries": total_entries,
                "expired_entries": expired_entries,
                "valid_entries": total_entries - expired_entries,
                "cache_hit_ratio": getattr(self, '_cache_hits', 0) / max(getattr(self, '_cache_requests', 1), 1)
            }
    
    def load_environment_configs(self, 
                                environments_dir: Union[str, Path],
                                environment: str = None) -> Dict[str, Dict[str, Any]]:
        """
        Charge toutes les configurations d'environnement.
        
        Args:
            environments_dir: Répertoire contenant les environnements
            environment: Environnement spécifique à charger (optionnel)
            
        Returns:
            Dictionnaire des configurations par environnement
        """
        environments_dir = Path(environments_dir)
        configs = {}
        
        # Environnements à charger
        if environment:
            env_dirs = [environments_dir / environment]
        else:
            env_dirs = [d for d in environments_dir.iterdir() if d.is_dir()]
        
        for env_dir in env_dirs:
            env_name = env_dir.name
            
            # Fichier de configuration principal
            main_config_file = env_dir / f"{env_name}.yml"
            if not main_config_file.exists():
                main_config_file = env_dir / f"{env_name}.yaml"
            
            if main_config_file.exists():
                # Chargement des overrides
                overrides_dir = env_dir / "overrides"
                overrides = []
                
                if overrides_dir.exists():
                    for override_file in overrides_dir.glob("*.yml"):
                        overrides.append(override_file)
                    for override_file in overrides_dir.glob("*.yaml"):
                        overrides.append(override_file)
                
                # Chargement de la configuration complète
                try:
                    config = self.load_config(
                        config_path=main_config_file,
                        overrides=overrides
                    )
                    configs[env_name] = config
                    logger.info(f"Configuration chargée pour l'environnement: {env_name}")
                except Exception as e:
                    logger.error(f"Erreur lors du chargement de {env_name}: {e}")
            else:
                logger.warning(f"Fichier de configuration principal non trouvé pour {env_name}")
        
        return configs


@lru_cache(maxsize=10)
def get_config_loader(cache_ttl_minutes: int = 15, enable_cache: bool = True) -> ConfigLoader:
    """
    Factory function pour obtenir un loader de configuration avec cache LRU.
    
    Args:
        cache_ttl_minutes: Durée de vie du cache en minutes
        enable_cache: Active le cache
        
    Returns:
        Instance de ConfigLoader
    """
    return ConfigLoader(
        cache_ttl=timedelta(minutes=cache_ttl_minutes),
        enable_cache=enable_cache
    )


def load_config_with_validation(config_path: Union[str, Path],
                               environment: str,
                               overrides: List[Union[str, Path]] = None,
                               validate: bool = True) -> Dict[str, Any]:
    """
    Charge et valide une configuration.
    
    Args:
        config_path: Chemin vers la configuration
        environment: Environnement cible
        overrides: Configurations de surcharge
        validate: Active la validation
        
    Returns:
        Configuration validée
        
    Raises:
        ValidationError: Si la validation échoue
    """
    from .config_validator import ConfigValidator, ValidationResult
    
    # Chargement de la configuration
    loader = get_config_loader()
    config = loader.load_config(config_path, overrides)
    
    # Validation si demandée
    if validate:
        validator = ConfigValidator()
        result = validator.validate_config(config, environment)
        
        if not result.is_valid:
            errors = "; ".join(result.errors + result.security_issues)
            raise ValueError(f"Configuration invalide: {errors}")
        
        if result.warnings:
            for warning in result.warnings:
                logger.warning(f"Configuration warning: {warning}")
    
    return config


__all__ = [
    "ConfigCache",
    "VariableInterpolator", 
    "ConfigMerger",
    "ConfigLoader",
    "get_config_loader",
    "load_config_with_validation"
]
