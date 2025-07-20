"""
Chargeur de Configuration Dynamique - Spotify AI Agent
======================================================

Système avancé de chargement et gestion des configurations avec
support du hot-reload, validation de schéma et environment-specific configs.

Fonctionnalités:
- Chargement dynamique de configurations multi-format (YAML, JSON, TOML)
- Hot-reload automatique avec validation
- Gestion des environnements (dev, staging, prod)
- Validation de schéma avec JSONSchema
- Chiffrement des données sensibles
- Configuration hiérarchique avec inheritance
- Cache intelligent et optimisation
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import yaml
import toml
import os
import hashlib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import jsonschema
from cryptography.fernet import Fernet
import redis.asyncio as redis


class ConfigFormat(str):
    """Formats de configuration supportés"""
    YAML = "yaml"
    JSON = "json"
    TOML = "toml"
    ENV = "env"


@dataclass
class ConfigSource:
    """Source de configuration"""
    path: Path
    format: ConfigFormat
    priority: int = 0
    encrypted: bool = False
    environment_specific: bool = False
    reload_on_change: bool = True
    last_modified: Optional[datetime] = None
    checksum: str = ""


@dataclass
class ConfigValidationResult:
    """Résultat de validation de configuration"""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    schema_version: str = ""
    validated_at: datetime = field(default_factory=datetime.utcnow)


class ConfigChangeHandler(FileSystemEventHandler):
    """Gestionnaire de changements de fichiers de configuration"""
    
    def __init__(self, config_loader):
        self.config_loader = config_loader
        self.logger = logging.getLogger(__name__)
    
    def on_modified(self, event):
        if not event.is_directory:
            asyncio.create_task(self.config_loader._handle_file_change(event.src_path))


class ConfigLoader:
    """Chargeur de configuration avancé"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration de base
        self.base_path = Path(config.get('base_path', '.'))
        self.environment = config.get('environment', 'dev')
        self.encryption_key = config.get('encryption_key')
        
        # Sources de configuration
        self.config_sources: List[ConfigSource] = []
        self.loaded_configs: Dict[str, Any] = {}
        self.config_cache: Dict[str, Any] = {}
        
        # Schémas de validation
        self.schemas: Dict[str, Dict[str, Any]] = {}
        
        # Surveillance des fichiers
        self.file_observer: Optional[Observer] = None
        self.change_handler = ConfigChangeHandler(self)
        
        # Cache Redis
        self.redis_client = None
        self.cache_ttl = timedelta(hours=config.get('cache_ttl_hours', 1))
        
        # Callbacks de changement
        self.change_callbacks: Dict[str, List[Callable]] = {}
        
        # Chiffrement
        self.cipher_suite = None
        if self.encryption_key:
            self.cipher_suite = Fernet(self.encryption_key.encode())
        
        # Métriques de performance
        self.performance_stats = {
            'load_times': [],
            'validation_times': [],
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    async def initialize(self):
        """Initialisation asynchrone du chargeur"""
        try:
            # Connexion Redis pour le cache
            if self.config.get('redis_url'):
                self.redis_client = redis.from_url(
                    self.config.get('redis_url'),
                    decode_responses=True
                )
            
            # Chargement des schémas de validation
            await self._load_validation_schemas()
            
            # Découverte et chargement des sources de configuration
            await self._discover_config_sources()
            await self._load_all_configurations()
            
            # Démarrage de la surveillance des fichiers
            await self._start_file_watching()
            
            self.logger.info("ConfigLoader initialisé avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation: {e}")
            raise
    
    async def get_config(self, key: str, default: Any = None) -> Any:
        """Récupération d'une valeur de configuration"""
        try:
            # Vérification du cache
            cache_key = f"config:{key}:{self.environment}"
            
            if cache_key in self.config_cache:
                self.performance_stats['cache_hits'] += 1
                return self.config_cache[cache_key]
            
            # Vérification du cache Redis
            if self.redis_client:
                cached_value = await self.redis_client.get(cache_key)
                if cached_value:
                    try:
                        value = json.loads(cached_value)
                        self.config_cache[cache_key] = value
                        self.performance_stats['cache_hits'] += 1
                        return value
                    except json.JSONDecodeError:
                        pass
            
            self.performance_stats['cache_misses'] += 1
            
            # Récupération depuis les configurations chargées
            value = self._get_nested_value(self.loaded_configs, key, default)
            
            # Mise en cache
            self.config_cache[cache_key] = value
            if self.redis_client:
                await self.redis_client.setex(
                    cache_key,
                    int(self.cache_ttl.total_seconds()),
                    json.dumps(value, default=str)
                )
            
            return value
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération de config {key}: {e}")
            return default
    
    async def set_config(self, key: str, value: Any, persist: bool = True) -> bool:
        """Modification d'une valeur de configuration"""
        try:
            # Mise à jour en mémoire
            self._set_nested_value(self.loaded_configs, key, value)
            
            # Invalidation du cache
            cache_key = f"config:{key}:{self.environment}"
            if cache_key in self.config_cache:
                del self.config_cache[cache_key]
            
            if self.redis_client:
                await self.redis_client.delete(cache_key)
            
            # Persistance si demandée
            if persist:
                await self._persist_config_change(key, value)
            
            # Déclenchement des callbacks
            await self._trigger_change_callbacks(key, value)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la modification de config {key}: {e}")
            return False
    
    async def validate_config(self, config_name: str = None) -> ConfigValidationResult:
        """Validation des configurations"""
        try:
            start_time = datetime.utcnow()
            result = ConfigValidationResult(valid=True)
            
            configs_to_validate = [config_name] if config_name else list(self.loaded_configs.keys())
            
            for config_key in configs_to_validate:
                config_data = self.loaded_configs.get(config_key, {})
                
                # Recherche du schéma approprié
                schema = self._find_schema_for_config(config_key)
                
                if schema:
                    try:
                        jsonschema.validate(config_data, schema)
                        result.schema_version = schema.get('version', 'unknown')
                    except jsonschema.ValidationError as e:
                        result.valid = False
                        result.errors.append(f"{config_key}: {e.message}")
                    except jsonschema.SchemaError as e:
                        result.warnings.append(f"Schéma invalide pour {config_key}: {e.message}")
                
                # Validations personnalisées
                custom_errors = await self._custom_validation(config_key, config_data)
                result.errors.extend(custom_errors)
            
            # Métriques de performance
            validation_time = (datetime.utcnow() - start_time).total_seconds()
            self.performance_stats['validation_times'].append(validation_time)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la validation: {e}")
            return ConfigValidationResult(
                valid=False,
                errors=[f"Erreur de validation: {e}"]
            )
    
    async def reload_configs(self, config_name: str = None) -> bool:
        """Rechargement des configurations"""
        try:
            if config_name:
                # Rechargement d'une configuration spécifique
                source = self._find_source_for_config(config_name)
                if source:
                    await self._load_config_from_source(source)
                else:
                    self.logger.warning(f"Source non trouvée pour la config {config_name}")
                    return False
            else:
                # Rechargement de toutes les configurations
                await self._load_all_configurations()
            
            # Validation après rechargement
            validation_result = await self.validate_config(config_name)
            if not validation_result.valid:
                self.logger.error(f"Validation échouée après rechargement: {validation_result.errors}")
                return False
            
            # Invalidation du cache
            await self._invalidate_cache()
            
            self.logger.info(f"Configurations rechargées avec succès: {config_name or 'toutes'}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors du rechargement: {e}")
            return False
    
    async def add_config_source(self, source: ConfigSource) -> bool:
        """Ajout d'une nouvelle source de configuration"""
        try:
            # Vérification de l'existence du fichier
            if not source.path.exists():
                self.logger.error(f"Fichier de configuration non trouvé: {source.path}")
                return False
            
            # Calcul du checksum
            source.checksum = await self._calculate_file_checksum(source.path)
            source.last_modified = datetime.fromtimestamp(source.path.stat().st_mtime)
            
            # Ajout à la liste des sources
            self.config_sources.append(source)
            
            # Tri par priorité
            self.config_sources.sort(key=lambda s: s.priority, reverse=True)
            
            # Chargement de la nouvelle source
            await self._load_config_from_source(source)
            
            # Ajout à la surveillance si nécessaire
            if source.reload_on_change and self.file_observer:
                self.file_observer.schedule(
                    self.change_handler,
                    str(source.path.parent),
                    recursive=False
                )
            
            self.logger.info(f"Source de configuration ajoutée: {source.path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'ajout de source: {e}")
            return False
    
    def register_change_callback(self, config_key: str, callback: Callable):
        """Enregistrement d'un callback de changement"""
        if config_key not in self.change_callbacks:
            self.change_callbacks[config_key] = []
        
        self.change_callbacks[config_key].append(callback)
        self.logger.debug(f"Callback enregistré pour {config_key}")
    
    async def get_environment_config(self, env: str = None) -> Dict[str, Any]:
        """Récupération de la configuration pour un environnement spécifique"""
        target_env = env or self.environment
        
        # Filtrage des configurations par environnement
        env_config = {}
        
        for key, value in self.loaded_configs.items():
            if isinstance(value, dict) and 'environments' in value:
                env_value = value.get('environments', {}).get(target_env, value.get('default'))
                if env_value is not None:
                    env_config[key] = env_value
            else:
                env_config[key] = value
        
        return env_config
    
    async def _discover_config_sources(self):
        """Découverte automatique des sources de configuration"""
        try:
            # Patterns de recherche
            patterns = [
                f"**/config.{self.environment}.yaml",
                f"**/config.{self.environment}.json",
                f"**/config.{self.environment}.toml",
                "**/config.yaml",
                "**/config.json",
                "**/config.toml",
                f"**/{self.environment}/*.yaml",
                f"**/{self.environment}/*.json",
                f"**/{self.environment}/*.toml"
            ]
            
            for pattern in patterns:
                for config_file in self.base_path.glob(pattern):
                    if config_file.is_file():
                        # Détermination du format
                        format_map = {
                            '.yaml': ConfigFormat.YAML,
                            '.yml': ConfigFormat.YAML,
                            '.json': ConfigFormat.JSON,
                            '.toml': ConfigFormat.TOML
                        }
                        
                        file_format = format_map.get(config_file.suffix.lower())
                        if not file_format:
                            continue
                        
                        # Détermination de la priorité
                        priority = 0
                        if self.environment in str(config_file):
                            priority += 10
                        if 'config' in config_file.name:
                            priority += 5
                        
                        source = ConfigSource(
                            path=config_file,
                            format=file_format,
                            priority=priority,
                            environment_specific=self.environment in str(config_file)
                        )
                        
                        await self.add_config_source(source)
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la découverte de configurations: {e}")
    
    async def _load_config_from_source(self, source: ConfigSource):
        """Chargement d'une configuration depuis une source"""
        try:
            start_time = datetime.utcnow()
            
            # Lecture du fichier
            content = source.path.read_text(encoding='utf-8')
            
            # Déchiffrement si nécessaire
            if source.encrypted and self.cipher_suite:
                content = self.cipher_suite.decrypt(content.encode()).decode()
            
            # Parsing selon le format
            if source.format == ConfigFormat.YAML:
                config_data = yaml.safe_load(content)
            elif source.format == ConfigFormat.JSON:
                config_data = json.loads(content)
            elif source.format == ConfigFormat.TOML:
                config_data = toml.loads(content)
            else:
                self.logger.warning(f"Format non supporté: {source.format}")
                return
            
            # Intégration dans la configuration globale
            config_name = source.path.stem
            self.loaded_configs[config_name] = config_data
            
            # Mise à jour des métadonnées de la source
            source.last_modified = datetime.fromtimestamp(source.path.stat().st_mtime)
            source.checksum = await self._calculate_file_checksum(source.path)
            
            # Métriques de performance
            load_time = (datetime.utcnow() - start_time).total_seconds()
            self.performance_stats['load_times'].append(load_time)
            
            self.logger.debug(f"Configuration chargée: {source.path}")
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de {source.path}: {e}")
    
    async def _handle_file_change(self, file_path: str):
        """Gestion des changements de fichiers"""
        try:
            path = Path(file_path)
            
            # Recherche de la source correspondante
            source = None
            for s in self.config_sources:
                if s.path == path:
                    source = s
                    break
            
            if not source or not source.reload_on_change:
                return
            
            # Vérification du checksum pour éviter les rechargements inutiles
            new_checksum = await self._calculate_file_checksum(path)
            if new_checksum == source.checksum:
                return
            
            self.logger.info(f"Changement détecté dans {file_path}, rechargement...")
            
            # Rechargement
            await self._load_config_from_source(source)
            
            # Validation
            validation_result = await self.validate_config()
            if not validation_result.valid:
                self.logger.error(f"Configuration invalide après rechargement: {validation_result.errors}")
                return
            
            # Invalidation du cache
            await self._invalidate_cache()
            
            # Déclenchement des callbacks
            config_name = source.path.stem
            await self._trigger_change_callbacks(config_name, self.loaded_configs.get(config_name))
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la gestion du changement: {e}")
    
    def _get_nested_value(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Récupération d'une valeur imbriquée avec notation pointée"""
        keys = key.split('.')
        current = data
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def _set_nested_value(self, data: Dict[str, Any], key: str, value: Any):
        """Modification d'une valeur imbriquée avec notation pointée"""
        keys = key.split('.')
        current = data
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    async def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calcul du checksum d'un fichier"""
        content = file_path.read_bytes()
        return hashlib.sha256(content).hexdigest()
    
    async def _invalidate_cache(self):
        """Invalidation du cache"""
        self.config_cache.clear()
        
        if self.redis_client:
            # Suppression des clés de cache Redis
            pattern = f"config:*:{self.environment}"
            async for key in self.redis_client.scan_iter(match=pattern):
                await self.redis_client.delete(key)
    
    async def _trigger_change_callbacks(self, config_key: str, new_value: Any):
        """Déclenchement des callbacks de changement"""
        if config_key in self.change_callbacks:
            for callback in self.change_callbacks[config_key]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(config_key, new_value)
                    else:
                        callback(config_key, new_value)
                except Exception as e:
                    self.logger.error(f"Erreur dans le callback pour {config_key}: {e}")
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Statistiques de performance du chargeur"""
        stats = self.performance_stats.copy()
        
        if stats['load_times']:
            stats['avg_load_time'] = sum(stats['load_times']) / len(stats['load_times'])
            stats['max_load_time'] = max(stats['load_times'])
        
        if stats['validation_times']:
            stats['avg_validation_time'] = sum(stats['validation_times']) / len(stats['validation_times'])
        
        cache_total = stats['cache_hits'] + stats['cache_misses']
        if cache_total > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / cache_total
        
        stats['total_sources'] = len(self.config_sources)
        stats['total_configs'] = len(self.loaded_configs)
        
        return stats
