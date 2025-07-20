"""
Advanced Configuration Manager for Scripts System

This module provides comprehensive configuration management with dynamic
loading, validation, inheritance, and real-time updates.

Version: 3.0.0
Developed by Spotify AI Agent Team
Lead Developer & AI Architect: Fahed Mlaiel
"""

import os
import json
import yaml
import toml
import asyncio
import hashlib
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable, Type
from dataclasses import dataclass, field
from pydantic import BaseModel, validator, Field
from cryptography.fernet import Fernet
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import structlog

from .enums import EnvironmentType, CloudProvider, DeploymentMode
from .constants import (
    DEFAULT_CONFIG_PATHS, ENCRYPTION_KEY_LENGTH, CONFIG_CACHE_TTL,
    ERROR_CODES, DEFAULT_TIMEOUTS
)
from .utils import (
    validate_json_schema, encrypt_sensitive_data, decrypt_sensitive_data,
    compute_file_hash, deep_merge_dict
)

logger = structlog.get_logger(__name__)

# ============================================================================
# Configuration Models
# ============================================================================

class BaseConfiguration(BaseModel):
    """Configuration de base avec validation"""
    
    class Config:
        extra = "allow"
        validate_assignment = True
        use_enum_values = True
    
    environment: EnvironmentType = EnvironmentType.DEVELOPMENT
    debug: bool = False
    version: str = "1.0.0"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('version')
    def validate_version(cls, v):
        """Valide le format de version"""
        if not v or not isinstance(v, str):
            raise ValueError("Version must be a non-empty string")
        return v

@dataclass
class ConfigSource:
    """Source de configuration"""
    name: str
    path: str
    format: str
    priority: int = 0
    encrypted: bool = False
    watch: bool = False
    cache_ttl: int = CONFIG_CACHE_TTL
    last_modified: Optional[datetime] = None
    checksum: Optional[str] = None

@dataclass
class ConfigContext:
    """Contexte de configuration"""
    environment: EnvironmentType
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    application: Optional[str] = None
    region: Optional[str] = None
    overrides: Dict[str, Any] = field(default_factory=dict)

class ConfigurationSchema(BaseModel):
    """Schéma de configuration avec validation avancée"""
    
    # Core settings
    application: Dict[str, Any] = Field(default_factory=dict)
    database: Dict[str, Any] = Field(default_factory=dict)
    cache: Dict[str, Any] = Field(default_factory=dict)
    security: Dict[str, Any] = Field(default_factory=dict)
    monitoring: Dict[str, Any] = Field(default_factory=dict)
    
    # Infrastructure
    deployment: Dict[str, Any] = Field(default_factory=dict)
    scaling: Dict[str, Any] = Field(default_factory=dict)
    networking: Dict[str, Any] = Field(default_factory=dict)
    
    # Features
    features: Dict[str, bool] = Field(default_factory=dict)
    integrations: Dict[str, Any] = Field(default_factory=dict)
    
    # ML/AI
    ml_models: Dict[str, Any] = Field(default_factory=dict)
    ai_services: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('database')
    def validate_database_config(cls, v):
        """Valide la configuration de base de données"""
        if not v:
            return v
        
        required_fields = ['host', 'port', 'database']
        for field in required_fields:
            if field not in v:
                logger.warning(f"Missing database field: {field}")
        
        return v

# ============================================================================
# Configuration Change Handlers
# ============================================================================

class ConfigChangeHandler:
    """Gestionnaire de changements de configuration"""
    
    def __init__(self):
        self.handlers: Dict[str, List[Callable]] = {}
        self.lock = threading.Lock()
    
    def register_handler(self, config_path: str, handler: Callable):
        """Enregistre un gestionnaire pour un chemin de configuration"""
        with self.lock:
            if config_path not in self.handlers:
                self.handlers[config_path] = []
            self.handlers[config_path].append(handler)
    
    def unregister_handler(self, config_path: str, handler: Callable):
        """Désenregistre un gestionnaire"""
        with self.lock:
            if config_path in self.handlers:
                try:
                    self.handlers[config_path].remove(handler)
                except ValueError:
                    pass
    
    async def notify_change(self, config_path: str, old_value: Any, new_value: Any):
        """Notifie les changements de configuration"""
        with self.lock:
            handlers = self.handlers.get(config_path, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(config_path, old_value, new_value)
                else:
                    handler(config_path, old_value, new_value)
            except Exception as e:
                logger.error(f"Error in config change handler: {e}")

class FileWatcher(FileSystemEventHandler):
    """Surveillant de fichiers de configuration"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        super().__init__()
    
    def on_modified(self, event):
        """Gestionnaire de modification de fichier"""
        if not event.is_directory:
            asyncio.create_task(
                self.config_manager.reload_config_file(event.src_path)
            )

# ============================================================================
# Advanced Configuration Manager
# ============================================================================

class AdvancedConfigurationManager:
    """Gestionnaire de configuration avancé avec fonctionnalités enterprise"""
    
    def __init__(self, 
                 config_dir: Optional[str] = None,
                 encryption_key: Optional[bytes] = None,
                 enable_watching: bool = True,
                 cache_enabled: bool = True):
        
        self.config_dir = Path(config_dir) if config_dir else Path("./configs")
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.fernet = Fernet(self.encryption_key)
        self.enable_watching = enable_watching
        self.cache_enabled = cache_enabled
        
        # Internal state
        self._config_cache: Dict[str, Dict[str, Any]] = {}
        self._config_sources: Dict[str, ConfigSource] = {}
        self._contexts: Dict[str, ConfigContext] = {}
        self._schema_validators: Dict[str, Type[BaseModel]] = {}
        self._change_handler = ConfigChangeHandler()
        self._lock = asyncio.Lock()
        
        # File watcher
        self._observer = None
        if enable_watching:
            self._setup_file_watcher()
        
        # Default context
        self._default_context = ConfigContext(
            environment=EnvironmentType.DEVELOPMENT
        )
        
        logger.info("Advanced Configuration Manager initialized")
    
    def _setup_file_watcher(self):
        """Configure le surveillant de fichiers"""
        try:
            self._observer = Observer()
            event_handler = FileWatcher(self)
            self._observer.schedule(
                event_handler, 
                str(self.config_dir), 
                recursive=True
            )
            self._observer.start()
            logger.info("File watcher started")
        except Exception as e:
            logger.error(f"Failed to setup file watcher: {e}")
    
    async def add_config_source(self, 
                               name: str,
                               path: str,
                               format: str = "auto",
                               priority: int = 0,
                               encrypted: bool = False,
                               watch: bool = False) -> bool:
        """Ajoute une source de configuration"""
        try:
            # Détection automatique du format
            if format == "auto":
                format = self._detect_format(path)
            
            # Validation du fichier
            if not Path(path).exists():
                logger.error(f"Config file not found: {path}")
                return False
            
            # Création de la source
            source = ConfigSource(
                name=name,
                path=path,
                format=format,
                priority=priority,
                encrypted=encrypted,
                watch=watch,
                last_modified=datetime.fromtimestamp(Path(path).stat().st_mtime),
                checksum=compute_file_hash(path)
            )
            
            self._config_sources[name] = source
            
            # Chargement initial
            await self._load_config_source(source)
            
            logger.info(f"Added config source: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add config source {name}: {e}")
            return False
    
    async def load_configuration(self, 
                               context: Optional[ConfigContext] = None,
                               force_reload: bool = False) -> Dict[str, Any]:
        """Charge la configuration complète avec contexte"""
        
        context = context or self._default_context
        cache_key = self._generate_cache_key(context)
        
        # Vérification du cache
        if not force_reload and self.cache_enabled and cache_key in self._config_cache:
            cached_config = self._config_cache[cache_key]
            if self._is_cache_valid(cached_config):
                return cached_config['data']
        
        async with self._lock:
            try:
                # Chargement de toutes les sources
                merged_config = {}
                
                # Tri des sources par priorité
                sorted_sources = sorted(
                    self._config_sources.values(),
                    key=lambda x: x.priority
                )
                
                for source in sorted_sources:
                    config_data = await self._load_config_source(source)
                    if config_data:
                        merged_config = deep_merge_dict(merged_config, config_data)
                
                # Application des overrides du contexte
                if context.overrides:
                    merged_config = deep_merge_dict(merged_config, context.overrides)
                
                # Application des transformations spécifiques à l'environnement
                merged_config = await self._apply_environment_transforms(
                    merged_config, context
                )
                
                # Validation du schéma
                await self._validate_configuration(merged_config)
                
                # Mise en cache
                if self.cache_enabled:
                    self._config_cache[cache_key] = {
                        'data': merged_config,
                        'timestamp': datetime.utcnow(),
                        'context': context
                    }
                
                logger.info(f"Configuration loaded for context: {context.environment}")
                return merged_config
                
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                raise
    
    async def get_config_value(self,
                             key_path: str,
                             default: Any = None,
                             context: Optional[ConfigContext] = None) -> Any:
        """Récupère une valeur de configuration par chemin"""
        
        config = await self.load_configuration(context)
        
        # Navigation dans la configuration avec chemin de clés
        keys = key_path.split('.')
        current = config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    async def set_config_value(self,
                             key_path: str,
                             value: Any,
                             context: Optional[ConfigContext] = None,
                             persist: bool = False) -> bool:
        """Définit une valeur de configuration"""
        
        try:
            context = context or self._default_context
            
            # Récupération de la configuration actuelle
            config = await self.load_configuration(context)
            
            # Modification de la valeur
            keys = key_path.split('.')
            current = config
            
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            old_value = current.get(keys[-1])
            current[keys[-1]] = value
            
            # Notification du changement
            await self._change_handler.notify_change(key_path, old_value, value)
            
            # Persistance si demandée
            if persist:
                await self._persist_configuration(config, context)
            
            # Invalidation du cache
            cache_key = self._generate_cache_key(context)
            if cache_key in self._config_cache:
                del self._config_cache[cache_key]
            
            logger.info(f"Configuration value set: {key_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set config value {key_path}: {e}")
            return False
    
    async def validate_configuration_schema(self,
                                          config: Dict[str, Any],
                                          schema_class: Type[BaseModel]) -> bool:
        """Valide la configuration contre un schéma"""
        
        try:
            schema_class(**config)
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def register_schema_validator(self, 
                                config_section: str,
                                validator_class: Type[BaseModel]):
        """Enregistre un validateur de schéma"""
        self._schema_validators[config_section] = validator_class
    
    def register_change_handler(self, config_path: str, handler: Callable):
        """Enregistre un gestionnaire de changement"""
        self._change_handler.register_handler(config_path, handler)
    
    async def export_configuration(self,
                                 context: Optional[ConfigContext] = None,
                                 format: str = "yaml",
                                 include_sensitive: bool = False) -> str:
        """Exporte la configuration dans un format donné"""
        
        config = await self.load_configuration(context)
        
        # Filtrage des données sensibles si nécessaire
        if not include_sensitive:
            config = await self._filter_sensitive_data(config)
        
        # Export selon le format
        if format.lower() == "json":
            return json.dumps(config, indent=2, default=str)
        elif format.lower() == "yaml":
            return yaml.dump(config, default_flow_style=False)
        elif format.lower() == "toml":
            return toml.dumps(config)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    async def reload_config_file(self, file_path: str):
        """Recharge un fichier de configuration"""
        
        # Recherche de la source correspondante
        source = None
        for src in self._config_sources.values():
            if src.path == file_path:
                source = src
                break
        
        if source:
            try:
                await self._load_config_source(source, force_reload=True)
                
                # Invalidation du cache
                self._config_cache.clear()
                
                logger.info(f"Reloaded config file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to reload config file {file_path}: {e}")
    
    def get_config_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques de configuration"""
        
        return {
            "sources_count": len(self._config_sources),
            "cached_contexts": len(self._config_cache),
            "schema_validators": len(self._schema_validators),
            "change_handlers": sum(len(handlers) for handlers in self._change_handler.handlers.values()),
            "watcher_active": self._observer is not None and self._observer.is_alive(),
            "encryption_enabled": self.encryption_key is not None,
            "cache_enabled": self.cache_enabled
        }
    
    async def _load_config_source(self, 
                                source: ConfigSource,
                                force_reload: bool = False) -> Optional[Dict[str, Any]]:
        """Charge une source de configuration"""
        
        try:
            file_path = Path(source.path)
            
            # Vérification de modification
            current_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            current_checksum = compute_file_hash(source.path)
            
            if (not force_reload and 
                source.last_modified and 
                current_mtime <= source.last_modified and
                current_checksum == source.checksum):
                return None  # Pas de changement
            
            # Lecture du fichier
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Décryptage si nécessaire
            if source.encrypted:
                content = self.fernet.decrypt(content.encode()).decode()
            
            # Parsing selon le format
            if source.format.lower() == "json":
                config_data = json.loads(content)
            elif source.format.lower() in ["yaml", "yml"]:
                config_data = yaml.safe_load(content)
            elif source.format.lower() == "toml":
                config_data = toml.loads(content)
            else:
                raise ValueError(f"Unsupported format: {source.format}")
            
            # Mise à jour des métadonnées
            source.last_modified = current_mtime
            source.checksum = current_checksum
            
            return config_data
            
        except Exception as e:
            logger.error(f"Failed to load config source {source.name}: {e}")
            return None
    
    def _detect_format(self, file_path: str) -> str:
        """Détecte le format d'un fichier de configuration"""
        
        extension = Path(file_path).suffix.lower()
        
        if extension in ['.json']:
            return "json"
        elif extension in ['.yaml', '.yml']:
            return "yaml"
        elif extension in ['.toml']:
            return "toml"
        else:
            return "yaml"  # Default
    
    def _generate_cache_key(self, context: ConfigContext) -> str:
        """Génère une clé de cache pour un contexte"""
        
        key_data = {
            "environment": context.environment.value,
            "tenant_id": context.tenant_id,
            "user_id": context.user_id,
            "application": context.application,
            "region": context.region,
            "overrides_hash": hashlib.md5(
                json.dumps(context.overrides, sort_keys=True).encode()
            ).hexdigest()
        }
        
        return hashlib.sha256(
            json.dumps(key_data, sort_keys=True).encode()
        ).hexdigest()
    
    def _is_cache_valid(self, cached_config: Dict[str, Any]) -> bool:
        """Vérifie si le cache est valide"""
        
        timestamp = cached_config.get('timestamp')
        if not timestamp:
            return False
        
        return datetime.utcnow() - timestamp < timedelta(seconds=CONFIG_CACHE_TTL)
    
    async def _apply_environment_transforms(self,
                                          config: Dict[str, Any],
                                          context: ConfigContext) -> Dict[str, Any]:
        """Applique les transformations spécifiques à l'environnement"""
        
        # Transformations par environnement
        if context.environment == EnvironmentType.PRODUCTION:
            # Désactiver le debug en production
            if 'debug' in config:
                config['debug'] = False
            
            # Augmenter les timeouts en production
            if 'timeouts' in config:
                for key, value in config['timeouts'].items():
                    if isinstance(value, (int, float)):
                        config['timeouts'][key] = value * 1.5
        
        elif context.environment == EnvironmentType.DEVELOPMENT:
            # Activer le debug en développement
            if 'debug' not in config:
                config['debug'] = True
        
        # Transformations par région
        if context.region:
            # Ajustements spécifiques à la région
            pass
        
        return config
    
    async def _validate_configuration(self, config: Dict[str, Any]):
        """Valide la configuration complète"""
        
        # Validation avec les schémas enregistrés
        for section, validator_class in self._schema_validators.items():
            if section in config:
                try:
                    validator_class(**config[section])
                except Exception as e:
                    logger.warning(f"Validation failed for section {section}: {e}")
        
        # Validations personnalisées
        await self._custom_validations(config)
    
    async def _custom_validations(self, config: Dict[str, Any]):
        """Validations personnalisées"""
        
        # Validation des URLs
        if 'database' in config and 'url' in config['database']:
            url = config['database']['url']
            if not url.startswith(('postgresql://', 'mysql://', 'sqlite://')):
                logger.warning("Database URL format may be incorrect")
        
        # Validation des ports
        for section in config.values():
            if isinstance(section, dict) and 'port' in section:
                port = section['port']
                if not isinstance(port, int) or not (1 <= port <= 65535):
                    logger.warning(f"Invalid port number: {port}")
    
    async def _filter_sensitive_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Filtre les données sensibles de la configuration"""
        
        sensitive_keys = [
            'password', 'secret', 'key', 'token', 'credential',
            'api_key', 'private_key', 'cert', 'certificate'
        ]
        
        def filter_dict(d):
            if isinstance(d, dict):
                return {
                    k: "***FILTERED***" if any(sensitive in k.lower() for sensitive in sensitive_keys)
                    else filter_dict(v) for k, v in d.items()
                }
            elif isinstance(d, list):
                return [filter_dict(item) for item in d]
            return d
        
        return filter_dict(config)
    
    async def _persist_configuration(self, 
                                   config: Dict[str, Any],
                                   context: ConfigContext):
        """Persiste la configuration dans les fichiers sources"""
        
        # Pour le moment, on sauvegarde dans un fichier spécifique au contexte
        filename = f"config_{context.environment.value}.yaml"
        filepath = self.config_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Configuration persisted to: {filepath}")
    
    async def cleanup(self):
        """Nettoyage des ressources"""
        
        if self._observer:
            self._observer.stop()
            self._observer.join()
        
        self._config_cache.clear()
        logger.info("Configuration manager cleaned up")

# ============================================================================
# Factory Functions
# ============================================================================

def create_config_manager(
    config_dir: Optional[str] = None,
    environment: EnvironmentType = EnvironmentType.DEVELOPMENT,
    enable_encryption: bool = True,
    enable_watching: bool = True
) -> AdvancedConfigurationManager:
    """Factory pour créer un gestionnaire de configuration"""
    
    encryption_key = Fernet.generate_key() if enable_encryption else None
    
    manager = AdvancedConfigurationManager(
        config_dir=config_dir,
        encryption_key=encryption_key,
        enable_watching=enable_watching
    )
    
    # Enregistrement des validateurs par défaut
    manager.register_schema_validator("application", ConfigurationSchema)
    
    return manager

async def load_config_from_env() -> Dict[str, Any]:
    """Charge la configuration depuis les variables d'environnement"""
    
    config = {}
    
    # Variables d'environnement communes
    env_mappings = {
        'APP_NAME': 'application.name',
        'APP_VERSION': 'application.version',
        'APP_DEBUG': 'application.debug',
        'DATABASE_URL': 'database.url',
        'REDIS_URL': 'cache.redis.url',
        'LOG_LEVEL': 'logging.level',
        'SECRET_KEY': 'security.secret_key'
    }
    
    for env_var, config_path in env_mappings.items():
        value = os.getenv(env_var)
        if value:
            # Navigation dans la structure de configuration
            keys = config_path.split('.')
            current = config
            
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Conversion du type si nécessaire
            if value.lower() in ['true', 'false']:
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            
            current[keys[-1]] = value
    
    return config

# Export principal
__all__ = [
    "AdvancedConfigurationManager",
    "BaseConfiguration",
    "ConfigurationSchema",
    "ConfigSource",
    "ConfigContext",
    "ConfigChangeHandler",
    "create_config_manager",
    "load_config_from_env"
]
