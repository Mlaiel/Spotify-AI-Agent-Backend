#!/usr/bin/env python3
"""
Configuration Parser and Manager for PagerDuty Integration.

Advanced configuration management system with support for multiple formats,
environment variable substitution, validation, and dynamic reloading.

Features:
- Multiple configuration formats (YAML, JSON, TOML, INI)
- Environment variable substitution and templating
- Configuration validation with Pydantic schemas
- Hierarchical configuration merging
- Dynamic configuration reloading
- Configuration encryption and security
- Configuration versioning and rollback
- Remote configuration sources
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type, Callable
from dataclasses import dataclass, field
import threading
import hashlib
from copy import deepcopy

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import toml
    TOML_AVAILABLE = True
except ImportError:
    TOML_AVAILABLE = False

from pydantic import BaseModel, ValidationError, Field

from .encryption import SecurityManager

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Base exception for configuration errors."""
    pass


class ConfigValidationError(ConfigError):
    """Exception raised for configuration validation errors."""
    pass


class ConfigFormatError(ConfigError):
    """Exception raised for configuration format errors."""
    pass


class ConfigNotFoundError(ConfigError):
    """Exception raised when configuration file is not found."""
    pass


@dataclass
class ConfigSource:
    """Configuration source definition."""
    name: str
    path: Optional[str] = None
    url: Optional[str] = None
    format: str = "yaml"
    priority: int = 0
    required: bool = True
    reload_on_change: bool = False
    last_modified: Optional[datetime] = None
    content_hash: Optional[str] = None


class PagerDutyConfig(BaseModel):
    """Pydantic model for PagerDuty configuration validation."""
    
    class APIConfig(BaseModel):
        token: str = Field(..., description="PagerDuty API token")
        base_url: str = Field("https://api.pagerduty.com", description="API base URL")
        timeout: int = Field(30, ge=1, le=300, description="Request timeout in seconds")
        retry_attempts: int = Field(3, ge=0, le=10, description="Number of retry attempts")
        rate_limit_per_minute: int = Field(960, ge=1, description="Rate limit per minute")
    
    class AlertingConfig(BaseModel):
        default_service_id: Optional[str] = Field(None, description="Default service ID")
        default_urgency: str = Field("high", regex="^(low|high)$", description="Default urgency")
        auto_resolve: bool = Field(True, description="Auto-resolve incidents")
        escalation_timeout: int = Field(1800, ge=60, description="Escalation timeout in seconds")
    
    class CacheConfig(BaseModel):
        enabled: bool = Field(True, description="Enable caching")
        redis_url: str = Field("redis://localhost:6379", description="Redis URL")
        default_ttl: int = Field(300, ge=60, description="Default TTL in seconds")
        max_size: int = Field(1000, ge=100, description="Maximum cache size")
    
    class SecurityConfig(BaseModel):
        encryption_enabled: bool = Field(True, description="Enable encryption")
        encryption_key: Optional[str] = Field(None, description="Encryption key")
        audit_logging: bool = Field(True, description="Enable audit logging")
        sensitive_fields: List[str] = Field(
            default=["token", "password", "secret", "key"],
            description="Fields to encrypt/mask"
        )
    
    api: APIConfig
    alerting: AlertingConfig = Field(default_factory=AlertingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    # Additional custom configuration
    custom: Dict[str, Any] = Field(default_factory=dict, description="Custom configuration")


class ConfigParser:
    """
    Advanced configuration parser and manager.
    
    Features:
    - Multiple format support (YAML, JSON, TOML, INI)
    - Environment variable substitution
    - Configuration validation
    - Hierarchical merging
    - Dynamic reloading
    - Configuration encryption
    """
    
    def __init__(self,
                 config_schema: Optional[Type[BaseModel]] = None,
                 default_format: str = "yaml",
                 env_var_pattern: str = r'\$\{([^}]+)\}',
                 enable_encryption: bool = False,
                 encryption_key: Optional[str] = None):
        """
        Initialize configuration parser.
        
        Args:
            config_schema: Pydantic model for validation
            default_format: Default configuration format
            env_var_pattern: Regex pattern for environment variables
            enable_encryption: Enable configuration encryption
            encryption_key: Encryption key for sensitive data
        """
        self.config_schema = config_schema or PagerDutyConfig
        self.default_format = default_format
        self.env_var_pattern = re.compile(env_var_pattern)
        self.enable_encryption = enable_encryption
        
        # Configuration sources
        self.sources: List[ConfigSource] = []
        self.config_data: Dict[str, Any] = {}
        self.parsed_config: Optional[BaseModel] = None
        
        # Security manager for encryption
        self.security_manager = None
        if enable_encryption:
            self.security_manager = SecurityManager(encryption_key)
        
        # File watching for auto-reload
        self.watchers: Dict[str, Any] = {}
        self.reload_callbacks: List[Callable] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Configuration history for rollback
        self.config_history: List[Dict[str, Any]] = []
        self.max_history_size = 10
        
        logger.info("Configuration parser initialized")
    
    def add_source(self, source: ConfigSource):
        """Add configuration source."""
        with self._lock:
            self.sources.append(source)
            self.sources.sort(key=lambda s: s.priority)
        
        logger.info(f"Added configuration source: {source.name}")
    
    def add_file_source(self, 
                       name: str, 
                       path: str, 
                       format: Optional[str] = None,
                       priority: int = 0,
                       required: bool = True,
                       reload_on_change: bool = False):
        """Add file-based configuration source."""
        if format is None:
            # Detect format from file extension
            ext = Path(path).suffix.lower()
            format_map = {
                '.yaml': 'yaml',
                '.yml': 'yaml',
                '.json': 'json',
                '.toml': 'toml',
                '.ini': 'ini'
            }
            format = format_map.get(ext, self.default_format)
        
        source = ConfigSource(
            name=name,
            path=path,
            format=format,
            priority=priority,
            required=required,
            reload_on_change=reload_on_change
        )
        
        self.add_source(source)
    
    def add_env_source(self, 
                      name: str, 
                      prefix: str = "PAGERDUTY_",
                      priority: int = 100):
        """Add environment variable source."""
        # Create virtual source for environment variables
        source = ConfigSource(
            name=name,
            format="env",
            priority=priority,
            required=False
        )
        source.env_prefix = prefix
        
        self.add_source(source)
    
    def _load_file_content(self, source: ConfigSource) -> Dict[str, Any]:
        """Load content from file source."""
        if not source.path or not os.path.exists(source.path):
            if source.required:
                raise ConfigNotFoundError(f"Required configuration file not found: {source.path}")
            return {}
        
        try:
            with open(source.path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Update source metadata
            stat = os.stat(source.path)
            source.last_modified = datetime.fromtimestamp(stat.st_mtime)
            source.content_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Parse content based on format
            if source.format.lower() in ['yaml', 'yml']:
                if not YAML_AVAILABLE:
                    raise ConfigFormatError("YAML library not available")
                return yaml.safe_load(content) or {}
            
            elif source.format.lower() == 'json':
                return json.loads(content)
            
            elif source.format.lower() == 'toml':
                if not TOML_AVAILABLE:
                    raise ConfigFormatError("TOML library not available")
                return toml.loads(content)
            
            elif source.format.lower() == 'ini':
                import configparser
                config = configparser.ConfigParser()
                config.read_string(content)
                return {section: dict(config[section]) for section in config.sections()}
            
            else:
                raise ConfigFormatError(f"Unsupported configuration format: {source.format}")
                
        except Exception as e:
            if source.required:
                raise ConfigError(f"Failed to load configuration from {source.path}: {e}")
            logger.warning(f"Failed to load optional configuration from {source.path}: {e}")
            return {}
    
    def _load_env_content(self, source: ConfigSource) -> Dict[str, Any]:
        """Load content from environment variables."""
        env_prefix = getattr(source, 'env_prefix', 'PAGERDUTY_')
        config = {}
        
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                # Remove prefix and convert to nested dict
                config_key = key[len(env_prefix):].lower()
                key_parts = config_key.split('_')
                
                # Build nested dictionary
                current = config
                for part in key_parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Try to parse value as JSON, fall back to string
                try:
                    current[key_parts[-1]] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    current[key_parts[-1]] = value
        
        return config
    
    def _substitute_env_vars(self, data: Any) -> Any:
        """Recursively substitute environment variables in configuration data."""
        if isinstance(data, str):
            def replace_var(match):
                var_name = match.group(1)
                default_value = None
                
                # Handle default values: ${VAR_NAME:default_value}
                if ':' in var_name:
                    var_name, default_value = var_name.split(':', 1)
                
                return os.getenv(var_name, default_value or match.group(0))
            
            return self.env_var_pattern.sub(replace_var, data)
        
        elif isinstance(data, dict):
            return {key: self._substitute_env_vars(value) for key, value in data.items()}
        
        elif isinstance(data, list):
            return [self._substitute_env_vars(item) for item in data]
        
        else:
            return data
    
    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration dictionaries recursively."""
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    def _encrypt_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive configuration data."""
        if not self.security_manager:
            return data
        
        # Get sensitive field patterns from schema or default
        sensitive_fields = ["token", "password", "secret", "key", "credential"]
        
        if hasattr(self.parsed_config, 'security') and hasattr(self.parsed_config.security, 'sensitive_fields'):
            sensitive_fields = self.parsed_config.security.sensitive_fields
        
        def encrypt_recursive(obj: Any, path: str = "") -> Any:
            if isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # Check if field should be encrypted
                    should_encrypt = any(
                        sensitive_field.lower() in key.lower()
                        for sensitive_field in sensitive_fields
                    )
                    
                    if should_encrypt and isinstance(value, str):
                        try:
                            result[key] = self.security_manager.encrypt(value.encode()).decode()
                        except Exception as e:
                            logger.warning(f"Failed to encrypt field {current_path}: {e}")
                            result[key] = value
                    else:
                        result[key] = encrypt_recursive(value, current_path)
                
                return result
            
            elif isinstance(obj, list):
                return [encrypt_recursive(item, f"{path}[{i}]") for i, item in enumerate(obj)]
            
            else:
                return obj
        
        return encrypt_recursive(data)
    
    def load(self) -> BaseModel:
        """Load and parse configuration from all sources."""
        with self._lock:
            # Save current config for history
            if self.config_data:
                self.config_history.append(deepcopy(self.config_data))
                # Limit history size
                if len(self.config_history) > self.max_history_size:
                    self.config_history.pop(0)
            
            # Load from all sources in priority order
            merged_config = {}
            
            for source in self.sources:
                try:
                    if source.format == "env":
                        source_data = self._load_env_content(source)
                    else:
                        source_data = self._load_file_content(source)
                    
                    if source_data:
                        # Substitute environment variables
                        source_data = self._substitute_env_vars(source_data)
                        
                        # Merge with existing configuration
                        merged_config = self._merge_config(merged_config, source_data)
                        
                        logger.debug(f"Loaded configuration from source: {source.name}")
                
                except Exception as e:
                    logger.error(f"Failed to load configuration from source {source.name}: {e}")
                    if source.required:
                        raise
            
            # Store raw configuration
            self.config_data = merged_config
            
            # Validate configuration
            try:
                self.parsed_config = self.config_schema(**merged_config)
                logger.info("Configuration loaded and validated successfully")
                
                # Encrypt sensitive data if enabled
                if self.enable_encryption:
                    self.config_data = self._encrypt_sensitive_data(self.config_data)
                
                return self.parsed_config
                
            except ValidationError as e:
                raise ConfigValidationError(f"Configuration validation failed: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated path.
        
        Args:
            key_path: Dot-separated path (e.g., 'api.timeout')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        if not self.parsed_config:
            self.load()
        
        try:
            current = self.parsed_config.dict()
            
            for key in key_path.split('.'):
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return default
            
            return current
            
        except (KeyError, AttributeError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """
        Set configuration value by dot-separated path.
        
        Args:
            key_path: Dot-separated path
            value: Value to set
        """
        with self._lock:
            if not self.config_data:
                self.config_data = {}
            
            keys = key_path.split('.')
            current = self.config_data
            
            # Navigate to parent of target key
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Set the value
            current[keys[-1]] = value
            
            # Revalidate configuration
            try:
                self.parsed_config = self.config_schema(**self.config_data)
                logger.debug(f"Configuration updated: {key_path} = {value}")
            except ValidationError as e:
                # Restore previous state
                self.rollback()
                raise ConfigValidationError(f"Invalid configuration value: {e}")
    
    def reload(self) -> BaseModel:
        """Reload configuration from sources."""
        logger.info("Reloading configuration")
        return self.load()
    
    def rollback(self) -> Optional[BaseModel]:
        """Rollback to previous configuration version."""
        with self._lock:
            if not self.config_history:
                logger.warning("No configuration history available for rollback")
                return None
            
            # Restore previous configuration
            self.config_data = self.config_history.pop()
            
            try:
                self.parsed_config = self.config_schema(**self.config_data)
                logger.info("Configuration rolled back successfully")
                return self.parsed_config
            except ValidationError as e:
                logger.error(f"Failed to rollback configuration: {e}")
                return None
    
    def export(self, format: str = "yaml", include_sensitive: bool = False) -> str:
        """
        Export configuration to string.
        
        Args:
            format: Export format (yaml, json, toml)
            include_sensitive: Include sensitive/encrypted data
            
        Returns:
            Configuration as string
        """
        if not self.parsed_config:
            self.load()
        
        export_data = self.parsed_config.dict()
        
        # Remove sensitive data if requested
        if not include_sensitive and self.enable_encryption:
            sensitive_fields = ["token", "password", "secret", "key", "credential"]
            
            def mask_sensitive(obj: Any) -> Any:
                if isinstance(obj, dict):
                    result = {}
                    for key, value in obj.items():
                        should_mask = any(
                            sensitive_field.lower() in key.lower()
                            for sensitive_field in sensitive_fields
                        )
                        
                        if should_mask:
                            result[key] = "***MASKED***"
                        else:
                            result[key] = mask_sensitive(value)
                    return result
                elif isinstance(obj, list):
                    return [mask_sensitive(item) for item in obj]
                else:
                    return obj
            
            export_data = mask_sensitive(export_data)
        
        # Export in requested format
        if format.lower() in ['yaml', 'yml']:
            if not YAML_AVAILABLE:
                raise ConfigFormatError("YAML library not available")
            return yaml.dump(export_data, default_flow_style=False, sort_keys=True)
        
        elif format.lower() == 'json':
            return json.dumps(export_data, indent=2, sort_keys=True)
        
        elif format.lower() == 'toml':
            if not TOML_AVAILABLE:
                raise ConfigFormatError("TOML library not available")
            return toml.dumps(export_data)
        
        else:
            raise ConfigFormatError(f"Unsupported export format: {format}")
    
    def add_reload_callback(self, callback: Callable[[BaseModel], None]):
        """Add callback to be called when configuration is reloaded."""
        self.reload_callbacks.append(callback)
    
    def start_file_watching(self):
        """Start watching configuration files for changes."""
        # This would require a file watching library like watchdog
        # For now, we'll implement a simple polling mechanism
        async def watch_files():
            while True:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                for source in self.sources:
                    if not source.reload_on_change or not source.path:
                        continue
                    
                    try:
                        if os.path.exists(source.path):
                            stat = os.stat(source.path)
                            last_modified = datetime.fromtimestamp(stat.st_mtime)
                            
                            if source.last_modified and last_modified > source.last_modified:
                                logger.info(f"Configuration file changed: {source.path}")
                                old_config = self.parsed_config
                                new_config = self.reload()
                                
                                # Call reload callbacks
                                for callback in self.reload_callbacks:
                                    try:
                                        callback(new_config)
                                    except Exception as e:
                                        logger.error(f"Reload callback failed: {e}")
                                
                                break
                    
                    except Exception as e:
                        logger.error(f"Error watching file {source.path}: {e}")
        
        # Start watching task
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(watch_files())
            logger.info("Started configuration file watching")
        except RuntimeError:
            logger.warning("No event loop available for file watching")
    
    def get_status(self) -> Dict[str, Any]:
        """Get configuration parser status."""
        with self._lock:
            return {
                'sources_count': len(self.sources),
                'sources': [
                    {
                        'name': source.name,
                        'path': source.path,
                        'format': source.format,
                        'priority': source.priority,
                        'required': source.required,
                        'reload_on_change': source.reload_on_change,
                        'last_modified': source.last_modified.isoformat() if source.last_modified else None
                    }
                    for source in self.sources
                ],
                'config_loaded': self.parsed_config is not None,
                'history_size': len(self.config_history),
                'encryption_enabled': self.enable_encryption,
                'reload_callbacks': len(self.reload_callbacks)
            }


# Global configuration instance
_global_config_parser = None

def get_config_parser() -> ConfigParser:
    """Get global configuration parser instance."""
    global _global_config_parser
    if _global_config_parser is None:
        _global_config_parser = ConfigParser()
    return _global_config_parser


def load_config(config_file: Optional[str] = None) -> PagerDutyConfig:
    """
    Load configuration from file or environment.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Parsed configuration
    """
    parser = get_config_parser()
    
    # Add default sources if none exist
    if not parser.sources:
        # Add file source if provided
        if config_file:
            parser.add_file_source("main", config_file, reload_on_change=True)
        
        # Add environment variables
        parser.add_env_source("env", priority=100)
        
        # Add default config files
        default_files = [
            "config.yaml",
            "config.yml", 
            "config.json",
            "/etc/pagerduty/config.yaml",
            os.path.expanduser("~/.pagerduty/config.yaml")
        ]
        
        for i, file_path in enumerate(default_files):
            if os.path.exists(file_path):
                parser.add_file_source(f"default_{i}", file_path, priority=i, required=False)
    
    return parser.load()


def get_config(key_path: str, default: Any = None) -> Any:
    """Get configuration value using global parser."""
    parser = get_config_parser()
    return parser.get(key_path, default)
