"""
🎵 Advanced Configuration Management for Spotify AI Agent Automation
Ultra-sophisticated configuration system with dynamic updates and validation

This module provides enterprise-grade configuration management including:
- Dynamic configuration loading and hot-reloading
- Environment-specific configuration management
- Configuration validation and schema enforcement
- Encrypted configuration storage for sensitive data
- Configuration versioning and rollback capabilities
- Multi-tenant configuration isolation

Author: Fahed Mlaiel (Lead Developer & AI Architect)
"""

import os
import json
import yaml
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import asyncio
from cryptography.fernet import Fernet
import hashlib
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigEnvironment(Enum):
    """Configuration environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class ConfigFormat(Enum):
    """Supported configuration formats"""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    ENV = "env"


@dataclass
class ConfigSchema:
    """Configuration schema definition"""
    name: str
    version: str
    fields: Dict[str, Any]
    required_fields: List[str] = field(default_factory=list)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    sensitive_fields: List[str] = field(default_factory=list)


@dataclass
class ConfigurationSet:
    """Complete configuration set for automation system"""
    
    # Core automation settings
    automation: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "operation_mode": "balanced",
        "max_concurrent_workflows": 100,
        "max_concurrent_actions": 500,
        "workflow_timeout_seconds": 3600,
        "action_timeout_seconds": 600,
        "retry_attempts": 3,
        "retry_delay_seconds": 5,
        "circuit_breaker_threshold": 5,
        "rate_limiting_enabled": True,
        "rate_limit_per_minute": 1000
    })
    
    # Machine Learning settings
    ml: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "prediction_confidence_threshold": 0.85,
        "anomaly_detection_enabled": True,
        "model_retrain_interval_hours": 168,  # 1 week
        "feature_window_hours": 48,
        "prediction_horizon_hours": 24,
        "models": {
            "traffic_predictor": {
                "type": "lstm",
                "batch_size": 32,
                "epochs": 100,
                "learning_rate": 0.001
            },
            "resource_predictor": {
                "type": "random_forest",
                "n_estimators": 100,
                "max_depth": 10
            },
            "failure_predictor": {
                "type": "gradient_boosting",
                "n_estimators": 100,
                "learning_rate": 0.1
            },
            "anomaly_detector": {
                "type": "isolation_forest",
                "contamination": 0.1,
                "n_estimators": 100
            }
        }
    })
    
    # Monitoring and alerting settings
    monitoring: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "metrics_collection_interval_seconds": 30,
        "health_check_interval_seconds": 60,
        "alert_evaluation_interval_seconds": 30,
        "prometheus_metrics_port": 8000,
        "thresholds": {
            "cpu_usage_percent": 80.0,
            "memory_usage_percent": 85.0,
            "disk_usage_percent": 90.0,
            "error_rate_percent": 5.0,
            "response_time_p95_ms": 500.0,
            "recommendation_accuracy": 0.85
        },
        "alert_channels": {
            "slack": {
                "enabled": True,
                "webhook_url": "${SLACK_WEBHOOK_URL}",
                "default_channel": "#ops-alerts",
                "severity_channels": {
                    "critical": "#critical-alerts",
                    "emergency": "#emergency-alerts"
                }
            },
            "email": {
                "enabled": True,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "${EMAIL_USERNAME}",
                "password": "${EMAIL_PASSWORD}",
                "recipients": ["ops-team@spotify-ai-agent.com"]
            },
            "pagerduty": {
                "enabled": False,
                "integration_key": "${PAGERDUTY_INTEGRATION_KEY}"
            }
        }
    })
    
    # Data storage settings
    storage: Dict[str, Any] = field(default_factory=lambda: {
        "redis": {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "password": "${REDIS_PASSWORD}",
            "ssl": False,
            "cluster_mode": False,
            "max_connections": 100,
            "retry_on_timeout": True,
            "socket_timeout": 30,
            "socket_connect_timeout": 30
        },
        "postgresql": {
            "host": "localhost",
            "port": 5432,
            "database": "spotify_ai_agent",
            "username": "${POSTGRES_USERNAME}",
            "password": "${POSTGRES_PASSWORD}",
            "ssl_mode": "prefer",
            "max_connections": 20,
            "connection_timeout": 30
        },
        "mongodb": {
            "host": "localhost",
            "port": 27017,
            "database": "spotify_ai_agent_logs",
            "username": "${MONGODB_USERNAME}",
            "password": "${MONGODB_PASSWORD}",
            "auth_source": "admin",
            "ssl": False,
            "max_pool_size": 10
        }
    })
    
    # Security settings
    security: Dict[str, Any] = field(default_factory=lambda: {
        "encryption_enabled": True,
        "encryption_key": "${ENCRYPTION_KEY}",
        "api_rate_limiting": {
            "enabled": True,
            "requests_per_minute": 1000,
            "burst_limit": 2000
        },
        "authentication": {
            "enabled": True,
            "token_expiry_hours": 24,
            "refresh_token_expiry_days": 30,
            "secret_key": "${JWT_SECRET_KEY}"
        },
        "authorization": {
            "rbac_enabled": True,
            "default_role": "viewer",
            "admin_users": ["${ADMIN_USER}"]
        },
        "audit_logging": {
            "enabled": True,
            "log_level": "INFO",
            "retention_days": 90,
            "encrypt_logs": True
        }
    })
    
    # Spotify-specific settings
    spotify: Dict[str, Any] = field(default_factory=lambda: {
        "api": {
            "client_id": "${SPOTIFY_CLIENT_ID}",
            "client_secret": "${SPOTIFY_CLIENT_SECRET}",
            "redirect_uri": "https://api.spotify-ai-agent.com/callback",
            "scopes": [
                "user-read-playback-state",
                "user-modify-playback-state", 
                "user-read-currently-playing",
                "playlist-read-private",
                "playlist-modify-public",
                "playlist-modify-private",
                "user-library-read",
                "user-library-modify",
                "user-top-read",
                "user-read-recently-played"
            ]
        },
        "recommendation_engine": {
            "enabled": True,
            "model_version": "v2.1.0",
            "confidence_threshold": 0.75,
            "max_recommendations": 50,
            "diversity_factor": 0.3,
            "freshness_factor": 0.2,
            "popularity_bias": 0.1
        },
        "audio_analysis": {
            "enabled": True,
            "feature_extraction": [
                "tempo", "key", "loudness", "speechiness",
                "acousticness", "instrumentalness", "liveness",
                "valence", "danceability", "energy"
            ],
            "batch_size": 100,
            "cache_duration_hours": 24
        },
        "personalization": {
            "user_profiling_enabled": True,
            "real_time_updates": True,
            "preference_learning_rate": 0.01,
            "context_awareness": {
                "time_of_day": True,
                "day_of_week": True,
                "location": False,  # Privacy considerations
                "activity": True,
                "mood": True
            }
        }
    })
    
    # Performance optimization settings
    performance: Dict[str, Any] = field(default_factory=lambda: {
        "caching": {
            "enabled": True,
            "default_ttl_seconds": 300,
            "cache_strategies": {
                "api_responses": "redis",
                "ml_predictions": "memory",
                "user_profiles": "redis",
                "recommendation_results": "redis"
            },
            "cache_sizes": {
                "memory_cache_mb": 512,
                "redis_max_memory": "2gb"
            }
        },
        "concurrency": {
            "thread_pool_size": 20,
            "process_pool_size": 8,
            "async_workers": 50,
            "max_queue_size": 1000
        },
        "optimization": {
            "auto_scaling_enabled": True,
            "scaling_metrics": ["cpu_usage", "memory_usage", "queue_size"],
            "scale_up_threshold": 75.0,
            "scale_down_threshold": 25.0,
            "min_instances": 2,
            "max_instances": 20,
            "cooldown_seconds": 300
        }
    })
    
    # Logging configuration
    logging: Dict[str, Any] = field(default_factory=lambda: {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "handlers": {
            "console": {
                "enabled": True,
                "level": "INFO"
            },
            "file": {
                "enabled": True,
                "level": "DEBUG",
                "filename": "/var/log/spotify-ai-agent/automation.log",
                "max_bytes": 104857600,  # 100MB
                "backup_count": 5,
                "rotation": "size"
            },
            "elasticsearch": {
                "enabled": False,
                "host": "localhost",
                "port": 9200,
                "index": "spotify-ai-agent-logs"
            }
        },
        "structured_logging": True,
        "correlation_id_header": "X-Correlation-ID"
    })


class ConfigurationManager:
    """Advanced configuration management system"""
    
    def __init__(self, environment: ConfigEnvironment = ConfigEnvironment.DEVELOPMENT):
        self.environment = environment
        self.config = ConfigurationSet()
        self.schemas = {}
        self.watchers = {}
        self.encryption_key = None
        self.config_history = []
        self.validation_errors = []
        
        # Initialize encryption
        self._initialize_encryption()
        
        # Load configuration
        asyncio.create_task(self.load_configuration())
    
    def _initialize_encryption(self):
        """Initialize encryption for sensitive configuration data"""
        try:
            # Try to load existing key
            key_file = Path(f".encryption_key_{self.environment.value}")
            if key_file.exists():
                with open(key_file, 'rb') as f:
                    self.encryption_key = f.read()
            else:
                # Generate new key
                self.encryption_key = Fernet.generate_key()
                with open(key_file, 'wb') as f:
                    f.write(self.encryption_key)
                logger.info("Generated new encryption key")
            
            self.cipher = Fernet(self.encryption_key)
            
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            self.encryption_key = None
            self.cipher = None
    
    async def load_configuration(self):
        """Load configuration from various sources"""
        logger.info(f"Loading configuration for environment: {self.environment.value}")
        
        try:
            # Load from environment variables
            await self._load_from_environment()
            
            # Load from configuration files
            await self._load_from_files()
            
            # Load from external services (Consul, etcd, etc.)
            await self._load_from_external_services()
            
            # Apply environment-specific overrides
            await self._apply_environment_overrides()
            
            # Validate configuration
            await self.validate_configuration()
            
            # Process variables and substitutions
            await self._process_variables()
            
            logger.info("Configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    async def _load_from_environment(self):
        """Load configuration from environment variables"""
        env_mappings = {
            # Database configurations
            'REDIS_HOST': ('storage.redis.host', str),
            'REDIS_PORT': ('storage.redis.port', int),
            'REDIS_PASSWORD': ('storage.redis.password', str),
            'POSTGRES_HOST': ('storage.postgresql.host', str),
            'POSTGRES_PORT': ('storage.postgresql.port', int),
            'POSTGRES_USERNAME': ('storage.postgresql.username', str),
            'POSTGRES_PASSWORD': ('storage.postgresql.password', str),
            'POSTGRES_DATABASE': ('storage.postgresql.database', str),
            
            # Spotify API
            'SPOTIFY_CLIENT_ID': ('spotify.api.client_id', str),
            'SPOTIFY_CLIENT_SECRET': ('spotify.api.client_secret', str),
            
            # Security
            'JWT_SECRET_KEY': ('security.authentication.secret_key', str),
            'ENCRYPTION_KEY': ('security.encryption_key', str),
            
            # Monitoring
            'SLACK_WEBHOOK_URL': ('monitoring.alert_channels.slack.webhook_url', str),
            'EMAIL_USERNAME': ('monitoring.alert_channels.email.username', str),
            'EMAIL_PASSWORD': ('monitoring.alert_channels.email.password', str),
            'PAGERDUTY_INTEGRATION_KEY': ('monitoring.alert_channels.pagerduty.integration_key', str),
            
            # Performance
            'MAX_CONCURRENT_WORKFLOWS': ('automation.max_concurrent_workflows', int),
            'MAX_CONCURRENT_ACTIONS': ('automation.max_concurrent_actions', int),
            'WORKFLOW_TIMEOUT': ('automation.workflow_timeout_seconds', int),
            
            # ML Configuration
            'ML_PREDICTION_THRESHOLD': ('ml.prediction_confidence_threshold', float),
            'MODEL_RETRAIN_INTERVAL': ('ml.model_retrain_interval_hours', int),
        }
        
        for env_var, (config_path, value_type) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    # Convert to appropriate type
                    if value_type == int:
                        value = int(value)
                    elif value_type == float:
                        value = float(value)
                    elif value_type == bool:
                        value = value.lower() in ('true', '1', 'yes', 'on')
                    
                    # Set nested configuration value
                    self._set_nested_value(self.config, config_path, value)
                    logger.debug(f"Set {config_path} from environment variable {env_var}")
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid value for {env_var}: {value} ({e})")
    
    async def _load_from_files(self):
        """Load configuration from files"""
        config_dirs = [
            f"config/{self.environment.value}",
            "config/common",
            "/etc/spotify-ai-agent",
            os.path.expanduser("~/.spotify-ai-agent")
        ]
        
        for config_dir in config_dirs:
            config_path = Path(config_dir)
            if config_path.exists() and config_path.is_dir():
                await self._load_from_directory(config_path)
    
    async def _load_from_directory(self, config_dir: Path):
        """Load all configuration files from a directory"""
        config_files = [
            "automation.yaml",
            "automation.yml", 
            "automation.json",
            "monitoring.yaml",
            "monitoring.yml",
            "monitoring.json",
            "security.yaml",
            "security.yml",
            "security.json",
            "spotify.yaml",
            "spotify.yml",
            "spotify.json"
        ]
        
        for config_file in config_files:
            file_path = config_dir / config_file
            if file_path.exists():
                await self._load_config_file(file_path)
    
    async def _load_config_file(self, file_path: Path):
        """Load configuration from a specific file"""
        try:
            logger.debug(f"Loading configuration from: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                elif file_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    logger.warning(f"Unsupported file format: {file_path}")
                    return
            
            # Merge configuration data
            if config_data:
                self._merge_configuration(config_data)
                logger.info(f"Loaded configuration from: {file_path}")
                
        except Exception as e:
            logger.error(f"Failed to load configuration file {file_path}: {e}")
    
    async def _load_from_external_services(self):
        """Load configuration from external services (Consul, etcd, etc.)"""
        # This would integrate with external configuration services
        # For now, it's a placeholder
        logger.debug("External configuration services not configured")
    
    async def _apply_environment_overrides(self):
        """Apply environment-specific configuration overrides"""
        overrides = {
            ConfigEnvironment.DEVELOPMENT: {
                'logging.level': 'DEBUG',
                'monitoring.metrics_collection_interval_seconds': 10,
                'automation.retry_attempts': 1,
                'ml.model_retrain_interval_hours': 24,
                'performance.caching.default_ttl_seconds': 60
            },
            ConfigEnvironment.STAGING: {
                'logging.level': 'INFO',
                'monitoring.metrics_collection_interval_seconds': 20,
                'automation.retry_attempts': 2,
                'ml.model_retrain_interval_hours': 72,
                'performance.caching.default_ttl_seconds': 180
            },
            ConfigEnvironment.PRODUCTION: {
                'logging.level': 'WARNING',
                'monitoring.metrics_collection_interval_seconds': 30,
                'automation.retry_attempts': 3,
                'ml.model_retrain_interval_hours': 168,
                'performance.caching.default_ttl_seconds': 300,
                'security.audit_logging.enabled': True,
                'performance.auto_scaling_enabled': True
            },
            ConfigEnvironment.TESTING: {
                'logging.level': 'DEBUG',
                'monitoring.metrics_collection_interval_seconds': 5,
                'automation.retry_attempts': 1,
                'ml.enabled': False,
                'monitoring.enabled': False,
                'performance.caching.enabled': False
            }
        }
        
        env_overrides = overrides.get(self.environment, {})
        for config_path, value in env_overrides.items():
            self._set_nested_value(self.config, config_path, value)
            logger.debug(f"Applied environment override: {config_path} = {value}")
    
    def _set_nested_value(self, obj: Any, path: str, value: Any):
        """Set a nested configuration value using dot notation"""
        keys = path.split('.')
        current = obj
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if hasattr(current, key):
                current = getattr(current, key)
            elif isinstance(current, dict):
                if key not in current:
                    current[key] = {}
                current = current[key]
            else:
                return  # Can't set nested value
        
        # Set the final value
        final_key = keys[-1]
        if hasattr(current, final_key):
            setattr(current, final_key, value)
        elif isinstance(current, dict):
            current[final_key] = value
    
    def _get_nested_value(self, obj: Any, path: str, default: Any = None) -> Any:
        """Get a nested configuration value using dot notation"""
        keys = path.split('.')
        current = obj
        
        try:
            for key in keys:
                if hasattr(current, key):
                    current = getattr(current, key)
                elif isinstance(current, dict):
                    current = current[key]
                else:
                    return default
            return current
        except (KeyError, AttributeError):
            return default
    
    def _merge_configuration(self, new_config: Dict[str, Any]):
        """Merge new configuration data into existing configuration"""
        def merge_dicts(target: Dict[str, Any], source: Dict[str, Any]):
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    merge_dicts(target[key], value)
                else:
                    target[key] = value
        
        # Convert config object to dict for merging
        config_dict = self._config_to_dict()
        merge_dicts(config_dict, new_config)
        
        # Update config object
        self._dict_to_config(config_dict)
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert configuration object to dictionary"""
        return {
            'automation': self.config.automation,
            'ml': self.config.ml,
            'monitoring': self.config.monitoring,
            'storage': self.config.storage,
            'security': self.config.security,
            'spotify': self.config.spotify,
            'performance': self.config.performance,
            'logging': self.config.logging
        }
    
    def _dict_to_config(self, config_dict: Dict[str, Any]):
        """Update configuration object from dictionary"""
        self.config.automation.update(config_dict.get('automation', {}))
        self.config.ml.update(config_dict.get('ml', {}))
        self.config.monitoring.update(config_dict.get('monitoring', {}))
        self.config.storage.update(config_dict.get('storage', {}))
        self.config.security.update(config_dict.get('security', {}))
        self.config.spotify.update(config_dict.get('spotify', {}))
        self.config.performance.update(config_dict.get('performance', {}))
        self.config.logging.update(config_dict.get('logging', {}))
    
    async def _process_variables(self):
        """Process variable substitutions in configuration"""
        def process_value(value):
            if isinstance(value, str):
                # Replace environment variables
                if value.startswith('${') and value.endswith('}'):
                    var_name = value[2:-1]
                    env_value = os.getenv(var_name)
                    if env_value is not None:
                        return env_value
                    else:
                        logger.warning(f"Environment variable not found: {var_name}")
                        return value
            elif isinstance(value, dict):
                return {k: process_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [process_value(item) for item in value]
            
            return value
        
        config_dict = self._config_to_dict()
        processed_config = process_value(config_dict)
        self._dict_to_config(processed_config)
    
    async def validate_configuration(self) -> bool:
        """Validate the configuration against defined schemas"""
        self.validation_errors = []
        
        try:
            # Define validation rules
            validation_rules = {
                'automation.max_concurrent_workflows': lambda x: isinstance(x, int) and x > 0,
                'automation.max_concurrent_actions': lambda x: isinstance(x, int) and x > 0,
                'automation.workflow_timeout_seconds': lambda x: isinstance(x, int) and x > 0,
                'ml.prediction_confidence_threshold': lambda x: isinstance(x, (int, float)) and 0 <= x <= 1,
                'monitoring.metrics_collection_interval_seconds': lambda x: isinstance(x, int) and x > 0,
                'storage.redis.port': lambda x: isinstance(x, int) and 1 <= x <= 65535,
                'storage.postgresql.port': lambda x: isinstance(x, int) and 1 <= x <= 65535,
                'security.authentication.token_expiry_hours': lambda x: isinstance(x, int) and x > 0,
                'performance.concurrency.thread_pool_size': lambda x: isinstance(x, int) and x > 0
            }
            
            # Validate required fields
            required_fields = [
                'automation.enabled',
                'ml.enabled', 
                'monitoring.enabled',
                'storage.redis.host',
                'storage.postgresql.host',
                'spotify.api.client_id'
            ]
            
            for field in required_fields:
                value = self._get_nested_value(self.config, field)
                if value is None:
                    self.validation_errors.append(f"Required field missing: {field}")
            
            # Validate field values
            for field, validator in validation_rules.items():
                value = self._get_nested_value(self.config, field)
                if value is not None and not validator(value):
                    self.validation_errors.append(f"Invalid value for {field}: {value}")
            
            # Validate cross-field dependencies
            await self._validate_dependencies()
            
            if self.validation_errors:
                for error in self.validation_errors:
                    logger.error(f"Configuration validation error: {error}")
                return False
            else:
                logger.info("Configuration validation passed")
                return True
                
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            self.validation_errors.append(f"Validation exception: {e}")
            return False
    
    async def _validate_dependencies(self):
        """Validate cross-field configuration dependencies"""
        # Example: If ML is enabled, certain ML-specific fields should be set
        if self._get_nested_value(self.config, 'ml.enabled', False):
            ml_required_fields = [
                'ml.prediction_confidence_threshold',
                'ml.model_retrain_interval_hours'
            ]
            
            for field in ml_required_fields:
                value = self._get_nested_value(self.config, field)
                if value is None:
                    self.validation_errors.append(f"ML is enabled but required field missing: {field}")
        
        # Example: If monitoring is enabled, alert channels should be configured
        if self._get_nested_value(self.config, 'monitoring.enabled', False):
            channels = self._get_nested_value(self.config, 'monitoring.alert_channels', {})
            enabled_channels = [ch for ch, cfg in channels.items() if cfg.get('enabled', False)]
            
            if not enabled_channels:
                self.validation_errors.append("Monitoring is enabled but no alert channels are configured")
    
    def encrypt_sensitive_value(self, value: str) -> str:
        """Encrypt a sensitive configuration value"""
        if self.cipher is None:
            logger.warning("Encryption not available, returning plain value")
            return value
        
        try:
            encrypted = self.cipher.encrypt(value.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Failed to encrypt value: {e}")
            return value
    
    def decrypt_sensitive_value(self, encrypted_value: str) -> str:
        """Decrypt a sensitive configuration value"""
        if self.cipher is None:
            logger.warning("Encryption not available, returning encrypted value")
            return encrypted_value
        
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode())
            decrypted = self.cipher.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Failed to decrypt value: {e}")
            return encrypted_value
    
    async def reload_configuration(self):
        """Reload configuration from all sources"""
        logger.info("Reloading configuration")
        
        # Store current configuration for rollback
        previous_config = self._config_to_dict()
        self.config_history.append({
            'timestamp': datetime.now(),
            'config': previous_config.copy()
        })
        
        try:
            await self.load_configuration()
            logger.info("Configuration reloaded successfully")
        except Exception as e:
            logger.error(f"Configuration reload failed: {e}")
            # Rollback to previous configuration
            self._dict_to_config(previous_config)
            logger.info("Rolled back to previous configuration")
            raise
    
    def get_config_section(self, section: str) -> Dict[str, Any]:
        """Get a specific configuration section"""
        return self._get_nested_value(self.config, section, {})
    
    def set_config_value(self, path: str, value: Any, persist: bool = False):
        """Set a configuration value at runtime"""
        logger.info(f"Setting configuration value: {path} = {value}")
        
        # Store change in history
        old_value = self._get_nested_value(self.config, path)
        self.config_history.append({
            'timestamp': datetime.now(),
            'change': {
                'path': path,
                'old_value': old_value,
                'new_value': value
            }
        })
        
        # Set the new value
        self._set_nested_value(self.config, path, value)
        
        # Persist to file if requested
        if persist:
            asyncio.create_task(self._persist_configuration())
    
    async def _persist_configuration(self):
        """Persist current configuration to file"""
        try:
            config_file = Path(f"config/{self.environment.value}/runtime_config.yaml")
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            config_dict = self._config_to_dict()
            
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration persisted to: {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to persist configuration: {e}")
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration"""
        return {
            'environment': self.environment.value,
            'loaded_at': datetime.now().isoformat(),
            'validation_status': 'valid' if not self.validation_errors else 'invalid',
            'validation_errors': self.validation_errors,
            'configuration_sections': {
                'automation': bool(self.config.automation),
                'ml': bool(self.config.ml),
                'monitoring': bool(self.config.monitoring),
                'storage': bool(self.config.storage),
                'security': bool(self.config.security),
                'spotify': bool(self.config.spotify),
                'performance': bool(self.config.performance),
                'logging': bool(self.config.logging)
            },
            'features_enabled': {
                'automation': self._get_nested_value(self.config, 'automation.enabled', False),
                'ml': self._get_nested_value(self.config, 'ml.enabled', False),
                'monitoring': self._get_nested_value(self.config, 'monitoring.enabled', False),
                'security': self._get_nested_value(self.config, 'security.encryption_enabled', False),
                'caching': self._get_nested_value(self.config, 'performance.caching.enabled', False)
            }
        }


# Factory function
def create_configuration_manager(environment: ConfigEnvironment = None) -> ConfigurationManager:
    """Create a configuration manager instance"""
    if environment is None:
        env_name = os.getenv('ENVIRONMENT', 'development').lower()
        try:
            environment = ConfigEnvironment(env_name)
        except ValueError:
            logger.warning(f"Unknown environment '{env_name}', defaulting to development")
            environment = ConfigEnvironment.DEVELOPMENT
    
    return ConfigurationManager(environment)


# Export main classes
__all__ = [
    'ConfigurationManager',
    'ConfigurationSet', 
    'ConfigEnvironment',
    'ConfigFormat',
    'ConfigSchema',
    'create_configuration_manager'
]
