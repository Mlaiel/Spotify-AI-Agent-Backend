"""
Enterprise Configuration Management Package for Spotify AI Agent Alert Algorithms

This package provides comprehensive configuration management for enterprise-grade
alert algorithms, including environment-specific settings, model parameters,
performance tuning, and operational configurations for music streaming platforms.

Author: Fahed Mlaiel (Expert Backend Developer & ML Engineer)
Version: 2.0.0 (Enterprise Edition)
"""

import os
import yaml
import json
import logging
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime
import threading
from functools import lru_cache

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Supported environments for configuration"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class ConfigFormat(Enum):
    """Supported configuration file formats"""
    YAML = "yaml"
    JSON = "json"
    TOML = "toml"


@dataclass
class ConfigMetadata:
    """Metadata for configuration tracking"""
    environment: Environment
    version: str
    last_updated: datetime
    loaded_at: datetime = field(default_factory=datetime.now)
    checksum: Optional[str] = None
    validation_status: bool = True
    source_file: Optional[str] = None


@dataclass
class PerformanceThresholds:
    """Performance thresholds for different environments"""
    max_latency_ms: int = 100
    min_accuracy: float = 0.95
    memory_limit_mb: int = 4096
    cpu_usage_limit_percent: int = 80
    throughput_requests_per_second: int = 1000
    cache_hit_rate_minimum: float = 0.85


@dataclass
class MusicStreamingConfig:
    """Music streaming platform specific configuration"""
    supported_quality_tiers: List[str] = field(default_factory=lambda: ['free', 'premium', 'family', 'student'])
    supported_audio_formats: List[str] = field(default_factory=lambda: ['mp3', 'flac', 'ogg', 'aac'])
    geographic_regions: List[str] = field(default_factory=lambda: ['US', 'EU', 'APAC', 'LATAM', 'MEA'])
    max_concurrent_streams_per_user: Dict[str, int] = field(default_factory=lambda: {
        'free': 1, 'premium': 1, 'family': 6, 'student': 1
    })
    bitrate_limits_kbps: Dict[str, int] = field(default_factory=lambda: {
        'free': 160, 'premium': 320, 'family': 320, 'student': 320
    })
    offline_download_limits: Dict[str, int] = field(default_factory=lambda: {
        'free': 0, 'premium': 10000, 'family': 10000, 'student': 10000
    })


@dataclass
class AlgorithmConfig:
    """Complete algorithm configuration"""
    environment: Environment
    performance_thresholds: PerformanceThresholds
    music_streaming: MusicStreamingConfig
    model_parameters: Dict[str, Any] = field(default_factory=dict)
    monitoring_config: Dict[str, Any] = field(default_factory=dict)
    security_config: Dict[str, Any] = field(default_factory=dict)
    cache_config: Dict[str, Any] = field(default_factory=dict)
    database_config: Dict[str, Any] = field(default_factory=dict)
    metadata: ConfigMetadata = field(default_factory=lambda: ConfigMetadata(
        environment=Environment.DEVELOPMENT,
        version="1.0.0",
        last_updated=datetime.now()
    ))


class ConfigurationManager:
    """Enterprise configuration manager with validation, caching, and hot reloading"""
    
    def __init__(self, environment: Environment = Environment.DEVELOPMENT, config_dir: Optional[str] = None):
        self.environment = environment
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent
        self._configs: Dict[str, AlgorithmConfig] = {}
        self._config_lock = threading.RLock()
        self._file_watchers: Dict[str, float] = {}  # file -> last_modified
        self._validation_rules: List[Callable[[Dict[str, Any]], bool]] = []
        
        # Initialize default validation rules
        self._setup_validation_rules()
        
        logger.info(f"Configuration manager initialized for environment: {environment.value}")
    
    def _setup_validation_rules(self):
        """Setup default validation rules for configuration"""
        
        def validate_performance_thresholds(config: Dict[str, Any]) -> bool:
            """Validate performance thresholds are reasonable"""
            if 'performance_thresholds' not in config:
                return True
            
            thresholds = config['performance_thresholds']
            if isinstance(thresholds, dict):
                max_latency = thresholds.get('max_latency_ms', 0)
                min_accuracy = thresholds.get('min_accuracy', 0)
                
                return 0 < max_latency <= 10000 and 0 <= min_accuracy <= 1.0
            return True
        
        def validate_music_streaming_config(config: Dict[str, Any]) -> bool:
            """Validate music streaming specific configuration"""
            if 'music_streaming' not in config:
                return True
            
            streaming = config['music_streaming']
            if isinstance(streaming, dict):
                bitrate_limits = streaming.get('bitrate_limits_kbps', {})
                if bitrate_limits:
                    # Validate bitrate limits are reasonable
                    return all(32 <= bitrate <= 2000 for bitrate in bitrate_limits.values())
            return True
        
        def validate_model_parameters(config: Dict[str, Any]) -> bool:
            """Validate model parameters"""
            if 'model_parameters' not in config:
                return True
            
            params = config['model_parameters']
            if isinstance(params, dict):
                # Validate contamination values for anomaly detection
                for model_config in params.values():
                    if isinstance(model_config, dict) and 'contamination' in model_config:
                        contamination = model_config['contamination']
                        if not (0 < contamination < 0.5):
                            return False
            return True
        
        self._validation_rules.extend([
            validate_performance_thresholds,
            validate_music_streaming_config,
            validate_model_parameters
        ])
    
    def add_validation_rule(self, rule: Callable[[Dict[str, Any]], bool]):
        """Add custom validation rule"""
        self._validation_rules.append(rule)
        logger.debug("Added custom validation rule")
    
    def load_config(self, config_name: Optional[str] = None) -> AlgorithmConfig:
        """Load configuration for current environment"""
        if config_name is None:
            config_name = f"algorithm_config_{self.environment.value}"
        
        with self._config_lock:
            if config_name in self._configs:
                # Check if file has been modified
                if self._should_reload_config(config_name):
                    logger.info(f"Configuration file modified, reloading: {config_name}")
                    del self._configs[config_name]
                else:
                    return self._configs[config_name]
            
            # Load configuration from file
            config_data = self._load_config_file(config_name)
            
            # Validate configuration
            if not self._validate_config(config_data):
                raise ValueError(f"Configuration validation failed for: {config_name}")
            
            # Create configuration object
            algorithm_config = self._create_config_object(config_data)
            
            # Cache configuration
            self._configs[config_name] = algorithm_config
            
            logger.info(f"Configuration loaded successfully: {config_name}")
            return algorithm_config
    
    def _load_config_file(self, config_name: str) -> Dict[str, Any]:
        """Load configuration from file"""
        # Try different file extensions
        for ext in ['yaml', 'yml', 'json']:
            config_file = self.config_dir / f"{config_name}.{ext}"
            if config_file.exists():
                # Update file watcher
                self._file_watchers[config_name] = config_file.stat().st_mtime
                
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        if ext in ['yaml', 'yml']:
                            return yaml.safe_load(f)
                        elif ext == 'json':
                            return json.load(f)
                except Exception as e:
                    logger.error(f"Error loading configuration file {config_file}: {e}")
                    raise
        
        raise FileNotFoundError(f"Configuration file not found: {config_name}")
    
    def _should_reload_config(self, config_name: str) -> bool:
        """Check if configuration file should be reloaded"""
        if config_name not in self._file_watchers:
            return True
        
        # Find the configuration file
        for ext in ['yaml', 'yml', 'json']:
            config_file = self.config_dir / f"{config_name}.{ext}"
            if config_file.exists():
                current_mtime = config_file.stat().st_mtime
                last_mtime = self._file_watchers.get(config_name, 0)
                return current_mtime > last_mtime
        
        return False
    
    def _validate_config(self, config_data: Dict[str, Any]) -> bool:
        """Validate configuration using all validation rules"""
        for rule in self._validation_rules:
            try:
                if not rule(config_data):
                    logger.error(f"Configuration validation failed for rule: {rule.__name__}")
                    return False
            except Exception as e:
                logger.error(f"Error in validation rule {rule.__name__}: {e}")
                return False
        
        return True
    
    def _create_config_object(self, config_data: Dict[str, Any]) -> AlgorithmConfig:
        """Create AlgorithmConfig object from configuration data"""
        
        # Extract performance thresholds
        perf_data = config_data.get('performance_thresholds', {})
        performance_thresholds = PerformanceThresholds(
            max_latency_ms=perf_data.get('max_latency_ms', 100),
            min_accuracy=perf_data.get('min_accuracy', 0.95),
            memory_limit_mb=perf_data.get('memory_limit_mb', 4096),
            cpu_usage_limit_percent=perf_data.get('cpu_usage_limit_percent', 80),
            throughput_requests_per_second=perf_data.get('throughput_requests_per_second', 1000),
            cache_hit_rate_minimum=perf_data.get('cache_hit_rate_minimum', 0.85)
        )
        
        # Extract music streaming configuration
        music_data = config_data.get('music_streaming', {})
        music_streaming = MusicStreamingConfig(
            supported_quality_tiers=music_data.get('supported_quality_tiers', ['free', 'premium', 'family', 'student']),
            supported_audio_formats=music_data.get('supported_audio_formats', ['mp3', 'flac', 'ogg', 'aac']),
            geographic_regions=music_data.get('geographic_regions', ['US', 'EU', 'APAC', 'LATAM', 'MEA']),
            max_concurrent_streams_per_user=music_data.get('max_concurrent_streams_per_user', {
                'free': 1, 'premium': 1, 'family': 6, 'student': 1
            }),
            bitrate_limits_kbps=music_data.get('bitrate_limits_kbps', {
                'free': 160, 'premium': 320, 'family': 320, 'student': 320
            }),
            offline_download_limits=music_data.get('offline_download_limits', {
                'free': 0, 'premium': 10000, 'family': 10000, 'student': 10000
            })
        )
        
        # Create metadata
        metadata = ConfigMetadata(
            environment=self.environment,
            version=config_data.get('version', '1.0.0'),
            last_updated=datetime.fromisoformat(config_data.get('last_updated', datetime.now().isoformat())),
            validation_status=True,
            source_file=f"algorithm_config_{self.environment.value}"
        )
        
        return AlgorithmConfig(
            environment=self.environment,
            performance_thresholds=performance_thresholds,
            music_streaming=music_streaming,
            model_parameters=config_data.get('model_parameters', {}),
            monitoring_config=config_data.get('monitoring', {}),
            security_config=config_data.get('security', {}),
            cache_config=config_data.get('caching', {}),
            database_config=config_data.get('database', {}),
            metadata=metadata
        )
    
    def get_algorithm_config(self, algorithm_name: str) -> Dict[str, Any]:
        """Get configuration for specific algorithm"""
        config = self.load_config()
        return config.model_parameters.get(algorithm_name, {})
    
    def get_performance_config(self) -> PerformanceThresholds:
        """Get performance configuration"""
        config = self.load_config()
        return config.performance_thresholds
    
    def get_music_streaming_config(self) -> MusicStreamingConfig:
        """Get music streaming configuration"""
        config = self.load_config()
        return config.music_streaming
    
    def update_config(self, config_updates: Dict[str, Any], config_name: Optional[str] = None):
        """Update configuration dynamically"""
        if config_name is None:
            config_name = f"algorithm_config_{self.environment.value}"
        
        with self._config_lock:
            if config_name in self._configs:
                # Apply updates to cached config
                current_config = self._configs[config_name]
                
                # Update model parameters
                if 'model_parameters' in config_updates:
                    current_config.model_parameters.update(config_updates['model_parameters'])
                
                # Update monitoring config
                if 'monitoring' in config_updates:
                    current_config.monitoring_config.update(config_updates['monitoring'])
                
                # Update metadata
                current_config.metadata.last_updated = datetime.now()
                
                logger.info(f"Configuration updated dynamically: {config_name}")
    
    def reload_all_configs(self):
        """Reload all cached configurations"""
        with self._config_lock:
            config_names = list(self._configs.keys())
            self._configs.clear()
            self._file_watchers.clear()
            
            for config_name in config_names:
                try:
                    self.load_config(config_name)
                    logger.info(f"Reloaded configuration: {config_name}")
                except Exception as e:
                    logger.error(f"Failed to reload configuration {config_name}: {e}")
    
    def get_config_status(self) -> Dict[str, Any]:
        """Get status of all loaded configurations"""
        with self._config_lock:
            status = {
                'environment': self.environment.value,
                'config_dir': str(self.config_dir),
                'loaded_configs': len(self._configs),
                'configs': {}
            }
            
            for name, config in self._configs.items():
                status['configs'][name] = {
                    'version': config.metadata.version,
                    'last_updated': config.metadata.last_updated.isoformat(),
                    'loaded_at': config.metadata.loaded_at.isoformat(),
                    'validation_status': config.metadata.validation_status,
                    'environment': config.environment.value
                }
            
            return status


@lru_cache(maxsize=1)
def get_config_manager(environment: Environment = Environment.DEVELOPMENT) -> ConfigurationManager:
    """Get cached configuration manager instance"""
    return ConfigurationManager(environment)


def load_algorithm_config(environment: Environment = Environment.DEVELOPMENT) -> AlgorithmConfig:
    """Convenience function to load algorithm configuration"""
    manager = get_config_manager(environment)
    return manager.load_config()


def get_model_config(model_name: str, environment: Environment = Environment.DEVELOPMENT) -> Dict[str, Any]:
    """Convenience function to get model-specific configuration"""
    manager = get_config_manager(environment)
    return manager.get_algorithm_config(model_name)


# Package exports
__all__ = [
    'Environment',
    'ConfigFormat',
    'ConfigMetadata',
    'PerformanceThresholds',
    'MusicStreamingConfig',
    'AlgorithmConfig',
    'ConfigurationManager',
    'get_config_manager',
    'load_algorithm_config',
    'get_model_config'
]

# Package metadata
__version__ = "2.0.0"
__author__ = "Configuration Management Team"
__description__ = "Enterprise configuration management for Spotify AI Agent alert algorithms"

# Configuration file types supported
SUPPORTED_CONFIG_FORMATS = [
    "yaml", "yml", "json", "toml", "ini", "env"
]

# Environment configuration files mapping
ENVIRONMENT_CONFIG_FILES = {
    "development": [
        "algorithm_config_development.yaml",
        "model_params_dev.yaml", 
        "performance_dev.yaml",
        "security_dev.yaml"
    ],
    "staging": [
        "algorithm_config_staging.yaml",
        "model_params_staging.yaml",
        "performance_staging.yaml", 
        "security_staging.yaml"
    ],
    "production": [
        "algorithm_config_production.yaml",
        "model_params_prod.yaml",
        "performance_prod.yaml",
        "security_prod.yaml"
    ],
    "testing": [
        "algorithm_config_testing.yaml",
        "model_params_test.yaml"
    ]
}

# Default configuration structure
DEFAULT_CONFIG_STRUCTURE = {
    "algorithms": {},
    "models": {},
    "performance": {},
    "security": {},
    "monitoring": {},
    "data_pipeline": {},
    "api": {},
    "business_rules": {}
}

def get_config_directory() -> Path:
    """
    Get the configuration directory path.
    
    Returns:
        Path object pointing to the configuration directory
    """
    return Path(__file__).parent

def list_config_files(environment: str = None) -> List[str]:
    """
    List available configuration files.
    
    Args:
        environment: Optional environment filter
        
    Returns:
        List of configuration file names
    """
    config_dir = get_config_directory()
    
    if environment and environment in ENVIRONMENT_CONFIG_FILES:
        return ENVIRONMENT_CONFIG_FILES[environment]
    
    # Return all configuration files
    config_files = []
    for files in ENVIRONMENT_CONFIG_FILES.values():
        config_files.extend(files)
    
    return list(set(config_files))

def validate_config_structure(config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure against expected schema.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        required_sections = DEFAULT_CONFIG_STRUCTURE.keys()
        
        for section in required_sections:
            if section not in config:
                logger.warning(f"Missing configuration section: {section}")
                return False
                
        logger.info("Configuration structure validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        return False

def get_config_file_path(filename: str) -> Path:
    """
    Get full path to a configuration file.
    
    Args:
        filename: Name of the configuration file
        
    Returns:
        Path object pointing to the configuration file
    """
    return get_config_directory() / filename

# Export public functions
__all__ = [
    "get_config_directory",
    "list_config_files", 
    "validate_config_structure",
    "get_config_file_path",
    "SUPPORTED_CONFIG_FORMATS",
    "ENVIRONMENT_CONFIG_FILES",
    "DEFAULT_CONFIG_STRUCTURE"
]
