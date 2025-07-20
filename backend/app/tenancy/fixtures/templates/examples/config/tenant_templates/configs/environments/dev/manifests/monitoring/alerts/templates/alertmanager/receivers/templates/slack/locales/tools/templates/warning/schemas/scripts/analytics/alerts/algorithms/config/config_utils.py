"""
Configuration Management Utilities for Spotify AI Agent

This module provides utility functions and helper classes for configuration
management, including configuration merging, environment-specific loading,
validation helpers, and configuration transformation utilities.

Author: Fahed Mlaiel (Expert Backend Developer & ML Engineer)
Version: 2.0.0 (Enterprise Edition)
"""

import os
import json
import yaml
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import base64
from copy import deepcopy
import re

logger = logging.getLogger(__name__)


class ConfigFormat(Enum):
    """Supported configuration file formats"""
    YAML = "yaml"
    JSON = "json"
    TOML = "toml"
    INI = "ini"
    ENV = "env"


class EnvironmentType(Enum):
    """Environment types for configuration loading"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"
    LOCAL = "local"


@dataclass
class ConfigurationMetadata:
    """Metadata about configuration"""
    file_path: str
    format_type: ConfigFormat
    last_modified: datetime
    size_bytes: int
    checksum: str
    environment: EnvironmentType
    version: str = "1.0.0"
    description: str = ""
    tags: List[str] = field(default_factory=list)


class ConfigurationTransformer:
    """Transforms configurations between different formats and structures"""
    
    def __init__(self):
        self.transformation_rules: Dict[str, Callable] = {}
        self._register_default_transformations()
    
    def _register_default_transformations(self):
        """Register default transformation rules"""
        self.transformation_rules.update({
            'env_var_substitution': self._substitute_env_vars,
            'base64_decode': self._decode_base64_values,
            'json_parse': self._parse_json_strings,
            'type_conversion': self._convert_types,
            'path_normalization': self._normalize_paths,
            'duration_parsing': self._parse_durations,
            'size_parsing': self._parse_sizes,
            'boolean_normalization': self._normalize_booleans
        })
    
    def transform(self, config: Dict[str, Any], rules: List[str] = None) -> Dict[str, Any]:
        """Apply transformation rules to configuration"""
        if rules is None:
            rules = list(self.transformation_rules.keys())
        
        transformed_config = deepcopy(config)
        
        for rule_name in rules:
            if rule_name in self.transformation_rules:
                try:
                    transformed_config = self.transformation_rules[rule_name](transformed_config)
                    logger.debug(f"Applied transformation rule: {rule_name}")
                except Exception as e:
                    logger.error(f"Error applying transformation rule {rule_name}: {e}")
        
        return transformed_config
    
    def _substitute_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute environment variables in configuration values"""
        def substitute_recursive(obj):
            if isinstance(obj, dict):
                return {k: substitute_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute_recursive(item) for item in obj]
            elif isinstance(obj, str):
                # Look for ${VAR_NAME} or ${VAR_NAME:default_value} patterns
                pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
                
                def replace_env_var(match):
                    var_name = match.group(1)
                    default_value = match.group(2) if match.group(2) is not None else ""
                    return os.getenv(var_name, default_value)
                
                return re.sub(pattern, replace_env_var, obj)
            else:
                return obj
        
        return substitute_recursive(config)
    
    def _decode_base64_values(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Decode base64 encoded values"""
        def decode_recursive(obj, path=""):
            if isinstance(obj, dict):
                return {k: decode_recursive(v, f"{path}.{k}" if path else k) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [decode_recursive(item, f"{path}[{i}]") for i, item in enumerate(obj)]
            elif isinstance(obj, str):
                # Check if field name suggests base64 encoded content
                if any(keyword in path.lower() for keyword in ['secret', 'key', 'token', 'credential']):
                    if obj.startswith('base64:'):
                        try:
                            decoded = base64.b64decode(obj[7:]).decode('utf-8')
                            logger.debug(f"Decoded base64 value at {path}")
                            return decoded
                        except Exception as e:
                            logger.warning(f"Failed to decode base64 value at {path}: {e}")
                return obj
            else:
                return obj
        
        return decode_recursive(config)
    
    def _parse_json_strings(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Parse JSON strings into objects"""
        def parse_recursive(obj):
            if isinstance(obj, dict):
                return {k: parse_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [parse_recursive(item) for item in obj]
            elif isinstance(obj, str):
                # Try to parse as JSON if it looks like JSON
                if obj.strip().startswith(('{', '[')):
                    try:
                        return json.loads(obj)
                    except json.JSONDecodeError:
                        pass
                return obj
            else:
                return obj
        
        return parse_recursive(config)
    
    def _convert_types(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert string values to appropriate types"""
        def convert_recursive(obj):
            if isinstance(obj, dict):
                return {k: convert_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_recursive(item) for item in obj]
            elif isinstance(obj, str):
                # Try to convert to appropriate type
                if obj.lower() in ('true', 'false'):
                    return obj.lower() == 'true'
                elif obj.lower() in ('null', 'none'):
                    return None
                elif obj.isdigit():
                    return int(obj)
                elif re.match(r'^\d+\.\d+$', obj):
                    return float(obj)
                return obj
            else:
                return obj
        
        return convert_recursive(config)
    
    def _normalize_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize file paths"""
        def normalize_recursive(obj, path=""):
            if isinstance(obj, dict):
                return {k: normalize_recursive(v, f"{path}.{k}" if path else k) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [normalize_recursive(item, f"{path}[{i}]") for i, item in enumerate(obj)]
            elif isinstance(obj, str):
                # Normalize paths for path-like fields
                if any(keyword in path.lower() for keyword in ['path', 'dir', 'file', 'location']):
                    return str(Path(obj).resolve())
                return obj
            else:
                return obj
        
        return normalize_recursive(config)
    
    def _parse_durations(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Parse duration strings into seconds"""
        def parse_duration(duration_str: str) -> int:
            """Parse duration string like '1h', '30m', '45s' into seconds"""
            if isinstance(duration_str, int):
                return duration_str
            
            if not isinstance(duration_str, str):
                return duration_str
            
            duration_str = duration_str.strip().lower()
            
            # Duration patterns
            patterns = {
                r'(\d+)s': 1,           # seconds
                r'(\d+)m': 60,          # minutes
                r'(\d+)h': 3600,        # hours
                r'(\d+)d': 86400,       # days
                r'(\d+)w': 604800,      # weeks
            }
            
            total_seconds = 0
            for pattern, multiplier in patterns.items():
                matches = re.findall(pattern, duration_str)
                for match in matches:
                    total_seconds += int(match) * multiplier
            
            return total_seconds if total_seconds > 0 else duration_str
        
        def parse_recursive(obj, path=""):
            if isinstance(obj, dict):
                return {k: parse_recursive(v, f"{path}.{k}" if path else k) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [parse_recursive(item, f"{path}[{i}]") for i, item in enumerate(obj)]
            elif isinstance(obj, str):
                # Parse duration for time-related fields
                if any(keyword in path.lower() for keyword in ['timeout', 'duration', 'interval', 'ttl', 'expiry']):
                    return parse_duration(obj)
                return obj
            else:
                return obj
        
        return parse_recursive(config)
    
    def _parse_sizes(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Parse size strings into bytes"""
        def parse_size(size_str: str) -> int:
            """Parse size string like '1GB', '512MB', '1024KB' into bytes"""
            if isinstance(size_str, int):
                return size_str
            
            if not isinstance(size_str, str):
                return size_str
            
            size_str = size_str.strip().upper()
            
            # Size patterns
            patterns = {
                r'(\d+)B': 1,                    # bytes
                r'(\d+)KB': 1024,                # kilobytes
                r'(\d+)MB': 1024 ** 2,           # megabytes
                r'(\d+)GB': 1024 ** 3,           # gigabytes
                r'(\d+)TB': 1024 ** 4,           # terabytes
            }
            
            for pattern, multiplier in patterns.items():
                match = re.match(pattern, size_str)
                if match:
                    return int(match.group(1)) * multiplier
            
            return size_str
        
        def parse_recursive(obj, path=""):
            if isinstance(obj, dict):
                return {k: parse_recursive(v, f"{path}.{k}" if path else k) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [parse_recursive(item, f"{path}[{i}]") for i, item in enumerate(obj)]
            elif isinstance(obj, str):
                # Parse size for size-related fields
                if any(keyword in path.lower() for keyword in ['size', 'limit', 'capacity', 'memory', 'storage']):
                    return parse_size(obj)
                return obj
            else:
                return obj
        
        return parse_recursive(config)
    
    def _normalize_booleans(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize boolean values"""
        def normalize_recursive(obj):
            if isinstance(obj, dict):
                return {k: normalize_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [normalize_recursive(item) for item in obj]
            elif isinstance(obj, str):
                lower_val = obj.lower()
                if lower_val in ('true', 'yes', 'on', '1', 'enabled'):
                    return True
                elif lower_val in ('false', 'no', 'off', '0', 'disabled'):
                    return False
                return obj
            else:
                return obj
        
        return normalize_recursive(config)


class ConfigurationMerger:
    """Merges multiple configuration dictionaries with conflict resolution"""
    
    def __init__(self, merge_strategy: str = "deep_merge"):
        self.merge_strategy = merge_strategy
        self.conflict_resolver = self._default_conflict_resolver
    
    def merge(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configuration dictionaries"""
        if not configs:
            return {}
        
        if len(configs) == 1:
            return deepcopy(configs[0])
        
        result = deepcopy(configs[0])
        
        for config in configs[1:]:
            result = self._merge_two_configs(result, config)
        
        return result
    
    def _merge_two_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries"""
        if self.merge_strategy == "deep_merge":
            return self._deep_merge(base, override)
        elif self.merge_strategy == "override":
            return self._override_merge(base, override)
        elif self.merge_strategy == "selective":
            return self._selective_merge(base, override)
        else:
            raise ValueError(f"Unknown merge strategy: {self.merge_strategy}")
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result:
                if isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = self._deep_merge(result[key], value)
                elif isinstance(result[key], list) and isinstance(value, list):
                    result[key] = self._merge_lists(result[key], value)
                else:
                    # Conflict detected, use resolver
                    result[key] = self.conflict_resolver(key, result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    def _override_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Override merge - override values take precedence"""
        result = deepcopy(base)
        result.update(deepcopy(override))
        return result
    
    def _selective_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Selective merge with specific rules for different key patterns"""
        result = deepcopy(base)
        
        for key, value in override.items():
            if key.endswith('_override'):
                # Override keys always take precedence
                result[key[:-9]] = deepcopy(value)
            elif key.endswith('_append'):
                # Append keys merge with existing lists
                base_key = key[:-7]
                if base_key in result and isinstance(result[base_key], list):
                    result[base_key].extend(value if isinstance(value, list) else [value])
                else:
                    result[base_key] = value if isinstance(value, list) else [value]
            else:
                # Standard deep merge
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = self._deep_merge(result[key], value)
                else:
                    result[key] = deepcopy(value)
        
        return result
    
    def _merge_lists(self, base_list: List[Any], override_list: List[Any]) -> List[Any]:
        """Merge two lists based on content type"""
        # If lists contain dictionaries with 'name' or 'id' fields, merge by key
        if (base_list and isinstance(base_list[0], dict) and 
            override_list and isinstance(override_list[0], dict)):
            
            # Try to find a key field
            key_field = None
            for field in ['name', 'id', 'key', 'type']:
                if field in base_list[0]:
                    key_field = field
                    break
            
            if key_field:
                # Merge by key field
                base_dict = {item[key_field]: item for item in base_list if key_field in item}
                for item in override_list:
                    if key_field in item:
                        if item[key_field] in base_dict:
                            # Merge existing item
                            base_dict[item[key_field]] = self._deep_merge(
                                base_dict[item[key_field]], item
                            )
                        else:
                            # Add new item
                            base_dict[item[key_field]] = item
                
                return list(base_dict.values())
        
        # Default: concatenate lists
        return base_list + override_list
    
    def _default_conflict_resolver(self, key: str, base_value: Any, override_value: Any) -> Any:
        """Default conflict resolution strategy"""
        # Override value takes precedence by default
        logger.debug(f"Conflict resolved for key '{key}': {base_value} -> {override_value}")
        return override_value


class ConfigurationLoader:
    """Loads configurations from various sources and formats"""
    
    def __init__(self, base_path: Optional[str] = None):
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.transformer = ConfigurationTransformer()
        self.merger = ConfigurationMerger()
        self._format_loaders = {
            ConfigFormat.YAML: self._load_yaml,
            ConfigFormat.JSON: self._load_json,
            ConfigFormat.TOML: self._load_toml,
            ConfigFormat.INI: self._load_ini,
            ConfigFormat.ENV: self._load_env
        }
    
    def load_config(self, config_path: str, format_type: ConfigFormat = None, 
                   transform: bool = True) -> Dict[str, Any]:
        """Load configuration from file"""
        config_file = Path(config_path)
        
        if not config_file.is_absolute():
            config_file = self.base_path / config_file
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        # Auto-detect format if not specified
        if format_type is None:
            format_type = self._detect_format(config_file)
        
        # Load configuration
        loader = self._format_loaders.get(format_type)
        if not loader:
            raise ValueError(f"Unsupported configuration format: {format_type}")
        
        config = loader(config_file)
        
        # Apply transformations
        if transform:
            config = self.transformer.transform(config)
        
        return config
    
    def load_environment_config(self, environment: EnvironmentType, 
                               config_name: str = "config") -> Dict[str, Any]:
        """Load environment-specific configuration"""
        configs = []
        
        # Load base configuration
        base_config_file = f"{config_name}.yaml"
        try:
            base_config = self.load_config(base_config_file)
            configs.append(base_config)
            logger.debug(f"Loaded base configuration: {base_config_file}")
        except FileNotFoundError:
            logger.warning(f"Base configuration file not found: {base_config_file}")
        
        # Load environment-specific configuration
        env_config_file = f"{config_name}_{environment.value}.yaml"
        try:
            env_config = self.load_config(env_config_file)
            configs.append(env_config)
            logger.debug(f"Loaded environment configuration: {env_config_file}")
        except FileNotFoundError:
            logger.warning(f"Environment configuration file not found: {env_config_file}")
        
        # Load local overrides
        local_config_file = f"{config_name}_local.yaml"
        try:
            local_config = self.load_config(local_config_file)
            configs.append(local_config)
            logger.debug(f"Loaded local configuration: {local_config_file}")
        except FileNotFoundError:
            logger.debug(f"No local configuration file found: {local_config_file}")
        
        if not configs:
            raise ValueError(f"No configuration files found for environment: {environment.value}")
        
        # Merge all configurations
        merged_config = self.merger.merge(*configs)
        
        # Set environment metadata
        merged_config['_metadata'] = {
            'environment': environment.value,
            'loaded_files': [str(self.base_path / f) for f in 
                           [base_config_file, env_config_file, local_config_file] 
                           if (self.base_path / f).exists()],
            'load_timestamp': datetime.now().isoformat()
        }
        
        return merged_config
    
    def _detect_format(self, config_file: Path) -> ConfigFormat:
        """Auto-detect configuration file format"""
        suffix = config_file.suffix.lower()
        
        format_map = {
            '.yaml': ConfigFormat.YAML,
            '.yml': ConfigFormat.YAML,
            '.json': ConfigFormat.JSON,
            '.toml': ConfigFormat.TOML,
            '.ini': ConfigFormat.INI,
            '.env': ConfigFormat.ENV
        }
        
        return format_map.get(suffix, ConfigFormat.YAML)
    
    def _load_yaml(self, config_file: Path) -> Dict[str, Any]:
        """Load YAML configuration file"""
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    
    def _load_json(self, config_file: Path) -> Dict[str, Any]:
        """Load JSON configuration file"""
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_toml(self, config_file: Path) -> Dict[str, Any]:
        """Load TOML configuration file"""
        try:
            import toml
            with open(config_file, 'r', encoding='utf-8') as f:
                return toml.load(f)
        except ImportError:
            raise ImportError("toml library required for TOML configuration files: pip install toml")
    
    def _load_ini(self, config_file: Path) -> Dict[str, Any]:
        """Load INI configuration file"""
        import configparser
        config = configparser.ConfigParser()
        config.read(config_file)
        
        # Convert to nested dictionary
        result = {}
        for section_name in config.sections():
            result[section_name] = dict(config[section_name])
        
        return result
    
    def _load_env(self, config_file: Path) -> Dict[str, Any]:
        """Load environment file"""
        config = {}
        
        with open(config_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        config[key.strip()] = value.strip().strip('"\'')
        
        return config


class ConfigurationGenerator:
    """Generates configuration templates and examples"""
    
    def __init__(self):
        self.templates = {
            'music_streaming_basic': self._music_streaming_basic_template,
            'music_streaming_enterprise': self._music_streaming_enterprise_template,
            'anomaly_detection': self._anomaly_detection_template,
            'monitoring': self._monitoring_template,
            'security': self._security_template
        }
    
    def generate_template(self, template_name: str, **kwargs) -> Dict[str, Any]:
        """Generate configuration template"""
        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        generator = self.templates[template_name]
        return generator(**kwargs)
    
    def _music_streaming_basic_template(self, **kwargs) -> Dict[str, Any]:
        """Generate basic music streaming configuration template"""
        return {
            'music_streaming': {
                'audio_quality': {
                    'bitrates': {
                        'premium': 320,
                        'high': 256,
                        'normal': 128,
                        'low': 96
                    },
                    'codecs': ['aac', 'mp3'],
                    'adaptive_streaming': True
                },
                'user_segments': {
                    'priority_levels': {
                        'premium': 1,
                        'family': 2,
                        'student': 3,
                        'free': 4
                    }
                }
            },
            'performance': {
                'cache_ttl': 300,
                'max_connections': 100,
                'timeout': 30
            }
        }
    
    def _music_streaming_enterprise_template(self, **kwargs) -> Dict[str, Any]:
        """Generate enterprise music streaming configuration template"""
        basic = self._music_streaming_basic_template(**kwargs)
        
        enterprise_features = {
            'music_streaming': {
                'recommendation_engine': {
                    'models': {
                        'collaborative_filtering': {
                            'embedding_dim': 128,
                            'learning_rate': 0.001
                        },
                        'content_based': {
                            'similarity_threshold': 0.75
                        }
                    }
                },
                'analytics': {
                    'real_time_analytics': True,
                    'user_behavior_tracking': True,
                    'revenue_optimization': True
                }
            },
            'security': {
                'encryption': {
                    'algorithm': 'AES-256-GCM',
                    'key_rotation_interval': '30d'
                },
                'authentication': {
                    'multi_factor_required': True,
                    'token_expiry_hours': 12
                }
            },
            'compliance': {
                'gdpr_enabled': True,
                'data_retention_days': 2555,
                'audit_logging': True
            }
        }
        
        merger = ConfigurationMerger()
        return merger.merge(basic, enterprise_features)
    
    def _anomaly_detection_template(self, **kwargs) -> Dict[str, Any]:
        """Generate anomaly detection configuration template"""
        return {
            'anomaly_detection': {
                'models': {
                    'isolation_forest': {
                        'contamination': 0.05,
                        'n_estimators': 200,
                        'random_state': 42
                    },
                    'autoencoder': {
                        'encoder_layers': [256, 128, 64],
                        'decoder_layers': [64, 128, 256],
                        'epochs': 100
                    }
                },
                'thresholds': {
                    'anomaly_score': 0.7,
                    'confidence_threshold': 0.8
                }
            }
        }
    
    def _monitoring_template(self, **kwargs) -> Dict[str, Any]:
        """Generate monitoring configuration template"""
        return {
            'monitoring': {
                'metrics': {
                    'collection_interval': 60,
                    'retention_days': 30
                },
                'alerting': {
                    'channels': ['email', 'slack'],
                    'thresholds': {
                        'critical': 0.95,
                        'warning': 0.8
                    }
                }
            }
        }
    
    def _security_template(self, **kwargs) -> Dict[str, Any]:
        """Generate security configuration template"""
        return {
            'security': {
                'encryption': {
                    'algorithm': 'AES-256-GCM',
                    'key_size': 256
                },
                'authentication': {
                    'token_expiry_hours': 24,
                    'refresh_token_expiry_days': 30
                },
                'network': {
                    'enforce_https': True,
                    'cors_enabled': True
                }
            }
        }


def calculate_config_checksum(config: Dict[str, Any]) -> str:
    """Calculate checksum for configuration"""
    config_str = json.dumps(config, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(config_str.encode('utf-8')).hexdigest()


def get_config_metadata(config_file_path: str) -> ConfigurationMetadata:
    """Get metadata for configuration file"""
    config_path = Path(config_file_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file_path}")
    
    # Detect format
    format_type = ConfigFormat.YAML
    suffix = config_path.suffix.lower()
    if suffix in ['.json']:
        format_type = ConfigFormat.JSON
    elif suffix in ['.toml']:
        format_type = ConfigFormat.TOML
    elif suffix in ['.ini']:
        format_type = ConfigFormat.INI
    elif suffix in ['.env']:
        format_type = ConfigFormat.ENV
    
    # Get file stats
    stat = config_path.stat()
    
    # Calculate checksum
    with open(config_path, 'rb') as f:
        content = f.read()
        checksum = hashlib.sha256(content).hexdigest()
    
    # Detect environment from filename
    environment = EnvironmentType.DEVELOPMENT
    if 'staging' in config_path.name:
        environment = EnvironmentType.STAGING
    elif 'production' in config_path.name or 'prod' in config_path.name:
        environment = EnvironmentType.PRODUCTION
    elif 'test' in config_path.name:
        environment = EnvironmentType.TESTING
    
    return ConfigurationMetadata(
        file_path=str(config_path),
        format_type=format_type,
        last_modified=datetime.fromtimestamp(stat.st_mtime),
        size_bytes=stat.st_size,
        checksum=checksum,
        environment=environment
    )


# Export all classes and functions
__all__ = [
    'ConfigFormat',
    'EnvironmentType',
    'ConfigurationMetadata',
    'ConfigurationTransformer',
    'ConfigurationMerger',
    'ConfigurationLoader',
    'ConfigurationGenerator',
    'calculate_config_checksum',
    'get_config_metadata'
]
