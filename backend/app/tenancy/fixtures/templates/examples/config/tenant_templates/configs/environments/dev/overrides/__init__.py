"""
Configuration Overrides Management System for Development Environment

This module provides an enterprise-grade configuration override system
for the development environment of the Spotify AI Agent platform.

Developed by Expert Team:
- Lead Dev + AI Architect
- Senior Backend Developer (Python/FastAPI/Django)
- ML Engineer (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Backend Security Specialist
- Microservices Architect
"""

import os
import yaml
import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
from functools import lru_cache
import aiofiles
from contextlib import asynccontextmanager

# Configuration constants
OVERRIDE_CACHE_TTL = 300  # 5 minutes
MAX_OVERRIDE_DEPTH = 10
SUPPORTED_FORMATS = ['yml', 'yaml', 'json']


class OverrideType(Enum):
    """Types of configuration overrides supported"""
    LOCAL = "local"
    DOCKER = "docker" 
    TESTING = "testing"
    CI_CD = "ci_cd"
    PROFILING = "profiling"
    DEBUGGING = "debugging"
    LOAD_TESTING = "load_testing"
    SECURITY_TESTING = "security_testing"


class OverridePriority(Enum):
    """Priority levels for configuration overrides"""
    LOWEST = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    HIGHEST = 4
    CRITICAL = 5


@dataclass
class OverrideMetadata:
    """Metadata for configuration overrides"""
    override_type: OverrideType
    priority: OverridePriority
    created_at: datetime
    last_modified: datetime
    checksum: str
    author: str
    description: str
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OverrideValidationResult:
    """Result of override validation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class OverrideValidationError(Exception):
    """Exception raised when override validation fails"""
    pass


class OverrideManager:
    """
    Enterprise-grade configuration override manager
    
    Features:
    - Hierarchical override system with priority management
    - Dynamic configuration loading with caching
    - Validation and integrity checks
    - Environment-specific conditional overrides
    - Performance monitoring and optimization
    - Security validation and compliance
    """
    
    def __init__(self, base_path: Path, cache_ttl: int = OVERRIDE_CACHE_TTL):
        self.base_path = Path(base_path)
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._override_metadata: Dict[str, OverrideMetadata] = {}
        self._validation_rules: Dict[str, Any] = {}
        self._logger = logging.getLogger(__name__)
        
        # Initialize override registry
        self._override_registry: Dict[OverrideType, Dict[str, Any]] = {
            override_type: {} for override_type in OverrideType
        }
        
        # Load validation rules
        self._load_validation_rules()
        
    async def initialize(self) -> None:
        """Initialize the override manager"""
        try:
            await self._scan_available_overrides()
            await self._validate_override_integrity()
            await self._build_dependency_graph()
            self._logger.info("Override manager initialized successfully")
        except Exception as e:
            self._logger.error(f"Failed to initialize override manager: {e}")
            raise
    
    async def _scan_available_overrides(self) -> None:
        """Scan and catalog all available override files"""
        override_files = []
        
        # Scan for override files
        for pattern in ['*.yml', '*.yaml', '*.json']:
            override_files.extend(self.base_path.glob(pattern))
        
        # Process each override file
        for file_path in override_files:
            try:
                override_type = self._determine_override_type(file_path)
                metadata = await self._extract_metadata(file_path)
                
                self._override_metadata[str(file_path)] = metadata
                self._override_registry[override_type][file_path.stem] = file_path
                
                self._logger.debug(f"Registered override: {file_path} as {override_type}")
                
            except Exception as e:
                self._logger.warning(f"Failed to process override file {file_path}: {e}")
    
    def _determine_override_type(self, file_path: Path) -> OverrideType:
        """Determine the type of override based on filename and content"""
        filename = file_path.stem.lower()
        
        type_mapping = {
            'local': OverrideType.LOCAL,
            'docker': OverrideType.DOCKER,
            'testing': OverrideType.TESTING,
            'test': OverrideType.TESTING,
            'ci': OverrideType.CI_CD,
            'cd': OverrideType.CI_CD,
            'cicd': OverrideType.CI_CD,
            'profile': OverrideType.PROFILING,
            'profiling': OverrideType.PROFILING,
            'debug': OverrideType.DEBUGGING,
            'debugging': OverrideType.DEBUGGING,
            'load': OverrideType.LOAD_TESTING,
            'loadtest': OverrideType.LOAD_TESTING,
            'security': OverrideType.SECURITY_TESTING,
            'sectest': OverrideType.SECURITY_TESTING
        }
        
        for key, override_type in type_mapping.items():
            if key in filename:
                return override_type
        
        # Default to local if no specific type found
        return OverrideType.LOCAL
    
    async def _extract_metadata(self, file_path: Path) -> OverrideMetadata:
        """Extract metadata from override file"""
        stat = file_path.stat()
        
        # Calculate file checksum
        async with aiofiles.open(file_path, 'rb') as f:
            content = await f.read()
            checksum = hashlib.sha256(content).hexdigest()
        
        # Load content to extract embedded metadata
        config_data = await self._load_file_content(file_path)
        embedded_metadata = config_data.get('_metadata', {})
        
        return OverrideMetadata(
            override_type=self._determine_override_type(file_path),
            priority=OverridePriority(embedded_metadata.get('priority', 2)),
            created_at=datetime.fromtimestamp(stat.st_ctime),
            last_modified=datetime.fromtimestamp(stat.st_mtime),
            checksum=checksum,
            author=embedded_metadata.get('author', 'unknown'),
            description=embedded_metadata.get('description', ''),
            tags=embedded_metadata.get('tags', []),
            dependencies=embedded_metadata.get('dependencies', []),
            conditions=embedded_metadata.get('conditions', {})
        )
    
    async def _load_file_content(self, file_path: Path) -> Dict[str, Any]:
        """Load and parse configuration file content"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            if file_path.suffix.lower() in ['.yml', '.yaml']:
                return yaml.safe_load(content) or {}
            elif file_path.suffix.lower() == '.json':
                return json.loads(content)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
                
        except Exception as e:
            self._logger.error(f"Failed to load {file_path}: {e}")
            return {}
    
    async def _validate_override_integrity(self) -> None:
        """Validate integrity and consistency of all overrides"""
        validation_errors = []
        
        for file_path, metadata in self._override_metadata.items():
            try:
                # Validate file still exists and hasn't been corrupted
                path_obj = Path(file_path)
                if not path_obj.exists():
                    validation_errors.append(f"Override file missing: {file_path}")
                    continue
                
                # Validate checksum
                async with aiofiles.open(path_obj, 'rb') as f:
                    content = await f.read()
                    current_checksum = hashlib.sha256(content).hexdigest()
                
                if current_checksum != metadata.checksum:
                    self._logger.warning(f"Checksum mismatch for {file_path}, updating metadata")
                    metadata.checksum = current_checksum
                    metadata.last_modified = datetime.now()
                
                # Validate override content
                validation_result = await self._validate_override_content(path_obj)
                if not validation_result.is_valid:
                    validation_errors.extend(validation_result.errors)
                
            except Exception as e:
                validation_errors.append(f"Validation error for {file_path}: {e}")
        
        if validation_errors:
            self._logger.error(f"Override validation errors: {validation_errors}")
            raise OverrideValidationError(f"Validation failed: {validation_errors}")
    
    async def _validate_override_content(self, file_path: Path) -> OverrideValidationResult:
        """Validate the content of an override file"""
        result = OverrideValidationResult(is_valid=True)
        
        try:
            content = await self._load_file_content(file_path)
            
            # Validate structure
            if not isinstance(content, dict):
                result.is_valid = False
                result.errors.append("Override content must be a dictionary")
                return result
            
            # Validate required sections
            required_sections = ['development_local', 'development_docker', 'development_testing']
            override_type = self._determine_override_type(file_path)
            
            if override_type == OverrideType.LOCAL and 'development_local' not in content:
                result.warnings.append("Missing 'development_local' section in local override")
            elif override_type == OverrideType.DOCKER and 'development_docker' not in content:
                result.warnings.append("Missing 'development_docker' section in docker override")
            elif override_type == OverrideType.TESTING and 'development_testing' not in content:
                result.warnings.append("Missing 'development_testing' section in testing override")
            
            # Validate security configurations
            await self._validate_security_config(content, result)
            
            # Validate database configurations
            await self._validate_database_config(content, result)
            
            # Validate API configurations
            await self._validate_api_config(content, result)
            
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Content validation error: {e}")
        
        return result
    
    async def _validate_security_config(self, content: Dict[str, Any], result: OverrideValidationResult) -> None:
        """Validate security-related configurations"""
        for section_name, section_content in content.items():
            if not isinstance(section_content, dict):
                continue
            
            # Check for exposed secrets
            security_config = section_content.get('security', {})
            if isinstance(security_config, dict):
                for key, value in security_config.items():
                    if isinstance(value, str) and ('password' in key.lower() or 'secret' in key.lower()):
                        if not value.startswith('${') and not value.startswith('{{'):
                            result.warnings.append(f"Potential hardcoded secret in {section_name}.security.{key}")
            
            # Check database passwords
            db_config = section_content.get('database', {})
            if isinstance(db_config, dict):
                for db_type, db_settings in db_config.items():
                    if isinstance(db_settings, dict):
                        password = db_settings.get('password', '')
                        if isinstance(password, str) and password and not password.startswith('${'):
                            result.warnings.append(f"Hardcoded database password in {section_name}.database.{db_type}")
    
    async def _validate_database_config(self, content: Dict[str, Any], result: OverrideValidationResult) -> None:
        """Validate database configurations"""
        for section_name, section_content in content.items():
            if not isinstance(section_content, dict):
                continue
            
            db_config = section_content.get('database', {})
            if not isinstance(db_config, dict):
                continue
            
            # Validate PostgreSQL config
            postgres_config = db_config.get('postgresql', {})
            if isinstance(postgres_config, dict):
                required_fields = ['host', 'port', 'database', 'username']
                for field in required_fields:
                    if field not in postgres_config:
                        result.warnings.append(f"Missing required PostgreSQL field '{field}' in {section_name}")
                
                # Validate pool settings
                pool_config = postgres_config.get('pool', {})
                if isinstance(pool_config, dict):
                    if pool_config.get('max_size', 0) <= pool_config.get('min_size', 0):
                        result.errors.append(f"Invalid pool configuration in {section_name}: max_size must be > min_size")
    
    async def _validate_api_config(self, content: Dict[str, Any], result: OverrideValidationResult) -> None:
        """Validate API configurations"""
        for section_name, section_content in content.items():
            if not isinstance(section_content, dict):
                continue
            
            api_config = section_content.get('application', {}).get('api', {})
            if not isinstance(api_config, dict):
                continue
            
            # Validate FastAPI config
            fastapi_config = api_config.get('fastapi', {})
            if isinstance(fastapi_config, dict):
                port = fastapi_config.get('port')
                if isinstance(port, int) and (port < 1024 or port > 65535):
                    result.warnings.append(f"Port {port} in {section_name} may cause permission issues")
                
                workers = fastapi_config.get('workers')
                if isinstance(workers, int) and workers > 8:
                    result.warnings.append(f"High worker count ({workers}) in {section_name} may cause resource issues")
    
    async def _build_dependency_graph(self) -> None:
        """Build dependency graph for override resolution order"""
        # Implementation for dependency resolution
        # This would build a graph of override dependencies and determine load order
        pass
    
    def _load_validation_rules(self) -> None:
        """Load validation rules for override configurations"""
        self._validation_rules = {
            'database': {
                'postgresql': {
                    'required_fields': ['host', 'port', 'database', 'username'],
                    'optional_fields': ['password', 'pool', 'features', 'ssl'],
                    'validation_patterns': {
                        'host': r'^[a-zA-Z0-9.-]+$',
                        'port': r'^[0-9]+$',
                        'database': r'^[a-zA-Z0-9_]+$'
                    }
                }
            },
            'api': {
                'fastapi': {
                    'required_fields': ['host', 'port'],
                    'optional_fields': ['workers', 'debug', 'auto_reload', 'cors'],
                    'port_range': (1024, 65535),
                    'max_workers': 16
                }
            }
        }
    
    @lru_cache(maxsize=128)
    def _get_cache_key(self, override_type: OverrideType, context: str) -> str:
        """Generate cache key for override configuration"""
        return f"{override_type.value}:{context}:{hash(str(sorted(os.environ.items())))}"
    
    async def get_override_configuration(
        self, 
        override_type: OverrideType, 
        context: Optional[str] = None,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Get merged configuration for specified override type
        
        Args:
            override_type: Type of override configuration to retrieve
            context: Additional context for configuration resolution
            force_refresh: Force refresh of cached configuration
            
        Returns:
            Merged configuration dictionary
        """
        cache_key = self._get_cache_key(override_type, context or "default")
        
        # Check cache if not forcing refresh
        if not force_refresh and cache_key in self._cache:
            cache_time = self._cache_timestamps.get(cache_key)
            if cache_time and (datetime.now() - cache_time).seconds < self.cache_ttl:
                return self._cache[cache_key]
        
        # Load and merge configurations
        merged_config = await self._merge_override_configurations(override_type, context)
        
        # Cache the result
        self._cache[cache_key] = merged_config
        self._cache_timestamps[cache_key] = datetime.now()
        
        return merged_config
    
    async def _merge_override_configurations(
        self, 
        override_type: OverrideType, 
        context: Optional[str]
    ) -> Dict[str, Any]:
        """Merge multiple override configurations based on priority"""
        base_config = {}
        
        # Get override files for the specified type
        override_files = self._override_registry.get(override_type, {})
        
        # Sort by priority
        sorted_overrides = sorted(
            override_files.items(),
            key=lambda x: self._override_metadata.get(str(x[1]), OverrideMetadata(
                override_type=override_type,
                priority=OverridePriority.MEDIUM,
                created_at=datetime.now(),
                last_modified=datetime.now(),
                checksum="",
                author="",
                description=""
            )).priority.value
        )
        
        # Merge configurations in priority order
        for name, file_path in sorted_overrides:
            try:
                file_config = await self._load_file_content(file_path)
                
                # Apply conditional logic
                if await self._should_apply_override(file_path, context):
                    base_config = self._deep_merge(base_config, file_config)
                    
            except Exception as e:
                self._logger.error(f"Failed to merge override {file_path}: {e}")
        
        # Apply environment variable substitutions
        resolved_config = await self._resolve_environment_variables(base_config)
        
        return resolved_config
    
    async def _should_apply_override(self, file_path: Path, context: Optional[str]) -> bool:
        """Determine if an override should be applied based on conditions"""
        metadata = self._override_metadata.get(str(file_path))
        if not metadata or not metadata.conditions:
            return True
        
        conditions = metadata.conditions
        
        # Check environment conditions
        if 'environment' in conditions:
            env_conditions = conditions['environment']
            for env_var, expected_value in env_conditions.items():
                actual_value = os.environ.get(env_var)
                if actual_value != expected_value:
                    return False
        
        # Check context conditions
        if 'context' in conditions and context:
            context_conditions = conditions['context']
            if isinstance(context_conditions, list):
                if context not in context_conditions:
                    return False
            elif isinstance(context_conditions, str):
                if context != context_conditions:
                    return False
        
        # Check time-based conditions
        if 'time_range' in conditions:
            time_range = conditions['time_range']
            current_time = datetime.now().time()
            start_time = datetime.strptime(time_range.get('start', '00:00'), '%H:%M').time()
            end_time = datetime.strptime(time_range.get('end', '23:59'), '%H:%M').time()
            
            if not (start_time <= current_time <= end_time):
                return False
        
        return True
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two configuration dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    async def _resolve_environment_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve environment variable references in configuration"""
        def resolve_value(value):
            if isinstance(value, str):
                # Handle ${VAR_NAME:-default_value} syntax
                if value.startswith('${') and value.endswith('}'):
                    var_expr = value[2:-1]
                    if ':-' in var_expr:
                        var_name, default_value = var_expr.split(':-', 1)
                        return os.environ.get(var_name, default_value)
                    else:
                        return os.environ.get(var_expr, value)
                # Handle {{VAR_NAME}} syntax
                elif value.startswith('{{') and value.endswith('}}'):
                    var_name = value[2:-2]
                    return os.environ.get(var_name, value)
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(item) for item in value]
            
            return value
        
        return resolve_value(config)
    
    async def create_override(
        self,
        override_type: OverrideType,
        name: str,
        configuration: Dict[str, Any],
        metadata: Optional[OverrideMetadata] = None
    ) -> Path:
        """Create a new override configuration file"""
        file_name = f"{name}.yml"
        file_path = self.base_path / file_name
        
        # Add metadata to configuration
        if metadata:
            configuration['_metadata'] = {
                'priority': metadata.priority.value,
                'author': metadata.author,
                'description': metadata.description,
                'tags': metadata.tags,
                'dependencies': metadata.dependencies,
                'conditions': metadata.conditions
            }
        
        # Write configuration file
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            yaml_content = yaml.dump(configuration, default_flow_style=False, indent=2)
            await f.write(yaml_content)
        
        # Update registry
        self._override_registry[override_type][name] = file_path
        
        # Extract and store metadata
        extracted_metadata = await self._extract_metadata(file_path)
        self._override_metadata[str(file_path)] = extracted_metadata
        
        self._logger.info(f"Created override configuration: {file_path}")
        return file_path
    
    async def update_override(
        self,
        file_path: Path,
        configuration: Dict[str, Any],
        merge_mode: bool = True
    ) -> None:
        """Update an existing override configuration"""
        if not file_path.exists():
            raise FileNotFoundError(f"Override file not found: {file_path}")
        
        if merge_mode:
            # Load existing configuration
            existing_config = await self._load_file_content(file_path)
            
            # Merge with new configuration
            merged_config = self._deep_merge(existing_config, configuration)
        else:
            merged_config = configuration
        
        # Write updated configuration
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            yaml_content = yaml.dump(merged_config, default_flow_style=False, indent=2)
            await f.write(yaml_content)
        
        # Update metadata
        updated_metadata = await self._extract_metadata(file_path)
        self._override_metadata[str(file_path)] = updated_metadata
        
        # Clear cache
        self._cache.clear()
        self._cache_timestamps.clear()
        
        self._logger.info(f"Updated override configuration: {file_path}")
    
    async def delete_override(self, file_path: Path) -> None:
        """Delete an override configuration file"""
        if file_path.exists():
            file_path.unlink()
        
        # Remove from registry and metadata
        str_path = str(file_path)
        if str_path in self._override_metadata:
            del self._override_metadata[str_path]
        
        # Remove from registry
        for override_type, files in self._override_registry.items():
            to_remove = [name for name, path in files.items() if path == file_path]
            for name in to_remove:
                del files[name]
        
        # Clear cache
        self._cache.clear()
        self._cache_timestamps.clear()
        
        self._logger.info(f"Deleted override configuration: {file_path}")
    
    async def get_override_metadata(self, file_path: Path) -> Optional[OverrideMetadata]:
        """Get metadata for a specific override file"""
        return self._override_metadata.get(str(file_path))
    
    async def list_overrides(
        self, 
        override_type: Optional[OverrideType] = None
    ) -> Dict[str, OverrideMetadata]:
        """List all available overrides with their metadata"""
        result = {}
        
        for file_path, metadata in self._override_metadata.items():
            if override_type is None or metadata.override_type == override_type:
                result[file_path] = metadata
        
        return result
    
    async def validate_all_overrides(self) -> Dict[str, OverrideValidationResult]:
        """Validate all override configurations"""
        results = {}
        
        for file_path in self._override_metadata.keys():
            path_obj = Path(file_path)
            if path_obj.exists():
                results[file_path] = await self._validate_override_content(path_obj)
        
        return results
    
    async def export_overrides(self, output_path: Path, format: str = 'yaml') -> None:
        """Export all override configurations to a single file"""
        all_overrides = {}
        
        for override_type in OverrideType:
            overrides = self._override_registry.get(override_type, {})
            for name, file_path in overrides.items():
                config = await self._load_file_content(file_path)
                all_overrides[f"{override_type.value}_{name}"] = config
        
        # Write exported configuration
        async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
            if format.lower() in ['yaml', 'yml']:
                content = yaml.dump(all_overrides, default_flow_style=False, indent=2)
            elif format.lower() == 'json':
                content = json.dumps(all_overrides, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            await f.write(content)
        
        self._logger.info(f"Exported all overrides to: {output_path}")
    
    async def import_overrides(self, input_path: Path, merge_mode: bool = True) -> None:
        """Import override configurations from a file"""
        imported_config = await self._load_file_content(input_path)
        
        for override_name, config in imported_config.items():
            # Parse override type from name
            type_str = override_name.split('_')[0]
            try:
                override_type = OverrideType(type_str)
            except ValueError:
                self._logger.warning(f"Unknown override type in import: {type_str}")
                continue
            
            # Create or update override
            name = '_'.join(override_name.split('_')[1:])
            file_path = self.base_path / f"{name}.yml"
            
            if file_path.exists() and merge_mode:
                await self.update_override(file_path, config, merge_mode=True)
            else:
                await self.create_override(override_type, name, config)
        
        self._logger.info(f"Imported overrides from: {input_path}")
    
    @asynccontextmanager
    async def temporary_override(
        self,
        override_type: OverrideType,
        configuration: Dict[str, Any],
        context: Optional[str] = None
    ):
        """Context manager for temporary configuration overrides"""
        temp_name = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        temp_file = None
        
        try:
            # Create temporary override
            temp_file = await self.create_override(override_type, temp_name, configuration)
            
            # Yield the temporary configuration
            yield await self.get_override_configuration(override_type, context, force_refresh=True)
            
        finally:
            # Clean up temporary override
            if temp_file and temp_file.exists():
                await self.delete_override(temp_file)
    
    async def get_configuration_diff(
        self,
        override_type1: OverrideType,
        override_type2: OverrideType,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get the difference between two override configurations"""
        config1 = await self.get_override_configuration(override_type1, context)
        config2 = await self.get_override_configuration(override_type2, context)
        
        def compute_diff(dict1, dict2, path=""):
            diff = {}
            
            # Keys only in dict1
            for key in dict1.keys() - dict2.keys():
                diff[f"{path}.{key}" if path else key] = {
                    'type': 'removed',
                    'value': dict1[key]
                }
            
            # Keys only in dict2
            for key in dict2.keys() - dict1.keys():
                diff[f"{path}.{key}" if path else key] = {
                    'type': 'added',
                    'value': dict2[key]
                }
            
            # Keys in both
            for key in dict1.keys() & dict2.keys():
                full_key = f"{path}.{key}" if path else key
                
                if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                    nested_diff = compute_diff(dict1[key], dict2[key], full_key)
                    diff.update(nested_diff)
                elif dict1[key] != dict2[key]:
                    diff[full_key] = {
                        'type': 'modified',
                        'old_value': dict1[key],
                        'new_value': dict2[key]
                    }
            
            return diff
        
        return compute_diff(config1, config2)
    
    async def cleanup_cache(self) -> None:
        """Clean up expired cache entries"""
        current_time = datetime.now()
        expired_keys = []
        
        for key, timestamp in self._cache_timestamps.items():
            if (current_time - timestamp).seconds > self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
            del self._cache_timestamps[key]
        
        self._logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the override manager"""
        return {
            'cache_size': len(self._cache),
            'cache_hit_ratio': self._calculate_cache_hit_ratio(),
            'total_overrides': len(self._override_metadata),
            'validation_errors': await self._count_validation_errors(),
            'last_validation': max(
                (metadata.last_modified for metadata in self._override_metadata.values()),
                default=datetime.min
            ).isoformat()
        }
    
    def _calculate_cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio"""
        # This would be implemented with actual cache hit/miss tracking
        return 0.95  # Placeholder
    
    async def _count_validation_errors(self) -> int:
        """Count total validation errors across all overrides"""
        validation_results = await self.validate_all_overrides()
        return sum(len(result.errors) for result in validation_results.values())


# Global override manager instance
_override_manager: Optional[OverrideManager] = None


async def get_override_manager(base_path: Optional[Path] = None) -> OverrideManager:
    """Get or create the global override manager instance"""
    global _override_manager
    
    if _override_manager is None:
        if base_path is None:
            base_path = Path(__file__).parent
        
        _override_manager = OverrideManager(base_path)
        await _override_manager.initialize()
    
    return _override_manager


async def get_development_override(
    override_type: OverrideType,
    context: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function to get development override configuration"""
    manager = await get_override_manager()
    return await manager.get_override_configuration(override_type, context)


async def create_development_override(
    override_type: OverrideType,
    name: str,
    configuration: Dict[str, Any],
    author: str = "Development Team",
    description: str = ""
) -> Path:
    """Convenience function to create development override configuration"""
    manager = await get_override_manager()
    
    metadata = OverrideMetadata(
        override_type=override_type,
        priority=OverridePriority.MEDIUM,
        created_at=datetime.now(),
        last_modified=datetime.now(),
        checksum="",
        author=author,
        description=description
    )
    
    return await manager.create_override(override_type, name, configuration, metadata)
