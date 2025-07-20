"""
Spotify AI Agent - Integration Factory & Dependency Injection
============================================================

Ultra-advanced factory pattern implementation for dynamic integration
instantiation, configuration management, and dependency injection.

This module provides sophisticated factory patterns for:
- Dynamic integration creation and configuration
- Dependency injection and service resolution
- Configuration validation and environment management
- Integration lifecycle management
- Plugin architecture and extensibility
- Multi-tenant integration provisioning
- Circuit breaker and retry policy configuration
- Health monitoring and metrics collection

Author: Expert Team - Lead Dev + AI Architect, Backend Senior, Microservices Architect
Version: 2.1.0
"""

import asyncio
import importlib
import inspect
from typing import Dict, List, Any, Optional, Type, Union, Callable, TypeVar, Generic
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import structlog
from datetime import datetime, timezone
import json
import yaml
import os
from pathlib import Path

from . import (
    BaseIntegration,
    IntegrationConfig,
    IntegrationType,
    IntegrationStatus,
    get_integration_registry
)

logger = structlog.get_logger(__name__)

T = TypeVar('T', bound=BaseIntegration)


class ConfigurationSource(Enum):
    """Sources for integration configuration."""
    ENVIRONMENT = "environment"
    FILE = "file"
    DATABASE = "database"
    VAULT = "vault"
    CONSUL = "consul"
    ETCD = "etcd"


class ProvisioningStrategy(Enum):
    """Strategies for integration provisioning."""
    EAGER = "eager"
    LAZY = "lazy"
    ON_DEMAND = "on_demand"
    SCHEDULED = "scheduled"


@dataclass
class IntegrationDependency:
    """Represents a dependency between integrations."""
    name: str
    type: IntegrationType
    required: bool = True
    version_constraint: Optional[str] = None
    initialization_order: int = 0


@dataclass
class FactoryConfig:
    """Configuration for the integration factory."""
    
    config_source: ConfigurationSource = ConfigurationSource.ENVIRONMENT
    config_path: Optional[str] = None
    provisioning_strategy: ProvisioningStrategy = ProvisioningStrategy.LAZY
    auto_discovery: bool = True
    hot_reload: bool = False
    validation_enabled: bool = True
    dependency_resolution: bool = True
    circuit_breaker_enabled: bool = True
    metrics_enabled: bool = True
    
    # Environment-specific settings
    environment: str = "development"
    debug_mode: bool = False
    
    # Security settings
    secrets_encryption: bool = True
    secrets_source: str = "environment"
    
    # Performance settings
    connection_pooling: bool = True
    async_initialization: bool = True
    batch_size: int = 10
    timeout: int = 60


class ConfigurationValidator:
    """Validates integration configurations."""
    
    def __init__(self):
        self.logger = logger.bind(component="config_validator")
        self.validation_rules: Dict[str, List[Callable]] = {}
    
    def add_validation_rule(self, integration_type: str, rule: Callable[[Dict[str, Any]], bool]):
        """Add a validation rule for an integration type."""
        if integration_type not in self.validation_rules:
            self.validation_rules[integration_type] = []
        self.validation_rules[integration_type].append(rule)
    
    def validate_config(self, integration_type: str, config: Dict[str, Any]) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Basic validation
        if not isinstance(config, dict):
            errors.append("Configuration must be a dictionary")
            return errors
        
        # Required fields validation
        required_fields = self._get_required_fields(integration_type)
        for field in required_fields:
            if field not in config:
                errors.append(f"Required field '{field}' is missing")
        
        # Type-specific validation rules
        if integration_type in self.validation_rules:
            for rule in self.validation_rules[integration_type]:
                try:
                    if not rule(config):
                        errors.append(f"Validation rule failed for {integration_type}")
                except Exception as e:
                    errors.append(f"Validation rule error: {str(e)}")
        
        return errors
    
    def _get_required_fields(self, integration_type: str) -> List[str]:
        """Get required fields for integration type."""
        common_fields = ["enabled"]
        
        type_specific_fields = {
            "spotify_api": ["client_id", "client_secret"],
            "aws_s3": ["region", "access_key_id", "secret_access_key"],
            "email_service": ["smtp_host", "smtp_port"],
            "database": ["host", "port", "database"]
        }
        
        return common_fields + type_specific_fields.get(integration_type, [])


class DependencyResolver:
    """Resolves dependencies between integrations."""
    
    def __init__(self):
        self.logger = logger.bind(component="dependency_resolver")
        self.dependency_graph: Dict[str, List[IntegrationDependency]] = {}
    
    def add_dependency(self, integration_name: str, dependency: IntegrationDependency):
        """Add a dependency for an integration."""
        if integration_name not in self.dependency_graph:
            self.dependency_graph[integration_name] = []
        self.dependency_graph[integration_name].append(dependency)
    
    def resolve_dependencies(self, integration_configs: Dict[str, IntegrationConfig]) -> List[str]:
        """Resolve dependencies and return initialization order."""
        
        # Build adjacency list
        graph = {}
        in_degree = {}
        
        for name in integration_configs:
            graph[name] = []
            in_degree[name] = 0
        
        # Add edges for dependencies
        for name, dependencies in self.dependency_graph.items():
            if name in integration_configs:
                for dep in dependencies:
                    if dep.name in integration_configs:
                        graph[dep.name].append(name)
                        in_degree[name] += 1
        
        # Topological sort using Kahn's algorithm
        queue = []
        for name, degree in in_degree.items():
            if degree == 0:
                queue.append(name)
        
        result = []
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for circular dependencies
        if len(result) != len(integration_configs):
            remaining = set(integration_configs.keys()) - set(result)
            self.logger.error(f"Circular dependency detected: {remaining}")
            raise ValueError(f"Circular dependency detected: {remaining}")
        
        return result


class ConfigurationLoader:
    """Loads integration configurations from various sources."""
    
    def __init__(self, factory_config: FactoryConfig):
        self.factory_config = factory_config
        self.logger = logger.bind(component="config_loader")
    
    async def load_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Load configurations from the configured source."""
        
        if self.factory_config.config_source == ConfigurationSource.ENVIRONMENT:
            return await self._load_from_environment()
        elif self.factory_config.config_source == ConfigurationSource.FILE:
            return await self._load_from_file()
        elif self.factory_config.config_source == ConfigurationSource.DATABASE:
            return await self._load_from_database()
        elif self.factory_config.config_source == ConfigurationSource.VAULT:
            return await self._load_from_vault()
        else:
            raise ValueError(f"Unsupported configuration source: {self.factory_config.config_source}")
    
    async def _load_from_environment(self) -> Dict[str, Dict[str, Any]]:
        """Load configurations from environment variables."""
        configs = {}
        
        # Look for environment variables with pattern: INTEGRATION_{NAME}_{SETTING}
        for key, value in os.environ.items():
            if key.startswith("INTEGRATION_"):
                parts = key.split("_", 2)
                if len(parts) >= 3:
                    integration_name = parts[1].lower()
                    setting_name = parts[2].lower()
                    
                    if integration_name not in configs:
                        configs[integration_name] = {}
                    
                    # Try to parse as JSON, fall back to string
                    try:
                        configs[integration_name][setting_name] = json.loads(value)
                    except json.JSONDecodeError:
                        configs[integration_name][setting_name] = value
        
        return configs
    
    async def _load_from_file(self) -> Dict[str, Dict[str, Any]]:
        """Load configurations from YAML/JSON file."""
        if not self.factory_config.config_path:
            raise ValueError("config_path must be specified for file-based configuration")
        
        config_path = Path(self.factory_config.config_path)
        
        if not config_path.exists():
            self.logger.warning(f"Configuration file not found: {config_path}")
            return {}
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            # Extract environment-specific configuration
            env_config = data.get(self.factory_config.environment, {})
            global_config = data.get('global', {})
            
            # Merge global and environment-specific configs
            merged_config = {}
            all_integrations = set(global_config.keys()) | set(env_config.keys())
            
            for integration in all_integrations:
                merged_config[integration] = {}
                merged_config[integration].update(global_config.get(integration, {}))
                merged_config[integration].update(env_config.get(integration, {}))
            
            return merged_config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration file: {str(e)}")
            raise
    
    async def _load_from_database(self) -> Dict[str, Dict[str, Any]]:
        """Load configurations from database."""
        # Implementation would depend on specific database schema
        # This is a placeholder for database-based configuration loading
        self.logger.info("Loading configurations from database")
        return {}
    
    async def _load_from_vault(self) -> Dict[str, Dict[str, Any]]:
        """Load configurations from HashiCorp Vault or similar secret management."""
        # Implementation would depend on specific vault system
        # This is a placeholder for vault-based configuration loading
        self.logger.info("Loading configurations from vault")
        return {}


class IntegrationDiscovery:
    """Discovers available integration classes."""
    
    def __init__(self):
        self.logger = logger.bind(component="integration_discovery")
        self.discovered_integrations: Dict[str, Type[BaseIntegration]] = {}
    
    async def discover_integrations(self, search_paths: List[str] = None) -> Dict[str, Type[BaseIntegration]]:
        """Discover integration classes in specified paths."""
        
        if search_paths is None:
            search_paths = [
                "integrations.external_apis",
                "integrations.cloud",
                "integrations.communication",
                "integrations.auth",
                "integrations.data_pipelines",
                "integrations.security",
                "integrations.monitoring"
            ]
        
        for search_path in search_paths:
            try:
                await self._discover_in_module(search_path)
            except Exception as e:
                self.logger.error(f"Failed to discover integrations in {search_path}: {str(e)}")
        
        self.logger.info(f"Discovered {len(self.discovered_integrations)} integrations")
        return self.discovered_integrations
    
    async def _discover_in_module(self, module_path: str):
        """Discover integrations in a specific module."""
        try:
            module = importlib.import_module(module_path)
            
            for name in dir(module):
                obj = getattr(module, name)
                
                if (inspect.isclass(obj) and 
                    issubclass(obj, BaseIntegration) and 
                    obj != BaseIntegration):
                    
                    integration_name = self._get_integration_name(obj)
                    self.discovered_integrations[integration_name] = obj
                    
                    self.logger.debug(f"Discovered integration: {integration_name} ({obj.__name__})")
        
        except ImportError as e:
            self.logger.warning(f"Could not import module {module_path}: {str(e)}")
    
    def _get_integration_name(self, integration_class: Type[BaseIntegration]) -> str:
        """Extract integration name from class."""
        # Remove 'Integration' suffix if present
        name = integration_class.__name__
        if name.endswith('Integration'):
            name = name[:-11]
        
        # Convert CamelCase to snake_case
        import re
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
        
        return name


class IntegrationFactory:
    """Factory for creating and managing integrations."""
    
    def __init__(self, factory_config: FactoryConfig = None):
        self.factory_config = factory_config or FactoryConfig()
        self.logger = logger.bind(component="integration_factory")
        
        # Components
        self.config_loader = ConfigurationLoader(self.factory_config)
        self.validator = ConfigurationValidator()
        self.dependency_resolver = DependencyResolver()
        self.discovery = IntegrationDiscovery()
        
        # State
        self.integration_classes: Dict[str, Type[BaseIntegration]] = {}
        self.configurations: Dict[str, Dict[str, Any]] = {}
        self.created_integrations: Dict[str, BaseIntegration] = {}
        
        # Metrics
        self.factory_metrics = {
            "integrations_created": 0,
            "integrations_failed": 0,
            "configuration_reloads": 0,
            "dependency_resolutions": 0
        }
    
    async def initialize(self):
        """Initialize the factory."""
        self.logger.info("Initializing integration factory")
        
        try:
            # Discover available integrations
            if self.factory_config.auto_discovery:
                self.integration_classes = await self.discovery.discover_integrations()
            
            # Load configurations
            self.configurations = await self.config_loader.load_configurations()
            
            # Validate configurations
            if self.factory_config.validation_enabled:
                await self._validate_configurations()
            
            # Set up dependency rules
            await self._setup_dependencies()
            
            self.logger.info("Integration factory initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize integration factory: {str(e)}")
            raise
    
    async def create_integration(self, 
                               name: str, 
                               tenant_id: str,
                               config_override: Optional[Dict[str, Any]] = None) -> Optional[BaseIntegration]:
        """Create a single integration instance."""
        
        try:
            # Get configuration
            config_data = self.configurations.get(name, {})
            if config_override:
                config_data.update(config_override)
            
            if not config_data:
                self.logger.error(f"No configuration found for integration: {name}")
                return None
            
            # Determine integration type and class
            integration_class = self._resolve_integration_class(name, config_data)
            if not integration_class:
                self.logger.error(f"Could not resolve integration class for: {name}")
                return None
            
            # Create integration config
            integration_config = self._build_integration_config(name, config_data)
            
            # Create integration instance
            integration = integration_class(integration_config, tenant_id)
            
            # Initialize if using eager provisioning
            if self.factory_config.provisioning_strategy == ProvisioningStrategy.EAGER:
                success = await integration.enable()
                if not success:
                    self.logger.error(f"Failed to initialize integration: {name}")
                    self.factory_metrics["integrations_failed"] += 1
                    return None
            
            self.created_integrations[f"{tenant_id}:{name}"] = integration
            self.factory_metrics["integrations_created"] += 1
            
            self.logger.info(f"Created integration: {name} for tenant: {tenant_id}")
            return integration
            
        except Exception as e:
            self.logger.error(f"Failed to create integration {name}: {str(e)}")
            self.factory_metrics["integrations_failed"] += 1
            return None
    
    async def create_integrations_for_tenant(self, tenant_id: str) -> Dict[str, BaseIntegration]:
        """Create all enabled integrations for a tenant."""
        
        created_integrations = {}
        
        try:
            # Resolve dependency order
            if self.factory_config.dependency_resolution:
                integration_configs = {
                    name: self._build_integration_config(name, config)
                    for name, config in self.configurations.items()
                    if config.get('enabled', True)
                }
                
                creation_order = self.dependency_resolver.resolve_dependencies(integration_configs)
                self.factory_metrics["dependency_resolutions"] += 1
            else:
                creation_order = [
                    name for name, config in self.configurations.items()
                    if config.get('enabled', True)
                ]
            
            # Create integrations in dependency order
            for name in creation_order:
                integration = await self.create_integration(name, tenant_id)
                if integration:
                    created_integrations[name] = integration
                else:
                    self.logger.warning(f"Failed to create integration {name} for tenant {tenant_id}")
            
            self.logger.info(f"Created {len(created_integrations)} integrations for tenant: {tenant_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to create integrations for tenant {tenant_id}: {str(e)}")
        
        return created_integrations
    
    async def reload_configurations(self):
        """Reload configurations from source."""
        try:
            self.configurations = await self.config_loader.load_configurations()
            
            if self.factory_config.validation_enabled:
                await self._validate_configurations()
            
            self.factory_metrics["configuration_reloads"] += 1
            self.logger.info("Configurations reloaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to reload configurations: {str(e)}")
            raise
    
    def register_integration_class(self, name: str, integration_class: Type[BaseIntegration]):
        """Manually register an integration class."""
        self.integration_classes[name] = integration_class
        self.logger.info(f"Registered integration class: {name}")
    
    def add_validation_rule(self, integration_type: str, rule: Callable[[Dict[str, Any]], bool]):
        """Add a validation rule for an integration type."""
        self.validator.add_validation_rule(integration_type, rule)
    
    def add_dependency(self, integration_name: str, dependency: IntegrationDependency):
        """Add a dependency for an integration."""
        self.dependency_resolver.add_dependency(integration_name, dependency)
    
    def get_factory_metrics(self) -> Dict[str, Any]:
        """Get factory metrics."""
        return {
            **self.factory_metrics,
            "discovered_integrations": len(self.integration_classes),
            "loaded_configurations": len(self.configurations),
            "created_integrations": len(self.created_integrations)
        }
    
    async def _validate_configurations(self):
        """Validate all loaded configurations."""
        for name, config in self.configurations.items():
            errors = self.validator.validate_config(name, config)
            if errors:
                self.logger.error(f"Configuration validation failed for {name}: {errors}")
                raise ValueError(f"Invalid configuration for {name}: {errors}")
    
    def _resolve_integration_class(self, name: str, config: Dict[str, Any]) -> Optional[Type[BaseIntegration]]:
        """Resolve the integration class for a given name and config."""
        
        # Try exact name match first
        if name in self.integration_classes:
            return self.integration_classes[name]
        
        # Try to match by type
        integration_type = config.get('type')
        if integration_type:
            type_mappings = {
                'spotify_api': 'spotify',
                'aws_s3': 'aws',
                'email_service': 'email',
                'websocket': 'websocket'
            }
            
            mapped_name = type_mappings.get(integration_type, integration_type)
            if mapped_name in self.integration_classes:
                return self.integration_classes[mapped_name]
        
        # Try fuzzy matching
        for class_name, integration_class in self.integration_classes.items():
            if name.lower() in class_name.lower() or class_name.lower() in name.lower():
                return integration_class
        
        return None
    
    def _build_integration_config(self, name: str, config_data: Dict[str, Any]) -> IntegrationConfig:
        """Build IntegrationConfig from configuration data."""
        
        # Determine integration type
        integration_type_str = config_data.get('type', name)
        try:
            integration_type = IntegrationType(integration_type_str)
        except ValueError:
            # Default to external API if type not recognized
            integration_type = IntegrationType.EXTERNAL_API
        
        return IntegrationConfig(
            name=name,
            type=integration_type,
            enabled=config_data.get('enabled', True),
            config=config_data.get('config', {}),
            secrets=config_data.get('secrets', {}),
            retry_policy=config_data.get('retry_policy', {}),
            rate_limits=config_data.get('rate_limits', {}),
            timeout=config_data.get('timeout', 30),
            health_check_interval=config_data.get('health_check_interval', 60),
            circuit_breaker_config=config_data.get('circuit_breaker_config', {})
        )
    
    async def _setup_dependencies(self):
        """Set up common integration dependencies."""
        
        # Example dependencies
        self.add_dependency("spotify_api", IntegrationDependency(
            name="auth_service",
            type=IntegrationType.AUTHENTICATION,
            required=True,
            initialization_order=1
        ))
        
        self.add_dependency("analytics_pipeline", IntegrationDependency(
            name="database",
            type=IntegrationType.DATA_PIPELINE,
            required=True,
            initialization_order=1
        ))
        
        self.add_dependency("email_notifications", IntegrationDependency(
            name="user_service",
            type=IntegrationType.EXTERNAL_API,
            required=False,
            initialization_order=2
        ))


# Global factory instance
_integration_factory: Optional[IntegrationFactory] = None


async def get_integration_factory(factory_config: FactoryConfig = None) -> IntegrationFactory:
    """Get or create the global integration factory."""
    global _integration_factory
    
    if _integration_factory is None:
        _integration_factory = IntegrationFactory(factory_config)
        await _integration_factory.initialize()
    
    return _integration_factory


async def create_integration_for_tenant(name: str, tenant_id: str, config_override: Dict[str, Any] = None) -> Optional[BaseIntegration]:
    """Convenience function to create an integration for a tenant."""
    factory = await get_integration_factory()
    return await factory.create_integration(name, tenant_id, config_override)


async def create_all_integrations_for_tenant(tenant_id: str) -> Dict[str, BaseIntegration]:
    """Convenience function to create all integrations for a tenant."""
    factory = await get_integration_factory()
    return await factory.create_integrations_for_tenant(tenant_id)


# Export main components
__all__ = [
    # Factory classes
    "IntegrationFactory",
    "FactoryConfig",
    "ConfigurationLoader",
    "ConfigurationValidator",
    "DependencyResolver",
    "IntegrationDiscovery",
    
    # Data classes
    "IntegrationDependency",
    
    # Enums
    "ConfigurationSource",
    "ProvisioningStrategy",
    
    # Factory functions
    "get_integration_factory",
    "create_integration_for_tenant",
    "create_all_integrations_for_tenant"
]
