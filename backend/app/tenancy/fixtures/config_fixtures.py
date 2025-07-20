"""
Spotify AI Agent - Configuration Fixtures
========================================

Enterprise configuration management system for 
multi-tenant Spotify AI Agent architecture.
"""

import asyncio
import json
import logging
import os
import yaml
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.tenancy.fixtures.base import BaseFixture, FixtureMetadata, FixtureType
from app.tenancy.fixtures.exceptions import (
    FixtureConfigError,
    FixtureValidationError,
    FixtureDataError
)
from app.tenancy.fixtures.constants import (
    DEFAULT_TENANT_LIMITS,
    PREMIUM_TENANT_LIMITS,
    ENTERPRISE_TENANT_LIMITS
)

logger = logging.getLogger(__name__)


class ConfigScope(Enum):
    """Configuration scope enumeration."""
    GLOBAL = "global"
    TENANT = "tenant"
    USER = "user"
    APPLICATION = "application"
    FEATURE = "feature"


class ConfigType(Enum):
    """Configuration type enumeration."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    JSON = "json"
    LIST = "list"
    ENCRYPTED = "encrypted"


class ConfigCategory(Enum):
    """Configuration category enumeration."""
    SYSTEM = "system"
    SECURITY = "security"
    INTEGRATION = "integration"
    AI_MODEL = "ai_model"
    BILLING = "billing"
    FEATURE_FLAG = "feature_flag"
    ANALYTICS = "analytics"
    NOTIFICATION = "notification"


@dataclass
class ConfigItem:
    """Individual configuration item."""
    key: str
    value: Any
    config_type: ConfigType
    category: ConfigCategory
    scope: ConfigScope
    tenant_id: Optional[str] = None
    user_id: Optional[UUID] = None
    description: Optional[str] = None
    is_sensitive: bool = False
    is_encrypted: bool = False
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        """Post-initialization validation and processing."""
        if self.config_type == ConfigType.ENCRYPTED:
            self.is_encrypted = True
            self.is_sensitive = True
        
        # Validate value type
        self._validate_value_type()
    
    def _validate_value_type(self) -> None:
        """Validate that value matches declared type."""
        if self.config_type == ConfigType.STRING and not isinstance(self.value, str):
            raise ValueError(f"Value must be string for key {self.key}")
        elif self.config_type == ConfigType.INTEGER and not isinstance(self.value, int):
            raise ValueError(f"Value must be integer for key {self.key}")
        elif self.config_type == ConfigType.FLOAT and not isinstance(self.value, (int, float)):
            raise ValueError(f"Value must be float for key {self.key}")
        elif self.config_type == ConfigType.BOOLEAN and not isinstance(self.value, bool):
            raise ValueError(f"Value must be boolean for key {self.key}")
        elif self.config_type == ConfigType.LIST and not isinstance(self.value, list):
            raise ValueError(f"Value must be list for key {self.key}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "key": self.key,
            "value": self.value if not self.is_sensitive else "[REDACTED]",
            "config_type": self.config_type.value,
            "category": self.category.value,
            "scope": self.scope.value,
            "tenant_id": self.tenant_id,
            "user_id": str(self.user_id) if self.user_id else None,
            "description": self.description,
            "is_sensitive": self.is_sensitive,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class ConfigurationSet(BaseModel):
    """Set of configuration items for a specific scope."""
    scope: ConfigScope
    tenant_id: Optional[str] = None
    user_id: Optional[UUID] = None
    items: List[ConfigItem] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_item(self, item: ConfigItem) -> None:
        """Add a configuration item to the set."""
        # Validate scope consistency
        if item.scope != self.scope:
            raise ValueError(f"Item scope {item.scope} doesn't match set scope {self.scope}")
        
        if self.scope == ConfigScope.TENANT and item.tenant_id != self.tenant_id:
            raise ValueError("Tenant ID mismatch for tenant-scoped configuration")
        
        # Check for duplicates
        existing_keys = {item.key for item in self.items}
        if item.key in existing_keys:
            raise ValueError(f"Configuration key already exists: {item.key}")
        
        self.items.append(item)
    
    def get_item(self, key: str) -> Optional[ConfigItem]:
        """Get configuration item by key."""
        for item in self.items:
            if item.key == key:
                return item
        return None
    
    def update_item(self, key: str, value: Any) -> bool:
        """Update configuration item value."""
        item = self.get_item(key)
        if item:
            item.value = value
            item.updated_at = datetime.now(timezone.utc)
            return True
        return False
    
    def remove_item(self, key: str) -> bool:
        """Remove configuration item by key."""
        for i, item in enumerate(self.items):
            if item.key == key:
                del self.items[i]
                return True
        return False
    
    def get_by_category(self, category: ConfigCategory) -> List[ConfigItem]:
        """Get all items in a specific category."""
        return [item for item in self.items if item.category == category]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "scope": self.scope.value,
            "tenant_id": self.tenant_id,
            "user_id": str(self.user_id) if self.user_id else None,
            "items": [item.to_dict() for item in self.items],
            "metadata": self.metadata
        }


class ConfigFixture(BaseFixture[ConfigurationSet]):
    """
    Fixture for managing configuration data.
    
    Handles:
    - Configuration loading from files
    - Configuration validation
    - Database storage
    - Encryption of sensitive values
    - Template-based configuration
    """
    
    def __init__(
        self,
        config_set: ConfigurationSet,
        config_files: Optional[List[str]] = None,
        **kwargs
    ):
        metadata = FixtureMetadata(
            fixture_type=FixtureType.CONFIG,
            tenant_id=config_set.tenant_id,
            name=f"config_{config_set.scope.value}_{config_set.tenant_id or 'global'}",
            description=f"Configuration setup for {config_set.scope.value} scope",
            tags={"config", config_set.scope.value}
        )
        super().__init__(metadata, **kwargs)
        
        self.config_set = config_set
        self.config_files = config_files or []
        
        logger.info(
            f"Initialized config fixture: {config_set.scope.value} "
            f"(tenant: {config_set.tenant_id})"
        )
    
    async def validate(self) -> bool:
        """
        Validate configuration set and items.
        
        Returns:
            bool: True if validation passes
            
        Raises:
            FixtureValidationError: If validation fails
        """
        errors = []
        
        try:
            # Validate configuration files
            for config_file in self.config_files:
                if not os.path.exists(config_file):
                    errors.append(f"Configuration file not found: {config_file}")
                else:
                    file_errors = await self._validate_config_file(config_file)
                    errors.extend(file_errors)
            
            # Validate configuration items
            for item in self.config_set.items:
                item_errors = await self._validate_config_item(item)
                errors.extend(item_errors)
            
            # Check for required configurations
            required_errors = await self._validate_required_configs()
            errors.extend(required_errors)
            
            # Check for conflicts with existing configurations
            session = await self.get_session()
            conflict_errors = await self._check_config_conflicts(session)
            errors.extend(conflict_errors)
            
            if errors:
                raise FixtureValidationError(
                    f"Configuration validation failed",
                    validation_errors=errors,
                    fixture_id=self.metadata.fixture_id,
                    tenant_id=self.metadata.tenant_id
                )
            
            logger.info(f"Configuration validation passed: {self.config_set.scope.value}")
            return True
            
        except Exception as e:
            if isinstance(e, FixtureValidationError):
                raise
            raise FixtureValidationError(
                f"Configuration validation error: {str(e)}",
                fixture_id=self.metadata.fixture_id,
                tenant_id=self.metadata.tenant_id
            )
    
    async def apply(self) -> ConfigurationSet:
        """
        Apply configuration set to database.
        
        Returns:
            ConfigurationSet: Applied configuration
            
        Raises:
            FixtureConfigError: If configuration application fails
        """
        try:
            session = await self.get_session()
            
            # Load additional configurations from files
            if self.config_files:
                await self._load_config_files()
                self.increment_processed(len(self.config_files))
            
            # Create configuration table if it doesn't exist
            await self._ensure_config_table(session)
            self.increment_processed()
            
            # Apply configuration items
            for item in self.config_set.items:
                await self._apply_config_item(session, item)
                self.increment_processed()
            
            await session.commit()
            
            logger.info(
                f"Configuration application completed: {self.config_set.scope.value} "
                f"({len(self.config_set.items)} items)"
            )
            return self.config_set
            
        except Exception as e:
            logger.error(f"Configuration application failed: {e}")
            await session.rollback()
            raise FixtureConfigError(
                f"Failed to apply configuration: {str(e)}",
                fixture_id=self.metadata.fixture_id,
                tenant_id=self.metadata.tenant_id
            )
    
    async def rollback(self) -> bool:
        """
        Rollback configuration changes.
        
        Returns:
            bool: True if rollback successful
        """
        try:
            session = await self.get_session()
            
            # Remove all configuration items for this scope
            delete_sql = """
            DELETE FROM configurations 
            WHERE scope = :scope 
            AND (:tenant_id IS NULL OR tenant_id = :tenant_id)
            AND (:user_id IS NULL OR user_id = :user_id)
            """
            
            await session.execute(
                text(delete_sql),
                {
                    "scope": self.config_set.scope.value,
                    "tenant_id": self.config_set.tenant_id,
                    "user_id": str(self.config_set.user_id) if self.config_set.user_id else None
                }
            )
            
            await session.commit()
            
            logger.info(f"Configuration rollback completed: {self.config_set.scope.value}")
            return True
            
        except Exception as e:
            logger.error(f"Configuration rollback failed: {e}")
            return False
    
    async def _validate_config_file(self, config_file: str) -> List[str]:
        """Validate configuration file format and content."""
        errors = []
        
        try:
            file_path = Path(config_file)
            
            if file_path.suffix.lower() == '.json':
                with open(config_file, 'r', encoding='utf-8') as f:
                    json.load(f)  # Validate JSON syntax
            elif file_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_file, 'r', encoding='utf-8') as f:
                    yaml.safe_load(f)  # Validate YAML syntax
            else:
                errors.append(f"Unsupported configuration file format: {config_file}")
                
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in {config_file}: {str(e)}")
        except yaml.YAMLError as e:
            errors.append(f"Invalid YAML in {config_file}: {str(e)}")
        except Exception as e:
            errors.append(f"Error reading configuration file {config_file}: {str(e)}")
        
        return errors
    
    async def _validate_config_item(self, item: ConfigItem) -> List[str]:
        """Validate individual configuration item."""
        errors = []
        
        # Validate key format
        if not item.key or not isinstance(item.key, str):
            errors.append(f"Invalid configuration key: {item.key}")
        
        # Validate value according to validation rules
        if item.validation_rules:
            validation_errors = await self._apply_validation_rules(item)
            errors.extend(validation_errors)
        
        # Category-specific validations
        if item.category == ConfigCategory.SECURITY:
            if not item.is_sensitive and 'password' in item.key.lower():
                self.add_warning(f"Security config {item.key} should be marked as sensitive")
        
        return errors
    
    async def _apply_validation_rules(self, item: ConfigItem) -> List[str]:
        """Apply validation rules to configuration item."""
        errors = []
        rules = item.validation_rules
        
        if 'min_length' in rules and isinstance(item.value, str):
            if len(item.value) < rules['min_length']:
                errors.append(f"Value too short for {item.key} (min: {rules['min_length']})")
        
        if 'max_length' in rules and isinstance(item.value, str):
            if len(item.value) > rules['max_length']:
                errors.append(f"Value too long for {item.key} (max: {rules['max_length']})")
        
        if 'min_value' in rules and isinstance(item.value, (int, float)):
            if item.value < rules['min_value']:
                errors.append(f"Value too small for {item.key} (min: {rules['min_value']})")
        
        if 'max_value' in rules and isinstance(item.value, (int, float)):
            if item.value > rules['max_value']:
                errors.append(f"Value too large for {item.key} (max: {rules['max_value']})")
        
        if 'allowed_values' in rules:
            if item.value not in rules['allowed_values']:
                errors.append(f"Invalid value for {item.key}, allowed: {rules['allowed_values']}")
        
        if 'regex' in rules and isinstance(item.value, str):
            import re
            if not re.match(rules['regex'], item.value):
                errors.append(f"Value format invalid for {item.key}")
        
        return errors
    
    async def _validate_required_configs(self) -> List[str]:
        """Validate that required configurations are present."""
        errors = []
        
        # Define required configurations by scope
        required_configs = {
            ConfigScope.GLOBAL: [
                "system.name",
                "system.version",
                "database.url"
            ],
            ConfigScope.TENANT: [
                "tenant.name",
                "tenant.tier",
                "billing.email"
            ],
            ConfigScope.APPLICATION: [
                "app.environment",
                "app.debug"
            ]
        }
        
        required_for_scope = required_configs.get(self.config_set.scope, [])
        existing_keys = {item.key for item in self.config_set.items}
        
        for required_key in required_for_scope:
            if required_key not in existing_keys:
                errors.append(f"Required configuration missing: {required_key}")
        
        return errors
    
    async def _check_config_conflicts(self, session: AsyncSession) -> List[str]:
        """Check for conflicts with existing configurations."""
        errors = []
        
        for item in self.config_set.items:
            # Check if configuration already exists
            result = await session.execute(
                text("""
                SELECT 1 FROM configurations 
                WHERE config_key = :key 
                AND scope = :scope 
                AND (:tenant_id IS NULL OR tenant_id = :tenant_id)
                """),
                {
                    "key": item.key,
                    "scope": item.scope.value,
                    "tenant_id": item.tenant_id
                }
            )
            
            if result.first():
                errors.append(f"Configuration already exists: {item.key}")
        
        return errors
    
    async def _load_config_files(self) -> None:
        """Load configuration items from files."""
        for config_file in self.config_files:
            file_path = Path(config_file)
            
            if file_path.suffix.lower() == '.json':
                await self._load_json_config(config_file)
            elif file_path.suffix.lower() in ['.yaml', '.yml']:
                await self._load_yaml_config(config_file)
            
            logger.info(f"Loaded configuration from: {config_file}")
    
    async def _load_json_config(self, config_file: str) -> None:
        """Load configuration from JSON file."""
        with open(config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        await self._process_config_data(data)
    
    async def _load_yaml_config(self, config_file: str) -> None:
        """Load configuration from YAML file."""
        with open(config_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        await self._process_config_data(data)
    
    async def _process_config_data(self, data: Dict[str, Any]) -> None:
        """Process configuration data and create ConfigItem objects."""
        def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
            """Flatten nested dictionary with dot notation."""
            items = {}
            for key, value in d.items():
                new_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    items.update(flatten_dict(value, new_key))
                else:
                    items[new_key] = value
            return items
        
        flattened = flatten_dict(data)
        
        for key, value in flattened.items():
            # Determine config type
            if isinstance(value, bool):
                config_type = ConfigType.BOOLEAN
            elif isinstance(value, int):
                config_type = ConfigType.INTEGER
            elif isinstance(value, float):
                config_type = ConfigType.FLOAT
            elif isinstance(value, list):
                config_type = ConfigType.LIST
            elif isinstance(value, dict):
                config_type = ConfigType.JSON
            else:
                config_type = ConfigType.STRING
            
            # Determine category based on key prefix
            category = self._determine_category(key)
            
            # Check if sensitive
            is_sensitive = any(sensitive_word in key.lower() 
                             for sensitive_word in ['password', 'secret', 'key', 'token'])
            
            item = ConfigItem(
                key=key,
                value=value,
                config_type=config_type,
                category=category,
                scope=self.config_set.scope,
                tenant_id=self.config_set.tenant_id,
                user_id=self.config_set.user_id,
                is_sensitive=is_sensitive
            )
            
            self.config_set.add_item(item)
    
    def _determine_category(self, key: str) -> ConfigCategory:
        """Determine configuration category based on key."""
        key_lower = key.lower()
        
        if any(word in key_lower for word in ['security', 'auth', 'password', 'token']):
            return ConfigCategory.SECURITY
        elif any(word in key_lower for word in ['spotify', 'api', 'webhook', 'integration']):
            return ConfigCategory.INTEGRATION
        elif any(word in key_lower for word in ['ai', 'model', 'ml', 'openai']):
            return ConfigCategory.AI_MODEL
        elif any(word in key_lower for word in ['billing', 'payment', 'stripe']):
            return ConfigCategory.BILLING
        elif any(word in key_lower for word in ['feature', 'flag', 'enable', 'disable']):
            return ConfigCategory.FEATURE_FLAG
        elif any(word in key_lower for word in ['analytics', 'metrics', 'tracking']):
            return ConfigCategory.ANALYTICS
        elif any(word in key_lower for word in ['notification', 'email', 'sms']):
            return ConfigCategory.NOTIFICATION
        else:
            return ConfigCategory.SYSTEM
    
    async def _ensure_config_table(self, session: AsyncSession) -> None:
        """Ensure configuration table exists."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS configurations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            config_key VARCHAR(255) NOT NULL,
            config_value TEXT,
            config_type VARCHAR(50) NOT NULL,
            category VARCHAR(50) NOT NULL,
            scope VARCHAR(50) NOT NULL,
            tenant_id VARCHAR(50),
            user_id UUID,
            description TEXT,
            is_sensitive BOOLEAN DEFAULT FALSE,
            is_encrypted BOOLEAN DEFAULT FALSE,
            validation_rules JSONB DEFAULT '{}',
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            UNIQUE(config_key, scope, COALESCE(tenant_id, ''), COALESCE(user_id::text, ''))
        )
        """
        
        await session.execute(text(create_table_sql))
        
        # Create indexes
        indexes_sql = [
            "CREATE INDEX IF NOT EXISTS idx_configurations_scope ON configurations(scope)",
            "CREATE INDEX IF NOT EXISTS idx_configurations_tenant_id ON configurations(tenant_id)",
            "CREATE INDEX IF NOT EXISTS idx_configurations_category ON configurations(category)",
            "CREATE INDEX IF NOT EXISTS idx_configurations_key ON configurations(config_key)"
        ]
        
        for sql in indexes_sql:
            await session.execute(text(sql))
    
    async def _apply_config_item(self, session: AsyncSession, item: ConfigItem) -> None:
        """Apply individual configuration item to database."""
        # Encrypt sensitive values if needed
        value_to_store = item.value
        if item.is_sensitive and not item.is_encrypted:
            value_to_store = await self._encrypt_value(str(item.value))
            item.is_encrypted = True
        
        insert_sql = """
        INSERT INTO configurations (
            config_key, config_value, config_type, category, scope,
            tenant_id, user_id, description, is_sensitive, is_encrypted,
            validation_rules, metadata
        ) VALUES (
            :key, :value, :type, :category, :scope,
            :tenant_id, :user_id, :description, :sensitive, :encrypted,
            :rules, :metadata
        )
        """
        
        await session.execute(
            text(insert_sql),
            {
                "key": item.key,
                "value": json.dumps(value_to_store) if item.config_type == ConfigType.JSON else str(value_to_store),
                "type": item.config_type.value,
                "category": item.category.value,
                "scope": item.scope.value,
                "tenant_id": item.tenant_id,
                "user_id": str(item.user_id) if item.user_id else None,
                "description": item.description,
                "sensitive": item.is_sensitive,
                "encrypted": item.is_encrypted,
                "rules": json.dumps(item.validation_rules),
                "metadata": json.dumps(item.metadata)
            }
        )
    
    async def _encrypt_value(self, value: str) -> str:
        """Encrypt sensitive configuration value."""
        # This would integrate with encryption service
        # For now, return base64 encoded value as placeholder
        import base64
        return base64.b64encode(value.encode()).decode()


class ConfigurationManager:
    """
    Central manager for configuration operations.
    
    Provides:
    - Configuration loading and caching
    - Template-based configuration
    - Environment-specific configurations
    - Configuration validation and updates
    """
    
    def __init__(self, config_path: str = "/app/config"):
        self.config_path = Path(config_path)
        self.cache: Dict[str, ConfigurationSet] = {}
        self.logger = logging.getLogger(f"{__name__}.ConfigurationManager")
    
    async def load_tenant_config(
        self,
        tenant_id: str,
        config_template: str = "tenant_config.yaml"
    ) -> ConfigurationSet:
        """Load configuration for a specific tenant."""
        cache_key = f"tenant:{tenant_id}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        config_set = ConfigurationSet(
            scope=ConfigScope.TENANT,
            tenant_id=tenant_id
        )
        
        # Load from template
        template_path = self.config_path / "templates" / config_template
        if template_path.exists():
            config_fixture = ConfigFixture(
                config_set=config_set,
                config_files=[str(template_path)]
            )
            await config_fixture._load_config_files()
        
        self.cache[cache_key] = config_set
        self.logger.info(f"Loaded configuration for tenant: {tenant_id}")
        
        return config_set
    
    async def create_from_template(
        self,
        template_name: str,
        variables: Dict[str, Any],
        scope: ConfigScope,
        tenant_id: Optional[str] = None
    ) -> ConfigurationSet:
        """Create configuration set from template with variable substitution."""
        template_path = self.config_path / "templates" / f"{template_name}.yaml"
        
        if not template_path.exists():
            raise FixtureConfigError(f"Template not found: {template_name}")
        
        # Load template
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        # Replace variables
        for key, value in variables.items():
            template_content = template_content.replace(f"{{{key}}}", str(value))
        
        # Parse processed template
        config_data = yaml.safe_load(template_content)
        
        # Create configuration set
        config_set = ConfigurationSet(
            scope=scope,
            tenant_id=tenant_id
        )
        
        # Process configuration data
        config_fixture = ConfigFixture(config_set=config_set)
        await config_fixture._process_config_data(config_data)
        
        self.logger.info(f"Created configuration from template: {template_name}")
        return config_set
    
    def clear_cache(self, tenant_id: Optional[str] = None) -> None:
        """Clear configuration cache."""
        if tenant_id:
            cache_key = f"tenant:{tenant_id}"
            self.cache.pop(cache_key, None)
        else:
            self.cache.clear()
        
        self.logger.info(f"Cleared configuration cache for tenant: {tenant_id or 'all'}")
    
    async def validate_config_set(self, config_set: ConfigurationSet) -> List[str]:
        """Validate a configuration set."""
        fixture = ConfigFixture(config_set=config_set)
        
        try:
            await fixture.validate()
            return []
        except FixtureValidationError as e:
            return e.validation_errors or [str(e)]
