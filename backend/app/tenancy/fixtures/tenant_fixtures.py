"""
Spotify AI Agent - Tenant-Specific Fixtures
==========================================

Comprehensive tenant fixture management for multi-tenant
Spotify AI Agent architecture with enterprise-grade features.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

from sqlalchemy import select, insert, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field, validator

from app.tenancy.fixtures.base import BaseFixture, FixtureMetadata, FixtureType
from app.tenancy.fixtures.exceptions import (
    FixtureValidationError,
    FixtureConflictError,
    FixtureDataError
)
from app.tenancy.fixtures.constants import (
    DEFAULT_TENANT_LIMITS,
    PREMIUM_TENANT_LIMITS,
    ENTERPRISE_TENANT_LIMITS
)

logger = logging.getLogger(__name__)


class TenantTier(Enum):
    """Tenant service tiers."""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class TenantStatus(Enum):
    """Tenant status enumeration."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    PENDING = "pending"
    ARCHIVED = "archived"


@dataclass
class TenantLimits:
    """Tenant resource limits configuration."""
    max_fixtures: int = 100
    max_data_size_mb: int = 1000
    max_concurrent_operations: int = 5
    api_calls_per_hour: int = 10000
    max_users: int = 10
    max_ai_requests_per_day: int = 1000
    storage_quota_gb: float = 10.0
    bandwidth_limit_mbps: float = 100.0
    
    @classmethod
    def for_tier(cls, tier: TenantTier) -> 'TenantLimits':
        """Create limits for a specific tenant tier."""
        limits_map = {
            TenantTier.FREE: DEFAULT_TENANT_LIMITS,
            TenantTier.BASIC: DEFAULT_TENANT_LIMITS,
            TenantTier.PREMIUM: PREMIUM_TENANT_LIMITS,
            TenantTier.ENTERPRISE: ENTERPRISE_TENANT_LIMITS
        }
        limits_config = limits_map.get(tier, DEFAULT_TENANT_LIMITS)
        return cls(**limits_config)


@dataclass
class TenantFeatures:
    """Tenant feature flags and capabilities."""
    ai_collaboration: bool = False
    advanced_analytics: bool = False
    real_time_sync: bool = False
    custom_models: bool = False
    api_access: bool = True
    white_labeling: bool = False
    sso_integration: bool = False
    audit_logging: bool = False
    priority_support: bool = False
    custom_integrations: bool = False
    
    @classmethod
    def for_tier(cls, tier: TenantTier) -> 'TenantFeatures':
        """Create features for a specific tenant tier."""
        if tier == TenantTier.FREE:
            return cls()
        elif tier == TenantTier.BASIC:
            return cls(
                ai_collaboration=True,
                api_access=True
            )
        elif tier == TenantTier.PREMIUM:
            return cls(
                ai_collaboration=True,
                advanced_analytics=True,
                real_time_sync=True,
                api_access=True,
                audit_logging=True
            )
        elif tier == TenantTier.ENTERPRISE:
            return cls(
                ai_collaboration=True,
                advanced_analytics=True,
                real_time_sync=True,
                custom_models=True,
                api_access=True,
                white_labeling=True,
                sso_integration=True,
                audit_logging=True,
                priority_support=True,
                custom_integrations=True
            )
        return cls()


class TenantConfiguration(BaseModel):
    """Comprehensive tenant configuration model."""
    tenant_id: str = Field(..., min_length=3, max_length=50)
    name: str = Field(..., min_length=1, max_length=100)
    domain: Optional[str] = Field(None, max_length=100)
    tier: TenantTier = Field(default=TenantTier.FREE)
    status: TenantStatus = Field(default=TenantStatus.PENDING)
    
    # Contact Information
    admin_email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    admin_name: str = Field(..., min_length=1, max_length=100)
    phone: Optional[str] = Field(None, max_length=20)
    
    # Billing Information
    billing_email: Optional[str] = Field(None, regex=r'^[^@]+@[^@]+\.[^@]+$')
    billing_address: Optional[Dict[str, str]] = None
    payment_method_id: Optional[str] = None
    
    # Technical Configuration
    database_schema: str = Field(..., min_length=3, max_length=50)
    redis_namespace: str = Field(..., min_length=3, max_length=50)
    s3_bucket: Optional[str] = Field(None, max_length=100)
    
    # Security Settings
    encryption_key_id: Optional[str] = None
    allowed_ips: List[str] = Field(default_factory=list)
    require_mfa: bool = Field(default=False)
    session_timeout_minutes: int = Field(default=480, ge=30, le=1440)
    
    # Integration Settings
    spotify_client_id: Optional[str] = None
    spotify_client_secret: Optional[str] = None
    webhook_url: Optional[str] = None
    webhook_secret: Optional[str] = None
    
    # AI Configuration
    ai_model_preferences: Dict[str, Any] = Field(default_factory=dict)
    content_filters: List[str] = Field(default_factory=list)
    language_preferences: List[str] = Field(default_factory=lambda: ["en"])
    
    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('tenant_id')
    def validate_tenant_id(cls, v):
        """Validate tenant ID format."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Tenant ID must contain only alphanumeric characters, hyphens, and underscores")
        return v.lower()
    
    @validator('database_schema')
    def validate_database_schema(cls, v, values):
        """Validate database schema name."""
        tenant_id = values.get('tenant_id', '')
        if not v.startswith(f"tenant_{tenant_id}"):
            return f"tenant_{tenant_id}_{v}"
        return v
    
    @validator('redis_namespace')
    def validate_redis_namespace(cls, v, values):
        """Validate Redis namespace."""
        tenant_id = values.get('tenant_id', '')
        if not v.startswith(f"tenant:{tenant_id}"):
            return f"tenant:{tenant_id}:{v}"
        return v


class TenantFixture(BaseFixture[TenantConfiguration]):
    """
    Fixture for creating and managing tenant configurations.
    
    Handles:
    - Tenant creation and initialization
    - Schema setup
    - Resource allocation
    - Feature configuration
    - Security setup
    """
    
    def __init__(
        self,
        tenant_config: TenantConfiguration,
        **kwargs
    ):
        metadata = FixtureMetadata(
            fixture_type=FixtureType.TENANT,
            tenant_id=tenant_config.tenant_id,
            name=f"tenant_setup_{tenant_config.tenant_id}",
            description=f"Tenant setup for {tenant_config.name}",
            tags={"tenant", "setup", tenant_config.tier.value}
        )
        super().__init__(metadata, **kwargs)
        
        self.tenant_config = tenant_config
        self.limits = TenantLimits.for_tier(tenant_config.tier)
        self.features = TenantFeatures.for_tier(tenant_config.tier)
        
        logger.info(f"Initialized tenant fixture for: {tenant_config.tenant_id}")
    
    async def validate(self) -> bool:
        """
        Validate tenant configuration and dependencies.
        
        Returns:
            bool: True if validation passes
            
        Raises:
            FixtureValidationError: If validation fails
        """
        errors = []
        
        try:
            # Validate basic configuration
            self.tenant_config.dict()  # Pydantic validation
            
            # Check for existing tenant
            session = await self.get_session()
            result = await session.execute(
                select(1).where(
                    # Assuming we have a tenants table
                    # This would need to be adapted to actual schema
                    f"SELECT 1 FROM tenants WHERE tenant_id = '{self.tenant_config.tenant_id}'"
                )
            )
            
            if result.first():
                errors.append(f"Tenant already exists: {self.tenant_config.tenant_id}")
            
            # Validate domain uniqueness if provided
            if self.tenant_config.domain:
                domain_result = await session.execute(
                    f"SELECT 1 FROM tenants WHERE domain = '{self.tenant_config.domain}'"
                )
                if domain_result.first():
                    errors.append(f"Domain already in use: {self.tenant_config.domain}")
            
            # Validate resource limits
            if self.tenant_config.tier == TenantTier.FREE and len(self.tenant_config.allowed_ips) > 5:
                errors.append("Free tier is limited to 5 allowed IP addresses")
            
            # Validate Spotify credentials if provided
            if (self.tenant_config.spotify_client_id and 
                not self.tenant_config.spotify_client_secret):
                errors.append("Spotify client secret required when client ID provided")
            
            if errors:
                raise FixtureValidationError(
                    f"Tenant validation failed for {self.tenant_config.tenant_id}",
                    validation_errors=errors,
                    fixture_id=self.metadata.fixture_id,
                    tenant_id=self.tenant_config.tenant_id
                )
            
            logger.info(f"Tenant validation passed: {self.tenant_config.tenant_id}")
            return True
            
        except Exception as e:
            if isinstance(e, FixtureValidationError):
                raise
            raise FixtureValidationError(
                f"Tenant validation error: {str(e)}",
                fixture_id=self.metadata.fixture_id,
                tenant_id=self.tenant_config.tenant_id
            )
    
    async def apply(self) -> TenantConfiguration:
        """
        Apply tenant configuration and create resources.
        
        Returns:
            TenantConfiguration: Applied configuration
            
        Raises:
            FixtureDataError: If data creation fails
        """
        try:
            session = await self.get_session()
            
            # Create tenant record
            await self._create_tenant_record(session)
            self.increment_processed()
            
            # Create database schema
            await self._create_database_schema(session)
            self.increment_processed()
            
            # Setup Redis namespace
            await self._setup_redis_namespace()
            self.increment_processed()
            
            # Create default data
            await self._create_default_data(session)
            self.increment_processed(5)  # Multiple default records
            
            # Setup security configuration
            await self._setup_security_config(session)
            self.increment_processed()
            
            # Configure features
            await self._configure_features(session)
            self.increment_processed()
            
            # Commit transaction
            await session.commit()
            
            logger.info(f"Tenant setup completed: {self.tenant_config.tenant_id}")
            return self.tenant_config
            
        except Exception as e:
            logger.error(f"Tenant setup failed: {self.tenant_config.tenant_id} - {e}")
            await session.rollback()
            raise FixtureDataError(
                f"Failed to create tenant: {str(e)}",
                fixture_id=self.metadata.fixture_id,
                tenant_id=self.tenant_config.tenant_id
            )
    
    async def rollback(self) -> bool:
        """
        Rollback tenant creation.
        
        Returns:
            bool: True if rollback successful
        """
        try:
            session = await self.get_session()
            
            # Remove tenant data in reverse order
            await self._cleanup_features(session)
            await self._cleanup_security_config(session)
            await self._cleanup_default_data(session)
            await self._cleanup_redis_namespace()
            await self._cleanup_database_schema(session)
            await self._remove_tenant_record(session)
            
            await session.commit()
            
            logger.info(f"Tenant rollback completed: {self.tenant_config.tenant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Tenant rollback failed: {self.tenant_config.tenant_id} - {e}")
            return False
    
    async def _create_tenant_record(self, session: AsyncSession) -> None:
        """Create the main tenant record."""
        tenant_data = {
            "tenant_id": self.tenant_config.tenant_id,
            "name": self.tenant_config.name,
            "domain": self.tenant_config.domain,
            "tier": self.tenant_config.tier.value,
            "status": self.tenant_config.status.value,
            "admin_email": self.tenant_config.admin_email,
            "admin_name": self.tenant_config.admin_name,
            "phone": self.tenant_config.phone,
            "billing_email": self.tenant_config.billing_email,
            "billing_address": json.dumps(self.tenant_config.billing_address) if self.tenant_config.billing_address else None,
            "database_schema": self.tenant_config.database_schema,
            "redis_namespace": self.tenant_config.redis_namespace,
            "s3_bucket": self.tenant_config.s3_bucket,
            "created_at": self.tenant_config.created_at,
            "updated_at": self.tenant_config.updated_at,
            "metadata": json.dumps(self.tenant_config.metadata)
        }
        
        # This would need to be adapted to actual table structure
        await session.execute(
            insert("tenants").values(**tenant_data)
        )
    
    async def _create_database_schema(self, session: AsyncSession) -> None:
        """Create tenant-specific database schema."""
        schema_name = self.tenant_config.database_schema
        
        # Create schema
        await session.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
        
        # Create tenant-specific tables
        await self._create_tenant_tables(session, schema_name)
        
        logger.info(f"Created database schema: {schema_name}")
    
    async def _create_tenant_tables(self, session: AsyncSession, schema_name: str) -> None:
        """Create tables within tenant schema."""
        tables_sql = [
            f"""
            CREATE TABLE {schema_name}.users (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                tenant_id VARCHAR(50) NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                name VARCHAR(100) NOT NULL,
                role VARCHAR(50) DEFAULT 'user',
                is_active BOOLEAN DEFAULT true,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
            """,
            f"""
            CREATE TABLE {schema_name}.spotify_connections (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID REFERENCES {schema_name}.users(id),
                spotify_user_id VARCHAR(100) UNIQUE NOT NULL,
                access_token TEXT,
                refresh_token TEXT,
                token_expires_at TIMESTAMP WITH TIME ZONE,
                scopes TEXT[],
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
            """,
            f"""
            CREATE TABLE {schema_name}.ai_sessions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID REFERENCES {schema_name}.users(id),
                session_type VARCHAR(50) NOT NULL,
                status VARCHAR(20) DEFAULT 'active',
                context JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
            """,
            f"""
            CREATE TABLE {schema_name}.content_generated (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                session_id UUID REFERENCES {schema_name}.ai_sessions(id),
                content_type VARCHAR(50) NOT NULL,
                content_data JSONB NOT NULL,
                metadata JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
            """
        ]
        
        for sql in tables_sql:
            await session.execute(sql)
    
    async def _setup_redis_namespace(self) -> None:
        """Setup Redis namespace for tenant."""
        # This would integrate with Redis client
        # For now, just log the operation
        logger.info(f"Setup Redis namespace: {self.tenant_config.redis_namespace}")
    
    async def _create_default_data(self, session: AsyncSession) -> None:
        """Create default data for tenant."""
        schema_name = self.tenant_config.database_schema
        
        # Create default admin user
        admin_user_data = {
            "tenant_id": self.tenant_config.tenant_id,
            "email": self.tenant_config.admin_email,
            "name": self.tenant_config.admin_name,
            "role": "admin"
        }
        
        await session.execute(
            f"INSERT INTO {schema_name}.users (tenant_id, email, name, role) VALUES "
            f"('{admin_user_data['tenant_id']}', '{admin_user_data['email']}', "
            f"'{admin_user_data['name']}', '{admin_user_data['role']}')"
        )
    
    async def _setup_security_config(self, session: AsyncSession) -> None:
        """Setup security configuration."""
        # This would create security policies, encryption keys, etc.
        logger.info(f"Setup security config for tenant: {self.tenant_config.tenant_id}")
    
    async def _configure_features(self, session: AsyncSession) -> None:
        """Configure tenant features."""
        # This would setup feature flags and capabilities
        logger.info(f"Configured features for tenant: {self.tenant_config.tenant_id}")
    
    # Cleanup methods for rollback
    async def _cleanup_features(self, session: AsyncSession) -> None:
        """Cleanup tenant features."""
        logger.info(f"Cleaning up features for tenant: {self.tenant_config.tenant_id}")
    
    async def _cleanup_security_config(self, session: AsyncSession) -> None:
        """Cleanup security configuration."""
        logger.info(f"Cleaning up security config for tenant: {self.tenant_config.tenant_id}")
    
    async def _cleanup_default_data(self, session: AsyncSession) -> None:
        """Cleanup default data."""
        schema_name = self.tenant_config.database_schema
        await session.execute(f"DELETE FROM {schema_name}.users")
    
    async def _cleanup_redis_namespace(self) -> None:
        """Cleanup Redis namespace."""
        logger.info(f"Cleaning up Redis namespace: {self.tenant_config.redis_namespace}")
    
    async def _cleanup_database_schema(self, session: AsyncSession) -> None:
        """Cleanup database schema."""
        schema_name = self.tenant_config.database_schema
        await session.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
    
    async def _remove_tenant_record(self, session: AsyncSession) -> None:
        """Remove tenant record."""
        await session.execute(
            f"DELETE FROM tenants WHERE tenant_id = '{self.tenant_config.tenant_id}'"
        )


class TenantDataLoader:
    """
    Utility class for loading tenant data from various sources.
    
    Supports:
    - JSON configuration files
    - YAML configuration files
    - Environment variables
    - External APIs
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.TenantDataLoader")
    
    async def load_from_json(self, file_path: str) -> List[TenantConfiguration]:
        """Load tenant configurations from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                data = [data]
            
            configs = []
            for item in data:
                config = TenantConfiguration(**item)
                configs.append(config)
            
            self.logger.info(f"Loaded {len(configs)} tenant configurations from {file_path}")
            return configs
            
        except Exception as e:
            self.logger.error(f"Failed to load tenant data from {file_path}: {e}")
            raise FixtureDataError(f"Failed to load tenant data: {str(e)}")
    
    async def load_from_template(
        self,
        template_data: Dict[str, Any],
        variables: Dict[str, Any]
    ) -> TenantConfiguration:
        """Load tenant configuration from template with variable substitution."""
        try:
            # Simple variable substitution
            config_str = json.dumps(template_data)
            for key, value in variables.items():
                config_str = config_str.replace(f"{{{key}}}", str(value))
            
            config_data = json.loads(config_str)
            config = TenantConfiguration(**config_data)
            
            self.logger.info(f"Created tenant configuration from template: {config.tenant_id}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to create tenant from template: {e}")
            raise FixtureDataError(f"Failed to create tenant from template: {str(e)}")
    
    async def validate_bulk_configs(
        self,
        configs: List[TenantConfiguration]
    ) -> List[str]:
        """Validate multiple tenant configurations for conflicts."""
        errors = []
        tenant_ids = set()
        domains = set()
        emails = set()
        
        for config in configs:
            # Check for duplicate tenant IDs
            if config.tenant_id in tenant_ids:
                errors.append(f"Duplicate tenant ID: {config.tenant_id}")
            tenant_ids.add(config.tenant_id)
            
            # Check for duplicate domains
            if config.domain:
                if config.domain in domains:
                    errors.append(f"Duplicate domain: {config.domain}")
                domains.add(config.domain)
            
            # Check for duplicate admin emails
            if config.admin_email in emails:
                errors.append(f"Duplicate admin email: {config.admin_email}")
            emails.add(config.admin_email)
        
        return errors
