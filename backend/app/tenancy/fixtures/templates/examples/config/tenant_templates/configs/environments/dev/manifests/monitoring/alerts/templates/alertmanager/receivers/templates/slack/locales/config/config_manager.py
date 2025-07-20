"""
Advanced Slack Configuration Manager for Multi-Tenant Alert System.

This module provides enterprise-grade configuration management for Slack alerting
with comprehensive multi-tenant support, caching, and performance optimization.

Features:
- Multi-tenant configuration isolation
- Dynamic configuration reloading
- Redis-based distributed caching
- Configuration validation and sanitization
- Audit logging and compliance
- Performance monitoring integration

Author: Fahed Mlaiel
Version: 1.0.0
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

import aioredis
from pydantic import BaseModel, Field, validator
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from cryptography.fernet import Fernet

from .constants import (
    DEFAULT_LOCALE,
    CACHE_TTL_CONFIG,
    MAX_RETRIES,
    ALERT_PRIORITIES
)
from .exceptions import (
    ConfigurationError,
    TenantNotFoundError,
    CacheError,
    ValidationError
)
from .security_manager import SecurityManager
from .performance_monitor import PerformanceMonitor
from .validation import ConfigValidator


@dataclass
class SlackChannelConfig:
    """Configuration for a Slack channel."""
    channel_id: str
    channel_name: str
    webhook_url: str
    token: str
    priority_level: str
    rate_limit: int = 100
    enabled: bool = True
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()


@dataclass
class TenantSlackConfig:
    """Complete Slack configuration for a tenant."""
    tenant_id: str
    environment: str
    default_locale: str
    channels: Dict[str, SlackChannelConfig]
    global_settings: Dict[str, Any]
    security_settings: Dict[str, Any]
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()


class SlackConfigManager:
    """
    Advanced Slack configuration manager with enterprise features.
    
    Provides comprehensive configuration management for Slack alerting
    system with multi-tenant support, caching, and security.
    """

    def __init__(
        self,
        redis_url: str,
        database_session: AsyncSession,
        encryption_key: Optional[str] = None,
        cache_ttl: int = CACHE_TTL_CONFIG,
        max_retries: int = MAX_RETRIES,
        enable_monitoring: bool = True
    ):
        """
        Initialize the Slack configuration manager.
        
        Args:
            redis_url: Redis connection URL for caching
            database_session: Database session for persistent storage
            encryption_key: Key for encrypting sensitive data
            cache_ttl: Cache time-to-live in seconds
            max_retries: Maximum retry attempts for operations
            enable_monitoring: Enable performance monitoring
        """
        self.redis_url = redis_url
        self.db_session = database_session
        self.cache_ttl = cache_ttl
        self.max_retries = max_retries
        
        # Initialize components
        self.security_manager = SecurityManager(encryption_key)
        self.config_validator = ConfigValidator()
        self.performance_monitor = PerformanceMonitor() if enable_monitoring else None
        
        # Redis connection
        self.redis_client: Optional[aioredis.Redis] = None
        
        # In-memory cache for frequently accessed configs
        self._memory_cache: Dict[str, TenantSlackConfig] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # Configuration change listeners
        self._change_listeners: List[callable] = []
        
        # Logging
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """Initialize the configuration manager."""
        try:
            # Connect to Redis
            self.redis_client = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )
            
            # Test Redis connection
            await self.redis_client.ping()
            self.logger.info("Redis connection established successfully")
            
            # Initialize performance monitoring
            if self.performance_monitor:
                await self.performance_monitor.initialize()
            
            self.logger.info("SlackConfigManager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SlackConfigManager: {e}")
            raise ConfigurationError(f"Initialization failed: {e}")

    async def get_tenant_config(
        self,
        tenant_id: str,
        environment: str = "production",
        use_cache: bool = True
    ) -> TenantSlackConfig:
        """
        Retrieve Slack configuration for a specific tenant.
        
        Args:
            tenant_id: Unique tenant identifier
            environment: Environment (dev, staging, production)
            use_cache: Whether to use caching
            
        Returns:
            Complete tenant Slack configuration
            
        Raises:
            TenantNotFoundError: If tenant configuration not found
            ConfigurationError: If configuration is invalid
        """
        cache_key = f"slack_config:{tenant_id}:{environment}"
        
        if self.performance_monitor:
            timer = self.performance_monitor.start_timer("get_tenant_config")
        
        try:
            # Check memory cache first
            if use_cache and cache_key in self._memory_cache:
                cache_time = self._cache_timestamps.get(cache_key)
                if cache_time and (datetime.utcnow() - cache_time).seconds < self.cache_ttl:
                    self.logger.debug(f"Config retrieved from memory cache for {tenant_id}")
                    return self._memory_cache[cache_key]
            
            # Check Redis cache
            if use_cache and self.redis_client:
                cached_config = await self.redis_client.get(cache_key)
                if cached_config:
                    config_data = json.loads(cached_config)
                    config = self._deserialize_config(config_data)
                    
                    # Update memory cache
                    self._memory_cache[cache_key] = config
                    self._cache_timestamps[cache_key] = datetime.utcnow()
                    
                    self.logger.debug(f"Config retrieved from Redis cache for {tenant_id}")
                    return config
            
            # Load from database
            config = await self._load_config_from_database(tenant_id, environment)
            
            if not config:
                raise TenantNotFoundError(f"Configuration not found for tenant {tenant_id}")
            
            # Validate configuration
            await self.config_validator.validate_tenant_config(config)
            
            # Cache the configuration
            if use_cache:
                await self._cache_config(cache_key, config)
            
            self.logger.info(f"Config loaded from database for {tenant_id}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to get tenant config for {tenant_id}: {e}")
            raise
        finally:
            if self.performance_monitor and 'timer' in locals():
                self.performance_monitor.end_timer(timer)

    async def create_tenant_config(
        self,
        tenant_id: str,
        environment: str,
        config_data: Dict[str, Any]
    ) -> TenantSlackConfig:
        """
        Create a new tenant Slack configuration.
        
        Args:
            tenant_id: Unique tenant identifier
            environment: Environment name
            config_data: Configuration data
            
        Returns:
            Created tenant configuration
            
        Raises:
            ValidationError: If configuration data is invalid
            ConfigurationError: If creation fails
        """
        try:
            # Validate input data
            validated_data = await self.config_validator.validate_create_data(config_data)
            
            # Create configuration object
            config = TenantSlackConfig(
                tenant_id=tenant_id,
                environment=environment,
                default_locale=validated_data.get('default_locale', DEFAULT_LOCALE),
                channels=self._build_channel_configs(validated_data.get('channels', {})),
                global_settings=validated_data.get('global_settings', {}),
                security_settings=validated_data.get('security_settings', {})
            )
            
            # Encrypt sensitive data
            config = await self.security_manager.encrypt_config(config)
            
            # Save to database
            await self._save_config_to_database(config)
            
            # Cache the new configuration
            cache_key = f"slack_config:{tenant_id}:{environment}"
            await self._cache_config(cache_key, config)
            
            # Notify listeners
            await self._notify_config_change('created', tenant_id, environment)
            
            self.logger.info(f"Created new config for tenant {tenant_id}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to create tenant config for {tenant_id}: {e}")
            raise ConfigurationError(f"Config creation failed: {e}")

    async def update_tenant_config(
        self,
        tenant_id: str,
        environment: str,
        updates: Dict[str, Any]
    ) -> TenantSlackConfig:
        """
        Update existing tenant Slack configuration.
        
        Args:
            tenant_id: Unique tenant identifier
            environment: Environment name
            updates: Configuration updates
            
        Returns:
            Updated tenant configuration
        """
        try:
            # Get current configuration
            current_config = await self.get_tenant_config(tenant_id, environment)
            
            # Validate updates
            validated_updates = await self.config_validator.validate_update_data(updates)
            
            # Apply updates
            updated_config = self._apply_config_updates(current_config, validated_updates)
            
            # Encrypt sensitive data
            updated_config = await self.security_manager.encrypt_config(updated_config)
            
            # Save to database
            await self._save_config_to_database(updated_config)
            
            # Invalidate and update cache
            cache_key = f"slack_config:{tenant_id}:{environment}"
            await self._invalidate_cache(cache_key)
            await self._cache_config(cache_key, updated_config)
            
            # Notify listeners
            await self._notify_config_change('updated', tenant_id, environment)
            
            self.logger.info(f"Updated config for tenant {tenant_id}")
            return updated_config
            
        except Exception as e:
            self.logger.error(f"Failed to update tenant config for {tenant_id}: {e}")
            raise ConfigurationError(f"Config update failed: {e}")

    async def delete_tenant_config(
        self,
        tenant_id: str,
        environment: str
    ) -> bool:
        """
        Delete tenant Slack configuration.
        
        Args:
            tenant_id: Unique tenant identifier
            environment: Environment name
            
        Returns:
            True if deletion successful
        """
        try:
            # Delete from database
            deleted = await self._delete_config_from_database(tenant_id, environment)
            
            if deleted:
                # Invalidate cache
                cache_key = f"slack_config:{tenant_id}:{environment}"
                await self._invalidate_cache(cache_key)
                
                # Notify listeners
                await self._notify_config_change('deleted', tenant_id, environment)
                
                self.logger.info(f"Deleted config for tenant {tenant_id}")
            
            return deleted
            
        except Exception as e:
            self.logger.error(f"Failed to delete tenant config for {tenant_id}: {e}")
            raise ConfigurationError(f"Config deletion failed: {e}")

    async def add_channel_config(
        self,
        tenant_id: str,
        environment: str,
        channel_name: str,
        channel_config: Dict[str, Any]
    ) -> SlackChannelConfig:
        """Add a new channel configuration to a tenant."""
        try:
            # Get current configuration
            config = await self.get_tenant_config(tenant_id, environment)
            
            # Validate channel configuration
            validated_config = await self.config_validator.validate_channel_config(channel_config)
            
            # Create channel config object
            channel = SlackChannelConfig(
                channel_id=validated_config['channel_id'],
                channel_name=channel_name,
                webhook_url=validated_config['webhook_url'],
                token=validated_config['token'],
                priority_level=validated_config.get('priority_level', 'medium'),
                rate_limit=validated_config.get('rate_limit', 100),
                enabled=validated_config.get('enabled', True)
            )
            
            # Add to tenant configuration
            config.channels[channel_name] = channel
            config.updated_at = datetime.utcnow()
            
            # Save updated configuration
            await self.update_tenant_config(tenant_id, environment, {
                'channels': {name: asdict(ch) for name, ch in config.channels.items()}
            })
            
            self.logger.info(f"Added channel {channel_name} to tenant {tenant_id}")
            return channel
            
        except Exception as e:
            self.logger.error(f"Failed to add channel config: {e}")
            raise ConfigurationError(f"Channel addition failed: {e}")

    async def remove_channel_config(
        self,
        tenant_id: str,
        environment: str,
        channel_name: str
    ) -> bool:
        """Remove a channel configuration from a tenant."""
        try:
            # Get current configuration
            config = await self.get_tenant_config(tenant_id, environment)
            
            if channel_name not in config.channels:
                return False
            
            # Remove channel
            del config.channels[channel_name]
            config.updated_at = datetime.utcnow()
            
            # Save updated configuration
            await self.update_tenant_config(tenant_id, environment, {
                'channels': {name: asdict(ch) for name, ch in config.channels.items()}
            })
            
            self.logger.info(f"Removed channel {channel_name} from tenant {tenant_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove channel config: {e}")
            raise ConfigurationError(f"Channel removal failed: {e}")

    async def list_tenant_configs(
        self,
        environment: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List all tenant configurations with pagination."""
        try:
            configs = await self._list_configs_from_database(environment, limit, offset)
            
            # Decrypt sensitive data for response
            decrypted_configs = []
            for config in configs:
                decrypted_config = await self.security_manager.decrypt_config(config)
                decrypted_configs.append({
                    'tenant_id': decrypted_config.tenant_id,
                    'environment': decrypted_config.environment,
                    'default_locale': decrypted_config.default_locale,
                    'channel_count': len(decrypted_config.channels),
                    'created_at': decrypted_config.created_at.isoformat(),
                    'updated_at': decrypted_config.updated_at.isoformat()
                })
            
            return decrypted_configs
            
        except Exception as e:
            self.logger.error(f"Failed to list tenant configs: {e}")
            raise ConfigurationError(f"Config listing failed: {e}")

    async def reload_config(
        self,
        tenant_id: str,
        environment: str
    ) -> TenantSlackConfig:
        """Force reload configuration from database, bypassing cache."""
        cache_key = f"slack_config:{tenant_id}:{environment}"
        await self._invalidate_cache(cache_key)
        return await self.get_tenant_config(tenant_id, environment, use_cache=False)

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the configuration manager."""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'components': {}
        }
        
        try:
            # Check Redis connection
            if self.redis_client:
                await self.redis_client.ping()
                health_status['components']['redis'] = 'healthy'
            else:
                health_status['components']['redis'] = 'unavailable'
            
            # Check database connection
            try:
                await self.db_session.execute("SELECT 1")
                health_status['components']['database'] = 'healthy'
            except Exception as e:
                health_status['components']['database'] = f'unhealthy: {e}'
                health_status['status'] = 'degraded'
            
            # Check memory cache stats
            health_status['components']['memory_cache'] = {
                'cached_configs': len(self._memory_cache),
                'cache_hits': getattr(self, '_cache_hits', 0),
                'cache_misses': getattr(self, '_cache_misses', 0)
            }
            
            # Performance metrics
            if self.performance_monitor:
                metrics = await self.performance_monitor.get_metrics()
                health_status['metrics'] = metrics
            
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['error'] = str(e)
        
        return health_status

    # Private helper methods
    
    def _build_channel_configs(self, channels_data: Dict[str, Any]) -> Dict[str, SlackChannelConfig]:
        """Build channel configuration objects from data."""
        channels = {}
        for name, data in channels_data.items():
            channels[name] = SlackChannelConfig(
                channel_id=data['channel_id'],
                channel_name=name,
                webhook_url=data['webhook_url'],
                token=data['token'],
                priority_level=data.get('priority_level', 'medium'),
                rate_limit=data.get('rate_limit', 100),
                enabled=data.get('enabled', True)
            )
        return channels

    def _apply_config_updates(
        self,
        config: TenantSlackConfig,
        updates: Dict[str, Any]
    ) -> TenantSlackConfig:
        """Apply updates to configuration object."""
        if 'default_locale' in updates:
            config.default_locale = updates['default_locale']
        
        if 'channels' in updates:
            config.channels = self._build_channel_configs(updates['channels'])
        
        if 'global_settings' in updates:
            config.global_settings.update(updates['global_settings'])
        
        if 'security_settings' in updates:
            config.security_settings.update(updates['security_settings'])
        
        config.updated_at = datetime.utcnow()
        return config

    async def _cache_config(self, cache_key: str, config: TenantSlackConfig) -> None:
        """Cache configuration in Redis and memory."""
        try:
            # Serialize configuration
            config_data = self._serialize_config(config)
            
            # Cache in Redis
            if self.redis_client:
                await self.redis_client.setex(
                    cache_key,
                    self.cache_ttl,
                    json.dumps(config_data)
                )
            
            # Cache in memory
            self._memory_cache[cache_key] = config
            self._cache_timestamps[cache_key] = datetime.utcnow()
            
        except Exception as e:
            self.logger.warning(f"Failed to cache config {cache_key}: {e}")

    async def _invalidate_cache(self, cache_key: str) -> None:
        """Invalidate cached configuration."""
        try:
            # Remove from Redis
            if self.redis_client:
                await self.redis_client.delete(cache_key)
            
            # Remove from memory
            self._memory_cache.pop(cache_key, None)
            self._cache_timestamps.pop(cache_key, None)
            
        except Exception as e:
            self.logger.warning(f"Failed to invalidate cache {cache_key}: {e}")

    def _serialize_config(self, config: TenantSlackConfig) -> Dict[str, Any]:
        """Serialize configuration for caching."""
        return {
            'tenant_id': config.tenant_id,
            'environment': config.environment,
            'default_locale': config.default_locale,
            'channels': {name: asdict(channel) for name, channel in config.channels.items()},
            'global_settings': config.global_settings,
            'security_settings': config.security_settings,
            'created_at': config.created_at.isoformat(),
            'updated_at': config.updated_at.isoformat()
        }

    def _deserialize_config(self, data: Dict[str, Any]) -> TenantSlackConfig:
        """Deserialize configuration from cache."""
        channels = {}
        for name, channel_data in data['channels'].items():
            channel_data['created_at'] = datetime.fromisoformat(channel_data['created_at'])
            channel_data['updated_at'] = datetime.fromisoformat(channel_data['updated_at'])
            channels[name] = SlackChannelConfig(**channel_data)
        
        return TenantSlackConfig(
            tenant_id=data['tenant_id'],
            environment=data['environment'],
            default_locale=data['default_locale'],
            channels=channels,
            global_settings=data['global_settings'],
            security_settings=data['security_settings'],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at'])
        )

    async def _notify_config_change(
        self,
        action: str,
        tenant_id: str,
        environment: str
    ) -> None:
        """Notify listeners of configuration changes."""
        for listener in self._change_listeners:
            try:
                await listener(action, tenant_id, environment)
            except Exception as e:
                self.logger.warning(f"Config change listener failed: {e}")

    def add_change_listener(self, listener: callable) -> None:
        """Add a configuration change listener."""
        self._change_listeners.append(listener)

    def remove_change_listener(self, listener: callable) -> None:
        """Remove a configuration change listener."""
        if listener in self._change_listeners:
            self._change_listeners.remove(listener)

    # Database operation placeholders (implement based on your ORM)
    
    async def _load_config_from_database(
        self,
        tenant_id: str,
        environment: str
    ) -> Optional[TenantSlackConfig]:
        """Load configuration from database."""
        # Implement database loading logic
        pass

    async def _save_config_to_database(self, config: TenantSlackConfig) -> None:
        """Save configuration to database."""
        # Implement database saving logic
        pass

    async def _delete_config_from_database(
        self,
        tenant_id: str,
        environment: str
    ) -> bool:
        """Delete configuration from database."""
        # Implement database deletion logic
        pass

    async def _list_configs_from_database(
        self,
        environment: Optional[str],
        limit: int,
        offset: int
    ) -> List[TenantSlackConfig]:
        """List configurations from database."""
        # Implement database listing logic
        pass

    async def close(self) -> None:
        """Clean up resources."""
        try:
            if self.redis_client:
                await self.redis_client.close()
            
            if self.performance_monitor:
                await self.performance_monitor.close()
            
            self.logger.info("SlackConfigManager closed successfully")
            
        except Exception as e:
            self.logger.error(f"Error closing SlackConfigManager: {e}")
