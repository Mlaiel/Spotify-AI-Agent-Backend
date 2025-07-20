#!/usr/bin/env python3
"""
Spotify AI Agent - Tenant Initialization Script
==============================================

Comprehensive tenant initialization script that sets up:
- Database schemas and tables
- Initial configuration data  
- Default user roles and permissions
- Spotify API integration
- AI model configurations
- Monitoring and analytics setup

Usage:
    python -m app.tenancy.fixtures.scripts.init_tenant --tenant-id mycompany --tier premium
    python init_tenant.py --tenant-id startup --tier basic --config custom_config.yaml

Author: Expert Development Team (Fahed Mlaiel)
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_async_session
from app.core.cache import get_redis_client
from app.tenancy.fixtures.base import FixtureManager
from app.tenancy.fixtures.tenant_fixtures import TenantFixture
from app.tenancy.fixtures.schema_fixtures import SchemaFixture
from app.tenancy.fixtures.config_fixtures import ConfigFixture
from app.tenancy.fixtures.data_loaders import SpotifyDataLoader, AIModelLoader
from app.tenancy.fixtures.validators import TenantValidator, DataValidator
from app.tenancy.fixtures.monitoring import FixtureMonitor
from app.tenancy.fixtures.utils import FixtureUtils, TenantUtils, ConfigUtils
from app.tenancy.fixtures.exceptions import FixtureError
from app.tenancy.fixtures.constants import SUPPORTED_TIERS, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class TenantInitializer:
    """
    Complete tenant initialization manager.
    
    Orchestrates the entire tenant setup process including:
    - Schema creation
    - Configuration setup
    - Initial data loading
    - Validation
    - Monitoring setup
    """
    
    def __init__(self, session: AsyncSession, redis_client=None):
        self.session = session
        self.redis_client = redis_client
        self.fixture_manager = FixtureManager(session, redis_client)
        self.monitor = FixtureMonitor(session, redis_client)
        
        # Initialize fixture components
        self.tenant_fixture = TenantFixture(session, redis_client)
        self.schema_fixture = SchemaFixture(session, redis_client)
        self.config_fixture = ConfigFixture(session, redis_client)
        
        # Initialize data loaders
        self.spotify_loader = SpotifyDataLoader(session, redis_client)
        self.ai_loader = AIModelLoader(session, redis_client)
        
        # Initialize validators
        self.tenant_validator = TenantValidator(session)
        self.data_validator = DataValidator(session)
    
    async def initialize_tenant(
        self,
        tenant_id: str,
        tier: str = "basic",
        config_overrides: Optional[Dict[str, Any]] = None,
        custom_data: Optional[Dict[str, Any]] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Initialize a complete tenant setup.
        
        Args:
            tenant_id: Unique tenant identifier
            tier: Subscription tier (free, basic, premium, enterprise)
            config_overrides: Custom configuration overrides
            custom_data: Custom initial data
            dry_run: If True, perform validation without actual changes
            
        Returns:
            Dictionary with initialization results and metrics
        """
        start_time = datetime.now(timezone.utc)
        init_result = {
            "tenant_id": tenant_id,
            "tier": tier,
            "status": "started",
            "start_time": start_time.isoformat(),
            "steps_completed": [],
            "steps_failed": [],
            "warnings": [],
            "resources_created": {}
        }
        
        try:
            # Step 1: Validate inputs
            logger.info(f"Starting tenant initialization for: {tenant_id} (tier: {tier})")
            await self._validate_inputs(tenant_id, tier)
            init_result["steps_completed"].append("input_validation")
            
            # Step 2: Check if tenant already exists
            exists = await self._check_tenant_exists(tenant_id)
            if exists and not dry_run:
                raise FixtureError(f"Tenant {tenant_id} already exists")
            
            if exists:
                init_result["warnings"].append("Tenant already exists (dry run mode)")
            
            init_result["steps_completed"].append("existence_check")
            
            # Step 3: Create database schema
            if not dry_run:
                schema_result = await self.schema_fixture.create_tenant_schema(tenant_id)
                init_result["resources_created"]["schema"] = schema_result
            else:
                logger.info(f"DRY RUN: Would create schema for {tenant_id}")
            
            init_result["steps_completed"].append("schema_creation")
            
            # Step 4: Setup tenant configuration
            config_data = await self._prepare_tenant_config(tenant_id, tier, config_overrides)
            
            if not dry_run:
                config_result = await self.config_fixture.setup_tenant_config(
                    tenant_id, config_data
                )
                init_result["resources_created"]["config"] = config_result
            else:
                logger.info(f"DRY RUN: Would setup config for {tenant_id}")
            
            init_result["steps_completed"].append("config_setup")
            
            # Step 5: Create tenant record
            if not dry_run:
                tenant_result = await self.tenant_fixture.create_tenant(
                    tenant_id, tier, config_data
                )
                init_result["resources_created"]["tenant"] = tenant_result
            else:
                logger.info(f"DRY RUN: Would create tenant record for {tenant_id}")
            
            init_result["steps_completed"].append("tenant_creation")
            
            # Step 6: Initialize data structures
            if not dry_run:
                await self._initialize_tenant_data(tenant_id, tier, custom_data)
            else:
                logger.info(f"DRY RUN: Would initialize data for {tenant_id}")
            
            init_result["steps_completed"].append("data_initialization")
            
            # Step 7: Setup Spotify integration
            if not dry_run:
                spotify_result = await self._setup_spotify_integration(tenant_id, tier)
                init_result["resources_created"]["spotify"] = spotify_result
            else:
                logger.info(f"DRY RUN: Would setup Spotify integration for {tenant_id}")
            
            init_result["steps_completed"].append("spotify_integration")
            
            # Step 8: Configure AI models
            if not dry_run:
                ai_result = await self._setup_ai_models(tenant_id, tier)
                init_result["resources_created"]["ai_models"] = ai_result
            else:
                logger.info(f"DRY RUN: Would configure AI models for {tenant_id}")
            
            init_result["steps_completed"].append("ai_configuration")
            
            # Step 9: Setup monitoring
            if not dry_run:
                monitoring_result = await self._setup_monitoring(tenant_id)
                init_result["resources_created"]["monitoring"] = monitoring_result
            else:
                logger.info(f"DRY RUN: Would setup monitoring for {tenant_id}")
            
            init_result["steps_completed"].append("monitoring_setup")
            
            # Step 10: Final validation
            if not dry_run:
                validation_result = await self._validate_tenant_setup(tenant_id)
                init_result["validation"] = validation_result
            else:
                logger.info(f"DRY RUN: Would validate setup for {tenant_id}")
            
            init_result["steps_completed"].append("final_validation")
            
            # Calculate timing and finalize
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            init_result.update({
                "status": "completed",
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "summary": await self._generate_summary(tenant_id, init_result)
            })
            
            logger.info(
                f"Tenant initialization completed for {tenant_id} in "
                f"{FixtureUtils.format_duration(duration)}"
            )
            
        except Exception as e:
            init_result["status"] = "failed"
            init_result["error"] = str(e)
            logger.error(f"Tenant initialization failed for {tenant_id}: {e}")
            raise
        
        return init_result
    
    async def _validate_inputs(self, tenant_id: str, tier: str) -> None:
        """Validate initialization inputs."""
        if not TenantUtils.validate_tenant_id(tenant_id):
            raise FixtureError(f"Invalid tenant ID format: {tenant_id}")
        
        if tier not in SUPPORTED_TIERS:
            raise FixtureError(f"Unsupported tier: {tier}")
    
    async def _check_tenant_exists(self, tenant_id: str) -> bool:
        """Check if tenant already exists."""
        return await self.tenant_fixture.tenant_exists(tenant_id)
    
    async def _prepare_tenant_config(
        self,
        tenant_id: str,
        tier: str,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Prepare tenant configuration."""
        # Start with tier-specific defaults
        tier_config = DEFAULT_CONFIG["tiers"][tier].copy()
        
        # Add tenant-specific settings
        tenant_config = {
            "tenant_id": tenant_id,
            "tier": tier,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "schema_name": TenantUtils.get_tenant_schema_name(tenant_id),
            "cache_namespace": TenantUtils.get_tenant_cache_namespace(tenant_id),
            "storage_path": str(TenantUtils.get_tenant_storage_path(tenant_id))
        }
        
        # Merge configurations
        config = ConfigUtils.merge_configs(tier_config, tenant_config)
        
        if config_overrides:
            config = ConfigUtils.merge_configs(config, config_overrides)
        
        return config
    
    async def _initialize_tenant_data(
        self,
        tenant_id: str,
        tier: str,
        custom_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize tenant data structures."""
        schema = TenantUtils.get_tenant_schema_name(tenant_id)
        
        # Create basic tables and indexes
        await self.schema_fixture.create_tenant_tables(tenant_id)
        await self.schema_fixture.create_tenant_indexes(tenant_id)
        
        # Insert default data
        default_data = {
            "roles": [
                {"name": "admin", "permissions": ["*"]},
                {"name": "user", "permissions": ["read", "create_content"]},
                {"name": "collaborator", "permissions": ["read", "write", "collaborate"]}
            ],
            "settings": {
                "ui_theme": "default",
                "language": "en",
                "timezone": "UTC"
            }
        }
        
        if custom_data:
            default_data.update(custom_data)
        
        # Load default data
        await self.fixture_manager.load_fixtures(tenant_id, default_data)
    
    async def _setup_spotify_integration(
        self,
        tenant_id: str,
        tier: str
    ) -> Dict[str, Any]:
        """Setup Spotify API integration."""
        spotify_config = {
            "tenant_id": tenant_id,
            "api_rate_limits": DEFAULT_CONFIG["tiers"][tier]["spotify"]["rate_limits"],
            "features_enabled": DEFAULT_CONFIG["tiers"][tier]["spotify"]["features"],
            "cache_ttl": DEFAULT_CONFIG["tiers"][tier]["spotify"]["cache_ttl"]
        }
        
        result = await self.spotify_loader.initialize_spotify_config(
            tenant_id, spotify_config
        )
        
        return result
    
    async def _setup_ai_models(
        self,
        tenant_id: str,
        tier: str
    ) -> Dict[str, Any]:
        """Configure AI models for tenant."""
        ai_config = {
            "tenant_id": tenant_id,
            "models_enabled": DEFAULT_CONFIG["tiers"][tier]["ai"]["models"],
            "processing_limits": DEFAULT_CONFIG["tiers"][tier]["ai"]["limits"],
            "quality_settings": DEFAULT_CONFIG["tiers"][tier]["ai"]["quality"]
        }
        
        result = await self.ai_loader.initialize_ai_config(
            tenant_id, ai_config
        )
        
        return result
    
    async def _setup_monitoring(self, tenant_id: str) -> Dict[str, Any]:
        """Setup monitoring for tenant."""
        monitoring_config = {
            "tenant_id": tenant_id,
            "metrics_enabled": True,
            "alerts_enabled": True,
            "retention_days": 30
        }
        
        result = await self.monitor.setup_tenant_monitoring(
            tenant_id, monitoring_config
        )
        
        return result
    
    async def _validate_tenant_setup(self, tenant_id: str) -> Dict[str, Any]:
        """Validate complete tenant setup."""
        validation_results = {}
        
        # Validate tenant data
        tenant_validation = await self.tenant_validator.validate_tenant(tenant_id)
        validation_results["tenant"] = tenant_validation
        
        # Validate data integrity
        data_validation = await self.data_validator.validate_data_integrity(tenant_id)
        validation_results["data_integrity"] = data_validation
        
        # Check resource limits
        resources = await TenantUtils.calculate_tenant_resources(self.session, tenant_id)
        validation_results["resources"] = resources
        
        return validation_results
    
    async def _generate_summary(
        self,
        tenant_id: str,
        init_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate initialization summary."""
        return {
            "tenant_id": tenant_id,
            "steps_total": len(init_result["steps_completed"]) + len(init_result["steps_failed"]),
            "steps_successful": len(init_result["steps_completed"]),
            "steps_failed": len(init_result["steps_failed"]),
            "warnings_count": len(init_result["warnings"]),
            "resources_created_count": len(init_result.get("resources_created", {})),
            "success_rate": len(init_result["steps_completed"]) / max(1, len(init_result["steps_completed"]) + len(init_result["steps_failed"]))
        }


async def init_tenant_fixtures(
    tenant_id: str,
    tier: str = "basic",
    config_file: Optional[str] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Main function to initialize tenant fixtures.
    
    Args:
        tenant_id: Tenant identifier
        tier: Subscription tier
        config_file: Optional configuration file path
        dry_run: Perform validation without changes
        
    Returns:
        Initialization results
    """
    # Load configuration overrides if provided
    config_overrides = None
    if config_file:
        config_path = Path(config_file)
        if config_path.exists():
            if config_path.suffix.lower() == '.yaml':
                config_overrides = FixtureUtils.load_yaml_file(config_path)
            elif config_path.suffix.lower() == '.json':
                config_overrides = FixtureUtils.load_json_file(config_path)
    
    # Get database session and Redis client
    async with get_async_session() as session:
        redis_client = await get_redis_client()
        
        try:
            initializer = TenantInitializer(session, redis_client)
            result = await initializer.initialize_tenant(
                tenant_id=tenant_id,
                tier=tier,
                config_overrides=config_overrides,
                dry_run=dry_run
            )
            
            return result
            
        finally:
            if redis_client:
                await redis_client.close()


def main():
    """Command line interface for tenant initialization."""
    parser = argparse.ArgumentParser(
        description="Initialize tenant with complete fixture setup"
    )
    
    parser.add_argument(
        "--tenant-id",
        required=True,
        help="Unique tenant identifier"
    )
    
    parser.add_argument(
        "--tier",
        choices=SUPPORTED_TIERS,
        default="basic",
        help="Subscription tier"
    )
    
    parser.add_argument(
        "--config",
        help="Configuration file path (YAML or JSON)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform validation without making changes"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        # Run initialization
        result = asyncio.run(
            init_tenant_fixtures(
                tenant_id=args.tenant_id,
                tier=args.tier,
                config_file=args.config,
                dry_run=args.dry_run
            )
        )
        
        # Display results
        print(f"\nTenant Initialization Results for '{args.tenant_id}':")
        print(f"Status: {result['status']}")
        print(f"Duration: {FixtureUtils.format_duration(result.get('duration_seconds', 0))}")
        print(f"Steps Completed: {len(result['steps_completed'])}")
        
        if result['steps_failed']:
            print(f"Steps Failed: {len(result['steps_failed'])}")
        
        if result['warnings']:
            print(f"Warnings: {len(result['warnings'])}")
        
        if result['status'] == 'completed':
            print("\n✅ Tenant initialization completed successfully!")
            sys.exit(0)
        else:
            print(f"\n❌ Tenant initialization failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Initialization interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
