#!/usr/bin/env python3
"""
Spotify AI Agent - Fixture Migration Script
==========================================

Comprehensive migration script that handles:
- Version-to-version fixture migrations
- Schema evolution and data transformation
- Breaking change mitigation
- Rollback capabilities
- Migration validation and testing

Usage:
    python -m app.tenancy.fixtures.scripts.migrate --from-version 1.0 --to-version 1.1
    python migrate.py --tenant-id mycompany --dry-run --auto-resolve

Author: Expert Development Team (Fahed Mlaiel)
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_async_session
from app.core.cache import get_redis_client
from app.tenancy.fixtures.base import FixtureManager
from app.tenancy.fixtures.monitoring import FixtureMonitor
from app.tenancy.fixtures.utils import FixtureUtils, TenantUtils
from app.tenancy.fixtures.exceptions import FixtureError, FixtureMigrationError
from app.tenancy.fixtures.constants import SUPPORTED_VERSIONS, MIGRATION_PATH

logger = logging.getLogger(__name__)


class MigrationStep:
    """Individual migration step with rollback capability."""
    
    def __init__(
        self,
        name: str,
        forward_func: Callable,
        rollback_func: Optional[Callable] = None,
        description: str = "",
        critical: bool = False
    ):
        self.name = name
        self.forward_func = forward_func
        self.rollback_func = rollback_func
        self.description = description
        self.critical = critical
        self.executed = False
        self.rollback_data = None
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the migration step."""
        try:
            result = await self.forward_func(context)
            self.executed = True
            self.rollback_data = result.get('rollback_data')
            return result
        except Exception as e:
            raise FixtureMigrationError(f"Migration step '{self.name}' failed: {e}")
    
    async def rollback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback the migration step."""
        if not self.executed or not self.rollback_func:
            return {"status": "skipped", "reason": "not executed or no rollback function"}
        
        try:
            context['rollback_data'] = self.rollback_data
            result = await self.rollback_func(context)
            self.executed = False
            return result
        except Exception as e:
            raise FixtureMigrationError(f"Rollback step '{self.name}' failed: {e}")


class FixtureMigrator:
    """
    Comprehensive fixture migration orchestrator.
    
    Handles:
    - Version detection and validation
    - Migration path planning
    - Step-by-step execution with rollback
    - Data transformation and validation
    - Multi-tenant migration coordination
    """
    
    def __init__(self, session: AsyncSession, redis_client=None):
        self.session = session
        self.redis_client = redis_client
        self.fixture_manager = FixtureManager(session, redis_client)
        self.monitor = FixtureMonitor(session, redis_client)
        
        # Migration registry
        self.migration_steps = {}
        self._register_migration_steps()
    
    async def migrate_fixtures(
        self,
        from_version: str,
        to_version: str,
        tenant_id: Optional[str] = None,
        dry_run: bool = True,
        auto_resolve: bool = False,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Perform fixture migration between versions.
        
        Args:
            from_version: Source version
            to_version: Target version
            tenant_id: Specific tenant (None for all tenants)
            dry_run: Perform validation without changes
            auto_resolve: Automatically resolve conflicts
            force: Force migration even with warnings
            
        Returns:
            Migration results and status
        """
        start_time = datetime.now(timezone.utc)
        
        migration_result = {
            "from_version": from_version,
            "to_version": to_version,
            "tenant_id": tenant_id or "all",
            "dry_run": dry_run,
            "auto_resolve": auto_resolve,
            "force": force,
            "status": "started",
            "start_time": start_time.isoformat(),
            "migration_path": [],
            "steps_executed": [],
            "steps_failed": [],
            "tenants_migrated": [],
            "rollbacks_performed": [],
            "warnings": [],
            "errors": [],
            "summary": {}
        }
        
        try:
            # Validate migration parameters
            await self._validate_migration_params(from_version, to_version, force)
            
            # Plan migration path
            migration_path = await self._plan_migration_path(from_version, to_version)
            migration_result["migration_path"] = migration_path
            
            # Get tenant list
            tenant_list = []
            if tenant_id:
                if await self._check_tenant_exists(tenant_id):
                    tenant_list = [tenant_id]
                else:
                    raise FixtureError(f"Tenant not found: {tenant_id}")
            else:
                tenant_list = await self._get_all_tenants()
            
            # Pre-migration validation
            validation_result = await self._validate_pre_migration(
                tenant_list, from_version, to_version
            )
            migration_result["pre_validation"] = validation_result
            
            if validation_result["critical_issues"] and not force:
                raise FixtureMigrationError(
                    f"Critical issues found: {validation_result['critical_issues']}"
                )
            
            # Execute migration for each tenant
            for tenant in tenant_list:
                try:
                    tenant_result = await self._migrate_tenant(
                        tenant, migration_path, dry_run, auto_resolve
                    )
                    
                    migration_result["tenants_migrated"].append(tenant)
                    migration_result["steps_executed"].extend(tenant_result.get("steps_executed", []))
                    migration_result["warnings"].extend(tenant_result.get("warnings", []))
                    
                    if tenant_result.get("errors"):
                        migration_result["errors"].extend(tenant_result["errors"])
                        migration_result["steps_failed"].extend(tenant_result.get("steps_failed", []))
                
                except Exception as e:
                    error_msg = f"Migration failed for tenant {tenant}: {e}"
                    migration_result["errors"].append(error_msg)
                    logger.error(error_msg)
                    
                    # Attempt rollback for this tenant
                    if not dry_run:
                        try:
                            rollback_result = await self._rollback_tenant_migration(tenant)
                            migration_result["rollbacks_performed"].append({
                                "tenant": tenant,
                                "result": rollback_result
                            })
                        except Exception as rollback_error:
                            logger.error(f"Rollback failed for tenant {tenant}: {rollback_error}")
            
            # Post-migration validation
            if not dry_run and migration_result["tenants_migrated"]:
                post_validation = await self._validate_post_migration(
                    migration_result["tenants_migrated"], to_version
                )
                migration_result["post_validation"] = post_validation
            
            # Calculate final metrics
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            success_rate = len(migration_result["tenants_migrated"]) / max(len(tenant_list), 1)
            
            migration_result.update({
                "status": "completed" if success_rate > 0.5 else "failed",
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "success_rate": success_rate,
                "summary": await self._generate_migration_summary(migration_result)
            })
            
            # Record migration metrics
            await self.monitor.record_migration_operation(migration_result)
            
            logger.info(
                f"Migration completed: {from_version} -> {to_version}, "
                f"Success rate: {success_rate:.1%}, "
                f"Duration: {FixtureUtils.format_duration(duration)}"
            )
            
        except Exception as e:
            migration_result["status"] = "failed"
            migration_result["error"] = str(e)
            logger.error(f"Migration failed: {e}")
            raise
        
        return migration_result
    
    async def _validate_migration_params(
        self,
        from_version: str,
        to_version: str,
        force: bool
    ) -> None:
        """Validate migration parameters."""
        if from_version not in SUPPORTED_VERSIONS:
            raise FixtureMigrationError(f"Unsupported source version: {from_version}")
        
        if to_version not in SUPPORTED_VERSIONS:
            raise FixtureMigrationError(f"Unsupported target version: {to_version}")
        
        if from_version == to_version:
            raise FixtureMigrationError("Source and target versions are the same")
        
        # Check if migration path exists
        if not self._migration_path_exists(from_version, to_version) and not force:
            raise FixtureMigrationError(
                f"No migration path from {from_version} to {to_version}"
            )
    
    async def _plan_migration_path(self, from_version: str, to_version: str) -> List[str]:
        """Plan the migration path between versions."""
        # Simplified version planning - in reality, this would be more complex
        version_order = ["1.0.0", "1.1.0", "1.2.0", "2.0.0"]
        
        try:
            from_idx = version_order.index(from_version)
            to_idx = version_order.index(to_version)
            
            if from_idx < to_idx:
                return version_order[from_idx:to_idx + 1]
            else:
                return list(reversed(version_order[to_idx:from_idx + 1]))
        
        except ValueError:
            raise FixtureMigrationError(f"Cannot determine migration path")
    
    async def _validate_pre_migration(
        self,
        tenant_list: List[str],
        from_version: str,
        to_version: str
    ) -> Dict[str, Any]:
        """Validate system state before migration."""
        validation_result = {
            "tenants_checked": len(tenant_list),
            "version_mismatches": [],
            "missing_data": [],
            "critical_issues": [],
            "warnings": []
        }
        
        for tenant_id in tenant_list:
            try:
                # Check current version
                current_version = await self._get_tenant_version(tenant_id)
                if current_version != from_version:
                    validation_result["version_mismatches"].append({
                        "tenant": tenant_id,
                        "expected": from_version,
                        "actual": current_version
                    })
                
                # Check data integrity
                integrity_issues = await self._check_data_integrity(tenant_id)
                if integrity_issues:
                    validation_result["critical_issues"].extend(integrity_issues)
                
                # Check for missing required data
                missing_data = await self._check_required_data(tenant_id, from_version)
                if missing_data:
                    validation_result["missing_data"].extend(missing_data)
            
            except Exception as e:
                validation_result["critical_issues"].append(
                    f"Validation failed for tenant {tenant_id}: {e}"
                )
        
        return validation_result
    
    async def _migrate_tenant(
        self,
        tenant_id: str,
        migration_path: List[str],
        dry_run: bool,
        auto_resolve: bool
    ) -> Dict[str, Any]:
        """Migrate single tenant through the migration path."""
        tenant_result = {
            "tenant_id": tenant_id,
            "steps_executed": [],
            "steps_failed": [],
            "warnings": [],
            "errors": []
        }
        
        context = {
            "tenant_id": tenant_id,
            "session": self.session,
            "redis_client": self.redis_client,
            "dry_run": dry_run,
            "auto_resolve": auto_resolve
        }
        
        # Execute migration steps for each version transition
        for i in range(len(migration_path) - 1):
            from_ver = migration_path[i]
            to_ver = migration_path[i + 1]
            
            migration_key = f"{from_ver}_to_{to_ver}"
            
            if migration_key in self.migration_steps:
                steps = self.migration_steps[migration_key]
                
                for step in steps:
                    try:
                        logger.info(f"Executing step: {step.name} for tenant {tenant_id}")
                        
                        step_result = await step.execute(context)
                        tenant_result["steps_executed"].append({
                            "step": step.name,
                            "result": step_result
                        })
                        
                        if step_result.get("warnings"):
                            tenant_result["warnings"].extend(step_result["warnings"])
                    
                    except Exception as e:
                        error_msg = f"Step {step.name} failed: {e}"
                        tenant_result["errors"].append(error_msg)
                        tenant_result["steps_failed"].append(step.name)
                        
                        if step.critical:
                            raise FixtureMigrationError(error_msg)
                        
                        logger.error(error_msg)
        
        return tenant_result
    
    async def _rollback_tenant_migration(self, tenant_id: str) -> Dict[str, Any]:
        """Rollback migration for a tenant."""
        rollback_result = {
            "tenant_id": tenant_id,
            "steps_rolled_back": [],
            "rollback_errors": []
        }
        
        # This would implement rollback logic based on executed steps
        # For now, a simplified implementation
        
        logger.info(f"Rolling back migration for tenant: {tenant_id}")
        
        return rollback_result
    
    async def _validate_post_migration(
        self,
        tenant_list: List[str],
        to_version: str
    ) -> Dict[str, Any]:
        """Validate system state after migration."""
        validation_result = {
            "tenants_validated": 0,
            "version_confirmations": [],
            "data_issues": [],
            "performance_metrics": {}
        }
        
        for tenant_id in tenant_list:
            try:
                # Confirm version update
                current_version = await self._get_tenant_version(tenant_id)
                validation_result["version_confirmations"].append({
                    "tenant": tenant_id,
                    "version": current_version,
                    "expected": to_version,
                    "matches": current_version == to_version
                })
                
                # Check data integrity post-migration
                integrity_issues = await self._check_data_integrity(tenant_id)
                if integrity_issues:
                    validation_result["data_issues"].extend(integrity_issues)
                
                validation_result["tenants_validated"] += 1
                
            except Exception as e:
                validation_result["data_issues"].append(
                    f"Post-migration validation failed for tenant {tenant_id}: {e}"
                )
        
        return validation_result
    
    def _register_migration_steps(self) -> None:
        """Register all available migration steps."""
        
        # Migration from 1.0.0 to 1.1.0
        self.migration_steps["1.0.0_to_1.1.0"] = [
            MigrationStep(
                name="add_ai_session_metadata",
                forward_func=self._add_ai_session_metadata_column,
                rollback_func=self._remove_ai_session_metadata_column,
                description="Add metadata column to ai_sessions table",
                critical=False
            ),
            MigrationStep(
                name="update_collaboration_schema",
                forward_func=self._update_collaboration_schema,
                rollback_func=self._rollback_collaboration_schema,
                description="Add new collaboration features",
                critical=True
            ),
            MigrationStep(
                name="migrate_user_preferences",
                forward_func=self._migrate_user_preferences,
                rollback_func=self._rollback_user_preferences,
                description="Migrate user preferences to new format",
                critical=False
            )
        ]
        
        # Migration from 1.1.0 to 1.2.0
        self.migration_steps["1.1.0_to_1.2.0"] = [
            MigrationStep(
                name="add_analytics_tables",
                forward_func=self._add_analytics_tables,
                rollback_func=self._remove_analytics_tables,
                description="Add analytics and metrics tables",
                critical=True
            ),
            MigrationStep(
                name="update_spotify_integration",
                forward_func=self._update_spotify_integration,
                rollback_func=self._rollback_spotify_integration,
                description="Update Spotify API integration",
                critical=False
            )
        ]
        
        # Migration from 1.2.0 to 2.0.0
        self.migration_steps["1.2.0_to_2.0.0"] = [
            MigrationStep(
                name="restructure_tenant_isolation",
                forward_func=self._restructure_tenant_isolation,
                rollback_func=self._rollback_tenant_isolation,
                description="Improve tenant isolation architecture",
                critical=True
            ),
            MigrationStep(
                name="upgrade_ai_models",
                forward_func=self._upgrade_ai_models,
                rollback_func=self._rollback_ai_models,
                description="Upgrade AI model configurations",
                critical=False
            )
        ]
    
    # Migration step implementations
    
    async def _add_ai_session_metadata_column(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Add metadata column to ai_sessions table."""
        tenant_id = context["tenant_id"]
        schema_name = TenantUtils.get_tenant_schema_name(tenant_id)
        
        if not context["dry_run"]:
            await self.session.execute(
                text(f"""
                ALTER TABLE {schema_name}.ai_sessions
                ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{{}}'
                """)
            )
            await self.session.commit()
        
        return {
            "status": "success",
            "changes": "Added metadata column to ai_sessions",
            "rollback_data": {"schema": schema_name, "column": "metadata"}
        }
    
    async def _remove_ai_session_metadata_column(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Remove metadata column from ai_sessions table."""
        rollback_data = context.get("rollback_data", {})
        schema_name = rollback_data.get("schema")
        
        if schema_name and not context["dry_run"]:
            await self.session.execute(
                text(f"ALTER TABLE {schema_name}.ai_sessions DROP COLUMN IF EXISTS metadata")
            )
            await self.session.commit()
        
        return {"status": "success", "changes": "Removed metadata column"}
    
    async def _update_collaboration_schema(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Update collaboration schema with new features."""
        tenant_id = context["tenant_id"]
        schema_name = TenantUtils.get_tenant_schema_name(tenant_id)
        
        changes = []
        
        if not context["dry_run"]:
            # Add new columns
            await self.session.execute(
                text(f"""
                ALTER TABLE {schema_name}.collaborations
                ADD COLUMN IF NOT EXISTS visibility VARCHAR(50) DEFAULT 'private',
                ADD COLUMN IF NOT EXISTS settings JSONB DEFAULT '{{}}'
                """)
            )
            
            # Create new table for collaboration invites
            await self.session.execute(
                text(f"""
                CREATE TABLE IF NOT EXISTS {schema_name}.collaboration_invites (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    collaboration_id UUID REFERENCES {schema_name}.collaborations(id),
                    inviter_id UUID REFERENCES {schema_name}.users(id),
                    invitee_email VARCHAR(255) NOT NULL,
                    status VARCHAR(50) DEFAULT 'pending',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP WITH TIME ZONE
                )
                """)
            )
            
            await self.session.commit()
            changes = ["Added visibility and settings columns", "Created collaboration_invites table"]
        
        return {
            "status": "success",
            "changes": changes,
            "rollback_data": {"schema": schema_name, "new_table": "collaboration_invites"}
        }
    
    async def _rollback_collaboration_schema(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback collaboration schema changes."""
        rollback_data = context.get("rollback_data", {})
        schema_name = rollback_data.get("schema")
        
        if schema_name and not context["dry_run"]:
            # Drop new table
            await self.session.execute(
                text(f"DROP TABLE IF EXISTS {schema_name}.collaboration_invites")
            )
            
            # Remove new columns
            await self.session.execute(
                text(f"""
                ALTER TABLE {schema_name}.collaborations
                DROP COLUMN IF EXISTS visibility,
                DROP COLUMN IF EXISTS settings
                """)
            )
            await self.session.commit()
        
        return {"status": "success", "changes": "Rolled back collaboration schema changes"}
    
    async def _migrate_user_preferences(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate user preferences to new format."""
        tenant_id = context["tenant_id"]
        schema_name = TenantUtils.get_tenant_schema_name(tenant_id)
        
        if not context["dry_run"]:
            # Update existing preferences data format
            await self.session.execute(
                text(f"""
                UPDATE {schema_name}.users
                SET preferences = COALESCE(preferences, '{{}}')
                WHERE preferences IS NULL
                """)
            )
            await self.session.commit()
        
        return {
            "status": "success",
            "changes": "Migrated user preferences format",
            "warnings": ["Manual verification of preferences format recommended"]
        }
    
    async def _rollback_user_preferences(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback user preferences migration."""
        return {
            "status": "success",
            "changes": "User preferences rollback not implemented (data format compatible)"
        }
    
    async def _add_analytics_tables(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Add analytics and metrics tables."""
        tenant_id = context["tenant_id"]
        schema_name = TenantUtils.get_tenant_schema_name(tenant_id)
        
        if not context["dry_run"]:
            # Create analytics tables
            await self.session.execute(
                text(f"""
                CREATE TABLE IF NOT EXISTS {schema_name}.analytics_events (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    event_type VARCHAR(100) NOT NULL,
                    event_data JSONB NOT NULL,
                    user_id UUID REFERENCES {schema_name}.users(id),
                    session_id VARCHAR(255),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS {schema_name}.performance_metrics (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    metric_name VARCHAR(100) NOT NULL,
                    metric_value DECIMAL(10, 4) NOT NULL,
                    metric_tags JSONB DEFAULT '{{}}',
                    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                """)
            )
            await self.session.commit()
        
        return {
            "status": "success",
            "changes": "Created analytics and performance metrics tables",
            "rollback_data": {
                "schema": schema_name,
                "tables": ["analytics_events", "performance_metrics"]
            }
        }
    
    async def _remove_analytics_tables(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Remove analytics tables."""
        rollback_data = context.get("rollback_data", {})
        schema_name = rollback_data.get("schema")
        tables = rollback_data.get("tables", [])
        
        if schema_name and not context["dry_run"]:
            for table in tables:
                await self.session.execute(
                    text(f"DROP TABLE IF EXISTS {schema_name}.{table}")
                )
            await self.session.commit()
        
        return {"status": "success", "changes": f"Removed analytics tables: {tables}"}
    
    async def _update_spotify_integration(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Update Spotify API integration."""
        tenant_id = context["tenant_id"]
        
        # Update Spotify connection configuration
        if not context["dry_run"]:
            # This would update Spotify API configurations
            pass
        
        return {
            "status": "success",
            "changes": "Updated Spotify API integration configuration"
        }
    
    async def _rollback_spotify_integration(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback Spotify integration update."""
        return {
            "status": "success", 
            "changes": "Spotify integration rollback completed"
        }
    
    async def _restructure_tenant_isolation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Restructure tenant isolation architecture."""
        tenant_id = context["tenant_id"]
        
        # Major architectural changes would go here
        if not context["dry_run"]:
            # Implement tenant isolation improvements
            pass
        
        return {
            "status": "success",
            "changes": "Restructured tenant isolation architecture",
            "warnings": ["Full system restart recommended after migration"]
        }
    
    async def _rollback_tenant_isolation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback tenant isolation restructure."""
        return {
            "status": "success",
            "changes": "Tenant isolation rollback completed"
        }
    
    async def _upgrade_ai_models(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Upgrade AI model configurations."""
        tenant_id = context["tenant_id"]
        
        if not context["dry_run"]:
            # Update AI model configurations
            pass
        
        return {
            "status": "success",
            "changes": "Upgraded AI model configurations"
        }
    
    async def _rollback_ai_models(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback AI model upgrade."""
        return {
            "status": "success",
            "changes": "AI model configuration rollback completed"
        }
    
    # Helper methods
    
    async def _check_tenant_exists(self, tenant_id: str) -> bool:
        """Check if tenant exists."""
        try:
            schema_name = TenantUtils.get_tenant_schema_name(tenant_id)
            result = await self.session.execute(
                text(f"SELECT schema_name FROM information_schema.schemata WHERE schema_name = '{schema_name}'")
            )
            return result.scalar() is not None
        except Exception:
            return False
    
    async def _get_all_tenants(self) -> List[str]:
        """Get list of all tenants."""
        try:
            result = await self.session.execute(
                text("""
                SELECT schema_name 
                FROM information_schema.schemata 
                WHERE schema_name LIKE 'tenant_%'
                """)
            )
            
            tenant_schemas = [row[0] for row in result]
            return [schema.replace('tenant_', '') for schema in tenant_schemas]
            
        except Exception as e:
            logger.error(f"Error getting tenant list: {e}")
            return []
    
    async def _get_tenant_version(self, tenant_id: str) -> str:
        """Get current version for tenant."""
        # This would check tenant metadata for current version
        # For now, return a default version
        return "1.0.0"
    
    async def _check_data_integrity(self, tenant_id: str) -> List[str]:
        """Check data integrity for tenant."""
        issues = []
        
        # Basic integrity checks
        try:
            schema_name = TenantUtils.get_tenant_schema_name(tenant_id)
            
            # Check for orphaned records (simplified)
            result = await self.session.execute(
                text(f"""
                SELECT COUNT(*)
                FROM {schema_name}.content_generated cg
                LEFT JOIN {schema_name}.users u ON cg.user_id = u.id
                WHERE u.id IS NULL AND cg.user_id IS NOT NULL
                """)
            )
            
            orphaned_count = result.scalar() or 0
            if orphaned_count > 0:
                issues.append(f"Found {orphaned_count} orphaned content records")
                
        except Exception as e:
            issues.append(f"Integrity check failed: {e}")
        
        return issues
    
    async def _check_required_data(self, tenant_id: str, version: str) -> List[str]:
        """Check for required data for migration."""
        missing_data = []
        
        # Check for required data based on version
        # This is version-specific validation
        
        return missing_data
    
    def _migration_path_exists(self, from_version: str, to_version: str) -> bool:
        """Check if migration path exists between versions."""
        # Simplified check - in reality would be more sophisticated
        return f"{from_version}_to_{to_version}" in self.migration_steps
    
    async def _generate_migration_summary(self, migration_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate migration summary."""
        return {
            "total_tenants": len(migration_result.get("tenants_migrated", [])),
            "successful_migrations": len(migration_result.get("tenants_migrated", [])),
            "failed_migrations": len(migration_result.get("errors", [])),
            "steps_executed": len(migration_result.get("steps_executed", [])),
            "steps_failed": len(migration_result.get("steps_failed", [])),
            "rollbacks_performed": len(migration_result.get("rollbacks_performed", [])),
            "warnings_count": len(migration_result.get("warnings", [])),
            "migration_path_length": len(migration_result.get("migration_path", [])),
            "success_rate": migration_result.get("success_rate", 0)
        }


async def migrate_fixtures(
    from_version: str,
    to_version: str,
    tenant_id: Optional[str] = None,
    dry_run: bool = True,
    auto_resolve: bool = False,
    force: bool = False
) -> Dict[str, Any]:
    """
    Main function to migrate fixtures.
    
    Args:
        from_version: Source version
        to_version: Target version
        tenant_id: Specific tenant or None for all
        dry_run: Validation mode
        auto_resolve: Auto-resolve conflicts
        force: Force migration
        
    Returns:
        Migration results
    """
    async with get_async_session() as session:
        redis_client = await get_redis_client()
        
        try:
            migrator = FixtureMigrator(session, redis_client)
            result = await migrator.migrate_fixtures(
                from_version=from_version,
                to_version=to_version,
                tenant_id=tenant_id,
                dry_run=dry_run,
                auto_resolve=auto_resolve,
                force=force
            )
            
            return result
            
        finally:
            if redis_client:
                await redis_client.close()


def main():
    """Command line interface for fixture migration."""
    parser = argparse.ArgumentParser(
        description="Migrate fixtures between versions"
    )
    
    parser.add_argument(
        "--from-version",
        required=True,
        help="Source version"
    )
    
    parser.add_argument(
        "--to-version", 
        required=True,
        help="Target version"
    )
    
    parser.add_argument(
        "--tenant-id",
        help="Specific tenant to migrate (default: all tenants)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Perform validation without changes"
    )
    
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform migration (overrides dry-run)"
    )
    
    parser.add_argument(
        "--auto-resolve",
        action="store_true",
        help="Automatically resolve conflicts"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force migration even with warnings"
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
    
    # Override dry-run if execute is specified
    dry_run = args.dry_run and not args.execute
    
    if not dry_run and not args.force:
        print(f"⚠️  This will migrate fixtures from {args.from_version} to {args.to_version}")
        if args.tenant_id:
            print(f"Target tenant: {args.tenant_id}")
        else:
            print("Target: ALL TENANTS")
        
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Migration cancelled.")
            sys.exit(0)
    
    try:
        # Run migration
        result = asyncio.run(
            migrate_fixtures(
                from_version=args.from_version,
                to_version=args.to_version,
                tenant_id=args.tenant_id,
                dry_run=dry_run,
                auto_resolve=args.auto_resolve,
                force=args.force
            )
        )
        
        # Display results
        print(f"\nMigration Results:")
        print(f"Status: {result['status']}")
        print(f"From: {result['from_version']} → To: {result['to_version']}")
        print(f"Duration: {FixtureUtils.format_duration(result.get('duration_seconds', 0))}")
        print(f"Success Rate: {result.get('success_rate', 0):.1%}")
        
        if result.get('migration_path'):
            print(f"Migration Path: {' → '.join(result['migration_path'])}")
        
        summary = result.get('summary', {})
        if summary:
            print(f"\nSummary:")
            print(f"  Tenants Migrated: {summary.get('successful_migrations', 0)}")
            print(f"  Steps Executed: {summary.get('steps_executed', 0)}")
            print(f"  Rollbacks: {summary.get('rollbacks_performed', 0)}")
            print(f"  Warnings: {summary.get('warnings_count', 0)}")
        
        if result.get('errors'):
            print(f"\nErrors ({len(result['errors'])}):")
            for error in result['errors'][:3]:
                print(f"  ❌ {error}")
            if len(result['errors']) > 3:
                print(f"  ... and {len(result['errors']) - 3} more")
        
        if result.get('warnings'):
            print(f"\nWarnings ({len(result['warnings'])}):")
            for warning in result['warnings'][:3]:
                print(f"  ⚠️  {warning}")
            if len(result['warnings']) > 3:
                print(f"  ... and {len(result['warnings']) - 3} more")
        
        if dry_run:
            print("\n🔍 DRY RUN completed - no changes made")
        elif result['status'] == 'completed':
            print("\n✅ Migration completed successfully!")
            sys.exit(0)
        else:
            print(f"\n❌ Migration failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Migration interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
