#!/usr/bin/env python3
"""
Spotify AI Agent - Template Migrations
=====================================

Advanced template versioning and migration system for handling
template schema evolution, data migration, and backward compatibility.

Features:
- Version-based migration chains
- Automatic schema detection and migration
- Rollback and recovery mechanisms
- Data transformation and validation
- Multi-tenant migration support
- Performance optimization during migrations

Migration Types:
- Schema migrations (structure changes)
- Data migrations (content transformation)
- Security migrations (security updates)
- Performance migrations (optimization)
- Compatibility migrations (API changes)

Author: Expert Development Team
"""

import json
import logging
import hashlib
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import semver

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select, insert, update, delete

from app.core.database import get_async_session
from app.core.redis import get_redis_client
from app.tenancy.fixtures.templates.validators import TemplateValidationEngine, ValidationReport

logger = logging.getLogger(__name__)


class MigrationType(Enum):
    """Types of template migrations."""
    SCHEMA = "schema"
    DATA = "data"
    SECURITY = "security"
    PERFORMANCE = "performance"
    COMPATIBILITY = "compatibility"


class MigrationStatus(Enum):
    """Migration execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class MigrationResult:
    """Result of migration execution."""
    success: bool
    migration_id: str
    from_version: str
    to_version: str
    execution_time_ms: float
    templates_migrated: int
    templates_failed: int
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    rollback_data: Optional[Dict[str, Any]] = None


@dataclass
class MigrationPlan:
    """Migration execution plan."""
    migrations: List['BaseMigration']
    total_templates: int
    estimated_time_seconds: float
    requires_downtime: bool
    rollback_plan: List['BaseMigration']


class BaseMigration(ABC):
    """Base class for all template migrations."""
    
    def __init__(self):
        self.id = self._generate_migration_id()
        self.from_version = "0.0.0"
        self.to_version = "1.0.0"
        self.migration_type = MigrationType.SCHEMA
        self.description = ""
        self.requires_downtime = False
        self.is_reversible = True
        self.dependencies: List[str] = []
        self.validation_engine = TemplateValidationEngine()
    
    @abstractmethod
    async def migrate_up(self, template: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Apply migration to template (forward migration)."""
        pass
    
    @abstractmethod
    async def migrate_down(self, template: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Rollback migration from template (reverse migration)."""
        pass
    
    async def can_migrate(self, template: Dict[str, Any]) -> bool:
        """Check if migration can be applied to template."""
        try:
            # Check version compatibility
            current_version = self._get_template_version(template)
            return semver.compare(current_version, self.from_version) >= 0
        except Exception as e:
            logger.warning(f"Version check failed for migration {self.id}: {str(e)}")
            return False
    
    async def validate_migration(self, original: Dict[str, Any], migrated: Dict[str, Any]) -> ValidationReport:
        """Validate migration result."""
        # Validate that migrated template is valid
        template_type = migrated.get('_metadata', {}).get('template_type', 'unknown')
        return self.validation_engine.validate_template(migrated, self.id, template_type)
    
    def _generate_migration_id(self) -> str:
        """Generate unique migration ID."""
        class_name = self.__class__.__name__
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{class_name}_{timestamp}"
    
    def _get_template_version(self, template: Dict[str, Any]) -> str:
        """Get template version."""
        metadata = template.get('_metadata', {})
        return metadata.get('template_version', '0.0.0')
    
    def _update_template_version(self, template: Dict[str, Any], version: str) -> Dict[str, Any]:
        """Update template version."""
        if '_metadata' not in template:
            template['_metadata'] = {}
        
        template['_metadata']['template_version'] = version
        template['_metadata']['last_migration'] = self.id
        template['_metadata']['migration_timestamp'] = datetime.now(timezone.utc).isoformat()
        
        return template


class SchemaEvolutionMigration(BaseMigration):
    """Migration for template schema evolution."""
    
    def __init__(self, schema_changes: Dict[str, Any]):
        super().__init__()
        self.migration_type = MigrationType.SCHEMA
        self.schema_changes = schema_changes
        self.description = "Schema evolution migration"
    
    async def migrate_up(self, template: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Apply schema changes to template."""
        migrated = template.copy()
        
        # Apply field additions
        for field_path, default_value in self.schema_changes.get('add_fields', {}).items():
            self._set_nested_field(migrated, field_path, default_value)
        
        # Apply field removals
        for field_path in self.schema_changes.get('remove_fields', []):
            self._remove_nested_field(migrated, field_path)
        
        # Apply field renames
        for old_path, new_path in self.schema_changes.get('rename_fields', {}).items():
            value = self._get_nested_field(migrated, old_path)
            if value is not None:
                self._set_nested_field(migrated, new_path, value)
                self._remove_nested_field(migrated, old_path)
        
        # Apply type conversions
        for field_path, target_type in self.schema_changes.get('convert_types', {}).items():
            current_value = self._get_nested_field(migrated, field_path)
            if current_value is not None:
                converted_value = self._convert_type(current_value, target_type)
                self._set_nested_field(migrated, field_path, converted_value)
        
        # Update version
        return self._update_template_version(migrated, self.to_version)
    
    async def migrate_down(self, template: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Reverse schema changes."""
        if not self.is_reversible:
            raise ValueError(f"Migration {self.id} is not reversible")
        
        migrated = template.copy()
        
        # Reverse type conversions
        for field_path, target_type in self.schema_changes.get('convert_types', {}).items():
            current_value = self._get_nested_field(migrated, field_path)
            if current_value is not None:
                # This is simplified - in practice, you'd need to store original types
                original_value = self._reverse_convert_type(current_value, target_type)
                self._set_nested_field(migrated, field_path, original_value)
        
        # Reverse field renames
        for old_path, new_path in self.schema_changes.get('rename_fields', {}).items():
            value = self._get_nested_field(migrated, new_path)
            if value is not None:
                self._set_nested_field(migrated, old_path, value)
                self._remove_nested_field(migrated, new_path)
        
        # Reverse field additions (remove them)
        for field_path in self.schema_changes.get('add_fields', {}).keys():
            self._remove_nested_field(migrated, field_path)
        
        # Reverse field removals (this requires backup data)
        # In practice, you'd need to store removed data during forward migration
        
        return self._update_template_version(migrated, self.from_version)
    
    def _get_nested_field(self, obj: Dict[str, Any], path: str) -> Any:
        """Get nested field value by dot notation path."""
        keys = path.split('.')
        current = obj
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def _set_nested_field(self, obj: Dict[str, Any], path: str, value: Any):
        """Set nested field value by dot notation path."""
        keys = path.split('.')
        current = obj
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _remove_nested_field(self, obj: Dict[str, Any], path: str):
        """Remove nested field by dot notation path."""
        keys = path.split('.')
        current = obj
        
        for key in keys[:-1]:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return  # Path doesn't exist
        
        if isinstance(current, dict) and keys[-1] in current:
            del current[keys[-1]]
    
    def _convert_type(self, value: Any, target_type: str) -> Any:
        """Convert value to target type."""
        try:
            if target_type == 'string':
                return str(value)
            elif target_type == 'integer':
                return int(value)
            elif target_type == 'float':
                return float(value)
            elif target_type == 'boolean':
                return bool(value)
            elif target_type == 'list':
                return list(value) if not isinstance(value, list) else value
            elif target_type == 'dict':
                return dict(value) if not isinstance(value, dict) else value
            else:
                return value
        except (ValueError, TypeError):
            return value  # Return original if conversion fails
    
    def _reverse_convert_type(self, value: Any, original_type: str) -> Any:
        """Reverse type conversion (simplified)."""
        # This is a simplified implementation
        # In practice, you'd need to track original types
        return self._convert_type(value, original_type)


class SecurityMigration(BaseMigration):
    """Migration for security updates and fixes."""
    
    def __init__(self, security_updates: Dict[str, Any]):
        super().__init__()
        self.migration_type = MigrationType.SECURITY
        self.security_updates = security_updates
        self.description = "Security update migration"
        self.requires_downtime = True  # Security updates often require downtime
    
    async def migrate_up(self, template: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Apply security updates to template."""
        migrated = template.copy()
        
        # Update encryption settings
        if 'encryption' in self.security_updates:
            encryption_config = self.security_updates['encryption']
            if '_security' not in migrated:
                migrated['_security'] = {}
            migrated['_security']['encryption'] = encryption_config
        
        # Update access control
        if 'access_control' in self.security_updates:
            access_config = self.security_updates['access_control']
            if '_security' not in migrated:
                migrated['_security'] = {}
            migrated['_security']['access_control'] = access_config
        
        # Sanitize sensitive data
        if 'sanitize_fields' in self.security_updates:
            for field_path in self.security_updates['sanitize_fields']:
                value = self._get_nested_field(migrated, field_path)
                if value and isinstance(value, str):
                    sanitized_value = self._sanitize_sensitive_data(value)
                    self._set_nested_field(migrated, field_path, sanitized_value)
        
        # Update password policies
        if 'password_policy' in self.security_updates:
            policy = self.security_updates['password_policy']
            if 'configuration' in migrated and 'security' in migrated['configuration']:
                migrated['configuration']['security']['password_policy'].update(policy)
        
        return self._update_template_version(migrated, self.to_version)
    
    async def migrate_down(self, template: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Rollback security updates (limited for security reasons)."""
        # Security migrations are typically not fully reversible
        # Only rollback non-critical changes
        
        migrated = template.copy()
        
        # Remove added security metadata (keep core security intact)
        if '_security' in migrated:
            # Only remove specific added fields, not core security
            for field in ['migration_security_metadata']:
                migrated['_security'].pop(field, None)
        
        return self._update_template_version(migrated, self.from_version)
    
    def _sanitize_sensitive_data(self, value: str) -> str:
        """Sanitize sensitive data in template."""
        # Remove or mask sensitive patterns
        import re
        
        # Mask email addresses
        value = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                      '****@****.***', value)
        
        # Mask phone numbers
        value = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '***-***-****', value)
        
        # Mask credit card numbers
        value = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', 
                      '****-****-****-****', value)
        
        return value
    
    def _get_nested_field(self, obj: Dict[str, Any], path: str) -> Any:
        """Get nested field (same as SchemaEvolutionMigration)."""
        keys = path.split('.')
        current = obj
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def _set_nested_field(self, obj: Dict[str, Any], path: str, value: Any):
        """Set nested field (same as SchemaEvolutionMigration)."""
        keys = path.split('.')
        current = obj
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value


class PerformanceMigration(BaseMigration):
    """Migration for performance optimizations."""
    
    def __init__(self, optimizations: Dict[str, Any]):
        super().__init__()
        self.migration_type = MigrationType.PERFORMANCE
        self.optimizations = optimizations
        self.description = "Performance optimization migration"
    
    async def migrate_up(self, template: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Apply performance optimizations."""
        migrated = template.copy()
        
        # Add caching configuration
        if 'caching' in self.optimizations:
            cache_config = self.optimizations['caching']
            if '_performance' not in migrated:
                migrated['_performance'] = {}
            migrated['_performance']['caching'] = cache_config
        
        # Add lazy loading configuration
        if 'lazy_loading' in self.optimizations:
            lazy_config = self.optimizations['lazy_loading']
            if '_performance' not in migrated:
                migrated['_performance'] = {}
            migrated['_performance']['lazy_loading'] = lazy_config
        
        # Optimize data structures
        if 'optimize_structures' in self.optimizations and self.optimizations['optimize_structures']:
            migrated = await self._optimize_data_structures(migrated)
        
        return self._update_template_version(migrated, self.to_version)
    
    async def migrate_down(self, template: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Remove performance optimizations."""
        migrated = template.copy()
        
        # Remove performance metadata
        if '_performance' in migrated:
            del migrated['_performance']
        
        return self._update_template_version(migrated, self.from_version)
    
    async def _optimize_data_structures(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize template data structures."""
        # Example optimization: convert repeated objects to references
        optimized = template.copy()
        
        # This is a simplified example
        # In practice, you'd implement sophisticated optimization algorithms
        
        return optimized


class MigrationRegistry:
    """Registry for managing template migrations."""
    
    def __init__(self):
        self.migrations: Dict[str, BaseMigration] = {}
        self.migration_chains: Dict[str, List[str]] = {}  # version -> list of migration IDs
    
    def register_migration(self, migration: BaseMigration):
        """Register a migration."""
        self.migrations[migration.id] = migration
        
        # Add to migration chain
        from_version = migration.from_version
        if from_version not in self.migration_chains:
            self.migration_chains[from_version] = []
        
        self.migration_chains[from_version].append(migration.id)
    
    def get_migration_path(self, from_version: str, to_version: str) -> List[BaseMigration]:
        """Get migration path between versions."""
        # Simplified implementation
        # In practice, you'd implement graph traversal to find optimal path
        
        migrations = []
        current_version = from_version
        
        while semver.compare(current_version, to_version) < 0:
            # Find next migration
            next_migrations = self.migration_chains.get(current_version, [])
            if not next_migrations:
                break
            
            # Choose first available migration (in practice, you'd be smarter)
            migration_id = next_migrations[0]
            migration = self.migrations[migration_id]
            migrations.append(migration)
            current_version = migration.to_version
        
        return migrations
    
    def get_rollback_path(self, from_version: str, to_version: str) -> List[BaseMigration]:
        """Get rollback path between versions."""
        forward_path = self.get_migration_path(to_version, from_version)
        return list(reversed(forward_path))


class MigrationExecutor:
    """Executes template migrations with rollback support."""
    
    def __init__(self, registry: MigrationRegistry):
        self.registry = registry
        self.validation_engine = TemplateValidationEngine()
    
    async def create_migration_plan(
        self,
        templates: List[Dict[str, Any]],
        target_version: str
    ) -> MigrationPlan:
        """Create migration execution plan."""
        
        # Group templates by current version
        version_groups = {}
        for template in templates:
            current_version = template.get('_metadata', {}).get('template_version', '0.0.0')
            if current_version not in version_groups:
                version_groups[current_version] = []
            version_groups[current_version].append(template)
        
        # Calculate migrations needed
        all_migrations = []
        requires_downtime = False
        estimated_time = 0.0
        
        for current_version, template_group in version_groups.items():
            if semver.compare(current_version, target_version) < 0:
                migrations = self.registry.get_migration_path(current_version, target_version)
                all_migrations.extend(migrations)
                
                # Check if any migration requires downtime
                if any(migration.requires_downtime for migration in migrations):
                    requires_downtime = True
                
                # Estimate time (simplified)
                estimated_time += len(migrations) * len(template_group) * 0.1  # 100ms per template per migration
        
        # Remove duplicate migrations
        unique_migrations = []
        seen_ids = set()
        for migration in all_migrations:
            if migration.id not in seen_ids:
                unique_migrations.append(migration)
                seen_ids.add(migration.id)
        
        # Create rollback plan
        rollback_plan = []
        for migration in reversed(unique_migrations):
            if migration.is_reversible:
                rollback_plan.append(migration)
        
        return MigrationPlan(
            migrations=unique_migrations,
            total_templates=len(templates),
            estimated_time_seconds=estimated_time,
            requires_downtime=requires_downtime,
            rollback_plan=rollback_plan
        )
    
    async def execute_migration_plan(
        self,
        plan: MigrationPlan,
        templates: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> List[MigrationResult]:
        """Execute migration plan."""
        
        results = []
        
        for migration in plan.migrations:
            result = await self._execute_single_migration(migration, templates, context)
            results.append(result)
            
            # If migration failed, consider stopping or rolling back
            if not result.success:
                logger.error(f"Migration {migration.id} failed: {result.error_message}")
                # In practice, you might want to rollback here
                break
        
        return results
    
    async def _execute_single_migration(
        self,
        migration: BaseMigration,
        templates: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> MigrationResult:
        """Execute single migration on templates."""
        
        start_time = datetime.now()
        templates_migrated = 0
        templates_failed = 0
        errors = []
        warnings = []
        rollback_data = {}
        
        try:
            for i, template in enumerate(templates):
                try:
                    # Check if migration can be applied
                    if not await migration.can_migrate(template):
                        warnings.append(f"Template {i} cannot be migrated with {migration.id}")
                        continue
                    
                    # Store original for rollback
                    if migration.is_reversible:
                        rollback_data[str(i)] = template.copy()
                    
                    # Apply migration
                    migrated_template = await migration.migrate_up(template, context)
                    
                    # Validate migration result
                    validation_report = await migration.validate_migration(template, migrated_template)
                    if not validation_report.is_valid:
                        templates_failed += 1
                        errors.append(f"Validation failed for template {i}: {validation_report.total_issues} issues")
                        continue
                    
                    # Update template in-place
                    templates[i] = migrated_template
                    templates_migrated += 1
                
                except Exception as e:
                    templates_failed += 1
                    errors.append(f"Template {i} migration failed: {str(e)}")
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return MigrationResult(
                success=templates_failed == 0,
                migration_id=migration.id,
                from_version=migration.from_version,
                to_version=migration.to_version,
                execution_time_ms=execution_time,
                templates_migrated=templates_migrated,
                templates_failed=templates_failed,
                error_message="; ".join(errors) if errors else None,
                warnings=warnings,
                rollback_data=rollback_data if rollback_data else None
            )
        
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return MigrationResult(
                success=False,
                migration_id=migration.id,
                from_version=migration.from_version,
                to_version=migration.to_version,
                execution_time_ms=execution_time,
                templates_migrated=templates_migrated,
                templates_failed=len(templates),
                error_message=str(e),
                rollback_data=rollback_data if rollback_data else None
            )
    
    async def rollback_migration(
        self,
        migration_result: MigrationResult,
        templates: List[Dict[str, Any]]
    ) -> MigrationResult:
        """Rollback a migration using stored rollback data."""
        
        if not migration_result.rollback_data:
            return MigrationResult(
                success=False,
                migration_id=migration_result.migration_id,
                from_version=migration_result.to_version,
                to_version=migration_result.from_version,
                execution_time_ms=0.0,
                templates_migrated=0,
                templates_failed=0,
                error_message="No rollback data available"
            )
        
        start_time = datetime.now()
        templates_restored = 0
        
        try:
            # Restore templates from rollback data
            for template_index, original_template in migration_result.rollback_data.items():
                index = int(template_index)
                if index < len(templates):
                    templates[index] = original_template
                    templates_restored += 1
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return MigrationResult(
                success=True,
                migration_id=migration_result.migration_id,
                from_version=migration_result.to_version,
                to_version=migration_result.from_version,
                execution_time_ms=execution_time,
                templates_migrated=templates_restored,
                templates_failed=0
            )
        
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return MigrationResult(
                success=False,
                migration_id=migration_result.migration_id,
                from_version=migration_result.to_version,
                to_version=migration_result.from_version,
                execution_time_ms=execution_time,
                templates_migrated=0,
                templates_failed=len(templates),
                error_message=str(e)
            )


class MigrationManager:
    """High-level migration management."""
    
    def __init__(self):
        self.registry = MigrationRegistry()
        self.executor = MigrationExecutor(self.registry)
        self._setup_default_migrations()
    
    def _setup_default_migrations(self):
        """Setup default migrations."""
        
        # Schema evolution: v1.0.0 -> v1.1.0
        schema_migration = SchemaEvolutionMigration({
            'add_fields': {
                '_metadata.schema_version': '2024.1',
                'performance.caching_enabled': True
            },
            'rename_fields': {
                'config': 'configuration'
            },
            'convert_types': {
                'status': 'string'
            }
        })
        schema_migration.from_version = "1.0.0"
        schema_migration.to_version = "1.1.0"
        
        # Security migration: v1.1.0 -> v1.2.0
        security_migration = SecurityMigration({
            'encryption': {
                'enabled': True,
                'algorithm': 'AES-256-GCM'
            },
            'access_control': {
                'default_policy': 'deny',
                'audit_enabled': True
            },
            'sanitize_fields': ['description', 'notes'],
            'password_policy': {
                'min_length': 12,
                'require_mfa': True
            }
        })
        security_migration.from_version = "1.1.0"
        security_migration.to_version = "1.2.0"
        
        # Performance migration: v1.2.0 -> v1.3.0
        performance_migration = PerformanceMigration({
            'caching': {
                'enabled': True,
                'ttl_seconds': 3600,
                'strategy': 'write-through'
            },
            'lazy_loading': {
                'enabled': True,
                'threshold_bytes': 10240
            },
            'optimize_structures': True
        })
        performance_migration.from_version = "1.2.0"
        performance_migration.to_version = "1.3.0"
        
        # Register migrations
        self.registry.register_migration(schema_migration)
        self.registry.register_migration(security_migration)
        self.registry.register_migration(performance_migration)
    
    async def migrate_templates_to_version(
        self,
        templates: List[Dict[str, Any]],
        target_version: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[MigrationResult], List[Dict[str, Any]]]:
        """Migrate templates to target version."""
        
        # Create migration plan
        plan = await self.executor.create_migration_plan(templates, target_version)
        
        logger.info(f"Migration plan created: {len(plan.migrations)} migrations for {plan.total_templates} templates")
        logger.info(f"Estimated time: {plan.estimated_time_seconds:.2f}s, Downtime required: {plan.requires_downtime}")
        
        # Execute migrations
        results = await self.executor.execute_migration_plan(plan, templates, context)
        
        return results, templates
    
    async def rollback_to_version(
        self,
        templates: List[Dict[str, Any]],
        target_version: str,
        migration_results: List[MigrationResult]
    ) -> List[MigrationResult]:
        """Rollback templates to target version."""
        
        rollback_results = []
        
        # Rollback in reverse order
        for result in reversed(migration_results):
            if semver.compare(result.to_version, target_version) > 0:
                rollback_result = await self.executor.rollback_migration(result, templates)
                rollback_results.append(rollback_result)
        
        return rollback_results
    
    def register_custom_migration(self, migration: BaseMigration):
        """Register custom migration."""
        self.registry.register_migration(migration)
    
    def get_migration_status(self, templates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get migration status for templates."""
        
        version_distribution = {}
        for template in templates:
            version = template.get('_metadata', {}).get('template_version', '0.0.0')
            if version not in version_distribution:
                version_distribution[version] = 0
            version_distribution[version] += 1
        
        # Find latest available version
        all_versions = set()
        for migration in self.registry.migrations.values():
            all_versions.add(migration.from_version)
            all_versions.add(migration.to_version)
        
        latest_version = max(all_versions, key=lambda v: semver.VersionInfo.parse(v)) if all_versions else "0.0.0"
        
        return {
            'total_templates': len(templates),
            'version_distribution': version_distribution,
            'latest_available_version': latest_version,
            'migrations_available': len(self.registry.migrations),
            'migration_chains': len(self.registry.migration_chains)
        }


# Global migration manager instance
migration_manager = MigrationManager()


# Utility functions
async def migrate_template_to_latest(
    template: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], List[MigrationResult]]:
    """Migrate single template to latest version."""
    
    templates = [template]
    latest_version = "1.3.0"  # This would be determined dynamically
    
    results, migrated_templates = await migration_manager.migrate_templates_to_version(
        templates, latest_version, context
    )
    
    return migrated_templates[0], results


async def check_migration_needed(template: Dict[str, Any]) -> bool:
    """Check if template needs migration."""
    current_version = template.get('_metadata', {}).get('template_version', '0.0.0')
    latest_version = "1.3.0"  # This would be determined dynamically
    
    return semver.compare(current_version, latest_version) < 0


def create_custom_migration(
    migration_class: type,
    from_version: str,
    to_version: str,
    **kwargs
) -> BaseMigration:
    """Factory function to create custom migrations."""
    
    migration = migration_class(**kwargs)
    migration.from_version = from_version
    migration.to_version = to_version
    
    return migration
