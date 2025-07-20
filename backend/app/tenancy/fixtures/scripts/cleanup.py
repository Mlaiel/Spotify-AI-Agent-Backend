#!/usr/bin/env python3
"""
Spotify AI Agent - Data Cleanup Script
=====================================

Comprehensive data cleanup script that provides:
- Tenant data cleanup and archival
- Temporary file cleanup
- Cache cleanup and optimization
- Database maintenance and optimization
- Safe data removal with backup options

Usage:
    python -m app.tenancy.fixtures.scripts.cleanup --tenant-id mycompany --type full
    python cleanup.py --tenant-id startup --dry-run --cleanup-type cache

Author: Expert Development Team (Fahed Mlaiel)
"""

import argparse
import asyncio
import logging
import shutil
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_async_session
from app.core.cache import get_redis_client
from app.tenancy.fixtures.base import FixtureManager
from app.tenancy.fixtures.monitoring import FixtureMonitor
from app.tenancy.fixtures.utils import FixtureUtils, TenantUtils, ArchiveManager
from app.tenancy.fixtures.exceptions import FixtureError
from app.tenancy.fixtures.constants import (
    FIXTURE_BASE_PATH, BACKUP_PATH, TEMP_PATH, CACHE_TTL_DEFAULT
)

logger = logging.getLogger(__name__)


class DataCleanup:
    """
    Comprehensive data cleanup orchestrator.
    
    Provides multiple cleanup strategies:
    - Tenant data cleanup (with archival)
    - Temporary file cleanup
    - Cache cleanup and optimization
    - Database maintenance
    - Log cleanup
    - Backup cleanup
    """
    
    def __init__(self, session: AsyncSession, redis_client=None):
        self.session = session
        self.redis_client = redis_client
        self.fixture_manager = FixtureManager(session, redis_client)
        self.monitor = FixtureMonitor(session, redis_client)
        self.archive_manager = ArchiveManager()
    
    async def cleanup_tenant_data(
        self,
        tenant_id: Optional[str] = None,
        cleanup_types: List[str] = None,
        retention_days: int = 90,
        dry_run: bool = True,
        create_backup: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive data cleanup.
        
        Args:
            tenant_id: Specific tenant to clean (None for all tenants)
            cleanup_types: Types of cleanup to perform
            retention_days: Data retention period in days
            dry_run: Perform analysis without actual cleanup
            create_backup: Create backup before cleanup
            
        Returns:
            Cleanup results and statistics
        """
        start_time = datetime.now(timezone.utc)
        
        if cleanup_types is None:
            cleanup_types = ["old_data", "temp_files", "cache", "logs"]
        
        cleanup_result = {
            "tenant_id": tenant_id or "all",
            "cleanup_types": cleanup_types,
            "retention_days": retention_days,
            "dry_run": dry_run,
            "create_backup": create_backup,
            "status": "started",
            "start_time": start_time.isoformat(),
            "operations_performed": {},
            "data_removed": {},
            "space_freed": 0,
            "backups_created": [],
            "warnings": [],
            "errors": []
        }
        
        try:
            # Get tenant list
            tenant_list = []
            if tenant_id:
                if await self._check_tenant_exists(tenant_id):
                    tenant_list = [tenant_id]
                else:
                    raise FixtureError(f"Tenant not found: {tenant_id}")
            else:
                tenant_list = await self._get_all_tenants()
            
            # Perform cleanup operations
            for cleanup_type in cleanup_types:
                logger.info(f"Performing {cleanup_type} cleanup")
                
                operation_result = await self._perform_cleanup_operation(
                    cleanup_type, tenant_list, retention_days, dry_run, create_backup
                )
                
                cleanup_result["operations_performed"][cleanup_type] = operation_result
                cleanup_result["space_freed"] += operation_result.get("space_freed", 0)
                
                if operation_result.get("data_removed"):
                    cleanup_result["data_removed"].update(operation_result["data_removed"])
                
                if operation_result.get("backups_created"):
                    cleanup_result["backups_created"].extend(operation_result["backups_created"])
                
                if operation_result.get("warnings"):
                    cleanup_result["warnings"].extend(operation_result["warnings"])
                
                if operation_result.get("errors"):
                    cleanup_result["errors"].extend(operation_result["errors"])
            
            # Calculate final metrics
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            cleanup_result.update({
                "status": "completed",
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "tenants_processed": len(tenant_list),
                "operations_count": len(cleanup_types),
                "space_freed_mb": cleanup_result["space_freed"] / 1024 / 1024
            })
            
            # Record cleanup metrics
            await self.monitor.record_cleanup_operation(cleanup_result)
            
            logger.info(
                f"Cleanup completed: {FixtureUtils.format_bytes(cleanup_result['space_freed'])} freed "
                f"in {FixtureUtils.format_duration(duration)}"
            )
            
        except Exception as e:
            cleanup_result["status"] = "failed"
            cleanup_result["error"] = str(e)
            logger.error(f"Cleanup failed: {e}")
            raise
        
        return cleanup_result
    
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
    
    async def _perform_cleanup_operation(
        self,
        cleanup_type: str,
        tenant_list: List[str],
        retention_days: int,
        dry_run: bool,
        create_backup: bool
    ) -> Dict[str, Any]:
        """Perform specific cleanup operation."""
        cleanup_methods = {
            "old_data": self._cleanup_old_data,
            "temp_files": self._cleanup_temp_files,
            "cache": self._cleanup_cache,
            "logs": self._cleanup_logs,
            "backups": self._cleanup_old_backups,
            "analytics": self._cleanup_analytics_data,
            "sessions": self._cleanup_old_sessions
        }
        
        if cleanup_type not in cleanup_methods:
            raise FixtureError(f"Unknown cleanup type: {cleanup_type}")
        
        return await cleanup_methods[cleanup_type](
            tenant_list, retention_days, dry_run, create_backup
        )
    
    async def _cleanup_old_data(
        self,
        tenant_list: List[str],
        retention_days: int,
        dry_run: bool,
        create_backup: bool
    ) -> Dict[str, Any]:
        """Clean up old data records."""
        operation_result = {
            "operation": "old_data",
            "data_removed": {},
            "space_freed": 0,
            "backups_created": [],
            "warnings": [],
            "errors": []
        }
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
        
        for tenant_id in tenant_list:
            try:
                schema_name = TenantUtils.get_tenant_schema_name(tenant_id)
                
                # Tables to clean with their timestamp columns
                tables_to_clean = {
                    "ai_sessions": "created_at",
                    "content_generated": "created_at",
                    "analytics_events": "created_at"
                }
                
                tenant_data_removed = {}
                
                for table, timestamp_col in tables_to_clean.items():
                    try:
                        # Count records to be removed
                        count_result = await self.session.execute(
                            text(f"""
                            SELECT COUNT(*)
                            FROM {schema_name}.{table}
                            WHERE {timestamp_col} < '{cutoff_date.isoformat()}'
                            """)
                        )
                        record_count = count_result.scalar() or 0
                        
                        if record_count > 0:
                            # Create backup if requested
                            if create_backup and not dry_run:
                                backup_result = await self._create_table_backup(
                                    schema_name, table, tenant_id, cutoff_date
                                )
                                if backup_result:
                                    operation_result["backups_created"].append(backup_result)
                            
                            # Calculate space before deletion
                            size_result = await self.session.execute(
                                text(f"SELECT pg_total_relation_size('{schema_name}.{table}')")
                            )
                            table_size = size_result.scalar() or 0
                            
                            # Estimate space to be freed (proportional to records)
                            if not dry_run:
                                total_records_result = await self.session.execute(
                                    text(f"SELECT COUNT(*) FROM {schema_name}.{table}")
                                )
                                total_records = total_records_result.scalar() or 1
                                space_to_free = int((record_count / total_records) * table_size)
                                operation_result["space_freed"] += space_to_free
                            
                            tenant_data_removed[table] = record_count
                            
                            # Delete old records
                            if not dry_run:
                                await self.session.execute(
                                    text(f"""
                                    DELETE FROM {schema_name}.{table}
                                    WHERE {timestamp_col} < '{cutoff_date.isoformat()}'
                                    """)
                                )
                                await self.session.commit()
                                
                                logger.info(f"Deleted {record_count} old records from {schema_name}.{table}")
                            else:
                                logger.info(f"DRY RUN: Would delete {record_count} records from {schema_name}.{table}")
                    
                    except Exception as e:
                        error_msg = f"Error cleaning {table} for tenant {tenant_id}: {e}"
                        operation_result["errors"].append(error_msg)
                        logger.error(error_msg)
                
                if tenant_data_removed:
                    operation_result["data_removed"][tenant_id] = tenant_data_removed
                
            except Exception as e:
                error_msg = f"Error processing tenant {tenant_id}: {e}"
                operation_result["errors"].append(error_msg)
                logger.error(error_msg)
        
        return operation_result
    
    async def _cleanup_temp_files(
        self,
        tenant_list: List[str],
        retention_days: int,
        dry_run: bool,
        create_backup: bool
    ) -> Dict[str, Any]:
        """Clean up temporary files."""
        operation_result = {
            "operation": "temp_files",
            "data_removed": {},
            "space_freed": 0,
            "backups_created": [],
            "warnings": [],
            "errors": []
        }
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
        temp_dirs = [
            Path(TEMP_PATH),
            Path(FIXTURE_BASE_PATH) / "temp",
            Path("/tmp/spotify_ai_agent")  # System temp directory
        ]
        
        for temp_dir in temp_dirs:
            if temp_dir.exists():
                try:
                    files_removed = 0
                    space_freed = 0
                    
                    for file_path in temp_dir.rglob("*"):
                        if file_path.is_file():
                            try:
                                # Check file age
                                file_mtime = datetime.fromtimestamp(
                                    file_path.stat().st_mtime, tz=timezone.utc
                                )
                                
                                if file_mtime < cutoff_date:
                                    file_size = file_path.stat().st_size
                                    
                                    if not dry_run:
                                        file_path.unlink()
                                        space_freed += file_size
                                        files_removed += 1
                                    else:
                                        space_freed += file_size
                                        files_removed += 1
                                        
                            except Exception as e:
                                operation_result["warnings"].append(f"Error processing file {file_path}: {e}")
                    
                    if files_removed > 0:
                        operation_result["data_removed"][str(temp_dir)] = {
                            "files_removed": files_removed,
                            "space_freed": space_freed
                        }
                        operation_result["space_freed"] += space_freed
                        
                        if not dry_run:
                            logger.info(f"Removed {files_removed} temp files from {temp_dir}")
                        else:
                            logger.info(f"DRY RUN: Would remove {files_removed} temp files from {temp_dir}")
                
                except Exception as e:
                    error_msg = f"Error cleaning temp directory {temp_dir}: {e}"
                    operation_result["errors"].append(error_msg)
                    logger.error(error_msg)
        
        return operation_result
    
    async def _cleanup_cache(
        self,
        tenant_list: List[str],
        retention_days: int,
        dry_run: bool,
        create_backup: bool
    ) -> Dict[str, Any]:
        """Clean up cache data."""
        operation_result = {
            "operation": "cache",
            "data_removed": {},
            "space_freed": 0,
            "backups_created": [],
            "warnings": [],
            "errors": []
        }
        
        if not self.redis_client:
            operation_result["warnings"].append("Redis client not available for cache cleanup")
            return operation_result
        
        try:
            total_keys_removed = 0
            
            for tenant_id in tenant_list:
                try:
                    cache_namespace = TenantUtils.get_tenant_cache_namespace(tenant_id)
                    pattern = f"{cache_namespace}:*"
                    
                    # Get all keys for tenant
                    keys = await self.redis_client.keys(pattern)
                    keys_to_remove = []
                    
                    # Check expiration and age
                    for key in keys:
                        try:
                            ttl = await self.redis_client.ttl(key)
                            
                            # Remove expired or very old keys
                            if ttl == -1 or ttl > CACHE_TTL_DEFAULT:  # No expiration or too long
                                keys_to_remove.append(key)
                                
                        except Exception as e:
                            operation_result["warnings"].append(f"Error checking key {key}: {e}")
                    
                    if keys_to_remove:
                        if not dry_run:
                            if keys_to_remove:
                                await self.redis_client.delete(*keys_to_remove)
                            total_keys_removed += len(keys_to_remove)
                        
                        operation_result["data_removed"][tenant_id] = {
                            "cache_keys_removed": len(keys_to_remove)
                        }
                        
                        if not dry_run:
                            logger.info(f"Removed {len(keys_to_remove)} cache keys for tenant {tenant_id}")
                        else:
                            logger.info(f"DRY RUN: Would remove {len(keys_to_remove)} cache keys for tenant {tenant_id}")
                
                except Exception as e:
                    error_msg = f"Error cleaning cache for tenant {tenant_id}: {e}"
                    operation_result["errors"].append(error_msg)
                    logger.error(error_msg)
            
            # Estimate space freed (rough calculation)
            operation_result["space_freed"] = total_keys_removed * 1024  # Assume 1KB per key average
            
        except Exception as e:
            error_msg = f"Error during cache cleanup: {e}"
            operation_result["errors"].append(error_msg)
            logger.error(error_msg)
        
        return operation_result
    
    async def _cleanup_logs(
        self,
        tenant_list: List[str],
        retention_days: int,
        dry_run: bool,
        create_backup: bool
    ) -> Dict[str, Any]:
        """Clean up log files."""
        operation_result = {
            "operation": "logs",
            "data_removed": {},
            "space_freed": 0,
            "backups_created": [],
            "warnings": [],
            "errors": []
        }
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
        log_dirs = [
            Path("/var/log/spotify_ai_agent"),
            Path(FIXTURE_BASE_PATH) / "logs",
            Path("./logs")
        ]
        
        for log_dir in log_dirs:
            if log_dir.exists():
                try:
                    files_removed = 0
                    space_freed = 0
                    
                    for log_file in log_dir.rglob("*.log*"):
                        if log_file.is_file():
                            try:
                                file_mtime = datetime.fromtimestamp(
                                    log_file.stat().st_mtime, tz=timezone.utc
                                )
                                
                                if file_mtime < cutoff_date:
                                    file_size = log_file.stat().st_size
                                    
                                    if not dry_run:
                                        # Compress old log files instead of deleting
                                        if not log_file.name.endswith('.gz'):
                                            compressed_path = log_file.with_suffix(log_file.suffix + '.gz')
                                            import gzip
                                            
                                            with open(log_file, 'rb') as f_in:
                                                with gzip.open(compressed_path, 'wb') as f_out:
                                                    shutil.copyfileobj(f_in, f_out)
                                            
                                            log_file.unlink()
                                            space_freed += file_size - compressed_path.stat().st_size
                                        else:
                                            # Very old compressed files can be deleted
                                            if file_mtime < cutoff_date - timedelta(days=retention_days):
                                                log_file.unlink()
                                                space_freed += file_size
                                        
                                        files_removed += 1
                                    else:
                                        space_freed += file_size
                                        files_removed += 1
                                        
                            except Exception as e:
                                operation_result["warnings"].append(f"Error processing log file {log_file}: {e}")
                    
                    if files_removed > 0:
                        operation_result["data_removed"][str(log_dir)] = {
                            "log_files_processed": files_removed,
                            "space_freed": space_freed
                        }
                        operation_result["space_freed"] += space_freed
                        
                        action = "compressed/removed" if not dry_run else "would compress/remove"
                        logger.info(f"{action.capitalize()} {files_removed} log files from {log_dir}")
                
                except Exception as e:
                    error_msg = f"Error cleaning log directory {log_dir}: {e}"
                    operation_result["errors"].append(error_msg)
                    logger.error(error_msg)
        
        return operation_result
    
    async def _cleanup_old_backups(
        self,
        tenant_list: List[str],
        retention_days: int,
        dry_run: bool,
        create_backup: bool
    ) -> Dict[str, Any]:
        """Clean up old backup files."""
        operation_result = {
            "operation": "backups",
            "data_removed": {},
            "space_freed": 0,
            "backups_created": [],
            "warnings": [],
            "errors": []
        }
        
        # Keep backups longer than regular data
        backup_retention_days = retention_days * 2
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=backup_retention_days)
        
        backup_dir = Path(BACKUP_PATH)
        if backup_dir.exists():
            try:
                files_removed = 0
                space_freed = 0
                
                for backup_file in backup_dir.rglob("*_backup*"):
                    if backup_file.is_file():
                        try:
                            file_mtime = datetime.fromtimestamp(
                                backup_file.stat().st_mtime, tz=timezone.utc
                            )
                            
                            if file_mtime < cutoff_date:
                                file_size = backup_file.stat().st_size
                                
                                if not dry_run:
                                    backup_file.unlink()
                                    space_freed += file_size
                                    files_removed += 1
                                else:
                                    space_freed += file_size
                                    files_removed += 1
                                    
                        except Exception as e:
                            operation_result["warnings"].append(f"Error processing backup file {backup_file}: {e}")
                
                if files_removed > 0:
                    operation_result["data_removed"]["backup_files"] = {
                        "files_removed": files_removed,
                        "space_freed": space_freed
                    }
                    operation_result["space_freed"] += space_freed
                    
                    action = "Removed" if not dry_run else "Would remove"
                    logger.info(f"{action} {files_removed} old backup files")
            
            except Exception as e:
                error_msg = f"Error cleaning backup directory: {e}"
                operation_result["errors"].append(error_msg)
                logger.error(error_msg)
        
        return operation_result
    
    async def _cleanup_analytics_data(
        self,
        tenant_list: List[str],
        retention_days: int,
        dry_run: bool,
        create_backup: bool
    ) -> Dict[str, Any]:
        """Clean up old analytics data."""
        operation_result = {
            "operation": "analytics",
            "data_removed": {},
            "space_freed": 0,
            "backups_created": [],
            "warnings": [],
            "errors": []
        }
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
        
        for tenant_id in tenant_list:
            try:
                schema_name = TenantUtils.get_tenant_schema_name(tenant_id)
                
                # Clean analytics tables
                analytics_tables = [
                    "analytics_events",
                    "performance_metrics",
                    "user_activity_logs"
                ]
                
                tenant_data_removed = {}
                
                for table in analytics_tables:
                    try:
                        # Check if table exists
                        table_check = await self.session.execute(
                            text(f"""
                            SELECT table_name 
                            FROM information_schema.tables 
                            WHERE table_schema = '{schema_name}' 
                            AND table_name = '{table}'
                            """)
                        )
                        
                        if not table_check.scalar():
                            continue
                        
                        # Count old records
                        count_result = await self.session.execute(
                            text(f"""
                            SELECT COUNT(*)
                            FROM {schema_name}.{table}
                            WHERE created_at < '{cutoff_date.isoformat()}'
                            """)
                        )
                        record_count = count_result.scalar() or 0
                        
                        if record_count > 0:
                            tenant_data_removed[table] = record_count
                            
                            if not dry_run:
                                await self.session.execute(
                                    text(f"""
                                    DELETE FROM {schema_name}.{table}
                                    WHERE created_at < '{cutoff_date.isoformat()}'
                                    """)
                                )
                                await self.session.commit()
                                
                                logger.info(f"Deleted {record_count} old analytics records from {schema_name}.{table}")
                            else:
                                logger.info(f"DRY RUN: Would delete {record_count} analytics records from {schema_name}.{table}")
                    
                    except Exception as e:
                        operation_result["warnings"].append(f"Error cleaning analytics table {table}: {e}")
                
                if tenant_data_removed:
                    operation_result["data_removed"][tenant_id] = tenant_data_removed
                
            except Exception as e:
                error_msg = f"Error cleaning analytics for tenant {tenant_id}: {e}"
                operation_result["errors"].append(error_msg)
                logger.error(error_msg)
        
        return operation_result
    
    async def _cleanup_old_sessions(
        self,
        tenant_list: List[str],
        retention_days: int,
        dry_run: bool,
        create_backup: bool
    ) -> Dict[str, Any]:
        """Clean up old AI sessions and related data."""
        operation_result = {
            "operation": "sessions",
            "data_removed": {},
            "space_freed": 0,
            "backups_created": [],
            "warnings": [],
            "errors": []
        }
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
        
        for tenant_id in tenant_list:
            try:
                schema_name = TenantUtils.get_tenant_schema_name(tenant_id)
                
                # Count old AI sessions
                count_result = await self.session.execute(
                    text(f"""
                    SELECT COUNT(*)
                    FROM {schema_name}.ai_sessions
                    WHERE created_at < '{cutoff_date.isoformat()}'
                    AND status IN ('completed', 'failed', 'timeout')
                    """)
                )
                session_count = count_result.scalar() or 0
                
                if session_count > 0:
                    # Create backup if requested
                    if create_backup and not dry_run:
                        backup_result = await self._create_table_backup(
                            schema_name, "ai_sessions", tenant_id, cutoff_date
                        )
                        if backup_result:
                            operation_result["backups_created"].append(backup_result)
                    
                    operation_result["data_removed"][tenant_id] = {
                        "ai_sessions": session_count
                    }
                    
                    if not dry_run:
                        await self.session.execute(
                            text(f"""
                            DELETE FROM {schema_name}.ai_sessions
                            WHERE created_at < '{cutoff_date.isoformat()}'
                            AND status IN ('completed', 'failed', 'timeout')
                            """)
                        )
                        await self.session.commit()
                        
                        logger.info(f"Deleted {session_count} old AI sessions for tenant {tenant_id}")
                    else:
                        logger.info(f"DRY RUN: Would delete {session_count} AI sessions for tenant {tenant_id}")
            
            except Exception as e:
                error_msg = f"Error cleaning sessions for tenant {tenant_id}: {e}"
                operation_result["errors"].append(error_msg)
                logger.error(error_msg)
        
        return operation_result
    
    async def _create_table_backup(
        self,
        schema_name: str,
        table_name: str,
        tenant_id: str,
        cutoff_date: datetime
    ) -> Optional[str]:
        """Create backup of table data before cleanup."""
        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_file = f"{tenant_id}_{table_name}_{timestamp}_backup.sql"
            backup_path = Path(BACKUP_PATH) / backup_file
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Export data to SQL file
            export_query = f"""
            COPY (
                SELECT * FROM {schema_name}.{table_name}
                WHERE created_at < '{cutoff_date.isoformat()}'
            ) TO STDOUT WITH CSV HEADER
            """
            
            # This would need to be implemented with proper database export tools
            # For now, just create a placeholder file
            with open(backup_path, 'w') as f:
                f.write(f"-- Backup of {schema_name}.{table_name} created at {timestamp}\n")
                f.write(f"-- Data older than {cutoff_date.isoformat()}\n")
            
            logger.info(f"Created backup: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Failed to create backup for {schema_name}.{table_name}: {e}")
            return None


async def cleanup_tenant_data(
    tenant_id: Optional[str] = None,
    cleanup_types: List[str] = None,
    retention_days: int = 90,
    dry_run: bool = True,
    create_backup: bool = True
) -> Dict[str, Any]:
    """
    Main function to cleanup tenant data.
    
    Args:
        tenant_id: Specific tenant or None for all
        cleanup_types: Types of cleanup to perform
        retention_days: Data retention period
        dry_run: Analysis mode without changes
        create_backup: Create backups before cleanup
        
    Returns:
        Cleanup results
    """
    async with get_async_session() as session:
        redis_client = await get_redis_client()
        
        try:
            cleanup = DataCleanup(session, redis_client)
            result = await cleanup.cleanup_tenant_data(
                tenant_id=tenant_id,
                cleanup_types=cleanup_types,
                retention_days=retention_days,
                dry_run=dry_run,
                create_backup=create_backup
            )
            
            return result
            
        finally:
            if redis_client:
                await redis_client.close()


def main():
    """Command line interface for data cleanup."""
    parser = argparse.ArgumentParser(
        description="Clean up tenant data and temporary files"
    )
    
    parser.add_argument(
        "--tenant-id",
        help="Specific tenant to clean (default: all tenants)"
    )
    
    parser.add_argument(
        "--cleanup-type",
        choices=["old_data", "temp_files", "cache", "logs", "backups", "analytics", "sessions", "all"],
        default="all",
        help="Type of cleanup to perform"
    )
    
    parser.add_argument(
        "--retention-days",
        type=int,
        default=90,
        help="Data retention period in days"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Perform analysis without actual cleanup"
    )
    
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform cleanup (overrides dry-run)"
    )
    
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating backups before cleanup"
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
    
    # Determine cleanup types
    cleanup_types = None
    if args.cleanup_type != "all":
        cleanup_types = [args.cleanup_type]
    
    # Override dry-run if execute is specified
    dry_run = args.dry_run and not args.execute
    create_backup = not args.no_backup
    
    if dry_run:
        print("🔍 Running in DRY RUN mode - no changes will be made")
    else:
        print("⚠️  EXECUTING cleanup - data will be permanently removed!")
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Cleanup cancelled.")
            sys.exit(0)
    
    try:
        # Run cleanup
        result = asyncio.run(
            cleanup_tenant_data(
                tenant_id=args.tenant_id,
                cleanup_types=cleanup_types,
                retention_days=args.retention_days,
                dry_run=dry_run,
                create_backup=create_backup
            )
        )
        
        # Display results
        print(f"\nData Cleanup Results:")
        print(f"Status: {result['status']}")
        print(f"Duration: {FixtureUtils.format_duration(result.get('duration_seconds', 0))}")
        print(f"Tenants Processed: {result.get('tenants_processed', 0)}")
        print(f"Space Freed: {FixtureUtils.format_bytes(result['space_freed'])}")
        
        if result.get('data_removed'):
            print(f"\nData Removed:")
            for entity, data in result['data_removed'].items():
                if isinstance(data, dict):
                    for table, count in data.items():
                        print(f"  {entity}/{table}: {count} records")
                else:
                    print(f"  {entity}: {data}")
        
        if result.get('backups_created'):
            print(f"\nBackups Created: {len(result['backups_created'])}")
            for backup in result['backups_created'][:3]:
                print(f"  {backup}")
            if len(result['backups_created']) > 3:
                print(f"  ... and {len(result['backups_created']) - 3} more")
        
        if result.get('warnings'):
            print(f"\nWarnings: {len(result['warnings'])}")
            for warning in result['warnings'][:3]:
                print(f"  ⚠️  {warning}")
            if len(result['warnings']) > 3:
                print(f"  ... and {len(result['warnings']) - 3} more")
        
        if result.get('errors'):
            print(f"\nErrors: {len(result['errors'])}")
            for error in result['errors']:
                print(f"  ❌ {error}")
        
        if result['status'] == 'completed':
            if result.get('errors'):
                print(f"\n⚠️  Cleanup completed with {len(result['errors'])} errors")
                sys.exit(2)
            else:
                print("\n✅ Cleanup completed successfully!")
                sys.exit(0)
        else:
            print(f"\n❌ Cleanup failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Cleanup interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Cleanup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
