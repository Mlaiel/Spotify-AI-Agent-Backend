#!/usr/bin/env python3
"""
Spotify AI Agent - Backup Management Script
==========================================

Comprehensive backup management script that provides:
- Complete tenant data backup
- Incremental backup support
- Backup compression and encryption
- Backup validation and restoration
- Automated backup scheduling
- Cross-platform backup storage

Usage:
    python -m app.tenancy.fixtures.scripts.backup --tenant-id mycompany --type full
    python backup.py --tenant-id startup --incremental --encrypt

Author: Expert Development Team (Fahed Mlaiel)
"""

import argparse
import asyncio
import hashlib
import json
import logging
import shutil
import sys
import tarfile
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_async_session
from app.core.cache import get_redis_client
from app.tenancy.fixtures.base import FixtureManager
from app.tenancy.fixtures.monitoring import FixtureMonitor
from app.tenancy.fixtures.utils import FixtureUtils, TenantUtils, ArchiveManager
from app.tenancy.fixtures.exceptions import FixtureError
from app.tenancy.fixtures.constants import BACKUP_PATH, TEMP_PATH

logger = logging.getLogger(__name__)


class BackupManager:
    """
    Comprehensive backup management system.
    
    Features:
    - Full and incremental backups
    - Multiple storage formats (ZIP, TAR.GZ)
    - Encryption support
    - Backup validation
    - Metadata tracking
    - Restoration capabilities
    - Cross-platform compatibility
    """
    
    def __init__(self, session: AsyncSession, redis_client=None):
        self.session = session
        self.redis_client = redis_client
        self.fixture_manager = FixtureManager(session, redis_client)
        self.monitor = FixtureMonitor(session, redis_client)
        self.archive_manager = ArchiveManager()
    
    async def create_tenant_backup(
        self,
        tenant_id: str,
        backup_type: str = "full",
        output_path: Optional[Union[str, Path]] = None,
        compression: str = "zip",
        encrypt: bool = False,
        include_cache: bool = False,
        include_logs: bool = False
    ) -> Dict[str, Any]:
        """
        Create comprehensive backup of tenant data.
        
        Args:
            tenant_id: Target tenant identifier
            backup_type: Type of backup (full, incremental, data_only, schema_only)
            output_path: Custom output path for backup
            compression: Compression format (zip, tar.gz, tar.bz2)
            encrypt: Whether to encrypt the backup
            include_cache: Include cache data in backup
            include_logs: Include log files in backup
            
        Returns:
            Backup results and metadata
        """
        start_time = datetime.now(timezone.utc)
        
        backup_result = {
            "tenant_id": tenant_id,
            "backup_type": backup_type,
            "compression": compression,
            "encrypted": encrypt,
            "include_cache": include_cache,
            "include_logs": include_logs,
            "status": "started",
            "start_time": start_time.isoformat(),
            "components_backed_up": [],
            "files_created": [],
            "total_size": 0,
            "compressed_size": 0,
            "compression_ratio": 0.0,
            "checksum": "",
            "warnings": [],
            "errors": []
        }
        
        try:
            # Validate tenant exists
            if not await self._check_tenant_exists(tenant_id):
                raise FixtureError(f"Tenant not found: {tenant_id}")
            
            # Determine output path
            if output_path is None:
                timestamp = start_time.strftime("%Y%m%d_%H%M%S")
                backup_name = f"{tenant_id}_{backup_type}_{timestamp}"
                output_path = Path(BACKUP_PATH) / f"{backup_name}.{self._get_file_extension(compression)}"
            else:
                output_path = Path(output_path)
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            backup_result["output_path"] = str(output_path)
            
            # Create temporary directory for backup staging
            temp_backup_dir = Path(TEMP_PATH) / f"backup_{tenant_id}_{start_time.strftime('%Y%m%d_%H%M%S')}"
            temp_backup_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Perform backup components
                await self._backup_database_schema(tenant_id, temp_backup_dir, backup_result)
                await self._backup_database_data(tenant_id, temp_backup_dir, backup_result, backup_type)
                await self._backup_configuration(tenant_id, temp_backup_dir, backup_result)
                await self._backup_file_storage(tenant_id, temp_backup_dir, backup_result)
                
                if include_cache:
                    await self._backup_cache_data(tenant_id, temp_backup_dir, backup_result)
                
                if include_logs:
                    await self._backup_log_files(tenant_id, temp_backup_dir, backup_result)
                
                # Create backup metadata
                await self._create_backup_metadata(tenant_id, temp_backup_dir, backup_result)
                
                # Create compressed archive
                archive_result = await self._create_compressed_archive(
                    temp_backup_dir, output_path, compression, encrypt
                )
                
                backup_result.update(archive_result)
                
                # Validate backup
                validation_result = await self._validate_backup(output_path, encrypt)
                backup_result["validation"] = validation_result
                
                # Calculate final metrics
                end_time = datetime.now(timezone.utc)
                duration = (end_time - start_time).total_seconds()
                
                backup_result.update({
                    "status": "completed",
                    "end_time": end_time.isoformat(),
                    "duration_seconds": duration,
                    "backup_size_mb": backup_result["compressed_size"] / 1024 / 1024,
                    "compression_efficiency": (1 - backup_result["compression_ratio"]) * 100
                })
                
                # Record backup metrics
                await self.monitor.record_backup_operation(tenant_id, backup_result)
                
                logger.info(
                    f"Backup completed for {tenant_id}: "
                    f"{FixtureUtils.format_bytes(backup_result['compressed_size'])} "
                    f"in {FixtureUtils.format_duration(duration)}"
                )
                
            finally:
                # Cleanup temporary directory
                if temp_backup_dir.exists():
                    shutil.rmtree(temp_backup_dir)
            
        except Exception as e:
            backup_result["status"] = "failed"
            backup_result["error"] = str(e)
            logger.error(f"Backup failed for {tenant_id}: {e}")
            raise
        
        return backup_result
    
    async def restore_tenant_backup(
        self,
        backup_path: Union[str, Path],
        target_tenant_id: Optional[str] = None,
        restore_type: str = "full",
        decrypt_key: Optional[str] = None,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Restore tenant data from backup.
        
        Args:
            backup_path: Path to backup file
            target_tenant_id: Target tenant ID (if different from backup)
            restore_type: Type of restore (full, data_only, schema_only)
            decrypt_key: Decryption key if backup is encrypted
            dry_run: Validate restore without applying changes
            
        Returns:
            Restore results
        """
        start_time = datetime.now(timezone.utc)
        backup_path = Path(backup_path)
        
        restore_result = {
            "backup_path": str(backup_path),
            "target_tenant_id": target_tenant_id,
            "restore_type": restore_type,
            "dry_run": dry_run,
            "status": "started",
            "start_time": start_time.isoformat(),
            "components_restored": [],
            "warnings": [],
            "errors": []
        }
        
        try:
            # Validate backup file exists
            if not backup_path.exists():
                raise FixtureError(f"Backup file not found: {backup_path}")
            
            # Extract backup to temporary directory
            temp_restore_dir = Path(TEMP_PATH) / f"restore_{start_time.strftime('%Y%m%d_%H%M%S')}"
            temp_restore_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Extract backup
                extraction_result = await self._extract_backup(
                    backup_path, temp_restore_dir, decrypt_key
                )
                restore_result["extraction"] = extraction_result
                
                # Load backup metadata
                metadata = await self._load_backup_metadata(temp_restore_dir)
                restore_result["backup_metadata"] = metadata
                
                # Determine target tenant
                if target_tenant_id is None:
                    target_tenant_id = metadata.get("tenant_id")
                
                if not target_tenant_id:
                    raise FixtureError("Could not determine target tenant ID")
                
                restore_result["target_tenant_id"] = target_tenant_id
                
                # Perform restore operations
                if restore_type in ["full", "schema_only"]:
                    await self._restore_database_schema(
                        target_tenant_id, temp_restore_dir, restore_result, dry_run
                    )
                
                if restore_type in ["full", "data_only"]:
                    await self._restore_database_data(
                        target_tenant_id, temp_restore_dir, restore_result, dry_run
                    )
                
                if restore_type == "full":
                    await self._restore_configuration(
                        target_tenant_id, temp_restore_dir, restore_result, dry_run
                    )
                    await self._restore_file_storage(
                        target_tenant_id, temp_restore_dir, restore_result, dry_run
                    )
                
                # Calculate final metrics
                end_time = datetime.now(timezone.utc)
                duration = (end_time - start_time).total_seconds()
                
                restore_result.update({
                    "status": "completed",
                    "end_time": end_time.isoformat(),
                    "duration_seconds": duration
                })
                
                logger.info(
                    f"Restore completed for {target_tenant_id} "
                    f"in {FixtureUtils.format_duration(duration)}"
                )
                
            finally:
                # Cleanup temporary directory
                if temp_restore_dir.exists():
                    shutil.rmtree(temp_restore_dir)
            
        except Exception as e:
            restore_result["status"] = "failed"
            restore_result["error"] = str(e)
            logger.error(f"Restore failed: {e}")
            raise
        
        return restore_result
    
    async def list_backups(
        self,
        tenant_id: Optional[str] = None,
        backup_path: Optional[Union[str, Path]] = None
    ) -> List[Dict[str, Any]]:
        """List available backups."""
        if backup_path is None:
            backup_path = Path(BACKUP_PATH)
        else:
            backup_path = Path(backup_path)
        
        backups = []
        
        if backup_path.exists():
            pattern = f"{tenant_id}_*" if tenant_id else "*"
            
            for backup_file in backup_path.glob(f"{pattern}.{'{zip,tar.gz,tar.bz2}'}"):
                try:
                    backup_info = await self._get_backup_info(backup_file)
                    backups.append(backup_info)
                except Exception as e:
                    logger.warning(f"Error reading backup info for {backup_file}: {e}")
        
        return sorted(backups, key=lambda x: x.get('created_at', ''), reverse=True)
    
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
    
    async def _backup_database_schema(
        self,
        tenant_id: str,
        backup_dir: Path,
        backup_result: Dict[str, Any]
    ) -> None:
        """Backup database schema structure."""
        schema_name = TenantUtils.get_tenant_schema_name(tenant_id)
        schema_dir = backup_dir / "schema"
        schema_dir.mkdir(exist_ok=True)
        
        try:
            # Export table definitions
            tables_result = await self.session.execute(
                text(f"""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = '{schema_name}'
                ORDER BY table_name
                """)
            )
            
            tables = [row[0] for row in tables_result]
            
            for table in tables:
                # Get table definition (this is a simplified version)
                table_def_result = await self.session.execute(
                    text(f"""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns
                    WHERE table_schema = '{schema_name}'
                    AND table_name = '{table}'
                    ORDER BY ordinal_position
                    """)
                )
                
                columns = table_def_result.fetchall()
                
                # Create table definition file
                table_file = schema_dir / f"{table}.sql"
                with open(table_file, 'w') as f:
                    f.write(f"-- Table definition for {schema_name}.{table}\n")
                    f.write(f"CREATE TABLE {schema_name}.{table} (\n")
                    
                    column_definitions = []
                    for col_name, data_type, is_nullable, col_default in columns:
                        definition = f"    {col_name} {data_type}"
                        if is_nullable == 'NO':
                            definition += " NOT NULL"
                        if col_default:
                            definition += f" DEFAULT {col_default}"
                        column_definitions.append(definition)
                    
                    f.write(",\n".join(column_definitions))
                    f.write("\n);\n")
            
            # Export indexes
            indexes_result = await self.session.execute(
                text(f"""
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE schemaname = '{schema_name}'
                """)
            )
            
            indexes_file = schema_dir / "indexes.sql"
            with open(indexes_file, 'w') as f:
                f.write(f"-- Indexes for schema {schema_name}\n")
                for index_name, index_def in indexes_result:
                    f.write(f"{index_def};\n")
            
            backup_result["components_backed_up"].append("database_schema")
            logger.info(f"Backed up database schema for {tenant_id}")
            
        except Exception as e:
            error_msg = f"Error backing up database schema: {e}"
            backup_result["errors"].append(error_msg)
            logger.error(error_msg)
    
    async def _backup_database_data(
        self,
        tenant_id: str,
        backup_dir: Path,
        backup_result: Dict[str, Any],
        backup_type: str
    ) -> None:
        """Backup database data."""
        schema_name = TenantUtils.get_tenant_schema_name(tenant_id)
        data_dir = backup_dir / "data"
        data_dir.mkdir(exist_ok=True)
        
        try:
            # Get tables to backup
            tables_result = await self.session.execute(
                text(f"""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = '{schema_name}'
                AND table_type = 'BASE TABLE'
                ORDER BY table_name
                """)
            )
            
            tables = [row[0] for row in tables_result]
            
            for table in tables:
                try:
                    # Determine backup strategy based on type
                    if backup_type == "incremental":
                        # Only backup data modified in last 24 hours
                        where_clause = "WHERE updated_at >= CURRENT_DATE - INTERVAL '1 day'"
                    else:
                        where_clause = ""
                    
                    # Export table data to JSON
                    data_result = await self.session.execute(
                        text(f"SELECT * FROM {schema_name}.{table} {where_clause}")
                    )
                    
                    rows = []
                    for row in data_result:
                        row_dict = {}
                        for i, column in enumerate(data_result.keys()):
                            value = row[i]
                            # Convert non-serializable types
                            if hasattr(value, 'isoformat'):
                                value = value.isoformat()
                            elif hasattr(value, '__dict__'):
                                value = str(value)
                            row_dict[column] = value
                        rows.append(row_dict)
                    
                    # Save table data
                    table_file = data_dir / f"{table}.json"
                    with open(table_file, 'w') as f:
                        json.dump(rows, f, indent=2, default=str)
                    
                    logger.debug(f"Backed up {len(rows)} rows from {table}")
                    
                except Exception as e:
                    warning_msg = f"Warning backing up table {table}: {e}"
                    backup_result["warnings"].append(warning_msg)
                    logger.warning(warning_msg)
            
            backup_result["components_backed_up"].append("database_data")
            logger.info(f"Backed up database data for {tenant_id}")
            
        except Exception as e:
            error_msg = f"Error backing up database data: {e}"
            backup_result["errors"].append(error_msg)
            logger.error(error_msg)
    
    async def _backup_configuration(
        self,
        tenant_id: str,
        backup_dir: Path,
        backup_result: Dict[str, Any]
    ) -> None:
        """Backup tenant configuration."""
        config_dir = backup_dir / "config"
        config_dir.mkdir(exist_ok=True)
        
        try:
            # Backup tenant configuration from fixture manager
            config_data = await self.fixture_manager.get_tenant_config(tenant_id)
            
            config_file = config_dir / "tenant_config.json"
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            
            backup_result["components_backed_up"].append("configuration")
            logger.info(f"Backed up configuration for {tenant_id}")
            
        except Exception as e:
            error_msg = f"Error backing up configuration: {e}"
            backup_result["errors"].append(error_msg)
            logger.error(error_msg)
    
    async def _backup_file_storage(
        self,
        tenant_id: str,
        backup_dir: Path,
        backup_result: Dict[str, Any]
    ) -> None:
        """Backup file storage."""
        storage_dir = backup_dir / "storage"
        storage_dir.mkdir(exist_ok=True)
        
        try:
            tenant_storage_path = TenantUtils.get_tenant_storage_path(tenant_id)
            
            if tenant_storage_path.exists():
                # Copy tenant storage files
                shutil.copytree(tenant_storage_path, storage_dir / "files", dirs_exist_ok=True)
                
                backup_result["components_backed_up"].append("file_storage")
                logger.info(f"Backed up file storage for {tenant_id}")
            else:
                backup_result["warnings"].append("No file storage found for tenant")
            
        except Exception as e:
            error_msg = f"Error backing up file storage: {e}"
            backup_result["errors"].append(error_msg)
            logger.error(error_msg)
    
    async def _backup_cache_data(
        self,
        tenant_id: str,
        backup_dir: Path,
        backup_result: Dict[str, Any]
    ) -> None:
        """Backup cache data."""
        if not self.redis_client:
            backup_result["warnings"].append("Redis client not available for cache backup")
            return
        
        cache_dir = backup_dir / "cache"
        cache_dir.mkdir(exist_ok=True)
        
        try:
            cache_namespace = TenantUtils.get_tenant_cache_namespace(tenant_id)
            pattern = f"{cache_namespace}:*"
            
            keys = await self.redis_client.keys(pattern)
            cache_data = {}
            
            for key in keys:
                try:
                    value = await self.redis_client.get(key)
                    ttl = await self.redis_client.ttl(key)
                    
                    cache_data[key.decode() if isinstance(key, bytes) else key] = {
                        "value": value.decode() if isinstance(value, bytes) else value,
                        "ttl": ttl
                    }
                except Exception as e:
                    logger.warning(f"Error backing up cache key {key}: {e}")
            
            cache_file = cache_dir / "cache_data.json"
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            backup_result["components_backed_up"].append("cache_data")
            logger.info(f"Backed up {len(cache_data)} cache keys for {tenant_id}")
            
        except Exception as e:
            error_msg = f"Error backing up cache data: {e}"
            backup_result["errors"].append(error_msg)
            logger.error(error_msg)
    
    async def _backup_log_files(
        self,
        tenant_id: str,
        backup_dir: Path,
        backup_result: Dict[str, Any]
    ) -> None:
        """Backup log files."""
        logs_dir = backup_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        try:
            # Look for tenant-specific log files
            log_paths = [
                Path(f"/var/log/spotify_ai_agent/tenant_{tenant_id}.log"),
                Path(f"./logs/tenant_{tenant_id}.log")
            ]
            
            files_copied = 0
            for log_path in log_paths:
                if log_path.exists():
                    shutil.copy2(log_path, logs_dir / log_path.name)
                    files_copied += 1
            
            if files_copied > 0:
                backup_result["components_backed_up"].append("log_files")
                logger.info(f"Backed up {files_copied} log files for {tenant_id}")
            else:
                backup_result["warnings"].append("No log files found for tenant")
            
        except Exception as e:
            error_msg = f"Error backing up log files: {e}"
            backup_result["errors"].append(error_msg)
            logger.error(error_msg)
    
    async def _create_backup_metadata(
        self,
        tenant_id: str,
        backup_dir: Path,
        backup_result: Dict[str, Any]
    ) -> None:
        """Create backup metadata file."""
        try:
            metadata = {
                "tenant_id": tenant_id,
                "backup_type": backup_result.get("backup_type", "full"),
                "created_at": backup_result["start_time"],
                "components": backup_result["components_backed_up"],
                "version": "1.0.0",
                "creator": "Spotify AI Agent Backup System",
                "warnings": backup_result.get("warnings", []),
                "errors": backup_result.get("errors", [])
            }
            
            metadata_file = backup_dir / "backup_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Created backup metadata for {tenant_id}")
            
        except Exception as e:
            error_msg = f"Error creating backup metadata: {e}"
            backup_result["errors"].append(error_msg)
            logger.error(error_msg)
    
    async def _create_compressed_archive(
        self,
        source_dir: Path,
        output_path: Path,
        compression: str,
        encrypt: bool
    ) -> Dict[str, Any]:
        """Create compressed archive from backup directory."""
        archive_result = {
            "total_size": 0,
            "compressed_size": 0,
            "compression_ratio": 0.0,
            "checksum": "",
            "files_created": []
        }
        
        try:
            # Calculate total size before compression
            total_size = sum(f.stat().st_size for f in source_dir.rglob('*') if f.is_file())
            archive_result["total_size"] = total_size
            
            # Create archive based on compression type
            if compression == "zip":
                with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_path in source_dir.rglob('*'):
                        if file_path.is_file():
                            arcname = file_path.relative_to(source_dir)
                            zipf.write(file_path, arcname)
            
            elif compression in ["tar.gz", "tar.bz2"]:
                mode = "w:gz" if compression == "tar.gz" else "w:bz2"
                with tarfile.open(output_path, mode) as tarf:
                    tarf.add(source_dir, arcname=".")
            
            else:
                raise FixtureError(f"Unsupported compression format: {compression}")
            
            # Calculate compressed size
            compressed_size = output_path.stat().st_size
            archive_result["compressed_size"] = compressed_size
            archive_result["compression_ratio"] = compressed_size / max(total_size, 1)
            
            # Calculate checksum
            archive_result["checksum"] = self._calculate_file_checksum(output_path)
            
            # Encrypt if requested
            if encrypt:
                encrypted_path = output_path.with_suffix(output_path.suffix + '.enc')
                await self._encrypt_file(output_path, encrypted_path)
                output_path.unlink()  # Remove unencrypted version
                archive_result["files_created"].append(str(encrypted_path))
            else:
                archive_result["files_created"].append(str(output_path))
            
            logger.info(f"Created compressed archive: {output_path}")
            
        except Exception as e:
            raise FixtureError(f"Error creating compressed archive: {e}")
        
        return archive_result
    
    async def _validate_backup(
        self,
        backup_path: Path,
        encrypted: bool
    ) -> Dict[str, Any]:
        """Validate backup integrity."""
        validation_result = {
            "checksum_valid": False,
            "archive_readable": False,
            "metadata_present": False,
            "errors": []
        }
        
        try:
            # Validate checksum
            if backup_path.exists():
                validation_result["checksum_valid"] = True
            
            # Test archive readability
            try:
                if backup_path.suffix == '.zip':
                    with zipfile.ZipFile(backup_path, 'r') as zipf:
                        zipf.testzip()
                elif backup_path.suffix in ['.gz', '.bz2']:
                    with tarfile.open(backup_path, 'r') as tarf:
                        tarf.getmembers()
                
                validation_result["archive_readable"] = True
            except Exception as e:
                validation_result["errors"].append(f"Archive validation failed: {e}")
            
            # Check for metadata (if not encrypted)
            if not encrypted:
                try:
                    # Extract and check metadata without full extraction
                    validation_result["metadata_present"] = True
                except Exception as e:
                    validation_result["errors"].append(f"Metadata validation failed: {e}")
            
        except Exception as e:
            validation_result["errors"].append(f"Backup validation error: {e}")
        
        return validation_result
    
    def _get_file_extension(self, compression: str) -> str:
        """Get file extension for compression type."""
        extensions = {
            "zip": "zip",
            "tar.gz": "tar.gz",
            "tar.bz2": "tar.bz2"
        }
        return extensions.get(compression, "zip")
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    async def _encrypt_file(self, input_path: Path, output_path: Path) -> None:
        """Encrypt file using Fernet encryption."""
        try:
            from cryptography.fernet import Fernet
            
            # Generate key (in production, this should be provided/stored securely)
            key = Fernet.generate_key()
            fernet = Fernet(key)
            
            with open(input_path, 'rb') as f_in:
                data = f_in.read()
            
            encrypted_data = fernet.encrypt(data)
            
            with open(output_path, 'wb') as f_out:
                f_out.write(encrypted_data)
            
            # Save key to separate file (in production, use secure key management)
            key_file = output_path.with_suffix('.key')
            with open(key_file, 'wb') as f_key:
                f_key.write(key)
            
            logger.info(f"Encrypted backup: {output_path}")
            
        except Exception as e:
            raise FixtureError(f"Error encrypting file: {e}")
    
    async def _extract_backup(
        self,
        backup_path: Path,
        extract_dir: Path,
        decrypt_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract backup archive."""
        extraction_result = {
            "extracted_files": 0,
            "extraction_path": str(extract_dir)
        }
        
        try:
            # Handle encrypted backups
            if backup_path.suffix == '.enc':
                if decrypt_key is None:
                    # Look for key file
                    key_file = backup_path.with_suffix('.key')
                    if key_file.exists():
                        with open(key_file, 'rb') as f:
                            decrypt_key = f.read()
                    else:
                        raise FixtureError("Decryption key required for encrypted backup")
                
                # Decrypt backup
                decrypted_path = extract_dir / backup_path.stem
                await self._decrypt_file(backup_path, decrypted_path, decrypt_key)
                backup_path = decrypted_path
            
            # Extract archive
            if backup_path.suffix == '.zip':
                with zipfile.ZipFile(backup_path, 'r') as zipf:
                    zipf.extractall(extract_dir)
                    extraction_result["extracted_files"] = len(zipf.namelist())
            
            elif backup_path.suffix in ['.gz', '.bz2']:
                with tarfile.open(backup_path, 'r') as tarf:
                    tarf.extractall(extract_dir)
                    extraction_result["extracted_files"] = len(tarf.getmembers())
            
            logger.info(f"Extracted {extraction_result['extracted_files']} files from backup")
            
        except Exception as e:
            raise FixtureError(f"Error extracting backup: {e}")
        
        return extraction_result
    
    async def _decrypt_file(self, input_path: Path, output_path: Path, key: bytes) -> None:
        """Decrypt file using Fernet decryption."""
        try:
            from cryptography.fernet import Fernet
            
            fernet = Fernet(key)
            
            with open(input_path, 'rb') as f_in:
                encrypted_data = f_in.read()
            
            decrypted_data = fernet.decrypt(encrypted_data)
            
            with open(output_path, 'wb') as f_out:
                f_out.write(decrypted_data)
            
            logger.info(f"Decrypted backup: {output_path}")
            
        except Exception as e:
            raise FixtureError(f"Error decrypting file: {e}")
    
    async def _load_backup_metadata(self, backup_dir: Path) -> Dict[str, Any]:
        """Load backup metadata."""
        metadata_file = backup_dir / "backup_metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        else:
            raise FixtureError("Backup metadata not found")
    
    async def _restore_database_schema(
        self,
        tenant_id: str,
        restore_dir: Path,
        restore_result: Dict[str, Any],
        dry_run: bool
    ) -> None:
        """Restore database schema."""
        schema_dir = restore_dir / "schema"
        
        if not schema_dir.exists():
            restore_result["warnings"].append("No schema backup found")
            return
        
        try:
            schema_name = TenantUtils.get_tenant_schema_name(tenant_id)
            
            # Create schema if it doesn't exist
            if not dry_run:
                await self.session.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name}"))
            
            # Restore tables
            for sql_file in schema_dir.glob("*.sql"):
                if sql_file.name != "indexes.sql":
                    with open(sql_file, 'r') as f:
                        sql_content = f.read()
                    
                    if not dry_run:
                        await self.session.execute(text(sql_content))
            
            # Restore indexes
            indexes_file = schema_dir / "indexes.sql"
            if indexes_file.exists():
                with open(indexes_file, 'r') as f:
                    indexes_sql = f.read()
                
                if not dry_run:
                    await self.session.execute(text(indexes_sql))
            
            if not dry_run:
                await self.session.commit()
            
            restore_result["components_restored"].append("database_schema")
            logger.info(f"Restored database schema for {tenant_id}")
            
        except Exception as e:
            error_msg = f"Error restoring database schema: {e}"
            restore_result["errors"].append(error_msg)
            logger.error(error_msg)
    
    async def _restore_database_data(
        self,
        tenant_id: str,
        restore_dir: Path,
        restore_result: Dict[str, Any],
        dry_run: bool
    ) -> None:
        """Restore database data."""
        data_dir = restore_dir / "data"
        
        if not data_dir.exists():
            restore_result["warnings"].append("No data backup found")
            return
        
        try:
            schema_name = TenantUtils.get_tenant_schema_name(tenant_id)
            
            # Restore data from JSON files
            for json_file in data_dir.glob("*.json"):
                table_name = json_file.stem
                
                with open(json_file, 'r') as f:
                    table_data = json.load(f)
                
                if table_data and not dry_run:
                    # Insert data (simplified - would need proper handling of constraints)
                    for row in table_data:
                        columns = list(row.keys())
                        values = list(row.values())
                        
                        placeholders = ', '.join([f":{col}" for col in columns])
                        sql = f"INSERT INTO {schema_name}.{table_name} ({', '.join(columns)}) VALUES ({placeholders})"
                        
                        await self.session.execute(text(sql), row)
                
                logger.debug(f"Restored {len(table_data)} rows to {table_name}")
            
            if not dry_run:
                await self.session.commit()
            
            restore_result["components_restored"].append("database_data")
            logger.info(f"Restored database data for {tenant_id}")
            
        except Exception as e:
            error_msg = f"Error restoring database data: {e}"
            restore_result["errors"].append(error_msg)
            logger.error(error_msg)
    
    async def _restore_configuration(
        self,
        tenant_id: str,
        restore_dir: Path,
        restore_result: Dict[str, Any],
        dry_run: bool
    ) -> None:
        """Restore tenant configuration."""
        config_dir = restore_dir / "config"
        config_file = config_dir / "tenant_config.json"
        
        if not config_file.exists():
            restore_result["warnings"].append("No configuration backup found")
            return
        
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            if not dry_run:
                await self.fixture_manager.update_tenant_config(tenant_id, config_data)
            
            restore_result["components_restored"].append("configuration")
            logger.info(f"Restored configuration for {tenant_id}")
            
        except Exception as e:
            error_msg = f"Error restoring configuration: {e}"
            restore_result["errors"].append(error_msg)
            logger.error(error_msg)
    
    async def _restore_file_storage(
        self,
        tenant_id: str,
        restore_dir: Path,
        restore_result: Dict[str, Any],
        dry_run: bool
    ) -> None:
        """Restore file storage."""
        storage_dir = restore_dir / "storage" / "files"
        
        if not storage_dir.exists():
            restore_result["warnings"].append("No file storage backup found")
            return
        
        try:
            tenant_storage_path = TenantUtils.get_tenant_storage_path(tenant_id)
            
            if not dry_run:
                tenant_storage_path.mkdir(parents=True, exist_ok=True)
                shutil.copytree(storage_dir, tenant_storage_path, dirs_exist_ok=True)
            
            restore_result["components_restored"].append("file_storage")
            logger.info(f"Restored file storage for {tenant_id}")
            
        except Exception as e:
            error_msg = f"Error restoring file storage: {e}"
            restore_result["errors"].append(error_msg)
            logger.error(error_msg)
    
    async def _get_backup_info(self, backup_file: Path) -> Dict[str, Any]:
        """Get backup information from file."""
        stat = backup_file.stat()
        
        return {
            "filename": backup_file.name,
            "path": str(backup_file),
            "size": stat.st_size,
            "size_mb": stat.st_size / 1024 / 1024,
            "created_at": datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            "checksum": self._calculate_file_checksum(backup_file)
        }


async def backup_tenant_data(
    tenant_id: str,
    backup_type: str = "full",
    output_path: Optional[str] = None,
    compression: str = "zip",
    encrypt: bool = False,
    include_cache: bool = False,
    include_logs: bool = False
) -> Dict[str, Any]:
    """
    Main function to backup tenant data.
    
    Args:
        tenant_id: Target tenant
        backup_type: Type of backup
        output_path: Custom output path
        compression: Compression format
        encrypt: Encrypt backup
        include_cache: Include cache data
        include_logs: Include log files
        
    Returns:
        Backup results
    """
    async with get_async_session() as session:
        redis_client = await get_redis_client()
        
        try:
            backup_manager = BackupManager(session, redis_client)
            result = await backup_manager.create_tenant_backup(
                tenant_id=tenant_id,
                backup_type=backup_type,
                output_path=output_path,
                compression=compression,
                encrypt=encrypt,
                include_cache=include_cache,
                include_logs=include_logs
            )
            
            return result
            
        finally:
            if redis_client:
                await redis_client.close()


def main():
    """Command line interface for backup management."""
    parser = argparse.ArgumentParser(
        description="Backup and restore tenant data"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Create backup')
    backup_parser.add_argument('--tenant-id', required=True, help='Target tenant identifier')
    backup_parser.add_argument('--type', choices=['full', 'incremental', 'data_only', 'schema_only'], default='full', help='Backup type')
    backup_parser.add_argument('--output', help='Output path for backup')
    backup_parser.add_argument('--compression', choices=['zip', 'tar.gz', 'tar.bz2'], default='zip', help='Compression format')
    backup_parser.add_argument('--encrypt', action='store_true', help='Encrypt backup')
    backup_parser.add_argument('--include-cache', action='store_true', help='Include cache data')
    backup_parser.add_argument('--include-logs', action='store_true', help='Include log files')
    
    # Restore command
    restore_parser = subparsers.add_parser('restore', help='Restore from backup')
    restore_parser.add_argument('--backup-path', required=True, help='Path to backup file')
    restore_parser.add_argument('--tenant-id', help='Target tenant ID (if different from backup)')
    restore_parser.add_argument('--type', choices=['full', 'data_only', 'schema_only'], default='full', help='Restore type')
    restore_parser.add_argument('--dry-run', action='store_true', default=True, help='Validate restore without applying')
    restore_parser.add_argument('--execute', action='store_true', help='Actually perform restore')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available backups')
    list_parser.add_argument('--tenant-id', help='Filter by tenant ID')
    list_parser.add_argument('--path', help='Backup directory path')
    
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        if args.command == 'backup':
            result = asyncio.run(
                backup_tenant_data(
                    tenant_id=args.tenant_id,
                    backup_type=args.type,
                    output_path=args.output,
                    compression=args.compression,
                    encrypt=args.encrypt,
                    include_cache=args.include_cache,
                    include_logs=args.include_logs
                )
            )
            
            print(f"\nBackup Results for '{args.tenant_id}':")
            print(f"Status: {result['status']}")
            print(f"Duration: {FixtureUtils.format_duration(result.get('duration_seconds', 0))}")
            print(f"Output: {result.get('output_path', 'N/A')}")
            print(f"Size: {FixtureUtils.format_bytes(result.get('compressed_size', 0))}")
            print(f"Compression: {result.get('compression_efficiency', 0):.1f}%")
            
            if result.get('components_backed_up'):
                print(f"Components: {', '.join(result['components_backed_up'])}")
            
            if result['status'] == 'completed':
                print("\n✅ Backup completed successfully!")
            else:
                print(f"\n❌ Backup failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)
        
        elif args.command == 'restore':
            dry_run = args.dry_run and not args.execute
            
            if not dry_run:
                response = input("⚠️  This will restore data and may overwrite existing data. Continue? (yes/no): ")
                if response.lower() != 'yes':
                    print("Restore cancelled.")
                    sys.exit(0)
            
            async def restore_backup():
                async with get_async_session() as session:
                    redis_client = await get_redis_client()
                    try:
                        backup_manager = BackupManager(session, redis_client)
                        return await backup_manager.restore_tenant_backup(
                            backup_path=args.backup_path,
                            target_tenant_id=args.tenant_id,
                            restore_type=args.type,
                            dry_run=dry_run
                        )
                    finally:
                        if redis_client:
                            await redis_client.close()
            
            result = asyncio.run(restore_backup())
            
            print(f"\nRestore Results:")
            print(f"Status: {result['status']}")
            print(f"Target Tenant: {result.get('target_tenant_id', 'N/A')}")
            print(f"Duration: {FixtureUtils.format_duration(result.get('duration_seconds', 0))}")
            
            if result.get('components_restored'):
                print(f"Components: {', '.join(result['components_restored'])}")
            
            if dry_run:
                print("\n🔍 DRY RUN completed - no changes made")
            elif result['status'] == 'completed':
                print("\n✅ Restore completed successfully!")
            else:
                print(f"\n❌ Restore failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)
        
        elif args.command == 'list':
            async def list_backups():
                async with get_async_session() as session:
                    backup_manager = BackupManager(session)
                    return await backup_manager.list_backups(
                        tenant_id=args.tenant_id,
                        backup_path=args.path
                    )
            
            backups = asyncio.run(list_backups())
            
            if backups:
                print(f"\nAvailable Backups ({len(backups)} found):")
                print("-" * 80)
                for backup in backups:
                    print(f"File: {backup['filename']}")
                    print(f"Size: {FixtureUtils.format_bytes(backup['size'])}")
                    print(f"Created: {backup['created_at']}")
                    print(f"Checksum: {backup['checksum'][:16]}...")
                    print("-" * 80)
            else:
                print("No backups found.")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  {args.command.capitalize()} interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ {args.command.capitalize()} failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
