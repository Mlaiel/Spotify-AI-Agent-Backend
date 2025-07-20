"""
Spotify AI Agent - Fixture Utilities
===================================

Comprehensive utility functions and helper classes
for fixture operations, data processing, and system management.
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import tempfile
import uuid
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, AsyncGenerator, Callable
from uuid import UUID

import aiofiles
import yaml
from cryptography.fernet import Fernet
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.tenancy.fixtures.exceptions import FixtureError, FixtureDataError
from app.tenancy.fixtures.constants import (
    FIXTURE_BASE_PATH,
    BACKUP_PATH,
    TEMP_PATH,
    TEMPLATE_ENCODING
)

logger = logging.getLogger(__name__)


class FixtureUtils:
    """
    General utility functions for fixture operations.
    
    Provides:
    - File operations
    - Data serialization
    - Encryption/decryption
    - Checksum calculation
    - Template processing
    """
    
    @staticmethod
    def generate_fixture_id() -> UUID:
        """Generate a new fixture ID."""
        return uuid.uuid4()
    
    @staticmethod
    def calculate_checksum(data: Union[str, bytes, Dict[str, Any]]) -> str:
        """Calculate SHA-256 checksum of data."""
        if isinstance(data, dict):
            data = json.dumps(data, sort_keys=True)
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return hashlib.sha256(data).hexdigest()
    
    @staticmethod
    async def read_file_async(file_path: Union[str, Path]) -> str:
        """Read file content asynchronously."""
        try:
            async with aiofiles.open(file_path, mode='r', encoding=TEMPLATE_ENCODING) as f:
                return await f.read()
        except Exception as e:
            raise FixtureError(f"Failed to read file {file_path}: {str(e)}")
    
    @staticmethod
    async def write_file_async(
        file_path: Union[str, Path],
        content: str,
        create_dirs: bool = True
    ) -> None:
        """Write content to file asynchronously."""
        try:
            file_path = Path(file_path)
            
            if create_dirs:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            
            async with aiofiles.open(file_path, mode='w', encoding=TEMPLATE_ENCODING) as f:
                await f.write(content)
        except Exception as e:
            raise FixtureError(f"Failed to write file {file_path}: {str(e)}")
    
    @staticmethod
    def load_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load JSON file synchronously."""
        try:
            with open(file_path, 'r', encoding=TEMPLATE_ENCODING) as f:
                return json.load(f)
        except Exception as e:
            raise FixtureError(f"Failed to load JSON file {file_path}: {str(e)}")
    
    @staticmethod
    def save_json_file(file_path: Union[str, Path], data: Dict[str, Any]) -> None:
        """Save data to JSON file synchronously."""
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding=TEMPLATE_ENCODING) as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            raise FixtureError(f"Failed to save JSON file {file_path}: {str(e)}")
    
    @staticmethod
    def load_yaml_file(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load YAML file synchronously."""
        try:
            with open(file_path, 'r', encoding=TEMPLATE_ENCODING) as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise FixtureError(f"Failed to load YAML file {file_path}: {str(e)}")
    
    @staticmethod
    def save_yaml_file(file_path: Union[str, Path], data: Dict[str, Any]) -> None:
        """Save data to YAML file synchronously."""
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding=TEMPLATE_ENCODING) as f:
                yaml.dump(data, f, default_flow_style=False)
        except Exception as e:
            raise FixtureError(f"Failed to save YAML file {file_path}: {str(e)}")
    
    @staticmethod
    def encrypt_string(data: str, key: Optional[bytes] = None) -> str:
        """Encrypt string data."""
        if key is None:
            key = Fernet.generate_key()
        
        fernet = Fernet(key)
        encrypted_data = fernet.encrypt(data.encode())
        return encrypted_data.decode()
    
    @staticmethod
    def decrypt_string(encrypted_data: str, key: bytes) -> str:
        """Decrypt string data."""
        fernet = Fernet(key)
        decrypted_data = fernet.decrypt(encrypted_data.encode())
        return decrypted_data.decode()
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe file system usage."""
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        if len(filename) > 255:
            filename = filename[:255]
        
        return filename.strip()
    
    @staticmethod
    def format_bytes(bytes_value: int) -> str:
        """Format bytes into human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in seconds to human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    @staticmethod
    async def batch_process(
        items: List[Any],
        processor: Callable,
        batch_size: int = 100,
        max_concurrent: int = 10
    ) -> List[Any]:
        """Process items in batches with concurrency control."""
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_batch(batch: List[Any]) -> List[Any]:
            async with semaphore:
                return await processor(batch)
        
        # Create batches
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        # Process batches concurrently
        tasks = [process_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks)
        
        # Flatten results
        for batch_result in batch_results:
            results.extend(batch_result)
        
        return results


class TenantUtils:
    """
    Tenant-specific utility functions.
    
    Provides:
    - Tenant validation
    - Schema management
    - Resource calculation
    - Migration utilities
    """
    
    @staticmethod
    def validate_tenant_id(tenant_id: str) -> bool:
        """Validate tenant ID format."""
        if not tenant_id or not isinstance(tenant_id, str):
            return False
        
        # Check length
        if len(tenant_id) < 3 or len(tenant_id) > 50:
            return False
        
        # Check format: lowercase alphanumeric with hyphens and underscores
        import re
        pattern = r'^[a-z0-9_-]+$'
        return bool(re.match(pattern, tenant_id))
    
    @staticmethod
    def generate_tenant_id(base_name: str) -> str:
        """Generate a valid tenant ID from base name."""
        # Convert to lowercase and replace invalid characters
        tenant_id = base_name.lower()
        tenant_id = ''.join(c if c.isalnum() or c in '-_' else '_' for c in tenant_id)
        
        # Remove consecutive underscores/hyphens
        import re
        tenant_id = re.sub(r'[_-]+', '_', tenant_id)
        
        # Ensure it starts with alphanumeric
        if tenant_id and not tenant_id[0].isalnum():
            tenant_id = 't' + tenant_id
        
        # Truncate if too long
        if len(tenant_id) > 50:
            tenant_id = tenant_id[:50]
        
        # Ensure minimum length
        if len(tenant_id) < 3:
            tenant_id = tenant_id + '_001'
        
        return tenant_id.rstrip('_-')
    
    @staticmethod
    def get_tenant_schema_name(tenant_id: str) -> str:
        """Get database schema name for tenant."""
        return f"tenant_{tenant_id}"
    
    @staticmethod
    def get_tenant_cache_namespace(tenant_id: str) -> str:
        """Get cache namespace for tenant."""
        return f"tenant:{tenant_id}"
    
    @staticmethod
    def get_tenant_storage_path(tenant_id: str) -> Path:
        """Get storage path for tenant."""
        return Path(FIXTURE_BASE_PATH) / "tenants" / tenant_id
    
    @staticmethod
    async def calculate_tenant_resources(
        session: AsyncSession,
        tenant_id: str
    ) -> Dict[str, Any]:
        """Calculate resource usage for tenant."""
        schema = TenantUtils.get_tenant_schema_name(tenant_id)
        
        try:
            # User count
            user_result = await session.execute(
                text(f"SELECT COUNT(*) FROM {schema}.users WHERE is_active = true")
            )
            user_count = user_result.scalar() or 0
            
            # Collaboration count
            collab_result = await session.execute(
                text(f"SELECT COUNT(*) FROM {schema}.collaborations WHERE status = 'active'")
            )
            collaboration_count = collab_result.scalar() or 0
            
            # Content count
            content_result = await session.execute(
                text(f"SELECT COUNT(*) FROM {schema}.content_generated")
            )
            content_count = content_result.scalar() or 0
            
            # Storage usage (approximate)
            storage_result = await session.execute(
                text(f"""
                SELECT 
                    pg_total_relation_size(schemaname||'.'||tablename) as size
                FROM pg_tables 
                WHERE schemaname = '{schema}'
                """)
            )
            storage_bytes = sum(row[0] for row in storage_result) or 0
            
            return {
                "users": user_count,
                "collaborations": collaboration_count,
                "content_items": content_count,
                "storage_bytes": storage_bytes,
                "storage_mb": storage_bytes / 1024 / 1024
            }
            
        except Exception as e:
            logger.error(f"Error calculating tenant resources for {tenant_id}: {e}")
            return {
                "users": 0,
                "collaborations": 0,
                "content_items": 0,
                "storage_bytes": 0,
                "storage_mb": 0
            }
    
    @staticmethod
    async def cleanup_tenant_data(
        session: AsyncSession,
        tenant_id: str,
        dry_run: bool = True
    ) -> Dict[str, int]:
        """Clean up tenant data (with dry run option)."""
        schema = TenantUtils.get_tenant_schema_name(tenant_id)
        cleanup_stats = {}
        
        tables_to_clean = [
            "content_generated",
            "ai_sessions",
            "collaboration_participants",
            "collaborations",
            "spotify_connections",
            "users"
        ]
        
        try:
            for table in tables_to_clean:
                # Count records to be deleted
                count_result = await session.execute(
                    text(f"SELECT COUNT(*) FROM {schema}.{table}")
                )
                record_count = count_result.scalar() or 0
                cleanup_stats[table] = record_count
                
                if not dry_run and record_count > 0:
                    await session.execute(text(f"DELETE FROM {schema}.{table}"))
            
            if not dry_run:
                await session.commit()
                logger.info(f"Cleaned up tenant data for: {tenant_id}")
            else:
                logger.info(f"Dry run cleanup for tenant: {tenant_id}")
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Error cleaning up tenant data for {tenant_id}: {e}")
            raise FixtureError(f"Tenant cleanup failed: {str(e)}")
        
        return cleanup_stats


class ValidationUtils:
    """
    Validation utility functions.
    
    Provides:
    - Data validation helpers
    - Schema validation
    - Business rule checks
    - Format validation
    """
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email address format."""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format."""
        import re
        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        return bool(re.match(pattern, url))
    
    @staticmethod
    def validate_uuid(uuid_string: str) -> bool:
        """Validate UUID format."""
        try:
            uuid.UUID(uuid_string)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
        """Validate data against JSON schema."""
        errors = []
        
        # Required fields
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Field types
        properties = schema.get("properties", {})
        for field, field_schema in properties.items():
            if field in data:
                field_errors = ValidationUtils._validate_field_type(
                    field, data[field], field_schema
                )
                errors.extend(field_errors)
        
        return errors
    
    @staticmethod
    def _validate_field_type(
        field_name: str,
        value: Any,
        field_schema: Dict[str, Any]
    ) -> List[str]:
        """Validate individual field type."""
        errors = []
        expected_type = field_schema.get("type")
        
        if expected_type == "string" and not isinstance(value, str):
            errors.append(f"Field {field_name} must be string")
        elif expected_type == "number" and not isinstance(value, (int, float)):
            errors.append(f"Field {field_name} must be number")
        elif expected_type == "integer" and not isinstance(value, int):
            errors.append(f"Field {field_name} must be integer")
        elif expected_type == "boolean" and not isinstance(value, bool):
            errors.append(f"Field {field_name} must be boolean")
        elif expected_type == "array" and not isinstance(value, list):
            errors.append(f"Field {field_name} must be array")
        elif expected_type == "object" and not isinstance(value, dict):
            errors.append(f"Field {field_name} must be object")
        
        # String validations
        if isinstance(value, str):
            min_length = field_schema.get("minLength")
            max_length = field_schema.get("maxLength")
            
            if min_length and len(value) < min_length:
                errors.append(f"Field {field_name} too short (min: {min_length})")
            
            if max_length and len(value) > max_length:
                errors.append(f"Field {field_name} too long (max: {max_length})")
            
            pattern = field_schema.get("pattern")
            if pattern:
                import re
                if not re.match(pattern, value):
                    errors.append(f"Field {field_name} does not match pattern")
        
        # Number validations
        if isinstance(value, (int, float)):
            minimum = field_schema.get("minimum")
            maximum = field_schema.get("maximum")
            
            if minimum is not None and value < minimum:
                errors.append(f"Field {field_name} below minimum ({minimum})")
            
            if maximum is not None and value > maximum:
                errors.append(f"Field {field_name} above maximum ({maximum})")
        
        return errors
    
    @staticmethod
    def sanitize_string(
        input_string: str,
        max_length: Optional[int] = None,
        allowed_chars: Optional[str] = None
    ) -> str:
        """Sanitize string input."""
        # Remove null bytes
        sanitized = input_string.replace('\x00', '')
        
        # Filter allowed characters
        if allowed_chars:
            sanitized = ''.join(c for c in sanitized if c in allowed_chars)
        
        # Truncate if needed
        if max_length and len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized.strip()


class ConfigUtils:
    """
    Configuration utility functions.
    
    Provides:
    - Configuration loading
    - Environment variable processing
    - Template variable substitution
    - Configuration validation
    """
    
    @staticmethod
    def load_config_from_env(prefix: str = "FIXTURE_") -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                
                # Try to convert to appropriate type
                if value.lower() in ('true', 'false'):
                    config[config_key] = value.lower() == 'true'
                elif value.isdigit():
                    config[config_key] = int(value)
                elif '.' in value and value.replace('.', '').isdigit():
                    config[config_key] = float(value)
                else:
                    config[config_key] = value
        
        return config
    
    @staticmethod
    def substitute_variables(
        template: str,
        variables: Dict[str, Any]
    ) -> str:
        """Substitute variables in template string."""
        result = template
        
        for key, value in variables.items():
            placeholder = f"{{{key}}}"
            result = result.replace(placeholder, str(value))
        
        return result
    
    @staticmethod
    def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configuration dictionaries."""
        merged = {}
        
        for config in configs:
            merged.update(config)
        
        return merged
    
    @staticmethod
    def validate_config_completeness(
        config: Dict[str, Any],
        required_keys: List[str]
    ) -> List[str]:
        """Validate that configuration has all required keys."""
        missing_keys = []
        
        for key in required_keys:
            if key not in config:
                missing_keys.append(key)
        
        return missing_keys


@asynccontextmanager
async def temporary_directory() -> AsyncGenerator[Path, None]:
    """Context manager for temporary directory."""
    temp_dir = Path(tempfile.mkdtemp(dir=TEMP_PATH))
    try:
        yield temp_dir
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


@asynccontextmanager
async def backup_context(
    source_path: Union[str, Path],
    backup_name: Optional[str] = None
) -> AsyncGenerator[Path, None]:
    """Context manager for creating backups."""
    source_path = Path(source_path)
    
    if backup_name is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_name = f"{source_path.name}_{timestamp}_backup"
    
    backup_path = Path(BACKUP_PATH) / backup_name
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create backup
    if source_path.is_file():
        shutil.copy2(source_path, backup_path)
    elif source_path.is_dir():
        shutil.copytree(source_path, backup_path)
    
    try:
        yield backup_path
    finally:
        # Optionally clean up backup
        pass


class ArchiveManager:
    """
    Manager for creating and extracting fixture archives.
    
    Provides:
    - Archive creation
    - Archive extraction
    - Compression optimization
    - Integrity verification
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ArchiveManager")
    
    async def create_archive(
        self,
        source_paths: List[Union[str, Path]],
        archive_path: Union[str, Path],
        compression: str = "zip"
    ) -> Dict[str, Any]:
        """Create archive from source paths."""
        archive_path = Path(archive_path)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        
        archive_info = {
            "archive_path": str(archive_path),
            "compression": compression,
            "files_included": [],
            "total_size": 0,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            if compression == "zip":
                with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for source_path in source_paths:
                        source_path = Path(source_path)
                        
                        if source_path.is_file():
                            zipf.write(source_path, source_path.name)
                            archive_info["files_included"].append(str(source_path))
                            archive_info["total_size"] += source_path.stat().st_size
                        
                        elif source_path.is_dir():
                            for file_path in source_path.rglob("*"):
                                if file_path.is_file():
                                    arcname = file_path.relative_to(source_path.parent)
                                    zipf.write(file_path, arcname)
                                    archive_info["files_included"].append(str(file_path))
                                    archive_info["total_size"] += file_path.stat().st_size
            
            # Calculate archive size and compression ratio
            archive_size = archive_path.stat().st_size
            archive_info["archive_size"] = archive_size
            archive_info["compression_ratio"] = archive_size / archive_info["total_size"] if archive_info["total_size"] > 0 else 0
            
            self.logger.info(f"Created archive: {archive_path} ({FixtureUtils.format_bytes(archive_size)})")
            
        except Exception as e:
            error_msg = f"Failed to create archive: {str(e)}"
            self.logger.error(error_msg)
            raise FixtureError(error_msg)
        
        return archive_info
    
    async def extract_archive(
        self,
        archive_path: Union[str, Path],
        extract_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """Extract archive to specified path."""
        archive_path = Path(archive_path)
        extract_path = Path(extract_path)
        extract_path.mkdir(parents=True, exist_ok=True)
        
        extraction_info = {
            "archive_path": str(archive_path),
            "extract_path": str(extract_path),
            "files_extracted": [],
            "extracted_at": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            if archive_path.suffix.lower() == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zipf:
                    zipf.extractall(extract_path)
                    extraction_info["files_extracted"] = zipf.namelist()
            
            self.logger.info(f"Extracted archive: {archive_path} to {extract_path}")
            
        except Exception as e:
            error_msg = f"Failed to extract archive: {str(e)}"
            self.logger.error(error_msg)
            raise FixtureError(error_msg)
        
        return extraction_info
    
    def verify_archive_integrity(self, archive_path: Union[str, Path]) -> bool:
        """Verify archive integrity."""
        archive_path = Path(archive_path)
        
        try:
            if archive_path.suffix.lower() == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zipf:
                    corrupt_files = zipf.testzip()
                    return corrupt_files is None
            
        except Exception as e:
            self.logger.error(f"Error verifying archive integrity: {e}")
            return False
        
        return True
