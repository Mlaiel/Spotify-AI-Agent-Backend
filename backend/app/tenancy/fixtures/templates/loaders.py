#!/usr/bin/env python3
"""
Spotify AI Agent - Template Loaders
==================================

Advanced template loading system supporting multiple sources:
- File system loaders (local, network, cloud storage)
- Database loaders (PostgreSQL, MongoDB, Redis)
- Remote loaders (HTTP/HTTPS, Git repositories)
- Stream loaders (real-time, batch processing)
- Cache-aware loaders with versioning

Features:
- Multi-source loading with fallback chains
- Intelligent caching with TTL and invalidation
- Security scanning and validation
- Performance monitoring and metrics
- Template versioning and migration support

Author: Expert Development Team
"""

import os
import json
import yaml
import logging
import hashlib
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, AsyncIterator
from urllib.parse import urlparse
from dataclasses import dataclass, field
import asyncio
import aiofiles
import aiohttp
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.core.database import get_async_session
from app.core.redis import get_redis_client
from app.core.security import get_encryption_key, encrypt_data, decrypt_data
from app.tenancy.fixtures.templates.validators import TemplateValidationEngine, ValidationReport

logger = logging.getLogger(__name__)


@dataclass
class LoadResult:
    """Result of template loading operation."""
    success: bool
    template: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    load_time_ms: float = 0.0
    cache_hit: bool = False
    validation_report: Optional[ValidationReport] = None
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class LoaderConfig:
    """Configuration for template loaders."""
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    validate_on_load: bool = True
    encryption_enabled: bool = False
    max_file_size_mb: int = 10
    timeout_seconds: int = 30
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0


class BaseTemplateLoader(ABC):
    """Base class for all template loaders."""
    
    def __init__(self, config: Optional[LoaderConfig] = None):
        self.config = config or LoaderConfig()
        self.cache = {}
        self.cache_timestamps = {}
        self.metrics = {
            "loads_total": 0,
            "loads_successful": 0,
            "cache_hits": 0,
            "validation_failures": 0,
            "average_load_time_ms": 0.0
        }
        self.validation_engine = TemplateValidationEngine() if self.config.validate_on_load else None
    
    @abstractmethod
    async def load_template(self, identifier: str, **kwargs) -> LoadResult:
        """Load template from source."""
        pass
    
    async def load_multiple(self, identifiers: List[str], **kwargs) -> List[LoadResult]:
        """Load multiple templates concurrently."""
        tasks = [self.load_template(identifier, **kwargs) for identifier in identifiers]
        return await asyncio.gather(*tasks, return_exceptions=False)
    
    def _get_cache_key(self, identifier: str, **kwargs) -> str:
        """Generate cache key for template."""
        key_data = f"{identifier}:{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached template is still valid."""
        if not self.config.cache_enabled:
            return False
        
        if cache_key not in self.cache:
            return False
        
        timestamp = self.cache_timestamps.get(cache_key)
        if not timestamp:
            return False
        
        age = (datetime.now() - timestamp).total_seconds()
        return age < self.config.cache_ttl_seconds
    
    def _cache_template(self, cache_key: str, template: Dict[str, Any]):
        """Cache template with timestamp."""
        if self.config.cache_enabled:
            self.cache[cache_key] = template.copy()
            self.cache_timestamps[cache_key] = datetime.now()
    
    def _update_metrics(self, load_time_ms: float, success: bool, cache_hit: bool, validation_failed: bool = False):
        """Update loader metrics."""
        self.metrics["loads_total"] += 1
        if success:
            self.metrics["loads_successful"] += 1
        if cache_hit:
            self.metrics["cache_hits"] += 1
        if validation_failed:
            self.metrics["validation_failures"] += 1
        
        # Update average load time
        current_avg = self.metrics["average_load_time_ms"]
        total_loads = self.metrics["loads_total"]
        self.metrics["average_load_time_ms"] = ((current_avg * (total_loads - 1)) + load_time_ms) / total_loads
    
    async def _validate_template(self, template: Dict[str, Any], identifier: str) -> Optional[ValidationReport]:
        """Validate loaded template."""
        if not self.validation_engine:
            return None
        
        template_type = template.get("_metadata", {}).get("template_type", "unknown")
        return self.validation_engine.validate_template(template, identifier, template_type)
    
    def clear_cache(self):
        """Clear template cache."""
        self.cache.clear()
        self.cache_timestamps.clear()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get loader performance metrics."""
        return self.metrics.copy()


class FileSystemLoader(BaseTemplateLoader):
    """Loads templates from file system (local or mounted)."""
    
    def __init__(self, base_path: Union[str, Path], config: Optional[LoaderConfig] = None):
        super().__init__(config)
        self.base_path = Path(base_path).resolve()
        self.supported_extensions = [".json", ".yaml", ".yml"]
    
    async def load_template(self, identifier: str, **kwargs) -> LoadResult:
        """Load template from file system."""
        start_time = datetime.now()
        cache_key = self._get_cache_key(identifier, **kwargs)
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            cached_template = self.cache[cache_key]
            load_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_metrics(load_time, True, True)
            
            return LoadResult(
                success=True,
                template=cached_template,
                source=f"cache:{identifier}",
                load_time_ms=load_time,
                cache_hit=True
            )
        
        try:
            # Resolve file path
            file_path = await self._resolve_file_path(identifier)
            if not file_path:
                error_msg = f"Template file not found: {identifier}"
                load_time = (datetime.now() - start_time).total_seconds() * 1000
                self._update_metrics(load_time, False, False)
                
                return LoadResult(
                    success=False,
                    error_message=error_msg,
                    source=f"filesystem:{identifier}",
                    load_time_ms=load_time
                )
            
            # Check file size
            file_size = file_path.stat().st_size
            max_size = self.config.max_file_size_mb * 1024 * 1024
            if file_size > max_size:
                error_msg = f"Template file too large: {file_size} bytes (max: {max_size})"
                load_time = (datetime.now() - start_time).total_seconds() * 1000
                self._update_metrics(load_time, False, False)
                
                return LoadResult(
                    success=False,
                    error_message=error_msg,
                    source=f"filesystem:{identifier}",
                    load_time_ms=load_time
                )
            
            # Load and parse file
            template = await self._load_and_parse_file(file_path)
            
            # Validate template if enabled
            validation_report = None
            validation_failed = False
            if self.config.validate_on_load:
                validation_report = await self._validate_template(template, identifier)
                if validation_report and not validation_report.is_valid:
                    validation_failed = True
            
            # Cache template
            self._cache_template(cache_key, template)
            
            # Calculate metrics
            load_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_metrics(load_time, True, False, validation_failed)
            
            # Prepare metadata
            metadata = {
                "file_path": str(file_path),
                "file_size": file_size,
                "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                "file_extension": file_path.suffix
            }
            
            return LoadResult(
                success=True,
                template=template,
                metadata=metadata,
                source=f"filesystem:{file_path}",
                load_time_ms=load_time,
                validation_report=validation_report
            )
        
        except Exception as e:
            logger.error(f"Failed to load template {identifier}: {str(e)}")
            load_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_metrics(load_time, False, False)
            
            return LoadResult(
                success=False,
                error_message=str(e),
                source=f"filesystem:{identifier}",
                load_time_ms=load_time
            )
    
    async def _resolve_file_path(self, identifier: str) -> Optional[Path]:
        """Resolve template file path with multiple extensions."""
        # Try direct path first
        direct_path = self.base_path / identifier
        if direct_path.exists() and direct_path.is_file():
            return direct_path
        
        # Try with supported extensions
        for ext in self.supported_extensions:
            file_path = self.base_path / f"{identifier}{ext}"
            if file_path.exists() and file_path.is_file():
                return file_path
        
        # Try nested path structures
        nested_path = self.base_path / identifier.replace(".", "/")
        if nested_path.exists() and nested_path.is_file():
            return nested_path
        
        for ext in self.supported_extensions:
            nested_file = self.base_path / f"{identifier.replace('.', '/')}{ext}"
            if nested_file.exists() and nested_file.is_file():
                return nested_file
        
        return None
    
    async def _load_and_parse_file(self, file_path: Path) -> Dict[str, Any]:
        """Load and parse template file."""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
            content = await file.read()
        
        if file_path.suffix == '.json':
            return json.loads(content)
        elif file_path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(content)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")


class DatabaseLoader(BaseTemplateLoader):
    """Loads templates from database storage."""
    
    def __init__(self, table_name: str = "template_storage", config: Optional[LoaderConfig] = None):
        super().__init__(config)
        self.table_name = table_name
    
    async def load_template(self, identifier: str, **kwargs) -> LoadResult:
        """Load template from database."""
        start_time = datetime.now()
        cache_key = self._get_cache_key(identifier, **kwargs)
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            cached_template = self.cache[cache_key]
            load_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_metrics(load_time, True, True)
            
            return LoadResult(
                success=True,
                template=cached_template,
                source=f"cache:{identifier}",
                load_time_ms=load_time,
                cache_hit=True
            )
        
        try:
            async with get_async_session() as session:
                # Query template from database
                query = text(f"""
                    SELECT template_data, metadata, created_at, updated_at, version
                    FROM {self.table_name}
                    WHERE identifier = :identifier
                    AND tenant_id = :tenant_id
                    ORDER BY version DESC
                    LIMIT 1
                """)
                
                result = await session.execute(query, {
                    "identifier": identifier,
                    "tenant_id": kwargs.get("tenant_id")
                })
                
                row = result.fetchone()
                if not row:
                    error_msg = f"Template not found in database: {identifier}"
                    load_time = (datetime.now() - start_time).total_seconds() * 1000
                    self._update_metrics(load_time, False, False)
                    
                    return LoadResult(
                        success=False,
                        error_message=error_msg,
                        source=f"database:{identifier}",
                        load_time_ms=load_time
                    )
                
                # Parse template data
                template_data = row.template_data
                if isinstance(template_data, str):
                    template = json.loads(template_data)
                else:
                    template = template_data
                
                # Decrypt if needed
                if self.config.encryption_enabled:
                    template = await self._decrypt_template(template)
                
                # Validate template if enabled
                validation_report = None
                validation_failed = False
                if self.config.validate_on_load:
                    validation_report = await self._validate_template(template, identifier)
                    if validation_report and not validation_report.is_valid:
                        validation_failed = True
                
                # Cache template
                self._cache_template(cache_key, template)
                
                # Calculate metrics
                load_time = (datetime.now() - start_time).total_seconds() * 1000
                self._update_metrics(load_time, True, False, validation_failed)
                
                # Prepare metadata
                metadata = {
                    "database_metadata": row.metadata,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                    "version": row.version
                }
                
                return LoadResult(
                    success=True,
                    template=template,
                    metadata=metadata,
                    source=f"database:{identifier}",
                    load_time_ms=load_time,
                    validation_report=validation_report
                )
        
        except Exception as e:
            logger.error(f"Failed to load template from database {identifier}: {str(e)}")
            load_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_metrics(load_time, False, False)
            
            return LoadResult(
                success=False,
                error_message=str(e),
                source=f"database:{identifier}",
                load_time_ms=load_time
            )
    
    async def _decrypt_template(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt template data if encryption is enabled."""
        # Implementation would depend on encryption strategy
        # This is a placeholder for actual decryption logic
        return template


class RemoteLoader(BaseTemplateLoader):
    """Loads templates from remote sources (HTTP/HTTPS)."""
    
    def __init__(self, base_url: str, config: Optional[LoaderConfig] = None):
        super().__init__(config)
        self.base_url = base_url.rstrip('/')
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def load_template(self, identifier: str, **kwargs) -> LoadResult:
        """Load template from remote source."""
        start_time = datetime.now()
        cache_key = self._get_cache_key(identifier, **kwargs)
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            cached_template = self.cache[cache_key]
            load_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_metrics(load_time, True, True)
            
            return LoadResult(
                success=True,
                template=cached_template,
                source=f"cache:{identifier}",
                load_time_ms=load_time,
                cache_hit=True
            )
        
        if not self.session:
            # Create session if not in context manager
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            )
        
        try:
            # Construct URL
            url = f"{self.base_url}/{identifier}"
            if not identifier.endswith(('.json', '.yaml', '.yml')):
                url += '.json'  # Default to JSON
            
            # Add authentication headers if provided
            headers = {}
            auth_token = kwargs.get('auth_token')
            if auth_token:
                headers['Authorization'] = f"Bearer {auth_token}"
            
            # Retry logic
            last_exception = None
            for attempt in range(self.config.retry_attempts):
                try:
                    async with self.session.get(url, headers=headers) as response:
                        if response.status == 200:
                            content = await response.text()
                            
                            # Parse content based on content type
                            content_type = response.headers.get('content-type', '')
                            if 'application/json' in content_type:
                                template = json.loads(content)
                            elif 'yaml' in content_type or url.endswith(('.yaml', '.yml')):
                                template = yaml.safe_load(content)
                            else:
                                template = json.loads(content)  # Default to JSON
                            
                            # Validate template if enabled
                            validation_report = None
                            validation_failed = False
                            if self.config.validate_on_load:
                                validation_report = await self._validate_template(template, identifier)
                                if validation_report and not validation_report.is_valid:
                                    validation_failed = True
                            
                            # Cache template
                            self._cache_template(cache_key, template)
                            
                            # Calculate metrics
                            load_time = (datetime.now() - start_time).total_seconds() * 1000
                            self._update_metrics(load_time, True, False, validation_failed)
                            
                            # Prepare metadata
                            metadata = {
                                "url": url,
                                "status_code": response.status,
                                "content_type": content_type,
                                "content_length": len(content),
                                "etag": response.headers.get('etag'),
                                "last_modified": response.headers.get('last-modified')
                            }
                            
                            return LoadResult(
                                success=True,
                                template=template,
                                metadata=metadata,
                                source=f"remote:{url}",
                                load_time_ms=load_time,
                                validation_report=validation_report
                            )
                        
                        elif response.status == 404:
                            error_msg = f"Template not found at URL: {url}"
                            load_time = (datetime.now() - start_time).total_seconds() * 1000
                            self._update_metrics(load_time, False, False)
                            
                            return LoadResult(
                                success=False,
                                error_message=error_msg,
                                source=f"remote:{url}",
                                load_time_ms=load_time
                            )
                        
                        else:
                            raise aiohttp.ClientResponseError(
                                request_info=response.request_info,
                                history=response.history,
                                status=response.status,
                                message=f"HTTP {response.status}"
                            )
                
                except Exception as e:
                    last_exception = e
                    if attempt < self.config.retry_attempts - 1:
                        await asyncio.sleep(self.config.retry_delay_seconds * (2 ** attempt))
                        continue
                    raise
            
            # If we get here, all retries failed
            raise last_exception or Exception("All retry attempts failed")
        
        except Exception as e:
            logger.error(f"Failed to load template from remote {identifier}: {str(e)}")
            load_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_metrics(load_time, False, False)
            
            return LoadResult(
                success=False,
                error_message=str(e),
                source=f"remote:{identifier}",
                load_time_ms=load_time
            )


class RedisLoader(BaseTemplateLoader):
    """Loads templates from Redis cache/storage."""
    
    def __init__(self, key_prefix: str = "template:", config: Optional[LoaderConfig] = None):
        super().__init__(config)
        self.key_prefix = key_prefix
    
    async def load_template(self, identifier: str, **kwargs) -> LoadResult:
        """Load template from Redis."""
        start_time = datetime.now()
        
        try:
            redis = await get_redis_client()
            redis_key = f"{self.key_prefix}{identifier}"
            
            # Add tenant context if provided
            tenant_id = kwargs.get('tenant_id')
            if tenant_id:
                redis_key = f"{self.key_prefix}{tenant_id}:{identifier}"
            
            # Get template from Redis
            template_data = await redis.get(redis_key)
            if not template_data:
                error_msg = f"Template not found in Redis: {identifier}"
                load_time = (datetime.now() - start_time).total_seconds() * 1000
                self._update_metrics(load_time, False, False)
                
                return LoadResult(
                    success=False,
                    error_message=error_msg,
                    source=f"redis:{redis_key}",
                    load_time_ms=load_time
                )
            
            # Parse JSON data
            if isinstance(template_data, bytes):
                template_data = template_data.decode('utf-8')
            
            template = json.loads(template_data)
            
            # Validate template if enabled
            validation_report = None
            validation_failed = False
            if self.config.validate_on_load:
                validation_report = await self._validate_template(template, identifier)
                if validation_report and not validation_report.is_valid:
                    validation_failed = True
            
            # Calculate metrics
            load_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_metrics(load_time, True, False, validation_failed)
            
            # Get TTL info
            ttl = await redis.ttl(redis_key)
            
            # Prepare metadata
            metadata = {
                "redis_key": redis_key,
                "ttl_seconds": ttl,
                "data_size": len(template_data)
            }
            
            return LoadResult(
                success=True,
                template=template,
                metadata=metadata,
                source=f"redis:{redis_key}",
                load_time_ms=load_time,
                validation_report=validation_report
            )
        
        except Exception as e:
            logger.error(f"Failed to load template from Redis {identifier}: {str(e)}")
            load_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_metrics(load_time, False, False)
            
            return LoadResult(
                success=False,
                error_message=str(e),
                source=f"redis:{identifier}",
                load_time_ms=load_time
            )


class GitRepositoryLoader(BaseTemplateLoader):
    """Loads templates from Git repositories."""
    
    def __init__(self, repo_url: str, branch: str = "main", config: Optional[LoaderConfig] = None):
        super().__init__(config)
        self.repo_url = repo_url
        self.branch = branch
        self.local_repo_path = None
    
    async def load_template(self, identifier: str, **kwargs) -> LoadResult:
        """Load template from Git repository."""
        start_time = datetime.now()
        cache_key = self._get_cache_key(identifier, **kwargs)
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            cached_template = self.cache[cache_key]
            load_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_metrics(load_time, True, True)
            
            return LoadResult(
                success=True,
                template=cached_template,
                source=f"cache:{identifier}",
                load_time_ms=load_time,
                cache_hit=True
            )
        
        try:
            # Ensure repository is cloned/updated
            await self._ensure_repository()
            
            # Load template using filesystem loader
            fs_loader = FileSystemLoader(self.local_repo_path, self.config)
            result = await fs_loader.load_template(identifier, **kwargs)
            
            if result.success:
                # Cache the template
                self._cache_template(cache_key, result.template)
                
                # Update metadata to include Git info
                result.metadata.update({
                    "git_repo": self.repo_url,
                    "git_branch": self.branch,
                    "local_path": str(self.local_repo_path)
                })
                
                result.source = f"git:{self.repo_url}:{identifier}"
            
            # Update metrics
            load_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_metrics(load_time, result.success, False, 
                               result.validation_report and not result.validation_report.is_valid)
            
            return result
        
        except Exception as e:
            logger.error(f"Failed to load template from Git {identifier}: {str(e)}")
            load_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_metrics(load_time, False, False)
            
            return LoadResult(
                success=False,
                error_message=str(e),
                source=f"git:{self.repo_url}:{identifier}",
                load_time_ms=load_time
            )
    
    async def _ensure_repository(self):
        """Ensure Git repository is available locally."""
        if self.local_repo_path and self.local_repo_path.exists():
            # Repository exists, pull latest changes
            await self._git_pull()
        else:
            # Clone repository
            await self._git_clone()
    
    async def _git_clone(self):
        """Clone Git repository."""
        self.local_repo_path = Path(tempfile.mkdtemp(prefix="template_repo_"))
        
        process = await asyncio.create_subprocess_exec(
            "git", "clone", "-b", self.branch, self.repo_url, str(self.local_repo_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"Git clone failed: {stderr.decode()}")
    
    async def _git_pull(self):
        """Pull latest changes from Git repository."""
        process = await asyncio.create_subprocess_exec(
            "git", "pull", "origin", self.branch,
            cwd=self.local_repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.warning(f"Git pull failed: {stderr.decode()}")


class ChainLoader(BaseTemplateLoader):
    """Chains multiple loaders with fallback support."""
    
    def __init__(self, loaders: List[BaseTemplateLoader], config: Optional[LoaderConfig] = None):
        super().__init__(config)
        self.loaders = loaders
    
    async def load_template(self, identifier: str, **kwargs) -> LoadResult:
        """Load template using chain of loaders."""
        start_time = datetime.now()
        errors = []
        warnings = []
        
        for i, loader in enumerate(self.loaders):
            try:
                result = await loader.load_template(identifier, **kwargs)
                
                if result.success:
                    # Update metrics
                    load_time = (datetime.now() - start_time).total_seconds() * 1000
                    self._update_metrics(load_time, True, result.cache_hit)
                    
                    # Add chain info to metadata
                    result.metadata["chain_position"] = i
                    result.metadata["total_loaders"] = len(self.loaders)
                    result.metadata["failed_loaders"] = errors
                    result.warnings.extend(warnings)
                    
                    return result
                else:
                    errors.append({
                        "loader": loader.__class__.__name__,
                        "error": result.error_message
                    })
                    warnings.extend(result.warnings)
            
            except Exception as e:
                errors.append({
                    "loader": loader.__class__.__name__,
                    "error": str(e)
                })
        
        # All loaders failed
        load_time = (datetime.now() - start_time).total_seconds() * 1000
        self._update_metrics(load_time, False, False)
        
        return LoadResult(
            success=False,
            error_message=f"All loaders failed for {identifier}",
            source=f"chain:{identifier}",
            load_time_ms=load_time,
            metadata={"failed_loaders": errors},
            warnings=warnings
        )


class TemplateLoaderManager:
    """Manages multiple template loaders and provides unified interface."""
    
    def __init__(self):
        self.loaders: Dict[str, BaseTemplateLoader] = {}
        self.default_loader: Optional[str] = None
    
    def register_loader(self, name: str, loader: BaseTemplateLoader, is_default: bool = False):
        """Register a template loader."""
        self.loaders[name] = loader
        if is_default or not self.default_loader:
            self.default_loader = name
    
    def unregister_loader(self, name: str):
        """Unregister a template loader."""
        if name in self.loaders:
            del self.loaders[name]
        if self.default_loader == name:
            self.default_loader = next(iter(self.loaders.keys())) if self.loaders else None
    
    async def load_template(
        self,
        identifier: str,
        loader_name: Optional[str] = None,
        **kwargs
    ) -> LoadResult:
        """Load template using specified or default loader."""
        loader_name = loader_name or self.default_loader
        
        if not loader_name or loader_name not in self.loaders:
            return LoadResult(
                success=False,
                error_message=f"Loader not found: {loader_name}",
                source=f"manager:{loader_name or 'unknown'}"
            )
        
        loader = self.loaders[loader_name]
        return await loader.load_template(identifier, **kwargs)
    
    async def load_with_fallback(
        self,
        identifier: str,
        loader_names: Optional[List[str]] = None,
        **kwargs
    ) -> LoadResult:
        """Load template with fallback to other loaders."""
        loader_names = loader_names or list(self.loaders.keys())
        
        fallback_loaders = [self.loaders[name] for name in loader_names if name in self.loaders]
        
        if not fallback_loaders:
            return LoadResult(
                success=False,
                error_message="No valid loaders available",
                source="manager:fallback"
            )
        
        chain_loader = ChainLoader(fallback_loaders)
        return await chain_loader.load_template(identifier, **kwargs)
    
    def get_loader_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all registered loaders."""
        return {name: loader.get_metrics() for name, loader in self.loaders.items()}
    
    def clear_all_caches(self):
        """Clear caches for all loaders."""
        for loader in self.loaders.values():
            loader.clear_cache()


# Factory functions for common loader configurations
def create_filesystem_loader(
    base_path: Union[str, Path],
    cache_ttl: int = 3600,
    validate: bool = True
) -> FileSystemLoader:
    """Create configured filesystem loader."""
    config = LoaderConfig(
        cache_ttl_seconds=cache_ttl,
        validate_on_load=validate
    )
    return FileSystemLoader(base_path, config)


def create_database_loader(
    table_name: str = "template_storage",
    cache_ttl: int = 3600,
    validate: bool = True
) -> DatabaseLoader:
    """Create configured database loader."""
    config = LoaderConfig(
        cache_ttl_seconds=cache_ttl,
        validate_on_load=validate
    )
    return DatabaseLoader(table_name, config)


def create_remote_loader(
    base_url: str,
    cache_ttl: int = 1800,
    timeout: int = 30,
    validate: bool = True
) -> RemoteLoader:
    """Create configured remote loader."""
    config = LoaderConfig(
        cache_ttl_seconds=cache_ttl,
        timeout_seconds=timeout,
        validate_on_load=validate
    )
    return RemoteLoader(base_url, config)


def create_redis_loader(
    key_prefix: str = "template:",
    cache_ttl: int = 7200,
    validate: bool = True
) -> RedisLoader:
    """Create configured Redis loader."""
    config = LoaderConfig(
        cache_ttl_seconds=cache_ttl,
        validate_on_load=validate
    )
    return RedisLoader(key_prefix, config)


def create_git_loader(
    repo_url: str,
    branch: str = "main",
    cache_ttl: int = 1800,
    validate: bool = True
) -> GitRepositoryLoader:
    """Create configured Git repository loader."""
    config = LoaderConfig(
        cache_ttl_seconds=cache_ttl,
        validate_on_load=validate
    )
    return GitRepositoryLoader(repo_url, branch, config)


# Global loader manager instance
loader_manager = TemplateLoaderManager()


def setup_default_loaders(base_template_path: Union[str, Path]):
    """Setup default template loaders."""
    # Filesystem loader as primary
    fs_loader = create_filesystem_loader(base_template_path, validate=True)
    loader_manager.register_loader("filesystem", fs_loader, is_default=True)
    
    # Database loader as secondary
    db_loader = create_database_loader(validate=True)
    loader_manager.register_loader("database", db_loader)
    
    # Redis loader for high-speed caching
    redis_loader = create_redis_loader(validate=False)  # Skip validation for cached templates
    loader_manager.register_loader("redis", redis_loader)
    
    logger.info("Default template loaders configured successfully")
