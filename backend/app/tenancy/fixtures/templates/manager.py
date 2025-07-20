#!/usr/bin/env python3
"""
Spotify AI Agent - Template Manager
==================================

Enterprise template management system providing:
- Template lifecycle management
- Dynamic template discovery and loading
- Template versioning and migration
- Real-time template synchronization
- Multi-tenant template isolation
- Advanced template analytics

Author: Expert Development Team
"""

import asyncio
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import aiofiles
import yaml
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.cache import get_redis_client
from app.core.database import get_async_session
from app.tenancy.fixtures.exceptions import TemplateError, TemplateNotFoundError
from app.tenancy.fixtures.utils import FixtureUtils, TenantUtils
from .engine import TemplateEngine, TemplateCache

logger = logging.getLogger(__name__)


class TemplateMetadata:
    """Template metadata management."""
    
    def __init__(self, template_path: str):
        self.template_path = Path(template_path)
        self.metadata = {}
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load template metadata from file or generate default."""
        metadata_file = self.template_path.with_suffix('.meta.json')
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load metadata for {self.template_path}: {e}")
                self.metadata = self._generate_default_metadata()
        else:
            self.metadata = self._generate_default_metadata()
    
    def _generate_default_metadata(self) -> Dict[str, Any]:
        """Generate default metadata for template."""
        stat = self.template_path.stat() if self.template_path.exists() else None
        
        return {
            "name": self.template_path.stem,
            "version": "1.0.0",
            "description": f"Template for {self.template_path.stem}",
            "category": self.template_path.parent.name,
            "format": self.template_path.suffix.lstrip('.'),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat() if stat else None,
            "size_bytes": stat.st_size if stat else 0,
            "schema_version": "2024.1",
            "tags": [],
            "required_context": [],
            "optional_context": [],
            "dependencies": [],
            "security_level": "standard",
            "cache_ttl": 3600
        }
    
    async def save_metadata(self) -> None:
        """Save metadata to file."""
        metadata_file = self.template_path.with_suffix('.meta.json')
        
        try:
            async with aiofiles.open(metadata_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(self.metadata, indent=2, ensure_ascii=False))
        except IOError as e:
            logger.error(f"Failed to save metadata for {self.template_path}: {e}")
            raise TemplateError(f"Failed to save template metadata: {e}")
    
    def update_metadata(self, updates: Dict[str, Any]) -> None:
        """Update metadata with new values."""
        self.metadata.update(updates)
        self.metadata["modified_at"] = datetime.now(timezone.utc).isoformat()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get template metadata."""
        return self.metadata.copy()


class TemplateRegistry:
    """Central registry for template management."""
    
    def __init__(self, base_directory: str):
        self.base_directory = Path(base_directory)
        self.templates = {}
        self.categories = {}
        self._initialize_registry()
    
    def _initialize_registry(self) -> None:
        """Initialize template registry."""
        if not self.base_directory.exists():
            self.base_directory.mkdir(parents=True, exist_ok=True)
        
        self._scan_templates()
    
    def _scan_templates(self) -> None:
        """Scan directory for templates and build registry."""
        self.templates.clear()
        self.categories.clear()
        
        for template_file in self.base_directory.rglob("*.jinja2"):
            self._register_template(template_file)
        
        for template_file in self.base_directory.rglob("*.json"):
            if not template_file.name.endswith(".meta.json"):
                self._register_template(template_file)
        
        for template_file in self.base_directory.rglob("*.yaml"):
            if not template_file.name.endswith(".meta.yaml"):
                self._register_template(template_file)
    
    def _register_template(self, template_path: Path) -> None:
        """Register a template in the registry."""
        try:
            relative_path = template_path.relative_to(self.base_directory)
            category = relative_path.parent.name if relative_path.parent != Path('.') else "root"
            
            metadata = TemplateMetadata(str(template_path))
            
            template_info = {
                "path": str(template_path),
                "relative_path": str(relative_path),
                "category": category,
                "metadata": metadata.get_metadata()
            }
            
            # Register template
            template_key = f"{category}/{template_path.stem}"
            self.templates[template_key] = template_info
            
            # Register category
            if category not in self.categories:
                self.categories[category] = []
            self.categories[category].append(template_key)
            
        except Exception as e:
            logger.error(f"Failed to register template {template_path}: {e}")
    
    def get_template_info(self, template_key: str) -> Optional[Dict[str, Any]]:
        """Get template information by key."""
        return self.templates.get(template_key)
    
    def get_templates_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all templates in a category."""
        template_keys = self.categories.get(category, [])
        return [self.templates[key] for key in template_keys]
    
    def get_all_categories(self) -> List[str]:
        """Get list of all template categories."""
        return list(self.categories.keys())
    
    def search_templates(
        self,
        query: str = None,
        category: str = None,
        tags: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Search templates by various criteria."""
        results = []
        
        for template_key, template_info in self.templates.items():
            metadata = template_info["metadata"]
            
            # Category filter
            if category and template_info["category"] != category:
                continue
            
            # Query filter
            if query:
                if query.lower() not in template_info["metadata"]["name"].lower():
                    if query.lower() not in template_info["metadata"].get("description", "").lower():
                        continue
            
            # Tags filter
            if tags:
                template_tags = metadata.get("tags", [])
                if not any(tag in template_tags for tag in tags):
                    continue
            
            results.append(template_info)
        
        return results
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        total_templates = len(self.templates)
        templates_by_category = {
            category: len(template_keys)
            for category, template_keys in self.categories.items()
        }
        
        total_size = sum(
            template_info["metadata"]["size_bytes"]
            for template_info in self.templates.values()
        )
        
        return {
            "total_templates": total_templates,
            "total_categories": len(self.categories),
            "templates_by_category": templates_by_category,
            "total_size_bytes": total_size,
            "total_size_formatted": FixtureUtils.format_size(total_size)
        }
    
    async def refresh_registry(self) -> Dict[str, Any]:
        """Refresh template registry by rescanning directory."""
        old_count = len(self.templates)
        
        self._scan_templates()
        
        new_count = len(self.templates)
        
        return {
            "templates_before": old_count,
            "templates_after": new_count,
            "templates_added": max(0, new_count - old_count),
            "templates_removed": max(0, old_count - new_count),
            "categories": len(self.categories)
        }


class TemplateManager:
    """
    Comprehensive template manager for enterprise operations.
    
    Provides:
    - Template CRUD operations
    - Template validation and security
    - Template versioning and backup
    - Template deployment and synchronization
    - Performance monitoring and analytics
    """
    
    def __init__(
        self,
        base_directory: str,
        session: Optional[AsyncSession] = None,
        redis_client=None
    ):
        self.base_directory = Path(base_directory)
        self.session = session
        self.redis_client = redis_client
        
        # Initialize components
        self.registry = TemplateRegistry(str(self.base_directory))
        self.engine = TemplateEngine(
            str(self.base_directory),
            redis_client=redis_client
        )
        
        # Create template directories
        self._initialize_directories()
        
        logger.info(f"Template manager initialized with base directory: {self.base_directory}")
    
    def _initialize_directories(self) -> None:
        """Initialize template directory structure."""
        categories = [
            "tenant", "user", "content", "ai_session", 
            "collaboration", "analytics", "notifications", "workflows"
        ]
        
        for category in categories:
            category_dir = self.base_directory / category
            category_dir.mkdir(parents=True, exist_ok=True)
    
    async def create_template(
        self,
        category: str,
        name: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        format_type: str = "json"
    ) -> Dict[str, Any]:
        """
        Create a new template.
        
        Args:
            category: Template category
            name: Template name
            content: Template content
            metadata: Optional metadata
            format_type: Template format (json, yaml, jinja2)
            
        Returns:
            Creation result with template information
        """
        try:
            # Validate category
            category_dir = self.base_directory / category
            category_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine file extension
            if format_type == "jinja2":
                template_file = category_dir / f"{name}.jinja2"
            else:
                template_file = category_dir / f"{name}.{format_type}.jinja2"
            
            # Check if template already exists
            if template_file.exists():
                raise TemplateError(f"Template {category}/{name} already exists")
            
            # Write template content
            async with aiofiles.open(template_file, 'w', encoding='utf-8') as f:
                await f.write(content)
            
            # Create metadata
            template_metadata = TemplateMetadata(str(template_file))
            if metadata:
                template_metadata.update_metadata(metadata)
            
            template_metadata.update_metadata({
                "category": category,
                "format": format_type,
                "created_by": "template_manager",
                "size_bytes": len(content.encode('utf-8'))
            })
            
            await template_metadata.save_metadata()
            
            # Refresh registry
            await self.registry.refresh_registry()
            
            # Validate template
            validation_result = await self.engine.validate_template(
                str(template_file.relative_to(self.base_directory))
            )
            
            result = {
                "status": "created",
                "category": category,
                "name": name,
                "path": str(template_file),
                "format": format_type,
                "size_bytes": len(content.encode('utf-8')),
                "validation": validation_result,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"Template created: {category}/{name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to create template {category}/{name}: {e}")
            raise TemplateError(f"Failed to create template: {e}")
    
    async def get_template(
        self,
        category: str,
        name: str,
        include_content: bool = False
    ) -> Dict[str, Any]:
        """Get template information and optionally content."""
        template_key = f"{category}/{name}"
        template_info = self.registry.get_template_info(template_key)
        
        if not template_info:
            raise TemplateNotFoundError(f"Template {template_key} not found")
        
        result = template_info.copy()
        
        if include_content:
            try:
                async with aiofiles.open(template_info["path"], 'r', encoding='utf-8') as f:
                    result["content"] = await f.read()
            except IOError as e:
                logger.error(f"Failed to read template content: {e}")
                result["content"] = None
        
        return result
    
    async def update_template(
        self,
        category: str,
        name: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update existing template."""
        template_key = f"{category}/{name}"
        template_info = self.registry.get_template_info(template_key)
        
        if not template_info:
            raise TemplateNotFoundError(f"Template {template_key} not found")
        
        template_path = Path(template_info["path"])
        
        # Backup existing template
        backup_result = await self._backup_template(template_path)
        
        try:
            # Update content if provided
            if content is not None:
                async with aiofiles.open(template_path, 'w', encoding='utf-8') as f:
                    await f.write(content)
            
            # Update metadata if provided
            if metadata:
                template_metadata = TemplateMetadata(str(template_path))
                template_metadata.update_metadata(metadata)
                await template_metadata.save_metadata()
            
            # Refresh registry
            await self.registry.refresh_registry()
            
            # Clear template cache
            relative_path = template_path.relative_to(self.base_directory)
            if self.engine.renderer.cache:
                await self.engine.renderer.cache.invalidate_template_cache(str(relative_path))
            
            result = {
                "status": "updated",
                "category": category,
                "name": name,
                "path": str(template_path),
                "backup_created": backup_result["backup_path"],
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"Template updated: {category}/{name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to update template {category}/{name}: {e}")
            raise TemplateError(f"Failed to update template: {e}")
    
    async def delete_template(
        self,
        category: str,
        name: str,
        create_backup: bool = True
    ) -> Dict[str, Any]:
        """Delete template with optional backup."""
        template_key = f"{category}/{name}"
        template_info = self.registry.get_template_info(template_key)
        
        if not template_info:
            raise TemplateNotFoundError(f"Template {template_key} not found")
        
        template_path = Path(template_info["path"])
        metadata_path = template_path.with_suffix('.meta.json')
        
        backup_path = None
        
        try:
            # Create backup if requested
            if create_backup:
                backup_result = await self._backup_template(template_path)
                backup_path = backup_result["backup_path"]
            
            # Delete template file
            if template_path.exists():
                template_path.unlink()
            
            # Delete metadata file
            if metadata_path.exists():
                metadata_path.unlink()
            
            # Refresh registry
            await self.registry.refresh_registry()
            
            # Clear template cache
            relative_path = template_path.relative_to(self.base_directory)
            if self.engine.renderer.cache:
                await self.engine.renderer.cache.invalidate_template_cache(str(relative_path))
            
            result = {
                "status": "deleted",
                "category": category,
                "name": name,
                "backup_created": backup_path is not None,
                "backup_path": backup_path,
                "deleted_at": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"Template deleted: {category}/{name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to delete template {category}/{name}: {e}")
            raise TemplateError(f"Failed to delete template: {e}")
    
    async def render_template(
        self,
        category: str,
        name: str,
        context: Dict[str, Any],
        tenant_id: Optional[str] = None
    ) -> str:
        """Render template with tenant-specific context."""
        # Enhance context with tenant data if provided
        if tenant_id:
            tenant_context = await self._get_tenant_context(tenant_id)
            context = {**tenant_context, **context}
        
        # Determine format from template info
        template_key = f"{category}/{name}"
        template_info = self.registry.get_template_info(template_key)
        
        if not template_info:
            raise TemplateNotFoundError(f"Template {template_key} not found")
        
        format_type = template_info["metadata"]["format"]
        
        return await self.engine.render_template(
            category=category,
            template_name=name,
            context=context,
            format_type=format_type
        )
    
    async def _get_tenant_context(self, tenant_id: str) -> Dict[str, Any]:
        """Get tenant-specific context for template rendering."""
        tenant_context = {
            "tenant_id": tenant_id,
            "tenant_schema": TenantUtils.get_tenant_schema_name(tenant_id)
        }
        
        # Get tenant data from database if session available
        if self.session:
            try:
                schema_name = TenantUtils.get_tenant_schema_name(tenant_id)
                
                # Get tenant configuration
                result = await self.session.execute(
                    text(f"SELECT * FROM {schema_name}.tenant_config LIMIT 1")
                )
                tenant_config = result.fetchone()
                
                if tenant_config:
                    tenant_context["tenant_config"] = dict(tenant_config._mapping)
                
            except Exception as e:
                logger.warning(f"Failed to load tenant context for {tenant_id}: {e}")
        
        return tenant_context
    
    async def _backup_template(self, template_path: Path) -> Dict[str, Any]:
        """Create backup of template."""
        backup_dir = self.base_directory / ".backups"
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{template_path.stem}_{timestamp}{template_path.suffix}"
        backup_path = backup_dir / backup_filename
        
        try:
            shutil.copy2(template_path, backup_path)
            
            # Also backup metadata if exists
            metadata_path = template_path.with_suffix('.meta.json')
            if metadata_path.exists():
                backup_metadata_path = backup_dir / f"{template_path.stem}_{timestamp}.meta.json"
                shutil.copy2(metadata_path, backup_metadata_path)
            
            return {
                "backup_path": str(backup_path),
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to create template backup: {e}")
            raise TemplateError(f"Failed to create backup: {e}")
    
    async def list_templates(
        self,
        category: Optional[str] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """List templates with optional filtering."""
        if category:
            templates = self.registry.get_templates_by_category(category)
        else:
            templates = list(self.registry.templates.values())
        
        if not include_metadata:
            # Remove metadata for lighter response
            templates = [
                {k: v for k, v in template.items() if k != "metadata"}
                for template in templates
            ]
        
        return templates
    
    async def search_templates(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search templates by various criteria."""
        return self.registry.search_templates(
            query=query,
            category=category,
            tags=tags
        )
    
    async def get_template_stats(self) -> Dict[str, Any]:
        """Get comprehensive template statistics."""
        registry_stats = self.registry.get_registry_stats()
        engine_stats = await self.engine.get_engine_stats()
        
        return {
            "registry": registry_stats,
            "engine": engine_stats,
            "base_directory": str(self.base_directory),
            "last_scan": datetime.now(timezone.utc).isoformat()
        }
    
    async def export_templates(
        self,
        export_path: str,
        categories: Optional[List[str]] = None,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Export templates to archive."""
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        exported_count = 0
        exported_categories = set()
        
        templates_to_export = []
        if categories:
            for category in categories:
                templates_to_export.extend(
                    self.registry.get_templates_by_category(category)
                )
        else:
            templates_to_export = list(self.registry.templates.values())
        
        for template_info in templates_to_export:
            try:
                source_path = Path(template_info["path"])
                category = template_info["category"]
                
                # Create category directory
                category_dir = export_path / category
                category_dir.mkdir(exist_ok=True)
                
                # Copy template
                dest_path = category_dir / source_path.name
                shutil.copy2(source_path, dest_path)
                
                # Copy metadata if requested
                if include_metadata:
                    metadata_path = source_path.with_suffix('.meta.json')
                    if metadata_path.exists():
                        dest_metadata_path = category_dir / metadata_path.name
                        shutil.copy2(metadata_path, dest_metadata_path)
                
                exported_count += 1
                exported_categories.add(category)
                
            except Exception as e:
                logger.error(f"Failed to export template {template_info['path']}: {e}")
        
        return {
            "exported_templates": exported_count,
            "exported_categories": list(exported_categories),
            "export_path": str(export_path),
            "include_metadata": include_metadata,
            "exported_at": datetime.now(timezone.utc).isoformat()
        }
    
    async def import_templates(
        self,
        import_path: str,
        overwrite_existing: bool = False,
        validate_templates: bool = True
    ) -> Dict[str, Any]:
        """Import templates from directory."""
        import_path = Path(import_path)
        
        if not import_path.exists():
            raise TemplateError(f"Import path does not exist: {import_path}")
        
        imported_count = 0
        skipped_count = 0
        errors = []
        
        for template_file in import_path.rglob("*.jinja2"):
            try:
                relative_path = template_file.relative_to(import_path)
                category = relative_path.parent.name if relative_path.parent != Path('.') else "root"
                
                dest_dir = self.base_directory / category
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                dest_path = dest_dir / template_file.name
                
                # Check if exists and handle overwrite
                if dest_path.exists() and not overwrite_existing:
                    skipped_count += 1
                    continue
                
                # Copy template
                shutil.copy2(template_file, dest_path)
                
                # Copy metadata if exists
                metadata_file = template_file.with_suffix('.meta.json')
                if metadata_file.exists():
                    dest_metadata = dest_path.with_suffix('.meta.json')
                    shutil.copy2(metadata_file, dest_metadata)
                
                # Validate if requested
                if validate_templates:
                    validation_result = await self.engine.validate_template(
                        str(dest_path.relative_to(self.base_directory))
                    )
                    if not validation_result["valid"]:
                        errors.append(f"Validation failed for {relative_path}")
                
                imported_count += 1
                
            except Exception as e:
                errors.append(f"Failed to import {template_file}: {e}")
                logger.error(f"Template import error: {e}")
        
        # Refresh registry
        await self.registry.refresh_registry()
        
        return {
            "imported_templates": imported_count,
            "skipped_templates": skipped_count,
            "errors": errors,
            "import_path": str(import_path),
            "overwrite_existing": overwrite_existing,
            "validate_templates": validate_templates,
            "imported_at": datetime.now(timezone.utc).isoformat()
        }
