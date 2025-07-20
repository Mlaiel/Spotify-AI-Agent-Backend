#!/usr/bin/env python3
"""
Spotify AI Agent - Template Engine
=================================

Advanced template engine with enterprise features:
- High-performance Jinja2 rendering
- Dynamic template compilation
- Advanced caching and optimization
- Security-first template processing
- Multi-format template support
- Real-time template reloading

Author: Expert Development Team
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable

import jinja2
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape
from jinja2.exceptions import TemplateError, TemplateSyntaxError

from app.core.cache import get_redis_client
from app.tenancy.fixtures.exceptions import TemplateRenderError, TemplateValidationError
from app.tenancy.fixtures.utils import FixtureUtils

logger = logging.getLogger(__name__)


class AdvancedTemplateEnvironment:
    """Advanced Jinja2 environment with custom filters and functions."""
    
    def __init__(self, template_dirs: List[str]):
        self.template_dirs = template_dirs
        self.env = self._create_environment()
        self._register_custom_filters()
        self._register_custom_functions()
    
    def _create_environment(self) -> Environment:
        """Create Jinja2 environment with security settings."""
        loader = FileSystemLoader(
            self.template_dirs,
            followlinks=False,  # Security: don't follow symlinks
            encoding='utf-8'
        )
        
        env = Environment(
            loader=loader,
            autoescape=select_autoescape(['html', 'xml', 'json']),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
            optimized=True,
            cache_size=1000,
            auto_reload=False  # Disable for production
        )
        
        # Security: Remove dangerous builtins
        env.globals.pop('range', None)
        env.globals.pop('dict', None)
        env.globals.pop('list', None)
        
        return env
    
    def _register_custom_filters(self) -> None:
        """Register custom Jinja2 filters for template processing."""
        
        @jinja2.pass_context
        def tenant_filter(context, value, tenant_id: str) -> str:
            """Filter value based on tenant context."""
            tenant_data = context.get('tenant_data', {})
            if tenant_id in tenant_data:
                return tenant_data[tenant_id].get(value, value)
            return value
        
        @jinja2.pass_context  
        def format_datetime(context, value, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
            """Format datetime with timezone awareness."""
            if isinstance(value, str):
                try:
                    value = datetime.fromisoformat(value.replace('Z', '+00:00'))
                except ValueError:
                    return value
            
            if isinstance(value, datetime):
                return value.strftime(format_str)
            return str(value)
        
        def json_pretty(value, indent: int = 2) -> str:
            """Pretty print JSON with proper indentation."""
            try:
                if isinstance(value, str):
                    value = json.loads(value)
                return json.dumps(value, indent=indent, ensure_ascii=False)
            except (json.JSONDecodeError, TypeError):
                return str(value)
        
        def yaml_format(value, default_style: str = None) -> str:
            """Format value as YAML."""
            try:
                return yaml.dump(
                    value,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False
                )
            except yaml.YAMLError:
                return str(value)
        
        def encrypt_sensitive(value, key: str = "default") -> str:
            """Encrypt sensitive values in templates."""
            # Simple encoding for demo - would use proper encryption
            encoded = hashlib.md5(f"{key}:{value}".encode()).hexdigest()
            return f"***{encoded[:8]}***"
        
        def sanitize_input(value) -> str:
            """Sanitize input to prevent injection."""
            if not isinstance(value, str):
                value = str(value)
            # Remove potentially dangerous characters
            dangerous_chars = ['<', '>', '&', '"', "'", '`', '$', '{', '}']
            for char in dangerous_chars:
                value = value.replace(char, '')
            return value.strip()
        
        # Register filters
        self.env.filters['tenant'] = tenant_filter
        self.env.filters['format_datetime'] = format_datetime
        self.env.filters['json_pretty'] = json_pretty
        self.env.filters['yaml_format'] = yaml_format
        self.env.filters['encrypt_sensitive'] = encrypt_sensitive
        self.env.filters['sanitize'] = sanitize_input
    
    def _register_custom_functions(self) -> None:
        """Register custom functions available in templates."""
        
        def generate_uuid() -> str:
            """Generate UUID for template use."""
            import uuid
            return str(uuid.uuid4())
        
        def current_timestamp() -> str:
            """Get current timestamp."""
            return datetime.now(timezone.utc).isoformat()
        
        def tenant_config(tenant_id: str, key: str, default: Any = None) -> Any:
            """Get tenant-specific configuration."""
            # Would fetch from database in real implementation
            return f"config_{tenant_id}_{key}" if default is None else default
        
        def feature_enabled(feature: str, tenant_id: str = None) -> bool:
            """Check if feature is enabled for tenant."""
            # Would check feature flags in real implementation
            return True  # Default enabled for demo
        
        def calculate_tier_limits(tier: str) -> Dict[str, Any]:
            """Calculate limits based on tier."""
            tier_limits = {
                "starter": {"users": 10, "storage_gb": 5, "ai_sessions": 100},
                "professional": {"users": 100, "storage_gb": 50, "ai_sessions": 1000},
                "enterprise": {"users": -1, "storage_gb": 500, "ai_sessions": -1}
            }
            return tier_limits.get(tier, tier_limits["starter"])
        
        # Register functions
        self.env.globals['generate_uuid'] = generate_uuid
        self.env.globals['current_timestamp'] = current_timestamp
        self.env.globals['tenant_config'] = tenant_config
        self.env.globals['feature_enabled'] = feature_enabled
        self.env.globals['calculate_tier_limits'] = calculate_tier_limits
    
    def get_template(self, template_name: str) -> jinja2.Template:
        """Get compiled template by name."""
        return self.env.get_template(template_name)
    
    def from_string(self, source: str) -> jinja2.Template:
        """Create template from string."""
        return self.env.from_string(source)


class TemplateCache:
    """High-performance template caching system."""
    
    def __init__(self, redis_client=None, max_memory_cache: int = 1000):
        self.redis_client = redis_client
        self.memory_cache = {}
        self.max_memory_cache = max_memory_cache
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }
    
    def _generate_cache_key(self, template_path: str, context_hash: str) -> str:
        """Generate cache key for template and context."""
        return f"template_cache:{hashlib.md5(f'{template_path}:{context_hash}'.encode()).hexdigest()}"
    
    def _hash_context(self, context: Dict[str, Any]) -> str:
        """Generate hash for template context."""
        # Sort context for consistent hashing
        context_str = json.dumps(context, sort_keys=True, default=str)
        return hashlib.md5(context_str.encode()).hexdigest()
    
    async def get_cached_template(
        self,
        template_path: str,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """Get cached rendered template."""
        context_hash = self._hash_context(context)
        cache_key = self._generate_cache_key(template_path, context_hash)
        
        # Try memory cache first
        if cache_key in self.memory_cache:
            self.cache_stats["hits"] += 1
            return self.memory_cache[cache_key]["content"]
        
        # Try Redis cache
        if self.redis_client:
            try:
                cached = await self.redis_client.get(cache_key)
                if cached:
                    self.cache_stats["hits"] += 1
                    # Store in memory cache for faster access
                    self._store_in_memory_cache(cache_key, cached.decode('utf-8'))
                    return cached.decode('utf-8')
            except Exception as e:
                logger.warning(f"Redis cache get failed: {e}")
        
        self.cache_stats["misses"] += 1
        return None
    
    async def cache_template(
        self,
        template_path: str,
        context: Dict[str, Any],
        rendered_content: str,
        ttl: int = 3600
    ) -> None:
        """Cache rendered template."""
        context_hash = self._hash_context(context)
        cache_key = self._generate_cache_key(template_path, context_hash)
        
        # Store in memory cache
        self._store_in_memory_cache(cache_key, rendered_content)
        
        # Store in Redis cache
        if self.redis_client:
            try:
                await self.redis_client.setex(cache_key, ttl, rendered_content)
            except Exception as e:
                logger.warning(f"Redis cache set failed: {e}")
    
    def _store_in_memory_cache(self, key: str, content: str) -> None:
        """Store content in memory cache with LRU eviction."""
        if len(self.memory_cache) >= self.max_memory_cache:
            # Simple LRU eviction - remove oldest entry
            oldest_key = min(
                self.memory_cache.keys(),
                key=lambda k: self.memory_cache[k]["timestamp"]
            )
            del self.memory_cache[oldest_key]
            self.cache_stats["evictions"] += 1
        
        self.memory_cache[key] = {
            "content": content,
            "timestamp": time.time()
        }
    
    async def invalidate_template_cache(self, template_path: str) -> None:
        """Invalidate all cached versions of a template."""
        # Remove from memory cache
        keys_to_remove = [
            key for key in self.memory_cache.keys()
            if template_path in key
        ]
        for key in keys_to_remove:
            del self.memory_cache[key]
        
        # Remove from Redis cache
        if self.redis_client:
            try:
                pattern = f"template_cache:*{template_path}*"
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
            except Exception as e:
                logger.warning(f"Redis cache invalidation failed: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "hit_rate_percent": round(hit_rate, 2),
            "total_hits": self.cache_stats["hits"],
            "total_misses": self.cache_stats["misses"],
            "total_evictions": self.cache_stats["evictions"],
            "memory_cache_size": len(self.memory_cache),
            "max_memory_cache": self.max_memory_cache
        }


class TemplateRenderer:
    """High-performance template renderer with advanced features."""
    
    def __init__(self, template_dirs: List[str], cache_enabled: bool = True):
        self.template_dirs = template_dirs
        self.cache_enabled = cache_enabled
        self.environment = AdvancedTemplateEnvironment(template_dirs)
        self.cache = TemplateCache() if cache_enabled else None
        self.render_stats = {
            "total_renders": 0,
            "cache_hits": 0,
            "errors": 0,
            "avg_render_time": 0.0
        }
    
    async def render_template(
        self,
        template_path: str,
        context: Dict[str, Any],
        cache_ttl: int = 3600
    ) -> str:
        """
        Render template with context and caching.
        
        Args:
            template_path: Path to template file
            context: Template context variables
            cache_ttl: Cache time-to-live in seconds
            
        Returns:
            Rendered template content
        """
        start_time = time.time()
        self.render_stats["total_renders"] += 1
        
        try:
            # Try cache first
            if self.cache_enabled and self.cache:
                cached_content = await self.cache.get_cached_template(template_path, context)
                if cached_content:
                    self.render_stats["cache_hits"] += 1
                    return cached_content
            
            # Load and render template
            template = self.environment.get_template(template_path)
            rendered_content = template.render(**context)
            
            # Cache the result
            if self.cache_enabled and self.cache:
                await self.cache.cache_template(
                    template_path, context, rendered_content, cache_ttl
                )
            
            # Update stats
            render_time = time.time() - start_time
            self._update_render_stats(render_time)
            
            return rendered_content
            
        except (TemplateError, TemplateSyntaxError) as e:
            self.render_stats["errors"] += 1
            logger.error(f"Template rendering failed for {template_path}: {e}")
            raise TemplateRenderError(f"Failed to render template {template_path}: {e}")
        
        except Exception as e:
            self.render_stats["errors"] += 1
            logger.error(f"Unexpected error rendering template {template_path}: {e}")
            raise TemplateRenderError(f"Unexpected error: {e}")
    
    async def render_string(
        self,
        template_string: str,
        context: Dict[str, Any]
    ) -> str:
        """Render template from string."""
        start_time = time.time()
        
        try:
            template = self.environment.from_string(template_string)
            rendered_content = template.render(**context)
            
            render_time = time.time() - start_time
            self._update_render_stats(render_time)
            
            return rendered_content
            
        except (TemplateError, TemplateSyntaxError) as e:
            self.render_stats["errors"] += 1
            raise TemplateRenderError(f"Failed to render template string: {e}")
    
    def _update_render_stats(self, render_time: float) -> None:
        """Update rendering statistics."""
        total_renders = self.render_stats["total_renders"]
        current_avg = self.render_stats["avg_render_time"]
        
        # Calculate new running average
        new_avg = ((current_avg * (total_renders - 1)) + render_time) / total_renders
        self.render_stats["avg_render_time"] = new_avg
    
    def get_render_stats(self) -> Dict[str, Any]:
        """Get rendering performance statistics."""
        cache_stats = self.cache.get_cache_stats() if self.cache else {}
        
        return {
            "rendering": self.render_stats,
            "caching": cache_stats,
            "cache_enabled": self.cache_enabled
        }
    
    async def precompile_templates(self, template_patterns: List[str]) -> Dict[str, Any]:
        """Precompile templates for faster rendering."""
        compilation_results = {
            "compiled": 0,
            "errors": 0,
            "templates": []
        }
        
        for pattern in template_patterns:
            try:
                # Find matching templates
                template_paths = []
                for template_dir in self.template_dirs:
                    template_dir_path = Path(template_dir)
                    if template_dir_path.exists():
                        template_paths.extend(template_dir_path.glob(pattern))
                
                # Compile templates
                for template_path in template_paths:
                    try:
                        relative_path = template_path.relative_to(template_dir_path)
                        template = self.environment.get_template(str(relative_path))
                        compilation_results["compiled"] += 1
                        compilation_results["templates"].append(str(relative_path))
                        
                    except Exception as e:
                        compilation_results["errors"] += 1
                        logger.error(f"Failed to compile template {template_path}: {e}")
                        
            except Exception as e:
                logger.error(f"Error processing template pattern {pattern}: {e}")
                compilation_results["errors"] += 1
        
        return compilation_results


class TemplateEngine:
    """
    Main template engine orchestrating all template operations.
    
    Provides:
    - Template rendering with caching
    - Dynamic template loading
    - Template validation and security
    - Performance monitoring
    - Multi-format support
    """
    
    def __init__(
        self,
        template_base_dir: str,
        redis_client=None,
        cache_enabled: bool = True
    ):
        self.template_base_dir = Path(template_base_dir)
        self.redis_client = redis_client
        self.cache_enabled = cache_enabled
        
        # Initialize template directories
        self.template_dirs = self._discover_template_directories()
        
        # Initialize renderer
        self.renderer = TemplateRenderer(
            template_dirs=self.template_dirs,
            cache_enabled=cache_enabled
        )
        
        logger.info(f"Template engine initialized with {len(self.template_dirs)} directories")
    
    def _discover_template_directories(self) -> List[str]:
        """Discover all template directories."""
        template_dirs = []
        
        if self.template_base_dir.exists():
            # Add main template directory
            template_dirs.append(str(self.template_base_dir))
            
            # Add category subdirectories
            for category_dir in self.template_base_dir.iterdir():
                if category_dir.is_dir():
                    template_dirs.append(str(category_dir))
        
        return template_dirs
    
    async def render_template(
        self,
        category: str,
        template_name: str,
        context: Dict[str, Any],
        format_type: str = "json"
    ) -> str:
        """
        Render template by category and name.
        
        Args:
            category: Template category (tenant, user, content, etc.)
            template_name: Template name without extension
            context: Template context variables
            format_type: Output format (json, yaml, etc.)
            
        Returns:
            Rendered template content
        """
        # Construct template path
        if format_type in ["jinja2", "liquid"]:
            template_path = f"{category}/{template_name}.{format_type}"
        else:
            template_path = f"{category}/{template_name}.{format_type}.jinja2"
        
        # Add engine context
        enhanced_context = self._enhance_context(context, category)
        
        # Render template
        return await self.renderer.render_template(
            template_path=template_path,
            context=enhanced_context
        )
    
    async def render_template_string(
        self,
        template_string: str,
        context: Dict[str, Any],
        category: str = "general"
    ) -> str:
        """Render template from string content."""
        enhanced_context = self._enhance_context(context, category)
        return await self.renderer.render_string(template_string, enhanced_context)
    
    def _enhance_context(self, context: Dict[str, Any], category: str) -> Dict[str, Any]:
        """Enhance template context with engine-provided variables."""
        enhanced_context = context.copy()
        
        # Add engine metadata
        enhanced_context.update({
            "_template_engine": {
                "version": "1.0.0",
                "category": category,
                "render_time": datetime.now(timezone.utc).isoformat(),
                "cache_enabled": self.cache_enabled
            },
            "_utils": {
                "format_size": FixtureUtils.format_size,
                "format_duration": FixtureUtils.format_duration,
                "generate_id": FixtureUtils.generate_id
            }
        })
        
        return enhanced_context
    
    async def validate_template(self, template_path: str) -> Dict[str, Any]:
        """Validate template syntax and security."""
        validation_result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "security_issues": []
        }
        
        try:
            # Load template to check syntax
            template = self.renderer.environment.get_template(template_path)
            validation_result["valid"] = True
            
            # Basic security checks
            template_source = template.source
            
            # Check for dangerous patterns
            dangerous_patterns = [
                "import os",
                "import sys", 
                "__import__",
                "exec(",
                "eval(",
                "open(",
                "file("
            ]
            
            for pattern in dangerous_patterns:
                if pattern in template_source:
                    validation_result["security_issues"].append(
                        f"Potentially dangerous pattern found: {pattern}"
                    )
            
            # Check for best practices
            if "{% autoescape false %}" in template_source:
                validation_result["warnings"].append(
                    "Autoescape disabled - ensure manual escaping is used"
                )
            
        except (TemplateError, TemplateSyntaxError) as e:
            validation_result["errors"].append(str(e))
        
        except Exception as e:
            validation_result["errors"].append(f"Unexpected validation error: {e}")
        
        return validation_result
    
    async def get_available_templates(self) -> Dict[str, List[str]]:
        """Get list of all available templates by category."""
        templates_by_category = {}
        
        for template_dir in self.template_dirs:
            template_dir_path = Path(template_dir)
            category_name = template_dir_path.name
            
            if category_name == self.template_base_dir.name:
                category_name = "root"
            
            templates = []
            if template_dir_path.exists():
                for template_file in template_dir_path.glob("*.jinja2"):
                    templates.append(template_file.stem)
                for template_file in template_dir_path.glob("*.json"):
                    if not template_file.name.endswith(".jinja2"):
                        templates.append(template_file.stem)
            
            if templates:
                templates_by_category[category_name] = sorted(templates)
        
        return templates_by_category
    
    async def get_engine_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics."""
        render_stats = self.renderer.get_render_stats()
        available_templates = await self.get_available_templates()
        
        total_templates = sum(len(templates) for templates in available_templates.values())
        
        return {
            "template_directories": len(self.template_dirs),
            "total_templates": total_templates,
            "templates_by_category": {
                category: len(templates) 
                for category, templates in available_templates.items()
            },
            "performance": render_stats,
            "cache_enabled": self.cache_enabled
        }
    
    async def reload_templates(self) -> Dict[str, Any]:
        """Reload all templates and clear cache."""
        reload_result = {
            "templates_reloaded": 0,
            "cache_cleared": False,
            "errors": []
        }
        
        try:
            # Clear template cache
            if self.renderer.cache:
                # Clear all template caches
                for template_dir in self.template_dirs:
                    template_dir_path = Path(template_dir)
                    if template_dir_path.exists():
                        for template_file in template_dir_path.glob("**/*.jinja2"):
                            relative_path = template_file.relative_to(template_dir_path)
                            await self.renderer.cache.invalidate_template_cache(str(relative_path))
                
                reload_result["cache_cleared"] = True
            
            # Rediscover template directories
            self.template_dirs = self._discover_template_directories()
            
            # Recreate renderer
            self.renderer = TemplateRenderer(
                template_dirs=self.template_dirs,
                cache_enabled=self.cache_enabled
            )
            
            # Count reloaded templates
            available_templates = await self.get_available_templates()
            reload_result["templates_reloaded"] = sum(
                len(templates) for templates in available_templates.values()
            )
            
        except Exception as e:
            reload_result["errors"].append(str(e))
            logger.error(f"Template reload failed: {e}")
        
        return reload_result
