"""
Advanced Template Engine for Dynamic Slack Alert Generation.

This module provides a sophisticated template engine for generating dynamic
Slack alert messages with support for complex templating, conditional logic,
custom filters, and performance optimization.

Features:
- Jinja2-based template engine with custom extensions
- Dynamic template compilation and caching
- Conditional rendering and complex logic
- Custom filters for Slack-specific formatting
- Template inheritance and composition
- Variable context management
- Performance monitoring and optimization
- Template validation and error handling

Author: Fahed Mlaiel
Version: 1.0.0
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
from pathlib import Path
from collections import ChainMap
import hashlib

from jinja2 import (
    Environment,
    FileSystemLoader,
    DictLoader,
    BaseLoader,
    Template,
    TemplateError,
    UndefinedError,
    select_autoescape
)
from jinja2.ext import Extension
from markupsafe import Markup

from .constants import (
    TEMPLATE_CACHE_TTL,
    MAX_TEMPLATE_SIZE,
    ALERT_PRIORITIES,
    SLACK_EMOJI_MAP
)
from .exceptions import (
    TemplateError as CustomTemplateError,
    TemplateNotFoundError,
    TemplateCompilationError,
    TemplateRenderError
)
from .performance_monitor import PerformanceMonitor


@dataclass
class TemplateMetadata:
    """Metadata for a template."""
    name: str
    path: str
    version: str
    author: str
    description: str
    variables: List[str]
    dependencies: List[str]
    created_at: datetime
    updated_at: datetime
    compiled_at: Optional[datetime] = None
    cache_key: Optional[str] = None


@dataclass
class RenderContext:
    """Context for template rendering."""
    template_name: str
    variables: Dict[str, Any]
    locale: str
    tenant_id: str
    environment: str
    alert_priority: str
    timestamp: datetime
    user_context: Optional[Dict[str, Any]] = None


class SlackExtension(Extension):
    """Custom Jinja2 extension for Slack-specific functionality."""
    
    def __init__(self, environment):
        super().__init__(environment)
        environment.filters.update({
            'slack_mention': self.slack_mention,
            'slack_channel': self.slack_channel,
            'slack_emoji': self.slack_emoji,
            'slack_code': self.slack_code,
            'slack_quote': self.slack_quote,
            'slack_link': self.slack_link,
            'priority_color': self.priority_color,
            'format_duration': self.format_duration,
            'format_metric': self.format_metric,
            'truncate_text': self.truncate_text
        })

    def slack_mention(self, user_id: str, display_name: Optional[str] = None) -> str:
        """Create Slack user mention."""
        if display_name:
            return f"<@{user_id}|{display_name}>"
        return f"<@{user_id}>"

    def slack_channel(self, channel_id: str, display_name: Optional[str] = None) -> str:
        """Create Slack channel mention."""
        if display_name:
            return f"<#{channel_id}|{display_name}>"
        return f"<#{channel_id}>"

    def slack_emoji(self, emoji_name: str) -> str:
        """Create Slack emoji."""
        # Map common emoji names to Slack format
        mapped_emoji = SLACK_EMOJI_MAP.get(emoji_name, emoji_name)
        return f":{mapped_emoji}:"

    def slack_code(self, text: str, language: Optional[str] = None) -> str:
        """Create Slack code block."""
        if language:
            return f"```{language}\n{text}\n```"
        return f"`{text}`"

    def slack_quote(self, text: str) -> str:
        """Create Slack quote block."""
        lines = text.split('\n')
        quoted_lines = [f"> {line}" for line in lines]
        return '\n'.join(quoted_lines)

    def slack_link(self, url: str, text: Optional[str] = None) -> str:
        """Create Slack link."""
        if text:
            return f"<{url}|{text}>"
        return f"<{url}>"

    def priority_color(self, priority: str) -> str:
        """Get color for alert priority."""
        color_map = {
            'critical': 'danger',
            'high': 'warning',
            'medium': 'good',
            'low': '#36a64f',
            'info': '#2196F3'
        }
        return color_map.get(priority.lower(), 'good')

    def format_duration(self, seconds: Union[int, float]) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            remaining_seconds = int(seconds % 60)
            return f"{minutes}m {remaining_seconds}s"
        else:
            hours = int(seconds / 3600)
            remaining_minutes = int((seconds % 3600) / 60)
            return f"{hours}h {remaining_minutes}m"

    def format_metric(self, value: Union[int, float], unit: str = "") -> str:
        """Format metric value with proper units."""
        if isinstance(value, float):
            if value >= 1000000:
                return f"{value/1000000:.1f}M {unit}".strip()
            elif value >= 1000:
                return f"{value/1000:.1f}K {unit}".strip()
            else:
                return f"{value:.1f} {unit}".strip()
        else:
            if value >= 1000000:
                return f"{value//1000000}M {unit}".strip()
            elif value >= 1000:
                return f"{value//1000}K {unit}".strip()
            else:
                return f"{value} {unit}".strip()

    def truncate_text(self, text: str, max_length: int = 100, suffix: str = "...") -> str:
        """Truncate text to maximum length."""
        if len(text) <= max_length:
            return text
        return text[:max_length - len(suffix)] + suffix


class TemplateEngine:
    """
    Advanced template engine for dynamic Slack alert generation.
    
    Provides comprehensive template management with caching, validation,
    and performance optimization for Slack alert message generation.
    """

    def __init__(
        self,
        templates_path: Optional[str] = None,
        cache_ttl: int = TEMPLATE_CACHE_TTL,
        enable_monitoring: bool = True,
        enable_autoescape: bool = True,
        max_template_size: int = MAX_TEMPLATE_SIZE
    ):
        """
        Initialize the template engine.
        
        Args:
            templates_path: Path to template files directory
            cache_ttl: Template cache time-to-live in seconds
            enable_monitoring: Enable performance monitoring
            enable_autoescape: Enable auto-escaping for security
            max_template_size: Maximum template size in bytes
        """
        self.templates_path = Path(templates_path) if templates_path else Path(__file__).parent / "templates"
        self.cache_ttl = cache_ttl
        self.enable_monitoring = enable_monitoring
        self.max_template_size = max_template_size
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor() if enable_monitoring else None
        
        # Template cache
        self.template_cache: Dict[str, Template] = {}
        self.metadata_cache: Dict[str, TemplateMetadata] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Template string cache for dynamic templates
        self.string_templates: Dict[str, str] = {}
        
        # Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_path)),
            autoescape=select_autoescape(['html', 'xml']) if enable_autoescape else False,
            extensions=[SlackExtension],
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=False
        )
        
        # Add global functions
        self.env.globals.update({
            'now': datetime.now,
            'utcnow': datetime.utcnow,
            'range': range,
            'len': len,
            'min': min,
            'max': max,
            'sum': sum,
            'abs': abs,
            'round': round
        })
        
        # Template validation patterns
        self.validation_patterns = {
            'slack_mention': re.compile(r'<@[A-Z0-9]+(\|[^>]+)?>'),
            'slack_channel': re.compile(r'<#[A-Z0-9]+(\|[^>]+)?>'),
            'slack_emoji': re.compile(r':[a-zA-Z0-9_+-]+:'),
            'slack_code': re.compile(r'```[\s\S]*?```|`[^`]+`'),
            'slack_quote': re.compile(r'^> .+', re.MULTILINE)
        }
        
        # Logging
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """Initialize the template engine."""
        try:
            # Create templates directory if it doesn't exist
            self.templates_path.mkdir(parents=True, exist_ok=True)
            
            # Load template metadata
            await self._load_template_metadata()
            
            # Precompile frequently used templates
            await self._precompile_templates()
            
            # Initialize performance monitoring
            if self.performance_monitor:
                await self.performance_monitor.initialize()
            
            self.logger.info("TemplateEngine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize TemplateEngine: {e}")
            raise CustomTemplateError(f"Initialization failed: {e}")

    async def render_template(
        self,
        template_name: str,
        context: RenderContext,
        use_cache: bool = True
    ) -> str:
        """
        Render a template with the given context.
        
        Args:
            template_name: Name of the template to render
            context: Rendering context with variables
            use_cache: Whether to use template caching
            
        Returns:
            Rendered template string
            
        Raises:
            TemplateNotFoundError: If template not found
            TemplateRenderError: If rendering fails
        """
        if self.performance_monitor:
            timer = self.performance_monitor.start_timer("render_template")
        
        try:
            # Get compiled template
            template = await self._get_template(template_name, use_cache)
            
            # Prepare rendering context
            render_vars = await self._prepare_render_context(context)
            
            # Render template
            rendered = template.render(render_vars)
            
            # Validate rendered output
            await self._validate_rendered_output(rendered, context)
            
            self.logger.debug(f"Successfully rendered template: {template_name}")
            return rendered
            
        except TemplateError as e:
            self.logger.error(f"Template rendering failed for {template_name}: {e}")
            raise TemplateRenderError(f"Failed to render template {template_name}: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error rendering template {template_name}: {e}")
            raise TemplateRenderError(f"Unexpected rendering error: {e}")
        finally:
            if self.performance_monitor and 'timer' in locals():
                self.performance_monitor.end_timer(timer)

    async def render_string_template(
        self,
        template_string: str,
        context: RenderContext,
        template_name: Optional[str] = None
    ) -> str:
        """
        Render a template from string.
        
        Args:
            template_string: Template content as string
            context: Rendering context with variables
            template_name: Optional name for caching
            
        Returns:
            Rendered template string
        """
        try:
            # Generate cache key if template name provided
            cache_key = None
            if template_name:
                cache_key = f"string_template:{template_name}"
                
                # Check if template is cached
                if cache_key in self.template_cache:
                    template = self.template_cache[cache_key]
                else:
                    # Compile and cache template
                    template = self.env.from_string(template_string)
                    self.template_cache[cache_key] = template
                    self.cache_timestamps[cache_key] = datetime.utcnow()
            else:
                # Compile template without caching
                template = self.env.from_string(template_string)
            
            # Prepare rendering context
            render_vars = await self._prepare_render_context(context)
            
            # Render template
            rendered = template.render(render_vars)
            
            # Validate rendered output
            await self._validate_rendered_output(rendered, context)
            
            return rendered
            
        except Exception as e:
            self.logger.error(f"Failed to render string template: {e}")
            raise TemplateRenderError(f"String template rendering failed: {e}")

    async def create_template(
        self,
        name: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> TemplateMetadata:
        """
        Create a new template.
        
        Args:
            name: Template name
            content: Template content
            metadata: Template metadata
            
        Returns:
            Created template metadata
        """
        try:
            # Validate template content
            await self._validate_template_content(content)
            
            # Create template file
            template_file = self.templates_path / f"{name}.j2"
            
            with open(template_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Create metadata
            template_metadata = TemplateMetadata(
                name=name,
                path=str(template_file),
                version=metadata.get('version', '1.0.0'),
                author=metadata.get('author', 'Unknown'),
                description=metadata.get('description', ''),
                variables=metadata.get('variables', []),
                dependencies=metadata.get('dependencies', []),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Save metadata
            await self._save_template_metadata(template_metadata)
            
            # Compile template
            template = await self._compile_template(name, content)
            self.template_cache[name] = template
            self.cache_timestamps[name] = datetime.utcnow()
            
            self.logger.info(f"Created template: {name}")
            return template_metadata
            
        except Exception as e:
            self.logger.error(f"Failed to create template {name}: {e}")
            raise CustomTemplateError(f"Template creation failed: {e}")

    async def update_template(
        self,
        name: str,
        content: str,
        metadata_updates: Optional[Dict[str, Any]] = None
    ) -> TemplateMetadata:
        """Update an existing template."""
        try:
            # Check if template exists
            if name not in self.metadata_cache:
                raise TemplateNotFoundError(f"Template not found: {name}")
            
            # Validate template content
            await self._validate_template_content(content)
            
            # Update template file
            template_file = self.templates_path / f"{name}.j2"
            
            with open(template_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Update metadata
            template_metadata = self.metadata_cache[name]
            template_metadata.updated_at = datetime.utcnow()
            
            if metadata_updates:
                for key, value in metadata_updates.items():
                    if hasattr(template_metadata, key):
                        setattr(template_metadata, key, value)
            
            # Save updated metadata
            await self._save_template_metadata(template_metadata)
            
            # Recompile template
            template = await self._compile_template(name, content)
            self.template_cache[name] = template
            self.cache_timestamps[name] = datetime.utcnow()
            
            self.logger.info(f"Updated template: {name}")
            return template_metadata
            
        except Exception as e:
            self.logger.error(f"Failed to update template {name}: {e}")
            raise CustomTemplateError(f"Template update failed: {e}")

    async def delete_template(self, name: str) -> bool:
        """Delete a template."""
        try:
            # Remove template file
            template_file = self.templates_path / f"{name}.j2"
            if template_file.exists():
                template_file.unlink()
            
            # Remove metadata file
            metadata_file = self.templates_path / f"{name}.metadata.json"
            if metadata_file.exists():
                metadata_file.unlink()
            
            # Remove from caches
            self.template_cache.pop(name, None)
            self.metadata_cache.pop(name, None)
            self.cache_timestamps.pop(name, None)
            
            self.logger.info(f"Deleted template: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete template {name}: {e}")
            return False

    async def list_templates(self) -> List[TemplateMetadata]:
        """List all available templates."""
        return list(self.metadata_cache.values())

    async def get_template_metadata(self, name: str) -> Optional[TemplateMetadata]:
        """Get metadata for a specific template."""
        return self.metadata_cache.get(name)

    async def validate_template(self, name: str) -> Dict[str, Any]:
        """
        Validate a template for syntax and best practices.
        
        Args:
            name: Template name to validate
            
        Returns:
            Validation result with errors and warnings
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        try:
            # Get template
            template_file = self.templates_path / f"{name}.j2"
            if not template_file.exists():
                validation_result['valid'] = False
                validation_result['errors'].append(f"Template file not found: {name}")
                return validation_result
            
            # Read template content
            with open(template_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Validate syntax
            try:
                self.env.parse(content)
            except TemplateError as e:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Syntax error: {e}")
            
            # Check template size
            if len(content) > self.max_template_size:
                validation_result['warnings'].append(
                    f"Template size ({len(content)} bytes) exceeds recommended maximum ({self.max_template_size} bytes)"
                )
            
            # Check for undefined variables
            undefined_vars = self._find_undefined_variables(content)
            if undefined_vars:
                validation_result['warnings'].extend([
                    f"Potentially undefined variable: {var}" for var in undefined_vars
                ])
            
            # Check for best practices
            suggestions = self._check_template_best_practices(content)
            validation_result['suggestions'].extend(suggestions)
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Validation error: {e}")
        
        return validation_result

    async def get_template_variables(self, name: str) -> List[str]:
        """Extract all variables used in a template."""
        try:
            template_file = self.templates_path / f"{name}.j2"
            if not template_file.exists():
                return []
            
            with open(template_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse template to extract variables
            ast = self.env.parse(content)
            variables = set()
            
            for node in ast.find_all():
                if hasattr(node, 'name') and isinstance(node.name, str):
                    variables.add(node.name)
            
            return sorted(list(variables))
            
        except Exception as e:
            self.logger.error(f"Failed to extract variables from template {name}: {e}")
            return []

    async def clear_cache(self, template_name: Optional[str] = None) -> None:
        """Clear template cache."""
        if template_name:
            # Clear specific template
            self.template_cache.pop(template_name, None)
            self.cache_timestamps.pop(template_name, None)
        else:
            # Clear all templates
            self.template_cache.clear()
            self.cache_timestamps.clear()
        
        self.logger.info(f"Cleared template cache for: {template_name or 'all templates'}")

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get template cache statistics."""
        stats = {
            'cached_templates': len(self.template_cache),
            'cache_timestamps': {
                name: timestamp.isoformat() 
                for name, timestamp in self.cache_timestamps.items()
            },
            'total_metadata': len(self.metadata_cache)
        }
        
        # Calculate cache efficiency if monitoring is enabled
        if self.performance_monitor:
            metrics = await self.performance_monitor.get_metrics()
            if 'template_cache_hits' in metrics and 'template_cache_misses' in metrics:
                hits = metrics['template_cache_hits']
                misses = metrics['template_cache_misses']
                total = hits + misses
                stats['cache_hit_ratio'] = hits / total if total > 0 else 0
        
        return stats

    # Private helper methods
    
    async def _get_template(self, name: str, use_cache: bool = True) -> Template:
        """Get compiled template with caching."""
        # Check cache first
        if use_cache and name in self.template_cache:
            cache_time = self.cache_timestamps.get(name)
            if cache_time and (datetime.utcnow() - cache_time).seconds < self.cache_ttl:
                return self.template_cache[name]
        
        # Load template from file
        try:
            template_file = self.templates_path / f"{name}.j2"
            if not template_file.exists():
                raise TemplateNotFoundError(f"Template not found: {name}")
            
            with open(template_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Compile template
            template = await self._compile_template(name, content)
            
            # Cache template
            if use_cache:
                self.template_cache[name] = template
                self.cache_timestamps[name] = datetime.utcnow()
            
            return template
            
        except Exception as e:
            self.logger.error(f"Failed to load template {name}: {e}")
            raise TemplateNotFoundError(f"Template loading failed: {e}")

    async def _compile_template(self, name: str, content: str) -> Template:
        """Compile template with error handling."""
        try:
            # Validate content size
            if len(content) > self.max_template_size:
                raise TemplateCompilationError(
                    f"Template {name} exceeds maximum size ({self.max_template_size} bytes)"
                )
            
            # Compile template
            template = self.env.from_string(content, template_class=Template)
            
            # Update metadata
            if name in self.metadata_cache:
                self.metadata_cache[name].compiled_at = datetime.utcnow()
                self.metadata_cache[name].cache_key = self._generate_cache_key(content)
            
            return template
            
        except TemplateError as e:
            raise TemplateCompilationError(f"Failed to compile template {name}: {e}")

    async def _prepare_render_context(self, context: RenderContext) -> Dict[str, Any]:
        """Prepare context variables for template rendering."""
        render_vars = {
            # Context variables
            **context.variables,
            
            # Metadata
            'template_name': context.template_name,
            'locale': context.locale,
            'tenant_id': context.tenant_id,
            'environment': context.environment,
            'alert_priority': context.alert_priority,
            'timestamp': context.timestamp,
            
            # User context
            'user': context.user_context or {},
            
            # Utility functions
            'now': datetime.now(),
            'utcnow': datetime.utcnow(),
            
            # Slack-specific helpers
            'slack': {
                'mention': lambda user_id, name=None: f"<@{user_id}|{name}>" if name else f"<@{user_id}>",
                'channel': lambda channel_id, name=None: f"<#{channel_id}|{name}>" if name else f"<#{channel_id}>",
                'emoji': lambda name: f":{name}:",
                'code': lambda text, lang=None: f"```{lang}\n{text}\n```" if lang else f"`{text}`",
                'link': lambda url, text=None: f"<{url}|{text}>" if text else f"<{url}>"
            }
        }
        
        return render_vars

    async def _validate_rendered_output(self, rendered: str, context: RenderContext) -> None:
        """Validate rendered template output."""
        # Check output length (Slack message limits)
        if len(rendered) > 40000:  # Slack limit
            self.logger.warning(f"Rendered template exceeds Slack message limit: {len(rendered)} characters")
        
        # Check for potential security issues
        if '<script>' in rendered.lower() or 'javascript:' in rendered.lower():
            self.logger.warning("Rendered template contains potential security risks")
        
        # Validate Slack-specific formatting
        for pattern_name, pattern in self.validation_patterns.items():
            matches = pattern.findall(rendered)
            if matches:
                self.logger.debug(f"Found {len(matches)} {pattern_name} patterns in rendered output")

    async def _validate_template_content(self, content: str) -> None:
        """Validate template content before saving."""
        # Check template size
        if len(content) > self.max_template_size:
            raise CustomTemplateError(f"Template exceeds maximum size ({self.max_template_size} bytes)")
        
        # Check syntax
        try:
            self.env.parse(content)
        except TemplateError as e:
            raise TemplateCompilationError(f"Template syntax error: {e}")
        
        # Check for dangerous patterns
        dangerous_patterns = [
            r'__import__',
            r'exec\s*\(',
            r'eval\s*\(',
            r'open\s*\(',
            r'file\s*\('
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                raise CustomTemplateError(f"Template contains potentially dangerous pattern: {pattern}")

    def _find_undefined_variables(self, content: str) -> List[str]:
        """Find potentially undefined variables in template."""
        try:
            ast = self.env.parse(content)
            undefined_vars = set()
            
            # This is a simplified implementation
            # In practice, you would need more sophisticated AST analysis
            variable_pattern = re.compile(r'{{\s*([a-zA-Z_][a-zA-Z0-9_]*)')
            matches = variable_pattern.findall(content)
            
            for match in matches:
                if match not in self.env.globals:
                    undefined_vars.add(match)
            
            return sorted(list(undefined_vars))
            
        except Exception:
            return []

    def _check_template_best_practices(self, content: str) -> List[str]:
        """Check template against best practices."""
        suggestions = []
        
        # Check for hardcoded values
        if re.search(r'(http://|https://)[^\s\'"]+', content):
            suggestions.append("Consider using variables for URLs instead of hardcoding them")
        
        # Check for long lines
        lines = content.split('\n')
        long_lines = [i+1 for i, line in enumerate(lines) if len(line) > 120]
        if long_lines:
            suggestions.append(f"Consider breaking long lines (lines: {', '.join(map(str, long_lines))})")
        
        # Check for complex expressions
        complex_expressions = re.findall(r'{%[^%]*if[^%]*%}', content)
        if len(complex_expressions) > 5:
            suggestions.append("Template has many conditional statements, consider simplifying")
        
        return suggestions

    def _generate_cache_key(self, content: str) -> str:
        """Generate cache key for template content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

    async def _load_template_metadata(self) -> None:
        """Load template metadata from files."""
        try:
            for metadata_file in self.templates_path.glob("*.metadata.json"):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                metadata = TemplateMetadata(
                    name=data['name'],
                    path=data['path'],
                    version=data['version'],
                    author=data['author'],
                    description=data['description'],
                    variables=data['variables'],
                    dependencies=data['dependencies'],
                    created_at=datetime.fromisoformat(data['created_at']),
                    updated_at=datetime.fromisoformat(data['updated_at']),
                    compiled_at=datetime.fromisoformat(data['compiled_at']) if data.get('compiled_at') else None,
                    cache_key=data.get('cache_key')
                )
                
                self.metadata_cache[metadata.name] = metadata
                
        except Exception as e:
            self.logger.warning(f"Failed to load template metadata: {e}")

    async def _save_template_metadata(self, metadata: TemplateMetadata) -> None:
        """Save template metadata to file."""
        try:
            metadata_file = self.templates_path / f"{metadata.name}.metadata.json"
            
            data = {
                'name': metadata.name,
                'path': metadata.path,
                'version': metadata.version,
                'author': metadata.author,
                'description': metadata.description,
                'variables': metadata.variables,
                'dependencies': metadata.dependencies,
                'created_at': metadata.created_at.isoformat(),
                'updated_at': metadata.updated_at.isoformat(),
                'compiled_at': metadata.compiled_at.isoformat() if metadata.compiled_at else None,
                'cache_key': metadata.cache_key
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            # Update cache
            self.metadata_cache[metadata.name] = metadata
            
        except Exception as e:
            self.logger.error(f"Failed to save template metadata for {metadata.name}: {e}")

    async def _precompile_templates(self) -> None:
        """Precompile frequently used templates."""
        try:
            # Get list of template files
            template_files = list(self.templates_path.glob("*.j2"))
            
            for template_file in template_files:
                template_name = template_file.stem
                
                try:
                    with open(template_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Compile template
                    template = await self._compile_template(template_name, content)
                    
                    # Cache template
                    self.template_cache[template_name] = template
                    self.cache_timestamps[template_name] = datetime.utcnow()
                    
                except Exception as e:
                    self.logger.warning(f"Failed to precompile template {template_name}: {e}")
            
            self.logger.info(f"Precompiled {len(self.template_cache)} templates")
            
        except Exception as e:
            self.logger.warning(f"Failed to precompile templates: {e}")

    async def close(self) -> None:
        """Clean up resources."""
        try:
            if self.performance_monitor:
                await self.performance_monitor.close()
            
            self.logger.info("TemplateEngine closed successfully")
            
        except Exception as e:
            self.logger.error(f"Error closing TemplateEngine: {e}")
