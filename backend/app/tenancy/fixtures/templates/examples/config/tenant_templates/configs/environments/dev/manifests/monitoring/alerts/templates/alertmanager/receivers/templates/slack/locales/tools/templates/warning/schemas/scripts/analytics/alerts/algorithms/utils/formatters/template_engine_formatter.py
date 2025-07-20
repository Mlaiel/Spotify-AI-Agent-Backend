"""
Spotify AI Agent - Advanced Template Engine & Dynamic Content Formatters
=====================================================================

Ultra-advanced template engine with dynamic content generation, conditional logic,
macro systems, and intelligent content adaptation for various output formats.

This module handles sophisticated formatting for:
- Jinja2 template engine with custom filters and functions
- Dynamic content generation based on data patterns
- Conditional logic and branching templates
- Macro systems for reusable template components
- Multi-format output (HTML, JSON, XML, Markdown, PDF)
- Template inheritance and composition
- Real-time template compilation and caching
- A/B testing for template variations
- Accessibility-aware template generation
- Multi-language template systems with cultural adaptation

Author: Fahed Mlaiel & Spotify Template Engineering Team
Version: 2.1.0
"""

import asyncio
import json
import re
import hashlib
import base64
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import structlog
from jinja2 import Environment, FileSystemLoader, select_autoescape, meta
from jinja2.sandbox import SandboxedEnvironment
from jinja2.exceptions import TemplateError, TemplateSyntaxError
import yaml
from pathlib import Path

logger = structlog.get_logger(__name__)


class OutputFormat(Enum):
    """Supported output formats."""
    HTML = "html"
    JSON = "json"
    XML = "xml"
    MARKDOWN = "markdown"
    PDF = "pdf"
    CSV = "csv"
    YAML = "yaml"
    PLAIN_TEXT = "plain_text"
    LATEX = "latex"
    SVG = "svg"


class TemplateType(Enum):
    """Template types."""
    ALERT = "alert"
    REPORT = "report"
    EMAIL = "email"
    DASHBOARD = "dashboard"
    NOTIFICATION = "notification"
    EXPORT = "export"
    PRESENTATION = "presentation"
    DOCUMENTATION = "documentation"


class ContentStrategy(Enum):
    """Content generation strategies."""
    STATIC = "static"
    DYNAMIC = "dynamic"
    ADAPTIVE = "adaptive"
    PERSONALIZED = "personalized"
    CONTEXTUAL = "contextual"


@dataclass
class TemplateConfig:
    """Template configuration."""
    
    name: str
    template_type: TemplateType
    output_format: OutputFormat
    content_strategy: ContentStrategy
    template_path: str
    cache_ttl: int = 3600  # seconds
    auto_escape: bool = True
    sandbox_mode: bool = True
    accessibility_enabled: bool = True
    multi_language: bool = False
    a_b_testing: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "template_type": self.template_type.value,
            "output_format": self.output_format.value,
            "content_strategy": self.content_strategy.value,
            "template_path": self.template_path,
            "cache_ttl": self.cache_ttl,
            "auto_escape": self.auto_escape,
            "sandbox_mode": self.sandbox_mode,
            "accessibility_enabled": self.accessibility_enabled,
            "multi_language": self.multi_language,
            "a_b_testing": self.a_b_testing,
            "metadata": self.metadata
        }


@dataclass
class TemplateContext:
    """Template rendering context."""
    
    data: Dict[str, Any]
    user_context: Dict[str, Any] = field(default_factory=dict)
    environment_vars: Dict[str, Any] = field(default_factory=dict)
    localization: Dict[str, Any] = field(default_factory=dict)
    accessibility_options: Dict[str, Any] = field(default_factory=dict)
    theme_config: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for template variables."""
        return {
            "data": self.data,
            "user": self.user_context,
            "env": self.environment_vars,
            "i18n": self.localization,
            "a11y": self.accessibility_options,
            "theme": self.theme_config,
            "timestamp": self.timestamp,
            "now": self.timestamp
        }


@dataclass
class RenderedTemplate:
    """Container for rendered template result."""
    
    content: str
    output_format: OutputFormat
    template_name: str
    render_time_ms: float
    cache_key: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    accessibility_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "output_format": self.output_format.value,
            "template_name": self.template_name,
            "render_time_ms": self.render_time_ms,
            "cache_key": self.cache_key,
            "metadata": self.metadata,
            "warnings": self.warnings,
            "accessibility_score": self.accessibility_score
        }


class SpotifyTemplateEngine:
    """Advanced template engine with Spotify-specific optimizations."""
    
    def __init__(self, tenant_id: str, config: Optional[Dict[str, Any]] = None):
        self.tenant_id = tenant_id
        self.config = config or {}
        self.logger = logger.bind(tenant_id=tenant_id, component="template_engine")
        
        # Template cache
        self._template_cache: Dict[str, Any] = {}
        self._compiled_cache: Dict[str, Any] = {}
        
        # Initialize Jinja2 environment
        self._init_jinja_environment()
        
        # Register custom filters and functions
        self._register_custom_filters()
        self._register_custom_functions()
        
        # A/B testing configuration
        self.ab_testing_enabled = config.get('ab_testing_enabled', False)
        self.ab_variants: Dict[str, List[str]] = {}
        
        # Performance monitoring
        self.render_stats: Dict[str, Any] = {
            "total_renders": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_render_time": 0.0,
            "errors": 0
        }
    
    def _init_jinja_environment(self) -> None:
        """Initialize Jinja2 environment with security and performance optimizations."""
        
        template_dirs = self.config.get('template_directories', [
            'templates',
            'templates/alerts',
            'templates/reports',
            'templates/emails'
        ])
        
        # Use sandboxed environment for security
        if self.config.get('sandbox_mode', True):
            self.jinja_env = SandboxedEnvironment(
                loader=FileSystemLoader(template_dirs),
                autoescape=select_autoescape(['html', 'xml']),
                enable_async=True,
                cache_size=self.config.get('cache_size', 1000),
                auto_reload=self.config.get('auto_reload', False)
            )
        else:
            self.jinja_env = Environment(
                loader=FileSystemLoader(template_dirs),
                autoescape=select_autoescape(['html', 'xml']),
                enable_async=True,
                cache_size=self.config.get('cache_size', 1000),
                auto_reload=self.config.get('auto_reload', False)
            )
        
        # Configure environment settings
        self.jinja_env.trim_blocks = True
        self.jinja_env.lstrip_blocks = True
        self.jinja_env.keep_trailing_newline = True
    
    def _register_custom_filters(self) -> None:
        """Register custom Jinja2 filters for Spotify-specific formatting."""
        
        @self.jinja_env.filter
        def spotify_number(value: Union[int, float], precision: int = 2) -> str:
            """Format numbers in Spotify style (e.g., 1.2M, 3.4K)."""
            try:
                num = float(value)
                if num >= 1_000_000_000:
                    return f"{num / 1_000_000_000:.{precision}f}B"
                elif num >= 1_000_000:
                    return f"{num / 1_000_000:.{precision}f}M"
                elif num >= 1_000:
                    return f"{num / 1_000:.{precision}f}K"
                else:
                    return f"{num:.{precision}f}"
            except (ValueError, TypeError):
                return str(value)
        
        @self.jinja_env.filter
        def spotify_duration(seconds: Union[int, float]) -> str:
            """Format duration in Spotify style (e.g., 3:45, 1:23:45)."""
            try:
                total_seconds = int(seconds)
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                secs = total_seconds % 60
                
                if hours > 0:
                    return f"{hours}:{minutes:02d}:{secs:02d}"
                else:
                    return f"{minutes}:{secs:02d}"
            except (ValueError, TypeError):
                return str(seconds)
        
        @self.jinja_env.filter
        def spotify_popularity(score: Union[int, float]) -> str:
            """Convert popularity score to visual representation."""
            try:
                popularity = float(score)
                if popularity >= 80:
                    return "🔥🔥🔥🔥🔥 Viral"
                elif popularity >= 60:
                    return "🔥🔥🔥🔥 Popular"
                elif popularity >= 40:
                    return "🔥🔥🔥 Trending"
                elif popularity >= 20:
                    return "🔥🔥 Growing"
                else:
                    return "🔥 Emerging"
            except (ValueError, TypeError):
                return "Unknown"
        
        @self.jinja_env.filter
        def to_chart_data(data: List[Dict[str, Any]], x_field: str, y_field: str) -> str:
            """Convert data to Chart.js format."""
            try:
                chart_data = {
                    "labels": [item.get(x_field, '') for item in data],
                    "datasets": [{
                        "data": [item.get(y_field, 0) for item in data],
                        "backgroundColor": "#1DB954",
                        "borderColor": "#169C47"
                    }]
                }
                return json.dumps(chart_data)
            except Exception:
                return "{}"
        
        @self.jinja_env.filter
        def accessibility_alt(text: str, context: str = "") -> str:
            """Generate accessibility-friendly alt text."""
            if not text:
                return f"Image: {context}" if context else "Image"
            
            # Clean and enhance alt text
            clean_text = re.sub(r'[^\w\s-]', '', text).strip()
            if context:
                return f"{context}: {clean_text}"
            return clean_text
        
        @self.jinja_env.filter
        def sanitize_html(text: str) -> str:
            """Sanitize HTML content for security."""
            # Basic HTML sanitization (in production, use a proper library like bleach)
            text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.IGNORECASE | re.DOTALL)
            text = re.sub(r'on\w+="[^"]*"', '', text, flags=re.IGNORECASE)
            return text
        
        @self.jinja_env.filter
        def format_percentage(value: Union[int, float], precision: int = 1) -> str:
            """Format percentage values."""
            try:
                return f"{float(value):.{precision}f}%"
            except (ValueError, TypeError):
                return "0.0%"
        
        @self.jinja_env.filter
        def truncate_smart(text: str, length: int = 100, suffix: str = "...") -> str:
            """Smart text truncation that respects word boundaries."""
            if len(text) <= length:
                return text
            
            # Find the last space within the length limit
            truncated = text[:length].rsplit(' ', 1)[0]
            return f"{truncated}{suffix}"
        
        @self.jinja_env.filter
        def highlight_search(text: str, query: str) -> str:
            """Highlight search terms in text."""
            if not query:
                return text
            
            pattern = re.compile(re.escape(query), re.IGNORECASE)
            return pattern.sub(f'<mark>{query}</mark>', text)
    
    def _register_custom_functions(self) -> None:
        """Register custom Jinja2 global functions."""
        
        @self.jinja_env.global_function
        def get_spotify_color(name: str) -> str:
            """Get Spotify brand colors."""
            colors = {
                'spotify_green': '#1DB954',
                'spotify_green_light': '#1ED760',
                'spotify_black': '#191414',
                'spotify_white': '#FFFFFF',
                'spotify_gray': '#535353',
                'spotify_gray_light': '#B3B3B3'
            }
            return colors.get(name, '#1DB954')
        
        @self.jinja_env.global_function
        def generate_chart_id() -> str:
            """Generate unique chart ID."""
            return f"chart_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
        
        @self.jinja_env.global_function
        def format_currency(amount: Union[int, float], currency: str = "USD") -> str:
            """Format currency values."""
            try:
                formatted_amount = f"{float(amount):,.2f}"
                currency_symbols = {
                    'USD': '$',
                    'EUR': '€',
                    'GBP': '£',
                    'JPY': '¥'
                }
                symbol = currency_symbols.get(currency.upper(), currency)
                return f"{symbol}{formatted_amount}"
            except (ValueError, TypeError):
                return f"{currency} 0.00"
        
        @self.jinja_env.global_function
        def calculate_growth_rate(current: Union[int, float], previous: Union[int, float]) -> float:
            """Calculate growth rate percentage."""
            try:
                if previous == 0:
                    return 100.0 if current > 0 else 0.0
                return ((current - previous) / previous) * 100
            except (ValueError, TypeError, ZeroDivisionError):
                return 0.0
        
        @self.jinja_env.global_function
        def get_time_ago(timestamp: datetime) -> str:
            """Get human-readable time ago string."""
            try:
                now = datetime.now(timezone.utc)
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                
                diff = now - timestamp
                
                if diff.days > 30:
                    return f"{diff.days // 30} month{'s' if diff.days // 30 != 1 else ''} ago"
                elif diff.days > 0:
                    return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
                elif diff.seconds > 3600:
                    hours = diff.seconds // 3600
                    return f"{hours} hour{'s' if hours != 1 else ''} ago"
                elif diff.seconds > 60:
                    minutes = diff.seconds // 60
                    return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
                else:
                    return "Just now"
            except Exception:
                return "Unknown"
        
        @self.jinja_env.global_function
        def generate_qr_code(data: str, size: int = 200) -> str:
            """Generate QR code data URL (placeholder implementation)."""
            # In production, this would generate actual QR codes
            encoded_data = base64.b64encode(data.encode()).decode()
            return f"data:image/svg+xml;base64,{encoded_data}"
        
        @self.jinja_env.global_function
        def accessibility_label(element_type: str, content: str) -> str:
            """Generate accessibility labels for UI elements."""
            labels = {
                'button': f"Button: {content}",
                'link': f"Link to {content}",
                'image': f"Image showing {content}",
                'chart': f"Chart displaying {content}",
                'table': f"Data table: {content}"
            }
            return labels.get(element_type, content)
    
    async def render_template(self, 
                            template_config: TemplateConfig,
                            context: TemplateContext) -> RenderedTemplate:
        """Render template with given configuration and context."""
        
        start_time = datetime.now()
        
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(template_config, context)
            
            # Check cache first
            if cached_result := self._get_from_cache(cache_key, template_config.cache_ttl):
                self.render_stats["cache_hits"] += 1
                return cached_result
            
            self.render_stats["cache_misses"] += 1
            
            # Load and compile template
            template = await self._load_template(template_config)
            
            # Prepare template variables
            template_vars = await self._prepare_template_variables(context, template_config)
            
            # Render template
            if template_config.sandbox_mode:
                rendered_content = await template.render_async(**template_vars)
            else:
                rendered_content = template.render(**template_vars)
            
            # Post-process content
            processed_content = await self._post_process_content(
                rendered_content, template_config, context
            )
            
            # Calculate render time
            render_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Calculate accessibility score if enabled
            accessibility_score = None
            if template_config.accessibility_enabled:
                accessibility_score = await self._calculate_accessibility_score(processed_content)
            
            # Create result
            result = RenderedTemplate(
                content=processed_content,
                output_format=template_config.output_format,
                template_name=template_config.name,
                render_time_ms=render_time,
                cache_key=cache_key,
                accessibility_score=accessibility_score,
                metadata={
                    "template_type": template_config.template_type.value,
                    "content_strategy": template_config.content_strategy.value,
                    "render_timestamp": datetime.now(timezone.utc).isoformat(),
                    "tenant_id": self.tenant_id
                }
            )
            
            # Cache result
            self._cache_result(cache_key, result, template_config.cache_ttl)
            
            # Update statistics
            self.render_stats["total_renders"] += 1
            self._update_avg_render_time(render_time)
            
            self.logger.info(
                "Template rendered successfully",
                template_name=template_config.name,
                render_time_ms=render_time,
                cache_key=cache_key
            )
            
            return result
            
        except Exception as e:
            self.render_stats["errors"] += 1
            self.logger.error(
                "Template rendering failed",
                template_name=template_config.name,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def _load_template(self, config: TemplateConfig):
        """Load and compile template."""
        
        template_key = f"{config.name}_{config.template_path}"
        
        # Check compiled cache
        if template_key in self._compiled_cache:
            return self._compiled_cache[template_key]
        
        try:
            # Load template from file system
            template = self.jinja_env.get_template(config.template_path)
            
            # Cache compiled template
            self._compiled_cache[template_key] = template
            
            return template
            
        except TemplateError as e:
            self.logger.error(
                "Template loading failed",
                template_path=config.template_path,
                error=str(e)
            )
            raise
    
    async def _prepare_template_variables(self, 
                                        context: TemplateContext,
                                        config: TemplateConfig) -> Dict[str, Any]:
        """Prepare template variables from context."""
        
        variables = context.to_dict()
        
        # Add template-specific variables
        variables.update({
            "template_name": config.name,
            "template_type": config.template_type.value,
            "output_format": config.output_format.value,
            "tenant_id": self.tenant_id,
            "config": config.metadata
        })
        
        # Add dynamic content if strategy requires it
        if config.content_strategy == ContentStrategy.DYNAMIC:
            variables.update(await self._generate_dynamic_content(context, config))
        elif config.content_strategy == ContentStrategy.PERSONALIZED:
            variables.update(await self._generate_personalized_content(context, config))
        elif config.content_strategy == ContentStrategy.ADAPTIVE:
            variables.update(await self._generate_adaptive_content(context, config))
        
        # Add A/B testing variants if enabled
        if config.a_b_testing and self.ab_testing_enabled:
            variables.update(await self._get_ab_testing_variants(config, context))
        
        return variables
    
    async def _generate_dynamic_content(self, 
                                      context: TemplateContext,
                                      config: TemplateConfig) -> Dict[str, Any]:
        """Generate dynamic content based on data patterns."""
        
        dynamic_content = {}
        data = context.data
        
        # Auto-generate insights based on data
        if isinstance(data, dict):
            # Generate summary statistics
            numeric_fields = {k: v for k, v in data.items() 
                            if isinstance(v, (int, float))}
            
            if numeric_fields:
                dynamic_content["auto_insights"] = {
                    "max_value": max(numeric_fields.values()),
                    "min_value": min(numeric_fields.values()),
                    "avg_value": sum(numeric_fields.values()) / len(numeric_fields),
                    "total_fields": len(numeric_fields)
                }
            
            # Generate trend indicators
            if "metrics" in data and isinstance(data["metrics"], list):
                metrics = data["metrics"]
                if len(metrics) >= 2:
                    latest = metrics[-1]
                    previous = metrics[-2]
                    
                    if isinstance(latest, dict) and isinstance(previous, dict):
                        trends = {}
                        for key in latest:
                            if key in previous and isinstance(latest[key], (int, float)):
                                change = latest[key] - previous[key]
                                trends[f"{key}_trend"] = "up" if change > 0 else "down" if change < 0 else "stable"
                                trends[f"{key}_change"] = change
                        
                        dynamic_content["trends"] = trends
        
        return dynamic_content
    
    async def _generate_personalized_content(self, 
                                           context: TemplateContext,
                                           config: TemplateConfig) -> Dict[str, Any]:
        """Generate personalized content based on user context."""
        
        personalized_content = {}
        user_context = context.user_context
        
        # User preferences
        if "preferences" in user_context:
            prefs = user_context["preferences"]
            personalized_content["user_prefs"] = prefs
            
            # Customize content based on preferences
            if prefs.get("theme") == "dark":
                personalized_content["theme_colors"] = {
                    "background": "#191414",
                    "text": "#FFFFFF",
                    "accent": "#1DB954"
                }
            else:
                personalized_content["theme_colors"] = {
                    "background": "#FFFFFF",
                    "text": "#191414",
                    "accent": "#1DB954"
                }
        
        # User role-based content
        if "role" in user_context:
            role = user_context["role"]
            if role == "admin":
                personalized_content["show_admin_features"] = True
            elif role == "analyst":
                personalized_content["show_analytics_features"] = True
            elif role == "viewer":
                personalized_content["show_basic_features"] = True
        
        # Localization
        if "locale" in user_context:
            locale = user_context["locale"]
            personalized_content["locale_specific"] = {
                "date_format": "%Y-%m-%d" if locale.startswith("en") else "%d.%m.%Y",
                "number_format": "," if locale.startswith("en") else ".",
                "currency_symbol": "$" if locale.startswith("en_US") else "€"
            }
        
        return personalized_content
    
    async def _generate_adaptive_content(self, 
                                       context: TemplateContext,
                                       config: TemplateConfig) -> Dict[str, Any]:
        """Generate adaptive content based on context and environment."""
        
        adaptive_content = {}
        
        # Time-based adaptations
        now = context.timestamp
        hour = now.hour
        
        if 6 <= hour < 12:
            adaptive_content["time_context"] = {
                "greeting": "Good morning",
                "energy_level": "high",
                "suggested_actions": ["Review overnight metrics", "Plan daily activities"]
            }
        elif 12 <= hour < 18:
            adaptive_content["time_context"] = {
                "greeting": "Good afternoon",
                "energy_level": "medium",
                "suggested_actions": ["Check progress", "Address urgent items"]
            }
        else:
            adaptive_content["time_context"] = {
                "greeting": "Good evening",
                "energy_level": "low",
                "suggested_actions": ["Review daily summary", "Plan tomorrow"]
            }
        
        # Device-based adaptations
        if "device_info" in context.environment_vars:
            device = context.environment_vars["device_info"]
            if device.get("type") == "mobile":
                adaptive_content["layout"] = {
                    "mobile_optimized": True,
                    "column_count": 1,
                    "font_size": "large"
                }
            elif device.get("type") == "tablet":
                adaptive_content["layout"] = {
                    "tablet_optimized": True,
                    "column_count": 2,
                    "font_size": "medium"
                }
            else:
                adaptive_content["layout"] = {
                    "desktop_optimized": True,
                    "column_count": 3,
                    "font_size": "normal"
                }
        
        return adaptive_content
    
    async def _get_ab_testing_variants(self, 
                                     config: TemplateConfig,
                                     context: TemplateContext) -> Dict[str, Any]:
        """Get A/B testing variants for template."""
        
        variants = {}
        
        # Generate user hash for consistent variant assignment
        user_id = context.user_context.get("user_id", "anonymous")
        user_hash = hashlib.md5(f"{user_id}_{config.name}".encode()).hexdigest()
        variant_selector = int(user_hash[:8], 16) % 100
        
        # Define A/B test variants for different template types
        if config.template_type == TemplateType.ALERT:
            if variant_selector < 50:
                variants["alert_style"] = "compact"
                variants["alert_priority"] = "high"
            else:
                variants["alert_style"] = "detailed"
                variants["alert_priority"] = "normal"
        
        elif config.template_type == TemplateType.EMAIL:
            if variant_selector < 50:
                variants["email_layout"] = "single_column"
                variants["cta_style"] = "button"
            else:
                variants["email_layout"] = "two_column"
                variants["cta_style"] = "link"
        
        elif config.template_type == TemplateType.DASHBOARD:
            if variant_selector < 50:
                variants["chart_type"] = "line"
                variants["color_scheme"] = "vibrant"
            else:
                variants["chart_type"] = "bar"
                variants["color_scheme"] = "muted"
        
        variants["ab_variant_id"] = f"variant_{variant_selector}"
        return variants
    
    async def _post_process_content(self, 
                                  content: str,
                                  config: TemplateConfig,
                                  context: TemplateContext) -> str:
        """Post-process rendered content."""
        
        processed = content
        
        # Format-specific post-processing
        if config.output_format == OutputFormat.HTML:
            processed = await self._post_process_html(processed, config, context)
        elif config.output_format == OutputFormat.JSON:
            processed = await self._post_process_json(processed, config, context)
        elif config.output_format == OutputFormat.MARKDOWN:
            processed = await self._post_process_markdown(processed, config, context)
        
        # Accessibility enhancements
        if config.accessibility_enabled:
            processed = await self._enhance_accessibility(processed, config)
        
        # Security sanitization
        processed = await self._sanitize_content(processed, config)
        
        return processed
    
    async def _post_process_html(self, 
                               content: str,
                               config: TemplateConfig,
                               context: TemplateContext) -> str:
        """Post-process HTML content."""
        
        # Add responsive meta tags if not present
        if "<meta name=\"viewport\"" not in content and "<html" in content:
            viewport_meta = '<meta name="viewport" content="width=device-width, initial-scale=1">'
            content = content.replace("<head>", f"<head>\n{viewport_meta}")
        
        # Add Spotify brand CSS if requested
        if config.metadata.get("include_spotify_branding", False):
            spotify_css = """
            <style>
                :root {
                    --spotify-green: #1DB954;
                    --spotify-black: #191414;
                    --spotify-white: #FFFFFF;
                }
                .spotify-button {
                    background-color: var(--spotify-green);
                    color: var(--spotify-white);
                    border: none;
                    padding: 12px 24px;
                    border-radius: 500px;
                    font-weight: bold;
                }
            </style>
            """
            content = content.replace("</head>", f"{spotify_css}\n</head>")
        
        return content
    
    async def _post_process_json(self, 
                               content: str,
                               config: TemplateConfig,
                               context: TemplateContext) -> str:
        """Post-process JSON content."""
        
        try:
            # Validate and pretty-print JSON
            json_data = json.loads(content)
            return json.dumps(json_data, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            self.logger.warning("Invalid JSON in template output", template_name=config.name)
            return content
    
    async def _post_process_markdown(self, 
                                   content: str,
                                   config: TemplateConfig,
                                   context: TemplateContext) -> str:
        """Post-process Markdown content."""
        
        # Add table of contents if requested
        if config.metadata.get("include_toc", False):
            toc = self._generate_markdown_toc(content)
            if toc:
                content = f"{toc}\n\n{content}"
        
        # Ensure proper spacing around headers
        content = re.sub(r'\n(#{1,6}\s)', r'\n\n\1', content)
        
        return content
    
    def _generate_markdown_toc(self, content: str) -> str:
        """Generate table of contents for Markdown."""
        
        headers = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
        if not headers:
            return ""
        
        toc_lines = ["## Table of Contents\n"]
        
        for level, title in headers:
            indent = "  " * (len(level) - 1)
            anchor = re.sub(r'[^\w\s-]', '', title).strip().lower().replace(' ', '-')
            toc_lines.append(f"{indent}- [{title}](#{anchor})")
        
        return "\n".join(toc_lines)
    
    async def _enhance_accessibility(self, 
                                   content: str,
                                   config: TemplateConfig) -> str:
        """Enhance content for accessibility."""
        
        if config.output_format == OutputFormat.HTML:
            # Add alt text to images without it
            content = re.sub(
                r'<img([^>]+)(?<!alt="[^"]*")(?<!alt=\'[^\']*\')>',
                r'<img\1 alt="Image">',
                content
            )
            
            # Add ARIA labels to interactive elements
            content = re.sub(
                r'<button([^>]*)>',
                r'<button\1 role="button">',
                content
            )
            
            # Ensure proper heading hierarchy
            content = self._fix_heading_hierarchy(content)
        
        return content
    
    def _fix_heading_hierarchy(self, content: str) -> str:
        """Fix HTML heading hierarchy for accessibility."""
        
        # Find all headings
        headings = re.findall(r'<h([1-6])[^>]*>', content)
        
        if not headings:
            return content
        
        # Check if hierarchy is proper (should start with h1 and increment by 1)
        current_level = 1
        replacements = {}
        
        for i, level in enumerate(headings):
            level_num = int(level)
            if level_num > current_level + 1:
                # Skip levels - fix by reducing to proper level
                new_level = current_level + 1
                replacements[f'h{level_num}'] = f'h{new_level}'
                current_level = new_level
            else:
                current_level = level_num
        
        # Apply replacements
        for old, new in replacements.items():
            content = content.replace(f'<{old}', f'<{new}')
            content = content.replace(f'</{old}>', f'</{new}>')
        
        return content
    
    async def _sanitize_content(self, 
                              content: str,
                              config: TemplateConfig) -> str:
        """Sanitize content for security."""
        
        if config.output_format == OutputFormat.HTML:
            # Remove dangerous script tags
            content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.IGNORECASE | re.DOTALL)
            
            # Remove event handlers
            content = re.sub(r'\s*on\w+\s*=\s*["\'][^"\']*["\']', '', content, flags=re.IGNORECASE)
            
            # Remove javascript: links
            content = re.sub(r'href\s*=\s*["\']javascript:[^"\']*["\']', 'href="#"', content, flags=re.IGNORECASE)
        
        return content
    
    async def _calculate_accessibility_score(self, content: str) -> float:
        """Calculate accessibility score for content."""
        
        score = 100.0
        
        if content.strip().startswith('<'):  # HTML content
            # Check for common accessibility issues
            
            # Images without alt text
            img_tags = re.findall(r'<img[^>]*>', content)
            img_without_alt = [img for img in img_tags if 'alt=' not in img]
            if img_without_alt:
                score -= len(img_without_alt) * 5
            
            # Missing heading structure
            if '<h1' not in content.lower():
                score -= 10
            
            # Links without descriptive text
            link_texts = re.findall(r'<a[^>]*>([^<]+)</a>', content)
            generic_links = [text for text in link_texts if text.lower().strip() in ['click here', 'read more', 'link']]
            if generic_links:
                score -= len(generic_links) * 3
            
            # Missing form labels
            input_tags = re.findall(r'<input[^>]*>', content)
            inputs_without_labels = [inp for inp in input_tags if 'aria-label=' not in inp and 'id=' not in inp]
            if inputs_without_labels:
                score -= len(inputs_without_labels) * 5
            
            # Color contrast (simplified check)
            if 'color:' in content and 'background' not in content:
                score -= 5  # Potential contrast issue
        
        return max(0.0, min(100.0, score))
    
    def _generate_cache_key(self, 
                          config: TemplateConfig,
                          context: TemplateContext) -> str:
        """Generate cache key for template rendering."""
        
        # Create hash from configuration and context
        config_hash = hashlib.md5(json.dumps(config.to_dict(), sort_keys=True).encode()).hexdigest()[:8]
        
        # Include relevant context data in hash
        context_data = {
            "user_id": context.user_context.get("user_id"),
            "locale": context.localization.get("locale"),
            "theme": context.theme_config.get("theme"),
            "data_hash": hashlib.md5(str(context.data).encode()).hexdigest()[:8]
        }
        context_hash = hashlib.md5(json.dumps(context_data, sort_keys=True).encode()).hexdigest()[:8]
        
        return f"{self.tenant_id}_{config.name}_{config_hash}_{context_hash}"
    
    def _get_from_cache(self, cache_key: str, ttl: int) -> Optional[RenderedTemplate]:
        """Get rendered template from cache."""
        
        if cache_key not in self._template_cache:
            return None
        
        cached_data = self._template_cache[cache_key]
        cache_time = cached_data.get("timestamp")
        
        if cache_time and (datetime.now() - cache_time).total_seconds() < ttl:
            return cached_data.get("result")
        
        # Remove expired cache entry
        del self._template_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: RenderedTemplate, ttl: int) -> None:
        """Cache rendered template result."""
        
        self._template_cache[cache_key] = {
            "result": result,
            "timestamp": datetime.now(),
            "ttl": ttl
        }
        
        # Clean up expired cache entries periodically
        if len(self._template_cache) > 1000:
            self._cleanup_cache()
    
    def _cleanup_cache(self) -> None:
        """Clean up expired cache entries."""
        
        now = datetime.now()
        expired_keys = []
        
        for key, data in self._template_cache.items():
            cache_time = data.get("timestamp")
            ttl = data.get("ttl", 3600)
            
            if cache_time and (now - cache_time).total_seconds() > ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._template_cache[key]
        
        self.logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _update_avg_render_time(self, render_time: float) -> None:
        """Update average render time statistics."""
        
        total_renders = self.render_stats["total_renders"]
        if total_renders == 1:
            self.render_stats["avg_render_time"] = render_time
        else:
            current_avg = self.render_stats["avg_render_time"]
            new_avg = ((current_avg * (total_renders - 1)) + render_time) / total_renders
            self.render_stats["avg_render_time"] = new_avg
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get template engine statistics."""
        
        cache_hit_rate = 0.0
        total_requests = self.render_stats["cache_hits"] + self.render_stats["cache_misses"]
        if total_requests > 0:
            cache_hit_rate = (self.render_stats["cache_hits"] / total_requests) * 100
        
        return {
            "total_renders": self.render_stats["total_renders"],
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "avg_render_time_ms": f"{self.render_stats['avg_render_time']:.2f}",
            "error_count": self.render_stats["errors"],
            "cached_templates": len(self._template_cache),
            "compiled_templates": len(self._compiled_cache)
        }


# Factory function for creating template engine formatters
def create_template_formatter(
    tenant_id: str,
    config: Optional[Dict[str, Any]] = None
) -> SpotifyTemplateEngine:
    """
    Factory function to create template engine formatter.
    
    Args:
        tenant_id: Tenant identifier
        config: Configuration dictionary
        
    Returns:
        Configured template engine instance
    """
    return SpotifyTemplateEngine(tenant_id, config or {})
