#!/usr/bin/env python3
"""
Enterprise Slack Template Manager - Advanced Industrial Grade
Developed by: Fahed Mlaiel (Lead Dev + AI Architect)

This module provides advanced template management for Slack alert notifications
with enterprise-grade features including dynamic template selection, A/B testing,
internationalization, and real-time template optimization.
"""

import os
import json
import yaml
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import jinja2
from jinja2 import Environment, FileSystemLoader, select_autoescape
import aiofiles
import hashlib
from abc import ABC, abstractmethod

# Configure advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/spotify-ai-agent/template-manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels with priority scores"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class TemplateType(Enum):
    """Template types for different alert scenarios"""
    CRITICAL = "critical"
    WARNING = "warning"
    RESOLVED = "resolved"
    ML_ALERT = "ml_alert"
    SECURITY_ALERT = "security_alert"
    PERFORMANCE_ALERT = "performance_alert"
    INFRASTRUCTURE_ALERT = "infrastructure_alert"
    INCIDENT = "incident"
    DIGEST = "digest"


class TemplateFormat(Enum):
    """Template output formats"""
    TEXT = "text"
    BLOCKS = "blocks"
    MARKDOWN = "markdown"
    HTML = "html"


@dataclass
class TemplateContext:
    """Comprehensive template context with enterprise features"""
    alert: Dict[str, Any]
    environment: str
    tenant_id: str
    language: str = "en"
    format_type: TemplateFormat = TemplateFormat.TEXT
    user_preferences: Optional[Dict[str, Any]] = None
    a_b_test_variant: Optional[str] = None
    template_version: str = "latest"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemplateMetrics:
    """Template performance and engagement metrics"""
    template_id: str
    render_time_ms: float
    engagement_score: float
    click_through_rate: float
    resolution_correlation: float
    user_feedback_score: float
    error_rate: float
    usage_count: int
    last_updated: datetime


class TemplateRenderer(ABC):
    """Abstract base class for template renderers"""
    
    @abstractmethod
    async def render(self, template_path: str, context: TemplateContext) -> str:
        """Render template with given context"""
        pass


class JinjaTemplateRenderer(TemplateRenderer):
    """Advanced Jinja2 template renderer with enterprise features"""
    
    def __init__(self, template_dirs: List[str]):
        self.template_dirs = template_dirs
        self.env = Environment(
            loader=FileSystemLoader(template_dirs),
            autoescape=select_autoescape(['html', 'xml']),
            enable_async=True,
            cache_size=1000,
            auto_reload=True
        )
        self._setup_custom_filters()
        self._setup_custom_functions()
    
    def _setup_custom_filters(self):
        """Setup custom Jinja2 filters for advanced template functionality"""
        
        def format_date(date_value, format_type='iso'):
            """Advanced date formatting filter"""
            if isinstance(date_value, str):
                date_value = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
            
            formats = {
                'iso': '%Y-%m-%dT%H:%M:%SZ',
                'short': '%Y-%m-%d %H:%M',
                'full': '%A, %B %d, %Y at %I:%M %p',
                'time': '%H:%M:%S',
                'date': '%Y-%m-%d'
            }
            return date_value.strftime(formats.get(format_type, formats['iso']))
        
        def duration_format(seconds):
            """Format duration in human readable format"""
            if not seconds:
                return "Unknown"
            
            seconds = int(seconds)
            if seconds < 60:
                return f"{seconds}s"
            elif seconds < 3600:
                return f"{seconds // 60}m {seconds % 60}s"
            elif seconds < 86400:
                hours = seconds // 3600
                minutes = (seconds % 3600) // 60
                return f"{hours}h {minutes}m"
            else:
                days = seconds // 86400
                hours = (seconds % 86400) // 3600
                return f"{days}d {hours}h"
        
        def truncate_smart(text, max_length=100):
            """Smart truncation that preserves word boundaries"""
            if len(text) <= max_length:
                return text
            
            truncated = text[:max_length]
            last_space = truncated.rfind(' ')
            if last_space > max_length * 0.8:
                truncated = truncated[:last_space]
            return truncated + "..."
        
        def get_severity_emoji(severity):
            """Get emoji for alert severity"""
            emoji_map = {
                'critical': '🚨',
                'high': '⚠️', 
                'medium': '🔶',
                'low': '💡',
                'info': 'ℹ️'
            }
            return emoji_map.get(severity.lower(), '❓')
        
        def ternary(condition, true_value, false_value):
            """Ternary operator filter"""
            return true_value if condition else false_value
        
        # Register filters
        self.env.filters['format_date'] = format_date
        self.env.filters['duration_format'] = duration_format
        self.env.filters['truncate_smart'] = truncate_smart
        self.env.filters['get_severity_emoji'] = get_severity_emoji
        self.env.filters['ternary'] = ternary
    
    def _setup_custom_functions(self):
        """Setup custom global functions for templates"""
        
        def now():
            """Get current datetime"""
            return datetime.utcnow()
        
        def format_time(time_str):
            """Format time string"""
            try:
                dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                return dt.strftime('%H:%M')
            except:
                return time_str
        
        # Register global functions
        self.env.globals['now'] = now
        self.env.globals['format_time'] = format_time
    
    async def render(self, template_path: str, context: TemplateContext) -> str:
        """Render template with comprehensive error handling and metrics"""
        start_time = datetime.utcnow()
        
        try:
            template = self.env.get_template(template_path)
            
            # Prepare context variables
            template_vars = {
                'alert': context.alert,
                'environment': context.environment,
                'tenant_id': context.tenant_id,
                'language': context.language,
                'dashboard_url': self._get_dashboard_url(context.environment),
                'metrics_url': self._get_metrics_url(context.environment),
                'logs_url': self._get_logs_url(context.environment),
                'tracing_url': self._get_tracing_url(context.environment),
                'runbook_url': self._get_runbook_url(context.environment),
                'system_version': context.metadata.get('system_version', 'latest'),
                **context.metadata
            }
            
            # Add user preferences if available
            if context.user_preferences:
                template_vars.update(context.user_preferences)
            
            # Render template
            rendered = await template.render_async(**template_vars)
            
            # Calculate render time
            render_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            logger.info(f"Template {template_path} rendered successfully in {render_time:.2f}ms")
            
            return rendered
            
        except Exception as e:
            logger.error(f"Error rendering template {template_path}: {str(e)}")
            raise TemplateRenderError(f"Failed to render template: {str(e)}")
    
    def _get_dashboard_url(self, environment: str) -> str:
        """Get dashboard URL for environment"""
        base_urls = {
            'production': 'https://monitoring.spotify-ai-agent.com',
            'staging': 'https://monitoring-staging.spotify-ai-agent.com',
            'development': 'https://monitoring-dev.spotify-ai-agent.com'
        }
        return base_urls.get(environment, base_urls['development'])
    
    def _get_metrics_url(self, environment: str) -> str:
        """Get metrics URL for environment"""
        base_urls = {
            'production': 'https://grafana.spotify-ai-agent.com',
            'staging': 'https://grafana-staging.spotify-ai-agent.com',
            'development': 'https://grafana-dev.spotify-ai-agent.com'
        }
        return base_urls.get(environment, base_urls['development'])
    
    def _get_logs_url(self, environment: str) -> str:
        """Get logs URL for environment"""
        base_urls = {
            'production': 'https://kibana.spotify-ai-agent.com',
            'staging': 'https://kibana-staging.spotify-ai-agent.com',
            'development': 'https://kibana-dev.spotify-ai-agent.com'
        }
        return base_urls.get(environment, base_urls['development'])
    
    def _get_tracing_url(self, environment: str) -> str:
        """Get tracing URL for environment"""
        base_urls = {
            'production': 'https://jaeger.spotify-ai-agent.com',
            'staging': 'https://jaeger-staging.spotify-ai-agent.com',
            'development': 'https://jaeger-dev.spotify-ai-agent.com'
        }
        return base_urls.get(environment, base_urls['development'])
    
    def _get_runbook_url(self, environment: str) -> str:
        """Get runbook URL for environment"""
        base_urls = {
            'production': 'https://runbooks.spotify-ai-agent.com',
            'staging': 'https://runbooks-staging.spotify-ai-agent.com',
            'development': 'https://runbooks-dev.spotify-ai-agent.com'
        }
        return base_urls.get(environment, base_urls['development'])


class TemplateSelector:
    """Advanced template selector with ML-powered optimization"""
    
    def __init__(self, template_dir: str):
        self.template_dir = Path(template_dir)
        self.metrics_cache = {}
        self.a_b_tests = {}
    
    async def select_template(self, context: TemplateContext) -> str:
        """Select optimal template based on context and metrics"""
        
        # Determine base template type
        template_type = self._determine_template_type(context.alert)
        
        # Get language-specific template
        language_suffix = f"_{context.language}"
        format_suffix = f"_{context.format_type.value}"
        
        # Build template path
        base_name = f"{template_type.value}{language_suffix}{format_suffix}"
        template_path = f"{base_name}.j2"
        
        # Check for A/B test variant
        if context.a_b_test_variant:
            variant_path = f"{base_name}_{context.a_b_test_variant}.j2"
            if (self.template_dir / variant_path).exists():
                template_path = variant_path
        
        # Validate template exists
        full_path = self.template_dir / template_path
        if not full_path.exists():
            # Fallback to English if localized version doesn't exist
            fallback_path = f"{template_type.value}_en{format_suffix}.j2"
            if (self.template_dir / fallback_path).exists():
                template_path = fallback_path
            else:
                raise TemplateNotFoundError(f"Template not found: {template_path}")
        
        logger.info(f"Selected template: {template_path} for context: {context}")
        return template_path
    
    def _determine_template_type(self, alert: Dict[str, Any]) -> TemplateType:
        """Determine template type based on alert characteristics"""
        
        # Check alert status
        if alert.get('status') == 'resolved':
            return TemplateType.RESOLVED
        
        # Check alert category/type
        alert_type = alert.get('type', '').lower()
        if 'ml' in alert_type or 'model' in alert_type:
            return TemplateType.ML_ALERT
        elif 'security' in alert_type or 'intrusion' in alert_type:
            return TemplateType.SECURITY_ALERT
        elif 'performance' in alert_type or 'latency' in alert_type:
            return TemplateType.PERFORMANCE_ALERT
        elif 'infrastructure' in alert_type or 'node' in alert_type:
            return TemplateType.INFRASTRUCTURE_ALERT
        elif alert.get('incident_id'):
            return TemplateType.INCIDENT
        
        # Check severity
        severity = alert.get('severity', '').lower()
        if severity in ['critical', 'emergency']:
            return TemplateType.CRITICAL
        else:
            return TemplateType.WARNING


class SlackTemplateManager:
    """Enterprise-grade Slack template manager with advanced features"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.template_dirs = self.config.get('template_directories', [])
        self.renderer = JinjaTemplateRenderer(self.template_dirs)
        self.selector = TemplateSelector(self.template_dirs[0])
        self.metrics = {}
        self.cache = {}
        
    async def render_alert_message(self, 
                                 alert: Dict[str, Any],
                                 environment: str,
                                 tenant_id: str,
                                 language: str = "en",
                                 format_type: TemplateFormat = TemplateFormat.TEXT,
                                 user_preferences: Optional[Dict[str, Any]] = None) -> str:
        """Render alert message with comprehensive context and optimization"""
        
        # Create template context
        context = TemplateContext(
            alert=alert,
            environment=environment,
            tenant_id=tenant_id,
            language=language,
            format_type=format_type,
            user_preferences=user_preferences or {},
            metadata={
                'system_version': self.config.get('system_version', 'latest'),
                'tenant_config': await self._get_tenant_config(tenant_id)
            }
        )
        
        # Select appropriate template
        template_path = await self.selector.select_template(context)
        
        # Check cache
        cache_key = self._generate_cache_key(template_path, context)
        if cache_key in self.cache:
            logger.info(f"Returning cached template for key: {cache_key}")
            return self.cache[cache_key]
        
        # Render template
        try:
            rendered_message = await self.renderer.render(template_path, context)
            
            # Cache result
            self.cache[cache_key] = rendered_message
            
            # Update metrics
            await self._update_template_metrics(template_path, context)
            
            return rendered_message
            
        except Exception as e:
            logger.error(f"Failed to render template {template_path}: {str(e)}")
            # Fallback to basic template
            return await self._render_fallback_template(context)
    
    async def _get_tenant_config(self, tenant_id: str) -> Dict[str, Any]:
        """Get tenant-specific configuration"""
        # This would typically fetch from a database or configuration service
        return {
            'branding': {
                'primary_color': '#1DB954',
                'logo_url': f'https://cdn.spotify-ai-agent.com/tenants/{tenant_id}/logo.png'
            },
            'notifications': {
                'escalation_enabled': True,
                'auto_resolve_enabled': True
            }
        }
    
    def _generate_cache_key(self, template_path: str, context: TemplateContext) -> str:
        """Generate cache key for template rendering"""
        key_data = {
            'template': template_path,
            'alert_id': context.alert.get('alert_id'),
            'environment': context.environment,
            'language': context.language,
            'format': context.format_type.value
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    async def _update_template_metrics(self, template_path: str, context: TemplateContext):
        """Update template performance metrics"""
        # Implementation would track template usage, performance, and effectiveness
        pass
    
    async def _render_fallback_template(self, context: TemplateContext) -> str:
        """Render basic fallback template when primary template fails"""
        fallback_template = """
🚨 **ALERT** - {{ alert.context.service_name }}

**Environment**: {{ environment | upper }}
**Severity**: {{ alert.severity | upper }}
**Description**: {{ alert.description }}
**Time**: {{ alert.created_at }}

Alert ID: {{ alert.alert_id }}

*Spotify AI Agent Monitoring*
        """.strip()
        
        env = Environment()
        template = env.from_string(fallback_template)
        return template.render(
            alert=context.alert,
            environment=context.environment
        )
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {str(e)}")
            return {}


class TemplateRenderError(Exception):
    """Exception raised when template rendering fails"""
    pass


class TemplateNotFoundError(Exception):
    """Exception raised when template file is not found"""
    pass


# Example usage and factory functions
async def create_slack_template_manager(config_path: str = None) -> SlackTemplateManager:
    """Factory function to create SlackTemplateManager instance"""
    if not config_path:
        config_path = os.environ.get('TEMPLATE_CONFIG_PATH', 'template_config.yaml')
    
    return SlackTemplateManager(config_path)


async def render_slack_alert(alert_data: Dict[str, Any], 
                           environment: str,
                           tenant_id: str = "default",
                           **kwargs) -> str:
    """Convenience function to render Slack alert message"""
    manager = await create_slack_template_manager()
    return await manager.render_alert_message(
        alert=alert_data,
        environment=environment,
        tenant_id=tenant_id,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    async def main():
        # Sample alert data
        sample_alert = {
            "alert_id": "alert-123456",
            "title": "High CPU Usage Detected",
            "description": "CPU usage has exceeded 90% for the past 5 minutes",
            "severity": "critical",
            "status": "firing",
            "created_at": datetime.utcnow().isoformat(),
            "context": {
                "service_name": "spotify-ai-recommender",
                "component": "recommendation-engine",
                "instance_id": "i-0123456789abcdef0",
                "cluster_name": "production-us-east-1"
            },
            "metrics": {
                "cpu_usage": "92%",
                "memory_usage": "78%",
                "load_average": "4.2"
            },
            "ai_insights": {
                "recommended_actions": [
                    "Scale up the instance to handle increased load",
                    "Check for memory leaks in the application",
                    "Review recent deployment for performance issues"
                ],
                "confidence_score": 85
            }
        }
        
        # Render alert message
        message = await render_slack_alert(
            alert_data=sample_alert,
            environment="production",
            tenant_id="spotify-main",
            language="en",
            format_type=TemplateFormat.TEXT
        )
        
        print("Rendered Slack Alert Message:")
        print("=" * 50)
        print(message)
    
    # Run example
    asyncio.run(main())
