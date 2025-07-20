#!/usr/bin/env python3
"""
Notification Builder for PagerDuty Integration.

Advanced notification and message building system with template support,
rich formatting, multi-channel delivery, and dynamic content generation.

Features:
- Template-based notification building
- Rich text formatting and markup
- Multi-channel support (email, SMS, Slack, etc.)
- Dynamic content generation and placeholders
- Internationalization and localization
- Notification prioritization and routing
- Template validation and testing
- Performance optimization and caching
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import html
import markdown
from jinja2 import Template, Environment, DictLoader, TemplateError

logger = logging.getLogger(__name__)


class TemplateError(Exception):
    """Base exception for template errors."""
    pass


class NotificationError(Exception):
    """Base exception for notification errors."""
    pass


class DeliveryChannelError(NotificationError):
    """Exception raised for delivery channel errors."""
    pass


class NotificationChannel(Enum):
    """Supported notification channels."""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    MOBILE_PUSH = "mobile_push"


class NotificationPriority(Enum):
    """Notification priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class MessageFormat(Enum):
    """Message format types."""
    PLAIN_TEXT = "plain_text"
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    XML = "xml"


@dataclass
class NotificationTemplate:
    """Notification template definition."""
    name: str
    channel: NotificationChannel
    format: MessageFormat
    subject_template: Optional[str] = None
    body_template: str = ""
    variables: Dict[str, Any] = field(default_factory=dict)
    filters: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class NotificationContext:
    """Context data for notification generation."""
    recipient: str
    channel: NotificationChannel
    priority: NotificationPriority = NotificationPriority.NORMAL
    locale: str = "en_US"
    timezone: str = "UTC"
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NotificationMessage:
    """Generated notification message."""
    recipient: str
    channel: NotificationChannel
    priority: NotificationPriority
    subject: Optional[str] = None
    body: str = ""
    format: MessageFormat = MessageFormat.PLAIN_TEXT
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None


class NotificationBuilder:
    """
    Advanced notification builder with template support.
    
    Features:
    - Jinja2-based templating
    - Multi-format message generation
    - Dynamic content and variables
    - Template validation and testing
    - Caching and performance optimization
    - Internationalization support
    """
    
    def __init__(self,
                 template_directory: Optional[str] = None,
                 enable_caching: bool = True,
                 default_locale: str = "en_US",
                 default_timezone: str = "UTC"):
        """
        Initialize notification builder.
        
        Args:
            template_directory: Directory containing template files
            enable_caching: Enable template caching
            default_locale: Default locale for messages
            default_timezone: Default timezone for timestamps
        """
        self.template_directory = template_directory
        self.enable_caching = enable_caching
        self.default_locale = default_locale
        self.default_timezone = default_timezone
        
        # Template storage
        self.templates: Dict[str, NotificationTemplate] = {}
        self.template_cache: Dict[str, Template] = {}
        
        # Jinja2 environment
        self.jinja_env = Environment(
            loader=DictLoader({}),
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Register built-in filters and functions
        self._register_builtin_filters()
        
        # Channel-specific formatters
        self.channel_formatters: Dict[NotificationChannel, Callable] = {}
        self._register_channel_formatters()
        
        # Localization support
        self.localization_data: Dict[str, Dict[str, str]] = {}
        self._load_default_localization()
        
        logger.info("Notification builder initialized")
    
    def _register_builtin_filters(self):
        """Register built-in Jinja2 filters."""
        def format_datetime(value, format_string="%Y-%m-%d %H:%M:%S"):
            """Format datetime values."""
            if isinstance(value, str):
                try:
                    value = datetime.fromisoformat(value.replace('Z', '+00:00'))
                except ValueError:
                    return value
            
            if isinstance(value, datetime):
                return value.strftime(format_string)
            
            return str(value)
        
        def format_duration(seconds):
            """Format duration in seconds to human-readable format."""
            if not isinstance(seconds, (int, float)):
                return str(seconds)
            
            hours, remainder = divmod(int(seconds), 3600)
            minutes, seconds = divmod(remainder, 60)
            
            if hours > 0:
                return f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                return f"{minutes}m {seconds}s"
            else:
                return f"{seconds}s"
        
        def format_number(value, decimals=2):
            """Format numeric values."""
            try:
                return f"{float(value):.{decimals}f}"
            except (ValueError, TypeError):
                return str(value)
        
        def truncate_text(text, max_length=100, suffix="..."):
            """Truncate text to maximum length."""
            text = str(text)
            if len(text) <= max_length:
                return text
            return text[:max_length - len(suffix)] + suffix
        
        def escape_html(text):
            """Escape HTML entities."""
            return html.escape(str(text))
        
        def markdown_to_html(text):
            """Convert Markdown to HTML."""
            return markdown.markdown(str(text))
        
        # Register filters
        self.jinja_env.filters.update({
            'datetime': format_datetime,
            'duration': format_duration,
            'number': format_number,
            'truncate': truncate_text,
            'escape_html': escape_html,
            'markdown': markdown_to_html
        })
        
        # Register global functions
        def now():
            """Get current datetime."""
            return datetime.utcnow()
        
        def localize(key, locale=None):
            """Get localized string."""
            locale = locale or self.default_locale
            return self.localization_data.get(locale, {}).get(key, key)
        
        self.jinja_env.globals.update({
            'now': now,
            'localize': localize
        })
    
    def _register_channel_formatters(self):
        """Register channel-specific formatters."""
        def format_for_slack(message: NotificationMessage) -> NotificationMessage:
            """Format message for Slack."""
            # Convert to Slack markdown format
            body = message.body
            
            # Convert common markdown to Slack format
            body = re.sub(r'\*\*(.*?)\*\*', r'*\1*', body)  # Bold
            body = re.sub(r'__(.*?)__', r'_\1_', body)      # Italic
            body = re.sub(r'`(.*?)`', r'`\1`', body)        # Code
            
            message.body = body
            message.format = MessageFormat.MARKDOWN
            return message
        
        def format_for_email(message: NotificationMessage) -> NotificationMessage:
            """Format message for email."""
            if message.format == MessageFormat.MARKDOWN:
                # Convert markdown to HTML for email
                message.body = markdown.markdown(message.body)
                message.format = MessageFormat.HTML
            
            return message
        
        def format_for_sms(message: NotificationMessage) -> NotificationMessage:
            """Format message for SMS."""
            # Strip formatting and truncate for SMS
            body = re.sub(r'[*_`#]', '', message.body)  # Remove markdown
            body = re.sub(r'<[^>]+>', '', body)         # Remove HTML tags
            body = re.sub(r'\s+', ' ', body).strip()    # Normalize whitespace
            
            # Truncate to SMS length limit
            if len(body) > 160:
                body = body[:157] + "..."
            
            message.body = body
            message.format = MessageFormat.PLAIN_TEXT
            return message
        
        def format_for_pagerduty(message: NotificationMessage) -> NotificationMessage:
            """Format message for PagerDuty."""
            # Ensure plain text format for PagerDuty
            if message.format != MessageFormat.PLAIN_TEXT:
                body = re.sub(r'<[^>]+>', '', message.body)  # Remove HTML
                body = re.sub(r'[*_`#]', '', body)           # Remove markdown
                message.body = body
                message.format = MessageFormat.PLAIN_TEXT
            
            return message
        
        self.channel_formatters.update({
            NotificationChannel.SLACK: format_for_slack,
            NotificationChannel.EMAIL: format_for_email,
            NotificationChannel.SMS: format_for_sms,
            NotificationChannel.PAGERDUTY: format_for_pagerduty
        })
    
    def _load_default_localization(self):
        """Load default localization strings."""
        self.localization_data.update({
            'en_US': {
                'incident_created': 'Incident Created',
                'incident_resolved': 'Incident Resolved',
                'alert_triggered': 'Alert Triggered',
                'service_down': 'Service Down',
                'high_priority': 'High Priority',
                'critical_priority': 'Critical Priority',
                'click_here': 'Click here',
                'view_details': 'View Details',
                'acknowledge': 'Acknowledge',
                'resolve': 'Resolve'
            },
            'fr_FR': {
                'incident_created': 'Incident Créé',
                'incident_resolved': 'Incident Résolu',
                'alert_triggered': 'Alerte Déclenchée',
                'service_down': 'Service Indisponible',
                'high_priority': 'Priorité Élevée',
                'critical_priority': 'Priorité Critique',
                'click_here': 'Cliquez ici',
                'view_details': 'Voir les Détails',
                'acknowledge': 'Acquitter',
                'resolve': 'Résoudre'
            },
            'de_DE': {
                'incident_created': 'Vorfall Erstellt',
                'incident_resolved': 'Vorfall Gelöst',
                'alert_triggered': 'Alarm Ausgelöst',
                'service_down': 'Service Nicht Verfügbar',
                'high_priority': 'Hohe Priorität',
                'critical_priority': 'Kritische Priorität',
                'click_here': 'Hier klicken',
                'view_details': 'Details Anzeigen',
                'acknowledge': 'Bestätigen',
                'resolve': 'Lösen'
            }
        })
    
    def register_template(self, template: NotificationTemplate):
        """Register a notification template."""
        self.templates[template.name] = template
        
        # Update Jinja2 loader
        template_dict = {}
        for name, tmpl in self.templates.items():
            if tmpl.subject_template:
                template_dict[f"{name}_subject"] = tmpl.subject_template
            template_dict[f"{name}_body"] = tmpl.body_template
        
        self.jinja_env.loader = DictLoader(template_dict)
        
        # Clear cache
        self.template_cache.clear()
        
        logger.debug(f"Registered notification template: {template.name}")
    
    def register_channel_formatter(self, channel: NotificationChannel, formatter: Callable):
        """Register custom channel formatter."""
        self.channel_formatters[channel] = formatter
        logger.debug(f"Registered channel formatter: {channel.value}")
    
    def add_localization(self, locale: str, translations: Dict[str, str]):
        """Add localization translations."""
        if locale not in self.localization_data:
            self.localization_data[locale] = {}
        
        self.localization_data[locale].update(translations)
        logger.debug(f"Added localization for locale: {locale}")
    
    def build_notification(self, 
                          template_name: str, 
                          context: NotificationContext) -> NotificationMessage:
        """
        Build notification from template and context.
        
        Args:
            template_name: Name of template to use
            context: Notification context with variables
            
        Returns:
            Generated notification message
        """
        try:
            # Get template
            if template_name not in self.templates:
                raise TemplateError(f"Template not found: {template_name}")
            
            template = self.templates[template_name]
            
            # Merge variables
            variables = {
                **template.variables,
                **context.variables,
                'recipient': context.recipient,
                'channel': context.channel.value,
                'priority': context.priority.value,
                'locale': context.locale,
                'timezone': context.timezone
            }
            
            # Render subject
            subject = None
            if template.subject_template:
                subject_tmpl = self._get_jinja_template(f"{template_name}_subject")
                subject = subject_tmpl.render(**variables).strip()
            
            # Render body
            body_tmpl = self._get_jinja_template(f"{template_name}_body")
            body = body_tmpl.render(**variables).strip()
            
            # Create message
            message = NotificationMessage(
                recipient=context.recipient,
                channel=context.channel,
                priority=context.priority,
                subject=subject,
                body=body,
                format=template.format,
                metadata={
                    **template.metadata,
                    **context.metadata,
                    'template_name': template_name,
                    'locale': context.locale
                }
            )
            
            # Apply channel-specific formatting
            if context.channel in self.channel_formatters:
                message = self.channel_formatters[context.channel](message)
            
            # Set expiration if specified
            if template.metadata.get('expires_after_hours'):
                hours = int(template.metadata['expires_after_hours'])
                message.expires_at = datetime.utcnow() + timedelta(hours=hours)
            
            logger.debug(f"Built notification for template: {template_name}")
            return message
            
        except Exception as e:
            logger.error(f"Failed to build notification: {e}")
            raise NotificationError(f"Failed to build notification: {e}")
    
    def _get_jinja_template(self, template_name: str) -> Template:
        """Get Jinja2 template with caching."""
        if not self.enable_caching or template_name not in self.template_cache:
            try:
                template = self.jinja_env.get_template(template_name)
                if self.enable_caching:
                    self.template_cache[template_name] = template
                return template
            except TemplateError as e:
                raise TemplateError(f"Template error for '{template_name}': {e}")
        
        return self.template_cache[template_name]
    
    def validate_template(self, template: NotificationTemplate) -> List[str]:
        """
        Validate template syntax and variables.
        
        Args:
            template: Template to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        try:
            # Test subject template
            if template.subject_template:
                try:
                    Template(template.subject_template)
                except TemplateError as e:
                    errors.append(f"Subject template error: {e}")
            
            # Test body template
            try:
                Template(template.body_template)
            except TemplateError as e:
                errors.append(f"Body template error: {e}")
            
            # Test rendering with sample data
            test_variables = {
                'test_var': 'test_value',
                'timestamp': datetime.utcnow(),
                'number': 42,
                'bool_value': True
            }
            
            if template.subject_template:
                try:
                    Template(template.subject_template).render(**test_variables)
                except Exception as e:
                    errors.append(f"Subject rendering error: {e}")
            
            try:
                Template(template.body_template).render(**test_variables)
            except Exception as e:
                errors.append(f"Body rendering error: {e}")
                
        except Exception as e:
            errors.append(f"Validation error: {e}")
        
        return errors
    
    def test_template(self, 
                     template_name: str, 
                     test_variables: Dict[str, Any]) -> NotificationMessage:
        """
        Test template with provided variables.
        
        Args:
            template_name: Name of template to test
            test_variables: Variables for testing
            
        Returns:
            Generated test message
        """
        if template_name not in self.templates:
            raise TemplateError(f"Template not found: {template_name}")
        
        template = self.templates[template_name]
        
        # Create test context
        context = NotificationContext(
            recipient="test@example.com",
            channel=template.channel,
            priority=NotificationPriority.NORMAL,
            variables=test_variables
        )
        
        return self.build_notification(template_name, context)
    
    def get_template_variables(self, template_name: str) -> List[str]:
        """
        Extract variables used in template.
        
        Args:
            template_name: Name of template to analyze
            
        Returns:
            List of variable names used in template
        """
        if template_name not in self.templates:
            raise TemplateError(f"Template not found: {template_name}")
        
        template = self.templates[template_name]
        variables = set()
        
        # Parse templates to find variables
        from jinja2 import meta
        
        if template.subject_template:
            ast = self.jinja_env.parse(template.subject_template)
            variables.update(meta.find_undeclared_variables(ast))
        
        ast = self.jinja_env.parse(template.body_template)
        variables.update(meta.find_undeclared_variables(ast))
        
        # Remove built-in variables
        builtin_vars = {'now', 'localize', 'recipient', 'channel', 'priority', 'locale', 'timezone'}
        variables -= builtin_vars
        
        return sorted(list(variables))
    
    def create_default_templates(self):
        """Create default notification templates."""
        # PagerDuty incident template
        pagerduty_incident_template = NotificationTemplate(
            name="pagerduty_incident",
            channel=NotificationChannel.PAGERDUTY,
            format=MessageFormat.PLAIN_TEXT,
            subject_template="{{ localize('incident_created') }}: {{ title }}",
            body_template="""
{{ localize('incident_created') }}: {{ title }}

Service: {{ service_name }}
Severity: {{ severity }}
Priority: {{ priority }}
Created: {{ created_at | datetime }}

Description:
{{ description }}

{% if url %}
{{ localize('view_details') }}: {{ url }}
{% endif %}
            """.strip(),
            variables={
                'title': 'Sample Incident',
                'service_name': 'Sample Service',
                'severity': 'high',
                'description': 'Sample incident description',
                'url': 'https://example.com/incident'
            }
        )
        
        # Email alert template
        email_alert_template = NotificationTemplate(
            name="email_alert",
            channel=NotificationChannel.EMAIL,
            format=MessageFormat.HTML,
            subject_template="{{ localize('alert_triggered') }}: {{ alert_name }}",
            body_template="""
<h2>{{ localize('alert_triggered') }}</h2>

<p><strong>Alert:</strong> {{ alert_name }}</p>
<p><strong>Severity:</strong> <span style="color: {% if severity == 'critical' %}red{% elif severity == 'high' %}orange{% else %}black{% endif %}">{{ severity | title }}</span></p>
<p><strong>Source:</strong> {{ source }}</p>
<p><strong>Time:</strong> {{ timestamp | datetime }}</p>

<h3>Details</h3>
<p>{{ message | markdown }}</p>

{% if actions %}
<h3>Actions</h3>
<ul>
{% for action in actions %}
    <li><a href="{{ action.url }}">{{ action.label }}</a></li>
{% endfor %}
</ul>
{% endif %}

<hr>
<p><small>This is an automated alert from Spotify AI Agent monitoring system.</small></p>
            """.strip(),
            variables={
                'alert_name': 'Sample Alert',
                'severity': 'high',
                'source': 'monitoring',
                'message': 'Sample alert message',
                'actions': [
                    {'label': 'Acknowledge', 'url': 'https://example.com/ack'},
                    {'label': 'View Dashboard', 'url': 'https://example.com/dashboard'}
                ]
            }
        )
        
        # Slack notification template
        slack_notification_template = NotificationTemplate(
            name="slack_notification",
            channel=NotificationChannel.SLACK,
            format=MessageFormat.MARKDOWN,
            body_template="""
:warning: *{{ localize('alert_triggered') }}*

*Alert:* {{ alert_name }}
*Severity:* {{ severity | title }}
*Source:* {{ source }}
*Time:* {{ timestamp | datetime }}

*Details:*
{{ message | truncate(200) }}

{% if url %}
<{{ url }}|{{ localize('view_details') }}>
{% endif %}
            """.strip(),
            variables={
                'alert_name': 'Sample Alert',
                'severity': 'high',
                'source': 'monitoring',
                'message': 'Sample alert message',
                'url': 'https://example.com/alert'
            }
        )
        
        # Register templates
        self.register_template(pagerduty_incident_template)
        self.register_template(email_alert_template)
        self.register_template(slack_notification_template)
        
        logger.info("Created default notification templates")
    
    def get_available_templates(self) -> List[str]:
        """Get list of available template names."""
        return list(self.templates.keys())
    
    def get_template_info(self, template_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific template."""
        if template_name not in self.templates:
            return None
        
        template = self.templates[template_name]
        
        return {
            'name': template.name,
            'channel': template.channel.value,
            'format': template.format.value,
            'description': template.description,
            'variables': self.get_template_variables(template_name),
            'metadata': template.metadata
        }
    
    async def build_notification_async(self, *args, **kwargs) -> NotificationMessage:
        """Async version of build_notification."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.build_notification, *args, **kwargs)


# Global notification builder instance
_global_notification_builder = None

def get_notification_builder() -> NotificationBuilder:
    """Get global notification builder instance."""
    global _global_notification_builder
    if _global_notification_builder is None:
        _global_notification_builder = NotificationBuilder()
        _global_notification_builder.create_default_templates()
    return _global_notification_builder


# Convenience functions
def build_pagerduty_notification(incident_data: Dict[str, Any], 
                                recipient: str) -> NotificationMessage:
    """Build PagerDuty incident notification."""
    builder = get_notification_builder()
    context = NotificationContext(
        recipient=recipient,
        channel=NotificationChannel.PAGERDUTY,
        priority=NotificationPriority.HIGH,
        variables=incident_data
    )
    return builder.build_notification("pagerduty_incident", context)


def build_email_alert(alert_data: Dict[str, Any], 
                     recipient: str) -> NotificationMessage:
    """Build email alert notification."""
    builder = get_notification_builder()
    context = NotificationContext(
        recipient=recipient,
        channel=NotificationChannel.EMAIL,
        priority=NotificationPriority.NORMAL,
        variables=alert_data
    )
    return builder.build_notification("email_alert", context)
