"""
Alert Templates System
=====================

Ce module implémente le système de templates pour les alertes de sécurité
du Spotify AI Agent multi-tenant.

Auteur: Fahed Mlaiel
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from jinja2 import Environment, BaseLoader, select_autoescape, Template
from jinja2.exceptions import TemplateError
import yaml
import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession
from .core import SecurityLevel, SecurityEvent
from .processors import AlertContext

logger = logging.getLogger(__name__)


class TemplateType(Enum):
    """Types de templates disponibles"""
    SLACK = "slack"
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    TEAMS = "teams"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    CUSTOM = "custom"


class TemplateFormat(Enum):
    """Formats de templates"""
    JINJA2 = "jinja2"
    MUSTACHE = "mustache"
    PLAIN_TEXT = "plain_text"
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"


class LocaleCode(Enum):
    """Codes de locales supportés"""
    EN = "en"
    FR = "fr"
    DE = "de"
    ES = "es"
    IT = "it"
    PT = "pt"
    NL = "nl"
    SV = "sv"
    NO = "no"
    DA = "da"
    FI = "fi"


@dataclass
class TemplateVariable:
    """Variable de template"""
    name: str
    type: str
    description: str
    required: bool = True
    default_value: Any = None
    validation_regex: Optional[str] = None
    
    def validate(self, value: Any) -> bool:
        """Valide une valeur pour cette variable"""
        if self.required and value is None:
            return False
        
        if self.validation_regex and isinstance(value, str):
            return bool(re.match(self.validation_regex, value))
        
        return True


@dataclass
class TemplateMetadata:
    """Métadonnées de template"""
    name: str
    description: str
    version: str
    author: str
    created_at: datetime
    updated_at: datetime
    tags: List[str] = field(default_factory=list)
    category: str = ""
    
    # Variables du template
    variables: List[TemplateVariable] = field(default_factory=list)
    
    # Configuration
    supports_html: bool = False
    supports_markdown: bool = False
    max_length: Optional[int] = None
    min_severity: SecurityLevel = SecurityLevel.LOW


@dataclass
class AlertTemplate:
    """Template d'alerte"""
    template_id: str
    template_type: TemplateType
    template_format: TemplateFormat
    locale: LocaleCode
    
    # Contenu du template
    subject_template: str = ""
    body_template: str = ""
    html_template: str = ""
    
    # Métadonnées
    metadata: TemplateMetadata = field(default_factory=lambda: TemplateMetadata(
        name="", description="", version="1.0.0", author="",
        created_at=datetime.utcnow(), updated_at=datetime.utcnow()
    ))
    
    # Configuration
    enabled: bool = True
    tenant_specific: bool = False
    tenant_id: Optional[str] = None
    
    # Conditions d'utilisation
    severity_filter: List[SecurityLevel] = field(default_factory=list)
    event_type_filter: List[str] = field(default_factory=list)
    
    # Template compilé (cache)
    _compiled_subject: Optional[Template] = None
    _compiled_body: Optional[Template] = None
    _compiled_html: Optional[Template] = None


class SlackTemplateBuilder:
    """
    Constructeur de templates Slack avancés
    """
    
    def __init__(self):
        self.blocks = []
        self.attachments = []
        
    def add_header(self, text: str, emoji: str = "🔒") -> 'SlackTemplateBuilder':
        """Ajoute un header à l'alerte"""
        self.blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} {text}",
                "emoji": True
            }
        })
        return self
    
    def add_section(self, text: str, fields: List[Dict[str, str]] = None) -> 'SlackTemplateBuilder':
        """Ajoute une section avec texte et champs"""
        block = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": text
            }
        }
        
        if fields:
            block["fields"] = [
                {
                    "type": "mrkdwn",
                    "text": f"*{field['title']}*\n{field['value']}"
                }
                for field in fields
            ]
        
        self.blocks.append(block)
        return self
    
    def add_context(self, elements: List[str]) -> 'SlackTemplateBuilder':
        """Ajoute un bloc de contexte"""
        self.blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": element
                }
                for element in elements
            ]
        })
        return self
    
    def add_divider(self) -> 'SlackTemplateBuilder':
        """Ajoute un séparateur"""
        self.blocks.append({"type": "divider"})
        return self
    
    def add_actions(self, actions: List[Dict[str, Any]]) -> 'SlackTemplateBuilder':
        """Ajoute des boutons d'action"""
        self.blocks.append({
            "type": "actions",
            "elements": actions
        })
        return self
    
    def build(self) -> Dict[str, Any]:
        """Construit le message Slack final"""
        return {
            "blocks": self.blocks,
            "attachments": self.attachments
        }


class EmailTemplateBuilder:
    """
    Constructeur de templates Email avec HTML et texte
    """
    
    def __init__(self):
        self.html_parts = []
        self.text_parts = []
        self.styles = {
            "body": "font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5;",
            "container": "max-width: 600px; margin: 0 auto; background-color: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1);",
            "header": "background-color: #ff4444; color: white; padding: 20px; text-align: center;",
            "content": "padding: 20px;",
            "alert": "background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 4px; padding: 15px; margin: 10px 0;",
            "footer": "background-color: #f8f9fa; padding: 15px; text-align: center; font-size: 12px; color: #666;"
        }
        
    def add_header(self, title: str, subtitle: str = "") -> 'EmailTemplateBuilder':
        """Ajoute un header email"""
        html = f"""
        <div style="{self.styles['header']}">
            <h1 style="margin: 0; font-size: 24px;">{title}</h1>
            {f'<p style="margin: 10px 0 0 0; opacity: 0.9;">{subtitle}</p>' if subtitle else ''}
        </div>
        """
        
        text = f"{title}\n{'=' * len(title)}\n"
        if subtitle:
            text += f"{subtitle}\n"
        text += "\n"
        
        self.html_parts.append(html)
        self.text_parts.append(text)
        return self
    
    def add_alert_box(self, severity: SecurityLevel, message: str) -> 'EmailTemplateBuilder':
        """Ajoute une boîte d'alerte colorée"""
        severity_colors = {
            SecurityLevel.LOW: "#d1ecf1",
            SecurityLevel.MEDIUM: "#fff3cd", 
            SecurityLevel.HIGH: "#f8d7da",
            SecurityLevel.CRITICAL: "#f5c6cb"
        }
        
        severity_icons = {
            SecurityLevel.LOW: "ℹ️",
            SecurityLevel.MEDIUM: "⚠️",
            SecurityLevel.HIGH: "🚨",
            SecurityLevel.CRITICAL: "🔥"
        }
        
        color = severity_colors.get(severity, "#fff3cd")
        icon = severity_icons.get(severity, "⚠️")
        
        html = f"""
        <div style="background-color: {color}; border-radius: 4px; padding: 15px; margin: 15px 0;">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 24px; margin-right: 10px;">{icon}</span>
                <div>
                    <strong>Security Alert - {severity.value.upper()}</strong>
                    <p style="margin: 5px 0 0 0;">{message}</p>
                </div>
            </div>
        </div>
        """
        
        text = f"\n{icon} SECURITY ALERT - {severity.value.upper()}\n{message}\n\n"
        
        self.html_parts.append(html)
        self.text_parts.append(text)
        return self
    
    def add_details_table(self, details: Dict[str, str]) -> 'EmailTemplateBuilder':
        """Ajoute un tableau de détails"""
        html = """
        <table style="width: 100%; border-collapse: collapse; margin: 15px 0;">
            <thead>
                <tr style="background-color: #f8f9fa;">
                    <th style="padding: 10px; text-align: left; border: 1px solid #dee2e6;">Field</th>
                    <th style="padding: 10px; text-align: left; border: 1px solid #dee2e6;">Value</th>
                </tr>
            </thead>
            <tbody>
        """
        
        text = "\nDetails:\n" + "-" * 50 + "\n"
        
        for key, value in details.items():
            html += f"""
                <tr>
                    <td style="padding: 8px; border: 1px solid #dee2e6; font-weight: bold;">{key}</td>
                    <td style="padding: 8px; border: 1px solid #dee2e6;">{value}</td>
                </tr>
            """
            text += f"{key}: {value}\n"
        
        html += """
            </tbody>
        </table>
        """
        text += "\n"
        
        self.html_parts.append(html)
        self.text_parts.append(text)
        return self
    
    def add_action_buttons(self, buttons: List[Dict[str, str]]) -> 'EmailTemplateBuilder':
        """Ajoute des boutons d'action"""
        html = '<div style="text-align: center; margin: 20px 0;">'
        
        for button in buttons:
            color = button.get('color', '#007bff')
            html += f"""
            <a href="{button['url']}" style="display: inline-block; margin: 5px; padding: 10px 20px; 
               background-color: {color}; color: white; text-decoration: none; border-radius: 4px;">
               {button['text']}
            </a>
            """
        
        html += '</div>'
        
        text = "\nActions:\n"
        for button in buttons:
            text += f"- {button['text']}: {button['url']}\n"
        text += "\n"
        
        self.html_parts.append(html)
        self.text_parts.append(text)
        return self
    
    def add_footer(self, footer_text: str) -> 'EmailTemplateBuilder':
        """Ajoute un footer"""
        html = f"""
        <div style="{self.styles['footer']}">
            {footer_text}
        </div>
        """
        
        text = f"\n{'-' * 50}\n{footer_text}\n"
        
        self.html_parts.append(html)
        self.text_parts.append(text)
        return self
    
    def build(self) -> Dict[str, str]:
        """Construit l'email final"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Security Alert</title>
        </head>
        <body style="{self.styles['body']}">
            <div style="{self.styles['container']}">
                <div style="{self.styles['content']}">
                    {''.join(self.html_parts)}
                </div>
            </div>
        </body>
        </html>
        """
        
        text_content = ''.join(self.text_parts)
        
        return {
            "html": html_content,
            "text": text_content
        }


class TemplateEngine:
    """
    Moteur de templates principal avec support Jinja2
    """
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.jinja_env = Environment(
            loader=BaseLoader(),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Fonctions personnalisées pour templates
        self.jinja_env.globals.update({
            'format_datetime': self._format_datetime,
            'severity_emoji': self._get_severity_emoji,
            'severity_color': self._get_severity_color,
            'truncate_text': self._truncate_text,
            'capitalize_words': self._capitalize_words
        })
        
        # Cache des templates compilés
        self.template_cache = {}
        
    def _format_datetime(self, dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S UTC") -> str:
        """Formate une datetime"""
        return dt.strftime(format_str)
    
    def _get_severity_emoji(self, severity: SecurityLevel) -> str:
        """Retourne l'emoji pour une sévérité"""
        emojis = {
            SecurityLevel.LOW: "ℹ️",
            SecurityLevel.MEDIUM: "⚠️",
            SecurityLevel.HIGH: "🚨",
            SecurityLevel.CRITICAL: "🔥"
        }
        return emojis.get(severity, "⚠️")
    
    def _get_severity_color(self, severity: SecurityLevel) -> str:
        """Retourne la couleur pour une sévérité"""
        colors = {
            SecurityLevel.LOW: "#36a64f",
            SecurityLevel.MEDIUM: "#ff9f00",
            SecurityLevel.HIGH: "#ff4444",
            SecurityLevel.CRITICAL: "#8b0000"
        }
        return colors.get(severity, "#ff9f00")
    
    def _truncate_text(self, text: str, max_length: int = 100, suffix: str = "...") -> str:
        """Tronque un texte"""
        if len(text) <= max_length:
            return text
        return text[:max_length - len(suffix)] + suffix
    
    def _capitalize_words(self, text: str) -> str:
        """Met en forme title case"""
        return text.title()
    
    async def compile_template(self, template: AlertTemplate) -> AlertTemplate:
        """Compile les templates Jinja2"""
        try:
            if template.subject_template:
                template._compiled_subject = self.jinja_env.from_string(template.subject_template)
            
            if template.body_template:
                template._compiled_body = self.jinja_env.from_string(template.body_template)
            
            if template.html_template:
                template._compiled_html = self.jinja_env.from_string(template.html_template)
            
            return template
            
        except TemplateError as e:
            logger.error(f"Template compilation error for {template.template_id}: {e}")
            raise
    
    async def render_template(self, template: AlertTemplate, context: Dict[str, Any]) -> Dict[str, str]:
        """Rend un template avec le contexte fourni"""
        if not any([template._compiled_subject, template._compiled_body, template._compiled_html]):
            template = await self.compile_template(template)
        
        result = {}
        
        try:
            if template._compiled_subject:
                result['subject'] = template._compiled_subject.render(**context)
            
            if template._compiled_body:
                result['body'] = template._compiled_body.render(**context)
            
            if template._compiled_html:
                result['html'] = template._compiled_html.render(**context)
            
            return result
            
        except TemplateError as e:
            logger.error(f"Template rendering error for {template.template_id}: {e}")
            raise
    
    async def validate_template_context(self, template: AlertTemplate, context: Dict[str, Any]) -> List[str]:
        """Valide le contexte pour un template"""
        errors = []
        
        for variable in template.metadata.variables:
            value = context.get(variable.name)
            
            if not variable.validate(value):
                if variable.required and value is None:
                    errors.append(f"Required variable '{variable.name}' is missing")
                elif variable.validation_regex and isinstance(value, str):
                    errors.append(f"Variable '{variable.name}' does not match pattern: {variable.validation_regex}")
        
        return errors


class TemplateManager:
    """
    Gestionnaire de templates avec support multi-locale et tenant
    """
    
    def __init__(self, redis_client: aioredis.Redis, db_session: AsyncSession):
        self.redis = redis_client
        self.db = db_session
        self.template_engine = TemplateEngine(redis_client)
        self.templates: Dict[str, AlertTemplate] = {}
        
        # Templates par défaut
        self.default_templates = {}
        
    async def initialize(self):
        """Initialise le gestionnaire de templates"""
        await self._load_default_templates()
        await self._load_custom_templates()
        logger.info("TemplateManager initialized")
    
    async def _load_default_templates(self):
        """Charge les templates par défaut"""
        # Template Slack par défaut
        slack_template = AlertTemplate(
            template_id="default_slack_en",
            template_type=TemplateType.SLACK,
            template_format=TemplateFormat.JSON,
            locale=LocaleCode.EN,
            subject_template="🔒 Security Alert: {{ alert.title }}",
            body_template="""
{
    "text": "{{ severity_emoji(alert.severity) }} Security Alert: {{ alert.title }}",
    "blocks": [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "{{ severity_emoji(alert.severity) }} Security Alert",
                "emoji": true
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*{{ alert.title }}*\\n{{ alert.message }}"
            },
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": "*Severity:*\\n{{ alert.severity.value|upper }}"
                },
                {
                    "type": "mrkdwn",
                    "text": "*Tenant:*\\n{{ alert.tenant_id }}"
                },
                {
                    "type": "mrkdwn",
                    "text": "*Event ID:*\\n{{ alert.event_id }}"
                },
                {
                    "type": "mrkdwn",
                    "text": "*Time:*\\n{{ format_datetime(alert.created_at) }}"
                }
            ]
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": "Achiri Security System"
                }
            ]
        }
    ]
}
            """,
            metadata=TemplateMetadata(
                name="Default Slack Template",
                description="Template Slack par défaut pour alertes de sécurité",
                version="1.0.0",
                author="Fahed Mlaiel",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                variables=[
                    TemplateVariable("alert", "AlertContext", "Contexte de l'alerte"),
                ]
            )
        )
        
        # Template Email par défaut
        email_template = AlertTemplate(
            template_id="default_email_en",
            template_type=TemplateType.EMAIL,
            template_format=TemplateFormat.HTML,
            locale=LocaleCode.EN,
            subject_template="🔒 Security Alert: {{ alert.title }}",
            body_template="""
Security Alert: {{ alert.title }}

Severity: {{ alert.severity.value|upper }}
Tenant: {{ alert.tenant_id }}
Event ID: {{ alert.event_id }}
Timestamp: {{ format_datetime(alert.created_at) }}

Details:
{{ alert.message }}

{% if alert.details %}
Additional Information:
{% for key, value in alert.details.items() %}
- {{ key }}: {{ value }}
{% endfor %}
{% endif %}

--
Achiri Security System
Generated at {{ format_datetime(now()) }}
            """,
            html_template="""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Security Alert</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 600px; margin: 0 auto; background-color: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .header { background-color: {{ severity_color(alert.severity) }}; color: white; padding: 20px; text-align: center; }
        .content { padding: 20px; }
        .alert-box { background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 4px; padding: 15px; margin: 15px 0; }
        .details-table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        .details-table th, .details-table td { padding: 8px; border: 1px solid #dee2e6; text-align: left; }
        .details-table th { background-color: #f8f9fa; font-weight: bold; }
        .footer { background-color: #f8f9fa; padding: 15px; text-align: center; font-size: 12px; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ severity_emoji(alert.severity) }} Security Alert</h1>
            <p>{{ alert.title }}</p>
        </div>
        <div class="content">
            <div class="alert-box">
                <strong>{{ alert.severity.value|upper }} SEVERITY ALERT</strong>
                <p>{{ alert.message }}</p>
            </div>
            
            <table class="details-table">
                <tr>
                    <th>Field</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Severity</td>
                    <td>{{ alert.severity.value|upper }}</td>
                </tr>
                <tr>
                    <td>Tenant</td>
                    <td>{{ alert.tenant_id }}</td>
                </tr>
                <tr>
                    <td>Event ID</td>
                    <td>{{ alert.event_id }}</td>
                </tr>
                <tr>
                    <td>Timestamp</td>
                    <td>{{ format_datetime(alert.created_at) }}</td>
                </tr>
            </table>
            
            {% if alert.details %}
            <h3>Additional Details</h3>
            <ul>
            {% for key, value in alert.details.items() %}
                <li><strong>{{ key }}:</strong> {{ value }}</li>
            {% endfor %}
            </ul>
            {% endif %}
        </div>
        <div class="footer">
            Achiri Security System<br>
            Generated at {{ format_datetime(now()) }}
        </div>
    </div>
</body>
</html>
            """,
            metadata=TemplateMetadata(
                name="Default Email Template",
                description="Template Email par défaut pour alertes de sécurité",
                version="1.0.0",
                author="Fahed Mlaiel",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                supports_html=True,
                variables=[
                    TemplateVariable("alert", "AlertContext", "Contexte de l'alerte"),
                ]
            )
        )
        
        # Template Webhook par défaut
        webhook_template = AlertTemplate(
            template_id="default_webhook_en",
            template_type=TemplateType.WEBHOOK,
            template_format=TemplateFormat.JSON,
            locale=LocaleCode.EN,
            body_template="""
{
    "alert_id": "{{ alert.alert_id }}",
    "event_id": "{{ alert.event_id }}",
    "tenant_id": "{{ alert.tenant_id }}",
    "severity": "{{ alert.severity.value }}",
    "title": "{{ alert.title }}",
    "message": "{{ alert.message }}",
    "timestamp": "{{ alert.created_at.isoformat() }}",
    "details": {{ alert.details|tojson if alert.details else '{}' }},
    "metadata": {
        "source": "achiri_security_system",
        "version": "1.0.0",
        "generated_at": "{{ now().isoformat() }}"
    }
}
            """,
            metadata=TemplateMetadata(
                name="Default Webhook Template",
                description="Template Webhook par défaut pour alertes de sécurité",
                version="1.0.0",
                author="Fahed Mlaiel",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                variables=[
                    TemplateVariable("alert", "AlertContext", "Contexte de l'alerte"),
                ]
            )
        )
        
        # Stockage des templates par défaut
        self.default_templates = {
            "default_slack_en": slack_template,
            "default_email_en": email_template,
            "default_webhook_en": webhook_template
        }
        
        # Compilation des templates
        for template in self.default_templates.values():
            await self.template_engine.compile_template(template)
    
    async def _load_custom_templates(self):
        """Charge les templates personnalisés depuis Redis"""
        templates_key = "security:templates"
        template_ids = await self.redis.smembers(templates_key)
        
        for template_id in template_ids:
            template_data_key = f"security:template:{template_id.decode()}"
            template_data = await self.redis.get(template_data_key)
            
            if template_data:
                try:
                    data = json.loads(template_data)
                    template = AlertTemplate(**data)
                    await self.template_engine.compile_template(template)
                    self.templates[template.template_id] = template
                    
                except Exception as e:
                    logger.error(f"Error loading template {template_id}: {e}")
    
    async def get_template(self, template_type: TemplateType, locale: LocaleCode = LocaleCode.EN, 
                          tenant_id: Optional[str] = None, severity: Optional[SecurityLevel] = None) -> Optional[AlertTemplate]:
        """Récupère le meilleur template pour les critères donnés"""
        
        # Recherche d'un template spécifique au tenant
        if tenant_id:
            tenant_template = await self._find_template(
                template_type=template_type,
                locale=locale,
                tenant_id=tenant_id,
                severity=severity
            )
            if tenant_template:
                return tenant_template
        
        # Recherche d'un template global
        global_template = await self._find_template(
            template_type=template_type,
            locale=locale,
            severity=severity
        )
        if global_template:
            return global_template
        
        # Fallback vers template par défaut
        default_key = f"default_{template_type.value}_{locale.value}"
        if default_key in self.default_templates:
            return self.default_templates[default_key]
        
        # Fallback vers template par défaut en anglais
        fallback_key = f"default_{template_type.value}_en"
        return self.default_templates.get(fallback_key)
    
    async def _find_template(self, template_type: TemplateType, locale: LocaleCode,
                           tenant_id: Optional[str] = None, severity: Optional[SecurityLevel] = None) -> Optional[AlertTemplate]:
        """Trouve un template correspondant aux critères"""
        
        for template in self.templates.values():
            # Vérification du type
            if template.template_type != template_type:
                continue
            
            # Vérification de la locale
            if template.locale != locale:
                continue
            
            # Vérification du tenant
            if tenant_id and template.tenant_specific and template.tenant_id != tenant_id:
                continue
            
            # Vérification de la sévérité
            if severity and template.severity_filter and severity not in template.severity_filter:
                continue
            
            # Vérification de l'activation
            if not template.enabled:
                continue
            
            return template
        
        return None
    
    async def render_alert(self, alert_context: AlertContext, template_type: TemplateType, 
                          locale: LocaleCode = LocaleCode.EN) -> Optional[Dict[str, str]]:
        """Rend une alerte avec le template approprié"""
        
        # Récupération du template
        template = await self.get_template(
            template_type=template_type,
            locale=locale,
            tenant_id=alert_context.tenant_id,
            severity=alert_context.severity
        )
        
        if not template:
            logger.warning(f"No template found for {template_type.value} in {locale.value}")
            return None
        
        # Construction du contexte pour le template
        template_context = {
            'alert': alert_context,
            'now': datetime.utcnow,
            'tenant_id': alert_context.tenant_id,
            'severity': alert_context.severity,
            'locale': locale.value
        }
        
        # Validation du contexte
        validation_errors = await self.template_engine.validate_template_context(template, template_context)
        if validation_errors:
            logger.error(f"Template context validation errors: {validation_errors}")
            return None
        
        # Rendu du template
        try:
            result = await self.template_engine.render_template(template, template_context)
            return result
            
        except Exception as e:
            logger.error(f"Template rendering error: {e}")
            return None
    
    async def create_template(self, template: AlertTemplate) -> bool:
        """Crée un nouveau template"""
        try:
            # Compilation du template
            await self.template_engine.compile_template(template)
            
            # Stockage en Redis
            template_data = {
                "template_id": template.template_id,
                "template_type": template.template_type.value,
                "template_format": template.template_format.value,
                "locale": template.locale.value,
                "subject_template": template.subject_template,
                "body_template": template.body_template,
                "html_template": template.html_template,
                "enabled": template.enabled,
                "tenant_specific": template.tenant_specific,
                "tenant_id": template.tenant_id,
                "severity_filter": [s.value for s in template.severity_filter],
                "event_type_filter": template.event_type_filter,
                "metadata": {
                    "name": template.metadata.name,
                    "description": template.metadata.description,
                    "version": template.metadata.version,
                    "author": template.metadata.author,
                    "created_at": template.metadata.created_at.isoformat(),
                    "updated_at": template.metadata.updated_at.isoformat(),
                    "tags": template.metadata.tags,
                    "category": template.metadata.category,
                    "supports_html": template.metadata.supports_html,
                    "supports_markdown": template.metadata.supports_markdown,
                    "max_length": template.metadata.max_length,
                    "min_severity": template.metadata.min_severity.value
                }
            }
            
            # Sauvegarde
            templates_key = "security:templates"
            template_data_key = f"security:template:{template.template_id}"
            
            await self.redis.sadd(templates_key, template.template_id)
            await self.redis.set(template_data_key, json.dumps(template_data))
            
            # Ajout au cache
            self.templates[template.template_id] = template
            
            logger.info(f"Template {template.template_id} created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating template: {e}")
            return False
    
    async def delete_template(self, template_id: str) -> bool:
        """Supprime un template"""
        try:
            templates_key = "security:templates"
            template_data_key = f"security:template:{template_id}"
            
            await self.redis.srem(templates_key, template_id)
            await self.redis.delete(template_data_key)
            
            if template_id in self.templates:
                del self.templates[template_id]
            
            logger.info(f"Template {template_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting template: {e}")
            return False
    
    async def list_templates(self, template_type: Optional[TemplateType] = None, 
                           locale: Optional[LocaleCode] = None) -> List[AlertTemplate]:
        """Liste les templates disponibles"""
        templates = list(self.templates.values()) + list(self.default_templates.values())
        
        if template_type:
            templates = [t for t in templates if t.template_type == template_type]
        
        if locale:
            templates = [t for t in templates if t.locale == locale]
        
        return templates
