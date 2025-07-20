"""
Advanced Email Templates Management System

This module provides comprehensive email template management with AI-powered
content generation, multi-language support, and enterprise-grade features.

Version: 3.0.0
Developed by Spotify AI Agent Team
"""

import asyncio
import json
import yaml
import html
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import structlog
import aiofiles
from jinja2 import Environment, FileSystemLoader, Template, select_autoescape
from markupsafe import Markup
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import smtplib
import ssl

logger = structlog.get_logger(__name__)

# ============================================================================
# Email Template Enums and Models
# ============================================================================

class EmailType(Enum):
    """Types d'emails supportés"""
    ALERT = "alert"
    NOTIFICATION = "notification"
    REPORT = "report"
    MARKETING = "marketing"
    TRANSACTIONAL = "transactional"
    SYSTEM = "system"
    SECURITY = "security"
    COMPLIANCE = "compliance"

class EmailPriority(Enum):
    """Priorités d'email"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"

class EmailFormat(Enum):
    """Formats d'email"""
    HTML = "html"
    TEXT = "text"
    MULTIPART = "multipart"

class LanguageCode(Enum):
    """Codes de langue supportés"""
    EN = "en"
    FR = "fr"
    DE = "de"
    ES = "es"
    IT = "it"
    PT = "pt"
    ZH = "zh"
    JA = "ja"
    KO = "ko"
    AR = "ar"

@dataclass
class EmailTemplate:
    """Modèle d'email avec métadonnées"""
    id: str
    name: str
    type: EmailType
    subject_template: str
    body_template: str
    format: EmailFormat = EmailFormat.HTML
    language: LanguageCode = LanguageCode.EN
    priority: EmailPriority = EmailPriority.NORMAL
    tags: List[str] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    css_styles: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    author: str = "System"
    description: str = ""
    attachments: List[str] = field(default_factory=list)

@dataclass
class EmailContext:
    """Contexte pour le rendu d'email"""
    recipient: str
    sender: str
    data: Dict[str, Any] = field(default_factory=dict)
    language: LanguageCode = LanguageCode.EN
    timezone: str = "UTC"
    personalization: Dict[str, Any] = field(default_factory=dict)
    tracking_params: Dict[str, str] = field(default_factory=dict)

@dataclass
class RenderedEmail:
    """Email rendu avec tout le contenu"""
    subject: str
    body: str
    format: EmailFormat
    recipients: List[str]
    sender: str
    reply_to: Optional[str] = None
    attachments: List[str] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)
    tracking_id: Optional[str] = None
    render_time: datetime = field(default_factory=datetime.utcnow)

# ============================================================================
# Advanced Email Template Manager
# ============================================================================

class AdvancedEmailTemplateManager:
    """Gestionnaire avancé de templates email avec IA et multi-langue"""
    
    def __init__(self,
                 templates_dir: str,
                 cache_enabled: bool = True,
                 ai_optimization: bool = True,
                 multi_language: bool = True):
        
        self.templates_dir = Path(templates_dir)
        self.cache_enabled = cache_enabled
        self.ai_optimization = ai_optimization
        self.multi_language = multi_language
        
        # Template storage
        self.templates: Dict[str, EmailTemplate] = {}
        self.compiled_templates: Dict[str, Template] = {}
        self.template_cache: Dict[str, RenderedEmail] = {}
        
        # Jinja2 environment with security
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filters
        self._setup_custom_filters()
        
        # AI and analytics
        self.ai_analytics = {}
        self.performance_metrics = {}
        self.personalization_rules = {}
        
        # Multi-language support
        self.translations: Dict[str, Dict[str, str]] = {}
        self.language_fallbacks = {
            LanguageCode.FR: LanguageCode.EN,
            LanguageCode.DE: LanguageCode.EN,
            LanguageCode.ES: LanguageCode.EN,
        }
        
        # Event listeners
        self.event_listeners: Dict[str, List[Callable]] = {}
        
        # Initialize
        asyncio.create_task(self._initialize())
        
        logger.info("Advanced Email Template Manager initialized")
    
    async def _initialize(self):
        """Initialisation du gestionnaire"""
        
        # Création des répertoires
        await self._ensure_directories()
        
        # Chargement des templates
        await self.load_all_templates()
        
        # Chargement des traductions
        if self.multi_language:
            await self._load_translations()
        
        # Configuration des règles de personnalisation
        await self._setup_personalization_rules()
        
        logger.info("Email template manager initialization completed")
    
    async def _ensure_directories(self):
        """Assure que les répertoires nécessaires existent"""
        
        directories = [
            self.templates_dir,
            self.templates_dir / "layouts",
            self.templates_dir / "partials",
            self.templates_dir / "assets" / "css",
            self.templates_dir / "assets" / "images",
            self.templates_dir / "translations",
            self.templates_dir / "compiled"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_custom_filters(self):
        """Configure les filtres Jinja2 personnalisés"""
        
        @self.jinja_env.filter('format_datetime')
        def format_datetime(value, format_str="%Y-%m-%d %H:%M:%S", timezone="UTC"):
            """Formate une datetime selon le timezone"""
            if isinstance(value, str):
                value = datetime.fromisoformat(value)
            return value.strftime(format_str)
        
        @self.jinja_env.filter('format_currency')
        def format_currency(value, currency="USD", locale="en_US"):
            """Formate une valeur monétaire"""
            try:
                return f"{currency} {value:,.2f}"
            except:
                return str(value)
        
        @self.jinja_env.filter('truncate_smart')
        def truncate_smart(value, length=100, suffix="..."):
            """Troncature intelligente qui respecte les mots"""
            if len(value) <= length:
                return value
            truncated = value[:length].rsplit(' ', 1)[0]
            return truncated + suffix
        
        @self.jinja_env.filter('highlight_keywords')
        def highlight_keywords(text, keywords, css_class="highlight"):
            """Met en évidence des mots-clés dans le texte"""
            for keyword in keywords:
                pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                text = pattern.sub(f'<span class="{css_class}">{keyword}</span>', text)
            return Markup(text)
        
        @self.jinja_env.filter('personalize')
        def personalize(text, context):
            """Personnalise le texte selon le contexte utilisateur"""
            if not isinstance(context, dict):
                return text
            
            # Personnalisation basique
            for key, value in context.get('personalization', {}).items():
                text = text.replace(f'{{{key}}}', str(value))
            
            return text
    
    async def create_template(self,
                            template_id: str,
                            name: str,
                            type: EmailType,
                            subject_template: str,
                            body_template: str,
                            **kwargs) -> bool:
        """Crée un nouveau template email"""
        
        try:
            template = EmailTemplate(
                id=template_id,
                name=name,
                type=type,
                subject_template=subject_template,
                body_template=body_template,
                **kwargs
            )
            
            # Validation du template
            if not await self._validate_template(template):
                return False
            
            # Compilation du template
            compiled_subject = self.jinja_env.from_string(subject_template)
            compiled_body = self.jinja_env.from_string(body_template)
            
            # Stockage
            self.templates[template_id] = template
            self.compiled_templates[f"{template_id}_subject"] = compiled_subject
            self.compiled_templates[f"{template_id}_body"] = compiled_body
            
            # Sauvegarde sur disque
            await self._save_template_to_disk(template)
            
            # Optimisation IA si activée
            if self.ai_optimization:
                await self._optimize_template_with_ai(template_id)
            
            # Événement de création
            await self._trigger_event("template_created", template)
            
            logger.info(f"Email template created: {template_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create template {template_id}: {e}")
            return False
    
    async def render_email(self,
                         template_id: str,
                         context: EmailContext) -> Optional[RenderedEmail]:
        """Rend un email à partir d'un template"""
        
        try:
            if template_id not in self.templates:
                logger.error(f"Template not found: {template_id}")
                return None
            
            template = self.templates[template_id]
            
            # Vérification du cache
            cache_key = f"{template_id}_{hash(str(context.data))}"
            if self.cache_enabled and cache_key in self.template_cache:
                cached = self.template_cache[cache_key]
                if datetime.utcnow() - cached.render_time < timedelta(hours=1):
                    return cached
            
            # Préparation du contexte de rendu
            render_context = await self._prepare_render_context(context)
            
            # Application des traductions
            if self.multi_language and context.language != LanguageCode.EN:
                render_context = await self._apply_translations(
                    render_context, context.language
                )
            
            # Rendu du sujet
            subject_template = self.compiled_templates[f"{template_id}_subject"]
            rendered_subject = subject_template.render(**render_context)
            
            # Rendu du corps
            body_template = self.compiled_templates[f"{template_id}_body"]
            rendered_body = body_template.render(**render_context)
            
            # Post-traitement avec IA
            if self.ai_optimization:
                rendered_subject, rendered_body = await self._ai_post_process(
                    rendered_subject, rendered_body, context
                )
            
            # Création de l'email rendu
            rendered_email = RenderedEmail(
                subject=rendered_subject,
                body=rendered_body,
                format=template.format,
                recipients=[context.recipient],
                sender=context.sender,
                tracking_id=self._generate_tracking_id(),
                headers=self._generate_headers(template, context)
            )
            
            # Mise en cache
            if self.cache_enabled:
                self.template_cache[cache_key] = rendered_email
            
            # Métriques
            await self._record_render_metrics(template_id, context)
            
            # Événement de rendu
            await self._trigger_event("email_rendered", rendered_email)
            
            return rendered_email
            
        except Exception as e:
            logger.error(f"Failed to render email template {template_id}: {e}")
            return None
    
    async def bulk_render(self,
                        template_id: str,
                        contexts: List[EmailContext],
                        batch_size: int = 100) -> List[RenderedEmail]:
        """Rendu en lot avec optimisation de performance"""
        
        rendered_emails = []
        
        # Traitement par lots
        for i in range(0, len(contexts), batch_size):
            batch = contexts[i:i + batch_size]
            
            # Rendu parallèle du lot
            tasks = [self.render_email(template_id, ctx) for ctx in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filtrage des résultats valides
            for result in batch_results:
                if isinstance(result, RenderedEmail):
                    rendered_emails.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Bulk render error: {result}")
            
            # Pause entre les lots pour éviter la surcharge
            if i + batch_size < len(contexts):
                await asyncio.sleep(0.1)
        
        logger.info(f"Bulk rendered {len(rendered_emails)} emails from {len(contexts)} contexts")
        return rendered_emails
    
    async def personalize_template(self,
                                 template_id: str,
                                 user_profile: Dict[str, Any]) -> str:
        """Personnalise un template selon le profil utilisateur"""
        
        if template_id not in self.templates:
            return template_id
        
        # Analyse du profil utilisateur
        user_preferences = await self._analyze_user_preferences(user_profile)
        
        # Sélection de la variante optimale
        if self.ai_optimization:
            variant = await self._select_optimal_variant(template_id, user_preferences)
            return variant or template_id
        
        return template_id
    
    async def a_b_test_templates(self,
                               template_a: str,
                               template_b: str,
                               contexts: List[EmailContext],
                               split_ratio: float = 0.5) -> Dict[str, Any]:
        """Test A/B entre deux templates"""
        
        import random
        
        # Division des contextes
        random.shuffle(contexts)
        split_index = int(len(contexts) * split_ratio)
        
        group_a = contexts[:split_index]
        group_b = contexts[split_index:]
        
        # Rendu des groupes
        results_a = await self.bulk_render(template_a, group_a)
        results_b = await self.bulk_render(template_b, group_b)
        
        # Analyse des résultats
        metrics = {
            "template_a": {
                "id": template_a,
                "count": len(results_a),
                "success_rate": len(results_a) / len(group_a) if group_a else 0
            },
            "template_b": {
                "id": template_b,
                "count": len(results_b),
                "success_rate": len(results_b) / len(group_b) if group_b else 0
            },
            "test_date": datetime.utcnow(),
            "total_contexts": len(contexts)
        }
        
        return metrics
    
    async def load_all_templates(self) -> int:
        """Charge tous les templates depuis le disque"""
        
        loaded_count = 0
        
        # Recherche des fichiers de template
        for template_file in self.templates_dir.rglob("*.yaml"):
            try:
                async with aiofiles.open(template_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    template_data = yaml.safe_load(content)
                
                # Conversion en objet EmailTemplate
                template = EmailTemplate(**template_data)
                self.templates[template.id] = template
                
                # Compilation
                subject_tpl = self.jinja_env.from_string(template.subject_template)
                body_tpl = self.jinja_env.from_string(template.body_template)
                
                self.compiled_templates[f"{template.id}_subject"] = subject_tpl
                self.compiled_templates[f"{template.id}_body"] = body_tpl
                
                loaded_count += 1
                
            except Exception as e:
                logger.error(f"Failed to load template {template_file}: {e}")
        
        logger.info(f"Loaded {loaded_count} email templates")
        return loaded_count
    
    async def export_templates(self, format: str = "yaml") -> str:
        """Exporte tous les templates dans un format donné"""
        
        export_data = {}
        
        for template_id, template in self.templates.items():
            template_dict = {
                "id": template.id,
                "name": template.name,
                "type": template.type.value,
                "subject_template": template.subject_template,
                "body_template": template.body_template,
                "format": template.format.value,
                "language": template.language.value,
                "priority": template.priority.value,
                "tags": template.tags,
                "variables": template.variables,
                "css_styles": template.css_styles,
                "created_at": template.created_at.isoformat(),
                "updated_at": template.updated_at.isoformat(),
                "version": template.version,
                "author": template.author,
                "description": template.description
            }
            export_data[template_id] = template_dict
        
        if format.lower() == "json":
            return json.dumps(export_data, indent=2, default=str)
        elif format.lower() == "yaml":
            return yaml.dump(export_data, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_template_analytics(self, template_id: str) -> Dict[str, Any]:
        """Retourne les analytics d'un template"""
        
        return self.ai_analytics.get(template_id, {
            "renders_count": 0,
            "success_rate": 0.0,
            "avg_render_time": 0.0,
            "popular_variables": [],
            "language_distribution": {},
            "performance_score": 0.0
        })
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques de performance globales"""
        
        total_templates = len(self.templates)
        total_renders = sum(
            self.ai_analytics.get(tid, {}).get("renders_count", 0)
            for tid in self.templates.keys()
        )
        
        return {
            "total_templates": total_templates,
            "total_renders": total_renders,
            "cache_hit_rate": self._calculate_cache_hit_rate(),
            "avg_render_time": self._calculate_avg_render_time(),
            "memory_usage": self._estimate_memory_usage(),
            "most_used_templates": self._get_most_used_templates(),
            "language_distribution": self._get_language_distribution()
        }
    
    async def cleanup_cache(self, max_age_hours: int = 24):
        """Nettoie le cache des templates anciens"""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        cleaned_count = 0
        for cache_key in list(self.template_cache.keys()):
            cached_email = self.template_cache[cache_key]
            if cached_email.render_time < cutoff_time:
                del self.template_cache[cache_key]
                cleaned_count += 1
        
        logger.info(f"Cleaned {cleaned_count} cached templates")
    
    def register_event_listener(self, event_type: str, listener: Callable):
        """Enregistre un listener d'événements"""
        
        if event_type not in self.event_listeners:
            self.event_listeners[event_type] = []
        
        self.event_listeners[event_type].append(listener)
    
    async def _validate_template(self, template: EmailTemplate) -> bool:
        """Valide un template email"""
        
        # Validation basique
        if not template.id or not template.name:
            return False
        
        if not template.subject_template or not template.body_template:
            return False
        
        # Validation de la syntaxe Jinja2
        try:
            self.jinja_env.from_string(template.subject_template)
            self.jinja_env.from_string(template.body_template)
        except Exception as e:
            logger.error(f"Template syntax error: {e}")
            return False
        
        # Validation du HTML si format HTML
        if template.format == EmailFormat.HTML:
            if not self._validate_html(template.body_template):
                return False
        
        return True
    
    def _validate_html(self, html_content: str) -> bool:
        """Valide le contenu HTML"""
        
        # Vérifications basiques
        required_tags = ['html', 'head', 'body']
        for tag in required_tags:
            if f'<{tag}' not in html_content.lower():
                logger.warning(f"Missing required HTML tag: {tag}")
        
        # Vérification des balises fermées
        open_tags = re.findall(r'<([^/\s>]+)', html_content)
        close_tags = re.findall(r'</([^>\s]+)', html_content)
        
        for tag in open_tags:
            if tag.lower() not in ['img', 'br', 'hr', 'meta', 'link']:
                if tag not in close_tags:
                    logger.warning(f"Unclosed HTML tag: {tag}")
        
        return True
    
    async def _save_template_to_disk(self, template: EmailTemplate):
        """Sauvegarde un template sur disque"""
        
        template_data = {
            "id": template.id,
            "name": template.name,
            "type": template.type.value,
            "subject_template": template.subject_template,
            "body_template": template.body_template,
            "format": template.format.value,
            "language": template.language.value,
            "priority": template.priority.value,
            "tags": template.tags,
            "variables": template.variables,
            "css_styles": template.css_styles,
            "created_at": template.created_at.isoformat(),
            "updated_at": template.updated_at.isoformat(),
            "version": template.version,
            "author": template.author,
            "description": template.description
        }
        
        file_path = self.templates_dir / f"{template.id}.yaml"
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(yaml.dump(template_data, default_flow_style=False))
    
    async def _prepare_render_context(self, context: EmailContext) -> Dict[str, Any]:
        """Prépare le contexte de rendu"""
        
        render_context = {
            "recipient": context.recipient,
            "sender": context.sender,
            "current_date": datetime.utcnow(),
            "language": context.language.value,
            "timezone": context.timezone,
            **context.data,
            **context.personalization
        }
        
        # Ajout des fonctions utilitaires
        render_context["utils"] = {
            "format_date": lambda d: d.strftime("%Y-%m-%d"),
            "format_time": lambda d: d.strftime("%H:%M:%S"),
            "capitalize": lambda s: s.capitalize(),
            "upper": lambda s: s.upper(),
            "lower": lambda s: s.lower()
        }
        
        return render_context
    
    async def _trigger_event(self, event_type: str, data: Any):
        """Déclenche un événement"""
        
        listeners = self.event_listeners.get(event_type, [])
        for listener in listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(data)
                else:
                    listener(data)
            except Exception as e:
                logger.error(f"Event listener error: {e}")
    
    def _generate_tracking_id(self) -> str:
        """Génère un ID de tracking unique"""
        import uuid
        return str(uuid.uuid4())
    
    def _generate_headers(self, template: EmailTemplate, context: EmailContext) -> Dict[str, str]:
        """Génère les headers d'email"""
        
        headers = {
            "X-Template-ID": template.id,
            "X-Template-Version": template.version,
            "X-Priority": template.priority.value,
            "X-Language": context.language.value
        }
        
        # Ajout des paramètres de tracking
        for key, value in context.tracking_params.items():
            headers[f"X-Track-{key}"] = value
        
        return headers

# ============================================================================
# Template Builder Class
# ============================================================================

class EmailTemplateBuilder:
    """Builder pour créer des templates email facilement"""
    
    def __init__(self):
        self._template = EmailTemplate(
            id="",
            name="",
            type=EmailType.NOTIFICATION,
            subject_template="",
            body_template=""
        )
    
    def with_id(self, template_id: str) -> 'EmailTemplateBuilder':
        self._template.id = template_id
        return self
    
    def with_name(self, name: str) -> 'EmailTemplateBuilder':
        self._template.name = name
        return self
    
    def with_type(self, email_type: EmailType) -> 'EmailTemplateBuilder':
        self._template.type = email_type
        return self
    
    def with_subject(self, subject: str) -> 'EmailTemplateBuilder':
        self._template.subject_template = subject
        return self
    
    def with_body(self, body: str) -> 'EmailTemplateBuilder':
        self._template.body_template = body
        return self
    
    def with_format(self, format: EmailFormat) -> 'EmailTemplateBuilder':
        self._template.format = format
        return self
    
    def with_language(self, language: LanguageCode) -> 'EmailTemplateBuilder':
        self._template.language = language
        return self
    
    def with_priority(self, priority: EmailPriority) -> 'EmailTemplateBuilder':
        self._template.priority = priority
        return self
    
    def with_tags(self, tags: List[str]) -> 'EmailTemplateBuilder':
        self._template.tags = tags
        return self
    
    def with_variables(self, variables: Dict[str, Any]) -> 'EmailTemplateBuilder':
        self._template.variables = variables
        return self
    
    def with_css(self, css_styles: str) -> 'EmailTemplateBuilder':
        self._template.css_styles = css_styles
        return self
    
    def build(self) -> EmailTemplate:
        return self._template

# ============================================================================
# Factory Functions
# ============================================================================

def create_email_template_manager(
    templates_dir: str,
    enable_ai: bool = True,
    enable_cache: bool = True
) -> AdvancedEmailTemplateManager:
    """Factory pour créer un gestionnaire de templates"""
    
    return AdvancedEmailTemplateManager(
        templates_dir=templates_dir,
        cache_enabled=enable_cache,
        ai_optimization=enable_ai,
        multi_language=True
    )

def create_alert_template() -> EmailTemplate:
    """Crée un template d'alerte par défaut"""
    
    return EmailTemplateBuilder() \
        .with_id("default_alert") \
        .with_name("Default Alert Template") \
        .with_type(EmailType.ALERT) \
        .with_subject("🚨 Alert: {{ alert.name }} - {{ alert.severity }}") \
        .with_body("""
        <html>
        <head>
            <title>Alert Notification</title>
        </head>
        <body>
            <h2>Alert: {{ alert.name }}</h2>
            <p><strong>Severity:</strong> {{ alert.severity }}</p>
            <p><strong>Time:</strong> {{ alert.timestamp }}</p>
            <p><strong>Description:</strong> {{ alert.description }}</p>
            <p><strong>Source:</strong> {{ alert.source }}</p>
        </body>
        </html>
        """) \
        .with_format(EmailFormat.HTML) \
        .with_priority(EmailPriority.HIGH) \
        .build()

# Export des classes principales
__all__ = [
    "AdvancedEmailTemplateManager",
    "EmailTemplate",
    "EmailContext",
    "RenderedEmail",
    "EmailTemplateBuilder",
    "EmailType",
    "EmailPriority",
    "EmailFormat",
    "LanguageCode",
    "create_email_template_manager",
    "create_alert_template"
]
