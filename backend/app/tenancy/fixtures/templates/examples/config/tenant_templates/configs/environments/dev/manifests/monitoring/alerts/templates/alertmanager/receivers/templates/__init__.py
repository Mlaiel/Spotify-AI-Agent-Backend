"""
Advanced Email Templates Management System

This module provides comprehensive email template management with AI-powered
personalization, multi-tenant support, and enterprise-grade features.

Version: 3.0.0
Developed by Spotify AI Agent Team
Lead Developer & AI Architect: Fahed Mlaiel
"""

import asyncio
import json
import hashlib
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import structlog
import aiofiles
from jinja2 import Environment, FileSystemLoader, select_autoescape, Template
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.base import MimeBase
from email import encoders
import smtplib
import aiosmtplib
from premailer import transform
import bleach
from bs4 import BeautifulSoup
from markupsafe import Markup
import cssutils
from PIL import Image
import qrcode
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from ..utils import (
    validate_email, generate_secure_token, encrypt_sensitive_data,
    decrypt_sensitive_data, sanitize_html_content
)
from ..constants import (
    EMAIL_TEMPLATE_TYPES, DEFAULT_SMTP_SETTINGS, TEMPLATE_CACHE_TTL,
    AI_MODEL_CONFIGS, SUPPORTED_LANGUAGES
)
from ..enums import (
    EnvironmentType, NotificationChannel, ThreatLevel,
    LogLevel, PriorityLevel, DeploymentStatus
)

logger = structlog.get_logger(__name__)

# ============================================================================
# Email Template Models and Enums
# ============================================================================

class EmailTemplateType(Enum):
    """Types de templates email"""
    ALERT = "alert"
    NOTIFICATION = "notification"
    REPORT = "report"
    MARKETING = "marketing"
    TRANSACTIONAL = "transactional"
    SYSTEM = "system"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    ONBOARDING = "onboarding"
    RECOVERY = "recovery"

class EmailPriority(Enum):
    """Priorités d'email"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"

class EmailStatus(Enum):
    """Statuts d'email"""
    DRAFT = "draft"
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    OPENED = "opened"
    CLICKED = "clicked"
    BOUNCED = "bounced"
    FAILED = "failed"

class PersonalizationLevel(Enum):
    """Niveaux de personnalisation"""
    BASIC = "basic"
    ADVANCED = "advanced"
    AI_POWERED = "ai_powered"
    HYPER_PERSONALIZED = "hyper_personalized"

@dataclass
class EmailTemplate:
    """Template d'email avancé"""
    id: str
    name: str
    type: EmailTemplateType
    subject_template: str
    html_content: str
    text_content: str
    language: str = "en"
    version: str = "1.0.0"
    tenant_id: Optional[str] = None
    created_by: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    personalization_rules: List[Dict[str, Any]] = field(default_factory=list)
    ai_settings: Dict[str, Any] = field(default_factory=dict)
    attachments: List[str] = field(default_factory=list)
    tracking_enabled: bool = True
    a_b_test_config: Optional[Dict[str, Any]] = None

@dataclass
class EmailRecipient:
    """Destinataire d'email"""
    email: str
    name: Optional[str] = None
    tenant_id: Optional[str] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    segments: List[str] = field(default_factory=list)
    engagement_score: float = 0.0
    last_interaction: Optional[datetime] = None
    custom_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EmailCampaign:
    """Campagne d'email"""
    id: str
    name: str
    template_id: str
    recipients: List[EmailRecipient]
    scheduled_time: Optional[datetime] = None
    priority: EmailPriority = EmailPriority.NORMAL
    personalization_level: PersonalizationLevel = PersonalizationLevel.BASIC
    a_b_test_enabled: bool = False
    tracking_config: Dict[str, Any] = field(default_factory=dict)
    delivery_options: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EmailAnalytics:
    """Analytiques d'email"""
    campaign_id: str
    total_sent: int = 0
    delivered: int = 0
    opened: int = 0
    clicked: int = 0
    bounced: int = 0
    unsubscribed: int = 0
    engagement_rate: float = 0.0
    conversion_rate: float = 0.0
    revenue_generated: float = 0.0
    geographical_data: Dict[str, int] = field(default_factory=dict)
    device_data: Dict[str, int] = field(default_factory=dict)
    time_data: Dict[str, int] = field(default_factory=dict)

# ============================================================================
# AI-Powered Email Template Manager
# ============================================================================

class AdvancedEmailTemplateManager:
    """Gestionnaire de templates email avancé avec IA et personnalisation"""
    
    def __init__(self,
                 templates_dir: str,
                 smtp_config: Optional[Dict[str, Any]] = None,
                 ai_models_enabled: bool = True,
                 multi_tenant: bool = True):
        
        self.templates_dir = Path(templates_dir)
        self.smtp_config = smtp_config or DEFAULT_SMTP_SETTINGS
        self.ai_models_enabled = ai_models_enabled
        self.multi_tenant = multi_tenant
        
        # Template storage
        self.templates: Dict[str, EmailTemplate] = {}
        self.template_cache: Dict[str, Dict[str, Any]] = {}
        self.campaigns: Dict[str, EmailCampaign] = {}
        self.analytics: Dict[str, EmailAnalytics] = {}
        
        # AI Components
        self.sentiment_analyzer = None
        self.text_generator = None
        self.personalization_model = None
        self.engagement_predictor = None
        
        # Jinja2 Environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.templates_dir),
            autoescape=select_autoescape(['html', 'xml']),
            enable_async=True
        )
        
        # Email tracking
        self.tracking_pixels: Dict[str, Dict[str, Any]] = {}
        self.link_tracking: Dict[str, Dict[str, Any]] = {}
        
        # A/B Testing
        self.ab_tests: Dict[str, Dict[str, Any]] = {}
        
        # Personalization engine
        self.personalization_engine = PersonalizationEngine()
        
        # Initialize AI models if enabled
        if ai_models_enabled:
            asyncio.create_task(self._initialize_ai_models())
        
        # Setup directories
        self._setup_directories()
        
        logger.info("Advanced Email Template Manager initialized")
    
    def _setup_directories(self):
        """Configure les répertoires nécessaires"""
        
        directories = [
            "templates/html",
            "templates/text", 
            "templates/partials",
            "assets/images",
            "assets/css",
            "assets/fonts",
            "campaigns",
            "analytics",
            "exports"
        ]
        
        for directory in directories:
            (self.templates_dir / directory).mkdir(parents=True, exist_ok=True)
    
    async def _initialize_ai_models(self):
        """Initialise les modèles IA"""
        
        try:
            # Analyseur de sentiment
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Générateur de texte
            self.text_generator = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Modèle de personnalisation personnalisé
            self.personalization_model = await self._load_personalization_model()
            
            # Prédicteur d'engagement
            self.engagement_predictor = await self._load_engagement_model()
            
            logger.info("AI models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI models: {e}")
            self.ai_models_enabled = False
    
    async def create_template(self,
                            name: str,
                            template_type: EmailTemplateType,
                            subject_template: str,
                            html_content: str,
                            text_content: str,
                            language: str = "en",
                            tenant_id: Optional[str] = None,
                            metadata: Optional[Dict[str, Any]] = None) -> str:
        """Crée un nouveau template email"""
        
        try:
            template_id = str(uuid.uuid4())
            
            # Validation du contenu
            if not await self._validate_template_content(html_content, text_content):
                raise ValueError("Invalid template content")
            
            # Optimisation IA du contenu si activée
            if self.ai_models_enabled:
                html_content = await self._optimize_content_with_ai(html_content)
                subject_template = await self._optimize_subject_with_ai(subject_template)
            
            # Création du template
            template = EmailTemplate(
                id=template_id,
                name=name,
                type=template_type,
                subject_template=subject_template,
                html_content=html_content,
                text_content=text_content,
                language=language,
                tenant_id=tenant_id,
                metadata=metadata or {},
                ai_settings={
                    "content_optimized": self.ai_models_enabled,
                    "personalization_enabled": True,
                    "engagement_prediction": True
                }
            )
            
            # Sauvegarde
            self.templates[template_id] = template
            await self._save_template_to_disk(template)
            
            # Génération des variantes A/B si demandé
            if metadata and metadata.get("enable_ab_testing"):
                await self._generate_ab_variants(template)
            
            logger.info(f"Template created: {template_id}")
            return template_id
            
        except Exception as e:
            logger.error(f"Failed to create template: {e}")
            raise
    
    async def render_template(self,
                            template_id: str,
                            recipient: EmailRecipient,
                            context: Optional[Dict[str, Any]] = None,
                            personalization_level: PersonalizationLevel = PersonalizationLevel.BASIC) -> Dict[str, str]:
        """Rend un template avec personnalisation IA"""
        
        try:
            template = self.templates.get(template_id)
            if not template:
                raise ValueError(f"Template not found: {template_id}")
            
            # Préparation du contexte
            render_context = await self._prepare_render_context(
                template, recipient, context or {}, personalization_level
            )
            
            # Rendu avec Jinja2
            html_template = self.jinja_env.from_string(template.html_content)
            text_template = self.jinja_env.from_string(template.text_content)
            subject_template = self.jinja_env.from_string(template.subject_template)
            
            rendered_html = await html_template.render_async(**render_context)
            rendered_text = await text_template.render_async(**render_context)
            rendered_subject = await subject_template.render_async(**render_context)
            
            # Post-traitement IA
            if self.ai_models_enabled and personalization_level in [
                PersonalizationLevel.AI_POWERED, 
                PersonalizationLevel.HYPER_PERSONALIZED
            ]:
                rendered_html = await self._apply_ai_personalization(
                    rendered_html, recipient, personalization_level
                )
                rendered_subject = await self._optimize_subject_for_recipient(
                    rendered_subject, recipient
                )
            
            # Optimisation CSS inline
            rendered_html = transform(rendered_html)
            
            # Ajout du tracking si activé
            if template.tracking_enabled:
                rendered_html = await self._add_tracking_pixels(
                    rendered_html, template_id, recipient.email
                )
                rendered_html = await self._add_link_tracking(
                    rendered_html, template_id, recipient.email
                )
            
            # Sanitization de sécurité
            rendered_html = bleach.clean(
                rendered_html,
                tags=bleach.ALLOWED_TAGS + ['img', 'div', 'span', 'table', 'tr', 'td'],
                attributes={
                    '*': ['class', 'style'],
                    'img': ['src', 'alt', 'width', 'height'],
                    'a': ['href', 'title']
                }
            )
            
            return {
                "subject": rendered_subject,
                "html": rendered_html,
                "text": rendered_text,
                "metadata": {
                    "template_id": template_id,
                    "recipient_email": recipient.email,
                    "personalization_level": personalization_level.value,
                    "ai_enhanced": self.ai_models_enabled,
                    "rendered_at": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to render template {template_id}: {e}")
            raise
    
    async def send_email(self,
                        template_id: str,
                        recipients: List[EmailRecipient],
                        context: Optional[Dict[str, Any]] = None,
                        priority: EmailPriority = EmailPriority.NORMAL,
                        delivery_time: Optional[datetime] = None) -> str:
        """Envoie un email avec le template spécifié"""
        
        try:
            campaign_id = str(uuid.uuid4())
            
            # Création de la campagne
            campaign = EmailCampaign(
                id=campaign_id,
                name=f"Campaign_{template_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                template_id=template_id,
                recipients=recipients,
                scheduled_time=delivery_time,
                priority=priority
            )
            
            self.campaigns[campaign_id] = campaign
            
            # Envoi immédiat ou programmé
            if delivery_time and delivery_time > datetime.utcnow():
                # Programmer l'envoi
                asyncio.create_task(
                    self._schedule_email_delivery(campaign, context)
                )
                logger.info(f"Email scheduled for {delivery_time}: {campaign_id}")
            else:
                # Envoi immédiat
                await self._execute_email_campaign(campaign, context)
                logger.info(f"Email sent immediately: {campaign_id}")
            
            return campaign_id
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            raise
    
    async def _execute_email_campaign(self,
                                    campaign: EmailCampaign,
                                    context: Optional[Dict[str, Any]] = None):
        """Exécute une campagne email"""
        
        analytics = EmailAnalytics(campaign_id=campaign.id)
        
        try:
            template = self.templates[campaign.template_id]
            
            # Configuration SMTP
            smtp_client = aiosmtplib.SMTP(
                hostname=self.smtp_config["host"],
                port=self.smtp_config["port"],
                use_tls=self.smtp_config.get("use_tls", True)
            )
            
            await smtp_client.connect()
            
            if self.smtp_config.get("username"):
                await smtp_client.login(
                    self.smtp_config["username"],
                    self.smtp_config["password"]
                )
            
            # Envoi par lot pour optimiser les performances
            batch_size = 100
            for i in range(0, len(campaign.recipients), batch_size):
                batch = campaign.recipients[i:i + batch_size]
                
                tasks = []
                for recipient in batch:
                    task = self._send_single_email(
                        smtp_client, template, recipient, context, analytics
                    )
                    tasks.append(task)
                
                # Envoi concurrent du lot
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Pause entre les lots pour éviter le spam
                if i + batch_size < len(campaign.recipients):
                    await asyncio.sleep(1)
            
            await smtp_client.quit()
            
            # Calcul des métriques finales
            analytics.engagement_rate = (
                (analytics.opened + analytics.clicked) / analytics.total_sent * 100
                if analytics.total_sent > 0 else 0
            )
            
            self.analytics[campaign.id] = analytics
            
            logger.info(f"Campaign completed: {campaign.id}")
            
        except Exception as e:
            logger.error(f"Campaign execution failed: {e}")
            raise
    
    async def _send_single_email(self,
                               smtp_client: aiosmtplib.SMTP,
                               template: EmailTemplate,
                               recipient: EmailRecipient,
                               context: Optional[Dict[str, Any]],
                               analytics: EmailAnalytics):
        """Envoie un email individuel"""
        
        try:
            # Rendu du template
            rendered = await self.render_template(
                template.id, recipient, context
            )
            
            # Création du message MIME
            msg = MimeMultipart('alternative')
            msg['Subject'] = rendered['subject']
            msg['From'] = self.smtp_config['from_email']
            msg['To'] = recipient.email
            msg['Message-ID'] = f"<{uuid.uuid4()}@{self.smtp_config['domain']}>"
            
            # Ajout des parties texte et HTML
            text_part = MimeText(rendered['text'], 'plain', 'utf-8')
            html_part = MimeText(rendered['html'], 'html', 'utf-8')
            
            msg.attach(text_part)
            msg.attach(html_part)
            
            # Ajout des pièces jointes si nécessaire
            for attachment_path in template.attachments:
                await self._add_attachment(msg, attachment_path)
            
            # Envoi
            await smtp_client.send_message(msg)
            
            analytics.total_sent += 1
            analytics.delivered += 1
            
            # Logging de l'envoi
            logger.debug(f"Email sent to {recipient.email}")
            
        except Exception as e:
            analytics.total_sent += 1
            logger.error(f"Failed to send email to {recipient.email}: {e}")
    
    async def create_email_campaign(self,
                                  name: str,
                                  template_id: str,
                                  recipient_segments: List[str],
                                  schedule_time: Optional[datetime] = None,
                                  a_b_test_config: Optional[Dict[str, Any]] = None) -> str:
        """Crée une campagne email avancée"""
        
        try:
            campaign_id = str(uuid.uuid4())
            
            # Récupération des destinataires par segments
            recipients = await self._get_recipients_by_segments(recipient_segments)
            
            # Configuration A/B test si demandée
            if a_b_test_config:
                recipients = await self._setup_ab_test_groups(recipients, a_b_test_config)
            
            # Création de la campagne
            campaign = EmailCampaign(
                id=campaign_id,
                name=name,
                template_id=template_id,
                recipients=recipients,
                scheduled_time=schedule_time,
                a_b_test_enabled=bool(a_b_test_config),
                tracking_config={
                    "open_tracking": True,
                    "click_tracking": True,
                    "conversion_tracking": True,
                    "geographic_tracking": True
                }
            )
            
            self.campaigns[campaign_id] = campaign
            
            logger.info(f"Email campaign created: {campaign_id}")
            return campaign_id
            
        except Exception as e:
            logger.error(f"Failed to create email campaign: {e}")
            raise
    
    async def analyze_email_performance(self,
                                      campaign_id: str,
                                      time_range: Optional[Dict[str, datetime]] = None) -> Dict[str, Any]:
        """Analyse les performances d'une campagne"""
        
        try:
            analytics = self.analytics.get(campaign_id)
            if not analytics:
                raise ValueError(f"Analytics not found for campaign: {campaign_id}")
            
            # Métriques de base
            basic_metrics = {
                "total_sent": analytics.total_sent,
                "delivered": analytics.delivered,
                "opened": analytics.opened,
                "clicked": analytics.clicked,
                "bounced": analytics.bounced,
                "unsubscribed": analytics.unsubscribed,
                "delivery_rate": analytics.delivered / analytics.total_sent * 100 if analytics.total_sent > 0 else 0,
                "open_rate": analytics.opened / analytics.delivered * 100 if analytics.delivered > 0 else 0,
                "click_rate": analytics.clicked / analytics.opened * 100 if analytics.opened > 0 else 0,
                "bounce_rate": analytics.bounced / analytics.total_sent * 100 if analytics.total_sent > 0 else 0,
                "unsubscribe_rate": analytics.unsubscribed / analytics.delivered * 100 if analytics.delivered > 0 else 0
            }
            
            # Analyses avancées avec IA
            advanced_analysis = {}
            if self.ai_models_enabled:
                advanced_analysis = await self._perform_ai_analysis(campaign_id, analytics)
            
            # Recommandations d'optimisation
            recommendations = await self._generate_optimization_recommendations(
                campaign_id, basic_metrics, advanced_analysis
            )
            
            # Comparaison avec les benchmarks
            benchmark_comparison = await self._compare_with_benchmarks(
                basic_metrics, self.campaigns[campaign_id].template_id
            )
            
            return {
                "campaign_id": campaign_id,
                "basic_metrics": basic_metrics,
                "advanced_analysis": advanced_analysis,
                "geographical_breakdown": analytics.geographical_data,
                "device_breakdown": analytics.device_data,
                "time_breakdown": analytics.time_data,
                "recommendations": recommendations,
                "benchmark_comparison": benchmark_comparison,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze email performance: {e}")
            raise
    
    async def optimize_template_with_ai(self,
                                      template_id: str,
                                      optimization_goals: List[str]) -> Dict[str, Any]:
        """Optimise un template avec l'IA"""
        
        if not self.ai_models_enabled:
            raise ValueError("AI models not enabled")
        
        try:
            template = self.templates.get(template_id)
            if not template:
                raise ValueError(f"Template not found: {template_id}")
            
            optimization_results = {}
            
            # Optimisation du sujet
            if "subject" in optimization_goals:
                optimized_subjects = await self._generate_subject_variants(
                    template.subject_template
                )
                optimization_results["subject_variants"] = optimized_subjects
            
            # Optimisation du contenu
            if "content" in optimization_goals:
                optimized_content = await self._optimize_content_structure(
                    template.html_content
                )
                optimization_results["content_optimization"] = optimized_content
            
            # Optimisation de l'engagement
            if "engagement" in optimization_goals:
                engagement_score = await self._predict_engagement_score(template)
                optimization_results["engagement_prediction"] = engagement_score
            
            # Recommandations personnalisées
            if "personalization" in optimization_goals:
                personalization_rules = await self._generate_personalization_rules(
                    template
                )
                optimization_results["personalization_rules"] = personalization_rules
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Failed to optimize template with AI: {e}")
            raise
    
    def get_template_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques des templates"""
        
        total_templates = len(self.templates)
        templates_by_type = {}
        templates_by_language = {}
        
        for template in self.templates.values():
            # Par type
            type_key = template.type.value
            templates_by_type[type_key] = templates_by_type.get(type_key, 0) + 1
            
            # Par langue
            lang_key = template.language
            templates_by_language[lang_key] = templates_by_language.get(lang_key, 0) + 1
        
        return {
            "total_templates": total_templates,
            "templates_by_type": templates_by_type,
            "templates_by_language": templates_by_language,
            "total_campaigns": len(self.campaigns),
            "ai_models_enabled": self.ai_models_enabled,
            "multi_tenant_enabled": self.multi_tenant,
            "cache_size": len(self.template_cache)
        }
    
    async def export_templates(self,
                             format: str = "json",
                             template_ids: Optional[List[str]] = None) -> str:
        """Exporte les templates dans le format spécifié"""
        
        try:
            templates_to_export = []
            
            if template_ids:
                templates_to_export = [
                    self.templates[tid] for tid in template_ids 
                    if tid in self.templates
                ]
            else:
                templates_to_export = list(self.templates.values())
            
            export_data = {
                "exported_at": datetime.utcnow().isoformat(),
                "total_templates": len(templates_to_export),
                "templates": []
            }
            
            for template in templates_to_export:
                template_data = {
                    "id": template.id,
                    "name": template.name,
                    "type": template.type.value,
                    "subject_template": template.subject_template,
                    "html_content": template.html_content,
                    "text_content": template.text_content,
                    "language": template.language,
                    "version": template.version,
                    "created_at": template.created_at.isoformat(),
                    "metadata": template.metadata
                }
                export_data["templates"].append(template_data)
            
            # Export selon le format
            if format.lower() == "json":
                return json.dumps(export_data, indent=2, ensure_ascii=False)
            elif format.lower() == "yaml":
                import yaml
                return yaml.dump(export_data, default_flow_style=False, allow_unicode=True)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Failed to export templates: {e}")
            raise
    
    async def cleanup(self):
        """Nettoyage des ressources"""
        
        # Nettoyage du cache
        self.template_cache.clear()
        
        # Sauvegarde des données importantes
        await self._save_analytics_to_disk()
        await self._save_campaigns_to_disk()
        
        logger.info("Email Template Manager cleaned up")

# ============================================================================
# Personalization Engine
# ============================================================================

class PersonalizationEngine:
    """Moteur de personnalisation avancé"""
    
    def __init__(self):
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.behavioral_patterns: Dict[str, Dict[str, Any]] = {}
        self.content_variants: Dict[str, List[str]] = {}
    
    async def analyze_user_behavior(self, user_email: str, interactions: List[Dict[str, Any]]):
        """Analyse le comportement utilisateur"""
        
        # Analyse des patterns d'interaction
        time_patterns = self._analyze_time_patterns(interactions)
        content_preferences = self._analyze_content_preferences(interactions)
        engagement_trends = self._analyze_engagement_trends(interactions)
        
        self.behavioral_patterns[user_email] = {
            "time_patterns": time_patterns,
            "content_preferences": content_preferences,
            "engagement_trends": engagement_trends,
            "last_analysis": datetime.utcnow()
        }
    
    async def generate_personalized_content(self,
                                          base_content: str,
                                          user_profile: Dict[str, Any]) -> str:
        """Génère du contenu personnalisé"""
        
        # Logique de personnalisation basée sur le profil utilisateur
        personalized_content = base_content
        
        # Personnalisation par préférences
        if user_profile.get("preferences"):
            personalized_content = self._apply_preference_personalization(
                personalized_content, user_profile["preferences"]
            )
        
        # Personnalisation par comportement
        if user_profile.get("behavior_data"):
            personalized_content = self._apply_behavioral_personalization(
                personalized_content, user_profile["behavior_data"]
            )
        
        return personalized_content

# ============================================================================
# Factory Functions
# ============================================================================

def create_email_template_manager(
    templates_dir: str,
    smtp_config: Optional[Dict[str, Any]] = None,
    enable_ai: bool = True
) -> AdvancedEmailTemplateManager:
    """Factory pour créer un gestionnaire de templates email"""
    
    return AdvancedEmailTemplateManager(
        templates_dir=templates_dir,
        smtp_config=smtp_config,
        ai_models_enabled=enable_ai
    )

# Export principal
__all__ = [
    "AdvancedEmailTemplateManager",
    "EmailTemplate",
    "EmailRecipient",
    "EmailCampaign",
    "EmailAnalytics",
    "EmailTemplateType",
    "EmailPriority",
    "EmailStatus",
    "PersonalizationLevel",
    "PersonalizationEngine",
    "create_email_template_manager"
]
