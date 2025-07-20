"""
Gestionnaire de templates avancé pour notifications
==================================================

Système ultra-sophistiqué de templates avec support multilingue,
A/B testing, versioning, et optimisations de performance.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union, Set
from pathlib import Path
from abc import ABC, abstractmethod
import re
from functools import lru_cache

import aiofiles
import yaml
from jinja2 import (
    Environment, FileSystemLoader, DictLoader, 
    Template, TemplateNotFound, select_autoescape
)
from jinja2.meta import find_undeclared_variables
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
import aioredis
from babel import Locale, dates, numbers
from babel.support import Translations
import markdown
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont
import qrcode

from .models import NotificationTemplate, NotificationMetrics
from .schemas import NotificationCreateSchema, NotificationTemplateSchema, ChannelTypeEnum


class TemplateError(Exception):
    """Exception pour les erreurs de template"""
    pass


class TemplateValidationError(TemplateError):
    """Erreur de validation de template"""
    pass


class TemplateRenderError(TemplateError):
    """Erreur de rendu de template"""
    pass


class BaseTemplateRenderer(ABC):
    """Classe de base pour les renderers de template"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    @abstractmethod
    async def render(
        self,
        template_content: str,
        data: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> str:
        """Rendre un template avec les données"""
        pass
    
    @abstractmethod
    def validate_template(self, template_content: str) -> List[str]:
        """Valider un template et retourner les erreurs"""
        pass


class JinjaTemplateRenderer(BaseTemplateRenderer):
    """Renderer Jinja2 avancé avec fonctions personnalisées"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Configuration Jinja2
        self.jinja_env = Environment(
            loader=DictLoader({}),  # Loader vide, on charge dynamiquement
            autoescape=select_autoescape(['html', 'xml']),
            enable_async=True,
            trim_blocks=True,
            lstrip_blocks=True,
            cache_size=400,  # Cache des templates compilés
        )
        
        # Ajouter des fonctions personnalisées
        self._setup_custom_functions()
        
        # Cache pour les templates compilés
        self._template_cache: Dict[str, Template] = {}
    
    def _setup_custom_functions(self):
        """Ajouter des fonctions personnalisées à Jinja2"""
        
        # Fonctions de formatage
        self.jinja_env.globals.update({
            'format_datetime': self._format_datetime,
            'format_number': self._format_number,
            'format_currency': self._format_currency,
            'format_duration': self._format_duration,
            'truncate_text': self._truncate_text,
            'markdown_to_html': self._markdown_to_html,
            'generate_qr_code': self._generate_qr_code,
            'create_button': self._create_button,
            'get_priority_color': self._get_priority_color,
            'get_priority_emoji': self._get_priority_emoji,
            'url_encode': self._url_encode,
            'base64_encode': self._base64_encode,
            'hash_text': self._hash_text,
        })
        
        # Filtres personnalisés
        self.jinja_env.filters.update({
            'smart_truncate': self._smart_truncate,
            'remove_html': self._remove_html,
            'highlight_keywords': self._highlight_keywords,
            'format_list': self._format_list,
            'conditional_format': self._conditional_format,
        })
        
        # Tests personnalisés
        self.jinja_env.tests.update({
            'high_priority': lambda x: x in ['high', 'critical', 'emergency'],
            'valid_email': lambda x: re.match(r'^[^@]+@[^@]+\.[^@]+$', str(x)) is not None,
            'valid_url': lambda x: re.match(r'^https?://', str(x)) is not None,
        })
    
    async def render(
        self,
        template_content: str,
        data: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> str:
        """Rendre un template avec cache intelligent"""
        
        # Créer une clé de cache basée sur le contenu du template
        cache_key = hash(template_content)
        
        # Vérifier le cache
        if cache_key not in self._template_cache:
            try:
                self._template_cache[cache_key] = self.jinja_env.from_string(template_content)
            except Exception as e:
                raise TemplateRenderError(f"Erreur de compilation du template: {e}")
        
        template = self._template_cache[cache_key]
        
        # Préparer le contexte complet
        render_context = {
            'now': datetime.now(timezone.utc),
            'today': datetime.now(timezone.utc).date(),
            'locale': context.get('locale', 'en') if context else 'en',
            'timezone': context.get('timezone', 'UTC') if context else 'UTC',
            **data
        }
        
        if context:
            render_context.update(context)
        
        try:
            # Rendu asynchrone
            rendered = await template.render_async(**render_context)
            return rendered.strip()
        except Exception as e:
            raise TemplateRenderError(f"Erreur de rendu du template: {e}")
    
    def validate_template(self, template_content: str) -> List[str]:
        """Valider un template et retourner les erreurs"""
        errors = []
        
        try:
            # Tenter de compiler le template
            template = self.jinja_env.from_string(template_content)
            
            # Vérifier les variables non déclarées
            ast = self.jinja_env.parse(template_content)
            undeclared = find_undeclared_variables(ast)
            
            # Variables système autorisées
            allowed_globals = {
                'now', 'today', 'locale', 'timezone',
                'format_datetime', 'format_number', 'format_currency',
                'format_duration', 'truncate_text', 'markdown_to_html',
                'generate_qr_code', 'create_button', 'get_priority_color',
                'get_priority_emoji', 'url_encode', 'base64_encode', 'hash_text'
            }
            
            undefined_vars = undeclared - allowed_globals
            if undefined_vars:
                errors.append(f"Variables non définies: {', '.join(undefined_vars)}")
            
            # Vérifier la syntaxe avec des données de test
            test_data = self._generate_test_data()
            try:
                template.render(**test_data)
            except Exception as e:
                errors.append(f"Erreur de rendu test: {e}")
        
        except Exception as e:
            errors.append(f"Erreur de syntaxe: {e}")
        
        return errors
    
    def _generate_test_data(self) -> Dict[str, Any]:
        """Générer des données de test pour validation"""
        return {
            'title': 'Test Notification',
            'message': 'This is a test message for template validation.',
            'priority': 'normal',
            'recipient_name': 'John Doe',
            'recipient_email': 'john.doe@example.com',
            'timestamp': datetime.now(timezone.utc),
            'correlation_id': 'test-correlation-123',
            'source_system': 'test-system',
            'metadata': {'key': 'value'},
            'tags': ['test', 'validation'],
            'now': datetime.now(timezone.utc),
            'today': datetime.now(timezone.utc).date(),
            'locale': 'en',
            'timezone': 'UTC'
        }
    
    # Fonctions personnalisées pour templates
    
    def _format_datetime(self, dt: datetime, format_str: str = 'medium', locale: str = 'en') -> str:
        """Formater une date/heure selon la locale"""
        if not isinstance(dt, datetime):
            return str(dt)
        
        try:
            babel_locale = Locale(locale)
            if format_str == 'short':
                return dates.format_datetime(dt, format='short', locale=babel_locale)
            elif format_str == 'medium':
                return dates.format_datetime(dt, format='medium', locale=babel_locale)
            elif format_str == 'long':
                return dates.format_datetime(dt, format='long', locale=babel_locale)
            else:
                return dt.strftime(format_str)
        except:
            return dt.strftime('%Y-%m-%d %H:%M:%S')
    
    def _format_number(self, number: Union[int, float], locale: str = 'en') -> str:
        """Formater un nombre selon la locale"""
        try:
            babel_locale = Locale(locale)
            return numbers.format_number(number, locale=babel_locale)
        except:
            return str(number)
    
    def _format_currency(self, amount: float, currency: str = 'USD', locale: str = 'en') -> str:
        """Formater une devise selon la locale"""
        try:
            babel_locale = Locale(locale)
            return numbers.format_currency(amount, currency, locale=babel_locale)
        except:
            return f"{amount} {currency}"
    
    def _format_duration(self, seconds: int) -> str:
        """Formater une durée en format lisible"""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds // 60}m {seconds % 60}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"
    
    def _truncate_text(self, text: str, length: int = 100, suffix: str = '...') -> str:
        """Tronquer un texte intelligemment"""
        if len(text) <= length:
            return text
        
        # Chercher le dernier espace avant la limite
        truncated = text[:length]
        last_space = truncated.rfind(' ')
        if last_space > length * 0.8:  # Au moins 80% de la longueur cible
            truncated = truncated[:last_space]
        
        return truncated + suffix
    
    def _markdown_to_html(self, md_text: str) -> str:
        """Convertir Markdown en HTML"""
        try:
            html = markdown.markdown(
                md_text,
                extensions=['extra', 'codehilite', 'toc']
            )
            return html
        except:
            return md_text
    
    def _generate_qr_code(self, data: str, size: int = 200) -> str:
        """Générer un QR code en base64"""
        try:
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(data)
            qr.make(fit=True)
            
            img = qr.make_image(fill_color="black", back_color="white")
            img = img.resize((size, size))
            
            # Convertir en base64
            import io
            import base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/png;base64,{img_str}"
        except:
            return ""
    
    def _create_button(
        self,
        text: str,
        url: str,
        style: str = 'primary',
        size: str = 'medium'
    ) -> str:
        """Créer un bouton HTML stylé"""
        
        styles = {
            'primary': 'background-color: #1DB954; color: white;',
            'secondary': 'background-color: #6c757d; color: white;',
            'success': 'background-color: #28a745; color: white;',
            'danger': 'background-color: #dc3545; color: white;',
            'warning': 'background-color: #ffc107; color: black;',
            'info': 'background-color: #17a2b8; color: white;',
        }
        
        sizes = {
            'small': 'padding: 8px 12px; font-size: 12px;',
            'medium': 'padding: 10px 16px; font-size: 14px;',
            'large': 'padding: 12px 20px; font-size: 16px;',
        }
        
        base_style = (
            'display: inline-block; text-decoration: none; '
            'border-radius: 4px; border: none; cursor: pointer; '
            'text-align: center; font-weight: bold;'
        )
        
        button_style = base_style + styles.get(style, styles['primary']) + sizes.get(size, sizes['medium'])
        
        return f'<a href="{url}" style="{button_style}">{text}</a>'
    
    def _get_priority_color(self, priority: str) -> str:
        """Obtenir la couleur selon la priorité"""
        colors = {
            'low': '#6c757d',
            'normal': '#007bff',
            'high': '#ffc107',
            'critical': '#fd7e14',
            'emergency': '#dc3545'
        }
        return colors.get(priority.lower(), '#007bff')
    
    def _get_priority_emoji(self, priority: str) -> str:
        """Obtenir l'emoji selon la priorité"""
        emojis = {
            'low': 'ℹ️',
            'normal': '🔔',
            'high': '⚠️',
            'critical': '🚨',
            'emergency': '🔥'
        }
        return emojis.get(priority.lower(), '🔔')
    
    def _url_encode(self, text: str) -> str:
        """Encoder une URL"""
        import urllib.parse
        return urllib.parse.quote(text)
    
    def _base64_encode(self, text: str) -> str:
        """Encoder en base64"""
        import base64
        return base64.b64encode(text.encode()).decode()
    
    def _hash_text(self, text: str, algorithm: str = 'md5') -> str:
        """Hasher un texte"""
        import hashlib
        if algorithm == 'md5':
            return hashlib.md5(text.encode()).hexdigest()
        elif algorithm == 'sha256':
            return hashlib.sha256(text.encode()).hexdigest()
        else:
            return text
    
    # Filtres personnalisés
    
    def _smart_truncate(self, text: str, length: int = 100) -> str:
        """Troncature intelligente au niveau des mots"""
        if len(text) <= length:
            return text
        
        # Chercher le dernier espace
        truncated = text[:length]
        last_space = truncated.rfind(' ')
        if last_space > 0:
            truncated = truncated[:last_space]
        
        return truncated + '...'
    
    def _remove_html(self, text: str) -> str:
        """Supprimer les tags HTML"""
        try:
            soup = BeautifulSoup(text, 'html.parser')
            return soup.get_text()
        except:
            return text
    
    def _highlight_keywords(self, text: str, keywords: List[str]) -> str:
        """Surligner des mots-clés"""
        for keyword in keywords:
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            text = pattern.sub(f'<mark>{keyword}</mark>', text)
        return text
    
    def _format_list(self, items: List[str], style: str = 'bullet') -> str:
        """Formater une liste"""
        if not items:
            return ''
        
        if style == 'bullet':
            return '\n'.join([f'• {item}' for item in items])
        elif style == 'numbered':
            return '\n'.join([f'{i+1}. {item}' for i, item in enumerate(items)])
        elif style == 'comma':
            return ', '.join(items)
        else:
            return '\n'.join(items)
    
    def _conditional_format(self, value: Any, condition: str, true_format: str, false_format: str = '') -> str:
        """Formatage conditionnel"""
        try:
            # Évaluer la condition de manière sécurisée
            if condition == 'empty':
                result = not value
            elif condition == 'positive':
                result = float(value) > 0
            elif condition == 'negative':
                result = float(value) < 0
            elif condition == 'zero':
                result = float(value) == 0
            else:
                result = bool(value)
            
            return true_format if result else false_format
        except:
            return str(value)


class NotificationTemplateManager:
    """Gestionnaire principal des templates de notification"""
    
    def __init__(
        self,
        db_session: AsyncSession,
        redis_client: aioredis.Redis,
        config: Dict[str, Any]
    ):
        self.db = db_session
        self.redis = redis_client
        self.config = config
        self.logger = logging.getLogger("TemplateManager")
        
        # Renderers par type de contenu
        self.renderers = {
            'jinja2': JinjaTemplateRenderer(config),
            'markdown': JinjaTemplateRenderer(config),  # Markdown avec Jinja2
            'html': JinjaTemplateRenderer(config),
            'text': JinjaTemplateRenderer(config),
        }
        
        # Cache des templates compilés
        self._template_cache: Dict[str, NotificationTemplate] = {}
        self._cache_ttl = config.get('template_cache_ttl', 3600)  # 1 heure
        
        # Système de fallback
        self._fallback_templates = {}
        
        # A/B Testing
        self._ab_test_weights: Dict[str, float] = {}
    
    async def initialize(self):
        """Initialisation du gestionnaire"""
        await self._load_fallback_templates()
        await self._initialize_ab_test_weights()
    
    async def _load_fallback_templates(self):
        """Charger les templates de fallback depuis les fichiers"""
        fallback_dir = Path(self.config.get('fallback_templates_dir', 'templates/fallback'))
        
        if fallback_dir.exists():
            for template_file in fallback_dir.glob('*.yaml'):
                try:
                    async with aiofiles.open(template_file, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        template_data = yaml.safe_load(content)
                        
                        key = f"{template_data['channel_type']}_{template_data['locale']}"
                        self._fallback_templates[key] = template_data
                        
                except Exception as e:
                    self.logger.error(f"Erreur chargement template fallback {template_file}: {e}")
    
    async def _initialize_ab_test_weights(self):
        """Initialiser les poids pour A/B testing"""
        # Récupérer les templates avec A/B testing actif
        query = select(NotificationTemplate).where(
            and_(
                NotificationTemplate.is_active == True,
                NotificationTemplate.ab_test_group.isnot(None)
            )
        )
        
        result = await self.db.execute(query)
        templates = result.scalars().all()
        
        # Grouper par groupe A/B
        ab_groups = {}
        for template in templates:
            group = template.ab_test_group
            if group not in ab_groups:
                ab_groups[group] = []
            ab_groups[group].append(template)
        
        # Calculer les poids normalisés
        for group, group_templates in ab_groups.items():
            total_weight = sum(t.ab_test_weight for t in group_templates)
            if total_weight > 0:
                for template in group_templates:
                    normalized_weight = template.ab_test_weight / total_weight
                    self._ab_test_weights[str(template.id)] = normalized_weight
    
    async def get_template(
        self,
        template_id: Optional[str] = None,
        channel_type: ChannelTypeEnum = None,
        locale: str = 'en',
        tenant_id: str = None,
        context: Dict[str, Any] = None
    ) -> Optional[NotificationTemplate]:
        """Récupérer un template avec cache et fallback"""
        
        # Cache key
        cache_key = f"template:{template_id or 'auto'}:{channel_type}:{locale}:{tenant_id}"
        
        # Vérifier le cache Redis
        cached = await self.redis.get(cache_key)
        if cached:
            try:
                template_data = json.loads(cached)
                return self._deserialize_template(template_data)
            except:
                pass
        
        # Récupérer depuis la base de données
        template = None
        
        if template_id:
            # Template spécifique
            template = await self._get_template_by_id(template_id, tenant_id)
        else:
            # Auto-sélection basée sur le canal et la locale
            template = await self._auto_select_template(channel_type, locale, tenant_id, context)
        
        # Fallback si aucun template trouvé
        if not template:
            template = await self._get_fallback_template(channel_type, locale)
        
        # Mettre en cache
        if template:
            serialized = self._serialize_template(template)
            await self.redis.setex(cache_key, self._cache_ttl, json.dumps(serialized))
        
        return template
    
    async def _get_template_by_id(self, template_id: str, tenant_id: str) -> Optional[NotificationTemplate]:
        """Récupérer un template par ID"""
        query = select(NotificationTemplate).where(
            and_(
                NotificationTemplate.id == template_id,
                NotificationTemplate.tenant_id == tenant_id,
                NotificationTemplate.is_active == True
            )
        )
        
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def _auto_select_template(
        self,
        channel_type: ChannelTypeEnum,
        locale: str,
        tenant_id: str,
        context: Dict[str, Any] = None
    ) -> Optional[NotificationTemplate]:
        """Sélection automatique de template avec A/B testing"""
        
        # Requête de base
        query = select(NotificationTemplate).where(
            and_(
                NotificationTemplate.channel_type == channel_type.value,
                NotificationTemplate.tenant_id == tenant_id,
                NotificationTemplate.is_active == True
            )
        )
        
        # Filtrer par locale avec fallback
        locale_filter = or_(
            NotificationTemplate.locale == locale,
            NotificationTemplate.locale == locale[:2],  # Langue sans région
            NotificationTemplate.locale == 'en'  # Fallback anglais
        )
        query = query.where(locale_filter)
        
        # Ordonner par priorité (locale exacte d'abord, puis usage)
        query = query.order_by(
            NotificationTemplate.locale == locale,  # Locale exacte en premier
            NotificationTemplate.usage_count.desc(),
            NotificationTemplate.success_rate.desc()
        )
        
        result = await self.db.execute(query)
        candidates = result.scalars().all()
        
        if not candidates:
            return None
        
        # A/B Testing si applicable
        ab_candidates = [t for t in candidates if t.ab_test_group]
        if ab_candidates and context and context.get('user_id'):
            selected = await self._select_ab_test_template(ab_candidates, context['user_id'])
            if selected:
                return selected
        
        # Retourner le meilleur candidat
        return candidates[0]
    
    async def _select_ab_test_template(
        self,
        templates: List[NotificationTemplate],
        user_id: str
    ) -> Optional[NotificationTemplate]:
        """Sélectionner un template pour A/B testing"""
        
        # Grouper par groupe A/B
        ab_groups = {}
        for template in templates:
            group = template.ab_test_group
            if group not in ab_groups:
                ab_groups[group] = []
            ab_groups[group].append(template)
        
        # Sélectionner un groupe (déterministe basé sur user_id)
        import hashlib
        if len(ab_groups) > 1:
            hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            group_index = hash_value % len(ab_groups)
            selected_group = list(ab_groups.keys())[group_index]
        else:
            selected_group = list(ab_groups.keys())[0]
        
        group_templates = ab_groups[selected_group]
        
        # Sélectionner un template dans le groupe selon les poids
        import random
        weights = [self._ab_test_weights.get(str(t.id), 1.0) for t in group_templates]
        selected = random.choices(group_templates, weights=weights)[0]
        
        return selected
    
    async def _get_fallback_template(
        self,
        channel_type: ChannelTypeEnum,
        locale: str
    ) -> Optional[NotificationTemplate]:
        """Récupérer un template de fallback"""
        
        # Chercher dans les fallbacks chargés
        keys_to_try = [
            f"{channel_type.value}_{locale}",
            f"{channel_type.value}_{locale[:2]}",
            f"{channel_type.value}_en",
            f"default_{locale}",
            f"default_en"
        ]
        
        for key in keys_to_try:
            if key in self._fallback_templates:
                template_data = self._fallback_templates[key]
                
                # Créer un template temporaire
                template = NotificationTemplate(
                    id=f"fallback_{key}",
                    name=f"Fallback {channel_type.value} {locale}",
                    channel_type=channel_type.value,
                    locale=locale,
                    body_template=template_data.get('body_template', '{{message}}'),
                    subject_template=template_data.get('subject_template'),
                    html_template=template_data.get('html_template'),
                    is_active=True,
                    tenant_id='fallback'
                )
                
                return template
        
        # Template de fallback ultime
        return NotificationTemplate(
            id="ultimate_fallback",
            name="Ultimate Fallback",
            channel_type=channel_type.value,
            locale=locale,
            body_template="{{title}}\n\n{{message}}",
            is_active=True,
            tenant_id='fallback'
        )
    
    async def render_template(
        self,
        template: NotificationTemplate,
        data: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, str]:
        """Rendre un template complet"""
        
        # Déterminer le renderer à utiliser
        renderer_type = self.config.get('default_renderer', 'jinja2')
        renderer = self.renderers.get(renderer_type, self.renderers['jinja2'])
        
        # Contexte de rendu
        render_context = {
            'locale': template.locale,
            'channel_type': template.channel_type,
            'template_version': template.version,
            **(context or {})
        }
        
        # Rendre chaque partie du template
        result = {}
        
        try:
            # Sujet
            if template.subject_template:
                result['subject'] = await renderer.render(
                    template.subject_template,
                    data,
                    render_context
                )
            
            # Corps principal
            if template.body_template:
                result['body'] = await renderer.render(
                    template.body_template,
                    data,
                    render_context
                )
            
            # HTML
            if template.html_template:
                result['html'] = await renderer.render(
                    template.html_template,
                    data,
                    render_context
                )
            
            # Métadonnées
            if template.metadata_template:
                metadata_json = await renderer.render(
                    json.dumps(template.metadata_template),
                    data,
                    render_context
                )
                result['metadata'] = json.loads(metadata_json)
            
            # Mettre à jour les statistiques
            await self._update_template_stats(template.id, success=True)
            
            return result
        
        except Exception as e:
            self.logger.error(f"Erreur de rendu template {template.id}: {e}")
            await self._update_template_stats(template.id, success=False)
            raise TemplateRenderError(f"Erreur de rendu: {e}")
    
    async def _update_template_stats(self, template_id: str, success: bool):
        """Mettre à jour les statistiques du template"""
        
        # Requête atomique pour mettre à jour les stats
        if success:
            query = f"""
                UPDATE notification_templates 
                SET usage_count = usage_count + 1,
                    success_rate = (success_rate * usage_count + 1) / (usage_count + 1)
                WHERE id = '{template_id}'
            """
        else:
            query = f"""
                UPDATE notification_templates 
                SET usage_count = usage_count + 1,
                    success_rate = (success_rate * usage_count) / (usage_count + 1)
                WHERE id = '{template_id}'
            """
        
        try:
            await self.db.execute(query)
            await self.db.commit()
        except Exception as e:
            self.logger.error(f"Erreur mise à jour stats template: {e}")
    
    def _serialize_template(self, template: NotificationTemplate) -> Dict[str, Any]:
        """Sérialiser un template pour le cache"""
        return {
            'id': str(template.id),
            'name': template.name,
            'channel_type': template.channel_type,
            'version': template.version,
            'locale': template.locale,
            'subject_template': template.subject_template,
            'body_template': template.body_template,
            'html_template': template.html_template,
            'metadata_template': template.metadata_template,
            'ab_test_group': template.ab_test_group,
            'ab_test_weight': template.ab_test_weight,
            'is_active': template.is_active,
            'tenant_id': template.tenant_id
        }
    
    def _deserialize_template(self, data: Dict[str, Any]) -> NotificationTemplate:
        """Désérialiser un template depuis le cache"""
        template = NotificationTemplate()
        template.id = data['id']
        template.name = data['name']
        template.channel_type = data['channel_type']
        template.version = data['version']
        template.locale = data['locale']
        template.subject_template = data['subject_template']
        template.body_template = data['body_template']
        template.html_template = data['html_template']
        template.metadata_template = data['metadata_template']
        template.ab_test_group = data['ab_test_group']
        template.ab_test_weight = data['ab_test_weight']
        template.is_active = data['is_active']
        template.tenant_id = data['tenant_id']
        return template
    
    async def validate_template(self, template_data: NotificationTemplateSchema) -> List[str]:
        """Valider un template avant sauvegarde"""
        errors = []
        
        # Validation syntaxique
        renderer = self.renderers.get('jinja2')
        
        if template_data.body_template:
            body_errors = renderer.validate_template(template_data.body_template)
            errors.extend([f"Body: {error}" for error in body_errors])
        
        if template_data.subject_template:
            subject_errors = renderer.validate_template(template_data.subject_template)
            errors.extend([f"Subject: {error}" for error in subject_errors])
        
        if template_data.html_template:
            html_errors = renderer.validate_template(template_data.html_template)
            errors.extend([f"HTML: {error}" for error in html_errors])
        
        # Validation des variables requises
        if template_data.required_variables:
            for template_part in [template_data.body_template, template_data.subject_template]:
                if template_part:
                    template_vars = set(re.findall(r'\{\{(\w+)\}\}', template_part))
                    missing_required = template_data.required_variables - template_vars
                    if missing_required:
                        errors.append(f"Variables requises manquantes: {missing_required}")
        
        return errors
    
    async def create_template(self, template_data: NotificationTemplateSchema) -> NotificationTemplate:
        """Créer un nouveau template"""
        
        # Validation
        validation_errors = await self.validate_template(template_data)
        if validation_errors:
            raise TemplateValidationError(f"Erreurs de validation: {validation_errors}")
        
        # Créer l'enregistrement
        template = NotificationTemplate(
            **template_data.dict(exclude={'id', 'created_at', 'updated_at'})
        )
        template.created_at = datetime.now(timezone.utc)
        
        self.db.add(template)
        await self.db.commit()
        await self.db.refresh(template)
        
        # Invalider le cache
        await self._invalidate_template_cache(template.tenant_id, template.channel_type)
        
        return template
    
    async def _invalidate_template_cache(self, tenant_id: str, channel_type: str):
        """Invalider le cache des templates"""
        pattern = f"template:*:{channel_type}:*:{tenant_id}"
        keys = await self.redis.keys(pattern)
        if keys:
            await self.redis.delete(*keys)
