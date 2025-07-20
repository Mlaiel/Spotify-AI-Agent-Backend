"""
Slack Template Engine - Moteur de templates dynamiques pour alertes Slack
Support multi-langue et templates conditionnels avec intelligence contextuelle
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from jinja2 import Environment, FileSystemLoader, Template, select_autoescape
import yaml
from babel import Locale
from babel.dates import format_datetime
from babel.numbers import format_decimal


class TemplateType(str, Enum):
    """Types de templates disponibles"""
    STANDARD = "standard"
    CRITICAL = "critical"
    SUMMARY = "summary"
    RESOLUTION = "resolution"
    ACKNOWLEDGMENT = "acknowledgment"
    ESCALATION = "escalation"
    MAINTENANCE = "maintenance"
    CUSTOM = "custom"


class MessageFormat(str, Enum):
    """Formats de message Slack"""
    SIMPLE = "simple"
    RICH = "rich"
    ATTACHMENT = "attachment"
    BLOCKS = "blocks"
    INTERACTIVE = "interactive"


@dataclass
class TemplateContext:
    """Contexte pour le rendu des templates"""
    alert_data: Dict[str, Any]
    tenant_config: Dict[str, Any]
    environment: str
    language: str = "fr"
    timezone: str = "Europe/Paris"
    custom_variables: Dict[str, Any] = None

    def __post_init__(self):
        if self.custom_variables is None:
            self.custom_variables = {}


class SlackTemplateEngine:
    """
    Moteur de templates Slack avancé avec:
    - Support multi-langue (FR/EN/DE)
    - Templates conditionnels intelligents
    - Formatage adaptatif selon le contexte
    - Cache optimisé pour les performances
    - Validation et sécurisation des templates
    """

    def __init__(self, templates_dir: Optional[str] = None):
        self.templates_dir = templates_dir or self._get_default_templates_dir()
        self.jinja_env = None
        self.template_cache = {}
        self.config_cache = {}
        self.logger = logging.getLogger(__name__)
        
        # Configuration par défaut
        self.default_config = {
            "languages": ["fr", "en", "de"],
            "default_language": "fr",
            "timezone": "Europe/Paris",
            "date_format": "medium",
            "number_format": "standard"
        }

    async def initialize(self):
        """Initialise le moteur de templates"""
        try:
            # Configuration de Jinja2
            self.jinja_env = Environment(
                loader=FileSystemLoader(self.templates_dir),
                autoescape=select_autoescape(['html', 'xml']),
                enable_async=True,
                trim_blocks=True,
                lstrip_blocks=True
            )
            
            # Ajout des filtres personnalisés
            self._register_custom_filters()
            
            # Chargement des templates de base
            await self._load_base_templates()
            
            # Validation des templates
            await self._validate_templates()
            
            self.logger.info("SlackTemplateEngine initialisé avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation du moteur de templates: {e}")
            raise

    async def render_template(
        self,
        template_type: TemplateType,
        context: TemplateContext,
        format_type: MessageFormat = MessageFormat.BLOCKS
    ) -> Dict[str, Any]:
        """
        Rend un template Slack avec le contexte fourni
        
        Args:
            template_type: Type de template à utiliser
            context: Contexte de rendu
            format_type: Format du message Slack
            
        Returns:
            Message Slack formaté
        """
        try:
            # Sélection du template approprié
            template_name = self._get_template_name(template_type, context.language, format_type)
            template = await self._get_template(template_name)
            
            # Enrichissement du contexte
            enriched_context = await self._enrich_context(context)
            
            # Rendu du template
            rendered_content = await template.render_async(**enriched_context)
            
            # Parse du JSON résultant
            if format_type in [MessageFormat.BLOCKS, MessageFormat.ATTACHMENT]:
                message = json.loads(rendered_content)
            else:
                message = {"text": rendered_content}
            
            # Validation du message Slack
            await self._validate_slack_message(message)
            
            # Ajout des métadonnées
            message["metadata"] = {
                "template_type": template_type.value,
                "language": context.language,
                "format": format_type.value,
                "generated_at": datetime.utcnow().isoformat()
            }
            
            return message
            
        except Exception as e:
            self.logger.error(f"Erreur lors du rendu du template {template_type}: {e}")
            # Fallback vers template minimal
            return await self._render_fallback_template(context)

    async def render_custom_template(
        self,
        template_content: str,
        context: TemplateContext,
        format_type: MessageFormat = MessageFormat.BLOCKS
    ) -> Dict[str, Any]:
        """Rend un template personnalisé"""
        try:
            # Création du template à la volée
            template = self.jinja_env.from_string(template_content)
            
            # Enrichissement du contexte
            enriched_context = await self._enrich_context(context)
            
            # Rendu
            rendered_content = await template.render_async(**enriched_context)
            
            if format_type in [MessageFormat.BLOCKS, MessageFormat.ATTACHMENT]:
                message = json.loads(rendered_content)
            else:
                message = {"text": rendered_content}
            
            await self._validate_slack_message(message)
            
            return message
            
        except Exception as e:
            self.logger.error(f"Erreur lors du rendu du template personnalisé: {e}")
            return await self._render_fallback_template(context)

    async def get_available_templates(self, language: str = "fr") -> List[Dict[str, Any]]:
        """Récupère la liste des templates disponibles"""
        templates = []
        
        for template_type in TemplateType:
            for format_type in MessageFormat:
                template_name = self._get_template_name(template_type, language, format_type)
                template_path = Path(self.templates_dir) / template_name
                
                if template_path.exists():
                    templates.append({
                        "type": template_type.value,
                        "format": format_type.value,
                        "language": language,
                        "name": template_name,
                        "path": str(template_path)
                    })
        
        return templates

    async def validate_template_syntax(self, template_content: str) -> Dict[str, Any]:
        """Valide la syntaxe d'un template"""
        try:
            # Test de compilation
            template = self.jinja_env.from_string(template_content)
            
            # Test de rendu avec données minimales
            test_context = {
                "alert": {
                    "title": "Test Alert",
                    "description": "Test Description",
                    "severity": "warning"
                },
                "tenant": {"name": "Test Tenant"},
                "timestamp": datetime.utcnow()
            }
            
            rendered = await template.render_async(**test_context)
            
            # Validation JSON si nécessaire
            if template_content.strip().startswith('{'):
                json.loads(rendered)
            
            return {
                "valid": True,
                "message": "Template valide",
                "rendered_sample": rendered[:200] + "..." if len(rendered) > 200 else rendered
            }
            
        except Exception as e:
            return {
                "valid": False,
                "message": f"Erreur de syntaxe: {str(e)}",
                "error_type": type(e).__name__
            }

    def _get_default_templates_dir(self) -> str:
        """Récupère le répertoire par défaut des templates"""
        current_dir = Path(__file__).parent
        return str(current_dir / "templates")

    def _get_template_name(
        self,
        template_type: TemplateType,
        language: str,
        format_type: MessageFormat
    ) -> str:
        """Génère le nom du fichier template"""
        return f"{template_type.value}_{language}_{format_type.value}.j2"

    async def _get_template(self, template_name: str) -> Template:
        """Récupère un template avec cache"""
        if template_name not in self.template_cache:
            try:
                template = self.jinja_env.get_template(template_name)
                self.template_cache[template_name] = template
            except Exception as e:
                self.logger.warning(f"Template {template_name} non trouvé: {e}")
                # Fallback vers template par défaut
                fallback_name = f"standard_fr_blocks.j2"
                if fallback_name in self.template_cache:
                    return self.template_cache[fallback_name]
                else:
                    template = self.jinja_env.get_template(fallback_name)
                    self.template_cache[fallback_name] = template
                    return template
        
        return self.template_cache[template_name]

    async def _enrich_context(self, context: TemplateContext) -> Dict[str, Any]:
        """Enrichit le contexte de rendu"""
        enriched = {
            # Données de base
            "alert": context.alert_data,
            "tenant": context.tenant_config,
            "environment": context.environment,
            "language": context.language,
            
            # Fonctions utilitaires
            "now": datetime.utcnow(),
            "format_date": lambda dt: self._format_date(dt, context.language, context.timezone),
            "format_number": lambda num: self._format_number(num, context.language),
            "get_severity_color": self._get_severity_color,
            "get_severity_emoji": self._get_severity_emoji,
            "get_priority_level": self._get_priority_level,
            
            # Variables personnalisées
            **context.custom_variables
        }
        
        # Ajout des traductions
        translations = await self._load_translations(context.language)
        enriched["t"] = translations
        enriched["_"] = lambda key, **kwargs: self._translate(key, translations, **kwargs)
        
        return enriched

    def _register_custom_filters(self):
        """Enregistre les filtres personnalisés Jinja2"""
        
        @self.jinja_env.filter('severity_color')
        def severity_color_filter(severity: str) -> str:
            return self._get_severity_color(severity)
        
        @self.jinja_env.filter('severity_emoji')
        def severity_emoji_filter(severity: str) -> str:
            return self._get_severity_emoji(severity)
        
        @self.jinja_env.filter('truncate_smart')
        def truncate_smart_filter(text: str, length: int = 100) -> str:
            if len(text) <= length:
                return text
            return text[:length-3] + "..."
        
        @self.jinja_env.filter('format_duration')
        def format_duration_filter(seconds: int) -> str:
            if seconds < 60:
                return f"{seconds}s"
            elif seconds < 3600:
                return f"{seconds//60}m {seconds%60}s"
            else:
                hours = seconds // 3600
                minutes = (seconds % 3600) // 60
                return f"{hours}h {minutes}m"
        
        @self.jinja_env.filter('json_pretty')
        def json_pretty_filter(obj: Any) -> str:
            return json.dumps(obj, indent=2, ensure_ascii=False)

    def _get_severity_color(self, severity: str) -> str:
        """Retourne la couleur associée à une sévérité"""
        colors = {
            "info": "#36a64f",      # Vert
            "warning": "#ff9500",   # Orange  
            "error": "#ff0000",     # Rouge
            "critical": "#8B0000",  # Rouge foncé
            "emergency": "#800080"  # Violet
        }
        return colors.get(severity.lower(), "#808080")

    def _get_severity_emoji(self, severity: str) -> str:
        """Retourne l'emoji associé à une sévérité"""
        emojis = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "critical": "🔥",
            "emergency": "🚨"
        }
        return emojis.get(severity.lower(), "📌")

    def _get_priority_level(self, severity: str) -> int:
        """Retourne le niveau de priorité numérique"""
        levels = {
            "info": 1,
            "warning": 2,
            "error": 3,
            "critical": 4,
            "emergency": 5
        }
        return levels.get(severity.lower(), 1)

    def _format_date(self, dt: datetime, language: str, timezone: str) -> str:
        """Formate une date selon la locale"""
        try:
            locale = Locale(language)
            return format_datetime(dt, locale=locale, tzinfo=timezone)
        except:
            return dt.strftime("%Y-%m-%d %H:%M:%S")

    def _format_number(self, number: Union[int, float], language: str) -> str:
        """Formate un nombre selon la locale"""
        try:
            locale = Locale(language)
            return format_decimal(number, locale=locale)
        except:
            return str(number)

    async def _load_translations(self, language: str) -> Dict[str, str]:
        """Charge les traductions pour une langue"""
        cache_key = f"translations_{language}"
        
        if cache_key not in self.config_cache:
            translations_file = Path(self.templates_dir) / "translations" / f"{language}.yaml"
            
            try:
                with open(translations_file, 'r', encoding='utf-8') as f:
                    translations = yaml.safe_load(f)
                self.config_cache[cache_key] = translations
            except FileNotFoundError:
                # Fallback vers traductions par défaut
                self.config_cache[cache_key] = await self._get_default_translations()
        
        return self.config_cache[cache_key]

    def _translate(self, key: str, translations: Dict[str, str], **kwargs) -> str:
        """Traduit une clé avec interpolation"""
        text = translations.get(key, key)
        if kwargs:
            try:
                return text.format(**kwargs)
            except:
                return text
        return text

    async def _get_default_translations(self) -> Dict[str, str]:
        """Retourne les traductions par défaut"""
        return {
            "alert_title": "Alerte {severity}",
            "alert_description": "Description de l'alerte",
            "service": "Service",
            "component": "Composant",
            "environment": "Environnement",
            "timestamp": "Horodatage",
            "view_details": "Voir les détails",
            "acknowledge": "Acquitter",
            "resolve": "Résoudre",
            "escalate": "Escalader"
        }

    async def _load_base_templates(self):
        """Charge les templates de base dans le cache"""
        base_templates = [
            "standard_fr_blocks.j2",
            "critical_fr_blocks.j2",
            "summary_fr_blocks.j2"
        ]
        
        for template_name in base_templates:
            try:
                template = self.jinja_env.get_template(template_name)
                self.template_cache[template_name] = template
            except Exception as e:
                self.logger.warning(f"Template de base {template_name} non trouvé: {e}")

    async def _validate_templates(self):
        """Valide tous les templates disponibles"""
        templates_dir = Path(self.templates_dir)
        if not templates_dir.exists():
            # Création du répertoire et des templates par défaut
            await self._create_default_templates()
            return
        
        for template_file in templates_dir.glob("*.j2"):
            try:
                self.jinja_env.get_template(template_file.name)
            except Exception as e:
                self.logger.warning(f"Template invalide {template_file.name}: {e}")

    async def _validate_slack_message(self, message: Dict[str, Any]):
        """Valide un message Slack"""
        # Validation basique de la structure
        if not isinstance(message, dict):
            raise ValueError("Le message doit être un dictionnaire")
        
        # Vérification des limites Slack
        if "text" in message and len(message["text"]) > 40000:
            raise ValueError("Le texte dépasse la limite Slack de 40000 caractères")
        
        if "blocks" in message and len(message["blocks"]) > 50:
            raise ValueError("Trop de blocs (maximum 50)")

    async def _render_fallback_template(self, context: TemplateContext) -> Dict[str, Any]:
        """Rend un template de fallback minimal"""
        alert = context.alert_data
        
        return {
            "text": f"🚨 *Alerte {alert.get('severity', 'unknown').upper()}*\n"
                   f"*Service:* {alert.get('service_name', 'Unknown')}\n"
                   f"*Titre:* {alert.get('title', 'Alerte sans titre')}\n"
                   f"*Description:* {alert.get('description', 'Aucune description')}\n"
                   f"*Environnement:* {context.environment}\n"
                   f"*Timestamp:* {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
            "metadata": {
                "template_type": "fallback",
                "language": context.language,
                "generated_at": datetime.utcnow().isoformat()
            }
        }

    async def _create_default_templates(self):
        """Crée les templates par défaut"""
        templates_dir = Path(self.templates_dir)
        templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Template standard en blocs
        standard_template = """{
    "blocks": [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "{{ get_severity_emoji(alert.severity) }} Alerte {{ alert.severity|upper }}"
            }
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": "*Service:*\\n{{ alert.service_name }}"
                },
                {
                    "type": "mrkdwn", 
                    "text": "*Environnement:*\\n{{ environment }}"
                },
                {
                    "type": "mrkdwn",
                    "text": "*Composant:*\\n{{ alert.component }}"
                },
                {
                    "type": "mrkdwn",
                    "text": "*Timestamp:*\\n{{ format_date(now) }}"
                }
            ]
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*{{ alert.title }}*\\n{{ alert.description }}"
            }
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Acquitter"
                    },
                    "style": "primary",
                    "action_id": "acknowledge_alert"
                },
                {
                    "type": "button", 
                    "text": {
                        "type": "plain_text",
                        "text": "Résoudre"
                    },
                    "style": "danger",
                    "action_id": "resolve_alert"
                }
            ]
        }
    ]
}"""
        
        with open(templates_dir / "standard_fr_blocks.j2", 'w', encoding='utf-8') as f:
            f.write(standard_template)
        
        # Répertoire des traductions
        translations_dir = templates_dir / "translations"
        translations_dir.mkdir(exist_ok=True)
        
        # Traductions françaises
        fr_translations = {
            "alert_title": "Alerte {severity}",
            "service": "Service",
            "component": "Composant", 
            "environment": "Environnement",
            "timestamp": "Horodatage",
            "acknowledge": "Acquitter",
            "resolve": "Résoudre"
        }
        
        with open(translations_dir / "fr.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(fr_translations, f, default_flow_style=False, allow_unicode=True)
