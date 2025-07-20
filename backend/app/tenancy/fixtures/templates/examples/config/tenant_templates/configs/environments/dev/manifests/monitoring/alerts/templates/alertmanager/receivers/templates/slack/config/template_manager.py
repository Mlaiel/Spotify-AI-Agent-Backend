"""
Gestionnaire de Templates Slack Ultra-Avancé
============================================

Module de gestion des templates de messages Slack pour AlertManager.
Fournit un système de templating flexible, performant et personnalisable
pour tous types d'alertes du Spotify AI Agent.

Développé par l'équipe Backend Senior sous la direction de Fahed Mlaiel.
"""

import os
import json
import yaml
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import jinja2
from jinja2 import Environment, FileSystemLoader, DictLoader, select_autoescape
import redis.asyncio as redis
from pydantic import BaseModel, Field, validator
import hashlib

from . import SlackSeverity, SlackChannelType
from .utils import SlackUtils

logger = logging.getLogger(__name__)

@dataclass
class SlackAttachment:
    """Représente un attachement Slack."""
    
    color: str = "#36a64f"  # Couleur par défaut (vert)
    title: Optional[str] = None
    title_link: Optional[str] = None
    text: Optional[str] = None
    fields: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    footer: Optional[str] = None
    footer_icon: Optional[str] = None
    timestamp: Optional[int] = None
    
    def add_field(self, title: str, value: str, short: bool = True):
        """Ajoute un champ à l'attachement."""
        self.fields.append({
            "title": title,
            "value": value,
            "short": short
        })
    
    def add_action(self, name: str, text: str, type: str = "button", url: Optional[str] = None):
        """Ajoute une action à l'attachement."""
        action = {
            "name": name,
            "text": text,
            "type": type
        }
        if url:
            action["url"] = url
        self.actions.append(action)

@dataclass
class SlackBlock:
    """Représente un block Slack (Blocks Kit)."""
    
    type: str
    text: Optional[Dict[str, Any]] = None
    elements: List[Dict[str, Any]] = field(default_factory=list)
    accessory: Optional[Dict[str, Any]] = None
    fields: List[Dict[str, Any]] = field(default_factory=list)
    
    @classmethod
    def create_section(cls, text: str, markdown: bool = True):
        """Crée un block section."""
        return cls(
            type="section",
            text={
                "type": "mrkdwn" if markdown else "plain_text",
                "text": text
            }
        )
    
    @classmethod
    def create_divider(cls):
        """Crée un block divider."""
        return cls(type="divider")
    
    @classmethod
    def create_header(cls, text: str):
        """Crée un block header."""
        return cls(
            type="header",
            text={
                "type": "plain_text",
                "text": text
            }
        )

class SlackTemplate(BaseModel):
    """Modèle de template Slack."""
    
    id: str = Field(..., description="ID unique du template")
    name: str = Field(..., description="Nom du template")
    description: str = Field(default="", description="Description du template")
    severity: SlackSeverity = Field(default=SlackSeverity.INFO)
    channel_type: SlackChannelType = Field(default=SlackChannelType.ALERTS)
    
    # Template principal
    template_content: str = Field(..., description="Contenu du template Jinja2")
    
    # Paramètres
    default_params: Dict[str, Any] = Field(default_factory=dict)
    required_params: List[str] = Field(default_factory=list)
    
    # Configuration
    use_blocks: bool = Field(default=True, description="Utiliser Blocks Kit")
    use_attachments: bool = Field(default=False, description="Utiliser les attachements legacy")
    
    # Métadonnées
    version: str = Field(default="1.0.0")
    author: str = Field(default="System")
    tags: List[str] = Field(default_factory=list)
    
    # Validation
    enabled: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('template_content')
    def validate_template_content(cls, v):
        if not v.strip():
            raise ValueError("Le contenu du template ne peut pas être vide")
        return v

class SlackTemplateManager:
    """
    Gestionnaire ultra-avancé des templates Slack.
    
    Fonctionnalités:
    - Templates Jinja2 avec syntaxe avancée
    - Cache Redis pour performances optimales
    - Validation stricte des templates
    - Support Blocks Kit et attachements legacy
    - Templates dynamiques par tenant/environnement
    - Préprocesseurs et postprocesseurs personnalisés
    - Gestion des versions et migrations
    - Métriques détaillées de rendu
    """
    
    def __init__(self,
                 templates_path: Optional[str] = None,
                 redis_client: Optional[redis.Redis] = None,
                 cache_ttl: int = 3600):
        """
        Initialise le gestionnaire de templates.
        
        Args:
            templates_path: Chemin vers les templates sur disque
            redis_client: Client Redis pour le cache
            cache_ttl: TTL du cache en secondes
        """
        self.templates_path = templates_path or self._get_default_templates_path()
        self.redis_client = redis_client
        self.cache_ttl = cache_ttl
        
        # Environment Jinja2
        self.jinja_env = self._create_jinja_environment()
        
        # Cache local des templates
        self._template_cache: Dict[str, SlackTemplate] = {}
        self._compiled_cache: Dict[str, jinja2.Template] = {}
        
        # Filtres et fonctions personnalisés
        self._custom_filters: Dict[str, Callable] = {}
        self._custom_functions: Dict[str, Callable] = {}
        
        # Métriques
        self.metrics = {
            'templates_loaded': 0,
            'renders_total': 0,
            'renders_cached': 0,
            'render_errors': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'validation_errors': 0
        }
        
        # Configuration des couleurs par sévérité
        self.severity_colors = {
            SlackSeverity.CRITICAL: "#d63031",  # Rouge
            SlackSeverity.HIGH: "#fd79a8",      # Rose
            SlackSeverity.MEDIUM: "#fdcb6e",    # Orange
            SlackSeverity.LOW: "#55a3ff",       # Bleu
            SlackSeverity.INFO: "#00b894"       # Vert
        }
        
        # Initialisation
        self._initialize()
    
    def _get_default_templates_path(self) -> str:
        """Retourne le chemin par défaut des templates."""
        return str(Path(__file__).parent / "templates")
    
    def _create_jinja_environment(self) -> jinja2.Environment:
        """Crée l'environnement Jinja2."""
        # Créer le répertoire de templates s'il n'existe pas
        templates_dir = Path(self.templates_path)
        templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration Jinja2
        env = Environment(
            loader=FileSystemLoader(self.templates_path),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=jinja2.StrictUndefined
        )
        
        # Ajouter les filtres personnalisés
        self._register_custom_filters(env)
        
        # Ajouter les fonctions globales
        self._register_global_functions(env)
        
        return env
    
    def _register_custom_filters(self, env: jinja2.Environment):
        """Enregistre les filtres personnalisés."""
        
        @env.filter('slack_escape')
        def slack_escape(text: str) -> str:
            """Échappe les caractères spéciaux Slack."""
            if not isinstance(text, str):
                text = str(text)
            
            replacements = {
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;',
                "'": '&#x27;'
            }
            
            for char, replacement in replacements.items():
                text = text.replace(char, replacement)
            
            return text
        
        @env.filter('format_duration')
        def format_duration(seconds: Union[int, float]) -> str:
            """Formate une durée en secondes en format lisible."""
            if not isinstance(seconds, (int, float)):
                return str(seconds)
            
            if seconds < 60:
                return f"{seconds:.1f}s"
            elif seconds < 3600:
                minutes = seconds / 60
                return f"{minutes:.1f}m"
            else:
                hours = seconds / 3600
                return f"{hours:.1f}h"
        
        @env.filter('format_bytes')
        def format_bytes(bytes_value: Union[int, float]) -> str:
            """Formate une taille en bytes en format lisible."""
            if not isinstance(bytes_value, (int, float)):
                return str(bytes_value)
            
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if bytes_value < 1024.0:
                    return f"{bytes_value:.1f} {unit}"
                bytes_value /= 1024.0
            
            return f"{bytes_value:.1f} PB"
        
        @env.filter('severity_color')
        def severity_color(severity: Union[str, SlackSeverity]) -> str:
            """Retourne la couleur associée à une sévérité."""
            if isinstance(severity, str):
                try:
                    severity = SlackSeverity(severity.lower())
                except ValueError:
                    return self.severity_colors[SlackSeverity.INFO]
            
            return self.severity_colors.get(severity, self.severity_colors[SlackSeverity.INFO])
        
        @env.filter('truncate_smart')
        def truncate_smart(text: str, length: int = 100, suffix: str = "...") -> str:
            """Tronque intelligemment un texte."""
            if not isinstance(text, str) or len(text) <= length:
                return text
            
            # Essayer de tronquer au niveau d'un mot
            truncated = text[:length - len(suffix)]
            last_space = truncated.rfind(' ')
            
            if last_space > length * 0.8:  # Si l'espace est assez proche de la fin
                truncated = truncated[:last_space]
            
            return truncated + suffix
    
    def _register_global_functions(self, env: jinja2.Environment):
        """Enregistre les fonctions globales."""
        
        def now():
            """Retourne la date/heure actuelle."""
            return datetime.utcnow()
        
        def timestamp():
            """Retourne le timestamp Unix actuel."""
            return int(datetime.utcnow().timestamp())
        
        def format_timestamp(ts: Union[int, float, datetime], format_str: str = "%Y-%m-%d %H:%M:%S UTC"):
            """Formate un timestamp."""
            if isinstance(ts, datetime):
                dt = ts
            else:
                dt = datetime.fromtimestamp(float(ts))
            
            return dt.strftime(format_str)
        
        def create_link(url: str, text: str) -> str:
            """Crée un lien Slack."""
            return f"<{url}|{text}>"
        
        def create_mention(user_id: str) -> str:
            """Crée une mention utilisateur."""
            return f"<@{user_id}>"
        
        def create_channel_mention(channel_id: str) -> str:
            """Crée une mention de canal."""
            return f"<#{channel_id}>"
        
        # Ajouter les fonctions à l'environnement
        env.globals.update({
            'now': now,
            'timestamp': timestamp,
            'format_timestamp': format_timestamp,
            'create_link': create_link,
            'create_mention': create_mention,
            'create_channel_mention': create_channel_mention
        })
    
    def _initialize(self):
        """Initialise le gestionnaire."""
        try:
            # Créer les templates par défaut
            asyncio.create_task(self._create_default_templates())
            
            # Charger les templates existants
            asyncio.create_task(self._load_templates())
            
            logger.info("SlackTemplateManager initialisé")
            
        except Exception as e:
            logger.error(f"Erreur initialisation SlackTemplateManager: {e}")
    
    async def _create_default_templates(self):
        """Crée les templates par défaut."""
        default_templates = [
            {
                "id": "alert_critical",
                "name": "Alerte Critique",
                "description": "Template pour les alertes critiques",
                "severity": SlackSeverity.CRITICAL,
                "template_content": self._get_critical_alert_template(),
                "required_params": ["alertname", "severity", "description"]
            },
            {
                "id": "alert_high",
                "name": "Alerte Haute",
                "description": "Template pour les alertes de priorité haute",
                "severity": SlackSeverity.HIGH,
                "template_content": self._get_high_alert_template(),
                "required_params": ["alertname", "severity", "description"]
            },
            {
                "id": "alert_medium",
                "name": "Alerte Moyenne",
                "description": "Template pour les alertes de priorité moyenne",
                "severity": SlackSeverity.MEDIUM,
                "template_content": self._get_medium_alert_template(),
                "required_params": ["alertname", "severity", "description"]
            },
            {
                "id": "alert_resolved",
                "name": "Alerte Résolue",
                "description": "Template pour les alertes résolues",
                "severity": SlackSeverity.INFO,
                "template_content": self._get_resolved_alert_template(),
                "required_params": ["alertname", "resolved_at"]
            },
            {
                "id": "system_notification",
                "name": "Notification Système",
                "description": "Template pour les notifications système",
                "severity": SlackSeverity.INFO,
                "template_content": self._get_system_notification_template(),
                "required_params": ["title", "message"]
            }
        ]
        
        for template_data in default_templates:
            await self.register_template(SlackTemplate(**template_data))
    
    def _get_critical_alert_template(self) -> str:
        """Template pour alertes critiques."""
        return '''
{
  "blocks": [
    {
      "type": "header",
      "text": {
        "type": "plain_text",
        "text": "🚨 ALERTE CRITIQUE - {{ alertname | upper }}"
      }
    },
    {
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": "*Sévérité:* {{ severity | severity_color }} {{ severity | upper }}\\n*Description:* {{ description }}\\n*Heure:* {{ format_timestamp(timestamp()) }}"
      }
    },
    {% if instance %}
    {
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": "*Instance:* `{{ instance }}`"
      }
    },
    {% endif %}
    {% if labels %}
    {
      "type": "section",
      "fields": [
        {% for key, value in labels.items() %}
        {
          "type": "mrkdwn",
          "text": "*{{ key | title }}:*\\n{{ value }}"
        }{% if not loop.last %},{% endif %}
        {% endfor %}
      ]
    },
    {% endif %}
    {
      "type": "divider"
    },
    {
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": "🔗 *Actions rapides:*"
      },
      "accessory": {
        "type": "button",
        "text": {
          "type": "plain_text",
          "text": "Voir Dashboard"
        },
        "url": "{{ dashboard_url | default('https://grafana.example.com') }}"
      }
    }
  ]
}
'''
    
    def _get_high_alert_template(self) -> str:
        """Template pour alertes hautes."""
        return '''
{
  "blocks": [
    {
      "type": "header",
      "text": {
        "type": "plain_text",
        "text": "⚠️ ALERTE HAUTE - {{ alertname }}"
      }
    },
    {
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": "*Sévérité:* {{ severity | severity_color }} {{ severity | upper }}\\n*Description:* {{ description | truncate_smart(200) }}\\n*Heure:* {{ format_timestamp(timestamp()) }}"
      }
    },
    {% if annotations %}
    {
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": "*Détails:*\\n{% for key, value in annotations.items() %}• *{{ key | title }}:* {{ value }}\\n{% endfor %}"
      }
    },
    {% endif %}
    {
      "type": "context",
      "elements": [
        {
          "type": "mrkdwn",
          "text": "🏷️ *Labels:* {% for key, value in labels.items() %}`{{ key }}={{ value }}`{% if not loop.last %}, {% endif %}{% endfor %}"
        }
      ]
    }
  ]
}
'''
    
    def _get_medium_alert_template(self) -> str:
        """Template pour alertes moyennes."""
        return '''
{
  "blocks": [
    {
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": "📊 *{{ alertname }}* - {{ severity | upper }}\\n{{ description | truncate_smart(150) }}"
      }
    },
    {
      "type": "context",
      "elements": [
        {
          "type": "mrkdwn",
          "text": "⏰ {{ format_timestamp(timestamp()) }} | 🎯 {{ instance | default('N/A') }}"
        }
      ]
    }
  ]
}
'''
    
    def _get_resolved_alert_template(self) -> str:
        """Template pour alertes résolues."""
        return '''
{
  "blocks": [
    {
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": "✅ *RÉSOLU* - {{ alertname }}\\n*Résolu à:* {{ format_timestamp(resolved_at) }}\\n*Durée:* {{ format_duration(duration | default(0)) }}"
      }
    }
  ]
}
'''
    
    def _get_system_notification_template(self) -> str:
        """Template pour notifications système."""
        return '''
{
  "blocks": [
    {
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": "🔔 *{{ title }}*\\n{{ message }}"
      }
    },
    {
      "type": "context",
      "elements": [
        {
          "type": "mrkdwn",
          "text": "{{ format_timestamp(timestamp()) }} | Spotify AI Agent"
        }
      ]
    }
  ]
}
'''
    
    async def _load_templates(self):
        """Charge les templates depuis le disque et Redis."""
        try:
            # Charger depuis Redis
            await self._load_templates_from_redis()
            
            # Charger depuis le disque
            await self._load_templates_from_disk()
            
            self.metrics['templates_loaded'] = len(self._template_cache)
            logger.info(f"{self.metrics['templates_loaded']} templates chargés")
            
        except Exception as e:
            logger.error(f"Erreur chargement templates: {e}")
    
    async def _load_templates_from_redis(self):
        """Charge les templates depuis Redis."""
        try:
            if not self.redis_client:
                return
            
            # Récupérer tous les templates
            pattern = "slack_template:*"
            keys = await self.redis_client.keys(pattern)
            
            for key in keys:
                template_data = await self.redis_client.hgetall(key)
                if template_data:
                    # Convertir les bytes en strings
                    template_dict = {k.decode(): v.decode() for k, v in template_data.items()}
                    
                    # Parser les champs JSON
                    for json_field in ['default_params', 'required_params', 'tags']:
                        if json_field in template_dict:
                            template_dict[json_field] = json.loads(template_dict[json_field])
                    
                    # Convertir les dates
                    for date_field in ['created_at', 'updated_at']:
                        if date_field in template_dict:
                            template_dict[date_field] = datetime.fromisoformat(template_dict[date_field])
                    
                    # Convertir les enums
                    if 'severity' in template_dict:
                        template_dict['severity'] = SlackSeverity(template_dict['severity'])
                    
                    if 'channel_type' in template_dict:
                        template_dict['channel_type'] = SlackChannelType(template_dict['channel_type'])
                    
                    template = SlackTemplate(**template_dict)
                    self._template_cache[template.id] = template
                    
        except Exception as e:
            logger.error(f"Erreur chargement templates Redis: {e}")
    
    async def _load_templates_from_disk(self):
        """Charge les templates depuis le disque."""
        try:
            templates_dir = Path(self.templates_path)
            if not templates_dir.exists():
                return
            
            # Parcourir tous les fichiers .yaml et .json
            for file_path in templates_dir.glob("**/*.{yaml,yml,json}"):
                await self._load_template_file(file_path)
                
        except Exception as e:
            logger.error(f"Erreur chargement templates disque: {e}")
    
    async def _load_template_file(self, file_path: Path):
        """Charge un fichier template."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            # Valider et créer le template
            template = SlackTemplate(**data)
            
            # Ajouter au cache si pas déjà présent
            if template.id not in self._template_cache:
                self._template_cache[template.id] = template
                logger.debug(f"Template {template.id} chargé depuis {file_path}")
                
        except Exception as e:
            logger.error(f"Erreur chargement fichier template {file_path}: {e}")
            self.metrics['validation_errors'] += 1
    
    async def register_template(self, template: SlackTemplate) -> bool:
        """
        Enregistre un nouveau template.
        
        Args:
            template: Template à enregistrer
            
        Returns:
            True si succès, False sinon
        """
        try:
            # Valider le template
            is_valid, errors = await self.validate_template(template)
            if not is_valid:
                logger.error(f"Template {template.id} invalide: {errors}")
                return False
            
            # Mettre à jour le timestamp
            template.updated_at = datetime.utcnow()
            
            # Ajouter au cache
            self._template_cache[template.id] = template
            
            # Persister en Redis
            if self.redis_client:
                await self._persist_template(template)
            
            # Sauvegarder sur disque
            await self._save_template_to_disk(template)
            
            # Invalider le cache compilé
            if template.id in self._compiled_cache:
                del self._compiled_cache[template.id]
            
            logger.info(f"Template {template.id} enregistré")
            return True
            
        except Exception as e:
            logger.error(f"Erreur enregistrement template {template.id}: {e}")
            return False
    
    async def validate_template(self, template: SlackTemplate) -> tuple[bool, List[str]]:
        """
        Valide un template.
        
        Args:
            template: Template à valider
            
        Returns:
            Tuple (is_valid, errors)
        """
        errors = []
        
        try:
            # Validation Pydantic de base
            template.dict()
            
            # Valider la syntaxe Jinja2
            try:
                self.jinja_env.from_string(template.template_content)
            except jinja2.TemplateSyntaxError as e:
                errors.append(f"Erreur syntaxe Jinja2: {e}")
            
            # Valider le JSON de sortie avec des données de test
            test_data = {param: f"test_{param}" for param in template.required_params}
            test_data.update(template.default_params)
            
            try:
                rendered = await self._render_template_content(template, test_data)
                json.loads(rendered)
            except (json.JSONDecodeError, Exception) as e:
                errors.append(f"Le template ne produit pas un JSON valide: {e}")
            
            # Vérifier les paramètres requis
            if not template.required_params:
                errors.append("Aucun paramètre requis défini")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Erreur validation: {e}")
            return False, errors
    
    async def _render_template_content(self, template: SlackTemplate, data: Dict[str, Any]) -> str:
        """Rend le contenu d'un template."""
        try:
            # Vérifier le cache compilé
            if template.id in self._compiled_cache:
                compiled_template = self._compiled_cache[template.id]
            else:
                # Compiler le template
                compiled_template = self.jinja_env.from_string(template.template_content)
                self._compiled_cache[template.id] = compiled_template
            
            # Fusionner avec les paramètres par défaut
            render_data = {**template.default_params, **data}
            
            # Rendre le template
            rendered = compiled_template.render(**render_data)
            
            return rendered.strip()
            
        except Exception as e:
            logger.error(f"Erreur rendu template {template.id}: {e}")
            raise
    
    async def render_template(self,
                            template_id: str,
                            data: Dict[str, Any],
                            tenant_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Rend un template avec les données fournies.
        
        Args:
            template_id: ID du template
            data: Données pour le rendu
            tenant_id: ID du tenant (optionnel)
            
        Returns:
            Message Slack formaté ou None
        """
        start_time = datetime.utcnow()
        
        try:
            # Rechercher le template
            template = await self.get_template(template_id, tenant_id)
            if not template:
                logger.error(f"Template {template_id} introuvable")
                return None
            
            # Vérifier les paramètres requis
            missing_params = [param for param in template.required_params if param not in data]
            if missing_params:
                logger.error(f"Paramètres manquants pour template {template_id}: {missing_params}")
                return None
            
            # Vérifier le cache rendu
            cache_key = self._get_render_cache_key(template_id, data, tenant_id)
            cached_result = await self._get_cached_render(cache_key)
            if cached_result:
                self.metrics['renders_cached'] += 1
                return cached_result
            
            # Rendre le template
            rendered_content = await self._render_template_content(template, data)
            
            # Parser le JSON
            result = json.loads(rendered_content)
            
            # Mettre en cache
            await self._cache_render(cache_key, result)
            
            # Métriques
            self.metrics['renders_total'] += 1
            render_time = (datetime.utcnow() - start_time).total_seconds()
            
            logger.debug(f"Template {template_id} rendu en {render_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.metrics['render_errors'] += 1
            logger.error(f"Erreur rendu template {template_id}: {e}")
            return None
    
    async def get_template(self, template_id: str, tenant_id: Optional[str] = None) -> Optional[SlackTemplate]:
        """
        Récupère un template par ID.
        
        Args:
            template_id: ID du template
            tenant_id: ID du tenant (pour templates spécifiques)
            
        Returns:
            Template ou None
        """
        try:
            # Chercher d'abord un template spécifique au tenant
            if tenant_id:
                tenant_template_id = f"{tenant_id}_{template_id}"
                if tenant_template_id in self._template_cache:
                    return self._template_cache[tenant_template_id]
            
            # Chercher le template global
            if template_id in self._template_cache:
                return self._template_cache[template_id]
            
            # Essayer de charger depuis Redis
            template = await self._load_template_from_redis(template_id, tenant_id)
            if template:
                return template
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur récupération template {template_id}: {e}")
            return None
    
    async def _load_template_from_redis(self, template_id: str, tenant_id: Optional[str] = None) -> Optional[SlackTemplate]:
        """Charge un template depuis Redis."""
        try:
            if not self.redis_client:
                return None
            
            # Essayer avec le tenant d'abord
            if tenant_id:
                key = f"slack_template:{tenant_id}_{template_id}"
                template_data = await self.redis_client.hgetall(key)
                if template_data:
                    return self._parse_redis_template(template_data)
            
            # Essayer le template global
            key = f"slack_template:{template_id}"
            template_data = await self.redis_client.hgetall(key)
            if template_data:
                return self._parse_redis_template(template_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur chargement template Redis {template_id}: {e}")
            return None
    
    def _parse_redis_template(self, template_data: Dict) -> SlackTemplate:
        """Parse les données template depuis Redis."""
        # Convertir les bytes en strings
        template_dict = {k.decode(): v.decode() for k, v in template_data.items()}
        
        # Parser les champs JSON
        for json_field in ['default_params', 'required_params', 'tags']:
            if json_field in template_dict:
                template_dict[json_field] = json.loads(template_dict[json_field])
        
        # Convertir les dates
        for date_field in ['created_at', 'updated_at']:
            if date_field in template_dict:
                template_dict[date_field] = datetime.fromisoformat(template_dict[date_field])
        
        # Convertir les enums
        if 'severity' in template_dict:
            template_dict['severity'] = SlackSeverity(template_dict['severity'])
        
        if 'channel_type' in template_dict:
            template_dict['channel_type'] = SlackChannelType(template_dict['channel_type'])
        
        return SlackTemplate(**template_dict)
    
    def _get_render_cache_key(self, template_id: str, data: Dict[str, Any], tenant_id: Optional[str] = None) -> str:
        """Génère une clé de cache pour un rendu."""
        data_hash = hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
        tenant_part = f":{tenant_id}" if tenant_id else ""
        return f"render:{template_id}{tenant_part}:{data_hash}"
    
    async def _get_cached_render(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Récupère un rendu depuis le cache."""
        try:
            if not self.redis_client:
                return None
            
            cached_data = await self.redis_client.get(f"slack_render:{cache_key}")
            if cached_data:
                self.metrics['cache_hits'] += 1
                return json.loads(cached_data.decode())
            
            self.metrics['cache_misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"Erreur récupération cache render: {e}")
            return None
    
    async def _cache_render(self, cache_key: str, result: Dict[str, Any]):
        """Met en cache un rendu."""
        try:
            if self.redis_client:
                await self.redis_client.setex(
                    f"slack_render:{cache_key}",
                    self.cache_ttl,
                    json.dumps(result)
                )
                
        except Exception as e:
            logger.error(f"Erreur mise en cache render: {e}")
    
    async def _persist_template(self, template: SlackTemplate):
        """Persiste un template en Redis."""
        try:
            if not self.redis_client:
                return
            
            key = f"slack_template:{template.id}"
            data = {
                'id': template.id,
                'name': template.name,
                'description': template.description,
                'severity': template.severity.value,
                'channel_type': template.channel_type.value,
                'template_content': template.template_content,
                'default_params': json.dumps(template.default_params),
                'required_params': json.dumps(template.required_params),
                'use_blocks': str(template.use_blocks),
                'use_attachments': str(template.use_attachments),
                'version': template.version,
                'author': template.author,
                'tags': json.dumps(template.tags),
                'enabled': str(template.enabled),
                'created_at': template.created_at.isoformat(),
                'updated_at': template.updated_at.isoformat()
            }
            
            await self.redis_client.hset(key, mapping=data)
            await self.redis_client.expire(key, self.cache_ttl * 24)  # 24x le TTL normal
            
        except Exception as e:
            logger.error(f"Erreur persistance template {template.id}: {e}")
    
    async def _save_template_to_disk(self, template: SlackTemplate):
        """Sauvegarde un template sur disque."""
        try:
            templates_dir = Path(self.templates_path)
            templates_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = templates_dir / f"{template.id}.yaml"
            
            template_data = template.dict()
            # Convertir les enums en strings
            template_data['severity'] = template_data['severity'].value
            template_data['channel_type'] = template_data['channel_type'].value
            # Convertir les dates en ISO format
            template_data['created_at'] = template_data['created_at'].isoformat()
            template_data['updated_at'] = template_data['updated_at'].isoformat()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(template_data, f, default_flow_style=False, indent=2)
            
            logger.debug(f"Template {template.id} sauvegardé sur disque")
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde template {template.id}: {e}")
    
    async def list_templates(self, 
                           tenant_id: Optional[str] = None,
                           severity: Optional[SlackSeverity] = None,
                           enabled_only: bool = True) -> List[SlackTemplate]:
        """
        Liste les templates disponibles.
        
        Args:
            tenant_id: Filtrer par tenant
            severity: Filtrer par sévérité
            enabled_only: Seulement les templates activés
            
        Returns:
            Liste des templates
        """
        try:
            templates = []
            
            for template in self._template_cache.values():
                # Filtrer par tenant
                if tenant_id and not template.id.startswith(f"{tenant_id}_"):
                    continue
                
                # Filtrer par sévérité
                if severity and template.severity != severity:
                    continue
                
                # Filtrer par état
                if enabled_only and not template.enabled:
                    continue
                
                templates.append(template)
            
            # Trier par nom
            templates.sort(key=lambda t: t.name)
            
            return templates
            
        except Exception as e:
            logger.error(f"Erreur listage templates: {e}")
            return []
    
    async def delete_template(self, template_id: str, tenant_id: Optional[str] = None) -> bool:
        """
        Supprime un template.
        
        Args:
            template_id: ID du template
            tenant_id: ID du tenant
            
        Returns:
            True si succès, False sinon
        """
        try:
            # Construire l'ID complet si tenant spécifié
            full_template_id = f"{tenant_id}_{template_id}" if tenant_id else template_id
            
            # Supprimer du cache
            if full_template_id in self._template_cache:
                del self._template_cache[full_template_id]
            
            # Supprimer du cache compilé
            if full_template_id in self._compiled_cache:
                del self._compiled_cache[full_template_id]
            
            # Supprimer de Redis
            if self.redis_client:
                await self.redis_client.delete(f"slack_template:{full_template_id}")
            
            # Supprimer du disque
            file_path = Path(self.templates_path) / f"{full_template_id}.yaml"
            if file_path.exists():
                file_path.unlink()
            
            logger.info(f"Template {full_template_id} supprimé")
            return True
            
        except Exception as e:
            logger.error(f"Erreur suppression template {template_id}: {e}")
            return False
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques du gestionnaire."""
        cache_hit_rate = 0
        if self.metrics['cache_hits'] + self.metrics['cache_misses'] > 0:
            cache_hit_rate = self.metrics['cache_hits'] / (self.metrics['cache_hits'] + self.metrics['cache_misses'])
        
        return {
            **self.metrics,
            'cache_hit_rate': cache_hit_rate,
            'templates_in_cache': len(self._template_cache),
            'compiled_templates': len(self._compiled_cache),
            'templates_path': self.templates_path,
            'jinja_filters': len(self.jinja_env.filters),
            'jinja_globals': len(self.jinja_env.globals)
        }
    
    def add_custom_filter(self, name: str, filter_func: Callable):
        """Ajoute un filtre personnalisé."""
        self.jinja_env.filters[name] = filter_func
        self._custom_filters[name] = filter_func
        logger.info(f"Filtre personnalisé '{name}' ajouté")
    
    def add_custom_function(self, name: str, func: Callable):
        """Ajoute une fonction personnalisée."""
        self.jinja_env.globals[name] = func
        self._custom_functions[name] = func
        logger.info(f"Fonction personnalisée '{name}' ajoutée")
    
    async def reload_templates(self):
        """Recharge tous les templates."""
        try:
            # Vider les caches
            self._template_cache.clear()
            self._compiled_cache.clear()
            
            # Recharger
            await self._load_templates()
            
            logger.info("Templates rechargés")
            
        except Exception as e:
            logger.error(f"Erreur rechargement templates: {e}")
    
    def __repr__(self) -> str:
        return f"SlackTemplateManager(templates={len(self._template_cache)}, path='{self.templates_path}')"
