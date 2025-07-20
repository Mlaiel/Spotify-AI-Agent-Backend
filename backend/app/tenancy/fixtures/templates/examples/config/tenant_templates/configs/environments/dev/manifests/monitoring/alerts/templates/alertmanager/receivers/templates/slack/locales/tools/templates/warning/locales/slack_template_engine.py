#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spotify AI Agent - Slack Template Engine pour Alerting

Moteur de templates Slack avancé pour génération de messages d'alerte
contextualisés et interactifs avec support multi-tenant.

Fonctionnalités:
- Génération de messages Slack avec Blocks API
- Templates adaptatifs selon la sévérité et le contexte
- Support des attachments, boutons d'action et menus
- Formatage intelligent avec markdown et emojis
- Rate limiting et anti-spam intégré
- Threading automatique des alertes similaires
- Métriques et monitoring des envois

Architecture:
- Template Engine basé sur Jinja2
- Builder Pattern pour construction des messages
- Strategy Pattern pour différents formats
- Cache intelligent pour templates compilés
- Queue asynchrone pour envois groupés

Cas d'usage:
- Alertes critiques avec escalation
- Notifications de performance IA
- Rapports de synthèse automatiques
- Messages interactifs avec actions
- Threading des alertes corrélées

Utilisation:
    engine = SlackTemplateEngine()
    
    message = engine.render_alert_message(
        alert_data={
            "severity": "critical",
            "type": "ai_model_performance",
            "context": {...}
        },
        channel_config={
            "channel": "#alerts-critical",
            "mention_users": ["@here"]
        }
    )
"""

import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import re
import logging
from urllib.parse import quote

# Imports externes
import structlog
from jinja2 import Environment, BaseLoader, select_autoescape
from prometheus_client import Counter, Histogram, Gauge
import redis
import asyncio
from ratelimiter import RateLimiter

# Imports internes
from .config import (
    SLACK_CHANNEL_MAPPING,
    SLACK_FORMATTING_RULES,
    ALERT_SEVERITY_LEVELS,
    get_severity_config
)

# Configuration logging
logger = structlog.get_logger(__name__)

# Métriques Prometheus
slack_message_generation_total = Counter(
    'spotify_ai_slack_message_generation_total',
    'Nombre total de messages Slack générés',
    ['tenant_id', 'severity', 'channel', 'template_type']
)

slack_message_generation_duration = Histogram(
    'spotify_ai_slack_message_generation_seconds',
    'Durée de génération des messages Slack',
    ['template_type', 'complexity_level']
)

slack_template_cache_hits = Counter(
    'spotify_ai_slack_template_cache_hits_total',
    'Hits du cache des templates Slack',
    ['template_type']
)

slack_rate_limit_hits = Counter(
    'spotify_ai_slack_rate_limit_hits_total',
    'Nombre de fois où la rate limit a été atteinte',
    ['channel']
)

active_slack_threads = Gauge(
    'spotify_ai_active_slack_threads',
    'Nombre de threads Slack actifs'
)

class MessageComplexity(Enum):
    """Niveaux de complexité des messages."""
    SIMPLE = "simple"          # Texte basique
    STRUCTURED = "structured"  # Blocks et fields
    INTERACTIVE = "interactive" # Boutons et menus
    RICH = "rich"             # Attachments, images, graphiques

class ThreadStrategy(Enum):
    """Stratégies de threading."""
    NONE = "none"              # Pas de threading
    BY_ALERT_TYPE = "by_type"  # Thread par type d'alerte
    BY_SERVICE = "by_service"  # Thread par service
    BY_SEVERITY = "by_severity" # Thread par sévérité
    CORRELATION_ID = "correlation" # Thread par ID de corrélation

@dataclass
class SlackMessageOptions:
    """Options de configuration d'un message Slack."""
    channel: str
    username: str = "Spotify AI Alert Bot"
    icon_emoji: str = ":robot_face:"
    icon_url: str = ""
    
    # Threading
    thread_strategy: ThreadStrategy = ThreadStrategy.BY_ALERT_TYPE
    thread_ts: Optional[str] = None
    
    # Mentions
    mention_users: List[str] = field(default_factory=list)
    mention_here: bool = False
    mention_channel: bool = False
    
    # Formatage
    unfurl_links: bool = False
    unfurl_media: bool = False
    parse: str = "full"  # none, full
    
    # Rate limiting
    rate_limit_key: str = ""
    max_per_minute: int = 10
    
    # Métadonnées
    correlation_id: str = ""
    alert_id: str = ""
    tenant_id: str = ""

@dataclass
class SlackBlock:
    """Représentation d'un block Slack."""
    type: str
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit le block en dictionnaire Slack."""
        result = {"type": self.type}
        result.update(self.data)
        return result

@dataclass
class SlackField:
    """Champ Slack pour sections."""
    title: str
    value: str
    short: bool = True

@dataclass
class SlackAction:
    """Action Slack (bouton, menu)."""
    type: str  # button, select, etc.
    text: str
    value: str = ""
    url: str = ""
    style: str = "default"  # default, primary, danger
    confirm: Optional[Dict[str, str]] = None

class SlackMessageBuilder:
    """
    Builder pour construction de messages Slack complexes.
    
    Facilite la création de messages avec blocks, attachments,
    et éléments interactifs.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Réinitialise le builder."""
        self.text = ""
        self.blocks = []
        self.attachments = []
        self.thread_ts = None
        self.options = SlackMessageOptions(channel="")
        return self
    
    def set_text(self, text: str) -> 'SlackMessageBuilder':
        """Définit le texte principal (fallback)."""
        self.text = text
        return self
    
    def set_options(self, options: SlackMessageOptions) -> 'SlackMessageBuilder':
        """Définit les options du message."""
        self.options = options
        return self
    
    def add_header(self, text: str, emoji: str = "") -> 'SlackMessageBuilder':
        """Ajoute un header."""
        header_text = f"{emoji} {text}" if emoji else text
        block = SlackBlock("header", {
            "text": {
                "type": "plain_text",
                "text": header_text
            }
        })
        self.blocks.append(block)
        return self
    
    def add_section(
        self,
        text: str,
        fields: List[SlackField] = None,
        accessory: Dict[str, Any] = None
    ) -> 'SlackMessageBuilder':
        """Ajoute une section avec texte et champs optionnels."""
        section_data = {
            "text": {
                "type": "mrkdwn",
                "text": text
            }
        }
        
        # Ajout des champs
        if fields:
            section_data["fields"] = [
                {
                    "type": "mrkdwn",
                    "text": f"*{field.title}*\n{field.value}"
                }
                for field in fields
            ]
        
        # Ajout d'accessoire (bouton, image, etc.)
        if accessory:
            section_data["accessory"] = accessory
        
        block = SlackBlock("section", section_data)
        self.blocks.append(block)
        return self
    
    def add_divider(self) -> 'SlackMessageBuilder':
        """Ajoute un séparateur."""
        block = SlackBlock("divider")
        self.blocks.append(block)
        return self
    
    def add_context(self, elements: List[str]) -> 'SlackMessageBuilder':
        """Ajoute des éléments de contexte."""
        context_elements = [
            {
                "type": "mrkdwn", 
                "text": element
            }
            for element in elements
        ]
        
        block = SlackBlock("context", {
            "elements": context_elements
        })
        self.blocks.append(block)
        return self
    
    def add_actions(self, actions: List[SlackAction]) -> 'SlackMessageBuilder':
        """Ajoute des boutons d'action."""
        elements = []
        
        for action in actions:
            if action.type == "button":
                element = {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": action.text
                    },
                    "value": action.value
                }
                
                if action.url:
                    element["url"] = action.url
                
                if action.style != "default":
                    element["style"] = action.style
                
                if action.confirm:
                    element["confirm"] = {
                        "title": {"type": "plain_text", "text": action.confirm.get("title", "Confirmer")},
                        "text": {"type": "plain_text", "text": action.confirm.get("text", "Êtes-vous sûr?")},
                        "confirm": {"type": "plain_text", "text": action.confirm.get("confirm", "Oui")},
                        "deny": {"type": "plain_text", "text": action.confirm.get("deny", "Non")}
                    }
                
                elements.append(element)
        
        if elements:
            block = SlackBlock("actions", {
                "elements": elements
            })
            self.blocks.append(block)
        
        return self
    
    def add_rich_text(
        self,
        title: str,
        description: str,
        color: str = "good",
        fields: List[SlackField] = None,
        footer: str = "",
        timestamp: datetime = None
    ) -> 'SlackMessageBuilder':
        """Ajoute un attachment riche (legacy mais utile)."""
        attachment = {
            "color": color,
            "title": title,
            "text": description,
            "mrkdwn_in": ["text", "fields"]
        }
        
        if fields:
            attachment["fields"] = [
                {
                    "title": field.title,
                    "value": field.value,
                    "short": field.short
                }
                for field in fields
            ]
        
        if footer:
            attachment["footer"] = footer
        
        if timestamp:
            attachment["ts"] = int(timestamp.timestamp())
        
        self.attachments.append(attachment)
        return self
    
    def add_chart_image(
        self,
        image_url: str,
        title: str = "",
        alt_text: str = ""
    ) -> 'SlackMessageBuilder':
        """Ajoute une image de graphique."""
        block = SlackBlock("image", {
            "image_url": image_url,
            "alt_text": alt_text or title or "Graphique"
        })
        
        if title:
            # Ajout d'un contexte avec le titre
            self.add_context([f"📊 {title}"])
        
        self.blocks.append(block)
        return self
    
    def set_thread(self, thread_ts: str) -> 'SlackMessageBuilder':
        """Définit le thread parent."""
        self.thread_ts = thread_ts
        return self
    
    def build(self) -> Dict[str, Any]:
        """Construit le message final."""
        message = {
            "channel": self.options.channel,
            "text": self.text,
            "username": self.options.username,
            "icon_emoji": self.options.icon_emoji,
            "unfurl_links": self.options.unfurl_links,
            "unfurl_media": self.options.unfurl_media
        }
        
        # Ajout des blocks
        if self.blocks:
            message["blocks"] = [block.to_dict() for block in self.blocks]
        
        # Ajout des attachments
        if self.attachments:
            message["attachments"] = self.attachments
        
        # Threading
        if self.thread_ts:
            message["thread_ts"] = self.thread_ts
        
        # Icon URL prioritaire sur emoji
        if self.options.icon_url:
            message["icon_url"] = self.options.icon_url
            del message["icon_emoji"]
        
        return message

class SlackTemplateEngine:
    """
    Moteur de templates Slack pour Spotify AI Agent.
    
    Génère des messages Slack contextualisés avec support
    du threading, rate limiting et templates adaptatifs.
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        
        # Cache des templates compilés
        self._template_cache = {}
        self._cache_lock = asyncio.Lock()
        
        # Rate limiters par canal
        self._rate_limiters = {}
        
        # Thread tracking
        self._active_threads = {}
        
        # Builder réutilisable
        self.builder = SlackMessageBuilder()
        
        # Initialisation Jinja2
        self._init_jinja_env()
        
        # Templates par défaut
        self._load_default_templates()
        
        logger.info("SlackTemplateEngine initialisé")
    
    def _init_jinja_env(self):
        """Initialise l'environnement Jinja2."""
        class StringLoader(BaseLoader):
            """Loader pour templates en string."""
            def __init__(self, templates):
                self.templates = templates
            
            def get_source(self, environment, template):
                if template in self.templates:
                    source = self.templates[template]
                    return source, None, lambda: True
                raise FileNotFoundError(template)
        
        self.jinja_env = Environment(
            loader=StringLoader({}),
            autoescape=select_autoescape(),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Filtres custom
        self.jinja_env.filters.update({
            'slack_escape': self._slack_escape,
            'slack_link': self._slack_link,
            'slack_mention': self._slack_mention,
            'truncate_smart': self._truncate_smart,
            'format_number': self._format_number,
            'format_duration': self._format_duration,
            'severity_emoji': self._severity_emoji,
            'ai_context_emoji': self._ai_context_emoji
        })
    
    def _load_default_templates(self):
        """Charge les templates par défaut."""
        self.default_templates = {
            "alert_critical": """
🚨 *ALERTE CRITIQUE* - {{ alert_type|title }}

*Service:* {{ service_name|default('N/A') }}
*Tenant:* {{ tenant_name|default(tenant_id) }}
*Environnement:* {{ environment|upper }}

{{ message }}

{% if metrics %}
*Métriques:*
{% for metric in metrics %}
• {{ metric.name }}: {{ metric.value|format_number }} {{ metric.unit|default('') }}
{% endfor %}
{% endif %}

{% if actions %}
*Actions recommandées:*
{% for action in actions %}
• {{ action.description }}
{% endfor %}
{% endif %}
            """,
            
            "alert_warning": """
⚠️ *Alerte* - {{ alert_type|title }}

{{ message }}

*Détails:*
• Service: {{ service_name|default('N/A') }}
• Valeur: {{ metric_value|format_number }} {{ metric_unit|default('') }}
• Seuil: {{ threshold|format_number }} {{ metric_unit|default('') }}
            """,
            
            "alert_info": """
ℹ️ {{ message }}

{% if details %}
{{ details }}
{% endif %}
            """,
            
            "ai_performance": """
{{ severity_emoji(severity) }} *Performance IA* - {{ model_name }}

*Modèle:* {{ model_name }}
*Précision:* {{ accuracy_score }}%
*Latence:* {{ latency_ms }}ms
*Utilisateurs affectés:* {{ affected_users|format_number }}

{% if trend %}
*Tendance:* {{ '📈' if trend == 'improving' else '📉' }} {{ trend|title }}
{% endif %}
            """,
            
            "summary_daily": """
📊 *Résumé Quotidien* - {{ date }}

*Alertes générées:* {{ alert_count|format_number }}
*Incidents résolus:* {{ resolved_count|format_number }}
*Temps de résolution moyen:* {{ avg_resolution_time|format_duration }}

*Top Services:*
{% for service in top_services %}
• {{ service.name }}: {{ service.alert_count }} alertes
{% endfor %}
            """
        }
    
    def render_alert_message(
        self,
        alert_data: Dict[str, Any],
        channel_config: Dict[str, Any],
        template_name: str = None
    ) -> Dict[str, Any]:
        """
        Génère un message Slack complet pour une alerte.
        
        Args:
            alert_data: Données de l'alerte
            channel_config: Configuration du canal Slack
            template_name: Nom du template à utiliser (optionnel)
            
        Returns:
            Message Slack formaté prêt à envoyer
        """
        start_time = time.time()
        
        # Extraction des informations principales
        severity = alert_data.get('severity', 'info')
        alert_type = alert_data.get('alert_type', 'unknown')
        tenant_id = alert_data.get('tenant_id', 'system')
        
        try:
            # Détermination du template
            if not template_name:
                template_name = self._select_template(alert_data)
            
            # Configuration du canal
            channel = channel_config.get('channel', '#alerts')
            
            # Options du message
            options = SlackMessageOptions(
                channel=channel,
                mention_users=channel_config.get('mention_users', []),
                mention_here=channel_config.get('mention_here', False),
                mention_channel=channel_config.get('mention_channel', False),
                rate_limit_key=f"{channel}:{tenant_id}",
                max_per_minute=channel_config.get('rate_limit', 10),
                correlation_id=alert_data.get('correlation_id', ''),
                alert_id=alert_data.get('alert_id', ''),
                tenant_id=tenant_id
            )
            
            # Vérification rate limiting
            if not self._check_rate_limit(options):
                slack_rate_limit_hits.labels(channel=channel).inc()
                logger.warning(
                    "Rate limit atteinte pour canal Slack",
                    channel=channel,
                    tenant_id=tenant_id
                )
                return None
            
            # Génération du message selon la complexité
            complexity = self._determine_complexity(alert_data, channel_config)
            
            if complexity == MessageComplexity.SIMPLE:
                message = self._render_simple_message(alert_data, options, template_name)
            elif complexity == MessageComplexity.STRUCTURED:
                message = self._render_structured_message(alert_data, options)
            elif complexity == MessageComplexity.INTERACTIVE:
                message = self._render_interactive_message(alert_data, options)
            else:  # RICH
                message = self._render_rich_message(alert_data, options)
            
            # Threading automatique
            thread_ts = self._get_thread_ts(alert_data, options)
            if thread_ts:
                message['thread_ts'] = thread_ts
            
            # Métriques
            slack_message_generation_total.labels(
                tenant_id=tenant_id,
                severity=severity,
                channel=channel,
                template_type=template_name
            ).inc()
            
            slack_message_generation_duration.labels(
                template_type=template_name,
                complexity_level=complexity.value
            ).observe(time.time() - start_time)
            
            logger.info(
                "Message Slack généré avec succès",
                template=template_name,
                complexity=complexity.value,
                channel=channel,
                tenant_id=tenant_id,
                has_thread=thread_ts is not None
            )
            
            return message
            
        except Exception as e:
            logger.error(
                "Erreur génération message Slack",
                template=template_name,
                channel=channel,
                tenant_id=tenant_id,
                error=str(e)
            )
            
            # Message d'erreur de fallback
            return self._create_fallback_message(alert_data, options, str(e))
    
    def _select_template(self, alert_data: Dict[str, Any]) -> str:
        """Sélectionne le template approprié selon les données."""
        severity = alert_data.get('severity', 'info')
        alert_type = alert_data.get('alert_type', '')
        
        # Templates spécialisés selon le type
        if 'ai_model' in alert_type or 'ai_inference' in alert_type:
            return 'ai_performance'
        elif severity in ['critical', 'emergency']:
            return 'alert_critical'
        elif severity == 'warning':
            return 'alert_warning'
        else:
            return 'alert_info'
    
    def _determine_complexity(
        self,
        alert_data: Dict[str, Any],
        channel_config: Dict[str, Any]
    ) -> MessageComplexity:
        """Détermine la complexité du message selon le contexte."""
        severity = alert_data.get('severity', 'info')
        has_actions = bool(alert_data.get('actions', []))
        has_metrics = bool(alert_data.get('metrics', []))
        has_attachments = bool(alert_data.get('attachments', []))
        
        # Messages interactifs pour alertes critiques avec actions
        if severity in ['critical', 'emergency'] and has_actions:
            return MessageComplexity.INTERACTIVE
        
        # Messages riches pour données complexes
        if has_attachments or len(alert_data.get('metrics', [])) > 5:
            return MessageComplexity.RICH
        
        # Messages structurés avec métriques
        if has_metrics or severity in ['warning', 'critical']:
            return MessageComplexity.STRUCTURED
        
        # Messages simples par défaut
        return MessageComplexity.SIMPLE
    
    def _render_simple_message(
        self,
        alert_data: Dict[str, Any],
        options: SlackMessageOptions,
        template_name: str
    ) -> Dict[str, Any]:
        """Génère un message Slack simple (texte seulement)."""
        template_source = self.default_templates.get(template_name, "{{ message }}")
        
        try:
            template = self.jinja_env.from_string(template_source)
            text = template.render(**alert_data)
            
            # Ajout des mentions si nécessaire
            if options.mention_here:
                text = "<!here> " + text
            elif options.mention_channel:
                text = "<!channel> " + text
            elif options.mention_users:
                mentions = " ".join(f"<@{user}>" for user in options.mention_users)
                text = f"{mentions} {text}"
            
            return {
                "channel": options.channel,
                "text": text,
                "username": options.username,
                "icon_emoji": options.icon_emoji
            }
            
        except Exception as e:
            logger.error(f"Erreur rendu template simple", template=template_name, error=str(e))
            return {
                "channel": options.channel,
                "text": f"Erreur affichage alerte: {alert_data.get('message', 'N/A')}",
                "username": options.username
            }
    
    def _render_structured_message(
        self,
        alert_data: Dict[str, Any],
        options: SlackMessageOptions
    ) -> Dict[str, Any]:
        """Génère un message Slack structuré avec blocks."""
        severity = alert_data.get('severity', 'info')
        severity_config = get_severity_config(severity)
        
        # Construction avec le builder
        self.builder.reset().set_options(options)
        
        # Header avec emoji de sévérité
        title = alert_data.get('title', alert_data.get('message', 'Alerte'))
        self.builder.add_header(title, severity_config.get('emoji', ''))
        
        # Section principale
        message = alert_data.get('description', alert_data.get('message', ''))
        fields = []
        
        # Ajout des champs contextuels
        context = alert_data.get('context', {})
        if context.get('service_name'):
            fields.append(SlackField("Service", context['service_name']))
        if context.get('environment'):
            fields.append(SlackField("Environnement", context['environment'].upper()))
        if context.get('metric_value'):
            value_str = f"{context['metric_value']}"
            if context.get('metric_unit'):
                value_str += f" {context['metric_unit']}"
            fields.append(SlackField("Valeur", value_str))
        
        self.builder.add_section(message, fields)
        
        # Métriques détaillées si disponibles
        metrics = alert_data.get('metrics', [])
        if metrics:
            metric_fields = [
                SlackField(metric.get('name', 'N/A'), f"{metric.get('value', 'N/A')} {metric.get('unit', '')}")
                for metric in metrics[:8]  # Limite Slack
            ]
            
            if metric_fields:
                self.builder.add_divider()
                self.builder.add_section("📊 *Métriques détaillées*", metric_fields)
        
        # Contexte temporel
        timestamp = alert_data.get('timestamp')
        tenant_id = alert_data.get('tenant_id', '')
        alert_id = alert_data.get('alert_id', '')
        
        context_elements = []
        if timestamp:
            context_elements.append(f"⏰ {timestamp}")
        if tenant_id:
            context_elements.append(f"🏢 {tenant_id}")
        if alert_id:
            context_elements.append(f"🆔 {alert_id[:8]}")
        
        if context_elements:
            self.builder.add_context(context_elements)
        
        return self.builder.build()
    
    def _render_interactive_message(
        self,
        alert_data: Dict[str, Any],
        options: SlackMessageOptions
    ) -> Dict[str, Any]:
        """Génère un message Slack interactif avec boutons."""
        # Base structurée
        message = self._render_structured_message(alert_data, options)
        
        # Ajout des actions
        actions_data = alert_data.get('actions', [])
        if actions_data:
            actions = []
            
            for action_data in actions_data[:5]:  # Limite Slack: 5 boutons
                action = SlackAction(
                    type="button",
                    text=action_data.get('label', 'Action'),
                    url=action_data.get('url', ''),
                    value=action_data.get('value', ''),
                    style=action_data.get('style', 'default')
                )
                
                # Confirmation pour actions dangereuses
                if action.style == 'danger':
                    action.confirm = {
                        'title': 'Confirmer l\'action',
                        'text': 'Cette action est irréversible.',
                        'confirm': 'Continuer',
                        'deny': 'Annuler'
                    }
                
                actions.append(action)
            
            # Mise à jour du builder avec les actions
            self.builder.blocks = [SlackBlock(block['type'], {k:v for k,v in block.items() if k != 'type'}) 
                                 for block in message.get('blocks', [])]
            self.builder.add_actions(actions)
            message = self.builder.build()
        
        return message
    
    def _render_rich_message(
        self,
        alert_data: Dict[str, Any],
        options: SlackMessageOptions
    ) -> Dict[str, Any]:
        """Génère un message Slack riche avec attachments."""
        # Base interactive
        message = self._render_interactive_message(alert_data, options)
        
        # Ajout d'attachments pour données complexes
        attachments = alert_data.get('attachments', [])
        
        if attachments:
            slack_attachments = []
            
            for attachment in attachments:
                if attachment.get('type') == 'chart':
                    # Graphique intégré
                    chart_attachment = {
                        'color': get_severity_config(alert_data.get('severity', 'info')).get('color'),
                        'title': attachment.get('title', 'Graphique'),
                        'title_link': attachment.get('url', ''),
                        'image_url': attachment.get('image_url', ''),
                        'footer': 'Spotify AI Agent Monitoring',
                        'ts': int(time.time())
                    }
                    slack_attachments.append(chart_attachment)
                
                elif attachment.get('type') == 'log':
                    # Logs d'erreur
                    log_attachment = {
                        'color': '#ff0000',
                        'title': 'Logs d\'erreur',
                        'text': f"```{attachment.get('content', 'N/A')}```",
                        'mrkdwn_in': ['text']
                    }
                    slack_attachments.append(log_attachment)
            
            if slack_attachments:
                message['attachments'] = slack_attachments
        
        return message
    
    def _get_thread_ts(
        self,
        alert_data: Dict[str, Any],
        options: SlackMessageOptions
    ) -> Optional[str]:
        """Détermine le timestamp de thread pour l'alerte."""
        if not self.redis_client:
            return None
        
        thread_strategy = options.thread_strategy
        
        # Clé de thread selon la stratégie
        if thread_strategy == ThreadStrategy.BY_ALERT_TYPE:
            thread_key = f"thread:{options.channel}:{alert_data.get('alert_type', 'unknown')}"
        elif thread_strategy == ThreadStrategy.BY_SERVICE:
            service = alert_data.get('context', {}).get('service_name', 'unknown')
            thread_key = f"thread:{options.channel}:service:{service}"
        elif thread_strategy == ThreadStrategy.BY_SEVERITY:
            thread_key = f"thread:{options.channel}:{alert_data.get('severity', 'info')}"
        elif thread_strategy == ThreadStrategy.CORRELATION_ID:
            correlation_id = alert_data.get('correlation_id', '')
            if correlation_id:
                thread_key = f"thread:{options.channel}:correlation:{correlation_id}"
            else:
                return None
        else:
            return None
        
        try:
            # Récupération du thread existant
            thread_ts = self.redis_client.get(thread_key)
            
            if thread_ts:
                # Vérification que le thread n'est pas trop ancien (24h)
                thread_age = time.time() - float(thread_ts)
                if thread_age < 86400:  # 24 heures
                    active_slack_threads.inc()
                    return thread_ts.decode('utf-8')
                else:
                    # Thread expiré, suppression
                    self.redis_client.delete(thread_key)
            
            return None
            
        except Exception as e:
            logger.debug(f"Erreur récupération thread", key=thread_key, error=str(e))
            return None
    
    def update_thread_timestamp(
        self,
        channel: str,
        thread_key: str,
        message_ts: str
    ):
        """Met à jour le timestamp d'un thread après envoi."""
        if self.redis_client:
            try:
                full_key = f"thread:{channel}:{thread_key}"
                self.redis_client.setex(full_key, 86400, message_ts)  # 24h TTL
            except Exception as e:
                logger.debug(f"Erreur mise à jour thread", key=full_key, error=str(e))
    
    def _check_rate_limit(self, options: SlackMessageOptions) -> bool:
        """Vérifie les limites de taux pour un canal."""
        rate_key = options.rate_limit_key or options.channel
        
        if rate_key not in self._rate_limiters:
            self._rate_limiters[rate_key] = RateLimiter(
                max_calls=options.max_per_minute,
                period=60
            )
        
        limiter = self._rate_limiters[rate_key]
        
        try:
            with limiter:
                return True
        except:
            return False
    
    def _create_fallback_message(
        self,
        alert_data: Dict[str, Any],
        options: SlackMessageOptions,
        error: str
    ) -> Dict[str, Any]:
        """Crée un message de fallback en cas d'erreur."""
        return {
            "channel": options.channel,
            "text": f"⚠️ Erreur génération alerte: {error}\n\nDonnées brutes: {json.dumps(alert_data, indent=2)[:500]}",
            "username": options.username,
            "icon_emoji": ":warning:"
        }
    
    # Filtres Jinja2 pour Slack
    
    def _slack_escape(self, text: str) -> str:
        """Échappe les caractères spéciaux Slack."""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;'))
    
    def _slack_link(self, url: str, text: str = None) -> str:
        """Crée un lien Slack formaté."""
        if not text:
            text = url
        return f"<{url}|{text}>"
    
    def _slack_mention(self, user: str) -> str:
        """Crée une mention utilisateur Slack."""
        if user.startswith('@'):
            user = user[1:]
        return f"<@{user}>"
    
    def _truncate_smart(self, text: str, length: int = 100) -> str:
        """Troncature intelligente pour Slack."""
        if len(text) <= length:
            return text
        
        truncated = text[:length]
        last_space = truncated.rfind(' ')
        
        if last_space > length * 0.7:
            return truncated[:last_space] + "..."
        
        return truncated + "..."
    
    def _format_number(self, value: Union[int, float]) -> str:
        """Formate un nombre pour affichage."""
        if isinstance(value, float):
            if value >= 1000000:
                return f"{value/1000000:.1f}M"
            elif value >= 1000:
                return f"{value/1000:.1f}K"
            else:
                return f"{value:.2f}"
        else:
            if value >= 1000000:
                return f"{value//1000000}M"
            elif value >= 1000:
                return f"{value//1000}K"
            else:
                return str(value)
    
    def _format_duration(self, seconds: Union[int, float]) -> str:
        """Formate une durée en texte lisible."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.0f}m"
        elif seconds < 86400:
            return f"{seconds/3600:.1f}h"
        else:
            return f"{seconds/86400:.1f}j"
    
    def _severity_emoji(self, severity: str) -> str:
        """Retourne l'emoji pour une sévérité."""
        return get_severity_config(severity).get('emoji', '❓')
    
    def _ai_context_emoji(self, context_type: str) -> str:
        """Retourne l'emoji pour un contexte IA."""
        emoji_map = {
            'music': '🎵',
            'recommendation': '🎯',
            'generation': '🎼',
            'analysis': '📊',
            'training': '🎓',
            'inference': '🧠'
        }
        return emoji_map.get(context_type, '🤖')

# Fonctions utilitaires

def create_quick_alert(
    message: str,
    severity: str = "info",
    channel: str = "#alerts"
) -> Dict[str, Any]:
    """Création rapide d'un message d'alerte Slack."""
    engine = SlackTemplateEngine()
    
    alert_data = {
        'message': message,
        'severity': severity,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    channel_config = {
        'channel': channel
    }
    
    return engine.render_alert_message(alert_data, channel_config)

def create_ai_performance_alert(
    model_name: str,
    accuracy_drop: float,
    affected_users: int,
    tenant_id: str,
    channel: str = "#ml-alerts"
) -> Dict[str, Any]:
    """Création d'une alerte de performance IA."""
    engine = SlackTemplateEngine()
    
    severity = "critical" if accuracy_drop > 10 else "warning"
    
    alert_data = {
        'alert_type': 'ai_model_performance',
        'severity': severity,
        'model_name': model_name,
        'accuracy_drop': accuracy_drop,
        'affected_users': affected_users,
        'tenant_id': tenant_id,
        'timestamp': datetime.utcnow().isoformat(),
        'actions': [
            {
                'label': 'Redémarrer Modèle',
                'url': f'/api/models/{model_name}/restart',
                'style': 'primary'
            },
            {
                'label': 'Voir Métriques',
                'url': f'/monitoring/models/{model_name}',
                'style': 'default'
            }
        ]
    }
    
    channel_config = {
        'channel': channel,
        'mention_users': ['@ml-team'] if severity == 'critical' else []
    }
    
    return engine.render_alert_message(alert_data, channel_config)
