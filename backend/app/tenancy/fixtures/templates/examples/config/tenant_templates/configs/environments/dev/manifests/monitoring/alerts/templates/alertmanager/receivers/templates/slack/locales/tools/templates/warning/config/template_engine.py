"""
Moteur de Templates d'Alertes - Spotify AI Agent
===============================================

Moteur ultra-avancé pour la génération de templates d'alertes personnalisés
avec support Jinja2, cache intelligent et optimisation des performances.

Auteur: Équipe d'experts dirigée par Fahed Mlaiel
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import jinja2
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import re
from abc import ABC, abstractmethod

# Configuration du logging
logger = logging.getLogger(__name__)

@dataclass
class TemplateContext:
    """Contexte pour le rendu des templates."""
    alert_id: str
    tenant_id: str
    level: str
    message: str
    timestamp: datetime
    source: str
    metadata: Dict[str, Any]
    escalation_count: int = 0
    previous_alerts: List[str] = None
    
    def __post_init__(self):
        if self.previous_alerts is None:
            self.previous_alerts = []

@dataclass
class TemplateConfig:
    """Configuration d'un template."""
    template_id: str
    name: str
    description: str
    format_type: str  # slack, email, webhook, sms
    template_content: str
    variables: List[str]
    filters: List[str]
    macros: Dict[str, str]
    cache_ttl: int
    version: str
    created_at: datetime
    updated_at: datetime

class TemplateRenderer(ABC):
    """Interface abstraite pour les renderers de templates."""
    
    @abstractmethod
    def render(self, template_content: str, context: TemplateContext) -> str:
        """Rend un template avec le contexte donné."""
        pass
    
    @abstractmethod
    def validate_template(self, template_content: str) -> bool:
        """Valide la syntaxe d'un template."""
        pass

class JinjaTemplateRenderer(TemplateRenderer):
    """Renderer Jinja2 avec fonctions personnalisées."""
    
    def __init__(self):
        """Initialise le renderer Jinja2."""
        self.env = jinja2.Environment(
            loader=jinja2.BaseLoader(),
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Ajout des filtres personnalisés
        self._register_custom_filters()
        self._register_custom_functions()
    
    def _register_custom_filters(self):
        """Enregistre les filtres personnalisés Jinja2."""
        
        def format_timestamp(value, format_str='%Y-%m-%d %H:%M:%S'):
            """Filtre pour formater les timestamps."""
            if isinstance(value, str):
                value = datetime.fromisoformat(value.replace('Z', '+00:00'))
            return value.strftime(format_str)
        
        def truncate_smart(value, length=100, suffix='...'):
            """Filtre pour tronquer intelligemment le texte."""
            if len(value) <= length:
                return value
            
            # Trouve le dernier espace avant la limite
            truncated = value[:length]
            last_space = truncated.rfind(' ')
            
            if last_space > length * 0.8:  # Si l'espace est proche de la fin
                return truncated[:last_space] + suffix
            else:
                return truncated + suffix
        
        def highlight_keywords(value, keywords, highlight_format='**{}**'):
            """Filtre pour mettre en évidence des mots-clés."""
            if not keywords:
                return value
            
            result = value
            for keyword in keywords:
                pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                result = pattern.sub(highlight_format.format(keyword), result)
            
            return result
        
        def severity_color(level):
            """Filtre pour obtenir la couleur selon la sévérité."""
            colors = {
                'CRITICAL': '#FF0000',
                'HIGH': '#FF8C00',
                'WARNING': '#FFD700',
                'INFO': '#00CED1',
                'DEBUG': '#808080'
            }
            return colors.get(level.upper(), '#808080')
        
        def format_duration(seconds):
            """Filtre pour formater une durée."""
            if seconds < 60:
                return f"{seconds}s"
            elif seconds < 3600:
                minutes = seconds // 60
                return f"{minutes}m"
            else:
                hours = seconds // 3600
                minutes = (seconds % 3600) // 60
                return f"{hours}h {minutes}m"
        
        # Enregistrement des filtres
        self.env.filters['format_timestamp'] = format_timestamp
        self.env.filters['truncate_smart'] = truncate_smart
        self.env.filters['highlight_keywords'] = highlight_keywords
        self.env.filters['severity_color'] = severity_color
        self.env.filters['format_duration'] = format_duration
    
    def _register_custom_functions(self):
        """Enregistre les fonctions personnalisées Jinja2."""
        
        def get_escalation_message(escalation_count):
            """Génère un message d'escalade selon le nombre."""
            if escalation_count == 0:
                return ""
            elif escalation_count == 1:
                return "🔔 ESCALADE NIVEAU 1"
            elif escalation_count == 2:
                return "🚨 ESCALADE NIVEAU 2 - ATTENTION REQUISE"
            else:
                return f"🆘 ESCALADE CRITIQUE NIVEAU {escalation_count}"
        
        def generate_alert_link(alert_id, tenant_id):
            """Génère un lien vers l'alerte."""
            base_url = os.getenv('DASHBOARD_URL', 'https://spotify-ai-agent.com')
            return f"{base_url}/alerts/{tenant_id}/{alert_id}"
        
        def format_metadata(metadata, max_items=5):
            """Formate les métadonnées pour l'affichage."""
            if not metadata:
                return "Aucune métadonnée"
            
            items = []
            for key, value in list(metadata.items())[:max_items]:
                items.append(f"• {key}: {value}")
            
            if len(metadata) > max_items:
                items.append(f"... et {len(metadata) - max_items} autres")
            
            return '\n'.join(items)
        
        # Enregistrement des fonctions globales
        self.env.globals['get_escalation_message'] = get_escalation_message
        self.env.globals['generate_alert_link'] = generate_alert_link
        self.env.globals['format_metadata'] = format_metadata
        self.env.globals['now'] = datetime.now
    
    def render(self, template_content: str, context: TemplateContext) -> str:
        """Rend un template avec le contexte donné."""
        try:
            template = self.env.from_string(template_content)
            context_dict = asdict(context)
            context_dict['timestamp'] = context.timestamp.isoformat()
            
            rendered = template.render(**context_dict)
            return rendered.strip()
            
        except jinja2.TemplateError as e:
            logger.error(f"Erreur rendu template: {e}")
            return f"Erreur de rendu: {str(e)}"
        except Exception as e:
            logger.error(f"Erreur inattendue rendu template: {e}")
            return f"Erreur inattendue: {str(e)}"
    
    def validate_template(self, template_content: str) -> bool:
        """Valide la syntaxe d'un template."""
        try:
            self.env.from_string(template_content)
            return True
        except jinja2.TemplateError:
            return False

class TemplateCache:
    """Cache optimisé pour les templates compilés."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = {}
        self.access_times = {}
        self.lock = threading.RLock()
    
    def get_template_hash(self, template_content: str) -> str:
        """Génère un hash pour le contenu du template."""
        return hashlib.sha256(template_content.encode()).hexdigest()[:16]
    
    def get(self, template_hash: str) -> Optional[jinja2.Template]:
        """Récupère un template compilé du cache."""
        with self.lock:
            if template_hash in self.cache:
                template_data = self.cache[template_hash]
                
                # Vérification de l'expiration
                if datetime.now() < template_data['expires_at']:
                    self.access_times[template_hash] = datetime.now()
                    return template_data['template']
                else:
                    # Template expiré
                    del self.cache[template_hash]
                    del self.access_times[template_hash]
            
            return None
    
    def set(self, template_hash: str, template: jinja2.Template, ttl: Optional[int] = None) -> bool:
        """Met en cache un template compilé."""
        ttl = ttl or self.default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)
        
        with self.lock:
            # Nettoyage si le cache est plein
            if len(self.cache) >= self.max_size:
                self._cleanup_cache()
            
            self.cache[template_hash] = {
                'template': template,
                'expires_at': expires_at,
                'created_at': datetime.now()
            }
            self.access_times[template_hash] = datetime.now()
            
            return True
    
    def _cleanup_cache(self):
        """Nettoie le cache en supprimant les entrées les moins récemment utilisées."""
        if not self.access_times:
            return
        
        # Tri par temps d'accès (LRU)
        sorted_items = sorted(self.access_times.items(), key=lambda x: x[1])
        items_to_remove = len(sorted_items) // 4  # Supprime 25% des éléments
        
        for template_hash, _ in sorted_items[:items_to_remove]:
            self.cache.pop(template_hash, None)
            self.access_times.pop(template_hash, None)

class AlertTemplateEngine:
    """
    Moteur de templates ultra-avancé pour les alertes Warning.
    
    Fonctionnalités:
    - Support Jinja2 avec filtres et fonctions personnalisés
    - Cache intelligent avec LRU et expiration
    - Templates prédéfinis pour Slack, Email, Webhook
    - Validation et précompilation des templates
    - Optimisation des performances avec compilation asynchrone
    - Metrics et monitoring intégrés
    """
    
    def __init__(self, templates_dir: str = None, cache_size: int = 1000):
        """Initialise le moteur de templates."""
        self.templates_dir = Path(templates_dir or os.getcwd()) / "templates"
        self.templates_dir.mkdir(exist_ok=True)
        
        # Initialisation des composants
        self.renderer = JinjaTemplateRenderer()
        self.cache = TemplateCache(max_size=cache_size)
        self.templates_config = {}
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Métriques
        self.metrics = {
            'templates_rendered': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'render_errors': 0,
            'validation_errors': 0
        }
        self.metrics_lock = threading.RLock()
        
        # Chargement des templates par défaut
        self._load_default_templates()
        
        logger.info("AlertTemplateEngine initialisé avec succès")
    
    def _load_default_templates(self):
        """Charge les templates par défaut pour chaque format."""
        
        # Template Slack par défaut
        slack_template = """
🚨 **{{ level }} Alert** - Spotify AI Agent

**Message:** {{ message | truncate_smart(200) }}
**Source:** {{ source }}
**Time:** {{ timestamp | format_timestamp('%Y-%m-%d %H:%M:%S UTC') }}
**Tenant:** {{ tenant_id }}

{% if escalation_count > 0 %}
{{ get_escalation_message(escalation_count) }}
{% endif %}

**Details:**
{{ format_metadata(metadata, 8) }}

{% if previous_alerts %}
**Recent Related Alerts:** {{ previous_alerts | length }}
{% endif %}

🔗 [View Alert]({{ generate_alert_link(alert_id, tenant_id) }})
        """.strip()
        
        # Template Email par défaut
        email_template = """
Subject: [{{ level }}] Spotify AI Agent Alert - {{ message | truncate_smart(50) }}

Bonjour,

Une alerte de niveau {{ level }} a été détectée dans le système Spotify AI Agent.

DÉTAILS DE L'ALERTE:
===================
ID de l'alerte: {{ alert_id }}
Niveau: {{ level }}
Message: {{ message }}
Source: {{ source }}
Tenant: {{ tenant_id }}
Horodatage: {{ timestamp | format_timestamp('%d/%m/%Y à %H:%M:%S UTC') }}

{% if escalation_count > 0 %}
ESCALADE: {{ get_escalation_message(escalation_count) }}
Cette alerte a été escaladée {{ escalation_count }} fois.
{% endif %}

MÉTADONNÉES:
============
{{ format_metadata(metadata, 10) }}

{% if previous_alerts %}
ALERTES CONNEXES:
================
{{ previous_alerts | length }} alertes similaires détectées récemment.
{% endif %}

Vous pouvez consulter les détails complets de cette alerte en cliquant sur le lien suivant:
{{ generate_alert_link(alert_id, tenant_id) }}

Cordialement,
L'équipe Spotify AI Agent
        """.strip()
        
        # Template Webhook JSON par défaut
        webhook_template = """
{
  "alert_id": "{{ alert_id }}",
  "tenant_id": "{{ tenant_id }}",
  "level": "{{ level }}",
  "message": "{{ message | replace('"', '\\"') }}",
  "source": "{{ source }}",
  "timestamp": "{{ timestamp | format_timestamp('%Y-%m-%dT%H:%M:%S.%fZ') }}",
  "escalation_count": {{ escalation_count }},
  "metadata": {{ metadata | tojson }},
  "previous_alerts_count": {{ previous_alerts | length }},
  "alert_url": "{{ generate_alert_link(alert_id, tenant_id) }}",
  "severity_color": "{{ level | severity_color }}",
  "escalation_message": "{{ get_escalation_message(escalation_count) }}"
}
        """.strip()
        
        # Enregistrement des templates par défaut
        default_templates = {
            "warning_slack_default": TemplateConfig(
                template_id="warning_slack_default",
                name="Warning Slack Default",
                description="Template par défaut pour les alertes Warning sur Slack",
                format_type="slack",
                template_content=slack_template,
                variables=["alert_id", "tenant_id", "level", "message", "source", "timestamp"],
                filters=["truncate_smart", "format_timestamp", "format_metadata"],
                macros={"escalation": "get_escalation_message", "link": "generate_alert_link"},
                cache_ttl=3600,
                version="1.0.0",
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            "warning_email_default": TemplateConfig(
                template_id="warning_email_default",
                name="Warning Email Default",
                description="Template par défaut pour les alertes Warning par email",
                format_type="email",
                template_content=email_template,
                variables=["alert_id", "tenant_id", "level", "message", "source", "timestamp"],
                filters=["truncate_smart", "format_timestamp", "format_metadata"],
                macros={"escalation": "get_escalation_message", "link": "generate_alert_link"},
                cache_ttl=3600,
                version="1.0.0",
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            "warning_webhook_default": TemplateConfig(
                template_id="warning_webhook_default",
                name="Warning Webhook Default",
                description="Template par défaut pour les alertes Warning via webhook",
                format_type="webhook",
                template_content=webhook_template,
                variables=["alert_id", "tenant_id", "level", "message", "source", "timestamp"],
                filters=["tojson", "format_timestamp", "severity_color", "replace"],
                macros={"escalation": "get_escalation_message", "link": "generate_alert_link"},
                cache_ttl=3600,
                version="1.0.0",
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        }
        
        self.templates_config.update(default_templates)
    
    def render_alert_template(self, template_id: str, context: TemplateContext) -> str:
        """Rend un template d'alerte avec le contexte donné."""
        
        if template_id not in self.templates_config:
            self._increment_metric('render_errors')
            raise ValueError(f"Template non trouvé: {template_id}")
        
        template_config = self.templates_config[template_id]
        template_content = template_config.template_content
        
        # Vérification du cache
        template_hash = self.cache.get_template_hash(template_content)
        cached_template = self.cache.get(template_hash)
        
        if cached_template:
            self._increment_metric('cache_hits')
            try:
                context_dict = asdict(context)
                context_dict['timestamp'] = context.timestamp.isoformat()
                rendered = cached_template.render(**context_dict)
                self._increment_metric('templates_rendered')
                return rendered.strip()
            except Exception as e:
                logger.error(f"Erreur rendu template en cache: {e}")
                self._increment_metric('render_errors')
                return f"Erreur de rendu: {str(e)}"
        else:
            self._increment_metric('cache_misses')
        
        # Rendu avec compilation et mise en cache
        try:
            # Compilation du template
            compiled_template = self.renderer.env.from_string(template_content)
            
            # Mise en cache du template compilé
            self.cache.set(template_hash, compiled_template, template_config.cache_ttl)
            
            # Rendu
            context_dict = asdict(context)
            context_dict['timestamp'] = context.timestamp.isoformat()
            rendered = compiled_template.render(**context_dict)
            
            self._increment_metric('templates_rendered')
            return rendered.strip()
            
        except Exception as e:
            logger.error(f"Erreur rendu template {template_id}: {e}")
            self._increment_metric('render_errors')
            return f"Erreur de rendu template {template_id}: {str(e)}"
    
    def register_template(self, template_config: TemplateConfig) -> bool:
        """Enregistre un nouveau template."""
        
        # Validation du template
        if not self.renderer.validate_template(template_config.template_content):
            self._increment_metric('validation_errors')
            raise ValueError(f"Template invalide: {template_config.template_id}")
        
        # Enregistrement
        self.templates_config[template_config.template_id] = template_config
        
        logger.info(f"Template enregistré: {template_config.template_id}")
        return True
    
    def get_template_for_format(self, format_type: str, tenant_id: str = None) -> str:
        """Récupère l'ID du template approprié pour un format donné."""
        
        # Recherche d'un template spécifique au tenant
        if tenant_id:
            tenant_template_id = f"warning_{format_type}_{tenant_id}"
            if tenant_template_id in self.templates_config:
                return tenant_template_id
        
        # Template par défaut pour le format
        default_template_id = f"warning_{format_type}_default"
        if default_template_id in self.templates_config:
            return default_template_id
        
        # Fallback sur le template Slack par défaut
        return "warning_slack_default"
    
    def create_template_context(self, alert_id: str, tenant_id: str, level: str,
                              message: str, source: str, metadata: Dict[str, Any] = None,
                              escalation_count: int = 0, previous_alerts: List[str] = None) -> TemplateContext:
        """Crée un contexte de template à partir des paramètres."""
        
        return TemplateContext(
            alert_id=alert_id,
            tenant_id=tenant_id,
            level=level,
            message=message,
            timestamp=datetime.now(),
            source=source,
            metadata=metadata or {},
            escalation_count=escalation_count,
            previous_alerts=previous_alerts or []
        )
    
    def _increment_metric(self, metric_name: str):
        """Incrémente une métrique."""
        with self.metrics_lock:
            self.metrics[metric_name] = self.metrics.get(metric_name, 0) + 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques du moteur."""
        with self.metrics_lock:
            metrics = self.metrics.copy()
            metrics['templates_count'] = len(self.templates_config)
            metrics['cache_size'] = len(self.cache.cache)
            return metrics
    
    def cleanup(self):
        """Nettoie les ressources."""
        self.executor.shutdown(wait=True)
        logger.info("AlertTemplateEngine nettoyé avec succès")

# Factory function
def create_alert_template_engine(templates_dir: str = None, cache_size: int = 1000) -> AlertTemplateEngine:
    """Factory function pour créer un moteur de templates."""
    return AlertTemplateEngine(templates_dir, cache_size)
