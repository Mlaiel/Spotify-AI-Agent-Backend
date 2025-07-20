"""
Slack Template Engine - Moteur de génération de templates Slack pour Alertmanager
Système avancé de génération de templates multi-tenant avec localisation
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, Template, select_autoescape
import redis
from dataclasses import dataclass, asdict
from enum import Enum

from .locale_manager import LocaleManager
from .template_validator import SlackTemplateValidator


class AlertSeverity(Enum):
    """Niveaux de sévérité des alertes"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    RECOVERY = "recovery"


class SlackColor(Enum):
    """Codes couleur Slack pour les attachments"""
    CRITICAL = "#FF0000"  # Rouge
    WARNING = "#FFA500"   # Orange
    INFO = "#36A64F"      # Vert
    RECOVERY = "#00FF00"  # Vert clair


@dataclass
class AlertContext:
    """Contexte d'une alerte pour la génération de template"""
    alert_name: str
    severity: AlertSeverity
    tenant_id: str
    instance: str
    timestamp: datetime
    description: str
    runbook_url: Optional[str] = None
    dashboard_url: Optional[str] = None
    labels: Optional[Dict[str, str]] = None
    annotations: Optional[Dict[str, str]] = None
    fingerprint: Optional[str] = None
    generator_url: Optional[str] = None


class SlackTemplateEngine:
    """
    Moteur de génération de templates Slack pour Alertmanager
    
    Fonctionnalités :
    - Génération de templates multi-tenant
    - Support de localisation
    - Cache Redis pour performances
    - Validation des payloads
    - Customisation par tenant
    """

    def __init__(
        self,
        templates_dir: str = "/templates",
        redis_client: Optional[redis.Redis] = None,
        cache_ttl: int = 3600,
        locale_manager: Optional[LocaleManager] = None,
        validator: Optional[SlackTemplateValidator] = None
    ):
        self.templates_dir = Path(templates_dir)
        self.cache_ttl = cache_ttl
        self.redis_client = redis_client
        self.locale_manager = locale_manager or LocaleManager()
        self.validator = validator or SlackTemplateValidator()
        
        # Configuration Jinja2
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Ajout de fonctions personnalisées à Jinja2
        self._register_template_functions()
        
        self.logger = logging.getLogger(__name__)

    def _register_template_functions(self):
        """Enregistre les fonctions personnalisées pour les templates Jinja2"""
        
        def format_timestamp(timestamp: datetime, locale: str = "fr-FR") -> str:
            """Formate un timestamp selon la locale"""
            return self.locale_manager.format_datetime(timestamp, locale)
        
        def severity_color(severity: str) -> str:
            """Retourne la couleur Slack pour une sévérité"""
            try:
                return SlackColor[severity.upper()].value
            except KeyError:
                return SlackColor.INFO.value
        
        def truncate_text(text: str, max_length: int = 100) -> str:
            """Tronque un texte avec ellipses"""
            if len(text) <= max_length:
                return text
            return text[:max_length-3] + "..."
        
        def escape_markdown(text: str) -> str:
            """Échape les caractères markdown Slack"""
            escape_chars = ['*', '_', '`', '~']
            for char in escape_chars:
                text = text.replace(char, f'\\{char}')
            return text

        # Enregistrement des fonctions
        self.jinja_env.globals.update({
            'format_timestamp': format_timestamp,
            'severity_color': severity_color,
            'truncate_text': truncate_text,
            'escape_markdown': escape_markdown,
            'now': datetime.now
        })

    def _get_cache_key(self, template_type: str, tenant_id: str, locale: str) -> str:
        """Génère une clé de cache pour un template"""
        return f"slack_template:{template_type}:{tenant_id}:{locale}"

    def _get_cached_template(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Récupère un template depuis le cache Redis"""
        if not self.redis_client:
            return None
        
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data.decode('utf-8'))
        except Exception as e:
            self.logger.warning(f"Erreur lors de la récupération du cache: {e}")
        
        return None

    def _cache_template(self, cache_key: str, template_data: Dict[str, Any]):
        """Met en cache un template dans Redis"""
        if not self.redis_client:
            return
        
        try:
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(template_data, default=str)
            )
        except Exception as e:
            self.logger.warning(f"Erreur lors de la mise en cache: {e}")

    def _load_template_file(self, template_path: str) -> Template:
        """Charge un fichier template Jinja2"""
        try:
            return self.jinja_env.get_template(template_path)
        except Exception as e:
            self.logger.error(f"Impossible de charger le template {template_path}: {e}")
            raise

    def _get_tenant_config(self, tenant_id: str) -> Dict[str, Any]:
        """Récupère la configuration spécifique d'un tenant"""
        # Configuration par défaut
        default_config = {
            "brand_name": "Spotify AI Agent",
            "brand_color": "#1DB954",
            "brand_logo": "https://spotify-ai-agent.com/logo.png",
            "support_channel": "#support",
            "escalation_channel": "#ops-critical",
            "custom_fields": {},
            "footer_text": "Spotify AI Agent Monitoring",
            "timezone": "Europe/Paris"
        }
        
        # Ici, on récupérerait la config depuis la base de données
        # Pour l'instant, on retourne la config par défaut
        return default_config

    def generate_alert_template(
        self,
        alert_context: AlertContext,
        locale: str = "fr-FR",
        template_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Génère un template Slack pour une alerte
        
        Args:
            alert_context: Contexte de l'alerte
            locale: Locale pour la localisation
            template_override: Template personnalisé optionnel
            
        Returns:
            Dict contenant le payload Slack formaté
        """
        
        # Génération de la clé de cache
        cache_key = self._get_cache_key(
            alert_context.severity.value,
            alert_context.tenant_id,
            locale
        )
        
        # Vérification du cache
        cached_template = self._get_cached_template(cache_key)
        if cached_template and not template_override:
            # Mise à jour des données dynamiques
            cached_template.update(self._get_dynamic_context(alert_context, locale))
            return cached_template
        
        # Chargement de la configuration tenant
        tenant_config = self._get_tenant_config(alert_context.tenant_id)
        
        # Détermination du template à utiliser
        template_path = template_override or self._get_template_path(alert_context.severity)
        
        # Chargement du template
        template = self._load_template_file(template_path)
        
        # Préparation du contexte complet
        template_context = self._prepare_template_context(
            alert_context, tenant_config, locale
        )
        
        # Rendu du template
        rendered_template = template.render(**template_context)
        slack_payload = json.loads(rendered_template)
        
        # Validation du payload
        if self.validator.validate_slack_payload(slack_payload):
            # Mise en cache
            self._cache_template(cache_key, slack_payload)
            return slack_payload
        else:
            raise ValueError("Le payload Slack généré n'est pas valide")

    def _get_template_path(self, severity: AlertSeverity) -> str:
        """Détermine le chemin du template selon la sévérité"""
        template_mapping = {
            AlertSeverity.CRITICAL: "critical/alert_critical.json.j2",
            AlertSeverity.WARNING: "warning/alert_warning.json.j2", 
            AlertSeverity.INFO: "info/alert_info.json.j2",
            AlertSeverity.RECOVERY: "core/recovery.json.j2"
        }
        
        return template_mapping.get(severity, "core/alert_base.json.j2")

    def _prepare_template_context(
        self,
        alert_context: AlertContext,
        tenant_config: Dict[str, Any],
        locale: str
    ) -> Dict[str, Any]:
        """Prépare le contexte complet pour le rendu du template"""
        
        # Localisation des textes
        localized_texts = self.locale_manager.get_localized_texts(
            "slack_templates", locale
        )
        
        return {
            # Données de l'alerte
            "alert": asdict(alert_context),
            
            # Configuration du tenant
            "tenant": tenant_config,
            
            # Textes localisés
            "texts": localized_texts,
            
            # Métadonnées
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "locale": locale,
                "template_version": "1.0.0",
                "engine_version": "2.0.0"
            },
            
            # Utilitaires
            "utils": {
                "severity_color": SlackColor[alert_context.severity.name].value,
                "timestamp_formatted": self.locale_manager.format_datetime(
                    alert_context.timestamp, locale
                ),
                "urgency_emoji": self._get_urgency_emoji(alert_context.severity)
            }
        }

    def _get_dynamic_context(self, alert_context: AlertContext, locale: str) -> Dict[str, Any]:
        """Génère le contexte dynamique pour mise à jour du cache"""
        return {
            "timestamp": alert_context.timestamp.isoformat(),
            "fingerprint": alert_context.fingerprint,
            "instance": alert_context.instance,
            "formatted_timestamp": self.locale_manager.format_datetime(
                alert_context.timestamp, locale
            )
        }

    def _get_urgency_emoji(self, severity: AlertSeverity) -> str:
        """Retourne l'emoji correspondant à la sévérité"""
        emoji_mapping = {
            AlertSeverity.CRITICAL: "🚨",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.RECOVERY: "✅"
        }
        
        return emoji_mapping.get(severity, "❓")

    def generate_batch_templates(
        self,
        alert_contexts: List[AlertContext],
        locale: str = "fr-FR"
    ) -> List[Dict[str, Any]]:
        """Génère des templates en lot pour améliorer les performances"""
        
        templates = []
        for alert_context in alert_contexts:
            try:
                template = self.generate_alert_template(alert_context, locale)
                templates.append(template)
            except Exception as e:
                self.logger.error(
                    f"Erreur lors de la génération du template pour {alert_context.alert_name}: {e}"
                )
                continue
        
        return templates

    def clear_cache(self, pattern: Optional[str] = None):
        """Vide le cache des templates"""
        if not self.redis_client:
            return
        
        try:
            if pattern:
                keys = self.redis_client.keys(pattern)
            else:
                keys = self.redis_client.keys("slack_template:*")
            
            if keys:
                self.redis_client.delete(*keys)
                self.logger.info(f"Cache vidé: {len(keys)} clés supprimées")
        except Exception as e:
            self.logger.error(f"Erreur lors du vidage du cache: {e}")

    def get_template_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques d'utilisation des templates"""
        stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_generations": 0,
            "error_count": 0,
            "avg_generation_time": 0.0
        }
        
        # Ici, on récupérerait les vraies statistiques depuis Redis/BDD
        return stats
