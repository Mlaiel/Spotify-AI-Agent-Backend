#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gestionnaire de Templates Slack Avancé avec Rendu Jinja2

Ce module fournit un système de rendu de templates Slack sophistiqué avec:
- Templates Jinja2 avec macros et héritages
- Cache intelligent multi-niveau
- Validation de templates
- Support des attachments riches
- Formatage contextuel par tenant
- Métriques de performance
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import jinja2
from jinja2 import Environment, FileSystemLoader, BaseLoader, TemplateNotFound
from jinja2.sandbox import SandboxedEnvironment
import structlog
from prometheus_client import Counter, Histogram, Gauge
from dataclasses import dataclass, field
import hashlib

logger = structlog.get_logger(__name__)

# Métriques Prometheus
TEMPLATE_RENDER_REQUESTS = Counter(
    'slack_template_render_requests_total',
    'Total template render requests',
    ['template', 'tenant', 'status']
)

TEMPLATE_RENDER_DURATION = Histogram(
    'slack_template_render_duration_seconds',
    'Template rendering duration',
    ['template', 'type']
)

TEMPLATE_CACHE_HITS = Counter(
    'slack_template_cache_hits_total',
    'Template cache hits',
    ['cache_type', 'template']
)

ACTIVE_TEMPLATES = Gauge(
    'slack_templates_active_total',
    'Number of active templates'
)

@dataclass
class SlackTemplate:
    """Représentation d'un template Slack avec métadonnées."""
    name: str
    path: str
    content: str
    variables: List[str] = field(default_factory=list)
    macros: List[str] = field(default_factory=list)
    extends: Optional[str] = None
    version: str = "1.0.0"
    description: Optional[str] = None
    author: Optional[str] = None
    last_modified: Optional[datetime] = None
    checksum: Optional[str] = None
    
    def __post_init__(self):
        if self.checksum is None:
            self.checksum = hashlib.md5(self.content.encode()).hexdigest()

@dataclass
class RenderContext:
    """Contexte de rendu avec données enrichies."""
    alert: Dict[str, Any]
    tenant: Dict[str, Any]
    config: Dict[str, Any]
    locale: str
    timezone: str = "UTC"
    timestamp: Optional[datetime] = None
    user_context: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

class SlackTemplateRenderer:
    """
    Moteur de rendu de templates Slack avec Jinja2 sécurisé.
    """
    
    def __init__(self, config: Dict[str, Any], localization_engine=None):
        self.config = config
        self.localization_engine = localization_engine
        self.logger = logger.bind(component="template_renderer")
        
        # Configuration
        self.templates_dir = Path(__file__).parent / "templates"
        self.cache_ttl = config.get("cache_ttl", 300)
        self.auto_reload = config.get("auto_reload", True)
        self.validation_enabled = config.get("validation_enabled", True)
        self.sandbox_enabled = config.get("sandbox_enabled", True)
        
        # Stockage interne
        self._templates: Dict[str, SlackTemplate] = {}
        self._compiled_templates: Dict[str, jinja2.Template] = {}
        self._template_cache: Dict[str, str] = {}
        
        # Configuration Jinja2
        self._setup_jinja_environment()
        
        # Lock pour thread-safety
        self._lock = asyncio.Lock()
        
        # Initialisation
        asyncio.create_task(self._initialize())
    
    def _setup_jinja_environment(self):
        """Configure l'environnement Jinja2 avec filtres et fonctions."""
        # Utiliser l'environnement sécurisé si activé
        if self.sandbox_enabled:
            self.jinja_env = SandboxedEnvironment(
                loader=FileSystemLoader(str(self.templates_dir)),
                auto_reload=self.auto_reload,
                cache_size=1000,
                trim_blocks=True,
                lstrip_blocks=True
            )
        else:
            self.jinja_env = Environment(
                loader=FileSystemLoader(str(self.templates_dir)),
                auto_reload=self.auto_reload,
                cache_size=1000,
                trim_blocks=True,
                lstrip_blocks=True
            )
        
        # Ajouter les filtres personnalisés
        self._register_custom_filters()
        
        # Ajouter les fonctions globales
        self._register_global_functions()
    
    def _register_custom_filters(self):
        """Enregistre les filtres Jinja2 personnalisés."""
        
        @self.jinja_env.filter('localize')
        def localize_filter(text: str, locale: Optional[str] = None, **kwargs):
            """Filtre de localisation."""
            if self.localization_engine:
                # Exécuter de manière synchrone (limitation Jinja2)
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(
                    self.localization_engine.localize(text, locale, kwargs)
                )
            return text
        
        @self.jinja_env.filter('format_datetime')
        def format_datetime_filter(dt: datetime, format_str: str = None, tz: str = None):
            """Filtre de formatage de date/heure."""
            if not isinstance(dt, datetime):
                return str(dt)
            
            if tz:
                # Conversion de timezone si nécessaire
                try:
                    import pytz
                    target_tz = pytz.timezone(tz)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    dt = dt.astimezone(target_tz)
                except:
                    pass
            
            if format_str:
                return dt.strftime(format_str)
            return dt.isoformat()
        
        @self.jinja_env.filter('severity_color')
        def severity_color_filter(severity: str) -> str:
            """Retourne la couleur Slack selon la sévérité."""
            color_map = {
                'critical': 'danger',
                'warning': 'warning',
                'info': 'good',
                'unknown': '#808080'
            }
            return color_map.get(severity.lower(), '#808080')
        
        @self.jinja_env.filter('severity_emoji')
        def severity_emoji_filter(severity: str) -> str:
            """Retourne l'emoji selon la sévérité."""
            emoji_map = {
                'critical': '🚨',
                'warning': '⚠️',
                'info': 'ℹ️',
                'resolved': '✅',
                'unknown': '❓'
            }
            return emoji_map.get(severity.lower(), '❓')
        
        @self.jinja_env.filter('truncate_smart')
        def truncate_smart_filter(text: str, length: int = 100, suffix: str = '...') -> str:
            """Troncature intelligente qui évite de couper au milieu d'un mot."""
            if len(text) <= length:
                return text
            
            truncated = text[:length - len(suffix)]
            # Chercher le dernier espace
            last_space = truncated.rfind(' ')
            if last_space > length * 0.8:  # Si l'espace n'est pas trop loin
                truncated = truncated[:last_space]
            
            return truncated + suffix
        
        @self.jinja_env.filter('json_pretty')
        def json_pretty_filter(obj: Any, indent: int = 2) -> str:
            """Formatage JSON avec indentation."""
            try:
                return json.dumps(obj, indent=indent, ensure_ascii=False)
            except:
                return str(obj)
    
    def _register_global_functions(self):
        """Enregistre les fonctions globales Jinja2."""
        
        def now(tz: str = "UTC") -> datetime:
            """Retourne la date/heure actuelle."""
            if tz == "UTC":
                return datetime.now(timezone.utc)
            try:
                import pytz
                target_tz = pytz.timezone(tz)
                return datetime.now(target_tz)
            except:
                return datetime.now(timezone.utc)
        
        def url_encode(text: str) -> str:
            """Encode une URL."""
            from urllib.parse import quote
            return quote(str(text))
        
        def build_alert_url(alert: Dict[str, Any], base_url: str = "") -> str:
            """Construit l'URL d'une alerte."""
            fingerprint = alert.get('fingerprint', '')
            return f"{base_url}/alerts/{fingerprint}"
        
        def build_runbook_url(alert: Dict[str, Any], base_url: str = "") -> str:
            """Construit l'URL du runbook."""
            runbook = alert.get('annotations', {}).get('runbook_url')
            if runbook:
                return runbook
            
            service = alert.get('labels', {}).get('service', 'unknown')
            return f"{base_url}/runbooks/{service}"
        
        # Enregistrement des fonctions
        self.jinja_env.globals.update({
            'now': now,
            'url_encode': url_encode,
            'build_alert_url': build_alert_url,
            'build_runbook_url': build_runbook_url
        })
    
    async def _initialize(self):
        """Initialise le renderer avec chargement des templates."""
        try:
            await self._load_templates()
            await self._validate_templates()
            
            self.logger.info(
                "Renderer de templates initialisé",
                templates_count=len(self._templates)
            )
            
            ACTIVE_TEMPLATES.set(len(self._templates))
            
        except Exception as e:
            self.logger.error("Erreur lors de l'initialisation", error=str(e))
            raise
    
    async def _load_templates(self):
        """Charge tous les templates depuis le répertoire."""
        if not self.templates_dir.exists():
            self.templates_dir.mkdir(parents=True, exist_ok=True)
            await self._create_default_templates()
        
        # Parcourir les fichiers de templates
        for template_file in self.templates_dir.rglob("*.j2"):
            await self._load_template_file(template_file)
    
    async def _load_template_file(self, template_file: Path):
        """Charge un fichier de template spécifique."""
        try:
            relative_path = template_file.relative_to(self.templates_dir)
            template_name = str(relative_path).replace('.j2', '')
            
            with open(template_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Analyser le template pour extraire les métadonnées
            variables = self._extract_template_variables(content)
            macros = self._extract_template_macros(content)
            extends = self._extract_template_extends(content)
            
            template = SlackTemplate(
                name=template_name,
                path=str(template_file),
                content=content,
                variables=variables,
                macros=macros,
                extends=extends,
                last_modified=datetime.fromtimestamp(template_file.stat().st_mtime)
            )
            
            self._templates[template_name] = template
            
            # Compiler le template
            try:
                compiled = self.jinja_env.from_string(content)
                self._compiled_templates[template_name] = compiled
            except Exception as e:
                self.logger.error(
                    "Erreur compilation template",
                    template=template_name,
                    error=str(e)
                )
            
            self.logger.debug(
                "Template chargé",
                name=template_name,
                variables=len(variables),
                macros=len(macros)
            )
            
        except Exception as e:
            self.logger.error(
                "Erreur chargement template",
                file=str(template_file),
                error=str(e)
            )
    
    def _extract_template_variables(self, content: str) -> List[str]:
        """Extrait les variables utilisées dans un template."""
        import re
        
        # Pattern pour les variables Jinja2
        variable_pattern = re.compile(r'\{\{\s*([a-zA-Z_][a-zA-Z0-9_\.]*)')
        variables = set()
        
        for match in variable_pattern.finditer(content):
            var_name = match.group(1).split('.')[0]  # Prendre la racine
            variables.add(var_name)
        
        return list(variables)
    
    def _extract_template_macros(self, content: str) -> List[str]:
        """Extrait les macros définies dans un template."""
        import re
        
        macro_pattern = re.compile(r'\{%\s*macro\s+([a-zA-Z_][a-zA-Z0-9_]*)')
        macros = []
        
        for match in macro_pattern.finditer(content):
            macros.append(match.group(1))
        
        return macros
    
    def _extract_template_extends(self, content: str) -> Optional[str]:
        """Extrait le template parent si extends est utilisé."""
        import re
        
        extends_pattern = re.compile(r'\{%\s*extends\s+["\']([^"\']+)["\']')
        match = extends_pattern.search(content)
        
        return match.group(1) if match else None
    
    async def _create_default_templates(self):
        """Crée les templates par défaut si le répertoire est vide."""
        default_templates = {
            "alert_critical.json.j2": self._get_critical_alert_template(),
            "alert_warning.json.j2": self._get_warning_alert_template(),
            "alert_info.json.j2": self._get_info_alert_template(),
            "alert_resolved.json.j2": self._get_resolved_alert_template(),
            "macros/common.j2": self._get_common_macros_template()
        }
        
        for template_name, content in default_templates.items():
            template_path = self.templates_dir / template_name
            template_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write(content)
    
    def _get_critical_alert_template(self) -> str:
        """Template pour alertes critiques."""
        return """{
  "channel": "{{ config.channel }}",
  "username": "{{ config.bot_name | default('Spotify AI Agent') }}",
  "icon_emoji": ":rotating_light:",
  "attachments": [{
    "color": "danger",
    "pretext": "{{ 'alerts.critical_detected' | localize(locale) }}",
    "title": "{{ alert.labels.service | title }} - {{ alert.annotations.summary }}",
    "title_link": "{{ build_alert_url(alert, config.base_url) }}",
    "text": "{{ alert.annotations.description | truncate_smart(200) }}",
    "fields": [
      {
        "title": "{{ 'fields.severity' | localize(locale) }}",
        "value": "{{ alert.labels.severity | upper }} {{ alert.labels.severity | severity_emoji }}",
        "short": true
      },
      {
        "title": "{{ 'fields.tenant' | localize(locale) }}",
        "value": "{{ tenant.name | default(alert.labels.tenant) }}",
        "short": true
      },
      {
        "title": "{{ 'fields.environment' | localize(locale) }}",
        "value": "{{ alert.labels.environment | upper }}",
        "short": true
      },
      {
        "title": "{{ 'fields.timestamp' | localize(locale) }}",
        "value": "{{ alert.startsAt | format_datetime(tenant.date_format, tenant.timezone) }}",
        "short": true
      }
    ],
    "actions": [
      {
        "type": "button",
        "text": "{{ 'actions.view_runbook' | localize(locale) }}",
        "url": "{{ build_runbook_url(alert, config.base_url) }}",
        "style": "primary"
      },
      {
        "type": "button",
        "text": "{{ 'actions.acknowledge' | localize(locale) }}",
        "url": "{{ config.ack_url }}/{{ alert.fingerprint }}",
        "style": "default"
      }
    ],
    "footer": "Spotify AI Agent",
    "footer_icon": "https://spotify-ai-agent.com/icon.png",
    "ts": {{ (alert.startsAt | format_datetime | as_timestamp) | int }}
  }]
}"""
    
    def _get_warning_alert_template(self) -> str:
        """Template pour alertes d'avertissement."""
        return """{
  "channel": "{{ config.channel }}",
  "username": "{{ config.bot_name | default('Spotify AI Agent') }}",
  "icon_emoji": ":warning:",
  "attachments": [{
    "color": "warning",
    "pretext": "{{ 'alerts.warning_detected' | localize(locale) }}",
    "title": "{{ alert.labels.service | title }} - {{ alert.annotations.summary }}",
    "title_link": "{{ build_alert_url(alert, config.base_url) }}",
    "text": "{{ alert.annotations.description | truncate_smart(150) }}",
    "fields": [
      {
        "title": "{{ 'fields.severity' | localize(locale) }}",
        "value": "{{ alert.labels.severity | upper }} {{ alert.labels.severity | severity_emoji }}",
        "short": true
      },
      {
        "title": "{{ 'fields.tenant' | localize(locale) }}",
        "value": "{{ tenant.name | default(alert.labels.tenant) }}",
        "short": true
      }
    ],
    "footer": "Spotify AI Agent",
    "ts": {{ (alert.startsAt | format_datetime | as_timestamp) | int }}
  }]
}"""
    
    def _get_info_alert_template(self) -> str:
        """Template pour alertes d'information."""
        return """{
  "channel": "{{ config.channel }}",
  "username": "{{ config.bot_name | default('Spotify AI Agent') }}",
  "icon_emoji": ":information_source:",
  "attachments": [{
    "color": "good",
    "title": "{{ alert.annotations.summary }}",
    "text": "{{ alert.annotations.description }}",
    "footer": "Spotify AI Agent",
    "ts": {{ (alert.startsAt | format_datetime | as_timestamp) | int }}
  }]
}"""
    
    def _get_resolved_alert_template(self) -> str:
        """Template pour alertes résolues."""
        return """{
  "channel": "{{ config.channel }}",
  "username": "{{ config.bot_name | default('Spotify AI Agent') }}",
  "icon_emoji": ":white_check_mark:",
  "attachments": [{
    "color": "good",
    "pretext": "{{ 'alerts.resolved' | localize(locale) }}",
    "title": "{{ alert.labels.service | title }} - {{ alert.annotations.summary }}",
    "fields": [
      {
        "title": "{{ 'fields.duration' | localize(locale) }}",
        "value": "{{ ((alert.endsAt | format_datetime | as_timestamp) - (alert.startsAt | format_datetime | as_timestamp)) | duration_format }}",
        "short": true
      }
    ],
    "footer": "Spotify AI Agent",
    "ts": {{ (alert.endsAt | format_datetime | as_timestamp) | int }}
  }]
}"""
    
    def _get_common_macros_template(self) -> str:
        """Macros communes réutilisables."""
        return """{% macro render_field(title, value, short=true) %}
{
  "title": "{{ title | localize(locale) }}",
  "value": "{{ value }}",
  "short": {{ short | lower }}
}
{% endmacro %}

{% macro render_action_button(text, url, style="default") %}
{
  "type": "button", 
  "text": "{{ text | localize(locale) }}",
  "url": "{{ url }}",
  "style": "{{ style }}"
}
{% endmacro %}

{% macro severity_badge(severity) %}
{{ severity | upper }} {{ severity | severity_emoji }}
{% endmacro %}

{% macro tenant_info(tenant, alert) %}
{{ tenant.name | default(alert.labels.tenant) }}
{% if tenant.support_tier %}({{ tenant.support_tier | upper }}){% endif %}
{% endmacro %}"""
    
    async def _validate_templates(self):
        """Valide tous les templates chargés."""
        if not self.validation_enabled:
            return
        
        errors = []
        
        for template_name, template in self._templates.items():
            try:
                # Tenter de compiler le template
                self.jinja_env.from_string(template.content)
                
                # Vérifier les variables requises
                required_vars = {'alert', 'config', 'locale'}
                template_vars = set(template.variables)
                
                missing_vars = required_vars - template_vars
                if missing_vars:
                    errors.append(f"Template {template_name}: variables manquantes {missing_vars}")
                
            except Exception as e:
                errors.append(f"Template {template_name}: erreur de compilation - {str(e)}")
        
        if errors:
            self.logger.warning(
                "Erreurs de validation des templates",
                errors=errors
            )
    
    async def render_template(
        self,
        template_name: str,
        context: RenderContext,
        cache_enabled: bool = True
    ) -> str:
        """
        Rend un template avec le contexte fourni.
        
        Args:
            template_name: Nom du template à rendre
            context: Contexte de rendu
            cache_enabled: Activer le cache
            
        Returns:
            JSON rendu pour Slack
        """
        start_time = datetime.utcnow()
        
        try:
            # Vérifier le cache d'abord
            if cache_enabled:
                cache_key = self._build_cache_key(template_name, context)
                cached_result = self._template_cache.get(cache_key)
                
                if cached_result:
                    TEMPLATE_CACHE_HITS.labels(
                        cache_type="memory",
                        template=template_name
                    ).inc()
                    return cached_result
            
            # Récupérer le template compilé
            compiled_template = self._compiled_templates.get(template_name)
            if not compiled_template:
                raise TemplateNotFound(f"Template {template_name} non trouvé")
            
            # Préparer le contexte de rendu
            render_vars = {
                'alert': context.alert,
                'tenant': context.tenant,
                'config': context.config,
                'locale': context.locale,
                'timezone': context.timezone,
                'timestamp': context.timestamp,
                'user_context': context.user_context or {}
            }
            
            # Rendu du template
            rendered = await self._render_template_async(compiled_template, render_vars)
            
            # Validation du JSON résultant
            try:
                json.loads(rendered)  # Vérifier que c'est du JSON valide
            except json.JSONDecodeError as e:
                raise ValueError(f"Template rendu invalide (JSON): {str(e)}")
            
            # Mise en cache
            if cache_enabled:
                self._template_cache[cache_key] = rendered
                # Nettoyer le cache si trop volumineux
                if len(self._template_cache) > 1000:
                    # Supprimer les 200 plus anciens
                    old_keys = list(self._template_cache.keys())[:200]
                    for key in old_keys:
                        del self._template_cache[key]
            
            # Métriques
            duration = (datetime.utcnow() - start_time).total_seconds()
            TEMPLATE_RENDER_DURATION.labels(
                template=template_name,
                type="success"
            ).observe(duration)
            
            TEMPLATE_RENDER_REQUESTS.labels(
                template=template_name,
                tenant=context.tenant.get('id', 'unknown'),
                status="success"
            ).inc()
            
            return rendered
            
        except Exception as e:
            self.logger.error(
                "Erreur lors du rendu de template",
                template=template_name,
                error=str(e)
            )
            
            TEMPLATE_RENDER_REQUESTS.labels(
                template=template_name,
                tenant=context.tenant.get('id', 'unknown'),
                status="error"
            ).inc()
            
            raise
    
    async def _render_template_async(self, template: jinja2.Template, context: Dict[str, Any]) -> str:
        """Rend un template de manière asynchrone."""
        # Jinja2 est synchrone, on utilise un executor pour ne pas bloquer
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, template.render, context)
    
    def _build_cache_key(self, template_name: str, context: RenderContext) -> str:
        """Construit une clé de cache unique."""
        # Utiliser des éléments significatifs pour la clé
        key_parts = [
            template_name,
            context.locale,
            context.tenant.get('id', ''),
            str(hash(str(context.alert.get('fingerprint', ''))))
        ]
        
        return "|".join(key_parts)
    
    async def list_templates(self) -> List[Dict[str, Any]]:
        """Liste tous les templates disponibles avec métadonnées."""
        templates_info = []
        
        for name, template in self._templates.items():
            templates_info.append({
                "name": name,
                "path": template.path,
                "variables": template.variables,
                "macros": template.macros,
                "extends": template.extends,
                "version": template.version,
                "description": template.description,
                "last_modified": template.last_modified.isoformat() if template.last_modified else None,
                "checksum": template.checksum
            })
        
        return templates_info
    
    async def validate_template_syntax(self, template_content: str) -> Dict[str, Any]:
        """Valide la syntaxe d'un template."""
        try:
            # Tenter de compiler
            compiled = self.jinja_env.from_string(template_content)
            
            # Extraire les variables
            variables = self._extract_template_variables(template_content)
            macros = self._extract_template_macros(template_content)
            
            return {
                "valid": True,
                "variables": variables,
                "macros": macros,
                "message": "Template valide"
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "message": f"Erreur de syntaxe: {str(e)}"
            }
    
    async def reload_templates(self):
        """Recharge tous les templates depuis le disque."""
        async with self._lock:
            try:
                # Vider les caches
                self._templates.clear()
                self._compiled_templates.clear()
                self._template_cache.clear()
                
                # Recharger
                await self._load_templates()
                await self._validate_templates()
                
                ACTIVE_TEMPLATES.set(len(self._templates))
                
                self.logger.info(
                    "Templates rechargés",
                    count=len(self._templates)
                )
                
            except Exception as e:
                self.logger.error("Erreur lors du rechargement", error=str(e))
                raise
    
    async def get_template_preview(
        self,
        template_name: str,
        sample_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Génère un aperçu de template avec des données d'exemple."""
        # Données d'exemple par défaut
        if sample_data is None:
            sample_data = {
                "alert": {
                    "status": "firing",
                    "labels": {
                        "severity": "critical",
                        "service": "ml-engine",
                        "tenant": "spotify_premium",
                        "environment": "production"
                    },
                    "annotations": {
                        "summary": "Haute utilisation CPU détectée",
                        "description": "Le service ML Engine affiche une utilisation CPU de 95% depuis 5 minutes"
                    },
                    "startsAt": "2025-01-18T10:30:00Z",
                    "fingerprint": "abc123def456"
                },
                "tenant": {
                    "id": "spotify_premium",
                    "name": "Spotify Premium",
                    "timezone": "Europe/Paris",
                    "support_tier": "premium"
                },
                "config": {
                    "channel": "#alerts-premium",
                    "bot_name": "Spotify AI Agent",
                    "base_url": "https://monitoring.spotify-ai.com",
                    "ack_url": "https://monitoring.spotify-ai.com/ack"
                },
                "locale": "fr_FR"
            }
        
        context = RenderContext(
            alert=sample_data["alert"],
            tenant=sample_data["tenant"],
            config=sample_data["config"],
            locale=sample_data["locale"]
        )
        
        return await self.render_template(template_name, context, cache_enabled=False)
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérifie la santé du renderer."""
        try:
            # Test de rendu simple
            test_template = "{{ 'test' }}"
            test_compiled = self.jinja_env.from_string(test_template)
            test_result = test_compiled.render()
            
            return {
                "status": "healthy",
                "templates_count": len(self._templates),
                "compiled_count": len(self._compiled_templates),
                "cache_size": len(self._template_cache),
                "jinja_version": jinja2.__version__,
                "sandbox_enabled": self.sandbox_enabled,
                "test_render": test_result,
                "last_check": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.utcnow().isoformat()
            }

# Factory function
def create_template_renderer(config: Dict[str, Any], localization_engine=None) -> SlackTemplateRenderer:
    """Crée une instance du renderer de templates."""
    return SlackTemplateRenderer(config, localization_engine)
