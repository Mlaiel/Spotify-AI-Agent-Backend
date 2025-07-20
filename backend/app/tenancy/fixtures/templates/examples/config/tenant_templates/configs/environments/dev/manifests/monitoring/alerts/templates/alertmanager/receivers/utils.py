"""
Utilitaires pour la gestion des receivers d'alertes.

Ce module fournit des classes utilitaires pour la validation,
le rendu de templates, le throttling, les métriques et la sécurité.
"""

import asyncio
import logging
import time
import hashlib
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from collections import defaultdict, deque
import yaml
import jinja2
from cryptography.fernet import Fernet
import aioredis
from prometheus_client import Counter, Histogram, Gauge
import hvac

from .models import (
    ReceiverConfig,
    AlertContext,
    NotificationResult,
    ChannelType,
    AlertSeverity
)
from .exceptions import (
    ReceiverConfigError,
    TemplateRenderError,
    ThrottleError,
    SecurityError
)

logger = logging.getLogger(__name__)

class ReceiverValidator:
    """Validateur pour les configurations de receivers"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Expressions régulières pour validation
        self.url_pattern = re.compile(
            r'^https?://'  # http:// ou https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domaine...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...ou IP
            r'(?::\d+)?'  # port optionnel
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        self.email_pattern = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )
    
    async def validate_config(self, config: Dict[str, Any], tenant_id: str) -> bool:
        """Valide une configuration complète"""
        try:
            self.logger.info(f"Validating config for tenant {tenant_id}")
            
            # Vérifier la structure de base
            if 'tenants' not in config:
                raise ReceiverConfigError("Missing 'tenants' section in config")
            
            tenant_config = config['tenants'].get(tenant_id)
            if not tenant_config:
                raise ReceiverConfigError(f"No configuration found for tenant {tenant_id}")
            
            # Valider les receivers
            receivers = tenant_config.get('receivers', [])
            if not receivers:
                self.logger.warning(f"No receivers configured for tenant {tenant_id}")
                return True
            
            for i, receiver_config in enumerate(receivers):
                await self._validate_receiver(receiver_config, i)
            
            # Valider les politiques d'escalade
            escalation_policies = tenant_config.get('escalation_policies', [])
            for i, policy in enumerate(escalation_policies):
                await self._validate_escalation_policy(policy, i, receivers)
            
            self.logger.info(f"Config validation successful for tenant {tenant_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Config validation failed: {e}")
            raise ReceiverConfigError(f"Validation failed: {e}")
    
    async def _validate_receiver(self, receiver_config: Dict[str, Any], index: int):
        """Valide la configuration d'un receiver"""
        
        # Champs obligatoires
        required_fields = ['name', 'channel_type', 'config']
        for field in required_fields:
            if field not in receiver_config:
                raise ReceiverConfigError(f"Receiver {index}: Missing required field '{field}'")
        
        name = receiver_config['name']
        channel_type = receiver_config['channel_type']
        config = receiver_config['config']
        
        # Valider le nom
        if not isinstance(name, str) or not name.strip():
            raise ReceiverConfigError(f"Receiver {index}: Invalid name")
        
        # Valider le type de canal
        try:
            channel_enum = ChannelType(channel_type)
        except ValueError:
            raise ReceiverConfigError(f"Receiver {name}: Unsupported channel type '{channel_type}'")
        
        # Validation spécifique par type
        await self._validate_channel_config(name, channel_enum, config)
        
        # Valider les sélecteurs optionnels
        self._validate_selectors(name, receiver_config.get('label_selectors', {}))
        self._validate_selectors(name, receiver_config.get('annotation_selectors', {}))
        
        # Valider les paramètres de retry
        self._validate_retry_config(name, receiver_config)
    
    async def _validate_channel_config(self, name: str, channel_type: ChannelType, config: Dict[str, Any]):
        """Valide la configuration spécifique d'un canal"""
        
        if channel_type == ChannelType.SLACK:
            await self._validate_slack_config(name, config)
        elif channel_type == ChannelType.EMAIL:
            await self._validate_email_config(name, config)
        elif channel_type == ChannelType.PAGERDUTY:
            await self._validate_pagerduty_config(name, config)
        elif channel_type == ChannelType.WEBHOOK:
            await self._validate_webhook_config(name, config)
        elif channel_type == ChannelType.TEAMS:
            await self._validate_teams_config(name, config)
        elif channel_type == ChannelType.DISCORD:
            await self._validate_discord_config(name, config)
    
    async def _validate_slack_config(self, name: str, config: Dict[str, Any]):
        """Valide la configuration Slack"""
        if 'webhook_url' not in config:
            raise ReceiverConfigError(f"Receiver {name}: Missing 'webhook_url' for Slack")
        
        webhook_url = config['webhook_url']
        if not self._is_valid_url(webhook_url):
            raise ReceiverConfigError(f"Receiver {name}: Invalid Slack webhook URL")
        
        # Vérifier que c'est bien une URL Slack
        if 'hooks.slack.com' not in webhook_url:
            self.logger.warning(f"Receiver {name}: webhook_url doesn't look like a Slack URL")
    
    async def _validate_email_config(self, name: str, config: Dict[str, Any]):
        """Valide la configuration Email"""
        required_fields = ['smtp_server', 'recipients', 'from_address']
        for field in required_fields:
            if field not in config:
                raise ReceiverConfigError(f"Receiver {name}: Missing '{field}' for Email")
        
        # Valider les adresses email
        recipients = config['recipients']
        if not isinstance(recipients, list) or not recipients:
            raise ReceiverConfigError(f"Receiver {name}: 'recipients' must be a non-empty list")
        
        for email in recipients:
            if not self._is_valid_email(email):
                raise ReceiverConfigError(f"Receiver {name}: Invalid email address '{email}'")
        
        # Valider l'adresse d'expéditeur
        from_address = config['from_address']
        if not self._is_valid_email(from_address):
            raise ReceiverConfigError(f"Receiver {name}: Invalid from_address '{from_address}'")
        
        # Valider le port SMTP
        smtp_port = config.get('smtp_port', 587)
        if not isinstance(smtp_port, int) or not (1 <= smtp_port <= 65535):
            raise ReceiverConfigError(f"Receiver {name}: Invalid SMTP port '{smtp_port}'")
    
    async def _validate_pagerduty_config(self, name: str, config: Dict[str, Any]):
        """Valide la configuration PagerDuty"""
        if 'integration_key' not in config:
            raise ReceiverConfigError(f"Receiver {name}: Missing 'integration_key' for PagerDuty")
        
        integration_key = config['integration_key']
        if not isinstance(integration_key, str) or len(integration_key) != 32:
            raise ReceiverConfigError(f"Receiver {name}: Invalid PagerDuty integration key format")
    
    async def _validate_webhook_config(self, name: str, config: Dict[str, Any]):
        """Valide la configuration Webhook"""
        if 'url' not in config:
            raise ReceiverConfigError(f"Receiver {name}: Missing 'url' for Webhook")
        
        url = config['url']
        if not self._is_valid_url(url):
            raise ReceiverConfigError(f"Receiver {name}: Invalid webhook URL")
        
        # Valider la méthode HTTP
        method = config.get('method', 'POST')
        if method not in ['POST', 'PUT', 'PATCH']:
            raise ReceiverConfigError(f"Receiver {name}: Invalid HTTP method '{method}'")
    
    async def _validate_teams_config(self, name: str, config: Dict[str, Any]):
        """Valide la configuration Teams"""
        if 'webhook_url' not in config:
            raise ReceiverConfigError(f"Receiver {name}: Missing 'webhook_url' for Teams")
        
        webhook_url = config['webhook_url']
        if not self._is_valid_url(webhook_url):
            raise ReceiverConfigError(f"Receiver {name}: Invalid Teams webhook URL")
        
        if 'office.com' not in webhook_url and 'outlook.office365.com' not in webhook_url:
            self.logger.warning(f"Receiver {name}: webhook_url doesn't look like a Teams URL")
    
    async def _validate_discord_config(self, name: str, config: Dict[str, Any]):
        """Valide la configuration Discord"""
        if 'webhook_url' not in config:
            raise ReceiverConfigError(f"Receiver {name}: Missing 'webhook_url' for Discord")
        
        webhook_url = config['webhook_url']
        if not self._is_valid_url(webhook_url):
            raise ReceiverConfigError(f"Receiver {name}: Invalid Discord webhook URL")
        
        if 'discord.com' not in webhook_url and 'discordapp.com' not in webhook_url:
            self.logger.warning(f"Receiver {name}: webhook_url doesn't look like a Discord URL")
    
    def _validate_selectors(self, name: str, selectors: Dict[str, str]):
        """Valide les sélecteurs de labels/annotations"""
        if not isinstance(selectors, dict):
            raise ReceiverConfigError(f"Receiver {name}: Selectors must be a dictionary")
        
        for key, value in selectors.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ReceiverConfigError(f"Receiver {name}: Selector keys and values must be strings")
    
    def _validate_retry_config(self, name: str, receiver_config: Dict[str, Any]):
        """Valide la configuration de retry"""
        max_retry = receiver_config.get('max_retry_attempts', 3)
        if not isinstance(max_retry, int) or not (0 <= max_retry <= 10):
            raise ReceiverConfigError(f"Receiver {name}: max_retry_attempts must be 0-10")
        
        timeout = receiver_config.get('timeout_seconds', 30)
        if not isinstance(timeout, int) or not (5 <= timeout <= 300):
            raise ReceiverConfigError(f"Receiver {name}: timeout_seconds must be 5-300")
    
    async def _validate_escalation_policy(self, policy: Dict[str, Any], index: int, receivers: List[Dict[str, Any]]):
        """Valide une politique d'escalade"""
        required_fields = ['name', 'escalation_receivers']
        for field in required_fields:
            if field not in policy:
                raise ReceiverConfigError(f"Escalation policy {index}: Missing '{field}'")
        
        # Vérifier que les receivers d'escalade existent
        escalation_receivers = policy['escalation_receivers']
        if not isinstance(escalation_receivers, list) or not escalation_receivers:
            raise ReceiverConfigError(f"Escalation policy {index}: escalation_receivers must be a non-empty list")
        
        receiver_names = {r['name'] for r in receivers}
        for receiver_name in escalation_receivers:
            if receiver_name not in receiver_names:
                raise ReceiverConfigError(f"Escalation policy {index}: Unknown receiver '{receiver_name}'")
    
    def _is_valid_url(self, url: str) -> bool:
        """Vérifie si une URL est valide"""
        return bool(self.url_pattern.match(url))
    
    def _is_valid_email(self, email: str) -> bool:
        """Vérifie si une adresse email est valide"""
        return bool(self.email_pattern.match(email))

class TemplateRenderer:
    """Gestionnaire de rendu de templates Jinja2"""
    
    def __init__(self, template_dir: Optional[str] = None):
        self.template_dir = template_dir or "templates"
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialiser l'environnement Jinja2
        try:
            self.env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(self.template_dir),
                autoescape=True,
                trim_blocks=True,
                lstrip_blocks=True,
                undefined=jinja2.StrictUndefined
            )
        except Exception:
            # Fallback avec template par défaut
            self.env = jinja2.Environment(
                loader=jinja2.DictLoader({}),
                autoescape=True,
                trim_blocks=True,
                lstrip_blocks=True,
                undefined=jinja2.StrictUndefined
            )
        
        # Ajouter des fonctions personnalisées
        self._register_template_functions()
        
        # Cache des templates rendus
        self.template_cache: Dict[str, jinja2.Template] = {}
    
    def _register_template_functions(self):
        """Enregistre les fonctions personnalisées pour les templates"""
        self.env.globals.update({
            'format_timestamp': self._format_timestamp,
            'severity_color': self._get_severity_color,
            'severity_emoji': self._get_severity_emoji,
            'truncate_text': self._truncate_text,
            'join_labels': self._join_labels,
            'format_duration': self._format_duration,
            'highlight_text': self._highlight_text,
            'escape_markdown': self._escape_markdown,
            'generate_alert_id': self._generate_alert_id
        })
    
    def _format_timestamp(self, timestamp: datetime, format_str: str = "%Y-%m-%d %H:%M:%S UTC") -> str:
        """Formate un timestamp"""
        return timestamp.strftime(format_str)
    
    def _get_severity_color(self, severity: str) -> str:
        """Retourne une couleur selon la sévérité"""
        colors = {
            'critical': '#FF0000',
            'high': '#FF8C00',
            'medium': '#FFD700',
            'low': '#32CD32',
            'info': '#4169E1'
        }
        return colors.get(severity.lower(), '#808080')
    
    def _get_severity_emoji(self, severity: str) -> str:
        """Retourne un emoji selon la sévérité"""
        emojis = {
            'critical': '🚨',
            'high': '⚠️',
            'medium': '⚡',
            'low': 'ℹ️',
            'info': '📢'
        }
        return emojis.get(severity.lower(), '❓')
    
    def _truncate_text(self, text: str, max_length: int = 100, suffix: str = "...") -> str:
        """Tronque un texte"""
        if len(text) <= max_length:
            return text
        return text[:max_length-len(suffix)] + suffix
    
    def _join_labels(self, labels: Dict[str, str], separator: str = ", ", format_str: str = "{key}={value}") -> str:
        """Joint les labels en une chaîne"""
        return separator.join([
            format_str.format(key=k, value=v) 
            for k, v in labels.items()
        ])
    
    def _format_duration(self, start_time: datetime, end_time: Optional[datetime] = None) -> str:
        """Formate une durée"""
        if end_time is None:
            end_time = datetime.utcnow()
        
        duration = end_time - start_time
        
        if duration.days > 0:
            return f"{duration.days}d {duration.seconds//3600}h"
        elif duration.seconds >= 3600:
            hours = duration.seconds // 3600
            minutes = (duration.seconds % 3600) // 60
            return f"{hours}h {minutes}m"
        elif duration.seconds >= 60:
            minutes = duration.seconds // 60
            seconds = duration.seconds % 60
            return f"{minutes}m {seconds}s"
        else:
            return f"{duration.seconds}s"
    
    def _highlight_text(self, text: str, keywords: List[str], format_str: str = "**{text}**") -> str:
        """Met en évidence des mots-clés dans un texte"""
        result = text
        for keyword in keywords:
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            result = pattern.sub(format_str.format(text=keyword), result)
        return result
    
    def _escape_markdown(self, text: str) -> str:
        """Échappe les caractères Markdown"""
        markdown_chars = ['*', '_', '`', '[', ']', '(', ')', '#', '+', '-', '.', '!']
        for char in markdown_chars:
            text = text.replace(char, f"\\{char}")
        return text
    
    def _generate_alert_id(self, prefix: str = "alert") -> str:
        """Génère un ID unique pour une alerte"""
        import uuid
        return f"{prefix}_{uuid.uuid4().hex[:8]}"
    
    async def render_template(
        self,
        channel_type: str,
        template_name: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Rend un template pour un type de canal donné"""
        try:
            # Construire le nom du fichier template
            template_file = f"{channel_type}/{template_name}.j2"
            
            # Vérifier le cache
            cache_key = f"{channel_type}:{template_name}"
            if cache_key in self.template_cache:
                template = self.template_cache[cache_key]
            else:
                try:
                    template = self.env.get_template(template_file)
                    self.template_cache[cache_key] = template
                except jinja2.TemplateNotFound:
                    # Utiliser le template par défaut
                    template = await self._get_default_template(channel_type)
                    self.template_cache[cache_key] = template
            
            # Rendre le template
            rendered = template.render(**context)
            
            # Parser le JSON si nécessaire
            if channel_type in ['slack', 'webhook', 'teams', 'discord']:
                try:
                    return json.loads(rendered)
                except json.JSONDecodeError:
                    # Si ce n'est pas du JSON, retourner tel quel
                    return {'message': rendered}
            else:
                return {'message': rendered}
                
        except Exception as e:
            self.logger.error(f"Template rendering failed: {e}")
            raise TemplateRenderError(f"Failed to render template {template_file}: {e}")
    
    async def _get_default_template(self, channel_type: str) -> jinja2.Template:
        """Retourne un template par défaut pour le type de canal"""
        default_templates = {
            'slack': '''
            {
                "text": "{{ severity_emoji(alert.severity) }} {{ alert.name }}",
                "attachments": [
                    {
                        "color": "{{ severity_color(alert.severity) }}",
                        "title": "{{ alert.name }}",
                        "text": "{{ alert.annotations.summary or 'No summary available' }}",
                        "fields": [
                            {
                                "title": "Severity",
                                "value": "{{ alert.severity.upper() }}",
                                "short": true
                            },
                            {
                                "title": "Status",
                                "value": "{{ alert.status.upper() }}",
                                "short": true
                            },
                            {
                                "title": "Tenant",
                                "value": "{{ alert.tenant_id }}",
                                "short": true
                            }
                        ],
                        "footer": "AlertManager",
                        "ts": {{ alert.starts_at.timestamp() | int }}
                    }
                ]
            }
            ''',
            
            'email': '''
            Subject: [{{ alert.severity.upper() }}] {{ alert.name }}
            
            Alert: {{ alert.name }}
            Severity: {{ alert.severity.upper() }}
            Status: {{ alert.status.upper() }}
            Tenant: {{ alert.tenant_id }}
            Time: {{ format_timestamp(alert.starts_at) }}
            
            Summary: {{ alert.annotations.summary or 'No summary available' }}
            
            {% if alert.annotations.description %}
            Description: {{ alert.annotations.description }}
            {% endif %}
            
            {% if alert.labels %}
            Labels:
            {% for key, value in alert.labels.items() %}
            - {{ key }}: {{ value }}
            {% endfor %}
            {% endif %}
            ''',
            
            'webhook': '''
            {
                "alert_name": "{{ alert.name }}",
                "severity": "{{ alert.severity }}",
                "status": "{{ alert.status }}",
                "tenant_id": "{{ alert.tenant_id }}",
                "timestamp": "{{ alert.starts_at.isoformat() }}",
                "summary": "{{ alert.annotations.summary or '' }}",
                "labels": {{ alert.labels | tojson }},
                "annotations": {{ alert.annotations | tojson }}
            }
            '''
        }
        
        template_content = default_templates.get(channel_type, default_templates['webhook'])
        return jinja2.Template(template_content, environment=self.env)
    
    def clear_cache(self):
        """Vide le cache des templates"""
        self.template_cache.clear()
        self.logger.info("Template cache cleared")

class NotificationThrottler:
    """Gestionnaire de throttling des notifications"""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url
        self.redis_client: Optional[aioredis.Redis] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Cache local pour le throttling
        self.local_cache: Dict[str, deque] = defaultdict(deque)
        self.cache_ttl = 3600  # 1 heure
    
    async def initialize(self):
        """Initialise la connexion Redis si configurée"""
        if self.redis_url:
            try:
                self.redis_client = await aioredis.from_url(self.redis_url)
                await self.redis_client.ping()
                self.logger.info("Redis connection established for throttling")
            except Exception as e:
                self.logger.warning(f"Redis connection failed, using local cache: {e}")
                self.redis_client = None
    
    async def is_throttled(self, alert_context: AlertContext) -> bool:
        """Vérifie si l'alerte doit être throttlée"""
        try:
            throttle_key = self._generate_throttle_key(alert_context)
            current_time = time.time()
            
            if self.redis_client:
                return await self._check_redis_throttle(throttle_key, current_time)
            else:
                return await self._check_local_throttle(throttle_key, current_time)
                
        except Exception as e:
            self.logger.error(f"Throttle check failed: {e}")
            # En cas d'erreur, ne pas throttler
            return False
    
    def _generate_throttle_key(self, alert_context: AlertContext) -> str:
        """Génère une clé de throttling basée sur l'alerte"""
        # Utiliser le fingerprint de l'alerte comme clé de base
        base_key = f"throttle:{alert_context.tenant_id}:{alert_context.fingerprint}"
        return base_key
    
    async def _check_redis_throttle(self, throttle_key: str, current_time: float) -> bool:
        """Vérifie le throttling avec Redis"""
        try:
            # Utiliser une fenêtre glissante de 5 minutes
            window_size = 300  # 5 minutes
            max_notifications = 3  # Max 3 notifications par fenêtre
            
            # Ajouter la notification actuelle
            pipe = self.redis_client.pipeline()
            pipe.zadd(throttle_key, {str(current_time): current_time})
            
            # Supprimer les anciennes entrées
            pipe.zremrangebyscore(throttle_key, 0, current_time - window_size)
            
            # Compter les notifications dans la fenêtre
            pipe.zcard(throttle_key)
            
            # Définir l'expiration
            pipe.expire(throttle_key, window_size)
            
            results = await pipe.execute()
            count = results[2]
            
            return count > max_notifications
            
        except Exception as e:
            self.logger.error(f"Redis throttle check failed: {e}")
            return False
    
    async def _check_local_throttle(self, throttle_key: str, current_time: float) -> bool:
        """Vérifie le throttling avec le cache local"""
        window_size = 300  # 5 minutes
        max_notifications = 3
        
        # Nettoyer les anciennes entrées
        timestamps = self.local_cache[throttle_key]
        while timestamps and timestamps[0] < current_time - window_size:
            timestamps.popleft()
        
        # Ajouter la notification actuelle
        timestamps.append(current_time)
        
        # Limiter la taille du cache
        if len(timestamps) > max_notifications * 2:
            timestamps = deque(list(timestamps)[-max_notifications:])
            self.local_cache[throttle_key] = timestamps
        
        return len(timestamps) > max_notifications
    
    async def get_throttle_status(self, alert_context: AlertContext) -> Dict[str, Any]:
        """Retourne le statut de throttling pour une alerte"""
        throttle_key = self._generate_throttle_key(alert_context)
        current_time = time.time()
        
        if self.redis_client:
            try:
                window_size = 300
                count = await self.redis_client.zcount(
                    throttle_key, 
                    current_time - window_size, 
                    current_time
                )
                
                next_reset = current_time + window_size
                return {
                    'is_throttled': count > 3,
                    'count': count,
                    'window_size': window_size,
                    'next_reset': datetime.fromtimestamp(next_reset)
                }
            except Exception as e:
                self.logger.error(f"Failed to get throttle status: {e}")
        
        # Fallback local
        timestamps = self.local_cache.get(throttle_key, deque())
        window_size = 300
        
        # Compter les notifications récentes
        recent_count = sum(1 for ts in timestamps if ts >= current_time - window_size)
        
        return {
            'is_throttled': recent_count > 3,
            'count': recent_count,
            'window_size': window_size,
            'next_reset': datetime.fromtimestamp(current_time + window_size)
        }

class MetricsCollector:
    """Collecteur de métriques Prometheus"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._init_metrics()
    
    def _init_metrics(self):
        """Initialise les métriques Prometheus"""
        self.notifications_total = Counter(
            'alertmanager_notifications_total',
            'Total number of notifications sent',
            ['tenant_id', 'receiver_name', 'channel_type', 'status']
        )
        
        self.notification_duration = Histogram(
            'alertmanager_notification_duration_seconds',
            'Time spent sending notifications',
            ['tenant_id', 'receiver_name', 'channel_type'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        )
        
        self.escalation_events = Counter(
            'alertmanager_escalation_events_total',
            'Total number of escalation events',
            ['tenant_id', 'escalation_policy', 'escalation_level']
        )
        
        self.throttle_events = Counter(
            'alertmanager_throttle_events_total',
            'Total number of throttled notifications',
            ['tenant_id', 'alert_name']
        )
        
        self.template_render_duration = Histogram(
            'alertmanager_template_render_duration_seconds',
            'Time spent rendering templates',
            ['channel_type', 'template_name'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        )
        
        self.receiver_health_score = Gauge(
            'alertmanager_receiver_health_score',
            'Health score of receivers (0-1)',
            ['tenant_id', 'receiver_name']
        )
    
    def record_notification(
        self,
        tenant_id: str,
        receiver_name: str,
        channel_type: str,
        success: bool,
        duration: float
    ):
        """Enregistre une métrique de notification"""
        status = 'success' if success else 'failure'
        
        self.notifications_total.labels(
            tenant_id=tenant_id,
            receiver_name=receiver_name,
            channel_type=channel_type,
            status=status
        ).inc()
        
        self.notification_duration.labels(
            tenant_id=tenant_id,
            receiver_name=receiver_name,
            channel_type=channel_type
        ).observe(duration)
    
    def record_escalation(self, tenant_id: str, policy_name: str, level: int):
        """Enregistre une métrique d'escalade"""
        self.escalation_events.labels(
            tenant_id=tenant_id,
            escalation_policy=policy_name,
            escalation_level=str(level)
        ).inc()
    
    def record_throttle(self, tenant_id: str, alert_name: str):
        """Enregistre une métrique de throttling"""
        self.throttle_events.labels(
            tenant_id=tenant_id,
            alert_name=alert_name
        ).inc()
    
    def record_template_render(self, channel_type: str, template_name: str, duration: float):
        """Enregistre une métrique de rendu de template"""
        self.template_render_duration.labels(
            channel_type=channel_type,
            template_name=template_name
        ).observe(duration)
    
    def update_receiver_health(self, tenant_id: str, receiver_name: str, health_score: float):
        """Met à jour le score de santé d'un receiver"""
        self.receiver_health_score.labels(
            tenant_id=tenant_id,
            receiver_name=receiver_name
        ).set(health_score)

class SecretManager:
    """Gestionnaire de secrets avec support Vault"""
    
    def __init__(self, vault_url: Optional[str] = None, vault_token: Optional[str] = None):
        self.vault_url = vault_url
        self.vault_token = vault_token
        self.vault_client: Optional[hvac.Client] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Chiffrement local pour les secrets
        self.cipher_key = Fernet.generate_key()
        self.cipher = Fernet(self.cipher_key)
        
        # Cache des secrets
        self.secret_cache: Dict[str, Tuple[str, float]] = {}
        self.cache_ttl = 300  # 5 minutes
    
    async def initialize(self):
        """Initialise la connexion Vault si configurée"""
        if self.vault_url and self.vault_token:
            try:
                self.vault_client = hvac.Client(url=self.vault_url, token=self.vault_token)
                if self.vault_client.is_authenticated():
                    self.logger.info("Vault connection established")
                else:
                    self.logger.warning("Vault authentication failed")
                    self.vault_client = None
            except Exception as e:
                self.logger.warning(f"Vault connection failed: {e}")
                self.vault_client = None
    
    async def resolve_secrets(self, config: Dict[str, Any], tenant_id: str) -> Dict[str, Any]:
        """Résout les références de secrets dans la configuration"""
        resolved_config = {}
        
        for key, value in config.items():
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                # C'est une référence de secret
                secret_ref = value[2:-1]  # Enlever ${ et }
                resolved_value = await self._get_secret(secret_ref, tenant_id)
                resolved_config[key] = resolved_value
            elif isinstance(value, dict):
                # Récursion pour les objets imbriqués
                resolved_config[key] = await self.resolve_secrets(value, tenant_id)
            else:
                resolved_config[key] = value
        
        return resolved_config
    
    async def _get_secret(self, secret_ref: str, tenant_id: str) -> str:
        """Récupère un secret depuis Vault ou les variables d'environnement"""
        # Vérifier le cache
        cache_key = f"{tenant_id}:{secret_ref}"
        if cache_key in self.secret_cache:
            encrypted_value, timestamp = self.secret_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return self.cipher.decrypt(encrypted_value.encode()).decode()
        
        secret_value = None
        
        # Essayer Vault d'abord
        if self.vault_client:
            try:
                secret_value = await self._get_vault_secret(secret_ref, tenant_id)
            except Exception as e:
                self.logger.warning(f"Failed to get secret from Vault: {e}")
        
        # Fallback vers les variables d'environnement
        if secret_value is None:
            import os
            secret_value = os.getenv(secret_ref)
        
        if secret_value is None:
            raise SecurityError(f"Secret not found: {secret_ref}")
        
        # Mettre en cache (chiffré)
        encrypted_value = self.cipher.encrypt(secret_value.encode()).decode()
        self.secret_cache[cache_key] = (encrypted_value, time.time())
        
        return secret_value
    
    async def _get_vault_secret(self, secret_ref: str, tenant_id: str) -> Optional[str]:
        """Récupère un secret depuis Vault"""
        if not self.vault_client:
            return None
        
        try:
            # Construire le chemin du secret
            secret_path = f"secret/tenants/{tenant_id}/{secret_ref}"
            
            # Récupérer le secret
            response = self.vault_client.secrets.kv.v2.read_secret_version(path=secret_path)
            
            if response and 'data' in response and 'data' in response['data']:
                return response['data']['data'].get('value')
            
            return None
            
        except Exception as e:
            self.logger.error(f"Vault secret retrieval failed: {e}")
            return None
    
    def clear_cache(self):
        """Vide le cache des secrets"""
        self.secret_cache.clear()
        self.logger.info("Secret cache cleared")

class AuditLogger:
    """Logger d'audit pour les notifications"""
    
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.logger = logging.getLogger(f"{__name__}.audit.{tenant_id}")
        
        # Configuration spécifique pour l'audit
        self.audit_logger = logging.getLogger(f"audit.notifications.{tenant_id}")
        self.audit_logger.setLevel(logging.INFO)
        
        # Handler pour fichier d'audit
        audit_file = Path(f"logs/audit/notifications_{tenant_id}.log")
        audit_file.parent.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(audit_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        if not self.audit_logger.handlers:
            self.audit_logger.addHandler(handler)
    
    async def log_notification(
        self,
        alert_context: AlertContext,
        results: Dict[str, NotificationResult]
    ):
        """Log une notification dans l'audit"""
        audit_entry = {
            'tenant_id': self.tenant_id,
            'alert_id': alert_context.alert_id,
            'alert_name': alert_context.alert_name,
            'severity': alert_context.severity.value,
            'status': alert_context.status,
            'fingerprint': alert_context.fingerprint,
            'timestamp': datetime.utcnow().isoformat(),
            'results': {}
        }
        
        # Ajouter les résultats de notification
        for receiver_name, result in results.items():
            audit_entry['results'][receiver_name] = {
                'success': result.success,
                'duration': result.duration,
                'error_message': result.error_message,
                'response_status': result.response_status
            }
        
        # Log l'entrée d'audit
        self.audit_logger.info(json.dumps(audit_entry, ensure_ascii=False))
    
    async def log_escalation(
        self,
        alert_context: AlertContext,
        policy_name: str,
        escalation_level: int
    ):
        """Log une escalade dans l'audit"""
        audit_entry = {
            'tenant_id': self.tenant_id,
            'alert_id': alert_context.alert_id,
            'alert_name': alert_context.alert_name,
            'action': 'escalation',
            'policy_name': policy_name,
            'escalation_level': escalation_level,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.audit_logger.info(json.dumps(audit_entry, ensure_ascii=False))
    
    async def log_throttle(self, alert_context: AlertContext, throttle_reason: str):
        """Log un throttling dans l'audit"""
        audit_entry = {
            'tenant_id': self.tenant_id,
            'alert_id': alert_context.alert_id,
            'alert_name': alert_context.alert_name,
            'action': 'throttled',
            'reason': throttle_reason,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.audit_logger.info(json.dumps(audit_entry, ensure_ascii=False))
