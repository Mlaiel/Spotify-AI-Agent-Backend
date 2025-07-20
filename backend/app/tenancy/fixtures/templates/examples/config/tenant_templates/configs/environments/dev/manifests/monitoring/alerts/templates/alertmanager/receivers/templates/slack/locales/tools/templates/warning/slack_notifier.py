"""
Slack Notifier - Notifications Slack Avancées pour Spotify AI Agent
Support des messages enrichis, templates dynamiques et intégrations avancées
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

import aiohttp
from prometheus_client import Counter, Histogram
from jinja2 import Environment, DictLoader

from .schemas import AlertLevel, SlackMessageTemplate
from .utils import SecurityUtils, AttachmentBuilder


class SlackChannel(Enum):
    """Canaux Slack prédéfinis"""
    CRITICAL_ALERTS = "#critical-alerts"
    GENERAL_ALERTS = "#alerts"
    MONITORING = "#monitoring"
    ML_ALERTS = "#ml-alerts"
    SECURITY_ALERTS = "#security-alerts"
    DEVOPS = "#devops"


@dataclass
class SlackAttachment:
    """Structure d'un attachment Slack"""
    color: str
    title: str
    text: str
    fields: List[Dict[str, str]]
    timestamp: int
    footer: str
    footer_icon: str


class SlackNotifier:
    """
    Notificateur Slack avancé avec fonctionnalités :
    - Templates de messages dynamiques
    - Attachments enrichis avec métriques
    - Rate limiting intelligent
    - Retry automatique avec backoff
    - Support multi-canal
    - Formatage conditionnel basé sur la sévérité
    - Intégration avec les threads
    - Support des actions interactives
    """
    
    def __init__(
        self,
        webhook_urls: Dict[str, str],
        bot_token: Optional[str] = None,
        default_channel: str = "#alerts",
        tenant_id: str = "",
        config: Dict[str, Any] = None
    ):
        self.webhook_urls = webhook_urls
        self.bot_token = bot_token
        self.default_channel = default_channel
        self.tenant_id = tenant_id
        self.config = config or {}
        
        # Logger avec contexte tenant
        self.logger = logging.getLogger(f"slack_notifier.{tenant_id}")
        
        # Métriques Prometheus
        self.message_counter = Counter(
            'slack_messages_total',
            'Total Slack messages sent',
            ['tenant_id', 'channel', 'level', 'status']
        )
        
        self.message_duration = Histogram(
            'slack_message_duration_seconds',
            'Time to send Slack messages',
            ['tenant_id', 'channel']
        )
        
        # Templates Jinja2 pour les messages
        self.template_env = Environment(
            loader=DictLoader(self._load_message_templates())
        )
        
        # Configuration des couleurs par niveau
        self.level_colors = {
            AlertLevel.CRITICAL: "#FF0000",  # Rouge vif
            AlertLevel.HIGH: "#FF8C00",      # Orange foncé
            AlertLevel.WARNING: "#FFD700",   # Jaune/Or
            AlertLevel.INFO: "#00CED1",      # Turquoise
            AlertLevel.DEBUG: "#808080"      # Gris
        }
        
        # Mapping des canaux par niveau
        self.level_channels = {
            AlertLevel.CRITICAL: SlackChannel.CRITICAL_ALERTS.value,
            AlertLevel.HIGH: SlackChannel.GENERAL_ALERTS.value,
            AlertLevel.WARNING: SlackChannel.GENERAL_ALERTS.value,
            AlertLevel.INFO: SlackChannel.MONITORING.value,
            AlertLevel.DEBUG: SlackChannel.MONITORING.value
        }
        
        # Session HTTP réutilisable
        self.session = None
        
    async def __aenter__(self):
        """Context manager pour gestion des sessions HTTP"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': f'SpotifyAI-AlertManager/{self.tenant_id}'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Fermeture des sessions HTTP"""
        if self.session:
            await self.session.close()
    
    async def send_alert_notification(
        self,
        alert_id: str,
        level: AlertLevel,
        message: str,
        context: Dict[str, Any],
        tags: Optional[Dict[str, str]] = None,
        processed_data: Optional[Dict[str, Any]] = None,
        thread_ts: Optional[str] = None
    ) -> bool:
        """
        Envoie une notification d'alerte Slack avec formatage enrichi
        
        Args:
            alert_id: ID unique de l'alerte
            level: Niveau d'alerte
            message: Message principal
            context: Contexte de l'alerte
            tags: Tags additionnels
            processed_data: Données traitées par le warning processor
            thread_ts: Timestamp du thread parent (pour les réponses)
            
        Returns:
            bool: True si envoyé avec succès
        """
        try:
            # Sélection du canal basé sur le niveau et la configuration
            channel = self._select_channel(level, tags)
            
            # Construction du message Slack
            slack_message = await self._build_alert_message(
                alert_id=alert_id,
                level=level,
                message=message,
                context=context,
                tags=tags,
                processed_data=processed_data
            )
            
            # Ajout du thread si spécifié
            if thread_ts:
                slack_message['thread_ts'] = thread_ts
            
            # Envoi du message
            success = await self._send_message(channel, slack_message)
            
            # Métriques
            self.message_counter.labels(
                tenant_id=self.tenant_id,
                channel=channel,
                level=level.value,
                status='sent' if success else 'failed'
            ).inc()
            
            # Log d'activité
            if success:
                self.logger.info(
                    f"Alert notification sent to Slack",
                    extra={
                        "alert_id": alert_id,
                        "level": level.value,
                        "channel": channel,
                        "tenant_id": self.tenant_id
                    }
                )
            else:
                self.logger.error(
                    f"Failed to send alert notification to Slack",
                    extra={
                        "alert_id": alert_id,
                        "level": level.value,
                        "channel": channel,
                        "tenant_id": self.tenant_id
                    }
                )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending Slack notification: {str(e)}", exc_info=True)
            return False
    
    async def send_rich_message(
        self,
        template_name: str,
        data: Dict[str, Any],
        channel: Optional[str] = None,
        blocks: Optional[List[Dict]] = None,
        attachments: Optional[List[SlackAttachment]] = None
    ) -> bool:
        """
        Envoie un message enrichi avec template personnalisé
        
        Args:
            template_name: Nom du template à utiliser
            data: Données pour le template
            channel: Canal de destination
            blocks: Blocks Slack personnalisés
            attachments: Attachments personnalisés
            
        Returns:
            bool: True si envoyé avec succès
        """
        try:
            # Rendu du template
            template = self.template_env.get_template(template_name)
            rendered_content = template.render(**data, tenant_id=self.tenant_id)
            
            # Construction du message
            message = {
                'text': rendered_content,
                'channel': channel or self.default_channel
            }
            
            # Ajout des blocks si fournis
            if blocks:
                message['blocks'] = blocks
            
            # Ajout des attachments
            if attachments:
                message['attachments'] = [
                    self._attachment_to_dict(att) for att in attachments
                ]
            
            # Envoi
            return await self._send_message(message['channel'], message)
            
        except Exception as e:
            self.logger.error(f"Error sending rich message: {str(e)}", exc_info=True)
            return False
    
    async def send_batch(
        self,
        messages: List[Dict[str, Any]],
        max_concurrent: int = 5
    ) -> Dict[str, bool]:
        """
        Envoie plusieurs messages en batch avec contrôle de la concurrence
        
        Args:
            messages: Liste des messages à envoyer
            max_concurrent: Nombre maximum d'envois simultanés
            
        Returns:
            Dict[str, bool]: Résultats par message
        """
        results = {}
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def send_single(msg_data: Dict[str, Any]) -> tuple:
            async with semaphore:
                msg_id = msg_data.get('id', f"msg_{len(results)}")
                channel = msg_data.get('channel', self.default_channel)
                success = await self._send_message(channel, msg_data)
                return msg_id, success
        
        try:
            # Exécution parallèle avec limitation
            tasks = [send_single(msg) for msg in messages]
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Traitement des résultats
            for result in completed_results:
                if isinstance(result, Exception):
                    self.logger.error(f"Batch send error: {str(result)}")
                    continue
                
                msg_id, success = result
                results[msg_id] = success
            
            success_count = sum(1 for v in results.values() if v)
            self.logger.info(
                f"Batch send completed: {success_count}/{len(messages)} successful"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch send: {str(e)}", exc_info=True)
            return results
    
    async def send_acknowledgment_notification(
        self,
        alert_id: str,
        user_id: str,
        tenant_id: str
    ) -> bool:
        """
        Envoie une notification d'acquittement d'alerte
        
        Args:
            alert_id: ID de l'alerte acquittée
            user_id: ID de l'utilisateur qui acquitte
            tenant_id: ID du tenant
            
        Returns:
            bool: True si envoyé avec succès
        """
        try:
            message = {
                'text': f"✅ Alert Acknowledged",
                'channel': self.default_channel,
                'attachments': [{
                    'color': '#36A64F',  # Vert
                    'title': f'Alert {alert_id[:8]}... acknowledged',
                    'fields': [
                        {
                            'title': 'Acknowledged by',
                            'value': f'<@{user_id}>',
                            'short': True
                        },
                        {
                            'title': 'Tenant',
                            'value': tenant_id,
                            'short': True
                        },
                        {
                            'title': 'Timestamp',
                            'value': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
                            'short': True
                        }
                    ],
                    'footer': 'Spotify AI Agent Alert System',
                    'ts': int(datetime.utcnow().timestamp())
                }]
            }
            
            return await self._send_message(message['channel'], message)
            
        except Exception as e:
            self.logger.error(f"Error sending acknowledgment notification: {str(e)}")
            return False
    
    async def send_resolution_notification(
        self,
        alert_id: str,
        user_id: str,
        resolution_note: str,
        tenant_id: str
    ) -> bool:
        """
        Envoie une notification de résolution d'alerte
        
        Args:
            alert_id: ID de l'alerte résolue
            user_id: ID de l'utilisateur qui résout
            resolution_note: Note de résolution
            tenant_id: ID du tenant
            
        Returns:
            bool: True si envoyé avec succès
        """
        try:
            message = {
                'text': f"✅ Alert Resolved",
                'channel': self.default_channel,
                'attachments': [{
                    'color': '#2ECC71',  # Vert plus foncé
                    'title': f'Alert {alert_id[:8]}... resolved',
                    'text': resolution_note if resolution_note else 'No resolution note provided',
                    'fields': [
                        {
                            'title': 'Resolved by',
                            'value': f'<@{user_id}>',
                            'short': True
                        },
                        {
                            'title': 'Tenant',
                            'value': tenant_id,
                            'short': True
                        },
                        {
                            'title': 'Resolution Time',
                            'value': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
                            'short': True
                        }
                    ],
                    'footer': 'Spotify AI Agent Alert System',
                    'ts': int(datetime.utcnow().timestamp())
                }]
            }
            
            return await self._send_message(message['channel'], message)
            
        except Exception as e:
            self.logger.error(f"Error sending resolution notification: {str(e)}")
            return False
    
    async def send_metric_alert(
        self,
        metric_name: str,
        current_value: float,
        threshold_value: float,
        service_name: str,
        tags: Dict[str, str]
    ) -> bool:
        """
        Envoie une alerte spécifique aux métriques
        
        Args:
            metric_name: Nom de la métrique
            current_value: Valeur actuelle
            threshold_value: Seuil dépassé
            service_name: Nom du service
            tags: Tags additionnels
            
        Returns:
            bool: True si envoyé avec succès
        """
        try:
            # Détermination de la couleur basée sur la gravité
            severity_ratio = current_value / threshold_value
            if severity_ratio >= 2.0:
                color = self.level_colors[AlertLevel.CRITICAL]
                icon = "🚨"
            elif severity_ratio >= 1.5:
                color = self.level_colors[AlertLevel.HIGH]
                icon = "⚠️"
            else:
                color = self.level_colors[AlertLevel.WARNING]
                icon = "⚡"
            
            message = {
                'text': f"{icon} Metric Alert: {metric_name}",
                'channel': self.level_channels[AlertLevel.WARNING],
                'attachments': [{
                    'color': color,
                    'title': f'Metric Threshold Exceeded: {metric_name}',
                    'fields': [
                        {
                            'title': 'Current Value',
                            'value': f'{current_value:.2f}',
                            'short': True
                        },
                        {
                            'title': 'Threshold',
                            'value': f'{threshold_value:.2f}',
                            'short': True
                        },
                        {
                            'title': 'Service',
                            'value': service_name,
                            'short': True
                        },
                        {
                            'title': 'Tenant',
                            'value': self.tenant_id,
                            'short': True
                        }
                    ] + [
                        {
                            'title': k.title(),
                            'value': v,
                            'short': True
                        }
                        for k, v in tags.items()
                    ],
                    'footer': 'Spotify AI Agent Metrics',
                    'ts': int(datetime.utcnow().timestamp())
                }]
            }
            
            return await self._send_message(message['channel'], message)
            
        except Exception as e:
            self.logger.error(f"Error sending metric alert: {str(e)}")
            return False
    
    # Méthodes privées
    
    def _load_message_templates(self) -> Dict[str, str]:
        """Charge les templates de messages Slack"""
        return {
            'ml_model_drift': """
🤖 **ML Model Drift Detected**

Model: {{ model_name }}
Drift Score: {{ drift_score }}
Threshold: {{ threshold }}
Environment: {{ environment }}

Current model performance may be degraded. Consider retraining.
            """.strip(),
            
            'api_performance': """
🚀 **API Performance Alert**

Endpoint: {{ endpoint }}
Response Time: {{ response_time }}ms
Error Rate: {{ error_rate }}%
Tenant: {{ tenant_id }}

Performance metrics outside acceptable thresholds.
            """.strip(),
            
            'security_alert': """
🔒 **Security Alert**

Type: {{ alert_type }}
Source IP: {{ source_ip }}
User: {{ user_id or 'Unknown' }}
Tenant: {{ tenant_id }}

{{ description }}
            """.strip(),
            
            'system_health': """
💓 **System Health Alert**

Service: {{ service_name }}
Status: {{ status }}
CPU: {{ cpu_usage }}%
Memory: {{ memory_usage }}%
Disk: {{ disk_usage }}%

{{ additional_info }}
            """.strip()
        }
    
    def _select_channel(self, level: AlertLevel, tags: Optional[Dict[str, str]]) -> str:
        """Sélectionne le canal approprié basé sur le niveau et les tags"""
        # Canal spécifique pour la sécurité
        if tags and tags.get('category') == 'security':
            return SlackChannel.SECURITY_ALERTS.value
        
        # Canal spécifique pour ML
        if tags and tags.get('category') == 'ml':
            return SlackChannel.ML_ALERTS.value
        
        # Canal basé sur le niveau
        return self.level_channels.get(level, self.default_channel)
    
    async def _build_alert_message(
        self,
        alert_id: str,
        level: AlertLevel,
        message: str,
        context: Dict[str, Any],
        tags: Optional[Dict[str, str]],
        processed_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Construit un message Slack enrichi pour une alerte"""
        
        # Icône basée sur le niveau
        level_icons = {
            AlertLevel.CRITICAL: "🚨",
            AlertLevel.HIGH: "⚠️",
            AlertLevel.WARNING: "⚡",
            AlertLevel.INFO: "ℹ️",
            AlertLevel.DEBUG: "🔍"
        }
        
        icon = level_icons.get(level, "📢")
        color = self.level_colors.get(level, "#808080")
        
        # Construction des champs d'information
        fields = [
            {
                'title': 'Alert ID',
                'value': f'`{alert_id[:8]}...`',
                'short': True
            },
            {
                'title': 'Level',
                'value': level.value,
                'short': True
            },
            {
                'title': 'Service',
                'value': context.get('service_name', 'Unknown'),
                'short': True
            },
            {
                'title': 'Environment',
                'value': context.get('environment', 'Unknown'),
                'short': True
            },
            {
                'title': 'Tenant',
                'value': self.tenant_id,
                'short': True
            },
            {
                'title': 'Timestamp',
                'value': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
                'short': True
            }
        ]
        
        # Ajout des tags comme champs
        if tags:
            for key, value in tags.items():
                fields.append({
                    'title': key.title(),
                    'value': value,
                    'short': True
                })
        
        # Ajout des données traitées importantes
        if processed_data:
            if 'severity_score' in processed_data:
                fields.append({
                    'title': 'Severity Score',
                    'value': f"{processed_data['severity_score']:.2f}",
                    'short': True
                })
            
            if 'recommendation' in processed_data:
                fields.append({
                    'title': 'Recommendation',
                    'value': processed_data['recommendation'],
                    'short': False
                })
        
        # Construction du message complet
        slack_message = {
            'text': f"{icon} {level.value} Alert",
            'attachments': [{
                'color': color,
                'title': f'{icon} {level.value} Alert',
                'text': message,
                'fields': fields,
                'footer': 'Spotify AI Agent Alert System',
                'footer_icon': 'https://platform.slack-edge.com/img/default_application_icon.png',
                'ts': int(datetime.utcnow().timestamp())
            }]
        }
        
        # Ajout d'actions pour les alertes critiques
        if level in [AlertLevel.CRITICAL, AlertLevel.HIGH]:
            slack_message['attachments'][0]['actions'] = [
                {
                    'type': 'button',
                    'text': 'Acknowledge',
                    'style': 'primary',
                    'value': f'ack_{alert_id}'
                },
                {
                    'type': 'button',
                    'text': 'Resolve',
                    'style': 'good',
                    'value': f'resolve_{alert_id}'
                },
                {
                    'type': 'button',
                    'text': 'Escalate',
                    'style': 'danger',
                    'value': f'escalate_{alert_id}'
                }
            ]
        
        return slack_message
    
    async def _send_message(self, channel: str, message: Dict[str, Any]) -> bool:
        """Envoie un message Slack avec retry automatique"""
        if not self.session:
            await self.__aenter__()
        
        message['channel'] = channel
        
        # Sélection de l'URL webhook appropriée
        webhook_url = self.webhook_urls.get(channel) or self.webhook_urls.get('default')
        if not webhook_url:
            self.logger.error(f"No webhook URL configured for channel {channel}")
            return False
        
        # Retry avec backoff exponentiel
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with self.session.post(
                    webhook_url,
                    json=message,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status == 200:
                        return True
                    elif response.status == 429:  # Rate limited
                        retry_after = int(response.headers.get('Retry-After', 60))
                        self.logger.warning(f"Rate limited, retrying after {retry_after}s")
                        await asyncio.sleep(retry_after)
                    else:
                        response_text = await response.text()
                        self.logger.error(
                            f"Slack API error: {response.status} - {response_text}"
                        )
                        
            except Exception as e:
                self.logger.error(f"Error sending to Slack (attempt {attempt + 1}): {str(e)}")
                
                if attempt < max_retries - 1:
                    # Backoff exponentiel: 1s, 2s, 4s
                    await asyncio.sleep(2 ** attempt)
        
        return False
    
    def _attachment_to_dict(self, attachment: SlackAttachment) -> Dict[str, Any]:
        """Convertit un SlackAttachment en dictionnaire"""
        return {
            'color': attachment.color,
            'title': attachment.title,
            'text': attachment.text,
            'fields': attachment.fields,
            'ts': attachment.timestamp,
            'footer': attachment.footer,
            'footer_icon': attachment.footer_icon
        }
