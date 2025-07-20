"""
Factories pour la création et gestion des receivers d'alertes.

Ce module implémente le pattern Factory pour créer des instances
de receivers spécialisés selon le type de canal de notification.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type
import aiohttp
import json
from datetime import datetime
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.utils import formataddr

from .models import (
    ReceiverConfig,
    AlertContext,
    NotificationResult,
    ChannelType,
    SlackReceiverConfig,
    EmailReceiverConfig,
    PagerDutyReceiverConfig,
    WebhookReceiverConfig
)
from .exceptions import NotificationError, ConfigurationError

logger = logging.getLogger(__name__)

class BaseReceiverFactory(ABC):
    """Factory de base pour tous les types de receivers"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def send_notification(
        self,
        receiver: ReceiverConfig,
        message: Dict[str, Any],
        alert_context: AlertContext,
        session: aiohttp.ClientSession
    ) -> Dict[str, Any]:
        """Envoie une notification via ce receiver"""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Valide la configuration du receiver"""
        pass
    
    def format_error_message(self, error: Exception, context: str = "") -> str:
        """Formate un message d'erreur standardisé"""
        return f"{context}: {type(error).__name__}: {str(error)}"

class SlackReceiverFactory(BaseReceiverFactory):
    """Factory pour les receivers Slack"""
    
    async def send_notification(
        self,
        receiver: ReceiverConfig,
        message: Dict[str, Any],
        alert_context: AlertContext,
        session: aiohttp.ClientSession
    ) -> Dict[str, Any]:
        """Envoie une notification Slack"""
        try:
            config = SlackReceiverConfig(**receiver.config)
            
            # Construire le payload Slack
            payload = await self._build_slack_payload(config, message, alert_context)
            
            # Envoyer via webhook
            webhook_url = config.webhook_url.get_secret_value()
            
            async with session.post(
                webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=aiohttp.ClientTimeout(total=receiver.timeout_seconds)
            ) as response:
                response_text = await response.text()
                
                if response.status == 200:
                    self.logger.info(f"Slack notification sent successfully to {receiver.name}")
                    return {
                        'status': 'success',
                        'response': response_text,
                        'webhook_url': webhook_url[:50] + "..." if len(webhook_url) > 50 else webhook_url
                    }
                else:
                    error_msg = f"Slack API error: {response.status} - {response_text}"
                    self.logger.error(error_msg)
                    raise NotificationError(error_msg)
                    
        except Exception as e:
            error_msg = self.format_error_message(e, "Slack notification failed")
            self.logger.error(error_msg)
            raise NotificationError(error_msg)
    
    async def _build_slack_payload(
        self,
        config: SlackReceiverConfig,
        message: Dict[str, Any],
        alert_context: AlertContext
    ) -> Dict[str, Any]:
        """Construit le payload pour l'API Slack"""
        
        # Couleur selon la sévérité
        severity_colors = {
            'critical': '#FF0000',
            'high': '#FF8C00', 
            'medium': '#FFD700',
            'low': '#32CD32',
            'info': '#4169E1'
        }
        
        color = severity_colors.get(alert_context.severity.value, '#808080')
        
        # Construire l'attachment principal
        attachment = {
            'color': color,
            'title': f"🚨 {alert_context.alert_name}",
            'title_link': alert_context.dashboard_url,
            'text': alert_context.annotations.get('summary', ''),
            'fallback': f"{alert_context.severity.value.upper()}: {alert_context.alert_name}",
            'fields': [],
            'footer': 'Spotify AI Agent AlertManager',
            'footer_icon': 'https://prometheus.io/assets/prometheus_logo_grey.svg',
            'ts': int(alert_context.starts_at.timestamp())
        }
        
        # Ajouter les champs
        attachment['fields'].extend([
            {
                'title': 'Sévérité',
                'value': alert_context.severity.value.upper(),
                'short': True
            },
            {
                'title': 'Statut',
                'value': '🔥 FIRING' if alert_context.status == 'firing' else '✅ RESOLVED',
                'short': True
            },
            {
                'title': 'Tenant',
                'value': alert_context.tenant_id,
                'short': True
            }
        ])
        
        # Ajouter les labels principaux
        important_labels = ['service', 'environment', 'region', 'cluster']
        for label in important_labels:
            if label in alert_context.labels:
                attachment['fields'].append({
                    'title': label.title(),
                    'value': alert_context.labels[label],
                    'short': True
                })
        
        # Description détaillée si disponible
        if 'description' in alert_context.annotations:
            attachment['fields'].append({
                'title': 'Description',
                'value': alert_context.annotations['description'][:500],
                'short': False
            })
        
        # Actions
        actions = []
        if alert_context.silence_url:
            actions.append({
                'type': 'button',
                'text': '🔇 Silence',
                'url': alert_context.silence_url,
                'style': 'primary'
            })
        
        if alert_context.dashboard_url:
            actions.append({
                'type': 'button', 
                'text': '📊 Dashboard',
                'url': alert_context.dashboard_url
            })
        
        if actions:
            attachment['actions'] = actions
        
        # Payload final
        payload = {
            'username': config.username,
            'icon_emoji': config.icon_emoji,
            'attachments': [attachment]
        }
        
        # Canal spécifique
        if config.channel:
            payload['channel'] = config.channel
        
        # Mentions
        mentions = []
        if config.mention_here and alert_context.severity.value in ['critical', 'high']:
            mentions.append('<!here>')
        if config.mention_channel and alert_context.severity.value == 'critical':
            mentions.append('<!channel>')
        
        for user in config.mention_users:
            mentions.append(f'<@{user}>')
        
        for group in config.mention_groups:
            mentions.append(f'<!subteam^{group}>')
        
        if mentions:
            payload['text'] = ' '.join(mentions)
        
        return payload
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Valide la configuration Slack"""
        try:
            SlackReceiverConfig(**config)
            return True
        except Exception as e:
            self.logger.error(f"Slack config validation failed: {e}")
            return False

class EmailReceiverFactory(BaseReceiverFactory):
    """Factory pour les receivers Email"""
    
    async def send_notification(
        self,
        receiver: ReceiverConfig,
        message: Dict[str, Any],
        alert_context: AlertContext,
        session: aiohttp.ClientSession
    ) -> Dict[str, Any]:
        """Envoie une notification par email"""
        try:
            config = EmailReceiverConfig(**receiver.config)
            
            # Construire le message email
            msg = await self._build_email_message(config, message, alert_context)
            
            # Envoyer via SMTP
            result = await self._send_smtp_email(config, msg)
            
            self.logger.info(f"Email notification sent to {len(config.recipients)} recipients")
            return result
            
        except Exception as e:
            error_msg = self.format_error_message(e, "Email notification failed")
            self.logger.error(error_msg)
            raise NotificationError(error_msg)
    
    async def _build_email_message(
        self,
        config: EmailReceiverConfig,
        message: Dict[str, Any],
        alert_context: AlertContext
    ) -> MimeMultipart:
        """Construit le message email"""
        
        # Message multipart
        msg = MimeMultipart('alternative')
        
        # Sujet
        subject_template = config.subject_template or "[{severity}] {alert_name} - {tenant_id}"
        subject = subject_template.format(
            severity=alert_context.severity.value.upper(),
            alert_name=alert_context.alert_name,
            tenant_id=alert_context.tenant_id,
            status=alert_context.status.upper()
        )
        msg['Subject'] = subject
        
        # Expéditeur
        msg['From'] = formataddr((config.from_name, config.from_address))
        
        # Destinataires
        msg['To'] = ', '.join(config.recipients)
        if config.cc_recipients:
            msg['Cc'] = ', '.join(config.cc_recipients)
        
        # Headers personnalisés
        msg['X-Alert-Name'] = alert_context.alert_name
        msg['X-Alert-Severity'] = alert_context.severity.value
        msg['X-Tenant-ID'] = alert_context.tenant_id
        msg['X-Alert-Fingerprint'] = alert_context.fingerprint
        
        # Contenu texte
        text_content = await self._build_text_content(alert_context)
        msg.attach(MimeText(text_content, 'plain', 'utf-8'))
        
        # Contenu HTML si activé
        if config.html_template:
            html_content = await self._build_html_content(alert_context)
            msg.attach(MimeText(html_content, 'html', 'utf-8'))
        
        return msg
    
    async def _build_text_content(self, alert_context: AlertContext) -> str:
        """Construit le contenu texte de l'email"""
        lines = [
            f"ALERTE: {alert_context.alert_name}",
            f"Sévérité: {alert_context.severity.value.upper()}",
            f"Statut: {alert_context.status.upper()}",
            f"Tenant: {alert_context.tenant_id}",
            f"Début: {alert_context.starts_at.isoformat()}",
            "",
            "Résumé:",
            alert_context.annotations.get('summary', 'Aucun résumé disponible'),
            ""
        ]
        
        if 'description' in alert_context.annotations:
            lines.extend([
                "Description:",
                alert_context.annotations['description'],
                ""
            ])
        
        if alert_context.labels:
            lines.append("Labels:")
            for key, value in alert_context.labels.items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        if alert_context.dashboard_url:
            lines.extend([
                "Liens:",
                f"Dashboard: {alert_context.dashboard_url}"
            ])
        
        if alert_context.silence_url:
            lines.append(f"Silence: {alert_context.silence_url}")
        
        return '\n'.join(lines)
    
    async def _build_html_content(self, alert_context: AlertContext) -> str:
        """Construit le contenu HTML de l'email"""
        
        # Couleur selon la sévérité
        severity_colors = {
            'critical': '#FF0000',
            'high': '#FF8C00',
            'medium': '#FFD700', 
            'low': '#32CD32',
            'info': '#4169E1'
        }
        
        color = severity_colors.get(alert_context.severity.value, '#808080')
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                .alert-header {{ 
                    background-color: {color}; 
                    color: white; 
                    padding: 15px; 
                    border-radius: 5px 5px 0 0; 
                }}
                .alert-body {{ 
                    border: 1px solid {color}; 
                    padding: 15px; 
                    border-radius: 0 0 5px 5px; 
                }}
                .label {{ font-weight: bold; }}
                .value {{ margin-left: 10px; }}
                .section {{ margin: 15px 0; }}
                .button {{ 
                    display: inline-block; 
                    background-color: {color}; 
                    color: white; 
                    padding: 10px 15px; 
                    text-decoration: none; 
                    border-radius: 3px; 
                    margin: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="alert-header">
                <h2>🚨 {alert_context.alert_name}</h2>
                <p>Sévérité: {alert_context.severity.value.upper()} | Statut: {alert_context.status.upper()}</p>
            </div>
            
            <div class="alert-body">
                <div class="section">
                    <div class="label">Tenant:</div>
                    <div class="value">{alert_context.tenant_id}</div>
                </div>
                
                <div class="section">
                    <div class="label">Début:</div>
                    <div class="value">{alert_context.starts_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</div>
                </div>
                
                <div class="section">
                    <div class="label">Résumé:</div>
                    <div class="value">{alert_context.annotations.get('summary', 'Aucun résumé disponible')}</div>
                </div>
        """
        
        if 'description' in alert_context.annotations:
            html += f"""
                <div class="section">
                    <div class="label">Description:</div>
                    <div class="value">{alert_context.annotations['description']}</div>
                </div>
            """
        
        if alert_context.labels:
            html += '<div class="section"><div class="label">Labels:</div>'
            for key, value in alert_context.labels.items():
                html += f'<div class="value">{key}: {value}</div>'
            html += '</div>'
        
        # Boutons d'action
        html += '<div class="section">'
        if alert_context.dashboard_url:
            html += f'<a href="{alert_context.dashboard_url}" class="button">📊 Dashboard</a>'
        if alert_context.silence_url:
            html += f'<a href="{alert_context.silence_url}" class="button">🔇 Silence</a>'
        html += '</div>'
        
        html += """
            </div>
            <br>
            <small>Spotify AI Agent AlertManager</small>
        </body>
        </html>
        """
        
        return html
    
    async def _send_smtp_email(self, config: EmailReceiverConfig, msg: MimeMultipart) -> Dict[str, Any]:
        """Envoie l'email via SMTP"""
        
        # Exécuter dans un thread pour éviter de bloquer
        loop = asyncio.get_event_loop()
        
        def _smtp_send():
            if config.use_ssl:
                server = smtplib.SMTP_SSL(config.smtp_server, config.smtp_port)
            else:
                server = smtplib.SMTP(config.smtp_server, config.smtp_port)
                if config.use_tls:
                    server.starttls()
            
            if config.username and config.password:
                server.login(config.username, config.password.get_secret_value())
            
            # Tous les destinataires
            recipients = list(config.recipients)
            recipients.extend(config.cc_recipients)
            recipients.extend(config.bcc_recipients)
            
            server.send_message(msg, to_addrs=recipients)
            server.quit()
            
            return {
                'status': 'success',
                'recipients_count': len(recipients),
                'message_id': msg.get('Message-ID')
            }
        
        return await loop.run_in_executor(None, _smtp_send)
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Valide la configuration Email"""
        try:
            EmailReceiverConfig(**config)
            return True
        except Exception as e:
            self.logger.error(f"Email config validation failed: {e}")
            return False

class PagerDutyReceiverFactory(BaseReceiverFactory):
    """Factory pour les receivers PagerDuty"""
    
    PAGERDUTY_API_URL = "https://events.pagerduty.com/v2/enqueue"
    
    async def send_notification(
        self,
        receiver: ReceiverConfig,
        message: Dict[str, Any],
        alert_context: AlertContext,
        session: aiohttp.ClientSession
    ) -> Dict[str, Any]:
        """Envoie une notification PagerDuty"""
        try:
            config = PagerDutyReceiverConfig(**receiver.config)
            
            # Construire l'événement PagerDuty
            event = await self._build_pagerduty_event(config, alert_context)
            
            # Envoyer à l'API PagerDuty
            async with session.post(
                self.PAGERDUTY_API_URL,
                json=event,
                headers={'Content-Type': 'application/json'},
                timeout=aiohttp.ClientTimeout(total=receiver.timeout_seconds)
            ) as response:
                response_data = await response.json()
                
                if response.status == 202:
                    self.logger.info(f"PagerDuty notification sent: {response_data.get('dedup_key')}")
                    return {
                        'status': 'success',
                        'dedup_key': response_data.get('dedup_key'),
                        'incident_key': response_data.get('incident_key')
                    }
                else:
                    error_msg = f"PagerDuty API error: {response.status} - {response_data}"
                    self.logger.error(error_msg)
                    raise NotificationError(error_msg)
                    
        except Exception as e:
            error_msg = self.format_error_message(e, "PagerDuty notification failed")
            self.logger.error(error_msg)
            raise NotificationError(error_msg)
    
    async def _build_pagerduty_event(
        self,
        config: PagerDutyReceiverConfig,
        alert_context: AlertContext
    ) -> Dict[str, Any]:
        """Construit l'événement PagerDuty"""
        
        # Mapping des sévérités
        severity_mapping = {
            'critical': 'critical',
            'high': 'error',
            'medium': 'warning',
            'low': 'info',
            'info': 'info'
        }
        
        # Déterminer l'action
        event_action = 'resolve' if alert_context.status == 'resolved' else 'trigger'
        
        event = {
            'routing_key': config.integration_key.get_secret_value(),
            'event_action': event_action,
            'dedup_key': config.dedup_key or alert_context.fingerprint,
            'client': config.client,
            'client_url': config.client_url,
            'payload': {
                'summary': f"{alert_context.alert_name} - {alert_context.annotations.get('summary', '')}",
                'source': alert_context.tenant_id,
                'severity': severity_mapping.get(alert_context.severity.value, 'info'),
                'timestamp': alert_context.starts_at.isoformat(),
                'component': alert_context.labels.get('service', 'unknown'),
                'group': alert_context.labels.get('environment', 'unknown'),
                'class': alert_context.labels.get('alertname', alert_context.alert_name),
                'custom_details': {
                    'tenant_id': alert_context.tenant_id,
                    'alert_name': alert_context.alert_name,
                    'severity': alert_context.severity.value,
                    'labels': alert_context.labels,
                    'annotations': alert_context.annotations,
                    'fingerprint': alert_context.fingerprint,
                    **config.custom_details
                }
            }
        }
        
        # Ajouter les liens
        if config.links or alert_context.dashboard_url:
            event['links'] = list(config.links)
            if alert_context.dashboard_url:
                event['links'].append({
                    'href': alert_context.dashboard_url,
                    'text': 'Dashboard'
                })
        
        # Ajouter les images
        if config.images:
            event['images'] = config.images
        
        return event
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Valide la configuration PagerDuty"""
        try:
            PagerDutyReceiverConfig(**config)
            return True
        except Exception as e:
            self.logger.error(f"PagerDuty config validation failed: {e}")
            return False

class WebhookReceiverFactory(BaseReceiverFactory):
    """Factory pour les receivers Webhook génériques"""
    
    async def send_notification(
        self,
        receiver: ReceiverConfig,
        message: Dict[str, Any],
        alert_context: AlertContext,
        session: aiohttp.ClientSession
    ) -> Dict[str, Any]:
        """Envoie une notification via webhook"""
        try:
            config = WebhookReceiverConfig(**receiver.config)
            
            # Construire le payload
            payload = await self._build_webhook_payload(config, alert_context)
            
            # Préparer les headers
            headers = dict(config.headers)
            headers['Content-Type'] = config.content_type
            
            # Authentification
            await self._add_authentication(config, headers, session)
            
            # Envoyer la requête
            async with session.request(
                config.method,
                config.url,
                json=payload if config.content_type == 'application/json' else None,
                data=payload if config.content_type != 'application/json' else None,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=receiver.timeout_seconds)
            ) as response:
                response_text = await response.text()
                
                if 200 <= response.status < 300:
                    self.logger.info(f"Webhook notification sent to {config.url}")
                    return {
                        'status': 'success',
                        'status_code': response.status,
                        'response': response_text[:500]  # Limiter la taille
                    }
                elif response.status in config.retry_status_codes:
                    error_msg = f"Webhook retryable error: {response.status} - {response_text}"
                    self.logger.warning(error_msg)
                    raise NotificationError(error_msg)
                else:
                    error_msg = f"Webhook error: {response.status} - {response_text}"
                    self.logger.error(error_msg)
                    raise NotificationError(error_msg)
                    
        except Exception as e:
            error_msg = self.format_error_message(e, "Webhook notification failed")
            self.logger.error(error_msg)
            raise NotificationError(error_msg)
    
    async def _build_webhook_payload(
        self,
        config: WebhookReceiverConfig,
        alert_context: AlertContext
    ) -> Dict[str, Any]:
        """Construit le payload pour le webhook"""
        
        if config.payload_template:
            # Utiliser le template personnalisé
            from jinja2 import Template
            template = Template(config.payload_template)
            payload_str = template.render(alert=alert_context.to_dict())
            return json.loads(payload_str)
        else:
            # Payload par défaut
            return {
                'alert_name': alert_context.alert_name,
                'severity': alert_context.severity.value,
                'status': alert_context.status,
                'tenant_id': alert_context.tenant_id,
                'labels': alert_context.labels,
                'annotations': alert_context.annotations,
                'starts_at': alert_context.starts_at.isoformat(),
                'ends_at': alert_context.ends_at.isoformat() if alert_context.ends_at else None,
                'fingerprint': alert_context.fingerprint,
                'dashboard_url': alert_context.dashboard_url,
                'silence_url': alert_context.silence_url,
                'generator_url': alert_context.generator_url
            }
    
    async def _add_authentication(
        self,
        config: WebhookReceiverConfig,
        headers: Dict[str, str],
        session: aiohttp.ClientSession
    ):
        """Ajoute l'authentification aux headers"""
        
        if config.auth_type == 'basic' and config.username and config.password:
            import base64
            credentials = base64.b64encode(
                f"{config.username}:{config.password.get_secret_value()}".encode()
            ).decode()
            headers['Authorization'] = f'Basic {credentials}'
            
        elif config.auth_type == 'bearer' and config.token:
            headers['Authorization'] = f'Bearer {config.token.get_secret_value()}'
            
        elif config.auth_type == 'api_key' and config.api_key:
            headers[config.api_key_header] = config.api_key.get_secret_value()
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Valide la configuration Webhook"""
        try:
            WebhookReceiverConfig(**config)
            return True
        except Exception as e:
            self.logger.error(f"Webhook config validation failed: {e}")
            return False

class TeamsReceiverFactory(BaseReceiverFactory):
    """Factory pour les receivers Microsoft Teams"""
    
    async def send_notification(
        self,
        receiver: ReceiverConfig,
        message: Dict[str, Any],
        alert_context: AlertContext,
        session: aiohttp.ClientSession
    ) -> Dict[str, Any]:
        """Envoie une notification Teams"""
        try:
            webhook_url = receiver.config.get('webhook_url')
            if not webhook_url:
                raise ConfigurationError("webhook_url is required for Teams receiver")
            
            # Construire la carte adaptative Teams
            card = await self._build_teams_card(alert_context)
            
            async with session.post(
                webhook_url,
                json=card,
                headers={'Content-Type': 'application/json'},
                timeout=aiohttp.ClientTimeout(total=receiver.timeout_seconds)
            ) as response:
                response_text = await response.text()
                
                if response.status == 200:
                    self.logger.info(f"Teams notification sent successfully")
                    return {'status': 'success', 'response': response_text}
                else:
                    error_msg = f"Teams API error: {response.status} - {response_text}"
                    self.logger.error(error_msg)
                    raise NotificationError(error_msg)
                    
        except Exception as e:
            error_msg = self.format_error_message(e, "Teams notification failed")
            self.logger.error(error_msg)
            raise NotificationError(error_msg)
    
    async def _build_teams_card(self, alert_context: AlertContext) -> Dict[str, Any]:
        """Construit une carte adaptive pour Teams"""
        
        severity_colors = {
            'critical': 'Attention',
            'high': 'Warning',
            'medium': 'Accent',
            'low': 'Good',
            'info': 'Default'
        }
        
        color = severity_colors.get(alert_context.severity.value, 'Default')
        
        card = {
            "@type": "MessageCard",
            "@context": "https://schema.org/extensions",
            "themeColor": "FF0000" if alert_context.severity.value == 'critical' else "FF8C00",
            "summary": f"{alert_context.alert_name} - {alert_context.severity.value.upper()}",
            "sections": [
                {
                    "activityTitle": f"🚨 {alert_context.alert_name}",
                    "activitySubtitle": f"Sévérité: {alert_context.severity.value.upper()} | Tenant: {alert_context.tenant_id}",
                    "facts": [
                        {"name": "Statut", "value": alert_context.status.upper()},
                        {"name": "Début", "value": alert_context.starts_at.strftime('%Y-%m-%d %H:%M:%S UTC')},
                        {"name": "Résumé", "value": alert_context.annotations.get('summary', 'N/A')}
                    ]
                }
            ]
        }
        
        # Ajouter les labels importants
        if alert_context.labels:
            for key, value in list(alert_context.labels.items())[:5]:  # Limiter à 5
                card["sections"][0]["facts"].append({"name": key.title(), "value": value})
        
        # Actions
        if alert_context.dashboard_url or alert_context.silence_url:
            card["potentialAction"] = []
            
            if alert_context.dashboard_url:
                card["potentialAction"].append({
                    "@type": "OpenUri",
                    "name": "📊 Voir Dashboard",
                    "targets": [{"os": "default", "uri": alert_context.dashboard_url}]
                })
            
            if alert_context.silence_url:
                card["potentialAction"].append({
                    "@type": "OpenUri", 
                    "name": "🔇 Silence",
                    "targets": [{"os": "default", "uri": alert_context.silence_url}]
                })
        
        return card
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Valide la configuration Teams"""
        return 'webhook_url' in config

class DiscordReceiverFactory(BaseReceiverFactory):
    """Factory pour les receivers Discord"""
    
    async def send_notification(
        self,
        receiver: ReceiverConfig,
        message: Dict[str, Any],
        alert_context: AlertContext,
        session: aiohttp.ClientSession
    ) -> Dict[str, Any]:
        """Envoie une notification Discord"""
        try:
            webhook_url = receiver.config.get('webhook_url')
            if not webhook_url:
                raise ConfigurationError("webhook_url is required for Discord receiver")
            
            # Construire l'embed Discord
            embed = await self._build_discord_embed(alert_context)
            payload = {"embeds": [embed]}
            
            # Mentions si critique
            if alert_context.severity.value in ['critical', 'high']:
                mentions = receiver.config.get('mentions', [])
                if mentions:
                    payload["content"] = " ".join([f"<@{mention}>" for mention in mentions])
            
            async with session.post(
                webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=aiohttp.ClientTimeout(total=receiver.timeout_seconds)
            ) as response:
                if response.status == 204:  # Discord retourne 204 pour succès
                    self.logger.info(f"Discord notification sent successfully")
                    return {'status': 'success'}
                else:
                    response_text = await response.text()
                    error_msg = f"Discord API error: {response.status} - {response_text}"
                    self.logger.error(error_msg)
                    raise NotificationError(error_msg)
                    
        except Exception as e:
            error_msg = self.format_error_message(e, "Discord notification failed")
            self.logger.error(error_msg)
            raise NotificationError(error_msg)
    
    async def _build_discord_embed(self, alert_context: AlertContext) -> Dict[str, Any]:
        """Construit un embed Discord"""
        
        severity_colors = {
            'critical': 0xFF0000,  # Rouge
            'high': 0xFF8C00,      # Orange
            'medium': 0xFFD700,    # Jaune
            'low': 0x32CD32,       # Vert
            'info': 0x4169E1       # Bleu
        }
        
        color = severity_colors.get(alert_context.severity.value, 0x808080)
        
        embed = {
            "title": f"🚨 {alert_context.alert_name}",
            "description": alert_context.annotations.get('summary', ''),
            "color": color,
            "timestamp": alert_context.starts_at.isoformat(),
            "footer": {
                "text": "Spotify AI Agent AlertManager",
                "icon_url": "https://prometheus.io/assets/prometheus_logo_grey.svg"
            },
            "fields": [
                {
                    "name": "Sévérité",
                    "value": alert_context.severity.value.upper(),
                    "inline": True
                },
                {
                    "name": "Statut", 
                    "value": "🔥 FIRING" if alert_context.status == 'firing' else "✅ RESOLVED",
                    "inline": True
                },
                {
                    "name": "Tenant",
                    "value": alert_context.tenant_id,
                    "inline": True
                }
            ]
        }
        
        # Ajouter les labels principaux
        important_labels = ['service', 'environment', 'region']
        for label in important_labels:
            if label in alert_context.labels:
                embed["fields"].append({
                    "name": label.title(),
                    "value": alert_context.labels[label],
                    "inline": True
                })
        
        # Description détaillée
        if 'description' in alert_context.annotations:
            embed["fields"].append({
                "name": "Description",
                "value": alert_context.annotations['description'][:1024],  # Limite Discord
                "inline": False
            })
        
        # URL du dashboard
        if alert_context.dashboard_url:
            embed["url"] = alert_context.dashboard_url
        
        return embed
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Valide la configuration Discord"""
        return 'webhook_url' in config

# Registry des factories
_RECEIVER_FACTORIES: Dict[ChannelType, Type[BaseReceiverFactory]] = {
    ChannelType.SLACK: SlackReceiverFactory,
    ChannelType.EMAIL: EmailReceiverFactory,
    ChannelType.PAGERDUTY: PagerDutyReceiverFactory,
    ChannelType.WEBHOOK: WebhookReceiverFactory,
    ChannelType.TEAMS: TeamsReceiverFactory,
    ChannelType.DISCORD: DiscordReceiverFactory
}

def get_receiver_factory(channel_type: ChannelType) -> BaseReceiverFactory:
    """Retourne la factory appropriée pour le type de canal"""
    factory_class = _RECEIVER_FACTORIES.get(channel_type)
    if not factory_class:
        raise NotificationError(f"Unsupported channel type: {channel_type}")
    
    return factory_class()

def register_receiver_factory(channel_type: ChannelType, factory_class: Type[BaseReceiverFactory]):
    """Enregistre une nouvelle factory pour un type de canal"""
    _RECEIVER_FACTORIES[channel_type] = factory_class

def get_supported_channel_types() -> List[ChannelType]:
    """Retourne la liste des types de canaux supportés"""
    return list(_RECEIVER_FACTORIES.keys())
