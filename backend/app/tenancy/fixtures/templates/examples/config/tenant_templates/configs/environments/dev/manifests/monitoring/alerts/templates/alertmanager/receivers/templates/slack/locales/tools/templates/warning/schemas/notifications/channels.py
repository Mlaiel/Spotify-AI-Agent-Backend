"""
Services de canaux de notification ultra-avancés
===============================================

Implémentations spécialisées pour chaque canal avec retry intelligent,
circuit breakers, et optimisations de performance.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod
import re
from pathlib import Path

import httpx
import aioredis
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.errors import SlackApiError
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import aiosmtplib
from twilio.rest import Client as TwilioClient
import firebase_admin
from firebase_admin import messaging
from jinja2 import Template
import backoff
from circuitbreaker import circuit
from tenacity import retry, stop_after_attempt, wait_exponential

from .services import BaseNotificationService, NotificationServiceError
from .schemas import *


class SlackNotificationService(BaseNotificationService):
    """Service avancé pour les notifications Slack"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client: Optional[AsyncWebClient] = None
        self.bot_token = config.get('bot_token')
        self.user_token = config.get('user_token')
        self.signing_secret = config.get('signing_secret')
        
        # Templates Slack avancés
        self.default_blocks_template = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🔔 {{title}}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "{{message}}"
                }
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "Priority: *{{priority}}* | Source: {{source_system}}"
                    }
                ]
            }
        ]
    
    async def initialize(self):
        """Initialisation du client Slack"""
        await super().initialize()
        
        if not self.bot_token:
            raise NotificationServiceError("Token Slack bot requis")
        
        self.client = AsyncWebClient(token=self.bot_token)
        
        # Tester la connexion
        try:
            response = await self.client.auth_test()
            self.logger.info(f"Slack connecté pour l'équipe: {response['team']}")
        except SlackApiError as e:
            raise NotificationServiceError(f"Erreur d'authentification Slack: {e}")
    
    @circuit(failure_threshold=5, recovery_timeout=60)
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def send_notification(
        self,
        notification: NotificationCreateSchema,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Envoyer une notification Slack avec retry et circuit breaker"""
        
        if not self.client:
            raise NotificationServiceError("Client Slack non initialisé")
        
        # Préparer le message
        message_data = await self._prepare_slack_message(notification, context)
        
        try:
            # Vérifier les limites de taux
            rate_limit_key = f"slack_rate_limit:{notification.recipients[0].id}"
            if not await self._check_rate_limit(rate_limit_key, 1, 1):  # 1 message par seconde
                raise NotificationServiceError("Limite de taux Slack dépassée")
            
            # Envoyer le message
            if message_data.get('channel', '').startswith('@'):
                # Message direct
                user_id = message_data['channel'][1:]  # Retirer le @
                response = await self.client.chat_postMessage(
                    channel=user_id,
                    **message_data
                )
            else:
                # Canal public/privé
                response = await self.client.chat_postMessage(**message_data)
            
            # Traiter la réponse
            if response['ok']:
                result = {
                    'success': True,
                    'external_id': response['ts'],  # Timestamp du message
                    'channel': response['channel'],
                    'message_ts': response['ts'],
                    'response': response.data
                }
                
                # Ajouter des réactions pour les priorités élevées
                if notification.priority in [NotificationPriorityEnum.CRITICAL, NotificationPriorityEnum.EMERGENCY]:
                    await self._add_priority_reactions(response['channel'], response['ts'])
                
                # Programmer un rappel si nécessaire
                if self._should_schedule_reminder(notification):
                    await self._schedule_reminder(response['channel'], response['ts'], notification)
                
                return result
            else:
                raise NotificationServiceError(f"Erreur Slack: {response.get('error', 'Erreur inconnue')}")
        
        except SlackApiError as e:
            self.logger.error(f"Erreur API Slack: {e}")
            raise NotificationServiceError(f"Erreur Slack API: {e}")
        except Exception as e:
            self.logger.error(f"Erreur lors de l'envoi Slack: {e}")
            raise NotificationServiceError(f"Erreur envoi Slack: {e}")
    
    async def _prepare_slack_message(
        self,
        notification: NotificationCreateSchema,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Préparer le message Slack formaté"""
        
        recipient = notification.recipients[0]  # Premier destinataire
        channel_config = notification.channels[0]  # Premier canal
        
        # Déterminer le canal de destination
        channel = channel_config.slack_channel or recipient.slack_user_id or recipient.id
        
        # Préparer les données de base
        message_data = {
            'channel': channel,
            'username': self.config.get('bot_name', 'Spotify AI Agent'),
            'icon_emoji': self._get_priority_emoji(notification.priority),
        }
        
        # Utiliser les blocs Slack si configurés
        if channel_config.slack_blocks:
            # Blocs personnalisés
            blocks = await self._render_slack_blocks(channel_config.slack_blocks, notification)
            message_data['blocks'] = blocks
        else:
            # Template par défaut
            blocks = await self._render_slack_blocks(self.default_blocks_template, notification)
            message_data['blocks'] = blocks
            
            # Fallback text pour les notifications
            message_data['text'] = f"{notification.title}: {notification.message}"
        
        # Thread si spécifié
        if channel_config.slack_thread_ts:
            message_data['thread_ts'] = channel_config.slack_thread_ts
        
        # Autres options
        message_data['unfurl_links'] = self.config.get('unfurl_links', True)
        message_data['unfurl_media'] = self.config.get('unfurl_media', True)
        
        return message_data
    
    async def _render_slack_blocks(
        self,
        blocks_template: List[Dict],
        notification: NotificationCreateSchema
    ) -> List[Dict]:
        """Rendre les blocs Slack avec les données de notification"""
        
        # Données pour le rendu
        template_data = {
            'title': notification.title,
            'message': notification.message,
            'priority': notification.priority.value.upper(),
            'source_system': notification.source_system or 'Unknown',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'correlation_id': notification.correlation_id,
            **notification.template_data
        }
        
        rendered_blocks = []
        
        for block in blocks_template:
            # Sérialiser en JSON pour pouvoir utiliser Jinja2
            block_json = json.dumps(block)
            
            # Rendre avec Jinja2
            template = Template(block_json)
            rendered_json = template.render(**template_data)
            
            # Désérialiser
            rendered_block = json.loads(rendered_json)
            rendered_blocks.append(rendered_block)
        
        # Ajouter des blocs dynamiques selon la priorité
        if notification.priority in [NotificationPriorityEnum.CRITICAL, NotificationPriorityEnum.EMERGENCY]:
            rendered_blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"⚠️ *ATTENTION: Notification {notification.priority.value.upper()}*"
                }
            })
        
        # Ajouter des actions si configurées
        if notification.metadata.get('actions'):
            action_block = {
                "type": "actions",
                "elements": []
            }
            
            for action in notification.metadata['actions']:
                action_block['elements'].append({
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": action['label']
                    },
                    "value": action['value'],
                    "action_id": action.get('action_id', action['value'])
                })
            
            rendered_blocks.append(action_block)
        
        return rendered_blocks
    
    def _get_priority_emoji(self, priority: NotificationPriorityEnum) -> str:
        """Obtenir l'emoji selon la priorité"""
        emoji_map = {
            NotificationPriorityEnum.LOW: ':information_source:',
            NotificationPriorityEnum.NORMAL: ':bell:',
            NotificationPriorityEnum.HIGH: ':warning:',
            NotificationPriorityEnum.CRITICAL: ':rotating_light:',
            NotificationPriorityEnum.EMERGENCY: ':fire:'
        }
        return emoji_map.get(priority, ':bell:')
    
    async def _add_priority_reactions(self, channel: str, message_ts: str):
        """Ajouter des réactions pour les messages prioritaires"""
        reactions = ['warning', 'exclamation']
        
        for reaction in reactions:
            try:
                await self.client.reactions_add(
                    channel=channel,
                    timestamp=message_ts,
                    name=reaction
                )
            except SlackApiError as e:
                self.logger.warning(f"Impossible d'ajouter la réaction {reaction}: {e}")
    
    def _should_schedule_reminder(self, notification: NotificationCreateSchema) -> bool:
        """Déterminer si un rappel doit être programmé"""
        return (
            notification.priority in [NotificationPriorityEnum.CRITICAL, NotificationPriorityEnum.EMERGENCY] and
            notification.metadata.get('reminder_enabled', False)
        )
    
    async def _schedule_reminder(self, channel: str, original_ts: str, notification: NotificationCreateSchema):
        """Programmer un rappel pour les notifications critiques"""
        reminder_delay = notification.metadata.get('reminder_delay_minutes', 15)
        
        # Utiliser Redis pour programmer le rappel
        reminder_data = {
            'channel': channel,
            'original_ts': original_ts,
            'notification_id': str(notification.correlation_id or 'unknown'),
            'reminder_text': f"⏰ Rappel: {notification.title}"
        }
        
        await self.redis_client.setex(
            f"slack_reminder:{original_ts}",
            reminder_delay * 60,
            json.dumps(reminder_data)
        )


class EmailNotificationService(BaseNotificationService):
    """Service avancé pour les notifications email"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.smtp_host = config.get('smtp_host', 'localhost')
        self.smtp_port = config.get('smtp_port', 587)
        self.smtp_username = config.get('smtp_username')
        self.smtp_password = config.get('smtp_password')
        self.use_tls = config.get('use_tls', True)
        self.from_email = config.get('from_email', 'noreply@spotify-ai-agent.com')
        self.from_name = config.get('from_name', 'Spotify AI Agent')
    
    @circuit(failure_threshold=3, recovery_timeout=120)
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def send_notification(
        self,
        notification: NotificationCreateSchema,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Envoyer une notification email"""
        
        recipient = notification.recipients[0]
        channel_config = notification.channels[0]
        
        if not recipient.email:
            raise NotificationServiceError("Adresse email manquante pour le destinataire")
        
        # Préparer le message email
        email_data = await self._prepare_email_message(notification, recipient, channel_config)
        
        try:
            # Envoyer via SMTP asynchrone
            await aiosmtplib.send(
                email_data['message'],
                hostname=self.smtp_host,
                port=self.smtp_port,
                username=self.smtp_username,
                password=self.smtp_password,
                use_tls=self.use_tls
            )
            
            return {
                'success': True,
                'external_id': email_data['message_id'],
                'recipient_email': recipient.email,
                'subject': email_data['subject']
            }
        
        except Exception as e:
            self.logger.error(f"Erreur lors de l'envoi email: {e}")
            raise NotificationServiceError(f"Erreur envoi email: {e}")
    
    async def _prepare_email_message(
        self,
        notification: NotificationCreateSchema,
        recipient: RecipientSchema,
        channel_config: ChannelConfigSchema
    ) -> Dict[str, Any]:
        """Préparer le message email formaté"""
        
        # Créer le message multipart
        message = MIMEMultipart('alternative')
        
        # Headers
        message['From'] = f"{self.from_name} <{self.from_email}>"
        message['To'] = recipient.email
        message['Subject'] = notification.title
        message['Message-ID'] = f"<{uuid4()}@{self.from_email.split('@')[1]}>"
        
        # Reply-To si configuré
        if channel_config.email_reply_to:
            message['Reply-To'] = channel_config.email_reply_to
        
        # Corps du message en texte plain
        text_body = self._render_text_template(notification)
        text_part = MIMEText(text_body, 'plain', 'utf-8')
        message.attach(text_part)
        
        # Corps du message en HTML
        html_body = self._render_html_template(notification)
        html_part = MIMEText(html_body, 'html', 'utf-8')
        message.attach(html_part)
        
        # Pièces jointes si configurées
        if channel_config.email_attachments:
            for attachment_path in channel_config.email_attachments:
                await self._add_attachment(message, attachment_path)
        
        # Headers personnalisés
        priority_headers = self._get_priority_headers(notification.priority)
        for header, value in priority_headers.items():
            message[header] = value
        
        return {
            'message': message,
            'message_id': message['Message-ID'],
            'subject': notification.title
        }
    
    def _render_text_template(self, notification: NotificationCreateSchema) -> str:
        """Rendre le template texte"""
        template_data = {
            'title': notification.title,
            'message': notification.message,
            'priority': notification.priority.value.upper(),
            'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
            **notification.template_data
        }
        
        template = Template("""
{{ title }}

{{ message }}

---
Priorité: {{ priority }}
Envoyé le: {{ timestamp }}
Système: Spotify AI Agent

Pour ne plus recevoir ces notifications, contactez votre administrateur.
        """.strip())
        
        return template.render(**template_data)
    
    def _render_html_template(self, notification: NotificationCreateSchema) -> str:
        """Rendre le template HTML"""
        template_data = {
            'title': notification.title,
            'message': notification.message.replace('\n', '<br>'),
            'priority': notification.priority.value.upper(),
            'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
            'priority_color': self._get_priority_color(notification.priority),
            **notification.template_data
        }
        
        template = Template("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background-color: {{ priority_color }}; color: white; padding: 20px; border-radius: 5px 5px 0 0; }
        .content { background-color: #f9f9f9; padding: 20px; border-radius: 0 0 5px 5px; }
        .priority { font-weight: bold; color: {{ priority_color }}; }
        .footer { margin-top: 20px; font-size: 12px; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ title }}</h1>
        </div>
        <div class="content">
            <p>{{ message }}</p>
            <hr>
            <p><strong>Priorité:</strong> <span class="priority">{{ priority }}</span></p>
            <p><strong>Envoyé le:</strong> {{ timestamp }}</p>
        </div>
        <div class="footer">
            <p>Système: Spotify AI Agent</p>
            <p>Pour ne plus recevoir ces notifications, contactez votre administrateur.</p>
        </div>
    </div>
</body>
</html>
        """.strip())
        
        return template.render(**template_data)
    
    def _get_priority_color(self, priority: NotificationPriorityEnum) -> str:
        """Obtenir la couleur selon la priorité"""
        color_map = {
            NotificationPriorityEnum.LOW: '#6c757d',      # Gris
            NotificationPriorityEnum.NORMAL: '#007bff',    # Bleu
            NotificationPriorityEnum.HIGH: '#ffc107',      # Jaune
            NotificationPriorityEnum.CRITICAL: '#fd7e14',  # Orange
            NotificationPriorityEnum.EMERGENCY: '#dc3545' # Rouge
        }
        return color_map.get(priority, '#007bff')
    
    def _get_priority_headers(self, priority: NotificationPriorityEnum) -> Dict[str, str]:
        """Obtenir les headers selon la priorité"""
        headers = {}
        
        if priority in [NotificationPriorityEnum.CRITICAL, NotificationPriorityEnum.EMERGENCY]:
            headers['Importance'] = 'high'
            headers['Priority'] = 'urgent'
            headers['X-MSMail-Priority'] = 'High'
        elif priority == NotificationPriorityEnum.HIGH:
            headers['Importance'] = 'high'
            headers['Priority'] = 'normal'
        elif priority == NotificationPriorityEnum.LOW:
            headers['Importance'] = 'low'
            headers['Priority'] = 'non-urgent'
        
        return headers
    
    async def _add_attachment(self, message: MIMEMultipart, attachment_path: str):
        """Ajouter une pièce jointe au message"""
        try:
            path = Path(attachment_path)
            if not path.exists():
                self.logger.warning(f"Pièce jointe non trouvée: {attachment_path}")
                return
            
            with open(path, 'rb') as f:
                attachment = MIMEBase('application', 'octet-stream')
                attachment.set_payload(f.read())
                encoders.encode_base64(attachment)
                attachment.add_header(
                    'Content-Disposition',
                    f'attachment; filename="{path.name}"'
                )
                message.attach(attachment)
        
        except Exception as e:
            self.logger.error(f"Erreur lors de l'ajout de la pièce jointe: {e}")


class SMSNotificationService(BaseNotificationService):
    """Service pour les notifications SMS via Twilio"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.account_sid = config.get('twilio_account_sid')
        self.auth_token = config.get('twilio_auth_token')
        self.from_phone = config.get('from_phone')
        self.client: Optional[TwilioClient] = None
    
    async def initialize(self):
        """Initialisation du client Twilio"""
        await super().initialize()
        
        if not all([self.account_sid, self.auth_token, self.from_phone]):
            raise NotificationServiceError("Configuration Twilio incomplète")
        
        self.client = TwilioClient(self.account_sid, self.auth_token)
    
    @circuit(failure_threshold=3, recovery_timeout=180)
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def send_notification(
        self,
        notification: NotificationCreateSchema,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Envoyer une notification SMS"""
        
        recipient = notification.recipients[0]
        channel_config = notification.channels[0]
        
        if not recipient.phone:
            raise NotificationServiceError("Numéro de téléphone manquant")
        
        # Préparer le message SMS
        sms_body = self._prepare_sms_message(notification, channel_config)
        
        try:
            # Envoyer le SMS dans un thread pour éviter le blocage
            loop = asyncio.get_event_loop()
            message = await loop.run_in_executor(
                None,
                lambda: self.client.messages.create(
                    body=sms_body,
                    from_=self.from_phone,
                    to=recipient.phone
                )
            )
            
            return {
                'success': True,
                'external_id': message.sid,
                'recipient_phone': recipient.phone,
                'status': message.status
            }
        
        except Exception as e:
            self.logger.error(f"Erreur lors de l'envoi SMS: {e}")
            raise NotificationServiceError(f"Erreur envoi SMS: {e}")
    
    def _prepare_sms_message(
        self,
        notification: NotificationCreateSchema,
        channel_config: ChannelConfigSchema
    ) -> str:
        """Préparer le message SMS (limité à 160 caractères)"""
        
        # Template de base pour SMS
        template_data = {
            'title': notification.title,
            'message': notification.message,
            'priority': notification.priority.value.upper(),
            **notification.template_data
        }
        
        # Format court pour SMS
        if notification.priority in [NotificationPriorityEnum.CRITICAL, NotificationPriorityEnum.EMERGENCY]:
            sms_text = f"🚨 {notification.title}\n{notification.message}"
        else:
            sms_text = f"{notification.title}\n{notification.message}"
        
        # Limiter à 160 caractères
        if len(sms_text) > 160:
            # Tronquer le message en gardant le titre
            max_message_length = 160 - len(notification.title) - 10  # Marge pour "..." et titre
            truncated_message = notification.message[:max_message_length] + "..."
            sms_text = f"{notification.title}\n{truncated_message}"
        
        return sms_text


class PushNotificationService(BaseNotificationService):
    """Service pour les notifications push via Firebase"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.firebase_credentials = config.get('firebase_credentials_path')
        self.firebase_app = None
    
    async def initialize(self):
        """Initialisation de Firebase"""
        await super().initialize()
        
        if not self.firebase_credentials:
            raise NotificationServiceError("Credentials Firebase requis")
        
        try:
            cred = firebase_admin.credentials.Certificate(self.firebase_credentials)
            self.firebase_app = firebase_admin.initialize_app(cred)
        except Exception as e:
            raise NotificationServiceError(f"Erreur initialisation Firebase: {e}")
    
    @circuit(failure_threshold=3, recovery_timeout=120)
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def send_notification(
        self,
        notification: NotificationCreateSchema,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Envoyer une notification push"""
        
        recipient = notification.recipients[0]
        channel_config = notification.channels[0]
        
        # Device token requis pour push
        device_token = recipient.metadata.get('device_token')
        if not device_token:
            raise NotificationServiceError("Device token manquant pour notification push")
        
        # Préparer le message push
        push_message = self._prepare_push_message(notification, channel_config, device_token)
        
        try:
            # Envoyer via Firebase
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: messaging.send(push_message)
            )
            
            return {
                'success': True,
                'external_id': response,
                'device_token': device_token,
                'platform': recipient.metadata.get('platform', 'unknown')
            }
        
        except Exception as e:
            self.logger.error(f"Erreur lors de l'envoi push: {e}")
            raise NotificationServiceError(f"Erreur envoi push: {e}")
    
    def _prepare_push_message(
        self,
        notification: NotificationCreateSchema,
        channel_config: ChannelConfigSchema,
        device_token: str
    ) -> messaging.Message:
        """Préparer le message push Firebase"""
        
        # Configuration de base
        push_notification = messaging.Notification(
            title=notification.title,
            body=notification.message
        )
        
        # Données additionnelles
        data = {
            'notification_id': str(notification.correlation_id or ''),
            'priority': notification.priority.value,
            'source_system': notification.source_system or 'spotify-ai-agent',
            **notification.metadata
        }
        
        # Configuration Android
        android_config = messaging.AndroidConfig(
            ttl=timedelta(hours=1),
            priority='high' if notification.priority in [
                NotificationPriorityEnum.CRITICAL,
                NotificationPriorityEnum.EMERGENCY
            ] else 'normal',
            notification=messaging.AndroidNotification(
                icon='notification_icon',
                color='#1DB954',  # Couleur Spotify
                sound=channel_config.push_sound or 'default',
                click_action='FLUTTER_NOTIFICATION_CLICK'
            )
        )
        
        # Configuration iOS
        apns_config = messaging.APNSConfig(
            payload=messaging.APNSPayload(
                aps=messaging.Aps(
                    badge=channel_config.push_badge_count,
                    sound=channel_config.push_sound or 'default',
                    category=channel_config.push_category
                )
            )
        )
        
        return messaging.Message(
            notification=push_notification,
            data=data,
            token=device_token,
            android=android_config,
            apns=apns_config
        )


class WebhookNotificationService(BaseNotificationService):
    """Service pour les notifications webhook"""
    
    @circuit(failure_threshold=5, recovery_timeout=60)
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def send_notification(
        self,
        notification: NotificationCreateSchema,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Envoyer une notification webhook"""
        
        channel_config = notification.channels[0]
        
        if not channel_config.webhook_url:
            raise NotificationServiceError("URL webhook manquante")
        
        # Préparer les données du webhook
        webhook_data = self._prepare_webhook_payload(notification)
        
        # Headers
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'Spotify-AI-Agent/1.0',
            **channel_config.webhook_headers
        }
        
        # Authentification
        if channel_config.webhook_auth:
            auth_type = channel_config.webhook_auth.get('type', 'bearer')
            if auth_type == 'bearer':
                headers['Authorization'] = f"Bearer {channel_config.webhook_auth['token']}"
            elif auth_type == 'basic':
                import base64
                credentials = f"{channel_config.webhook_auth['username']}:{channel_config.webhook_auth['password']}"
                encoded = base64.b64encode(credentials.encode()).decode()
                headers['Authorization'] = f"Basic {encoded}"
        
        try:
            # Envoyer la requête
            response = await self.http_client.request(
                method=channel_config.webhook_method or 'POST',
                url=str(channel_config.webhook_url),
                json=webhook_data,
                headers=headers,
                timeout=30.0
            )
            
            response.raise_for_status()
            
            return {
                'success': True,
                'external_id': f"webhook_{datetime.now(timezone.utc).timestamp()}",
                'status_code': response.status_code,
                'response_data': response.text[:1000]  # Limiter la taille
            }
        
        except httpx.HTTPStatusError as e:
            self.logger.error(f"Erreur HTTP webhook: {e}")
            raise NotificationServiceError(f"Erreur HTTP webhook: {e.response.status_code}")
        except Exception as e:
            self.logger.error(f"Erreur lors de l'envoi webhook: {e}")
            raise NotificationServiceError(f"Erreur envoi webhook: {e}")
    
    def _prepare_webhook_payload(self, notification: NotificationCreateSchema) -> Dict[str, Any]:
        """Préparer le payload du webhook"""
        
        return {
            'id': str(notification.correlation_id or uuid4()),
            'title': notification.title,
            'message': notification.message,
            'priority': notification.priority.value,
            'recipients': [
                {
                    'id': r.id,
                    'type': r.type,
                    'display_name': r.display_name
                } for r in notification.recipients
            ],
            'metadata': notification.metadata,
            'tags': notification.tags,
            'source_system': notification.source_system,
            'source_event': notification.source_event,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'template_data': notification.template_data
        }
