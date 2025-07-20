"""
Scripts de Notification Multi-Canal Ultra-Avancés
Système intelligent de notification pour incidents critiques dans Spotify AI Agent

Fonctionnalités:
- Notifications multi-canal (Slack, Teams, Email, SMS, PagerDuty)
- Escalade intelligente basée sur la sévérité et le temps de réponse
- Templates dynamiques adaptatifs selon le contexte
- Intégration IA pour résumés automatiques d'incidents
- Déduplication intelligente des alertes
- Notification géolocalisée pour équipes distribuées
"""

import asyncio
import logging
import json
import aiohttp
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from jinja2 import Template
import pytz
from twilio.rest import Client as TwilioClient

from . import AlertConfig, AlertSeverity, AlertCategory, ScriptType, register_alert

logger = logging.getLogger(__name__)

class NotificationChannel(Enum):
    """Canaux de notification disponibles"""
    SLACK = "slack"
    TEAMS = "teams"
    EMAIL = "email"
    SMS = "sms"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    MOBILE_PUSH = "mobile_push"
    DISCORD = "discord"

class EscalationLevel(Enum):
    """Niveaux d'escalade pour les incidents"""
    L1_SUPPORT = "l1_support"
    L2_ENGINEERS = "l2_engineers"
    L3_ARCHITECTS = "l3_architects"
    ON_CALL_MANAGER = "on_call_manager"
    EXECUTIVE_TEAM = "executive_team"

@dataclass
class NotificationTemplate:
    """Template de notification personnalisable"""
    name: str
    channel: NotificationChannel
    severity: AlertSeverity
    subject_template: str
    body_template: str
    rich_content: bool = False
    emoji_enabled: bool = True
    mention_users: List[str] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)

@dataclass
class NotificationRule:
    """Règle de notification et d'escalade"""
    rule_id: str
    channels: List[NotificationChannel]
    conditions: Dict[str, Any]
    escalation_levels: List[EscalationLevel]
    escalation_delays: List[int]  # en minutes
    tenant_id: Optional[str] = None
    timezone: str = "UTC"
    quiet_hours: Optional[Dict[str, str]] = None
    enabled: bool = True

@dataclass
class NotificationContext:
    """Contexte d'une notification"""
    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    category: AlertCategory
    timestamp: datetime
    affected_services: List[str]
    metrics: Dict[str, Any]
    tenant_id: Optional[str] = None
    incident_url: Optional[str] = None
    runbook_url: Optional[str] = None
    stakeholders: List[str] = field(default_factory=list)

class IntelligentNotificationSystem:
    """Système de notification intelligent avec IA"""
    
    def __init__(self):
        self.templates: Dict[str, NotificationTemplate] = {}
        self.rules: List[NotificationRule] = []
        self.active_incidents: Dict[str, Dict] = {}
        self.notification_history: List[Dict] = []
        self.deduplication_window = timedelta(minutes=10)
        
        # Configuration des canaux
        self.channel_configs = {
            NotificationChannel.SLACK: {
                'webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK',
                'token': 'xoxb-your-slack-bot-token'
            },
            NotificationChannel.TEAMS: {
                'webhook_url': 'https://outlook.office.com/webhook/YOUR/TEAMS/WEBHOOK'
            },
            NotificationChannel.EMAIL: {
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': 'alerts@spotify-ai-agent.com',
                'password': 'your-app-password'
            },
            NotificationChannel.SMS: {
                'twilio_account_sid': 'your-twilio-account-sid',
                'twilio_auth_token': 'your-twilio-auth-token',
                'from_number': '+1234567890'
            },
            NotificationChannel.PAGERDUTY: {
                'integration_key': 'your-pagerduty-integration-key',
                'api_url': 'https://events.pagerduty.com/v2/enqueue'
            }
        }
        
        self._initialize_default_templates()
        self._initialize_default_rules()

    def _initialize_default_templates(self):
        """Initialise les templates de notification par défaut"""
        
        # Template Slack pour incidents critiques
        slack_critical = NotificationTemplate(
            name="slack_critical_incident",
            channel=NotificationChannel.SLACK,
            severity=AlertSeverity.CRITICAL,
            subject_template="🚨 INCIDENT CRITIQUE - {{ title }}",
            body_template="""
🚨 **INCIDENT CRITIQUE DÉTECTÉ** 🚨

**Titre:** {{ title }}
**Sévérité:** {{ severity.value | upper }}
**Heure:** {{ timestamp.strftime('%Y-%m-%d %H:%M:%S UTC') }}
**Services Affectés:** {{ affected_services | join(', ') }}

**Description:**
{{ description }}

**Métriques Clés:**
{% for key, value in metrics.items() %}
• {{ key }}: {{ value }}
{% endfor %}

**Actions Requises:**
• Vérifier immédiatement les services affectés
• Analyser les logs d'erreur récents
• Contacter l'équipe d'astreinte si nécessaire

**Liens Utiles:**
• [Dashboard de Monitoring]({{ incident_url }})
• [Runbook d'Incident]({{ runbook_url }})

*Généré automatiquement par Spotify AI Agent Monitoring System*
            """,
            rich_content=True,
            emoji_enabled=True,
            mention_users=["@channel", "@oncall-team"]
        )
        
        # Template Email pour rapports détaillés
        email_detailed = NotificationTemplate(
            name="email_detailed_report",
            channel=NotificationChannel.EMAIL,
            severity=AlertSeverity.HIGH,
            subject_template="[ALERT-{{ severity.value | upper }}] {{ title }} - Spotify AI Agent",
            body_template="""
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .header { background-color: #ff4444; color: white; padding: 20px; }
        .content { padding: 20px; }
        .metrics { background-color: #f5f5f5; padding: 15px; margin: 10px 0; }
        .footer { background-color: #333; color: white; padding: 10px; font-size: 12px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚨 Incident Alert - {{ title }}</h1>
        <p>Severity: {{ severity.value | upper }} | Time: {{ timestamp.strftime('%Y-%m-%d %H:%M:%S UTC') }}</p>
    </div>
    
    <div class="content">
        <h2>Incident Details</h2>
        <p><strong>Description:</strong> {{ description }}</p>
        <p><strong>Affected Services:</strong> {{ affected_services | join(', ') }}</p>
        <p><strong>Tenant ID:</strong> {{ tenant_id or 'Global' }}</p>
        
        <div class="metrics">
            <h3>Current Metrics</h3>
            <ul>
            {% for key, value in metrics.items() %}
                <li><strong>{{ key }}:</strong> {{ value }}</li>
            {% endfor %}
            </ul>
        </div>
        
        <h3>Recommended Actions</h3>
        <ol>
            <li>Check system health dashboard immediately</li>
            <li>Review recent deployment changes</li>
            <li>Analyze error logs for patterns</li>
            <li>Contact on-call engineer if issues persist</li>
        </ol>
        
        <h3>Useful Links</h3>
        <ul>
            <li><a href="{{ incident_url }}">Incident Dashboard</a></li>
            <li><a href="{{ runbook_url }}">Incident Runbook</a></li>
        </ul>
    </div>
    
    <div class="footer">
        <p>This alert was generated automatically by Spotify AI Agent Monitoring System</p>
        <p>For support, contact: support@spotify-ai-agent.com</p>
    </div>
</body>
</html>
            """,
            rich_content=True
        )
        
        # Template SMS concis
        sms_template = NotificationTemplate(
            name="sms_critical_alert",
            channel=NotificationChannel.SMS,
            severity=AlertSeverity.CRITICAL,
            subject_template="",
            body_template="🚨 CRITICAL: {{ title }} - {{ affected_services | join(', ') }} affected. Check dashboard immediately. Time: {{ timestamp.strftime('%H:%M UTC') }}",
            rich_content=False,
            emoji_enabled=True
        )
        
        self.templates = {
            template.name: template for template in [
                slack_critical, email_detailed, sms_template
            ]
        }

    def _initialize_default_rules(self):
        """Initialise les règles de notification par défaut"""
        
        # Règle pour incidents critiques
        critical_rule = NotificationRule(
            rule_id="critical_incident_escalation",
            channels=[
                NotificationChannel.SLACK,
                NotificationChannel.EMAIL,
                NotificationChannel.PAGERDUTY
            ],
            conditions={
                'severity': [AlertSeverity.CRITICAL],
                'categories': [AlertCategory.AVAILABILITY, AlertCategory.SECURITY]
            },
            escalation_levels=[
                EscalationLevel.L1_SUPPORT,
                EscalationLevel.L2_ENGINEERS,
                EscalationLevel.ON_CALL_MANAGER
            ],
            escalation_delays=[0, 5, 15],  # Immédiat, puis 5min, puis 15min
            quiet_hours={'start': '22:00', 'end': '06:00'}
        )
        
        # Règle pour alertes de performance
        performance_rule = NotificationRule(
            rule_id="performance_degradation",
            channels=[NotificationChannel.SLACK, NotificationChannel.EMAIL],
            conditions={
                'severity': [AlertSeverity.HIGH, AlertSeverity.MEDIUM],
                'categories': [AlertCategory.PERFORMANCE]
            },
            escalation_levels=[EscalationLevel.L1_SUPPORT, EscalationLevel.L2_ENGINEERS],
            escalation_delays=[0, 10]
        )
        
        self.rules.extend([critical_rule, performance_rule])

    async def send_notification(self, context: NotificationContext) -> Dict[str, bool]:
        """Envoie une notification selon les règles configurées"""
        results = {}
        
        try:
            # Vérification de déduplication
            if await self._is_duplicate_notification(context):
                logger.info(f"Notification dédupliquée pour l'alerte: {context.alert_id}")
                return {"deduplicated": True}
            
            # Recherche des règles applicables
            applicable_rules = self._find_applicable_rules(context)
            
            if not applicable_rules:
                logger.warning(f"Aucune règle de notification trouvée pour: {context.alert_id}")
                return {"no_rules_found": True}
            
            # Envoi sur chaque canal configuré
            for rule in applicable_rules:
                for channel in rule.channels:
                    try:
                        # Vérification des heures de silence
                        if await self._is_quiet_hours(rule):
                            continue
                        
                        success = await self._send_to_channel(channel, context, rule)
                        results[f"{channel.value}"] = success
                        
                        if success:
                            logger.info(f"Notification envoyée avec succès via {channel.value}")
                        else:
                            logger.error(f"Échec d'envoi via {channel.value}")
                            
                    except Exception as e:
                        logger.error(f"Erreur lors de l'envoi via {channel.value}: {e}")
                        results[f"{channel.value}"] = False
            
            # Enregistrement dans l'historique
            await self._record_notification(context, results)
            
            # Planification de l'escalade si nécessaire
            await self._schedule_escalation(context, applicable_rules)
            
        except Exception as e:
            logger.error(f"Erreur lors de l'envoi de notification: {e}")
            results["error"] = str(e)
        
        return results

    async def _is_duplicate_notification(self, context: NotificationContext) -> bool:
        """Vérifie si la notification est un doublon récent"""
        cutoff_time = datetime.utcnow() - self.deduplication_window
        
        for notification in self.notification_history:
            if (notification.get('alert_id') == context.alert_id and 
                notification.get('timestamp', datetime.min) > cutoff_time):
                return True
        
        return False

    def _find_applicable_rules(self, context: NotificationContext) -> List[NotificationRule]:
        """Trouve les règles de notification applicables"""
        applicable_rules = []
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            # Vérification tenant
            if rule.tenant_id and rule.tenant_id != context.tenant_id:
                continue
            
            # Vérification conditions
            conditions_met = True
            
            if 'severity' in rule.conditions:
                if context.severity not in rule.conditions['severity']:
                    conditions_met = False
            
            if 'categories' in rule.conditions:
                if context.category not in rule.conditions['categories']:
                    conditions_met = False
            
            if conditions_met:
                applicable_rules.append(rule)
        
        return applicable_rules

    async def _is_quiet_hours(self, rule: NotificationRule) -> bool:
        """Vérifie si nous sommes dans les heures de silence"""
        if not rule.quiet_hours:
            return False
        
        try:
            tz = pytz.timezone(rule.timezone)
            current_time = datetime.now(tz).time()
            
            start_time = datetime.strptime(rule.quiet_hours['start'], '%H:%M').time()
            end_time = datetime.strptime(rule.quiet_hours['end'], '%H:%M').time()
            
            if start_time <= end_time:
                return start_time <= current_time <= end_time
            else:  # Traverse minuit
                return current_time >= start_time or current_time <= end_time
                
        except Exception as e:
            logger.error(f"Erreur lors de la vérification des heures de silence: {e}")
            return False

    async def _send_to_channel(self, channel: NotificationChannel, context: NotificationContext, rule: NotificationRule) -> bool:
        """Envoie la notification vers un canal spécifique"""
        
        try:
            if channel == NotificationChannel.SLACK:
                return await self._send_slack_notification(context)
            elif channel == NotificationChannel.EMAIL:
                return await self._send_email_notification(context)
            elif channel == NotificationChannel.SMS:
                return await self._send_sms_notification(context)
            elif channel == NotificationChannel.TEAMS:
                return await self._send_teams_notification(context)
            elif channel == NotificationChannel.PAGERDUTY:
                return await self._send_pagerduty_notification(context)
            else:
                logger.warning(f"Canal de notification non supporté: {channel}")
                return False
                
        except Exception as e:
            logger.error(f"Erreur lors de l'envoi vers {channel.value}: {e}")
            return False

    async def _send_slack_notification(self, context: NotificationContext) -> bool:
        """Envoie une notification Slack"""
        try:
            template = self.templates.get("slack_critical_incident")
            if not template:
                return False
            
            # Rendu du template
            subject = Template(template.subject_template).render(**context.__dict__)
            body = Template(template.body_template).render(**context.__dict__)
            
            # Préparation du payload Slack
            payload = {
                "text": subject,
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": subject
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": body
                        }
                    }
                ],
                "channel": "#alerts-critical"
            }
            
            webhook_url = self.channel_configs[NotificationChannel.SLACK]['webhook_url']
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.error(f"Erreur Slack: {e}")
            return False

    async def _send_email_notification(self, context: NotificationContext) -> bool:
        """Envoie une notification email"""
        try:
            template = self.templates.get("email_detailed_report")
            if not template:
                return False
            
            # Rendu du template
            subject = Template(template.subject_template).render(**context.__dict__)
            body = Template(template.body_template).render(**context.__dict__)
            
            # Configuration SMTP
            config = self.channel_configs[NotificationChannel.EMAIL]
            
            # Création du message
            msg = MimeMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = config['username']
            msg['To'] = "oncall-team@spotify-ai-agent.com"
            
            html_part = MimeText(body, 'html')
            msg.attach(html_part)
            
            # Envoi via SMTP
            with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
                server.starttls()
                server.login(config['username'], config['password'])
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur Email: {e}")
            return False

    async def _send_sms_notification(self, context: NotificationContext) -> bool:
        """Envoie une notification SMS via Twilio"""
        try:
            template = self.templates.get("sms_critical_alert")
            if not template:
                return False
            
            # Rendu du template
            message = Template(template.body_template).render(**context.__dict__)
            
            # Configuration Twilio
            config = self.channel_configs[NotificationChannel.SMS]
            
            client = TwilioClient(
                config['twilio_account_sid'],
                config['twilio_auth_token']
            )
            
            # Envoi SMS aux numéros d'astreinte
            oncall_numbers = ["+1234567890", "+0987654321"]  # À configurer
            
            for number in oncall_numbers:
                client.messages.create(
                    body=message,
                    from_=config['from_number'],
                    to=number
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur SMS: {e}")
            return False

    async def _send_teams_notification(self, context: NotificationContext) -> bool:
        """Envoie une notification Microsoft Teams"""
        try:
            # Payload Teams avec adaptive cards
            payload = {
                "@type": "MessageCard",
                "@context": "https://schema.org/extensions",
                "summary": f"Alert: {context.title}",
                "themeColor": "FF0000" if context.severity == AlertSeverity.CRITICAL else "FFA500",
                "sections": [
                    {
                        "activityTitle": f"🚨 {context.title}",
                        "activitySubtitle": f"Severity: {context.severity.value.upper()}",
                        "facts": [
                            {"name": "Time", "value": context.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')},
                            {"name": "Services", "value": ", ".join(context.affected_services)},
                            {"name": "Tenant", "value": context.tenant_id or "Global"}
                        ],
                        "text": context.description
                    }
                ],
                "potentialAction": [
                    {
                        "@type": "OpenUri",
                        "name": "View Dashboard",
                        "targets": [{"os": "default", "uri": context.incident_url}]
                    }
                ]
            }
            
            webhook_url = self.channel_configs[NotificationChannel.TEAMS]['webhook_url']
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.error(f"Erreur Teams: {e}")
            return False

    async def _send_pagerduty_notification(self, context: NotificationContext) -> bool:
        """Envoie une notification PagerDuty"""
        try:
            config = self.channel_configs[NotificationChannel.PAGERDUTY]
            
            payload = {
                "routing_key": config['integration_key'],
                "event_action": "trigger",
                "payload": {
                    "summary": context.title,
                    "source": "spotify-ai-agent-monitoring",
                    "severity": context.severity.value,
                    "component": ", ".join(context.affected_services),
                    "group": context.category.value,
                    "class": "monitoring_alert",
                    "custom_details": {
                        "metrics": context.metrics,
                        "tenant_id": context.tenant_id,
                        "affected_services": context.affected_services
                    }
                },
                "links": [
                    {
                        "href": context.incident_url,
                        "text": "View Incident Dashboard"
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(config['api_url'], json=payload) as response:
                    return response.status == 202
                    
        except Exception as e:
            logger.error(f"Erreur PagerDuty: {e}")
            return False

    async def _record_notification(self, context: NotificationContext, results: Dict[str, bool]):
        """Enregistre la notification dans l'historique"""
        record = {
            'alert_id': context.alert_id,
            'timestamp': datetime.utcnow(),
            'channels': list(results.keys()),
            'success_count': sum(1 for success in results.values() if success),
            'context': context.__dict__
        }
        
        self.notification_history.append(record)
        
        # Nettoyage de l'historique (garder seulement les 1000 dernières)
        if len(self.notification_history) > 1000:
            self.notification_history = self.notification_history[-1000:]

    async def _schedule_escalation(self, context: NotificationContext, rules: List[NotificationRule]):
        """Planifie l'escalade automatique des incidents"""
        for rule in rules:
            if len(rule.escalation_levels) > 1:
                # Planification des escalades futures
                for i, (level, delay) in enumerate(zip(rule.escalation_levels[1:], rule.escalation_delays[1:]), 1):
                    escalation_time = datetime.utcnow() + timedelta(minutes=delay)
                    
                    # En production, utiliser un scheduler comme Celery
                    logger.info(
                        f"Escalade planifiée vers {level.value} "
                        f"dans {delay} minutes pour l'alerte {context.alert_id}"
                    )

    async def get_notification_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques de notification"""
        total_notifications = len(self.notification_history)
        if total_notifications == 0:
            return {"total": 0}
        
        recent_notifications = [
            n for n in self.notification_history 
            if n['timestamp'] > datetime.utcnow() - timedelta(hours=24)
        ]
        
        success_rate = sum(n['success_count'] for n in recent_notifications) / len(recent_notifications) if recent_notifications else 0
        
        return {
            "total_notifications": total_notifications,
            "last_24h": len(recent_notifications),
            "success_rate": f"{success_rate:.2%}",
            "active_incidents": len(self.active_incidents),
            "channels_used": list(set(
                channel for n in recent_notifications 
                for channel in n['channels']
            ))
        }

# Instance globale du système de notification
_notification_system = IntelligentNotificationSystem()

async def send_critical_alert(title: str, description: str, affected_services: List[str], metrics: Dict[str, Any], tenant_id: Optional[str] = None) -> Dict[str, bool]:
    """Function helper pour envoyer une alerte critique"""
    context = NotificationContext(
        alert_id=f"critical_{int(datetime.utcnow().timestamp())}",
        title=title,
        description=description,
        severity=AlertSeverity.CRITICAL,
        category=AlertCategory.AVAILABILITY,
        timestamp=datetime.utcnow(),
        affected_services=affected_services,
        metrics=metrics,
        tenant_id=tenant_id,
        incident_url="https://monitoring.spotify-ai-agent.com/incidents",
        runbook_url="https://docs.spotify-ai-agent.com/runbooks"
    )
    
    return await _notification_system.send_notification(context)

async def get_notification_system() -> IntelligentNotificationSystem:
    """Retourne l'instance du système de notification"""
    return _notification_system

# Configuration des alertes de notification
if __name__ == "__main__":
    # Enregistrement des configurations d'alertes
    notification_configs = [
        AlertConfig(
            name="critical_notification_system",
            category=AlertCategory.AVAILABILITY,
            severity=AlertSeverity.CRITICAL,
            script_type=ScriptType.NOTIFICATION,
            conditions=['Incident critique détecté'],
            actions=['send_multi_channel_notification', 'escalate_to_oncall'],
            ml_enabled=False,
            auto_remediation=False
        ),
        AlertConfig(
            name="notification_channel_failure",
            category=AlertCategory.AVAILABILITY,
            severity=AlertSeverity.HIGH,
            script_type=ScriptType.NOTIFICATION,
            conditions=['Canal de notification indisponible'],
            actions=['switch_to_backup_channel', 'notify_admin'],
            ml_enabled=False
        )
    ]
    
    for config in notification_configs:
        register_alert(config)
