"""
Slack Alert Formatter - Formatage intelligent des messages d'alerte Slack
Adaptation contextuelle et support multi-format avec templates dynamiques
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

from .slack_alert_manager import AlertData, AlertSeverity, AlertStatus


class MessageStyle(str, Enum):
    """Styles de message"""
    MINIMAL = "minimal"
    STANDARD = "standard"
    DETAILED = "detailed"
    EXECUTIVE = "executive"


class AlertActionType(str, Enum):
    """Types d'actions sur les alertes"""
    ACKNOWLEDGE = "acknowledge"
    RESOLVE = "resolve"
    ESCALATE = "escalate"
    SUPPRESS = "suppress"
    INVESTIGATE = "investigate"


@dataclass
class FormattingContext:
    """Contexte de formatage"""
    alert_data: AlertData
    style: MessageStyle = MessageStyle.STANDARD
    language: str = "fr"
    timezone: str = "Europe/Paris"
    include_actions: bool = True
    include_metadata: bool = True
    max_description_length: int = 500


class SlackAlertFormatter:
    """
    Formateur intelligent d'alertes Slack avec:
    - Adaptation automatique selon la sévérité
    - Support multi-style (minimal, standard, détaillé)
    - Actions interactives contextuelles
    - Formatage optimisé pour mobile
    - Support des threads et réponses
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configuration des couleurs par sévérité
        self.severity_colors = {
            AlertSeverity.INFO: "#36a64f",      # Vert
            AlertSeverity.WARNING: "#ff9500",   # Orange
            AlertSeverity.ERROR: "#ff0000",     # Rouge
            AlertSeverity.CRITICAL: "#8B0000",  # Rouge foncé
            AlertSeverity.EMERGENCY: "#800080"  # Violet
        }
        
        # Emojis par sévérité
        self.severity_emojis = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.ERROR: "❌",
            AlertSeverity.CRITICAL: "🔥",
            AlertSeverity.EMERGENCY: "🚨"
        }
        
        # Configuration des styles
        self.style_configs = {
            MessageStyle.MINIMAL: {
                "max_fields": 2,
                "include_description": False,
                "include_actions": False,
                "compact_layout": True
            },
            MessageStyle.STANDARD: {
                "max_fields": 6,
                "include_description": True,
                "include_actions": True,
                "compact_layout": False
            },
            MessageStyle.DETAILED: {
                "max_fields": 12,
                "include_description": True,
                "include_actions": True,
                "include_metadata": True,
                "compact_layout": False
            },
            MessageStyle.EXECUTIVE: {
                "max_fields": 4,
                "include_description": True,
                "include_actions": False,
                "focus_impact": True
            }
        }

    async def format_alert(
        self,
        alert_data: AlertData,
        template_type: str = "standard",
        language: str = "fr",
        style: MessageStyle = MessageStyle.STANDARD
    ) -> Dict[str, Any]:
        """
        Formate une alerte pour Slack
        
        Args:
            alert_data: Données de l'alerte
            template_type: Type de template
            language: Langue de formatage
            style: Style de message
            
        Returns:
            Message Slack formaté
        """
        try:
            context = FormattingContext(
                alert_data=alert_data,
                style=style,
                language=language
            )
            
            # Sélection du formateur selon la sévérité
            if alert_data.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
                return await self._format_critical_alert(context)
            elif alert_data.severity == AlertSeverity.ERROR:
                return await self._format_error_alert(context)
            elif alert_data.severity == AlertSeverity.WARNING:
                return await self._format_warning_alert(context)
            else:
                return await self._format_info_alert(context)
                
        except Exception as e:
            self.logger.error(f"Erreur lors du formatage de l'alerte: {e}")
            return await self._format_fallback_alert(alert_data)

    async def format_acknowledgment(
        self,
        alert_data: AlertData,
        user_id: str,
        note: Optional[str] = None,
        language: str = "fr"
    ) -> Dict[str, Any]:
        """Formate un message d'acquittement"""
        try:
            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"✅ *Alerte acquittée*\n"
                               f"*Titre:* {alert_data.title}\n"
                               f"*Par:* <@{user_id}>\n"
                               f"*Timestamp:* {self._format_timestamp(datetime.utcnow(), language)}"
                    }
                }
            ]
            
            if note:
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Note:* {note}"
                    }
                })
            
            # Actions de suivi
            blocks.append({
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Résoudre"},
                        "style": "primary",
                        "action_id": "resolve_alert",
                        "value": alert_data.alert_id
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Réactiver"},
                        "style": "danger",
                        "action_id": "reactivate_alert",
                        "value": alert_data.alert_id
                    }
                ]
            })
            
            return {
                "blocks": blocks,
                "thread_ts": alert_data.metadata.get("original_ts"),
                "unfurl_links": False
            }
            
        except Exception as e:
            self.logger.error(f"Erreur lors du formatage de l'acquittement: {e}")
            return {"text": f"✅ Alerte {alert_data.alert_id} acquittée par <@{user_id}>"}

    async def format_resolution(
        self,
        alert_data: AlertData,
        user_id: str,
        resolution_note: str,
        language: str = "fr"
    ) -> Dict[str, Any]:
        """Formate un message de résolution"""
        try:
            duration = datetime.utcnow() - alert_data.created_at
            duration_str = self._format_duration(duration)
            
            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"✅ *Alerte résolue*\n"
                               f"*Titre:* {alert_data.title}\n"
                               f"*Résolu par:* <@{user_id}>\n"
                               f"*Durée:* {duration_str}\n"
                               f"*Timestamp:* {self._format_timestamp(datetime.utcnow(), language)}"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Solution:* {resolution_note}"
                    }
                }
            ]
            
            # Statistiques si disponibles
            if alert_data.ai_insights:
                insights = alert_data.ai_insights
                if "similar_incidents" in insights:
                    blocks.append({
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": f"📊 {len(insights['similar_incidents'])} incidents similaires cette semaine"
                            }
                        ]
                    })
            
            return {
                "blocks": blocks,
                "thread_ts": alert_data.metadata.get("original_ts"),
                "unfurl_links": False
            }
            
        except Exception as e:
            self.logger.error(f"Erreur lors du formatage de la résolution: {e}")
            return {"text": f"✅ Alerte {alert_data.alert_id} résolue par <@{user_id}>: {resolution_note}"}

    async def format_escalation(
        self,
        alert_data: AlertData,
        escalation_level: int,
        target_users: List[str],
        language: str = "fr"
    ) -> Dict[str, Any]:
        """Formate un message d'escalade"""
        try:
            emoji = "🔥" if escalation_level >= 3 else "⬆️"
            urgency = "CRITIQUE" if escalation_level >= 3 else "URGENT"
            
            user_mentions = " ".join([f"<@{user}>" for user in target_users])
            
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{emoji} ESCALADE NIVEAU {escalation_level} - {urgency}"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Alerte:* {alert_data.title}\n"
                               f"*Service:* {alert_data.context.service_name}\n"
                               f"*Sévérité:* {alert_data.severity.value.upper()}\n"
                               f"*Durée:* {self._format_duration(datetime.utcnow() - alert_data.created_at)}\n"
                               f"*Assigné à:* {user_mentions}"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Description:* {alert_data.description[:300]}..."
                    }
                }
            ]
            
            # Actions d'escalade
            blocks.append({
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Prendre en charge"},
                        "style": "primary",
                        "action_id": "take_ownership",
                        "value": alert_data.alert_id
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Escalader encore"},
                        "style": "danger",
                        "action_id": "escalate_further",
                        "value": alert_data.alert_id
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Voir détails"},
                        "action_id": "view_details",
                        "value": alert_data.alert_id
                    }
                ]
            })
            
            return {
                "blocks": blocks,
                "unfurl_links": False
            }
            
        except Exception as e:
            self.logger.error(f"Erreur lors du formatage de l'escalade: {e}")
            return {"text": f"🔥 ESCALADE: Alerte {alert_data.alert_id} nécessite une attention immédiate!"}

    async def format_summary(
        self,
        alerts: List[AlertData],
        period: str = "24h",
        language: str = "fr"
    ) -> Dict[str, Any]:
        """Formate un résumé d'alertes"""
        try:
            total_alerts = len(alerts)
            critical_count = len([a for a in alerts if a.severity == AlertSeverity.CRITICAL])
            active_count = len([a for a in alerts if a.status == AlertStatus.ACTIVE])
            
            # Services les plus touchés
            service_counts = {}
            for alert in alerts:
                service = alert.context.service_name
                service_counts[service] = service_counts.get(service, 0) + 1
            
            top_services = sorted(service_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"📊 Résumé des alertes - {period}"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Total:*\n{total_alerts}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Critiques:*\n{critical_count}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Actives:*\n{active_count}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Résolues:*\n{total_alerts - active_count}"
                        }
                    ]
                }
            ]
            
            if top_services:
                service_text = "\n".join([f"• {service}: {count}" for service, count in top_services])
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Services les plus impactés:*\n{service_text}"
                    }
                })
            
            # Tendances
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"📈 Généré le {self._format_timestamp(datetime.utcnow(), language)}"
                    }
                ]
            })
            
            return {"blocks": blocks}
            
        except Exception as e:
            self.logger.error(f"Erreur lors du formatage du résumé: {e}")
            return {"text": f"📊 Résumé: {len(alerts)} alertes sur {period}"}

    async def _format_critical_alert(self, context: FormattingContext) -> Dict[str, Any]:
        """Formate une alerte critique"""
        alert = context.alert_data
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"🚨 ALERTE CRITIQUE - {alert.severity.value.upper()}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{alert.title}*\n{alert.description[:300]}..."
                },
                "accessory": {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "🚨 URGENT"},
                    "style": "danger",
                    "action_id": "emergency_response"
                }
            }
        ]
        
        # Champs critiques
        fields = [
            {"type": "mrkdwn", "text": f"*Service:*\n{alert.context.service_name}"},
            {"type": "mrkdwn", "text": f"*Environnement:*\n{alert.context.environment}"},
            {"type": "mrkdwn", "text": f"*Composant:*\n{alert.context.component}"},
            {"type": "mrkdwn", "text": f"*Timestamp:*\n{self._format_timestamp(alert.created_at, context.language)}"}
        ]
        
        blocks.append({"type": "section", "fields": fields})
        
        # Actions d'urgence
        if context.include_actions:
            blocks.append({
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "🚨 Prendre en charge"},
                        "style": "danger",
                        "action_id": "emergency_ack",
                        "value": alert.alert_id
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "📞 Appeler équipe"},
                        "style": "primary",
                        "action_id": "call_team",
                        "value": alert.alert_id
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "📊 Voir métriques"},
                        "action_id": "view_metrics",
                        "value": alert.alert_id
                    }
                ]
            })
        
        return {
            "blocks": blocks,
            "unfurl_links": False,
            "metadata": {
                "event_type": "critical_alert",
                "alert_id": alert.alert_id
            }
        }

    async def _format_error_alert(self, context: FormattingContext) -> Dict[str, Any]:
        """Formate une alerte d'erreur"""
        alert = context.alert_data
        
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"❌ *{alert.title}*\n{alert.description[:200]}..."
                },
                "accessory": {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Examiner"},
                    "style": "primary",
                    "action_id": "investigate_error"
                }
            }
        ]
        
        # Informations essentielles
        fields = [
            {"type": "mrkdwn", "text": f"*Service:*\n{alert.context.service_name}"},
            {"type": "mrkdwn", "text": f"*Sévérité:*\n{alert.severity.value}"},
            {"type": "mrkdwn", "text": f"*Environnement:*\n{alert.context.environment}"},
            {"type": "mrkdwn", "text": f"*Timestamp:*\n{self._format_timestamp(alert.created_at, context.language)}"}
        ]
        
        blocks.append({"type": "section", "fields": fields})
        
        # Actions standard
        if context.include_actions:
            blocks.append({
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Acquitter"},
                        "style": "primary",
                        "action_id": "acknowledge_alert",
                        "value": alert.alert_id
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Résoudre"},
                        "action_id": "resolve_alert",
                        "value": alert.alert_id
                    }
                ]
            })
        
        return {"blocks": blocks, "unfurl_links": False}

    async def _format_warning_alert(self, context: FormattingContext) -> Dict[str, Any]:
        """Formate une alerte d'avertissement"""
        alert = context.alert_data
        
        text = f"⚠️ *{alert.title}*\n"
        text += f"Service: {alert.context.service_name} | "
        text += f"Env: {alert.context.environment} | "
        text += f"{self._format_timestamp(alert.created_at, context.language)}"
        
        if context.style == MessageStyle.DETAILED:
            text += f"\n{alert.description}"
        
        message = {"text": text}
        
        if context.include_actions:
            message["attachments"] = [{
                "color": self.severity_colors[alert.severity],
                "actions": [
                    {
                        "name": "acknowledge",
                        "text": "Acquitter",
                        "type": "button",
                        "value": alert.alert_id
                    }
                ]
            }]
        
        return message

    async def _format_info_alert(self, context: FormattingContext) -> Dict[str, Any]:
        """Formate une alerte informative"""
        alert = context.alert_data
        
        if context.style == MessageStyle.MINIMAL:
            return {
                "text": f"ℹ️ {alert.title} | {alert.context.service_name}"
            }
        
        return {
            "text": f"ℹ️ *{alert.title}*\n"
                   f"Service: {alert.context.service_name}\n"
                   f"{alert.description[:150]}...",
            "unfurl_links": False
        }

    async def _format_fallback_alert(self, alert_data: AlertData) -> Dict[str, Any]:
        """Formatage de fallback en cas d'erreur"""
        emoji = self.severity_emojis.get(alert_data.severity, "📌")
        
        return {
            "text": f"{emoji} *{alert_data.title}*\n"
                   f"Service: {alert_data.context.service_name}\n"
                   f"Sévérité: {alert_data.severity.value}\n"
                   f"ID: {alert_data.alert_id}"
        }

    def _format_timestamp(self, dt: datetime, language: str) -> str:
        """Formate un timestamp selon la langue"""
        if language == "fr":
            return dt.strftime("%d/%m/%Y à %H:%M")
        elif language == "de":
            return dt.strftime("%d.%m.%Y um %H:%M")
        else:  # en
            return dt.strftime("%Y-%m-%d at %H:%M")

    def _format_duration(self, delta: timedelta) -> str:
        """Formate une durée"""
        total_seconds = int(delta.total_seconds())
        
        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes}m {seconds}s"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours}h {minutes}m"
