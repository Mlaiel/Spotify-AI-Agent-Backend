"""
Formateur de Messages Slack Ultra-Avancé
========================================

Module de formatage intelligent des messages Slack avec support complet
des Blocks Kit, attachements legacy, et formatage contextuel avancé.

Développé par l'équipe Backend Senior sous la direction de Fahed Mlaiel.
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import html
import urllib.parse
from pathlib import Path

from . import SlackSeverity, SlackChannelType
from .utils import SlackUtils

logger = logging.getLogger(__name__)

class MessageFormat(Enum):
    """Formats de message supportés."""
    BLOCKS = "blocks"
    ATTACHMENTS = "attachments"
    TEXT = "text"
    MIXED = "mixed"

class MessageStyle(Enum):
    """Styles de message."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    DETAILED = "detailed"
    RICH = "rich"

@dataclass
class FormattingOptions:
    """Options de formatage pour les messages."""
    
    format_type: MessageFormat = MessageFormat.BLOCKS
    style: MessageStyle = MessageStyle.STANDARD
    include_timestamp: bool = True
    include_severity_indicator: bool = True
    include_quick_actions: bool = True
    include_metadata: bool = True
    max_text_length: int = 3000
    max_fields: int = 10
    use_threading: bool = False
    mention_users: List[str] = field(default_factory=list)
    mention_channels: List[str] = field(default_factory=list)
    custom_emojis: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validation post-initialisation."""
        if self.max_text_length > 4000:  # Limite Slack
            self.max_text_length = 4000
        if self.max_fields > 50:  # Limite pratique
            self.max_fields = 50

@dataclass
class MessageContext:
    """Contexte pour le formatage des messages."""
    
    tenant_id: str
    environment: str = "production"
    source_system: str = "AlertManager"
    alert_id: Optional[str] = None
    dashboard_url: Optional[str] = None
    runbook_url: Optional[str] = None
    escalation_policy: Optional[str] = None
    timezone: str = "UTC"
    language: str = "fr"
    custom_data: Dict[str, Any] = field(default_factory=dict)

class SlackMessageFormatter:
    """
    Formateur ultra-avancé pour les messages Slack.
    
    Fonctionnalités:
    - Support complet Blocks Kit avec tous les composants
    - Formatage intelligent basé sur le contexte
    - Templates de formatage personnalisables
    - Adaptation automatique selon la plateforme
    - Internationalisation et localisation
    - Gestion des mentions et liens intelligents
    - Optimisation pour mobile et desktop
    - Validation stricte des formats Slack
    - Cache de formatage pour performances
    - Métriques de formatage détaillées
    """
    
    def __init__(self):
        """Initialise le formateur de messages."""
        
        # Configuration des couleurs par sévérité
        self.severity_colors = {
            SlackSeverity.CRITICAL: "#d63031",  # Rouge vif
            SlackSeverity.HIGH: "#fd79a8",      # Rose/Rouge
            SlackSeverity.MEDIUM: "#fdcb6e",    # Orange
            SlackSeverity.LOW: "#55a3ff",       # Bleu
            SlackSeverity.INFO: "#00b894"       # Vert/Teal
        }
        
        # Émojis par sévérité
        self.severity_emojis = {
            SlackSeverity.CRITICAL: "🚨",
            SlackSeverity.HIGH: "⚠️",
            SlackSeverity.MEDIUM: "📊",
            SlackSeverity.LOW: "ℹ️",
            SlackSeverity.INFO: "✅"
        }
        
        # Configuration des limites Slack
        self.limits = {
            'block_count': 50,
            'text_length': 3000,
            'field_count': 10,
            'attachment_count': 20,
            'action_count': 25
        }
        
        # Cache de formatage
        self._format_cache: Dict[str, Any] = {}
        self._cache_ttl = 300  # 5 minutes
        
        # Métriques
        self.metrics = {
            'messages_formatted': 0,
            'blocks_created': 0,
            'attachments_created': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'validation_errors': 0,
            'truncations': 0
        }
        
        # Templates prédéfinis
        self._load_predefined_templates()
        
        logger.info("SlackMessageFormatter initialisé")
    
    def _load_predefined_templates(self):
        """Charge les templates prédéfinis."""
        self.predefined_templates = {
            'alert_critical': self._get_critical_alert_template(),
            'alert_resolved': self._get_resolved_alert_template(),
            'system_notification': self._get_system_notification_template(),
            'deployment_notification': self._get_deployment_template(),
            'performance_alert': self._get_performance_alert_template()
        }
    
    def format_alert_message(self,
                           alert_data: Dict[str, Any],
                           severity: SlackSeverity,
                           context: MessageContext,
                           options: Optional[FormattingOptions] = None) -> Dict[str, Any]:
        """
        Formate un message d'alerte Slack.
        
        Args:
            alert_data: Données de l'alerte
            severity: Niveau de sévérité
            context: Contexte du message
            options: Options de formatage
            
        Returns:
            Message Slack formaté
        """
        self.metrics['messages_formatted'] += 1
        
        try:
            options = options or FormattingOptions()
            
            # Vérifier le cache
            cache_key = self._get_cache_key('alert', alert_data, severity, context, options)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                self.metrics['cache_hits'] += 1
                return cached_result
            
            self.metrics['cache_misses'] += 1
            
            # Sélectionner le template approprié
            template_name = self._select_alert_template(alert_data, severity)
            
            # Formater selon le type de format
            if options.format_type == MessageFormat.BLOCKS:
                result = self._format_blocks_message(alert_data, severity, context, options, template_name)
            elif options.format_type == MessageFormat.ATTACHMENTS:
                result = self._format_attachments_message(alert_data, severity, context, options)
            elif options.format_type == MessageFormat.TEXT:
                result = self._format_text_message(alert_data, severity, context, options)
            else:  # MIXED
                result = self._format_mixed_message(alert_data, severity, context, options)
            
            # Ajouter les métadonnées communes
            result = self._add_common_metadata(result, context, options)
            
            # Valider le message
            if self._validate_message(result):
                # Mettre en cache
                self._put_in_cache(cache_key, result)
                return result
            else:
                self.metrics['validation_errors'] += 1
                logger.error("Message formaté invalide, utilisation du fallback")
                return self._create_fallback_message(alert_data, severity, context)
                
        except Exception as e:
            logger.error(f"Erreur formatage message d'alerte: {e}")
            return self._create_fallback_message(alert_data, severity, context)
    
    def _select_alert_template(self, alert_data: Dict[str, Any], severity: SlackSeverity) -> str:
        """Sélectionne le template approprié pour l'alerte."""
        alert_name = alert_data.get('alertname', '').lower()
        
        # Alertes critiques
        if severity == SlackSeverity.CRITICAL:
            return 'alert_critical'
        
        # Alertes résolues
        if alert_data.get('status') == 'resolved':
            return 'alert_resolved'
        
        # Alertes de performance
        if any(keyword in alert_name for keyword in ['cpu', 'memory', 'disk', 'latency', 'performance']):
            return 'performance_alert'
        
        # Template par défaut
        return 'alert_critical'
    
    def _format_blocks_message(self,
                             alert_data: Dict[str, Any],
                             severity: SlackSeverity,
                             context: MessageContext,
                             options: FormattingOptions,
                             template_name: str) -> Dict[str, Any]:
        """Formate un message avec Blocks Kit."""
        self.metrics['blocks_created'] += 1
        
        blocks = []
        
        # 1. Header Block
        if options.style in [MessageStyle.STANDARD, MessageStyle.DETAILED, MessageStyle.RICH]:
            header_block = self._create_header_block(alert_data, severity, context, options)
            if header_block:
                blocks.append(header_block)
        
        # 2. Section principale
        main_section = self._create_main_section_block(alert_data, severity, context, options)
        blocks.append(main_section)
        
        # 3. Champs détaillés
        if options.style in [MessageStyle.DETAILED, MessageStyle.RICH]:
            fields_block = self._create_fields_block(alert_data, context, options)
            if fields_block:
                blocks.append(fields_block)
        
        # 4. Contexte et métadonnées
        if options.include_metadata and options.style in [MessageStyle.DETAILED, MessageStyle.RICH]:
            context_block = self._create_context_block(alert_data, context, options)
            if context_block:
                blocks.append(context_block)
        
        # 5. Divider
        if len(blocks) > 1:
            blocks.append({"type": "divider"})
        
        # 6. Actions rapides
        if options.include_quick_actions:
            actions_block = self._create_actions_block(alert_data, context, options)
            if actions_block:
                blocks.append(actions_block)
        
        # 7. Footer
        if options.style == MessageStyle.RICH:
            footer_block = self._create_footer_block(context, options)
            if footer_block:
                blocks.append(footer_block)
        
        # Limitation du nombre de blocks
        if len(blocks) > self.limits['block_count']:
            blocks = blocks[:self.limits['block_count']]
            self.metrics['truncations'] += 1
        
        message = {
            "blocks": blocks,
            "unfurl_links": False,
            "unfurl_media": False
        }
        
        # Ajouter du texte de fallback
        message["text"] = self._create_fallback_text(alert_data, severity)
        
        return message
    
    def _create_header_block(self,
                           alert_data: Dict[str, Any],
                           severity: SlackSeverity,
                           context: MessageContext,
                           options: FormattingOptions) -> Optional[Dict[str, Any]]:
        """Crée le block header."""
        try:
            emoji = self.severity_emojis.get(severity, "")
            alert_name = alert_data.get('alertname', 'Alerte Inconnue')
            severity_text = severity.value.upper()
            
            header_text = f"{emoji} {severity_text} - {alert_name}"
            
            # Limitation de longueur pour les headers
            if len(header_text) > 150:
                header_text = header_text[:147] + "..."
                self.metrics['truncations'] += 1
            
            return {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": header_text,
                    "emoji": True
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur création header block: {e}")
            return None
    
    def _create_main_section_block(self,
                                 alert_data: Dict[str, Any],
                                 severity: SlackSeverity,
                                 context: MessageContext,
                                 options: FormattingOptions) -> Dict[str, Any]:
        """Crée le block section principal."""
        # Construire le texte principal
        text_parts = []
        
        # Sévérité avec couleur
        severity_indicator = f"*Sévérité:* {self._format_severity_with_color(severity)}"
        text_parts.append(severity_indicator)
        
        # Description
        description = alert_data.get('description', alert_data.get('summary', ''))
        if description:
            description = self._escape_markdown(description)
            description = self._truncate_text(description, 500)
            text_parts.append(f"*Description:* {description}")
        
        # Instance/Source
        instance = alert_data.get('instance', alert_data.get('source', ''))
        if instance:
            text_parts.append(f"*Source:* `{instance}`")
        
        # Timestamp
        if options.include_timestamp:
            timestamp = self._format_timestamp(alert_data, context)
            text_parts.append(f"*Heure:* {timestamp}")
        
        # Durée si disponible
        if 'startsAt' in alert_data:
            duration = self._calculate_duration(alert_data['startsAt'])
            if duration:
                text_parts.append(f"*Durée:* {duration}")
        
        text = "\\n".join(text_parts)
        
        block = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": text
            }
        }
        
        # Ajouter une image/icône si spécifiée
        if context.custom_data.get('icon_url'):
            block["accessory"] = {
                "type": "image",
                "image_url": context.custom_data['icon_url'],
                "alt_text": "Alert Icon"
            }
        
        return block
    
    def _create_fields_block(self,
                           alert_data: Dict[str, Any],
                           context: MessageContext,
                           options: FormattingOptions) -> Optional[Dict[str, Any]]:
        """Crée un block avec des champs détaillés."""
        try:
            fields = []
            
            # Labels de l'alerte
            labels = alert_data.get('labels', {})
            priority_labels = ['service', 'environment', 'cluster', 'namespace', 'pod']
            
            # Ajouter les labels prioritaires d'abord
            for label in priority_labels:
                if label in labels and len(fields) < options.max_fields:
                    fields.append({
                        "type": "mrkdwn",
                        "text": f"*{label.title()}:*\\n{self._escape_markdown(str(labels[label]))}"
                    })
            
            # Ajouter les autres labels
            for key, value in labels.items():
                if key not in priority_labels and len(fields) < options.max_fields:
                    fields.append({
                        "type": "mrkdwn",
                        "text": f"*{key.title()}:*\\n{self._escape_markdown(str(value))}"
                    })
            
            # Annotations
            annotations = alert_data.get('annotations', {})
            for key, value in annotations.items():
                if key not in ['description', 'summary'] and len(fields) < options.max_fields:
                    value_truncated = self._truncate_text(str(value), 100)
                    fields.append({
                        "type": "mrkdwn",
                        "text": f"*{key.title()}:*\\n{self._escape_markdown(value_truncated)}"
                    })
            
            if not fields:
                return None
            
            return {
                "type": "section",
                "fields": fields
            }
            
        except Exception as e:
            logger.error(f"Erreur création fields block: {e}")
            return None
    
    def _create_context_block(self,
                            alert_data: Dict[str, Any],
                            context: MessageContext,
                            options: FormattingOptions) -> Optional[Dict[str, Any]]:
        """Crée un block de contexte."""
        try:
            elements = []
            
            # ID de l'alerte
            if context.alert_id:
                elements.append({
                    "type": "mrkdwn",
                    "text": f"🆔 *ID:* `{context.alert_id}`"
                })
            
            # Environnement
            elements.append({
                "type": "mrkdwn",
                "text": f"🌍 *Env:* {context.environment}"
            })
            
            # Tenant
            elements.append({
                "type": "mrkdwn",
                "text": f"🏢 *Tenant:* {context.tenant_id}"
            })
            
            # Source système
            elements.append({
                "type": "mrkdwn",
                "text": f"⚙️ *Source:* {context.source_system}"
            })
            
            if not elements:
                return None
            
            return {
                "type": "context",
                "elements": elements
            }
            
        except Exception as e:
            logger.error(f"Erreur création context block: {e}")
            return None
    
    def _create_actions_block(self,
                            alert_data: Dict[str, Any],
                            context: MessageContext,
                            options: FormattingOptions) -> Optional[Dict[str, Any]]:
        """Crée un block d'actions rapides."""
        try:
            elements = []
            
            # Bouton Dashboard
            if context.dashboard_url:
                elements.append({
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "📊 Dashboard",
                        "emoji": True
                    },
                    "url": context.dashboard_url,
                    "style": "primary"
                })
            
            # Bouton Runbook
            if context.runbook_url:
                elements.append({
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "📖 Runbook",
                        "emoji": True
                    },
                    "url": context.runbook_url
                })
            
            # Bouton Acknowledge (simulé avec URL)
            ack_url = context.custom_data.get('acknowledge_url')
            if ack_url:
                elements.append({
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "✅ Acknowledge",
                        "emoji": True
                    },
                    "url": ack_url,
                    "style": "danger"
                })
            
            # Bouton Silence (simulé avec URL)
            silence_url = context.custom_data.get('silence_url')
            if silence_url:
                elements.append({
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "🔇 Silence",
                        "emoji": True
                    },
                    "url": silence_url
                })
            
            if not elements:
                return None
            
            # Limiter le nombre d'actions
            if len(elements) > 5:
                elements = elements[:5]
                self.metrics['truncations'] += 1
            
            return {
                "type": "actions",
                "elements": elements
            }
            
        except Exception as e:
            logger.error(f"Erreur création actions block: {e}")
            return None
    
    def _create_footer_block(self,
                           context: MessageContext,
                           options: FormattingOptions) -> Optional[Dict[str, Any]]:
        """Crée un block footer."""
        try:
            footer_text = f"Spotify AI Agent | {context.source_system}"
            
            if options.include_timestamp:
                timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                footer_text += f" | {timestamp}"
            
            return {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": footer_text
                    }
                ]
            }
            
        except Exception as e:
            logger.error(f"Erreur création footer block: {e}")
            return None
    
    def _format_attachments_message(self,
                                  alert_data: Dict[str, Any],
                                  severity: SlackSeverity,
                                  context: MessageContext,
                                  options: FormattingOptions) -> Dict[str, Any]:
        """Formate un message avec des attachements legacy."""
        self.metrics['attachments_created'] += 1
        
        # Texte principal
        main_text = self._create_fallback_text(alert_data, severity)
        
        # Créer l'attachement principal
        attachment = {
            "color": self.severity_colors[severity],
            "title": f"{self.severity_emojis[severity]} {alert_data.get('alertname', 'Alerte')}",
            "text": alert_data.get('description', ''),
            "fields": [],
            "ts": int(datetime.utcnow().timestamp())
        }
        
        # Ajouter des champs
        if alert_data.get('instance'):
            attachment["fields"].append({
                "title": "Instance",
                "value": alert_data['instance'],
                "short": True
            })
        
        if context.environment:
            attachment["fields"].append({
                "title": "Environnement",
                "value": context.environment,
                "short": True
            })
        
        # Labels importants
        labels = alert_data.get('labels', {})
        for key in ['service', 'cluster', 'namespace']:
            if key in labels:
                attachment["fields"].append({
                    "title": key.title(),
                    "value": labels[key],
                    "short": True
                })
        
        # Actions
        if context.dashboard_url:
            attachment["actions"] = [
                {
                    "type": "button",
                    "text": "Dashboard",
                    "url": context.dashboard_url
                }
            ]
        
        return {
            "text": main_text,
            "attachments": [attachment]
        }
    
    def _format_text_message(self,
                           alert_data: Dict[str, Any],
                           severity: SlackSeverity,
                           context: MessageContext,
                           options: FormattingOptions) -> Dict[str, Any]:
        """Formate un message texte simple."""
        text_parts = []
        
        # Header
        emoji = self.severity_emojis.get(severity, "")
        header = f"{emoji} *{severity.value.upper()}* - {alert_data.get('alertname', 'Alerte')}"
        text_parts.append(header)
        
        # Description
        description = alert_data.get('description', '')
        if description:
            text_parts.append(f"📝 {description}")
        
        # Instance
        instance = alert_data.get('instance', '')
        if instance:
            text_parts.append(f"🖥️ Instance: `{instance}`")
        
        # Liens
        if context.dashboard_url:
            text_parts.append(f"📊 <{context.dashboard_url}|Dashboard>")
        
        if context.runbook_url:
            text_parts.append(f"📖 <{context.runbook_url}|Runbook>")
        
        # Footer
        if options.include_timestamp:
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            text_parts.append(f"⏰ {timestamp}")
        
        text = "\\n".join(text_parts)
        
        # Vérifier la longueur
        if len(text) > options.max_text_length:
            text = text[:options.max_text_length - 3] + "..."
            self.metrics['truncations'] += 1
        
        return {"text": text}
    
    def _format_mixed_message(self,
                            alert_data: Dict[str, Any],
                            severity: SlackSeverity,
                            context: MessageContext,
                            options: FormattingOptions) -> Dict[str, Any]:
        """Formate un message mixte (blocks + attachments)."""
        # Utiliser blocks pour le contenu principal
        blocks_msg = self._format_blocks_message(alert_data, severity, context, options)
        
        # Ajouter un attachement pour les détails supplémentaires
        attachment = {
            "color": self.severity_colors[severity],
            "fields": []
        }
        
        # Ajouter les labels comme champs d'attachement
        labels = alert_data.get('labels', {})
        for key, value in labels.items():
            if len(attachment["fields"]) < 10:  # Limite
                attachment["fields"].append({
                    "title": key,
                    "value": str(value),
                    "short": True
                })
        
        if attachment["fields"]:
            blocks_msg["attachments"] = [attachment]
        
        return blocks_msg
    
    def _format_severity_with_color(self, severity: SlackSeverity) -> str:
        """Formate la sévérité avec indicateur coloré."""
        emoji = self.severity_emojis.get(severity, "")
        return f"{emoji} {severity.value.upper()}"
    
    def _format_timestamp(self, alert_data: Dict[str, Any], context: MessageContext) -> str:
        """Formate un timestamp selon le contexte."""
        # Essayer de récupérer le timestamp de l'alerte
        timestamp_str = alert_data.get('startsAt', alert_data.get('timestamp'))
        
        if timestamp_str:
            try:
                if isinstance(timestamp_str, str):
                    # Parser ISO format ou autres formats courants
                    if 'T' in timestamp_str:
                        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        dt = datetime.fromtimestamp(float(timestamp_str))
                else:
                    dt = datetime.fromtimestamp(float(timestamp_str))
                
                # Formater selon la langue
                if context.language == 'fr':
                    return dt.strftime("%d/%m/%Y %H:%M:%S UTC")
                else:
                    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
                    
            except (ValueError, TypeError):
                pass
        
        # Fallback: timestamp actuel
        now = datetime.utcnow()
        if context.language == 'fr':
            return now.strftime("%d/%m/%Y %H:%M:%S UTC")
        else:
            return now.strftime("%Y-%m-%d %H:%M:%S UTC")
    
    def _calculate_duration(self, start_time: str) -> Optional[str]:
        """Calcule la durée depuis le début de l'alerte."""
        try:
            if isinstance(start_time, str):
                if 'T' in start_time:
                    start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                else:
                    start_dt = datetime.fromtimestamp(float(start_time))
            else:
                start_dt = datetime.fromtimestamp(float(start_time))
            
            duration = datetime.utcnow() - start_dt.replace(tzinfo=None)
            
            # Formater la durée
            if duration.total_seconds() < 60:
                return f"{int(duration.total_seconds())}s"
            elif duration.total_seconds() < 3600:
                minutes = int(duration.total_seconds() / 60)
                return f"{minutes}m"
            else:
                hours = int(duration.total_seconds() / 3600)
                minutes = int((duration.total_seconds() % 3600) / 60)
                return f"{hours}h {minutes}m"
                
        except (ValueError, TypeError):
            return None
    
    def _escape_markdown(self, text: str) -> str:
        """Échappe les caractères spéciaux Markdown."""
        if not isinstance(text, str):
            text = str(text)
        
        # Caractères à échapper pour Slack markdown
        escape_chars = ['*', '_', '`', '~', '\\']
        
        for char in escape_chars:
            text = text.replace(char, f'\\{char}')
        
        return text
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Tronque intelligemment un texte."""
        if len(text) <= max_length:
            return text
        
        # Essayer de tronquer à un mot complet
        truncated = text[:max_length - 3]
        last_space = truncated.rfind(' ')
        
        if last_space > max_length * 0.8:  # Si l'espace est assez proche de la fin
            truncated = truncated[:last_space]
        
        return truncated + "..."
    
    def _create_fallback_text(self, alert_data: Dict[str, Any], severity: SlackSeverity) -> str:
        """Crée un texte de fallback pour les clients non-compatibles."""
        emoji = self.severity_emojis.get(severity, "")
        alert_name = alert_data.get('alertname', 'Alerte')
        description = alert_data.get('description', '')
        
        fallback = f"{emoji} {severity.value.upper()}: {alert_name}"
        
        if description:
            fallback += f" - {description[:100]}"
            if len(description) > 100:
                fallback += "..."
        
        return fallback
    
    def _add_common_metadata(self,
                           message: Dict[str, Any],
                           context: MessageContext,
                           options: FormattingOptions) -> Dict[str, Any]:
        """Ajoute les métadonnées communes au message."""
        # Thread TS pour grouper les messages
        if options.use_threading and context.alert_id:
            message["thread_ts"] = context.alert_id
        
        # Mentions d'utilisateurs
        if options.mention_users:
            text_mentions = " ".join([f"<@{user}>" for user in options.mention_users])
            if "text" in message:
                message["text"] = text_mentions + " " + message["text"]
            else:
                message["text"] = text_mentions
        
        # Mentions de canaux
        if options.mention_channels:
            text_mentions = " ".join([f"<#{channel}>" for channel in options.mention_channels])
            if "text" in message:
                message["text"] = text_mentions + " " + message["text"]
            else:
                message["text"] = text_mentions
        
        return message
    
    def _validate_message(self, message: Dict[str, Any]) -> bool:
        """Valide un message Slack selon les spécifications."""
        try:
            # Vérifications de base
            if not isinstance(message, dict):
                return False
            
            # Vérifier la présence d'au moins un contenu
            has_content = any(key in message for key in ['text', 'blocks', 'attachments'])
            if not has_content:
                return False
            
            # Valider les blocks si présents
            if 'blocks' in message:
                blocks = message['blocks']
                if not isinstance(blocks, list):
                    return False
                
                if len(blocks) > self.limits['block_count']:
                    return False
                
                for block in blocks:
                    if not self._validate_block(block):
                        return False
            
            # Valider les attachments si présents
            if 'attachments' in message:
                attachments = message['attachments']
                if not isinstance(attachments, list):
                    return False
                
                if len(attachments) > self.limits['attachment_count']:
                    return False
            
            # Valider la longueur du texte
            if 'text' in message:
                if len(message['text']) > self.limits['text_length']:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur validation message: {e}")
            return False
    
    def _validate_block(self, block: Dict[str, Any]) -> bool:
        """Valide un block Slack."""
        try:
            if not isinstance(block, dict):
                return False
            
            # Type obligatoire
            if 'type' not in block:
                return False
            
            block_type = block['type']
            
            # Validation selon le type
            if block_type == 'section':
                # Au moins text ou fields requis
                return 'text' in block or 'fields' in block
            
            elif block_type == 'header':
                # Text obligatoire
                return 'text' in block and isinstance(block['text'], dict)
            
            elif block_type == 'actions':
                # Elements obligatoires
                elements = block.get('elements', [])
                return isinstance(elements, list) and len(elements) <= self.limits['action_count']
            
            elif block_type == 'context':
                # Elements obligatoires
                elements = block.get('elements', [])
                return isinstance(elements, list) and len(elements) <= 10
            
            elif block_type == 'divider':
                # Pas de validation spéciale
                return True
            
            return True
            
        except Exception:
            return False
    
    def _create_fallback_message(self,
                               alert_data: Dict[str, Any],
                               severity: SlackSeverity,
                               context: MessageContext) -> Dict[str, Any]:
        """Crée un message de fallback en cas d'erreur."""
        fallback_text = self._create_fallback_text(alert_data, severity)
        
        return {
            "text": fallback_text,
            "attachments": [
                {
                    "color": self.severity_colors[severity],
                    "text": "Message de fallback - erreur de formatage",
                    "fields": [
                        {
                            "title": "Tenant",
                            "value": context.tenant_id,
                            "short": True
                        },
                        {
                            "title": "Environnement",
                            "value": context.environment,
                            "short": True
                        }
                    ]
                }
            ]
        }
    
    def _get_cache_key(self, *args) -> str:
        """Génère une clé de cache basée sur les arguments."""
        import hashlib
        key_data = json.dumps(args, default=str, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Récupère un élément du cache."""
        if key in self._format_cache:
            cached_item = self._format_cache[key]
            if datetime.utcnow() < cached_item['expires_at']:
                return cached_item['data']
            else:
                del self._format_cache[key]
        return None
    
    def _put_in_cache(self, key: str, data: Dict[str, Any]):
        """Met un élément en cache."""
        self._format_cache[key] = {
            'data': data,
            'expires_at': datetime.utcnow() + timedelta(seconds=self._cache_ttl)
        }
        
        # Nettoyage du cache si trop gros
        if len(self._format_cache) > 1000:
            self._cleanup_cache()
    
    def _cleanup_cache(self):
        """Nettoie le cache expiré."""
        now = datetime.utcnow()
        expired_keys = [
            key for key, item in self._format_cache.items()
            if now >= item['expires_at']
        ]
        
        for key in expired_keys:
            del self._format_cache[key]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques du formateur."""
        cache_hit_rate = 0
        if self.metrics['cache_hits'] + self.metrics['cache_misses'] > 0:
            cache_hit_rate = self.metrics['cache_hits'] / (self.metrics['cache_hits'] + self.metrics['cache_misses'])
        
        return {
            **self.metrics,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self._format_cache),
            'predefined_templates': len(self.predefined_templates),
            'supported_formats': [f.value for f in MessageFormat],
            'supported_styles': [s.value for s in MessageStyle]
        }
    
    def _get_critical_alert_template(self) -> Dict[str, Any]:
        """Template pour alertes critiques."""
        return {
            "format": MessageFormat.BLOCKS,
            "style": MessageStyle.RICH,
            "include_actions": True,
            "include_escalation": True
        }
    
    def _get_resolved_alert_template(self) -> Dict[str, Any]:
        """Template pour alertes résolues."""
        return {
            "format": MessageFormat.BLOCKS,
            "style": MessageStyle.STANDARD,
            "color_override": "#00b894"  # Vert
        }
    
    def _get_system_notification_template(self) -> Dict[str, Any]:
        """Template pour notifications système."""
        return {
            "format": MessageFormat.TEXT,
            "style": MessageStyle.MINIMAL
        }
    
    def _get_deployment_template(self) -> Dict[str, Any]:
        """Template pour notifications de déploiement."""
        return {
            "format": MessageFormat.BLOCKS,
            "style": MessageStyle.DETAILED,
            "include_deployment_info": True
        }
    
    def _get_performance_alert_template(self) -> Dict[str, Any]:
        """Template pour alertes de performance."""
        return {
            "format": MessageFormat.BLOCKS,
            "style": MessageStyle.DETAILED,
            "include_metrics": True,
            "include_graphs": True
        }
    
    def __repr__(self) -> str:
        return f"SlackMessageFormatter(templates={len(self.predefined_templates)}, cache_size={len(self._format_cache)})"
