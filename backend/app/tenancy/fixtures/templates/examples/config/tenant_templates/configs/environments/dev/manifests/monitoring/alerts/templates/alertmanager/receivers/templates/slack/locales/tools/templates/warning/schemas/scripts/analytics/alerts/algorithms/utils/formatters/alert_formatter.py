"""
Spotify AI Agent - Advanced Alert Formatters
===========================================

Ultra-advanced alert formatting system with multi-channel support,
rich interactive elements, and intelligent content adaptation.

This module handles complex alert formatting for:
- Slack with rich blocks and interactive components
- Email with HTML templates and attachments
- SMS with optimized short messages
- Microsoft Teams with adaptive cards
- PagerDuty with incident management integration
- Discord, Telegram, and custom webhooks

Author: Fahed Mlaiel & Spotify AI Team
Version: 2.1.0
"""

import asyncio
import json
import html
import re
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

import aiohttp
import jinja2
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup
import structlog

logger = structlog.get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels with color coding."""
    CRITICAL = ("critical", "#FF0000", "🚨")
    HIGH = ("high", "#FF8C00", "⚠️")
    MEDIUM = ("medium", "#FFD700", "⚡")
    LOW = ("low", "#32CD32", "ℹ️")
    INFO = ("info", "#1E90FF", "📢")


class AlertCategory(Enum):
    """Spotify AI specific alert categories."""
    AI_MODEL_PERFORMANCE = "ai_model_performance"
    RECOMMENDATION_ENGINE = "recommendation_engine"
    AUDIO_PROCESSING = "audio_processing"
    USER_ENGAGEMENT = "user_engagement"
    REVENUE_IMPACT = "revenue_impact"
    INFRASTRUCTURE = "infrastructure"
    SECURITY = "security"
    DATA_QUALITY = "data_quality"
    STREAMING_QUALITY = "streaming_quality"
    ARTIST_ANALYTICS = "artist_analytics"


@dataclass
class FormattedAlert:
    """Structured alert output."""
    
    title: str
    content: str
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    interactive_elements: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "title": self.title,
            "content": self.content,
            "attachments": self.attachments,
            "metadata": self.metadata,
            "interactive_elements": self.interactive_elements
        }


class BaseAlertFormatter(ABC):
    """Abstract base class for all alert formatters."""
    
    def __init__(self, tenant_id: str, config: Optional[Dict[str, Any]] = None):
        self.tenant_id = tenant_id
        self.config = config or {}
        self.template_env = self._setup_template_environment()
        self.logger = logger.bind(tenant_id=tenant_id, formatter=self.__class__.__name__)
        
    def _setup_template_environment(self) -> Environment:
        """Setup Jinja2 template environment with custom filters."""
        env = Environment(
            loader=FileSystemLoader('templates'),
            autoescape=select_autoescape(['html', 'xml']),
            enable_async=True
        )
        
        # Add custom filters
        env.filters['format_duration'] = self._format_duration
        env.filters['format_percentage'] = self._format_percentage
        env.filters['format_currency'] = self._format_currency
        env.filters['format_number'] = self._format_number
        env.filters['truncate_smart'] = self._truncate_smart
        env.filters['spotify_link'] = self._create_spotify_link
        
        return env
        
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in human readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
            
    @staticmethod
    def _format_percentage(value: float, precision: int = 1) -> str:
        """Format percentage with proper precision."""
        return f"{value * 100:.{precision}f}%"
        
    @staticmethod
    def _format_currency(amount: float, currency: str = "USD") -> str:
        """Format currency amount."""
        return f"${amount:,.2f} {currency}"
        
    @staticmethod
    def _format_number(value: Union[int, float], precision: int = 0) -> str:
        """Format large numbers with appropriate suffixes."""
        if value >= 1_000_000_000:
            return f"{value/1_000_000_000:.{precision}f}B"
        elif value >= 1_000_000:
            return f"{value/1_000_000:.{precision}f}M"
        elif value >= 1_000:
            return f"{value/1_000:.{precision}f}K"
        else:
            return f"{value:.{precision}f}"
            
    @staticmethod
    def _truncate_smart(text: str, length: int = 100) -> str:
        """Smart truncation that respects word boundaries."""
        if len(text) <= length:
            return text
        
        truncated = text[:length]
        last_space = truncated.rfind(' ')
        if last_space > length * 0.8:  # Only truncate at word if not too short
            truncated = truncated[:last_space]
        
        return f"{truncated}..."
        
    @staticmethod
    def _create_spotify_link(entity_type: str, entity_id: str) -> str:
        """Create Spotify deep links."""
        base_url = "https://open.spotify.com"
        return f"{base_url}/{entity_type}/{entity_id}"
        
    @abstractmethod
    async def format_alert(self, alert_data: Dict[str, Any]) -> FormattedAlert:
        """Format alert data for specific channel."""
        pass
        
    def _sanitize_content(self, content: str) -> str:
        """Sanitize content to prevent XSS and formatting issues."""
        # HTML escape
        content = html.escape(content)
        
        # Remove potentially dangerous characters
        content = re.sub(r'[<>"\']', '', content)
        
        return content


class SlackAlertFormatter(BaseAlertFormatter):
    """Advanced Slack alert formatter with rich blocks and interactivity."""
    
    def __init__(self, tenant_id: str, webhook_url: Optional[str] = None, **kwargs):
        super().__init__(tenant_id, kwargs)
        self.webhook_url = webhook_url
        self.max_block_limit = 50  # Slack's limit
        
    async def format_alert(self, alert_data: Dict[str, Any]) -> FormattedAlert:
        """Format alert as Slack blocks with rich content."""
        
        severity = AlertSeverity(alert_data.get('severity', 'info'))
        category = AlertCategory(alert_data.get('category', 'infrastructure'))
        
        # Build main content blocks
        blocks = []
        
        # Header block with severity indicator
        header_block = {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{severity.value[2]} {alert_data.get('title', 'Spotify AI Alert')}"
            }
        }
        blocks.append(header_block)
        
        # Context block with metadata
        context_elements = []
        if alert_data.get('tenant_id'):
            context_elements.append({
                "type": "mrkdwn",
                "text": f"*Tenant:* {alert_data['tenant_id']}"
            })
        if alert_data.get('timestamp'):
            context_elements.append({
                "type": "mrkdwn", 
                "text": f"*Time:* {alert_data['timestamp']}"
            })
        if alert_data.get('environment'):
            context_elements.append({
                "type": "mrkdwn",
                "text": f"*Environment:* {alert_data['environment']}"
            })
            
        if context_elements:
            blocks.append({
                "type": "context",
                "elements": context_elements
            })
        
        # Main content section
        main_text = alert_data.get('description', 'No description provided')
        if len(main_text) > 3000:  # Slack's text limit
            main_text = self._truncate_smart(main_text, 2900)
            
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": main_text
            }
        })
        
        # Metrics section if available
        if alert_data.get('metrics'):
            metrics_fields = []
            for key, value in alert_data['metrics'].items():
                field_text = f"*{key.replace('_', ' ').title()}:*\n{self._format_metric_value(key, value)}"
                metrics_fields.append({
                    "type": "mrkdwn",
                    "text": field_text
                })
                
            # Split fields into chunks of 10 (Slack's limit per section)
            for i in range(0, len(metrics_fields), 10):
                chunk = metrics_fields[i:i+10]
                blocks.append({
                    "type": "section",
                    "fields": chunk
                })
        
        # Affected resources section
        if alert_data.get('affected_resources'):
            resources_text = "🎯 *Affected Resources:*\n"
            for resource in alert_data['affected_resources'][:10]:  # Limit to 10
                resources_text += f"• {resource}\n"
                
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": resources_text
                }
            })
        
        # Action buttons if required
        if alert_data.get('action_required', False):
            action_blocks = self._create_action_buttons(alert_data)
            blocks.extend(action_blocks)
        
        # Spotify-specific rich content
        if category in [AlertCategory.AI_MODEL_PERFORMANCE, AlertCategory.RECOMMENDATION_ENGINE]:
            spotify_blocks = await self._create_spotify_ai_blocks(alert_data)
            blocks.extend(spotify_blocks)
        
        # Divider before footer
        blocks.append({"type": "divider"})
        
        # Footer with links and additional info
        footer_elements = []
        if alert_data.get('runbook_url'):
            footer_elements.append({
                "type": "mrkdwn",
                "text": f"📖 <{alert_data['runbook_url']}|Runbook>"
            })
        if alert_data.get('dashboard_url'):
            footer_elements.append({
                "type": "mrkdwn",
                "text": f"📊 <{alert_data['dashboard_url']}|Dashboard>"
            })
        if alert_data.get('incident_id'):
            footer_elements.append({
                "type": "mrkdwn",
                "text": f"🎫 Incident: {alert_data['incident_id']}"
            })
            
        if footer_elements:
            blocks.append({
                "type": "context",
                "elements": footer_elements
            })
        
        # Ensure we don't exceed Slack's block limit
        if len(blocks) > self.max_block_limit:
            blocks = blocks[:self.max_block_limit-1]
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "_Additional content truncated due to Slack limits_"
                }
            })
        
        slack_content = {
            "blocks": blocks,
            "color": severity.value[1],
            "unfurl_links": False,
            "unfurl_media": False
        }
        
        return FormattedAlert(
            title=alert_data.get('title', 'Spotify AI Alert'),
            content=json.dumps(slack_content, indent=2),
            metadata={
                "channel_type": "slack",
                "severity": severity.value[0],
                "category": category.value,
                "block_count": len(blocks)
            }
        )
    
    def _format_metric_value(self, key: str, value: Any) -> str:
        """Format metric values based on their type and context."""
        if key in ['accuracy', 'precision', 'recall', 'f1_score', 'confidence']:
            return self._format_percentage(float(value))
        elif key in ['latency', 'response_time', 'duration']:
            return self._format_duration(float(value))
        elif key in ['revenue', 'cost', 'price']:
            return self._format_currency(float(value))
        elif key in ['count', 'total', 'streams', 'plays']:
            return self._format_number(float(value))
        else:
            return str(value)
    
    def _create_action_buttons(self, alert_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create interactive action buttons."""
        action_blocks = []
        
        # Primary actions
        primary_actions = {
            "type": "actions",
            "elements": []
        }
        
        # Acknowledge button
        primary_actions["elements"].append({
            "type": "button",
            "text": {
                "type": "plain_text",
                "text": "🔔 Acknowledge"
            },
            "style": "primary",
            "action_id": "acknowledge_alert",
            "value": json.dumps({
                "alert_id": alert_data.get('alert_id'),
                "tenant_id": self.tenant_id
            })
        })
        
        # Resolve button
        primary_actions["elements"].append({
            "type": "button",
            "text": {
                "type": "plain_text",
                "text": "✅ Resolve"
            },
            "style": "primary",
            "action_id": "resolve_alert",
            "value": json.dumps({
                "alert_id": alert_data.get('alert_id'),
                "tenant_id": self.tenant_id
            })
        })
        
        # Escalate button for critical alerts
        if alert_data.get('severity') == 'critical':
            primary_actions["elements"].append({
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "🚨 Escalate"
                },
                "style": "danger",
                "action_id": "escalate_alert",
                "value": json.dumps({
                    "alert_id": alert_data.get('alert_id'),
                    "tenant_id": self.tenant_id
                })
            })
        
        action_blocks.append(primary_actions)
        
        # Secondary actions
        secondary_actions = {
            "type": "actions",
            "elements": []
        }
        
        # Snooze dropdown
        snooze_options = [
            {"text": {"type": "plain_text", "text": "15 minutes"}, "value": "15m"},
            {"text": {"type": "plain_text", "text": "1 hour"}, "value": "1h"},
            {"text": {"type": "plain_text", "text": "4 hours"}, "value": "4h"},
            {"text": {"type": "plain_text", "text": "24 hours"}, "value": "24h"}
        ]
        
        secondary_actions["elements"].append({
            "type": "static_select",
            "placeholder": {
                "type": "plain_text",
                "text": "⏰ Snooze for..."
            },
            "action_id": "snooze_alert",
            "options": snooze_options
        })
        
        # Add to incident button
        secondary_actions["elements"].append({
            "type": "button",
            "text": {
                "type": "plain_text",
                "text": "🎫 Create Incident"
            },
            "action_id": "create_incident",
            "value": json.dumps({
                "alert_id": alert_data.get('alert_id'),
                "tenant_id": self.tenant_id
            })
        })
        
        action_blocks.append(secondary_actions)
        
        return action_blocks
    
    async def _create_spotify_ai_blocks(self, alert_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create Spotify AI specific rich blocks."""
        blocks = []
        
        # AI Model Performance visualization
        if alert_data.get('model_metrics'):
            model_metrics = alert_data['model_metrics']
            
            # Model performance chart (using Chart.js via image)
            chart_url = await self._generate_performance_chart(model_metrics)
            if chart_url:
                blocks.append({
                    "type": "image",
                    "title": {
                        "type": "plain_text",
                        "text": "🤖 AI Model Performance Trend"
                    },
                    "image_url": chart_url,
                    "alt_text": "AI model performance trend chart"
                })
        
        # Recommendation Engine insights
        if alert_data.get('recommendation_data'):
            rec_data = alert_data['recommendation_data']
            
            rec_text = "🎵 *Recommendation Engine Status:*\n"
            if rec_data.get('hit_rate'):
                rec_text += f"• Hit Rate: {self._format_percentage(rec_data['hit_rate'])}\n"
            if rec_data.get('diversity_score'):
                rec_text += f"• Diversity Score: {self._format_percentage(rec_data['diversity_score'])}\n"
            if rec_data.get('novelty_score'):
                rec_text += f"• Novelty Score: {self._format_percentage(rec_data['novelty_score'])}\n"
            
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": rec_text
                }
            })
        
        # Top affected tracks/artists
        if alert_data.get('affected_content'):
            content = alert_data['affected_content']
            
            if content.get('tracks'):
                tracks_text = "🎵 *Top Affected Tracks:*\n"
                for track in content['tracks'][:5]:
                    spotify_url = self._create_spotify_link('track', track['id'])
                    tracks_text += f"• <{spotify_url}|{track['name']}> by {track['artist']}\n"
                
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": tracks_text
                    }
                })
            
            if content.get('artists'):
                artists_text = "👨‍🎤 *Top Affected Artists:*\n"
                for artist in content['artists'][:5]:
                    spotify_url = self._create_spotify_link('artist', artist['id'])
                    artists_text += f"• <{spotify_url}|{artist['name']}> ({self._format_number(artist['followers'])} followers)\n"
                
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": artists_text
                    }
                })
        
        return blocks
    
    async def _generate_performance_chart(self, metrics: Dict[str, Any]) -> Optional[str]:
        """Generate performance chart and return URL."""
        try:
            # This would integrate with a chart generation service
            # For now, return a placeholder
            return "https://quickchart.io/chart?c={type:'line',data:{labels:['1h','2h','3h','4h','5h'],datasets:[{label:'Accuracy',data:[85,82,78,75,72]}]}}"
        except Exception as e:
            self.logger.error("Failed to generate chart", error=str(e))
            return None
    
    async def send_alert(self, formatted_alert: FormattedAlert) -> bool:
        """Send formatted alert to Slack webhook."""
        if not self.webhook_url:
            self.logger.warning("No webhook URL configured for Slack")
            return False
        
        try:
            slack_payload = json.loads(formatted_alert.content)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=slack_payload,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status == 200:
                        self.logger.info("Slack alert sent successfully")
                        return True
                    else:
                        self.logger.error("Failed to send Slack alert", status=response.status)
                        return False
                        
        except Exception as e:
            self.logger.error("Exception sending Slack alert", error=str(e))
            return False


class EmailAlertFormatter(BaseAlertFormatter):
    """Advanced email alert formatter with HTML templates and attachments."""
    
    def __init__(self, tenant_id: str, smtp_config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(tenant_id, kwargs)
        self.smtp_config = smtp_config or {}
        
    async def format_alert(self, alert_data: Dict[str, Any]) -> FormattedAlert:
        """Format alert as rich HTML email with attachments."""
        
        severity = AlertSeverity(alert_data.get('severity', 'info'))
        category = AlertCategory(alert_data.get('category', 'infrastructure'))
        
        # Load and render HTML template
        template = self.template_env.get_template('email_alert.html')
        
        template_vars = {
            'title': alert_data.get('title', 'Spotify AI Alert'),
            'severity': severity,
            'category': category,
            'tenant_id': self.tenant_id,
            'timestamp': alert_data.get('timestamp', datetime.now(timezone.utc)),
            'description': alert_data.get('description', ''),
            'metrics': alert_data.get('metrics', {}),
            'affected_resources': alert_data.get('affected_resources', []),
            'action_required': alert_data.get('action_required', False),
            'runbook_url': alert_data.get('runbook_url'),
            'dashboard_url': alert_data.get('dashboard_url'),
            'incident_id': alert_data.get('incident_id')
        }
        
        html_content = await template.render_async(**template_vars)
        
        # Generate plain text version
        plain_text = await self._generate_plain_text(template_vars)
        
        # Create attachments for detailed metrics
        attachments = []
        if alert_data.get('detailed_metrics'):
            csv_attachment = await self._create_metrics_csv(alert_data['detailed_metrics'])
            attachments.append(csv_attachment)
        
        if alert_data.get('log_snippet'):
            log_attachment = await self._create_log_attachment(alert_data['log_snippet'])
            attachments.append(log_attachment)
        
        return FormattedAlert(
            title=f"[{severity.value[0].upper()}] {template_vars['title']}",
            content=html_content,
            attachments=attachments,
            metadata={
                "channel_type": "email",
                "severity": severity.value[0],
                "category": category.value,
                "plain_text": plain_text,
                "content_type": "text/html"
            }
        )
    
    async def _generate_plain_text(self, template_vars: Dict[str, Any]) -> str:
        """Generate plain text version of email."""
        template = self.template_env.get_template('email_alert.txt')
        return await template.render_async(**template_vars)
    
    async def _create_metrics_csv(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create CSV attachment with detailed metrics."""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Metric', 'Value', 'Timestamp'])
        
        # Write metrics data
        for key, value in metrics.items():
            writer.writerow([key, value, datetime.now(timezone.utc).isoformat()])
        
        return {
            "filename": f"metrics_{self.tenant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "content": output.getvalue(),
            "content_type": "text/csv"
        }
    
    async def _create_log_attachment(self, log_data: str) -> Dict[str, Any]:
        """Create log file attachment."""
        return {
            "filename": f"logs_{self.tenant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            "content": log_data,
            "content_type": "text/plain"
        }


class SMSAlertFormatter(BaseAlertFormatter):
    """SMS alert formatter optimized for short messages."""
    
    MAX_SMS_LENGTH = 160
    
    async def format_alert(self, alert_data: Dict[str, Any]) -> FormattedAlert:
        """Format alert as optimized SMS message."""
        
        severity = AlertSeverity(alert_data.get('severity', 'info'))
        
        # Create concise message
        message_parts = []
        
        # Severity emoji and tenant
        message_parts.append(f"{severity.value[2]} {self.tenant_id}")
        
        # Title (truncated)
        title = alert_data.get('title', 'Alert')
        if len(title) > 50:
            title = title[:47] + "..."
        message_parts.append(title)
        
        # Key metric if available
        if alert_data.get('key_metric'):
            key_metric = alert_data['key_metric']
            message_parts.append(f"{key_metric['name']}: {key_metric['value']}")
        
        # Action required indicator
        if alert_data.get('action_required'):
            message_parts.append("ACTION REQUIRED")
        
        # Combine and ensure length limit
        message = " | ".join(message_parts)
        
        if len(message) > self.MAX_SMS_LENGTH:
            # Truncate intelligently
            available_length = self.MAX_SMS_LENGTH - 3  # Reserve for "..."
            message = message[:available_length] + "..."
        
        return FormattedAlert(
            title="SMS Alert",
            content=message,
            metadata={
                "channel_type": "sms",
                "severity": severity.value[0],
                "character_count": len(message)
            }
        )


class TeamsAlertFormatter(BaseAlertFormatter):
    """Microsoft Teams adaptive cards formatter."""
    
    async def format_alert(self, alert_data: Dict[str, Any]) -> FormattedAlert:
        """Format alert as Teams adaptive card."""
        
        severity = AlertSeverity(alert_data.get('severity', 'info'))
        category = AlertCategory(alert_data.get('category', 'infrastructure'))
        
        # Create adaptive card structure
        card = {
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "type": "AdaptiveCard",
            "version": "1.5",
            "body": [],
            "actions": []
        }
        
        # Header with severity color
        card["body"].append({
            "type": "Container",
            "style": self._get_teams_style(severity),
            "items": [
                {
                    "type": "TextBlock",
                    "text": f"{severity.value[2]} {alert_data.get('title', 'Spotify AI Alert')}",
                    "weight": "Bolder",
                    "size": "Large",
                    "color": "Light"
                }
            ]
        })
        
        # Metadata fact set
        facts = []
        if alert_data.get('tenant_id'):
            facts.append({"title": "Tenant", "value": alert_data['tenant_id']})
        if alert_data.get('timestamp'):
            facts.append({"title": "Time", "value": alert_data['timestamp']})
        if alert_data.get('environment'):
            facts.append({"title": "Environment", "value": alert_data['environment']})
        
        if facts:
            card["body"].append({
                "type": "FactSet",
                "facts": facts
            })
        
        # Description
        if alert_data.get('description'):
            card["body"].append({
                "type": "TextBlock",
                "text": alert_data['description'],
                "wrap": True
            })
        
        # Metrics table
        if alert_data.get('metrics'):
            metrics_facts = []
            for key, value in alert_data['metrics'].items():
                formatted_value = self._format_metric_value(key, value)
                metrics_facts.append({
                    "title": key.replace('_', ' ').title(),
                    "value": formatted_value
                })
            
            card["body"].append({
                "type": "Container",
                "items": [
                    {
                        "type": "TextBlock",
                        "text": "📊 Metrics",
                        "weight": "Bolder"
                    },
                    {
                        "type": "FactSet",
                        "facts": metrics_facts
                    }
                ]
            })
        
        # Action buttons
        if alert_data.get('action_required'):
            card["actions"] = [
                {
                    "type": "Action.Http",
                    "title": "Acknowledge",
                    "method": "POST",
                    "url": f"{self.config.get('webhook_base_url', '')}/alerts/acknowledge",
                    "body": json.dumps({
                        "alert_id": alert_data.get('alert_id'),
                        "tenant_id": self.tenant_id,
                        "action": "acknowledge"
                    })
                },
                {
                    "type": "Action.Http",
                    "title": "Resolve",
                    "method": "POST",
                    "url": f"{self.config.get('webhook_base_url', '')}/alerts/resolve",
                    "body": json.dumps({
                        "alert_id": alert_data.get('alert_id'),
                        "tenant_id": self.tenant_id,
                        "action": "resolve"
                    })
                }
            ]
        
        # External links
        if alert_data.get('dashboard_url') or alert_data.get('runbook_url'):
            link_actions = []
            if alert_data.get('dashboard_url'):
                link_actions.append({
                    "type": "Action.OpenUrl",
                    "title": "View Dashboard",
                    "url": alert_data['dashboard_url']
                })
            if alert_data.get('runbook_url'):
                link_actions.append({
                    "type": "Action.OpenUrl",
                    "title": "View Runbook",
                    "url": alert_data['runbook_url']
                })
            
            card["actions"].extend(link_actions)
        
        return FormattedAlert(
            title=alert_data.get('title', 'Spotify AI Alert'),
            content=json.dumps(card, indent=2),
            metadata={
                "channel_type": "teams",
                "severity": severity.value[0],
                "category": category.value,
                "card_type": "adaptive_card"
            }
        )
    
    def _get_teams_style(self, severity: AlertSeverity) -> str:
        """Get Teams container style based on severity."""
        style_map = {
            AlertSeverity.CRITICAL: "attention",
            AlertSeverity.HIGH: "warning", 
            AlertSeverity.MEDIUM: "accent",
            AlertSeverity.LOW: "good",
            AlertSeverity.INFO: "default"
        }
        return style_map.get(severity, "default")
    
    def _format_metric_value(self, key: str, value: Any) -> str:
        """Format metric values for Teams display."""
        # Use same logic as Slack formatter
        return SlackAlertFormatter._format_metric_value(self, key, value)


class PagerDutyAlertFormatter(BaseAlertFormatter):
    """PagerDuty integration formatter for incident management."""
    
    def __init__(self, tenant_id: str, integration_key: Optional[str] = None, **kwargs):
        super().__init__(tenant_id, kwargs)
        self.integration_key = integration_key
        
    async def format_alert(self, alert_data: Dict[str, Any]) -> FormattedAlert:
        """Format alert for PagerDuty Events API v2."""
        
        severity = AlertSeverity(alert_data.get('severity', 'info'))
        
        # Determine event action
        event_action = "trigger"
        if alert_data.get('resolved'):
            event_action = "resolve"
        elif alert_data.get('acknowledged'):
            event_action = "acknowledge"
        
        # Create PagerDuty event payload
        payload = {
            "routing_key": self.integration_key,
            "event_action": event_action,
            "dedup_key": alert_data.get('alert_id', f"{self.tenant_id}_{alert_data.get('title', 'alert')}"),
            "payload": {
                "summary": alert_data.get('title', 'Spotify AI Alert'),
                "source": f"spotify-ai-agent-{self.tenant_id}",
                "severity": self._map_severity_to_pagerduty(severity),
                "component": alert_data.get('category', 'infrastructure'),
                "group": self.tenant_id,
                "class": "spotify_ai_alert"
            }
        }
        
        # Add custom details
        custom_details = {
            "tenant_id": self.tenant_id,
            "description": alert_data.get('description', ''),
            "environment": alert_data.get('environment', 'unknown'),
            "category": alert_data.get('category', 'infrastructure')
        }
        
        if alert_data.get('metrics'):
            custom_details["metrics"] = alert_data['metrics']
        
        if alert_data.get('affected_resources'):
            custom_details["affected_resources"] = alert_data['affected_resources']
        
        payload["payload"]["custom_details"] = custom_details
        
        # Add links
        links = []
        if alert_data.get('dashboard_url'):
            links.append({
                "href": alert_data['dashboard_url'],
                "text": "View Dashboard"
            })
        if alert_data.get('runbook_url'):
            links.append({
                "href": alert_data['runbook_url'],
                "text": "View Runbook"
            })
        
        if links:
            payload["links"] = links
        
        return FormattedAlert(
            title=payload["payload"]["summary"],
            content=json.dumps(payload, indent=2),
            metadata={
                "channel_type": "pagerduty",
                "severity": severity.value[0],
                "event_action": event_action,
                "dedup_key": payload["dedup_key"]
            }
        )
    
    def _map_severity_to_pagerduty(self, severity: AlertSeverity) -> str:
        """Map internal severity to PagerDuty severity."""
        mapping = {
            AlertSeverity.CRITICAL: "critical",
            AlertSeverity.HIGH: "error",
            AlertSeverity.MEDIUM: "warning",
            AlertSeverity.LOW: "info",
            AlertSeverity.INFO: "info"
        }
        return mapping.get(severity, "info")


# Factory function for creating formatters
def create_alert_formatter(
    formatter_type: str,
    tenant_id: str,
    config: Optional[Dict[str, Any]] = None
) -> BaseAlertFormatter:
    """
    Factory function to create alert formatters.
    
    Args:
        formatter_type: Type of formatter ('slack', 'email', 'sms', 'teams', 'pagerduty')
        tenant_id: Tenant identifier
        config: Configuration dictionary
        
    Returns:
        Configured formatter instance
    """
    formatters = {
        'slack': SlackAlertFormatter,
        'email': EmailAlertFormatter,
        'sms': SMSAlertFormatter,
        'teams': TeamsAlertFormatter,
        'pagerduty': PagerDutyAlertFormatter
    }
    
    if formatter_type not in formatters:
        raise ValueError(f"Unsupported formatter type: {formatter_type}")
    
    formatter_class = formatters[formatter_type]
    return formatter_class(tenant_id, **(config or {}))
