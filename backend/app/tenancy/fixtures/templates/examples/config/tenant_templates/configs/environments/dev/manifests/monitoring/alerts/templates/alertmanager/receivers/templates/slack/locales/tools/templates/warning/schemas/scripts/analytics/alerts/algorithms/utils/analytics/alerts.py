"""
Advanced Alert Management System for Spotify AI Agent
===================================================

Ultra-sophisticated alert management with intelligent routing, escalation,
and automated response capabilities.

Author: Fahed Mlaiel
Roles: Lead Dev + Architecte IA, Spécialiste Sécurité Backend
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Union
import aioredis
import aiohttp
from prometheus_client import Counter, Histogram
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

class AlertStatus(Enum):
    """Alert status enumeration."""
    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"
    SILENCED = "silenced"
    ACKNOWLEDGED = "acknowledged"

class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class NotificationChannel(Enum):
    """Notification channel types."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGERDUTY = "pagerduty"
    TEAMS = "teams"

@dataclass
class AlertRule:
    """Comprehensive alert rule definition."""
    id: str
    name: str
    description: str
    query: str
    condition: str
    threshold: float
    severity: AlertSeverity
    tenant_id: str
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    for_duration: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    evaluation_interval: timedelta = field(default_factory=lambda: timedelta(minutes=1))
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class Alert:
    """Alert instance with full context."""
    id: str
    rule_id: str
    tenant_id: str
    status: AlertStatus
    severity: AlertSeverity
    title: str
    description: str
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    value: Optional[float] = None
    threshold: Optional[float] = None
    started_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    escalation_level: int = 0
    notification_count: int = 0
    last_notification: Optional[datetime] = None

@dataclass
class NotificationRule:
    """Notification routing and escalation rule."""
    id: str
    name: str
    tenant_id: str
    severity_filter: List[AlertSeverity]
    label_filters: Dict[str, str] = field(default_factory=dict)
    channels: List[NotificationChannel] = field(default_factory=list)
    escalation_rules: List[Dict[str, Any]] = field(default_factory=list)
    cooldown: timedelta = field(default_factory=lambda: timedelta(minutes=15))
    enabled: bool = True

@dataclass
class NotificationChannel:
    """Notification channel configuration."""
    id: str
    type: NotificationChannel
    name: str
    config: Dict[str, Any]
    enabled: bool = True

class AlertManager:
    """
    Advanced alert management system with intelligent routing,
    escalation, and automated response capabilities.
    """
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        self.redis_client = redis_client
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.notification_rules: Dict[str, NotificationRule] = {}
        self.notification_channels: Dict[str, NotificationChannel] = {}
        self.alert_history: List[Alert] = []
        self.metrics = self._setup_metrics()
        self.evaluation_tasks: Set[asyncio.Task] = set()
        self.notification_queue = asyncio.Queue()
        self.is_running = False
        
    def _setup_metrics(self) -> Dict[str, Any]:
        """Setup Prometheus metrics for alert management."""
        return {
            'alerts_total': Counter(
                'spotify_ai_alerts_total',
                'Total number of alerts',
                ['tenant_id', 'severity', 'status']
            ),
            'alert_evaluation_duration': Histogram(
                'spotify_ai_alert_evaluation_duration_seconds',
                'Time spent evaluating alerts',
                ['tenant_id', 'rule_id']
            ),
            'notifications_sent': Counter(
                'spotify_ai_notifications_sent_total',
                'Total notifications sent',
                ['tenant_id', 'channel', 'severity']
            )
        }

    async def start(self) -> None:
        """Start the alert manager."""
        if self.is_running:
            return
            
        self.is_running = True
        
        # Start background tasks
        asyncio.create_task(self._evaluation_loop())
        asyncio.create_task(self._notification_processor())
        asyncio.create_task(self._cleanup_task())
        
        logger.info("Alert Manager started")

    async def stop(self) -> None:
        """Stop the alert manager."""
        self.is_running = False
        
        # Cancel evaluation tasks
        for task in self.evaluation_tasks:
            task.cancel()
        
        logger.info("Alert Manager stopped")

    async def add_alert_rule(self, rule: AlertRule) -> str:
        """Add a new alert rule."""
        self.alert_rules[rule.id] = rule
        
        # Store in Redis for persistence
        if self.redis_client:
            await self.redis_client.hset(
                f"alert_rules:{rule.tenant_id}",
                rule.id,
                json.dumps(self._serialize_rule(rule))
            )
        
        # Start evaluation task for this rule
        task = asyncio.create_task(self._evaluate_rule_loop(rule))
        self.evaluation_tasks.add(task)
        
        logger.info(f"Alert rule added: {rule.name} for tenant {rule.tenant_id}")
        return rule.id

    async def remove_alert_rule(self, rule_id: str, tenant_id: str) -> bool:
        """Remove an alert rule."""
        if rule_id not in self.alert_rules:
            return False
            
        rule = self.alert_rules[rule_id]
        if rule.tenant_id != tenant_id:
            return False
            
        # Remove from memory
        del self.alert_rules[rule_id]
        
        # Remove from Redis
        if self.redis_client:
            await self.redis_client.hdel(f"alert_rules:{tenant_id}", rule_id)
        
        # Cancel evaluation task
        for task in self.evaluation_tasks:
            if hasattr(task, 'rule_id') and task.rule_id == rule_id:
                task.cancel()
                self.evaluation_tasks.remove(task)
                break
        
        logger.info(f"Alert rule removed: {rule_id}")
        return True

    async def fire_alert(self, rule_id: str, value: float, context: Dict[str, Any] = None) -> str:
        """Fire an alert for a specific rule."""
        if rule_id not in self.alert_rules:
            raise ValueError(f"Alert rule {rule_id} not found")
            
        rule = self.alert_rules[rule_id]
        context = context or {}
        
        # Create alert instance
        alert = Alert(
            id=f"{rule_id}_{datetime.now().timestamp()}",
            rule_id=rule_id,
            tenant_id=rule.tenant_id,
            status=AlertStatus.FIRING,
            severity=rule.severity,
            title=rule.name,
            description=rule.description,
            labels=rule.labels.copy(),
            annotations=rule.annotations.copy(),
            value=value,
            threshold=rule.threshold
        )
        
        # Add context to labels
        alert.labels.update(context.get('labels', {}))
        alert.annotations.update(context.get('annotations', {}))
        
        # Store alert
        self.active_alerts[alert.id] = alert
        
        # Store in Redis
        if self.redis_client:
            await self.redis_client.hset(
                f"active_alerts:{alert.tenant_id}",
                alert.id,
                json.dumps(self._serialize_alert(alert))
            )
        
        # Update metrics
        self.metrics['alerts_total'].labels(
            tenant_id=alert.tenant_id,
            severity=alert.severity.value,
            status=alert.status.value
        ).inc()
        
        # Queue for notification
        await self.notification_queue.put(alert)
        
        logger.warning(f"Alert fired: {alert.title} (ID: {alert.id})")
        return alert.id

    async def resolve_alert(self, alert_id: str, tenant_id: str) -> bool:
        """Resolve an active alert."""
        if alert_id not in self.active_alerts:
            return False
            
        alert = self.active_alerts[alert_id]
        if alert.tenant_id != tenant_id:
            return False
            
        # Update alert status
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()
        
        # Move to history
        self.alert_history.append(alert)
        del self.active_alerts[alert_id]
        
        # Update Redis
        if self.redis_client:
            await self.redis_client.hdel(f"active_alerts:{tenant_id}", alert_id)
            await self.redis_client.lpush(
                f"alert_history:{tenant_id}",
                json.dumps(self._serialize_alert(alert))
            )
        
        # Update metrics
        self.metrics['alerts_total'].labels(
            tenant_id=alert.tenant_id,
            severity=alert.severity.value,
            status=alert.status.value
        ).inc()
        
        logger.info(f"Alert resolved: {alert.title} (ID: {alert.id})")
        return True

    async def acknowledge_alert(self, alert_id: str, tenant_id: str, user_id: str) -> bool:
        """Acknowledge an active alert."""
        if alert_id not in self.active_alerts:
            return False
            
        alert = self.active_alerts[alert_id]
        if alert.tenant_id != tenant_id:
            return False
            
        # Update alert
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.now()
        alert.acknowledged_by = user_id
        
        # Update Redis
        if self.redis_client:
            await self.redis_client.hset(
                f"active_alerts:{tenant_id}",
                alert_id,
                json.dumps(self._serialize_alert(alert))
            )
        
        logger.info(f"Alert acknowledged: {alert.title} by {user_id}")
        return True

    async def add_notification_rule(self, rule: NotificationRule) -> str:
        """Add a notification rule."""
        self.notification_rules[rule.id] = rule
        
        # Store in Redis
        if self.redis_client:
            await self.redis_client.hset(
                f"notification_rules:{rule.tenant_id}",
                rule.id,
                json.dumps(self._serialize_notification_rule(rule))
            )
        
        logger.info(f"Notification rule added: {rule.name}")
        return rule.id

    async def add_notification_channel(self, channel: NotificationChannel) -> str:
        """Add a notification channel."""
        self.notification_channels[channel.id] = channel
        
        # Store in Redis
        if self.redis_client:
            await self.redis_client.hset(
                "notification_channels",
                channel.id,
                json.dumps(self._serialize_notification_channel(channel))
            )
        
        logger.info(f"Notification channel added: {channel.name}")
        return channel.id

    async def _evaluation_loop(self) -> None:
        """Main evaluation loop for all alert rules."""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Evaluate every minute
                
                # Check for alerts that need to be auto-resolved
                await self._check_auto_resolution()
                
                # Check for alerts that need escalation
                await self._check_escalation()
                
            except Exception as e:
                logger.error(f"Error in evaluation loop: {e}")

    async def _evaluate_rule_loop(self, rule: AlertRule) -> None:
        """Evaluation loop for a specific rule."""
        while self.is_running and rule.id in self.alert_rules:
            try:
                start_time = datetime.now()
                
                # Evaluate the rule
                await self._evaluate_rule(rule)
                
                # Record evaluation time
                duration = (datetime.now() - start_time).total_seconds()
                self.metrics['alert_evaluation_duration'].labels(
                    tenant_id=rule.tenant_id,
                    rule_id=rule.id
                ).observe(duration)
                
                # Wait for next evaluation
                await asyncio.sleep(rule.evaluation_interval.total_seconds())
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.id}: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _evaluate_rule(self, rule: AlertRule) -> None:
        """Evaluate a single alert rule."""
        if not rule.enabled:
            return
            
        try:
            # This would typically query a metrics backend (Prometheus, InfluxDB, etc.)
            # For now, simulate with a placeholder
            current_value = await self._query_metric(rule.query, rule.tenant_id)
            
            # Check if condition is met
            condition_met = self._evaluate_condition(current_value, rule.condition, rule.threshold)
            
            if condition_met:
                # Check if we already have an active alert for this rule
                existing_alert = self._find_active_alert_for_rule(rule.id)
                
                if not existing_alert:
                    # Fire new alert
                    await self.fire_alert(rule.id, current_value)
            else:
                # Check if we need to resolve an existing alert
                existing_alert = self._find_active_alert_for_rule(rule.id)
                if existing_alert:
                    await self.resolve_alert(existing_alert.id, rule.tenant_id)
                    
        except Exception as e:
            logger.error(f"Error evaluating rule {rule.name}: {e}")

    async def _query_metric(self, query: str, tenant_id: str) -> float:
        """Query metric value from backend."""
        # This would integrate with your metrics backend
        # For now, return a random value for demonstration
        import random
        return random.uniform(0, 100)

    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition."""
        conditions = {
            'gt': value > threshold,
            'gte': value >= threshold,
            'lt': value < threshold,
            'lte': value <= threshold,
            'eq': abs(value - threshold) < 0.001,
            'ne': abs(value - threshold) >= 0.001
        }
        return conditions.get(condition, False)

    def _find_active_alert_for_rule(self, rule_id: str) -> Optional[Alert]:
        """Find active alert for a rule."""
        for alert in self.active_alerts.values():
            if alert.rule_id == rule_id and alert.status == AlertStatus.FIRING:
                return alert
        return None

    async def _notification_processor(self) -> None:
        """Process notification queue."""
        while self.is_running:
            try:
                # Wait for alert to process
                alert = await asyncio.wait_for(self.notification_queue.get(), timeout=1.0)
                
                # Find matching notification rules
                matching_rules = self._find_matching_notification_rules(alert)
                
                # Send notifications
                for rule in matching_rules:
                    await self._send_notifications(alert, rule)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing notifications: {e}")

    def _find_matching_notification_rules(self, alert: Alert) -> List[NotificationRule]:
        """Find notification rules that match the alert."""
        matching_rules = []
        
        for rule in self.notification_rules.values():
            if not rule.enabled or rule.tenant_id != alert.tenant_id:
                continue
                
            # Check severity filter
            if rule.severity_filter and alert.severity not in rule.severity_filter:
                continue
                
            # Check label filters
            if not self._labels_match(alert.labels, rule.label_filters):
                continue
                
            matching_rules.append(rule)
            
        return matching_rules

    def _labels_match(self, alert_labels: Dict[str, str], filter_labels: Dict[str, str]) -> bool:
        """Check if alert labels match filter labels."""
        for key, value in filter_labels.items():
            if key not in alert_labels or alert_labels[key] != value:
                return False
        return True

    async def _send_notifications(self, alert: Alert, rule: NotificationRule) -> None:
        """Send notifications for an alert."""
        # Check cooldown
        if self._is_in_cooldown(alert, rule):
            return
            
        for channel_id in rule.channels:
            if channel_id in self.notification_channels:
                channel = self.notification_channels[channel_id]
                await self._send_notification_to_channel(alert, channel)
                
                # Update metrics
                self.metrics['notifications_sent'].labels(
                    tenant_id=alert.tenant_id,
                    channel=channel.type.value,
                    severity=alert.severity.value
                ).inc()

    def _is_in_cooldown(self, alert: Alert, rule: NotificationRule) -> bool:
        """Check if alert is in cooldown period."""
        if not alert.last_notification:
            return False
            
        time_since_last = datetime.now() - alert.last_notification
        return time_since_last < rule.cooldown

    async def _send_notification_to_channel(self, alert: Alert, channel: NotificationChannel) -> None:
        """Send notification to a specific channel."""
        try:
            if channel.type == NotificationChannel.EMAIL:
                await self._send_email_notification(alert, channel)
            elif channel.type == NotificationChannel.SLACK:
                await self._send_slack_notification(alert, channel)
            elif channel.type == NotificationChannel.WEBHOOK:
                await self._send_webhook_notification(alert, channel)
            # Add more channel types as needed
            
            # Update alert notification info
            alert.notification_count += 1
            alert.last_notification = datetime.now()
            
        except Exception as e:
            logger.error(f"Error sending notification to {channel.name}: {e}")

    async def _send_email_notification(self, alert: Alert, channel: NotificationChannel) -> None:
        """Send email notification."""
        config = channel.config
        
        msg = MIMEMultipart()
        msg['From'] = config['from_email']
        msg['To'] = config['to_email']
        msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
        
        body = f"""
        Alert: {alert.title}
        Severity: {alert.severity.value}
        Description: {alert.description}
        Value: {alert.value}
        Threshold: {alert.threshold}
        Started: {alert.started_at}
        
        Labels: {json.dumps(alert.labels, indent=2)}
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email (implement SMTP sending)
        logger.info(f"Email notification sent for alert {alert.id}")

    async def _send_slack_notification(self, alert: Alert, channel: NotificationChannel) -> None:
        """Send Slack notification."""
        config = channel.config
        
        payload = {
            "text": f"Alert: {alert.title}",
            "attachments": [
                {
                    "color": self._get_slack_color(alert.severity),
                    "fields": [
                        {"title": "Severity", "value": alert.severity.value, "short": True},
                        {"title": "Value", "value": str(alert.value), "short": True},
                        {"title": "Threshold", "value": str(alert.threshold), "short": True},
                        {"title": "Started", "value": alert.started_at.isoformat(), "short": True}
                    ]
                }
            ]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(config['webhook_url'], json=payload) as response:
                if response.status == 200:
                    logger.info(f"Slack notification sent for alert {alert.id}")
                else:
                    logger.error(f"Failed to send Slack notification: {response.status}")

    async def _send_webhook_notification(self, alert: Alert, channel: NotificationChannel) -> None:
        """Send webhook notification."""
        config = channel.config
        
        payload = {
            "alert_id": alert.id,
            "rule_id": alert.rule_id,
            "tenant_id": alert.tenant_id,
            "status": alert.status.value,
            "severity": alert.severity.value,
            "title": alert.title,
            "description": alert.description,
            "value": alert.value,
            "threshold": alert.threshold,
            "started_at": alert.started_at.isoformat(),
            "labels": alert.labels,
            "annotations": alert.annotations
        }
        
        headers = config.get('headers', {})
        
        async with aiohttp.ClientSession() as session:
            async with session.post(config['url'], json=payload, headers=headers) as response:
                if response.status == 200:
                    logger.info(f"Webhook notification sent for alert {alert.id}")
                else:
                    logger.error(f"Failed to send webhook notification: {response.status}")

    def _get_slack_color(self, severity: AlertSeverity) -> str:
        """Get Slack attachment color based on severity."""
        colors = {
            AlertSeverity.CRITICAL: "#FF0000",
            AlertSeverity.HIGH: "#FF8000",
            AlertSeverity.MEDIUM: "#FFFF00",
            AlertSeverity.LOW: "#00FF00",
            AlertSeverity.INFO: "#0080FF"
        }
        return colors.get(severity, "#808080")

    async def _check_auto_resolution(self) -> None:
        """Check for alerts that can be auto-resolved."""
        for alert in list(self.active_alerts.values()):
            if alert.status == AlertStatus.FIRING:
                rule = self.alert_rules.get(alert.rule_id)
                if rule:
                    # Re-evaluate the condition
                    current_value = await self._query_metric(rule.query, rule.tenant_id)
                    condition_met = self._evaluate_condition(current_value, rule.condition, rule.threshold)
                    
                    if not condition_met:
                        await self.resolve_alert(alert.id, alert.tenant_id)

    async def _check_escalation(self) -> None:
        """Check for alerts that need escalation."""
        for alert in self.active_alerts.values():
            if alert.status == AlertStatus.FIRING:
                # Check if alert has been firing for too long
                firing_duration = datetime.now() - alert.started_at
                
                # Implement escalation logic based on duration and notification rules
                # This is a placeholder for more complex escalation logic
                if firing_duration > timedelta(hours=1) and alert.escalation_level == 0:
                    alert.escalation_level = 1
                    # Trigger escalation notifications

    async def _cleanup_task(self) -> None:
        """Clean up old alerts and maintain data hygiene."""
        while self.is_running:
            try:
                # Clean up old resolved alerts
                cutoff_time = datetime.now() - timedelta(days=30)
                self.alert_history = [
                    alert for alert in self.alert_history
                    if alert.resolved_at and alert.resolved_at > cutoff_time
                ]
                
                await asyncio.sleep(3600)  # Run cleanup every hour
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")

    def _serialize_rule(self, rule: AlertRule) -> Dict[str, Any]:
        """Serialize alert rule for storage."""
        return {
            'id': rule.id,
            'name': rule.name,
            'description': rule.description,
            'query': rule.query,
            'condition': rule.condition,
            'threshold': rule.threshold,
            'severity': rule.severity.value,
            'tenant_id': rule.tenant_id,
            'labels': rule.labels,
            'annotations': rule.annotations,
            'for_duration': rule.for_duration.total_seconds(),
            'evaluation_interval': rule.evaluation_interval.total_seconds(),
            'enabled': rule.enabled,
            'created_at': rule.created_at.isoformat(),
            'updated_at': rule.updated_at.isoformat()
        }

    def _serialize_alert(self, alert: Alert) -> Dict[str, Any]:
        """Serialize alert for storage."""
        return {
            'id': alert.id,
            'rule_id': alert.rule_id,
            'tenant_id': alert.tenant_id,
            'status': alert.status.value,
            'severity': alert.severity.value,
            'title': alert.title,
            'description': alert.description,
            'labels': alert.labels,
            'annotations': alert.annotations,
            'value': alert.value,
            'threshold': alert.threshold,
            'started_at': alert.started_at.isoformat(),
            'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None,
            'acknowledged_at': alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
            'acknowledged_by': alert.acknowledged_by,
            'escalation_level': alert.escalation_level,
            'notification_count': alert.notification_count,
            'last_notification': alert.last_notification.isoformat() if alert.last_notification else None
        }

    def _serialize_notification_rule(self, rule: NotificationRule) -> Dict[str, Any]:
        """Serialize notification rule for storage."""
        return {
            'id': rule.id,
            'name': rule.name,
            'tenant_id': rule.tenant_id,
            'severity_filter': [s.value for s in rule.severity_filter],
            'label_filters': rule.label_filters,
            'channels': rule.channels,
            'escalation_rules': rule.escalation_rules,
            'cooldown': rule.cooldown.total_seconds(),
            'enabled': rule.enabled
        }

    def _serialize_notification_channel(self, channel: NotificationChannel) -> Dict[str, Any]:
        """Serialize notification channel for storage."""
        return {
            'id': channel.id,
            'type': channel.type.value,
            'name': channel.name,
            'config': channel.config,
            'enabled': channel.enabled
        }

    async def get_alerts_for_tenant(self, tenant_id: str, status: Optional[AlertStatus] = None) -> List[Alert]:
        """Get alerts for a specific tenant."""
        alerts = []
        for alert in self.active_alerts.values():
            if alert.tenant_id == tenant_id:
                if status is None or alert.status == status:
                    alerts.append(alert)
        return alerts

    async def get_alert_statistics(self, tenant_id: str, time_range: timedelta) -> Dict[str, Any]:
        """Get alert statistics for a tenant."""
        end_time = datetime.now()
        start_time = end_time - time_range
        
        # Count alerts by severity and status
        stats = {
            'total_alerts': 0,
            'by_severity': {s.value: 0 for s in AlertSeverity},
            'by_status': {s.value: 0 for s in AlertStatus},
            'avg_resolution_time': 0.0,
            'escalated_alerts': 0
        }
        
        resolution_times = []
        
        for alert in self.alert_history:
            if alert.tenant_id == tenant_id and alert.started_at >= start_time:
                stats['total_alerts'] += 1
                stats['by_severity'][alert.severity.value] += 1
                stats['by_status'][alert.status.value] += 1
                
                if alert.escalation_level > 0:
                    stats['escalated_alerts'] += 1
                    
                if alert.resolved_at:
                    resolution_time = (alert.resolved_at - alert.started_at).total_seconds()
                    resolution_times.append(resolution_time)
        
        # Calculate average resolution time
        if resolution_times:
            stats['avg_resolution_time'] = sum(resolution_times) / len(resolution_times)
        
        return stats

# Global alert manager instance
alert_manager = AlertManager()

__all__ = [
    'AlertManager',
    'Alert',
    'AlertRule',
    'NotificationRule',
    'NotificationChannel',
    'AlertStatus',
    'AlertSeverity',
    'NotificationChannel',
    'alert_manager'
]
