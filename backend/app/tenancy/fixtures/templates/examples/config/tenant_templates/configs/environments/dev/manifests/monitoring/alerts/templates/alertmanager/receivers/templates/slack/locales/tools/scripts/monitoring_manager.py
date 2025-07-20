#!/usr/bin/env python3
"""
Spotify AI Agent - Advanced Monitoring Manager
==============================================

Enterprise-grade monitoring management system providing:
- Real-time multi-tenant monitoring
- Alertmanager configuration and integration
- Slack notification management
- Advanced metrics collection and analysis
- Performance optimization and auto-scaling
- Security monitoring and compliance

Author: Fahed Mlaiel (Lead Developer + AI Architect)
Team: Expert Development Team
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import yaml
import redis
import prometheus_client
from prometheus_client import Gauge, Counter, Histogram
import aiohttp
import asyncpg
from pydantic import BaseModel, Field, validator
from sqlalchemy import create_engine, text
from slack_sdk import WebhookClient
from slack_sdk.webhook import WebhookResponse

# Configure logging
logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"


class MonitoringStatus(str, Enum):
    """Monitoring system status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    ERROR = "error"


@dataclass
class TenantMetrics:
    """Tenant-specific metrics data"""
    tenant_id: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    api_response_time: float
    active_users: int
    database_connections: int
    error_rate: float
    timestamp: datetime


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    expression: str
    severity: AlertSeverity
    threshold: float
    duration: timedelta
    labels: Dict[str, str]
    annotations: Dict[str, str]
    enabled: bool = True


class MonitoringConfig(BaseModel):
    """Monitoring system configuration"""
    tenant_id: str = Field(..., min_length=1)
    environment: str = Field(default="development")
    prometheus_url: str = Field(default="http://localhost:9090")
    alertmanager_url: str = Field(default="http://localhost:9093")
    redis_url: str = Field(default="redis://localhost:6379")
    database_url: str = Field(default="postgresql://user:pass@localhost/db")
    metrics_retention: int = Field(default=2592000, ge=86400)  # 30 days
    collection_interval: int = Field(default=15, ge=5)  # seconds
    alert_evaluation_interval: int = Field(default=60, ge=15)  # seconds
    
    @validator('environment')
    def validate_environment(cls, v):
        valid_envs = ['development', 'staging', 'production']
        if v not in valid_envs:
            raise ValueError(f'Environment must be one of {valid_envs}')
        return v


class MonitoringManager:
    """
    Advanced multi-tenant monitoring manager with enterprise features
    """
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.tenant_id = config.tenant_id
        self.environment = config.environment
        self.status = MonitoringStatus.INACTIVE
        
        # Initialize components
        self._init_redis()
        self._init_prometheus_metrics()
        self._init_database()
        
        # Alert rules storage
        self.alert_rules: Dict[str, AlertRule] = {}
        
        # Metrics storage
        self.tenant_metrics: Dict[str, TenantMetrics] = {}
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._alert_evaluation_task: Optional[asyncio.Task] = None
        
        logger.info(f"MonitoringManager initialized for tenant {self.tenant_id}")

    def _init_redis(self):
        """Initialize Redis connection for caching and message queuing"""
        try:
            self.redis_client = redis.from_url(
                self.config.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        self.metrics = {
            'cpu_usage': Gauge(
                'tenant_cpu_usage_percent',
                'CPU usage percentage per tenant',
                ['tenant_id', 'environment']
            ),
            'memory_usage': Gauge(
                'tenant_memory_usage_percent', 
                'Memory usage percentage per tenant',
                ['tenant_id', 'environment']
            ),
            'disk_usage': Gauge(
                'tenant_disk_usage_percent',
                'Disk usage percentage per tenant', 
                ['tenant_id', 'environment']
            ),
            'api_response_time': Histogram(
                'tenant_api_response_time_seconds',
                'API response time per tenant',
                ['tenant_id', 'environment', 'endpoint']
            ),
            'active_users': Gauge(
                'tenant_active_users_count',
                'Number of active users per tenant',
                ['tenant_id', 'environment']
            ),
            'database_connections': Gauge(
                'tenant_database_connections_count',
                'Number of database connections per tenant',
                ['tenant_id', 'environment']
            ),
            'error_rate': Gauge(
                'tenant_error_rate_percent',
                'Error rate percentage per tenant',
                ['tenant_id', 'environment']
            ),
            'alerts_triggered': Counter(
                'tenant_alerts_triggered_total',
                'Total alerts triggered per tenant',
                ['tenant_id', 'environment', 'severity']
            )
        }
        logger.info("Prometheus metrics initialized")

    def _init_database(self):
        """Initialize database connection"""
        try:
            self.engine = create_engine(self.config.database_url)
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    async def start(self) -> bool:
        """Start the monitoring system"""
        try:
            self.status = MonitoringStatus.ACTIVE
            
            # Start monitoring tasks
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._alert_evaluation_task = asyncio.create_task(self._alert_evaluation_loop())
            
            # Load default alert rules
            await self._load_default_alert_rules()
            
            logger.info(f"Monitoring system started for tenant {self.tenant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start monitoring system: {e}")
            self.status = MonitoringStatus.ERROR
            return False

    async def stop(self) -> bool:
        """Stop the monitoring system"""
        try:
            self.status = MonitoringStatus.INACTIVE
            
            # Cancel background tasks
            if self._monitoring_task:
                self._monitoring_task.cancel()
            if self._alert_evaluation_task:
                self._alert_evaluation_task.cancel()
                
            logger.info(f"Monitoring system stopped for tenant {self.tenant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop monitoring system: {e}")
            return False

    async def _monitoring_loop(self):
        """Main monitoring loop for collecting metrics"""
        while self.status == MonitoringStatus.ACTIVE:
            try:
                # Collect system metrics
                metrics = await self._collect_tenant_metrics()
                
                # Update Prometheus metrics
                await self._update_prometheus_metrics(metrics)
                
                # Store metrics in Redis
                await self._store_metrics_redis(metrics)
                
                # Store metrics in database
                await self._store_metrics_database(metrics)
                
                await asyncio.sleep(self.config.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.config.collection_interval)

    async def _alert_evaluation_loop(self):
        """Alert evaluation loop"""
        while self.status == MonitoringStatus.ACTIVE:
            try:
                # Evaluate all alert rules
                for rule_name, rule in self.alert_rules.items():
                    if rule.enabled:
                        await self._evaluate_alert_rule(rule)
                
                await asyncio.sleep(self.config.alert_evaluation_interval)
                
            except Exception as e:
                logger.error(f"Error in alert evaluation loop: {e}")
                await asyncio.sleep(self.config.alert_evaluation_interval)

    async def _collect_tenant_metrics(self) -> TenantMetrics:
        """Collect metrics for the tenant"""
        # Simulate metrics collection - replace with actual implementation
        import psutil
        import random
        
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = TenantMetrics(
            tenant_id=self.tenant_id,
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            disk_usage=(disk.used / disk.total) * 100,
            api_response_time=random.uniform(0.05, 0.3),  # Simulated
            active_users=random.randint(10, 1000),  # Simulated
            database_connections=random.randint(5, 50),  # Simulated
            error_rate=random.uniform(0, 5),  # Simulated
            timestamp=datetime.utcnow()
        )
        
        self.tenant_metrics[self.tenant_id] = metrics
        return metrics

    async def _update_prometheus_metrics(self, metrics: TenantMetrics):
        """Update Prometheus metrics"""
        labels = [self.tenant_id, self.environment]
        
        self.metrics['cpu_usage'].labels(*labels).set(metrics.cpu_usage)
        self.metrics['memory_usage'].labels(*labels).set(metrics.memory_usage)
        self.metrics['disk_usage'].labels(*labels).set(metrics.disk_usage)
        self.metrics['active_users'].labels(*labels).set(metrics.active_users)
        self.metrics['database_connections'].labels(*labels).set(metrics.database_connections)
        self.metrics['error_rate'].labels(*labels).set(metrics.error_rate)

    async def _store_metrics_redis(self, metrics: TenantMetrics):
        """Store metrics in Redis for quick access"""
        key = f"metrics:{self.tenant_id}:latest"
        value = json.dumps(asdict(metrics), default=str)
        
        # Store with TTL
        self.redis_client.setex(key, 3600, value)  # 1 hour TTL
        
        # Store in time series
        ts_key = f"metrics:{self.tenant_id}:timeseries"
        timestamp = int(metrics.timestamp.timestamp())
        self.redis_client.zadd(ts_key, {value: timestamp})
        
        # Keep only recent data
        cutoff = timestamp - self.config.metrics_retention
        self.redis_client.zremrangebyscore(ts_key, 0, cutoff)

    async def _store_metrics_database(self, metrics: TenantMetrics):
        """Store metrics in database for long-term storage"""
        query = """
        INSERT INTO tenant_metrics (
            tenant_id, cpu_usage, memory_usage, disk_usage,
            api_response_time, active_users, database_connections,
            error_rate, timestamp
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9
        )
        """
        
        try:
            # Use asyncpg for async database operations
            conn = await asyncpg.connect(self.config.database_url)
            await conn.execute(
                query,
                metrics.tenant_id,
                metrics.cpu_usage,
                metrics.memory_usage,
                metrics.disk_usage,
                metrics.api_response_time,
                metrics.active_users,
                metrics.database_connections,
                metrics.error_rate,
                metrics.timestamp
            )
            await conn.close()
        except Exception as e:
            logger.error(f"Failed to store metrics in database: {e}")

    async def _load_default_alert_rules(self):
        """Load default alert rules"""
        default_rules = [
            AlertRule(
                name="high_cpu_usage",
                expression="cpu_usage > threshold",
                severity=AlertSeverity.WARNING,
                threshold=80.0,
                duration=timedelta(minutes=5),
                labels={"team": "infrastructure", "service": "monitoring"},
                annotations={"description": "High CPU usage detected for tenant {tenant_id}"}
            ),
            AlertRule(
                name="high_memory_usage",
                expression="memory_usage > threshold", 
                severity=AlertSeverity.WARNING,
                threshold=85.0,
                duration=timedelta(minutes=5),
                labels={"team": "infrastructure", "service": "monitoring"},
                annotations={"description": "High memory usage detected for tenant {tenant_id}"}
            ),
            AlertRule(
                name="high_error_rate",
                expression="error_rate > threshold",
                severity=AlertSeverity.CRITICAL,
                threshold=10.0,
                duration=timedelta(minutes=2),
                labels={"team": "sre", "service": "api"},
                annotations={"description": "High error rate detected for tenant {tenant_id}"}
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.name] = rule
            
        logger.info(f"Loaded {len(default_rules)} default alert rules")

    async def _evaluate_alert_rule(self, rule: AlertRule):
        """Evaluate a single alert rule"""
        try:
            metrics = self.tenant_metrics.get(self.tenant_id)
            if not metrics:
                return
                
            # Simple rule evaluation - extend for more complex expressions
            triggered = False
            current_value = 0.0
            
            if "cpu_usage" in rule.expression:
                current_value = metrics.cpu_usage
                triggered = current_value > rule.threshold
            elif "memory_usage" in rule.expression:
                current_value = metrics.memory_usage  
                triggered = current_value > rule.threshold
            elif "error_rate" in rule.expression:
                current_value = metrics.error_rate
                triggered = current_value > rule.threshold
                
            if triggered:
                await self._trigger_alert(rule, current_value)
                
        except Exception as e:
            logger.error(f"Failed to evaluate alert rule {rule.name}: {e}")

    async def _trigger_alert(self, rule: AlertRule, current_value: float):
        """Trigger an alert"""
        alert_data = {
            "rule_name": rule.name,
            "tenant_id": self.tenant_id,
            "severity": rule.severity.value,
            "threshold": rule.threshold,
            "current_value": current_value,
            "timestamp": datetime.utcnow().isoformat(),
            "labels": rule.labels,
            "annotations": rule.annotations
        }
        
        # Store alert in Redis
        alert_key = f"alerts:{self.tenant_id}:{rule.name}:{int(time.time())}"
        self.redis_client.setex(alert_key, 86400, json.dumps(alert_data))  # 24h TTL
        
        # Update metrics
        self.metrics['alerts_triggered'].labels(
            self.tenant_id, 
            self.environment, 
            rule.severity.value
        ).inc()
        
        logger.warning(f"Alert triggered: {rule.name} for tenant {self.tenant_id}")

    def get_tenant_metrics(self) -> Optional[TenantMetrics]:
        """Get latest metrics for the tenant"""
        return self.tenant_metrics.get(self.tenant_id)

    def get_status(self) -> MonitoringStatus:
        """Get monitoring system status"""
        return self.status

    async def add_alert_rule(self, rule: AlertRule) -> bool:
        """Add a new alert rule"""
        try:
            self.alert_rules[rule.name] = rule
            logger.info(f"Added alert rule: {rule.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add alert rule {rule.name}: {e}")
            return False

    async def remove_alert_rule(self, rule_name: str) -> bool:
        """Remove an alert rule"""
        try:
            if rule_name in self.alert_rules:
                del self.alert_rules[rule_name]
                logger.info(f"Removed alert rule: {rule_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove alert rule {rule_name}: {e}")
            return False


class AlertmanagerConfigManager:
    """
    Manages Alertmanager configuration for multi-tenant environments
    """
    
    def __init__(self, alertmanager_url: str, config_path: str = "/etc/alertmanager"):
        self.alertmanager_url = alertmanager_url
        self.config_path = config_path
        self.config = {}
        
    async def load_config(self) -> Dict[str, Any]:
        """Load Alertmanager configuration"""
        try:
            config_file = f"{self.config_path}/alertmanager.yml"
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
            return self.config
        except Exception as e:
            logger.error(f"Failed to load Alertmanager config: {e}")
            return {}

    async def update_tenant_config(self, tenant_id: str, config: Dict[str, Any]) -> bool:
        """Update configuration for a specific tenant"""
        try:
            # Implement tenant-specific configuration updates
            if 'route' not in self.config:
                self.config['route'] = {'group_by': ['alertname'], 'routes': []}
                
            # Add tenant-specific route
            tenant_route = {
                'match': {'tenant_id': tenant_id},
                'receiver': f'slack-{tenant_id}',
                'group_wait': '10s',
                'group_interval': '5m',
                'repeat_interval': '12h'
            }
            
            self.config['route']['routes'].append(tenant_route)
            
            # Save configuration
            await self._save_config()
            await self._reload_alertmanager()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update tenant config for {tenant_id}: {e}")
            return False

    async def _save_config(self):
        """Save configuration to file"""
        config_file = f"{self.config_path}/alertmanager.yml"
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    async def _reload_alertmanager(self):
        """Reload Alertmanager configuration"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.alertmanager_url}/-/reload") as response:
                    if response.status == 200:
                        logger.info("Alertmanager configuration reloaded")
                    else:
                        logger.error(f"Failed to reload Alertmanager: {response.status}")
        except Exception as e:
            logger.error(f"Failed to reload Alertmanager: {e}")


class SlackNotificationManager:
    """
    Advanced Slack notification manager with multi-tenant support
    """
    
    def __init__(self, webhook_url: str, default_channel: str = "#alerts"):
        self.webhook_url = webhook_url
        self.default_channel = default_channel
        self.client = WebhookClient(webhook_url)
        self.enabled = False
        
    def enable_notifications(self):
        """Enable Slack notifications"""
        self.enabled = True
        logger.info("Slack notifications enabled")

    def disable_notifications(self):
        """Disable Slack notifications"""
        self.enabled = False
        logger.info("Slack notifications disabled")

    async def send_alert(self, alert_data: Dict[str, Any], channel: Optional[str] = None) -> bool:
        """Send alert notification to Slack"""
        if not self.enabled:
            return False
            
        try:
            channel = channel or self.default_channel
            
            # Format alert message
            message = self._format_alert_message(alert_data)
            
            # Send to Slack
            response: WebhookResponse = self.client.send(
                text=message['text'],
                blocks=message.get('blocks', []),
                channel=channel
            )
            
            if response.status_code == 200:
                logger.info(f"Alert sent to Slack channel {channel}")
                return True
            else:
                logger.error(f"Failed to send alert to Slack: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False

    def _format_alert_message(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format alert data into Slack message"""
        severity = alert_data.get('severity', 'info')
        tenant_id = alert_data.get('tenant_id', 'unknown')
        rule_name = alert_data.get('rule_name', 'unknown')
        current_value = alert_data.get('current_value', 0)
        threshold = alert_data.get('threshold', 0)
        
        # Emoji mapping for severity
        emoji_map = {
            'critical': '🚨',
            'warning': '⚠️', 
            'info': 'ℹ️',
            'debug': '🐛'
        }
        
        emoji = emoji_map.get(severity, 'ℹ️')
        
        # Color mapping for severity
        color_map = {
            'critical': '#FF0000',
            'warning': '#FFA500',
            'info': '#00FF00',
            'debug': '#0000FF'
        }
        
        color = color_map.get(severity, '#808080')
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} Alert: {rule_name.replace('_', ' ').title()}"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Tenant:*\n{tenant_id}"
                    },
                    {
                        "type": "mrkdwn", 
                        "text": f"*Severity:*\n{severity.upper()}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Current Value:*\n{current_value:.2f}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Threshold:*\n{threshold:.2f}"
                    }
                ]
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Triggered at {alert_data.get('timestamp', 'unknown')}"
                    }
                ]
            }
        ]
        
        return {
            "text": f"{emoji} Alert: {rule_name} for tenant {tenant_id}",
            "blocks": blocks
        }
