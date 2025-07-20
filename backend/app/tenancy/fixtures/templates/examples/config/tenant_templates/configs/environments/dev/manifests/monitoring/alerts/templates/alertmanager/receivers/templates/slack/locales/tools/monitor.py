#!/usr/bin/env python3
"""
Enterprise Monitoring and Alerting Tool
Advanced monitoring system with intelligent alerting and anomaly detection.

This tool provides comprehensive monitoring capabilities:
- Real-time metrics collection
- Intelligent alerting
- Anomaly detection
- Performance analysis
- Automated remediation
- SLA monitoring
"""

import asyncio
import json
import logging
import time
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import sys
import os
from dataclasses import dataclass, asdict
from enum import Enum

# Add the schemas directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'schemas'))

from monitoring_schemas import (
    MonitoringConfigSchema,
    MetricConfigSchema,
    HealthCheckConfigSchema
)
from alert_schemas import (
    AlertRuleSchema,
    AlertConditionSchema,
    AlertThresholdSchema
)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class MetricData:
    """Metric data structure."""
    timestamp: datetime
    value: float
    labels: Dict[str, str]


@dataclass
class Alert:
    """Alert data structure."""
    alert_id: str
    rule_name: str
    severity: AlertSeverity
    message: str
    tenant_id: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    labels: Dict[str, str]


class MetricsCollector:
    """Advanced metrics collection engine."""
    
    def __init__(self, config: MonitoringConfigSchema):
        self.config = config
        self.logger = logging.getLogger("metrics_collector")
        self.metrics_buffer: Dict[str, List[MetricData]] = {}
    
    async def collect_system_metrics(self, tenant_id: str) -> Dict[str, MetricData]:
        """Collect system-level metrics."""
        metrics = {}
        
        # CPU metrics
        cpu_usage = await self._get_cpu_usage(tenant_id)
        metrics["cpu_usage_percent"] = MetricData(
            timestamp=datetime.utcnow(),
            value=cpu_usage,
            labels={"tenant_id": tenant_id, "metric_type": "system"}
        )
        
        # Memory metrics
        memory_usage = await self._get_memory_usage(tenant_id)
        metrics["memory_usage_percent"] = MetricData(
            timestamp=datetime.utcnow(),
            value=memory_usage,
            labels={"tenant_id": tenant_id, "metric_type": "system"}
        )
        
        # Disk metrics
        disk_usage = await self._get_disk_usage(tenant_id)
        metrics["disk_usage_percent"] = MetricData(
            timestamp=datetime.utcnow(),
            value=disk_usage,
            labels={"tenant_id": tenant_id, "metric_type": "system"}
        )
        
        # Network metrics
        network_throughput = await self._get_network_throughput(tenant_id)
        metrics["network_throughput_mbps"] = MetricData(
            timestamp=datetime.utcnow(),
            value=network_throughput,
            labels={"tenant_id": tenant_id, "metric_type": "network"}
        )
        
        return metrics
    
    async def collect_application_metrics(self, tenant_id: str) -> Dict[str, MetricData]:
        """Collect application-level metrics."""
        metrics = {}
        
        # Response time metrics
        response_time = await self._get_response_time(tenant_id)
        metrics["response_time_ms"] = MetricData(
            timestamp=datetime.utcnow(),
            value=response_time,
            labels={"tenant_id": tenant_id, "metric_type": "application"}
        )
        
        # Request rate metrics
        request_rate = await self._get_request_rate(tenant_id)
        metrics["request_rate_per_second"] = MetricData(
            timestamp=datetime.utcnow(),
            value=request_rate,
            labels={"tenant_id": tenant_id, "metric_type": "application"}
        )
        
        # Error rate metrics
        error_rate = await self._get_error_rate(tenant_id)
        metrics["error_rate_percent"] = MetricData(
            timestamp=datetime.utcnow(),
            value=error_rate,
            labels={"tenant_id": tenant_id, "metric_type": "application"}
        )
        
        # Database metrics
        db_connection_count = await self._get_db_connection_count(tenant_id)
        metrics["db_connection_count"] = MetricData(
            timestamp=datetime.utcnow(),
            value=db_connection_count,
            labels={"tenant_id": tenant_id, "metric_type": "database"}
        )
        
        return metrics
    
    async def collect_business_metrics(self, tenant_id: str) -> Dict[str, MetricData]:
        """Collect business-level metrics."""
        metrics = {}
        
        # User activity metrics
        active_users = await self._get_active_users(tenant_id)
        metrics["active_users_count"] = MetricData(
            timestamp=datetime.utcnow(),
            value=active_users,
            labels={"tenant_id": tenant_id, "metric_type": "business"}
        )
        
        # Transaction metrics
        transaction_volume = await self._get_transaction_volume(tenant_id)
        metrics["transaction_volume"] = MetricData(
            timestamp=datetime.utcnow(),
            value=transaction_volume,
            labels={"tenant_id": tenant_id, "metric_type": "business"}
        )
        
        # Revenue metrics
        revenue_rate = await self._get_revenue_rate(tenant_id)
        metrics["revenue_rate_per_hour"] = MetricData(
            timestamp=datetime.utcnow(),
            value=revenue_rate,
            labels={"tenant_id": tenant_id, "metric_type": "business"}
        )
        
        return metrics
    
    # Metric collection methods (simulated for demo)
    async def _get_cpu_usage(self, tenant_id: str) -> float:
        """Get CPU usage percentage."""
        # Simulate varying CPU usage
        import random
        return random.uniform(20.0, 85.0)
    
    async def _get_memory_usage(self, tenant_id: str) -> float:
        """Get memory usage percentage."""
        import random
        return random.uniform(30.0, 80.0)
    
    async def _get_disk_usage(self, tenant_id: str) -> float:
        """Get disk usage percentage."""
        import random
        return random.uniform(40.0, 70.0)
    
    async def _get_network_throughput(self, tenant_id: str) -> float:
        """Get network throughput in Mbps."""
        import random
        return random.uniform(10.0, 100.0)
    
    async def _get_response_time(self, tenant_id: str) -> float:
        """Get average response time in milliseconds."""
        import random
        return random.uniform(50.0, 500.0)
    
    async def _get_request_rate(self, tenant_id: str) -> float:
        """Get request rate per second."""
        import random
        return random.uniform(10.0, 1000.0)
    
    async def _get_error_rate(self, tenant_id: str) -> float:
        """Get error rate percentage."""
        import random
        return random.uniform(0.1, 5.0)
    
    async def _get_db_connection_count(self, tenant_id: str) -> float:
        """Get database connection count."""
        import random
        return random.uniform(5.0, 50.0)
    
    async def _get_active_users(self, tenant_id: str) -> float:
        """Get active users count."""
        import random
        return random.uniform(100.0, 10000.0)
    
    async def _get_transaction_volume(self, tenant_id: str) -> float:
        """Get transaction volume."""
        import random
        return random.uniform(50.0, 5000.0)
    
    async def _get_revenue_rate(self, tenant_id: str) -> float:
        """Get revenue rate per hour."""
        import random
        return random.uniform(1000.0, 50000.0)


class AnomalyDetector:
    """Advanced anomaly detection engine."""
    
    def __init__(self, sensitivity: float = 0.95):
        self.sensitivity = sensitivity
        self.baseline_data: Dict[str, List[float]] = {}
        self.logger = logging.getLogger("anomaly_detector")
    
    def update_baseline(self, metric_name: str, values: List[float]):
        """Update baseline data for a metric."""
        if metric_name not in self.baseline_data:
            self.baseline_data[metric_name] = []
        
        # Keep last 1000 values for baseline
        self.baseline_data[metric_name].extend(values)
        self.baseline_data[metric_name] = self.baseline_data[metric_name][-1000:]
    
    def detect_anomaly(self, metric_name: str, current_value: float) -> Tuple[bool, float]:
        """Detect anomaly using statistical analysis."""
        if metric_name not in self.baseline_data or len(self.baseline_data[metric_name]) < 10:
            return False, 0.0
        
        baseline = self.baseline_data[metric_name]
        
        # Calculate statistical measures
        mean = statistics.mean(baseline)
        std_dev = statistics.stdev(baseline) if len(baseline) > 1 else 0
        
        # Z-score based anomaly detection
        if std_dev == 0:
            return False, 0.0
        
        z_score = abs(current_value - mean) / std_dev
        threshold = 2.5  # 2.5 standard deviations
        
        is_anomaly = z_score > threshold
        confidence = min(z_score / threshold, 1.0) if is_anomaly else 0.0
        
        if is_anomaly:
            self.logger.warning(
                f"Anomaly detected for {metric_name}: "
                f"value={current_value}, mean={mean:.2f}, z_score={z_score:.2f}"
            )
        
        return is_anomaly, confidence
    
    def detect_trend_anomaly(self, metric_name: str, values: List[float]) -> bool:
        """Detect trend-based anomalies."""
        if len(values) < 5:
            return False
        
        # Simple trend analysis
        recent_values = values[-5:]
        older_values = values[-10:-5] if len(values) >= 10 else values[:-5]
        
        if not older_values:
            return False
        
        recent_avg = statistics.mean(recent_values)
        older_avg = statistics.mean(older_values)
        
        # Detect significant trend changes
        change_percent = abs(recent_avg - older_avg) / older_avg * 100
        
        return change_percent > 50  # 50% change threshold


class AlertManager:
    """Advanced alert management system."""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.logger = logging.getLogger("alert_manager")
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_channels = self._setup_notification_channels()
    
    def _setup_notification_channels(self) -> Dict[str, Any]:
        """Setup notification channels."""
        return {
            "slack": {"webhook_url": "https://hooks.slack.com/services/..."},
            "email": {"smtp_server": "smtp.example.com", "port": 587},
            "pagerduty": {"service_key": "your-pagerduty-key"},
            "webhook": {"url": "https://your-webhook-endpoint.com"}
        }
    
    async def evaluate_alert_rules(self, tenant_id: str, metrics: Dict[str, MetricData]) -> List[Alert]:
        """Evaluate alert rules against current metrics."""
        alerts = []
        
        # Define alert rules (in practice, these would be loaded from configuration)
        alert_rules = [
            {
                "name": "high_cpu_usage",
                "metric": "cpu_usage_percent",
                "condition": "greater_than",
                "threshold": 80.0,
                "severity": AlertSeverity.WARNING,
                "duration_minutes": 5
            },
            {
                "name": "critical_cpu_usage",
                "metric": "cpu_usage_percent",
                "condition": "greater_than",
                "threshold": 95.0,
                "severity": AlertSeverity.CRITICAL,
                "duration_minutes": 1
            },
            {
                "name": "high_memory_usage",
                "metric": "memory_usage_percent",
                "condition": "greater_than",
                "threshold": 85.0,
                "severity": AlertSeverity.WARNING,
                "duration_minutes": 5
            },
            {
                "name": "high_error_rate",
                "metric": "error_rate_percent",
                "condition": "greater_than",
                "threshold": 5.0,
                "severity": AlertSeverity.CRITICAL,
                "duration_minutes": 2
            },
            {
                "name": "slow_response_time",
                "metric": "response_time_ms",
                "condition": "greater_than",
                "threshold": 1000.0,
                "severity": AlertSeverity.WARNING,
                "duration_minutes": 3
            }
        ]
        
        for rule in alert_rules:
            metric_name = rule["metric"]
            if metric_name in metrics:
                metric_data = metrics[metric_name]
                
                # Evaluate condition
                alert_triggered = self._evaluate_condition(
                    metric_data.value,
                    rule["condition"],
                    rule["threshold"]
                )
                
                if alert_triggered:
                    alert = Alert(
                        alert_id=f"{rule['name']}_{tenant_id}_{int(time.time())}",
                        rule_name=rule["name"],
                        severity=rule["severity"],
                        message=f"{rule['name']}: {metric_name} is {metric_data.value:.2f}, threshold is {rule['threshold']}",
                        tenant_id=tenant_id,
                        metric_name=metric_name,
                        current_value=metric_data.value,
                        threshold_value=rule["threshold"],
                        timestamp=datetime.utcnow(),
                        labels=metric_data.labels
                    )
                    alerts.append(alert)
        
        return alerts
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition."""
        if condition == "greater_than":
            return value > threshold
        elif condition == "less_than":
            return value < threshold
        elif condition == "equal_to":
            return abs(value - threshold) < 0.001
        elif condition == "not_equal_to":
            return abs(value - threshold) >= 0.001
        else:
            return False
    
    async def process_alerts(self, alerts: List[Alert]):
        """Process and manage alerts."""
        for alert in alerts:
            # Check if alert is already active
            if alert.alert_id in self.active_alerts:
                continue
            
            # Add to active alerts
            self.active_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)
            
            # Log alert
            self.logger.warning(f"Alert triggered: {alert.message}")
            
            # Send notifications
            await self._send_notifications(alert)
            
            # Trigger automated remediation if configured
            await self._trigger_automated_remediation(alert)
    
    async def _send_notifications(self, alert: Alert):
        """Send alert notifications to configured channels."""
        notification_tasks = []
        
        # Determine notification channels based on severity
        channels = self._get_notification_channels_for_severity(alert.severity)
        
        for channel in channels:
            if channel == "slack":
                notification_tasks.append(self._send_slack_notification(alert))
            elif channel == "email":
                notification_tasks.append(self._send_email_notification(alert))
            elif channel == "pagerduty":
                notification_tasks.append(self._send_pagerduty_notification(alert))
            elif channel == "webhook":
                notification_tasks.append(self._send_webhook_notification(alert))
        
        # Send all notifications concurrently
        if notification_tasks:
            await asyncio.gather(*notification_tasks)
    
    def _get_notification_channels_for_severity(self, severity: AlertSeverity) -> List[str]:
        """Get notification channels based on alert severity."""
        if severity == AlertSeverity.EMERGENCY:
            return ["slack", "email", "pagerduty", "webhook"]
        elif severity == AlertSeverity.CRITICAL:
            return ["slack", "email", "pagerduty"]
        elif severity == AlertSeverity.WARNING:
            return ["slack", "email"]
        else:
            return ["slack"]
    
    async def _send_slack_notification(self, alert: Alert):
        """Send Slack notification."""
        self.logger.info(f"Sending Slack notification for alert: {alert.alert_id}")
        
        # Simulate Slack notification
        await asyncio.sleep(0.5)
        
        # In a real implementation, this would send an HTTP request to Slack webhook
        slack_message = {
            "text": f"🚨 Alert: {alert.rule_name}",
            "attachments": [
                {
                    "color": "danger" if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY] else "warning",
                    "fields": [
                        {"title": "Tenant", "value": alert.tenant_id, "short": True},
                        {"title": "Metric", "value": alert.metric_name, "short": True},
                        {"title": "Current Value", "value": f"{alert.current_value:.2f}", "short": True},
                        {"title": "Threshold", "value": f"{alert.threshold_value:.2f}", "short": True},
                        {"title": "Time", "value": alert.timestamp.isoformat(), "short": False}
                    ]
                }
            ]
        }
    
    async def _send_email_notification(self, alert: Alert):
        """Send email notification."""
        self.logger.info(f"Sending email notification for alert: {alert.alert_id}")
        await asyncio.sleep(0.3)
    
    async def _send_pagerduty_notification(self, alert: Alert):
        """Send PagerDuty notification."""
        self.logger.info(f"Sending PagerDuty notification for alert: {alert.alert_id}")
        await asyncio.sleep(0.4)
    
    async def _send_webhook_notification(self, alert: Alert):
        """Send webhook notification."""
        self.logger.info(f"Sending webhook notification for alert: {alert.alert_id}")
        await asyncio.sleep(0.2)
    
    async def _trigger_automated_remediation(self, alert: Alert):
        """Trigger automated remediation actions."""
        remediation_actions = {
            "high_cpu_usage": self._remediate_high_cpu,
            "high_memory_usage": self._remediate_high_memory,
            "high_error_rate": self._remediate_high_errors,
            "slow_response_time": self._remediate_slow_response
        }
        
        if alert.rule_name in remediation_actions:
            self.logger.info(f"Triggering automated remediation for: {alert.rule_name}")
            await remediation_actions[alert.rule_name](alert)
    
    async def _remediate_high_cpu(self, alert: Alert):
        """Remediate high CPU usage."""
        self.logger.info("Executing CPU remediation: scaling up instances")
        # Simulate auto-scaling
        await asyncio.sleep(2)
    
    async def _remediate_high_memory(self, alert: Alert):
        """Remediate high memory usage."""
        self.logger.info("Executing memory remediation: clearing caches")
        # Simulate cache clearing
        await asyncio.sleep(1)
    
    async def _remediate_high_errors(self, alert: Alert):
        """Remediate high error rate."""
        self.logger.info("Executing error remediation: circuit breaker activation")
        # Simulate circuit breaker
        await asyncio.sleep(1)
    
    async def _remediate_slow_response(self, alert: Alert):
        """Remediate slow response times."""
        self.logger.info("Executing response time remediation: cache warm-up")
        # Simulate cache warm-up
        await asyncio.sleep(1)


class MonitoringOrchestrator:
    """Main monitoring orchestration engine."""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.logger = self._setup_logging()
        self.config = self._load_config()
        self.metrics_collector = MetricsCollector(self.config)
        self.anomaly_detector = AnomalyDetector()
        self.alert_manager = AlertManager(config_path)
        self.running = False
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging."""
        logger = logging.getLogger("monitoring_orchestrator")
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(
            f"monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _load_config(self) -> MonitoringConfigSchema:
        """Load monitoring configuration."""
        try:
            # Create default config if file doesn't exist
            default_config = {
                "collection_interval_seconds": 60,
                "retention_days": 30,
                "enable_anomaly_detection": True,
                "enable_automated_remediation": True
            }
            
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
            else:
                config_data = default_config
            
            return MonitoringConfigSchema(**config_data)
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    async def start_monitoring(self, tenant_id: str):
        """Start continuous monitoring for a tenant."""
        self.logger.info(f"Starting monitoring for tenant: {tenant_id}")
        self.running = True
        
        collection_interval = getattr(self.config, 'collection_interval_seconds', 60)
        
        while self.running:
            try:
                await self._monitoring_cycle(tenant_id)
                await asyncio.sleep(collection_interval)
            except Exception as e:
                self.logger.error(f"Monitoring cycle failed: {e}")
                await asyncio.sleep(5)  # Short sleep before retry
    
    async def _monitoring_cycle(self, tenant_id: str):
        """Execute one monitoring cycle."""
        start_time = time.time()
        
        # Collect all metrics
        system_metrics = await self.metrics_collector.collect_system_metrics(tenant_id)
        app_metrics = await self.metrics_collector.collect_application_metrics(tenant_id)
        business_metrics = await self.metrics_collector.collect_business_metrics(tenant_id)
        
        all_metrics = {**system_metrics, **app_metrics, **business_metrics}
        
        # Update anomaly detection baselines
        for metric_name, metric_data in all_metrics.items():
            self.anomaly_detector.update_baseline(metric_name, [metric_data.value])
        
        # Detect anomalies
        anomalies_detected = []
        for metric_name, metric_data in all_metrics.items():
            is_anomaly, confidence = self.anomaly_detector.detect_anomaly(
                metric_name, metric_data.value
            )
            if is_anomaly:
                anomalies_detected.append((metric_name, metric_data.value, confidence))
        
        # Evaluate alert rules
        alerts = await self.alert_manager.evaluate_alert_rules(tenant_id, all_metrics)
        
        # Process alerts
        if alerts:
            await self.alert_manager.process_alerts(alerts)
        
        # Log monitoring summary
        cycle_duration = time.time() - start_time
        self.logger.info(
            f"Monitoring cycle completed for {tenant_id}: "
            f"{len(all_metrics)} metrics collected, "
            f"{len(anomalies_detected)} anomalies detected, "
            f"{len(alerts)} alerts triggered, "
            f"duration: {cycle_duration:.2f}s"
        )
    
    async def stop_monitoring(self):
        """Stop monitoring."""
        self.logger.info("Stopping monitoring")
        self.running = False
    
    async def get_metrics_summary(self, tenant_id: str) -> Dict[str, Any]:
        """Get current metrics summary."""
        system_metrics = await self.metrics_collector.collect_system_metrics(tenant_id)
        app_metrics = await self.metrics_collector.collect_application_metrics(tenant_id)
        business_metrics = await self.metrics_collector.collect_business_metrics(tenant_id)
        
        all_metrics = {**system_metrics, **app_metrics, **business_metrics}
        
        summary = {
            "tenant_id": tenant_id,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics_count": len(all_metrics),
            "active_alerts": len(self.alert_manager.active_alerts),
            "metrics": {
                name: {
                    "value": data.value,
                    "timestamp": data.timestamp.isoformat(),
                    "labels": data.labels
                }
                for name, data in all_metrics.items()
            }
        }
        
        return summary


async def main():
    """Main entry point for monitoring tool."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enterprise Monitoring and Alerting Tool")
    parser.add_argument("--config", default="monitoring_config.json", help="Monitoring configuration file")
    parser.add_argument("--tenant-id", required=True, help="Tenant ID to monitor")
    parser.add_argument("--duration", type=int, default=0, help="Monitoring duration in seconds (0 = infinite)")
    parser.add_argument("--summary", action="store_true", help="Show metrics summary and exit")
    
    args = parser.parse_args()
    
    # Initialize monitoring orchestrator
    orchestrator = MonitoringOrchestrator(args.config)
    
    if args.summary:
        # Show metrics summary
        summary = await orchestrator.get_metrics_summary(args.tenant_id)
        print(json.dumps(summary, indent=2))
        return
    
    # Start monitoring
    if args.duration > 0:
        # Run for specified duration
        monitoring_task = asyncio.create_task(
            orchestrator.start_monitoring(args.tenant_id)
        )
        await asyncio.sleep(args.duration)
        await orchestrator.stop_monitoring()
        monitoring_task.cancel()
    else:
        # Run indefinitely
        try:
            await orchestrator.start_monitoring(args.tenant_id)
        except KeyboardInterrupt:
            await orchestrator.stop_monitoring()
            print("Monitoring stopped by user")


if __name__ == "__main__":
    asyncio.run(main())
