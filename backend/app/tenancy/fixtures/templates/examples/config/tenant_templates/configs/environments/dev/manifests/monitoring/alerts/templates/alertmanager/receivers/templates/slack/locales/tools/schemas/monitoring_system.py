#!/usr/bin/env python3
"""
Real-time monitoring and metrics collection system for schema configurations.

This module provides comprehensive monitoring capabilities including performance tracking,
health monitoring, alerting, and AI-powered anomaly detection.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsCollector:
    """Advanced metrics collection and analysis."""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=1000))
        self.alert_thresholds = {}
        self.anomaly_detector = AnomalyDetector()
        self.running = False
        
        # Initialize default thresholds
        self._setup_default_thresholds()
        
    def _setup_default_thresholds(self):
        """Setup default alert thresholds."""
        self.alert_thresholds = {
            'validation_time_ms': {'warning': 100, 'critical': 500},
            'error_rate': {'warning': 0.01, 'critical': 0.05},
            'memory_usage_mb': {'warning': 100, 'critical': 200},
            'cpu_usage_percent': {'warning': 70, 'critical': 90},
            'deployment_duration_seconds': {'warning': 300, 'critical': 600},
            'health_check_failures': {'warning': 3, 'critical': 5}
        }
    
    def record_metric(self, metric_name: str, value: float, tags: Dict[str, str] = None):
        """Record a metric value."""
        timestamp = datetime.utcnow()
        
        metric_entry = {
            'value': value,
            'timestamp': timestamp,
            'tags': tags or {}
        }
        
        self.metrics_buffer[metric_name].append(metric_entry)
        
        # Clean old metrics
        self._cleanup_old_metrics(metric_name)
        
        # Check for alerts
        self._check_alerts(metric_name, value, tags)
        
        # Feed to anomaly detector
        self.anomaly_detector.add_data_point(metric_name, value, timestamp)
    
    def _cleanup_old_metrics(self, metric_name: str):
        """Remove metrics older than retention period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.retention_hours)
        buffer = self.metrics_buffer[metric_name]
        
        while buffer and buffer[0]['timestamp'] < cutoff_time:
            buffer.popleft()
    
    def _check_alerts(self, metric_name: str, value: float, tags: Dict[str, str]):
        """Check if metric value triggers alerts."""
        thresholds = self.alert_thresholds.get(metric_name)
        if not thresholds:
            return
        
        alert_level = None
        if value >= thresholds.get('critical', float('inf')):
            alert_level = 'critical'
        elif value >= thresholds.get('warning', float('inf')):
            alert_level = 'warning'
        
        if alert_level:
            self._trigger_alert(metric_name, value, alert_level, tags)
    
    def _trigger_alert(self, metric_name: str, value: float, level: str, tags: Dict[str, str]):
        """Trigger an alert."""
        alert = {
            'metric': metric_name,
            'value': value,
            'level': level,
            'timestamp': datetime.utcnow(),
            'tags': tags,
            'message': f"{level.upper()}: {metric_name} = {value}"
        }
        
        logger.warning(f"ALERT {level.upper()}: {metric_name} = {value} (tags: {tags})")
        
        # In real implementation, send to alerting system
        asyncio.create_task(self._send_alert(alert))
    
    async def _send_alert(self, alert: Dict[str, Any]):
        """Send alert to notification systems."""
        # Simulate sending alert
        await asyncio.sleep(0.1)
        # Would integrate with Slack, email, PagerDuty, etc.
    
    def get_metric_summary(self, metric_name: str, duration_minutes: int = 60) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=duration_minutes)
        buffer = self.metrics_buffer[metric_name]
        
        recent_values = [
            entry['value'] for entry in buffer 
            if entry['timestamp'] >= cutoff_time
        ]
        
        if not recent_values:
            return {'error': 'No data available'}
        
        return {
            'count': len(recent_values),
            'min': min(recent_values),
            'max': max(recent_values),
            'mean': statistics.mean(recent_values),
            'median': statistics.median(recent_values),
            'p95': self._percentile(recent_values, 0.95),
            'p99': self._percentile(recent_values, 0.99),
            'std_dev': statistics.stdev(recent_values) if len(recent_values) > 1 else 0
        }
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """Get summary for all metrics."""
        summary = {}
        for metric_name in self.metrics_buffer.keys():
            summary[metric_name] = self.get_metric_summary(metric_name)
        return summary


class AnomalyDetector:
    """AI-powered anomaly detection for metrics."""
    
    def __init__(self, window_size: int = 100, sensitivity: float = 2.0):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.data_windows = defaultdict(lambda: deque(maxlen=window_size))
        self.anomalies = deque(maxlen=1000)
    
    def add_data_point(self, metric_name: str, value: float, timestamp: datetime):
        """Add data point and check for anomalies."""
        window = self.data_windows[metric_name]
        window.append((value, timestamp))
        
        if len(window) >= 20:  # Need minimum data for anomaly detection
            is_anomaly = self._detect_anomaly(metric_name, value)
            if is_anomaly:
                self._record_anomaly(metric_name, value, timestamp)
    
    def _detect_anomaly(self, metric_name: str, value: float) -> bool:
        """Detect if value is anomalous using statistical methods."""
        window = self.data_windows[metric_name]
        values = [v for v, _ in window]
        
        if len(values) < 10:
            return False
        
        # Calculate z-score
        mean_val = statistics.mean(values[:-1])  # Exclude current value
        std_val = statistics.stdev(values[:-1]) if len(values) > 2 else 0
        
        if std_val == 0:
            return False
        
        z_score = abs(value - mean_val) / std_val
        return z_score > self.sensitivity
    
    def _record_anomaly(self, metric_name: str, value: float, timestamp: datetime):
        """Record detected anomaly."""
        anomaly = {
            'metric': metric_name,
            'value': value,
            'timestamp': timestamp,
            'severity': self._calculate_severity(metric_name, value)
        }
        
        self.anomalies.append(anomaly)
        logger.warning(f"ANOMALY DETECTED: {metric_name} = {value}")
    
    def _calculate_severity(self, metric_name: str, value: float) -> str:
        """Calculate anomaly severity."""
        # Simplified severity calculation
        window = self.data_windows[metric_name]
        values = [v for v, _ in window]
        
        if len(values) < 10:
            return 'low'
        
        mean_val = statistics.mean(values[:-1])
        std_val = statistics.stdev(values[:-1])
        
        if std_val == 0:
            return 'low'
        
        z_score = abs(value - mean_val) / std_val
        
        if z_score > 4.0:
            return 'critical'
        elif z_score > 3.0:
            return 'high'
        elif z_score > 2.5:
            return 'medium'
        else:
            return 'low'
    
    def get_recent_anomalies(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get anomalies from recent time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            anomaly for anomaly in self.anomalies
            if anomaly['timestamp'] >= cutoff_time
        ]


class PerformanceMonitor:
    """Monitor performance of schema operations."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.active_operations = {}
    
    def start_operation(self, operation_id: str, operation_type: str, metadata: Dict[str, Any] = None):
        """Start monitoring an operation."""
        self.active_operations[operation_id] = {
            'type': operation_type,
            'start_time': time.time(),
            'metadata': metadata or {}
        }
    
    def end_operation(self, operation_id: str, success: bool = True, error: str = None):
        """End monitoring an operation."""
        if operation_id not in self.active_operations:
            logger.warning(f"Operation {operation_id} not found in active operations")
            return
        
        operation = self.active_operations.pop(operation_id)
        duration = time.time() - operation['start_time']
        
        # Record metrics
        self.metrics.record_metric(
            f"{operation['type']}_duration_seconds",
            duration,
            tags=operation['metadata']
        )
        
        self.metrics.record_metric(
            f"{operation['type']}_success_rate",
            1.0 if success else 0.0,
            tags=operation['metadata']
        )
        
        if not success and error:
            self.metrics.record_metric(
                f"{operation['type']}_error_count",
                1.0,
                tags={'error_type': error, **operation['metadata']}
            )
        
        logger.info(f"Operation {operation_id} completed in {duration:.3f}s (success: {success})")
    
    def get_operation_stats(self, operation_type: str, duration_minutes: int = 60) -> Dict[str, Any]:
        """Get statistics for operation type."""
        duration_stats = self.metrics.get_metric_summary(
            f"{operation_type}_duration_seconds", 
            duration_minutes
        )
        
        # Calculate success rate
        cutoff_time = datetime.utcnow() - timedelta(minutes=duration_minutes)
        success_buffer = self.metrics.metrics_buffer[f"{operation_type}_success_rate"]
        
        recent_success = [
            entry['value'] for entry in success_buffer
            if entry['timestamp'] >= cutoff_time
        ]
        
        success_rate = statistics.mean(recent_success) if recent_success else 0.0
        
        return {
            'duration_stats': duration_stats,
            'success_rate': success_rate,
            'total_operations': len(recent_success)
        }


class HealthChecker:
    """System health monitoring."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.health_checks = {}
        self.running = False
    
    def register_health_check(self, name: str, check_func, interval_seconds: int = 60):
        """Register a health check function."""
        self.health_checks[name] = {
            'func': check_func,
            'interval': interval_seconds,
            'last_run': 0,
            'last_result': None
        }
    
    async def run_health_checks(self):
        """Run all registered health checks."""
        self.running = True
        
        while self.running:
            current_time = time.time()
            
            for name, check in self.health_checks.items():
                if current_time - check['last_run'] >= check['interval']:
                    try:
                        result = await check['func']()
                        check['last_result'] = result
                        check['last_run'] = current_time
                        
                        # Record health metric
                        self.metrics.record_metric(
                            f"health_check_{name}",
                            1.0 if result.get('healthy', False) else 0.0,
                            tags={'component': name}
                        )
                        
                    except Exception as e:
                        logger.error(f"Health check {name} failed: {e}")
                        self.metrics.record_metric(
                            f"health_check_{name}",
                            0.0,
                            tags={'component': name, 'error': str(e)}
                        )
            
            await asyncio.sleep(10)  # Check every 10 seconds
    
    def stop_health_checks(self):
        """Stop health check monitoring."""
        self.running = False
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        status = {
            'overall_healthy': True,
            'checks': {},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        for name, check in self.health_checks.items():
            result = check['last_result']
            if result:
                status['checks'][name] = result
                if not result.get('healthy', False):
                    status['overall_healthy'] = False
            else:
                status['checks'][name] = {'healthy': False, 'message': 'Never executed'}
                status['overall_healthy'] = False
        
        return status


# Example health check functions
async def database_health_check() -> Dict[str, Any]:
    """Check database connectivity and performance."""
    try:
        # Simulate database check
        await asyncio.sleep(0.1)
        
        return {
            'healthy': True,
            'response_time_ms': 45,
            'connection_pool_size': 10,
            'active_connections': 3
        }
    except Exception as e:
        return {
            'healthy': False,
            'error': str(e)
        }


async def redis_health_check() -> Dict[str, Any]:
    """Check Redis connectivity and performance."""
    try:
        # Simulate Redis check
        await asyncio.sleep(0.05)
        
        return {
            'healthy': True,
            'response_time_ms': 12,
            'memory_usage_mb': 128,
            'connected_clients': 5
        }
    except Exception as e:
        return {
            'healthy': False,
            'error': str(e)
        }


async def schema_validation_health_check() -> Dict[str, Any]:
    """Check schema validation system health."""
    try:
        # Test schema validation
        from . import validate_with_schema
        
        test_data = {
            'locale': 'en_US',
            'language_code': 'en',
            'country_code': 'US',
            'display_name': 'English (United States)',
            'native_name': 'English (United States)',
            'date_format': 'MM/DD/YYYY',
            'time_format': 'hh:mm:ss a',
            'number_format': '1,234.56',
            'currency_code': 'USD',
            'currency_symbol': '$',
            'currency_position': 'before',
            'default_timezone': 'America/New_York'
        }
        
        result = validate_with_schema('locale_config', test_data)
        
        return {
            'healthy': True,
            'validation_successful': True,
            'schemas_available': len(list_available_schemas())
        }
    except Exception as e:
        return {
            'healthy': False,
            'error': str(e)
        }


class MonitoringDashboard:
    """Real-time monitoring dashboard."""
    
    def __init__(self, metrics_collector: MetricsCollector, health_checker: HealthChecker):
        self.metrics = metrics_collector
        self.health = health_checker
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get all dashboard data."""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'health_status': self.health.get_health_status(),
            'metrics_summary': self.metrics.get_all_metrics_summary(),
            'recent_anomalies': self.metrics.anomaly_detector.get_recent_anomalies(),
            'system_info': self._get_system_info()
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        import psutil
        import os
        
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'process_id': os.getpid(),
                'uptime_seconds': time.time() - psutil.Process().create_time()
            }
        except:
            return {'error': 'System info unavailable'}


async def main():
    """Main monitoring loop."""
    # Initialize components
    metrics_collector = MetricsCollector()
    performance_monitor = PerformanceMonitor(metrics_collector)
    health_checker = HealthChecker(metrics_collector)
    dashboard = MonitoringDashboard(metrics_collector, health_checker)
    
    # Register health checks
    health_checker.register_health_check('database', database_health_check, 30)
    health_checker.register_health_check('redis', redis_health_check, 30)
    health_checker.register_health_check('schema_validation', schema_validation_health_check, 60)
    
    # Start health checking
    health_task = asyncio.create_task(health_checker.run_health_checks())
    
    try:
        # Simulate some operations and metrics
        for i in range(100):
            # Simulate validation operations
            op_id = f"validation_{i}"
            performance_monitor.start_operation(op_id, "schema_validation", {"schema": "tenant_config"})
            
            # Simulate processing time
            await asyncio.sleep(0.1)
            
            # Random success/failure
            import random
            success = random.random() > 0.05  # 5% failure rate
            performance_monitor.end_operation(op_id, success)
            
            # Record some metrics
            metrics_collector.record_metric("validation_time_ms", random.uniform(50, 200))
            metrics_collector.record_metric("memory_usage_mb", random.uniform(80, 150))
            metrics_collector.record_metric("cpu_usage_percent", random.uniform(20, 80))
            
            if i % 10 == 0:
                # Print dashboard data periodically
                dashboard_data = dashboard.get_dashboard_data()
                print(f"\n=== Dashboard Update {i} ===")
                print(f"Health: {'✅' if dashboard_data['health_status']['overall_healthy'] else '❌'}")
                print(f"Anomalies: {len(dashboard_data['recent_anomalies'])}")
                print(f"CPU: {dashboard_data['system_info'].get('cpu_percent', 'N/A')}%")
                print(f"Memory: {dashboard_data['system_info'].get('memory_percent', 'N/A')}%")
            
            await asyncio.sleep(1)
    
    finally:
        health_checker.stop_health_checks()
        health_task.cancel()
        try:
            await health_task
        except asyncio.CancelledError:
            pass


if __name__ == '__main__':
    asyncio.run(main())
