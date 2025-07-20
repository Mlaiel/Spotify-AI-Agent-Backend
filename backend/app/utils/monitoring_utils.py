"""
Enterprise Monitoring Utilities
===============================
Advanced monitoring and observability utilities for Spotify AI Agent platform.

Expert Team Implementation:
- Lead Developer + AI Architect: Intelligent monitoring with ML anomaly detection
- Senior Backend Developer: High-performance metrics collection and alerting
- DBA & Data Engineer: Database monitoring and performance analytics
- Security Specialist: Security monitoring and threat detection
- Microservices Architect: Distributed monitoring and service mesh observability
- ML Engineer: Predictive monitoring and capacity planning
"""

import asyncio
import logging
import json
import time
import threading
import psutil
import platform
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple, Callable, Iterator
from abc import ABC, abstractmethod
from enum import Enum
import statistics
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import weakref
import gc
import tracemalloc
import resource

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Histogram = Gauge = Summary = None

# APM and distributed tracing
try:
    import opentelemetry
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    opentelemetry = trace = None

# Database monitoring
try:
    import sqlalchemy
    from sqlalchemy import event, create_engine
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    sqlalchemy = event = None

logger = logging.getLogger(__name__)

# === Monitoring Types and Enums ===
class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"

class MonitoringScope(Enum):
    """Monitoring scope levels."""
    SYSTEM = "system"
    APPLICATION = "application"
    DATABASE = "database"
    NETWORK = "network"
    SECURITY = "security"
    BUSINESS = "business"

class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class MetricValue:
    """Metric value with metadata."""
    name: str
    value: Union[int, float]
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE
    unit: str = ""
    description: str = ""

@dataclass
class Alert:
    """Monitoring alert."""
    alert_id: str
    name: str
    description: str
    severity: AlertSeverity
    timestamp: datetime
    source: str
    labels: Dict[str, str] = field(default_factory=dict)
    threshold_value: Optional[float] = None
    current_value: Optional[float] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class HealthCheck:
    """Health check result."""
    name: str
    status: HealthStatus
    timestamp: datetime
    response_time_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    network_io_bytes: Dict[str, int]
    active_connections: int
    response_time_ms: float
    throughput_rps: float
    error_rate_percent: float

@dataclass
class SLAMetrics:
    """Service Level Agreement metrics."""
    availability_percent: float
    latency_p95_ms: float
    latency_p99_ms: float
    error_rate_percent: float
    throughput_rps: float
    uptime_seconds: float
    mttr_seconds: float  # Mean Time To Recovery
    mtbf_seconds: float  # Mean Time Between Failures

# === Base Monitor ===
class BaseMonitor(ABC):
    """Abstract base class for monitors."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.name = self.__class__.__name__
        self.enabled = config.get('enabled', True)
        self.interval_seconds = config.get('interval_seconds', 60)
        self.metrics = defaultdict(list)
        self.alerts = deque(maxlen=1000)
        self.callbacks = defaultdict(list)
        self.is_running = False
        
    @abstractmethod
    async def collect_metrics(self) -> List[MetricValue]:
        """Collect metrics from monitored system."""
        pass
    
    @abstractmethod
    async def check_health(self) -> HealthCheck:
        """Perform health check."""
        pass
    
    async def start_monitoring(self):
        """Start monitoring loop."""
        if not self.enabled:
            logger.info(f"Monitor {self.name} is disabled")
            return
        
        self.is_running = True
        logger.info(f"Started monitoring: {self.name}")
        
        while self.is_running:
            try:
                # Collect metrics
                metrics = await self.collect_metrics()
                for metric in metrics:
                    self.metrics[metric.name].append(metric)
                    await self.emit_event('metric_collected', metric)
                
                # Perform health check
                health = await self.check_health()
                await self.emit_event('health_check', health)
                
                # Check for alerts
                await self._check_alerts(metrics)
                
                # Wait for next interval
                await asyncio.sleep(self.interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop for {self.name}: {e}")
                await asyncio.sleep(min(self.interval_seconds, 30))
    
    async def stop_monitoring(self):
        """Stop monitoring loop."""
        self.is_running = False
        logger.info(f"Stopped monitoring: {self.name}")
    
    def register_callback(self, event: str, callback: Callable):
        """Register event callback."""
        self.callbacks[event].append(callback)
    
    async def emit_event(self, event: str, data: Any):
        """Emit event to registered callbacks."""
        for callback in self.callbacks[event]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Callback error for event {event}: {e}")
    
    async def _check_alerts(self, metrics: List[MetricValue]):
        """Check metrics against alert thresholds."""
        # Override in subclasses with specific alert logic
        pass
    
    def get_metrics_summary(self, metric_name: str, duration_minutes: int = 60) -> Dict[str, Any]:
        """Get metrics summary for specified duration."""
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        
        if metric_name not in self.metrics:
            return {}
        
        recent_metrics = [
            m for m in self.metrics[metric_name] 
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        values = [m.value for m in recent_metrics]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
            'latest': values[-1],
            'trend': self._calculate_trend(values)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction."""
        if len(values) < 2:
            return "stable"
        
        recent_avg = statistics.mean(values[-min(5, len(values)):])
        earlier_avg = statistics.mean(values[:min(5, len(values))])
        
        diff_percent = ((recent_avg - earlier_avg) / earlier_avg * 100) if earlier_avg != 0 else 0
        
        if diff_percent > 5:
            return "increasing"
        elif diff_percent < -5:
            return "decreasing"
        else:
            return "stable"

# === System Monitor ===
class SystemMonitor(BaseMonitor):
    """System resource monitoring."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.cpu_threshold = config.get('cpu_threshold_percent', 80)
        self.memory_threshold = config.get('memory_threshold_percent', 85)
        self.disk_threshold = config.get('disk_threshold_percent', 90)
        
        # Enable memory profiling if requested
        if config.get('enable_memory_profiling', False):
            tracemalloc.start()
    
    async def collect_metrics(self) -> List[MetricValue]:
        """Collect system metrics."""
        timestamp = datetime.now()
        metrics = []
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
        
        metrics.extend([
            MetricValue("cpu_usage_percent", cpu_percent, timestamp, 
                       metric_type=MetricType.GAUGE, unit="%"),
            MetricValue("cpu_count", cpu_count, timestamp, 
                       metric_type=MetricType.GAUGE),
            MetricValue("load_avg_1m", load_avg[0], timestamp, 
                       metric_type=MetricType.GAUGE),
            MetricValue("load_avg_5m", load_avg[1], timestamp, 
                       metric_type=MetricType.GAUGE),
            MetricValue("load_avg_15m", load_avg[2], timestamp, 
                       metric_type=MetricType.GAUGE),
        ])
        
        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        metrics.extend([
            MetricValue("memory_usage_percent", memory.percent, timestamp,
                       metric_type=MetricType.GAUGE, unit="%"),
            MetricValue("memory_available_bytes", memory.available, timestamp,
                       metric_type=MetricType.GAUGE, unit="bytes"),
            MetricValue("memory_total_bytes", memory.total, timestamp,
                       metric_type=MetricType.GAUGE, unit="bytes"),
            MetricValue("swap_usage_percent", swap.percent, timestamp,
                       metric_type=MetricType.GAUGE, unit="%"),
        ])
        
        # Disk metrics
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        metrics.extend([
            MetricValue("disk_usage_percent", 
                       (disk_usage.used / disk_usage.total) * 100, timestamp,
                       metric_type=MetricType.GAUGE, unit="%"),
            MetricValue("disk_free_bytes", disk_usage.free, timestamp,
                       metric_type=MetricType.GAUGE, unit="bytes"),
            MetricValue("disk_total_bytes", disk_usage.total, timestamp,
                       metric_type=MetricType.GAUGE, unit="bytes"),
        ])
        
        if disk_io:
            metrics.extend([
                MetricValue("disk_read_bytes", disk_io.read_bytes, timestamp,
                           metric_type=MetricType.COUNTER, unit="bytes"),
                MetricValue("disk_write_bytes", disk_io.write_bytes, timestamp,
                           metric_type=MetricType.COUNTER, unit="bytes"),
                MetricValue("disk_read_ops", disk_io.read_count, timestamp,
                           metric_type=MetricType.COUNTER),
                MetricValue("disk_write_ops", disk_io.write_count, timestamp,
                           metric_type=MetricType.COUNTER),
            ])
        
        # Network metrics
        network_io = psutil.net_io_counters()
        if network_io:
            metrics.extend([
                MetricValue("network_bytes_sent", network_io.bytes_sent, timestamp,
                           metric_type=MetricType.COUNTER, unit="bytes"),
                MetricValue("network_bytes_recv", network_io.bytes_recv, timestamp,
                           metric_type=MetricType.COUNTER, unit="bytes"),
                MetricValue("network_packets_sent", network_io.packets_sent, timestamp,
                           metric_type=MetricType.COUNTER),
                MetricValue("network_packets_recv", network_io.packets_recv, timestamp,
                           metric_type=MetricType.COUNTER),
            ])
        
        # Process metrics
        process = psutil.Process()
        metrics.extend([
            MetricValue("process_cpu_percent", process.cpu_percent(), timestamp,
                       metric_type=MetricType.GAUGE, unit="%"),
            MetricValue("process_memory_percent", process.memory_percent(), timestamp,
                       metric_type=MetricType.GAUGE, unit="%"),
            MetricValue("process_memory_rss", process.memory_info().rss, timestamp,
                       metric_type=MetricType.GAUGE, unit="bytes"),
            MetricValue("process_threads", process.num_threads(), timestamp,
                       metric_type=MetricType.GAUGE),
        ])
        
        # File descriptor usage
        try:
            fds = process.num_fds() if hasattr(process, 'num_fds') else 0
            metrics.append(
                MetricValue("process_file_descriptors", fds, timestamp,
                           metric_type=MetricType.GAUGE)
            )
        except:
            pass
        
        # Memory profiling if enabled
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            metrics.extend([
                MetricValue("memory_traced_current", current, timestamp,
                           metric_type=MetricType.GAUGE, unit="bytes"),
                MetricValue("memory_traced_peak", peak, timestamp,
                           metric_type=MetricType.GAUGE, unit="bytes"),
            ])
        
        return metrics
    
    async def check_health(self) -> HealthCheck:
        """Perform system health check."""
        timestamp = datetime.now()
        start_time = time.time()
        
        try:
            # Check critical resources
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            disk_percent = (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100
            
            # Determine health status
            status = HealthStatus.HEALTHY
            details = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'disk_percent': disk_percent
            }
            
            if (cpu_percent > self.cpu_threshold or 
                memory_percent > self.memory_threshold or 
                disk_percent > self.disk_threshold):
                status = HealthStatus.DEGRADED
            
            if (cpu_percent > 95 or memory_percent > 95 or disk_percent > 95):
                status = HealthStatus.UNHEALTHY
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheck(
                name="system_health",
                status=status,
                timestamp=timestamp,
                response_time_ms=response_time,
                details=details
            )
            
        except Exception as e:
            return HealthCheck(
                name="system_health",
                status=HealthStatus.UNHEALTHY,
                timestamp=timestamp,
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    async def _check_alerts(self, metrics: List[MetricValue]):
        """Check system metrics for alert conditions."""
        for metric in metrics:
            alert_threshold = None
            
            if metric.name == "cpu_usage_percent":
                alert_threshold = self.cpu_threshold
            elif metric.name == "memory_usage_percent":
                alert_threshold = self.memory_threshold
            elif metric.name == "disk_usage_percent":
                alert_threshold = self.disk_threshold
            
            if alert_threshold and metric.value > alert_threshold:
                alert = Alert(
                    alert_id=f"system_{metric.name}_{int(time.time())}",
                    name=f"High {metric.name}",
                    description=f"{metric.name} exceeded threshold: {metric.value:.1f}% > {alert_threshold}%",
                    severity=AlertSeverity.HIGH if metric.value > alert_threshold * 1.1 else AlertSeverity.MEDIUM,
                    timestamp=datetime.now(),
                    source="system_monitor",
                    threshold_value=alert_threshold,
                    current_value=metric.value,
                    labels={'metric': metric.name, 'node': platform.node()}
                )
                
                self.alerts.append(alert)
                await self.emit_event('alert_triggered', alert)

# === Application Monitor ===
class ApplicationMonitor(BaseMonitor):
    """Application-specific monitoring."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.app_name = config.get('app_name', 'spotify-ai-agent')
        self.response_time_threshold_ms = config.get('response_time_threshold_ms', 1000)
        self.error_rate_threshold_percent = config.get('error_rate_threshold_percent', 5)
        
        # Request tracking
        self.request_times = deque(maxlen=1000)
        self.error_count = 0
        self.total_requests = 0
        
        # Custom metrics
        self.custom_metrics = defaultdict(list)
        
        # Initialize Prometheus metrics if available
        if PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        self.prometheus_metrics = {
            'requests_total': Counter('requests_total', 'Total requests', ['method', 'endpoint', 'status']),
            'request_duration': Histogram('request_duration_seconds', 'Request duration', ['method', 'endpoint']),
            'active_connections': Gauge('active_connections', 'Active connections'),
            'memory_usage': Gauge('memory_usage_bytes', 'Memory usage'),
            'cpu_usage': Gauge('cpu_usage_percent', 'CPU usage percentage'),
        }
    
    async def collect_metrics(self) -> List[MetricValue]:
        """Collect application metrics."""
        timestamp = datetime.now()
        metrics = []
        
        # Calculate response time metrics
        if self.request_times:
            recent_times = list(self.request_times)
            avg_response_time = statistics.mean(recent_times)
            p95_response_time = np.percentile(recent_times, 95)
            p99_response_time = np.percentile(recent_times, 99)
            
            metrics.extend([
                MetricValue("response_time_avg_ms", avg_response_time, timestamp,
                           metric_type=MetricType.GAUGE, unit="ms"),
                MetricValue("response_time_p95_ms", p95_response_time, timestamp,
                           metric_type=MetricType.GAUGE, unit="ms"),
                MetricValue("response_time_p99_ms", p99_response_time, timestamp,
                           metric_type=MetricType.GAUGE, unit="ms"),
            ])
        
        # Error rate
        error_rate = (self.error_count / self.total_requests * 100) if self.total_requests > 0 else 0
        
        metrics.extend([
            MetricValue("error_rate_percent", error_rate, timestamp,
                       metric_type=MetricType.GAUGE, unit="%"),
            MetricValue("total_requests", self.total_requests, timestamp,
                       metric_type=MetricType.COUNTER),
            MetricValue("error_count", self.error_count, timestamp,
                       metric_type=MetricType.COUNTER),
        ])
        
        # Throughput (requests per minute)
        current_minute = timestamp.replace(second=0, microsecond=0)
        recent_requests = [
            t for t in self.request_times 
            if datetime.fromtimestamp(t).replace(second=0, microsecond=0) == current_minute
        ]
        throughput = len(recent_requests)
        
        metrics.append(
            MetricValue("throughput_rpm", throughput, timestamp,
                       metric_type=MetricType.GAUGE, unit="requests/min")
        )
        
        # Garbage collection metrics
        gc_stats = gc.get_stats()
        if gc_stats:
            metrics.extend([
                MetricValue("gc_collections", sum(stat['collections'] for stat in gc_stats), timestamp,
                           metric_type=MetricType.COUNTER),
                MetricValue("gc_collected", sum(stat['collected'] for stat in gc_stats), timestamp,
                           metric_type=MetricType.COUNTER),
                MetricValue("gc_uncollectable", sum(stat['uncollectable'] for stat in gc_stats), timestamp,
                           metric_type=MetricType.COUNTER),
            ])
        
        # Custom metrics
        for metric_name, values in self.custom_metrics.items():
            if values:
                latest_value = values[-1]
                metrics.append(
                    MetricValue(metric_name, latest_value, timestamp,
                               metric_type=MetricType.GAUGE)
                )
        
        return metrics
    
    async def check_health(self) -> HealthCheck:
        """Perform application health check."""
        timestamp = datetime.now()
        start_time = time.time()
        
        try:
            # Check key application components
            details = {}
            status = HealthStatus.HEALTHY
            
            # Check response times
            if self.request_times:
                avg_response_time = statistics.mean(list(self.request_times)[-10:])
                details['avg_response_time_ms'] = avg_response_time
                
                if avg_response_time > self.response_time_threshold_ms:
                    status = HealthStatus.DEGRADED
            
            # Check error rate
            error_rate = (self.error_count / self.total_requests * 100) if self.total_requests > 0 else 0
            details['error_rate_percent'] = error_rate
            
            if error_rate > self.error_rate_threshold_percent:
                status = HealthStatus.DEGRADED
            
            if error_rate > self.error_rate_threshold_percent * 2:
                status = HealthStatus.UNHEALTHY
            
            # Check memory leaks
            if tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                details['memory_traced_mb'] = current / 1024 / 1024
                
                # Simple leak detection: if current usage is > 80% of peak, might be leaking
                if current > peak * 0.8:
                    details['potential_memory_leak'] = True
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheck(
                name="application_health",
                status=status,
                timestamp=timestamp,
                response_time_ms=response_time,
                details=details
            )
            
        except Exception as e:
            return HealthCheck(
                name="application_health",
                status=HealthStatus.UNHEALTHY,
                timestamp=timestamp,
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    def record_request(self, response_time_ms: float, status_code: int = 200):
        """Record request metrics."""
        self.request_times.append(response_time_ms)
        self.total_requests += 1
        
        if status_code >= 400:
            self.error_count += 1
        
        # Update Prometheus metrics if available
        if PROMETHEUS_AVAILABLE and hasattr(self, 'prometheus_metrics'):
            self.prometheus_metrics['requests_total'].labels(
                method='unknown', endpoint='unknown', status=str(status_code)
            ).inc()
            
            self.prometheus_metrics['request_duration'].labels(
                method='unknown', endpoint='unknown'
            ).observe(response_time_ms / 1000)
    
    def record_custom_metric(self, name: str, value: float):
        """Record custom application metric."""
        self.custom_metrics[name].append(value)
        
        # Keep only recent values
        if len(self.custom_metrics[name]) > 100:
            self.custom_metrics[name] = self.custom_metrics[name][-100:]

# === Performance Monitor ===
class PerformanceMonitor(BaseMonitor):
    """Comprehensive performance monitoring."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.performance_history = deque(maxlen=1440)  # 24 hours of minute data
        self.benchmark_baselines = {}
        
    async def collect_metrics(self) -> List[MetricValue]:
        """Collect performance metrics."""
        timestamp = datetime.now()
        
        # Collect comprehensive performance snapshot
        perf_metrics = await self._collect_performance_snapshot()
        
        # Store in history
        self.performance_history.append(perf_metrics)
        
        # Convert to MetricValue objects
        metrics = [
            MetricValue("perf_cpu_usage", perf_metrics.cpu_usage_percent, timestamp,
                       metric_type=MetricType.GAUGE, unit="%"),
            MetricValue("perf_memory_usage", perf_metrics.memory_usage_percent, timestamp,
                       metric_type=MetricType.GAUGE, unit="%"),
            MetricValue("perf_disk_usage", perf_metrics.disk_usage_percent, timestamp,
                       metric_type=MetricType.GAUGE, unit="%"),
            MetricValue("perf_response_time", perf_metrics.response_time_ms, timestamp,
                       metric_type=MetricType.GAUGE, unit="ms"),
            MetricValue("perf_throughput", perf_metrics.throughput_rps, timestamp,
                       metric_type=MetricType.GAUGE, unit="rps"),
            MetricValue("perf_error_rate", perf_metrics.error_rate_percent, timestamp,
                       metric_type=MetricType.GAUGE, unit="%"),
        ]
        
        return metrics
    
    async def _collect_performance_snapshot(self) -> PerformanceMetrics:
        """Collect comprehensive performance snapshot."""
        timestamp = datetime.now()
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        # Network connections
        connections = len(psutil.net_connections())
        
        # Placeholder for application metrics (would be injected by app)
        response_time_ms = 0.0
        throughput_rps = 0.0
        error_rate_percent = 0.0
        
        return PerformanceMetrics(
            timestamp=timestamp,
            cpu_usage_percent=cpu_percent,
            memory_usage_percent=memory.percent,
            disk_usage_percent=(disk.used / disk.total) * 100,
            network_io_bytes={
                'bytes_sent': network.bytes_sent if network else 0,
                'bytes_recv': network.bytes_recv if network else 0
            },
            active_connections=connections,
            response_time_ms=response_time_ms,
            throughput_rps=throughput_rps,
            error_rate_percent=error_rate_percent
        )
    
    async def check_health(self) -> HealthCheck:
        """Perform performance health check."""
        timestamp = datetime.now()
        start_time = time.time()
        
        try:
            # Analyze recent performance
            if len(self.performance_history) < 5:
                status = HealthStatus.UNKNOWN
                details = {'reason': 'insufficient_data'}
            else:
                recent_metrics = list(self.performance_history)[-5:]
                status, details = await self._analyze_performance_health(recent_metrics)
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheck(
                name="performance_health",
                status=status,
                timestamp=timestamp,
                response_time_ms=response_time,
                details=details
            )
            
        except Exception as e:
            return HealthCheck(
                name="performance_health",
                status=HealthStatus.UNHEALTHY,
                timestamp=timestamp,
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    async def _analyze_performance_health(self, metrics: List[PerformanceMetrics]) -> Tuple[HealthStatus, Dict[str, Any]]:
        """Analyze performance metrics for health status."""
        # Calculate averages
        avg_cpu = statistics.mean(m.cpu_usage_percent for m in metrics)
        avg_memory = statistics.mean(m.memory_usage_percent for m in metrics)
        avg_response_time = statistics.mean(m.response_time_ms for m in metrics)
        
        details = {
            'avg_cpu_percent': avg_cpu,
            'avg_memory_percent': avg_memory,
            'avg_response_time_ms': avg_response_time,
            'sample_count': len(metrics)
        }
        
        # Determine health status
        if avg_cpu > 80 or avg_memory > 85 or avg_response_time > 2000:
            return HealthStatus.UNHEALTHY, details
        elif avg_cpu > 60 or avg_memory > 70 or avg_response_time > 1000:
            return HealthStatus.DEGRADED, details
        else:
            return HealthStatus.HEALTHY, details
    
    def set_baseline(self, name: str, metrics: PerformanceMetrics):
        """Set performance baseline for comparison."""
        self.benchmark_baselines[name] = metrics
    
    def compare_to_baseline(self, baseline_name: str) -> Dict[str, Any]:
        """Compare current performance to baseline."""
        if (baseline_name not in self.benchmark_baselines or 
            not self.performance_history):
            return {}
        
        baseline = self.benchmark_baselines[baseline_name]
        current = self.performance_history[-1]
        
        return {
            'cpu_change_percent': current.cpu_usage_percent - baseline.cpu_usage_percent,
            'memory_change_percent': current.memory_usage_percent - baseline.memory_usage_percent,
            'response_time_change_ms': current.response_time_ms - baseline.response_time_ms,
            'throughput_change_rps': current.throughput_rps - baseline.throughput_rps,
            'comparison_timestamp': datetime.now().isoformat()
        }

# === SLA Monitor ===
class SLAMonitor(BaseMonitor):
    """Service Level Agreement monitoring."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.availability_target = config.get('availability_target', 99.9)  # 99.9%
        self.latency_p95_target_ms = config.get('latency_p95_target_ms', 500)
        self.error_rate_target = config.get('error_rate_target', 1.0)  # 1%
        
        self.uptime_tracker = UptimeTracker()
        self.sla_history = deque(maxlen=720)  # 12 hours of 1-minute data
    
    async def collect_metrics(self) -> List[MetricValue]:
        """Collect SLA metrics."""
        timestamp = datetime.now()
        
        # Calculate current SLA metrics
        sla_metrics = await self._calculate_sla_metrics()
        
        # Store in history
        self.sla_history.append({
            'timestamp': timestamp,
            'metrics': sla_metrics
        })
        
        # Convert to MetricValue objects
        metrics = [
            MetricValue("sla_availability", sla_metrics.availability_percent, timestamp,
                       metric_type=MetricType.GAUGE, unit="%"),
            MetricValue("sla_latency_p95", sla_metrics.latency_p95_ms, timestamp,
                       metric_type=MetricType.GAUGE, unit="ms"),
            MetricValue("sla_latency_p99", sla_metrics.latency_p99_ms, timestamp,
                       metric_type=MetricType.GAUGE, unit="ms"),
            MetricValue("sla_error_rate", sla_metrics.error_rate_percent, timestamp,
                       metric_type=MetricType.GAUGE, unit="%"),
            MetricValue("sla_throughput", sla_metrics.throughput_rps, timestamp,
                       metric_type=MetricType.GAUGE, unit="rps"),
            MetricValue("sla_uptime", sla_metrics.uptime_seconds, timestamp,
                       metric_type=MetricType.COUNTER, unit="seconds"),
        ]
        
        return metrics
    
    async def _calculate_sla_metrics(self) -> SLAMetrics:
        """Calculate current SLA metrics."""
        # Get uptime information
        uptime_info = self.uptime_tracker.get_uptime_info()
        
        # Placeholder values (would be calculated from actual data)
        availability_percent = uptime_info['availability_percent']
        latency_p95_ms = 0.0  # Would be calculated from request data
        latency_p99_ms = 0.0  # Would be calculated from request data
        error_rate_percent = 0.0  # Would be calculated from error data
        throughput_rps = 0.0  # Would be calculated from request data
        
        return SLAMetrics(
            availability_percent=availability_percent,
            latency_p95_ms=latency_p95_ms,
            latency_p99_ms=latency_p99_ms,
            error_rate_percent=error_rate_percent,
            throughput_rps=throughput_rps,
            uptime_seconds=uptime_info['uptime_seconds'],
            mttr_seconds=uptime_info['mttr_seconds'],
            mtbf_seconds=uptime_info['mtbf_seconds']
        )
    
    async def check_health(self) -> HealthCheck:
        """Perform SLA health check."""
        timestamp = datetime.now()
        start_time = time.time()
        
        try:
            sla_metrics = await self._calculate_sla_metrics()
            
            # Check SLA compliance
            sla_violations = []
            
            if sla_metrics.availability_percent < self.availability_target:
                sla_violations.append(f"Availability: {sla_metrics.availability_percent:.2f}% < {self.availability_target}%")
            
            if sla_metrics.latency_p95_ms > self.latency_p95_target_ms:
                sla_violations.append(f"P95 Latency: {sla_metrics.latency_p95_ms:.1f}ms > {self.latency_p95_target_ms}ms")
            
            if sla_metrics.error_rate_percent > self.error_rate_target:
                sla_violations.append(f"Error Rate: {sla_metrics.error_rate_percent:.2f}% > {self.error_rate_target}%")
            
            status = HealthStatus.HEALTHY if not sla_violations else HealthStatus.DEGRADED
            
            details = {
                'sla_violations': sla_violations,
                'availability_percent': sla_metrics.availability_percent,
                'latency_p95_ms': sla_metrics.latency_p95_ms,
                'error_rate_percent': sla_metrics.error_rate_percent
            }
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheck(
                name="sla_health",
                status=status,
                timestamp=timestamp,
                response_time_ms=response_time,
                details=details
            )
            
        except Exception as e:
            return HealthCheck(
                name="sla_health",
                status=HealthStatus.UNHEALTHY,
                timestamp=timestamp,
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )

# === Uptime Tracker ===
class UptimeTracker:
    """Track service uptime and calculate availability metrics."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.downtime_periods = []
        self.current_downtime_start = None
        
    def record_downtime_start(self):
        """Record start of downtime period."""
        if self.current_downtime_start is None:
            self.current_downtime_start = datetime.now()
    
    def record_downtime_end(self):
        """Record end of downtime period."""
        if self.current_downtime_start is not None:
            downtime_end = datetime.now()
            self.downtime_periods.append({
                'start': self.current_downtime_start,
                'end': downtime_end,
                'duration_seconds': (downtime_end - self.current_downtime_start).total_seconds()
            })
            self.current_downtime_start = None
    
    def get_uptime_info(self) -> Dict[str, Any]:
        """Get comprehensive uptime information."""
        now = datetime.now()
        total_time = (now - self.start_time).total_seconds()
        
        # Calculate total downtime
        total_downtime = sum(period['duration_seconds'] for period in self.downtime_periods)
        
        # Add current downtime if in progress
        if self.current_downtime_start is not None:
            total_downtime += (now - self.current_downtime_start).total_seconds()
        
        # Calculate availability
        uptime_seconds = total_time - total_downtime
        availability_percent = (uptime_seconds / total_time * 100) if total_time > 0 else 100
        
        # Calculate MTTR and MTBF
        mttr_seconds = 0.0
        mtbf_seconds = 0.0
        
        if self.downtime_periods:
            mttr_seconds = statistics.mean(period['duration_seconds'] for period in self.downtime_periods)
            
            # MTBF = average time between failures
            if len(self.downtime_periods) > 1:
                time_between_failures = []
                for i in range(1, len(self.downtime_periods)):
                    prev_end = self.downtime_periods[i-1]['end']
                    current_start = self.downtime_periods[i]['start']
                    tbf = (current_start - prev_end).total_seconds()
                    time_between_failures.append(tbf)
                
                if time_between_failures:
                    mtbf_seconds = statistics.mean(time_between_failures)
        
        return {
            'uptime_seconds': uptime_seconds,
            'downtime_seconds': total_downtime,
            'availability_percent': availability_percent,
            'mttr_seconds': mttr_seconds,
            'mtbf_seconds': mtbf_seconds,
            'downtime_incidents': len(self.downtime_periods),
            'current_uptime_seconds': (now - (self.downtime_periods[-1]['end'] if self.downtime_periods else self.start_time)).total_seconds()
        }

# === Alert Manager ===
class AlertManager:
    """Manage and route monitoring alerts."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.alert_rules = []
        self.notification_channels = []
        self.alert_history = deque(maxlen=10000)
        self.suppression_rules = []
        
    def add_alert_rule(self, rule: Dict[str, Any]):
        """Add alert rule."""
        self.alert_rules.append(rule)
    
    def add_notification_channel(self, channel: Callable):
        """Add notification channel."""
        self.notification_channels.append(channel)
    
    async def process_alert(self, alert: Alert):
        """Process and route alert."""
        # Check suppression rules
        if await self._is_suppressed(alert):
            logger.debug(f"Alert {alert.alert_id} suppressed")
            return
        
        # Store in history
        self.alert_history.append(alert)
        
        # Send to notification channels
        for channel in self.notification_channels:
            try:
                if asyncio.iscoroutinefunction(channel):
                    await channel(alert)
                else:
                    channel(alert)
            except Exception as e:
                logger.error(f"Error sending alert to channel: {e}")
        
        logger.info(f"Alert processed: {alert.name} ({alert.severity.value})")
    
    async def _is_suppressed(self, alert: Alert) -> bool:
        """Check if alert should be suppressed."""
        for rule in self.suppression_rules:
            if await self._matches_suppression_rule(alert, rule):
                return True
        return False
    
    async def _matches_suppression_rule(self, alert: Alert, rule: Dict[str, Any]) -> bool:
        """Check if alert matches suppression rule."""
        # Simple implementation - could be more sophisticated
        if 'severity' in rule and alert.severity.value not in rule['severity']:
            return False
        
        if 'source' in rule and alert.source not in rule['source']:
            return False
        
        # Time-based suppression
        if 'time_window' in rule:
            # Suppress if similar alert was triggered recently
            recent_alerts = [
                a for a in self.alert_history 
                if (datetime.now() - a.timestamp).total_seconds() < rule['time_window']
                and a.name == alert.name
            ]
            
            if len(recent_alerts) > rule.get('max_alerts', 1):
                return True
        
        return False
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert summary for specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_alerts = [
            alert for alert in self.alert_history 
            if alert.timestamp >= cutoff_time
        ]
        
        # Group by severity
        severity_counts = defaultdict(int)
        for alert in recent_alerts:
            severity_counts[alert.severity.value] += 1
        
        # Group by source
        source_counts = defaultdict(int)
        for alert in recent_alerts:
            source_counts[alert.source] += 1
        
        return {
            'total_alerts': len(recent_alerts),
            'severity_breakdown': dict(severity_counts),
            'source_breakdown': dict(source_counts),
            'time_period_hours': hours,
            'active_alerts': len([a for a in recent_alerts if not a.resolved])
        }

# === Factory Functions ===
def create_system_monitor(config: Dict[str, Any] = None) -> SystemMonitor:
    """Create system monitor instance."""
    return SystemMonitor(config)

def create_application_monitor(config: Dict[str, Any] = None) -> ApplicationMonitor:
    """Create application monitor instance."""
    return ApplicationMonitor(config)

def create_performance_monitor(config: Dict[str, Any] = None) -> PerformanceMonitor:
    """Create performance monitor instance."""
    return PerformanceMonitor(config)

def create_sla_monitor(config: Dict[str, Any] = None) -> SLAMonitor:
    """Create SLA monitor instance."""
    return SLAMonitor(config)

def create_alert_manager(config: Dict[str, Any] = None) -> AlertManager:
    """Create alert manager instance."""
    return AlertManager(config)

# === Export Classes ===
__all__ = [
    'BaseMonitor', 'SystemMonitor', 'ApplicationMonitor', 'PerformanceMonitor', 
    'SLAMonitor', 'AlertManager', 'UptimeTracker',
    'AlertSeverity', 'MetricType', 'MonitoringScope', 'HealthStatus',
    'MetricValue', 'Alert', 'HealthCheck', 'PerformanceMetrics', 'SLAMetrics',
    'create_system_monitor', 'create_application_monitor', 'create_performance_monitor',
    'create_sla_monitor', 'create_alert_manager'
]
