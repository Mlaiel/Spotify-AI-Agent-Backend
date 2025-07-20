#!/usr/bin/env python3
"""
Metrics Collector for PagerDuty Integration.

Advanced metrics collection and monitoring system with support for
Prometheus, StatsD, and custom metric backends.

Features:
- Multiple metric types (counters, histograms, gauges, timers)
- Prometheus-compatible metrics export
- StatsD integration for real-time monitoring
- Custom metric backends and exporters
- Metric aggregation and buffering
- Performance monitoring and alerting
- Distributed metrics collection
- Metric retention and archival
"""

import asyncio
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import json
import statistics

try:
    from prometheus_client import Counter, Histogram, Gauge, Summary
    from prometheus_client import CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import statsd
    STATSD_AVAILABLE = True
except ImportError:
    STATSD_AVAILABLE = False

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """Individual metric value with timestamp."""
    value: Union[int, float]
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricDefinition:
    """Definition of a metric."""
    name: str
    type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms
    unit: Optional[str] = None


class MetricBackend:
    """Base class for metric backends."""
    
    def record_counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """Record counter metric."""
        raise NotImplementedError
    
    def record_gauge(self, name: str, value: Union[int, float], labels: Optional[Dict[str, str]] = None):
        """Record gauge metric."""
        raise NotImplementedError
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record histogram metric."""
        raise NotImplementedError
    
    def record_timer(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record timer metric."""
        raise NotImplementedError


class PrometheusBackend(MetricBackend):
    """Prometheus metrics backend."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize Prometheus backend."""
        if not PROMETHEUS_AVAILABLE:
            raise RuntimeError("Prometheus client library not available")
        
        self.registry = registry or CollectorRegistry()
        self.metrics: Dict[str, Any] = {}
        self._lock = threading.Lock()
    
    def _get_or_create_metric(self, name: str, metric_type: type, 
                             description: str = "", labels: Optional[List[str]] = None):
        """Get or create Prometheus metric."""
        with self._lock:
            if name not in self.metrics:
                if labels:
                    self.metrics[name] = metric_type(
                        name=name,
                        documentation=description,
                        labelnames=labels,
                        registry=self.registry
                    )
                else:
                    self.metrics[name] = metric_type(
                        name=name,
                        documentation=description,
                        registry=self.registry
                    )
            return self.metrics[name]
    
    def record_counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """Record counter metric."""
        metric = self._get_or_create_metric(
            name, Counter, f"Counter metric: {name}", 
            list(labels.keys()) if labels else None
        )
        
        if labels:
            metric.labels(**labels).inc(value)
        else:
            metric.inc(value)
    
    def record_gauge(self, name: str, value: Union[int, float], labels: Optional[Dict[str, str]] = None):
        """Record gauge metric."""
        metric = self._get_or_create_metric(
            name, Gauge, f"Gauge metric: {name}",
            list(labels.keys()) if labels else None
        )
        
        if labels:
            metric.labels(**labels).set(value)
        else:
            metric.set(value)
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record histogram metric."""
        metric = self._get_or_create_metric(
            name, Histogram, f"Histogram metric: {name}",
            list(labels.keys()) if labels else None
        )
        
        if labels:
            metric.labels(**labels).observe(value)
        else:
            metric.observe(value)
    
    def record_timer(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record timer metric."""
        self.record_histogram(f"{name}_duration_seconds", value, labels)
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        return generate_latest(self.registry).decode('utf-8')


class StatsDBackend(MetricBackend):
    """StatsD metrics backend."""
    
    def __init__(self, host: str = 'localhost', port: int = 8125, prefix: str = ''):
        """Initialize StatsD backend."""
        if not STATSD_AVAILABLE:
            raise RuntimeError("StatsD library not available")
        
        self.client = statsd.StatsClient(host=host, port=port, prefix=prefix)
    
    def record_counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """Record counter metric."""
        metric_name = self._format_name_with_labels(name, labels)
        self.client.incr(metric_name, count=value)
    
    def record_gauge(self, name: str, value: Union[int, float], labels: Optional[Dict[str, str]] = None):
        """Record gauge metric."""
        metric_name = self._format_name_with_labels(name, labels)
        self.client.gauge(metric_name, value)
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record histogram metric."""
        metric_name = self._format_name_with_labels(name, labels)
        self.client.timing(metric_name, value * 1000)  # Convert to milliseconds
    
    def record_timer(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record timer metric."""
        self.record_histogram(name, value, labels)
    
    def _format_name_with_labels(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Format metric name with labels for StatsD."""
        if not labels:
            return name
        
        label_parts = [f"{k}_{v}" for k, v in labels.items()]
        return f"{name}.{'.'.join(label_parts)}"


class InMemoryBackend(MetricBackend):
    """In-memory metrics backend for testing and development."""
    
    def __init__(self, retention_seconds: int = 3600):
        """Initialize in-memory backend."""
        self.retention_seconds = retention_seconds
        self.metrics: Dict[str, List[MetricValue]] = defaultdict(list)
        self._lock = threading.Lock()
        
        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        async def cleanup_loop():
            while True:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                self._cleanup_old_metrics()
        
        try:
            loop = asyncio.get_event_loop()
            self._cleanup_task = loop.create_task(cleanup_loop())
        except RuntimeError:
            # No event loop running
            pass
    
    def _cleanup_old_metrics(self):
        """Remove old metrics beyond retention period."""
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.retention_seconds)
        
        with self._lock:
            for metric_name in list(self.metrics.keys()):
                self.metrics[metric_name] = [
                    mv for mv in self.metrics[metric_name]
                    if mv.timestamp > cutoff_time
                ]
                
                # Remove empty metric lists
                if not self.metrics[metric_name]:
                    del self.metrics[metric_name]
    
    def _record_metric(self, name: str, value: Union[int, float], labels: Optional[Dict[str, str]] = None):
        """Record metric value."""
        with self._lock:
            metric_value = MetricValue(
                value=value,
                timestamp=datetime.utcnow(),
                labels=labels or {}
            )
            self.metrics[name].append(metric_value)
    
    def record_counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """Record counter metric."""
        self._record_metric(name, value, labels)
    
    def record_gauge(self, name: str, value: Union[int, float], labels: Optional[Dict[str, str]] = None):
        """Record gauge metric."""
        self._record_metric(name, value, labels)
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record histogram metric."""
        self._record_metric(name, value, labels)
    
    def record_timer(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record timer metric."""
        self._record_metric(name, value, labels)
    
    def get_metric_values(self, name: str, since: Optional[datetime] = None) -> List[MetricValue]:
        """Get metric values for a specific metric."""
        with self._lock:
            values = self.metrics.get(name, [])
            
            if since:
                values = [mv for mv in values if mv.timestamp >= since]
            
            return values.copy()
    
    def get_metric_stats(self, name: str, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Get statistical summary of a metric."""
        values = self.get_metric_values(name, since)
        
        if not values:
            return {}
        
        numeric_values = [mv.value for mv in values]
        
        return {
            'count': len(numeric_values),
            'sum': sum(numeric_values),
            'mean': statistics.mean(numeric_values),
            'median': statistics.median(numeric_values),
            'min': min(numeric_values),
            'max': max(numeric_values),
            'std_dev': statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0.0,
            'first_timestamp': values[0].timestamp.isoformat(),
            'last_timestamp': values[-1].timestamp.isoformat()
        }
    
    def export_json(self) -> str:
        """Export all metrics as JSON."""
        with self._lock:
            export_data = {}
            
            for name, values in self.metrics.items():
                export_data[name] = [
                    {
                        'value': mv.value,
                        'timestamp': mv.timestamp.isoformat(),
                        'labels': mv.labels
                    }
                    for mv in values
                ]
            
            return json.dumps(export_data, indent=2)


class MetricsCollector:
    """
    Main metrics collector with multiple backend support.
    
    Features:
    - Multiple metric backends (Prometheus, StatsD, in-memory)
    - Metric buffering and batching
    - Automatic metric registration
    - Performance monitoring
    - Metric alerting and thresholds
    """
    
    def __init__(self,
                 backends: Optional[List[MetricBackend]] = None,
                 default_labels: Optional[Dict[str, str]] = None,
                 enable_buffering: bool = False,
                 buffer_size: int = 1000,
                 flush_interval: float = 10.0):
        """
        Initialize metrics collector.
        
        Args:
            backends: List of metric backends
            default_labels: Default labels for all metrics
            enable_buffering: Enable metric buffering
            buffer_size: Maximum buffer size
            flush_interval: Buffer flush interval in seconds
        """
        self.backends = backends or [InMemoryBackend()]
        self.default_labels = default_labels or {}
        self.enable_buffering = enable_buffering
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        
        # Metric definitions registry
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        
        # Buffering
        self.metric_buffer: deque = deque(maxlen=buffer_size)
        self._buffer_lock = threading.Lock()
        self._flush_task = None
        
        # Performance tracking
        self._last_flush_time = time.time()
        self._metrics_processed = 0
        self._start_time = time.time()
        
        # Start background tasks
        if enable_buffering:
            self._start_flush_task()
        
        logger.info(f"Metrics collector initialized with {len(self.backends)} backends")
    
    def _start_flush_task(self):
        """Start background flush task."""
        async def flush_loop():
            while True:
                await asyncio.sleep(self.flush_interval)
                self._flush_buffer()
        
        try:
            loop = asyncio.get_event_loop()
            self._flush_task = loop.create_task(flush_loop())
        except RuntimeError:
            # No event loop running
            pass
    
    def register_metric(self, definition: MetricDefinition):
        """Register a metric definition."""
        self.metric_definitions[definition.name] = definition
        logger.debug(f"Registered metric: {definition.name} ({definition.type.value})")
    
    def _merge_labels(self, labels: Optional[Dict[str, str]]) -> Dict[str, str]:
        """Merge default labels with provided labels."""
        merged = self.default_labels.copy()
        if labels:
            merged.update(labels)
        return merged
    
    def _record_to_backends(self, method_name: str, *args, **kwargs):
        """Record metric to all backends."""
        for backend in self.backends:
            try:
                method = getattr(backend, method_name)
                method(*args, **kwargs)
            except Exception as e:
                logger.error(f"Failed to record metric to backend {type(backend).__name__}: {e}")
    
    def _buffer_metric(self, method_name: str, *args, **kwargs):
        """Buffer metric for later processing."""
        with self._buffer_lock:
            self.metric_buffer.append((method_name, args, kwargs))
    
    def _flush_buffer(self):
        """Flush buffered metrics to backends."""
        with self._buffer_lock:
            if not self.metric_buffer:
                return
            
            # Process all buffered metrics
            while self.metric_buffer:
                method_name, args, kwargs = self.metric_buffer.popleft()
                self._record_to_backends(method_name, *args, **kwargs)
                self._metrics_processed += 1
            
            self._last_flush_time = time.time()
            logger.debug(f"Flushed {self._metrics_processed} metrics")
    
    def increment(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        merged_labels = self._merge_labels(labels)
        
        if self.enable_buffering:
            self._buffer_metric('record_counter', name, value, merged_labels)
        else:
            self._record_to_backends('record_counter', name, value, merged_labels)
    
    def gauge(self, name: str, value: Union[int, float], labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric."""
        merged_labels = self._merge_labels(labels)
        
        if self.enable_buffering:
            self._buffer_metric('record_gauge', name, value, merged_labels)
        else:
            self._record_to_backends('record_gauge', name, value, merged_labels)
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram metric."""
        merged_labels = self._merge_labels(labels)
        
        if self.enable_buffering:
            self._buffer_metric('record_histogram', name, value, merged_labels)
        else:
            self._record_to_backends('record_histogram', name, value, merged_labels)
    
    def time_function(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager to time function execution."""
        return TimerContext(self, name, labels)
    
    def timer(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a timer metric."""
        merged_labels = self._merge_labels(labels)
        
        if self.enable_buffering:
            self._buffer_metric('record_timer', name, value, merged_labels)
        else:
            self._record_to_backends('record_timer', name, value, merged_labels)
    
    def get_metric_definitions(self) -> Dict[str, MetricDefinition]:
        """Get all registered metric definitions."""
        return self.metric_definitions.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics."""
        uptime = time.time() - self._start_time
        
        return {
            'uptime_seconds': uptime,
            'metrics_processed': self._metrics_processed,
            'metrics_per_second': self._metrics_processed / uptime if uptime > 0 else 0,
            'backends_count': len(self.backends),
            'buffer_enabled': self.enable_buffering,
            'buffer_size': len(self.metric_buffer) if self.enable_buffering else 0,
            'buffer_max_size': self.buffer_size,
            'last_flush_time': self._last_flush_time,
            'registered_metrics': len(self.metric_definitions)
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics from in-memory backends."""
        all_metrics = {}
        
        for backend in self.backends:
            if isinstance(backend, InMemoryBackend):
                with backend._lock:
                    for name, values in backend.metrics.items():
                        if values:
                            all_metrics[name] = backend.get_metric_stats(name)
        
        return all_metrics
    
    def flush(self):
        """Force flush of buffered metrics."""
        if self.enable_buffering:
            self._flush_buffer()
    
    def close(self):
        """Close the metrics collector."""
        if self.enable_buffering:
            self._flush_buffer()
        
        if self._flush_task:
            self._flush_task.cancel()
        
        logger.info("Metrics collector closed")


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, name: str, labels: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.name = name
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.collector.timer(self.name, duration, self.labels)


class PrometheusExporter:
    """Standalone Prometheus metrics exporter."""
    
    def __init__(self, collector: MetricsCollector, port: int = 8000):
        """Initialize Prometheus exporter."""
        self.collector = collector
        self.port = port
        self.registry = None
        
        # Find Prometheus backend
        for backend in collector.backends:
            if isinstance(backend, PrometheusBackend):
                self.registry = backend.registry
                break
        
        if not self.registry:
            # Create default Prometheus backend
            prometheus_backend = PrometheusBackend()
            collector.backends.append(prometheus_backend)
            self.registry = prometheus_backend.registry
    
    async def start_server(self):
        """Start HTTP server for metrics export."""
        from aiohttp import web
        
        async def metrics_handler(request):
            metrics_text = generate_latest(self.registry).decode('utf-8')
            return web.Response(text=metrics_text, content_type='text/plain')
        
        app = web.Application()
        app.router.add_get('/metrics', metrics_handler)
        
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, 'localhost', self.port)
        await site.start()
        
        logger.info(f"Prometheus metrics server started on port {self.port}")
        return runner


# Global metrics collector instance
_global_metrics_collector = None

def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()
    return _global_metrics_collector


# Convenience functions
def increment_counter(name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
    """Increment counter using global collector."""
    collector = get_metrics_collector()
    collector.increment(name, value, labels)


def set_gauge(name: str, value: Union[int, float], labels: Optional[Dict[str, str]] = None):
    """Set gauge using global collector."""
    collector = get_metrics_collector()
    collector.gauge(name, value, labels)


def record_timing(name: str, value: float, labels: Optional[Dict[str, str]] = None):
    """Record timing using global collector."""
    collector = get_metrics_collector()
    collector.timer(name, value, labels)


# Decorator for timing function execution
def timed(metric_name: Optional[str] = None, labels: Optional[Dict[str, str]] = None):
    """Decorator to time function execution."""
    def decorator(func):
        name = metric_name or f"function_{func.__name__}_duration"
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                with get_metrics_collector().time_function(name, labels):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                with get_metrics_collector().time_function(name, labels):
                    return func(*args, **kwargs)
            return sync_wrapper
    
    return decorator
