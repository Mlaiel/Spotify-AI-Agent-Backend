"""
Performance Monitoring System for Slack Alert Configuration.

This module provides comprehensive performance monitoring capabilities
including metrics collection, performance analysis, alerting, and
optimization recommendations for the Slack alert system.

Author: Fahed Mlaiel
Version: 1.0.0
"""

import asyncio
import time
import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import psutil
import threading

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from .constants import METRICS_CONFIG
from .exceptions import PerformanceError, PerformanceThresholdExceededError


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Performance alert information."""
    alert_id: str
    metric_name: str
    threshold_type: str  # 'above', 'below', 'equals'
    threshold_value: float
    current_value: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class Timer:
    """High-precision timer for performance measurement."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration: Optional[float] = None
    
    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.perf_counter()
    
    def stop(self) -> float:
        """Stop the timer and return duration."""
        if self.start_time is None:
            raise ValueError("Timer not started")
        
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        return self.duration
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.
    
    Provides real-time performance monitoring, metrics collection,
    alerting, and optimization recommendations.
    """

    def __init__(
        self,
        enable_prometheus: bool = True,
        metrics_port: int = 9090,
        metrics_path: str = "/metrics",
        alert_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
        retention_hours: int = 24
    ):
        """
        Initialize the performance monitor.
        
        Args:
            enable_prometheus: Enable Prometheus metrics export
            metrics_port: Port for Prometheus metrics server
            metrics_path: Path for Prometheus metrics endpoint
            alert_thresholds: Performance alert thresholds
            retention_hours: Hours to retain metrics data
        """
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.metrics_port = metrics_port
        self.metrics_path = metrics_path
        self.retention_hours = retention_hours
        
        # Metrics storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        
        # Performance alerts
        self.alert_thresholds = alert_thresholds or self._get_default_thresholds()
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_history: List[PerformanceAlert] = []
        
        # System monitoring
        self.system_metrics_enabled = True
        self.system_metrics_interval = 60  # seconds
        self.system_monitor_task: Optional[asyncio.Task] = None
        
        # Prometheus metrics
        self.prometheus_registry = None
        self.prometheus_metrics = {}
        self.prometheus_server_task: Optional[asyncio.Task] = None
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        # Logging
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """Initialize the performance monitor."""
        try:
            # Initialize Prometheus if enabled
            if self.enable_prometheus:
                await self._initialize_prometheus()
            
            # Start system monitoring
            if self.system_metrics_enabled:
                self.system_monitor_task = asyncio.create_task(self._system_monitor_loop())
            
            self.logger.info("PerformanceMonitor initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize PerformanceMonitor: {e}")
            raise PerformanceError(f"Initialization failed: {e}")

    def start_timer(self, name: str) -> Timer:
        """Start a new performance timer."""
        timer = Timer(name)
        timer.start()
        return timer

    def end_timer(self, timer: Timer) -> float:
        """End a performance timer and record the measurement."""
        duration = timer.stop()
        
        with self._lock:
            self.timers[timer.name].append(duration)
            
            # Update Prometheus histogram if available
            if self.enable_prometheus and timer.name in self.prometheus_metrics:
                self.prometheus_metrics[timer.name].observe(duration)
        
        # Check for performance alerts
        asyncio.create_task(self._check_timer_alert(timer.name, duration))
        
        return duration

    @asynccontextmanager
    async def timer_context(self, name: str):
        """Context manager for timing operations."""
        timer = self.start_timer(name)
        try:
            yield timer
        finally:
            self.end_timer(timer)

    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        with self._lock:
            self.counters[name] += value
            
            # Update Prometheus counter if available
            if self.enable_prometheus and name in self.prometheus_metrics:
                if tags:
                    self.prometheus_metrics[name].labels(**tags).inc(value)
                else:
                    self.prometheus_metrics[name].inc(value)

    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric value."""
        with self._lock:
            self.gauges[name] = value
            
            # Update Prometheus gauge if available
            if self.enable_prometheus and name in self.prometheus_metrics:
                if tags:
                    self.prometheus_metrics[name].labels(**tags).set(value)
                else:
                    self.prometheus_metrics[name].set(value)

    def record_metric(
        self,
        name: str,
        value: float,
        unit: str = "",
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a custom metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.utcnow(),
            tags=tags or {},
            metadata=metadata or {}
        )
        
        with self._lock:
            self.metrics[name].append(metric)
        
        # Check for alerts
        asyncio.create_task(self._check_metric_alert(name, value))

    async def get_metrics(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics."""
        with self._lock:
            if name:
                # Get specific metric
                if name in self.metrics:
                    metrics_list = list(self.metrics[name])
                    return {
                        "name": name,
                        "count": len(metrics_list),
                        "latest": metrics_list[-1] if metrics_list else None,
                        "values": [m.value for m in metrics_list[-100:]]  # Last 100 values
                    }
                else:
                    return {}
            else:
                # Get all metrics summary
                summary = {}
                
                # Timer metrics
                timer_summary = {}
                for timer_name, durations in self.timers.items():
                    if durations:
                        timer_summary[timer_name] = {
                            "count": len(durations),
                            "avg": statistics.mean(durations),
                            "min": min(durations),
                            "max": max(durations),
                            "p95": statistics.quantiles(durations, n=20)[18] if len(durations) > 20 else max(durations),
                            "p99": statistics.quantiles(durations, n=100)[98] if len(durations) > 100 else max(durations)
                        }
                summary["timers"] = timer_summary
                
                # Counter metrics
                summary["counters"] = dict(self.counters)
                
                # Gauge metrics
                summary["gauges"] = dict(self.gauges)
                
                # Custom metrics
                custom_summary = {}
                for metric_name, metric_list in self.metrics.items():
                    if metric_list:
                        values = [m.value for m in metric_list]
                        custom_summary[metric_name] = {
                            "count": len(values),
                            "latest": values[-1],
                            "avg": statistics.mean(values),
                            "min": min(values),
                            "max": max(values)
                        }
                summary["custom"] = custom_summary
                
                return summary

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # Network metrics
            network_io = psutil.net_io_counters()
            
            return {
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "frequency": cpu_freq.current if cpu_freq else None
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "percent": memory.percent
                },
                "swap": {
                    "total": swap.total,
                    "used": swap.used,
                    "free": swap.free,
                    "percent": swap.percent
                },
                "disk": {
                    "total": disk_usage.total,
                    "used": disk_usage.used,
                    "free": disk_usage.free,
                    "percent": (disk_usage.used / disk_usage.total) * 100,
                    "read_bytes": disk_io.read_bytes if disk_io else 0,
                    "write_bytes": disk_io.write_bytes if disk_io else 0
                },
                "network": {
                    "bytes_sent": network_io.bytes_sent,
                    "bytes_recv": network_io.bytes_recv,
                    "packets_sent": network_io.packets_sent,
                    "packets_recv": network_io.packets_recv
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system metrics: {e}")
            return {}

    async def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        try:
            metrics = await self.get_metrics()
            system_metrics = await self.get_system_metrics()
            
            # Performance analysis
            analysis = await self._analyze_performance()
            
            # Active alerts
            active_alerts = list(self.active_alerts.values())
            
            # Recommendations
            recommendations = await self._generate_recommendations()
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": metrics,
                "system": system_metrics,
                "analysis": analysis,
                "alerts": {
                    "active": [alert.__dict__ for alert in active_alerts],
                    "total_active": len(active_alerts),
                    "history_count": len(self.alert_history)
                },
                "recommendations": recommendations,
                "health_score": await self._calculate_health_score()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {e}")
            return {"error": str(e)}

    async def record_security_event(
        self,
        action: str,
        result: str,
        user_id: str,
        tenant_id: str
    ) -> None:
        """Record security-related performance event."""
        self.increment_counter(f"security_{action}_{result}")
        
        # Record timing if applicable
        if result == "success":
            self.record_metric(
                f"security_{action}_success_rate",
                1.0,
                "ratio",
                tags={"tenant_id": tenant_id}
            )
        else:
            self.record_metric(
                f"security_{action}_failure_rate",
                1.0,
                "ratio",
                tags={"tenant_id": tenant_id}
            )

    async def optimize_performance(self) -> Dict[str, Any]:
        """Analyze and optimize system performance."""
        try:
            optimizations = []
            
            # Analyze timer performance
            timer_analysis = await self._analyze_timers()
            if timer_analysis["slow_operations"]:
                optimizations.extend(timer_analysis["recommendations"])
            
            # Analyze memory usage
            system_metrics = await self.get_system_metrics()
            memory_percent = system_metrics.get("memory", {}).get("percent", 0)
            
            if memory_percent > 80:
                optimizations.append({
                    "type": "memory",
                    "issue": "High memory usage",
                    "current": f"{memory_percent:.1f}%",
                    "recommendation": "Consider increasing memory or optimizing cache usage"
                })
            
            # Analyze cache performance
            cache_analysis = await self._analyze_cache_performance()
            if cache_analysis["recommendations"]:
                optimizations.extend(cache_analysis["recommendations"])
            
            return {
                "optimizations": optimizations,
                "priority_actions": [opt for opt in optimizations if opt.get("priority") == "high"],
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Performance optimization failed: {e}")
            return {"error": str(e)}

    # Private helper methods
    
    async def _initialize_prometheus(self) -> None:
        """Initialize Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            self.logger.warning("Prometheus client not available")
            return
        
        try:
            self.prometheus_registry = CollectorRegistry()
            
            # Define common metrics
            self.prometheus_metrics = {
                # Timers as histograms
                "config_operation_duration": Histogram(
                    "config_operation_duration_seconds",
                    "Duration of configuration operations",
                    ["operation"],
                    registry=self.prometheus_registry
                ),
                "template_render_duration": Histogram(
                    "template_render_duration_seconds",
                    "Duration of template rendering",
                    ["template"],
                    registry=self.prometheus_registry
                ),
                "locale_lookup_duration": Histogram(
                    "locale_lookup_duration_seconds",
                    "Duration of locale lookups",
                    ["locale"],
                    registry=self.prometheus_registry
                ),
                
                # Counters
                "config_operations_total": Counter(
                    "config_operations_total",
                    "Total configuration operations",
                    ["operation", "result"],
                    registry=self.prometheus_registry
                ),
                "template_renders_total": Counter(
                    "template_renders_total",
                    "Total template renders",
                    ["template", "result"],
                    registry=self.prometheus_registry
                ),
                "security_events_total": Counter(
                    "security_events_total",
                    "Total security events",
                    ["action", "result"],
                    registry=self.prometheus_registry
                ),
                
                # Gauges
                "active_sessions": Gauge(
                    "active_sessions",
                    "Number of active sessions",
                    registry=self.prometheus_registry
                ),
                "cache_hit_ratio": Gauge(
                    "cache_hit_ratio",
                    "Cache hit ratio",
                    ["cache_type"],
                    registry=self.prometheus_registry
                ),
                "system_cpu_percent": Gauge(
                    "system_cpu_percent",
                    "System CPU usage percentage",
                    registry=self.prometheus_registry
                ),
                "system_memory_percent": Gauge(
                    "system_memory_percent",
                    "System memory usage percentage",
                    registry=self.prometheus_registry
                )
            }
            
            # Start metrics server
            if METRICS_CONFIG.get("enable_prometheus", True):
                start_http_server(self.metrics_port, registry=self.prometheus_registry)
                self.logger.info(f"Prometheus metrics server started on port {self.metrics_port}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Prometheus: {e}")

    async def _system_monitor_loop(self) -> None:
        """Background loop for system monitoring."""
        while True:
            try:
                system_metrics = await self.get_system_metrics()
                
                # Record system metrics
                if system_metrics:
                    self.set_gauge("system_cpu_percent", system_metrics["cpu"]["percent"])
                    self.set_gauge("system_memory_percent", system_metrics["memory"]["percent"])
                    self.set_gauge("system_disk_percent", system_metrics["disk"]["percent"])
                
                # Check system alerts
                await self._check_system_alerts(system_metrics)
                
                await asyncio.sleep(self.system_metrics_interval)
                
            except Exception as e:
                self.logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(self.system_metrics_interval)

    async def _check_timer_alert(self, name: str, duration: float) -> None:
        """Check if a timer duration exceeds thresholds."""
        try:
            thresholds = self.alert_thresholds.get("timers", {})
            threshold = thresholds.get(name, thresholds.get("default", 5.0))
            
            if duration > threshold:
                alert_id = f"timer_{name}_{int(time.time())}"
                alert = PerformanceAlert(
                    alert_id=alert_id,
                    metric_name=name,
                    threshold_type="above",
                    threshold_value=threshold,
                    current_value=duration,
                    severity="medium" if duration < threshold * 2 else "high",
                    message=f"Timer {name} duration {duration:.3f}s exceeds threshold {threshold:.3f}s",
                    timestamp=datetime.utcnow()
                )
                
                self.active_alerts[alert_id] = alert
                self.alert_history.append(alert)
                
                self.logger.warning(f"Performance alert: {alert.message}")
                
        except Exception as e:
            self.logger.error(f"Timer alert check failed: {e}")

    async def _check_metric_alert(self, name: str, value: float) -> None:
        """Check if a metric value exceeds thresholds."""
        try:
            thresholds = self.alert_thresholds.get("metrics", {})
            threshold_config = thresholds.get(name)
            
            if not threshold_config:
                return
            
            threshold_value = threshold_config.get("value")
            threshold_type = threshold_config.get("type", "above")
            
            if threshold_value is None:
                return
            
            alert_triggered = False
            
            if threshold_type == "above" and value > threshold_value:
                alert_triggered = True
            elif threshold_type == "below" and value < threshold_value:
                alert_triggered = True
            elif threshold_type == "equals" and abs(value - threshold_value) < 0.001:
                alert_triggered = True
            
            if alert_triggered:
                alert_id = f"metric_{name}_{int(time.time())}"
                alert = PerformanceAlert(
                    alert_id=alert_id,
                    metric_name=name,
                    threshold_type=threshold_type,
                    threshold_value=threshold_value,
                    current_value=value,
                    severity=threshold_config.get("severity", "medium"),
                    message=f"Metric {name} value {value} {threshold_type} threshold {threshold_value}",
                    timestamp=datetime.utcnow()
                )
                
                self.active_alerts[alert_id] = alert
                self.alert_history.append(alert)
                
                self.logger.warning(f"Performance alert: {alert.message}")
                
        except Exception as e:
            self.logger.error(f"Metric alert check failed: {e}")

    async def _check_system_alerts(self, system_metrics: Dict[str, Any]) -> None:
        """Check system metrics for alerts."""
        try:
            # CPU usage alert
            cpu_percent = system_metrics.get("cpu", {}).get("percent", 0)
            if cpu_percent > 80:
                alert_id = f"system_cpu_{int(time.time())}"
                alert = PerformanceAlert(
                    alert_id=alert_id,
                    metric_name="system_cpu_percent",
                    threshold_type="above",
                    threshold_value=80.0,
                    current_value=cpu_percent,
                    severity="high" if cpu_percent > 90 else "medium",
                    message=f"High CPU usage: {cpu_percent:.1f}%",
                    timestamp=datetime.utcnow()
                )
                
                if alert_id not in self.active_alerts:
                    self.active_alerts[alert_id] = alert
                    self.alert_history.append(alert)
            
            # Memory usage alert
            memory_percent = system_metrics.get("memory", {}).get("percent", 0)
            if memory_percent > 85:
                alert_id = f"system_memory_{int(time.time())}"
                alert = PerformanceAlert(
                    alert_id=alert_id,
                    metric_name="system_memory_percent",
                    threshold_type="above",
                    threshold_value=85.0,
                    current_value=memory_percent,
                    severity="high" if memory_percent > 95 else "medium",
                    message=f"High memory usage: {memory_percent:.1f}%",
                    timestamp=datetime.utcnow()
                )
                
                if alert_id not in self.active_alerts:
                    self.active_alerts[alert_id] = alert
                    self.alert_history.append(alert)
                    
        except Exception as e:
            self.logger.error(f"System alerts check failed: {e}")

    def _get_default_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """Get default performance alert thresholds."""
        return {
            "timers": {
                "default": 5.0,  # 5 seconds
                "get_tenant_config": 1.0,
                "render_template": 0.5,
                "encrypt_data": 0.1,
                "decrypt_data": 0.1
            },
            "metrics": {
                "cache_hit_ratio": {
                    "value": 0.8,
                    "type": "below",
                    "severity": "medium"
                },
                "error_rate": {
                    "value": 0.05,
                    "type": "above", 
                    "severity": "high"
                }
            },
            "system": {
                "cpu_percent": {
                    "value": 80.0,
                    "type": "above",
                    "severity": "medium"
                },
                "memory_percent": {
                    "value": 85.0,
                    "type": "above",
                    "severity": "high"
                }
            }
        }

    async def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze current performance state."""
        try:
            analysis = {
                "overall_health": "good",
                "bottlenecks": [],
                "trends": {},
                "efficiency_score": 0.85
            }
            
            # Analyze timer performance
            with self._lock:
                slow_timers = []
                for timer_name, durations in self.timers.items():
                    if durations:
                        avg_duration = statistics.mean(durations)
                        if avg_duration > 1.0:  # Slow operations > 1 second
                            slow_timers.append({
                                "name": timer_name,
                                "avg_duration": avg_duration,
                                "count": len(durations)
                            })
                
                if slow_timers:
                    analysis["bottlenecks"].extend(slow_timers)
                    analysis["overall_health"] = "degraded"
            
            # Check active alerts
            if len(self.active_alerts) > 5:
                analysis["overall_health"] = "poor"
            elif len(self.active_alerts) > 0:
                analysis["overall_health"] = "warning"
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Performance analysis failed: {e}")
            return {"error": str(e)}

    async def _analyze_timers(self) -> Dict[str, Any]:
        """Analyze timer performance."""
        try:
            analysis = {
                "slow_operations": [],
                "recommendations": []
            }
            
            with self._lock:
                for timer_name, durations in self.timers.items():
                    if not durations:
                        continue
                    
                    avg_duration = statistics.mean(durations)
                    max_duration = max(durations)
                    
                    if avg_duration > 2.0:  # Average > 2 seconds
                        analysis["slow_operations"].append({
                            "name": timer_name,
                            "avg_duration": avg_duration,
                            "max_duration": max_duration,
                            "count": len(durations)
                        })
                        
                        analysis["recommendations"].append({
                            "type": "performance",
                            "issue": f"Slow operation: {timer_name}",
                            "current": f"{avg_duration:.2f}s average",
                            "recommendation": f"Optimize {timer_name} operation",
                            "priority": "high" if avg_duration > 5.0 else "medium"
                        })
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Timer analysis failed: {e}")
            return {"slow_operations": [], "recommendations": []}

    async def _analyze_cache_performance(self) -> Dict[str, Any]:
        """Analyze cache performance."""
        try:
            analysis = {
                "hit_ratios": {},
                "recommendations": []
            }
            
            # This would typically analyze actual cache metrics
            # For now, we'll provide general recommendations
            
            # Check if cache hit ratio metrics exist
            with self._lock:
                cache_metrics = {k: v for k, v in self.gauges.items() if "cache" in k.lower()}
            
            if not cache_metrics:
                analysis["recommendations"].append({
                    "type": "cache",
                    "issue": "No cache metrics available",
                    "recommendation": "Implement cache performance monitoring",
                    "priority": "medium"
                })
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Cache analysis failed: {e}")
            return {"hit_ratios": {}, "recommendations": []}

    async def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate performance optimization recommendations."""
        try:
            recommendations = []
            
            # System resource recommendations
            system_metrics = await self.get_system_metrics()
            
            cpu_percent = system_metrics.get("cpu", {}).get("percent", 0)
            if cpu_percent > 70:
                recommendations.append({
                    "type": "system",
                    "category": "cpu",
                    "issue": f"High CPU usage: {cpu_percent:.1f}%",
                    "recommendation": "Consider CPU optimization or scaling",
                    "priority": "high" if cpu_percent > 90 else "medium"
                })
            
            memory_percent = system_metrics.get("memory", {}).get("percent", 0)
            if memory_percent > 80:
                recommendations.append({
                    "type": "system",
                    "category": "memory",
                    "issue": f"High memory usage: {memory_percent:.1f}%",
                    "recommendation": "Consider memory optimization or increase RAM",
                    "priority": "high" if memory_percent > 95 else "medium"
                })
            
            # Performance pattern recommendations
            metrics = await self.get_metrics()
            
            # Check for cache performance
            if "cache_hit_ratio" in metrics.get("gauges", {}):
                hit_ratio = metrics["gauges"]["cache_hit_ratio"]
                if hit_ratio < 0.8:
                    recommendations.append({
                        "type": "cache",
                        "category": "hit_ratio",
                        "issue": f"Low cache hit ratio: {hit_ratio:.2f}",
                        "recommendation": "Optimize cache strategy or increase cache size",
                        "priority": "medium"
                    })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Recommendations generation failed: {e}")
            return []

    async def _calculate_health_score(self) -> float:
        """Calculate overall system health score (0-1)."""
        try:
            score = 1.0
            
            # Factor in active alerts
            alert_penalty = len(self.active_alerts) * 0.1
            score -= min(alert_penalty, 0.5)  # Max 50% penalty for alerts
            
            # Factor in system resources
            system_metrics = await self.get_system_metrics()
            
            cpu_percent = system_metrics.get("cpu", {}).get("percent", 0)
            memory_percent = system_metrics.get("memory", {}).get("percent", 0)
            
            # CPU penalty
            if cpu_percent > 80:
                score -= (cpu_percent - 80) / 100  # Up to 20% penalty
            
            # Memory penalty
            if memory_percent > 80:
                score -= (memory_percent - 80) / 100  # Up to 20% penalty
            
            # Factor in error rates
            metrics = await self.get_metrics()
            error_counters = {k: v for k, v in metrics.get("counters", {}).items() if "error" in k.lower()}
            
            if error_counters:
                total_errors = sum(error_counters.values())
                if total_errors > 10:
                    score -= min(total_errors / 1000, 0.2)  # Up to 20% penalty for errors
            
            return max(score, 0.0)  # Ensure score doesn't go below 0
            
        except Exception as e:
            self.logger.error(f"Health score calculation failed: {e}")
            return 0.5  # Default moderate health score

    async def close(self) -> None:
        """Clean up resources."""
        try:
            # Stop system monitoring
            if self.system_monitor_task:
                self.system_monitor_task.cancel()
                try:
                    await self.system_monitor_task
                except asyncio.CancelledError:
                    pass
            
            # Stop Prometheus server
            if self.prometheus_server_task:
                self.prometheus_server_task.cancel()
                try:
                    await self.prometheus_server_task
                except asyncio.CancelledError:
                    pass
            
            self.logger.info("PerformanceMonitor closed successfully")
            
        except Exception as e:
            self.logger.error(f"Error closing PerformanceMonitor: {e}")
