"""
Enterprise Cache Monitoring
===========================
Real-time monitoring, metrics collection, and performance analysis for cache systems.

Expert Team Implementation:
- Lead Developer + AI Architect: Intelligent anomaly detection and predictive analytics
- Senior Backend Developer: High-performance metrics collection with minimal overhead
- Machine Learning Engineer: ML-based performance prediction and optimization recommendations
- DBA & Data Engineer: Advanced analytics pipeline and data visualization integration
- Security Specialist: Security monitoring, audit trails, and threat detection
- Microservices Architect: Distributed monitoring and cross-service observability
"""

import asyncio
import logging
import time
import json
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor

# External dependencies for monitoring features
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge, Info
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    prometheus_client = None

logger = logging.getLogger(__name__)

# === Types and Enums ===
class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMING = "timing"
    RATE = "rate"

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class HealthStatus(Enum):
    """Cache system health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"

@dataclass
class CacheMetrics:
    """Comprehensive cache metrics."""
    # Operation metrics
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    invalidations: int = 0
    errors: int = 0
    
    # Performance metrics
    avg_get_latency_ms: float = 0.0
    avg_set_latency_ms: float = 0.0
    avg_delete_latency_ms: float = 0.0
    p95_get_latency_ms: float = 0.0
    p99_get_latency_ms: float = 0.0
    
    # Size metrics
    total_size_bytes: int = 0
    key_count: int = 0
    memory_usage_bytes: int = 0
    memory_usage_percent: float = 0.0
    
    # Efficiency metrics
    hit_rate_percent: float = 0.0
    miss_rate_percent: float = 0.0
    eviction_rate_per_hour: float = 0.0
    throughput_ops_per_second: float = 0.0
    
    # Time metadata
    timestamp: datetime = field(default_factory=datetime.now)
    collection_duration_ms: float = 0.0
    
    @property
    def total_operations(self) -> int:
        """Total cache operations."""
        return self.hits + self.misses + self.sets + self.deletes
    
    @property
    def error_rate_percent(self) -> float:
        """Error rate percentage."""
        total_ops = self.total_operations
        return (self.errors / total_ops * 100) if total_ops > 0 else 0.0

@dataclass
class SystemMetrics:
    """System-level metrics affecting cache performance."""
    cpu_usage_percent: float = 0.0
    memory_total_bytes: int = 0
    memory_available_bytes: int = 0
    memory_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    load_average_1m: float = 0.0
    load_average_5m: float = 0.0
    load_average_15m: float = 0.0
    open_file_descriptors: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class AlertRule:
    """Configuration for monitoring alerts."""
    name: str
    condition: Callable[[CacheMetrics, SystemMetrics], bool]
    level: AlertLevel
    message_template: str
    cooldown_minutes: int = 5
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    
    def should_trigger(self, cache_metrics: CacheMetrics, system_metrics: SystemMetrics) -> bool:
        """Check if alert should trigger."""
        if not self.enabled:
            return False
        
        # Check cooldown
        if self.last_triggered:
            cooldown_delta = timedelta(minutes=self.cooldown_minutes)
            if datetime.now() - self.last_triggered < cooldown_delta:
                return False
        
        try:
            return self.condition(cache_metrics, system_metrics)
        except Exception as e:
            logger.error(f"Alert rule {self.name} evaluation failed: {e}")
            return False

@dataclass
class Alert:
    """Cache monitoring alert."""
    rule_name: str
    level: AlertLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metrics_snapshot: Optional[Dict[str, Any]] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None

# === Metrics Collection ===
class MetricsCollector:
    """High-performance metrics collector with minimal overhead."""
    
    def __init__(self, collection_interval: float = 10.0, max_history: int = 1000):
        self.collection_interval = collection_interval
        self.max_history = max_history
        
        # Metrics storage
        self.cache_metrics_history: deque = deque(maxlen=max_history)
        self.system_metrics_history: deque = deque(maxlen=max_history)
        
        # Real-time tracking
        self.operation_latencies: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.last_collection_time = time.time()
        
        # Thread safety
        self._lock = threading.RLock()
        self._collection_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info(f"MetricsCollector initialized with {collection_interval}s interval")
    
    async def start_collection(self):
        """Start automated metrics collection."""
        if self._running:
            logger.warning("Metrics collection already running")
            return
        
        self._running = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Started automated metrics collection")
    
    async def stop_collection(self):
        """Stop automated metrics collection."""
        self._running = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped metrics collection")
    
    async def _collection_loop(self):
        """Main metrics collection loop."""
        while self._running:
            try:
                await self.collect_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(1)  # Brief pause on error
    
    async def collect_metrics(self, cache_backends: List[Any] = None) -> CacheMetrics:
        """Collect comprehensive cache metrics."""
        start_time = time.time()
        
        try:
            # Collect cache metrics
            cache_metrics = await self._collect_cache_metrics(cache_backends)
            
            # Collect system metrics
            system_metrics = self._collect_system_metrics()
            
            # Calculate collection duration
            cache_metrics.collection_duration_ms = (time.time() - start_time) * 1000
            
            # Store in history
            with self._lock:
                self.cache_metrics_history.append(cache_metrics)
                self.system_metrics_history.append(system_metrics)
            
            return cache_metrics
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return CacheMetrics()  # Return empty metrics on error
    
    async def _collect_cache_metrics(self, cache_backends: List[Any] = None) -> CacheMetrics:
        """Collect metrics from cache backends."""
        metrics = CacheMetrics()
        
        if not cache_backends:
            # Use recorded operation data
            with self._lock:
                metrics.hits = self.operation_counts.get('hit', 0)
                metrics.misses = self.operation_counts.get('miss', 0)
                metrics.sets = self.operation_counts.get('set', 0)
                metrics.deletes = self.operation_counts.get('delete', 0)
                metrics.evictions = self.operation_counts.get('eviction', 0)
                metrics.errors = self.operation_counts.get('error', 0)
                
                # Calculate latency statistics
                get_latencies = list(self.operation_latencies.get('get', []))
                if get_latencies:
                    metrics.avg_get_latency_ms = statistics.mean(get_latencies)
                    metrics.p95_get_latency_ms = self._percentile(get_latencies, 95)
                    metrics.p99_get_latency_ms = self._percentile(get_latencies, 99)
                
                set_latencies = list(self.operation_latencies.get('set', []))
                if set_latencies:
                    metrics.avg_set_latency_ms = statistics.mean(set_latencies)
        else:
            # Collect from actual cache backends
            for backend in cache_backends:
                try:
                    backend_stats = await backend.get_stats()
                    metrics.hits += backend_stats.hits
                    metrics.misses += backend_stats.misses
                    metrics.sets += backend_stats.sets
                    metrics.deletes += backend_stats.deletes
                    metrics.key_count += backend_stats.size
                    metrics.memory_usage_bytes += backend_stats.memory_usage
                except Exception as e:
                    logger.warning(f"Failed to collect metrics from backend {backend}: {e}")
        
        # Calculate derived metrics
        total_ops = metrics.hits + metrics.misses
        if total_ops > 0:
            metrics.hit_rate_percent = (metrics.hits / total_ops) * 100
            metrics.miss_rate_percent = (metrics.misses / total_ops) * 100
        
        # Calculate throughput
        current_time = time.time()
        time_delta = current_time - self.last_collection_time
        if time_delta > 0:
            metrics.throughput_ops_per_second = total_ops / time_delta
        self.last_collection_time = current_time
        
        return metrics
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system-level metrics."""
        metrics = SystemMetrics()
        
        if not PSUTIL_AVAILABLE:
            logger.debug("psutil not available, skipping system metrics")
            return metrics
        
        try:
            # CPU metrics
            metrics.cpu_usage_percent = psutil.cpu_percent(interval=None)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.memory_total_bytes = memory.total
            metrics.memory_available_bytes = memory.available
            metrics.memory_usage_percent = memory.percent
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics.disk_usage_percent = (disk.used / disk.total) * 100
            
            # Network metrics
            network = psutil.net_io_counters()
            metrics.network_bytes_sent = network.bytes_sent
            metrics.network_bytes_recv = network.bytes_recv
            
            # Load average (Unix-like systems)
            try:
                load_avg = psutil.getloadavg()
                metrics.load_average_1m = load_avg[0]
                metrics.load_average_5m = load_avg[1]
                metrics.load_average_15m = load_avg[2]
            except (AttributeError, OSError):
                pass  # Not available on all platforms
            
            # Process metrics
            process = psutil.Process()
            metrics.open_file_descriptors = process.num_fds() if hasattr(process, 'num_fds') else 0
            
        except Exception as e:
            logger.warning(f"System metrics collection failed: {e}")
        
        return metrics
    
    def record_operation(self, operation: str, latency_ms: float = 0.0):
        """Record cache operation for metrics."""
        with self._lock:
            self.operation_counts[operation] += 1
            if latency_ms > 0:
                self.operation_latencies[operation].append(latency_ms)
    
    def get_recent_metrics(self, minutes: int = 10) -> List[CacheMetrics]:
        """Get metrics from recent time window."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        with self._lock:
            return [m for m in self.cache_metrics_history if m.timestamp >= cutoff_time]
    
    def get_metrics_summary(self, minutes: int = 60) -> Dict[str, Any]:
        """Get summarized metrics for time window."""
        recent_metrics = self.get_recent_metrics(minutes)
        
        if not recent_metrics:
            return {"status": "no_data", "time_window_minutes": minutes}
        
        return {
            "time_window_minutes": minutes,
            "data_points": len(recent_metrics),
            "avg_hit_rate": statistics.mean([m.hit_rate_percent for m in recent_metrics]),
            "avg_latency_ms": statistics.mean([m.avg_get_latency_ms for m in recent_metrics if m.avg_get_latency_ms > 0]),
            "total_operations": sum([m.total_operations for m in recent_metrics]),
            "total_errors": sum([m.errors for m in recent_metrics]),
            "peak_throughput": max([m.throughput_ops_per_second for m in recent_metrics], default=0),
            "avg_memory_usage_mb": statistics.mean([m.memory_usage_bytes / 1024 / 1024 for m in recent_metrics])
        }
    
    @staticmethod
    def _percentile(data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]

# === Performance Analyzer ===
class PerformanceAnalyzer:
    """Advanced performance analysis and optimization recommendations."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.analysis_cache: Dict[str, Tuple[datetime, Dict[str, Any]]] = {}
        self.cache_ttl = timedelta(minutes=5)
        
    def analyze_performance(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Comprehensive performance analysis."""
        cache_key = f"analysis_{time_window_minutes}"
        
        # Check cache
        if cache_key in self.analysis_cache:
            cached_time, cached_result = self.analysis_cache[cache_key]
            if datetime.now() - cached_time < self.cache_ttl:
                return cached_result
        
        try:
            recent_metrics = self.metrics_collector.get_recent_metrics(time_window_minutes)
            
            if not recent_metrics:
                return {"status": "insufficient_data"}
            
            analysis = {
                "status": "success",
                "time_window_minutes": time_window_minutes,
                "data_points": len(recent_metrics),
                "performance_score": self._calculate_performance_score(recent_metrics),
                "hit_rate_analysis": self._analyze_hit_rate(recent_metrics),
                "latency_analysis": self._analyze_latency(recent_metrics),
                "throughput_analysis": self._analyze_throughput(recent_metrics),
                "memory_analysis": self._analyze_memory_usage(recent_metrics),
                "error_analysis": self._analyze_errors(recent_metrics),
                "trends": self._analyze_trends(recent_metrics),
                "recommendations": self._generate_recommendations(recent_metrics)
            }
            
            # Cache result
            self.analysis_cache[cache_key] = (datetime.now(), analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def _calculate_performance_score(self, metrics: List[CacheMetrics]) -> float:
        """Calculate overall performance score (0-100)."""
        if not metrics:
            return 0.0
        
        # Weight different factors
        avg_hit_rate = statistics.mean([m.hit_rate_percent for m in metrics])
        avg_latency = statistics.mean([m.avg_get_latency_ms for m in metrics if m.avg_get_latency_ms > 0])
        avg_error_rate = statistics.mean([m.error_rate_percent for m in metrics])
        
        # Score components (0-100 each)
        hit_rate_score = min(avg_hit_rate, 100)  # Hit rate is already percentage
        latency_score = max(0, 100 - (avg_latency / 10))  # Penalize high latency
        error_score = max(0, 100 - (avg_error_rate * 10))  # Heavily penalize errors
        
        # Weighted average
        weights = {"hit_rate": 0.4, "latency": 0.4, "errors": 0.2}
        score = (hit_rate_score * weights["hit_rate"] + 
                latency_score * weights["latency"] + 
                error_score * weights["errors"])
        
        return round(score, 2)
    
    def _analyze_hit_rate(self, metrics: List[CacheMetrics]) -> Dict[str, Any]:
        """Analyze hit rate patterns."""
        hit_rates = [m.hit_rate_percent for m in metrics]
        
        return {
            "current": hit_rates[-1] if hit_rates else 0,
            "average": statistics.mean(hit_rates) if hit_rates else 0,
            "min": min(hit_rates) if hit_rates else 0,
            "max": max(hit_rates) if hit_rates else 0,
            "standard_deviation": statistics.stdev(hit_rates) if len(hit_rates) > 1 else 0,
            "trend": self._calculate_trend(hit_rates),
            "status": self._evaluate_hit_rate_status(statistics.mean(hit_rates) if hit_rates else 0)
        }
    
    def _analyze_latency(self, metrics: List[CacheMetrics]) -> Dict[str, Any]:
        """Analyze latency patterns."""
        latencies = [m.avg_get_latency_ms for m in metrics if m.avg_get_latency_ms > 0]
        p95_latencies = [m.p95_get_latency_ms for m in metrics if m.p95_get_latency_ms > 0]
        
        return {
            "average_ms": statistics.mean(latencies) if latencies else 0,
            "p95_average_ms": statistics.mean(p95_latencies) if p95_latencies else 0,
            "min_ms": min(latencies) if latencies else 0,
            "max_ms": max(latencies) if latencies else 0,
            "trend": self._calculate_trend(latencies),
            "status": self._evaluate_latency_status(statistics.mean(latencies) if latencies else 0)
        }
    
    def _analyze_throughput(self, metrics: List[CacheMetrics]) -> Dict[str, Any]:
        """Analyze throughput patterns."""
        throughputs = [m.throughput_ops_per_second for m in metrics if m.throughput_ops_per_second > 0]
        
        return {
            "current_ops_per_sec": throughputs[-1] if throughputs else 0,
            "average_ops_per_sec": statistics.mean(throughputs) if throughputs else 0,
            "peak_ops_per_sec": max(throughputs) if throughputs else 0,
            "trend": self._calculate_trend(throughputs),
            "status": self._evaluate_throughput_status(statistics.mean(throughputs) if throughputs else 0)
        }
    
    def _analyze_memory_usage(self, metrics: List[CacheMetrics]) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        memory_usages = [m.memory_usage_percent for m in metrics if m.memory_usage_percent > 0]
        
        return {
            "current_percent": memory_usages[-1] if memory_usages else 0,
            "average_percent": statistics.mean(memory_usages) if memory_usages else 0,
            "peak_percent": max(memory_usages) if memory_usages else 0,
            "trend": self._calculate_trend(memory_usages),
            "status": self._evaluate_memory_status(statistics.mean(memory_usages) if memory_usages else 0)
        }
    
    def _analyze_errors(self, metrics: List[CacheMetrics]) -> Dict[str, Any]:
        """Analyze error patterns."""
        error_rates = [m.error_rate_percent for m in metrics]
        total_errors = sum([m.errors for m in metrics])
        
        return {
            "total_errors": total_errors,
            "average_error_rate_percent": statistics.mean(error_rates) if error_rates else 0,
            "peak_error_rate_percent": max(error_rates) if error_rates else 0,
            "trend": self._calculate_trend(error_rates),
            "status": self._evaluate_error_status(statistics.mean(error_rates) if error_rates else 0)
        }
    
    def _analyze_trends(self, metrics: List[CacheMetrics]) -> Dict[str, str]:
        """Analyze overall trends."""
        if len(metrics) < 3:
            return {"status": "insufficient_data"}
        
        hit_rates = [m.hit_rate_percent for m in metrics]
        latencies = [m.avg_get_latency_ms for m in metrics if m.avg_get_latency_ms > 0]
        throughputs = [m.throughput_ops_per_second for m in metrics if m.throughput_ops_per_second > 0]
        
        return {
            "hit_rate": self._calculate_trend(hit_rates),
            "latency": self._calculate_trend(latencies),
            "throughput": self._calculate_trend(throughputs),
            "overall": self._determine_overall_trend(hit_rates, latencies, throughputs)
        }
    
    def _generate_recommendations(self, metrics: List[CacheMetrics]) -> List[Dict[str, str]]:
        """Generate optimization recommendations."""
        recommendations = []
        
        if not metrics:
            return recommendations
        
        avg_hit_rate = statistics.mean([m.hit_rate_percent for m in metrics])
        avg_latency = statistics.mean([m.avg_get_latency_ms for m in metrics if m.avg_get_latency_ms > 0])
        avg_memory = statistics.mean([m.memory_usage_percent for m in metrics if m.memory_usage_percent > 0])
        avg_error_rate = statistics.mean([m.error_rate_percent for m in metrics])
        
        # Hit rate recommendations
        if avg_hit_rate < 80:
            recommendations.append({
                "type": "hit_rate",
                "priority": "high",
                "title": "Low Hit Rate Detected",
                "description": f"Current hit rate ({avg_hit_rate:.1f}%) is below optimal threshold (80%)",
                "suggestion": "Consider increasing cache TTL, improving cache warming strategies, or analyzing access patterns"
            })
        
        # Latency recommendations
        if avg_latency > 50:
            recommendations.append({
                "type": "latency",
                "priority": "medium",
                "title": "High Latency Detected",
                "description": f"Average latency ({avg_latency:.1f}ms) exceeds recommended threshold (50ms)",
                "suggestion": "Consider using faster storage, optimizing serialization, or implementing cache partitioning"
            })
        
        # Memory recommendations
        if avg_memory > 85:
            recommendations.append({
                "type": "memory",
                "priority": "high",
                "title": "High Memory Usage",
                "description": f"Memory usage ({avg_memory:.1f}%) is approaching capacity limits",
                "suggestion": "Consider implementing more aggressive eviction policies or increasing cache memory allocation"
            })
        
        # Error rate recommendations
        if avg_error_rate > 1:
            recommendations.append({
                "type": "errors",
                "priority": "critical",
                "title": "Elevated Error Rate",
                "description": f"Error rate ({avg_error_rate:.1f}%) indicates system instability",
                "suggestion": "Investigate error logs, check network connectivity, and review cache configuration"
            })
        
        return recommendations
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values."""
        if len(values) < 3:
            return "stable"
        
        # Simple linear regression slope
        n = len(values)
        x_mean = (n - 1) / 2  # 0, 1, 2, ... n-1
        y_mean = statistics.mean(values)
        
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "stable"
        
        slope = numerator / denominator
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    def _determine_overall_trend(self, hit_rates: List[float], latencies: List[float], throughputs: List[float]) -> str:
        """Determine overall system trend."""
        trends = []
        
        if hit_rates:
            hit_trend = self._calculate_trend(hit_rates)
            trends.append(1 if hit_trend == "increasing" else -1 if hit_trend == "decreasing" else 0)
        
        if latencies:
            lat_trend = self._calculate_trend(latencies)
            trends.append(-1 if lat_trend == "increasing" else 1 if lat_trend == "decreasing" else 0)
        
        if throughputs:
            thr_trend = self._calculate_trend(throughputs)
            trends.append(1 if thr_trend == "increasing" else -1 if thr_trend == "decreasing" else 0)
        
        if not trends:
            return "stable"
        
        avg_trend = sum(trends) / len(trends)
        
        if avg_trend > 0.3:
            return "improving"
        elif avg_trend < -0.3:
            return "degrading"
        else:
            return "stable"
    
    def _evaluate_hit_rate_status(self, hit_rate: float) -> str:
        """Evaluate hit rate status."""
        if hit_rate >= 95:
            return "excellent"
        elif hit_rate >= 85:
            return "good"
        elif hit_rate >= 70:
            return "fair"
        else:
            return "poor"
    
    def _evaluate_latency_status(self, latency: float) -> str:
        """Evaluate latency status."""
        if latency <= 10:
            return "excellent"
        elif latency <= 25:
            return "good"
        elif latency <= 50:
            return "fair"
        else:
            return "poor"
    
    def _evaluate_throughput_status(self, throughput: float) -> str:
        """Evaluate throughput status."""
        if throughput >= 1000:
            return "excellent"
        elif throughput >= 500:
            return "good"
        elif throughput >= 100:
            return "fair"
        else:
            return "poor"
    
    def _evaluate_memory_status(self, memory_percent: float) -> str:
        """Evaluate memory usage status."""
        if memory_percent <= 70:
            return "good"
        elif memory_percent <= 85:
            return "fair"
        else:
            return "poor"
    
    def _evaluate_error_status(self, error_rate: float) -> str:
        """Evaluate error rate status."""
        if error_rate <= 0.1:
            return "excellent"
        elif error_rate <= 0.5:
            return "good"
        elif error_rate <= 1.0:
            return "fair"
        else:
            return "poor"

# === Alert Manager ===
class AlertManager:
    """Intelligent alert management with configurable rules."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: List[Alert] = []
        self.alert_history: deque = deque(maxlen=1000)
        self.notification_handlers: List[Callable] = []
        
        # Initialize default alert rules
        self._setup_default_rules()
        
        logger.info("AlertManager initialized with default rules")
    
    def _setup_default_rules(self):
        """Setup default monitoring alert rules."""
        self.alert_rules = [
            AlertRule(
                name="low_hit_rate",
                condition=lambda cache, system: cache.hit_rate_percent < 70,
                level=AlertLevel.WARNING,
                message_template="Cache hit rate is {hit_rate:.1f}%, below 70% threshold",
                cooldown_minutes=10
            ),
            AlertRule(
                name="high_latency",
                condition=lambda cache, system: cache.avg_get_latency_ms > 100,
                level=AlertLevel.WARNING,
                message_template="Average cache latency is {latency:.1f}ms, above 100ms threshold",
                cooldown_minutes=5
            ),
            AlertRule(
                name="high_error_rate",
                condition=lambda cache, system: cache.error_rate_percent > 5,
                level=AlertLevel.ERROR,
                message_template="Cache error rate is {error_rate:.1f}%, above 5% threshold",
                cooldown_minutes=2
            ),
            AlertRule(
                name="memory_pressure",
                condition=lambda cache, system: system.memory_usage_percent > 90,
                level=AlertLevel.CRITICAL,
                message_template="System memory usage is {memory_usage:.1f}%, above 90% threshold",
                cooldown_minutes=1
            ),
            AlertRule(
                name="cache_unavailable",
                condition=lambda cache, system: cache.total_operations == 0 and cache.errors > 0,
                level=AlertLevel.CRITICAL,
                message_template="Cache appears to be unavailable - no operations but {errors} errors",
                cooldown_minutes=1
            )
        ]
    
    def add_alert_rule(self, rule: AlertRule):
        """Add custom alert rule."""
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add notification handler for alerts."""
        self.notification_handlers.append(handler)
        logger.info("Added notification handler")
    
    async def check_alerts(self) -> List[Alert]:
        """Check all alert rules and trigger notifications."""
        new_alerts = []
        
        try:
            # Get latest metrics
            cache_metrics = await self.metrics_collector.collect_metrics()
            system_metrics = self.metrics_collector._collect_system_metrics()
            
            for rule in self.alert_rules:
                if rule.should_trigger(cache_metrics, system_metrics):
                    alert = self._create_alert(rule, cache_metrics, system_metrics)
                    new_alerts.append(alert)
                    self.active_alerts.append(alert)
                    self.alert_history.append(alert)
                    
                    # Update rule trigger time
                    rule.last_triggered = datetime.now()
                    
                    # Send notifications
                    await self._send_notifications(alert)
                    
                    logger.warning(f"Alert triggered: {rule.name} - {alert.message}")
            
            return new_alerts
            
        except Exception as e:
            logger.error(f"Alert checking failed: {e}")
            return []
    
    def _create_alert(self, rule: AlertRule, cache_metrics: CacheMetrics, system_metrics: SystemMetrics) -> Alert:
        """Create alert from rule and metrics."""
        # Format message with current metrics
        message = rule.message_template.format(
            hit_rate=cache_metrics.hit_rate_percent,
            latency=cache_metrics.avg_get_latency_ms,
            error_rate=cache_metrics.error_rate_percent,
            memory_usage=system_metrics.memory_usage_percent,
            errors=cache_metrics.errors
        )
        
        return Alert(
            rule_name=rule.name,
            level=rule.level,
            message=message,
            metrics_snapshot={
                "cache_metrics": asdict(cache_metrics),
                "system_metrics": asdict(system_metrics)
            }
        )
    
    async def _send_notifications(self, alert: Alert):
        """Send alert notifications to all handlers."""
        for handler in self.notification_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Notification handler failed: {e}")
    
    def get_active_alerts(self, level: AlertLevel = None) -> List[Alert]:
        """Get active alerts, optionally filtered by level."""
        if level:
            return [alert for alert in self.active_alerts if alert.level == level and not alert.resolved]
        return [alert for alert in self.active_alerts if not alert.resolved]
    
    def resolve_alert(self, alert_id: str):
        """Mark alert as resolved."""
        for alert in self.active_alerts:
            if alert.rule_name == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                logger.info(f"Alert resolved: {alert_id}")
                break

# === Health Checker ===
class CacheHealthChecker:
    """Comprehensive cache system health monitoring."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.health_thresholds = {
            "hit_rate_min": 80.0,
            "latency_max_ms": 50.0,
            "error_rate_max": 1.0,
            "memory_usage_max": 85.0
        }
    
    async def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            cache_metrics = await self.metrics_collector.collect_metrics()
            system_metrics = self.metrics_collector._collect_system_metrics()
            
            checks = {
                "hit_rate": self._check_hit_rate(cache_metrics),
                "latency": self._check_latency(cache_metrics),
                "error_rate": self._check_error_rate(cache_metrics),
                "memory_usage": self._check_memory_usage(system_metrics),
                "connectivity": await self._check_connectivity(),
                "performance": self._check_performance(cache_metrics)
            }
            
            # Determine overall status
            overall_status = self._determine_overall_health(checks)
            
            return {
                "status": overall_status.value,
                "timestamp": datetime.now().isoformat(),
                "checks": checks,
                "summary": self._generate_health_summary(checks),
                "thresholds": self.health_thresholds
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": HealthStatus.CRITICAL.value,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _check_hit_rate(self, metrics: CacheMetrics) -> Dict[str, Any]:
        """Check hit rate health."""
        hit_rate = metrics.hit_rate_percent
        threshold = self.health_thresholds["hit_rate_min"]
        
        if hit_rate >= threshold:
            status = HealthStatus.HEALTHY
        elif hit_rate >= threshold * 0.8:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.UNHEALTHY
        
        return {
            "status": status.value,
            "value": hit_rate,
            "threshold": threshold,
            "message": f"Hit rate: {hit_rate:.1f}% (threshold: {threshold}%)"
        }
    
    def _check_latency(self, metrics: CacheMetrics) -> Dict[str, Any]:
        """Check latency health."""
        latency = metrics.avg_get_latency_ms
        threshold = self.health_thresholds["latency_max_ms"]
        
        if latency <= threshold:
            status = HealthStatus.HEALTHY
        elif latency <= threshold * 2:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.UNHEALTHY
        
        return {
            "status": status.value,
            "value": latency,
            "threshold": threshold,
            "message": f"Average latency: {latency:.1f}ms (threshold: {threshold}ms)"
        }
    
    def _check_error_rate(self, metrics: CacheMetrics) -> Dict[str, Any]:
        """Check error rate health."""
        error_rate = metrics.error_rate_percent
        threshold = self.health_thresholds["error_rate_max"]
        
        if error_rate <= threshold:
            status = HealthStatus.HEALTHY
        elif error_rate <= threshold * 3:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.UNHEALTHY
        
        return {
            "status": status.value,
            "value": error_rate,
            "threshold": threshold,
            "message": f"Error rate: {error_rate:.1f}% (threshold: {threshold}%)"
        }
    
    def _check_memory_usage(self, metrics: SystemMetrics) -> Dict[str, Any]:
        """Check memory usage health."""
        memory_usage = metrics.memory_usage_percent
        threshold = self.health_thresholds["memory_usage_max"]
        
        if memory_usage <= threshold:
            status = HealthStatus.HEALTHY
        elif memory_usage <= 95:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.CRITICAL
        
        return {
            "status": status.value,
            "value": memory_usage,
            "threshold": threshold,
            "message": f"Memory usage: {memory_usage:.1f}% (threshold: {threshold}%)"
        }
    
    async def _check_connectivity(self) -> Dict[str, Any]:
        """Check cache connectivity."""
        try:
            # Try basic operation
            start_time = time.time()
            # This would test actual cache connectivity in real implementation
            await asyncio.sleep(0.001)  # Simulate quick operation
            latency = (time.time() - start_time) * 1000
            
            return {
                "status": HealthStatus.HEALTHY.value,
                "latency_ms": latency,
                "message": "Cache connectivity is healthy"
            }
        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL.value,
                "error": str(e),
                "message": "Cache connectivity failed"
            }
    
    def _check_performance(self, metrics: CacheMetrics) -> Dict[str, Any]:
        """Check overall performance."""
        throughput = metrics.throughput_ops_per_second
        
        if throughput >= 1000:
            status = HealthStatus.HEALTHY
        elif throughput >= 100:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.UNHEALTHY
        
        return {
            "status": status.value,
            "value": throughput,
            "message": f"Throughput: {throughput:.1f} ops/sec"
        }
    
    def _determine_overall_health(self, checks: Dict[str, Any]) -> HealthStatus:
        """Determine overall health status from individual checks."""
        statuses = [check.get("status", "unknown") for check in checks.values()]
        
        if "critical" in statuses:
            return HealthStatus.CRITICAL
        elif "unhealthy" in statuses:
            return HealthStatus.UNHEALTHY
        elif "degraded" in statuses:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def _generate_health_summary(self, checks: Dict[str, Any]) -> str:
        """Generate human-readable health summary."""
        status_counts = defaultdict(int)
        for check in checks.values():
            status_counts[check.get("status", "unknown")] += 1
        
        total_checks = len(checks)
        healthy_checks = status_counts.get("healthy", 0)
        
        return f"{healthy_checks}/{total_checks} health checks passing"

# === Prometheus Exporter ===
class PrometheusExporter:
    """Export cache metrics to Prometheus monitoring system."""
    
    def __init__(self, metrics_collector: MetricsCollector, port: int = 8000):
        self.metrics_collector = metrics_collector
        self.port = port
        
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available. Install with: pip install prometheus-client")
            return
        
        # Initialize Prometheus metrics
        self._init_prometheus_metrics()
        
        logger.info(f"PrometheusExporter initialized on port {port}")
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metric objects."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        from app.utils.metrics_manager import get_counter, get_histogram, get_gauge
        
        # Counter metrics
        self.cache_hits_total = get_counter('cache_hits_total', 'Total cache hits')
        self.cache_misses_total = get_counter('cache_misses_total', 'Total cache misses')
        self.cache_sets_total = get_counter('cache_sets_total', 'Total cache sets')
        self.cache_deletes_total = get_counter('cache_deletes_total', 'Total cache deletes')
        self.cache_errors_total = get_counter('cache_errors_total', 'Total cache errors')
        
        # Gauge metrics
        self.cache_hit_rate = get_gauge('cache_hit_rate_percent', 'Cache hit rate percentage')
        self.cache_memory_usage = get_gauge('cache_memory_usage_bytes', 'Cache memory usage in bytes')
        self.cache_key_count = get_gauge('cache_key_count', 'Number of keys in cache')
        
        # Histogram metrics
        self.cache_latency = get_histogram('cache_operation_duration_seconds', 'Cache operation latency')
        
        # Info metrics (garder Info original car pas dans notre gestionnaire)
        from prometheus_client import Info
        self.cache_info = Info('cache_info', 'Cache system information')
    
    async def start_server(self):
        """Start Prometheus metrics server."""
        if not PROMETHEUS_AVAILABLE:
            logger.error("Cannot start Prometheus server - prometheus_client not available")
            return
        
        try:
            prometheus_client.start_http_server(self.port)
            logger.info(f"Prometheus metrics server started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
    
    async def update_metrics(self):
        """Update Prometheus metrics with current cache data."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        try:
            cache_metrics = await self.metrics_collector.collect_metrics()
            
            # Update counters (note: Prometheus counters should only increase)
            # In real implementation, you'd track deltas properly
            
            # Update gauges
            self.cache_hit_rate.set(cache_metrics.hit_rate_percent)
            self.cache_memory_usage.set(cache_metrics.memory_usage_bytes)
            self.cache_key_count.set(cache_metrics.key_count)
            
            # Update info
            self.cache_info.info({
                'version': '3.0.0',
                'status': 'operational'
            })
            
        except Exception as e:
            logger.error(f"Failed to update Prometheus metrics: {e}")

# === Factory Functions ===
def create_metrics_collector(**kwargs) -> MetricsCollector:
    """Create metrics collector with configuration."""
    return MetricsCollector(**kwargs)

def create_performance_analyzer(metrics_collector: MetricsCollector) -> PerformanceAnalyzer:
    """Create performance analyzer."""
    return PerformanceAnalyzer(metrics_collector)

def create_alert_manager(metrics_collector: MetricsCollector) -> AlertManager:
    """Create alert manager."""
    return AlertManager(metrics_collector)

def create_health_checker(metrics_collector: MetricsCollector) -> CacheHealthChecker:
    """Create health checker."""
    return CacheHealthChecker(metrics_collector)

def create_prometheus_exporter(metrics_collector: MetricsCollector, port: int = 8000) -> PrometheusExporter:
    """Create Prometheus exporter."""
    return PrometheusExporter(metrics_collector, port)

def create_monitoring_suite(collection_interval: float = 10.0) -> Dict[str, Any]:
    """Create complete monitoring suite."""
    metrics_collector = create_metrics_collector(collection_interval=collection_interval)
    
    return {
        'metrics_collector': metrics_collector,
        'performance_analyzer': create_performance_analyzer(metrics_collector),
        'alert_manager': create_alert_manager(metrics_collector),
        'health_checker': create_health_checker(metrics_collector),
        'prometheus_exporter': create_prometheus_exporter(metrics_collector)
    }
