"""
Advanced Performance Metrics and Health Monitoring System.

This module provides comprehensive performance monitoring, health checks,
and system diagnostics for the Spotify AI Agent monitoring algorithms.

Features:
- Real-time performance metrics collection
- Algorithm health status monitoring
- Resource utilization tracking
- Bottleneck detection and analysis
- Performance optimization recommendations
- Circuit breaker pattern implementation
"""

import asyncio
import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import statistics
from collections import deque, defaultdict
import json

from prometheus_client import Counter, Histogram, Gauge, Summary
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"

class MetricType(Enum):
    """Performance metric types."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    ACCURACY = "accuracy"

@dataclass
class PerformanceSnapshot:
    """Performance metrics snapshot."""
    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    disk_io_mb_per_sec: float
    network_io_mb_per_sec: float
    active_connections: int
    algorithm_latencies: Dict[str, float]
    error_rates: Dict[str, float]
    throughput_rps: Dict[str, float]

@dataclass
class HealthMetrics:
    """Comprehensive health metrics."""
    status: HealthStatus
    score: float  # 0-100
    details: Dict[str, Any]
    recommendations: List[str]
    last_check: datetime
    trend_direction: str  # "improving", "stable", "declining"

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    success_threshold: int = 3
    timeout_duration: float = 30.0

class CircuitBreaker:
    """Circuit breaker for algorithm resilience."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        self._lock = asyncio.Lock()
        
        # Metrics
        self.state_gauge = Gauge(
            f'circuit_breaker_state_{name}',
            f'Circuit breaker state for {name}',
            ['algorithm']
        )
        
        self.failure_counter = Counter(
            f'circuit_breaker_failures_{name}',
            f'Circuit breaker failures for {name}',
            ['algorithm']
        )
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        
        async with self._lock:
            # Check if circuit is open
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info(f"Circuit breaker {self.name} moved to HALF_OPEN")
                else:
                    raise Exception(f"Circuit breaker {self.name} is OPEN")
            
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.timeout_duration
                )
                
                await self._on_success()
                return result
                
            except Exception as e:
                await self._on_failure()
                raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt to reset."""
        if self.last_failure_time is None:
            return True
        
        return (
            datetime.now() - self.last_failure_time
        ).total_seconds() > self.config.recovery_timeout
    
    async def _on_success(self):
        """Handle successful execution."""
        self.last_success_time = datetime.now()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(f"Circuit breaker {self.name} moved to CLOSED")
        
        self._update_metrics()
    
    async def _on_failure(self):
        """Handle failed execution."""
        self.last_failure_time = datetime.now()
        self.failure_count += 1
        self.success_count = 0
        
        self.failure_counter.labels(algorithm=self.name).inc()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker {self.name} moved to OPEN")
        
        self._update_metrics()
    
    def _update_metrics(self):
        """Update Prometheus metrics."""
        state_value = {
            CircuitBreakerState.CLOSED: 0,
            CircuitBreakerState.HALF_OPEN: 1,
            CircuitBreakerState.OPEN: 2
        }[self.state]
        
        self.state_gauge.labels(algorithm=self.name).set(state_value)

class PerformanceMonitor:
    """Advanced performance monitoring system."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self.snapshots: deque = deque(maxlen=window_size)
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.monitoring_active = False
        self.monitor_task = None
        
        # Performance metrics
        self.latency_histogram = Histogram(
            'algorithm_performance_latency_seconds',
            'Algorithm execution latency',
            ['algorithm_name', 'model_name'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
        )
        
        self.throughput_gauge = Gauge(
            'algorithm_performance_throughput_rps',
            'Algorithm throughput requests per second',
            ['algorithm_name', 'model_name']
        )
        
        self.error_rate_gauge = Gauge(
            'algorithm_performance_error_rate',
            'Algorithm error rate percentage',
            ['algorithm_name', 'model_name']
        )
        
        self.resource_usage_gauge = Gauge(
            'algorithm_performance_resource_usage',
            'Algorithm resource usage percentage',
            ['algorithm_name', 'model_name', 'resource_type']
        )
    
    def add_circuit_breaker(self, algorithm_name: str, 
                           config: Optional[CircuitBreakerConfig] = None):
        """Add circuit breaker for algorithm."""
        if config is None:
            config = CircuitBreakerConfig()
        
        self.circuit_breakers[algorithm_name] = CircuitBreaker(
            algorithm_name, config
        )
    
    async def execute_with_monitoring(self, algorithm_name: str, 
                                     func: Callable, *args, **kwargs):
        """Execute function with comprehensive monitoring."""
        
        start_time = time.time()
        success = False
        
        try:
            # Use circuit breaker if available
            if algorithm_name in self.circuit_breakers:
                result = await self.circuit_breakers[algorithm_name].call(
                    func, *args, **kwargs
                )
            else:
                result = await func(*args, **kwargs)
            
            success = True
            return result
            
        except Exception as e:
            logger.error(f"Algorithm {algorithm_name} execution failed: {e}")
            raise
            
        finally:
            # Record metrics
            execution_time = time.time() - start_time
            self._record_execution_metrics(
                algorithm_name, execution_time, success
            )
    
    def _record_execution_metrics(self, algorithm_name: str, 
                                 execution_time: float, success: bool):
        """Record execution metrics."""
        
        # Update latency
        self.latency_histogram.labels(
            algorithm_name=algorithm_name,
            model_name='default'
        ).observe(execution_time)
        
        # Update error rate
        current_time = time.time()
        error_key = f"{algorithm_name}_errors"
        success_key = f"{algorithm_name}_success"
        
        if success:
            self.metrics_history[success_key].append(current_time)
        else:
            self.metrics_history[error_key].append(current_time)
        
        # Calculate error rate (last minute)
        recent_errors = sum(
            1 for t in self.metrics_history[error_key]
            if current_time - t <= 60
        )
        recent_successes = sum(
            1 for t in self.metrics_history[success_key]
            if current_time - t <= 60
        )
        
        total_requests = recent_errors + recent_successes
        error_rate = (recent_errors / total_requests * 100) if total_requests > 0 else 0
        
        self.error_rate_gauge.labels(
            algorithm_name=algorithm_name,
            model_name='default'
        ).set(error_rate)
        
        # Calculate throughput (requests per second)
        throughput = total_requests / 60 if total_requests > 0 else 0
        self.throughput_gauge.labels(
            algorithm_name=algorithm_name,
            model_name='default'
        ).set(throughput)
    
    async def start_monitoring(self, interval: float = 10.0):
        """Start background performance monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_task = asyncio.create_task(
            self._monitoring_loop(interval)
        )
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self, interval: float):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                snapshot = self._take_performance_snapshot()
                self.snapshots.append(snapshot)
                
                # Update resource usage metrics
                for algorithm_name in self.circuit_breakers.keys():
                    self.resource_usage_gauge.labels(
                        algorithm_name=algorithm_name,
                        model_name='default',
                        resource_type='cpu'
                    ).set(snapshot.cpu_percent)
                    
                    self.resource_usage_gauge.labels(
                        algorithm_name=algorithm_name,
                        model_name='default',
                        resource_type='memory'
                    ).set(snapshot.memory_mb)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval)
    
    def _take_performance_snapshot(self) -> PerformanceSnapshot:
        """Take current performance snapshot."""
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_mb = memory.used / 1024 / 1024
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_io_mb_per_sec = 0
        if disk_io and len(self.snapshots) > 0:
            last_snapshot = self.snapshots[-1]
            time_diff = (datetime.now() - last_snapshot.timestamp).total_seconds()
            if time_diff > 0:
                disk_diff = (disk_io.read_bytes + disk_io.write_bytes) - (
                    last_snapshot.disk_io_mb_per_sec * time_diff * 1024 * 1024
                )
                disk_io_mb_per_sec = disk_diff / (1024 * 1024) / time_diff
        
        # Network I/O
        network_io = psutil.net_io_counters()
        network_io_mb_per_sec = 0
        if network_io and len(self.snapshots) > 0:
            last_snapshot = self.snapshots[-1]
            time_diff = (datetime.now() - last_snapshot.timestamp).total_seconds()
            if time_diff > 0:
                network_diff = (network_io.bytes_sent + network_io.bytes_recv) - (
                    last_snapshot.network_io_mb_per_sec * time_diff * 1024 * 1024
                )
                network_io_mb_per_sec = network_diff / (1024 * 1024) / time_diff
        
        # Active connections
        active_connections = len(psutil.net_connections())
        
        return PerformanceSnapshot(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            disk_io_mb_per_sec=disk_io_mb_per_sec,
            network_io_mb_per_sec=network_io_mb_per_sec,
            active_connections=active_connections,
            algorithm_latencies={},  # Will be filled by algorithm executions
            error_rates={},
            throughput_rps={}
        )
    
    def get_health_status(self) -> HealthMetrics:
        """Calculate overall health status."""
        
        if len(self.snapshots) == 0:
            return HealthMetrics(
                status=HealthStatus.UNHEALTHY,
                score=0.0,
                details={'error': 'No performance data available'},
                recommendations=['Start performance monitoring'],
                last_check=datetime.now(),
                trend_direction="unknown"
            )
        
        latest_snapshot = self.snapshots[-1]
        
        # Calculate health score
        health_score = 100.0
        details = {}
        recommendations = []
        
        # CPU health (weight: 25%)
        cpu_score = max(0, 100 - latest_snapshot.cpu_percent)
        health_score = health_score * 0.75 + cpu_score * 0.25
        details['cpu_usage_percent'] = latest_snapshot.cpu_percent
        
        if latest_snapshot.cpu_percent > 80:
            recommendations.append("High CPU usage detected - consider scaling")
        
        # Memory health (weight: 25%)
        memory_usage_percent = (latest_snapshot.memory_mb / 
                              (psutil.virtual_memory().total / 1024 / 1024)) * 100
        memory_score = max(0, 100 - memory_usage_percent)
        health_score = health_score * 0.75 + memory_score * 0.25
        details['memory_usage_percent'] = memory_usage_percent
        
        if memory_usage_percent > 80:
            recommendations.append("High memory usage detected - check for leaks")
        
        # Error rates (weight: 30%)
        avg_error_rate = 0
        if len(self.snapshots) >= 5:
            error_rates = []
            for cb in self.circuit_breakers.values():
                if cb.failure_count > 0:
                    error_rates.append(cb.failure_count)
            
            if error_rates:
                avg_error_rate = statistics.mean(error_rates)
        
        error_score = max(0, 100 - avg_error_rate * 10)
        health_score = health_score * 0.7 + error_score * 0.3
        details['average_error_rate'] = avg_error_rate
        
        if avg_error_rate > 5:
            recommendations.append("High error rate detected - investigate algorithms")
        
        # Circuit breaker status (weight: 20%)
        open_breakers = sum(
            1 for cb in self.circuit_breakers.values()
            if cb.state == CircuitBreakerState.OPEN
        )
        
        if open_breakers > 0:
            health_score *= 0.5  # Significant penalty for open breakers
            recommendations.append(f"{open_breakers} circuit breakers are open")
        
        details['open_circuit_breakers'] = open_breakers
        details['total_circuit_breakers'] = len(self.circuit_breakers)
        
        # Determine status
        if health_score >= 90:
            status = HealthStatus.HEALTHY
        elif health_score >= 70:
            status = HealthStatus.DEGRADED
        elif health_score >= 40:
            status = HealthStatus.UNHEALTHY
        else:
            status = HealthStatus.CRITICAL
        
        # Calculate trend
        trend_direction = "stable"
        if len(self.snapshots) >= 10:
            recent_scores = []
            for i in range(-10, 0):
                snapshot = self.snapshots[i]
                # Simplified health calculation for trend
                snapshot_score = 100 - snapshot.cpu_percent * 0.5 - memory_usage_percent * 0.3
                recent_scores.append(snapshot_score)
            
            if len(recent_scores) >= 5:
                early_avg = statistics.mean(recent_scores[:5])
                late_avg = statistics.mean(recent_scores[-5:])
                
                if late_avg > early_avg + 5:
                    trend_direction = "improving"
                elif late_avg < early_avg - 5:
                    trend_direction = "declining"
        
        return HealthMetrics(
            status=status,
            score=health_score,
            details=details,
            recommendations=recommendations,
            last_check=datetime.now(),
            trend_direction=trend_direction
        )
    
    def get_performance_summary(self, window_minutes: int = 15) -> Dict[str, Any]:
        """Get performance summary for the specified window."""
        
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_snapshots = [
            s for s in self.snapshots
            if s.timestamp >= cutoff_time
        ]
        
        if not recent_snapshots:
            return {'error': 'No recent performance data'}
        
        # Calculate statistics
        cpu_values = [s.cpu_percent for s in recent_snapshots]
        memory_values = [s.memory_mb for s in recent_snapshots]
        
        return {
            'window_minutes': window_minutes,
            'sample_count': len(recent_snapshots),
            'cpu_usage': {
                'min': min(cpu_values),
                'max': max(cpu_values),
                'avg': statistics.mean(cpu_values),
                'median': statistics.median(cpu_values)
            },
            'memory_usage_mb': {
                'min': min(memory_values),
                'max': max(memory_values),
                'avg': statistics.mean(memory_values),
                'median': statistics.median(memory_values)
            },
            'circuit_breakers': {
                name: {
                    'state': cb.state.value,
                    'failure_count': cb.failure_count,
                    'last_failure_time': cb.last_failure_time.isoformat() if cb.last_failure_time else None
                }
                for name, cb in self.circuit_breakers.items()
            },
            'health_status': self.get_health_status().__dict__
        }
    
    async def export_metrics_to_redis(self, redis_client: aioredis.Redis,
                                     key_prefix: str = "perf_metrics"):
        """Export performance metrics to Redis."""
        
        try:
            health_status = self.get_health_status()
            performance_summary = self.get_performance_summary()
            
            metrics_data = {
                'health_status': health_status.__dict__,
                'performance_summary': performance_summary,
                'timestamp': datetime.now().isoformat(),
                'circuit_breakers_count': len(self.circuit_breakers)
            }
            
            # Convert datetime objects to strings
            def serialize_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return obj
            
            serialized_data = json.dumps(metrics_data, default=serialize_datetime)
            
            await redis_client.setex(
                f"{key_prefix}:latest",
                300,  # 5 minutes TTL
                serialized_data
            )
            
            # Also store historical data
            await redis_client.lpush(
                f"{key_prefix}:history",
                serialized_data
            )
            
            # Keep only last 100 entries
            await redis_client.ltrim(f"{key_prefix}:history", 0, 99)
            
        except Exception as e:
            logger.error(f"Failed to export metrics to Redis: {e}")

# Global performance monitor instance
PERFORMANCE_MONITOR = PerformanceMonitor()

__all__ = [
    'PerformanceMonitor',
    'CircuitBreaker',
    'CircuitBreakerConfig',
    'HealthStatus',
    'HealthMetrics',
    'PerformanceSnapshot',
    'PERFORMANCE_MONITOR'
]
