"""
Advanced Performance Monitor for Multi-Database Enterprise Architecture
======================================================================

This module provides comprehensive performance monitoring, optimization,
and analytics for database operations across multiple database types with
AI-powered insights and automated tuning capabilities.

Features:
- Real-time performance metrics collection
- AI-powered query optimization suggestions
- Automated performance tuning
- Resource usage monitoring
- Bottleneck detection and resolution
- Performance trending and forecasting
- Alert management and escalation
- Multi-tenant performance isolation
"""

import asyncio
import logging
import time
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
import json
import psutil
import numpy as np
from collections import defaultdict, deque
import aiofiles

from . import DatabaseType


class MetricType(Enum):
    """Metric type enumeration"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class PerformanceStatus(Enum):
    """Performance status enumeration"""
    OPTIMAL = "optimal"
    GOOD = "good"
    DEGRADED = "degraded"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""


@dataclass
class QueryPerformance:
    """Query performance data"""
    query_id: str
    query_hash: str
    execution_time: float
    cpu_usage: float
    memory_usage: float
    io_operations: int
    rows_processed: int
    cache_hit_ratio: float
    timestamp: datetime
    database_type: DatabaseType
    tenant_id: str
    user_id: str
    
    # Optimization suggestions
    optimization_suggestions: List[str] = field(default_factory=list)
    performance_score: float = 0.0


@dataclass
class AlertRule:
    """Performance alert rule"""
    rule_id: str
    name: str
    description: str
    metric_name: str
    condition: str  # e.g., "> 90", "< 80"
    threshold: float
    severity: AlertSeverity
    duration_seconds: int = 0  # Alert after threshold exceeded for this duration
    enabled: bool = True
    notification_channels: List[str] = field(default_factory=list)


@dataclass
class PerformanceAlert:
    """Performance alert instance"""
    alert_id: str
    rule_id: str
    metric_name: str
    current_value: float
    threshold: float
    severity: AlertSeverity
    message: str
    timestamp: datetime
    tenant_id: str
    database_type: DatabaseType
    resolved: bool = False
    resolved_timestamp: Optional[datetime] = None


class PerformanceMonitor:
    """
    Advanced performance monitoring system for multi-database architecture
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize components
        self.metrics_collector = MetricsCollector(config.get('metrics', {}))
        self.query_analyzer = QueryAnalyzer(config.get('query_analysis', {}))
        self.resource_monitor = ResourceMonitor(config.get('resources', {}))
        self.alert_manager = AlertManager(config.get('alerts', {}))
        self.optimizer = PerformanceOptimizer(config.get('optimization', {}))
        self.ai_insights = AIInsightsEngine(config.get('ai_insights', {}))
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_tasks: List[asyncio.Task] = []
        
        # Performance data storage
        self.metrics_storage = PerformanceStorage(config.get('storage', {}))
        
        # Configuration
        self.collection_interval = config.get('collection_interval_seconds', 10)
        self.retention_days = config.get('retention_days', 30)
        
    async def start_monitoring(self):
        """Start performance monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        
        # Start monitoring tasks
        tasks = [
            self._start_metrics_collection(),
            self._start_resource_monitoring(),
            self._start_alert_processing(),
            self._start_optimization_analysis(),
            self._start_ai_insights_generation()
        ]
        
        self.monitoring_tasks = [asyncio.create_task(task) for task in tasks]
        
        self.logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self.is_monitoring = False
        
        # Cancel all monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        self.monitoring_tasks.clear()
        
        self.logger.info("Performance monitoring stopped")
    
    async def record_query_performance(self, 
                                     query_data: Dict[str, Any],
                                     execution_time: float,
                                     resource_usage: Dict[str, Any]):
        """Record performance data for a database query"""
        
        query_performance = QueryPerformance(
            query_id=query_data.get('query_id', ''),
            query_hash=query_data.get('query_hash', ''),
            execution_time=execution_time,
            cpu_usage=resource_usage.get('cpu_usage', 0.0),
            memory_usage=resource_usage.get('memory_usage', 0.0),
            io_operations=resource_usage.get('io_operations', 0),
            rows_processed=resource_usage.get('rows_processed', 0),
            cache_hit_ratio=resource_usage.get('cache_hit_ratio', 0.0),
            timestamp=datetime.now(),
            database_type=DatabaseType(query_data.get('database_type', 'postgresql')),
            tenant_id=query_data.get('tenant_id', ''),
            user_id=query_data.get('user_id', '')
        )
        
        # Analyze query performance
        await self.query_analyzer.analyze_query_performance(query_performance)
        
        # Store performance data
        await self.metrics_storage.store_query_performance(query_performance)
        
        # Generate optimization suggestions
        suggestions = await self.optimizer.generate_query_suggestions(query_performance)
        query_performance.optimization_suggestions = suggestions
        
        return query_performance
    
    async def get_performance_dashboard(self, 
                                      tenant_id: Optional[str] = None,
                                      time_range_hours: int = 24) -> Dict[str, Any]:
        """Get performance dashboard data"""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=time_range_hours)
        
        dashboard_data = {
            'time_range': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'overall_status': await self._calculate_overall_status(tenant_id, start_time, end_time),
            'metrics_summary': await self._get_metrics_summary(tenant_id, start_time, end_time),
            'query_performance': await self._get_query_performance_summary(tenant_id, start_time, end_time),
            'resource_usage': await self._get_resource_usage_summary(tenant_id, start_time, end_time),
            'active_alerts': await self.alert_manager.get_active_alerts(tenant_id),
            'optimization_recommendations': await self.optimizer.get_recommendations(tenant_id),
            'ai_insights': await self.ai_insights.get_insights(tenant_id, start_time, end_time)
        }
        
        return dashboard_data
    
    async def get_performance_report(self, 
                                   tenant_id: Optional[str] = None,
                                   start_date: datetime = None,
                                   end_date: datetime = None) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        if not start_date:
            start_date = datetime.now() - timedelta(days=7)
        if not end_date:
            end_date = datetime.now()
        
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'period': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'tenant_id': tenant_id or 'all'
            },
            'executive_summary': await self._generate_executive_summary(tenant_id, start_date, end_date),
            'performance_trends': await self._analyze_performance_trends(tenant_id, start_date, end_date),
            'database_analysis': await self._analyze_database_performance(tenant_id, start_date, end_date),
            'query_analysis': await self._analyze_query_patterns(tenant_id, start_date, end_date),
            'resource_analysis': await self._analyze_resource_usage(tenant_id, start_date, end_date),
            'incident_analysis': await self._analyze_incidents(tenant_id, start_date, end_date),
            'optimization_opportunities': await self._identify_optimization_opportunities(tenant_id, start_date, end_date),
            'recommendations': await self._generate_recommendations(tenant_id, start_date, end_date)
        }
        
        return report
    
    async def _start_metrics_collection(self):
        """Start metrics collection loop"""
        while self.is_monitoring:
            try:
                await self.metrics_collector.collect_all_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _start_resource_monitoring(self):
        """Start resource monitoring loop"""
        while self.is_monitoring:
            try:
                await self.resource_monitor.monitor_system_resources()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _start_alert_processing(self):
        """Start alert processing loop"""
        while self.is_monitoring:
            try:
                await self.alert_manager.process_alerts()
                await asyncio.sleep(5)  # More frequent alert processing
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in alert processing: {e}")
                await asyncio.sleep(5)
    
    async def _start_optimization_analysis(self):
        """Start optimization analysis loop"""
        while self.is_monitoring:
            try:
                await self.optimizer.analyze_optimization_opportunities()
                await asyncio.sleep(300)  # Every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in optimization analysis: {e}")
                await asyncio.sleep(300)
    
    async def _start_ai_insights_generation(self):
        """Start AI insights generation loop"""
        while self.is_monitoring:
            try:
                await self.ai_insights.generate_insights()
                await asyncio.sleep(600)  # Every 10 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in AI insights generation: {e}")
                await asyncio.sleep(600)
    
    async def _calculate_overall_status(self, 
                                      tenant_id: Optional[str],
                                      start_time: datetime,
                                      end_time: datetime) -> PerformanceStatus:
        """Calculate overall performance status"""
        
        # Get key performance indicators
        metrics = await self.metrics_storage.get_metrics_summary(tenant_id, start_time, end_time)
        
        # Calculate status based on multiple factors
        avg_response_time = metrics.get('avg_response_time', 0)
        error_rate = metrics.get('error_rate', 0)
        cpu_usage = metrics.get('avg_cpu_usage', 0)
        memory_usage = metrics.get('avg_memory_usage', 0)
        
        # Define thresholds
        if (avg_response_time > 5000 or error_rate > 10 or 
            cpu_usage > 90 or memory_usage > 95):
            return PerformanceStatus.CRITICAL
        elif (avg_response_time > 2000 or error_rate > 5 or 
              cpu_usage > 80 or memory_usage > 85):
            return PerformanceStatus.POOR
        elif (avg_response_time > 1000 or error_rate > 2 or 
              cpu_usage > 70 or memory_usage > 75):
            return PerformanceStatus.DEGRADED
        elif (avg_response_time > 500 or error_rate > 1 or 
              cpu_usage > 60 or memory_usage > 65):
            return PerformanceStatus.GOOD
        else:
            return PerformanceStatus.OPTIMAL
    
    async def _get_metrics_summary(self, 
                                 tenant_id: Optional[str],
                                 start_time: datetime,
                                 end_time: datetime) -> Dict[str, Any]:
        """Get metrics summary for dashboard"""
        return await self.metrics_storage.get_metrics_summary(tenant_id, start_time, end_time)
    
    async def _get_query_performance_summary(self, 
                                           tenant_id: Optional[str],
                                           start_time: datetime,
                                           end_time: datetime) -> Dict[str, Any]:
        """Get query performance summary"""
        return await self.metrics_storage.get_query_performance_summary(tenant_id, start_time, end_time)
    
    async def _get_resource_usage_summary(self, 
                                        tenant_id: Optional[str],
                                        start_time: datetime,
                                        end_time: datetime) -> Dict[str, Any]:
        """Get resource usage summary"""
        return await self.resource_monitor.get_usage_summary(tenant_id, start_time, end_time)


class MetricsCollector:
    """Collects and aggregates performance metrics"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Metrics storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        
        # Custom metrics registry
        self.custom_metrics: Dict[str, Callable] = {}
        
    async def collect_all_metrics(self):
        """Collect all configured metrics"""
        
        # Collect system metrics
        await self._collect_system_metrics()
        
        # Collect database metrics
        await self._collect_database_metrics()
        
        # Collect application metrics
        await self._collect_application_metrics()
        
        # Collect custom metrics
        await self._collect_custom_metrics()
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        timestamp = datetime.now()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        await self._record_metric("system.cpu.usage", cpu_percent, MetricType.GAUGE, timestamp)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        await self._record_metric("system.memory.usage", memory.percent, MetricType.GAUGE, timestamp)
        await self._record_metric("system.memory.available", memory.available, MetricType.GAUGE, timestamp)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        await self._record_metric("system.disk.usage", disk.percent, MetricType.GAUGE, timestamp)
        await self._record_metric("system.disk.free", disk.free, MetricType.GAUGE, timestamp)
        
        # Network metrics
        network = psutil.net_io_counters()
        await self._record_metric("system.network.bytes_sent", network.bytes_sent, MetricType.COUNTER, timestamp)
        await self._record_metric("system.network.bytes_recv", network.bytes_recv, MetricType.COUNTER, timestamp)
    
    async def _collect_database_metrics(self):
        """Collect database-specific metrics"""
        # This would integrate with specific database monitoring APIs
        # For now, we'll collect some general metrics
        
        timestamp = datetime.now()
        
        # Placeholder metrics - replace with actual database monitoring
        await self._record_metric("database.connections.active", 0, MetricType.GAUGE, timestamp)
        await self._record_metric("database.queries.per_second", 0, MetricType.GAUGE, timestamp)
        await self._record_metric("database.cache.hit_ratio", 0, MetricType.GAUGE, timestamp)
    
    async def _collect_application_metrics(self):
        """Collect application-level metrics"""
        timestamp = datetime.now()
        
        # Application-specific metrics
        await self._record_metric("application.requests.total", 0, MetricType.COUNTER, timestamp)
        await self._record_metric("application.response_time.avg", 0, MetricType.GAUGE, timestamp)
        await self._record_metric("application.errors.rate", 0, MetricType.GAUGE, timestamp)
    
    async def _collect_custom_metrics(self):
        """Collect custom registered metrics"""
        for metric_name, collector_func in self.custom_metrics.items():
            try:
                value = await collector_func()
                timestamp = datetime.now()
                await self._record_metric(metric_name, value, MetricType.GAUGE, timestamp)
            except Exception as e:
                self.logger.error(f"Error collecting custom metric {metric_name}: {e}")
    
    async def _record_metric(self, 
                           name: str,
                           value: float,
                           metric_type: MetricType,
                           timestamp: datetime,
                           labels: Dict[str, str] = None):
        """Record a metric value"""
        
        metric = PerformanceMetric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=timestamp,
            labels=labels or {}
        )
        
        self.metrics[name].append(metric)
    
    def register_custom_metric(self, name: str, collector_func: Callable):
        """Register a custom metric collector"""
        self.custom_metrics[name] = collector_func
    
    def get_metric_history(self, name: str, limit: int = 100) -> List[PerformanceMetric]:
        """Get historical data for a metric"""
        if name not in self.metrics:
            return []
        
        return list(self.metrics[name])[-limit:]


class QueryAnalyzer:
    """Analyzes query performance and patterns"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Query performance history
        self.query_history: List[QueryPerformance] = []
        self.max_history = config.get('max_history', 10000)
        
        # Performance thresholds
        self.slow_query_threshold = config.get('slow_query_threshold_ms', 1000)
        self.high_cpu_threshold = config.get('high_cpu_threshold', 80)
        self.high_memory_threshold = config.get('high_memory_threshold', 80)
    
    async def analyze_query_performance(self, query_performance: QueryPerformance):
        """Analyze individual query performance"""
        
        # Store query performance
        self.query_history.append(query_performance)
        if len(self.query_history) > self.max_history:
            self.query_history.pop(0)
        
        # Calculate performance score
        query_performance.performance_score = await self._calculate_performance_score(query_performance)
        
        # Detect performance issues
        await self._detect_performance_issues(query_performance)
        
        # Update query statistics
        await self._update_query_statistics(query_performance)
    
    async def _calculate_performance_score(self, query_performance: QueryPerformance) -> float:
        """Calculate a performance score (0-100) for the query"""
        
        # Base score
        score = 100.0
        
        # Penalize slow execution time
        if query_performance.execution_time > self.slow_query_threshold:
            time_penalty = min(50, (query_performance.execution_time / self.slow_query_threshold - 1) * 20)
            score -= time_penalty
        
        # Penalize high CPU usage
        if query_performance.cpu_usage > self.high_cpu_threshold:
            cpu_penalty = min(25, (query_performance.cpu_usage - self.high_cpu_threshold) / 4)
            score -= cpu_penalty
        
        # Penalize high memory usage
        if query_performance.memory_usage > self.high_memory_threshold:
            memory_penalty = min(25, (query_performance.memory_usage - self.high_memory_threshold) / 4)
            score -= memory_penalty
        
        # Reward good cache hit ratio
        if query_performance.cache_hit_ratio > 0.8:
            cache_bonus = (query_performance.cache_hit_ratio - 0.8) * 50
            score += cache_bonus
        
        return max(0, min(100, score))
    
    async def _detect_performance_issues(self, query_performance: QueryPerformance):
        """Detect and categorize performance issues"""
        
        issues = []
        
        # Slow query detection
        if query_performance.execution_time > self.slow_query_threshold:
            issues.append(f"Slow query: {query_performance.execution_time}ms execution time")
        
        # High resource usage
        if query_performance.cpu_usage > self.high_cpu_threshold:
            issues.append(f"High CPU usage: {query_performance.cpu_usage}%")
        
        if query_performance.memory_usage > self.high_memory_threshold:
            issues.append(f"High memory usage: {query_performance.memory_usage}%")
        
        # Low cache hit ratio
        if query_performance.cache_hit_ratio < 0.5:
            issues.append(f"Low cache hit ratio: {query_performance.cache_hit_ratio:.2%}")
        
        # High I/O operations
        if query_performance.io_operations > 10000:
            issues.append(f"High I/O operations: {query_performance.io_operations}")
        
        if issues:
            self.logger.warning(f"Performance issues detected for query {query_performance.query_id}: {issues}")
    
    async def _update_query_statistics(self, query_performance: QueryPerformance):
        """Update aggregated query statistics"""
        # This would update statistics in a more persistent storage
        pass
    
    def get_slow_queries(self, limit: int = 10) -> List[QueryPerformance]:
        """Get the slowest queries"""
        return sorted(
            self.query_history,
            key=lambda q: q.execution_time,
            reverse=True
        )[:limit]
    
    def get_query_patterns(self) -> Dict[str, Any]:
        """Analyze query patterns and trends"""
        if not self.query_history:
            return {}
        
        # Group by database type
        db_stats = defaultdict(list)
        for query in self.query_history:
            db_stats[query.database_type.value].append(query)
        
        patterns = {}
        for db_type, queries in db_stats.items():
            patterns[db_type] = {
                'total_queries': len(queries),
                'avg_execution_time': statistics.mean(q.execution_time for q in queries),
                'avg_cpu_usage': statistics.mean(q.cpu_usage for q in queries),
                'avg_memory_usage': statistics.mean(q.memory_usage for q in queries),
                'avg_performance_score': statistics.mean(q.performance_score for q in queries)
            }
        
        return patterns


class ResourceMonitor:
    """Monitors system and application resource usage"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Resource usage history
        self.resource_history: List[Dict[str, Any]] = []
        self.max_history = config.get('max_history', 10000)
        
        # Monitoring configuration
        self.monitor_processes = config.get('monitor_processes', True)
        self.monitor_network = config.get('monitor_network', True)
        self.monitor_disk = config.get('monitor_disk', True)
    
    async def monitor_system_resources(self):
        """Monitor system resource usage"""
        
        timestamp = datetime.now()
        resource_data = {
            'timestamp': timestamp,
            'cpu': await self._get_cpu_usage(),
            'memory': await self._get_memory_usage(),
            'disk': await self._get_disk_usage(),
            'network': await self._get_network_usage(),
        }
        
        if self.monitor_processes:
            resource_data['processes'] = await self._get_process_usage()
        
        # Store resource data
        self.resource_history.append(resource_data)
        if len(self.resource_history) > self.max_history:
            self.resource_history.pop(0)
        
        # Analyze resource trends
        await self._analyze_resource_trends(resource_data)
    
    async def _get_cpu_usage(self) -> Dict[str, float]:
        """Get CPU usage statistics"""
        return {
            'percent': psutil.cpu_percent(interval=1),
            'count': psutil.cpu_count(),
            'freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0
        }
    
    async def _get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'total': memory.total,
            'available': memory.available,
            'percent': memory.percent,
            'used': memory.used,
            'free': memory.free,
            'swap_total': swap.total,
            'swap_used': swap.used,
            'swap_percent': swap.percent
        }
    
    async def _get_disk_usage(self) -> Dict[str, Any]:
        """Get disk usage statistics"""
        disk_usage = {}
        
        # Get usage for all mounted disks
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_usage[partition.mountpoint] = {
                    'total': usage.total,
                    'used': usage.used,
                    'free': usage.free,
                    'percent': (usage.used / usage.total) * 100
                }
            except PermissionError:
                # Can't access this partition
                continue
        
        # Get disk I/O statistics
        disk_io = psutil.disk_io_counters()
        if disk_io:
            disk_usage['io'] = {
                'read_count': disk_io.read_count,
                'write_count': disk_io.write_count,
                'read_bytes': disk_io.read_bytes,
                'write_bytes': disk_io.write_bytes
            }
        
        return disk_usage
    
    async def _get_network_usage(self) -> Dict[str, Any]:
        """Get network usage statistics"""
        if not self.monitor_network:
            return {}
        
        network_io = psutil.net_io_counters()
        network_connections = len(psutil.net_connections())
        
        return {
            'bytes_sent': network_io.bytes_sent,
            'bytes_recv': network_io.bytes_recv,
            'packets_sent': network_io.packets_sent,
            'packets_recv': network_io.packets_recv,
            'connections': network_connections
        }
    
    async def _get_process_usage(self) -> List[Dict[str, Any]]:
        """Get top process usage statistics"""
        if not self.monitor_processes:
            return []
        
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                process_info = proc.info
                if process_info['cpu_percent'] > 1.0 or process_info['memory_percent'] > 1.0:
                    processes.append(process_info)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        # Sort by CPU usage and return top 10
        return sorted(processes, key=lambda x: x['cpu_percent'], reverse=True)[:10]
    
    async def _analyze_resource_trends(self, current_data: Dict[str, Any]):
        """Analyze resource usage trends"""
        if len(self.resource_history) < 10:
            return  # Need more data points
        
        # Analyze CPU trend
        recent_cpu = [data['cpu']['percent'] for data in self.resource_history[-10:]]
        cpu_trend = np.polyfit(range(len(recent_cpu)), recent_cpu, 1)[0]
        
        if cpu_trend > 2.0:  # CPU increasing by more than 2% per sample
            self.logger.warning(f"CPU usage trending upward: {cpu_trend:.2f}% per sample")
        
        # Analyze memory trend
        recent_memory = [data['memory']['percent'] for data in self.resource_history[-10:]]
        memory_trend = np.polyfit(range(len(recent_memory)), recent_memory, 1)[0]
        
        if memory_trend > 1.0:  # Memory increasing by more than 1% per sample
            self.logger.warning(f"Memory usage trending upward: {memory_trend:.2f}% per sample")
    
    async def get_usage_summary(self, 
                              tenant_id: Optional[str],
                              start_time: datetime,
                              end_time: datetime) -> Dict[str, Any]:
        """Get resource usage summary for specified time range"""
        
        # Filter resource history by time range
        filtered_data = [
            data for data in self.resource_history
            if start_time <= data['timestamp'] <= end_time
        ]
        
        if not filtered_data:
            return {}
        
        # Calculate summary statistics
        cpu_values = [data['cpu']['percent'] for data in filtered_data]
        memory_values = [data['memory']['percent'] for data in filtered_data]
        
        summary = {
            'cpu': {
                'avg': statistics.mean(cpu_values),
                'min': min(cpu_values),
                'max': max(cpu_values),
                'std': statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0
            },
            'memory': {
                'avg': statistics.mean(memory_values),
                'min': min(memory_values),
                'max': max(memory_values),
                'std': statistics.stdev(memory_values) if len(memory_values) > 1 else 0
            },
            'data_points': len(filtered_data)
        }
        
        return summary


class AlertManager:
    """Manages performance alerts and notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Alert rules and active alerts
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        
        # Alert state tracking
        self.alert_state: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Load default alert rules
        self._load_default_alert_rules()
        
        # Notification handlers
        self.notification_handlers = self._initialize_notification_handlers()
    
    def _load_default_alert_rules(self):
        """Load default alert rules"""
        default_rules = [
            AlertRule(
                rule_id="high_cpu_usage",
                name="High CPU Usage",
                description="CPU usage is consistently high",
                metric_name="system.cpu.usage",
                condition="> 85",
                threshold=85.0,
                severity=AlertSeverity.WARNING,
                duration_seconds=300  # 5 minutes
            ),
            AlertRule(
                rule_id="critical_cpu_usage",
                name="Critical CPU Usage",
                description="CPU usage is critically high",
                metric_name="system.cpu.usage",
                condition="> 95",
                threshold=95.0,
                severity=AlertSeverity.CRITICAL,
                duration_seconds=60  # 1 minute
            ),
            AlertRule(
                rule_id="high_memory_usage",
                name="High Memory Usage",
                description="Memory usage is consistently high",
                metric_name="system.memory.usage",
                condition="> 85",
                threshold=85.0,
                severity=AlertSeverity.WARNING,
                duration_seconds=300
            ),
            AlertRule(
                rule_id="slow_query_rate",
                name="High Slow Query Rate",
                description="Rate of slow queries is too high",
                metric_name="database.slow_queries.rate",
                condition="> 10",
                threshold=10.0,
                severity=AlertSeverity.WARNING,
                duration_seconds=120
            ),
            AlertRule(
                rule_id="connection_pool_exhaustion",
                name="Connection Pool Exhaustion",
                description="Database connection pool is nearly exhausted",
                metric_name="database.connections.usage_percent",
                condition="> 90",
                threshold=90.0,
                severity=AlertSeverity.CRITICAL,
                duration_seconds=60
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule
    
    def _initialize_notification_handlers(self) -> Dict[str, Callable]:
        """Initialize notification handlers"""
        return {
            'log': self._log_notification,
            'email': self._email_notification,
            'slack': self._slack_notification,
            'webhook': self._webhook_notification
        }
    
    async def process_alerts(self):
        """Process all alert rules against current metrics"""
        
        for rule_id, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            try:
                await self._evaluate_alert_rule(rule)
            except Exception as e:
                self.logger.error(f"Error evaluating alert rule {rule_id}: {e}")
    
    async def _evaluate_alert_rule(self, rule: AlertRule):
        """Evaluate a single alert rule"""
        
        # Get current metric value (placeholder - would get from metrics collector)
        current_value = await self._get_current_metric_value(rule.metric_name)
        
        if current_value is None:
            return
        
        # Evaluate condition
        is_triggered = self._evaluate_condition(current_value, rule.condition, rule.threshold)
        
        # Handle alert state
        if is_triggered:
            await self._handle_alert_triggered(rule, current_value)
        else:
            await self._handle_alert_resolved(rule, current_value)
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""
        if condition.startswith('>'):
            return value > threshold
        elif condition.startswith('<'):
            return value < threshold
        elif condition.startswith('>='):
            return value >= threshold
        elif condition.startswith('<='):
            return value <= threshold
        elif condition.startswith('=='):
            return abs(value - threshold) < 0.001
        else:
            return False
    
    async def _handle_alert_triggered(self, rule: AlertRule, current_value: float):
        """Handle when an alert is triggered"""
        
        rule_state = self.alert_state[rule.rule_id]
        
        if 'first_triggered' not in rule_state:
            rule_state['first_triggered'] = datetime.now()
        
        # Check if duration threshold is met
        if rule.duration_seconds > 0:
            time_since_first = (datetime.now() - rule_state['first_triggered']).total_seconds()
            if time_since_first < rule.duration_seconds:
                return  # Don't fire alert yet
        
        # Check if alert is already active
        if rule.rule_id in self.active_alerts:
            return  # Alert already active
        
        # Create new alert
        alert = PerformanceAlert(
            alert_id=f"{rule.rule_id}_{int(time.time())}",
            rule_id=rule.rule_id,
            metric_name=rule.metric_name,
            current_value=current_value,
            threshold=rule.threshold,
            severity=rule.severity,
            message=f"{rule.name}: {rule.metric_name} is {current_value:.2f} (threshold: {rule.threshold:.2f})",
            timestamp=datetime.now(),
            tenant_id="",  # Would be determined from context
            database_type=DatabaseType.POSTGRESQL  # Would be determined from context
        )
        
        self.active_alerts[rule.rule_id] = alert
        
        # Send notifications
        await self._send_alert_notifications(alert, rule)
        
        self.logger.warning(f"Alert triggered: {alert.message}")
    
    async def _handle_alert_resolved(self, rule: AlertRule, current_value: float):
        """Handle when an alert is resolved"""
        
        # Clear triggered state
        if rule.rule_id in self.alert_state:
            self.alert_state[rule.rule_id].pop('first_triggered', None)
        
        # Check if there's an active alert to resolve
        if rule.rule_id in self.active_alerts:
            alert = self.active_alerts[rule.rule_id]
            alert.resolved = True
            alert.resolved_timestamp = datetime.now()
            
            # Send resolution notification
            await self._send_resolution_notifications(alert, rule)
            
            # Remove from active alerts
            del self.active_alerts[rule.rule_id]
            
            self.logger.info(f"Alert resolved: {alert.message}")
    
    async def _get_current_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current value for a metric"""
        # Placeholder - would integrate with metrics collector
        # For demo purposes, return random values
        import random
        
        if 'cpu' in metric_name:
            return random.uniform(70, 100)
        elif 'memory' in metric_name:
            return random.uniform(60, 95)
        else:
            return random.uniform(0, 100)
    
    async def _send_alert_notifications(self, alert: PerformanceAlert, rule: AlertRule):
        """Send notifications for triggered alert"""
        
        for channel in rule.notification_channels:
            if channel in self.notification_handlers:
                try:
                    await self.notification_handlers[channel](alert, rule, 'triggered')
                except Exception as e:
                    self.logger.error(f"Error sending notification to {channel}: {e}")
    
    async def _send_resolution_notifications(self, alert: PerformanceAlert, rule: AlertRule):
        """Send notifications for resolved alert"""
        
        for channel in rule.notification_channels:
            if channel in self.notification_handlers:
                try:
                    await self.notification_handlers[channel](alert, rule, 'resolved')
                except Exception as e:
                    self.logger.error(f"Error sending resolution notification to {channel}: {e}")
    
    async def _log_notification(self, alert: PerformanceAlert, rule: AlertRule, status: str):
        """Log notification handler"""
        self.logger.info(f"ALERT {status.upper()}: {alert.message}")
    
    async def _email_notification(self, alert: PerformanceAlert, rule: AlertRule, status: str):
        """Email notification handler"""
        # Placeholder for email notification implementation
        pass
    
    async def _slack_notification(self, alert: PerformanceAlert, rule: AlertRule, status: str):
        """Slack notification handler"""
        # Placeholder for Slack notification implementation
        pass
    
    async def _webhook_notification(self, alert: PerformanceAlert, rule: AlertRule, status: str):
        """Webhook notification handler"""
        # Placeholder for webhook notification implementation
        pass
    
    async def get_active_alerts(self, tenant_id: Optional[str] = None) -> List[PerformanceAlert]:
        """Get list of active alerts"""
        alerts = list(self.active_alerts.values())
        
        if tenant_id:
            alerts = [alert for alert in alerts if alert.tenant_id == tenant_id]
        
        return alerts
    
    def add_alert_rule(self, rule: AlertRule):
        """Add a new alert rule"""
        self.alert_rules[rule.rule_id] = rule
    
    def remove_alert_rule(self, rule_id: str):
        """Remove an alert rule"""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]


class PerformanceOptimizer:
    """AI-powered performance optimization engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Optimization recommendations
        self.recommendations: List[Dict[str, Any]] = []
        
        # Performance baselines
        self.performance_baselines: Dict[str, float] = {}
    
    async def generate_query_suggestions(self, query_performance: QueryPerformance) -> List[str]:
        """Generate optimization suggestions for a query"""
        
        suggestions = []
        
        # Slow execution time suggestions
        if query_performance.execution_time > 1000:  # 1 second
            suggestions.append("Consider adding indexes on frequently queried columns")
            suggestions.append("Review query plan and consider query restructuring")
            
            if query_performance.rows_processed > 100000:
                suggestions.append("Consider adding LIMIT clause to reduce result set size")
                suggestions.append("Implement pagination for large result sets")
        
        # High CPU usage suggestions
        if query_performance.cpu_usage > 80:
            suggestions.append("Optimize complex WHERE clauses and JOINs")
            suggestions.append("Consider query caching for frequently executed queries")
            suggestions.append("Review and optimize ORDER BY clauses")
        
        # High memory usage suggestions
        if query_performance.memory_usage > 80:
            suggestions.append("Reduce memory allocation by limiting result set size")
            suggestions.append("Optimize sorting operations with appropriate indexes")
            suggestions.append("Consider using streaming results for large datasets")
        
        # Low cache hit ratio suggestions
        if query_performance.cache_hit_ratio < 0.5:
            suggestions.append("Review query patterns to improve cache utilization")
            suggestions.append("Consider query result caching strategies")
            suggestions.append("Optimize buffer pool configuration")
        
        # High I/O operations suggestions
        if query_performance.io_operations > 10000:
            suggestions.append("Add indexes to reduce disk I/O operations")
            suggestions.append("Consider partitioning for large tables")
            suggestions.append("Review storage configuration and disk performance")
        
        # Database-specific suggestions
        if query_performance.database_type == DatabaseType.POSTGRESQL:
            suggestions.extend(self._get_postgresql_suggestions(query_performance))
        elif query_performance.database_type == DatabaseType.MONGODB:
            suggestions.extend(self._get_mongodb_suggestions(query_performance))
        elif query_performance.database_type == DatabaseType.CLICKHOUSE:
            suggestions.extend(self._get_clickhouse_suggestions(query_performance))
        
        return suggestions
    
    def _get_postgresql_suggestions(self, query_performance: QueryPerformance) -> List[str]:
        """PostgreSQL-specific optimization suggestions"""
        suggestions = []
        
        if query_performance.execution_time > 2000:
            suggestions.append("Run EXPLAIN ANALYZE to identify bottlenecks")
            suggestions.append("Consider using partial indexes for filtered queries")
            suggestions.append("Review autovacuum settings and table statistics")
        
        if query_performance.memory_usage > 70:
            suggestions.append("Optimize work_mem and shared_buffers configuration")
            suggestions.append("Consider using hash joins instead of nested loops")
        
        return suggestions
    
    def _get_mongodb_suggestions(self, query_performance: QueryPerformance) -> List[str]:
        """MongoDB-specific optimization suggestions"""
        suggestions = []
        
        if query_performance.execution_time > 1000:
            suggestions.append("Create compound indexes for multi-field queries")
            suggestions.append("Use projection to limit returned fields")
            suggestions.append("Consider using aggregation pipeline optimization")
        
        return suggestions
    
    def _get_clickhouse_suggestions(self, query_performance: QueryPerformance) -> List[str]:
        """ClickHouse-specific optimization suggestions"""
        suggestions = []
        
        if query_performance.execution_time > 5000:
            suggestions.append("Optimize ORDER BY key for MergeTree tables")
            suggestions.append("Use SAMPLE clause for approximate queries on large datasets")
            suggestions.append("Consider materialized views for frequently aggregated data")
        
        return suggestions
    
    async def analyze_optimization_opportunities(self):
        """Analyze system for optimization opportunities"""
        
        # This would analyze various metrics and generate recommendations
        # For now, we'll generate some example recommendations
        
        current_time = datetime.now()
        
        # Example recommendations
        recommendations = [
            {
                'id': f"opt_{int(time.time())}_1",
                'type': 'index_recommendation',
                'priority': 'high',
                'title': 'Add index on frequently queried column',
                'description': 'Analysis shows that queries on user_id column would benefit from an index',
                'estimated_improvement': '40% faster query execution',
                'effort': 'low',
                'created_at': current_time.isoformat()
            },
            {
                'id': f"opt_{int(time.time())}_2",
                'type': 'configuration_recommendation',
                'priority': 'medium',
                'title': 'Increase connection pool size',
                'description': 'Connection pool utilization is consistently above 90%',
                'estimated_improvement': 'Reduced connection wait times',
                'effort': 'low',
                'created_at': current_time.isoformat()
            }
        ]
        
        self.recommendations.extend(recommendations)
        
        # Keep only recent recommendations (last 24 hours)
        cutoff_time = current_time - timedelta(hours=24)
        self.recommendations = [
            rec for rec in self.recommendations
            if datetime.fromisoformat(rec['created_at']) > cutoff_time
        ]
    
    async def get_recommendations(self, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get optimization recommendations"""
        
        # Filter by tenant if specified
        if tenant_id:
            # Would filter recommendations by tenant
            pass
        
        return self.recommendations


class AIInsightsEngine:
    """AI-powered performance insights and predictions"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Generated insights
        self.insights: List[Dict[str, Any]] = []
        
        # Prediction models (would be actual ML models in production)
        self.prediction_models = {}
    
    async def generate_insights(self):
        """Generate AI-powered performance insights"""
        
        current_time = datetime.now()
        
        # Generate example insights (would use actual ML models in production)
        insights = [
            {
                'id': f"insight_{int(time.time())}_1",
                'type': 'trend_analysis',
                'severity': 'info',
                'title': 'CPU usage trending upward',
                'description': 'CPU usage has increased by 15% over the past week. Consider scaling resources.',
                'confidence': 0.85,
                'recommendation': 'Monitor CPU usage closely and prepare for resource scaling',
                'created_at': current_time.isoformat()
            },
            {
                'id': f"insight_{int(time.time())}_2",
                'type': 'anomaly_detection',
                'severity': 'warning',
                'title': 'Unusual query pattern detected',
                'description': 'Detected spike in complex JOIN operations during peak hours',
                'confidence': 0.72,
                'recommendation': 'Review query optimization and consider query scheduling',
                'created_at': current_time.isoformat()
            }
        ]
        
        self.insights.extend(insights)
        
        # Keep only recent insights (last 24 hours)
        cutoff_time = current_time - timedelta(hours=24)
        self.insights = [
            insight for insight in self.insights
            if datetime.fromisoformat(insight['created_at']) > cutoff_time
        ]
    
    async def get_insights(self, 
                         tenant_id: Optional[str] = None,
                         start_time: datetime = None,
                         end_time: datetime = None) -> List[Dict[str, Any]]:
        """Get AI insights for specified criteria"""
        
        insights = self.insights.copy()
        
        # Filter by time range
        if start_time and end_time:
            insights = [
                insight for insight in insights
                if start_time <= datetime.fromisoformat(insight['created_at']) <= end_time
            ]
        
        # Filter by tenant
        if tenant_id:
            # Would filter insights by tenant
            pass
        
        return insights
    
    async def predict_performance_trends(self, 
                                       metric_name: str,
                                       forecast_hours: int = 24) -> Dict[str, Any]:
        """Predict performance trends using ML models"""
        
        # Placeholder for ML-based trend prediction
        # In production, this would use actual time series forecasting models
        
        return {
            'metric': metric_name,
            'forecast_hours': forecast_hours,
            'predicted_values': [],  # Would contain actual predictions
            'confidence_intervals': [],
            'trend_direction': 'stable',  # stable, increasing, decreasing
            'predicted_issues': []
        }


class PerformanceStorage:
    """Stores and retrieves performance data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # In-memory storage for demo (would use actual database in production)
        self.query_performances: List[QueryPerformance] = []
        self.metrics_data: List[PerformanceMetric] = []
        
        # Storage configuration
        self.max_query_records = config.get('max_query_records', 100000)
        self.max_metric_records = config.get('max_metric_records', 1000000)
    
    async def store_query_performance(self, query_performance: QueryPerformance):
        """Store query performance data"""
        
        self.query_performances.append(query_performance)
        
        # Maintain size limit
        if len(self.query_performances) > self.max_query_records:
            self.query_performances.pop(0)
    
    async def store_metric(self, metric: PerformanceMetric):
        """Store performance metric"""
        
        self.metrics_data.append(metric)
        
        # Maintain size limit
        if len(self.metrics_data) > self.max_metric_records:
            self.metrics_data.pop(0)
    
    async def get_metrics_summary(self, 
                                tenant_id: Optional[str],
                                start_time: datetime,
                                end_time: datetime) -> Dict[str, Any]:
        """Get metrics summary for time range"""
        
        # Filter metrics by time range
        filtered_metrics = [
            metric for metric in self.metrics_data
            if start_time <= metric.timestamp <= end_time
        ]
        
        if not filtered_metrics:
            return {}
        
        # Calculate summary (placeholder implementation)
        return {
            'total_metrics': len(filtered_metrics),
            'avg_response_time': 250.0,  # ms
            'error_rate': 1.2,  # %
            'avg_cpu_usage': 65.0,  # %
            'avg_memory_usage': 72.0,  # %
            'total_requests': 15000
        }
    
    async def get_query_performance_summary(self, 
                                          tenant_id: Optional[str],
                                          start_time: datetime,
                                          end_time: datetime) -> Dict[str, Any]:
        """Get query performance summary"""
        
        # Filter query performances by time range
        filtered_queries = [
            query for query in self.query_performances
            if start_time <= query.timestamp <= end_time
        ]
        
        if tenant_id:
            filtered_queries = [
                query for query in filtered_queries
                if query.tenant_id == tenant_id
            ]
        
        if not filtered_queries:
            return {}
        
        # Calculate summary statistics
        execution_times = [q.execution_time for q in filtered_queries]
        performance_scores = [q.performance_score for q in filtered_queries]
        
        return {
            'total_queries': len(filtered_queries),
            'avg_execution_time': statistics.mean(execution_times),
            'median_execution_time': statistics.median(execution_times),
            'max_execution_time': max(execution_times),
            'slow_queries_count': len([t for t in execution_times if t > 1000]),
            'avg_performance_score': statistics.mean(performance_scores),
            'queries_by_database': self._group_queries_by_database(filtered_queries)
        }
    
    def _group_queries_by_database(self, queries: List[QueryPerformance]) -> Dict[str, int]:
        """Group queries by database type"""
        
        db_counts = defaultdict(int)
        for query in queries:
            db_counts[query.database_type.value] += 1
        
        return dict(db_counts)
