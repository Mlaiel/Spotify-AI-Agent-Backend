"""
Enterprise Monitoring Integration for Spotify AI Agent Alert Algorithms

This module provides comprehensive monitoring capabilities integrating with Prometheus,
custom metrics collection, performance tracking, and specialized monitoring for
music streaming platform operations.

Author: Fahed Mlaiel (Expert Backend Developer & ML Engineer)
Version: 2.0.0 (Enterprise Edition)
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable

import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry
import logging
import psutil
import threading

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics for monitoring"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels for monitoring"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class MonitoringConfig:
    """Configuration for monitoring system"""
    prometheus_enabled: bool = True
    prometheus_port: int = 8000
    metrics_prefix: str = "spotify_ai_agent"
    collection_interval: int = 60  # seconds
    custom_labels: Dict[str, str] = field(default_factory=lambda: {
        'service': 'spotify-ai-agent',
        'component': 'alert-algorithms',
        'environment': 'production'
    })
    music_streaming_metrics: Dict[str, Any] = field(default_factory=lambda: {
        'track_user_engagement': True,
        'track_audio_quality': True,
        'track_revenue_impact': True,
        'track_cdn_performance': True,
        'track_search_performance': True,
    })


@dataclass
class MetricData:
    """Data structure for metric information"""
    name: str
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    description: str = ""
    metric_type: MetricType = MetricType.GAUGE


class MetricsCollector(ABC):
    """Abstract base class for metrics collectors"""
    
    @abstractmethod
    def collect_metrics(self) -> List[MetricData]:
        """Collect metrics and return as list of MetricData"""
        pass
    
    @abstractmethod
    def get_collector_name(self) -> str:
        """Get name of the collector"""
        pass


class PrometheusMetricsManager:
    """Manager for Prometheus metrics integration"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.registry = CollectorRegistry()
        self.metrics: Dict[str, Any] = {}
        self._collectors: List[MetricsCollector] = []
        self._is_running = False
        self._collection_task: Optional[asyncio.Task] = None
        
        # Initialize core metrics
        self._initialize_core_metrics()
        self._initialize_music_streaming_metrics()
    
    def _initialize_core_metrics(self):
        """Initialize core algorithm performance metrics"""
        self.metrics.update({
            # Algorithm performance metrics
            'algorithm_execution_time': Histogram(
                f'{self.config.metrics_prefix}_algorithm_execution_seconds',
                'Algorithm execution time in seconds',
                ['algorithm_name', 'operation'],
                registry=self.registry
            ),
            'algorithm_success_rate': Gauge(
                f'{self.config.metrics_prefix}_algorithm_success_rate',
                'Algorithm success rate (0-1)',
                ['algorithm_name'],
                registry=self.registry
            ),
            'alerts_processed_total': Counter(
                f'{self.config.metrics_prefix}_alerts_processed_total',
                'Total number of alerts processed',
                ['severity', 'source', 'status'],
                registry=self.registry
            ),
            'anomalies_detected_total': Counter(
                f'{self.config.metrics_prefix}_anomalies_detected_total',
                'Total number of anomalies detected',
                ['model_name', 'severity', 'confidence_level'],
                registry=self.registry
            ),
            'model_accuracy': Gauge(
                f'{self.config.metrics_prefix}_model_accuracy',
                'ML model accuracy score',
                ['model_name', 'dataset'],
                registry=self.registry
            ),
            'prediction_latency': Histogram(
                f'{self.config.metrics_prefix}_prediction_latency_seconds',
                'ML prediction latency in seconds',
                ['model_name'],
                registry=self.registry
            ),
            'cache_hit_rate': Gauge(
                f'{self.config.metrics_prefix}_cache_hit_rate',
                'Cache hit rate (0-1)',
                ['cache_type'],
                registry=self.registry
            ),
            'memory_usage_bytes': Gauge(
                f'{self.config.metrics_prefix}_memory_usage_bytes',
                'Memory usage in bytes',
                ['component'],
                registry=self.registry
            ),
            'cpu_usage_percent': Gauge(
                f'{self.config.metrics_prefix}_cpu_usage_percent',
                'CPU usage percentage',
                ['component'],
                registry=self.registry
            ),
        })
    
    def _initialize_music_streaming_metrics(self):
        """Initialize music streaming specific metrics"""
        if not self.config.music_streaming_metrics.get('track_user_engagement', False):
            return
        
        streaming_metrics = {
            # User engagement metrics
            'user_session_duration': Histogram(
                f'{self.config.metrics_prefix}_user_session_duration_seconds',
                'User session duration in seconds',
                ['user_segment', 'region'],
                registry=self.registry
            ),
            'track_skip_rate': Gauge(
                f'{self.config.metrics_prefix}_track_skip_rate',
                'Track skip rate (0-1)',
                ['genre', 'user_segment'],
                registry=self.registry
            ),
            'playlist_completion_rate': Gauge(
                f'{self.config.metrics_prefix}_playlist_completion_rate',
                'Playlist completion rate (0-1)',
                ['playlist_type', 'user_segment'],
                registry=self.registry
            ),
            
            # Audio quality metrics
            'audio_bitrate_kbps': Gauge(
                f'{self.config.metrics_prefix}_audio_bitrate_kbps',
                'Audio bitrate in kbps',
                ['quality_tier', 'region'],
                registry=self.registry
            ),
            'buffering_events_total': Counter(
                f'{self.config.metrics_prefix}_buffering_events_total',
                'Total buffering events',
                ['severity', 'cause', 'region'],
                registry=self.registry
            ),
            'audio_latency_ms': Histogram(
                f'{self.config.metrics_prefix}_audio_latency_milliseconds',
                'Audio latency in milliseconds',
                ['region', 'cdn_node'],
                registry=self.registry
            ),
            
            # Revenue impact metrics
            'revenue_per_user': Gauge(
                f'{self.config.metrics_prefix}_revenue_per_user_usd',
                'Revenue per user in USD',
                ['user_segment', 'region', 'subscription_type'],
                registry=self.registry
            ),
            'churn_risk_score': Gauge(
                f'{self.config.metrics_prefix}_churn_risk_score',
                'User churn risk score (0-1)',
                ['user_segment', 'prediction_horizon'],
                registry=self.registry
            ),
            'premium_conversion_rate': Gauge(
                f'{self.config.metrics_prefix}_premium_conversion_rate',
                'Free to premium conversion rate (0-1)',
                ['region', 'acquisition_channel'],
                registry=self.registry
            ),
            
            # Search and discovery metrics
            'search_query_latency': Histogram(
                f'{self.config.metrics_prefix}_search_query_latency_milliseconds',
                'Search query latency in milliseconds',
                ['query_type', 'result_count'],
                registry=self.registry
            ),
            'recommendation_accuracy': Gauge(
                f'{self.config.metrics_prefix}_recommendation_accuracy',
                'Recommendation system accuracy (0-1)',
                ['algorithm_type', 'user_segment'],
                registry=self.registry
            ),
            'discovery_engagement_rate': Gauge(
                f'{self.config.metrics_prefix}_discovery_engagement_rate',
                'Discovery feature engagement rate (0-1)',
                ['feature_type', 'user_segment'],
                registry=self.registry
            ),
        }
        
        self.metrics.update(streaming_metrics)
    
    def register_collector(self, collector: MetricsCollector):
        """Register a custom metrics collector"""
        self._collectors.append(collector)
        logger.info(f"Registered metrics collector: {collector.get_collector_name()}")
    
    def record_algorithm_execution(self, algorithm_name: str, operation: str, execution_time: float):
        """Record algorithm execution time"""
        self.metrics['algorithm_execution_time'].labels(
            algorithm_name=algorithm_name, 
            operation=operation
        ).observe(execution_time)
    
    def record_alert_processed(self, severity: str, source: str, status: str):
        """Record processed alert"""
        self.metrics['alerts_processed_total'].labels(
            severity=severity,
            source=source,
            status=status
        ).inc()
    
    def record_anomaly_detected(self, model_name: str, severity: str, confidence: float):
        """Record detected anomaly"""
        confidence_level = "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
        self.metrics['anomalies_detected_total'].labels(
            model_name=model_name,
            severity=severity,
            confidence_level=confidence_level
        ).inc()
    
    def update_model_accuracy(self, model_name: str, dataset: str, accuracy: float):
        """Update model accuracy metric"""
        self.metrics['model_accuracy'].labels(
            model_name=model_name,
            dataset=dataset
        ).set(accuracy)
    
    def record_prediction_latency(self, model_name: str, latency: float):
        """Record prediction latency"""
        self.metrics['prediction_latency'].labels(
            model_name=model_name
        ).observe(latency)
    
    def update_cache_hit_rate(self, cache_type: str, hit_rate: float):
        """Update cache hit rate"""
        self.metrics['cache_hit_rate'].labels(
            cache_type=cache_type
        ).set(hit_rate)
    
    # Music streaming specific metric recording methods
    
    def record_user_session(self, duration: float, user_segment: str, region: str):
        """Record user session metrics"""
        if 'user_session_duration' in self.metrics:
            self.metrics['user_session_duration'].labels(
                user_segment=user_segment,
                region=region
            ).observe(duration)
    
    def update_track_skip_rate(self, genre: str, user_segment: str, skip_rate: float):
        """Update track skip rate"""
        if 'track_skip_rate' in self.metrics:
            self.metrics['track_skip_rate'].labels(
                genre=genre,
                user_segment=user_segment
            ).set(skip_rate)
    
    def record_buffering_event(self, severity: str, cause: str, region: str):
        """Record buffering event"""
        if 'buffering_events_total' in self.metrics:
            self.metrics['buffering_events_total'].labels(
                severity=severity,
                cause=cause,
                region=region
            ).inc()
    
    def update_audio_quality(self, bitrate: float, latency: float, region: str, quality_tier: str, cdn_node: str):
        """Update audio quality metrics"""
        if 'audio_bitrate_kbps' in self.metrics:
            self.metrics['audio_bitrate_kbps'].labels(
                quality_tier=quality_tier,
                region=region
            ).set(bitrate)
        
        if 'audio_latency_ms' in self.metrics:
            self.metrics['audio_latency_ms'].labels(
                region=region,
                cdn_node=cdn_node
            ).observe(latency)
    
    def update_revenue_metrics(self, revenue_per_user: float, user_segment: str, region: str, subscription_type: str):
        """Update revenue metrics"""
        if 'revenue_per_user' in self.metrics:
            self.metrics['revenue_per_user'].labels(
                user_segment=user_segment,
                region=region,
                subscription_type=subscription_type
            ).set(revenue_per_user)
    
    def update_churn_risk(self, user_segment: str, prediction_horizon: str, risk_score: float):
        """Update churn risk score"""
        if 'churn_risk_score' in self.metrics:
            self.metrics['churn_risk_score'].labels(
                user_segment=user_segment,
                prediction_horizon=prediction_horizon
            ).set(risk_score)
    
    def record_search_query(self, query_type: str, result_count: str, latency: float):
        """Record search query metrics"""
        if 'search_query_latency' in self.metrics:
            self.metrics['search_query_latency'].labels(
                query_type=query_type,
                result_count=result_count
            ).observe(latency)
    
    def update_recommendation_accuracy(self, algorithm_type: str, user_segment: str, accuracy: float):
        """Update recommendation accuracy"""
        if 'recommendation_accuracy' in self.metrics:
            self.metrics['recommendation_accuracy'].labels(
                algorithm_type=algorithm_type,
                user_segment=user_segment
            ).set(accuracy)
    
    async def start_collection(self):
        """Start automatic metrics collection"""
        if self._is_running:
            return
        
        self._is_running = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Started metrics collection")
    
    async def stop_collection(self):
        """Stop automatic metrics collection"""
        self._is_running = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped metrics collection")
    
    async def _collection_loop(self):
        """Main collection loop"""
        while self._is_running:
            try:
                await self._collect_system_metrics()
                await self._collect_custom_metrics()
                await asyncio.sleep(self.config.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(5)  # Short delay before retrying
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # Memory usage
            memory_info = psutil.virtual_memory()
            self.metrics['memory_usage_bytes'].labels(
                component='system'
            ).set(memory_info.used)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent()
            self.metrics['cpu_usage_percent'].labels(
                component='system'
            ).set(cpu_percent)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _collect_custom_metrics(self):
        """Collect metrics from registered collectors"""
        for collector in self._collectors:
            try:
                metrics = collector.collect_metrics()
                for metric in metrics:
                    await self._process_custom_metric(metric)
            except Exception as e:
                logger.error(f"Error collecting metrics from {collector.get_collector_name()}: {e}")
    
    async def _process_custom_metric(self, metric: MetricData):
        """Process a custom metric"""
        try:
            # Create metric if it doesn't exist
            metric_key = f"custom_{metric.name}"
            if metric_key not in self.metrics:
                self._create_dynamic_metric(metric_key, metric)
            
            # Update metric value
            prometheus_metric = self.metrics[metric_key]
            if hasattr(prometheus_metric, 'labels'):
                prometheus_metric.labels(**metric.labels).set(metric.value)
            else:
                prometheus_metric.set(metric.value)
                
        except Exception as e:
            logger.error(f"Error processing custom metric {metric.name}: {e}")
    
    def _create_dynamic_metric(self, metric_key: str, metric: MetricData):
        """Create a dynamic metric based on MetricData"""
        try:
            if metric.metric_type == MetricType.GAUGE:
                self.metrics[metric_key] = Gauge(
                    f'{self.config.metrics_prefix}_{metric.name}',
                    metric.description or f'Custom metric: {metric.name}',
                    list(metric.labels.keys()),
                    registry=self.registry
                )
            elif metric.metric_type == MetricType.COUNTER:
                self.metrics[metric_key] = Counter(
                    f'{self.config.metrics_prefix}_{metric.name}',
                    metric.description or f'Custom counter: {metric.name}',
                    list(metric.labels.keys()),
                    registry=self.registry
                )
            elif metric.metric_type == MetricType.HISTOGRAM:
                self.metrics[metric_key] = Histogram(
                    f'{self.config.metrics_prefix}_{metric.name}',
                    metric.description or f'Custom histogram: {metric.name}',
                    list(metric.labels.keys()),
                    registry=self.registry
                )
            elif metric.metric_type == MetricType.SUMMARY:
                self.metrics[metric_key] = Summary(
                    f'{self.config.metrics_prefix}_{metric.name}',
                    metric.description or f'Custom summary: {metric.name}',
                    list(metric.labels.keys()),
                    registry=self.registry
                )
        except Exception as e:
            logger.error(f"Error creating dynamic metric {metric_key}: {e}")
    
    def get_metrics_output(self) -> str:
        """Get Prometheus-formatted metrics output"""
        return prometheus_client.generate_latest(self.registry)
    
    def start_http_server(self):
        """Start Prometheus HTTP server"""
        if self.config.prometheus_enabled:
            prometheus_client.start_http_server(
                self.config.prometheus_port,
                registry=self.registry
            )
            logger.info(f"Prometheus HTTP server started on port {self.config.prometheus_port}")


class AlgorithmPerformanceCollector(MetricsCollector):
    """Collector for algorithm-specific performance metrics"""
    
    def __init__(self):
        self.algorithm_stats: Dict[str, Dict[str, Any]] = {}
    
    def record_algorithm_performance(self, algorithm_name: str, execution_time: float, success: bool, input_size: int):
        """Record algorithm performance data"""
        if algorithm_name not in self.algorithm_stats:
            self.algorithm_stats[algorithm_name] = {
                'total_executions': 0,
                'successful_executions': 0,
                'total_time': 0.0,
                'avg_input_size': 0,
                'last_execution': datetime.now()
            }
        
        stats = self.algorithm_stats[algorithm_name]
        stats['total_executions'] += 1
        stats['total_time'] += execution_time
        stats['avg_input_size'] = (stats['avg_input_size'] + input_size) / 2
        stats['last_execution'] = datetime.now()
        
        if success:
            stats['successful_executions'] += 1
    
    def collect_metrics(self) -> List[MetricData]:
        """Collect algorithm performance metrics"""
        metrics = []
        
        for algorithm_name, stats in self.algorithm_stats.items():
            # Success rate
            success_rate = stats['successful_executions'] / stats['total_executions'] if stats['total_executions'] > 0 else 0
            metrics.append(MetricData(
                name='algorithm_success_rate',
                value=success_rate,
                labels={'algorithm_name': algorithm_name},
                description='Algorithm success rate'
            ))
            
            # Average execution time
            avg_time = stats['total_time'] / stats['total_executions'] if stats['total_executions'] > 0 else 0
            metrics.append(MetricData(
                name='algorithm_avg_execution_time',
                value=avg_time,
                labels={'algorithm_name': algorithm_name},
                description='Average algorithm execution time'
            ))
            
            # Total executions
            metrics.append(MetricData(
                name='algorithm_total_executions',
                value=stats['total_executions'],
                labels={'algorithm_name': algorithm_name},
                description='Total algorithm executions',
                metric_type=MetricType.COUNTER
            ))
        
        return metrics
    
    def get_collector_name(self) -> str:
        return "AlgorithmPerformanceCollector"


class MusicStreamingMetricsCollector(MetricsCollector):
    """Collector for music streaming specific metrics"""
    
    def __init__(self):
        self.streaming_stats: Dict[str, Any] = {
            'total_sessions': 0,
            'total_tracks_played': 0,
            'total_skips': 0,
            'total_revenue': 0.0,
            'active_users': set(),
            'quality_issues': 0,
        }
    
    def record_user_activity(self, user_id: str, tracks_played: int, skips: int, session_revenue: float):
        """Record user activity"""
        self.streaming_stats['total_sessions'] += 1
        self.streaming_stats['total_tracks_played'] += tracks_played
        self.streaming_stats['total_skips'] += skips
        self.streaming_stats['total_revenue'] += session_revenue
        self.streaming_stats['active_users'].add(user_id)
    
    def record_quality_issue(self):
        """Record audio quality issue"""
        self.streaming_stats['quality_issues'] += 1
    
    def collect_metrics(self) -> List[MetricData]:
        """Collect music streaming metrics"""
        metrics = []
        
        # Skip rate
        skip_rate = self.streaming_stats['total_skips'] / self.streaming_stats['total_tracks_played'] if self.streaming_stats['total_tracks_played'] > 0 else 0
        metrics.append(MetricData(
            name='global_skip_rate',
            value=skip_rate,
            description='Global track skip rate'
        ))
        
        # Active users
        metrics.append(MetricData(
            name='active_users_count',
            value=len(self.streaming_stats['active_users']),
            description='Number of active users'
        ))
        
        # Revenue per session
        revenue_per_session = self.streaming_stats['total_revenue'] / self.streaming_stats['total_sessions'] if self.streaming_stats['total_sessions'] > 0 else 0
        metrics.append(MetricData(
            name='revenue_per_session',
            value=revenue_per_session,
            description='Average revenue per session'
        ))
        
        # Quality issues rate
        quality_issue_rate = self.streaming_stats['quality_issues'] / self.streaming_stats['total_sessions'] if self.streaming_stats['total_sessions'] > 0 else 0
        metrics.append(MetricData(
            name='quality_issue_rate',
            value=quality_issue_rate,
            description='Rate of quality issues per session'
        ))
        
        return metrics
    
    def get_collector_name(self) -> str:
        return "MusicStreamingMetricsCollector"


# Global monitoring manager instance
_monitoring_manager: Optional[PrometheusMetricsManager] = None


def get_monitoring_manager(config: Optional[MonitoringConfig] = None) -> PrometheusMetricsManager:
    """Get or create global monitoring manager instance"""
    global _monitoring_manager
    
    if _monitoring_manager is None:
        if config is None:
            config = MonitoringConfig()
        _monitoring_manager = PrometheusMetricsManager(config)
    
    return _monitoring_manager


async def initialize_monitoring(config: Optional[MonitoringConfig] = None) -> PrometheusMetricsManager:
    """Initialize monitoring system"""
    manager = get_monitoring_manager(config)
    
    # Register default collectors
    manager.register_collector(AlgorithmPerformanceCollector())
    manager.register_collector(MusicStreamingMetricsCollector())
    
    # Start collection and HTTP server
    await manager.start_collection()
    manager.start_http_server()
    
    logger.info("Monitoring system initialized successfully")
    return manager


async def shutdown_monitoring():
    """Shutdown monitoring system"""
    global _monitoring_manager
    
    if _monitoring_manager is not None:
        await _monitoring_manager.stop_collection()
        _monitoring_manager = None
        logger.info("Monitoring system shut down")
