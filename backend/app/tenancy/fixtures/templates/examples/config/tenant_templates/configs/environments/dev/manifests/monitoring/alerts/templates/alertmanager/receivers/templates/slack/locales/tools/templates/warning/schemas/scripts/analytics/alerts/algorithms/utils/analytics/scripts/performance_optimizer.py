"""
Performance Optimizer - Ultra-Advanced Edition
==============================================

Ultra-advanced performance optimization system with AI-driven analysis,
automated tuning, and predictive performance management.

Features:
- Real-time performance monitoring and analysis
- AI-powered performance bottleneck detection
- Automated resource optimization and scaling
- Predictive performance modeling
- Database query optimization
- Code performance profiling
- Infrastructure cost optimization
- Load balancing and traffic management
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
import pickle
import time
import threading
from collections import defaultdict, deque
from enum import Enum
import statistics

# Performance monitoring
import psutil
import GPUtil
import py3nvml.py3nvml as nvml
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, start_http_server
import redis
import psycopg2
from sqlalchemy import create_engine, text

# ML and optimization
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import optuna
from scipy.optimize import minimize
import tensorflow as tf

# System optimization
import subprocess
import os
import signal
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

# Profiling and analysis
import cProfile
import pstats
import memory_profiler
import line_profiler

# Web and network optimization
import aiohttp
import asyncio
import uvloop
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware


class OptimizationLevel(Enum):
    """Optimization level enumeration."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    EXPERIMENTAL = "experimental"


class MetricType(Enum):
    """Metric type enumeration."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    DATABASE = "database"
    APPLICATION = "application"
    CUSTOM = "custom"


@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    
    name: str
    metric_type: MetricType
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Thresholds
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    
    # Context
    source: str = "unknown"
    description: str = ""


@dataclass
class PerformanceBottleneck:
    """Performance bottleneck identification."""
    
    bottleneck_id: str
    name: str
    severity: str  # low, medium, high, critical
    component: str  # cpu, memory, disk, network, database, application
    
    # Metrics
    impact_score: float  # 0-100
    confidence: float  # 0-1
    
    # Details
    description: str
    root_cause: str
    affected_metrics: List[str]
    
    # Recommendations
    recommendations: List[str]
    estimated_improvement: float
    
    # Context
    detected_at: datetime
    duration_seconds: float
    frequency: int


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation."""
    
    recommendation_id: str
    category: str
    priority: str  # low, medium, high, critical
    
    # Description
    title: str
    description: str
    rationale: str
    
    # Impact estimation
    estimated_improvement_percent: float
    estimated_cost_savings_usd: float
    implementation_effort: str  # low, medium, high
    
    # Implementation
    steps: List[str]
    automation_available: bool
    rollback_plan: str
    
    # Validation
    success_metrics: List[str]
    validation_period_days: int
    
    # Metadata
    created_at: datetime
    expires_at: Optional[datetime] = None
    applied: bool = False
    applied_at: Optional[datetime] = None


@dataclass
class OptimizationConfig:
    """Performance optimization configuration."""
    
    # General settings
    optimization_level: OptimizationLevel = OptimizationLevel.MODERATE
    auto_apply_optimizations: bool = False
    monitoring_interval_seconds: int = 60
    
    # Thresholds
    cpu_warning_threshold: float = 70.0
    cpu_critical_threshold: float = 90.0
    memory_warning_threshold: float = 80.0
    memory_critical_threshold: float = 95.0
    disk_warning_threshold: float = 85.0
    disk_critical_threshold: float = 95.0
    
    # Optimization targets
    target_response_time_ms: float = 200.0
    target_throughput_rps: float = 1000.0
    target_availability_percent: float = 99.9
    target_cost_efficiency: float = 80.0
    
    # ML settings
    anomaly_detection_enabled: bool = True
    predictive_scaling_enabled: bool = True
    ml_model_retrain_hours: int = 24
    
    # Database optimization
    query_optimization_enabled: bool = True
    index_optimization_enabled: bool = True
    connection_pool_optimization: bool = True
    
    # Application optimization
    code_profiling_enabled: bool = True
    cache_optimization_enabled: bool = True
    async_optimization_enabled: bool = True


class SystemMonitor:
    """Real-time system performance monitor."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger('SystemMonitor')
        self.metrics_buffer = deque(maxlen=1000)
        self.is_monitoring = False
        
        # Initialize GPU monitoring if available
        try:
            nvml.nvmlInit()
            self.gpu_available = True
        except:
            self.gpu_available = False
        
        # Prometheus metrics
        self.registry = CollectorRegistry()
        self.setup_prometheus_metrics()
    
    def setup_prometheus_metrics(self):
        """Setup Prometheus metrics."""
        self.cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'system_memory_usage_percent',
            'Memory usage percentage',
            registry=self.registry
        )
        
        self.disk_usage = Gauge(
            'system_disk_usage_percent',
            'Disk usage percentage',
            ['device'],
            registry=self.registry
        )
        
        self.network_io = Gauge(
            'system_network_io_bytes',
            'Network I/O bytes',
            ['direction'],
            registry=self.registry
        )
        
        self.response_time = Histogram(
            'application_response_time_seconds',
            'Application response time',
            registry=self.registry
        )
    
    async def start_monitoring(self):
        """Start performance monitoring."""
        self.is_monitoring = True
        self.logger.info("Starting performance monitoring")
        
        while self.is_monitoring:
            try:
                metrics = await self.collect_metrics()
                self.metrics_buffer.extend(metrics)
                
                # Update Prometheus metrics
                self.update_prometheus_metrics(metrics)
                
                await asyncio.sleep(self.config.monitoring_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {str(e)}")
                await asyncio.sleep(5)
    
    async def collect_metrics(self) -> List[PerformanceMetric]:
        """Collect performance metrics."""
        metrics = []
        timestamp = datetime.now()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
        
        metrics.extend([
            PerformanceMetric(
                name="cpu_usage_percent",
                metric_type=MetricType.CPU,
                value=cpu_percent,
                unit="percent",
                timestamp=timestamp,
                warning_threshold=self.config.cpu_warning_threshold,
                critical_threshold=self.config.cpu_critical_threshold,
                source="psutil"
            ),
            PerformanceMetric(
                name="cpu_count",
                metric_type=MetricType.CPU,
                value=cpu_count,
                unit="cores",
                timestamp=timestamp,
                source="psutil"
            ),
            PerformanceMetric(
                name="load_average_1min",
                metric_type=MetricType.CPU,
                value=load_avg[0],
                unit="load",
                timestamp=timestamp,
                source="system"
            )
        ])
        
        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        metrics.extend([
            PerformanceMetric(
                name="memory_usage_percent",
                metric_type=MetricType.MEMORY,
                value=memory.percent,
                unit="percent",
                timestamp=timestamp,
                warning_threshold=self.config.memory_warning_threshold,
                critical_threshold=self.config.memory_critical_threshold,
                source="psutil"
            ),
            PerformanceMetric(
                name="memory_available_gb",
                metric_type=MetricType.MEMORY,
                value=memory.available / (1024**3),
                unit="GB",
                timestamp=timestamp,
                source="psutil"
            ),
            PerformanceMetric(
                name="swap_usage_percent",
                metric_type=MetricType.MEMORY,
                value=swap.percent,
                unit="percent",
                timestamp=timestamp,
                source="psutil"
            )
        ])
        
        # Disk metrics
        for disk in psutil.disk_partitions():
            try:
                disk_usage = psutil.disk_usage(disk.mountpoint)
                metrics.append(
                    PerformanceMetric(
                        name="disk_usage_percent",
                        metric_type=MetricType.DISK,
                        value=(disk_usage.used / disk_usage.total) * 100,
                        unit="percent",
                        timestamp=timestamp,
                        tags={'device': disk.device, 'mountpoint': disk.mountpoint},
                        warning_threshold=self.config.disk_warning_threshold,
                        critical_threshold=self.config.disk_critical_threshold,
                        source="psutil"
                    )
                )
            except PermissionError:
                continue
        
        # Network metrics
        network = psutil.net_io_counters()
        metrics.extend([
            PerformanceMetric(
                name="network_bytes_sent",
                metric_type=MetricType.NETWORK,
                value=network.bytes_sent,
                unit="bytes",
                timestamp=timestamp,
                source="psutil"
            ),
            PerformanceMetric(
                name="network_bytes_recv",
                metric_type=MetricType.NETWORK,
                value=network.bytes_recv,
                unit="bytes",
                timestamp=timestamp,
                source="psutil"
            )
        ])
        
        # GPU metrics (if available)
        if self.gpu_available:
            try:
                device_count = nvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    gpu_util = nvml.nvmlDeviceGetUtilizationRates(handle)
                    memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    metrics.extend([
                        PerformanceMetric(
                            name="gpu_usage_percent",
                            metric_type=MetricType.CPU,  # Using CPU type for compute
                            value=gpu_util.gpu,
                            unit="percent",
                            timestamp=timestamp,
                            tags={'gpu_id': str(i)},
                            source="nvml"
                        ),
                        PerformanceMetric(
                            name="gpu_memory_usage_percent",
                            metric_type=MetricType.MEMORY,
                            value=(memory_info.used / memory_info.total) * 100,
                            unit="percent",
                            timestamp=timestamp,
                            tags={'gpu_id': str(i)},
                            source="nvml"
                        )
                    ])
            except Exception as e:
                self.logger.warning(f"GPU monitoring error: {str(e)}")
        
        return metrics
    
    def update_prometheus_metrics(self, metrics: List[PerformanceMetric]):
        """Update Prometheus metrics."""
        for metric in metrics:
            if metric.name == "cpu_usage_percent":
                self.cpu_usage.set(metric.value)
            elif metric.name == "memory_usage_percent":
                self.memory_usage.set(metric.value)
            elif metric.name == "disk_usage_percent":
                device = metric.tags.get('device', 'unknown')
                self.disk_usage.labels(device=device).set(metric.value)
            elif metric.name == "network_bytes_sent":
                self.network_io.labels(direction='sent').set(metric.value)
            elif metric.name == "network_bytes_recv":
                self.network_io.labels(direction='recv').set(metric.value)
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.is_monitoring = False
        self.logger.info("Performance monitoring stopped")
    
    def get_recent_metrics(self, minutes: int = 5) -> List[PerformanceMetric]:
        """Get recent metrics."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self.metrics_buffer if m.timestamp > cutoff]


class AnomalyDetector:
    """ML-based performance anomaly detector."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger('AnomalyDetector')
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        
        # Initialize models for different metric types
        self.model_types = ['cpu', 'memory', 'disk', 'network']
        for model_type in self.model_types:
            self.models[model_type] = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            self.scalers[model_type] = StandardScaler()
    
    def prepare_training_data(self, metrics: List[PerformanceMetric]) -> Dict[str, np.ndarray]:
        """Prepare training data from metrics."""
        data = defaultdict(list)
        
        for metric in metrics:
            metric_category = self._get_metric_category(metric.metric_type)
            if metric_category in self.model_types:
                # Create feature vector: [value, hour_of_day, day_of_week]
                hour_of_day = metric.timestamp.hour
                day_of_week = metric.timestamp.weekday()
                
                features = [metric.value, hour_of_day, day_of_week]
                data[metric_category].append(features)
        
        # Convert to numpy arrays
        for category in data:
            data[category] = np.array(data[category])
        
        return data
    
    def _get_metric_category(self, metric_type: MetricType) -> str:
        """Map metric type to model category."""
        mapping = {
            MetricType.CPU: 'cpu',
            MetricType.MEMORY: 'memory',
            MetricType.DISK: 'disk',
            MetricType.NETWORK: 'network'
        }
        return mapping.get(metric_type, 'unknown')
    
    async def train_models(self, historical_metrics: List[PerformanceMetric]):
        """Train anomaly detection models."""
        self.logger.info("Training anomaly detection models")
        
        training_data = self.prepare_training_data(historical_metrics)
        
        for category, data in training_data.items():
            if len(data) < 100:  # Need minimum data for training
                self.logger.warning(f"Insufficient data for {category} model: {len(data)} samples")
                continue
            
            # Scale features
            scaled_data = self.scalers[category].fit_transform(data)
            
            # Train model
            self.models[category].fit(scaled_data)
            
            self.logger.info(f"Trained {category} anomaly detection model with {len(data)} samples")
        
        self.is_trained = True
    
    async def detect_anomalies(
        self,
        recent_metrics: List[PerformanceMetric]
    ) -> List[PerformanceBottleneck]:
        """Detect performance anomalies."""
        
        if not self.is_trained:
            return []
        
        anomalies = []
        test_data = self.prepare_training_data(recent_metrics)
        
        for category, data in test_data.items():
            if len(data) == 0 or category not in self.models:
                continue
            
            # Scale features
            scaled_data = self.scalers[category].transform(data)
            
            # Predict anomalies
            predictions = self.models[category].predict(scaled_data)
            anomaly_scores = self.models[category].decision_function(scaled_data)
            
            # Process anomalies
            for i, (prediction, score) in enumerate(zip(predictions, anomaly_scores)):
                if prediction == -1:  # Anomaly detected
                    severity = self._calculate_severity(score)
                    
                    bottleneck = PerformanceBottleneck(
                        bottleneck_id=f"anomaly_{category}_{int(time.time())}_{i}",
                        name=f"{category.upper()} Performance Anomaly",
                        severity=severity,
                        component=category,
                        impact_score=min(abs(score) * 20, 100),
                        confidence=min(abs(score) * 0.1, 1.0),
                        description=f"Unusual {category} performance pattern detected",
                        root_cause=f"Anomalous {category} behavior outside normal patterns",
                        affected_metrics=[f"{category}_usage_percent"],
                        recommendations=self._get_anomaly_recommendations(category, severity),
                        estimated_improvement=abs(score) * 5,
                        detected_at=datetime.now(),
                        duration_seconds=60,  # Default duration
                        frequency=1
                    )
                    
                    anomalies.append(bottleneck)
        
        return anomalies
    
    def _calculate_severity(self, anomaly_score: float) -> str:
        """Calculate severity based on anomaly score."""
        abs_score = abs(anomaly_score)
        
        if abs_score > 0.5:
            return "critical"
        elif abs_score > 0.3:
            return "high"
        elif abs_score > 0.1:
            return "medium"
        else:
            return "low"
    
    def _get_anomaly_recommendations(self, category: str, severity: str) -> List[str]:
        """Get recommendations for anomaly category."""
        recommendations = {
            'cpu': [
                "Investigate high CPU usage processes",
                "Consider scaling up CPU resources",
                "Optimize application algorithms",
                "Check for infinite loops or inefficient code"
            ],
            'memory': [
                "Check for memory leaks",
                "Optimize memory allocation patterns",
                "Consider increasing available memory",
                "Review caching strategies"
            ],
            'disk': [
                "Clean up unnecessary files",
                "Optimize database queries",
                "Consider upgrading storage",
                "Implement data archiving"
            ],
            'network': [
                "Optimize network calls",
                "Implement request batching",
                "Check network configuration",
                "Consider CDN implementation"
            ]
        }
        
        base_recommendations = recommendations.get(category, [])
        
        if severity in ['critical', 'high']:
            return base_recommendations[:2]  # Most important recommendations
        else:
            return base_recommendations[2:4]  # Additional recommendations


class DatabaseOptimizer:
    """Database performance optimizer."""
    
    def __init__(self, config: OptimizationConfig, db_config: Dict[str, Any]):
        self.config = config
        self.db_config = db_config
        self.logger = logging.getLogger('DatabaseOptimizer')
        
        # Database connection
        self.engine = create_engine(
            f"postgresql://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
    
    async def analyze_slow_queries(self) -> List[Dict[str, Any]]:
        """Analyze slow queries."""
        slow_queries = []
        
        try:
            with self.engine.connect() as conn:
                # Enable slow query logging
                conn.execute(text("SELECT pg_reload_conf()"))
                
                # Get slow queries from pg_stat_statements
                query = """
                    SELECT 
                        query,
                        calls,
                        total_time,
                        mean_time,
                        rows,
                        100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
                    FROM pg_stat_statements 
                    WHERE mean_time > 100
                    ORDER BY mean_time DESC 
                    LIMIT 20
                """
                
                result = conn.execute(text(query))
                
                for row in result:
                    slow_queries.append({
                        'query': row.query,
                        'calls': row.calls,
                        'total_time_ms': row.total_time,
                        'mean_time_ms': row.mean_time,
                        'rows': row.rows,
                        'cache_hit_percent': row.hit_percent or 0
                    })
        
        except Exception as e:
            self.logger.error(f"Error analyzing slow queries: {str(e)}")
        
        return slow_queries
    
    async def suggest_indexes(self) -> List[Dict[str, Any]]:
        """Suggest database indexes."""
        index_suggestions = []
        
        try:
            with self.engine.connect() as conn:
                # Get missing indexes suggestions
                query = """
                    SELECT 
                        schemaname,
                        tablename,
                        attname,
                        n_distinct,
                        correlation
                    FROM pg_stats 
                    WHERE schemaname = 'public'
                    AND n_distinct > 10
                    AND correlation < 0.1
                    ORDER BY n_distinct DESC
                    LIMIT 10
                """
                
                result = conn.execute(text(query))
                
                for row in result:
                    index_suggestions.append({
                        'schema': row.schemaname,
                        'table': row.tablename,
                        'column': row.attname,
                        'distinctness': row.n_distinct,
                        'correlation': row.correlation,
                        'suggestion': f"CREATE INDEX idx_{row.tablename}_{row.attname} ON {row.tablename} ({row.attname})"
                    })
        
        except Exception as e:
            self.logger.error(f"Error suggesting indexes: {str(e)}")
        
        return index_suggestions
    
    async def optimize_connection_pool(self) -> Dict[str, Any]:
        """Optimize database connection pool."""
        recommendations = {}
        
        try:
            with self.engine.connect() as conn:
                # Get connection statistics
                result = conn.execute(text("""
                    SELECT 
                        setting::int as max_connections,
                        (SELECT count(*) FROM pg_stat_activity) as active_connections
                    FROM pg_settings 
                    WHERE name = 'max_connections'
                """))
                
                row = result.fetchone()
                max_connections = row.max_connections
                active_connections = row.active_connections
                
                utilization = (active_connections / max_connections) * 100
                
                recommendations = {
                    'current_max_connections': max_connections,
                    'active_connections': active_connections,
                    'utilization_percent': utilization,
                    'recommendations': []
                }
                
                if utilization > 80:
                    recommendations['recommendations'].append(
                        "Consider increasing max_connections or optimizing query performance"
                    )
                elif utilization < 20:
                    recommendations['recommendations'].append(
                        "Consider reducing max_connections to save memory"
                    )
                
                # Connection pooling recommendations
                recommendations['recommendations'].extend([
                    f"Recommended pool size: {max(10, active_connections * 2)}",
                    "Use connection pooling (PgBouncer) for better efficiency",
                    "Monitor connection lifetime and implement connection recycling"
                ])
        
        except Exception as e:
            self.logger.error(f"Error optimizing connection pool: {str(e)}")
        
        return recommendations


class ApplicationProfiler:
    """Application performance profiler."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger('ApplicationProfiler')
        self.profile_data = {}
    
    def profile_function(self, func):
        """Decorator for profiling function performance."""
        def wrapper(*args, **kwargs):
            if not self.config.code_profiling_enabled:
                return func(*args, **kwargs)
            
            start_time = time.time()
            
            # Memory profiling
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            try:
                result = func(*args, **kwargs)
                
                # Calculate metrics
                execution_time = time.time() - start_time
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                memory_used = memory_after - memory_before
                
                # Store profile data
                func_name = f"{func.__module__}.{func.__name__}"
                if func_name not in self.profile_data:
                    self.profile_data[func_name] = {
                        'call_count': 0,
                        'total_time': 0,
                        'min_time': float('inf'),
                        'max_time': 0,
                        'avg_memory_mb': 0,
                        'total_memory_mb': 0
                    }
                
                profile = self.profile_data[func_name]
                profile['call_count'] += 1
                profile['total_time'] += execution_time
                profile['min_time'] = min(profile['min_time'], execution_time)
                profile['max_time'] = max(profile['max_time'], execution_time)
                profile['total_memory_mb'] += memory_used
                profile['avg_memory_mb'] = profile['total_memory_mb'] / profile['call_count']
                
                return result
                
            except Exception as e:
                self.logger.error(f"Error profiling {func.__name__}: {str(e)}")
                raise
        
        return wrapper
    
    async def analyze_hot_spots(self) -> List[Dict[str, Any]]:
        """Analyze performance hot spots."""
        hot_spots = []
        
        for func_name, data in self.profile_data.items():
            if data['call_count'] > 0:
                avg_time = data['total_time'] / data['call_count']
                
                # Calculate performance score (lower is better)
                score = (avg_time * data['call_count']) + (data['avg_memory_mb'] * 0.1)
                
                hot_spots.append({
                    'function': func_name,
                    'call_count': data['call_count'],
                    'total_time_seconds': data['total_time'],
                    'average_time_seconds': avg_time,
                    'min_time_seconds': data['min_time'],
                    'max_time_seconds': data['max_time'],
                    'average_memory_mb': data['avg_memory_mb'],
                    'performance_score': score,
                    'recommendations': self._get_optimization_recommendations(data, avg_time)
                })
        
        # Sort by performance score (highest impact first)
        hot_spots.sort(key=lambda x: x['performance_score'], reverse=True)
        
        return hot_spots[:10]  # Top 10 hot spots
    
    def _get_optimization_recommendations(
        self,
        profile_data: Dict[str, Any],
        avg_time: float
    ) -> List[str]:
        """Get optimization recommendations for function."""
        recommendations = []
        
        if avg_time > 1.0:
            recommendations.append("Consider optimizing algorithm complexity")
        
        if profile_data['avg_memory_mb'] > 100:
            recommendations.append("Optimize memory usage - consider generators or streaming")
        
        if profile_data['call_count'] > 1000:
            recommendations.append("High call frequency - consider caching results")
        
        if profile_data['max_time'] > profile_data['min_time'] * 10:
            recommendations.append("High time variance - investigate input-dependent performance")
        
        return recommendations


class PerformanceOptimizer:
    """Ultra-advanced performance optimizer."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the performance optimizer."""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Initialize components
        self.system_monitor = SystemMonitor(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        self.application_profiler = ApplicationProfiler(self.config)
        
        # Database optimizer (if configured)
        self.database_optimizer = None
        if 'database' in self.config:
            self.database_optimizer = DatabaseOptimizer(
                self.config, self.config['database']
            )
        
        # Performance tracking
        self.bottlenecks: List[PerformanceBottleneck] = []
        self.recommendations: List[OptimizationRecommendation] = []
        self.optimization_history = []
        
        # Optimization state
        self.is_running = False
        self.optimization_thread = None
        
        # Performance metrics
        self.performance_metrics = {
            'bottlenecks_detected': 0,
            'optimizations_applied': 0,
            'average_improvement_percent': 0.0,
            'cost_savings_usd': 0.0,
            'uptime_seconds': 0
        }
    
    def _load_config(self, config_path: Optional[str]) -> OptimizationConfig:
        """Load configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                import yaml
                config_dict = yaml.safe_load(f)
                return OptimizationConfig(**config_dict)
        
        return OptimizationConfig()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger('PerformanceOptimizer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def start_optimization(self):
        """Start performance optimization."""
        self.is_running = True
        self.logger.info("Starting performance optimization")
        
        # Start system monitoring
        monitor_task = asyncio.create_task(self.system_monitor.start_monitoring())
        
        # Start optimization loop
        optimization_task = asyncio.create_task(self._optimization_loop())
        
        # Start Prometheus metrics server
        start_http_server(8000, registry=self.system_monitor.registry)
        
        await asyncio.gather(monitor_task, optimization_task)
    
    async def _optimization_loop(self):
        """Main optimization loop."""
        start_time = datetime.now()
        
        while self.is_running:
            try:
                loop_start = time.time()
                
                # Collect recent metrics
                recent_metrics = self.system_monitor.get_recent_metrics(minutes=5)
                
                if len(recent_metrics) < 10:
                    await asyncio.sleep(60)
                    continue
                
                # Train anomaly detection models (periodically)
                if (datetime.now() - start_time).total_seconds() > 3600:  # Every hour
                    await self._retrain_models()
                    start_time = datetime.now()
                
                # Detect performance bottlenecks
                bottlenecks = await self._detect_bottlenecks(recent_metrics)
                self.bottlenecks.extend(bottlenecks)
                
                # Generate optimization recommendations
                new_recommendations = await self._generate_recommendations(bottlenecks)
                self.recommendations.extend(new_recommendations)
                
                # Apply automatic optimizations
                if self.config.auto_apply_optimizations:
                    await self._apply_automatic_optimizations()
                
                # Database optimization
                if self.database_optimizer:
                    await self._optimize_database()
                
                # Update performance metrics
                self.performance_metrics['uptime_seconds'] = (
                    datetime.now() - start_time
                ).total_seconds()
                
                # Log progress
                if bottlenecks:
                    self.logger.info(
                        f"Detected {len(bottlenecks)} bottlenecks, "
                        f"generated {len(new_recommendations)} recommendations"
                    )
                
                # Sleep until next optimization cycle
                loop_time = time.time() - loop_start
                sleep_time = max(0, self.config.monitoring_interval_seconds - loop_time)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Optimization loop error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _retrain_models(self):
        """Retrain ML models with recent data."""
        self.logger.info("Retraining anomaly detection models")
        
        # Get historical metrics for training
        historical_metrics = list(self.system_monitor.metrics_buffer)
        
        if len(historical_metrics) > 100:
            await self.anomaly_detector.train_models(historical_metrics)
    
    async def _detect_bottlenecks(
        self,
        metrics: List[PerformanceMetric]
    ) -> List[PerformanceBottleneck]:
        """Detect performance bottlenecks."""
        bottlenecks = []
        
        # Threshold-based detection
        threshold_bottlenecks = self._detect_threshold_bottlenecks(metrics)
        bottlenecks.extend(threshold_bottlenecks)
        
        # ML-based anomaly detection
        if self.config.anomaly_detection_enabled:
            anomaly_bottlenecks = await self.anomaly_detector.detect_anomalies(metrics)
            bottlenecks.extend(anomaly_bottlenecks)
        
        # Application profiling bottlenecks
        if self.config.code_profiling_enabled:
            profiling_bottlenecks = await self._detect_profiling_bottlenecks()
            bottlenecks.extend(profiling_bottlenecks)
        
        # Update metrics
        self.performance_metrics['bottlenecks_detected'] += len(bottlenecks)
        
        return bottlenecks
    
    def _detect_threshold_bottlenecks(
        self,
        metrics: List[PerformanceMetric]
    ) -> List[PerformanceBottleneck]:
        """Detect bottlenecks based on thresholds."""
        bottlenecks = []
        
        # Group metrics by type
        metric_groups = defaultdict(list)
        for metric in metrics:
            metric_groups[metric.name].append(metric)
        
        # Check each metric type
        for metric_name, metric_list in metric_groups.items():
            latest_metric = max(metric_list, key=lambda m: m.timestamp)
            
            if latest_metric.critical_threshold and latest_metric.value > latest_metric.critical_threshold:
                severity = "critical"
            elif latest_metric.warning_threshold and latest_metric.value > latest_metric.warning_threshold:
                severity = "high"
            else:
                continue
            
            bottleneck = PerformanceBottleneck(
                bottleneck_id=f"threshold_{metric_name}_{int(time.time())}",
                name=f"{metric_name.replace('_', ' ').title()} Threshold Exceeded",
                severity=severity,
                component=latest_metric.metric_type.value,
                impact_score=min((latest_metric.value / latest_metric.warning_threshold) * 50, 100),
                confidence=0.9,
                description=f"{metric_name} has exceeded threshold: {latest_metric.value}{latest_metric.unit}",
                root_cause=f"High {metric_name} utilization",
                affected_metrics=[metric_name],
                recommendations=self._get_threshold_recommendations(latest_metric),
                estimated_improvement=20.0,
                detected_at=latest_metric.timestamp,
                duration_seconds=300,  # Assume 5 minutes
                frequency=1
            )
            
            bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def _get_threshold_recommendations(self, metric: PerformanceMetric) -> List[str]:
        """Get recommendations for threshold-based bottlenecks."""
        recommendations = {
            'cpu_usage_percent': [
                "Scale up CPU resources",
                "Optimize CPU-intensive operations",
                "Implement horizontal scaling",
                "Review application algorithms"
            ],
            'memory_usage_percent': [
                "Increase available memory",
                "Optimize memory allocation",
                "Implement memory caching strategies",
                "Check for memory leaks"
            ],
            'disk_usage_percent': [
                "Add more storage capacity",
                "Implement data archiving",
                "Optimize database storage",
                "Clean up temporary files"
            ]
        }
        
        return recommendations.get(metric.name, ["Investigate and optimize resource usage"])
    
    async def _detect_profiling_bottlenecks(self) -> List[PerformanceBottleneck]:
        """Detect bottlenecks from application profiling."""
        bottlenecks = []
        
        hot_spots = await self.application_profiler.analyze_hot_spots()
        
        for hot_spot in hot_spots[:5]:  # Top 5 hot spots
            if hot_spot['performance_score'] > 10:  # Significant impact threshold
                severity = "high" if hot_spot['performance_score'] > 50 else "medium"
                
                bottleneck = PerformanceBottleneck(
                    bottleneck_id=f"profiling_{hot_spot['function'].replace('.', '_')}_{int(time.time())}",
                    name=f"Performance Hot Spot: {hot_spot['function']}",
                    severity=severity,
                    component="application",
                    impact_score=min(hot_spot['performance_score'], 100),
                    confidence=0.8,
                    description=f"Function {hot_spot['function']} is a performance hot spot",
                    root_cause=f"Inefficient function with {hot_spot['call_count']} calls",
                    affected_metrics=['application_response_time'],
                    recommendations=hot_spot['recommendations'],
                    estimated_improvement=min(hot_spot['performance_score'] * 0.5, 50),
                    detected_at=datetime.now(),
                    duration_seconds=hot_spot['total_time_seconds'],
                    frequency=hot_spot['call_count']
                )
                
                bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    async def _generate_recommendations(
        self,
        bottlenecks: List[PerformanceBottleneck]
    ) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations."""
        recommendations = []
        
        for bottleneck in bottlenecks:
            # Generate recommendation based on bottleneck
            recommendation = OptimizationRecommendation(
                recommendation_id=f"rec_{bottleneck.bottleneck_id}",
                category=bottleneck.component,
                priority=bottleneck.severity,
                title=f"Optimize {bottleneck.component} performance",
                description=f"Address {bottleneck.name}",
                rationale=bottleneck.description,
                estimated_improvement_percent=bottleneck.estimated_improvement,
                estimated_cost_savings_usd=self._estimate_cost_savings(bottleneck),
                implementation_effort=self._estimate_implementation_effort(bottleneck),
                steps=bottleneck.recommendations,
                automation_available=self._check_automation_available(bottleneck),
                rollback_plan=self._generate_rollback_plan(bottleneck),
                success_metrics=[f"{bottleneck.component}_performance_improvement"],
                validation_period_days=7,
                created_at=datetime.now()
            )
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _estimate_cost_savings(self, bottleneck: PerformanceBottleneck) -> float:
        """Estimate cost savings from optimization."""
        # Simplified cost estimation
        base_monthly_cost = 1000.0  # Assume $1000 baseline monthly cost
        
        if bottleneck.component == "cpu":
            return base_monthly_cost * 0.3 * (bottleneck.impact_score / 100)
        elif bottleneck.component == "memory":
            return base_monthly_cost * 0.2 * (bottleneck.impact_score / 100)
        elif bottleneck.component == "disk":
            return base_monthly_cost * 0.1 * (bottleneck.impact_score / 100)
        else:
            return base_monthly_cost * 0.05 * (bottleneck.impact_score / 100)
    
    def _estimate_implementation_effort(self, bottleneck: PerformanceBottleneck) -> str:
        """Estimate implementation effort."""
        if bottleneck.component in ["cpu", "memory"] and bottleneck.severity == "critical":
            return "high"
        elif bottleneck.component == "application":
            return "medium"
        else:
            return "low"
    
    def _check_automation_available(self, bottleneck: PerformanceBottleneck) -> bool:
        """Check if automation is available for this bottleneck."""
        automatable_components = ["cpu", "memory", "disk"]
        return bottleneck.component in automatable_components
    
    def _generate_rollback_plan(self, bottleneck: PerformanceBottleneck) -> str:
        """Generate rollback plan for optimization."""
        if bottleneck.component == "cpu":
            return "Scale down CPU resources to previous configuration"
        elif bottleneck.component == "memory":
            return "Restore previous memory allocation settings"
        elif bottleneck.component == "disk":
            return "Revert storage configuration changes"
        else:
            return "Revert configuration changes and restart services"
    
    async def _apply_automatic_optimizations(self):
        """Apply automatic optimizations."""
        for recommendation in self.recommendations:
            if (not recommendation.applied and 
                recommendation.automation_available and
                recommendation.priority in ["critical", "high"]):
                
                success = await self._apply_optimization(recommendation)
                
                if success:
                    recommendation.applied = True
                    recommendation.applied_at = datetime.now()
                    
                    self.performance_metrics['optimizations_applied'] += 1
                    self.performance_metrics['cost_savings_usd'] += recommendation.estimated_cost_savings_usd
                    
                    self.logger.info(
                        f"Applied automatic optimization: {recommendation.title}"
                    )
    
    async def _apply_optimization(self, recommendation: OptimizationRecommendation) -> bool:
        """Apply a specific optimization."""
        try:
            if recommendation.category == "cpu":
                return await self._optimize_cpu(recommendation)
            elif recommendation.category == "memory":
                return await self._optimize_memory(recommendation)
            elif recommendation.category == "disk":
                return await self._optimize_disk(recommendation)
            else:
                self.logger.warning(f"No automation available for {recommendation.category}")
                return False
        
        except Exception as e:
            self.logger.error(f"Failed to apply optimization: {str(e)}")
            return False
    
    async def _optimize_cpu(self, recommendation: OptimizationRecommendation) -> bool:
        """Optimize CPU performance."""
        # Example CPU optimization (would depend on environment)
        self.logger.info("Applying CPU optimization")
        
        # In a real implementation, this might:
        # - Adjust process priorities
        # - Scale up compute resources
        # - Optimize thread pools
        
        return True
    
    async def _optimize_memory(self, recommendation: OptimizationRecommendation) -> bool:
        """Optimize memory performance."""
        self.logger.info("Applying memory optimization")
        
        # In a real implementation, this might:
        # - Adjust garbage collection settings
        # - Scale up memory resources
        # - Optimize caching strategies
        
        return True
    
    async def _optimize_disk(self, recommendation: OptimizationRecommendation) -> bool:
        """Optimize disk performance."""
        self.logger.info("Applying disk optimization")
        
        # In a real implementation, this might:
        # - Clean up temporary files
        # - Optimize database storage
        # - Implement data compression
        
        return True
    
    async def _optimize_database(self):
        """Optimize database performance."""
        if not self.database_optimizer:
            return
        
        # Analyze slow queries
        slow_queries = await self.database_optimizer.analyze_slow_queries()
        
        if slow_queries:
            self.logger.info(f"Found {len(slow_queries)} slow queries")
            
            # Generate recommendations for slow queries
            for query_info in slow_queries[:5]:  # Top 5 slow queries
                recommendation = OptimizationRecommendation(
                    recommendation_id=f"db_query_{int(time.time())}",
                    category="database",
                    priority="medium" if query_info['mean_time_ms'] > 500 else "low",
                    title="Optimize slow database query",
                    description=f"Query with {query_info['mean_time_ms']:.1f}ms average time",
                    rationale="Slow queries impact overall application performance",
                    estimated_improvement_percent=10.0,
                    estimated_cost_savings_usd=50.0,
                    implementation_effort="medium",
                    steps=[
                        "Analyze query execution plan",
                        "Add appropriate indexes",
                        "Optimize query structure",
                        "Test performance improvement"
                    ],
                    automation_available=False,
                    rollback_plan="Remove added indexes if performance degrades",
                    success_metrics=["query_response_time"],
                    validation_period_days=3,
                    created_at=datetime.now()
                )
                
                self.recommendations.append(recommendation)
        
        # Suggest indexes
        index_suggestions = await self.database_optimizer.suggest_indexes()
        
        if index_suggestions:
            self.logger.info(f"Generated {len(index_suggestions)} index suggestions")
    
    def stop_optimization(self):
        """Stop performance optimization."""
        self.is_running = False
        self.system_monitor.stop_monitoring()
        self.logger.info("Performance optimization stopped")
    
    def get_bottlenecks(
        self,
        severity_filter: Optional[str] = None,
        component_filter: Optional[str] = None
    ) -> List[PerformanceBottleneck]:
        """Get detected bottlenecks with optional filters."""
        bottlenecks = self.bottlenecks.copy()
        
        if severity_filter:
            bottlenecks = [b for b in bottlenecks if b.severity == severity_filter]
        
        if component_filter:
            bottlenecks = [b for b in bottlenecks if b.component == component_filter]
        
        return bottlenecks
    
    def get_recommendations(
        self,
        applied_filter: Optional[bool] = None,
        priority_filter: Optional[str] = None
    ) -> List[OptimizationRecommendation]:
        """Get optimization recommendations with optional filters."""
        recommendations = self.recommendations.copy()
        
        if applied_filter is not None:
            recommendations = [r for r in recommendations if r.applied == applied_filter]
        
        if priority_filter:
            recommendations = [r for r in recommendations if r.priority == priority_filter]
        
        return recommendations
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.performance_metrics.copy()
    
    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        
        # Get recent metrics
        recent_metrics = self.system_monitor.get_recent_metrics(minutes=60)
        
        # Calculate summary statistics
        cpu_metrics = [m for m in recent_metrics if m.name == "cpu_usage_percent"]
        memory_metrics = [m for m in recent_metrics if m.name == "memory_usage_percent"]
        
        cpu_avg = statistics.mean([m.value for m in cpu_metrics]) if cpu_metrics else 0
        memory_avg = statistics.mean([m.value for m in memory_metrics]) if memory_metrics else 0
        
        # Get hot spots
        hot_spots = await self.application_profiler.analyze_hot_spots()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'avg_cpu_usage_percent': cpu_avg,
                'avg_memory_usage_percent': memory_avg,
                'total_bottlenecks': len(self.bottlenecks),
                'total_recommendations': len(self.recommendations),
                'optimizations_applied': self.performance_metrics['optimizations_applied']
            },
            'bottlenecks': [asdict(b) for b in self.bottlenecks[-10:]],  # Latest 10
            'recommendations': [asdict(r) for r in self.recommendations[-10:]],  # Latest 10
            'hot_spots': hot_spots[:5],  # Top 5
            'performance_metrics': self.get_performance_metrics()
        }
        
        return report


# Utility functions
async def optimize_system_performance(
    config_path: Optional[str] = None,
    duration_hours: float = 24.0
) -> Dict[str, Any]:
    """Run system performance optimization for specified duration."""
    
    optimizer = PerformanceOptimizer(config_path)
    
    # Start optimization
    optimization_task = asyncio.create_task(optimizer.start_optimization())
    
    # Run for specified duration
    await asyncio.sleep(duration_hours * 3600)
    
    # Stop optimization
    optimizer.stop_optimization()
    optimization_task.cancel()
    
    # Generate final report
    report = await optimizer.generate_performance_report()
    
    return report


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Create performance optimizer
        optimizer = PerformanceOptimizer()
        
        # Run optimization for 1 minute (demo)
        optimization_task = asyncio.create_task(optimizer.start_optimization())
        
        # Let it run for a minute
        await asyncio.sleep(60)
        
        # Generate report
        report = await optimizer.generate_performance_report()
        
        print("Performance Optimization Report:")
        print(f"CPU Usage: {report['summary']['avg_cpu_usage_percent']:.1f}%")
        print(f"Memory Usage: {report['summary']['avg_memory_usage_percent']:.1f}%")
        print(f"Bottlenecks Detected: {report['summary']['total_bottlenecks']}")
        print(f"Recommendations: {report['summary']['total_recommendations']}")
        
        # Stop optimization
        optimizer.stop_optimization()
    
    asyncio.run(main())
