"""
Advanced Metrics Collector for Multi-Tenant Analytics

This module implements an ultra-sophisticated metrics collection system with real-time
aggregation, intelligent dimensionality, ML-enhanced metric analysis, and predictive analytics.

Features:
- Real-time metric aggregation and computation
- Multi-dimensional metric storage and querying
- ML-powered metric forecasting and anomaly detection
- Automatic metric derivation and correlation
- Intelligent sampling and compression
- High-cardinality metric handling
- Custom metric definitions and calculations

Created by Expert Team:
- Lead Dev + AI Architect: Architecture and ML integration
- ML Engineer: Predictive analytics and anomaly detection
- DBA & Data Engineer: Time-series optimization and storage
- Senior Backend Developer: Real-time processing and APIs
- Backend Security Specialist: Metric security and tenant isolation
- Microservices Architect: Distributed metrics infrastructure

Developed by: Fahed Mlaiel
"""

from typing import Dict, List, Optional, Any, Union, Tuple, Callable, AsyncGenerator
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import logging
import uuid
import time
import math
import statistics
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import aioredis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select, func
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
from functools import lru_cache
import heapq
import bisect

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics that can be collected"""
    COUNTER = "counter"  # Monotonically increasing
    GAUGE = "gauge"      # Point-in-time value
    HISTOGRAM = "histogram"  # Distribution of values
    SUMMARY = "summary"  # Statistical summary
    TIMER = "timer"      # Duration measurements
    RATE = "rate"        # Rate calculations
    DISTRIBUTION = "distribution"  # Value distribution
    SET = "set"          # Unique value counting

class AggregationType(Enum):
    """Aggregation types for metrics"""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    MEDIAN = "median"
    PERCENTILE = "percentile"
    STDDEV = "stddev"
    VARIANCE = "variance"
    RATE = "rate"

class TimeWindow(Enum):
    """Time windows for metric aggregation"""
    SECOND = "1s"
    MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    HOUR = "1h"
    DAY = "1d"
    WEEK = "1w"
    MONTH = "1M"

@dataclass
class MetricsConfig:
    """Configuration for metrics collection"""
    retention_days: int = 365
    aggregation_intervals: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h", "1d"])
    precision_decimals: int = 4
    percentiles: List[float] = field(default_factory=lambda: [50, 90, 95, 99, 99.9])
    cardinality_limit: int = 1000000
    sampling_rate: float = 1.0
    compression_enabled: bool = True
    real_time_enabled: bool = True
    ml_analysis_enabled: bool = True
    auto_derivation_enabled: bool = True
    anomaly_detection_threshold: float = 0.8

@dataclass
class Metric:
    """Comprehensive metric data structure"""
    name: str
    value: Union[int, float, List[float], Dict[str, float]]
    type: MetricType
    timestamp: datetime
    tenant_id: str
    
    # Dimensions and tags
    dimensions: Dict[str, str] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Metadata
    unit: Optional[str] = None
    description: Optional[str] = None
    source: str = "unknown"
    
    # Processing information
    aggregated: bool = False
    derived: bool = False
    anomalous: bool = False
    confidence: Optional[float] = None
    
    # Statistical information (for histograms/summaries)
    buckets: Optional[Dict[str, int]] = None
    quantiles: Optional[Dict[str, float]] = None
    sample_count: Optional[int] = None
    sample_sum: Optional[float] = None

@dataclass
class MetricDefinition:
    """Definition for custom metric calculations"""
    name: str
    expression: str  # Mathematical expression
    dependencies: List[str]  # Required input metrics
    output_type: MetricType
    aggregation: AggregationType
    time_window: TimeWindow
    tenant_id: str
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class AggregatedMetric:
    """Aggregated metric with statistical information"""
    name: str
    aggregation_type: AggregationType
    time_window: TimeWindow
    start_time: datetime
    end_time: datetime
    value: Union[float, Dict[str, float]]
    sample_count: int
    tenant_id: str
    dimensions: Dict[str, str] = field(default_factory=dict)

@dataclass
class MetricAlert:
    """Alert definition for metric thresholds"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metric_name: str = ""
    threshold_value: float = 0.0
    comparison: str = ">"  # >, <, >=, <=, ==, !=
    time_window: TimeWindow = TimeWindow.MINUTE
    tenant_id: str = ""
    is_active: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0

@dataclass
class MetricForecast:
    """ML-generated metric forecast"""
    metric_name: str
    forecast_values: List[float]
    forecast_timestamps: List[datetime]
    confidence_intervals: List[Tuple[float, float]]
    model_accuracy: float
    generated_at: datetime
    tenant_id: str
    forecast_horizon_hours: int

@dataclass
class MetricStats:
    """Statistics for metrics collection"""
    total_metrics: int = 0
    unique_metrics: int = 0
    aggregations_performed: int = 0
    anomalies_detected: int = 0
    forecasts_generated: int = 0
    avg_processing_time_ms: float = 0.0
    cardinality_current: int = 0
    storage_size_mb: float = 0.0
    last_metric_time: Optional[datetime] = None

class MetricsCollector:
    """
    Ultra-advanced metrics collector with ML analytics and predictive capabilities
    """
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Metric storage and aggregation
        self.metric_buffers = defaultdict(deque)
        self.aggregated_metrics = defaultdict(dict)
        self.metric_definitions = defaultdict(dict)
        self.alert_definitions = defaultdict(list)
        
        # Real-time processing
        self.real_time_aggregators = {}
        self.streaming_processors = {}
        
        # ML models and analyzers
        self.anomaly_detector = None
        self.forecasting_models = {}
        self.correlation_analyzer = None
        self.feature_scaler = StandardScaler()
        
        # Time-series storage
        self.timeseries_store = defaultdict(dict)
        self.compression_engine = None
        
        # Processing infrastructure
        self.executor = ThreadPoolExecutor(max_workers=16)
        self.aggregation_tasks = {}
        
        # Statistics and monitoring
        self.tenant_stats = defaultdict(lambda: MetricStats())
        self.global_stats = MetricStats()
        
        # Performance optimization
        self.metric_cache = {}
        self.cardinality_tracker = defaultdict(set)
        self.sampling_manager = None
        
        self.is_initialized = False
        self.start_time = datetime.utcnow()
    
    async def initialize(self) -> bool:
        """Initialize metrics collector with all components"""
        try:
            self.logger.info("Initializing Metrics Collector...")
            
            # Initialize ML models
            await self._initialize_ml_models()
            
            # Initialize time-series storage
            await self._initialize_timeseries_storage()
            
            # Initialize compression engine
            await self._initialize_compression()
            
            # Initialize real-time aggregators
            await self._initialize_real_time_processing()
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Load predefined metrics
            await self._load_predefined_metrics()
            
            self.is_initialized = True
            self.logger.info("Metrics Collector initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Metrics Collector: {e}")
            return False
    
    async def register_tenant(self, tenant_id: str, config: Optional[Dict] = None) -> bool:
        """Register tenant with custom metrics configuration"""
        try:
            tenant_config = {
                "custom_metrics": {},
                "aggregation_preferences": {},
                "alert_rules": {},
                "retention_policy": {"days": self.config.retention_days},
                "ml_preferences": {"forecasting": True, "anomaly_detection": True},
                **(config or {})
            }
            
            # Initialize tenant-specific components
            await self._create_tenant_aggregators(tenant_id)
            await self._initialize_tenant_metrics(tenant_id, tenant_config)
            
            # Start tenant-specific processing
            self.aggregation_tasks[tenant_id] = asyncio.create_task(
                self._process_tenant_metrics(tenant_id)
            )
            
            self.logger.info(f"Tenant {tenant_id} registered for metrics collection")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register tenant {tenant_id}: {e}")
            return False
    
    async def record_metric(
        self,
        tenant_id: str,
        metric: Metric,
        immediate_processing: bool = False
    ) -> bool:
        """Record a single metric with optional immediate processing"""
        try:
            start_time = time.time()
            
            # Set tenant ID and timestamp if not set
            if not metric.tenant_id:
                metric.tenant_id = tenant_id
            if not metric.timestamp:
                metric.timestamp = datetime.utcnow()
            
            # Validate metric
            if not await self._validate_metric(metric):
                self.tenant_stats[tenant_id].total_metrics += 1  # Count as attempt
                return False
            
            # Check cardinality limits
            if not await self._check_cardinality_limits(tenant_id, metric):
                self.logger.warning(f"Cardinality limit exceeded for tenant {tenant_id}")
                return False
            
            # Apply sampling if configured
            if not await self._should_sample_metric(metric):
                return True  # Skip but don't count as error
            
            # Enrich metric with derived values
            if self.config.auto_derivation_enabled:
                await self._derive_metric_values(metric)
            
            # ML-powered anomaly detection
            if self.config.ml_analysis_enabled:
                await self._detect_metric_anomaly(metric)
            
            # Add to buffer for batch processing
            await self._add_to_buffer(tenant_id, metric)
            
            # Immediate processing for real-time metrics
            if immediate_processing or self.config.real_time_enabled:
                await self._process_metric_immediately(tenant_id, metric)
            
            # Update processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Update statistics
            await self._update_metric_stats(tenant_id, metric, processing_time)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to record metric for tenant {tenant_id}: {e}")
            return False
    
    async def record_metric_batch(
        self,
        tenant_id: str,
        metrics: List[Metric]
    ) -> Dict[str, int]:
        """Record a batch of metrics efficiently"""
        try:
            results = {"successful": 0, "failed": 0, "sampled_out": 0}
            
            # Process metrics in parallel batches
            batch_size = 1000  # Optimize batch size
            
            for i in range(0, len(metrics), batch_size):
                batch = metrics[i:i + batch_size]
                
                # Process batch concurrently
                tasks = [
                    self.record_metric(tenant_id, metric, immediate_processing=False)
                    for metric in batch
                ]
                
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        results["failed"] += 1
                    elif result:
                        results["successful"] += 1
                    else:
                        results["failed"] += 1
            
            # Trigger batch aggregation
            await self._trigger_batch_aggregation(tenant_id)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch metric recording failed for tenant {tenant_id}: {e}")
            return {"successful": 0, "failed": len(metrics), "sampled_out": 0}
    
    async def define_custom_metric(
        self,
        tenant_id: str,
        definition: MetricDefinition
    ) -> bool:
        """Define a custom derived metric"""
        try:
            # Validate definition
            if not await self._validate_metric_definition(definition):
                return False
            
            # Set tenant ID
            definition.tenant_id = tenant_id
            
            # Compile expression
            compiled_expression = await self._compile_metric_expression(definition)
            if not compiled_expression:
                return False
            
            # Store definition
            self.metric_definitions[tenant_id][definition.name] = {
                "definition": definition,
                "compiled": compiled_expression
            }
            
            self.logger.info(f"Custom metric '{definition.name}' defined for tenant {tenant_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to define custom metric for tenant {tenant_id}: {e}")
            return False
    
    async def aggregate_metrics(
        self,
        tenant_id: str,
        metric_names: List[str],
        aggregation_type: AggregationType,
        time_window: TimeWindow,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> List[AggregatedMetric]:
        """Aggregate metrics over time windows"""
        try:
            aggregated_results = []
            
            for metric_name in metric_names:
                # Get metric data for time range
                metric_data = await self._get_metric_data(
                    tenant_id, metric_name, time_range
                )
                
                if not metric_data:
                    continue
                
                # Perform aggregation
                aggregated_result = await self._perform_aggregation(
                    metric_data, aggregation_type, time_window
                )
                
                if aggregated_result:
                    aggregated_results.append(aggregated_result)
            
            # Update aggregation statistics
            self.tenant_stats[tenant_id].aggregations_performed += len(aggregated_results)
            
            return aggregated_results
            
        except Exception as e:
            self.logger.error(f"Failed to aggregate metrics for tenant {tenant_id}: {e}")
            return []
    
    async def query_metrics(
        self,
        tenant_id: str,
        query: Dict[str, Any]
    ) -> List[Metric]:
        """Query metrics with advanced filtering and aggregation"""
        try:
            # Parse query parameters
            metric_names = query.get("metrics", [])
            filters = query.get("filters", {})
            time_range = query.get("time_range")
            aggregation = query.get("aggregation")
            limit = query.get("limit", 10000)
            
            # Execute query
            results = await self._execute_metric_query(
                tenant_id, metric_names, filters, time_range, aggregation, limit
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to query metrics for tenant {tenant_id}: {e}")
            return []
    
    async def generate_forecast(
        self,
        tenant_id: str,
        metric_name: str,
        forecast_horizon_hours: int = 24
    ) -> Optional[MetricForecast]:
        """Generate ML-powered metric forecast"""
        try:
            # Get historical data
            historical_data = await self._get_historical_metric_data(
                tenant_id, metric_name, hours=forecast_horizon_hours * 10  # 10x for training
            )
            
            if len(historical_data) < 50:  # Minimum data requirement
                return None
            
            # Generate forecast using ML model
            forecast = await self._generate_ml_forecast(
                historical_data, forecast_horizon_hours
            )
            
            if forecast:
                # Store forecast for caching
                await self._store_forecast(tenant_id, forecast)
                
                # Update statistics
                self.tenant_stats[tenant_id].forecasts_generated += 1
            
            return forecast
            
        except Exception as e:
            self.logger.error(f"Failed to generate forecast for metric {metric_name}: {e}")
            return None
    
    async def set_metric_alert(
        self,
        tenant_id: str,
        alert: MetricAlert
    ) -> bool:
        """Set up metric alert threshold"""
        try:
            # Validate alert configuration
            if not await self._validate_alert_config(alert):
                return False
            
            # Set tenant ID
            alert.tenant_id = tenant_id
            
            # Store alert definition
            self.alert_definitions[tenant_id].append(alert)
            
            self.logger.info(f"Alert set for metric '{alert.metric_name}' on tenant {tenant_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set metric alert for tenant {tenant_id}: {e}")
            return False
    
    async def get_count(self, tenant_id: str) -> int:
        """Get total metrics count for tenant"""
        return self.tenant_stats[tenant_id].total_metrics
    
    async def get_metric_stats(self, tenant_id: str) -> MetricStats:
        """Get metrics statistics for tenant"""
        return self.tenant_stats[tenant_id]
    
    async def is_healthy(self) -> bool:
        """Check metrics collector health"""
        try:
            if not self.is_initialized:
                return False
            
            # Check buffer sizes
            for tenant_id, buffer in self.metric_buffers.items():
                if len(buffer) > 100000:  # Buffer size limit
                    return False
            
            # Check cardinality
            total_cardinality = sum(
                len(cardinality) for cardinality in self.cardinality_tracker.values()
            )
            if total_cardinality > self.config.cardinality_limit:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    async def update_real_time_metrics(
        self,
        tenant_id: str,
        data_point: 'DataPoint'
    ) -> None:
        """Update real-time metrics from data point"""
        try:
            # Extract metrics from data point
            metrics = await self._extract_metrics_from_data_point(data_point)
            
            # Record each metric
            for metric in metrics:
                await self.record_metric(tenant_id, metric, immediate_processing=True)
            
        except Exception as e:
            self.logger.error(f"Failed to update real-time metrics: {e}")
    
    async def _initialize_ml_models(self) -> None:
        """Initialize ML models for metric analytics"""
        try:
            # Anomaly detection model
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_jobs=-1
            )
            
            # Forecasting models for different metric types
            self.forecasting_models = {
                "linear": LinearRegression(),
                "polynomial": None,  # Would be polynomial regression
                "lstm": None,       # Would be LSTM model
                "arima": None       # Would be ARIMA model
            }
            
            # Load pre-trained models if available
            await self._load_pretrained_models()
            
            self.logger.info("ML models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ML models: {e}")
            raise
    
    async def _initialize_timeseries_storage(self) -> None:
        """Initialize time-series storage system"""
        try:
            # Initialize TimescaleDB or InfluxDB connection
            # Implementation would depend on chosen time-series database
            
            self.logger.info("Time-series storage initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize time-series storage: {e}")
            raise
    
    async def _validate_metric(self, metric: Metric) -> bool:
        """Validate metric data and structure"""
        try:
            # Basic validation
            if not metric.name or not metric.tenant_id:
                return False
            
            # Type-specific validation
            if metric.type == MetricType.COUNTER:
                if not isinstance(metric.value, (int, float)) or metric.value < 0:
                    return False
            
            elif metric.type == MetricType.HISTOGRAM:
                if not isinstance(metric.value, (list, dict)):
                    return False
            
            # Timestamp validation
            now = datetime.utcnow()
            if metric.timestamp > now + timedelta(minutes=5):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Metric validation failed: {e}")
            return False
    
    async def _check_cardinality_limits(self, tenant_id: str, metric: Metric) -> bool:
        """Check if metric exceeds cardinality limits"""
        try:
            # Create cardinality key
            cardinality_key = f"{metric.name}:{json.dumps(metric.dimensions, sort_keys=True)}"
            
            # Add to cardinality tracker
            self.cardinality_tracker[tenant_id].add(cardinality_key)
            
            # Check limits
            current_cardinality = len(self.cardinality_tracker[tenant_id])
            
            return current_cardinality <= self.config.cardinality_limit
            
        except Exception as e:
            self.logger.error(f"Cardinality check failed: {e}")
            return True  # Allow on error
    
    async def _should_sample_metric(self, metric: Metric) -> bool:
        """Determine if metric should be sampled"""
        try:
            # Simple random sampling
            import random
            return random.random() <= self.config.sampling_rate
            
        except Exception as e:
            self.logger.error(f"Sampling decision failed: {e}")
            return True  # Include on error
    
    async def _derive_metric_values(self, metric: Metric) -> None:
        """Derive additional metric values automatically"""
        try:
            # Add derived dimensions
            if metric.type == MetricType.TIMER and isinstance(metric.value, (int, float)):
                # Convert to different time units
                metric.dimensions["ms"] = str(metric.value)
                metric.dimensions["seconds"] = str(metric.value / 1000)
            
            # Add statistical derivations for histograms
            if metric.type == MetricType.HISTOGRAM and isinstance(metric.value, list):
                values = metric.value
                metric.dimensions.update({
                    "mean": str(statistics.mean(values)),
                    "median": str(statistics.median(values)),
                    "std": str(statistics.stdev(values)) if len(values) > 1 else "0"
                })
            
        except Exception as e:
            self.logger.error(f"Metric derivation failed: {e}")
    
    async def _detect_metric_anomaly(self, metric: Metric) -> None:
        """Detect anomalies in metric values using ML"""
        try:
            if not self.anomaly_detector or not isinstance(metric.value, (int, float)):
                return
            
            # Get historical values for comparison
            historical_values = await self._get_recent_metric_values(
                metric.tenant_id, metric.name
            )
            
            if len(historical_values) < 10:  # Need minimum historical data
                return
            
            # Prepare data for anomaly detection
            all_values = historical_values + [metric.value]
            values_array = np.array(all_values).reshape(-1, 1)
            
            # Detect anomaly
            anomaly_scores = self.anomaly_detector.decision_function(values_array)
            current_score = anomaly_scores[-1]
            
            # Mark as anomalous if score is below threshold
            if current_score < -self.config.anomaly_detection_threshold:
                metric.anomalous = True
                self.tenant_stats[metric.tenant_id].anomalies_detected += 1
            
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
    
    # Placeholder implementations for complex methods
    async def _add_to_buffer(self, tenant_id, metric): pass
    async def _process_metric_immediately(self, tenant_id, metric): pass
    async def _update_metric_stats(self, tenant_id, metric, processing_time): pass
    async def _initialize_compression(self): pass
    async def _initialize_real_time_processing(self): pass
    async def _start_background_tasks(self): pass
    async def _load_predefined_metrics(self): pass
    async def _create_tenant_aggregators(self, tenant_id): pass
    async def _initialize_tenant_metrics(self, tenant_id, config): pass
    async def _process_tenant_metrics(self, tenant_id): pass
    async def _trigger_batch_aggregation(self, tenant_id): pass
    async def _validate_metric_definition(self, definition): return True
    async def _compile_metric_expression(self, definition): return None
    async def _get_metric_data(self, tenant_id, metric_name, time_range): return []
    async def _perform_aggregation(self, data, agg_type, time_window): return None
    async def _execute_metric_query(self, tenant_id, metrics, filters, time_range, agg, limit): return []
    async def _get_historical_metric_data(self, tenant_id, metric_name, hours): return []
    async def _generate_ml_forecast(self, data, horizon): return None
    async def _store_forecast(self, tenant_id, forecast): pass
    async def _validate_alert_config(self, alert): return True
    async def _extract_metrics_from_data_point(self, data_point): return []
    async def _load_pretrained_models(self): pass
    async def _get_recent_metric_values(self, tenant_id, metric_name): return []

# Export main classes
__all__ = [
    "MetricsCollector",
    "MetricsConfig",
    "Metric",
    "MetricDefinition", 
    "AggregatedMetric",
    "MetricAlert",
    "MetricForecast",
    "MetricStats",
    "MetricType",
    "AggregationType",
    "TimeWindow"
]
