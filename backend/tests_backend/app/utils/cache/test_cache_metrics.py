"""
Tests for cache metrics and monitoring in Spotify AI Agent

Comprehensive testing suite for cache performance metrics, monitoring,
alerting, and performance analytics.

Developed by Expert Team:
- Lead Dev + AI Architect
- Senior Backend Developer (Python/FastAPI/Django)
- Machine Learning Engineer (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Backend Security Specialist
- Microservices Architect
"""

import pytest
import time
import asyncio
import threading
from unittest.mock import Mock, patch
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

from app.utils.cache.metrics import (
    CacheMetrics, CacheMetricsCollector, CachePerformanceAnalyzer,
    CacheHealthMonitor, CacheAlertManager, MetricsAggregator,
    PerformanceBenchmark, CacheStatsTracker
)
from app.utils.cache.backends.memory_backend import MemoryCacheBackend
from app.utils.cache.backends.redis_backend import RedisCacheBackend
from app.utils.cache.manager import CacheManager
from app.utils.cache.events import CacheEvent


@dataclass
class MockMetricsData:
    """Mock metrics data for testing"""
    timestamp: datetime
    operation_type: str
    key: str
    hit: bool
    latency_ms: float
    size_bytes: int
    backend: str = "memory"


class TestCacheMetrics:
    """Test basic cache metrics functionality"""
    
    @pytest.fixture
    def cache_backend(self):
        """Cache backend fixture"""
        return MemoryCacheBackend(max_size=1000, default_ttl=3600)
    
    @pytest.fixture
    def cache_manager(self, cache_backend):
        """Cache manager fixture"""
        return CacheManager(default_backend=cache_backend)
    
    @pytest.fixture
    def metrics_collector(self, cache_manager):
        """Metrics collector fixture"""
        return CacheMetricsCollector(cache_manager)
    
    def test_cache_metrics_initialization(self):
        """Test cache metrics object initialization"""
        metrics = CacheMetrics()
        
        assert metrics.total_operations == 0
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0
        assert metrics.hit_rate == 0.0
        assert metrics.miss_rate == 0.0
        assert metrics.total_latency_ms == 0.0
        assert metrics.avg_latency_ms == 0.0
    
    def test_cache_metrics_hit_rate_calculation(self):
        """Test hit rate calculation"""
        metrics = CacheMetrics()
        
        # Record some hits and misses
        metrics.record_hit("key1", 1.5, 100)
        metrics.record_hit("key2", 2.0, 150)
        metrics.record_miss("key3", 0.5)
        metrics.record_miss("key4", 0.8)
        
        assert metrics.total_operations == 4
        assert metrics.cache_hits == 2
        assert metrics.cache_misses == 2
        assert metrics.hit_rate == 0.5
        assert metrics.miss_rate == 0.5
    
    def test_cache_metrics_latency_calculation(self):
        """Test latency calculation"""
        metrics = CacheMetrics()
        
        # Record operations with different latencies
        metrics.record_hit("key1", 10.0, 100)
        metrics.record_hit("key2", 20.0, 200)
        metrics.record_miss("key3", 5.0)
        
        assert metrics.total_latency_ms == 35.0
        assert metrics.avg_latency_ms == 35.0 / 3
    
    def test_cache_metrics_size_tracking(self):
        """Test cache size tracking"""
        metrics = CacheMetrics()
        
        # Record operations with different sizes
        metrics.record_hit("key1", 1.0, 100)
        metrics.record_hit("key2", 1.0, 200)
        metrics.record_set("key3", 1.0, 300)
        
        assert metrics.total_size_bytes == 600
        assert metrics.avg_entry_size_bytes == 600 / 3
    
    def test_cache_metrics_time_window(self):
        """Test time window-based metrics"""
        metrics = CacheMetrics(time_window_seconds=60)
        
        # Record operations at different times
        now = datetime.now()
        metrics.record_hit_at_time("key1", 1.0, 100, now - timedelta(seconds=30))
        metrics.record_hit_at_time("key2", 1.0, 100, now - timedelta(seconds=90))  # Outside window
        metrics.record_hit_at_time("key3", 1.0, 100, now)
        
        # Only recent operations should be counted
        recent_metrics = metrics.get_recent_metrics()
        assert recent_metrics.cache_hits == 2  # key1 and key3
    
    def test_cache_metrics_reset(self):
        """Test metrics reset functionality"""
        metrics = CacheMetrics()
        
        # Record some data
        metrics.record_hit("key1", 1.0, 100)
        metrics.record_miss("key2", 1.0)
        
        assert metrics.total_operations > 0
        
        # Reset metrics
        metrics.reset()
        
        assert metrics.total_operations == 0
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0


class TestCacheMetricsCollector:
    """Test cache metrics collector"""
    
    @pytest.fixture
    def cache_backend(self):
        """Cache backend fixture"""
        return MemoryCacheBackend(max_size=1000, default_ttl=3600)
    
    @pytest.fixture
    def cache_manager(self, cache_backend):
        """Cache manager fixture"""
        return CacheManager(default_backend=cache_backend)
    
    @pytest.fixture
    def metrics_collector(self, cache_manager):
        """Metrics collector fixture"""
        return CacheMetricsCollector(
            cache_manager=cache_manager,
            collection_interval=0.1,  # Fast collection for testing
            enable_detailed_metrics=True
        )
    
    def test_metrics_collector_initialization(self, cache_manager):
        """Test metrics collector initialization"""
        collector = CacheMetricsCollector(
            cache_manager=cache_manager,
            collection_interval=1.0,
            enable_detailed_metrics=True
        )
        
        assert collector.cache_manager == cache_manager
        assert collector.collection_interval == 1.0
        assert collector.enable_detailed_metrics is True
        assert collector.is_running is False
    
    def test_metrics_collector_operation_tracking(self, metrics_collector, cache_manager):
        """Test tracking of cache operations"""
        collector = metrics_collector
        
        # Start collecting metrics
        collector.start()
        
        try:
            # Perform cache operations
            cache_manager.set("test_key1", "value1")
            cache_manager.set("test_key2", "value2")
            cache_manager.get("test_key1")  # Hit
            cache_manager.get("nonexistent")  # Miss
            cache_manager.delete("test_key1")
            
            # Wait for metrics collection
            time.sleep(0.2)
            
            # Get collected metrics
            metrics = collector.get_current_metrics()
            
            assert metrics.total_operations >= 5
            assert metrics.cache_hits >= 1
            assert metrics.cache_misses >= 1
            
        finally:
            collector.stop()
    
    def test_metrics_collector_performance_tracking(self, metrics_collector, cache_manager):
        """Test performance metrics tracking"""
        collector = metrics_collector
        collector.start()
        
        try:
            # Perform operations with measurable latency
            large_data = "x" * 10000  # 10KB data
            
            start_time = time.time()
            for i in range(50):
                cache_manager.set(f"perf_key_{i}", large_data)
            
            for i in range(50):
                cache_manager.get(f"perf_key_{i}")
            
            duration = time.time() - start_time
            
            # Wait for metrics collection
            time.sleep(0.2)
            
            # Get performance metrics
            perf_metrics = collector.get_performance_metrics()
            
            assert "avg_set_latency_ms" in perf_metrics
            assert "avg_get_latency_ms" in perf_metrics
            assert "operations_per_second" in perf_metrics
            assert perf_metrics["operations_per_second"] > 0
            
        finally:
            collector.stop()
    
    def test_metrics_collector_memory_tracking(self, metrics_collector, cache_manager):
        """Test memory usage tracking"""
        collector = metrics_collector
        collector.start()
        
        try:
            # Add data with known sizes
            data_sizes = [1000, 2000, 3000, 4000, 5000]  # Bytes
            for i, size in enumerate(data_sizes):
                data = "x" * size
                cache_manager.set(f"memory_key_{i}", data)
            
            # Wait for metrics collection
            time.sleep(0.2)
            
            # Get memory metrics
            memory_metrics = collector.get_memory_metrics()
            
            assert "total_memory_bytes" in memory_metrics
            assert "cache_entry_count" in memory_metrics
            assert "avg_entry_size_bytes" in memory_metrics
            assert memory_metrics["cache_entry_count"] >= 5
            assert memory_metrics["total_memory_bytes"] > 0
            
        finally:
            collector.stop()
    
    def test_metrics_collector_backend_specific_metrics(self, cache_manager):
        """Test backend-specific metrics collection"""
        # Mock Redis backend for testing
        from fakeredis import FakeRedis
        redis_client = FakeRedis(decode_responses=False)
        redis_backend = RedisCacheBackend(client=redis_client)
        
        cache_manager.register_backend("redis", redis_backend)
        
        collector = CacheMetricsCollector(
            cache_manager=cache_manager,
            enable_backend_metrics=True
        )
        
        collector.start()
        
        try:
            # Perform operations on different backends
            cache_manager.set("memory_key", "value", backend="memory")
            cache_manager.set("redis_key", "value", backend="redis")
            
            cache_manager.get("memory_key", backend="memory")
            cache_manager.get("redis_key", backend="redis")
            
            # Wait for metrics collection
            time.sleep(0.2)
            
            # Get backend-specific metrics
            backend_metrics = collector.get_backend_metrics()
            
            assert "memory" in backend_metrics
            assert "redis" in backend_metrics
            
            for backend_name, metrics in backend_metrics.items():
                assert "operations" in metrics
                assert "hits" in metrics
                assert "misses" in metrics
                
        finally:
            collector.stop()
    
    def test_metrics_collector_key_pattern_analysis(self, metrics_collector, cache_manager):
        """Test key pattern analysis"""
        collector = metrics_collector
        collector.start()
        
        try:
            # Set keys with different patterns
            patterns = {
                "user:": 10,
                "track:": 15,
                "playlist:": 5,
                "session:": 8
            }
            
            for pattern, count in patterns.items():
                for i in range(count):
                    key = f"{pattern}{i}"
                    cache_manager.set(key, f"value_{i}")
                    cache_manager.get(key)  # Generate access
            
            # Wait for metrics collection
            time.sleep(0.2)
            
            # Get key pattern analysis
            pattern_metrics = collector.get_key_pattern_metrics()
            
            assert len(pattern_metrics) >= 4
            
            for pattern, stats in pattern_metrics.items():
                assert "access_count" in stats
                assert "hit_rate" in stats
                assert stats["access_count"] > 0
                
        finally:
            collector.stop()


class TestCachePerformanceAnalyzer:
    """Test cache performance analyzer"""
    
    @pytest.fixture
    def performance_analyzer(self):
        """Performance analyzer fixture"""
        return CachePerformanceAnalyzer(
            analysis_window_minutes=5,
            enable_trend_analysis=True,
            enable_anomaly_detection=True
        )
    
    def test_performance_analyzer_latency_analysis(self, performance_analyzer):
        """Test latency analysis"""
        analyzer = performance_analyzer
        
        # Generate sample latency data
        latencies = [1.0, 1.2, 1.1, 1.3, 1.0, 10.0, 1.1, 1.2]  # One outlier
        
        for i, latency in enumerate(latencies):
            metrics_data = MockMetricsData(
                timestamp=datetime.now() - timedelta(seconds=i),
                operation_type="get",
                key=f"key_{i}",
                hit=True,
                latency_ms=latency,
                size_bytes=100
            )
            analyzer.add_metrics_data(metrics_data)
        
        # Analyze latency
        latency_analysis = analyzer.analyze_latency()
        
        assert "avg_latency_ms" in latency_analysis
        assert "p95_latency_ms" in latency_analysis
        assert "p99_latency_ms" in latency_analysis
        assert "outliers" in latency_analysis
        
        # Should detect the 10.0ms outlier
        assert len(latency_analysis["outliers"]) >= 1
    
    def test_performance_analyzer_hit_rate_trends(self, performance_analyzer):
        """Test hit rate trend analysis"""
        analyzer = performance_analyzer
        
        # Generate trending hit rate data (declining over time)
        for i in range(100):
            hit_probability = 0.9 - (i * 0.005)  # Declining from 90% to 40%
            is_hit = i % 100 < (hit_probability * 100)
            
            metrics_data = MockMetricsData(
                timestamp=datetime.now() - timedelta(minutes=i),
                operation_type="get",
                key=f"key_{i}",
                hit=is_hit,
                latency_ms=1.0,
                size_bytes=100
            )
            analyzer.add_metrics_data(metrics_data)
        
        # Analyze hit rate trends
        trend_analysis = analyzer.analyze_hit_rate_trends()
        
        assert "trend_direction" in trend_analysis
        assert "trend_strength" in trend_analysis
        assert "current_hit_rate" in trend_analysis
        assert "projected_hit_rate" in trend_analysis
        
        # Should detect declining trend
        assert trend_analysis["trend_direction"] == "declining"
    
    def test_performance_analyzer_throughput_analysis(self, performance_analyzer):
        """Test throughput analysis"""
        analyzer = performance_analyzer
        
        # Generate variable throughput data
        current_time = datetime.now()
        
        # High throughput period
        for i in range(100):
            metrics_data = MockMetricsData(
                timestamp=current_time - timedelta(seconds=i),
                operation_type="get",
                key=f"high_key_{i}",
                hit=True,
                latency_ms=1.0,
                size_bytes=100
            )
            analyzer.add_metrics_data(metrics_data)
        
        # Low throughput period
        for i in range(20):
            metrics_data = MockMetricsData(
                timestamp=current_time - timedelta(seconds=i + 200),
                operation_type="get",
                key=f"low_key_{i}",
                hit=True,
                latency_ms=1.0,
                size_bytes=100
            )
            analyzer.add_metrics_data(metrics_data)
        
        # Analyze throughput
        throughput_analysis = analyzer.analyze_throughput()
        
        assert "current_ops_per_second" in throughput_analysis
        assert "peak_ops_per_second" in throughput_analysis
        assert "avg_ops_per_second" in throughput_analysis
        assert "throughput_variance" in throughput_analysis
    
    def test_performance_analyzer_anomaly_detection(self, performance_analyzer):
        """Test anomaly detection"""
        analyzer = performance_analyzer
        
        # Generate normal data
        for i in range(50):
            metrics_data = MockMetricsData(
                timestamp=datetime.now() - timedelta(seconds=i),
                operation_type="get",
                key=f"normal_key_{i}",
                hit=True,
                latency_ms=1.0 + (i % 3) * 0.1,  # Small variance
                size_bytes=100
            )
            analyzer.add_metrics_data(metrics_data)
        
        # Add anomalous data
        anomaly_data = MockMetricsData(
            timestamp=datetime.now(),
            operation_type="get",
            key="anomaly_key",
            hit=True,
            latency_ms=50.0,  # Very high latency
            size_bytes=100
        )
        analyzer.add_metrics_data(anomaly_data)
        
        # Detect anomalies
        anomalies = analyzer.detect_anomalies()
        
        assert len(anomalies) >= 1
        assert anomalies[0]["type"] == "latency_spike"
        assert anomalies[0]["severity"] in ["low", "medium", "high"]
    
    def test_performance_analyzer_capacity_prediction(self, performance_analyzer):
        """Test capacity prediction"""
        analyzer = performance_analyzer
        
        # Generate growing load data
        for i in range(100):
            # Increasing operation count over time
            ops_count = 10 + i  # Growing from 10 to 110 ops
            
            for j in range(ops_count):
                metrics_data = MockMetricsData(
                    timestamp=datetime.now() - timedelta(hours=i),
                    operation_type="get",
                    key=f"capacity_key_{i}_{j}",
                    hit=True,
                    latency_ms=1.0 + (ops_count * 0.01),  # Latency increases with load
                    size_bytes=100
                )
                analyzer.add_metrics_data(metrics_data)
        
        # Predict capacity
        capacity_prediction = analyzer.predict_capacity_needs()
        
        assert "projected_peak_load" in capacity_prediction
        assert "time_to_capacity_limit" in capacity_prediction
        assert "recommended_scaling_factor" in capacity_prediction
        assert "confidence_level" in capacity_prediction


class TestCacheHealthMonitor:
    """Test cache health monitoring"""
    
    @pytest.fixture
    def cache_backend(self):
        """Cache backend fixture"""
        return MemoryCacheBackend(max_size=1000, default_ttl=3600)
    
    @pytest.fixture
    def cache_manager(self, cache_backend):
        """Cache manager fixture"""
        return CacheManager(default_backend=cache_backend)
    
    @pytest.fixture
    def health_monitor(self, cache_manager):
        """Health monitor fixture"""
        return CacheHealthMonitor(
            cache_manager=cache_manager,
            check_interval_seconds=0.1,  # Fast checks for testing
            health_thresholds={
                "hit_rate_threshold": 0.8,
                "latency_threshold_ms": 10.0,
                "error_rate_threshold": 0.05
            }
        )
    
    def test_health_monitor_initialization(self, cache_manager):
        """Test health monitor initialization"""
        monitor = CacheHealthMonitor(
            cache_manager=cache_manager,
            check_interval_seconds=1.0,
            health_thresholds={"hit_rate_threshold": 0.9}
        )
        
        assert monitor.cache_manager == cache_manager
        assert monitor.check_interval_seconds == 1.0
        assert monitor.health_thresholds["hit_rate_threshold"] == 0.9
    
    def test_health_monitor_basic_checks(self, health_monitor, cache_manager):
        """Test basic health checks"""
        monitor = health_monitor
        
        # Set up cache with good performance
        for i in range(20):
            cache_manager.set(f"health_key_{i}", f"value_{i}")
        
        # Generate hits (good hit rate)
        for i in range(18):
            cache_manager.get(f"health_key_{i}")
        
        # Generate few misses
        cache_manager.get("nonexistent_1")
        cache_manager.get("nonexistent_2")
        
        # Run health check
        health_status = monitor.check_health()
        
        assert "overall_status" in health_status
        assert "hit_rate_status" in health_status
        assert "latency_status" in health_status
        assert "error_rate_status" in health_status
        assert "backend_status" in health_status
        
        # Should be healthy with good hit rate
        assert health_status["overall_status"] in ["healthy", "warning"]
    
    def test_health_monitor_unhealthy_detection(self, health_monitor, cache_manager):
        """Test detection of unhealthy conditions"""
        monitor = health_monitor
        
        # Create unhealthy condition (low hit rate)
        for i in range(10):
            cache_manager.set(f"unhealthy_key_{i}", f"value_{i}")
        
        # Generate mostly misses (poor hit rate)
        for i in range(2):
            cache_manager.get(f"unhealthy_key_{i}")  # 2 hits
        
        for i in range(20):
            cache_manager.get(f"nonexistent_{i}")  # 20 misses
        
        # Run health check
        health_status = monitor.check_health()
        
        # Should detect poor hit rate
        assert health_status["hit_rate_status"] in ["warning", "critical"]
        assert health_status["overall_status"] in ["warning", "critical"]
    
    def test_health_monitor_continuous_monitoring(self, health_monitor, cache_manager):
        """Test continuous health monitoring"""
        monitor = health_monitor
        
        health_reports = []
        
        def health_callback(status):
            health_reports.append(status)
        
        # Start monitoring
        monitor.start_monitoring(callback=health_callback)
        
        try:
            # Perform operations over time
            for i in range(5):
                cache_manager.set(f"monitor_key_{i}", f"value_{i}")
                cache_manager.get(f"monitor_key_{i}")
                time.sleep(0.05)
            
            # Wait for health reports
            time.sleep(0.3)
            
            # Check that health reports were generated
            assert len(health_reports) >= 1
            
            for report in health_reports:
                assert "overall_status" in report
                assert "timestamp" in report
                
        finally:
            monitor.stop_monitoring()
    
    def test_health_monitor_trend_analysis(self, health_monitor, cache_manager):
        """Test health trend analysis"""
        monitor = health_monitor
        
        # Generate declining performance over time
        for batch in range(5):
            hit_rate = 0.9 - (batch * 0.15)  # Declining hit rate
            
            # Set some keys
            for i in range(10):
                cache_manager.set(f"trend_key_{batch}_{i}", f"value_{i}")
            
            # Generate hits based on hit rate
            for i in range(10):
                if i < (hit_rate * 10):
                    cache_manager.get(f"trend_key_{batch}_{i}")  # Hit
                else:
                    cache_manager.get(f"nonexistent_{batch}_{i}")  # Miss
            
            # Record health check
            monitor.check_health()
            time.sleep(0.05)
        
        # Analyze trends
        trend_analysis = monitor.analyze_health_trends()
        
        assert "hit_rate_trend" in trend_analysis
        assert "latency_trend" in trend_analysis
        assert "overall_trend" in trend_analysis
        
        # Should detect declining hit rate
        assert trend_analysis["hit_rate_trend"]["direction"] in ["declining", "stable"]


class TestCacheAlertManager:
    """Test cache alert management"""
    
    @pytest.fixture
    def alert_manager(self):
        """Alert manager fixture"""
        return CacheAlertManager(
            alert_thresholds={
                "critical_hit_rate": 0.5,
                "warning_hit_rate": 0.7,
                "critical_latency_ms": 100.0,
                "warning_latency_ms": 50.0,
                "critical_error_rate": 0.1,
                "warning_error_rate": 0.05
            },
            cooldown_minutes=1,
            enable_escalation=True
        )
    
    def test_alert_manager_threshold_alerts(self, alert_manager):
        """Test threshold-based alerts"""
        manager = alert_manager
        
        # Mock metrics that trigger alerts
        critical_metrics = {
            "hit_rate": 0.3,  # Below critical threshold
            "avg_latency_ms": 150.0,  # Above critical threshold
            "error_rate": 0.15  # Above critical threshold
        }
        
        alerts = manager.check_thresholds(critical_metrics)
        
        assert len(alerts) >= 3
        
        # Check for critical alerts
        critical_alerts = [a for a in alerts if a["severity"] == "critical"]
        assert len(critical_alerts) >= 3
        
        alert_types = [a["type"] for a in alerts]
        assert "low_hit_rate" in alert_types
        assert "high_latency" in alert_types
        assert "high_error_rate" in alert_types
    
    def test_alert_manager_alert_cooldown(self, alert_manager):
        """Test alert cooldown mechanism"""
        manager = alert_manager
        
        # Trigger same alert twice
        metrics = {"hit_rate": 0.3}
        
        alerts1 = manager.check_thresholds(metrics)
        alerts2 = manager.check_thresholds(metrics)  # Immediate retry
        
        # First should generate alert, second should be suppressed
        assert len(alerts1) >= 1
        assert len(alerts2) == 0  # Suppressed by cooldown
    
    def test_alert_manager_escalation(self, alert_manager):
        """Test alert escalation"""
        manager = alert_manager
        
        # Trigger warning alert multiple times
        warning_metrics = {"hit_rate": 0.6}  # Warning level
        
        for i in range(5):
            alerts = manager.check_thresholds(warning_metrics)
            time.sleep(0.1)
        
        # Check escalation history
        escalation_info = manager.get_escalation_info("low_hit_rate")
        
        assert escalation_info["escalation_count"] > 0
        assert escalation_info["current_severity"] in ["warning", "critical"]
    
    def test_alert_manager_custom_handlers(self, alert_manager):
        """Test custom alert handlers"""
        manager = alert_manager
        
        # Custom alert handler
        handled_alerts = []
        
        def custom_handler(alert):
            handled_alerts.append(alert)
        
        # Register handler
        manager.register_alert_handler("low_hit_rate", custom_handler)
        
        # Trigger alert
        metrics = {"hit_rate": 0.3}
        manager.check_thresholds(metrics)
        
        # Check that custom handler was called
        assert len(handled_alerts) >= 1
        assert handled_alerts[0]["type"] == "low_hit_rate"
    
    def test_alert_manager_notification_integration(self, alert_manager):
        """Test notification system integration"""
        manager = alert_manager
        
        # Mock notification system
        notifications = []
        
        def mock_notify(alert):
            notifications.append({
                "type": alert["type"],
                "severity": alert["severity"],
                "message": alert["message"]
            })
        
        manager.set_notification_handler(mock_notify)
        
        # Trigger critical alert
        metrics = {"hit_rate": 0.2, "avg_latency_ms": 200.0}
        manager.check_thresholds(metrics)
        
        # Check notifications
        assert len(notifications) >= 2
        
        severities = [n["severity"] for n in notifications]
        assert "critical" in severities


class TestMetricsAggregator:
    """Test metrics aggregation"""
    
    @pytest.fixture
    def metrics_aggregator(self):
        """Metrics aggregator fixture"""
        return MetricsAggregator(
            aggregation_intervals=[60, 300, 3600],  # 1min, 5min, 1hour
            retention_days=7,
            enable_percentiles=True
        )
    
    def test_metrics_aggregator_basic_aggregation(self, metrics_aggregator):
        """Test basic metrics aggregation"""
        aggregator = metrics_aggregator
        
        # Add sample metrics data
        base_time = datetime.now()
        
        for i in range(100):
            timestamp = base_time - timedelta(seconds=i)
            metrics_data = MockMetricsData(
                timestamp=timestamp,
                operation_type="get",
                key=f"agg_key_{i}",
                hit=i % 3 != 0,  # 2/3 hit rate
                latency_ms=1.0 + (i % 10) * 0.1,
                size_bytes=100 + (i % 50)
            )
            aggregator.add_metrics_data(metrics_data)
        
        # Get aggregated metrics
        aggregated = aggregator.get_aggregated_metrics(interval_seconds=60)
        
        assert "total_operations" in aggregated
        assert "hit_rate" in aggregated
        assert "avg_latency_ms" in aggregated
        assert "p95_latency_ms" in aggregated
        assert "total_size_bytes" in aggregated
        
        # Check hit rate calculation
        assert abs(aggregated["hit_rate"] - (2/3)) < 0.1
    
    def test_metrics_aggregator_time_series(self, metrics_aggregator):
        """Test time series aggregation"""
        aggregator = metrics_aggregator
        
        # Generate time series data
        base_time = datetime.now()
        
        for hour in range(6):
            for minute in range(60):
                timestamp = base_time - timedelta(hours=hour, minutes=minute)
                
                # Vary hit rate by hour
                hit_rate = 0.8 + (hour % 3) * 0.1
                is_hit = minute < (hit_rate * 60)
                
                metrics_data = MockMetricsData(
                    timestamp=timestamp,
                    operation_type="get",
                    key=f"ts_key_{hour}_{minute}",
                    hit=is_hit,
                    latency_ms=1.0,
                    size_bytes=100
                )
                aggregator.add_metrics_data(metrics_data)
        
        # Get time series
        time_series = aggregator.get_time_series(
            start_time=base_time - timedelta(hours=6),
            end_time=base_time,
            interval_seconds=3600  # 1-hour intervals
        )
        
        assert len(time_series) >= 6
        
        for point in time_series:
            assert "timestamp" in point
            assert "hit_rate" in point
            assert "operation_count" in point
    
    def test_metrics_aggregator_percentile_calculation(self, metrics_aggregator):
        """Test percentile calculation"""
        aggregator = metrics_aggregator
        
        # Add data with known distribution
        latencies = [1.0] * 90 + [10.0] * 9 + [100.0] * 1  # 90%, 9%, 1%
        
        for i, latency in enumerate(latencies):
            metrics_data = MockMetricsData(
                timestamp=datetime.now() - timedelta(seconds=i),
                operation_type="get",
                key=f"perc_key_{i}",
                hit=True,
                latency_ms=latency,
                size_bytes=100
            )
            aggregator.add_metrics_data(metrics_data)
        
        # Get percentiles
        percentiles = aggregator.get_percentiles(["p50", "p90", "p95", "p99"])
        
        assert percentiles["p50"] <= 1.5  # Median should be ~1.0
        assert percentiles["p90"] >= 1.0  # 90th percentile
        assert percentiles["p95"] >= 10.0  # 95th percentile should be ~10.0
        assert percentiles["p99"] >= 50.0  # 99th percentile should be high


class TestPerformanceBenchmark:
    """Test cache performance benchmarking"""
    
    @pytest.fixture
    def cache_backend(self):
        """Cache backend fixture"""
        return MemoryCacheBackend(max_size=10000, default_ttl=3600)
    
    @pytest.fixture
    def cache_manager(self, cache_backend):
        """Cache manager fixture"""
        return CacheManager(default_backend=cache_backend)
    
    @pytest.fixture
    def benchmark(self, cache_manager):
        """Performance benchmark fixture"""
        return PerformanceBenchmark(cache_manager)
    
    def test_benchmark_set_operations(self, benchmark):
        """Test benchmarking of set operations"""
        bench = benchmark
        
        # Benchmark set operations
        results = bench.benchmark_set_operations(
            num_operations=1000,
            data_size_bytes=100,
            concurrent_threads=1
        )
        
        assert "total_time_seconds" in results
        assert "operations_per_second" in results
        assert "avg_latency_ms" in results
        assert "p95_latency_ms" in results
        assert "success_rate" in results
        
        assert results["operations_per_second"] > 0
        assert results["success_rate"] >= 0.95  # Should be very reliable
    
    def test_benchmark_get_operations(self, benchmark, cache_manager):
        """Test benchmarking of get operations"""
        bench = benchmark
        
        # Pre-populate cache
        for i in range(1000):
            cache_manager.set(f"bench_key_{i}", f"value_{i}")
        
        # Benchmark get operations
        results = bench.benchmark_get_operations(
            num_operations=1000,
            hit_rate=0.8,  # 80% hits
            concurrent_threads=1
        )
        
        assert "total_time_seconds" in results
        assert "operations_per_second" in results
        assert "hit_rate_achieved" in results
        assert "avg_hit_latency_ms" in results
        assert "avg_miss_latency_ms" in results
        
        # Hit rate should be close to target
        assert abs(results["hit_rate_achieved"] - 0.8) < 0.1
    
    def test_benchmark_mixed_workload(self, benchmark):
        """Test benchmarking of mixed workloads"""
        bench = benchmark
        
        # Define workload pattern
        workload = {
            "set_ratio": 0.2,
            "get_ratio": 0.7,
            "delete_ratio": 0.1
        }
        
        results = bench.benchmark_mixed_workload(
            num_operations=1000,
            workload_pattern=workload,
            concurrent_threads=2
        )
        
        assert "total_time_seconds" in results
        assert "operations_per_second" in results
        assert "operation_breakdown" in results
        
        breakdown = results["operation_breakdown"]
        assert "set" in breakdown
        assert "get" in breakdown
        assert "delete" in breakdown
        
        # Check that ratios are approximately correct
        total_ops = sum(breakdown.values())
        set_ratio = breakdown["set"] / total_ops
        assert abs(set_ratio - 0.2) < 0.1
    
    def test_benchmark_concurrent_access(self, benchmark):
        """Test benchmarking of concurrent access"""
        bench = benchmark
        
        # Benchmark with multiple threads
        results = bench.benchmark_concurrent_access(
            num_threads=4,
            operations_per_thread=250,
            data_size_bytes=200
        )
        
        assert "total_time_seconds" in results
        assert "operations_per_second" in results
        assert "thread_results" in results
        assert "concurrency_efficiency" in results
        
        # Check thread results
        thread_results = results["thread_results"]
        assert len(thread_results) == 4
        
        for thread_result in thread_results:
            assert "thread_id" in thread_result
            assert "operations_completed" in thread_result
            assert "success_rate" in thread_result
    
    def test_benchmark_memory_pressure(self, benchmark):
        """Test benchmarking under memory pressure"""
        bench = benchmark
        
        # Test with data that will fill cache to capacity
        results = bench.benchmark_memory_pressure(
            cache_size_entries=5000,  # Half of cache capacity
            data_size_bytes=1000,
            access_pattern="sequential"
        )
        
        assert "eviction_rate" in results
        assert "memory_efficiency" in results
        assert "performance_degradation" in results
        
        # Should have some evictions
        assert results["eviction_rate"] >= 0


@pytest.mark.performance
class TestCacheMetricsPerformance:
    """Performance tests for cache metrics system"""
    
    def test_metrics_collection_overhead(self):
        """Test overhead of metrics collection"""
        backend = MemoryCacheBackend(max_size=10000)
        cache_manager = CacheManager(default_backend=backend)
        
        # Measure without metrics
        start_time = time.time()
        for i in range(1000):
            cache_manager.set(f"no_metrics_key_{i}", f"value_{i}")
            cache_manager.get(f"no_metrics_key_{i}")
        no_metrics_duration = time.time() - start_time
        
        # Enable metrics collection
        metrics_collector = CacheMetricsCollector(cache_manager)
        metrics_collector.start()
        
        try:
            # Measure with metrics
            start_time = time.time()
            for i in range(1000):
                cache_manager.set(f"with_metrics_key_{i}", f"value_{i}")
                cache_manager.get(f"with_metrics_key_{i}")
            with_metrics_duration = time.time() - start_time
            
        finally:
            metrics_collector.stop()
        
        # Metrics overhead should be minimal
        overhead_ratio = with_metrics_duration / no_metrics_duration
        assert overhead_ratio < 2.0, f"Metrics overhead too high: {overhead_ratio}x"
    
    def test_metrics_memory_usage(self):
        """Test memory usage of metrics collection"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        backend = MemoryCacheBackend(max_size=10000)
        cache_manager = CacheManager(default_backend=backend)
        metrics_collector = CacheMetricsCollector(cache_manager)
        
        metrics_collector.start()
        
        try:
            # Generate lots of metrics data
            for i in range(10000):
                cache_manager.set(f"memory_test_key_{i}", f"value_{i}")
                cache_manager.get(f"memory_test_key_{i}")
            
            # Wait for metrics processing
            time.sleep(1)
            
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable
            assert memory_increase < 100 * 1024 * 1024  # Less than 100MB
            
        finally:
            metrics_collector.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not performance"])
