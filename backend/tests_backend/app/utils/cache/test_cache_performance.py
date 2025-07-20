"""
Tests for cache performance and benchmarking in Spotify AI Agent

Comprehensive testing suite for cache performance analysis, load testing,
benchmarking, optimization and capacity planning.

Developed by Expert Team:
- Lead Dev + AI Architect
- Senior Backend Developer (Python/FastAPI/Django)
- Machine Learning Engineer (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Backend Security Specialist
- Microservices Architect
"""

import pytest
import asyncio
import time
import psutil
import threading
import statistics
from unittest.mock import Mock, patch
from typing import Any, Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from app.utils.cache.performance import (
    CachePerformanceBenchmark, CacheLoadTester, CacheProfiler,
    PerformanceMetricsCollector, CacheCapacityPlanner, CacheOptimizer,
    LatencyAnalyzer, ThroughputAnalyzer, MemoryAnalyzer
)
from app.utils.cache.backends import RedisCacheBackend, MemoryCacheBackend
from app.utils.cache.manager import CacheManager


@dataclass
class BenchmarkResult:
    """Benchmark result data structure"""
    operation: str
    duration: float
    throughput: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    memory_usage: int
    cpu_usage: float
    error_rate: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadTestConfig:
    """Load test configuration"""
    concurrent_users: int = 100
    duration_seconds: int = 60
    ramp_up_seconds: int = 10
    operations_per_second: int = 1000
    data_size_bytes: int = 1024
    cache_hit_ratio: float = 0.8
    read_write_ratio: float = 0.7  # 70% reads, 30% writes


class TestCachePerformanceBenchmark:
    """Test cache performance benchmarking"""
    
    @pytest.fixture
    def benchmark(self):
        """Benchmark fixture"""
        return CachePerformanceBenchmark(
            warmup_iterations=100,
            test_iterations=1000,
            collect_detailed_metrics=True,
            enable_profiling=True
        )
    
    @pytest.fixture
    def redis_backend(self):
        """Redis backend fixture for testing"""
        from fakeredis import FakeRedis
        redis_client = FakeRedis()
        return RedisCacheBackend(
            redis_client=redis_client,
            default_ttl=3600
        )
    
    @pytest.fixture
    def memory_backend(self):
        """Memory backend fixture"""
        return MemoryCacheBackend(
            max_size=10000,
            default_ttl=3600
        )
    
    def test_basic_operation_benchmark(self, benchmark, memory_backend):
        """Test basic cache operation benchmarking"""
        bench = benchmark
        cache = memory_backend
        
        # Benchmark set operations
        set_result = bench.benchmark_operation(
            operation=lambda: cache.set("test_key", "test_value"),
            operation_name="cache_set",
            iterations=1000
        )
        
        assert set_result.operation == "cache_set"
        assert set_result.duration > 0
        assert set_result.throughput > 0
        assert set_result.latency_p50 >= 0
        
        # Benchmark get operations
        get_result = bench.benchmark_operation(
            operation=lambda: cache.get("test_key"),
            operation_name="cache_get",
            iterations=1000
        )
        
        assert get_result.operation == "cache_get"
        assert get_result.duration > 0
        assert get_result.throughput > 0
    
    def test_async_operation_benchmark(self, benchmark):
        """Test async operation benchmarking"""
        bench = benchmark
        
        async def async_operation():
            await asyncio.sleep(0.001)  # Simulate async work
            return "result"
        
        result = bench.benchmark_async_operation(
            async_operation,
            operation_name="async_test",
            iterations=100
        )
        
        assert result.operation == "async_test"
        assert result.duration > 0
        assert result.throughput > 0
    
    def test_concurrent_benchmark(self, benchmark, memory_backend):
        """Test concurrent operation benchmarking"""
        bench = benchmark
        cache = memory_backend
        
        def concurrent_operation(thread_id: int):
            for i in range(10):
                cache.set(f"key_{thread_id}_{i}", f"value_{thread_id}_{i}")
                cache.get(f"key_{thread_id}_{i}")
        
        result = bench.benchmark_concurrent_operations(
            operation=concurrent_operation,
            operation_name="concurrent_cache_ops",
            num_threads=10,
            operations_per_thread=10
        )
        
        assert result.operation == "concurrent_cache_ops"
        assert result.throughput > 0
        assert "concurrency_level" in result.metadata
    
    def test_memory_usage_benchmark(self, benchmark, memory_backend):
        """Test memory usage benchmarking"""
        bench = benchmark
        cache = memory_backend
        
        # Benchmark memory usage during operations
        def memory_intensive_operation():
            large_data = "x" * 10000  # 10KB data
            for i in range(100):
                cache.set(f"memory_test_{i}", large_data)
        
        result = bench.benchmark_with_memory_tracking(
            operation=memory_intensive_operation,
            operation_name="memory_intensive_cache"
        )
        
        assert result.memory_usage > 0
        assert "memory_delta" in result.metadata
        assert "peak_memory" in result.metadata
    
    def test_latency_distribution_analysis(self, benchmark, memory_backend):
        """Test latency distribution analysis"""
        bench = benchmark
        cache = memory_backend
        
        # Generate operations with varying latencies
        def variable_latency_operation():
            import random
            # Simulate variable processing time
            time.sleep(random.uniform(0.001, 0.01))
            cache.set("latency_test", "value")
            return cache.get("latency_test")
        
        result = bench.analyze_latency_distribution(
            operation=variable_latency_operation,
            operation_name="variable_latency",
            samples=200
        )
        
        assert result.latency_p50 > 0
        assert result.latency_p95 > result.latency_p50
        assert result.latency_p99 > result.latency_p95
        assert "latency_distribution" in result.metadata
    
    def test_throughput_scaling_benchmark(self, benchmark, memory_backend):
        """Test throughput scaling analysis"""
        bench = benchmark
        cache = memory_backend
        
        # Test throughput at different concurrency levels
        def throughput_operation():
            cache.set("throughput_test", "value")
            return cache.get("throughput_test")
        
        scaling_results = bench.benchmark_throughput_scaling(
            operation=throughput_operation,
            operation_name="throughput_scaling",
            concurrency_levels=[1, 2, 4, 8, 16],
            duration_seconds=5
        )
        
        assert len(scaling_results) == 5
        assert all(result.throughput > 0 for result in scaling_results)
        
        # Check that results include concurrency metadata
        for i, result in enumerate(scaling_results):
            expected_concurrency = [1, 2, 4, 8, 16][i]
            assert result.metadata["concurrency_level"] == expected_concurrency


class TestCacheLoadTester:
    """Test cache load testing"""
    
    @pytest.fixture
    def load_tester(self):
        """Load tester fixture"""
        return CacheLoadTester(
            enable_metrics_collection=True,
            enable_real_time_monitoring=True,
            failure_threshold=0.05  # 5% failure rate threshold
        )
    
    @pytest.fixture
    def test_config(self):
        """Load test configuration fixture"""
        return LoadTestConfig(
            concurrent_users=50,
            duration_seconds=30,
            ramp_up_seconds=5,
            operations_per_second=500
        )
    
    def test_basic_load_test(self, load_tester, memory_backend, test_config):
        """Test basic load testing"""
        tester = load_tester
        cache = memory_backend
        config = test_config
        
        # Define test scenarios
        def read_operation():
            return cache.get(f"key_{hash(time.time()) % 1000}")
        
        def write_operation():
            key = f"key_{hash(time.time()) % 1000}"
            return cache.set(key, f"value_{time.time()}")
        
        # Run load test
        results = tester.run_load_test(
            read_operation=read_operation,
            write_operation=write_operation,
            config=config
        )
        
        assert results["total_operations"] > 0
        assert results["duration"] <= config.duration_seconds + 5  # Allow some overhead
        assert results["average_throughput"] > 0
        assert results["error_rate"] <= 1.0
    
    def test_mixed_workload_test(self, load_tester, memory_backend):
        """Test mixed workload load testing"""
        tester = load_tester
        cache = memory_backend
        
        # Define different operation types
        operations = {
            "read_small": (lambda: cache.get("small_key"), 0.4),  # 40% weight
            "read_large": (lambda: cache.get("large_key"), 0.2),  # 20% weight
            "write_small": (lambda: cache.set("small_key", "small_value"), 0.3),  # 30% weight
            "write_large": (lambda: cache.set("large_key", "x" * 10000), 0.1)  # 10% weight
        }
        
        config = LoadTestConfig(
            concurrent_users=20,
            duration_seconds=15,
            operations_per_second=200
        )
        
        results = tester.run_mixed_workload_test(operations, config)
        
        assert "operation_breakdown" in results
        assert len(results["operation_breakdown"]) == 4
        assert all(op in results["operation_breakdown"] for op in operations.keys())
    
    def test_stress_test(self, load_tester, memory_backend):
        """Test stress testing to find breaking points"""
        tester = load_tester
        cache = memory_backend
        
        def stress_operation():
            key = f"stress_key_{hash(time.time()) % 100}"
            cache.set(key, "x" * 1000)  # 1KB data
            return cache.get(key)
        
        # Gradually increase load until failure
        stress_results = tester.run_stress_test(
            operation=stress_operation,
            initial_load=100,
            max_load=2000,
            load_increment=200,
            test_duration=10
        )
        
        assert "breaking_point" in stress_results
        assert "max_stable_load" in stress_results
        assert stress_results["breaking_point"] > 0
    
    def test_endurance_test(self, load_tester, memory_backend):
        """Test endurance/soak testing"""
        tester = load_tester
        cache = memory_backend
        
        def endurance_operation():
            import random
            key = f"endurance_key_{random.randint(1, 1000)}"
            cache.set(key, f"value_{time.time()}")
            return cache.get(key)
        
        config = LoadTestConfig(
            concurrent_users=10,
            duration_seconds=60,  # 1 minute endurance test
            operations_per_second=100
        )
        
        results = tester.run_endurance_test(endurance_operation, config)
        
        assert "memory_trend" in results
        assert "performance_degradation" in results
        assert "stability_score" in results
        assert results["stability_score"] >= 0
    
    def test_load_test_with_monitoring(self, load_tester, memory_backend):
        """Test load testing with real-time monitoring"""
        tester = load_tester
        cache = memory_backend
        
        # Enable monitoring
        monitoring_data = []
        
        def monitor_callback(metrics):
            monitoring_data.append(metrics)
        
        tester.set_monitoring_callback(monitor_callback)
        
        def monitored_operation():
            cache.set("monitor_key", "monitor_value")
            return cache.get("monitor_key")
        
        config = LoadTestConfig(
            concurrent_users=5,
            duration_seconds=20,
            operations_per_second=50
        )
        
        results = tester.run_load_test_with_monitoring(
            monitored_operation, 
            config,
            monitoring_interval=2  # Monitor every 2 seconds
        )
        
        assert len(monitoring_data) > 0
        assert "real_time_metrics" in results
        assert all("timestamp" in metric for metric in monitoring_data)
    
    def test_distributed_load_test(self, load_tester):
        """Test distributed load testing simulation"""
        tester = load_tester
        
        # Simulate multiple cache backends
        backends = [
            MemoryCacheBackend(max_size=1000),
            MemoryCacheBackend(max_size=1000),
            MemoryCacheBackend(max_size=1000)
        ]
        
        def distributed_operation(backend_id):
            backend = backends[backend_id % len(backends)]
            key = f"dist_key_{backend_id}_{time.time()}"
            backend.set(key, f"value_{backend_id}")
            return backend.get(key)
        
        config = LoadTestConfig(
            concurrent_users=30,  # 10 users per backend
            duration_seconds=20,
            operations_per_second=150
        )
        
        results = tester.run_distributed_load_test(
            distributed_operation,
            num_backends=len(backends),
            config=config
        )
        
        assert "backend_performance" in results
        assert len(results["backend_performance"]) == len(backends)
        assert "load_distribution" in results


class TestCacheProfiler:
    """Test cache profiling"""
    
    @pytest.fixture
    def profiler(self):
        """Profiler fixture"""
        return CacheProfiler(
            enable_cpu_profiling=True,
            enable_memory_profiling=True,
            enable_io_profiling=True,
            sampling_interval=0.01
        )
    
    def test_cpu_profiling(self, profiler, memory_backend):
        """Test CPU profiling of cache operations"""
        prof = profiler
        cache = memory_backend
        
        def cpu_intensive_operation():
            # Simulate CPU-intensive cache operations
            for i in range(1000):
                key = f"cpu_test_{i}"
                value = "x" * 100
                cache.set(key, value)
                retrieved = cache.get(key)
                # Simulate some processing
                processed = retrieved.upper() if retrieved else ""
        
        profile_result = prof.profile_cpu_usage(
            cpu_intensive_operation,
            duration_seconds=5
        )
        
        assert "cpu_percent" in profile_result
        assert "function_calls" in profile_result
        assert "hotspots" in profile_result
        assert profile_result["cpu_percent"] >= 0
    
    def test_memory_profiling(self, profiler, memory_backend):
        """Test memory profiling of cache operations"""
        prof = profiler
        cache = memory_backend
        
        def memory_intensive_operation():
            # Create memory-intensive cache operations
            large_data = []
            for i in range(100):
                data = "x" * 10000  # 10KB per item
                large_data.append(data)
                cache.set(f"memory_test_{i}", data)
        
        profile_result = prof.profile_memory_usage(
            memory_intensive_operation,
            track_objects=True
        )
        
        assert "peak_memory_mb" in profile_result
        assert "memory_growth_mb" in profile_result
        assert "object_counts" in profile_result
        assert profile_result["peak_memory_mb"] > 0
    
    def test_io_profiling(self, profiler):
        """Test I/O profiling (simulated for Redis-like operations)"""
        prof = profiler
        
        # Simulate I/O operations
        def io_operation():
            import tempfile
            import os
            
            # Simulate file-based cache operations
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                for i in range(100):
                    temp_file.write(f"cache_data_{i}\n".encode())
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
                
                temp_file.seek(0)
                data = temp_file.read()
            
            os.unlink(temp_file.name)
            return len(data)
        
        profile_result = prof.profile_io_operations(
            io_operation,
            track_file_operations=True
        )
        
        assert "io_read_bytes" in profile_result
        assert "io_write_bytes" in profile_result
        assert "io_operations_count" in profile_result
    
    def test_comprehensive_profiling(self, profiler, memory_backend):
        """Test comprehensive profiling"""
        prof = profiler
        cache = memory_backend
        
        def comprehensive_operation():
            # Mix of operations
            import json
            import hashlib
            
            for i in range(200):
                # CPU work
                data = {"id": i, "value": "x" * 1000}
                json_data = json.dumps(data)
                hash_value = hashlib.md5(json_data.encode()).hexdigest()
                
                # Cache operations
                cache.set(f"comp_key_{i}", json_data)
                retrieved = cache.get(f"comp_key_{i}")
                
                # More CPU work
                if retrieved:
                    parsed = json.loads(retrieved)
                    assert parsed["id"] == i
        
        profile_result = prof.profile_comprehensive(
            comprehensive_operation,
            include_call_graph=True
        )
        
        assert "cpu_profile" in profile_result
        assert "memory_profile" in profile_result
        assert "execution_timeline" in profile_result
        assert "bottlenecks" in profile_result


class TestPerformanceMetricsCollector:
    """Test performance metrics collection"""
    
    @pytest.fixture
    def metrics_collector(self):
        """Metrics collector fixture"""
        return PerformanceMetricsCollector(
            collection_interval=1.0,
            enable_system_metrics=True,
            enable_cache_metrics=True,
            buffer_size=1000
        )
    
    def test_basic_metrics_collection(self, metrics_collector, memory_backend):
        """Test basic metrics collection"""
        collector = metrics_collector
        cache = memory_backend
        
        # Start collection
        collector.start_collection()
        
        # Perform cache operations
        for i in range(100):
            cache.set(f"metrics_key_{i}", f"value_{i}")
            cache.get(f"metrics_key_{i}")
        
        time.sleep(2)  # Let metrics collect
        
        # Stop collection
        collector.stop_collection()
        
        # Get collected metrics
        metrics = collector.get_metrics()
        
        assert len(metrics) > 0
        assert "timestamp" in metrics[0]
        assert "cache_operations" in metrics[0]
        assert "system_cpu" in metrics[0]
        assert "system_memory" in metrics[0]
    
    def test_custom_metrics_collection(self, metrics_collector):
        """Test custom metrics collection"""
        collector = metrics_collector
        
        # Define custom metrics
        def custom_metric_calculator():
            return {
                "custom_counter": time.time(),
                "custom_gauge": psutil.cpu_percent(),
                "custom_histogram": [1, 2, 3, 4, 5]
            }
        
        collector.add_custom_metric("custom_metrics", custom_metric_calculator)
        
        # Collect metrics
        collector.start_collection()
        time.sleep(2)
        collector.stop_collection()
        
        metrics = collector.get_metrics()
        
        assert len(metrics) > 0
        assert "custom_metrics" in metrics[0]
        assert "custom_counter" in metrics[0]["custom_metrics"]
    
    def test_metrics_aggregation(self, metrics_collector, memory_backend):
        """Test metrics aggregation"""
        collector = metrics_collector
        cache = memory_backend
        
        # Collect metrics over time
        collector.start_collection()
        
        for i in range(50):
            cache.set(f"agg_key_{i}", f"value_{i}")
            time.sleep(0.1)
        
        collector.stop_collection()
        
        # Aggregate metrics
        aggregated = collector.aggregate_metrics(
            time_window_seconds=60,
            aggregation_functions=["avg", "min", "max", "sum", "count"]
        )
        
        assert "cache_operations" in aggregated
        assert "avg" in aggregated["cache_operations"]
        assert "max" in aggregated["cache_operations"]
        assert aggregated["cache_operations"]["count"] > 0
    
    def test_metrics_alerting(self, metrics_collector):
        """Test metrics-based alerting"""
        collector = metrics_collector
        
        alerts_triggered = []
        
        def alert_handler(alert):
            alerts_triggered.append(alert)
        
        collector.set_alert_handler(alert_handler)
        
        # Set alert conditions
        collector.add_alert_condition(
            name="high_cpu",
            condition=lambda m: m.get("system_cpu", 0) > 80,
            message="CPU usage is high"
        )
        
        collector.add_alert_condition(
            name="high_memory",
            condition=lambda m: m.get("system_memory", 0) > 90,
            message="Memory usage is high"
        )
        
        # Simulate high CPU
        with patch('psutil.cpu_percent', return_value=85):
            collector.start_collection()
            time.sleep(2)
            collector.stop_collection()
        
        # Check if alerts were triggered
        assert len(alerts_triggered) > 0
        assert any(alert["name"] == "high_cpu" for alert in alerts_triggered)


class TestCacheCapacityPlanner:
    """Test cache capacity planning"""
    
    @pytest.fixture
    def capacity_planner(self):
        """Capacity planner fixture"""
        return CacheCapacityPlanner(
            enable_growth_prediction=True,
            enable_cost_analysis=True,
            planning_horizon_days=90
        )
    
    def test_usage_growth_prediction(self, capacity_planner):
        """Test cache usage growth prediction"""
        planner = capacity_planner
        
        # Historical usage data (simulated)
        historical_data = []
        base_usage = 1000
        
        for day in range(60):  # 60 days of data
            daily_growth = day * 10  # Linear growth
            usage = base_usage + daily_growth + (day % 7) * 50  # Weekly pattern
            
            historical_data.append({
                "date": datetime.now() - timedelta(days=60-day),
                "memory_usage_mb": usage,
                "operation_count": usage * 100,
                "unique_keys": usage // 10
            })
        
        # Predict future usage
        prediction = planner.predict_usage_growth(
            historical_data,
            prediction_days=30,
            include_seasonality=True
        )
        
        assert "predicted_memory_mb" in prediction
        assert "predicted_operations" in prediction
        assert "confidence_interval" in prediction
        assert len(prediction["predicted_memory_mb"]) == 30
    
    def test_capacity_recommendations(self, capacity_planner):
        """Test capacity recommendations"""
        planner = capacity_planner
        
        # Current usage metrics
        current_metrics = {
            "memory_usage_mb": 2000,
            "cpu_utilization": 0.65,
            "operations_per_second": 5000,
            "cache_hit_ratio": 0.85,
            "peak_memory_mb": 2500,
            "average_response_time_ms": 2.5
        }
        
        # Get recommendations
        recommendations = planner.get_capacity_recommendations(
            current_metrics,
            growth_rate=0.15,  # 15% monthly growth
            target_utilization=0.70
        )
        
        assert "memory_recommendation" in recommendations
        assert "cpu_recommendation" in recommendations
        assert "scaling_timeline" in recommendations
        assert recommendations["memory_recommendation"]["recommended_mb"] > 2000
    
    def test_cost_analysis(self, capacity_planner):
        """Test cost analysis for different capacity options"""
        planner = capacity_planner
        
        # Define cost models
        cost_models = {
            "memory": {"cost_per_gb_per_month": 10},
            "cpu": {"cost_per_core_per_month": 50},
            "storage": {"cost_per_gb_per_month": 5},
            "network": {"cost_per_gb_transfer": 0.1}
        }
        
        planner.set_cost_models(cost_models)
        
        # Current and projected usage
        usage_scenarios = [
            {"memory_gb": 2, "cpu_cores": 2, "storage_gb": 10, "transfer_gb_month": 100},
            {"memory_gb": 4, "cpu_cores": 4, "storage_gb": 20, "transfer_gb_month": 200},
            {"memory_gb": 8, "cpu_cores": 8, "storage_gb": 40, "transfer_gb_month": 400}
        ]
        
        cost_analysis = planner.analyze_costs(usage_scenarios, months=12)
        
        assert len(cost_analysis) == 3
        assert all("monthly_cost" in analysis for analysis in cost_analysis)
        assert all("annual_cost" in analysis for analysis in cost_analysis)
        assert all("cost_breakdown" in analysis for analysis in cost_analysis)
    
    def test_bottleneck_identification(self, capacity_planner):
        """Test bottleneck identification"""
        planner = capacity_planner
        
        # System metrics indicating bottlenecks
        system_metrics = {
            "cpu_utilization": 0.95,  # High CPU
            "memory_utilization": 0.60,  # Normal memory
            "disk_io_utilization": 0.80,  # High disk I/O
            "network_utilization": 0.30,  # Low network
            "cache_hit_ratio": 0.65,  # Low hit ratio
            "average_response_time_ms": 15  # High response time
        }
        
        bottlenecks = planner.identify_bottlenecks(system_metrics)
        
        assert len(bottlenecks) > 0
        assert any(b["component"] == "cpu" for b in bottlenecks)
        assert any(b["component"] == "disk_io" for b in bottlenecks)
        
        # Check severity and recommendations
        cpu_bottleneck = next(b for b in bottlenecks if b["component"] == "cpu")
        assert cpu_bottleneck["severity"] == "high"
        assert "recommendations" in cpu_bottleneck
    
    def test_scaling_strategy_planning(self, capacity_planner):
        """Test scaling strategy planning"""
        planner = capacity_planner
        
        # Current system state
        current_state = {
            "memory_gb": 4,
            "cpu_cores": 4,
            "instances": 2,
            "operations_per_second": 10000,
            "response_time_ms": 5
        }
        
        # Target requirements
        target_requirements = {
            "operations_per_second": 50000,  # 5x increase
            "max_response_time_ms": 3,  # Better performance
            "availability": 0.999  # High availability
        }
        
        scaling_strategy = planner.plan_scaling_strategy(
            current_state, 
            target_requirements,
            constraints={"max_instances": 10, "budget_monthly": 5000}
        )
        
        assert "strategy_type" in scaling_strategy  # vertical, horizontal, or hybrid
        assert "scaling_steps" in scaling_strategy
        assert "cost_estimate" in scaling_strategy
        assert "timeline" in scaling_strategy
        
        # Verify scaling steps are logical
        steps = scaling_strategy["scaling_steps"]
        assert len(steps) > 0
        assert all("action" in step for step in steps)
        assert all("resources" in step for step in steps)


class TestCacheOptimizer:
    """Test cache optimization"""
    
    @pytest.fixture
    def optimizer(self):
        """Optimizer fixture"""
        return CacheOptimizer(
            enable_automatic_tuning=True,
            optimization_interval_seconds=60,
            enable_ml_optimization=True
        )
    
    def test_ttl_optimization(self, optimizer, memory_backend):
        """Test TTL optimization"""
        opt = optimizer
        cache = memory_backend
        
        # Generate access patterns for TTL optimization
        access_patterns = []
        
        # Simulate different access patterns
        for i in range(1000):
            key = f"ttl_key_{i % 100}"  # 100 unique keys, repeated access
            access_time = time.time() - (i * 0.1)  # 0.1 second intervals
            access_patterns.append({
                "key": key,
                "access_time": access_time,
                "operation": "get" if i % 3 == 0 else "set"
            })
        
        # Optimize TTL based on access patterns
        ttl_recommendations = opt.optimize_ttl(access_patterns)
        
        assert "recommended_ttl_seconds" in ttl_recommendations
        assert "confidence_score" in ttl_recommendations
        assert "access_pattern_analysis" in ttl_recommendations
        assert ttl_recommendations["recommended_ttl_seconds"] > 0
    
    def test_eviction_policy_optimization(self, optimizer):
        """Test eviction policy optimization"""
        opt = optimizer
        
        # Simulate cache access patterns
        access_history = []
        
        # Create different access patterns
        patterns = {
            "hot_keys": [f"hot_{i}" for i in range(10)],  # Frequently accessed
            "warm_keys": [f"warm_{i}" for i in range(50)],  # Moderately accessed
            "cold_keys": [f"cold_{i}" for i in range(1000)]  # Rarely accessed
        }
        
        # Generate access history
        import random
        for _ in range(10000):
            # Hot keys: 60% of accesses
            if random.random() < 0.6:
                key = random.choice(patterns["hot_keys"])
            # Warm keys: 30% of accesses
            elif random.random() < 0.9:
                key = random.choice(patterns["warm_keys"])
            # Cold keys: 10% of accesses
            else:
                key = random.choice(patterns["cold_keys"])
            
            access_history.append({
                "key": key,
                "timestamp": time.time() - random.uniform(0, 3600),  # Last hour
                "operation": random.choice(["get", "set"])
            })
        
        # Optimize eviction policy
        eviction_analysis = opt.optimize_eviction_policy(
            access_history,
            cache_size_limit=100,
            policies_to_test=["lru", "lfu", "fifo", "random"]
        )
        
        assert "best_policy" in eviction_analysis
        assert "policy_performance" in eviction_analysis
        assert eviction_analysis["best_policy"] in ["lru", "lfu", "fifo", "random"]
        
        # Verify performance metrics
        for policy in eviction_analysis["policy_performance"]:
            assert "hit_ratio" in eviction_analysis["policy_performance"][policy]
            assert "eviction_count" in eviction_analysis["policy_performance"][policy]
    
    def test_memory_optimization(self, optimizer, memory_backend):
        """Test memory usage optimization"""
        opt = optimizer
        cache = memory_backend
        
        # Fill cache with test data
        for i in range(1000):
            key = f"memory_key_{i}"
            value = f"value_{i}_" + "x" * (i % 100)  # Variable size values
            cache.set(key, value)
        
        # Analyze memory usage
        memory_analysis = opt.analyze_memory_usage(cache)
        
        assert "total_memory_bytes" in memory_analysis
        assert "average_key_size" in memory_analysis
        assert "average_value_size" in memory_analysis
        assert "memory_efficiency" in memory_analysis
        
        # Get optimization recommendations
        memory_recommendations = opt.optimize_memory_usage(memory_analysis)
        
        assert "compression_opportunity" in memory_recommendations
        assert "key_optimization" in memory_recommendations
        assert "value_optimization" in memory_recommendations
    
    def test_performance_tuning(self, optimizer, memory_backend):
        """Test automatic performance tuning"""
        opt = optimizer
        cache = memory_backend
        
        # Simulate performance metrics
        performance_metrics = {
            "average_response_time_ms": 10,
            "throughput_ops_per_second": 1000,
            "cpu_utilization": 0.8,
            "memory_utilization": 0.7,
            "cache_hit_ratio": 0.75,
            "error_rate": 0.02
        }
        
        # Get tuning recommendations
        tuning_recommendations = opt.tune_performance(
            current_metrics=performance_metrics,
            target_metrics={
                "max_response_time_ms": 5,
                "min_throughput_ops_per_second": 2000,
                "min_cache_hit_ratio": 0.9
            }
        )
        
        assert "parameter_adjustments" in tuning_recommendations
        assert "expected_improvements" in tuning_recommendations
        assert "implementation_priority" in tuning_recommendations
    
    def test_ml_based_optimization(self, optimizer):
        """Test ML-based optimization"""
        opt = optimizer
        
        # Historical performance data for ML training
        training_data = []
        
        for i in range(1000):
            # Simulate various cache configurations and their performance
            config = {
                "ttl_seconds": random.randint(300, 3600),
                "max_memory_mb": random.randint(100, 1000),
                "eviction_policy": random.choice(["lru", "lfu", "fifo"]),
                "compression_enabled": random.choice([True, False])
            }
            
            # Simulate performance based on config (simplified model)
            performance = {
                "hit_ratio": min(0.9, 0.5 + (config["ttl_seconds"] / 3600) * 0.3),
                "response_time_ms": max(1, 10 - (config["max_memory_mb"] / 1000) * 5),
                "throughput": config["max_memory_mb"] * 2
            }
            
            training_data.append({"config": config, "performance": performance})
        
        # Train ML model and get recommendations
        ml_recommendations = opt.ml_optimize(
            training_data,
            target_metrics={"hit_ratio": 0.85, "response_time_ms": 3}
        )
        
        assert "optimal_config" in ml_recommendations
        assert "predicted_performance" in ml_recommendations
        assert "confidence_score" in ml_recommendations
        
        # Verify optimal config is reasonable
        optimal_config = ml_recommendations["optimal_config"]
        assert 300 <= optimal_config["ttl_seconds"] <= 3600
        assert 100 <= optimal_config["max_memory_mb"] <= 1000


class TestIntegratedPerformanceSuite:
    """Integration tests for complete performance testing suite"""
    
    def test_complete_performance_analysis(self):
        """Test complete performance analysis workflow"""
        # Initialize all components
        benchmark = CachePerformanceBenchmark()
        load_tester = CacheLoadTester()
        profiler = CacheProfiler()
        metrics_collector = PerformanceMetricsCollector()
        capacity_planner = CacheCapacityPlanner()
        optimizer = CacheOptimizer()
        
        # Test cache backend
        cache = MemoryCacheBackend(max_size=1000)
        
        # 1. Benchmark basic operations
        def cache_operation():
            cache.set("perf_key", "perf_value")
            return cache.get("perf_key")
        
        benchmark_result = benchmark.benchmark_operation(
            cache_operation, "cache_ops", iterations=100
        )
        
        # 2. Run load test
        config = LoadTestConfig(
            concurrent_users=10,
            duration_seconds=10,
            operations_per_second=100
        )
        
        load_result = load_tester.run_load_test(
            read_operation=lambda: cache.get("load_key"),
            write_operation=lambda: cache.set("load_key", "load_value"),
            config=config
        )
        
        # 3. Profile performance
        profile_result = profiler.profile_comprehensive(cache_operation)
        
        # 4. Collect metrics
        metrics_collector.start_collection()
        time.sleep(2)
        metrics_collector.stop_collection()
        metrics = metrics_collector.get_metrics()
        
        # 5. Plan capacity
        current_metrics = {
            "memory_usage_mb": 100,
            "cpu_utilization": 0.5,
            "operations_per_second": load_result["average_throughput"]
        }
        
        capacity_recommendations = capacity_planner.get_capacity_recommendations(
            current_metrics, growth_rate=0.1
        )
        
        # 6. Optimize performance
        performance_metrics = {
            "average_response_time_ms": benchmark_result.latency_p50,
            "throughput_ops_per_second": benchmark_result.throughput,
            "cache_hit_ratio": 0.8
        }
        
        optimization_recommendations = optimizer.tune_performance(
            performance_metrics,
            target_metrics={"max_response_time_ms": 2, "min_throughput_ops_per_second": 2000}
        )
        
        # Verify all components worked
        assert benchmark_result.throughput > 0
        assert load_result["total_operations"] > 0
        assert "cpu_profile" in profile_result
        assert len(metrics) > 0
        assert "memory_recommendation" in capacity_recommendations
        assert "parameter_adjustments" in optimization_recommendations


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
