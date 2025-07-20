"""
Tests for cache manager in Spotify AI Agent

Comprehensive testing suite for the central cache manager that orchestrates
all caching operations, backend management, and monitoring.

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
import json
from unittest.mock import Mock, patch, AsyncMock
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass

from app.utils.cache.manager import CacheManager, CacheConfiguration
from app.utils.cache.backends.memory_backend import MemoryCacheBackend
from app.utils.cache.backends.redis_backend import RedisCacheBackend
from app.utils.cache.backends.hybrid_backend import HybridCacheBackend
from app.utils.cache.serializers import JSONSerializer, PickleSerializer
from app.utils.cache.exceptions import CacheError, CacheConfigurationError
from app.utils.cache.metrics import CacheMetrics
from app.utils.cache.events import CacheEventBus


@dataclass
class MockCacheConfig:
    """Mock cache configuration for testing"""
    default_ttl: int = 3600
    max_memory_size: int = 1000
    redis_url: Optional[str] = None
    enable_metrics: bool = True
    enable_events: bool = True
    compression_threshold: int = 1024
    serializer_type: str = "json"


class TestCacheManager:
    """Test central cache manager functionality"""
    
    @pytest.fixture
    def memory_backend(self):
        """Memory backend fixture"""
        return MemoryCacheBackend(max_size=1000, default_ttl=3600)
    
    @pytest.fixture
    def cache_config(self):
        """Cache configuration fixture"""
        return CacheConfiguration(
            default_ttl=3600,
            default_backend="memory",
            enable_metrics=True,
            enable_compression=False
        )
    
    @pytest.fixture
    def cache_manager(self, memory_backend, cache_config):
        """Cache manager fixture"""
        return CacheManager(
            default_backend=memory_backend,
            config=cache_config
        )
    
    def test_cache_manager_initialization(self, memory_backend):
        """Test cache manager initialization"""
        config = CacheConfiguration(
            default_ttl=1800,
            default_backend="memory",
            enable_metrics=True
        )
        
        manager = CacheManager(
            default_backend=memory_backend,
            config=config
        )
        
        assert manager.default_backend == memory_backend
        assert manager.config.default_ttl == 1800
        assert manager.config.enable_metrics is True
        assert manager.is_enabled is True
    
    def test_cache_manager_basic_operations(self, cache_manager):
        """Test basic cache operations through manager"""
        key = "test_key"
        value = {"data": "test_value", "number": 42}
        
        # Test set
        result = cache_manager.set(key, value)
        assert result is True
        
        # Test get
        retrieved = cache_manager.get(key)
        assert retrieved == value
        
        # Test exists
        assert cache_manager.exists(key) is True
        
        # Test delete
        result = cache_manager.delete(key)
        assert result is True
        assert cache_manager.exists(key) is False
    
    def test_cache_manager_with_ttl(self, cache_manager):
        """Test TTL handling through manager"""
        key = "ttl_test_key"
        value = "ttl_test_value"
        ttl = 10
        
        # Set with TTL
        cache_manager.set(key, value, ttl=ttl)
        
        # Check TTL
        remaining_ttl = cache_manager.get_ttl(key)
        assert remaining_ttl <= ttl
        assert remaining_ttl > 0
        
        # Test expire
        result = cache_manager.expire(key, 5)
        assert result is True
        
        new_ttl = cache_manager.get_ttl(key)
        assert new_ttl <= 5
    
    def test_cache_manager_bulk_operations(self, cache_manager):
        """Test bulk operations through manager"""
        data = {
            "bulk_key_1": "value_1",
            "bulk_key_2": {"nested": "value_2"},
            "bulk_key_3": [1, 2, 3, 4, 5]
        }
        
        # Test set_many
        result = cache_manager.set_many(data)
        assert result is True
        
        # Test get_many
        retrieved = cache_manager.get_many(list(data.keys()))
        assert len(retrieved) == 3
        for key, value in data.items():
            assert retrieved[key] == value
        
        # Test delete_many
        deleted_count = cache_manager.delete_many(list(data.keys()))
        assert deleted_count == 3
        
        # Verify deletion
        for key in data.keys():
            assert not cache_manager.exists(key)
    
    def test_cache_manager_increment_decrement(self, cache_manager):
        """Test increment/decrement operations"""
        key = "counter_key"
        
        # Test increment on non-existent key
        result = cache_manager.increment(key)
        assert result == 1
        
        # Test increment with delta
        result = cache_manager.increment(key, delta=5)
        assert result == 6
        
        # Test decrement
        result = cache_manager.decrement(key)
        assert result == 5
        
        # Test decrement with delta
        result = cache_manager.decrement(key, delta=2)
        assert result == 3
    
    def test_cache_manager_with_tags(self, cache_manager):
        """Test tag-based operations through manager"""
        # Set entries with tags
        cache_manager.set("user:123:profile", {"name": "John"}, tags=["user", "profile", "user:123"])
        cache_manager.set("user:123:settings", {"theme": "dark"}, tags=["user", "settings", "user:123"])
        cache_manager.set("user:456:profile", {"name": "Jane"}, tags=["user", "profile", "user:456"])
        cache_manager.set("global:config", {"version": "1.0"}, tags=["global", "config"])
        
        # Test tag-based invalidation
        invalidated_keys = cache_manager.invalidate_by_tag("user:123")
        
        assert "user:123:profile" in invalidated_keys
        assert "user:123:settings" in invalidated_keys
        assert "user:456:profile" not in invalidated_keys
        assert "global:config" not in invalidated_keys
        
        # Verify cache state
        assert not cache_manager.exists("user:123:profile")
        assert not cache_manager.exists("user:123:settings")
        assert cache_manager.exists("user:456:profile")
        assert cache_manager.exists("global:config")
    
    def test_cache_manager_clear(self, cache_manager):
        """Test cache clearing"""
        # Populate cache
        for i in range(10):
            cache_manager.set(f"clear_test_key_{i}", f"value_{i}")
        
        # Verify entries exist
        for i in range(10):
            assert cache_manager.exists(f"clear_test_key_{i}")
        
        # Clear cache
        result = cache_manager.clear()
        assert result is True
        
        # Verify cache is empty
        for i in range(10):
            assert not cache_manager.exists(f"clear_test_key_{i}")
    
    def test_cache_manager_health_check(self, cache_manager):
        """Test cache manager health check"""
        health_status = cache_manager.health_check()
        
        assert "status" in health_status
        assert "backend_status" in health_status
        assert "metrics" in health_status
        assert health_status["status"] in ["healthy", "degraded", "unhealthy"]
    
    def test_cache_manager_statistics(self, cache_manager):
        """Test cache statistics through manager"""
        # Perform operations to generate stats
        cache_manager.set("stats_key_1", "value_1")
        cache_manager.set("stats_key_2", "value_2")
        cache_manager.get("stats_key_1")  # Hit
        cache_manager.get("non_existent_key")  # Miss
        
        stats = cache_manager.get_statistics()
        
        assert "total_operations" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "hit_rate" in stats
        assert "backend_stats" in stats


class TestCacheManagerConfiguration:
    """Test cache manager configuration handling"""
    
    def test_cache_configuration_creation(self):
        """Test cache configuration creation"""
        config = CacheConfiguration(
            default_ttl=1800,
            default_backend="redis",
            enable_metrics=True,
            enable_compression=True,
            compression_threshold=2048,
            max_memory_usage=500 * 1024 * 1024  # 500MB
        )
        
        assert config.default_ttl == 1800
        assert config.default_backend == "redis"
        assert config.enable_metrics is True
        assert config.enable_compression is True
        assert config.compression_threshold == 2048
        assert config.max_memory_usage == 500 * 1024 * 1024
    
    def test_cache_configuration_validation(self):
        """Test cache configuration validation"""
        # Valid configuration
        valid_config = CacheConfiguration(
            default_ttl=3600,
            default_backend="memory"
        )
        assert valid_config.is_valid()
        
        # Invalid TTL
        with pytest.raises(CacheConfigurationError):
            CacheConfiguration(default_ttl=-1)
        
        # Invalid backend
        with pytest.raises(CacheConfigurationError):
            CacheConfiguration(default_backend="invalid_backend")
    
    def test_cache_configuration_from_dict(self):
        """Test creating configuration from dictionary"""
        config_dict = {
            "default_ttl": 1800,
            "default_backend": "redis",
            "enable_metrics": True,
            "enable_compression": False,
            "redis_url": "redis://localhost:6379/0"
        }
        
        config = CacheConfiguration.from_dict(config_dict)
        
        assert config.default_ttl == 1800
        assert config.default_backend == "redis"
        assert config.enable_metrics is True
        assert config.enable_compression is False
        assert config.redis_url == "redis://localhost:6379/0"
    
    def test_cache_configuration_to_dict(self):
        """Test converting configuration to dictionary"""
        config = CacheConfiguration(
            default_ttl=3600,
            default_backend="memory",
            enable_metrics=True
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["default_ttl"] == 3600
        assert config_dict["default_backend"] == "memory"
        assert config_dict["enable_metrics"] is True


class TestCacheManagerBackendManagement:
    """Test cache manager backend management"""
    
    @pytest.fixture
    def memory_backend(self):
        """Memory backend fixture"""
        return MemoryCacheBackend(max_size=100)
    
    @pytest.fixture
    def redis_backend(self):
        """Mock Redis backend fixture"""
        from fakeredis import FakeRedis
        client = FakeRedis(decode_responses=False)
        return RedisCacheBackend(client=client)
    
    def test_cache_manager_multiple_backends(self, memory_backend, redis_backend):
        """Test cache manager with multiple backends"""
        config = CacheConfiguration(default_backend="memory")
        
        manager = CacheManager(
            default_backend=memory_backend,
            config=config
        )
        
        # Register additional backend
        manager.register_backend("redis", redis_backend)
        
        # Test using different backends
        manager.set("memory_key", "memory_value", backend="memory")
        manager.set("redis_key", "redis_value", backend="redis")
        
        # Verify data is in correct backends
        assert manager.get("memory_key", backend="memory") == "memory_value"
        assert manager.get("redis_key", backend="redis") == "redis_value"
        
        # Cross-backend access should return None
        assert manager.get("memory_key", backend="redis") is None
        assert manager.get("redis_key", backend="memory") is None
    
    def test_cache_manager_backend_fallback(self, memory_backend):
        """Test backend fallback behavior"""
        # Create failing backend
        failing_backend = Mock()
        failing_backend.get.side_effect = Exception("Backend failed")
        failing_backend.set.side_effect = Exception("Backend failed")
        
        config = CacheConfiguration(
            default_backend="failing",
            enable_fallback=True,
            fallback_backend="memory"
        )
        
        manager = CacheManager(
            default_backend=failing_backend,
            config=config
        )
        manager.register_backend("memory", memory_backend)
        
        # Operations should fallback to memory backend
        result = manager.set("test_key", "test_value")
        assert result is True
        
        retrieved = manager.get("test_key")
        assert retrieved == "test_value"
    
    def test_cache_manager_backend_health_monitoring(self, memory_backend, redis_backend):
        """Test backend health monitoring"""
        config = CacheConfiguration(
            default_backend="memory",
            enable_health_monitoring=True,
            health_check_interval=1
        )
        
        manager = CacheManager(
            default_backend=memory_backend,
            config=config
        )
        manager.register_backend("redis", redis_backend)
        
        # Get health status
        health = manager.get_backend_health()
        
        assert "memory" in health
        assert "redis" in health
        assert health["memory"]["status"] == "healthy"
        assert health["redis"]["status"] == "healthy"
    
    def test_cache_manager_backend_load_balancing(self, memory_backend, redis_backend):
        """Test load balancing across backends"""
        config = CacheConfiguration(
            default_backend="memory",
            enable_load_balancing=True,
            load_balancing_strategy="round_robin"
        )
        
        manager = CacheManager(
            default_backend=memory_backend,
            config=config
        )
        manager.register_backend("redis", redis_backend)
        
        # Set multiple keys (should distribute across backends)
        for i in range(10):
            manager.set(f"lb_key_{i}", f"value_{i}")
        
        # Verify keys are distributed
        memory_count = 0
        redis_count = 0
        
        for i in range(10):
            if memory_backend.exists(f"lb_key_{i}"):
                memory_count += 1
            if redis_backend.exists(f"lb_key_{i}"):
                redis_count += 1
        
        # Both backends should have some keys
        assert memory_count > 0
        assert redis_count > 0
        assert memory_count + redis_count == 10


class TestCacheManagerMetrics:
    """Test cache manager metrics collection"""
    
    @pytest.fixture
    def cache_manager_with_metrics(self):
        """Cache manager with metrics enabled"""
        backend = MemoryCacheBackend(max_size=100)
        config = CacheConfiguration(
            default_backend="memory",
            enable_metrics=True,
            metrics_collection_interval=1
        )
        
        return CacheManager(default_backend=backend, config=config)
    
    def test_cache_manager_metrics_collection(self, cache_manager_with_metrics):
        """Test basic metrics collection"""
        manager = cache_manager_with_metrics
        
        # Perform operations
        manager.set("metrics_key_1", "value_1")
        manager.set("metrics_key_2", "value_2")
        manager.get("metrics_key_1")  # Hit
        manager.get("metrics_key_2")  # Hit
        manager.get("non_existent")   # Miss
        
        # Get metrics
        metrics = manager.get_metrics()
        
        assert metrics.total_operations >= 5
        assert metrics.cache_hits >= 2
        assert metrics.cache_misses >= 1
        assert metrics.hit_rate > 0
        assert metrics.miss_rate > 0
    
    def test_cache_manager_performance_metrics(self, cache_manager_with_metrics):
        """Test performance metrics collection"""
        manager = cache_manager_with_metrics
        
        # Perform timed operations
        start_time = time.time()
        for i in range(100):
            manager.set(f"perf_key_{i}", f"value_{i}")
        
        for i in range(100):
            manager.get(f"perf_key_{i}")
        
        duration = time.time() - start_time
        
        # Get performance metrics
        metrics = manager.get_performance_metrics()
        
        assert "avg_set_latency" in metrics
        assert "avg_get_latency" in metrics
        assert "operations_per_second" in metrics
        assert metrics["operations_per_second"] > 0
    
    def test_cache_manager_memory_metrics(self, cache_manager_with_metrics):
        """Test memory usage metrics"""
        manager = cache_manager_with_metrics
        
        # Add data to cache
        large_data = "x" * 1024  # 1KB data
        for i in range(10):
            manager.set(f"memory_test_key_{i}", large_data)
        
        # Get memory metrics
        metrics = manager.get_memory_metrics()
        
        assert "total_memory_usage" in metrics
        assert "cache_size" in metrics
        assert "memory_efficiency" in metrics
        assert metrics["cache_size"] == 10
        assert metrics["total_memory_usage"] > 0


class TestCacheManagerEvents:
    """Test cache manager event system"""
    
    @pytest.fixture
    def cache_manager_with_events(self):
        """Cache manager with events enabled"""
        backend = MemoryCacheBackend(max_size=100)
        config = CacheConfiguration(
            default_backend="memory",
            enable_events=True
        )
        
        return CacheManager(default_backend=backend, config=config)
    
    def test_cache_manager_event_subscription(self, cache_manager_with_events):
        """Test event subscription and handling"""
        manager = cache_manager_with_events
        events = []
        
        def event_handler(event):
            events.append(event)
        
        # Subscribe to events
        manager.subscribe_to_events("cache_set", event_handler)
        manager.subscribe_to_events("cache_get", event_handler)
        manager.subscribe_to_events("cache_delete", event_handler)
        
        # Perform operations
        manager.set("event_key", "event_value")
        manager.get("event_key")
        manager.delete("event_key")
        
        # Check events
        assert len(events) >= 3
        
        event_types = [event.event_type for event in events]
        assert "cache_set" in event_types
        assert "cache_get" in event_types
        assert "cache_delete" in event_types
    
    def test_cache_manager_event_filtering(self, cache_manager_with_events):
        """Test event filtering"""
        manager = cache_manager_with_events
        filtered_events = []
        
        def filtered_handler(event):
            if event.key.startswith("important:"):
                filtered_events.append(event)
        
        # Subscribe with filter
        manager.subscribe_to_events("cache_set", filtered_handler)
        
        # Set various keys
        manager.set("important:key1", "value1")
        manager.set("normal:key2", "value2")
        manager.set("important:key3", "value3")
        
        # Check filtered events
        assert len(filtered_events) == 2
        assert all(event.key.startswith("important:") for event in filtered_events)
    
    def test_cache_manager_event_performance_impact(self, cache_manager_with_events):
        """Test performance impact of event system"""
        manager = cache_manager_with_events
        
        # Measure performance without event handlers
        start_time = time.time()
        for i in range(1000):
            manager.set(f"no_events_key_{i}", f"value_{i}")
        no_events_duration = time.time() - start_time
        
        # Add event handlers
        def dummy_handler(event):
            pass  # Minimal processing
        
        manager.subscribe_to_events("cache_set", dummy_handler)
        
        # Measure performance with event handlers
        start_time = time.time()
        for i in range(1000):
            manager.set(f"with_events_key_{i}", f"value_{i}")
        with_events_duration = time.time() - start_time
        
        # Event overhead should be minimal
        overhead_ratio = with_events_duration / no_events_duration
        assert overhead_ratio < 2.0, f"Event overhead too high: {overhead_ratio}x"


class TestCacheManagerAsync:
    """Test cache manager async operations"""
    
    @pytest.fixture
    def async_cache_manager(self):
        """Async cache manager fixture"""
        backend = MemoryCacheBackend(max_size=100)
        config = CacheConfiguration(
            default_backend="memory",
            enable_async=True
        )
        
        return CacheManager(default_backend=backend, config=config)
    
    @pytest.mark.asyncio
    async def test_cache_manager_async_operations(self, async_cache_manager):
        """Test basic async cache operations"""
        manager = async_cache_manager
        
        # Test async set
        result = await manager.aset("async_key", "async_value")
        assert result is True
        
        # Test async get
        retrieved = await manager.aget("async_key")
        assert retrieved == "async_value"
        
        # Test async exists
        exists = await manager.aexists("async_key")
        assert exists is True
        
        # Test async delete
        deleted = await manager.adelete("async_key")
        assert deleted is True
        
        # Verify deletion
        exists_after = await manager.aexists("async_key")
        assert exists_after is False
    
    @pytest.mark.asyncio
    async def test_cache_manager_async_bulk_operations(self, async_cache_manager):
        """Test async bulk operations"""
        manager = async_cache_manager
        
        data = {
            "async_bulk_key_1": "value_1",
            "async_bulk_key_2": {"nested": "value_2"},
            "async_bulk_key_3": [1, 2, 3]
        }
        
        # Test async set_many
        result = await manager.aset_many(data)
        assert result is True
        
        # Test async get_many
        retrieved = await manager.aget_many(list(data.keys()))
        assert len(retrieved) == 3
        for key, value in data.items():
            assert retrieved[key] == value
        
        # Test async delete_many
        deleted_count = await manager.adelete_many(list(data.keys()))
        assert deleted_count == 3
    
    @pytest.mark.asyncio
    async def test_cache_manager_async_concurrent_access(self, async_cache_manager):
        """Test concurrent async access"""
        manager = async_cache_manager
        
        async def worker(worker_id: int):
            results = []
            for i in range(10):
                key = f"concurrent_key_{worker_id}_{i}"
                value = f"value_{worker_id}_{i}"
                
                # Set
                await manager.aset(key, value)
                
                # Get
                retrieved = await manager.aget(key)
                results.append(retrieved == value)
            
            return all(results)
        
        # Run multiple workers concurrently
        tasks = [worker(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        # All workers should succeed
        assert all(results)


class TestCacheManagerErrorHandling:
    """Test cache manager error handling"""
    
    @pytest.fixture
    def failing_backend(self):
        """Mock failing backend"""
        backend = Mock()
        backend.get.side_effect = Exception("Backend error")
        backend.set.side_effect = Exception("Backend error")
        backend.delete.side_effect = Exception("Backend error")
        backend.health_check.return_value = {"status": "unhealthy"}
        return backend
    
    def test_cache_manager_graceful_degradation(self, failing_backend):
        """Test graceful degradation on backend failures"""
        config = CacheConfiguration(
            default_backend="failing",
            fail_silently=True,
            enable_degraded_mode=True
        )
        
        manager = CacheManager(
            default_backend=failing_backend,
            config=config
        )
        
        # Operations should not raise exceptions
        result = manager.set("test_key", "test_value")
        assert result is False  # Failed but didn't raise
        
        retrieved = manager.get("test_key")
        assert retrieved is None  # Failed but didn't raise
    
    def test_cache_manager_error_propagation(self, failing_backend):
        """Test error propagation when fail_silently=False"""
        config = CacheConfiguration(
            default_backend="failing",
            fail_silently=False
        )
        
        manager = CacheManager(
            default_backend=failing_backend,
            config=config
        )
        
        # Operations should raise exceptions
        with pytest.raises(Exception, match="Backend error"):
            manager.set("test_key", "test_value")
        
        with pytest.raises(Exception, match="Backend error"):
            manager.get("test_key")
    
    def test_cache_manager_circuit_breaker(self, failing_backend):
        """Test circuit breaker pattern"""
        config = CacheConfiguration(
            default_backend="failing",
            enable_circuit_breaker=True,
            circuit_breaker_threshold=3,
            circuit_breaker_timeout=1
        )
        
        manager = CacheManager(
            default_backend=failing_backend,
            config=config
        )
        
        # First few operations should attempt backend
        for i in range(3):
            result = manager.set(f"test_key_{i}", f"value_{i}")
            assert result is False
        
        # Circuit should be open now
        assert manager.is_circuit_open()
        
        # Further operations should fail fast
        start_time = time.time()
        result = manager.set("fast_fail_key", "fast_fail_value")
        duration = time.time() - start_time
        
        assert result is False
        assert duration < 0.1  # Should fail fast


class TestCacheManagerIntegration:
    """Integration tests for cache manager"""
    
    @pytest.fixture
    def full_cache_manager(self):
        """Fully configured cache manager"""
        memory_backend = MemoryCacheBackend(max_size=100)
        
        config = CacheConfiguration(
            default_ttl=3600,
            default_backend="memory",
            enable_metrics=True,
            enable_events=True,
            enable_compression=True,
            compression_threshold=100
        )
        
        return CacheManager(default_backend=memory_backend, config=config)
    
    def test_cache_manager_end_to_end_workflow(self, full_cache_manager):
        """Test complete cache workflow"""
        manager = full_cache_manager
        
        # Set up event tracking
        events = []
        manager.subscribe_to_events("cache_set", lambda e: events.append(e))
        manager.subscribe_to_events("cache_get", lambda e: events.append(e))
        
        # Complex data operations
        user_data = {
            "user_id": 123,
            "profile": {
                "name": "John Doe",
                "preferences": {
                    "genres": ["rock", "pop"],
                    "features": {"valence": 0.8, "energy": 0.9}
                }
            },
            "history": [f"track_{i}" for i in range(50)]  # Large data for compression
        }
        
        # Set with tags
        manager.set("user:123:complete_profile", user_data, 
                   tags=["user:123", "profile", "complete"], ttl=1800)
        
        # Retrieve and verify
        retrieved = manager.get("user:123:complete_profile")
        assert retrieved == user_data
        
        # Check metrics
        metrics = manager.get_metrics()
        assert metrics.total_operations >= 2
        assert metrics.cache_hits >= 1
        
        # Check events
        assert len(events) >= 2
        
        # Tag-based invalidation
        invalidated = manager.invalidate_by_tag("user:123")
        assert "user:123:complete_profile" in invalidated
        
        # Verify invalidation
        assert not manager.exists("user:123:complete_profile")
        
        # Health check
        health = manager.health_check()
        assert health["status"] == "healthy"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
