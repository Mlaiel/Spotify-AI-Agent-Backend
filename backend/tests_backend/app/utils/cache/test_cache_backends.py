"""
Tests for cache backends in Spotify AI Agent

Comprehensive testing suite for various cache backend implementations
including Redis, Memory, Hybrid, and Distributed cache systems.

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
import pickle
from unittest.mock import Mock, patch, AsyncMock
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta

import redis
import redis.asyncio as aioredis
from redis.exceptions import ConnectionError, RedisError
from fakeredis import FakeRedis, FakeAsyncRedis

from app.utils.cache.backends.base import BaseCacheBackend
from app.utils.cache.backends.redis_backend import RedisCacheBackend, AsyncRedisCacheBackend
from app.utils.cache.backends.memory_backend import MemoryCacheBackend, LRUCacheBackend
from app.utils.cache.backends.hybrid_backend import HybridCacheBackend
from app.utils.cache.backends.distributed_backend import DistributedCacheBackend
from app.utils.cache.exceptions import CacheConnectionError, CacheOperationError
from app.utils.cache.serializers import JSONSerializer, PickleSerializer, CompressedSerializer


@dataclass
class CacheTestData:
    """Test data for cache operations"""
    simple_string: str = "test_value"
    simple_int: int = 42
    simple_dict: Dict = None
    complex_object: Dict = None
    large_data: bytes = None
    
    def __post_init__(self):
        self.simple_dict = {"key": "value", "number": 123}
        self.complex_object = {
            "user_id": "12345",
            "preferences": {
                "genres": ["rock", "pop", "electronic"],
                "artists": ["Artist1", "Artist2"],
                "features": {
                    "valence": 0.8,
                    "energy": 0.9,
                    "danceability": 0.7
                }
            },
            "history": [
                {"track_id": "track1", "timestamp": "2024-01-01T10:00:00Z"},
                {"track_id": "track2", "timestamp": "2024-01-01T10:03:00Z"},
            ]
        }
        self.large_data = b"x" * 1024 * 1024  # 1MB of data


class TestBaseCacheBackend:
    """Test base cache backend abstract class"""
    
    def test_base_backend_is_abstract(self):
        """Test that base backend cannot be instantiated directly"""
        with pytest.raises(TypeError):
            BaseCacheBackend()
    
    def test_backend_interface_methods(self):
        """Test that backend interface has required methods"""
        required_methods = [
            'get', 'set', 'delete', 'exists', 'clear',
            'get_many', 'set_many', 'delete_many',
            'increment', 'decrement', 'expire',
            'get_ttl', 'health_check'
        ]
        
        for method in required_methods:
            assert hasattr(BaseCacheBackend, method)
    
    def test_backend_async_interface(self):
        """Test async backend interface methods"""
        async_methods = [
            'aget', 'aset', 'adelete', 'aexists', 'aclear',
            'aget_many', 'aset_many', 'adelete_many',
            'aincrement', 'adecrement', 'aexpire',
            'aget_ttl', 'ahealth_check'
        ]
        
        for method in async_methods:
            assert hasattr(BaseCacheBackend, method)


class TestRedisCacheBackend:
    """Test Redis cache backend implementation"""
    
    @pytest.fixture
    def redis_client(self):
        """Mock Redis client fixture"""
        return FakeRedis(decode_responses=False)
    
    @pytest.fixture
    def async_redis_client(self):
        """Mock async Redis client fixture"""
        return FakeAsyncRedis(decode_responses=False)
    
    @pytest.fixture
    def redis_backend(self, redis_client):
        """Redis backend fixture"""
        return RedisCacheBackend(
            client=redis_client,
            serializer=JSONSerializer(),
            default_ttl=3600,
            key_prefix="test:"
        )
    
    @pytest.fixture
    def async_redis_backend(self, async_redis_client):
        """Async Redis backend fixture"""
        return AsyncRedisCacheBackend(
            client=async_redis_client,
            serializer=JSONSerializer(),
            default_ttl=3600,
            key_prefix="test:"
        )
    
    @pytest.fixture
    def test_data(self):
        """Test data fixture"""
        return CacheTestData()
    
    def test_redis_backend_initialization(self, redis_client):
        """Test Redis backend initialization"""
        backend = RedisCacheBackend(
            client=redis_client,
            serializer=JSONSerializer(),
            default_ttl=1800,
            key_prefix="spotify:",
            compression_enabled=True
        )
        
        assert backend.client == redis_client
        assert backend.default_ttl == 1800
        assert backend.key_prefix == "spotify:"
        assert backend.compression_enabled is True
    
    def test_set_and_get_simple_value(self, redis_backend, test_data):
        """Test setting and getting simple values"""
        key = "simple_key"
        value = test_data.simple_string
        
        # Set value
        result = redis_backend.set(key, value)
        assert result is True
        
        # Get value
        retrieved = redis_backend.get(key)
        assert retrieved == value
    
    def test_set_and_get_complex_object(self, redis_backend, test_data):
        """Test setting and getting complex objects"""
        key = "complex_key"
        value = test_data.complex_object
        
        # Set complex object
        result = redis_backend.set(key, value)
        assert result is True
        
        # Get complex object
        retrieved = redis_backend.get(key)
        assert retrieved == value
        assert retrieved["user_id"] == "12345"
        assert len(retrieved["preferences"]["genres"]) == 3
    
    def test_set_with_ttl(self, redis_backend, test_data):
        """Test setting values with TTL"""
        key = "ttl_key"
        value = test_data.simple_string
        ttl = 10
        
        # Set with TTL
        result = redis_backend.set(key, value, ttl=ttl)
        assert result is True
        
        # Check TTL
        remaining_ttl = redis_backend.get_ttl(key)
        assert remaining_ttl <= ttl
        assert remaining_ttl > 0
    
    def test_delete_key(self, redis_backend, test_data):
        """Test deleting keys"""
        key = "delete_key"
        value = test_data.simple_string
        
        # Set and verify exists
        redis_backend.set(key, value)
        assert redis_backend.exists(key) is True
        
        # Delete and verify
        result = redis_backend.delete(key)
        assert result is True
        assert redis_backend.exists(key) is False
    
    def test_get_nonexistent_key(self, redis_backend):
        """Test getting non-existent key returns None"""
        result = redis_backend.get("nonexistent_key")
        assert result is None
    
    def test_set_many_and_get_many(self, redis_backend, test_data):
        """Test bulk set and get operations"""
        data = {
            "key1": test_data.simple_string,
            "key2": test_data.simple_int,
            "key3": test_data.simple_dict
        }
        
        # Set many
        result = redis_backend.set_many(data)
        assert result is True
        
        # Get many
        retrieved = redis_backend.get_many(list(data.keys()))
        assert len(retrieved) == 3
        assert retrieved["key1"] == test_data.simple_string
        assert retrieved["key2"] == test_data.simple_int
        assert retrieved["key3"] == test_data.simple_dict
    
    def test_delete_many(self, redis_backend, test_data):
        """Test bulk delete operations"""
        keys = ["bulk_key1", "bulk_key2", "bulk_key3"]
        value = test_data.simple_string
        
        # Set multiple keys
        for key in keys:
            redis_backend.set(key, value)
        
        # Verify all exist
        for key in keys:
            assert redis_backend.exists(key) is True
        
        # Delete many
        result = redis_backend.delete_many(keys)
        assert result == len(keys)
        
        # Verify all deleted
        for key in keys:
            assert redis_backend.exists(key) is False
    
    def test_increment_and_decrement(self, redis_backend):
        """Test increment and decrement operations"""
        key = "counter_key"
        
        # Test increment on non-existent key
        result = redis_backend.increment(key)
        assert result == 1
        
        # Test increment with delta
        result = redis_backend.increment(key, delta=5)
        assert result == 6
        
        # Test decrement
        result = redis_backend.decrement(key)
        assert result == 5
        
        # Test decrement with delta
        result = redis_backend.decrement(key, delta=3)
        assert result == 2
    
    def test_expire_key(self, redis_backend, test_data):
        """Test setting expiration on existing key"""
        key = "expire_key"
        value = test_data.simple_string
        
        # Set key without TTL
        redis_backend.set(key, value)
        
        # Set expiration
        result = redis_backend.expire(key, 10)
        assert result is True
        
        # Check TTL
        ttl = redis_backend.get_ttl(key)
        assert ttl <= 10
        assert ttl > 0
    
    def test_clear_cache(self, redis_backend, test_data):
        """Test clearing all cache data"""
        # Set multiple keys
        for i in range(5):
            redis_backend.set(f"clear_key_{i}", test_data.simple_string)
        
        # Clear cache
        result = redis_backend.clear()
        assert result is True
        
        # Verify all keys are gone
        for i in range(5):
            assert redis_backend.exists(f"clear_key_{i}") is False
    
    def test_health_check(self, redis_backend):
        """Test health check functionality"""
        health_status = redis_backend.health_check()
        assert health_status["status"] == "healthy"
        assert "latency_ms" in health_status
        assert "memory_usage" in health_status
        assert "connected_clients" in health_status
    
    def test_key_prefix_handling(self, redis_client):
        """Test key prefix functionality"""
        backend = RedisCacheBackend(
            client=redis_client,
            serializer=JSONSerializer(),
            key_prefix="spotify:cache:"
        )
        
        key = "test_key"
        value = "test_value"
        
        backend.set(key, value)
        
        # Check that key is stored with prefix
        raw_keys = redis_client.keys("*")
        prefixed_key = f"spotify:cache:{key}".encode()
        assert prefixed_key in raw_keys
    
    def test_serialization_error_handling(self, redis_backend):
        """Test handling of serialization errors"""
        # Test with unserializable object
        class UnserializableClass:
            def __init__(self):
                self.func = lambda x: x  # Functions can't be JSON serialized
        
        key = "error_key"
        value = UnserializableClass()
        
        with pytest.raises(CacheOperationError):
            redis_backend.set(key, value)
    
    def test_connection_error_handling(self, redis_client):
        """Test handling of Redis connection errors"""
        backend = RedisCacheBackend(client=redis_client)
        
        # Mock connection error
        with patch.object(redis_client, 'set', side_effect=ConnectionError("Connection failed")):
            with pytest.raises(CacheConnectionError):
                backend.set("test_key", "test_value")
    
    @pytest.mark.asyncio
    async def test_async_redis_operations(self, async_redis_backend, test_data):
        """Test async Redis operations"""
        key = "async_key"
        value = test_data.complex_object
        
        # Async set
        result = await async_redis_backend.aset(key, value)
        assert result is True
        
        # Async get
        retrieved = await async_redis_backend.aget(key)
        assert retrieved == value
        
        # Async exists
        exists = await async_redis_backend.aexists(key)
        assert exists is True
        
        # Async delete
        deleted = await async_redis_backend.adelete(key)
        assert deleted is True
        
        # Verify deleted
        exists_after = await async_redis_backend.aexists(key)
        assert exists_after is False
    
    @pytest.mark.asyncio
    async def test_async_bulk_operations(self, async_redis_backend, test_data):
        """Test async bulk operations"""
        data = {
            "async_key1": test_data.simple_string,
            "async_key2": test_data.simple_int,
            "async_key3": test_data.simple_dict
        }
        
        # Async set many
        result = await async_redis_backend.aset_many(data)
        assert result is True
        
        # Async get many
        retrieved = await async_redis_backend.aget_many(list(data.keys()))
        assert len(retrieved) == 3
        for key, value in data.items():
            assert retrieved[key] == value
    
    @pytest.mark.asyncio
    async def test_async_health_check(self, async_redis_backend):
        """Test async health check"""
        health = await async_redis_backend.ahealth_check()
        assert health["status"] == "healthy"
        assert "latency_ms" in health


class TestMemoryCacheBackend:
    """Test memory cache backend implementation"""
    
    @pytest.fixture
    def memory_backend(self):
        """Memory backend fixture"""
        return MemoryCacheBackend(
            max_size=1000,
            default_ttl=3600,
            cleanup_interval=60
        )
    
    @pytest.fixture
    def lru_backend(self):
        """LRU memory backend fixture"""
        return LRUCacheBackend(
            max_size=100,
            default_ttl=1800
        )
    
    @pytest.fixture
    def test_data(self):
        """Test data fixture"""
        return CacheTestData()
    
    def test_memory_backend_initialization(self):
        """Test memory backend initialization"""
        backend = MemoryCacheBackend(
            max_size=500,
            default_ttl=1800,
            cleanup_interval=30
        )
        
        assert backend.max_size == 500
        assert backend.default_ttl == 1800
        assert backend.cleanup_interval == 30
    
    def test_memory_set_and_get(self, memory_backend, test_data):
        """Test memory cache set and get operations"""
        key = "memory_key"
        value = test_data.complex_object
        
        # Set value
        result = memory_backend.set(key, value)
        assert result is True
        
        # Get value
        retrieved = memory_backend.get(key)
        assert retrieved == value
    
    def test_memory_ttl_expiration(self, memory_backend, test_data):
        """Test TTL expiration in memory cache"""
        key = "ttl_key"
        value = test_data.simple_string
        
        # Set with short TTL
        memory_backend.set(key, value, ttl=1)
        
        # Verify exists
        assert memory_backend.exists(key) is True
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Verify expired
        assert memory_backend.exists(key) is False
        assert memory_backend.get(key) is None
    
    def test_memory_max_size_eviction(self, lru_backend, test_data):
        """Test max size eviction in memory cache"""
        # Fill cache to capacity
        for i in range(100):
            lru_backend.set(f"key_{i}", test_data.simple_string)
        
        # Add one more item (should evict oldest)
        lru_backend.set("new_key", test_data.simple_string)
        
        # Check that oldest item was evicted
        assert lru_backend.exists("key_0") is False
        assert lru_backend.exists("new_key") is True
    
    def test_lru_eviction_order(self, lru_backend):
        """Test LRU eviction order"""
        # Add items
        for i in range(5):
            lru_backend.set(f"lru_key_{i}", f"value_{i}")
        
        # Access some items to change LRU order
        lru_backend.get("lru_key_0")  # Make key_0 most recently used
        lru_backend.get("lru_key_2")  # Make key_2 second most recently used
        
        # Fill cache to trigger eviction
        for i in range(96):  # Fill remaining slots + 1 to trigger eviction
            lru_backend.set(f"fill_key_{i}", f"fill_value_{i}")
        
        # Check that least recently used items were evicted first
        assert lru_backend.exists("lru_key_0") is True  # Should still exist (most recent)
        assert lru_backend.exists("lru_key_2") is True  # Should still exist (second most recent)
        assert lru_backend.exists("lru_key_1") is False  # Should be evicted (least recent)
    
    def test_memory_cache_stats(self, memory_backend, test_data):
        """Test memory cache statistics"""
        # Perform operations
        memory_backend.set("stats_key1", test_data.simple_string)
        memory_backend.set("stats_key2", test_data.simple_int)
        memory_backend.get("stats_key1")  # Hit
        memory_backend.get("nonexistent")  # Miss
        
        stats = memory_backend.get_stats()
        assert stats["total_items"] >= 2
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1
        assert stats["hit_rate"] > 0
    
    def test_memory_cache_cleanup(self, memory_backend, test_data):
        """Test memory cache cleanup of expired items"""
        # Set items with short TTL
        for i in range(5):
            memory_backend.set(f"cleanup_key_{i}", test_data.simple_string, ttl=1)
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Trigger cleanup
        memory_backend._cleanup_expired()
        
        # Verify all expired items are cleaned up
        for i in range(5):
            assert memory_backend.exists(f"cleanup_key_{i}") is False
    
    def test_memory_thread_safety(self, memory_backend, test_data):
        """Test memory cache thread safety"""
        import threading
        import time
        
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(10):
                    key = f"thread_{worker_id}_key_{i}"
                    value = f"thread_{worker_id}_value_{i}"
                    
                    # Set
                    memory_backend.set(key, value)
                    
                    # Get
                    retrieved = memory_backend.get(key)
                    if retrieved == value:
                        results.append(True)
                    else:
                        results.append(False)
                        
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert all(results), "Some operations failed in threaded environment"


class TestHybridCacheBackend:
    """Test hybrid cache backend (memory + Redis)"""
    
    @pytest.fixture
    def redis_client(self):
        """Mock Redis client fixture"""
        return FakeRedis(decode_responses=False)
    
    @pytest.fixture
    def hybrid_backend(self, redis_client):
        """Hybrid backend fixture"""
        memory_backend = MemoryCacheBackend(max_size=100, default_ttl=3600)
        redis_backend = RedisCacheBackend(client=redis_client, default_ttl=7200)
        
        return HybridCacheBackend(
            l1_cache=memory_backend,
            l2_cache=redis_backend,
            write_through=True,
            promotion_threshold=2
        )
    
    @pytest.fixture
    def test_data(self):
        """Test data fixture"""
        return CacheTestData()
    
    def test_hybrid_cache_initialization(self, redis_client):
        """Test hybrid cache initialization"""
        memory_backend = MemoryCacheBackend(max_size=50)
        redis_backend = RedisCacheBackend(client=redis_client)
        
        hybrid = HybridCacheBackend(
            l1_cache=memory_backend,
            l2_cache=redis_backend,
            write_through=True,
            write_behind=False,
            promotion_threshold=3
        )
        
        assert hybrid.l1_cache == memory_backend
        assert hybrid.l2_cache == redis_backend
        assert hybrid.write_through is True
        assert hybrid.promotion_threshold == 3
    
    def test_hybrid_write_through(self, hybrid_backend, test_data):
        """Test write-through caching behavior"""
        key = "hybrid_key"
        value = test_data.complex_object
        
        # Set value (should write to both L1 and L2)
        result = hybrid_backend.set(key, value)
        assert result is True
        
        # Check both caches have the value
        assert hybrid_backend.l1_cache.exists(key) is True
        assert hybrid_backend.l2_cache.exists(key) is True
        
        # Get from hybrid should return from L1 (faster)
        retrieved = hybrid_backend.get(key)
        assert retrieved == value
    
    def test_hybrid_cache_miss_promotion(self, hybrid_backend, test_data):
        """Test cache miss and promotion from L2 to L1"""
        key = "promotion_key"
        value = test_data.simple_dict
        
        # Set only in L2 cache (simulating L1 miss)
        hybrid_backend.l2_cache.set(key, value)
        
        # First get (should promote to L1)
        retrieved = hybrid_backend.get(key)
        assert retrieved == value
        
        # Check that value is now in L1
        assert hybrid_backend.l1_cache.exists(key) is True
    
    def test_hybrid_l1_eviction_fallback(self, hybrid_backend, test_data):
        """Test fallback to L2 when L1 cache is full"""
        # Fill L1 cache to capacity
        for i in range(100):
            hybrid_backend.set(f"fill_key_{i}", test_data.simple_string)
        
        # Set additional items (should evict from L1 but remain in L2)
        for i in range(10):
            hybrid_backend.set(f"extra_key_{i}", test_data.simple_string)
        
        # Try to get an evicted item (should retrieve from L2)
        retrieved = hybrid_backend.get("fill_key_0")
        assert retrieved == test_data.simple_string
    
    def test_hybrid_cache_delete(self, hybrid_backend, test_data):
        """Test deletion from hybrid cache"""
        key = "delete_hybrid_key"
        value = test_data.simple_string
        
        # Set in both caches
        hybrid_backend.set(key, value)
        
        # Verify exists in both
        assert hybrid_backend.l1_cache.exists(key) is True
        assert hybrid_backend.l2_cache.exists(key) is True
        
        # Delete
        result = hybrid_backend.delete(key)
        assert result is True
        
        # Verify deleted from both
        assert hybrid_backend.l1_cache.exists(key) is False
        assert hybrid_backend.l2_cache.exists(key) is False
    
    def test_hybrid_cache_stats(self, hybrid_backend, test_data):
        """Test hybrid cache statistics aggregation"""
        # Perform operations
        hybrid_backend.set("stats_key1", test_data.simple_string)
        hybrid_backend.get("stats_key1")  # L1 hit
        hybrid_backend.l2_cache.set("stats_key2", test_data.simple_int)
        hybrid_backend.get("stats_key2")  # L1 miss, L2 hit
        hybrid_backend.get("nonexistent")  # Miss both
        
        stats = hybrid_backend.get_stats()
        assert "l1_stats" in stats
        assert "l2_stats" in stats
        assert "total_hits" in stats
        assert "total_misses" in stats
        assert "overall_hit_rate" in stats


class TestDistributedCacheBackend:
    """Test distributed cache backend"""
    
    @pytest.fixture
    def distributed_backend(self):
        """Distributed backend fixture with multiple nodes"""
        nodes = []
        for i in range(3):
            client = FakeRedis(decode_responses=False)
            backend = RedisCacheBackend(client=client, key_prefix=f"node{i}:")
            nodes.append(backend)
        
        return DistributedCacheBackend(
            nodes=nodes,
            replication_factor=2,
            consistency_level="quorum",
            hash_function="sha256"
        )
    
    @pytest.fixture
    def test_data(self):
        """Test data fixture"""
        return CacheTestData()
    
    def test_distributed_cache_initialization(self):
        """Test distributed cache initialization"""
        nodes = [Mock() for _ in range(3)]
        
        distributed = DistributedCacheBackend(
            nodes=nodes,
            replication_factor=2,
            consistency_level="strong",
            hash_function="md5"
        )
        
        assert len(distributed.nodes) == 3
        assert distributed.replication_factor == 2
        assert distributed.consistency_level == "strong"
    
    def test_distributed_consistent_hashing(self, distributed_backend, test_data):
        """Test consistent hashing for key distribution"""
        keys = [f"dist_key_{i}" for i in range(10)]
        
        # Set multiple keys
        for key in keys:
            result = distributed_backend.set(key, test_data.simple_string)
            assert result is True
        
        # Verify keys are distributed across nodes
        node_counts = [0, 0, 0]
        for key in keys:
            node_index = distributed_backend._get_primary_node(key)
            node_counts[node_index] += 1
        
        # Check that distribution is reasonably balanced
        assert all(count > 0 for count in node_counts), "Keys should be distributed across all nodes"
    
    def test_distributed_replication(self, distributed_backend, test_data):
        """Test data replication across nodes"""
        key = "replicated_key"
        value = test_data.complex_object
        
        # Set value
        result = distributed_backend.set(key, value)
        assert result is True
        
        # Check replication factor compliance
        node_index = distributed_backend._get_primary_node(key)
        replica_nodes = distributed_backend._get_replica_nodes(key)
        
        # Verify value exists on primary and replica nodes
        assert len(replica_nodes) >= distributed_backend.replication_factor - 1
        
        # Get value (should succeed even if one node fails)
        retrieved = distributed_backend.get(key)
        assert retrieved == value
    
    def test_distributed_node_failure_handling(self, distributed_backend, test_data):
        """Test handling of node failures"""
        key = "failure_test_key"
        value = test_data.simple_dict
        
        # Set value
        distributed_backend.set(key, value)
        
        # Simulate node failure by replacing one node with a broken mock
        broken_node = Mock()
        broken_node.get.side_effect = ConnectionError("Node failed")
        broken_node.set.side_effect = ConnectionError("Node failed")
        broken_node.health_check.return_value = {"status": "unhealthy"}
        
        original_node = distributed_backend.nodes[0]
        distributed_backend.nodes[0] = broken_node
        
        try:
            # Should still be able to get value from healthy replicas
            retrieved = distributed_backend.get(key)
            assert retrieved == value
        finally:
            # Restore original node
            distributed_backend.nodes[0] = original_node
    
    def test_distributed_quorum_consistency(self, distributed_backend, test_data):
        """Test quorum-based consistency"""
        key = "quorum_key"
        value = test_data.simple_string
        
        # Set with quorum consistency
        result = distributed_backend.set(key, value)
        assert result is True
        
        # Verify quorum read
        retrieved = distributed_backend.get(key)
        assert retrieved == value
        
        # Check that majority of nodes have the value
        successful_reads = 0
        for node in distributed_backend.nodes:
            try:
                node_value = node.get(key)
                if node_value == value:
                    successful_reads += 1
            except:
                pass
        
        quorum_size = len(distributed_backend.nodes) // 2 + 1
        assert successful_reads >= quorum_size
    
    def test_distributed_bulk_operations(self, distributed_backend, test_data):
        """Test distributed bulk operations"""
        data = {
            f"bulk_key_{i}": f"bulk_value_{i}"
            for i in range(20)
        }
        
        # Set many
        result = distributed_backend.set_many(data)
        assert result is True
        
        # Get many
        retrieved = distributed_backend.get_many(list(data.keys()))
        assert len(retrieved) == len(data)
        
        for key, value in data.items():
            assert retrieved[key] == value
    
    def test_distributed_health_monitoring(self, distributed_backend):
        """Test distributed cache health monitoring"""
        health = distributed_backend.health_check()
        
        assert "cluster_status" in health
        assert "node_health" in health
        assert "replication_status" in health
        assert len(health["node_health"]) == len(distributed_backend.nodes)
        
        # All nodes should be healthy in test environment
        for node_health in health["node_health"]:
            assert node_health["status"] == "healthy"


@pytest.mark.performance
class TestCacheBackendPerformance:
    """Performance tests for cache backends"""
    
    @pytest.fixture
    def performance_data(self):
        """Generate performance test data"""
        return {
            "small_data": "x" * 100,
            "medium_data": "x" * 10000,
            "large_data": "x" * 100000,
            "json_data": {"key": "value"} * 1000,
        }
    
    def test_redis_performance_benchmark(self, performance_data):
        """Benchmark Redis cache performance"""
        client = FakeRedis(decode_responses=False)
        backend = RedisCacheBackend(client=client)
        
        # Warm up
        for i in range(10):
            backend.set(f"warmup_{i}", performance_data["small_data"])
        
        # Benchmark set operations
        start_time = time.time()
        for i in range(1000):
            backend.set(f"perf_key_{i}", performance_data["medium_data"])
        set_duration = time.time() - start_time
        
        # Benchmark get operations
        start_time = time.time()
        for i in range(1000):
            backend.get(f"perf_key_{i}")
        get_duration = time.time() - start_time
        
        # Performance assertions (adjust based on environment)
        assert set_duration < 5.0, f"Set operations too slow: {set_duration}s"
        assert get_duration < 5.0, f"Get operations too slow: {get_duration}s"
        
        print(f"Redis Performance - Set: {set_duration:.3f}s, Get: {get_duration:.3f}s")
    
    def test_memory_cache_performance(self, performance_data):
        """Benchmark memory cache performance"""
        backend = MemoryCacheBackend(max_size=10000)
        
        # Benchmark set operations
        start_time = time.time()
        for i in range(1000):
            backend.set(f"mem_key_{i}", performance_data["medium_data"])
        set_duration = time.time() - start_time
        
        # Benchmark get operations
        start_time = time.time()
        for i in range(1000):
            backend.get(f"mem_key_{i}")
        get_duration = time.time() - start_time
        
        # Memory cache should be much faster
        assert set_duration < 1.0, f"Memory set operations too slow: {set_duration}s"
        assert get_duration < 0.5, f"Memory get operations too slow: {get_duration}s"
        
        print(f"Memory Performance - Set: {set_duration:.3f}s, Get: {get_duration:.3f}s")
    
    def test_cache_size_impact(self, performance_data):
        """Test performance impact of different data sizes"""
        client = FakeRedis(decode_responses=False)
        backend = RedisCacheBackend(client=client)
        
        sizes = ["small_data", "medium_data", "large_data"]
        results = {}
        
        for size_key in sizes:
            data = performance_data[size_key]
            
            # Benchmark for this size
            start_time = time.time()
            for i in range(100):
                backend.set(f"size_test_{size_key}_{i}", data)
            duration = time.time() - start_time
            
            results[size_key] = duration
            print(f"Size {size_key}: {duration:.3f}s for 100 operations")
        
        # Larger data should take longer, but not excessively
        assert results["small_data"] < results["medium_data"] < results["large_data"]


# Utility functions for testing
def create_test_cache_backend(backend_type: str, **kwargs):
    """Factory function to create test cache backends"""
    if backend_type == "redis":
        client = FakeRedis(decode_responses=False)
        return RedisCacheBackend(client=client, **kwargs)
    elif backend_type == "memory":
        return MemoryCacheBackend(**kwargs)
    elif backend_type == "lru":
        return LRUCacheBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


class MockCacheBackend(BaseCacheBackend):
    """Mock cache backend for testing"""
    
    def __init__(self):
        self.data = {}
        self.call_log = []
    
    def get(self, key: str) -> Any:
        self.call_log.append(("get", key))
        return self.data.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        self.call_log.append(("set", key, value, ttl))
        self.data[key] = value
        return True
    
    def delete(self, key: str) -> bool:
        self.call_log.append(("delete", key))
        return self.data.pop(key, None) is not None
    
    def exists(self, key: str) -> bool:
        self.call_log.append(("exists", key))
        return key in self.data
    
    def clear(self) -> bool:
        self.call_log.append(("clear",))
        self.data.clear()
        return True
    
    def health_check(self) -> Dict[str, Any]:
        return {"status": "healthy", "items": len(self.data)}


# Test configuration
@pytest.fixture(scope="session")
def cache_test_config():
    """Test configuration for cache tests"""
    return {
        "redis": {
            "host": "localhost",
            "port": 6379,
            "db": 15,  # Use separate DB for tests
            "decode_responses": False
        },
        "memory": {
            "max_size": 1000,
            "default_ttl": 3600
        },
        "hybrid": {
            "l1_max_size": 100,
            "l2_default_ttl": 7200,
            "write_through": True
        }
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
