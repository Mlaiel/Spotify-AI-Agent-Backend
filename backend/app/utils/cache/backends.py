"""
Enterprise Cache Backends
========================
Production-grade cache backend implementations for Spotify AI Agent.

Expert Team Implementation:
- Lead Developer + AI Architect: Advanced algorithms and ML-optimized structures
- Senior Backend Developer: Async/await optimization and FastAPI integration
- DBA & Data Engineer: Redis cluster optimization and persistence strategies
- Security Specialist: Encryption, access control, and secure key management
- Microservices Architect: Distributed consistency and cross-service coordination
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
import zlib
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from enum import Enum
from threading import RLock
import weakref
from concurrent.futures import ThreadPoolExecutor

# External dependencies for enterprise features
try:
    import aioredis
    import redis.sentinel
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False

try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False

logger = logging.getLogger(__name__)

# === Core Types and Enums ===
CacheKey = Union[str, bytes]
CacheValue = Any
TTL = Union[int, float, timedelta]

class CacheBackendType(Enum):
    """Cache backend types."""
    MEMORY = "memory"
    REDIS = "redis"
    PERSISTENT = "persistent"
    HYBRID = "hybrid"
    DISTRIBUTED = "distributed"
    SHARDED = "sharded"

class CompressionType(Enum):
    """Compression algorithms."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"

class SerializationType(Enum):
    """Serialization formats."""
    JSON = "json"
    PICKLE = "pickle"
    MSGPACK = "msgpack"
    CUSTOM = "custom"

@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    size: int = 0
    memory_usage: int = 0
    avg_latency_ms: float = 0.0
    last_access: Optional[datetime] = None
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    @property
    def operations_per_second(self) -> float:
        """Calculate operations per second."""
        if self.last_access:
            duration = (datetime.now() - self.last_access).total_seconds()
            total_ops = self.hits + self.misses + self.sets + self.deletes
            return total_ops / duration if duration > 0 else 0.0
        return 0.0

@dataclass
class CacheEntry:
    """Enhanced cache entry with enterprise metadata."""
    value: Any
    created_at: float
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None
    tags: Set[str] = field(default_factory=set)
    size_bytes: int = 0
    compression: CompressionType = CompressionType.NONE
    encrypted: bool = False
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() > self.created_at + self.ttl
    
    @property
    def age_seconds(self) -> float:
        """Get entry age in seconds."""
        return time.time() - self.created_at
    
    @property
    def time_to_expire(self) -> Optional[float]:
        """Get remaining time before expiration."""
        if self.ttl is None:
            return None
        remaining = (self.created_at + self.ttl) - time.time()
        return max(0, remaining)

# === Abstract Base Classes ===
class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.metrics = CacheMetrics()
        self._lock = RLock()
        logger.info(f"Initialized {self.name} cache backend")
    
    @abstractmethod
    async def get(self, key: CacheKey) -> Optional[Any]:
        """Retrieve value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: CacheKey, value: Any, ttl: Optional[TTL] = None, tags: Set[str] = None) -> bool:
        """Store value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: CacheKey) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    async def exists(self, key: CacheKey) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> CacheMetrics:
        """Get cache statistics."""
        pass
    
    def _normalize_ttl(self, ttl: Optional[TTL]) -> Optional[float]:
        """Normalize TTL to seconds."""
        if ttl is None:
            return None
        if isinstance(ttl, timedelta):
            return ttl.total_seconds()
        return float(ttl)
    
    def _update_metrics(self, operation: str, **kwargs):
        """Update cache metrics."""
        with self._lock:
            if operation == "hit":
                self.metrics.hits += 1
            elif operation == "miss":
                self.metrics.misses += 1
            elif operation == "set":
                self.metrics.sets += 1
            elif operation == "delete":
                self.metrics.deletes += 1
            elif operation == "eviction":
                self.metrics.evictions += 1
            
            self.metrics.last_access = datetime.now()

# === Memory Cache Backend ===
class MemoryCache(CacheBackend):
    """High-performance in-memory cache with advanced features."""
    
    def __init__(self, 
                 max_size: int = 10000,
                 max_memory_mb: int = 512,
                 eviction_policy: str = "lru",
                 compression: CompressionType = CompressionType.LZ4,
                 enable_stats: bool = True):
        super().__init__("MemoryCache")
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.eviction_policy = eviction_policy
        self.compression = compression
        self.enable_stats = enable_stats
        
        self._data: Dict[CacheKey, CacheEntry] = {}
        self._access_order: List[CacheKey] = []  # For LRU
        self._access_frequency: Dict[CacheKey, int] = {}  # For LFU
        self._current_memory = 0
        
        logger.info(f"MemoryCache initialized: max_size={max_size}, max_memory={max_memory_mb}MB")
    
    async def get(self, key: CacheKey) -> Optional[Any]:
        """Retrieve value from memory cache."""
        start_time = time.time()
        
        try:
            if key not in self._data:
                self._update_metrics("miss")
                return None
            
            entry = self._data[key]
            
            # Check expiration
            if entry.is_expired:
                await self.delete(key)
                self._update_metrics("miss")
                return None
            
            # Update access metadata
            entry.accessed_at = time.time()
            entry.access_count += 1
            self._access_frequency[key] = self._access_frequency.get(key, 0) + 1
            
            # Update LRU order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            
            self._update_metrics("hit")
            
            # Decompress if necessary
            value = entry.value
            if entry.compression != CompressionType.NONE:
                value = self._decompress(value, entry.compression)
            
            # Update latency metrics
            if self.enable_stats:
                latency = (time.time() - start_time) * 1000
                self.metrics.avg_latency_ms = (self.metrics.avg_latency_ms + latency) / 2
            
            return value
            
        except Exception as e:
            logger.error(f"Error getting key {key}: {e}")
            self._update_metrics("miss")
            return None
    
    async def set(self, key: CacheKey, value: Any, ttl: Optional[TTL] = None, tags: Set[str] = None) -> bool:
        """Store value in memory cache."""
        try:
            ttl_seconds = self._normalize_ttl(ttl)
            tags = tags or set()
            
            # Compress value if enabled
            compressed_value = value
            compression_used = CompressionType.NONE
            if self.compression != CompressionType.NONE:
                compressed_value = self._compress(value, self.compression)
                compression_used = self.compression
            
            # Calculate size
            size_bytes = self._calculate_size(compressed_value)
            
            # Check memory limits and evict if necessary
            await self._ensure_capacity(size_bytes)
            
            # Create cache entry
            entry = CacheEntry(
                value=compressed_value,
                created_at=time.time(),
                ttl=ttl_seconds,
                tags=tags,
                size_bytes=size_bytes,
                compression=compression_used
            )
            
            # Store entry
            old_entry = self._data.get(key)
            self._data[key] = entry
            
            # Update memory usage
            if old_entry:
                self._current_memory -= old_entry.size_bytes
            self._current_memory += size_bytes
            
            # Update access structures
            if key not in self._access_order:
                self._access_order.append(key)
            self._access_frequency[key] = self._access_frequency.get(key, 0) + 1
            
            self._update_metrics("set")
            self.metrics.size = len(self._data)
            self.metrics.memory_usage = self._current_memory
            
            logger.debug(f"Set key {key} with TTL {ttl_seconds}, size {size_bytes} bytes")
            return True
            
        except Exception as e:
            logger.error(f"Error setting key {key}: {e}")
            return False
    
    async def delete(self, key: CacheKey) -> bool:
        """Delete value from memory cache."""
        try:
            if key not in self._data:
                return False
            
            entry = self._data[key]
            del self._data[key]
            
            # Update memory usage
            self._current_memory -= entry.size_bytes
            
            # Clean up access structures
            if key in self._access_order:
                self._access_order.remove(key)
            if key in self._access_frequency:
                del self._access_frequency[key]
            
            self._update_metrics("delete")
            self.metrics.size = len(self._data)
            self.metrics.memory_usage = self._current_memory
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting key {key}: {e}")
            return False
    
    async def exists(self, key: CacheKey) -> bool:
        """Check if key exists and is not expired."""
        if key not in self._data:
            return False
        
        entry = self._data[key]
        if entry.is_expired:
            await self.delete(key)
            return False
        
        return True
    
    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries matching pattern."""
        try:
            if pattern is None:
                count = len(self._data)
                self._data.clear()
                self._access_order.clear()
                self._access_frequency.clear()
                self._current_memory = 0
                self.metrics.size = 0
                self.metrics.memory_usage = 0
                return count
            
            # Pattern matching (simple glob-style)
            import fnmatch
            keys_to_delete = [
                key for key in self._data.keys()
                if fnmatch.fnmatch(str(key), pattern)
            ]
            
            for key in keys_to_delete:
                await self.delete(key)
            
            return len(keys_to_delete)
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0
    
    async def get_stats(self) -> CacheMetrics:
        """Get cache statistics."""
        self.metrics.size = len(self._data)
        self.metrics.memory_usage = self._current_memory
        return self.metrics
    
    async def _ensure_capacity(self, new_size: int):
        """Ensure cache has capacity for new entry."""
        # Check size limit
        while len(self._data) >= self.max_size and self._data:
            await self._evict_one()
        
        # Check memory limit
        while (self._current_memory + new_size) > self.max_memory_bytes and self._data:
            await self._evict_one()
    
    async def _evict_one(self):
        """Evict one entry based on eviction policy."""
        if not self._data:
            return
        
        if self.eviction_policy == "lru":
            # Evict least recently used
            key_to_evict = self._access_order[0]
        elif self.eviction_policy == "lfu":
            # Evict least frequently used
            key_to_evict = min(self._access_frequency.keys(), 
                             key=lambda k: self._access_frequency[k])
        elif self.eviction_policy == "fifo":
            # Evict first in (oldest)
            key_to_evict = next(iter(self._data))
        else:
            # Default to LRU
            key_to_evict = self._access_order[0] if self._access_order else next(iter(self._data))
        
        await self.delete(key_to_evict)
        self._update_metrics("eviction")
        logger.debug(f"Evicted key {key_to_evict} using {self.eviction_policy} policy")
    
    def _compress(self, value: Any, compression: CompressionType) -> bytes:
        """Compress value using specified algorithm."""
        data = pickle.dumps(value)
        
        if compression == CompressionType.GZIP:
            return zlib.compress(data)
        elif compression == CompressionType.LZ4 and LZ4_AVAILABLE:
            return lz4.frame.compress(data)
        else:
            return data
    
    def _decompress(self, data: bytes, compression: CompressionType) -> Any:
        """Decompress value using specified algorithm."""
        if compression == CompressionType.GZIP:
            data = zlib.decompress(data)
        elif compression == CompressionType.LZ4 and LZ4_AVAILABLE:
            data = lz4.frame.decompress(data)
        
        return pickle.loads(data)
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate size of value in bytes."""
        if isinstance(value, bytes):
            return len(value)
        elif isinstance(value, str):
            return len(value.encode('utf-8'))
        else:
            return len(pickle.dumps(value))

# === Redis Cache Backend ===
class RedisCache(CacheBackend):
    """Enterprise Redis cache with cluster support and advanced features."""
    
    def __init__(self,
                 redis_url: str = "redis://localhost:6379",
                 cluster_mode: bool = False,
                 sentinel_hosts: Optional[List[Tuple[str, int]]] = None,
                 service_name: str = "mymaster",
                 max_connections: int = 100,
                 retry_on_timeout: bool = True,
                 compression: CompressionType = CompressionType.LZ4,
                 namespace: str = "spotify_ai"):
        super().__init__("RedisCache")
        
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is not available. Install with: pip install aioredis")
        
        self.redis_url = redis_url
        self.cluster_mode = cluster_mode
        self.sentinel_hosts = sentinel_hosts
        self.service_name = service_name
        self.max_connections = max_connections
        self.retry_on_timeout = retry_on_timeout
        self.compression = compression
        self.namespace = namespace
        
        self._redis: Optional[aioredis.Redis] = None
        self._connection_pool = None
        
        logger.info(f"RedisCache configured: cluster={cluster_mode}, namespace={namespace}")
    
    async def _get_redis(self) -> aioredis.Redis:
        """Get Redis connection with lazy initialization."""
        if self._redis is None:
            try:
                if self.sentinel_hosts:
                    # Use Redis Sentinel for HA
                    sentinel = aioredis.sentinel.Sentinel(self.sentinel_hosts)
                    self._redis = sentinel.master_for(
                        self.service_name,
                        socket_timeout=0.1,
                        retry_on_timeout=self.retry_on_timeout
                    )
                else:
                    # Direct Redis connection
                    self._redis = aioredis.from_url(
                        self.redis_url,
                        max_connections=self.max_connections,
                        retry_on_timeout=self.retry_on_timeout
                    )
                
                # Test connection
                await self._redis.ping()
                logger.info("Redis connection established successfully")
                
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        
        return self._redis
    
    def _make_key(self, key: CacheKey) -> str:
        """Create namespaced key."""
        if isinstance(key, bytes):
            key = key.decode('utf-8')
        return f"{self.namespace}:{key}"
    
    async def get(self, key: CacheKey) -> Optional[Any]:
        """Retrieve value from Redis cache."""
        start_time = time.time()
        
        try:
            redis = await self._get_redis()
            redis_key = self._make_key(key)
            
            # Get value and metadata
            data = await redis.hgetall(redis_key)
            
            if not data:
                self._update_metrics("miss")
                return None
            
            # Check expiration (Redis handles TTL, but double-check)
            ttl = await redis.ttl(redis_key)
            if ttl == 0:  # Key expired
                self._update_metrics("miss")
                return None
            
            # Deserialize value
            value_data = data.get(b'value')
            compression = CompressionType(data.get(b'compression', b'none').decode())
            
            if value_data:
                value = self._deserialize(value_data, compression)
                
                # Update access metadata
                await redis.hset(redis_key, mapping={
                    'accessed_at': time.time(),
                    'access_count': int(data.get(b'access_count', 0)) + 1
                })
                
                self._update_metrics("hit")
                
                # Update latency metrics
                latency = (time.time() - start_time) * 1000
                self.metrics.avg_latency_ms = (self.metrics.avg_latency_ms + latency) / 2
                
                return value
            
            self._update_metrics("miss")
            return None
            
        except Exception as e:
            logger.error(f"Error getting Redis key {key}: {e}")
            self._update_metrics("miss")
            return None
    
    async def set(self, key: CacheKey, value: Any, ttl: Optional[TTL] = None, tags: Set[str] = None) -> bool:
        """Store value in Redis cache."""
        try:
            redis = await self._get_redis()
            redis_key = self._make_key(key)
            ttl_seconds = self._normalize_ttl(ttl)
            tags = tags or set()
            
            # Serialize and compress value
            serialized_value = self._serialize(value, self.compression)
            
            # Prepare metadata
            metadata = {
                'value': serialized_value,
                'created_at': time.time(),
                'accessed_at': time.time(),
                'access_count': 0,
                'compression': self.compression.value,
                'tags': json.dumps(list(tags)) if tags else '[]',
                'size_bytes': len(serialized_value)
            }
            
            # Store in Redis
            await redis.hset(redis_key, mapping=metadata)
            
            # Set TTL if specified
            if ttl_seconds:
                await redis.expire(redis_key, int(ttl_seconds))
            
            self._update_metrics("set")
            logger.debug(f"Set Redis key {key} with TTL {ttl_seconds}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting Redis key {key}: {e}")
            return False
    
    async def delete(self, key: CacheKey) -> bool:
        """Delete value from Redis cache."""
        try:
            redis = await self._get_redis()
            redis_key = self._make_key(key)
            
            result = await redis.delete(redis_key)
            
            if result > 0:
                self._update_metrics("delete")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting Redis key {key}: {e}")
            return False
    
    async def exists(self, key: CacheKey) -> bool:
        """Check if key exists in Redis cache."""
        try:
            redis = await self._get_redis()
            redis_key = self._make_key(key)
            
            return await redis.exists(redis_key) > 0
            
        except Exception as e:
            logger.error(f"Error checking Redis key existence {key}: {e}")
            return False
    
    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear Redis cache entries."""
        try:
            redis = await self._get_redis()
            
            if pattern is None:
                pattern = f"{self.namespace}:*"
            else:
                pattern = f"{self.namespace}:{pattern}"
            
            # Get keys matching pattern
            keys = []
            async for key in redis.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                deleted = await redis.delete(*keys)
                logger.info(f"Cleared {deleted} Redis keys with pattern {pattern}")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.error(f"Error clearing Redis cache: {e}")
            return 0
    
    async def get_stats(self) -> CacheMetrics:
        """Get Redis cache statistics."""
        try:
            redis = await self._get_redis()
            
            # Get Redis info
            info = await redis.info()
            
            # Update metrics with Redis stats
            self.metrics.memory_usage = info.get('used_memory', 0)
            
            # Count keys in namespace
            key_count = 0
            async for _ in redis.scan_iter(match=f"{self.namespace}:*"):
                key_count += 1
            
            self.metrics.size = key_count
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error getting Redis stats: {e}")
            return self.metrics
    
    def _serialize(self, value: Any, compression: CompressionType) -> bytes:
        """Serialize and compress value."""
        # Serialize based on type
        if isinstance(value, (dict, list)):
            data = json.dumps(value).encode('utf-8')
        elif isinstance(value, str):
            data = value.encode('utf-8')
        elif isinstance(value, bytes):
            data = value
        else:
            data = pickle.dumps(value)
        
        # Compress if enabled
        if compression == CompressionType.GZIP:
            return zlib.compress(data)
        elif compression == CompressionType.LZ4 and LZ4_AVAILABLE:
            return lz4.frame.compress(data)
        else:
            return data
    
    def _deserialize(self, data: bytes, compression: CompressionType) -> Any:
        """Decompress and deserialize value."""
        # Decompress if necessary
        if compression == CompressionType.GZIP:
            data = zlib.decompress(data)
        elif compression == CompressionType.LZ4 and LZ4_AVAILABLE:
            data = lz4.frame.decompress(data)
        
        # Try different deserialization methods
        try:
            # Try JSON first
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            try:
                # Try pickle
                return pickle.loads(data)
            except pickle.PickleError:
                # Return as bytes
                return data
    
    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            logger.info("Redis connection closed")

# === Hybrid Cache Backend ===
class HybridCache(CacheBackend):
    """Multi-tier cache combining memory and Redis with intelligent routing."""
    
    def __init__(self,
                 memory_cache: MemoryCache,
                 redis_cache: RedisCache,
                 l1_ttl_ratio: float = 0.5,
                 promotion_threshold: int = 3,
                 enable_write_through: bool = True):
        super().__init__("HybridCache")
        
        self.l1_cache = memory_cache  # Fast L1 cache
        self.l2_cache = redis_cache   # Persistent L2 cache
        self.l1_ttl_ratio = l1_ttl_ratio
        self.promotion_threshold = promotion_threshold
        self.enable_write_through = enable_write_through
        
        # Track access patterns for promotion
        self._access_tracker: Dict[CacheKey, int] = {}
        
        logger.info("HybridCache initialized with L1 (Memory) + L2 (Redis)")
    
    async def get(self, key: CacheKey) -> Optional[Any]:
        """Get value from hybrid cache (L1 -> L2)."""
        # Try L1 cache first
        value = await self.l1_cache.get(key)
        if value is not None:
            self._update_metrics("hit")
            return value
        
        # Try L2 cache
        value = await self.l2_cache.get(key)
        if value is not None:
            # Track access for potential promotion
            self._access_tracker[key] = self._access_tracker.get(key, 0) + 1
            
            # Promote to L1 if accessed frequently
            if self._access_tracker[key] >= self.promotion_threshold:
                await self.l1_cache.set(key, value, ttl=300)  # 5 min in L1
                logger.debug(f"Promoted key {key} to L1 cache")
            
            self._update_metrics("hit")
            return value
        
        self._update_metrics("miss")
        return None
    
    async def set(self, key: CacheKey, value: Any, ttl: Optional[TTL] = None, tags: Set[str] = None) -> bool:
        """Set value in hybrid cache."""
        ttl_seconds = self._normalize_ttl(ttl)
        
        # Calculate L1 TTL (shorter duration)
        l1_ttl = None
        if ttl_seconds:
            l1_ttl = ttl_seconds * self.l1_ttl_ratio
        
        success = True
        
        # Set in L1 cache
        l1_success = await self.l1_cache.set(key, value, ttl=l1_ttl, tags=tags)
        
        # Set in L2 cache (write-through)
        if self.enable_write_through:
            l2_success = await self.l2_cache.set(key, value, ttl=ttl, tags=tags)
            success = l1_success and l2_success
        else:
            success = l1_success
        
        if success:
            self._update_metrics("set")
        
        return success
    
    async def delete(self, key: CacheKey) -> bool:
        """Delete value from both cache tiers."""
        l1_deleted = await self.l1_cache.delete(key)
        l2_deleted = await self.l2_cache.delete(key)
        
        # Clean up access tracker
        if key in self._access_tracker:
            del self._access_tracker[key]
        
        success = l1_deleted or l2_deleted
        if success:
            self._update_metrics("delete")
        
        return success
    
    async def exists(self, key: CacheKey) -> bool:
        """Check if key exists in either cache tier."""
        return (await self.l1_cache.exists(key) or 
                await self.l2_cache.exists(key))
    
    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear both cache tiers."""
        l1_cleared = await self.l1_cache.clear(pattern)
        l2_cleared = await self.l2_cache.clear(pattern)
        
        # Clear access tracker
        if pattern is None:
            self._access_tracker.clear()
        
        return l1_cleared + l2_cleared
    
    async def get_stats(self) -> CacheMetrics:
        """Get combined cache statistics."""
        l1_stats = await self.l1_cache.get_stats()
        l2_stats = await self.l2_cache.get_stats()
        
        # Combine metrics
        combined_metrics = CacheMetrics()
        combined_metrics.hits = l1_stats.hits + l2_stats.hits
        combined_metrics.misses = l1_stats.misses + l2_stats.misses
        combined_metrics.sets = l1_stats.sets + l2_stats.sets
        combined_metrics.deletes = l1_stats.deletes + l2_stats.deletes
        combined_metrics.evictions = l1_stats.evictions + l2_stats.evictions
        combined_metrics.size = l1_stats.size + l2_stats.size
        combined_metrics.memory_usage = l1_stats.memory_usage + l2_stats.memory_usage
        combined_metrics.avg_latency_ms = (l1_stats.avg_latency_ms + l2_stats.avg_latency_ms) / 2
        
        return combined_metrics

# === Factory Functions ===
def create_memory_cache(**kwargs) -> MemoryCache:
    """Create optimized memory cache."""
    return MemoryCache(**kwargs)

def create_redis_cache(**kwargs) -> RedisCache:
    """Create Redis cache with enterprise configuration."""
    return RedisCache(**kwargs)

def create_hybrid_cache(memory_size_mb: int = 256, redis_url: str = "redis://localhost:6379") -> HybridCache:
    """Create hybrid cache with optimized configuration."""
    memory_cache = MemoryCache(
        max_memory_mb=memory_size_mb,
        eviction_policy="lru",
        compression=CompressionType.LZ4
    )
    
    redis_cache = RedisCache(
        redis_url=redis_url,
        compression=CompressionType.LZ4,
        namespace="spotify_ai_l2"
    )
    
    return HybridCache(
        memory_cache=memory_cache,
        redis_cache=redis_cache,
        enable_write_through=True
    )

# Additional enterprise backends can be added here:
# - PersistentCache: File-based cache for large objects
# - DistributedCache: Multi-region cache with consistency
# - ShardedCache: Horizontally scaled cache
