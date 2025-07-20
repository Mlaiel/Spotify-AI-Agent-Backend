#!/usr/bin/env python3
"""
Cache Manager for PagerDuty Integration.

Advanced Redis-based caching system with intelligent TTL management,
cache warming, eviction policies, and performance optimization.

Features:
- Redis-based distributed caching
- Intelligent TTL management
- Cache warming and preloading
- LRU eviction policies
- Cache statistics and monitoring
- Compression for large values
- Encryption for sensitive data
- Multi-tier caching strategy
"""

import asyncio
import json
import logging
import pickle
import time
import zlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
import hashlib

try:
    import redis.asyncio as redis
    import redis as sync_redis
except ImportError:
    redis = None
    sync_redis = None

from .encryption import SecurityManager

logger = logging.getLogger(__name__)


class CacheError(Exception):
    """Base exception for cache-related errors."""
    pass


class CacheConnectionError(CacheError):
    """Exception raised when cache connection fails."""
    pass


class CacheSerializationError(CacheError):
    """Exception raised when serialization/deserialization fails."""
    pass


@dataclass
class CacheStats:
    """Cache statistics for monitoring."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    total_size: int = 0
    avg_ttl: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 0.0


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    size_bytes: int = 0
    compressed: bool = False
    encrypted: bool = False


class CacheManager:
    """
    Advanced cache manager with Redis backend.
    
    Features:
    - Multi-tier caching (memory + Redis)
    - Intelligent compression and encryption
    - Cache warming and preloading
    - Performance monitoring
    - Automatic expiration and cleanup
    """
    
    def __init__(self,
                 redis_url: str = "redis://localhost:6379",
                 default_ttl: int = 300,
                 max_memory_cache_size: int = 1000,
                 compression_threshold: int = 1024,
                 enable_encryption: bool = False,
                 key_prefix: str = "pagerduty:",
                 pool_size: int = 10):
        """
        Initialize cache manager.
        
        Args:
            redis_url: Redis connection URL
            default_ttl: Default TTL in seconds
            max_memory_cache_size: Max items in memory cache
            compression_threshold: Compress values larger than this
            enable_encryption: Enable encryption for sensitive data
            key_prefix: Prefix for all cache keys
            pool_size: Redis connection pool size
        """
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.max_memory_cache_size = max_memory_cache_size
        self.compression_threshold = compression_threshold
        self.enable_encryption = enable_encryption
        self.key_prefix = key_prefix
        self.pool_size = pool_size
        
        # Redis connections
        self.redis_client = None
        self.sync_redis_client = None
        
        # Memory cache for frequently accessed items
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        
        # Security manager for encryption
        self.security_manager = SecurityManager() if enable_encryption else None
        
        # Statistics
        self.stats = CacheStats()
        
        # Cache warming configuration
        self.warm_cache_patterns: List[str] = []
        self.warm_cache_callbacks: Dict[str, Callable] = {}
        
        logger.info(f"Cache manager initialized with Redis: {redis_url}")
    
    async def _get_redis_client(self):
        """Get or create async Redis client."""
        if self.redis_client is None:
            if redis is None:
                raise CacheConnectionError("Redis library not available")
            
            try:
                self.redis_client = redis.from_url(
                    self.redis_url,
                    max_connections=self.pool_size,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                # Test connection
                await self.redis_client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise CacheConnectionError(f"Redis connection failed: {e}")
        
        return self.redis_client
    
    def _get_sync_redis_client(self):
        """Get or create sync Redis client."""
        if self.sync_redis_client is None:
            if sync_redis is None:
                raise CacheConnectionError("Redis library not available")
            
            try:
                self.sync_redis_client = sync_redis.from_url(
                    self.redis_url,
                    max_connections=self.pool_size,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                # Test connection
                self.sync_redis_client.ping()
                logger.info("Sync Redis connection established")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise CacheConnectionError(f"Redis connection failed: {e}")
        
        return self.sync_redis_client
    
    def _make_key(self, key: str) -> str:
        """Create cache key with prefix."""
        return f"{self.key_prefix}{key}"
    
    def _serialize_value(self, value: Any, compress: bool = False, encrypt: bool = False) -> bytes:
        """Serialize value with optional compression and encryption."""
        try:
            # Serialize to JSON first, then pickle if needed
            try:
                serialized = json.dumps(value).encode('utf-8')
            except (TypeError, ValueError):
                serialized = pickle.dumps(value)
            
            # Compress if enabled and value is large enough
            if compress and len(serialized) > self.compression_threshold:
                serialized = zlib.compress(serialized)
            
            # Encrypt if enabled
            if encrypt and self.security_manager:
                serialized = self.security_manager.encrypt(serialized)
            
            return serialized
            
        except Exception as e:
            raise CacheSerializationError(f"Failed to serialize value: {e}")
    
    def _deserialize_value(self, data: bytes, compressed: bool = False, encrypted: bool = False) -> Any:
        """Deserialize value with optional decompression and decryption."""
        try:
            # Decrypt if needed
            if encrypted and self.security_manager:
                data = self.security_manager.decrypt(data)
            
            # Decompress if needed
            if compressed:
                data = zlib.decompress(data)
            
            # Try JSON first, then pickle
            try:
                return json.loads(data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return pickle.loads(data)
                
        except Exception as e:
            raise CacheSerializationError(f"Failed to deserialize value: {e}")
    
    def _update_memory_cache(self, key: str, entry: CacheEntry):
        """Update memory cache with LRU eviction."""
        # Remove if already exists
        if key in self.memory_cache:
            self.access_order.remove(key)
        
        # Add to cache
        self.memory_cache[key] = entry
        self.access_order.append(key)
        
        # Evict if over limit
        while len(self.memory_cache) > self.max_memory_cache_size:
            oldest_key = self.access_order.pop(0)
            del self.memory_cache[oldest_key]
    
    def _get_from_memory_cache(self, key: str) -> Optional[CacheEntry]:
        """Get value from memory cache."""
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            
            # Check if expired
            if entry.expires_at and datetime.utcnow() > entry.expires_at:
                del self.memory_cache[key]
                self.access_order.remove(key)
                return None
            
            # Update access info
            entry.access_count += 1
            entry.last_accessed = datetime.utcnow()
            
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            
            return entry
        
        return None
    
    async def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        try:
            cache_key = self._make_key(key)
            
            # Check memory cache first
            memory_entry = self._get_from_memory_cache(cache_key)
            if memory_entry:
                self.stats.hits += 1
                return memory_entry.value
            
            # Check Redis cache
            redis_client = await self._get_redis_client()
            
            # Get value and metadata
            pipe = redis_client.pipeline()
            pipe.get(cache_key)
            pipe.hgetall(f"{cache_key}:meta")
            results = await pipe.execute()
            
            data, meta = results
            
            if data is None:
                self.stats.misses += 1
                return default
            
            # Deserialize value
            compressed = meta.get(b'compressed') == b'true'
            encrypted = meta.get(b'encrypted') == b'true'
            
            value = self._deserialize_value(data, compressed, encrypted)
            
            # Create cache entry for memory cache
            expires_at = None
            if meta.get(b'expires_at'):
                expires_at = datetime.fromisoformat(meta[b'expires_at'].decode())
            
            entry = CacheEntry(
                key=cache_key,
                value=value,
                created_at=datetime.fromisoformat(meta[b'created_at'].decode()),
                expires_at=expires_at,
                compressed=compressed,
                encrypted=encrypted,
                size_bytes=len(data)
            )
            
            # Update memory cache
            self._update_memory_cache(cache_key, entry)
            
            self.stats.hits += 1
            return value
            
        except Exception as e:
            self.stats.errors += 1
            logger.error(f"Cache get error for key '{key}': {e}")
            return default
    
    async def set(self, 
                  key: str, 
                  value: Any, 
                  ttl: Optional[int] = None,
                  compress: Optional[bool] = None,
                  encrypt: Optional[bool] = None) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            compress: Force compression (auto-detect if None)
            encrypt: Force encryption (use default if None)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cache_key = self._make_key(key)
            ttl = ttl or self.default_ttl
            
            # Determine compression and encryption
            if compress is None:
                compress = True  # Let serialize method decide
            if encrypt is None:
                encrypt = self.enable_encryption
            
            # Serialize value
            serialized = self._serialize_value(value, compress, encrypt)
            actual_compressed = len(serialized) != len(str(value).encode())
            
            # Set expiration time
            expires_at = datetime.utcnow() + timedelta(seconds=ttl)
            
            # Store in Redis
            redis_client = await self._get_redis_client()
            
            pipe = redis_client.pipeline()
            pipe.setex(cache_key, ttl, serialized)
            
            # Store metadata
            meta = {
                'created_at': datetime.utcnow().isoformat(),
                'expires_at': expires_at.isoformat(),
                'compressed': str(actual_compressed).lower(),
                'encrypted': str(encrypt).lower(),
                'size_bytes': len(serialized)
            }
            pipe.hset(f"{cache_key}:meta", mapping=meta)
            pipe.expire(f"{cache_key}:meta", ttl)
            
            await pipe.execute()
            
            # Update memory cache
            entry = CacheEntry(
                key=cache_key,
                value=value,
                created_at=datetime.utcnow(),
                expires_at=expires_at,
                compressed=actual_compressed,
                encrypted=encrypt,
                size_bytes=len(serialized)
            )
            self._update_memory_cache(cache_key, entry)
            
            self.stats.sets += 1
            self.stats.total_size += len(serialized)
            return True
            
        except Exception as e:
            self.stats.errors += 1
            logger.error(f"Cache set error for key '{key}': {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            cache_key = self._make_key(key)
            
            # Remove from memory cache
            if cache_key in self.memory_cache:
                del self.memory_cache[cache_key]
                self.access_order.remove(cache_key)
            
            # Remove from Redis
            redis_client = await self._get_redis_client()
            
            pipe = redis_client.pipeline()
            pipe.delete(cache_key)
            pipe.delete(f"{cache_key}:meta")
            results = await pipe.execute()
            
            self.stats.deletes += 1
            return results[0] > 0
            
        except Exception as e:
            self.stats.errors += 1
            logger.error(f"Cache delete error for key '{key}': {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            cache_key = self._make_key(key)
            
            # Check memory cache first
            if cache_key in self.memory_cache:
                entry = self.memory_cache[cache_key]
                if not entry.expires_at or datetime.utcnow() <= entry.expires_at:
                    return True
            
            # Check Redis
            redis_client = await self._get_redis_client()
            return await redis_client.exists(cache_key) > 0
            
        except Exception as e:
            logger.error(f"Cache exists error for key '{key}': {e}")
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration time for a key."""
        try:
            cache_key = self._make_key(key)
            
            # Update memory cache
            if cache_key in self.memory_cache:
                entry = self.memory_cache[cache_key]
                entry.expires_at = datetime.utcnow() + timedelta(seconds=ttl)
            
            # Update Redis
            redis_client = await self._get_redis_client()
            return await redis_client.expire(cache_key, ttl)
            
        except Exception as e:
            logger.error(f"Cache expire error for key '{key}': {e}")
            return False
    
    async def clear(self, pattern: Optional[str] = None) -> int:
        """
        Clear cache entries.
        
        Args:
            pattern: Key pattern to match (None for all)
            
        Returns:
            Number of keys deleted
        """
        try:
            redis_client = await self._get_redis_client()
            
            if pattern:
                # Clear specific pattern
                search_pattern = self._make_key(pattern)
                keys = await redis_client.keys(search_pattern)
                meta_keys = await redis_client.keys(f"{search_pattern}:meta")
                all_keys = keys + meta_keys
            else:
                # Clear all cache keys
                keys = await redis_client.keys(f"{self.key_prefix}*")
                all_keys = keys
            
            if all_keys:
                deleted = await redis_client.delete(*all_keys)
            else:
                deleted = 0
            
            # Clear memory cache
            if pattern:
                # Clear matching keys from memory cache
                to_remove = [k for k in self.memory_cache.keys() 
                           if k.startswith(self._make_key(pattern.replace('*', '')))]
                for k in to_remove:
                    del self.memory_cache[k]
                    if k in self.access_order:
                        self.access_order.remove(k)
            else:
                # Clear all memory cache
                self.memory_cache.clear()
                self.access_order.clear()
            
            logger.info(f"Cleared {deleted} cache entries")
            return deleted
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return 0
    
    async def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        try:
            redis_client = await self._get_redis_client()
            info = await redis_client.info('memory')
            
            # Update total size from Redis info
            self.stats.total_size = info.get('used_memory', 0)
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return self.stats
    
    def add_warm_cache_pattern(self, pattern: str, callback: Callable[[str], Any]):
        """Add pattern for cache warming."""
        self.warm_cache_patterns.append(pattern)
        self.warm_cache_callbacks[pattern] = callback
    
    async def warm_cache(self):
        """Warm cache with predefined patterns."""
        for pattern in self.warm_cache_patterns:
            try:
                callback = self.warm_cache_callbacks[pattern]
                
                # Generate cache data
                cache_data = await callback(pattern)
                
                if isinstance(cache_data, dict):
                    for key, value in cache_data.items():
                        await self.set(key, value)
                
                logger.info(f"Cache warmed for pattern: {pattern}")
                
            except Exception as e:
                logger.error(f"Failed to warm cache for pattern '{pattern}': {e}")
    
    # Synchronous wrapper methods
    
    def get_sync(self, key: str, default: Any = None) -> Any:
        """Synchronous version of get."""
        return asyncio.run(self.get(key, default))
    
    def set_sync(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Synchronous version of set."""
        return asyncio.run(self.set(key, value, ttl))
    
    def delete_sync(self, key: str) -> bool:
        """Synchronous version of delete."""
        return asyncio.run(self.delete(key))
    
    # Context manager support
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def close(self):
        """Close Redis connections."""
        if self.redis_client:
            await self.redis_client.close()
        
        if self.sync_redis_client:
            self.sync_redis_client.close()
        
        logger.info("Cache manager closed")


# Singleton instance for global use
_cache_manager_instance = None

def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager_instance
    if _cache_manager_instance is None:
        _cache_manager_instance = CacheManager()
    return _cache_manager_instance


# Decorator for caching function results
def cached(ttl: int = 300, key_prefix: str = "func:"):
    """
    Decorator to cache function results.
    
    Args:
        ttl: Cache TTL in seconds
        key_prefix: Prefix for cache keys
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            cache = get_cache_manager()
            
            # Generate cache key
            key_data = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            cache_key = f"{key_prefix}{hashlib.md5(key_data.encode()).hexdigest()}"
            
            # Try to get from cache
            result = await cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, ttl)
            
            return result
        
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
