"""
Enterprise Caching System for Spotify AI Agent Alert Algorithms

This module provides intelligent caching mechanisms optimized for music streaming platforms,
including Redis-based caching, local cache with TTL, and specialized cache strategies for
real-time alert processing.

Author: Fahed Mlaiel (Expert Backend Developer & ML Engineer)
Version: 2.0.0 (Enterprise Edition)
"""

import asyncio
import hashlib
import json
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import redis.asyncio as redis
from cachetools import TTLCache, LRUCache
import logging

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategy types for different use cases"""
    REALTIME_ALERTS = "realtime_alerts"
    ML_MODELS = "ml_models"
    USER_BEHAVIOR = "user_behavior"
    AUDIO_QUALITY = "audio_quality"
    BUSINESS_METRICS = "business_metrics"


@dataclass
class CacheConfig:
    """Configuration for cache systems"""
    redis_url: str = "redis://localhost:6379/0"
    redis_pool_size: int = 50
    local_cache_size: int = 10000
    default_ttl: int = 3600  # 1 hour
    realtime_ttl: int = 60   # 1 minute for real-time data
    ml_model_ttl: int = 86400  # 24 hours for ML models
    compression_enabled: bool = True
    music_streaming_optimizations: Dict[str, Any] = field(default_factory=lambda: {
        'user_preference_ttl': 7200,     # 2 hours
        'playlist_cache_ttl': 1800,      # 30 minutes
        'audio_quality_cache_ttl': 300,  # 5 minutes
        'cdn_cache_ttl': 900,           # 15 minutes
        'revenue_metrics_ttl': 3600,    # 1 hour
    })


@dataclass
class CacheMetrics:
    """Cache performance metrics for monitoring"""
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    total_requests: int = 0
    cache_size: int = 0
    memory_usage_mb: float = 0.0
    avg_response_time_ms: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class CacheInterface(ABC):
    """Abstract interface for cache implementations"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries"""
        pass


class RedisCache(CacheInterface):
    """Redis-based cache implementation for distributed systems"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis_pool = None
        self._metrics = CacheMetrics()
        
    async def initialize(self):
        """Initialize Redis connection pool"""
        try:
            self.redis_pool = redis.ConnectionPool.from_url(
                self.config.redis_url,
                max_connections=self.config.redis_pool_size,
                decode_responses=False
            )
            self.redis_client = redis.Redis(connection_pool=self.redis_pool)
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            raise
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        start_time = time.time()
        try:
            self._metrics.total_requests += 1
            
            raw_value = await self.redis_client.get(key)
            if raw_value is None:
                self._metrics.miss_rate = (self._metrics.miss_rate * (self._metrics.total_requests - 1) + 1) / self._metrics.total_requests
                return None
            
            # Deserialize value
            if self.config.compression_enabled:
                value = pickle.loads(raw_value)
            else:
                value = json.loads(raw_value.decode('utf-8'))
            
            # Update metrics
            self._metrics.hit_rate = (self._metrics.hit_rate * (self._metrics.total_requests - 1) + 1) / self._metrics.total_requests
            response_time = (time.time() - start_time) * 1000
            self._metrics.avg_response_time_ms = (self._metrics.avg_response_time_ms + response_time) / 2
            
            logger.debug(f"Cache hit for key: {key}")
            return value
            
        except Exception as e:
            logger.error(f"Error getting value from Redis cache: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache"""
        try:
            # Serialize value
            if self.config.compression_enabled:
                serialized_value = pickle.dumps(value)
            else:
                serialized_value = json.dumps(value).encode('utf-8')
            
            # Set with TTL
            ttl = ttl or self.config.default_ttl
            await self.redis_client.setex(key, ttl, serialized_value)
            
            logger.debug(f"Cache set for key: {key} with TTL: {ttl}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting value in Redis cache: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache"""
        try:
            result = await self.redis_client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Error deleting key from Redis cache: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache"""
        try:
            return await self.redis_client.exists(key) > 0
        except Exception as e:
            logger.error(f"Error checking key existence in Redis cache: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all Redis cache entries"""
        try:
            await self.redis_client.flushdb()
            return True
        except Exception as e:
            logger.error(f"Error clearing Redis cache: {e}")
            return False
    
    async def get_metrics(self) -> CacheMetrics:
        """Get cache performance metrics"""
        try:
            info = await self.redis_client.info('memory')
            self._metrics.memory_usage_mb = info.get('used_memory', 0) / (1024 * 1024)
            self._metrics.cache_size = await self.redis_client.dbsize()
            self._metrics.last_updated = datetime.now()
        except Exception as e:
            logger.error(f"Error getting Redis cache metrics: {e}")
        
        return self._metrics


class LocalCache(CacheInterface):
    """Local in-memory cache implementation for high-speed access"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache = TTLCache(maxsize=config.local_cache_size, ttl=config.default_ttl)
        self._metrics = CacheMetrics()
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from local cache"""
        start_time = time.time()
        async with self._lock:
            self._metrics.total_requests += 1
            
            try:
                value = self.cache[key]
                self._metrics.hit_rate = (self._metrics.hit_rate * (self._metrics.total_requests - 1) + 1) / self._metrics.total_requests
                
                response_time = (time.time() - start_time) * 1000
                self._metrics.avg_response_time_ms = (self._metrics.avg_response_time_ms + response_time) / 2
                
                return value
            except KeyError:
                self._metrics.miss_rate = (self._metrics.miss_rate * (self._metrics.total_requests - 1) + 1) / self._metrics.total_requests
                return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in local cache"""
        async with self._lock:
            try:
                # TTLCache doesn't support per-item TTL, so we use default
                self.cache[key] = value
                return True
            except Exception as e:
                logger.error(f"Error setting value in local cache: {e}")
                return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from local cache"""
        async with self._lock:
            try:
                del self.cache[key]
                return True
            except KeyError:
                return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in local cache"""
        async with self._lock:
            return key in self.cache
    
    async def clear(self) -> bool:
        """Clear all local cache entries"""
        async with self._lock:
            self.cache.clear()
            return True


class MultiTierCache:
    """Multi-tier cache system combining local and Redis cache"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.local_cache = LocalCache(config)
        self.redis_cache = RedisCache(config)
        self._metrics = CacheMetrics()
    
    async def initialize(self):
        """Initialize all cache tiers"""
        await self.redis_cache.initialize()
    
    async def get(self, key: str, strategy: CacheStrategy = CacheStrategy.REALTIME_ALERTS) -> Optional[Any]:
        """Get value from multi-tier cache with strategy"""
        # Try local cache first
        value = await self.local_cache.get(key)
        if value is not None:
            logger.debug(f"Cache hit in local cache for key: {key}")
            return value
        
        # Try Redis cache
        value = await self.redis_cache.get(key)
        if value is not None:
            # Store in local cache for faster future access
            local_ttl = self._get_local_ttl(strategy)
            await self.local_cache.set(key, value, local_ttl)
            logger.debug(f"Cache hit in Redis cache for key: {key}")
            return value
        
        logger.debug(f"Cache miss for key: {key}")
        return None
    
    async def set(self, key: str, value: Any, strategy: CacheStrategy = CacheStrategy.REALTIME_ALERTS) -> bool:
        """Set value in multi-tier cache with strategy"""
        ttl = self._get_ttl(strategy)
        
        # Set in both caches
        redis_result = await self.redis_cache.set(key, value, ttl)
        local_result = await self.local_cache.set(key, value, ttl)
        
        return redis_result and local_result
    
    async def delete(self, key: str) -> bool:
        """Delete value from all cache tiers"""
        redis_result = await self.redis_cache.delete(key)
        local_result = await self.local_cache.delete(key)
        return redis_result or local_result
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in any cache tier"""
        return await self.local_cache.exists(key) or await self.redis_cache.exists(key)
    
    async def clear(self) -> bool:
        """Clear all cache tiers"""
        redis_result = await self.redis_cache.clear()
        local_result = await self.local_cache.clear()
        return redis_result and local_result
    
    def _get_ttl(self, strategy: CacheStrategy) -> int:
        """Get TTL based on cache strategy"""
        strategy_ttl_map = {
            CacheStrategy.REALTIME_ALERTS: self.config.realtime_ttl,
            CacheStrategy.ML_MODELS: self.config.ml_model_ttl,
            CacheStrategy.USER_BEHAVIOR: self.config.music_streaming_optimizations['user_preference_ttl'],
            CacheStrategy.AUDIO_QUALITY: self.config.music_streaming_optimizations['audio_quality_cache_ttl'],
            CacheStrategy.BUSINESS_METRICS: self.config.music_streaming_optimizations['revenue_metrics_ttl'],
        }
        return strategy_ttl_map.get(strategy, self.config.default_ttl)
    
    def _get_local_ttl(self, strategy: CacheStrategy) -> int:
        """Get local cache TTL (usually shorter than Redis TTL)"""
        base_ttl = self._get_ttl(strategy)
        return min(base_ttl, 300)  # Max 5 minutes for local cache
    
    async def get_metrics(self) -> Dict[str, CacheMetrics]:
        """Get metrics from all cache tiers"""
        return {
            'local': await self.local_cache.get_metrics() if hasattr(self.local_cache, 'get_metrics') else self.local_cache._metrics,
            'redis': await self.redis_cache.get_metrics(),
        }


class MusicStreamingCacheManager:
    """Specialized cache manager for music streaming platform optimizations"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache = MultiTierCache(config)
        
    async def initialize(self):
        """Initialize cache manager"""
        await self.cache.initialize()
    
    async def cache_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """Cache user preferences with optimized TTL"""
        key = f"user:preferences:{user_id}"
        return await self.cache.set(key, preferences, CacheStrategy.USER_BEHAVIOR)
    
    async def get_user_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get cached user preferences"""
        key = f"user:preferences:{user_id}"
        return await self.cache.get(key, CacheStrategy.USER_BEHAVIOR)
    
    async def cache_audio_quality_metrics(self, session_id: str, metrics: Dict[str, Any]) -> bool:
        """Cache audio quality metrics for real-time monitoring"""
        key = f"audio:quality:{session_id}"
        return await self.cache.set(key, metrics, CacheStrategy.AUDIO_QUALITY)
    
    async def get_audio_quality_metrics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get cached audio quality metrics"""
        key = f"audio:quality:{session_id}"
        return await self.cache.get(key, CacheStrategy.AUDIO_QUALITY)
    
    async def cache_playlist_recommendations(self, user_id: str, playlist_data: Dict[str, Any]) -> bool:
        """Cache playlist recommendations"""
        key = f"playlist:recommendations:{user_id}"
        return await self.cache.set(key, playlist_data, CacheStrategy.USER_BEHAVIOR)
    
    async def get_playlist_recommendations(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get cached playlist recommendations"""
        key = f"playlist:recommendations:{user_id}"
        return await self.cache.get(key, CacheStrategy.USER_BEHAVIOR)
    
    async def cache_revenue_metrics(self, metric_type: str, data: Dict[str, Any]) -> bool:
        """Cache business revenue metrics"""
        key = f"revenue:metrics:{metric_type}"
        return await self.cache.set(key, data, CacheStrategy.BUSINESS_METRICS)
    
    async def get_revenue_metrics(self, metric_type: str) -> Optional[Dict[str, Any]]:
        """Get cached revenue metrics"""
        key = f"revenue:metrics:{metric_type}"
        return await self.cache.get(key, CacheStrategy.BUSINESS_METRICS)
    
    async def cache_ml_model_predictions(self, model_name: str, input_hash: str, predictions: Any) -> bool:
        """Cache ML model predictions to avoid recomputation"""
        key = f"ml:predictions:{model_name}:{input_hash}"
        return await self.cache.set(key, predictions, CacheStrategy.ML_MODELS)
    
    async def get_ml_model_predictions(self, model_name: str, input_hash: str) -> Optional[Any]:
        """Get cached ML model predictions"""
        key = f"ml:predictions:{model_name}:{input_hash}"
        return await self.cache.get(key, CacheStrategy.ML_MODELS)
    
    async def invalidate_user_cache(self, user_id: str) -> bool:
        """Invalidate all cached data for a specific user"""
        keys_to_delete = [
            f"user:preferences:{user_id}",
            f"playlist:recommendations:{user_id}",
        ]
        
        results = []
        for key in keys_to_delete:
            results.append(await self.cache.delete(key))
        
        return all(results)
    
    async def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cache performance metrics"""
        cache_metrics = await self.cache.get_metrics()
        
        return {
            'cache_tiers': cache_metrics,
            'config': {
                'strategies': {
                    'realtime_ttl': self.config.realtime_ttl,
                    'ml_model_ttl': self.config.ml_model_ttl,
                    'user_preference_ttl': self.config.music_streaming_optimizations['user_preference_ttl'],
                    'audio_quality_ttl': self.config.music_streaming_optimizations['audio_quality_cache_ttl'],
                    'revenue_metrics_ttl': self.config.music_streaming_optimizations['revenue_metrics_ttl'],
                }
            },
            'last_updated': datetime.now().isoformat()
        }


def create_cache_key(prefix: str, *args, **kwargs) -> str:
    """Create a standardized cache key with hash for complex parameters"""
    key_parts = [prefix] + [str(arg) for arg in args]
    
    if kwargs:
        # Sort kwargs for consistent hashing
        sorted_kwargs = sorted(kwargs.items())
        kwargs_str = json.dumps(sorted_kwargs, sort_keys=True)
        kwargs_hash = hashlib.md5(kwargs_str.encode()).hexdigest()[:8]
        key_parts.append(kwargs_hash)
    
    return ":".join(key_parts)


# Global cache manager instance
_cache_manager: Optional[MusicStreamingCacheManager] = None


async def get_cache_manager(config: Optional[CacheConfig] = None) -> MusicStreamingCacheManager:
    """Get or create global cache manager instance"""
    global _cache_manager
    
    if _cache_manager is None:
        if config is None:
            config = CacheConfig()
        _cache_manager = MusicStreamingCacheManager(config)
        await _cache_manager.initialize()
    
    return _cache_manager


async def close_cache_manager():
    """Close global cache manager"""
    global _cache_manager
    
    if _cache_manager is not None:
        await _cache_manager.cache.clear()
        _cache_manager = None
