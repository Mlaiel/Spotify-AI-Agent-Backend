"""
Module cache avancé pour Spotify AI Agent.
Auto-discovery, versioning, et exports explicites.
"""

from .cache_manager import CacheManager
from .cache_strategies import (
    BaseCacheStrategy, LRUCacheStrategy, LFUCacheStrategy, AdaptiveMLCacheStrategy
)
from .invalidation_service import InvalidationService
from .redis_service import RedisCacheService
from .metrics import CacheMetrics
from .security import CacheSecurity

__version__ = "1.0.0"
__all__ = [
    "CacheManager",
    "BaseCacheStrategy", "LRUCacheStrategy", "LFUCacheStrategy", "AdaptiveMLCacheStrategy",
    "InvalidationService",
    "RedisCacheService",
    "CacheMetrics",
    "CacheSecurity",
]
