import logging
from typing import Any, Optional, Dict, Callable
from .cache_strategies import BaseCacheStrategy, LRUCacheStrategy, LFUCacheStrategy, AdaptiveMLCacheStrategy
from .redis_service import RedisCacheService
from .invalidation_service import InvalidationService
from .metrics import CacheMetrics
from .security import CacheSecurity

logger = logging.getLogger("cache_manager")

class CacheManager:
    """
    Orchestrateur de cache avancé, multi-backends, sécurisé, observable, partitionnable.
    - Supporte hooks d’audit, partitioning, fallback, warmup, monitoring.
    - Utilisable pour IA, analytics, Spotify, scoring, etc.
    """
    def __init__(self, backend: str = "redis", strategy: Optional[BaseCacheStrategy] = None, partition: Optional[str] = None, fallback: Optional[Callable[[str], Any]] = None, **kwargs):
        self.metrics = CacheMetrics()
        self.security = CacheSecurity()
        self.invalidator = InvalidationService()
        self.backend = backend
        self.partition = partition or "default"
        self.fallback = fallback
        if backend == "redis":
            self.engine = RedisCacheService(**kwargs)
        elif backend == "memory":
            self.engine = dict()
        else:
            raise ValueError(f"Backend {backend} non supporté")
        self.strategy = strategy or LRUCacheStrategy()
        logger.info(f"CacheManager initialisé avec backend={backend}, partition={self.partition}, stratégie={self.strategy.__class__.__name__}")

    def _partition_key(self, key: str) -> str:
        return f"{self.partition}:{key}"

    def get(self, key: str) -> Any:
        self.metrics.inc_get()
        pkey = self._partition_key(key)
        value = self.engine.get(pkey) if self.backend == "redis" else self.engine.get(pkey)
        if value:
            logger.debug(f"Cache hit: {pkey}")
            self.metrics.inc_hit()
            return self.security.decrypt(value)
        logger.debug(f"Cache miss: {pkey}")
        self.metrics.inc_miss()
        # Fallback logique métier
        if self.fallback:
            logger.info(f"Fallback déclenché pour {pkey}")
            fallback_value = self.fallback(key)
            if fallback_value is not None:
                self.set(key, fallback_value)
                return fallback_value
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        self.metrics.inc_set()
        value = self.security.encrypt(value)
        pkey = self._partition_key(key)
        if self.backend == "redis":
            self.engine.set(pkey, value, ex=ttl)
        else:
            self.engine[pkey] = value
        logger.debug(f"Cache set: {pkey} (ttl={ttl})")

    def invalidate(self, key: str):
        pkey = self._partition_key(key)
        self.invalidator.invalidate(pkey)
        if self.backend == "redis":
            self.engine.delete(pkey)
        else:
            self.engine.pop(pkey, None)
        logger.info(f"Cache invalidé: {pkey}")

    def clear(self):
        if self.backend == "redis":
            self.engine.flushdb()
        else:
            self.engine.clear()
        logger.warning("Cache global vidé")

    def warmup(self, data: Dict[str, Any], ttl: Optional[int] = None):
        for key, value in data.items():
            self.set(key, value, ttl=ttl)
        logger.info(f"Cache warmup terminé pour {len(data)} entrées dans la partition {self.partition}")
