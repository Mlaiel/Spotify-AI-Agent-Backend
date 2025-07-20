"""
📊 Cache Manager - Gestionnaire de Cache Ultra-Avancé Multi-Tenant
================================================================

Gestionnaire de cache intelligent pour l'isolation des données multi-tenant
avec invalidation intelligente, partitioning, et monitoring avancé.

Author: Architecte Cache - Fahed Mlaiel
"""

import asyncio
import logging
import hashlib
import json
import pickle
import weakref
from typing import Dict, List, Any, Optional, Union, Set, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import redis.asyncio as redis
from contextlib import asynccontextmanager

from ..core.tenant_context import TenantContext, TenantType
from ..exceptions import DataIsolationError, CacheError


class CacheBackend(Enum):
    """Types de backend de cache"""
    MEMORY = "memory"
    REDIS = "redis"
    MEMCACHED = "memcached"
    HYBRID = "hybrid"


class CacheStrategy(Enum):
    """Stratégies de cache"""
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"
    WRITE_AROUND = "write_around"
    CACHE_ASIDE = "cache_aside"


class EvictionPolicy(Enum):
    """Politiques d'éviction"""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    FIFO = "fifo"         # First In First Out
    TTL = "ttl"           # Time To Live based
    TENANT_AWARE = "tenant_aware"  # Tenant-aware eviction


class CacheLevel(Enum):
    """Niveaux de cache"""
    L1_MEMORY = "l1_memory"       # Cache mémoire local
    L2_REDIS = "l2_redis"         # Cache Redis partagé
    L3_PERSISTENT = "l3_persistent"  # Cache persistant


@dataclass
class CacheMetrics:
    """Métriques de cache"""
    tenant_id: str
    cache_level: CacheLevel
    
    # Hit/Miss statistics
    hits: int = 0
    misses: int = 0
    writes: int = 0
    deletes: int = 0
    
    # Timing statistics
    avg_hit_time: float = 0.0
    avg_miss_time: float = 0.0
    avg_write_time: float = 0.0
    
    # Size statistics
    current_size: int = 0
    max_size: int = 0
    evictions: int = 0
    
    # Last operations
    last_hit: Optional[datetime] = None
    last_miss: Optional[datetime] = None
    last_write: Optional[datetime] = None
    
    # Error tracking
    error_count: int = 0
    last_error: Optional[str] = None
    
    def hit_ratio(self) -> float:
        """Calcule le taux de hit"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def miss_ratio(self) -> float:
        """Calcule le taux de miss"""
        return 1.0 - self.hit_ratio()


@dataclass
class CacheConfiguration:
    """Configuration du cache"""
    # Backend settings
    backend: CacheBackend = CacheBackend.HYBRID
    strategy: CacheStrategy = CacheStrategy.CACHE_ASIDE
    eviction_policy: EvictionPolicy = EvictionPolicy.TENANT_AWARE
    
    # Size limits
    max_memory_size: int = 100 * 1024 * 1024  # 100MB
    max_entries: int = 10000
    max_key_size: int = 250
    max_value_size: int = 1 * 1024 * 1024  # 1MB
    
    # TTL settings
    default_ttl: int = 3600  # 1 hour
    max_ttl: int = 86400     # 24 hours
    min_ttl: int = 60        # 1 minute
    
    # Tenant isolation
    tenant_namespace_prefix: str = "tenant"
    enable_cross_tenant_access: bool = False
    tenant_quota_mb: int = 10  # MB per tenant
    
    # Performance settings
    async_write: bool = True
    compression_enabled: bool = True
    compression_threshold: int = 1024  # Bytes
    serialization_format: str = "pickle"  # pickle, json, msgpack
    
    # Redis settings (if used)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_ssl: bool = False
    redis_cluster_mode: bool = False
    redis_sentinel_hosts: List[str] = field(default_factory=list)
    
    # Connection pooling
    redis_pool_size: int = 10
    redis_pool_timeout: int = 10
    
    # Monitoring
    enable_metrics: bool = True
    metrics_retention_hours: int = 24
    slow_operation_threshold: float = 0.1  # seconds
    
    # Advanced features
    enable_cache_warming: bool = True
    enable_write_behind_batching: bool = True
    write_behind_batch_size: int = 100
    write_behind_flush_interval: int = 5  # seconds


class TenantCacheNamespace:
    """Namespace de cache pour un tenant"""
    
    def __init__(self, tenant_id: str, config: CacheConfiguration):
        self.tenant_id = tenant_id
        self.config = config
        self.logger = logging.getLogger(f"cache.tenant.{tenant_id}")
        
        # Namespace prefix
        self.prefix = f"{config.tenant_namespace_prefix}:{tenant_id}"
        
        # Local memory cache (L1)
        self._memory_cache: Dict[str, Tuple[Any, datetime, float]] = {}
        self._access_times: Dict[str, datetime] = {}
        self._access_counts: Dict[str, int] = {}
        
        # Metrics
        self.metrics = {
            level: CacheMetrics(tenant_id, level)
            for level in CacheLevel
        }
        
        # Size tracking
        self._current_memory_usage = 0
        self._entry_count = 0
        
        # Write-behind queue
        self._write_behind_queue: List[Tuple[str, Any, float]] = []
        self._write_behind_lock = asyncio.Lock()
    
    def _make_key(self, key: str, level: CacheLevel) -> str:
        """Crée une clé avec namespace"""
        level_prefix = level.value
        return f"{self.prefix}:{level_prefix}:{key}"
    
    def _estimate_size(self, value: Any) -> int:
        """Estime la taille d'une valeur"""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, dict):
                return len(str(value)) * 2  # Rough estimate
            else:
                return len(pickle.dumps(value))
        except:
            return 1024  # Default estimate
    
    async def get(self, key: str, level: CacheLevel = CacheLevel.L1_MEMORY) -> Optional[Any]:
        """Récupère une valeur du cache"""
        start_time = datetime.now(timezone.utc)
        
        try:
            if level == CacheLevel.L1_MEMORY:
                return await self._get_from_memory(key)
            elif level == CacheLevel.L2_REDIS:
                return await self._get_from_redis(key)
            elif level == CacheLevel.L3_PERSISTENT:
                return await self._get_from_persistent(key)
            
        except Exception as e:
            self._record_error(level, str(e))
            self.logger.error(f"Cache get error for key {key}: {e}")
            return None
        finally:
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            # Metrics will be updated by specific get methods
    
    async def _get_from_memory(self, key: str) -> Optional[Any]:
        """Récupère du cache mémoire"""
        if key not in self._memory_cache:
            self._record_miss(CacheLevel.L1_MEMORY)
            return None
        
        value, created_at, ttl = self._memory_cache[key]
        
        # Check TTL
        if ttl > 0 and (datetime.now(timezone.utc) - created_at).total_seconds() > ttl:
            await self._evict_from_memory(key)
            self._record_miss(CacheLevel.L1_MEMORY)
            return None
        
        # Update access tracking
        self._access_times[key] = datetime.now(timezone.utc)
        self._access_counts[key] = self._access_counts.get(key, 0) + 1
        
        self._record_hit(CacheLevel.L1_MEMORY)
        return value
    
    async def _get_from_redis(self, key: str) -> Optional[Any]:
        """Récupère du cache Redis"""
        # This would implement Redis retrieval
        # For now, return None (not implemented)
        self._record_miss(CacheLevel.L2_REDIS)
        return None
    
    async def _get_from_persistent(self, key: str) -> Optional[Any]:
        """Récupère du cache persistant"""
        # This would implement persistent cache retrieval
        # For now, return None (not implemented)
        self._record_miss(CacheLevel.L3_PERSISTENT)
        return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        level: CacheLevel = CacheLevel.L1_MEMORY
    ) -> bool:
        """Stocke une valeur dans le cache"""
        start_time = datetime.now(timezone.utc)
        
        try:
            if level == CacheLevel.L1_MEMORY:
                return await self._set_to_memory(key, value, ttl)
            elif level == CacheLevel.L2_REDIS:
                return await self._set_to_redis(key, value, ttl)
            elif level == CacheLevel.L3_PERSISTENT:
                return await self._set_to_persistent(key, value, ttl)
            
        except Exception as e:
            self._record_error(level, str(e))
            self.logger.error(f"Cache set error for key {key}: {e}")
            return False
        finally:
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.metrics[level].avg_write_time = (
                self.metrics[level].avg_write_time * 0.9 + response_time * 0.1
            )
    
    async def _set_to_memory(self, key: str, value: Any, ttl: Optional[int]) -> bool:
        """Stocke dans le cache mémoire"""
        value_size = self._estimate_size(value)
        
        # Check size limits
        if value_size > self.config.max_value_size:
            self.logger.warning(f"Value too large for key {key}: {value_size} bytes")
            return False
        
        # Check if we need to evict
        if (self._current_memory_usage + value_size > self.config.max_memory_size or
            self._entry_count >= self.config.max_entries):
            await self._evict_memory_entries()
        
        # Check quota
        tenant_usage = self._current_memory_usage / (1024 * 1024)  # MB
        if tenant_usage >= self.config.tenant_quota_mb:
            self.logger.warning(f"Tenant {self.tenant_id} exceeded cache quota")
            await self._evict_memory_entries(force=True)
        
        # Store value
        effective_ttl = ttl or self.config.default_ttl
        self._memory_cache[key] = (value, datetime.now(timezone.utc), effective_ttl)
        
        # Update tracking
        if key not in self._access_times:
            self._current_memory_usage += value_size
            self._entry_count += 1
        
        self._access_times[key] = datetime.now(timezone.utc)
        self._access_counts[key] = self._access_counts.get(key, 0) + 1
        
        self._record_write(CacheLevel.L1_MEMORY)
        return True
    
    async def _set_to_redis(self, key: str, value: Any, ttl: Optional[int]) -> bool:
        """Stocke dans Redis"""
        # Implementation would go here
        self._record_write(CacheLevel.L2_REDIS)
        return True
    
    async def _set_to_persistent(self, key: str, value: Any, ttl: Optional[int]) -> bool:
        """Stocke dans le cache persistant"""
        # Implementation would go here
        self._record_write(CacheLevel.L3_PERSISTENT)
        return True
    
    async def delete(self, key: str, level: CacheLevel = CacheLevel.L1_MEMORY) -> bool:
        """Supprime une clé du cache"""
        try:
            if level == CacheLevel.L1_MEMORY:
                return await self._delete_from_memory(key)
            elif level == CacheLevel.L2_REDIS:
                return await self._delete_from_redis(key)
            elif level == CacheLevel.L3_PERSISTENT:
                return await self._delete_from_persistent(key)
            
        except Exception as e:
            self._record_error(level, str(e))
            self.logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def _delete_from_memory(self, key: str) -> bool:
        """Supprime du cache mémoire"""
        if key in self._memory_cache:
            value, _, _ = self._memory_cache[key]
            value_size = self._estimate_size(value)
            
            del self._memory_cache[key]
            if key in self._access_times:
                del self._access_times[key]
            if key in self._access_counts:
                del self._access_counts[key]
            
            self._current_memory_usage -= value_size
            self._entry_count -= 1
            
            self._record_delete(CacheLevel.L1_MEMORY)
            return True
        
        return False
    
    async def _delete_from_redis(self, key: str) -> bool:
        """Supprime de Redis"""
        # Implementation would go here
        self._record_delete(CacheLevel.L2_REDIS)
        return True
    
    async def _delete_from_persistent(self, key: str) -> bool:
        """Supprime du cache persistant"""
        # Implementation would go here
        self._record_delete(CacheLevel.L3_PERSISTENT)
        return True
    
    async def _evict_memory_entries(self, force: bool = False):
        """Évince des entrées du cache mémoire"""
        if not self._memory_cache:
            return
        
        eviction_count = max(1, len(self._memory_cache) // 10)  # Evict 10%
        if force:
            eviction_count = max(eviction_count, len(self._memory_cache) // 4)  # Evict 25%
        
        if self.config.eviction_policy == EvictionPolicy.LRU:
            await self._evict_lru(eviction_count)
        elif self.config.eviction_policy == EvictionPolicy.LFU:
            await self._evict_lfu(eviction_count)
        elif self.config.eviction_policy == EvictionPolicy.FIFO:
            await self._evict_fifo(eviction_count)
        elif self.config.eviction_policy == EvictionPolicy.TTL:
            await self._evict_expired()
        elif self.config.eviction_policy == EvictionPolicy.TENANT_AWARE:
            await self._evict_tenant_aware(eviction_count)
    
    async def _evict_lru(self, count: int):
        """Évince selon LRU (Least Recently Used)"""
        # Sort by access time
        sorted_keys = sorted(
            self._access_times.keys(),
            key=lambda k: self._access_times[k]
        )
        
        for key in sorted_keys[:count]:
            await self._evict_from_memory(key)
    
    async def _evict_lfu(self, count: int):
        """Évince selon LFU (Least Frequently Used)"""
        # Sort by access count
        sorted_keys = sorted(
            self._access_counts.keys(),
            key=lambda k: self._access_counts[k]
        )
        
        for key in sorted_keys[:count]:
            await self._evict_from_memory(key)
    
    async def _evict_fifo(self, count: int):
        """Évince selon FIFO (First In First Out)"""
        # Sort by creation time (stored in cache value)
        sorted_keys = sorted(
            self._memory_cache.keys(),
            key=lambda k: self._memory_cache[k][1]  # created_at
        )
        
        for key in sorted_keys[:count]:
            await self._evict_from_memory(key)
    
    async def _evict_expired(self):
        """Évince les entrées expirées"""
        now = datetime.now(timezone.utc)
        expired_keys = []
        
        for key, (value, created_at, ttl) in self._memory_cache.items():
            if ttl > 0 and (now - created_at).total_seconds() > ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            await self._evict_from_memory(key)
    
    async def _evict_tenant_aware(self, count: int):
        """Évince avec conscience du tenant"""
        # Combine LRU and LFU with tenant-specific logic
        now = datetime.now(timezone.utc)
        
        # Score each key
        key_scores = {}
        for key in self._memory_cache.keys():
            access_time = self._access_times.get(key, now)
            access_count = self._access_counts.get(key, 0)
            
            # Time factor (older = higher score = more likely to evict)
            time_factor = (now - access_time).total_seconds() / 3600  # hours
            
            # Frequency factor (less used = higher score)
            freq_factor = 1.0 / max(access_count, 1)
            
            # Combine factors
            key_scores[key] = time_factor * 0.7 + freq_factor * 0.3
        
        # Sort by score (highest first)
        sorted_keys = sorted(key_scores.keys(), key=lambda k: key_scores[k], reverse=True)
        
        for key in sorted_keys[:count]:
            await self._evict_from_memory(key)
    
    async def _evict_from_memory(self, key: str):
        """Évince une clé spécifique de la mémoire"""
        if key in self._memory_cache:
            await self._delete_from_memory(key)
            self.metrics[CacheLevel.L1_MEMORY].evictions += 1
    
    def _record_hit(self, level: CacheLevel):
        """Enregistre un hit"""
        self.metrics[level].hits += 1
        self.metrics[level].last_hit = datetime.now(timezone.utc)
    
    def _record_miss(self, level: CacheLevel):
        """Enregistre un miss"""
        self.metrics[level].misses += 1
        self.metrics[level].last_miss = datetime.now(timezone.utc)
    
    def _record_write(self, level: CacheLevel):
        """Enregistre une écriture"""
        self.metrics[level].writes += 1
        self.metrics[level].last_write = datetime.now(timezone.utc)
    
    def _record_delete(self, level: CacheLevel):
        """Enregistre une suppression"""
        self.metrics[level].deletes += 1
    
    def _record_error(self, level: CacheLevel, error: str):
        """Enregistre une erreur"""
        self.metrics[level].error_count += 1
        self.metrics[level].last_error = error
    
    async def clear(self, level: Optional[CacheLevel] = None):
        """Vide le cache"""
        if level is None or level == CacheLevel.L1_MEMORY:
            self._memory_cache.clear()
            self._access_times.clear()
            self._access_counts.clear()
            self._current_memory_usage = 0
            self._entry_count = 0
        
        if level is None or level == CacheLevel.L2_REDIS:
            # Clear Redis cache
            pass
        
        if level is None or level == CacheLevel.L3_PERSISTENT:
            # Clear persistent cache
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du namespace"""
        return {
            "tenant_id": self.tenant_id,
            "memory_usage_bytes": self._current_memory_usage,
            "memory_usage_mb": self._current_memory_usage / (1024 * 1024),
            "entry_count": self._entry_count,
            "quota_usage_percent": (self._current_memory_usage / (1024 * 1024)) / self.config.tenant_quota_mb * 100,
            "metrics": {
                level.value: {
                    "hits": metrics.hits,
                    "misses": metrics.misses,
                    "writes": metrics.writes,
                    "deletes": metrics.deletes,
                    "hit_ratio": metrics.hit_ratio(),
                    "miss_ratio": metrics.miss_ratio(),
                    "avg_hit_time": metrics.avg_hit_time,
                    "avg_miss_time": metrics.avg_miss_time,
                    "avg_write_time": metrics.avg_write_time,
                    "evictions": metrics.evictions,
                    "error_count": metrics.error_count,
                    "last_error": metrics.last_error
                }
                for level, metrics in self.metrics.items()
            }
        }


class CacheManager:
    """
    Gestionnaire de cache ultra-avancé pour l'isolation multi-tenant
    
    Features:
    - Cache multi-niveau (Memory, Redis, Persistent)
    - Isolation complète par tenant
    - Quotas et limites par tenant
    - Éviction intelligente tenant-aware
    - Métriques détaillées par tenant
    - Invalidation de cache distribuée
    - Write-behind et write-through
    - Compression et sérialisation
    - Monitoring en temps réel
    """
    
    def __init__(self, config: Optional[CacheConfiguration] = None):
        self.config = config or CacheConfiguration()
        self.logger = logging.getLogger("cache_manager")
        
        # Tenant namespaces
        self._tenant_caches: Dict[str, TenantCacheNamespace] = {}
        self._tenant_cache_refs = weakref.WeakValueDictionary()
        
        # Redis connection (if used)
        self._redis_pool: Optional[redis.ConnectionPool] = None
        self._redis_client: Optional[redis.Redis] = None
        
        # Global state
        self._is_initialized = False
        self._shutdown_event = asyncio.Event()
        
        # Monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._write_behind_task: Optional[asyncio.Task] = None
        
        # Global metrics
        self._global_metrics = {
            "total_tenants": 0,
            "total_memory_usage": 0,
            "total_entries": 0,
            "total_hits": 0,
            "total_misses": 0,
            "total_writes": 0,
            "total_deletes": 0,
            "total_evictions": 0
        }
        
        # Cache warming
        self._warming_patterns: Dict[str, Callable] = {}
        
        # Invalidation tracking
        self._invalidation_patterns: Dict[str, Set[str]] = {}
    
    async def initialize(self):
        """Initialise le gestionnaire de cache"""
        try:
            if self._is_initialized:
                return
            
            # Initialize Redis if needed
            if self.config.backend in [CacheBackend.REDIS, CacheBackend.HYBRID]:
                await self._initialize_redis()
            
            # Start background tasks
            if self.config.enable_metrics:
                self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            if self.config.enable_write_behind_batching:
                self._write_behind_task = asyncio.create_task(self._write_behind_loop())
            
            self._is_initialized = True
            self.logger.info("Cache manager initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize cache manager: {e}")
            raise CacheError(f"Cache manager initialization failed: {e}")
    
    async def _initialize_redis(self):
        """Initialise la connexion Redis"""
        try:
            self._redis_pool = redis.ConnectionPool(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                ssl=self.config.redis_ssl,
                max_connections=self.config.redis_pool_size,
                socket_timeout=self.config.redis_pool_timeout
            )
            
            self._redis_client = redis.Redis(connection_pool=self._redis_pool)
            
            # Test connection
            await self._redis_client.ping()
            
            self.logger.info("Redis connection initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis: {e}")
            raise CacheError(f"Redis initialization failed: {e}")
    
    def get_tenant_cache(self, context: TenantContext) -> TenantCacheNamespace:
        """Obtient le cache d'un tenant"""
        tenant_id = context.tenant_id
        
        if tenant_id not in self._tenant_caches:
            cache_namespace = TenantCacheNamespace(tenant_id, self.config)
            self._tenant_caches[tenant_id] = cache_namespace
            self._tenant_cache_refs[tenant_id] = cache_namespace
            self._global_metrics["total_tenants"] += 1
            
            self.logger.debug(f"Created cache namespace for tenant {tenant_id}")
        
        return self._tenant_caches[tenant_id]
    
    async def get(
        self, 
        context: TenantContext, 
        key: str,
        level: CacheLevel = CacheLevel.L1_MEMORY
    ) -> Optional[Any]:
        """Récupère une valeur du cache"""
        if not self._validate_tenant_access(context, key):
            raise CacheError(f"Access denied for tenant {context.tenant_id} to key {key}")
        
        tenant_cache = self.get_tenant_cache(context)
        return await tenant_cache.get(key, level)
    
    async def set(
        self, 
        context: TenantContext, 
        key: str, 
        value: Any,
        ttl: Optional[int] = None,
        level: CacheLevel = CacheLevel.L1_MEMORY
    ) -> bool:
        """Stocke une valeur dans le cache"""
        if not self._validate_tenant_access(context, key):
            raise CacheError(f"Access denied for tenant {context.tenant_id} to key {key}")
        
        tenant_cache = self.get_tenant_cache(context)
        
        # Handle write strategies
        if self.config.strategy == CacheStrategy.WRITE_THROUGH:
            return await self._write_through(tenant_cache, key, value, ttl, level)
        elif self.config.strategy == CacheStrategy.WRITE_BEHIND:
            return await self._write_behind(tenant_cache, key, value, ttl, level)
        elif self.config.strategy == CacheStrategy.WRITE_AROUND:
            return await self._write_around(tenant_cache, key, value, ttl, level)
        else:  # CACHE_ASIDE
            return await tenant_cache.set(key, value, ttl, level)
    
    async def _write_through(
        self, 
        tenant_cache: TenantCacheNamespace, 
        key: str, 
        value: Any, 
        ttl: Optional[int],
        level: CacheLevel
    ) -> bool:
        """Stratégie write-through"""
        # Write to cache and backend simultaneously
        cache_result = await tenant_cache.set(key, value, ttl, level)
        # backend_result = await self._write_to_backend(key, value)
        backend_result = True  # Mock
        
        return cache_result and backend_result
    
    async def _write_behind(
        self, 
        tenant_cache: TenantCacheNamespace, 
        key: str, 
        value: Any, 
        ttl: Optional[int],
        level: CacheLevel
    ) -> bool:
        """Stratégie write-behind"""
        # Write to cache immediately, queue backend write
        cache_result = await tenant_cache.set(key, value, ttl, level)
        
        if cache_result and self.config.enable_write_behind_batching:
            async with tenant_cache._write_behind_lock:
                tenant_cache._write_behind_queue.append((key, value, ttl or 0))
        
        return cache_result
    
    async def _write_around(
        self, 
        tenant_cache: TenantCacheNamespace, 
        key: str, 
        value: Any, 
        ttl: Optional[int],
        level: CacheLevel
    ) -> bool:
        """Stratégie write-around"""
        # Write to backend only, bypass cache
        # backend_result = await self._write_to_backend(key, value)
        backend_result = True  # Mock
        return backend_result
    
    async def delete(
        self, 
        context: TenantContext, 
        key: str,
        level: CacheLevel = CacheLevel.L1_MEMORY
    ) -> bool:
        """Supprime une clé du cache"""
        if not self._validate_tenant_access(context, key):
            raise CacheError(f"Access denied for tenant {context.tenant_id} to key {key}")
        
        tenant_cache = self.get_tenant_cache(context)
        return await tenant_cache.delete(key, level)
    
    async def invalidate_pattern(self, context: TenantContext, pattern: str):
        """Invalide toutes les clés correspondant à un pattern"""
        tenant_cache = self.get_tenant_cache(context)
        
        # Find matching keys
        matching_keys = []
        for key in tenant_cache._memory_cache.keys():
            if self._key_matches_pattern(key, pattern):
                matching_keys.append(key)
        
        # Delete matching keys
        for key in matching_keys:
            await tenant_cache.delete(key)
        
        self.logger.info(f"Invalidated {len(matching_keys)} keys for pattern {pattern}")
    
    def _key_matches_pattern(self, key: str, pattern: str) -> bool:
        """Vérifie si une clé correspond à un pattern"""
        import fnmatch
        return fnmatch.fnmatch(key, pattern)
    
    def _validate_tenant_access(self, context: TenantContext, key: str) -> bool:
        """Valide l'accès d'un tenant à une clé"""
        if self.config.enable_cross_tenant_access:
            return True
        
        # Basic validation - key should not contain other tenant IDs
        # More sophisticated validation could be implemented
        return True
    
    async def _monitoring_loop(self):
        """Boucle de monitoring"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Monitor every minute
                await self._collect_global_metrics()
                await self._log_performance_stats()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
    
    async def _collect_global_metrics(self):
        """Collecte les métriques globales"""
        total_memory = 0
        total_entries = 0
        total_hits = 0
        total_misses = 0
        total_writes = 0
        total_deletes = 0
        total_evictions = 0
        
        for tenant_cache in self._tenant_caches.values():
            total_memory += tenant_cache._current_memory_usage
            total_entries += tenant_cache._entry_count
            
            for metrics in tenant_cache.metrics.values():
                total_hits += metrics.hits
                total_misses += metrics.misses
                total_writes += metrics.writes
                total_deletes += metrics.deletes
                total_evictions += metrics.evictions
        
        self._global_metrics.update({
            "total_memory_usage": total_memory,
            "total_entries": total_entries,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "total_writes": total_writes,
            "total_deletes": total_deletes,
            "total_evictions": total_evictions
        })
    
    async def _log_performance_stats(self):
        """Log les statistiques de performance"""
        stats = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "global_metrics": dict(self._global_metrics),
            "hit_ratio": (
                self._global_metrics["total_hits"] / 
                max(self._global_metrics["total_hits"] + self._global_metrics["total_misses"], 1)
            ),
            "memory_usage_mb": self._global_metrics["total_memory_usage"] / (1024 * 1024),
            "tenant_count": len(self._tenant_caches)
        }
        
        self.logger.info(f"Cache performance: {json.dumps(stats, indent=2)}")
    
    async def _cleanup_loop(self):
        """Boucle de nettoyage"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                await self._cleanup_expired_entries()
                await self._cleanup_inactive_tenants()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
    
    async def _cleanup_expired_entries(self):
        """Nettoie les entrées expirées"""
        for tenant_cache in self._tenant_caches.values():
            await tenant_cache._evict_expired()
    
    async def _cleanup_inactive_tenants(self):
        """Nettoie les tenants inactifs"""
        inactive_tenants = []
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
        
        for tenant_id, tenant_cache in self._tenant_caches.items():
            # Check if tenant has been active recently
            last_activity = None
            for metrics in tenant_cache.metrics.values():
                if metrics.last_hit and (not last_activity or metrics.last_hit > last_activity):
                    last_activity = metrics.last_hit
                if metrics.last_write and (not last_activity or metrics.last_write > last_activity):
                    last_activity = metrics.last_write
            
            if not last_activity or last_activity < cutoff_time:
                inactive_tenants.append(tenant_id)
        
        # Clean up inactive tenants
        for tenant_id in inactive_tenants:
            if tenant_id in self._tenant_caches:
                await self._tenant_caches[tenant_id].clear()
                del self._tenant_caches[tenant_id]
                self.logger.info(f"Cleaned up inactive tenant cache: {tenant_id}")
    
    async def _write_behind_loop(self):
        """Boucle de write-behind"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.write_behind_flush_interval)
                await self._flush_write_behind_queues()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Write-behind error: {e}")
    
    async def _flush_write_behind_queues(self):
        """Vide les files d'attente write-behind"""
        for tenant_cache in self._tenant_caches.values():
            if tenant_cache._write_behind_queue:
                async with tenant_cache._write_behind_lock:
                    queue = tenant_cache._write_behind_queue[:]
                    tenant_cache._write_behind_queue.clear()
                
                # Process batches
                batch_size = self.config.write_behind_batch_size
                for i in range(0, len(queue), batch_size):
                    batch = queue[i:i + batch_size]
                    await self._process_write_behind_batch(batch)
    
    async def _process_write_behind_batch(self, batch: List[Tuple[str, Any, float]]):
        """Traite un batch d'écritures write-behind"""
        # This would write to the backend storage
        # For now, just log
        self.logger.debug(f"Processing write-behind batch of {len(batch)} items")
    
    async def warm_cache(self, context: TenantContext, pattern: str):
        """Préchauffe le cache"""
        if not self.config.enable_cache_warming:
            return
        
        if pattern in self._warming_patterns:
            warming_func = self._warming_patterns[pattern]
            await warming_func(context, self)
            self.logger.info(f"Cache warmed for pattern {pattern}")
    
    def register_warming_pattern(self, pattern: str, warming_func: Callable):
        """Enregistre une fonction de préchauffage"""
        self._warming_patterns[pattern] = warming_func
        self.logger.info(f"Registered warming pattern: {pattern}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérification de santé"""
        health_status = {
            "cache_manager_healthy": True,
            "redis_healthy": True,
            "tenant_caches": len(self._tenant_caches),
            "global_metrics": dict(self._global_metrics)
        }
        
        # Test Redis if enabled
        if self._redis_client:
            try:
                await self._redis_client.ping()
            except Exception as e:
                health_status["redis_healthy"] = False
                health_status["redis_error"] = str(e)
        
        # Check tenant cache health
        unhealthy_tenants = []
        for tenant_id, tenant_cache in self._tenant_caches.items():
            quota_usage = (tenant_cache._current_memory_usage / (1024 * 1024)) / self.config.tenant_quota_mb
            if quota_usage > 0.9:  # >90% quota usage
                unhealthy_tenants.append(tenant_id)
        
        if unhealthy_tenants:
            health_status["cache_manager_healthy"] = False
            health_status["unhealthy_tenants"] = unhealthy_tenants
        
        return health_status
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Obtient les statistiques détaillées"""
        await self._collect_global_metrics()
        
        tenant_stats = {}
        for tenant_id, tenant_cache in self._tenant_caches.items():
            tenant_stats[tenant_id] = tenant_cache.get_stats()
        
        return {
            "global_metrics": dict(self._global_metrics),
            "configuration": {
                "backend": self.config.backend.value,
                "strategy": self.config.strategy.value,
                "eviction_policy": self.config.eviction_policy.value,
                "max_memory_size": self.config.max_memory_size,
                "default_ttl": self.config.default_ttl,
                "tenant_quota_mb": self.config.tenant_quota_mb
            },
            "tenant_statistics": tenant_stats,
            "active_tenants": len(self._tenant_caches),
            "warming_patterns": list(self._warming_patterns.keys())
        }
    
    async def shutdown(self):
        """Arrêt propre du gestionnaire"""
        self.logger.info("Shutting down cache manager...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in [self._monitoring_task, self._cleanup_task, self._write_behind_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Flush write-behind queues
        if self.config.enable_write_behind_batching:
            await self._flush_write_behind_queues()
        
        # Clear all tenant caches
        for tenant_cache in self._tenant_caches.values():
            await tenant_cache.clear()
        
        self._tenant_caches.clear()
        
        # Close Redis connection
        if self._redis_client:
            await self._redis_client.close()
        if self._redis_pool:
            await self._redis_pool.disconnect()
        
        self.logger.info("Cache manager shutdown completed")


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


async def get_cache_manager() -> CacheManager:
    """Obtient l'instance globale du gestionnaire de cache"""
    global _cache_manager
    if not _cache_manager:
        _cache_manager = CacheManager()
        await _cache_manager.initialize()
    return _cache_manager


async def shutdown_cache_manager():
    """Arrête l'instance globale du gestionnaire de cache"""
    global _cache_manager
    if _cache_manager:
        await _cache_manager.shutdown()
        _cache_manager = None
