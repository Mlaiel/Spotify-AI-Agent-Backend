"""
Advanced Cache Manager
=====================

Gestionnaire de cache multi-backend avec stratégies avancées.
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
import zlib
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import sys
sys.path.insert(0, '/workspaces/Achiri/spotify-ai-agent/backend')
import aioredis_compat as aioredis
from threading import RLock
import weakref

logger = logging.getLogger(__name__)

# === Types et constantes ===
CacheKey = Union[str, bytes]
CacheValue = Any
TTL = Union[int, float, timedelta]

class CacheBackend(Enum):
    MEMORY = "memory"
    REDIS = "redis"
    HYBRID = "hybrid"
    DISTRIBUTED = "distributed"

@dataclass
class CacheStats:
    """Statistiques de cache."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    size: int = 0
    memory_usage: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

@dataclass
class CacheEntry:
    """Entrée de cache avec métadonnées."""
    value: Any
    created_at: float
    accessed_at: float
    access_count: int
    ttl: Optional[float]
    tags: Set[str]
    
    @property
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() > self.created_at + self.ttl
    
    @property
    def age(self) -> float:
        return time.time() - self.created_at

# === Interface de base ===
class CacheBackendInterface(ABC):
    """Interface abstraite pour backends de cache."""
    
    @abstractmethod
    async def get(self, key: CacheKey) -> Optional[Any]:
        """Récupère une valeur."""
        pass
    
    @abstractmethod
    async def set(self, key: CacheKey, value: Any, ttl: Optional[TTL] = None) -> bool:
        """Stocke une valeur."""
        pass
    
    @abstractmethod
    async def delete(self, key: CacheKey) -> bool:
        """Supprime une valeur."""
        pass
    
    @abstractmethod
    async def exists(self, key: CacheKey) -> bool:
        """Vérifie l'existence d'une clé."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Vide le cache."""
        pass
    
    @abstractmethod
    async def keys(self, pattern: str = "*") -> List[CacheKey]:
        """Liste les clés correspondant au pattern."""
        pass
    
    @abstractmethod
    async def stats(self) -> CacheStats:
        """Retourne les statistiques."""
        pass

# === Backend mémoire ===
class MemoryCache(CacheBackendInterface):
    """
    Cache en mémoire avec LRU et TTL.
    """
    
    def __init__(self, max_size: int = 10000, default_ttl: Optional[float] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._data: Dict[CacheKey, CacheEntry] = {}
        self._access_order: List[CacheKey] = []
        self._lock = RLock()
        self._stats = CacheStats()
    
    async def get(self, key: CacheKey) -> Optional[Any]:
        """Récupère une valeur du cache mémoire."""
        with self._lock:
            if key not in self._data:
                self._stats.misses += 1
                return None
            
            entry = self._data[key]
            
            # Vérification TTL
            if entry.is_expired:
                await self._remove_entry(key)
                self._stats.misses += 1
                return None
            
            # Mise à jour des statistiques d'accès
            entry.accessed_at = time.time()
            entry.access_count += 1
            
            # Mise à jour de l'ordre LRU
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            
            self._stats.hits += 1
            return entry.value
    
    async def set(self, key: CacheKey, value: Any, ttl: Optional[TTL] = None) -> bool:
        """Stocke une valeur dans le cache mémoire."""
        with self._lock:
            # Conversion TTL
            if isinstance(ttl, timedelta):
                ttl = ttl.total_seconds()
            elif ttl is None:
                ttl = self.default_ttl
            
            # Éviction si nécessaire
            if len(self._data) >= self.max_size and key not in self._data:
                await self._evict_lru()
            
            # Création de l'entrée
            now = time.time()
            entry = CacheEntry(
                value=value,
                created_at=now,
                accessed_at=now,
                access_count=1,
                ttl=ttl,
                tags=set()
            )
            
            self._data[key] = entry
            
            # Mise à jour de l'ordre LRU
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            
            self._stats.sets += 1
            return True
    
    async def delete(self, key: CacheKey) -> bool:
        """Supprime une valeur du cache."""
        with self._lock:
            if key in self._data:
                await self._remove_entry(key)
                self._stats.deletes += 1
                return True
            return False
    
    async def exists(self, key: CacheKey) -> bool:
        """Vérifie l'existence d'une clé."""
        with self._lock:
            if key not in self._data:
                return False
            
            entry = self._data[key]
            if entry.is_expired:
                await self._remove_entry(key)
                return False
            
            return True
    
    async def clear(self) -> bool:
        """Vide complètement le cache."""
        with self._lock:
            self._data.clear()
            self._access_order.clear()
            return True
    
    async def keys(self, pattern: str = "*") -> List[CacheKey]:
        """Liste les clés correspondant au pattern."""
        import fnmatch
        
        with self._lock:
            # Nettoyage des entrées expirées
            await self._cleanup_expired()
            
            if pattern == "*":
                return list(self._data.keys())
            
            return [key for key in self._data.keys() if fnmatch.fnmatch(str(key), pattern)]
    
    async def stats(self) -> CacheStats:
        """Retourne les statistiques du cache."""
        with self._lock:
            await self._cleanup_expired()
            self._stats.size = len(self._data)
            self._stats.memory_usage = self._estimate_memory_usage()
            return self._stats
    
    async def _remove_entry(self, key: CacheKey):
        """Supprime une entrée du cache."""
        if key in self._data:
            del self._data[key]
        if key in self._access_order:
            self._access_order.remove(key)
    
    async def _evict_lru(self):
        """Évince l'entrée la moins récemment utilisée."""
        if self._access_order:
            lru_key = self._access_order[0]
            await self._remove_entry(lru_key)
            self._stats.evictions += 1
    
    async def _cleanup_expired(self):
        """Nettoie les entrées expirées."""
        expired_keys = []
        for key, entry in self._data.items():
            if entry.is_expired:
                expired_keys.append(key)
        
        for key in expired_keys:
            await self._remove_entry(key)
    
    def _estimate_memory_usage(self) -> int:
        """Estime l'usage mémoire en bytes."""
        total_size = 0
        for key, entry in self._data.items():
            try:
                total_size += len(pickle.dumps(key))
                total_size += len(pickle.dumps(entry.value))
                total_size += 200  # Métadonnées approximatives
            except:
                total_size += 1024  # Estimation par défaut
        return total_size

# === Backend Redis ===
class RedisCache(CacheBackendInterface):
    """
    Cache Redis avec support avancé.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379", prefix: str = "cache:"):
        self.redis_url = redis_url
        self.prefix = prefix
        self._redis: Optional[aioredis.Redis] = None
        self._stats = CacheStats()
    
    async def connect(self):
        """Initialise la connexion Redis."""
        if not self._redis:
            self._redis = aioredis.from_url(self.redis_url, decode_responses=False)
            await self._redis.ping()
    
    async def disconnect(self):
        """Ferme la connexion Redis."""
        if self._redis:
            await self._redis.close()
            self._redis = None
    
    def _make_key(self, key: CacheKey) -> str:
        """Crée une clé Redis avec préfixe."""
        return f"{self.prefix}{key}"
    
    async def get(self, key: CacheKey) -> Optional[Any]:
        """Récupère une valeur de Redis."""
        await self.connect()
        
        redis_key = self._make_key(key)
        data = await self._redis.get(redis_key)
        
        if data is None:
            self._stats.misses += 1
            return None
        
        try:
            # Désérialisation
            value = pickle.loads(data)
            self._stats.hits += 1
            return value
        except Exception as e:
            logger.error(f"Redis deserialization error: {e}")
            self._stats.misses += 1
            return None
    
    async def set(self, key: CacheKey, value: Any, ttl: Optional[TTL] = None) -> bool:
        """Stocke une valeur dans Redis."""
        await self.connect()
        
        try:
            # Sérialisation
            data = pickle.dumps(value)
            redis_key = self._make_key(key)
            
            # Conversion TTL
            if isinstance(ttl, timedelta):
                ttl = int(ttl.total_seconds())
            
            # Stockage avec TTL optionnel
            if ttl:
                await self._redis.setex(redis_key, int(ttl), data)
            else:
                await self._redis.set(redis_key, data)
            
            self._stats.sets += 1
            return True
            
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    async def delete(self, key: CacheKey) -> bool:
        """Supprime une valeur de Redis."""
        await self.connect()
        
        redis_key = self._make_key(key)
        deleted = await self._redis.delete(redis_key)
        
        if deleted:
            self._stats.deletes += 1
            return True
        return False
    
    async def exists(self, key: CacheKey) -> bool:
        """Vérifie l'existence d'une clé dans Redis."""
        await self.connect()
        
        redis_key = self._make_key(key)
        return bool(await self._redis.exists(redis_key))
    
    async def clear(self) -> bool:
        """Vide le cache (toutes les clés avec préfixe)."""
        await self.connect()
        
        pattern = f"{self.prefix}*"
        keys = await self._redis.keys(pattern)
        
        if keys:
            await self._redis.delete(*keys)
        
        return True
    
    async def keys(self, pattern: str = "*") -> List[CacheKey]:
        """Liste les clés correspondant au pattern."""
        await self.connect()
        
        redis_pattern = f"{self.prefix}{pattern}"
        redis_keys = await self._redis.keys(redis_pattern)
        
        # Suppression du préfixe
        return [key.decode().replace(self.prefix, "") for key in redis_keys]
    
    async def stats(self) -> CacheStats:
        """Retourne les statistiques Redis."""
        await self.connect()
        
        # Informations Redis
        info = await self._redis.info("memory")
        self._stats.memory_usage = info.get("used_memory", 0)
        
        # Compte des clés avec préfixe
        pattern = f"{self.prefix}*"
        keys = await self._redis.keys(pattern)
        self._stats.size = len(keys)
        
        return self._stats

# === Cache Hybride ===
class HybridCache(CacheBackendInterface):
    """
    Cache hybride combinant mémoire (L1) et Redis (L2).
    """
    
    def __init__(
        self,
        l1_cache: MemoryCache,
        l2_cache: RedisCache,
        l1_ttl: Optional[float] = 300,  # 5 minutes en L1
        promotion_threshold: int = 2  # Accès requis pour promotion en L1
    ):
        self.l1_cache = l1_cache
        self.l2_cache = l2_cache
        self.l1_ttl = l1_ttl
        self.promotion_threshold = promotion_threshold
        self._access_counts: Dict[CacheKey, int] = {}
        self._stats = CacheStats()
    
    async def get(self, key: CacheKey) -> Optional[Any]:
        """Récupère une valeur avec stratégie L1->L2."""
        
        # Tentative L1 (mémoire)
        value = await self.l1_cache.get(key)
        if value is not None:
            self._stats.hits += 1
            return value
        
        # Tentative L2 (Redis)
        value = await self.l2_cache.get(key)
        if value is not None:
            # Comptage d'accès pour promotion éventuelle
            self._access_counts[key] = self._access_counts.get(key, 0) + 1
            
            # Promotion en L1 si seuil atteint
            if self._access_counts[key] >= self.promotion_threshold:
                await self.l1_cache.set(key, value, self.l1_ttl)
                self._access_counts.pop(key, None)
            
            self._stats.hits += 1
            return value
        
        self._stats.misses += 1
        return None
    
    async def set(self, key: CacheKey, value: Any, ttl: Optional[TTL] = None) -> bool:
        """Stocke une valeur dans les deux niveaux."""
        
        # Stockage L2 (persistant)
        l2_success = await self.l2_cache.set(key, value, ttl)
        
        # Stockage L1 (temporaire)
        l1_ttl = min(self.l1_ttl, ttl) if ttl else self.l1_ttl
        l1_success = await self.l1_cache.set(key, value, l1_ttl)
        
        if l2_success or l1_success:
            self._stats.sets += 1
            return True
        
        return False
    
    async def delete(self, key: CacheKey) -> bool:
        """Supprime une valeur des deux niveaux."""
        l1_deleted = await self.l1_cache.delete(key)
        l2_deleted = await self.l2_cache.delete(key)
        self._access_counts.pop(key, None)
        
        if l1_deleted or l2_deleted:
            self._stats.deletes += 1
            return True
        
        return False
    
    async def exists(self, key: CacheKey) -> bool:
        """Vérifie l'existence dans l'un des niveaux."""
        return await self.l1_cache.exists(key) or await self.l2_cache.exists(key)
    
    async def clear(self) -> bool:
        """Vide les deux niveaux."""
        l1_cleared = await self.l1_cache.clear()
        l2_cleared = await self.l2_cache.clear()
        self._access_counts.clear()
        
        return l1_cleared and l2_cleared
    
    async def keys(self, pattern: str = "*") -> List[CacheKey]:
        """Combine les clés des deux niveaux."""
        l1_keys = set(await self.l1_cache.keys(pattern))
        l2_keys = set(await self.l2_cache.keys(pattern))
        return list(l1_keys.union(l2_keys))
    
    async def stats(self) -> CacheStats:
        """Combine les statistiques des deux niveaux."""
        l1_stats = await self.l1_cache.stats()
        l2_stats = await self.l2_cache.stats()
        
        combined_stats = CacheStats(
            hits=self._stats.hits,
            misses=self._stats.misses,
            sets=self._stats.sets,
            deletes=self._stats.deletes,
            evictions=l1_stats.evictions + l2_stats.evictions,
            size=l1_stats.size + l2_stats.size,
            memory_usage=l1_stats.memory_usage + l2_stats.memory_usage
        )
        
        return combined_stats

# === Gestionnaire principal ===
class CacheManager:
    """
    Gestionnaire principal de cache avec backend configurable.
    """
    
    def __init__(self, backend: CacheBackendInterface):
        self.backend = backend
        self._hooks: Dict[str, List[Callable]] = {
            'before_get': [],
            'after_get': [],
            'before_set': [],
            'after_set': [],
            'before_delete': [],
            'after_delete': []
        }
    
    def add_hook(self, event: str, callback: Callable):
        """Ajoute un hook pour un événement."""
        if event in self._hooks:
            self._hooks[event].append(callback)
    
    async def _execute_hooks(self, event: str, *args, **kwargs):
        """Exécute les hooks pour un événement."""
        for callback in self._hooks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args, **kwargs)
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Hook execution error: {e}")
    
    async def get(self, key: CacheKey, default: Any = None) -> Any:
        """Récupère une valeur avec hooks."""
        await self._execute_hooks('before_get', key)
        
        value = await self.backend.get(key)
        result = value if value is not None else default
        
        await self._execute_hooks('after_get', key, result)
        return result
    
    async def set(self, key: CacheKey, value: Any, ttl: Optional[TTL] = None, tags: Optional[Set[str]] = None) -> bool:
        """Stocke une valeur avec hooks et tags."""
        await self._execute_hooks('before_set', key, value, ttl)
        
        result = await self.backend.set(key, value, ttl)
        
        # Gestion des tags (si supporté par le backend)
        if tags and hasattr(self.backend, 'tag'):
            for tag in tags:
                await self.backend.tag(key, tag)
        
        await self._execute_hooks('after_set', key, value, result)
        return result
    
    async def delete(self, key: CacheKey) -> bool:
        """Supprime une valeur avec hooks."""
        await self._execute_hooks('before_delete', key)
        
        result = await self.backend.delete(key)
        
        await self._execute_hooks('after_delete', key, result)
        return result
    
    async def get_or_set(
        self,
        key: CacheKey,
        factory: Callable,
        ttl: Optional[TTL] = None,
        force_refresh: bool = False
    ) -> Any:
        """
        Récupère une valeur ou la génère si absente.
        
        Args:
            key: Clé de cache
            factory: Fonction/coroutine pour générer la valeur
            ttl: Durée de vie
            force_refresh: Force la régénération
            
        Returns:
            Valeur cachée ou générée
        """
        if not force_refresh:
            value = await self.get(key)
            if value is not None:
                return value
        
        # Génération de la valeur
        if asyncio.iscoroutinefunction(factory):
            new_value = await factory()
        else:
            new_value = factory()
        
        # Stockage en cache
        await self.set(key, new_value, ttl)
        
        return new_value
    
    async def delete_by_pattern(self, pattern: str) -> int:
        """Supprime toutes les clés correspondant au pattern."""
        keys = await self.backend.keys(pattern)
        deleted_count = 0
        
        for key in keys:
            if await self.delete(key):
                deleted_count += 1
        
        return deleted_count
    
    async def batch_get(self, keys: List[CacheKey]) -> Dict[CacheKey, Any]:
        """Récupère plusieurs valeurs en batch."""
        results = {}
        
        # Récupération en parallèle
        tasks = [self.get(key) for key in keys]
        values = await asyncio.gather(*tasks, return_exceptions=True)
        
        for key, value in zip(keys, values):
            if not isinstance(value, Exception) and value is not None:
                results[key] = value
        
        return results
    
    async def batch_set(self, items: Dict[CacheKey, Any], ttl: Optional[TTL] = None) -> Dict[CacheKey, bool]:
        """Stocke plusieurs valeurs en batch."""
        results = {}
        
        # Stockage en parallèle
        tasks = [self.set(key, value, ttl) for key, value in items.items()]
        success_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        for key, success in zip(items.keys(), success_list):
            results[key] = success if not isinstance(success, Exception) else False
        
        return results
    
    async def health_check(self) -> Dict[str, Any]:
        """Effectue un check de santé du cache."""
        start_time = time.time()
        
        try:
            # Test de base
            test_key = f"health_check_{int(time.time())}"
            test_value = "health_check_value"
            
            # Test set/get/delete
            set_success = await self.set(test_key, test_value, ttl=60)
            get_value = await self.get(test_key)
            delete_success = await self.delete(test_key)
            
            # Vérification
            health_ok = (
                set_success and
                get_value == test_value and
                delete_success
            )
            
            response_time = time.time() - start_time
            stats = await self.backend.stats()
            
            return {
                'healthy': health_ok,
                'response_time_ms': response_time * 1000,
                'backend_type': type(self.backend).__name__,
                'stats': {
                    'hit_rate': stats.hit_rate,
                    'size': stats.size,
                    'memory_usage_mb': stats.memory_usage / (1024 * 1024)
                }
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'response_time_ms': (time.time() - start_time) * 1000
            }

# === Cache Manager Avancé ===
class AdvancedCacheManager(CacheManager):
    """
    Gestionnaire de cache ultra-avancé avec fonctionnalités entreprise.
    
    Fonctionnalités :
    - Clustering et réplication
    - Compression automatique
    - Sérialisation avancée
    - Monitoring et métriques
    - Circuit breaker
    - Cache warming
    - Éviction intelligente
    """
    
    def __init__(
        self,
        backend: CacheBackendInterface,
        enable_compression: bool = True,
        compression_threshold: int = 1024,
        enable_metrics: bool = True,
        circuit_breaker_enabled: bool = True,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 60.0
    ):
        super().__init__(backend)
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold
        self.enable_metrics = enable_metrics
        
        # Circuit breaker
        self.circuit_breaker_enabled = circuit_breaker_enabled
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        self._failure_count = 0
        self._last_failure_time = 0
        self._circuit_state = "closed"  # closed, open, half-open
        
        # Métriques avancées
        self._advanced_metrics = {
            'compression_ratio': 0.0,
            'serialization_time': 0.0,
            'network_latency': 0.0,
            'cache_warming_hits': 0,
            'circuit_breaker_trips': 0
        }
        
        # Cache de second niveau (local)
        self._l2_cache = MemoryCache(max_size=1000, default_ttl=300)
        
        self.logger = logging.getLogger(__name__)
    
    async def get(self, key: CacheKey) -> Optional[Any]:
        """Récupération avec L2 cache et circuit breaker."""
        if self._is_circuit_open():
            # Essayer le cache L2 si le circuit est ouvert
            result = await self._l2_cache.get(key)
            if result is not None:
                self.logger.info(f"L2 cache hit for key: {key}")
                return result
            return None
        
        try:
            start_time = time.time()
            
            # Essayer d'abord le cache L2
            result = await self._l2_cache.get(key)
            if result is not None:
                return result
            
            # Récupération depuis le backend principal
            result = await super().get(key)
            
            # Mise à jour du cache L2 si réussite
            if result is not None:
                await self._l2_cache.set(key, result, ttl=300)
            
            # Métriques de latence
            latency = time.time() - start_time
            self._advanced_metrics['network_latency'] = latency
            
            self._reset_circuit_breaker()
            return result
            
        except Exception as e:
            self._record_failure()
            self.logger.error(f"Cache get error for key {key}: {e}")
            
            # Fallback vers L2 cache
            result = await self._l2_cache.get(key)
            if result is not None:
                return result
            
            raise
    
    async def set(self, key: CacheKey, value: Any, ttl: Optional[TTL] = None) -> bool:
        """Stockage avec compression et circuit breaker."""
        if self._is_circuit_open():
            # Stocker uniquement dans le cache L2
            return await self._l2_cache.set(key, value, ttl)
        
        try:
            start_time = time.time()
            
            # Compression si activée
            processed_value = value
            if self.enable_compression and self._should_compress(value):
                processed_value = await self._compress_value(value)
            
            # Stockage dans le backend principal
            result = await super().set(key, processed_value, ttl)
            
            # Stockage dans le cache L2
            if result:
                await self._l2_cache.set(key, value, ttl=min(ttl or 300, 300))
            
            # Métriques de sérialisation
            serialization_time = time.time() - start_time
            self._advanced_metrics['serialization_time'] = serialization_time
            
            self._reset_circuit_breaker()
            return result
            
        except Exception as e:
            self._record_failure()
            self.logger.error(f"Cache set error for key {key}: {e}")
            
            # Fallback vers L2 cache
            return await self._l2_cache.set(key, value, ttl)
    
    async def delete(self, key: CacheKey) -> bool:
        """Suppression avec L2 cache."""
        # Supprimer du cache L2
        await self._l2_cache.delete(key)
        
        if self._is_circuit_open():
            return True
        
        try:
            result = await super().delete(key)
            self._reset_circuit_breaker()
            return result
        except Exception as e:
            self._record_failure()
            self.logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def warm_cache(self, keys_and_values: Dict[CacheKey, Any], ttl: Optional[TTL] = None):
        """Préchauffage du cache."""
        self.logger.info(f"Warming cache with {len(keys_and_values)} entries")
        
        success_count = 0
        for key, value in keys_and_values.items():
            try:
                if await self.set(key, value, ttl):
                    success_count += 1
                    self._advanced_metrics['cache_warming_hits'] += 1
            except Exception as e:
                self.logger.error(f"Cache warming failed for key {key}: {e}")
        
        self.logger.info(f"Cache warming completed: {success_count}/{len(keys_and_values)} successful")
    
    async def get_advanced_stats(self) -> Dict[str, Any]:
        """Statistiques avancées."""
        base_stats = await self.stats()
        
        return {
            **base_stats,
            'advanced_metrics': self._advanced_metrics,
            'circuit_breaker': {
                'state': self._circuit_state,
                'failure_count': self._failure_count,
                'trips': self._advanced_metrics['circuit_breaker_trips']
            },
            'l2_cache': await self._l2_cache.stats()
        }
    
    def _should_compress(self, value: Any) -> bool:
        """Détermine si une valeur doit être compressée."""
        try:
            serialized = pickle.dumps(value)
            return len(serialized) > self.compression_threshold
        except:
            return False
    
    async def _compress_value(self, value: Any) -> bytes:
        """Compresse une valeur."""
        import gzip
        
        serialized = pickle.dumps(value)
        compressed = gzip.compress(serialized)
        
        # Calcul du ratio de compression
        original_size = len(serialized)
        compressed_size = len(compressed)
        ratio = (original_size - compressed_size) / original_size
        self._advanced_metrics['compression_ratio'] = ratio
        
        self.logger.debug(f"Compressed value: {original_size} -> {compressed_size} bytes ({ratio:.2%})")
        
        return compressed
    
    def _is_circuit_open(self) -> bool:
        """Vérifie si le circuit breaker est ouvert."""
        if not self.circuit_breaker_enabled:
            return False
        
        if self._circuit_state == "open":
            # Vérifier si on peut passer en half-open
            if time.time() - self._last_failure_time > self.circuit_breaker_timeout:
                self._circuit_state = "half-open"
                self.logger.info("Circuit breaker half-open")
                return False
            return True
        
        return False
    
    def _record_failure(self):
        """Enregistre un échec pour le circuit breaker."""
        if not self.circuit_breaker_enabled:
            return
        
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._failure_count >= self.circuit_breaker_threshold:
            self._circuit_state = "open"
            self._advanced_metrics['circuit_breaker_trips'] += 1
            self.logger.warning(f"Circuit breaker opened after {self._failure_count} failures")
    
    def _reset_circuit_breaker(self):
        """Réinitialise le circuit breaker."""
        if self._circuit_state in ["half-open", "open"]:
            self._circuit_state = "closed"
            self._failure_count = 0
            self.logger.info("Circuit breaker closed")


# === Factory Functions ===
def create_memory_cache_manager(max_size: int = 10000, default_ttl: Optional[float] = None) -> CacheManager:
    """Crée un gestionnaire de cache mémoire."""
    backend = MemoryCache(max_size=max_size, default_ttl=default_ttl)
    return CacheManager(backend)

def create_redis_cache_manager(redis_url: str = "redis://localhost:6379", prefix: str = "cache:") -> CacheManager:
    """Crée un gestionnaire de cache Redis."""
    backend = RedisCache(redis_url=redis_url, prefix=prefix)
    return CacheManager(backend)

def create_advanced_cache_manager(
    backend_type: str = "memory",
    **kwargs
) -> AdvancedCacheManager:
    """Crée un gestionnaire de cache avancé."""
    if backend_type == "memory":
        backend = MemoryCache(**kwargs)
    elif backend_type == "redis":
        backend = RedisCache(**kwargs)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
    
    return AdvancedCacheManager(backend)

# === Exports ===
__all__ = [
    'CacheKey', 'TTL', 'CacheStats', 'CacheEntry',
    'CacheBackendInterface', 'MemoryCache', 'RedisCache',
    'CacheManager', 'AdvancedCacheManager',
    'create_memory_cache_manager', 'create_redis_cache_manager', 'create_advanced_cache_manager'
]
