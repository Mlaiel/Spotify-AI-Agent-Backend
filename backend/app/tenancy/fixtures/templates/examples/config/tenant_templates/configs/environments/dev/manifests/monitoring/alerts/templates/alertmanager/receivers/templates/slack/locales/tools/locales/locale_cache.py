"""
Cache de Locales Distribué pour Spotify AI Agent
Système de cache multi-niveaux haute performance pour les locales
"""

import asyncio
import json
import pickle
import logging
import zlib
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict, defaultdict
import threading
import weakref
from abc import ABC, abstractmethod
import hashlib

from redis.asyncio import Redis

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration du cache"""
    local_cache_size: int = 1000
    local_ttl: int = 300  # 5 minutes
    distributed_ttl: int = 3600  # 1 heure
    compression_threshold: int = 1024  # 1KB
    compression_level: int = 6
    enable_statistics: bool = True
    enable_persistence: bool = False
    persistence_interval: int = 300  # 5 minutes
    max_key_size: int = 250
    max_value_size: int = 1024 * 1024  # 1MB
    eviction_policy: str = "lru"  # lru, lfu, ttl
    enable_encryption: bool = False


@dataclass
class CacheEntry:
    """Entrée de cache avec métadonnées"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    ttl: int
    compressed: bool = False
    encrypted: bool = False
    tenant_id: Optional[str] = None
    locale_code: Optional[str] = None
    version: int = 1
    
    @property
    def is_expired(self) -> bool:
        """Vérifie si l'entrée a expiré"""
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl)
    
    @property
    def age_seconds(self) -> float:
        """Âge de l'entrée en secondes"""
        return (datetime.now() - self.created_at).total_seconds()


class CacheBackend(ABC):
    """Interface pour les backends de cache"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Récupère une entrée"""
        pass
    
    @abstractmethod
    async def set(self, key: str, entry: CacheEntry) -> bool:
        """Stocke une entrée"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Supprime une entrée"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Vide le cache"""
        pass
    
    @abstractmethod
    async def keys(self, pattern: str = "*") -> List[str]:
        """Liste les clés"""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques"""
        pass


class MemoryCacheBackend(CacheBackend):
    """Backend de cache en mémoire"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._cache = OrderedDict()
        self._access_counts = defaultdict(int)
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'evictions': 0
        }
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Récupère une entrée du cache mémoire"""
        try:
            with self._lock:
                if key in self._cache:
                    entry = self._cache[key]
                    
                    # Vérifier l'expiration
                    if entry.is_expired:
                        del self._cache[key]
                        self._access_counts.pop(key, None)
                        self._stats['misses'] += 1
                        return None
                    
                    # Mettre à jour les statistiques d'accès
                    entry.last_accessed = datetime.now()
                    entry.access_count += 1
                    self._access_counts[key] += 1
                    
                    # Réorganiser pour LRU
                    if self.config.eviction_policy == "lru":
                        self._cache.move_to_end(key)
                    
                    self._stats['hits'] += 1
                    return entry
                
                self._stats['misses'] += 1
                return None
                
        except Exception as e:
            logger.error(f"Memory cache get error: {e}")
            return None
    
    async def set(self, key: str, entry: CacheEntry) -> bool:
        """Stocke une entrée dans le cache mémoire"""
        try:
            with self._lock:
                # Vérifier la taille
                if len(key) > self.config.max_key_size:
                    logger.warning(f"Key too large: {len(key)} > {self.config.max_key_size}")
                    return False
                
                if entry.size_bytes > self.config.max_value_size:
                    logger.warning(f"Value too large: {entry.size_bytes} > {self.config.max_value_size}")
                    return False
                
                # Éviction si nécessaire
                while len(self._cache) >= self.config.local_cache_size:
                    await self._evict_entry()
                
                self._cache[key] = entry
                self._access_counts[key] = entry.access_count
                self._stats['sets'] += 1
                
                return True
                
        except Exception as e:
            logger.error(f"Memory cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Supprime une entrée du cache mémoire"""
        try:
            with self._lock:
                if key in self._cache:
                    del self._cache[key]
                    self._access_counts.pop(key, None)
                    self._stats['deletes'] += 1
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Memory cache delete error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Vide le cache mémoire"""
        try:
            with self._lock:
                self._cache.clear()
                self._access_counts.clear()
                return True
                
        except Exception as e:
            logger.error(f"Memory cache clear error: {e}")
            return False
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Liste les clés du cache mémoire"""
        try:
            with self._lock:
                if pattern == "*":
                    return list(self._cache.keys())
                
                # Filtrage simple par pattern
                import fnmatch
                return [k for k in self._cache.keys() if fnmatch.fnmatch(k, pattern)]
                
        except Exception as e:
            logger.error(f"Memory cache keys error: {e}")
            return []
    
    async def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache mémoire"""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                'backend_type': 'memory',
                'size': len(self._cache),
                'max_size': self.config.local_cache_size,
                'hit_rate': hit_rate,
                'stats': dict(self._stats),
                'memory_usage': sum(entry.size_bytes for entry in self._cache.values())
            }
    
    async def _evict_entry(self):
        """Évince une entrée selon la politique configurée"""
        if not self._cache:
            return
        
        if self.config.eviction_policy == "lru":
            # Supprimer la plus ancienne (début de OrderedDict)
            key, entry = self._cache.popitem(last=False)
        elif self.config.eviction_policy == "lfu":
            # Supprimer celle avec le moins d'accès
            min_key = min(self._access_counts.items(), key=lambda x: x[1])[0]
            self._cache.pop(min_key)
            key = min_key
        elif self.config.eviction_policy == "ttl":
            # Supprimer celle qui expire le plus tôt
            earliest_key = None
            earliest_expiry = None
            
            for k, entry in self._cache.items():
                expiry = entry.created_at + timedelta(seconds=entry.ttl)
                if earliest_expiry is None or expiry < earliest_expiry:
                    earliest_expiry = expiry
                    earliest_key = k
            
            if earliest_key:
                self._cache.pop(earliest_key)
                key = earliest_key
            else:
                key, _ = self._cache.popitem(last=False)
        else:
            # Par défaut: LRU
            key, entry = self._cache.popitem(last=False)
        
        self._access_counts.pop(key, None)
        self._stats['evictions'] += 1
        logger.debug(f"Evicted cache entry: {key}")


class RedisCacheBackend(CacheBackend):
    """Backend de cache Redis"""
    
    def __init__(self, redis_client: Redis, config: CacheConfig):
        self.redis = redis_client
        self.config = config
        self._key_prefix = "locale_cache:"
        self._stats = defaultdict(int)
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Récupère une entrée depuis Redis"""
        try:
            redis_key = f"{self._key_prefix}{key}"
            data = await self.redis.get(redis_key)
            
            if data:
                entry_data = pickle.loads(data)
                entry = CacheEntry(**entry_data)
                
                # Vérifier l'expiration
                if entry.is_expired:
                    await self.redis.delete(redis_key)
                    self._stats['misses'] += 1
                    return None
                
                # Mettre à jour les stats d'accès
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                
                # Remettre à jour dans Redis
                await self._update_entry_stats(redis_key, entry)
                
                self._stats['hits'] += 1
                return entry
            
            self._stats['misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"Redis cache get error: {e}")
            self._stats['errors'] += 1
            return None
    
    async def set(self, key: str, entry: CacheEntry) -> bool:
        """Stocke une entrée dans Redis"""
        try:
            redis_key = f"{self._key_prefix}{key}"
            
            # Sérialiser l'entrée
            entry_data = {
                'key': entry.key,
                'value': entry.value,
                'created_at': entry.created_at,
                'last_accessed': entry.last_accessed,
                'access_count': entry.access_count,
                'size_bytes': entry.size_bytes,
                'ttl': entry.ttl,
                'compressed': entry.compressed,
                'encrypted': entry.encrypted,
                'tenant_id': entry.tenant_id,
                'locale_code': entry.locale_code,
                'version': entry.version
            }
            
            serialized = pickle.dumps(entry_data)
            
            # Compression si nécessaire
            if len(serialized) > self.config.compression_threshold:
                serialized = zlib.compress(serialized, self.config.compression_level)
                entry.compressed = True
            
            # Stocker avec TTL
            await self.redis.setex(redis_key, entry.ttl, serialized)
            
            self._stats['sets'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Redis cache set error: {e}")
            self._stats['errors'] += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """Supprime une entrée de Redis"""
        try:
            redis_key = f"{self._key_prefix}{key}"
            result = await self.redis.delete(redis_key)
            
            if result > 0:
                self._stats['deletes'] += 1
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Redis cache delete error: {e}")
            self._stats['errors'] += 1
            return False
    
    async def clear(self) -> bool:
        """Vide le cache Redis"""
        try:
            pattern = f"{self._key_prefix}*"
            keys = await self.redis.keys(pattern)
            
            if keys:
                await self.redis.delete(*keys)
            
            return True
            
        except Exception as e:
            logger.error(f"Redis cache clear error: {e}")
            self._stats['errors'] += 1
            return False
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Liste les clés Redis"""
        try:
            redis_pattern = f"{self._key_prefix}{pattern}"
            redis_keys = await self.redis.keys(redis_pattern)
            
            # Supprimer le préfixe
            keys = [key.decode().replace(self._key_prefix, "") for key in redis_keys]
            return keys
            
        except Exception as e:
            logger.error(f"Redis cache keys error: {e}")
            return []
    
    async def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques Redis"""
        try:
            info = await self.redis.info()
            
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                'backend_type': 'redis',
                'hit_rate': hit_rate,
                'stats': dict(self._stats),
                'redis_info': {
                    'used_memory': info.get('used_memory', 0),
                    'connected_clients': info.get('connected_clients', 0),
                    'total_commands_processed': info.get('total_commands_processed', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Redis stats error: {e}")
            return {'backend_type': 'redis', 'error': str(e)}
    
    async def _update_entry_stats(self, redis_key: str, entry: CacheEntry):
        """Met à jour les statistiques d'accès dans Redis"""
        try:
            # Mise à jour asynchrone pour éviter de bloquer
            asyncio.create_task(self._async_update_stats(redis_key, entry))
        except Exception as e:
            logger.warning(f"Error updating entry stats: {e}")
    
    async def _async_update_stats(self, redis_key: str, entry: CacheEntry):
        """Mise à jour asynchrone des statistiques"""
        try:
            entry_data = {
                'key': entry.key,
                'value': entry.value,
                'created_at': entry.created_at,
                'last_accessed': entry.last_accessed,
                'access_count': entry.access_count,
                'size_bytes': entry.size_bytes,
                'ttl': entry.ttl,
                'compressed': entry.compressed,
                'encrypted': entry.encrypted,
                'tenant_id': entry.tenant_id,
                'locale_code': entry.locale_code,
                'version': entry.version
            }
            
            serialized = pickle.dumps(entry_data)
            await self.redis.setex(redis_key, entry.ttl, serialized)
            
        except Exception as e:
            logger.warning(f"Async stats update error: {e}")


class LocaleCache:
    """Cache de locales multi-niveaux"""
    
    def __init__(
        self,
        config: CacheConfig,
        redis_client: Optional[Redis] = None
    ):
        self.config = config
        self._memory_backend = MemoryCacheBackend(config)
        self._redis_backend = RedisCacheBackend(redis_client, config) if redis_client else None
        self._observers = weakref.WeakSet()
        self._stats = defaultdict(int)
        self._lock = threading.RLock()
    
    async def get(
        self,
        key: str,
        tenant_id: Optional[str] = None,
        locale_code: Optional[str] = None
    ) -> Optional[Any]:
        """Récupère une valeur du cache multi-niveaux"""
        try:
            cache_key = self._build_key(key, tenant_id, locale_code)
            
            # Niveau 1: Cache mémoire
            entry = await self._memory_backend.get(cache_key)
            if entry:
                self._stats['l1_hits'] += 1
                await self._notify_observers('cache_hit', {
                    'level': 'memory',
                    'key': cache_key,
                    'tenant_id': tenant_id
                })
                return entry.value
            
            # Niveau 2: Cache Redis
            if self._redis_backend:
                entry = await self._redis_backend.get(cache_key)
                if entry:
                    self._stats['l2_hits'] += 1
                    
                    # Promouvoir vers le cache mémoire
                    memory_entry = CacheEntry(
                        key=cache_key,
                        value=entry.value,
                        created_at=entry.created_at,
                        last_accessed=datetime.now(),
                        access_count=entry.access_count + 1,
                        size_bytes=entry.size_bytes,
                        ttl=self.config.local_ttl,
                        tenant_id=tenant_id,
                        locale_code=locale_code
                    )
                    
                    await self._memory_backend.set(cache_key, memory_entry)
                    
                    await self._notify_observers('cache_hit', {
                        'level': 'redis',
                        'key': cache_key,
                        'tenant_id': tenant_id
                    })
                    return entry.value
            
            # Cache miss
            self._stats['total_misses'] += 1
            await self._notify_observers('cache_miss', {
                'key': cache_key,
                'tenant_id': tenant_id
            })
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self._stats['errors'] += 1
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        tenant_id: Optional[str] = None,
        locale_code: Optional[str] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """Stocke une valeur dans le cache multi-niveaux"""
        try:
            cache_key = self._build_key(key, tenant_id, locale_code)
            
            # Calculer la taille
            size_bytes = len(pickle.dumps(value))
            
            # Niveau 1: Cache mémoire
            memory_entry = CacheEntry(
                key=cache_key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                size_bytes=size_bytes,
                ttl=ttl or self.config.local_ttl,
                tenant_id=tenant_id,
                locale_code=locale_code
            )
            
            memory_success = await self._memory_backend.set(cache_key, memory_entry)
            
            # Niveau 2: Cache Redis
            redis_success = True
            if self._redis_backend:
                redis_entry = CacheEntry(
                    key=cache_key,
                    value=value,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    access_count=1,
                    size_bytes=size_bytes,
                    ttl=ttl or self.config.distributed_ttl,
                    tenant_id=tenant_id,
                    locale_code=locale_code
                )
                
                redis_success = await self._redis_backend.set(cache_key, redis_entry)
            
            if memory_success or redis_success:
                self._stats['total_sets'] += 1
                await self._notify_observers('cache_set', {
                    'key': cache_key,
                    'tenant_id': tenant_id,
                    'size_bytes': size_bytes
                })
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            self._stats['errors'] += 1
            return False
    
    async def delete(
        self,
        key: str,
        tenant_id: Optional[str] = None,
        locale_code: Optional[str] = None
    ) -> bool:
        """Supprime une entrée du cache"""
        try:
            cache_key = self._build_key(key, tenant_id, locale_code)
            
            # Supprimer de tous les niveaux
            memory_success = await self._memory_backend.delete(cache_key)
            redis_success = True
            
            if self._redis_backend:
                redis_success = await self._redis_backend.delete(cache_key)
            
            if memory_success or redis_success:
                self._stats['total_deletes'] += 1
                await self._notify_observers('cache_delete', {
                    'key': cache_key,
                    'tenant_id': tenant_id
                })
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def clear_tenant(self, tenant_id: str) -> bool:
        """Vide le cache pour un tenant spécifique"""
        try:
            pattern = f"*:{tenant_id}:*"
            
            # Obtenir les clés correspondantes
            memory_keys = await self._memory_backend.keys(pattern)
            
            # Supprimer du cache mémoire
            for key in memory_keys:
                await self._memory_backend.delete(key)
            
            # Supprimer du cache Redis
            if self._redis_backend:
                redis_keys = await self._redis_backend.keys(pattern)
                for key in redis_keys:
                    await self._redis_backend.delete(key)
            
            self._stats['tenant_clears'] += 1
            await self._notify_observers('tenant_cache_clear', {
                'tenant_id': tenant_id,
                'keys_cleared': len(memory_keys)
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Tenant cache clear error: {e}")
            return False
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache"""
        try:
            memory_stats = await self._memory_backend.get_stats()
            redis_stats = {}
            
            if self._redis_backend:
                redis_stats = await self._redis_backend.get_stats()
            
            total_requests = (
                self._stats['l1_hits'] + 
                self._stats['l2_hits'] + 
                self._stats['total_misses']
            )
            
            overall_hit_rate = 0
            if total_requests > 0:
                overall_hit_rate = (self._stats['l1_hits'] + self._stats['l2_hits']) / total_requests
            
            return {
                'overall_stats': {
                    'total_requests': total_requests,
                    'overall_hit_rate': overall_hit_rate,
                    'l1_hit_rate': self._stats['l1_hits'] / total_requests if total_requests > 0 else 0,
                    'l2_hit_rate': self._stats['l2_hits'] / total_requests if total_requests > 0 else 0,
                    'stats': dict(self._stats)
                },
                'memory_backend': memory_stats,
                'redis_backend': redis_stats,
                'config': {
                    'local_cache_size': self.config.local_cache_size,
                    'local_ttl': self.config.local_ttl,
                    'distributed_ttl': self.config.distributed_ttl,
                    'compression_threshold': self.config.compression_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {}
    
    def add_observer(self, observer):
        """Ajoute un observateur pour les événements de cache"""
        self._observers.add(observer)
    
    def remove_observer(self, observer):
        """Supprime un observateur"""
        self._observers.discard(observer)
    
    def _build_key(
        self,
        key: str,
        tenant_id: Optional[str] = None,
        locale_code: Optional[str] = None
    ) -> str:
        """Construit une clé de cache complète"""
        parts = [key]
        
        if tenant_id:
            parts.insert(0, f"tenant:{tenant_id}")
        
        if locale_code:
            parts.append(f"locale:{locale_code}")
        
        return ":".join(parts)
    
    async def _notify_observers(self, event_type: str, data: Dict[str, Any]):
        """Notifie les observateurs"""
        try:
            for observer in list(self._observers):
                try:
                    if hasattr(observer, 'on_cache_event'):
                        await observer.on_cache_event(event_type, data)
                except Exception as e:
                    logger.warning(f"Observer notification error: {e}")
        except Exception as e:
            logger.error(f"Error notifying observers: {e}")


class DistributedCache:
    """Cache distribué avec synchronisation entre instances"""
    
    def __init__(self, locale_cache: LocaleCache):
        self.locale_cache = locale_cache
        self._sync_enabled = True
        self._sync_interval = 60  # 1 minute
        self._sync_task = None
        self._running = False
    
    async def start_sync(self):
        """Démarre la synchronisation"""
        if not self._running:
            self._running = True
            self._sync_task = asyncio.create_task(self._sync_loop())
            logger.info("Distributed cache sync started")
    
    async def stop_sync(self):
        """Arrête la synchronisation"""
        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        logger.info("Distributed cache sync stopped")
    
    async def invalidate_globally(
        self,
        key: str,
        tenant_id: Optional[str] = None,
        locale_code: Optional[str] = None
    ):
        """Invalide une clé globalement sur toutes les instances"""
        try:
            # Supprimer localement
            await self.locale_cache.delete(key, tenant_id, locale_code)
            
            # Publier l'invalidation
            if self.locale_cache._redis_backend:
                invalidation_message = {
                    'action': 'invalidate',
                    'key': key,
                    'tenant_id': tenant_id,
                    'locale_code': locale_code,
                    'timestamp': datetime.now().isoformat()
                }
                
                await self.locale_cache._redis_backend.redis.publish(
                    'locale_cache_invalidation',
                    json.dumps(invalidation_message)
                )
            
        except Exception as e:
            logger.error(f"Global invalidation error: {e}")
    
    async def _sync_loop(self):
        """Boucle de synchronisation"""
        while self._running:
            try:
                await asyncio.sleep(self._sync_interval)
                await self._perform_sync()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync loop error: {e}")
    
    async def _perform_sync(self):
        """Effectue la synchronisation"""
        try:
            # Écouter les messages d'invalidation
            if self.locale_cache._redis_backend:
                # Implémentation de la synchronisation via Redis pub/sub
                pass
        except Exception as e:
            logger.error(f"Sync error: {e}")
