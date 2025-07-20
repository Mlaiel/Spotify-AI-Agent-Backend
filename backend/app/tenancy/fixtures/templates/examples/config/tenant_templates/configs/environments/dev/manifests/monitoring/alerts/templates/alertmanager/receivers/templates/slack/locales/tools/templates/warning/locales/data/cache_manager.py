"""
Gestionnaire de Cache Intelligent - Spotify AI Agent
===================================================

Module avancé de gestion de cache pour les données de localisation avec support
Redis, stratégies de cache adaptatif, invalidation intelligente et métriques
de performance pour optimiser les performances du système multi-tenant.

Fonctionnalités:
- Cache distribué Redis avec clustering
- Stratégies d'invalidation intelligentes  
- Cache adaptatif avec TTL dynamique
- Métriques de performance intégrées
- Support des patterns de cache avancés

Author: Fahed Mlaiel
"""

import asyncio
import json
import hashlib
import time
from typing import Dict, List, Optional, Any, Union, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import logging
import redis.asyncio as redis
from contextlib import asynccontextmanager
import pickle
from functools import wraps
import inspect

from . import LocaleType

T = TypeVar('T')


class CacheStrategy(Enum):
    """Stratégies de mise en cache"""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    TTL = "ttl"           # Time To Live
    ADAPTIVE = "adaptive"  # TTL adaptatif basé sur l'usage
    WRITE_THROUGH = "write_through"    # Écriture immédiate
    WRITE_BEHIND = "write_behind"      # Écriture différée


class CacheLevel(Enum):
    """Niveaux de cache"""
    L1_MEMORY = "l1_memory"      # Cache mémoire local
    L2_REDIS = "l2_redis"        # Cache Redis distribué
    L3_DATABASE = "l3_database"  # Base de données


@dataclass
class CacheMetrics:
    """Métriques de performance du cache"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    total_requests: int = 0
    average_response_time: float = 0.0
    hit_ratio: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CacheEntry:
    """Entrée de cache avec métadonnées"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl: Optional[timedelta] = None
    tags: List[str] = field(default_factory=list)
    compressed: bool = False
    serialization_method: str = "json"


class CacheManager:
    """Gestionnaire de cache intelligent multi-niveaux"""
    
    def __init__(
        self, 
        redis_client: Optional[redis.Redis] = None,
        default_ttl: timedelta = timedelta(hours=1),
        max_memory_entries: int = 10000,
        compression_threshold: int = 1024
    ):
        self.logger = logging.getLogger(__name__)
        self.redis_client = redis_client
        self.default_ttl = default_ttl
        self.max_memory_entries = max_memory_entries
        self.compression_threshold = compression_threshold
        
        # Cache mémoire L1
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._memory_access_order: List[str] = []
        
        # Métriques
        self._metrics = CacheMetrics()
        
        # Configuration des stratégies
        self._strategies: Dict[str, CacheStrategy] = {}
        self._ttl_adaptations: Dict[str, timedelta] = {}
        
        # Locks pour la concurrence
        self._locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
    
    async def get(
        self, 
        key: str, 
        default: Any = None,
        deserializer: Optional[Callable] = None
    ) -> Any:
        """Récupère une valeur du cache"""
        start_time = time.time()
        
        try:
            # Normalise la clé
            normalized_key = self._normalize_key(key)
            
            # Essaie le cache L1 (mémoire)
            value = await self._get_from_memory(normalized_key)
            if value is not None:
                self._update_metrics('hit', time.time() - start_time)
                return self._deserialize_value(value, deserializer)
            
            # Essaie le cache L2 (Redis)
            if self.redis_client:
                value = await self._get_from_redis(normalized_key)
                if value is not None:
                    # Met à jour le cache L1
                    await self._set_to_memory(normalized_key, value)
                    self._update_metrics('hit', time.time() - start_time)
                    return self._deserialize_value(value, deserializer)
            
            # Cache miss
            self._update_metrics('miss', time.time() - start_time)
            return default
            
        except Exception as e:
            self.logger.error(f"Cache get error for key {key}: {e}")
            self._update_metrics('miss', time.time() - start_time)
            return default
    
    async def set(
        self, 
        key: str, 
        value: Any,
        ttl: Optional[timedelta] = None,
        tags: Optional[List[str]] = None,
        strategy: CacheStrategy = CacheStrategy.TTL,
        serializer: Optional[Callable] = None
    ) -> bool:
        """Stocke une valeur dans le cache"""
        start_time = time.time()
        
        try:
            normalized_key = self._normalize_key(key)
            serialized_value = self._serialize_value(value, serializer)
            effective_ttl = ttl or self._get_adaptive_ttl(normalized_key)
            
            # Stocke dans le cache L1
            await self._set_to_memory(
                normalized_key, 
                serialized_value, 
                effective_ttl, 
                tags or []
            )
            
            # Stocke dans le cache L2 si disponible
            if self.redis_client:
                await self._set_to_redis(
                    normalized_key, 
                    serialized_value, 
                    effective_ttl
                )
            
            # Enregistre la stratégie
            self._strategies[normalized_key] = strategy
            
            self._update_metrics('set', time.time() - start_time)
            return True
            
        except Exception as e:
            self.logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Supprime une entrée du cache"""
        try:
            normalized_key = self._normalize_key(key)
            
            # Supprime du cache L1
            await self._delete_from_memory(normalized_key)
            
            # Supprime du cache L2
            if self.redis_client:
                await self._delete_from_redis(normalized_key)
            
            self._metrics.deletes += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalide toutes les entrées avec les tags spécifiés"""
        invalidated_count = 0
        
        try:
            # Invalide dans le cache L1
            keys_to_delete = []
            for cache_key, entry in self._memory_cache.items():
                if any(tag in entry.tags for tag in tags):
                    keys_to_delete.append(cache_key)
            
            for key in keys_to_delete:
                await self._delete_from_memory(key)
                invalidated_count += 1
            
            # Invalide dans le cache L2 (Redis)
            if self.redis_client:
                # Utilise un pattern pour trouver les clés avec tags
                pattern = f"*:tags:*"
                async for key in self.redis_client.scan_iter(match=pattern):
                    key_tags = await self.redis_client.smembers(key)
                    if any(tag.encode() in key_tags for tag in tags):
                        cache_key = key.decode().replace(':tags:', ':')
                        await self._delete_from_redis(cache_key)
                        invalidated_count += 1
            
            self.logger.info(f"Invalidated {invalidated_count} cache entries with tags: {tags}")
            return invalidated_count
            
        except Exception as e:
            self.logger.error(f"Cache invalidation error: {e}")
            return 0
    
    async def _get_from_memory(self, key: str) -> Optional[Any]:
        """Récupère depuis le cache mémoire L1"""
        entry = self._memory_cache.get(key)
        if not entry:
            return None
        
        # Vérifie l'expiration
        if entry.ttl and datetime.now(timezone.utc) - entry.created_at > entry.ttl:
            await self._delete_from_memory(key)
            return None
        
        # Met à jour les statistiques d'accès
        entry.last_accessed = datetime.now(timezone.utc)
        entry.access_count += 1
        
        # Met à jour l'ordre d'accès (pour LRU)
        if key in self._memory_access_order:
            self._memory_access_order.remove(key)
        self._memory_access_order.append(key)
        
        return entry.value
    
    async def _set_to_memory(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[timedelta] = None,
        tags: Optional[List[str]] = None
    ):
        """Stocke dans le cache mémoire L1"""
        # Vérifie si on doit éviction
        if len(self._memory_cache) >= self.max_memory_entries:
            await self._evict_memory_entries()
        
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(timezone.utc),
            last_accessed=datetime.now(timezone.utc),
            ttl=ttl,
            tags=tags or []
        )
        
        self._memory_cache[key] = entry
        
        # Met à jour l'ordre d'accès
        if key in self._memory_access_order:
            self._memory_access_order.remove(key)
        self._memory_access_order.append(key)
    
    async def _get_from_redis(self, key: str) -> Optional[Any]:
        """Récupère depuis le cache Redis L2"""
        try:
            # Récupère la valeur principale
            value = await self.redis_client.get(key)
            if value is None:
                return None
            
            # Met à jour les statistiques d'accès dans Redis
            await self.redis_client.hincrby(f"{key}:stats", "access_count", 1)
            await self.redis_client.hset(
                f"{key}:stats", 
                "last_accessed", 
                datetime.now(timezone.utc).isoformat()
            )
            
            return pickle.loads(value)
            
        except Exception as e:
            self.logger.error(f"Redis get error: {e}")
            return None
    
    async def _set_to_redis(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[timedelta] = None
    ):
        """Stocke dans le cache Redis L2"""
        try:
            # Sérialise la valeur
            serialized_value = pickle.dumps(value)
            
            # Stocke la valeur principale
            if ttl:
                await self.redis_client.setex(
                    key, 
                    int(ttl.total_seconds()), 
                    serialized_value
                )
            else:
                await self.redis_client.set(key, serialized_value)
            
            # Stocke les métadonnées
            stats_key = f"{key}:stats"
            await self.redis_client.hset(stats_key, mapping={
                "created_at": datetime.now(timezone.utc).isoformat(),
                "access_count": 0,
                "last_accessed": datetime.now(timezone.utc).isoformat()
            })
            
            if ttl:
                await self.redis_client.expire(stats_key, int(ttl.total_seconds()))
                
        except Exception as e:
            self.logger.error(f"Redis set error: {e}")
    
    async def _delete_from_memory(self, key: str):
        """Supprime du cache mémoire"""
        if key in self._memory_cache:
            del self._memory_cache[key]
        
        if key in self._memory_access_order:
            self._memory_access_order.remove(key)
        
        if key in self._strategies:
            del self._strategies[key]
    
    async def _delete_from_redis(self, key: str):
        """Supprime du cache Redis"""
        try:
            await self.redis_client.delete(key)
            await self.redis_client.delete(f"{key}:stats")
            await self.redis_client.delete(f"{key}:tags")
        except Exception as e:
            self.logger.error(f"Redis delete error: {e}")
    
    async def _evict_memory_entries(self, count: int = 100):
        """Éviction d'entrées du cache mémoire selon la stratégie LRU"""
        evicted = 0
        
        # Éviction LRU (Least Recently Used)
        while len(self._memory_cache) > (self.max_memory_entries - count) and self._memory_access_order:
            oldest_key = self._memory_access_order.pop(0)
            if oldest_key in self._memory_cache:
                del self._memory_cache[oldest_key]
                evicted += 1
        
        self._metrics.evictions += evicted
        self.logger.debug(f"Evicted {evicted} entries from memory cache")
    
    def _normalize_key(self, key: str) -> str:
        """Normalise une clé de cache"""
        # Ajoute un préfixe pour éviter les collisions
        return f"locale_cache:{hashlib.md5(key.encode()).hexdigest()}"
    
    def _serialize_value(self, value: Any, serializer: Optional[Callable] = None) -> Any:
        """Sérialise une valeur pour le stockage"""
        if serializer:
            return serializer(value)
        
        # Sérialisation par défaut
        if isinstance(value, (str, int, float, bool)):
            return value
        
        return json.dumps(value, default=str)
    
    def _deserialize_value(self, value: Any, deserializer: Optional[Callable] = None) -> Any:
        """Désérialise une valeur depuis le stockage"""
        if deserializer:
            return deserializer(value)
        
        # Désérialisation par défaut
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        
        return value
    
    def _get_adaptive_ttl(self, key: str) -> timedelta:
        """Calcule un TTL adaptatif basé sur l'historique d'usage"""
        # Utilise le TTL par défaut si pas d'historique
        if key not in self._ttl_adaptations:
            return self.default_ttl
        
        # Adapte le TTL selon l'usage (logique simplifiée)
        base_ttl = self._ttl_adaptations.get(key, self.default_ttl)
        
        # Augmente le TTL pour les clés fréquemment accédées
        entry = self._memory_cache.get(key)
        if entry and entry.access_count > 10:
            return base_ttl * 2
        
        return base_ttl
    
    def _update_metrics(self, operation: str, response_time: float):
        """Met à jour les métriques de performance"""
        if operation == 'hit':
            self._metrics.hits += 1
        elif operation == 'miss':
            self._metrics.misses += 1
        elif operation == 'set':
            self._metrics.sets += 1
        
        self._metrics.total_requests += 1
        
        # Calcule la moyenne mobile du temps de réponse
        alpha = 0.1  # Facteur de lissage
        self._metrics.average_response_time = (
            alpha * response_time + 
            (1 - alpha) * self._metrics.average_response_time
        )
        
        # Calcule le ratio de hits
        total_gets = self._metrics.hits + self._metrics.misses
        if total_gets > 0:
            self._metrics.hit_ratio = self._metrics.hits / total_gets
        
        self._metrics.last_updated = datetime.now(timezone.utc)
    
    def get_metrics(self) -> CacheMetrics:
        """Retourne les métriques actuelles"""
        return self._metrics
    
    async def warm_up(self, keys_and_values: Dict[str, Any]):
        """Préchauffe le cache avec des données prédéfinies"""
        for key, value in keys_and_values.items():
            await self.set(key, value)
        
        self.logger.info(f"Cache warmed up with {len(keys_and_values)} entries")
    
    async def clear_all(self):
        """Vide complètement le cache"""
        # Vide le cache mémoire
        self._memory_cache.clear()
        self._memory_access_order.clear()
        self._strategies.clear()
        
        # Vide le cache Redis (avec pattern)
        if self.redis_client:
            pattern = "locale_cache:*"
            async for key in self.redis_client.scan_iter(match=pattern):
                await self.redis_client.delete(key)
        
        # Reset des métriques
        self._metrics = CacheMetrics()
        
        self.logger.info("All cache levels cleared")


def cached(
    ttl: Optional[timedelta] = None,
    key_func: Optional[Callable] = None,
    tags: Optional[List[str]] = None
):
    """Décorateur pour mettre en cache les résultats de fonction"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Génère la clé de cache
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Génère une clé basée sur la fonction et les arguments
                func_name = f"{func.__module__}.{func.__name__}"
                args_hash = hashlib.md5(
                    str(args + tuple(sorted(kwargs.items()))).encode()
                ).hexdigest()
                cache_key = f"{func_name}:{args_hash}"
            
            # Essaie de récupérer depuis le cache
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Exécute la fonction et cache le résultat
            result = await func(*args, **kwargs) if inspect.iscoroutinefunction(func) else func(*args, **kwargs)
            await cache_manager.set(cache_key, result, ttl=ttl, tags=tags)
            
            return result
        
        return wrapper
    return decorator


# Instance globale du gestionnaire de cache
cache_manager = CacheManager()

__all__ = [
    "CacheStrategy",
    "CacheLevel", 
    "CacheMetrics",
    "CacheEntry",
    "CacheManager",
    "cached",
    "cache_manager"
]
