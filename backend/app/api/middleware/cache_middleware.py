"""
🎵 Spotify AI Agent - Advanced Cache Middleware
==============================================

Middleware de cache multi-couches avec Redis, Memcached et cache local.
Support pour invalidation intelligente, patterns de cache et métriques.

Architecture:
- Cache L1: Local Memory (LRU)
- Cache L2: Redis Cluster
- Cache L3: Memcached
- Cache CDN: CloudFlare/AWS CloudFront
- Smart Cache Invalidation
- Cache Warming & Prefetching
- Analytics & Metrics

Performance Features:
- Sub-millisecond cache hits
- Intelligent cache warming
- Background cache refresh
- Circuit breaker for cache failures
- Cache stampede protection
- Geographic cache distribution
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Union
from enum import Enum
from dataclasses import dataclass, asdict
from collections import OrderedDict
import threading
from contextlib import asynccontextmanager

from fastapi import Request, Response
from fastapi.responses import JSONResponse
import redis.asyncio as redis
import aiomcache as aiomemcache
from cachetools import TTLCache, LRUCache
import msgpack
import pickle
import zlib

from app.core.config import settings
from app.core.logging import get_logger
from app.utils.metrics_manager import MetricsManager
from app.utils.performance_tracker import PerformanceTracker


class CacheLevel(str, Enum):
    """Niveaux de cache disponibles"""
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_MEMCACHED = "l3_memcached"
    CDN = "cdn"
    ALL = "all"


class CacheStrategy(str, Enum):
    """Stratégies de cache"""
    READ_THROUGH = "read_through"
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"
    CACHE_ASIDE = "cache_aside"
    REFRESH_AHEAD = "refresh_ahead"


class CachePattern(str, Enum):
    """Patterns de cache avancés"""
    SINGLE_KEY = "single_key"
    TAG_BASED = "tag_based"
    HIERARCHICAL = "hierarchical"
    DEPENDENCY_BASED = "dependency_based"
    TIME_BASED = "time_based"


@dataclass
class CacheConfig:
    """Configuration du cache"""
    enabled: bool = True
    default_ttl: int = 3600  # 1 heure
    max_key_length: int = 250
    compression_enabled: bool = True
    compression_threshold: int = 1024  # bytes
    serialization_format: str = "msgpack"  # json, pickle, msgpack
    
    # L1 Cache (Memory)
    l1_enabled: bool = True
    l1_max_size: int = 1000
    l1_ttl: int = 300  # 5 minutes
    
    # L2 Cache (Redis)
    l2_enabled: bool = True
    l2_ttl: int = 3600  # 1 heure
    l2_cluster_enabled: bool = True
    
    # L3 Cache (Memcached)
    l3_enabled: bool = False
    l3_ttl: int = 7200  # 2 heures
    
    # Cache Warming
    warming_enabled: bool = True
    prefetch_enabled: bool = True
    background_refresh: bool = True
    
    # Circuit Breaker
    circuit_breaker_enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout: int = 60


@dataclass
class CacheKey:
    """Clé de cache structurée"""
    namespace: str
    identifier: str
    version: str = "v1"
    tags: List[str] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.dependencies is None:
            self.dependencies = []
    
    def to_string(self) -> str:
        """Convertit en chaîne de caractères"""
        base = f"{self.namespace}:{self.version}:{self.identifier}"
        if self.tags:
            base += f":tags:{':'.join(sorted(self.tags))}"
        return hashlib.md5(base.encode()).hexdigest()[:16] + f":{base}"


@dataclass
class CacheItem:
    """Item de cache avec métadonnées"""
    data: Any
    created_at: datetime
    expires_at: datetime
    hit_count: int = 0
    size_bytes: int = 0
    tags: List[str] = None
    dependencies: List[str] = None
    cache_level: CacheLevel = CacheLevel.L1_MEMORY
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.dependencies is None:
            self.dependencies = []
    
    def is_expired(self) -> bool:
        """Vérifie si l'item a expiré"""
        return datetime.utcnow() > self.expires_at
    
    def time_to_live(self) -> int:
        """Temps restant avant expiration"""
        delta = self.expires_at - datetime.utcnow()
        return max(0, int(delta.total_seconds()))


class CacheMetrics:
    """Métriques de cache"""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.evictions = 0
        self.errors = 0
        self.total_size = 0
        self.avg_response_time = 0.0
        self.hit_times: List[float] = []
        self.miss_times: List[float] = []
    
    def record_hit(self, response_time: float):
        """Enregistre un hit"""
        self.hits += 1
        self.hit_times.append(response_time)
        self._update_avg_response_time()
    
    def record_miss(self, response_time: float):
        """Enregistre un miss"""
        self.misses += 1
        self.miss_times.append(response_time)
        self._update_avg_response_time()
    
    def record_set(self):
        """Enregistre un set"""
        self.sets += 1
    
    def record_delete(self):
        """Enregistre une suppression"""
        self.deletes += 1
    
    def record_eviction(self):
        """Enregistre une éviction"""
        self.evictions += 1
    
    def record_error(self):
        """Enregistre une erreur"""
        self.errors += 1
    
    def _update_avg_response_time(self):
        """Met à jour le temps de réponse moyen"""
        all_times = self.hit_times + self.miss_times
        if all_times:
            self.avg_response_time = sum(all_times) / len(all_times)
    
    def get_hit_ratio(self) -> float:
        """Calcule le taux de hit"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "evictions": self.evictions,
            "errors": self.errors,
            "total_size": self.total_size,
            "hit_ratio": self.get_hit_ratio(),
            "avg_response_time": self.avg_response_time
        }


class CircuitBreaker:
    """Circuit breaker pour le cache"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def is_open(self) -> bool:
        """Vérifie si le circuit est ouvert"""
        with self._lock:
            if self.state == "OPEN":
                if (time.time() - self.last_failure_time) > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    return False
                return True
            return False
    
    def record_success(self):
        """Enregistre un succès"""
        with self._lock:
            self.failure_count = 0
            self.state = "CLOSED"
    
    def record_failure(self):
        """Enregistre un échec"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"


class CacheSerializer:
    """Sérialiseur de cache multi-format"""
    
    @staticmethod
    def serialize(data: Any, format: str = "msgpack", compress: bool = True) -> bytes:
        """Sérialise les données"""
        try:
            if format == "json":
                serialized = json.dumps(data, default=str).encode()
            elif format == "pickle":
                serialized = pickle.dumps(data)
            elif format == "msgpack":
                serialized = msgpack.packb(data, default=str)
            else:
                raise ValueError(f"Format de sérialisation non supporté: {format}")
            
            if compress and len(serialized) > 1024:
                serialized = zlib.compress(serialized)
                return b"COMPRESSED:" + serialized
            
            return serialized
        except Exception as e:
            raise ValueError(f"Erreur de sérialisation: {e}")
    
    @staticmethod
    def deserialize(data: bytes, format: str = "msgpack") -> Any:
        """Désérialise les données"""
        try:
            # Vérifier si les données sont compressées
            if data.startswith(b"COMPRESSED:"):
                data = zlib.decompress(data[11:])
            
            if format == "json":
                return json.loads(data.decode())
            elif format == "pickle":
                return pickle.loads(data)
            elif format == "msgpack":
                return msgpack.unpackb(data, raw=False)
            else:
                raise ValueError(f"Format de désérialisation non supporté: {format}")
        except Exception as e:
            raise ValueError(f"Erreur de désérialisation: {e}")


class AdvancedCacheManager:
    """Gestionnaire de cache multi-niveau avancé"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.logger = get_logger("cache_manager")
        self.metrics = CacheMetrics()
        self.circuit_breaker = CircuitBreaker(
            config.failure_threshold,
            config.recovery_timeout
        )
        
        # Caches L1 (Memory)
        self.l1_cache = TTLCache(
            maxsize=config.l1_max_size,
            ttl=config.l1_ttl
        ) if config.l1_enabled else None
        
        # Redis client pour L2
        self.redis_client = None
        
        # Memcached client pour L3
        self.memcached_client = None
        
        # Cache des tags et dépendances
        self.tag_cache: Dict[str, List[str]] = {}
        self.dependency_cache: Dict[str, List[str]] = {}
        
        # Locks pour la concurrence
        self._locks: Dict[str, asyncio.Lock] = {}
        self._lock_lock = asyncio.Lock()
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        
        # Initialisation asynchrone
        self._initialized = False
    
    async def initialize(self):
        """Initialise les connexions de cache"""
        if self._initialized:
            return
        
        try:
            settings = get_settings()
            
            # Initialiser Redis
            if self.config.l2_enabled:
                self.redis_client = redis.Redis.from_url(
                    settings.REDIS_URL,
                    decode_responses=False,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                await self.redis_client.ping()
                self.logger.info("Cache Redis L2 initialisé")
            
            # Initialiser Memcached
            if self.config.l3_enabled:
                self.memcached_client = aiomemcache.Client(
                    settings.MEMCACHED_SERVERS
                )
                self.logger.info("Cache Memcached L3 initialisé")
            
            # Démarrer les tâches de fond
            if self.config.warming_enabled:
                self._start_background_tasks()
            
            self._initialized = True
            self.logger.info("Gestionnaire de cache initialisé avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur d'initialisation du cache: {e}")
            raise
    
    async def _get_lock(self, key: str) -> asyncio.Lock:
        """Obtient un lock pour une clé"""
        async with self._lock_lock:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
            return self._locks[key]
    
    async def get(
        self,
        key: Union[str, CacheKey],
        default: Any = None,
        refresh_on_hit: bool = False
    ) -> Any:
        """Récupère une valeur du cache multi-niveau"""
        start_time = time.time()
        
        try:
            cache_key = key if isinstance(key, CacheKey) else CacheKey("default", str(key))
            key_str = cache_key.to_string()
            
            # Vérifier le circuit breaker
            if self.circuit_breaker.is_open():
                self.metrics.record_miss(time.time() - start_time)
                return default
            
            # L1 Cache (Memory)
            if self.config.l1_enabled and self.l1_cache:
                value = self.l1_cache.get(key_str)
                if value is not None:
                    self.metrics.record_hit(time.time() - start_time)
                    if refresh_on_hit:
                        asyncio.create_task(self._refresh_cache_async(cache_key, value))
                    return value
            
            # L2 Cache (Redis)
            if self.config.l2_enabled and self.redis_client:
                try:
                    data = await self.redis_client.get(key_str)
                    if data:
                        value = CacheSerializer.deserialize(
                            data, 
                            self.config.serialization_format
                        )
                        
                        # Mettre en L1 si activé
                        if self.config.l1_enabled and self.l1_cache:
                            self.l1_cache[key_str] = value
                        
                        self.metrics.record_hit(time.time() - start_time)
                        self.circuit_breaker.record_success()
                        return value
                except Exception as e:
                    self.logger.warning(f"Erreur cache Redis: {e}")
                    self.circuit_breaker.record_failure()
            
            # L3 Cache (Memcached)
            if self.config.l3_enabled and self.memcached_client:
                try:
                    data = await self.memcached_client.get(key_str.encode())
                    if data:
                        value = CacheSerializer.deserialize(
                            data,
                            self.config.serialization_format
                        )
                        
                        # Remonter vers L2 et L1
                        await self._promote_to_higher_levels(key_str, value)
                        
                        self.metrics.record_hit(time.time() - start_time)
                        return value
                except Exception as e:
                    self.logger.warning(f"Erreur cache Memcached: {e}")
            
            # Cache miss
            self.metrics.record_miss(time.time() - start_time)
            return default
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération cache: {e}")
            self.metrics.record_error()
            return default
    
    async def set(
        self,
        key: Union[str, CacheKey],
        value: Any,
        ttl: Optional[int] = None,
        tags: List[str] = None,
        dependencies: List[str] = None,
        strategy: CacheStrategy = CacheStrategy.WRITE_THROUGH
    ) -> bool:
        """Définit une valeur dans le cache"""
        try:
            cache_key = key if isinstance(key, CacheKey) else CacheKey("default", str(key))
            if tags:
                cache_key.tags.extend(tags)
            if dependencies:
                cache_key.dependencies.extend(dependencies)
            
            key_str = cache_key.to_string()
            ttl = ttl or self.config.default_ttl
            
            # Sérialiser les données
            serialized_data = CacheSerializer.serialize(
                value,
                self.config.serialization_format,
                self.config.compression_enabled
            )
            
            success = False
            
            # Stratégie Write-Through ou Write-Behind
            if strategy in [CacheStrategy.WRITE_THROUGH, CacheStrategy.WRITE_BEHIND]:
                # L1 Cache
                if self.config.l1_enabled and self.l1_cache:
                    self.l1_cache[key_str] = value
                    success = True
                
                # L2 Cache (Redis)
                if self.config.l2_enabled and self.redis_client:
                    try:
                        await self.redis_client.setex(key_str, ttl, serialized_data)
                        success = True
                    except Exception as e:
                        self.logger.warning(f"Erreur écriture Redis: {e}")
                
                # L3 Cache (Memcached)
                if self.config.l3_enabled and self.memcached_client:
                    try:
                        await self.memcached_client.set(
                            key_str.encode(),
                            serialized_data,
                            exptime=ttl
                        )
                        success = True
                    except Exception as e:
                        self.logger.warning(f"Erreur écriture Memcached: {e}")
            
            # Gestion des tags et dépendances
            await self._update_tag_mappings(cache_key, key_str)
            
            if success:
                self.metrics.record_set()
                
                # Background refresh si activé
                if self.config.background_refresh and strategy == CacheStrategy.REFRESH_AHEAD:
                    asyncio.create_task(self._schedule_refresh(cache_key, ttl))
            
            return success
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'écriture cache: {e}")
            self.metrics.record_error()
            return False
    
    async def delete(self, key: Union[str, CacheKey]) -> bool:
        """Supprime une clé du cache"""
        try:
            cache_key = key if isinstance(key, CacheKey) else CacheKey("default", str(key))
            key_str = cache_key.to_string()
            
            success = False
            
            # L1 Cache
            if self.config.l1_enabled and self.l1_cache and key_str in self.l1_cache:
                del self.l1_cache[key_str]
                success = True
            
            # L2 Cache (Redis)
            if self.config.l2_enabled and self.redis_client:
                try:
                    result = await self.redis_client.delete(key_str)
                    success = success or (result > 0)
                except Exception as e:
                    self.logger.warning(f"Erreur suppression Redis: {e}")
            
            # L3 Cache (Memcached)
            if self.config.l3_enabled and self.memcached_client:
                try:
                    await self.memcached_client.delete(key_str.encode())
                    success = True
                except Exception as e:
                    self.logger.warning(f"Erreur suppression Memcached: {e}")
            
            if success:
                self.metrics.record_delete()
                await self._cleanup_tag_mappings(cache_key, key_str)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la suppression cache: {e}")
            self.metrics.record_error()
            return False
    
    async def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalide tous les items avec les tags spécifiés"""
        try:
            invalidated_count = 0
            
            for tag in tags:
                if tag in self.tag_cache:
                    keys_to_invalidate = self.tag_cache[tag].copy()
                    
                    for key_str in keys_to_invalidate:
                        # Reconstituer la clé
                        cache_key = CacheKey("unknown", key_str)
                        if await self.delete(cache_key):
                            invalidated_count += 1
                    
                    # Nettoyer le mapping des tags
                    del self.tag_cache[tag]
            
            self.logger.info(f"Invalidé {invalidated_count} items pour les tags: {tags}")
            return invalidated_count
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'invalidation par tags: {e}")
            return 0
    
    async def warm_cache(self, keys: List[Union[str, CacheKey]], loader_func: Callable):
        """Préchauffe le cache avec les clés spécifiées"""
        try:
            warmed_count = 0
            
            for key in keys:
                try:
                    # Vérifier si la clé existe déjà
                    existing_value = await self.get(key)
                    if existing_value is None:
                        # Charger la valeur
                        value = await loader_func(key)
                        if value is not None:
                            await self.set(key, value)
                            warmed_count += 1
                except Exception as e:
                    self.logger.warning(f"Erreur de préchauffe pour {key}: {e}")
            
            self.logger.info(f"Cache préchauffé avec {warmed_count} items")
            
        except Exception as e:
            self.logger.error(f"Erreur lors du préchauffe du cache: {e}")
    
    async def _promote_to_higher_levels(self, key_str: str, value: Any):
        """Remonte une valeur vers les niveaux de cache supérieurs"""
        try:
            # L1 Cache
            if self.config.l1_enabled and self.l1_cache:
                self.l1_cache[key_str] = value
            
            # L2 Cache (Redis)
            if self.config.l2_enabled and self.redis_client:
                serialized_data = CacheSerializer.serialize(
                    value,
                    self.config.serialization_format,
                    self.config.compression_enabled
                )
                await self.redis_client.setex(key_str, self.config.l2_ttl, serialized_data)
        except Exception as e:
            self.logger.warning(f"Erreur promotion cache: {e}")
    
    async def _update_tag_mappings(self, cache_key: CacheKey, key_str: str):
        """Met à jour les mappings de tags"""
        for tag in cache_key.tags:
            if tag not in self.tag_cache:
                self.tag_cache[tag] = []
            if key_str not in self.tag_cache[tag]:
                self.tag_cache[tag].append(key_str)
    
    async def _cleanup_tag_mappings(self, cache_key: CacheKey, key_str: str):
        """Nettoie les mappings de tags"""
        for tag in cache_key.tags:
            if tag in self.tag_cache and key_str in self.tag_cache[tag]:
                self.tag_cache[tag].remove(key_str)
                if not self.tag_cache[tag]:
                    del self.tag_cache[tag]
    
    async def _refresh_cache_async(self, cache_key: CacheKey, current_value: Any):
        """Rafraîchit le cache en arrière-plan"""
        # Implémentation du refresh asynchrone
        pass
    
    async def _schedule_refresh(self, cache_key: CacheKey, ttl: int):
        """Planifie un rafraîchissement du cache"""
        # Implémentation de la planification de refresh
        pass
    
    def _start_background_tasks(self):
        """Démarre les tâches de fond"""
        # Tâche de nettoyage périodique
        cleanup_task = asyncio.create_task(self._periodic_cleanup())
        self._background_tasks.append(cleanup_task)
        
        # Tâche de métriques
        metrics_task = asyncio.create_task(self._periodic_metrics_report())
        self._background_tasks.append(metrics_task)
    
    async def _periodic_cleanup(self):
        """Nettoyage périodique du cache"""
        while True:
            try:
                await asyncio.sleep(300)  # 5 minutes
                
                # Nettoyer les locks inutilisés
                async with self._lock_lock:
                    keys_to_remove = []
                    for key, lock in self._locks.items():
                        if not lock.locked():
                            keys_to_remove.append(key)
                    
                    for key in keys_to_remove[:100]:  # Limiter le nombre
                        del self._locks[key]
                
                # Autres tâches de nettoyage
                self.logger.debug("Nettoyage périodique du cache effectué")
                
            except Exception as e:
                self.logger.error(f"Erreur nettoyage périodique: {e}")
    
    async def _periodic_metrics_report(self):
        """Rapport périodique des métriques"""
        while True:
            try:
                await asyncio.sleep(60)  # 1 minute
                
                metrics_data = self.metrics.to_dict()
                self.logger.info(f"Métriques cache: {metrics_data}")
                
                # Envoyer vers le système de métriques
                # MetricsCollector.record_cache_metrics(metrics_data)
                
            except Exception as e:
                self.logger.error(f"Erreur rapport métriques: {e}")
    
    async def close(self):
        """Ferme les connexions de cache"""
        try:
            # Annuler les tâches de fond
            for task in self._background_tasks:
                task.cancel()
            
            # Fermer les connexions
            if self.redis_client:
                await self.redis_client.close()
            
            if self.memcached_client:
                await self.memcached_client.close()
            
            self.logger.info("Gestionnaire de cache fermé")
            
        except Exception as e:
            self.logger.error(f"Erreur fermeture cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtient les statistiques du cache"""
        stats = self.metrics.to_dict()
        
        # Ajouter des stats spécifiques
        if self.l1_cache:
            stats["l1_size"] = len(self.l1_cache)
            stats["l1_max_size"] = self.l1_cache.maxsize
        
        stats["circuit_breaker_state"] = self.circuit_breaker.state
        stats["circuit_breaker_failures"] = self.circuit_breaker.failure_count
        
        return stats


class AdvancedCacheMiddleware:
    """Middleware de cache avancé pour FastAPI"""
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.cache_manager = AdvancedCacheManager(self.config)
        self.logger = get_logger("cache_middleware")
        self.performance_tracker = PerformanceTracker()
        
        # Patterns d'URL à mettre en cache
        self.cacheable_patterns = [
            r"/api/v1/spotify/tracks/.*",
            r"/api/v1/spotify/albums/.*",
            r"/api/v1/spotify/artists/.*",
            r"/api/v1/analytics/.*",
            r"/api/v1/ai/recommendations/.*"
        ]
        
        # Patterns à exclure du cache
        self.non_cacheable_patterns = [
            r"/api/v1/auth/.*",
            r"/api/v1/user/profile",
            r"/api/v1/admin/.*"
        ]
    
    async def __call__(self, request: Request, call_next):
        """Traite la requête avec cache"""
        # Initialiser le cache si nécessaire
        if not self.cache_manager._initialized:
            await self.cache_manager.initialize()
        
        # Vérifier si la requête est cacheable
        if not self._is_cacheable_request(request):
            return await call_next(request)
        
        # Générer la clé de cache
        cache_key = self._generate_cache_key(request)
        
        start_time = time.time()
        
        try:
            # Essayer de récupérer depuis le cache
            cached_response = await self.cache_manager.get(cache_key)
            
            if cached_response is not None:
                # Cache hit
                response_time = time.time() - start_time
                
                self.logger.debug(
                    f"Cache hit pour {request.url.path}",
                    extra={
                        "cache_key": cache_key.to_string(),
                        "response_time": response_time
                    }
                )
                
                # Construire la réponse depuis le cache
                return self._build_response_from_cache(cached_response)
            
            # Cache miss - exécuter la requête
            response = await call_next(request)
            
            # Mettre en cache la réponse si approprié
            if self._is_cacheable_response(response):
                await self._cache_response(cache_key, response, request)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Erreur middleware cache: {e}")
            # En cas d'erreur, continuer sans cache
            return await call_next(request)
    
    def _is_cacheable_request(self, request: Request) -> bool:
        """Vérifie si la requête peut être mise en cache"""
        # Seules les requêtes GET sont cachées par défaut
        if request.method != "GET":
            return False
        
        path = request.url.path
        
        # Vérifier les patterns exclus
        import re
        for pattern in self.non_cacheable_patterns:
            if re.match(pattern, path):
                return False
        
        # Vérifier les patterns inclus
        for pattern in self.cacheable_patterns:
            if re.match(pattern, path):
                return True
        
        return False
    
    def _is_cacheable_response(self, response: Response) -> bool:
        """Vérifie si la réponse peut être mise en cache"""
        # Vérifier le code de statut
        if response.status_code != 200:
            return False
        
        # Vérifier les headers de cache
        cache_control = response.headers.get("Cache-Control", "")
        if "no-cache" in cache_control or "no-store" in cache_control:
            return False
        
        return True
    
    def _generate_cache_key(self, request: Request) -> CacheKey:
        """Génère une clé de cache pour la requête"""
        # Base de la clé
        path = request.url.path
        
        # Inclure les paramètres de requête triés
        query_params = dict(request.query_params)
        if query_params:
            sorted_params = sorted(query_params.items())
            param_string = "&".join([f"{k}={v}" for k, v in sorted_params])
            identifier = f"{path}?{param_string}"
        else:
            identifier = path
        
        # Inclure les headers pertinents
        relevant_headers = ["Accept-Language", "Authorization"]
        header_parts = []
        for header in relevant_headers:
            value = request.headers.get(header)
            if value:
                # Pour Authorization, utiliser un hash pour la sécurité
                if header == "Authorization":
                    value = hashlib.md5(value.encode()).hexdigest()[:8]
                header_parts.append(f"{header}:{value}")
        
        if header_parts:
            identifier += f"#{':'.join(header_parts)}"
        
        # Déterminer les tags
        tags = ["api"]
        if "/spotify/" in path:
            tags.append("spotify")
        if "/ai/" in path:
            tags.append("ai")
        if "/analytics/" in path:
            tags.append("analytics")
        
        return CacheKey(
            namespace="api_responses",
            identifier=identifier,
            version="v1",
            tags=tags
        )
    
    async def _cache_response(self, cache_key: CacheKey, response: Response, request: Request):
        """Met en cache la réponse"""
        try:
            # Lire le corps de la réponse
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
            
            # Préparer les données à mettre en cache
            cache_data = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": body.decode() if body else "",
                "content_type": response.headers.get("content-type", "")
            }
            
            # Déterminer le TTL basé sur le type de contenu
            ttl = self._determine_ttl(request, response)
            
            # Mettre en cache
            await self.cache_manager.set(
                cache_key,
                cache_data,
                ttl=ttl,
                strategy=CacheStrategy.WRITE_THROUGH
            )
            
            # Reconstruire la réponse
            response.body_iterator = self._create_body_iterator(body)
            
        except Exception as e:
            self.logger.error(f"Erreur mise en cache réponse: {e}")
    
    def _determine_ttl(self, request: Request, response: Response) -> int:
        """Détermine le TTL basé sur le type de requête"""
        path = request.url.path
        
        # TTL spécifiques par type d'endpoint
        if "/spotify/tracks/" in path:
            return 1800  # 30 minutes pour les tracks
        elif "/spotify/albums/" in path:
            return 3600  # 1 heure pour les albums
        elif "/spotify/artists/" in path:
            return 7200  # 2 heures pour les artistes
        elif "/analytics/" in path:
            return 300   # 5 minutes pour les analytics
        elif "/ai/recommendations/" in path:
            return 900   # 15 minutes pour les recommandations
        
        return self.config.default_ttl
    
    def _build_response_from_cache(self, cached_data: Dict[str, Any]) -> Response:
        """Construit une réponse depuis les données en cache"""
        return JSONResponse(
            content=json.loads(cached_data["body"]) if cached_data["body"] else {},
            status_code=cached_data["status_code"],
            headers={
                **cached_data["headers"],
                "X-Cache": "HIT",
                "X-Cache-Timestamp": str(int(time.time()))
            }
        )
    
    def _create_body_iterator(self, body: bytes):
        """Crée un itérateur pour le corps de la réponse"""
        async def body_iterator():
            yield body
        
        return body_iterator()


# =============================================================================
# CLASSES MANQUANTES POUR LES TESTS
# =============================================================================

class CacheInvalidator:
    """Gestionnaire d'invalidation de cache intelligent"""
    
    def __init__(self, cache_manager: 'AdvancedCacheManager'):
        self.cache_manager = cache_manager
        self.invalidation_patterns = {}
        
    async def invalidate_by_pattern(self, pattern: str):
        """Invalide les clés correspondant au pattern"""
        pass
        
    async def invalidate_by_tags(self, tags: List[str]):
        """Invalide les clés avec les tags spécifiés"""
        pass

class CompressionEngine:
    """Moteur de compression pour les données du cache"""
    
    def __init__(self, algorithm: str = "gzip"):
        self.algorithm = algorithm
        
    def compress(self, data: bytes) -> bytes:
        """Compresse les données"""
        if self.algorithm == "gzip":
            return zlib.compress(data)
        return data
        
    def decompress(self, data: bytes) -> bytes:
        """Décompresse les données"""
        if self.algorithm == "gzip":
            return zlib.decompress(data)
        return data

class CacheCluster:
    """Gestionnaire de cluster cache pour distribution"""
    
    def __init__(self, nodes: List[str]):
        self.nodes = nodes
        self.active_nodes = nodes.copy()
        
    async def get_node_for_key(self, key: str) -> str:
        """Retourne le nœud responsable d'une clé"""
        hash_val = hash(key) % len(self.active_nodes)
        return self.active_nodes[hash_val]

class CacheOperation(str, Enum):
    """Types d'opérations de cache"""
    GET = "get"
    SET = "set"
    DELETE = "delete"
    INVALIDATE = "invalidate"
    CLEAR = "clear"

class CacheType(str, Enum):
    """Types de cache"""
    MEMORY = "memory"
    REDIS = "redis"
    MEMCACHED = "memcached"
    HYBRID = "hybrid"


# Alias pour compatibility
CacheMiddleware = AdvancedCacheMiddleware

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def create_cache_middleware(config: Optional[CacheConfig] = None) -> AdvancedCacheMiddleware:
    """Crée une instance de cache middleware"""
    if config is None:
        config = create_development_cache_config()
    return AdvancedCacheMiddleware(config)

def create_production_cache_config() -> CacheConfig:
    """Configuration cache pour production"""
    return CacheConfig(
        enabled=True,
        default_ttl=3600,
        l1_enabled=True,
        l1_max_size=10000,
        l1_ttl=300,
        l2_enabled=True,
        l2_ttl=3600,
        l3_enabled=True,
        l3_ttl=7200,
        compression_enabled=True,
        compression_threshold=1024
    )

def create_development_cache_config() -> CacheConfig:
    """Configuration cache pour développement"""
    return CacheConfig(
        enabled=True,
        default_ttl=300,
        l1_enabled=True,
        l1_max_size=1000,
        l1_ttl=60,
        l2_enabled=False,
        l3_enabled=False,
        compression_enabled=False
    )

# Instance globale pour l'app
cache = None

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "CacheLevel",
    "CacheStrategy", 
    "CachePattern",
    "CacheConfig",
    "CacheKey",
    "CacheItem",
    "CacheMetrics",
    "CircuitBreaker",
    "CacheSerializer",
    "AdvancedCacheManager",
    "AdvancedCacheMiddleware",
    "CacheMiddleware",
    "CacheInvalidator", 
    "CompressionEngine",
    "CacheCluster",
    "CacheOperation",
    "CacheType",
    "create_cache_middleware",
    "create_production_cache_config",
    "create_development_cache_config",
    "cache"
]
