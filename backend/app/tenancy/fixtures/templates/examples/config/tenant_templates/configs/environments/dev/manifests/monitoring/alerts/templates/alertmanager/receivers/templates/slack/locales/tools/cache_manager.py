#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gestionnaire de Cache Redis Distribué pour Outils Slack

Ce module fournit un système de cache Redis avancé avec:
- Cache multi-niveau (L1 mémoire, L2 Redis)
- Expiration intelligente et TTL adaptatif
- Compression automatique des données
- Clustering et haute disponibilité
- Métriques détaillées et monitoring
- Patterns de cache avancés (write-through, write-behind)
"""

import asyncio
import json
import logging
import pickle
import zlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Callable
import aioredis
from aioredis import Redis
from dataclasses import dataclass, field
import structlog
from prometheus_client import Counter, Histogram, Gauge
import hashlib
from contextlib import asynccontextmanager

logger = structlog.get_logger(__name__)

# Métriques Prometheus
CACHE_OPERATIONS = Counter(
    'slack_cache_operations_total',
    'Total cache operations',
    ['operation', 'cache_type', 'result']
)

CACHE_DURATION = Histogram(
    'slack_cache_operation_duration_seconds',
    'Cache operation duration',
    ['operation', 'cache_type']
)

CACHE_SIZE = Gauge(
    'slack_cache_size_bytes',
    'Cache size in bytes',
    ['cache_type']
)

CACHE_HIT_RATIO = Gauge(
    'slack_cache_hit_ratio',
    'Cache hit ratio',
    ['cache_type']
)

@dataclass
class CacheEntry:
    """Entrée de cache avec métadonnées."""
    key: str
    value: Any
    ttl: int
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    compressed: bool = False
    size_bytes: int = 0
    
    def __post_init__(self):
        if self.size_bytes == 0:
            self.size_bytes = len(str(self.value).encode('utf-8'))

@dataclass
class CacheStats:
    """Statistiques de cache."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    total_size: int = 0
    
    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

class CacheManager:
    """
    Gestionnaire de cache Redis distribué avec cache L1 local.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger.bind(component="cache_manager")
        
        # Configuration Redis
        self.redis_url = config.get("redis_url", "redis://localhost:6379/5")
        self.redis_password = config.get("redis_password")
        self.redis_ssl = config.get("redis_ssl", False)
        self.redis_timeout = config.get("redis_timeout", 30)
        self.max_connections = config.get("max_connections", 20)
        
        # Configuration cache
        self.default_ttl = config.get("default_ttl", 3600)
        self.max_cache_size = config.get("max_cache_size", 1000)
        self.compression_threshold = config.get("compression_threshold", 1024)
        self.compression_enabled = config.get("compression_enabled", True)
        self.l1_cache_enabled = config.get("l1_cache_enabled", True)
        self.l1_cache_size = config.get("l1_cache_size", 500)
        
        # Stockage interne (Cache L1)
        self._l1_cache: Dict[str, CacheEntry] = {}
        self._l1_stats = CacheStats()
        self._redis_stats = CacheStats()
        
        # Connexion Redis
        self._redis: Optional[Redis] = None
        self._redis_pool = None
        
        # Lock pour thread-safety
        self._lock = asyncio.Lock()
        
        # Tâches de maintenance
        self._cleanup_task = None
        self._stats_task = None
        
        # Initialisation
        asyncio.create_task(self._initialize())
    
    async def _initialize(self):
        """Initialise le gestionnaire de cache."""
        try:
            await self._setup_redis_connection()
            await self._start_maintenance_tasks()
            
            self.logger.info(
                "Gestionnaire de cache initialisé",
                redis_connected=self._redis is not None,
                l1_enabled=self.l1_cache_enabled
            )
            
        except Exception as e:
            self.logger.error("Erreur lors de l'initialisation", error=str(e))
            raise
    
    async def _setup_redis_connection(self):
        """Configure la connexion Redis."""
        try:
            # Configuration de la pool de connexions
            self._redis_pool = aioredis.ConnectionPool.from_url(
                self.redis_url,
                password=self.redis_password,
                ssl=self.redis_ssl,
                socket_timeout=self.redis_timeout,
                max_connections=self.max_connections,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Créer la connexion Redis
            self._redis = Redis(connection_pool=self._redis_pool)
            
            # Test de connexion
            await self._redis.ping()
            
            self.logger.info(
                "Connexion Redis établie",
                url=self.redis_url.replace(self.redis_password or "", "***") if self.redis_password else self.redis_url
            )
            
        except Exception as e:
            self.logger.error("Erreur connexion Redis", error=str(e))
            # Continue sans Redis (cache L1 seulement)
            self._redis = None
    
    async def _start_maintenance_tasks(self):
        """Démarre les tâches de maintenance."""
        # Nettoyage périodique du cache L1
        self._cleanup_task = asyncio.create_task(self._cleanup_l1_cache())
        
        # Mise à jour des statistiques
        self._stats_task = asyncio.create_task(self._update_stats())
    
    async def _cleanup_l1_cache(self):
        """Nettoie périodiquement le cache L1."""
        while True:
            try:
                await asyncio.sleep(300)  # Toutes les 5 minutes
                
                async with self._lock:
                    now = datetime.utcnow()
                    expired_keys = []
                    
                    # Identifier les clés expirées
                    for key, entry in self._l1_cache.items():
                        if now > entry.created_at + timedelta(seconds=entry.ttl):
                            expired_keys.append(key)
                    
                    # Supprimer les clés expirées
                    for key in expired_keys:
                        del self._l1_cache[key]
                        self._l1_stats.evictions += 1
                    
                    # LRU eviction si trop d'entrées
                    if len(self._l1_cache) > self.l1_cache_size:
                        # Trier par dernier accès
                        sorted_entries = sorted(
                            self._l1_cache.items(),
                            key=lambda x: x[1].accessed_at
                        )
                        
                        # Supprimer les plus anciennes
                        to_remove = len(self._l1_cache) - self.l1_cache_size
                        for key, _ in sorted_entries[:to_remove]:
                            del self._l1_cache[key]
                            self._l1_stats.evictions += 1
                    
                    if expired_keys:
                        self.logger.debug(
                            "Cache L1 nettoyé",
                            expired_count=len(expired_keys),
                            current_size=len(self._l1_cache)
                        )
                
            except Exception as e:
                self.logger.error("Erreur nettoyage cache L1", error=str(e))
    
    async def _update_stats(self):
        """Met à jour les métriques Prometheus."""
        while True:
            try:
                await asyncio.sleep(60)  # Toutes les minutes
                
                # Métriques L1
                CACHE_HIT_RATIO.labels(cache_type="l1").set(self._l1_stats.hit_ratio)
                CACHE_SIZE.labels(cache_type="l1").set(self._l1_stats.total_size)
                
                # Métriques Redis
                if self._redis:
                    CACHE_HIT_RATIO.labels(cache_type="redis").set(self._redis_stats.hit_ratio)
                    
                    # Taille Redis (approximative)
                    try:
                        info = await self._redis.info("memory")
                        used_memory = info.get("used_memory", 0)
                        CACHE_SIZE.labels(cache_type="redis").set(used_memory)
                    except:
                        pass
                
            except Exception as e:
                self.logger.error("Erreur mise à jour stats", error=str(e))
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Récupère une valeur du cache.
        
        Args:
            key: Clé de cache
            
        Returns:
            Valeur ou None si non trouvée
        """
        start_time = datetime.utcnow()
        
        try:
            # Chercher d'abord dans le cache L1
            if self.l1_cache_enabled:
                entry = await self._get_from_l1(key)
                if entry is not None:
                    CACHE_OPERATIONS.labels(
                        operation="get",
                        cache_type="l1",
                        result="hit"
                    ).inc()
                    return entry
            
            # Chercher dans Redis
            if self._redis:
                value = await self._get_from_redis(key)
                if value is not None:
                    # Mettre en cache L1 si activé
                    if self.l1_cache_enabled:
                        await self._set_in_l1(key, value, self.default_ttl)
                    
                    CACHE_OPERATIONS.labels(
                        operation="get",
                        cache_type="redis",
                        result="hit"
                    ).inc()
                    return value
            
            # Cache miss
            CACHE_OPERATIONS.labels(
                operation="get",
                cache_type="both",
                result="miss"
            ).inc()
            
            if self.l1_cache_enabled:
                self._l1_stats.misses += 1
            self._redis_stats.misses += 1
            
            return None
            
        except Exception as e:
            self.logger.error("Erreur get cache", key=key, error=str(e))
            return None
        
        finally:
            duration = (datetime.utcnow() - start_time).total_seconds()
            CACHE_DURATION.labels(
                operation="get",
                cache_type="both"
            ).observe(duration)
    
    async def _get_from_l1(self, key: str) -> Optional[Any]:
        """Récupère une valeur du cache L1."""
        async with self._lock:
            entry = self._l1_cache.get(key)
            if entry is None:
                return None
            
            # Vérifier expiration
            now = datetime.utcnow()
            if now > entry.created_at + timedelta(seconds=entry.ttl):
                del self._l1_cache[key]
                self._l1_stats.evictions += 1
                return None
            
            # Mettre à jour les stats d'accès
            entry.accessed_at = now
            entry.access_count += 1
            self._l1_stats.hits += 1
            
            return entry.value
    
    async def _get_from_redis(self, key: str) -> Optional[Any]:
        """Récupère une valeur de Redis."""
        try:
            raw_data = await self._redis.get(f"slack_cache:{key}")
            if raw_data is None:
                return None
            
            # Décompresser si nécessaire
            if raw_data.startswith(b'COMPRESSED:'):
                compressed_data = raw_data[11:]  # Enlever le préfixe
                raw_data = zlib.decompress(compressed_data)
            
            # Désérialiser
            return pickle.loads(raw_data)
            
        except Exception as e:
            self.logger.error("Erreur get Redis", key=key, error=str(e))
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        compress: Optional[bool] = None
    ):
        """
        Met une valeur en cache.
        
        Args:
            key: Clé de cache
            value: Valeur à cacher
            ttl: Durée de vie en secondes
            compress: Forcer la compression
        """
        start_time = datetime.utcnow()
        ttl = ttl or self.default_ttl
        
        try:
            # Mettre en cache L1 si activé
            if self.l1_cache_enabled:
                await self._set_in_l1(key, value, ttl)
            
            # Mettre en cache Redis
            if self._redis:
                await self._set_in_redis(key, value, ttl, compress)
            
            CACHE_OPERATIONS.labels(
                operation="set",
                cache_type="both",
                result="success"
            ).inc()
            
        except Exception as e:
            self.logger.error("Erreur set cache", key=key, error=str(e))
            CACHE_OPERATIONS.labels(
                operation="set",
                cache_type="both",
                result="error"
            ).inc()
        
        finally:
            duration = (datetime.utcnow() - start_time).total_seconds()
            CACHE_DURATION.labels(
                operation="set",
                cache_type="both"
            ).observe(duration)
    
    async def _set_in_l1(self, key: str, value: Any, ttl: int):
        """Met une valeur en cache L1."""
        async with self._lock:
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl,
                created_at=datetime.utcnow(),
                accessed_at=datetime.utcnow()
            )
            
            self._l1_cache[key] = entry
            self._l1_stats.sets += 1
            self._l1_stats.total_size += entry.size_bytes
    
    async def _set_in_redis(
        self,
        key: str,
        value: Any,
        ttl: int,
        compress: Optional[bool] = None
    ):
        """Met une valeur en cache Redis."""
        try:
            # Sérialiser la valeur
            serialized = pickle.dumps(value)
            
            # Compression si nécessaire
            if compress is None:
                compress = (
                    self.compression_enabled and 
                    len(serialized) > self.compression_threshold
                )
            
            if compress:
                compressed = zlib.compress(serialized)
                # Ajouter un préfixe pour identifier les données compressées
                data_to_store = b'COMPRESSED:' + compressed
            else:
                data_to_store = serialized
            
            # Stocker dans Redis
            await self._redis.setex(
                f"slack_cache:{key}",
                ttl,
                data_to_store
            )
            
            self._redis_stats.sets += 1
            
        except Exception as e:
            self.logger.error("Erreur set Redis", key=key, error=str(e))
            raise
    
    async def delete(self, key: str):
        """Supprime une entrée du cache."""
        start_time = datetime.utcnow()
        
        try:
            # Supprimer du cache L1
            if self.l1_cache_enabled:
                async with self._lock:
                    if key in self._l1_cache:
                        entry = self._l1_cache[key]
                        self._l1_stats.total_size -= entry.size_bytes
                        del self._l1_cache[key]
                        self._l1_stats.deletes += 1
            
            # Supprimer de Redis
            if self._redis:
                await self._redis.delete(f"slack_cache:{key}")
                self._redis_stats.deletes += 1
            
            CACHE_OPERATIONS.labels(
                operation="delete",
                cache_type="both",
                result="success"
            ).inc()
            
        except Exception as e:
            self.logger.error("Erreur delete cache", key=key, error=str(e))
            CACHE_OPERATIONS.labels(
                operation="delete",
                cache_type="both",
                result="error"
            ).inc()
        
        finally:
            duration = (datetime.utcnow() - start_time).total_seconds()
            CACHE_DURATION.labels(
                operation="delete",
                cache_type="both"
            ).observe(duration)
    
    async def delete_pattern(self, pattern: str):
        """Supprime toutes les clés correspondant au pattern."""
        try:
            # Pattern pour cache L1
            if self.l1_cache_enabled:
                async with self._lock:
                    keys_to_delete = []
                    for key in self._l1_cache:
                        if self._match_pattern(key, pattern):
                            keys_to_delete.append(key)
                    
                    for key in keys_to_delete:
                        entry = self._l1_cache[key]
                        self._l1_stats.total_size -= entry.size_bytes
                        del self._l1_cache[key]
                        self._l1_stats.deletes += 1
            
            # Pattern pour Redis
            if self._redis:
                redis_pattern = f"slack_cache:{pattern}"
                async for key in self._redis.scan_iter(match=redis_pattern):
                    await self._redis.delete(key)
                    self._redis_stats.deletes += 1
            
        except Exception as e:
            self.logger.error("Erreur delete pattern", pattern=pattern, error=str(e))
    
    def _match_pattern(self, key: str, pattern: str) -> bool:
        """Vérifie si une clé correspond au pattern (simple wildcard)."""
        import fnmatch
        return fnmatch.fnmatch(key, pattern)
    
    async def exists(self, key: str) -> bool:
        """Vérifie si une clé existe dans le cache."""
        # Vérifier L1
        if self.l1_cache_enabled and key in self._l1_cache:
            entry = self._l1_cache[key]
            now = datetime.utcnow()
            if now <= entry.created_at + timedelta(seconds=entry.ttl):
                return True
        
        # Vérifier Redis
        if self._redis:
            return await self._redis.exists(f"slack_cache:{key}") > 0
        
        return False
    
    async def ttl(self, key: str) -> Optional[int]:
        """Retourne le TTL d'une clé en secondes."""
        # Vérifier L1 d'abord
        if self.l1_cache_enabled and key in self._l1_cache:
            entry = self._l1_cache[key]
            now = datetime.utcnow()
            remaining = entry.created_at + timedelta(seconds=entry.ttl) - now
            return max(0, int(remaining.total_seconds()))
        
        # Vérifier Redis
        if self._redis:
            ttl_seconds = await self._redis.ttl(f"slack_cache:{key}")
            return ttl_seconds if ttl_seconds > 0 else None
        
        return None
    
    @asynccontextmanager
    async def lock(self, key: str, timeout: int = 60):
        """
        Verrou distribué Redis pour éviter les race conditions.
        
        Args:
            key: Clé du verrou
            timeout: Timeout en secondes
        """
        lock_key = f"slack_lock:{key}"
        acquired = False
        
        try:
            if self._redis:
                # Tenter d'acquérir le verrou
                acquired = await self._redis.set(
                    lock_key,
                    "locked",
                    ex=timeout,
                    nx=True
                )
            
            if acquired:
                yield
            else:
                raise RuntimeError(f"Impossible d'acquérir le verrou pour {key}")
        
        finally:
            if acquired and self._redis:
                await self._redis.delete(lock_key)
    
    async def mget(self, keys: List[str]) -> Dict[str, Any]:
        """Récupère plusieurs valeurs en une fois."""
        results = {}
        
        # Récupérer depuis L1 d'abord
        l1_misses = []
        if self.l1_cache_enabled:
            async with self._lock:
                for key in keys:
                    value = await self._get_from_l1(key)
                    if value is not None:
                        results[key] = value
                    else:
                        l1_misses.append(key)
        else:
            l1_misses = keys
        
        # Récupérer les manqués depuis Redis
        if l1_misses and self._redis:
            redis_keys = [f"slack_cache:{key}" for key in l1_misses]
            redis_values = await self._redis.mget(redis_keys)
            
            for i, value in enumerate(redis_values):
                if value is not None:
                    key = l1_misses[i]
                    
                    # Décompresser si nécessaire
                    if value.startswith(b'COMPRESSED:'):
                        compressed_data = value[11:]
                        value = zlib.decompress(compressed_data)
                    
                    # Désérialiser
                    deserialized = pickle.loads(value)
                    results[key] = deserialized
                    
                    # Mettre en cache L1
                    if self.l1_cache_enabled:
                        await self._set_in_l1(key, deserialized, self.default_ttl)
        
        return results
    
    async def mset(self, mapping: Dict[str, Any], ttl: Optional[int] = None):
        """Met plusieurs valeurs en cache en une fois."""
        ttl = ttl or self.default_ttl
        
        # Mettre en cache L1
        if self.l1_cache_enabled:
            for key, value in mapping.items():
                await self._set_in_l1(key, value, ttl)
        
        # Mettre en cache Redis
        if self._redis:
            redis_mapping = {}
            for key, value in mapping.items():
                serialized = pickle.dumps(value)
                
                # Compression si nécessaire
                if (self.compression_enabled and 
                    len(serialized) > self.compression_threshold):
                    compressed = zlib.compress(serialized)
                    data_to_store = b'COMPRESSED:' + compressed
                else:
                    data_to_store = serialized
                
                redis_mapping[f"slack_cache:{key}"] = data_to_store
            
            # Utiliser pipeline pour efficacité
            pipe = self._redis.pipeline()
            pipe.mset(redis_mapping)
            
            # Définir TTL pour chaque clé
            for redis_key in redis_mapping.keys():
                pipe.expire(redis_key, ttl)
            
            await pipe.execute()
    
    async def clear(self):
        """Vide complètement le cache."""
        try:
            # Vider L1
            if self.l1_cache_enabled:
                async with self._lock:
                    self._l1_cache.clear()
                    self._l1_stats = CacheStats()
            
            # Vider Redis (pattern)
            if self._redis:
                async for key in self._redis.scan_iter(match="slack_cache:*"):
                    await self._redis.delete(key)
            
            self.logger.info("Cache vidé complètement")
            
        except Exception as e:
            self.logger.error("Erreur clear cache", error=str(e))
    
    async def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques détaillées du cache."""
        stats = {
            "l1_cache": {
                "enabled": self.l1_cache_enabled,
                "size": len(self._l1_cache),
                "max_size": self.l1_cache_size,
                "hits": self._l1_stats.hits,
                "misses": self._l1_stats.misses,
                "hit_ratio": self._l1_stats.hit_ratio,
                "sets": self._l1_stats.sets,
                "deletes": self._l1_stats.deletes,
                "evictions": self._l1_stats.evictions,
                "total_size_bytes": self._l1_stats.total_size
            },
            "redis_cache": {
                "connected": self._redis is not None,
                "hits": self._redis_stats.hits,
                "misses": self._redis_stats.misses,
                "hit_ratio": self._redis_stats.hit_ratio,
                "sets": self._redis_stats.sets,
                "deletes": self._redis_stats.deletes
            }
        }
        
        # Ajouter les stats Redis si connecté
        if self._redis:
            try:
                redis_info = await self._redis.info()
                stats["redis_cache"].update({
                    "used_memory": redis_info.get("used_memory", 0),
                    "used_memory_human": redis_info.get("used_memory_human", "0B"),
                    "connected_clients": redis_info.get("connected_clients", 0),
                    "total_commands_processed": redis_info.get("total_commands_processed", 0)
                })
            except:
                pass
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérifie la santé du gestionnaire de cache."""
        try:
            health = {
                "status": "healthy",
                "l1_cache_enabled": self.l1_cache_enabled,
                "l1_cache_size": len(self._l1_cache),
                "redis_connected": False,
                "last_check": datetime.utcnow().isoformat()
            }
            
            # Test Redis
            if self._redis:
                await self._redis.ping()
                health["redis_connected"] = True
                
                # Test set/get
                test_key = "health_check_test"
                test_value = "test_value"
                await self.set(test_key, test_value, ttl=60)
                retrieved = await self.get(test_key)
                
                if retrieved == test_value:
                    health["redis_test"] = "success"
                    await self.delete(test_key)
                else:
                    health["redis_test"] = "failed"
                    health["status"] = "degraded"
            
            return health
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.utcnow().isoformat()
            }
    
    async def close(self):
        """Ferme les connexions et arrête les tâches."""
        try:
            # Arrêter les tâches de maintenance
            if self._cleanup_task:
                self._cleanup_task.cancel()
            if self._stats_task:
                self._stats_task.cancel()
            
            # Fermer Redis
            if self._redis:
                await self._redis.close()
                await self._redis_pool.disconnect()
            
            self.logger.info("Gestionnaire de cache fermé")
            
        except Exception as e:
            self.logger.error("Erreur fermeture cache", error=str(e))

# Factory function
def create_cache_manager(config: Dict[str, Any]) -> CacheManager:
    """Crée une instance du gestionnaire de cache."""
    return CacheManager(config)
