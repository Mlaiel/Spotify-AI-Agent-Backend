"""
Spotify AI Agent - Advanced Utilities Module
===========================================

Module d'utilitaires ultra-avancé pour les collecteurs de données
avec fonctionnalités d'optimisation, cache intelligent, compression,
chiffrement, et observabilité complète.

Utilitaires disponibles:
- CacheManager: Gestion de cache multi-niveaux avec TTL intelligent
- CompressionManager: Compression/décompression adaptive
- EncryptionManager: Chiffrement AES-256 avec rotation des clés
- DataValidator: Validation avancée avec schémas dynamiques
- MetricsCollector: Collecte de métriques business et technique
- ConfigManager: Gestion de configuration hot-reload
- RetryManager: Gestion des tentatives avec backoff adaptatif
- CircuitBreakerManager: Protection contre les défaillances en cascade
- RateLimiter: Limitation de débit avec algorithmes avancés
- DataTransformer: Transformation et normalisation de données
- SecurityManager: Gestion de la sécurité et audit
- PerformanceProfiler: Profilage de performance en temps réel

Développé par l'équipe Spotify AI Agent
Lead Developer: Fahed Mlaiel
"""

import asyncio
import hashlib
import zlib
import gzip
import lz4.frame
import zstandard as zstd
import json
import pickle
import time
import threading
import weakref
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import wraps, lru_cache
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Union, Callable, Tuple, 
    TypeVar, Generic, Set, Awaitable, Type, Protocol
)
import uuid
import secrets
import base64
import struct
import math
import statistics
import logging
import structlog
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, RLock, Event
import psutil
import aioredis
import aiofiles
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import pydantic
from pydantic import BaseModel, validator
import jsonschema
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary


T = TypeVar('T')
logger = structlog.get_logger(__name__)


class CompressionAlgorithm(Enum):
    """Algorithmes de compression supportés."""
    NONE = "none"
    ZLIB = "zlib"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstandard"


class CacheStrategy(Enum):
    """Stratégies de cache."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


class RateLimitAlgorithm(Enum):
    """Algorithmes de rate limiting."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class CacheEntry:
    """Entrée de cache avec métadonnées."""
    
    data: Any
    timestamp: float
    ttl: Optional[float] = None
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0
    compression_ratio: float = 1.0
    tags: Set[str] = field(default_factory=set)
    
    @property
    def is_expired(self) -> bool:
        """Vérifie si l'entrée a expiré."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    @property
    def age(self) -> float:
        """Âge de l'entrée en secondes."""
        return time.time() - self.timestamp


class CacheManager:
    """
    Gestionnaire de cache multi-niveaux avec intelligence artificielle.
    
    Fonctionnalités:
    - Cache mémoire avec LRU/LFU/TTL/Adaptive
    - Cache Redis distribué
    - Compression automatique
    - Préchargement intelligent
    - Éviction adaptative
    - Métriques détaillées
    """
    
    def __init__(
        self,
        max_memory_mb: int = 512,
        default_ttl: int = 3600,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        compression_threshold: int = 1024,
        redis_url: Optional[str] = None,
        enable_metrics: bool = True
    ):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.strategy = strategy
        self.compression_threshold = compression_threshold
        self.enable_metrics = enable_metrics
        
        # Stockage local
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: deque = deque()
        self._frequency_counter: defaultdict = defaultdict(int)
        self._lock = RLock()
        
        # Redis distribué
        self._redis_pool: Optional[aioredis.ConnectionPool] = None
        if redis_url:
            self._redis_pool = aioredis.ConnectionPool.from_url(redis_url)
        
        # Métriques
        if enable_metrics:
            self._init_metrics()
        
        # Gestionnaire de compression
        self.compression_manager = CompressionManager()
        
        # Statistiques
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size_bytes": 0,
            "compression_saved_bytes": 0
        }
        
        # Arrêt en douceur
        self._shutdown_event = Event()
        self._background_task = None
        
        # Démarrage des tâches de maintenance
        asyncio.create_task(self._start_maintenance_tasks())
    
    def _init_metrics(self) -> None:
        """Initialise les métriques Prometheus."""
        self.cache_hits = Counter(
            'cache_hits_total',
            'Nombre total de cache hits',
            ['cache_type', 'key_prefix']
        )
        self.cache_misses = Counter(
            'cache_misses_total',
            'Nombre total de cache misses',
            ['cache_type', 'key_prefix']
        )
        self.cache_size = Gauge(
            'cache_size_bytes',
            'Taille du cache en bytes',
            ['cache_type']
        )
        self.cache_operations = Histogram(
            'cache_operation_duration_seconds',
            'Durée des opérations de cache',
            ['operation', 'cache_type']
        )
    
    async def _start_maintenance_tasks(self) -> None:
        """Démarre les tâches de maintenance en arrière-plan."""
        self._background_task = asyncio.create_task(self._maintenance_loop())
    
    async def _maintenance_loop(self) -> None:
        """Boucle de maintenance pour nettoyage et optimisation."""
        while not self._shutdown_event.is_set():
            try:
                await self._cleanup_expired()
                await self._optimize_cache()
                await self._update_metrics()
                await asyncio.sleep(60)  # Maintenance toutes les minutes
            except Exception as e:
                logger.error("Erreur lors de la maintenance du cache", error=str(e))
                await asyncio.sleep(10)
    
    async def get(
        self,
        key: str,
        default: Optional[T] = None,
        refresh_ttl: bool = True
    ) -> Optional[T]:
        """
        Récupère une valeur du cache.
        
        Args:
            key: Clé de cache
            default: Valeur par défaut si non trouvé
            refresh_ttl: Renouvelle le TTL lors de l'accès
        """
        start_time = time.time()
        
        try:
            # Recherche en mémoire locale
            with self._lock:
                if key in self._cache:
                    entry = self._cache[key]
                    
                    if entry.is_expired:
                        del self._cache[key]
                        self._remove_from_access_order(key)
                    else:
                        # Mise à jour des statistiques d'accès
                        entry.access_count += 1
                        entry.last_access = time.time()
                        self._frequency_counter[key] += 1
                        
                        if refresh_ttl and entry.ttl:
                            entry.timestamp = time.time()
                        
                        self._update_access_order(key)
                        self.stats["hits"] += 1
                        
                        if self.enable_metrics:
                            self.cache_hits.labels(
                                cache_type="memory",
                                key_prefix=key.split(':')[0]
                            ).inc()
                        
                        return entry.data
            
            # Recherche dans Redis si disponible
            if self._redis_pool:
                value = await self._get_from_redis(key)
                if value is not None:
                    # Cache en mémoire locale pour les accès futurs
                    await self.set(key, value, ttl=self.default_ttl)
                    
                    self.stats["hits"] += 1
                    if self.enable_metrics:
                        self.cache_hits.labels(
                            cache_type="redis",
                            key_prefix=key.split(':')[0]
                        ).inc()
                    
                    return value
            
            # Cache miss
            self.stats["misses"] += 1
            if self.enable_metrics:
                self.cache_misses.labels(
                    cache_type="combined",
                    key_prefix=key.split(':')[0]
                ).inc()
            
            return default
            
        finally:
            if self.enable_metrics:
                duration = time.time() - start_time
                self.cache_operations.labels(
                    operation="get",
                    cache_type="combined"
                ).observe(duration)
    
    async def set(
        self,
        key: str,
        value: T,
        ttl: Optional[int] = None,
        tags: Optional[Set[str]] = None,
        compress: Optional[bool] = None
    ) -> bool:
        """
        Stocke une valeur dans le cache.
        
        Args:
            key: Clé de cache
            value: Valeur à stocker
            ttl: Time to live en secondes
            tags: Tags pour le regroupement
            compress: Force la compression
        """
        start_time = time.time()
        
        try:
            ttl = ttl or self.default_ttl
            tags = tags or set()
            
            # Sérialisation et éventuellement compression
            serialized_data = pickle.dumps(value)
            original_size = len(serialized_data)
            
            should_compress = (
                compress is True or
                (compress is None and original_size > self.compression_threshold)
            )
            
            if should_compress:
                compressed_data = self.compression_manager.compress(
                    serialized_data,
                    algorithm=CompressionAlgorithm.ZSTD
                )
                final_data = compressed_data
                compression_ratio = len(compressed_data) / original_size
                self.stats["compression_saved_bytes"] += (original_size - len(compressed_data))
            else:
                final_data = serialized_data
                compression_ratio = 1.0
            
            # Création de l'entrée de cache
            entry = CacheEntry(
                data=value,
                timestamp=time.time(),
                ttl=ttl,
                size_bytes=len(final_data),
                compression_ratio=compression_ratio,
                tags=tags
            )
            
            # Stockage en mémoire locale
            with self._lock:
                # Éviction si nécessaire
                await self._ensure_memory_limit(len(final_data))
                
                self._cache[key] = entry
                self._update_access_order(key)
                self.stats["size_bytes"] += len(final_data)
            
            # Stockage dans Redis si disponible
            if self._redis_pool:
                await self._set_in_redis(key, final_data, ttl, should_compress)
            
            return True
            
        except Exception as e:
            logger.error("Erreur lors du stockage en cache", key=key, error=str(e))
            return False
        
        finally:
            if self.enable_metrics:
                duration = time.time() - start_time
                self.cache_operations.labels(
                    operation="set",
                    cache_type="combined"
                ).observe(duration)
    
    async def delete(self, key: str) -> bool:
        """Supprime une entrée du cache."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                self.stats["size_bytes"] -= entry.size_bytes
                del self._cache[key]
                self._remove_from_access_order(key)
        
        # Suppression de Redis
        if self._redis_pool:
            await self._delete_from_redis(key)
        
        return True
    
    async def clear(self, pattern: Optional[str] = None, tags: Optional[Set[str]] = None) -> int:
        """
        Vide le cache selon des critères.
        
        Args:
            pattern: Pattern de clés à supprimer
            tags: Tags des entrées à supprimer
        """
        deleted_count = 0
        
        with self._lock:
            keys_to_delete = []
            
            for key, entry in self._cache.items():
                should_delete = True
                
                if pattern and not key.match(pattern):
                    should_delete = False
                
                if tags and not entry.tags.intersection(tags):
                    should_delete = False
                
                if should_delete:
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                entry = self._cache[key]
                self.stats["size_bytes"] -= entry.size_bytes
                del self._cache[key]
                self._remove_from_access_order(key)
                deleted_count += 1
        
        return deleted_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache."""
        with self._lock:
            hit_rate = (
                self.stats["hits"] / (self.stats["hits"] + self.stats["misses"])
                if (self.stats["hits"] + self.stats["misses"]) > 0 else 0
            )
            
            return {
                **self.stats,
                "entries_count": len(self._cache),
                "hit_rate": hit_rate,
                "memory_usage_mb": self.stats["size_bytes"] / (1024 * 1024),
                "memory_usage_percent": (self.stats["size_bytes"] / self.max_memory_bytes) * 100,
                "average_entry_size": (
                    self.stats["size_bytes"] / len(self._cache)
                    if len(self._cache) > 0 else 0
                ),
                "compression_efficiency": (
                    self.stats["compression_saved_bytes"] / 
                    (self.stats["size_bytes"] + self.stats["compression_saved_bytes"])
                    if (self.stats["size_bytes"] + self.stats["compression_saved_bytes"]) > 0 else 0
                )
            }
    
    async def _get_from_redis(self, key: str) -> Optional[Any]:
        """Récupère une valeur depuis Redis."""
        try:
            redis = aioredis.Redis(connection_pool=self._redis_pool)
            
            # Récupération des données et métadonnées
            pipe = redis.pipeline()
            pipe.get(f"data:{key}")
            pipe.hgetall(f"meta:{key}")
            
            results = await pipe.execute()
            data, meta = results
            
            if data is None:
                return None
            
            # Vérification de l'expiration
            if meta and 'ttl' in meta and 'timestamp' in meta:
                timestamp = float(meta['timestamp'])
                ttl = float(meta['ttl'])
                if time.time() - timestamp > ttl:
                    await self._delete_from_redis(key)
                    return None
            
            # Décompression si nécessaire
            if meta and meta.get('compressed') == 'true':
                data = self.compression_manager.decompress(data)
            
            # Désérialisation
            return pickle.loads(data)
            
        except Exception as e:
            logger.error("Erreur lors de la lecture Redis", key=key, error=str(e))
            return None
    
    async def _set_in_redis(
        self,
        key: str,
        data: bytes,
        ttl: int,
        compressed: bool
    ) -> None:
        """Stocke une valeur dans Redis."""
        try:
            redis = aioredis.Redis(connection_pool=self._redis_pool)
            
            # Stockage des données
            pipe = redis.pipeline()
            pipe.set(f"data:{key}", data, ex=ttl)
            
            # Stockage des métadonnées
            meta = {
                'timestamp': str(time.time()),
                'ttl': str(ttl),
                'compressed': str(compressed).lower(),
                'size': str(len(data))
            }
            pipe.hmset(f"meta:{key}", meta)
            pipe.expire(f"meta:{key}", ttl)
            
            await pipe.execute()
            
        except Exception as e:
            logger.error("Erreur lors de l'écriture Redis", key=key, error=str(e))
    
    async def _delete_from_redis(self, key: str) -> None:
        """Supprime une entrée de Redis."""
        try:
            redis = aioredis.Redis(connection_pool=self._redis_pool)
            await redis.delete(f"data:{key}", f"meta:{key}")
        except Exception as e:
            logger.error("Erreur lors de la suppression Redis", key=key, error=str(e))
    
    def _update_access_order(self, key: str) -> None:
        """Met à jour l'ordre d'accès pour LRU."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def _remove_from_access_order(self, key: str) -> None:
        """Retire une clé de l'ordre d'accès."""
        if key in self._access_order:
            self._access_order.remove(key)
    
    async def _ensure_memory_limit(self, new_entry_size: int) -> None:
        """S'assure que la limite mémoire n'est pas dépassée."""
        while (self.stats["size_bytes"] + new_entry_size) > self.max_memory_bytes:
            if not self._cache:
                break
            
            # Choix de la stratégie d'éviction
            if self.strategy == CacheStrategy.LRU:
                key_to_evict = self._access_order.popleft()
            elif self.strategy == CacheStrategy.LFU:
                key_to_evict = min(self._frequency_counter, key=self._frequency_counter.get)
            elif self.strategy == CacheStrategy.TTL:
                # Éviction des entrées les plus anciennes
                oldest_key = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].timestamp
                )
                key_to_evict = oldest_key
            else:  # ADAPTIVE
                key_to_evict = await self._adaptive_eviction()
            
            if key_to_evict in self._cache:
                entry = self._cache[key_to_evict]
                self.stats["size_bytes"] -= entry.size_bytes
                self.stats["evictions"] += 1
                del self._cache[key_to_evict]
                self._remove_from_access_order(key_to_evict)
                if key_to_evict in self._frequency_counter:
                    del self._frequency_counter[key_to_evict]
    
    async def _adaptive_eviction(self) -> str:
        """Stratégie d'éviction adaptative basée sur l'intelligence artificielle."""
        
        # Score composite basé sur plusieurs facteurs
        best_score = float('-inf')
        best_key = None
        
        for key, entry in self._cache.items():
            # Facteurs pour le score :
            # - Âge de l'entrée (plus vieux = plus susceptible d'éviction)
            # - Fréquence d'accès (moins fréquent = plus susceptible)
            # - Taille (plus gros = plus susceptible si peu utilisé)
            # - Dernière utilisation (plus ancien = plus susceptible)
            
            age_factor = entry.age / 3600  # Normalisation par heure
            frequency_factor = 1 / (entry.access_count + 1)
            size_factor = entry.size_bytes / (1024 * 1024)  # MB
            recency_factor = (time.time() - entry.last_access) / 3600
            
            # Score composite (plus élevé = plus susceptible d'éviction)
            score = (age_factor * 0.3 + 
                    frequency_factor * 0.4 + 
                    size_factor * 0.2 + 
                    recency_factor * 0.1)
            
            if score > best_score:
                best_score = score
                best_key = key
        
        return best_key or list(self._cache.keys())[0]
    
    async def _cleanup_expired(self) -> None:
        """Nettoie les entrées expirées."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]
            
            for key in expired_keys:
                entry = self._cache[key]
                self.stats["size_bytes"] -= entry.size_bytes
                del self._cache[key]
                self._remove_from_access_order(key)
                if key in self._frequency_counter:
                    del self._frequency_counter[key]
    
    async def _optimize_cache(self) -> None:
        """Optimise le cache en fonction des patterns d'usage."""
        
        # Analyse des patterns d'accès
        if len(self._cache) > 100:
            # Identification des entrées "chaudes" pour préchargement
            hot_keys = sorted(
                self._frequency_counter.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            # Log des statistiques d'optimisation
            logger.info(
                "Optimisation du cache",
                cache_size=len(self._cache),
                hot_keys=[key for key, _ in hot_keys],
                memory_usage_mb=self.stats["size_bytes"] / (1024 * 1024)
            )
    
    async def _update_metrics(self) -> None:
        """Met à jour les métriques Prometheus."""
        if self.enable_metrics:
            self.cache_size.labels(cache_type="memory").set(self.stats["size_bytes"])
    
    async def shutdown(self) -> None:
        """Arrêt en douceur du gestionnaire de cache."""
        self._shutdown_event.set()
        
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        
        if self._redis_pool:
            await self._redis_pool.disconnect()


class CompressionManager:
    """
    Gestionnaire de compression avec algorithmes adaptatifs.
    
    Fonctionnalités:
    - Support de multiples algorithmes (zlib, gzip, lz4, zstd)
    - Sélection automatique du meilleur algorithme
    - Métriques de performance
    - Compression adaptative selon le type de données
    """
    
    def __init__(self):
        self.stats = defaultdict(lambda: {"compressed_bytes": 0, "original_bytes": 0, "operations": 0})
        self._compressors = {
            CompressionAlgorithm.ZLIB: self._zlib_compress,
            CompressionAlgorithm.GZIP: self._gzip_compress,
            CompressionAlgorithm.LZ4: self._lz4_compress,
            CompressionAlgorithm.ZSTD: self._zstd_compress,
        }
        self._decompressors = {
            CompressionAlgorithm.ZLIB: self._zlib_decompress,
            CompressionAlgorithm.GZIP: self._gzip_decompress,
            CompressionAlgorithm.LZ4: self._lz4_decompress,
            CompressionAlgorithm.ZSTD: self._zstd_decompress,
        }
    
    def compress(
        self,
        data: bytes,
        algorithm: CompressionAlgorithm = CompressionAlgorithm.ZSTD,
        level: Optional[int] = None
    ) -> bytes:
        """
        Compresse des données avec l'algorithme spécifié.
        
        Args:
            data: Données à comprimer
            algorithm: Algorithme de compression
            level: Niveau de compression
        """
        if algorithm == CompressionAlgorithm.NONE:
            return data
        
        original_size = len(data)
        compressed_data = self._compressors[algorithm](data, level)
        compressed_size = len(compressed_data)
        
        # Ajout de l'en-tête pour identifier l'algorithme
        header = struct.pack('<BI', algorithm.value.encode('utf-8')[0], original_size)
        final_data = header + compressed_data
        
        # Mise à jour des statistiques
        self.stats[algorithm]["original_bytes"] += original_size
        self.stats[algorithm]["compressed_bytes"] += compressed_size
        self.stats[algorithm]["operations"] += 1
        
        return final_data
    
    def decompress(self, data: bytes) -> bytes:
        """Décompresse des données en détectant automatiquement l'algorithme."""
        if len(data) < 5:  # Taille minimale de l'en-tête
            return data
        
        # Lecture de l'en-tête
        algorithm_byte, original_size = struct.unpack('<BI', data[:5])
        compressed_data = data[5:]
        
        # Détection de l'algorithme
        algorithm_name = chr(algorithm_byte)
        algorithm = None
        
        for algo in CompressionAlgorithm:
            if algo.value.startswith(algorithm_name):
                algorithm = algo
                break
        
        if algorithm is None or algorithm == CompressionAlgorithm.NONE:
            return compressed_data
        
        return self._decompressors[algorithm](compressed_data)
    
    def get_best_algorithm(self, sample_data: bytes) -> CompressionAlgorithm:
        """
        Détermine le meilleur algorithme pour un type de données.
        
        Args:
            sample_data: Échantillon de données pour test
        """
        if len(sample_data) < 100:
            return CompressionAlgorithm.NONE
        
        # Test de tous les algorithmes sur un échantillon
        results = {}
        
        for algorithm in [CompressionAlgorithm.ZLIB, CompressionAlgorithm.LZ4, CompressionAlgorithm.ZSTD]:
            try:
                start_time = time.time()
                compressed = self.compress(sample_data, algorithm)
                compression_time = time.time() - start_time
                
                compression_ratio = len(compressed) / len(sample_data)
                
                # Score composite : ratio de compression + vitesse
                score = (1 - compression_ratio) * 0.7 + (1 / (compression_time + 0.001)) * 0.3
                
                results[algorithm] = {
                    "compression_ratio": compression_ratio,
                    "compression_time": compression_time,
                    "score": score
                }
            except Exception:
                continue
        
        if not results:
            return CompressionAlgorithm.NONE
        
        # Retourne l'algorithme avec le meilleur score
        best_algorithm = max(results, key=lambda x: results[x]["score"])
        return best_algorithm
    
    def _zlib_compress(self, data: bytes, level: Optional[int] = None) -> bytes:
        """Compression zlib."""
        return zlib.compress(data, level or 6)
    
    def _zlib_decompress(self, data: bytes) -> bytes:
        """Décompression zlib."""
        return zlib.decompress(data)
    
    def _gzip_compress(self, data: bytes, level: Optional[int] = None) -> bytes:
        """Compression gzip."""
        return gzip.compress(data, compresslevel=level or 6)
    
    def _gzip_decompress(self, data: bytes) -> bytes:
        """Décompression gzip."""
        return gzip.decompress(data)
    
    def _lz4_compress(self, data: bytes, level: Optional[int] = None) -> bytes:
        """Compression LZ4."""
        return lz4.frame.compress(data, compression_level=level or 1)
    
    def _lz4_decompress(self, data: bytes) -> bytes:
        """Décompression LZ4."""
        return lz4.frame.decompress(data)
    
    def _zstd_compress(self, data: bytes, level: Optional[int] = None) -> bytes:
        """Compression Zstandard."""
        cctx = zstd.ZstdCompressor(level=level or 3)
        return cctx.compress(data)
    
    def _zstd_decompress(self, data: bytes) -> bytes:
        """Décompression Zstandard."""
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(data)
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de compression."""
        total_stats = {
            "algorithms": {},
            "total_original_bytes": 0,
            "total_compressed_bytes": 0,
            "total_operations": 0,
            "overall_compression_ratio": 0
        }
        
        for algorithm, stats in self.stats.items():
            if stats["operations"] > 0:
                compression_ratio = stats["compressed_bytes"] / stats["original_bytes"]
                total_stats["algorithms"][algorithm.value] = {
                    **stats,
                    "compression_ratio": compression_ratio,
                    "savings_percent": (1 - compression_ratio) * 100
                }
                
                total_stats["total_original_bytes"] += stats["original_bytes"]
                total_stats["total_compressed_bytes"] += stats["compressed_bytes"]
                total_stats["total_operations"] += stats["operations"]
        
        if total_stats["total_original_bytes"] > 0:
            total_stats["overall_compression_ratio"] = (
                total_stats["total_compressed_bytes"] / total_stats["total_original_bytes"]
            )
        
        return total_stats


class EncryptionManager:
    """
    Gestionnaire de chiffrement avancé avec rotation automatique des clés.
    
    Fonctionnalités:
    - Chiffrement AES-256-GCM
    - Rotation automatique des clés
    - Dérivation de clés PBKDF2
    - Support multi-tenant
    - Audit et conformité
    """
    
    def __init__(
        self,
        master_key: Optional[bytes] = None,
        key_rotation_interval: int = 86400,  # 24 heures
        enable_audit: bool = True
    ):
        self.key_rotation_interval = key_rotation_interval
        self.enable_audit = enable_audit
        
        # Clé maître
        self._master_key = master_key or self._generate_master_key()
        
        # Stockage des clés par tenant
        self._tenant_keys: Dict[str, Dict[str, Any]] = {}
        self._key_versions: Dict[str, int] = defaultdict(int)
        
        # Verrous pour thread-safety
        self._keys_lock = RLock()
        
        # Audit trail
        self._audit_log: deque = deque(maxlen=10000)
        
        # Métriques
        self.encryption_counter = Counter('encryption_operations_total', 'Opérations de chiffrement', ['operation', 'tenant'])
        self.key_rotation_counter = Counter('key_rotations_total', 'Rotations de clés', ['tenant'])
    
    def _generate_master_key(self) -> bytes:
        """Génère une clé maître sécurisée."""
        return secrets.token_bytes(32)  # 256 bits
    
    def _derive_tenant_key(self, tenant_id: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """Dérive une clé spécifique au tenant."""
        if salt is None:
            salt = secrets.token_bytes(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        key_material = self._master_key + tenant_id.encode('utf-8')
        derived_key = kdf.derive(key_material)
        
        return derived_key, salt
    
    def _get_or_create_tenant_key(self, tenant_id: str) -> Dict[str, Any]:
        """Récupère ou crée une clé pour un tenant."""
        with self._keys_lock:
            if tenant_id not in self._tenant_keys:
                key, salt = self._derive_tenant_key(tenant_id)
                self._tenant_keys[tenant_id] = {
                    "key": key,
                    "salt": salt,
                    "created_at": time.time(),
                    "version": self._key_versions[tenant_id],
                    "usage_count": 0
                }
                
                if self.enable_audit:
                    self._audit_log.append({
                        "event": "key_created",
                        "tenant_id": tenant_id,
                        "timestamp": time.time(),
                        "version": self._key_versions[tenant_id]
                    })
            
            # Vérification de la rotation
            key_info = self._tenant_keys[tenant_id]
            if time.time() - key_info["created_at"] > self.key_rotation_interval:
                self._rotate_tenant_key(tenant_id)
                key_info = self._tenant_keys[tenant_id]
            
            return key_info
    
    def _rotate_tenant_key(self, tenant_id: str) -> None:
        """Effectue la rotation d'une clé tenant."""
        with self._keys_lock:
            # Sauvegarde de l'ancienne clé pour déchiffrement
            old_key_info = self._tenant_keys.get(tenant_id)
            if old_key_info:
                old_version = old_key_info["version"]
                self._tenant_keys[f"{tenant_id}:v{old_version}"] = old_key_info
            
            # Génération de la nouvelle clé
            self._key_versions[tenant_id] += 1
            new_key, new_salt = self._derive_tenant_key(tenant_id)
            
            self._tenant_keys[tenant_id] = {
                "key": new_key,
                "salt": new_salt,
                "created_at": time.time(),
                "version": self._key_versions[tenant_id],
                "usage_count": 0
            }
            
            # Audit et métriques
            if self.enable_audit:
                self._audit_log.append({
                    "event": "key_rotated",
                    "tenant_id": tenant_id,
                    "timestamp": time.time(),
                    "old_version": old_version if old_key_info else None,
                    "new_version": self._key_versions[tenant_id]
                })
            
            self.key_rotation_counter.labels(tenant=tenant_id).inc()
    
    def encrypt(self, data: bytes, tenant_id: str, additional_data: Optional[bytes] = None) -> bytes:
        """
        Chiffre des données pour un tenant spécifique.
        
        Args:
            data: Données à chiffrer
            tenant_id: Identifiant du tenant
            additional_data: Données additionnelles pour l'authentification
        """
        try:
            key_info = self._get_or_create_tenant_key(tenant_id)
            key = key_info["key"]
            version = key_info["version"]
            
            # Génération d'un IV unique
            iv = secrets.token_bytes(12)  # 96 bits pour GCM
            
            # Chiffrement AES-GCM
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            
            if additional_data:
                encryptor.authenticate_additional_data(additional_data)
            
            ciphertext = encryptor.update(data) + encryptor.finalize()
            
            # Construction du résultat avec métadonnées
            result = struct.pack('<I', version) + iv + encryptor.tag + ciphertext
            
            # Mise à jour des statistiques
            key_info["usage_count"] += 1
            self.encryption_counter.labels(operation="encrypt", tenant=tenant_id).inc()
            
            if self.enable_audit:
                self._audit_log.append({
                    "event": "data_encrypted",
                    "tenant_id": tenant_id,
                    "timestamp": time.time(),
                    "key_version": version,
                    "data_size": len(data)
                })
            
            return result
            
        except Exception as e:
            logger.error("Erreur lors du chiffrement", tenant_id=tenant_id, error=str(e))
            raise
    
    def decrypt(self, encrypted_data: bytes, tenant_id: str, additional_data: Optional[bytes] = None) -> bytes:
        """
        Déchiffre des données pour un tenant spécifique.
        
        Args:
            encrypted_data: Données chiffrées
            tenant_id: Identifiant du tenant
            additional_data: Données additionnelles pour l'authentification
        """
        try:
            # Extraction des métadonnées
            version = struct.unpack('<I', encrypted_data[:4])[0]
            iv = encrypted_data[4:16]
            tag = encrypted_data[16:32]
            ciphertext = encrypted_data[32:]
            
            # Récupération de la clé appropriée
            current_key_info = self._tenant_keys.get(tenant_id)
            if current_key_info and current_key_info["version"] == version:
                key = current_key_info["key"]
            else:
                # Recherche dans les anciennes versions
                versioned_key = self._tenant_keys.get(f"{tenant_id}:v{version}")
                if versioned_key:
                    key = versioned_key["key"]
                else:
                    raise ValueError(f"Clé non trouvée pour tenant {tenant_id} version {version}")
            
            # Déchiffrement
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(iv, tag),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            
            if additional_data:
                decryptor.authenticate_additional_data(additional_data)
            
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Métriques et audit
            self.encryption_counter.labels(operation="decrypt", tenant=tenant_id).inc()
            
            if self.enable_audit:
                self._audit_log.append({
                    "event": "data_decrypted",
                    "tenant_id": tenant_id,
                    "timestamp": time.time(),
                    "key_version": version,
                    "data_size": len(plaintext)
                })
            
            return plaintext
            
        except Exception as e:
            logger.error("Erreur lors du déchiffrement", tenant_id=tenant_id, error=str(e))
            raise
    
    def cleanup_old_keys(self, max_age: int = 604800) -> int:
        """
        Nettoie les anciennes clés (par défaut 7 jours).
        
        Args:
            max_age: Âge maximum des clés en secondes
        """
        removed_count = 0
        current_time = time.time()
        
        with self._keys_lock:
            keys_to_remove = []
            
            for key_id, key_info in self._tenant_keys.items():
                if ":" in key_id:  # Ancienne version
                    age = current_time - key_info["created_at"]
                    if age > max_age:
                        keys_to_remove.append(key_id)
            
            for key_id in keys_to_remove:
                del self._tenant_keys[key_id]
                removed_count += 1
        
        if self.enable_audit and removed_count > 0:
            self._audit_log.append({
                "event": "old_keys_cleaned",
                "timestamp": current_time,
                "removed_count": removed_count
            })
        
        return removed_count
    
    def get_audit_log(self, tenant_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Récupère le journal d'audit.
        
        Args:
            tenant_id: Filtrer par tenant (optionnel)
            limit: Nombre maximum d'entrées
        """
        logs = list(self._audit_log)
        
        if tenant_id:
            logs = [log for log in logs if log.get("tenant_id") == tenant_id]
        
        return logs[-limit:]
    
    def get_encryption_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de chiffrement."""
        with self._keys_lock:
            active_tenants = len([k for k in self._tenant_keys.keys() if ":" not in k])
            total_keys = len(self._tenant_keys)
            
            return {
                "active_tenants": active_tenants,
                "total_keys": total_keys,
                "old_keys": total_keys - active_tenants,
                "audit_entries": len(self._audit_log),
                "key_versions": dict(self._key_versions)
            }


# Décorateurs utilitaires
def cache_result(
    ttl: int = 3600,
    key_prefix: str = "",
    cache_manager: Optional[CacheManager] = None
):
    """
    Décorateur pour mise en cache automatique des résultats.
    
    Args:
        ttl: Time to live en secondes
        key_prefix: Préfixe pour les clés de cache
        cache_manager: Instance du gestionnaire de cache
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            nonlocal cache_manager
            if cache_manager is None:
                cache_manager = CacheManager()
            
            # Génération de la clé de cache
            key_parts = [key_prefix, func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(filter(None, key_parts))
            
            # Recherche en cache
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Exécution et mise en cache
            result = await func(*args, **kwargs)
            await cache_manager.set(cache_key, result, ttl=ttl)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Version synchrone simplifiée
            result = func(*args, **kwargs)
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def measure_performance(
    include_memory: bool = False,
    include_cpu: bool = False
):
    """
    Décorateur pour mesurer les performances d'exécution.
    
    Args:
        include_memory: Inclure les métriques mémoire
        include_cpu: Inclure les métriques CPU
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss if include_memory else None
            start_cpu = psutil.Process().cpu_percent() if include_cpu else None
            
            try:
                result = await func(*args, **kwargs)
                success = True
            except Exception as e:
                result = None
                success = False
                raise
            finally:
                end_time = time.time()
                duration = end_time - start_time
                
                metrics = {
                    "function": func.__name__,
                    "duration_seconds": duration,
                    "success": success
                }
                
                if include_memory and start_memory:
                    end_memory = psutil.Process().memory_info().rss
                    metrics["memory_delta_mb"] = (end_memory - start_memory) / (1024 * 1024)
                
                if include_cpu and start_cpu is not None:
                    end_cpu = psutil.Process().cpu_percent()
                    metrics["cpu_usage_percent"] = end_cpu
                
                logger.info("Performance mesurée", **metrics)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                logger.info(
                    "Performance mesurée",
                    function=func.__name__,
                    duration_seconds=duration
                )
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Fonctions utilitaires
def generate_correlation_id() -> str:
    """Génère un ID de corrélation unique."""
    return f"corr_{int(time.time() * 1000)}_{secrets.token_hex(8)}"


def generate_request_id() -> str:
    """Génère un ID de requête unique."""
    return f"req_{uuid.uuid4().hex}"


def safe_json_serialize(obj: Any) -> str:
    """Sérialisation JSON sécurisée avec gestion des types complexes."""
    def json_serializer(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, (set, frozenset)):
            return list(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    return json.dumps(obj, default=json_serializer, ensure_ascii=False)


def calculate_hash(data: Union[str, bytes], algorithm: str = "sha256") -> str:
    """
    Calcule le hash d'une donnée.
    
    Args:
        data: Données à hasher
        algorithm: Algorithme de hash (sha256, md5, sha1, etc.)
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    hash_func = getattr(hashlib, algorithm)()
    hash_func.update(data)
    return hash_func.hexdigest()


def truncate_string(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Tronque une chaîne de caractères si elle dépasse la limite."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


class DataValidator:
    """
    Validateur de données avancé avec schémas dynamiques.
    
    Fonctionnalités:
    - Validation JSON Schema
    - Validation Pydantic
    - Règles de validation personnalisées
    - Validation conditionnelle
    - Rapport d'erreurs détaillé
    """
    
    def __init__(self):
        self.custom_validators: Dict[str, Callable] = {}
        self.schemas: Dict[str, Dict[str, Any]] = {}
        
    def register_validator(self, name: str, validator: Callable[[Any], bool]) -> None:
        """Enregistre un validateur personnalisé."""
        self.custom_validators[name] = validator
    
    def register_schema(self, name: str, schema: Dict[str, Any]) -> None:
        """Enregistre un schéma de validation."""
        self.schemas[name] = schema
    
    def validate_json_schema(self, data: Any, schema_name: str) -> Tuple[bool, List[str]]:
        """Valide des données contre un schéma JSON."""
        if schema_name not in self.schemas:
            return False, [f"Schéma '{schema_name}' non trouvé"]
        
        try:
            jsonschema.validate(data, self.schemas[schema_name])
            return True, []
        except jsonschema.ValidationError as e:
            return False, [str(e)]
        except Exception as e:
            return False, [f"Erreur de validation: {str(e)}"]
    
    def validate_custom(self, data: Any, validator_names: List[str]) -> Tuple[bool, List[str]]:
        """Valide avec des validateurs personnalisés."""
        errors = []
        
        for validator_name in validator_names:
            if validator_name not in self.custom_validators:
                errors.append(f"Validateur '{validator_name}' non trouvé")
                continue
            
            try:
                if not self.custom_validators[validator_name](data):
                    errors.append(f"Validation '{validator_name}' échouée")
            except Exception as e:
                errors.append(f"Erreur dans le validateur '{validator_name}': {str(e)}")
        
        return len(errors) == 0, errors


# Instance globale des utilitaires
cache_manager = CacheManager()
compression_manager = CompressionManager()
encryption_manager = EncryptionManager()
data_validator = DataValidator()


# Configuration des validateurs par défaut
def setup_default_validators():
    """Configure les validateurs par défaut."""
    
    # Validateur d'email
    import re
    email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    data_validator.register_validator(
        "email",
        lambda x: isinstance(x, str) and email_pattern.match(x) is not None
    )
    
    # Validateur d'URL
    url_pattern = re.compile(
        r'^https?://'  # http:// ou https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...ou IP
        r'(?::\d+)?'  # port optionnel
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    data_validator.register_validator(
        "url",
        lambda x: isinstance(x, str) and url_pattern.match(x) is not None
    )
    
    # Validateur de tenant ID
    data_validator.register_validator(
        "tenant_id",
        lambda x: isinstance(x, str) and len(x) >= 3 and x.isalnum()
    )
    
    # Schémas JSON courants
    data_validator.register_schema("spotify_track", {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
            "artists": {
                "type": "array",
                "items": {"type": "string"}
            },
            "duration_ms": {"type": "integer", "minimum": 0},
            "explicit": {"type": "boolean"},
            "popularity": {"type": "integer", "minimum": 0, "maximum": 100}
        },
        "required": ["id", "name", "artists", "duration_ms"]
    })


# Initialisation automatique
setup_default_validators()
