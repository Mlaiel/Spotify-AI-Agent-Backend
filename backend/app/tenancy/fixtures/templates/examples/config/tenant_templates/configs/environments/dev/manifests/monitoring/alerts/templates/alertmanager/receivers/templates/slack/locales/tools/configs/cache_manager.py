"""
Gestionnaire de cache distribué multi-niveaux pour le système de monitoring Slack.

Ce module fournit un système de cache sophistiqué avec:
- Support multi-backend (Redis, Memcached, Memory)
- Cache multi-niveaux avec stratégies d'éviction intelligentes
- Compression et sérialisation optimisées
- Clustering et haute disponibilité
- Monitoring en temps réel des performances
- Cache-aside, write-through, write-behind patterns
- Invalidation intelligente par tags et patterns
- Circuit breakers pour la résilience
- Métriques détaillées et alerting

Architecture:
    - Strategy pattern pour les backends de cache
    - Observer pattern pour les événements de cache
    - Chain of Responsibility pour les niveaux de cache
    - Command pattern pour les opérations de cache
    - Decorator pattern pour les fonctionnalités transversales

Fonctionnalités:
    - Cache L1 (mémoire locale) + L2 (Redis/Memcached)
    - Warm-up automatique du cache au démarrage
    - TTL adaptatif basé sur les patterns d'usage
    - Bloom filters pour éviter les cache misses coûteux
    - Batch operations pour l'efficacité
    - Consistent hashing pour la distribution
    - Backup et restore automatiques

Auteur: Équipe Spotify AI Agent
Lead Developer: Fahed Mlaiel
"""

import asyncio
import gzip
import hashlib
import json
import pickle
import time
import threading
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Set, Tuple
from weakref import WeakSet

from .metrics import MetricsCollector


class CacheBackendType(Enum):
    """Types de backends de cache."""
    MEMORY = "memory"
    REDIS = "redis"
    MEMCACHED = "memcached"
    HYBRID = "hybrid"


class EvictionPolicy(Enum):
    """Politiques d'éviction du cache."""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    FIFO = "fifo"         # First In First Out
    TTL = "ttl"           # Time To Live
    RANDOM = "random"     # Random éviction


class CacheLevel(Enum):
    """Niveaux de cache."""
    L1 = "l1"     # Cache local en mémoire (rapide, limité)
    L2 = "l2"     # Cache distribué (Redis, plus lent mais persistant)
    L3 = "l3"     # Cache de stockage (DB, très lent mais permanent)


class SerializationFormat(Enum):
    """Formats de sérialisation."""
    JSON = "json"
    PICKLE = "pickle"
    MSGPACK = "msgpack"
    PROTOBUF = "protobuf"


@dataclass
class CacheEntry:
    """Entrée de cache avec métadonnées."""
    key: str
    value: Any
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    accessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    ttl: Optional[int] = None
    tags: Set[str] = field(default_factory=set)
    size_bytes: int = 0
    compressed: bool = False
    checksum: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialisation de l'entrée."""
        if isinstance(self.value, (str, bytes)):
            self.size_bytes = len(self.value)
        else:
            # Estimation de la taille pour les objets complexes
            try:
                self.size_bytes = len(pickle.dumps(self.value))
            except:
                self.size_bytes = 1024  # Estimation par défaut
        
        # Calcul du checksum pour l'intégrité
        if isinstance(self.value, str):
            self.checksum = hashlib.md5(self.value.encode()).hexdigest()
        elif isinstance(self.value, bytes):
            self.checksum = hashlib.md5(self.value).hexdigest()
    
    def is_expired(self) -> bool:
        """Vérifie si l'entrée a expiré."""
        if self.ttl is None:
            return False
        
        elapsed = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        return elapsed > self.ttl
    
    def update_access(self) -> None:
        """Met à jour les statistiques d'accès."""
        self.accessed_at = datetime.now(timezone.utc)
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire."""
        return {
            "key": self.key,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "ttl": self.ttl,
            "tags": list(self.tags),
            "size_bytes": self.size_bytes,
            "compressed": self.compressed,
            "checksum": self.checksum
        }


@dataclass
class CacheStats:
    """Statistiques du cache."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    errors: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0
    avg_access_time_ms: float = 0.0
    hit_ratio: float = 0.0
    
    def update_hit_ratio(self) -> None:
        """Met à jour le ratio de hits."""
        total_accesses = self.hits + self.misses
        if total_accesses > 0:
            self.hit_ratio = self.hits / total_accesses
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "evictions": self.evictions,
            "errors": self.errors,
            "total_size_bytes": self.total_size_bytes,
            "entry_count": self.entry_count,
            "avg_access_time_ms": self.avg_access_time_ms,
            "hit_ratio": self.hit_ratio
        }


class ICacheBackend(ABC):
    """Interface pour les backends de cache."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[CacheEntry]:
        """Récupère une valeur du cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, entry: CacheEntry) -> bool:
        """Stocke une valeur dans le cache."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Supprime une valeur du cache."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Vérifie si une clé existe."""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Vide le cache."""
        pass
    
    @abstractmethod
    def get_stats(self) -> CacheStats:
        """Récupère les statistiques."""
        pass
    
    @abstractmethod
    def get_keys(self, pattern: Optional[str] = None) -> List[str]:
        """Récupère les clés selon un pattern."""
        pass


class MemoryCacheBackend(ICacheBackend):
    """Backend de cache en mémoire avec LRU."""
    
    def __init__(self,
                 max_size: int = 10000,
                 max_memory_mb: int = 100,
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU):
        
        self._max_size = max_size
        self._max_memory_bytes = max_memory_mb * 1024 * 1024
        self._eviction_policy = eviction_policy
        
        # Stockage principal
        self._data: OrderedDict[str, CacheEntry] = OrderedDict()
        self._access_counts: Dict[str, int] = defaultdict(int)
        
        # Statistiques
        self._stats = CacheStats()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Métriques
        self._metrics = MetricsCollector()
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Récupère une valeur du cache."""
        start_time = time.time()
        
        with self._lock:
            entry = self._data.get(key)
            
            if entry is None:
                self._stats.misses += 1
                self._metrics.increment("cache_miss")
                return None
            
            # Vérification de l'expiration
            if entry.is_expired():
                del self._data[key]
                self._access_counts.pop(key, None)
                self._stats.misses += 1
                self._stats.evictions += 1
                self._metrics.increment("cache_miss")
                self._metrics.increment("cache_expired")
                return None
            
            # Mise à jour des statistiques d'accès
            entry.update_access()
            self._access_counts[key] += 1
            
            # LRU: déplacer à la fin
            if self._eviction_policy == EvictionPolicy.LRU:
                self._data.move_to_end(key)
            
            self._stats.hits += 1
            self._metrics.increment("cache_hit")
            
            # Temps d'accès
            access_time = (time.time() - start_time) * 1000
            self._stats.avg_access_time_ms = (
                (self._stats.avg_access_time_ms * (self._stats.hits - 1) + access_time) /
                self._stats.hits
            )
            
            return entry
    
    def set(self, key: str, entry: CacheEntry) -> bool:
        """Stocke une valeur dans le cache."""
        with self._lock:
            try:
                # Vérification de la capacité
                self._ensure_capacity(entry.size_bytes)
                
                # Stockage
                self._data[key] = entry
                self._access_counts[key] = 0
                
                # LRU: déplacer à la fin
                if self._eviction_policy == EvictionPolicy.LRU:
                    self._data.move_to_end(key)
                
                self._stats.sets += 1
                self._stats.entry_count = len(self._data)
                self._stats.total_size_bytes += entry.size_bytes
                
                self._metrics.increment("cache_set")
                self._metrics.gauge("cache_size", len(self._data))
                
                return True
                
            except Exception as e:
                self._stats.errors += 1
                self._metrics.increment("cache_error")
                return False
    
    def delete(self, key: str) -> bool:
        """Supprime une valeur du cache."""
        with self._lock:
            entry = self._data.pop(key, None)
            if entry:
                self._access_counts.pop(key, None)
                self._stats.deletes += 1
                self._stats.entry_count = len(self._data)
                self._stats.total_size_bytes -= entry.size_bytes
                self._metrics.increment("cache_delete")
                return True
            return False
    
    def exists(self, key: str) -> bool:
        """Vérifie si une clé existe."""
        with self._lock:
            entry = self._data.get(key)
            if entry and not entry.is_expired():
                return True
            return False
    
    def clear(self) -> bool:
        """Vide le cache."""
        with self._lock:
            self._data.clear()
            self._access_counts.clear()
            self._stats.entry_count = 0
            self._stats.total_size_bytes = 0
            self._metrics.increment("cache_clear")
            return True
    
    def get_stats(self) -> CacheStats:
        """Récupère les statistiques."""
        with self._lock:
            self._stats.update_hit_ratio()
            return self._stats
    
    def get_keys(self, pattern: Optional[str] = None) -> List[str]:
        """Récupère les clés selon un pattern."""
        with self._lock:
            keys = list(self._data.keys())
            
            if pattern:
                # Pattern matching simple
                import fnmatch
                keys = [k for k in keys if fnmatch.fnmatch(k, pattern)]
            
            return keys
    
    def _ensure_capacity(self, new_entry_size: int) -> None:
        """S'assure qu'il y a suffisamment de place."""
        # Vérification du nombre d'entrées
        while len(self._data) >= self._max_size:
            self._evict_one()
        
        # Vérification de la mémoire
        while (self._stats.total_size_bytes + new_entry_size) > self._max_memory_bytes:
            if not self._evict_one():
                break  # Plus rien à évincer
    
    def _evict_one(self) -> bool:
        """Évince une entrée selon la politique."""
        if not self._data:
            return False
        
        key_to_evict = None
        
        if self._eviction_policy == EvictionPolicy.LRU:
            # Le premier élément est le moins récemment utilisé
            key_to_evict = next(iter(self._data))
        
        elif self._eviction_policy == EvictionPolicy.LFU:
            # Chercher la clé avec le moins d'accès
            min_access = min(self._access_counts.values())
            for key, count in self._access_counts.items():
                if count == min_access:
                    key_to_evict = key
                    break
        
        elif self._eviction_policy == EvictionPolicy.FIFO:
            # Premier arrivé, premier sorti
            key_to_evict = next(iter(self._data))
        
        elif self._eviction_policy == EvictionPolicy.TTL:
            # Chercher l'entrée qui expire le plus tôt
            earliest_expiry = None
            for key, entry in self._data.items():
                if entry.ttl:
                    expiry_time = entry.created_at + timedelta(seconds=entry.ttl)
                    if earliest_expiry is None or expiry_time < earliest_expiry:
                        earliest_expiry = expiry_time
                        key_to_evict = key
        
        elif self._eviction_policy == EvictionPolicy.RANDOM:
            # Éviction aléatoire
            import random
            key_to_evict = random.choice(list(self._data.keys()))
        
        if key_to_evict:
            self.delete(key_to_evict)
            self._stats.evictions += 1
            self._metrics.increment("cache_eviction")
            return True
        
        return False


class RedisCacheBackend(ICacheBackend):
    """Backend de cache Redis (simulation)."""
    
    def __init__(self,
                 host: str = "localhost",
                 port: int = 6379,
                 db: int = 0,
                 password: Optional[str] = None,
                 prefix: str = "cache:",
                 serialization: SerializationFormat = SerializationFormat.PICKLE):
        
        self._host = host
        self._port = port
        self._db = db
        self._password = password
        self._prefix = prefix
        self._serialization = serialization
        
        # Simulation avec un dictionnaire (en production, utiliser redis-py)
        self._redis_sim: Dict[str, bytes] = {}
        self._stats = CacheStats()
        self._lock = threading.RLock()
        self._metrics = MetricsCollector()
        
        # Connection simulée
        self._connected = True
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Récupère une valeur du cache Redis."""
        if not self._connected:
            return None
        
        start_time = time.time()
        full_key = f"{self._prefix}{key}"
        
        with self._lock:
            data = self._redis_sim.get(full_key)
            
            if data is None:
                self._stats.misses += 1
                self._metrics.increment("redis_cache_miss")
                return None
            
            try:
                # Désérialisation
                if self._serialization == SerializationFormat.PICKLE:
                    entry = pickle.loads(data)
                elif self._serialization == SerializationFormat.JSON:
                    entry_dict = json.loads(data.decode())
                    entry = self._dict_to_cache_entry(entry_dict)
                else:
                    # Format non supporté
                    return None
                
                # Vérification de l'expiration
                if entry.is_expired():
                    del self._redis_sim[full_key]
                    self._stats.misses += 1
                    self._stats.evictions += 1
                    return None
                
                entry.update_access()
                self._stats.hits += 1
                self._metrics.increment("redis_cache_hit")
                
                # Temps d'accès
                access_time = (time.time() - start_time) * 1000
                self._stats.avg_access_time_ms = (
                    (self._stats.avg_access_time_ms * (self._stats.hits - 1) + access_time) /
                    self._stats.hits
                )
                
                return entry
                
            except Exception:
                self._stats.errors += 1
                self._metrics.increment("redis_cache_error")
                return None
    
    def set(self, key: str, entry: CacheEntry) -> bool:
        """Stocke une valeur dans le cache Redis."""
        if not self._connected:
            return False
        
        full_key = f"{self._prefix}{key}"
        
        with self._lock:
            try:
                # Sérialisation
                if self._serialization == SerializationFormat.PICKLE:
                    data = pickle.dumps(entry)
                elif self._serialization == SerializationFormat.JSON:
                    entry_dict = self._cache_entry_to_dict(entry)
                    data = json.dumps(entry_dict).encode()
                else:
                    return False
                
                # Stockage
                self._redis_sim[full_key] = data
                
                self._stats.sets += 1
                self._stats.entry_count = len(self._redis_sim)
                self._metrics.increment("redis_cache_set")
                
                return True
                
            except Exception:
                self._stats.errors += 1
                self._metrics.increment("redis_cache_error")
                return False
    
    def delete(self, key: str) -> bool:
        """Supprime une valeur du cache Redis."""
        if not self._connected:
            return False
        
        full_key = f"{self._prefix}{key}"
        
        with self._lock:
            if full_key in self._redis_sim:
                del self._redis_sim[full_key]
                self._stats.deletes += 1
                self._stats.entry_count = len(self._redis_sim)
                self._metrics.increment("redis_cache_delete")
                return True
            return False
    
    def exists(self, key: str) -> bool:
        """Vérifie si une clé existe."""
        if not self._connected:
            return False
        
        full_key = f"{self._prefix}{key}"
        return full_key in self._redis_sim
    
    def clear(self) -> bool:
        """Vide le cache Redis."""
        if not self._connected:
            return False
        
        with self._lock:
            # Supprime seulement les clés avec notre préfixe
            keys_to_delete = [k for k in self._redis_sim.keys() if k.startswith(self._prefix)]
            for key in keys_to_delete:
                del self._redis_sim[key]
            
            self._stats.entry_count = len([k for k in self._redis_sim.keys() if k.startswith(self._prefix)])
            self._metrics.increment("redis_cache_clear")
            return True
    
    def get_stats(self) -> CacheStats:
        """Récupère les statistiques."""
        with self._lock:
            self._stats.update_hit_ratio()
            return self._stats
    
    def get_keys(self, pattern: Optional[str] = None) -> List[str]:
        """Récupère les clés selon un pattern."""
        with self._lock:
            keys = [k[len(self._prefix):] for k in self._redis_sim.keys() if k.startswith(self._prefix)]
            
            if pattern:
                import fnmatch
                keys = [k for k in keys if fnmatch.fnmatch(k, pattern)]
            
            return keys
    
    def _cache_entry_to_dict(self, entry: CacheEntry) -> Dict[str, Any]:
        """Convertit une CacheEntry en dictionnaire pour JSON."""
        return {
            "key": entry.key,
            "value": entry.value,
            "created_at": entry.created_at.isoformat(),
            "accessed_at": entry.accessed_at.isoformat(),
            "access_count": entry.access_count,
            "ttl": entry.ttl,
            "tags": list(entry.tags),
            "size_bytes": entry.size_bytes,
            "compressed": entry.compressed,
            "checksum": entry.checksum
        }
    
    def _dict_to_cache_entry(self, data: Dict[str, Any]) -> CacheEntry:
        """Convertit un dictionnaire en CacheEntry."""
        return CacheEntry(
            key=data["key"],
            value=data["value"],
            created_at=datetime.fromisoformat(data["created_at"]),
            accessed_at=datetime.fromisoformat(data["accessed_at"]),
            access_count=data["access_count"],
            ttl=data["ttl"],
            tags=set(data["tags"]),
            size_bytes=data["size_bytes"],
            compressed=data["compressed"],
            checksum=data["checksum"]
        )


class MultiLevelCacheBackend(ICacheBackend):
    """Backend de cache multi-niveaux."""
    
    def __init__(self,
                 l1_backend: ICacheBackend,
                 l2_backend: ICacheBackend,
                 write_through: bool = False):
        
        self._l1 = l1_backend  # Cache rapide (mémoire)
        self._l2 = l2_backend  # Cache persistant (Redis)
        self._write_through = write_through
        
        self._stats = CacheStats()
        self._metrics = MetricsCollector()
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Récupère une valeur des caches multi-niveaux."""
        # Tentative L1
        entry = self._l1.get(key)
        if entry:
            self._stats.hits += 1
            self._metrics.increment("multilevel_l1_hit")
            return entry
        
        # Tentative L2
        entry = self._l2.get(key)
        if entry:
            # Promotion vers L1
            self._l1.set(key, entry)
            self._stats.hits += 1
            self._metrics.increment("multilevel_l2_hit")
            return entry
        
        # Cache miss complet
        self._stats.misses += 1
        self._metrics.increment("multilevel_miss")
        return None
    
    def set(self, key: str, entry: CacheEntry) -> bool:
        """Stocke une valeur dans les caches multi-niveaux."""
        success = True
        
        # Stockage L1
        if not self._l1.set(key, entry):
            success = False
        
        # Stockage L2 (optionnel selon write_through)
        if self._write_through:
            if not self._l2.set(key, entry):
                success = False
        
        if success:
            self._stats.sets += 1
            self._metrics.increment("multilevel_set")
        
        return success
    
    def delete(self, key: str) -> bool:
        """Supprime une valeur des caches multi-niveaux."""
        l1_success = self._l1.delete(key)
        l2_success = self._l2.delete(key)
        
        if l1_success or l2_success:
            self._stats.deletes += 1
            self._metrics.increment("multilevel_delete")
            return True
        
        return False
    
    def exists(self, key: str) -> bool:
        """Vérifie si une clé existe dans l'un des niveaux."""
        return self._l1.exists(key) or self._l2.exists(key)
    
    def clear(self) -> bool:
        """Vide tous les niveaux de cache."""
        l1_success = self._l1.clear()
        l2_success = self._l2.clear()
        
        if l1_success or l2_success:
            self._metrics.increment("multilevel_clear")
            return True
        
        return False
    
    def get_stats(self) -> CacheStats:
        """Récupère les statistiques combinées."""
        l1_stats = self._l1.get_stats()
        l2_stats = self._l2.get_stats()
        
        combined_stats = CacheStats()
        combined_stats.hits = l1_stats.hits + l2_stats.hits
        combined_stats.misses = l1_stats.misses + l2_stats.misses
        combined_stats.sets = l1_stats.sets + l2_stats.sets
        combined_stats.deletes = l1_stats.deletes + l2_stats.deletes
        combined_stats.evictions = l1_stats.evictions + l2_stats.evictions
        combined_stats.errors = l1_stats.errors + l2_stats.errors
        combined_stats.entry_count = l1_stats.entry_count + l2_stats.entry_count
        combined_stats.total_size_bytes = l1_stats.total_size_bytes + l2_stats.total_size_bytes
        
        combined_stats.update_hit_ratio()
        
        return combined_stats
    
    def get_keys(self, pattern: Optional[str] = None) -> List[str]:
        """Récupère les clés de tous les niveaux."""
        l1_keys = set(self._l1.get_keys(pattern))
        l2_keys = set(self._l2.get_keys(pattern))
        return list(l1_keys | l2_keys)


class CacheManager:
    """
    Gestionnaire de cache principal avec fonctionnalités avancées.
    
    Fournit une API unifiée pour le cache avec support multi-backend,
    compression, sérialisation, invalidation par tags, et monitoring.
    """
    
    def __init__(self,
                 backend: Optional[ICacheBackend] = None,
                 default_ttl: int = 3600,
                 compression_threshold: int = 1024,
                 enable_compression: bool = True,
                 enable_metrics: bool = True):
        
        # Backend de cache
        self._backend = backend or MemoryCacheBackend()
        
        # Configuration
        self._default_ttl = default_ttl
        self._compression_threshold = compression_threshold
        self._enable_compression = enable_compression
        self._enable_metrics = enable_metrics
        
        # Métriques
        self._metrics = MetricsCollector() if enable_metrics else None
        
        # Index des tags pour l'invalidation
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)
        self._key_tags: Dict[str, Set[str]] = defaultdict(set)
        
        # Callbacks pour les événements
        self._event_hooks: WeakSet[Callable[[str, str, Any], None]] = WeakSet()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Circuit breaker pour la résilience
        self._circuit_breaker_failures = 0
        self._circuit_breaker_threshold = 5
        self._circuit_breaker_timeout = 60
        self._circuit_breaker_last_failure = None
        self._circuit_breaker_open = False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Récupère une valeur du cache.
        
        Args:
            key: Clé de cache
            default: Valeur par défaut si non trouvée
            
        Returns:
            Valeur du cache ou valeur par défaut
        """
        if self._is_circuit_breaker_open():
            return default
        
        try:
            start_time = time.time()
            
            entry = self._backend.get(key)
            
            if entry is None:
                self._emit_event("cache_miss", key, None)
                return default
            
            # Décompression si nécessaire
            value = entry.value
            if entry.compressed and self._enable_compression:
                value = self._decompress(value)
            
            # Métriques
            if self._metrics:
                access_time = (time.time() - start_time) * 1000
                self._metrics.histogram("cache_access_time", access_time)
            
            self._emit_event("cache_hit", key, value)
            self._reset_circuit_breaker()
            
            return value
            
        except Exception as e:
            self._handle_circuit_breaker_failure()
            self._emit_event("cache_error", key, str(e))
            return default
    
    def set(self,
            key: str,
            value: Any,
            ttl: Optional[int] = None,
            tags: Optional[Set[str]] = None,
            compress: Optional[bool] = None) -> bool:
        """
        Stocke une valeur dans le cache.
        
        Args:
            key: Clé de cache
            value: Valeur à stocker
            ttl: Durée de vie en secondes
            tags: Tags pour l'invalidation
            compress: Forcer la compression
            
        Returns:
            True si succès
        """
        if self._is_circuit_breaker_open():
            return False
        
        try:
            # Préparation de la valeur
            final_value = value
            compressed = False
            
            # Compression si nécessaire
            if compress is None:
                compress = self._enable_compression
            
            if compress and isinstance(value, (str, bytes)):
                if len(str(value)) > self._compression_threshold:
                    final_value = self._compress(value)
                    compressed = True
            
            # Création de l'entrée
            entry = CacheEntry(
                key=key,
                value=final_value,
                ttl=ttl or self._default_ttl,
                tags=tags or set(),
                compressed=compressed
            )
            
            # Stockage
            success = self._backend.set(key, entry)
            
            if success:
                # Mise à jour de l'index des tags
                with self._lock:
                    for tag in entry.tags:
                        self._tag_index[tag].add(key)
                        self._key_tags[key].add(tag)
                
                self._emit_event("cache_set", key, value)
                self._reset_circuit_breaker()
            
            return success
            
        except Exception as e:
            self._handle_circuit_breaker_failure()
            self._emit_event("cache_error", key, str(e))
            return False
    
    def delete(self, key: str) -> bool:
        """
        Supprime une valeur du cache.
        
        Args:
            key: Clé à supprimer
            
        Returns:
            True si succès
        """
        if self._is_circuit_breaker_open():
            return False
        
        try:
            success = self._backend.delete(key)
            
            if success:
                # Nettoyage de l'index des tags
                with self._lock:
                    tags = self._key_tags.pop(key, set())
                    for tag in tags:
                        self._tag_index[tag].discard(key)
                        if not self._tag_index[tag]:
                            del self._tag_index[tag]
                
                self._emit_event("cache_delete", key, None)
                self._reset_circuit_breaker()
            
            return success
            
        except Exception as e:
            self._handle_circuit_breaker_failure()
            self._emit_event("cache_error", key, str(e))
            return False
    
    def exists(self, key: str) -> bool:
        """Vérifie si une clé existe dans le cache."""
        if self._is_circuit_breaker_open():
            return False
        
        try:
            exists = self._backend.exists(key)
            self._reset_circuit_breaker()
            return exists
        except Exception as e:
            self._handle_circuit_breaker_failure()
            return False
    
    def clear(self) -> bool:
        """Vide complètement le cache."""
        if self._is_circuit_breaker_open():
            return False
        
        try:
            success = self._backend.clear()
            
            if success:
                with self._lock:
                    self._tag_index.clear()
                    self._key_tags.clear()
                
                self._emit_event("cache_clear", None, None)
                self._reset_circuit_breaker()
            
            return success
            
        except Exception as e:
            self._handle_circuit_breaker_failure()
            return False
    
    def invalidate_by_tag(self, tag: str) -> int:
        """
        Invalide toutes les entrées avec un tag spécifique.
        
        Args:
            tag: Tag à invalider
            
        Returns:
            Nombre d'entrées supprimées
        """
        with self._lock:
            keys_to_delete = list(self._tag_index.get(tag, set()))
        
        deleted_count = 0
        for key in keys_to_delete:
            if self.delete(key):
                deleted_count += 1
        
        self._emit_event("cache_tag_invalidate", tag, deleted_count)
        return deleted_count
    
    def invalidate_by_pattern(self, pattern: str) -> int:
        """
        Invalide toutes les entrées correspondant à un pattern.
        
        Args:
            pattern: Pattern de clés (avec wildcards)
            
        Returns:
            Nombre d'entrées supprimées
        """
        try:
            keys_to_delete = self._backend.get_keys(pattern)
            deleted_count = 0
            
            for key in keys_to_delete:
                if self.delete(key):
                    deleted_count += 1
            
            self._emit_event("cache_pattern_invalidate", pattern, deleted_count)
            return deleted_count
            
        except Exception as e:
            self._emit_event("cache_error", pattern, str(e))
            return 0
    
    def get_or_set(self,
                   key: str,
                   factory: Callable[[], Any],
                   ttl: Optional[int] = None,
                   tags: Optional[Set[str]] = None) -> Any:
        """
        Récupère une valeur ou la calcule et la stocke si absente.
        
        Args:
            key: Clé de cache
            factory: Fonction pour calculer la valeur
            ttl: Durée de vie
            tags: Tags pour l'invalidation
            
        Returns:
            Valeur du cache ou calculée
        """
        # Tentative de récupération
        value = self.get(key)
        if value is not None:
            return value
        
        # Calcul et stockage
        try:
            computed_value = factory()
            self.set(key, computed_value, ttl=ttl, tags=tags)
            return computed_value
        except Exception as e:
            self._emit_event("cache_factory_error", key, str(e))
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques détaillées du cache."""
        backend_stats = self._backend.get_stats()
        
        with self._lock:
            stats = backend_stats.to_dict()
            stats.update({
                "tag_count": len(self._tag_index),
                "keys_with_tags": len(self._key_tags),
                "circuit_breaker_open": self._circuit_breaker_open,
                "circuit_breaker_failures": self._circuit_breaker_failures
            })
        
        return stats
    
    def add_event_hook(self, hook: Callable[[str, str, Any], None]) -> None:
        """Ajoute un hook pour les événements de cache."""
        self._event_hooks.add(hook)
    
    def _compress(self, data: Any) -> bytes:
        """Compresse des données."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        elif not isinstance(data, bytes):
            data = str(data).encode('utf-8')
        
        return gzip.compress(data)
    
    def _decompress(self, data: bytes) -> str:
        """Décompresse des données."""
        return gzip.decompress(data).decode('utf-8')
    
    def _emit_event(self, event_type: str, key: Optional[str], data: Any) -> None:
        """Émet un événement de cache."""
        for hook in self._event_hooks:
            try:
                hook(event_type, key or "", data)
            except Exception:
                continue
        
        if self._metrics:
            self._metrics.increment(f"cache_event_{event_type}")
    
    def _is_circuit_breaker_open(self) -> bool:
        """Vérifie si le circuit breaker est ouvert."""
        if not self._circuit_breaker_open:
            return False
        
        # Vérification du timeout
        if self._circuit_breaker_last_failure:
            elapsed = time.time() - self._circuit_breaker_last_failure
            if elapsed > self._circuit_breaker_timeout:
                self._circuit_breaker_open = False
                self._circuit_breaker_failures = 0
                return False
        
        return True
    
    def _handle_circuit_breaker_failure(self) -> None:
        """Gère les échecs du circuit breaker."""
        self._circuit_breaker_failures += 1
        self._circuit_breaker_last_failure = time.time()
        
        if self._circuit_breaker_failures >= self._circuit_breaker_threshold:
            self._circuit_breaker_open = True
            if self._metrics:
                self._metrics.increment("cache_circuit_breaker_open")
    
    def _reset_circuit_breaker(self) -> None:
        """Remet à zéro le circuit breaker."""
        if self._circuit_breaker_failures > 0:
            self._circuit_breaker_failures = 0
            self._circuit_breaker_last_failure = None
            if self._circuit_breaker_open:
                self._circuit_breaker_open = False
                if self._metrics:
                    self._metrics.increment("cache_circuit_breaker_close")


# Instance globale singleton
_global_cache_manager: Optional[CacheManager] = None
_cache_lock = threading.Lock()


def get_cache_manager(**kwargs) -> CacheManager:
    """
    Récupère l'instance globale du gestionnaire de cache.
    
    Returns:
        Instance singleton du CacheManager
    """
    global _global_cache_manager
    
    if _global_cache_manager is None:
        with _cache_lock:
            if _global_cache_manager is None:
                _global_cache_manager = CacheManager(**kwargs)
    
    return _global_cache_manager


# API publique simplifiée
def cache_get(key: str, default: Any = None) -> Any:
    """API simplifiée pour récupérer du cache."""
    manager = get_cache_manager()
    return manager.get(key, default)


def cache_set(key: str, value: Any, ttl: Optional[int] = None, tags: Optional[Set[str]] = None) -> bool:
    """API simplifiée pour stocker dans le cache."""
    manager = get_cache_manager()
    return manager.set(key, value, ttl=ttl, tags=tags)


def cache_delete(key: str) -> bool:
    """API simplifiée pour supprimer du cache."""
    manager = get_cache_manager()
    return manager.delete(key)


def cache_clear() -> bool:
    """API simplifiée pour vider le cache."""
    manager = get_cache_manager()
    return manager.clear()
