"""
Gestionnaire de cache distribué avancé pour le système de tenancy
Auteur: Fahed Mlaiel - Lead Dev & Architecte IA
"""

import asyncio
import json
import pickle
import gzip
import hashlib
import time
from typing import Dict, Any, List, Optional, Union, Callable, Type, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import aioredis
from abc import ABC, abstractmethod
import structlog

logger = structlog.get_logger(__name__)

class CacheStrategy(Enum):
    """Stratégies de cache"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"
    WRITE_AROUND = "write_around"

class CacheLevel(Enum):
    """Niveaux de cache"""
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_PERSISTENT = "l3_persistent"

class SerializationFormat(Enum):
    """Formats de sérialisation"""
    JSON = "json"
    PICKLE = "pickle"
    MSGPACK = "msgpack"
    PROTOBUF = "protobuf"

@dataclass
class CacheKey:
    """Clé de cache structurée"""
    namespace: str
    tenant_id: Optional[str] = None
    entity_type: str = "default"
    entity_id: str = ""
    version: str = "v1"
    tags: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        if not self.entity_id:
            self.entity_id = "default"
    
    def to_string(self) -> str:
        """Convertit la clé en chaîne"""
        parts = [self.namespace, self.version, self.entity_type]
        
        if self.tenant_id:
            parts.append(f"tenant:{self.tenant_id}")
            
        parts.append(self.entity_id)
        
        if self.tags:
            parts.append(f"tags:{','.join(sorted(self.tags))}")
        
        return ":".join(parts)
    
    def to_pattern(self) -> str:
        """Convertit en pattern pour la recherche"""
        pattern = f"{self.namespace}:{self.version}:{self.entity_type}"
        
        if self.tenant_id:
            pattern += f":tenant:{self.tenant_id}"
        else:
            pattern += ":tenant:*"
            
        pattern += ":*"
        return pattern
    
    @classmethod
    def from_string(cls, key_str: str) -> 'CacheKey':
        """Crée une clé depuis une chaîne"""
        parts = key_str.split(":")
        
        if len(parts) < 4:
            raise ValueError(f"Format de clé invalide: {key_str}")
        
        namespace = parts[0]
        version = parts[1]
        entity_type = parts[2]
        
        tenant_id = None
        entity_id = "default"
        tags = set()
        
        i = 3
        while i < len(parts):
            if parts[i] == "tenant" and i + 1 < len(parts):
                tenant_id = parts[i + 1]
                i += 2
            elif parts[i] == "tags" and i + 1 < len(parts):
                tags = set(parts[i + 1].split(","))
                i += 2
            else:
                entity_id = parts[i]
                i += 1
        
        return cls(
            namespace=namespace,
            tenant_id=tenant_id,
            entity_type=entity_type,
            entity_id=entity_id,
            version=version,
            tags=tags
        )

@dataclass
class CacheEntry:
    """Entrée de cache avec métadonnées"""
    key: CacheKey
    value: Any
    ttl: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    accessed_at: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    size_bytes: int = 0
    compression_enabled: bool = False
    serialization_format: SerializationFormat = SerializationFormat.JSON
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Vérifie si l'entrée a expiré"""
        if self.ttl is None:
            return False
        
        expiry_time = self.created_at + timedelta(seconds=self.ttl)
        return datetime.utcnow() > expiry_time
    
    def update_access(self) -> None:
        """Met à jour les statistiques d'accès"""
        self.accessed_at = datetime.utcnow()
        self.access_count += 1

class CacheSerializer:
    """Sérialiseur de cache avancé"""
    
    @staticmethod
    def serialize(value: Any, format: SerializationFormat, 
                 compress: bool = False) -> bytes:
        """Sérialise une valeur"""
        try:
            if format == SerializationFormat.JSON:
                data = json.dumps(value, default=str).encode('utf-8')
            elif format == SerializationFormat.PICKLE:
                data = pickle.dumps(value)
            else:
                # Fallback to JSON
                data = json.dumps(value, default=str).encode('utf-8')
            
            if compress:
                data = gzip.compress(data)
            
            return data
            
        except Exception as e:
            logger.error("Erreur de sérialisation", error=str(e), format=format.value)
            raise
    
    @staticmethod
    def deserialize(data: bytes, format: SerializationFormat, 
                   compressed: bool = False) -> Any:
        """Désérialise une valeur"""
        try:
            if compressed:
                data = gzip.decompress(data)
            
            if format == SerializationFormat.JSON:
                return json.loads(data.decode('utf-8'))
            elif format == SerializationFormat.PICKLE:
                return pickle.loads(data)
            else:
                # Fallback to JSON
                return json.loads(data.decode('utf-8'))
                
        except Exception as e:
            logger.error("Erreur de désérialisation", error=str(e), format=format.value)
            raise

class CacheBackend(ABC):
    """Interface de backend de cache"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[bytes]:
        """Récupère une valeur"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> bool:
        """Stocke une valeur"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Supprime une valeur"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Vérifie l'existence d'une clé"""
        pass
    
    @abstractmethod
    async def clear(self, pattern: Optional[str] = None) -> int:
        """Nettoie le cache"""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques"""
        pass

class MemoryCacheBackend(CacheBackend):
    """Backend de cache en mémoire"""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
    async def get(self, key: str) -> Optional[bytes]:
        """Récupère une valeur du cache mémoire"""
        if key not in self.cache:
            self.misses += 1
            return None
        
        entry = self.cache[key]
        
        if entry.is_expired():
            del self.cache[key]
            self.misses += 1
            return None
        
        entry.update_access()
        self.hits += 1
        
        return CacheSerializer.serialize(
            entry.value, 
            entry.serialization_format,
            entry.compression_enabled
        )
    
    async def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> bool:
        """Stocke une valeur en mémoire"""
        try:
            # Éviction si nécessaire
            if len(self.cache) >= self.max_size:
                await self._evict_lru()
            
            # Désérialisation pour stocker l'objet original
            deserialized_value = CacheSerializer.deserialize(
                value, SerializationFormat.JSON, False
            )
            
            cache_key = CacheKey.from_string(key)
            entry = CacheEntry(
                key=cache_key,
                value=deserialized_value,
                ttl=ttl,
                size_bytes=len(value)
            )
            
            self.cache[key] = entry
            return True
            
        except Exception as e:
            logger.error("Erreur lors du stockage en mémoire", error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Supprime une clé du cache mémoire"""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    async def exists(self, key: str) -> bool:
        """Vérifie l'existence d'une clé en mémoire"""
        return key in self.cache and not self.cache[key].is_expired()
    
    async def clear(self, pattern: Optional[str] = None) -> int:
        """Nettoie le cache mémoire"""
        if pattern is None:
            count = len(self.cache)
            self.cache.clear()
            return count
        
        # Suppression par pattern (simple)
        keys_to_delete = [
            key for key in self.cache.keys() 
            if pattern.replace("*", "") in key
        ]
        
        for key in keys_to_delete:
            del self.cache[key]
        
        return len(keys_to_delete)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques du cache mémoire"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "backend": "memory",
            "entries": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_percent": round(hit_rate, 2),
            "evictions": self.evictions,
            "memory_usage_bytes": sum(entry.size_bytes for entry in self.cache.values())
        }
    
    async def _evict_lru(self) -> None:
        """Éviction LRU (Least Recently Used)"""
        if not self.cache:
            return
        
        # Trouve l'entrée la moins récemment utilisée
        lru_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k].accessed_at
        )
        
        del self.cache[lru_key]
        self.evictions += 1

class RedisCacheBackend(CacheBackend):
    """Backend de cache Redis"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 db: int = 0, max_connections: int = 10):
        self.redis_url = redis_url
        self.db = db
        self.max_connections = max_connections
        self.redis_pool: Optional[aioredis.ConnectionPool] = None
        self.redis: Optional[aioredis.Redis] = None
        
    async def initialize(self) -> None:
        """Initialise la connexion Redis"""
        try:
            self.redis_pool = aioredis.ConnectionPool.from_url(
                self.redis_url,
                db=self.db,
                max_connections=self.max_connections,
                retry_on_timeout=True
            )
            
            self.redis = aioredis.Redis(connection_pool=self.redis_pool)
            
            # Test de connexion
            await self.redis.ping()
            logger.info("Connexion Redis établie")
            
        except Exception as e:
            logger.error("Erreur de connexion Redis", error=str(e))
            raise
    
    async def get(self, key: str) -> Optional[bytes]:
        """Récupère une valeur de Redis"""
        try:
            if not self.redis:
                await self.initialize()
            
            value = await self.redis.get(key)
            return value
            
        except Exception as e:
            logger.error("Erreur lors de la récupération Redis", key=key, error=str(e))
            return None
    
    async def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> bool:
        """Stocke une valeur dans Redis"""
        try:
            if not self.redis:
                await self.initialize()
            
            if ttl:
                await self.redis.setex(key, ttl, value)
            else:
                await self.redis.set(key, value)
            
            return True
            
        except Exception as e:
            logger.error("Erreur lors du stockage Redis", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Supprime une clé de Redis"""
        try:
            if not self.redis:
                await self.initialize()
            
            result = await self.redis.delete(key)
            return result > 0
            
        except Exception as e:
            logger.error("Erreur lors de la suppression Redis", key=key, error=str(e))
            return False
    
    async def exists(self, key: str) -> bool:
        """Vérifie l'existence d'une clé dans Redis"""
        try:
            if not self.redis:
                await self.initialize()
            
            result = await self.redis.exists(key)
            return result > 0
            
        except Exception as e:
            logger.error("Erreur lors de la vérification Redis", key=key, error=str(e))
            return False
    
    async def clear(self, pattern: Optional[str] = None) -> int:
        """Nettoie le cache Redis"""
        try:
            if not self.redis:
                await self.initialize()
            
            if pattern is None:
                # Attention: FLUSHDB supprime toute la base
                return await self.redis.flushdb()
            
            # Suppression par pattern
            keys = await self.redis.keys(pattern)
            if keys:
                return await self.redis.delete(*keys)
            
            return 0
            
        except Exception as e:
            logger.error("Erreur lors du nettoyage Redis", pattern=pattern, error=str(e))
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques Redis"""
        try:
            if not self.redis:
                await self.initialize()
            
            info = await self.redis.info()
            
            return {
                "backend": "redis",
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "redis_version": info.get("redis_version", "unknown")
            }
            
        except Exception as e:
            logger.error("Erreur lors de la récupération des stats Redis", error=str(e))
            return {"backend": "redis", "error": str(e)}
    
    async def close(self) -> None:
        """Ferme la connexion Redis"""
        if self.redis:
            await self.redis.close()
        if self.redis_pool:
            await self.redis_pool.disconnect()

class DistributedCache:
    """Cache distribué multi-niveaux"""
    
    def __init__(self, 
                 l1_backend: Optional[CacheBackend] = None,
                 l2_backend: Optional[CacheBackend] = None,
                 default_ttl: int = 3600,
                 compression_threshold: int = 1024):
        self.l1_backend = l1_backend or MemoryCacheBackend()
        self.l2_backend = l2_backend
        self.default_ttl = default_ttl
        self.compression_threshold = compression_threshold
        self.stats = {
            "operations": 0,
            "l1_hits": 0,
            "l2_hits": 0,
            "misses": 0,
            "errors": 0
        }
    
    async def get(self, key: CacheKey, 
                 deserialize: bool = True) -> Optional[Any]:
        """Récupère une valeur du cache distribué"""
        self.stats["operations"] += 1
        key_str = key.to_string()
        
        try:
            # Tentative L1 (mémoire)
            value = await self.l1_backend.get(key_str)
            if value is not None:
                self.stats["l1_hits"] += 1
                if deserialize:
                    return CacheSerializer.deserialize(
                        value, SerializationFormat.JSON, False
                    )
                return value
            
            # Tentative L2 (Redis)
            if self.l2_backend:
                value = await self.l2_backend.get(key_str)
                if value is not None:
                    self.stats["l2_hits"] += 1
                    
                    # Remontée vers L1
                    await self.l1_backend.set(key_str, value)
                    
                    if deserialize:
                        return CacheSerializer.deserialize(
                            value, SerializationFormat.JSON, False
                        )
                    return value
            
            self.stats["misses"] += 1
            return None
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error("Erreur lors de la récupération", key=key_str, error=str(e))
            return None
    
    async def set(self, key: CacheKey, value: Any, 
                 ttl: Optional[int] = None) -> bool:
        """Stocke une valeur dans le cache distribué"""
        self.stats["operations"] += 1
        key_str = key.to_string()
        ttl = ttl or self.default_ttl
        
        try:
            # Sérialisation
            serialized_value = CacheSerializer.serialize(
                value, SerializationFormat.JSON, 
                len(str(value)) > self.compression_threshold
            )
            
            success = True
            
            # Stockage L1
            l1_success = await self.l1_backend.set(key_str, serialized_value, ttl)
            if not l1_success:
                success = False
            
            # Stockage L2
            if self.l2_backend:
                l2_success = await self.l2_backend.set(key_str, serialized_value, ttl)
                if not l2_success:
                    success = False
            
            return success
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error("Erreur lors du stockage", key=key_str, error=str(e))
            return False
    
    async def delete(self, key: CacheKey) -> bool:
        """Supprime une valeur du cache distribué"""
        key_str = key.to_string()
        
        try:
            success = True
            
            # Suppression L1
            l1_success = await self.l1_backend.delete(key_str)
            if not l1_success:
                success = False
            
            # Suppression L2
            if self.l2_backend:
                l2_success = await self.l2_backend.delete(key_str)
                if not l2_success:
                    success = False
            
            return success
            
        except Exception as e:
            logger.error("Erreur lors de la suppression", key=key_str, error=str(e))
            return False
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalide toutes les clés correspondant au pattern"""
        try:
            total_deleted = 0
            
            # Invalidation L1
            l1_deleted = await self.l1_backend.clear(pattern)
            total_deleted += l1_deleted
            
            # Invalidation L2
            if self.l2_backend:
                l2_deleted = await self.l2_backend.clear(pattern)
                total_deleted += l2_deleted
            
            return total_deleted
            
        except Exception as e:
            logger.error("Erreur lors de l'invalidation", pattern=pattern, error=str(e))
            return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques du cache"""
        try:
            stats = {
                "distributed_cache": self.stats.copy()
            }
            
            # Stats L1
            l1_stats = await self.l1_backend.get_stats()
            stats["l1_cache"] = l1_stats
            
            # Stats L2
            if self.l2_backend:
                l2_stats = await self.l2_backend.get_stats()
                stats["l2_cache"] = l2_stats
            
            # Calculs globaux
            total_operations = self.stats["operations"]
            if total_operations > 0:
                stats["distributed_cache"]["total_hit_rate"] = round(
                    (self.stats["l1_hits"] + self.stats["l2_hits"]) / total_operations * 100, 2
                )
            
            return stats
            
        except Exception as e:
            logger.error("Erreur lors de la récupération des stats", error=str(e))
            return {"error": str(e)}

class CacheManager:
    """Gestionnaire de cache principal"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.cache: Optional[DistributedCache] = None
        self.is_connected = False
        
    async def initialize(self) -> None:
        """Initialise le gestionnaire de cache"""
        try:
            # Configuration par défaut
            redis_url = self.config.get("redis_url", "redis://localhost:6379")
            redis_db = self.config.get("redis_db", 0)
            memory_max_size = self.config.get("memory_max_size", 1000)
            default_ttl = self.config.get("default_ttl", 3600)
            
            # Backends
            l1_backend = MemoryCacheBackend(max_size=memory_max_size)
            
            l2_backend = None
            if self.config.get("redis_enabled", True):
                l2_backend = RedisCacheBackend(redis_url, redis_db)
                await l2_backend.initialize()
            
            # Cache distribué
            self.cache = DistributedCache(
                l1_backend=l1_backend,
                l2_backend=l2_backend,
                default_ttl=default_ttl
            )
            
            self.is_connected = True
            logger.info("Gestionnaire de cache initialisé")
            
        except Exception as e:
            logger.error("Erreur lors de l'initialisation du cache", error=str(e))
            raise
    
    async def get_tenant_config(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Récupère la configuration d'un tenant depuis le cache"""
        key = CacheKey(
            namespace="tenancy",
            tenant_id=tenant_id,
            entity_type="config",
            entity_id="main"
        )
        
        return await self.cache.get(key)
    
    async def set_tenant_config(self, tenant_id: str, config: Dict[str, Any],
                               ttl: Optional[int] = None) -> bool:
        """Stocke la configuration d'un tenant dans le cache"""
        key = CacheKey(
            namespace="tenancy",
            tenant_id=tenant_id,
            entity_type="config",
            entity_id="main"
        )
        
        return await self.cache.set(key, config, ttl)
    
    async def invalidate_tenant(self, tenant_id: str) -> int:
        """Invalide toutes les données cache d'un tenant"""
        pattern = CacheKey(
            namespace="tenancy",
            tenant_id=tenant_id,
            entity_type="*",
            entity_id="*"
        ).to_pattern()
        
        return await self.cache.invalidate_pattern(pattern)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques du cache"""
        if not self.cache:
            return {"error": "Cache non initialisé"}
        
        return await self.cache.get_cache_stats()
    
    async def cleanup(self) -> None:
        """Nettoie les ressources du cache"""
        if self.cache and self.cache.l2_backend:
            if hasattr(self.cache.l2_backend, 'close'):
                await self.cache.l2_backend.close()
        
        self.is_connected = False
        logger.info("Gestionnaire de cache nettoyé")

# Instance globale du gestionnaire de cache
cache_manager = CacheManager()

# Fonctions utilitaires
async def cache_result(key: CacheKey, func: Callable, *args, 
                      ttl: Optional[int] = None, **kwargs) -> Any:
    """Décorateur de cache pour les résultats de fonction"""
    if not cache_manager.cache:
        return await func(*args, **kwargs)
    
    # Tentative de récupération depuis le cache
    cached_result = await cache_manager.cache.get(key)
    if cached_result is not None:
        return cached_result
    
    # Exécution de la fonction et mise en cache
    result = await func(*args, **kwargs)
    await cache_manager.cache.set(key, result, ttl)
    
    return result

def cache_key_for_tenant(tenant_id: str, entity_type: str, 
                        entity_id: str = "default") -> CacheKey:
    """Crée une clé de cache pour un tenant"""
    return CacheKey(
        namespace="tenancy",
        tenant_id=tenant_id,
        entity_type=entity_type,
        entity_id=entity_id
    )

def cache_key_for_user(tenant_id: str, user_id: str, 
                      data_type: str = "profile") -> CacheKey:
    """Crée une clé de cache pour un utilisateur"""
    return CacheKey(
        namespace="users",
        tenant_id=tenant_id,
        entity_type=data_type,
        entity_id=user_id
    )
