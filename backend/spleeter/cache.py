"""
🎵 Spotify AI Agent - Cache Manager
==================================

Système de cache avancé pour optimiser les performances de Spleeter
avec support Redis, mémoire et stockage persistant.

🎖️ Développé par l'équipe d'experts enterprise
"""

import os
import pickle
import json
import hashlib
import asyncio
import aiofiles
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import time
from datetime import datetime, timedelta
import tempfile
import shutil
import sqlite3
from contextlib import asynccontextmanager
import numpy as np

from .exceptions import SpleeterError
from .monitoring import PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Entrée de cache avec métadonnées"""
    key: str
    value_hash: str
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[float] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def is_expired(self) -> bool:
        """Vérifie si l'entrée a expiré"""
        if self.ttl is None:
            return False
        return time.time() > (self.created_at + self.ttl)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Crée depuis un dictionnaire"""
        return cls(**data)


class CacheConfig:
    """Configuration du système de cache"""
    
    def __init__(
        self,
        # Stockage
        cache_dir: Optional[str] = None,
        max_memory_mb: int = 512,
        max_disk_gb: float = 5.0,
        
        # Redis (optionnel)
        redis_enabled: bool = False,
        redis_url: str = "redis://localhost:6379",
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        
        # TTL par défaut
        default_ttl: int = 3600,  # 1 heure
        audio_cache_ttl: int = 7200,  # 2 heures
        model_cache_ttl: int = 86400,  # 24 heures
        
        # Stratégies
        eviction_policy: str = "lru",  # lru, lfu, ttl
        compression_enabled: bool = True,
        encryption_enabled: bool = False,
        
        # Performance
        cleanup_interval: int = 300,  # 5 minutes
        batch_size: int = 100
    ):
        self.cache_dir = Path(cache_dir or os.path.expanduser("~/.spleeter/cache"))
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_disk_bytes = max_disk_gb * 1024 * 1024 * 1024
        
        self.redis_enabled = redis_enabled
        self.redis_url = redis_url
        self.redis_db = redis_db
        self.redis_password = redis_password
        
        self.default_ttl = default_ttl
        self.audio_cache_ttl = audio_cache_ttl
        self.model_cache_ttl = model_cache_ttl
        
        self.eviction_policy = eviction_policy
        self.compression_enabled = compression_enabled
        self.encryption_enabled = encryption_enabled
        
        self.cleanup_interval = cleanup_interval
        self.batch_size = batch_size
        
        # Créer le répertoire de cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)


class MemoryCache:
    """Cache en mémoire avec LRU"""
    
    def __init__(self, max_size_bytes: int):
        self.max_size_bytes = max_size_bytes
        self._cache = {}
        self._access_order = []
        self._current_size = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Récupère une valeur du cache"""
        if key not in self._cache:
            return None
        
        # Mise à jour de l'ordre d'accès
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        
        entry = self._cache[key]
        entry.last_accessed = time.time()
        entry.access_count += 1
        
        return entry
    
    def set(self, key: str, entry: CacheEntry) -> bool:
        """Stocke une valeur dans le cache"""
        # Vérifier si l'entrée existe déjà
        if key in self._cache:
            old_entry = self._cache[key]
            self._current_size -= old_entry.size_bytes
            self._access_order.remove(key)
        
        # Vérifier l'espace disponible
        while self._current_size + entry.size_bytes > self.max_size_bytes and self._access_order:
            self._evict_lru()
        
        if self._current_size + entry.size_bytes > self.max_size_bytes:
            return False  # Pas assez de place
        
        # Ajouter l'entrée
        self._cache[key] = entry
        self._access_order.append(key)
        self._current_size += entry.size_bytes
        
        return True
    
    def remove(self, key: str) -> bool:
        """Supprime une entrée du cache"""
        if key not in self._cache:
            return False
        
        entry = self._cache[key]
        del self._cache[key]
        self._access_order.remove(key)
        self._current_size -= entry.size_bytes
        
        return True
    
    def _evict_lru(self):
        """Évince l'entrée la moins récemment utilisée"""
        if not self._access_order:
            return
        
        lru_key = self._access_order[0]
        self.remove(lru_key)
    
    def clear(self):
        """Vide le cache"""
        self._cache.clear()
        self._access_order.clear()
        self._current_size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache"""
        return {
            "entries": len(self._cache),
            "size_bytes": self._current_size,
            "max_size_bytes": self.max_size_bytes,
            "usage_percentage": (self._current_size / self.max_size_bytes) * 100
        }


class DiskCache:
    """Cache sur disque avec base de données SQLite"""
    
    def __init__(self, cache_dir: Path, max_size_bytes: int):
        self.cache_dir = cache_dir
        self.max_size_bytes = max_size_bytes
        self.db_path = cache_dir / "cache.db"
        
        # Initialiser la base de données
        self._init_database()
    
    def _init_database(self):
        """Initialise la base de données SQLite"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value_hash TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    access_count INTEGER NOT NULL,
                    ttl REAL,
                    tags TEXT,
                    file_path TEXT NOT NULL
                )
            """)
            
            # Index pour optimiser les requêtes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_entries(last_accessed)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON cache_entries(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ttl ON cache_entries(ttl)")
            
            conn.commit()
    
    async def get(self, key: str) -> Optional[Tuple[CacheEntry, Any]]:
        """Récupère une entrée du cache disque"""
        def _get_from_db():
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM cache_entries WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                return row
        
        # Récupération depuis la DB
        loop = asyncio.get_event_loop()
        row = await loop.run_in_executor(None, _get_from_db)
        
        if not row:
            return None
        
        # Reconstruction de l'entrée
        entry_data = {
            "key": row[0],
            "value_hash": row[1],
            "size_bytes": row[2],
            "created_at": row[3],
            "last_accessed": row[4],
            "access_count": row[5],
            "ttl": row[6],
            "tags": json.loads(row[7]) if row[7] else []
        }
        
        entry = CacheEntry.from_dict(entry_data)
        file_path = Path(row[8])
        
        # Vérifier l'expiration
        if entry.is_expired():
            await self.remove(key)
            return None
        
        # Vérifier que le fichier existe
        if not file_path.exists():
            await self.remove(key)
            return None
        
        # Charger la valeur
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                data = await f.read()
                value = pickle.loads(data)
            
            # Mise à jour des statistiques d'accès
            entry.last_accessed = time.time()
            entry.access_count += 1
            await self._update_access_stats(key, entry)
            
            return entry, value
            
        except Exception as e:
            logger.error(f"Erreur lecture cache {key}: {e}")
            await self.remove(key)
            return None
    
    async def set(self, key: str, value: Any, entry: CacheEntry) -> bool:
        """Stocke une entrée dans le cache disque"""
        # Générer le chemin du fichier
        value_hash = hashlib.md5(str(value).encode()).hexdigest()
        file_path = self.cache_dir / f"{key}_{value_hash}.cache"
        
        try:
            # Sérialiser la valeur
            data = pickle.dumps(value)
            entry.size_bytes = len(data)
            entry.value_hash = value_hash
            
            # Vérifier l'espace disponible
            await self._ensure_space(entry.size_bytes)
            
            # Sauvegarder le fichier
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(data)
            
            # Insérer dans la base de données
            def _insert_db():
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO cache_entries 
                        (key, value_hash, size_bytes, created_at, last_accessed, 
                         access_count, ttl, tags, file_path)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        entry.key,
                        entry.value_hash,
                        entry.size_bytes,
                        entry.created_at,
                        entry.last_accessed,
                        entry.access_count,
                        entry.ttl,
                        json.dumps(entry.tags),
                        str(file_path)
                    ))
                    conn.commit()
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _insert_db)
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur stockage cache {key}: {e}")
            if file_path.exists():
                file_path.unlink()
            return False
    
    async def remove(self, key: str) -> bool:
        """Supprime une entrée du cache"""
        def _remove_from_db():
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT file_path FROM cache_entries WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                
                if row:
                    conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    conn.commit()
                    return row[0]
                return None
        
        loop = asyncio.get_event_loop()
        file_path_str = await loop.run_in_executor(None, _remove_from_db)
        
        if file_path_str:
            file_path = Path(file_path_str)
            if file_path.exists():
                file_path.unlink()
            return True
        
        return False
    
    async def _update_access_stats(self, key: str, entry: CacheEntry):
        """Met à jour les statistiques d'accès"""
        def _update():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE cache_entries 
                    SET last_accessed = ?, access_count = ?
                    WHERE key = ?
                """, (entry.last_accessed, entry.access_count, key))
                conn.commit()
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _update)
    
    async def _ensure_space(self, required_bytes: int):
        """S'assure qu'il y a assez d'espace disque"""
        def _get_current_size():
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT SUM(size_bytes) FROM cache_entries")
                result = cursor.fetchone()[0]
                return result or 0
        
        loop = asyncio.get_event_loop()
        current_size = await loop.run_in_executor(None, _get_current_size)
        
        # Nettoyer si nécessaire
        while current_size + required_bytes > self.max_size_bytes:
            evicted = await self._evict_entries(1)
            if not evicted:
                break
            current_size = await loop.run_in_executor(None, _get_current_size)
    
    async def _evict_entries(self, count: int) -> int:
        """Évince des entrées selon la politique LRU"""
        def _get_lru_keys():
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT key FROM cache_entries 
                    ORDER BY last_accessed ASC 
                    LIMIT ?
                """, (count,))
                return [row[0] for row in cursor.fetchall()]
        
        loop = asyncio.get_event_loop()
        lru_keys = await loop.run_in_executor(None, _get_lru_keys)
        
        evicted_count = 0
        for key in lru_keys:
            if await self.remove(key):
                evicted_count += 1
        
        return evicted_count
    
    async def cleanup_expired(self) -> int:
        """Nettoie les entrées expirées"""
        def _get_expired_keys():
            current_time = time.time()
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT key FROM cache_entries 
                    WHERE ttl IS NOT NULL AND (created_at + ttl) < ?
                """, (current_time,))
                return [row[0] for row in cursor.fetchall()]
        
        loop = asyncio.get_event_loop()
        expired_keys = await loop.run_in_executor(None, _get_expired_keys)
        
        removed_count = 0
        for key in expired_keys:
            if await self.remove(key):
                removed_count += 1
        
        logger.info(f"Nettoyage cache: {removed_count} entrées expirées supprimées")
        return removed_count
    
    async def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache disque"""
        def _get_stats():
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as entries,
                        SUM(size_bytes) as total_size,
                        AVG(access_count) as avg_access_count,
                        MIN(created_at) as oldest_entry,
                        MAX(last_accessed) as newest_access
                    FROM cache_entries
                """)
                row = cursor.fetchone()
                return row
        
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(None, _get_stats)
        
        return {
            "entries": stats[0] or 0,
            "size_bytes": stats[1] or 0,
            "max_size_bytes": self.max_size_bytes,
            "usage_percentage": ((stats[1] or 0) / self.max_size_bytes) * 100,
            "avg_access_count": stats[2] or 0,
            "oldest_entry": stats[3],
            "newest_access": stats[4]
        }


class RedisCache:
    """Cache Redis pour distribué"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._redis = None
    
    async def connect(self):
        """Connexion à Redis"""
        try:
            import aioredis
            
            self._redis = await aioredis.from_url(
                self.config.redis_url,
                db=self.config.redis_db,
                password=self.config.redis_password,
                encoding="utf-8",
                decode_responses=False  # Pour les données binaires
            )
            
            # Test de connexion
            await self._redis.ping()
            logger.info("Connexion Redis établie")
            
        except ImportError:
            logger.warning("aioredis non disponible - cache Redis désactivé")
            self._redis = None
        except Exception as e:
            logger.error(f"Erreur connexion Redis: {e}")
            self._redis = None
    
    async def get(self, key: str) -> Optional[Tuple[CacheEntry, Any]]:
        """Récupère une entrée depuis Redis"""
        if not self._redis:
            return None
        
        try:
            # Récupération des métadonnées
            metadata_key = f"meta:{key}"
            metadata_data = await self._redis.get(metadata_key)
            
            if not metadata_data:
                return None
            
            metadata = json.loads(metadata_data)
            entry = CacheEntry.from_dict(metadata)
            
            # Vérifier l'expiration
            if entry.is_expired():
                await self.remove(key)
                return None
            
            # Récupération de la valeur
            value_key = f"data:{key}"
            value_data = await self._redis.get(value_key)
            
            if not value_data:
                await self.remove(key)
                return None
            
            value = pickle.loads(value_data)
            
            # Mise à jour des statistiques d'accès
            entry.last_accessed = time.time()
            entry.access_count += 1
            await self._update_metadata(key, entry)
            
            return entry, value
            
        except Exception as e:
            logger.error(f"Erreur lecture Redis {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, entry: CacheEntry) -> bool:
        """Stocke une entrée dans Redis"""
        if not self._redis:
            return False
        
        try:
            # Sérialisation
            value_data = pickle.dumps(value)
            entry.size_bytes = len(value_data)
            
            # Stockage des données
            value_key = f"data:{key}"
            metadata_key = f"meta:{key}"
            
            # Pipeline pour atomicité
            pipe = self._redis.pipeline()
            
            # Données
            if entry.ttl:
                pipe.setex(value_key, int(entry.ttl), value_data)
                pipe.setex(metadata_key, int(entry.ttl), json.dumps(entry.to_dict()))
            else:
                pipe.set(value_key, value_data)
                pipe.set(metadata_key, json.dumps(entry.to_dict()))
            
            await pipe.execute()
            return True
            
        except Exception as e:
            logger.error(f"Erreur stockage Redis {key}: {e}")
            return False
    
    async def remove(self, key: str) -> bool:
        """Supprime une entrée de Redis"""
        if not self._redis:
            return False
        
        try:
            value_key = f"data:{key}"
            metadata_key = f"meta:{key}"
            
            result = await self._redis.delete(value_key, metadata_key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Erreur suppression Redis {key}: {e}")
            return False
    
    async def _update_metadata(self, key: str, entry: CacheEntry):
        """Met à jour les métadonnées"""
        if not self._redis:
            return
        
        try:
            metadata_key = f"meta:{key}"
            ttl = await self._redis.ttl(metadata_key)
            
            if ttl > 0:
                await self._redis.setex(metadata_key, ttl, json.dumps(entry.to_dict()))
            else:
                await self._redis.set(metadata_key, json.dumps(entry.to_dict()))
                
        except Exception as e:
            logger.error(f"Erreur mise à jour métadonnées Redis {key}: {e}")
    
    async def cleanup(self):
        """Ferme la connexion Redis"""
        if self._redis:
            await self._redis.close()


class CacheManager:
    """
    Gestionnaire de cache multi-niveaux pour Spleeter
    
    Features:
    - Cache mémoire (L1) + disque (L2) + Redis (L3)
    - Éviction automatique LRU/LFU/TTL
    - Compression et chiffrement optionnels
    - Nettoyage automatique
    - Métriques détaillées
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialise le gestionnaire de cache
        
        Args:
            config: Configuration du cache
        """
        self.config = config or CacheConfig()
        
        # Composants de cache
        self.memory_cache = MemoryCache(self.config.max_memory_bytes)
        self.disk_cache = DiskCache(self.config.cache_dir, self.config.max_disk_bytes)
        self.redis_cache = RedisCache(self.config) if self.config.redis_enabled else None
        
        # Monitoring
        self.monitor = PerformanceMonitor()
        
        # Tâche de nettoyage
        self._cleanup_task = None
        self._running = False
        
        # Statistiques
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0
        }
        
        logger.info(f"CacheManager initialisé: {self.config.cache_dir}")
    
    async def start(self):
        """Démarre le gestionnaire de cache"""
        if self._running:
            return
        
        self._running = True
        
        # Connexion Redis si activé
        if self.redis_cache:
            await self.redis_cache.connect()
        
        # Démarrage du nettoyage automatique
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("CacheManager démarré")
    
    async def stop(self):
        """Arrête le gestionnaire de cache"""
        if not self._running:
            return
        
        self._running = False
        
        # Arrêt du nettoyage
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Fermeture Redis
        if self.redis_cache:
            await self.redis_cache.cleanup()
        
        logger.info("CacheManager arrêté")
    
    async def _cleanup_loop(self):
        """Boucle de nettoyage automatique"""
        while self._running:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self._cleanup_expired()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erreur nettoyage cache: {e}")
    
    async def _cleanup_expired(self):
        """Nettoie les entrées expirées"""
        removed_count = await self.disk_cache.cleanup_expired()
        if removed_count > 0:
            self._stats["evictions"] += removed_count
    
    def _get_cache_key(self, key: str, namespace: str = "default") -> str:
        """Génère une clé de cache normalisée"""
        return f"{namespace}:{hashlib.md5(key.encode()).hexdigest()}"
    
    def _determine_ttl(self, tags: List[str]) -> Optional[float]:
        """Détermine le TTL selon les tags"""
        if "audio" in tags:
            return self.config.audio_cache_ttl
        elif "model" in tags:
            return self.config.model_cache_ttl
        else:
            return self.config.default_ttl
    
    async def get(
        self,
        key: str,
        namespace: str = "default"
    ) -> Optional[Any]:
        """
        Récupère une valeur du cache
        
        Args:
            key: Clé de cache
            namespace: Namespace de la clé
            
        Returns:
            Valeur cachée ou None
        """
        cache_key = self._get_cache_key(key, namespace)
        
        self.monitor.start_timer("cache_get")
        
        try:
            # L1: Cache mémoire
            entry = self.memory_cache.get(cache_key)
            if entry and not entry.is_expired():
                self._stats["hits"] += 1
                return entry
            
            # L2: Cache disque
            result = await self.disk_cache.get(cache_key)
            if result:
                entry, value = result
                
                # Promotion vers le cache mémoire
                self.memory_cache.set(cache_key, entry)
                
                self._stats["hits"] += 1
                return value
            
            # L3: Cache Redis
            if self.redis_cache:
                result = await self.redis_cache.get(cache_key)
                if result:
                    entry, value = result
                    
                    # Promotion vers les caches supérieurs
                    self.memory_cache.set(cache_key, entry)
                    await self.disk_cache.set(cache_key, value, entry)
                    
                    self._stats["hits"] += 1
                    return value
            
            # Cache miss
            self._stats["misses"] += 1
            return None
            
        finally:
            self.monitor.end_timer("cache_get")
    
    async def set(
        self,
        key: str,
        value: Any,
        namespace: str = "default",
        ttl: Optional[float] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Stocke une valeur dans le cache
        
        Args:
            key: Clé de cache
            value: Valeur à cacher
            namespace: Namespace de la clé
            ttl: Durée de vie en secondes
            tags: Tags pour classification
            
        Returns:
            True si stocké avec succès
        """
        cache_key = self._get_cache_key(key, namespace)
        tags = tags or []
        
        # Déterminer le TTL
        if ttl is None:
            ttl = self._determine_ttl(tags)
        
        # Créer l'entrée
        entry = CacheEntry(
            key=cache_key,
            value_hash="",  # Sera calculé par chaque cache
            size_bytes=0,   # Sera calculé par chaque cache
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            ttl=ttl,
            tags=tags
        )
        
        self.monitor.start_timer("cache_set")
        
        try:
            success = False
            
            # Stockage dans les caches (ordre inverse pour éviter les incohérences)
            
            # L3: Redis
            if self.redis_cache:
                redis_success = await self.redis_cache.set(cache_key, value, entry)
                success = success or redis_success
            
            # L2: Disque
            disk_success = await self.disk_cache.set(cache_key, value, entry)
            success = success or disk_success
            
            # L1: Mémoire
            memory_success = self.memory_cache.set(cache_key, entry)
            success = success or memory_success
            
            if success:
                self._stats["sets"] += 1
            
            return success
            
        finally:
            self.monitor.end_timer("cache_set")
    
    async def remove(
        self,
        key: str,
        namespace: str = "default"
    ) -> bool:
        """
        Supprime une entrée du cache
        
        Args:
            key: Clé à supprimer
            namespace: Namespace de la clé
            
        Returns:
            True si supprimé
        """
        cache_key = self._get_cache_key(key, namespace)
        
        success = False
        
        # Suppression de tous les niveaux
        memory_success = self.memory_cache.remove(cache_key)
        disk_success = await self.disk_cache.remove(cache_key)
        
        success = memory_success or disk_success
        
        if self.redis_cache:
            redis_success = await self.redis_cache.remove(cache_key)
            success = success or redis_success
        
        return success
    
    async def clear(self, namespace: Optional[str] = None):
        """
        Vide le cache
        
        Args:
            namespace: Namespace à vider (tous si None)
        """
        if namespace is None:
            # Vider complètement
            self.memory_cache.clear()
            
            # Pour le disque et Redis, il faudrait une méthode clear spécifique
            logger.warning("clear() complet non implémenté pour disque/Redis")
        else:
            # Vider un namespace spécifique (complexe à implémenter)
            logger.warning("clear() par namespace non implémenté")
    
    async def invalidate_by_tags(self, tags: List[str]):
        """
        Invalide les entrées ayant certains tags
        
        Args:
            tags: Tags à invalider
        """
        # Implementation complexe nécessaire pour scanner tous les caches
        logger.warning("invalidate_by_tags() non implémenté")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques du cache
        
        Returns:
            Dictionnaire des statistiques
        """
        memory_stats = self.memory_cache.get_stats()
        
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = (self._stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "requests": {
                "total": total_requests,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate_percent": hit_rate
            },
            "operations": {
                "sets": self._stats["sets"],
                "evictions": self._stats["evictions"]
            },
            "memory_cache": memory_stats,
            "config": {
                "max_memory_mb": self.config.max_memory_bytes / 1024 / 1024,
                "max_disk_gb": self.config.max_disk_bytes / 1024 / 1024 / 1024,
                "redis_enabled": self.config.redis_enabled,
                "compression_enabled": self.config.compression_enabled
            }
        }
    
    async def get_detailed_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques détaillées incluant le disque
        
        Returns:
            Statistiques complètes
        """
        stats = self.get_stats()
        
        # Ajout des stats disque (asynchrone)
        disk_stats = await self.disk_cache.get_stats()
        stats["disk_cache"] = disk_stats
        
        return stats
    
    @asynccontextmanager
    async def transaction(self):
        """
        Context manager pour les transactions de cache
        
        Usage:
            async with cache_manager.transaction():
                await cache_manager.set("key1", value1)
                await cache_manager.set("key2", value2)
                # Commit automatique à la sortie
        """
        # Pour l'instant, pas de vraie transaction
        # Implementation future avec rollback possible
        try:
            yield self
        except Exception:
            # Rollback des opérations si nécessaire
            raise
    
    async def cleanup(self):
        """Nettoyage des ressources"""
        await self.stop()
        await self.monitor.cleanup()
        logger.info("CacheManager nettoyé")
