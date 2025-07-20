"""
Optimiseur de Locales Avancé pour Spotify AI Agent
Système d'optimisation et de cache intelligent des locales
"""

import asyncio
import json
import logging
import zlib
import pickle
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref
import gc
from pathlib import Path

from redis.asyncio import Redis
import aiofiles

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration de l'optimiseur"""
    cache_size: int = 1000
    cache_ttl: int = 3600
    compression_enabled: bool = True
    compression_level: int = 6
    deduplication_enabled: bool = True
    memory_threshold: float = 0.8  # 80% de la mémoire
    cleanup_interval: int = 300  # 5 minutes
    preload_popular: bool = True
    popularity_threshold: int = 10
    batch_optimization: bool = True
    lazy_loading: bool = True
    adaptive_caching: bool = True


@dataclass
class CacheEntry:
    """Entrée de cache optimisée"""
    data: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    compressed: bool = False
    tenant_id: Optional[str] = None
    locale_code: Optional[str] = None
    version: str = "1.0"
    
    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.last_accessed, str):
            self.last_accessed = datetime.fromisoformat(self.last_accessed)


class LocaleOptimizer:
    """Optimiseur principal des locales"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self._cache = OrderedDict()
        self._access_stats = defaultdict(int)
        self._size_stats = defaultdict(int)
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._cleanup_task = None
        self._running = False
        self._memory_usage = 0
        self._deduplication_map = {}
        
        # Métriques de performance
        self._metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'compressions': 0,
            'decompressions': 0,
            'cleanups': 0,
            'memory_optimizations': 0
        }
    
    async def start(self):
        """Démarre l'optimiseur"""
        if not self._running:
            self._running = True
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Locale optimizer started")
    
    async def stop(self):
        """Arrête l'optimiseur"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        self._executor.shutdown(wait=True)
        logger.info("Locale optimizer stopped")
    
    async def get_optimized_data(
        self, 
        key: str, 
        loader_func: callable,
        tenant_id: Optional[str] = None,
        locale_code: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Récupère des données optimisées avec cache intelligent"""
        try:
            # Vérifier le cache
            cached_entry = await self._get_from_cache(key)
            if cached_entry:
                self._metrics['cache_hits'] += 1
                await self._update_access_stats(key, cached_entry)
                return cached_entry.data
            
            self._metrics['cache_misses'] += 1
            
            # Charger les données
            data = await loader_func()
            if data is None:
                return None
            
            # Optimiser et mettre en cache
            await self._store_optimized(key, data, tenant_id, locale_code)
            
            return data
            
        except Exception as e:
            logger.error(f"Optimization error for key {key}: {e}")
            return None
    
    async def optimize_data(self, data: Dict[str, Any]) -> Tuple[bytes, bool]:
        """Optimise les données (compression, déduplication)"""
        try:
            # Sérialisation
            serialized = pickle.dumps(data)
            original_size = len(serialized)
            
            compressed = False
            if self.config.compression_enabled and original_size > 1024:  # > 1KB
                compressed_data = zlib.compress(
                    serialized, 
                    level=self.config.compression_level
                )
                
                # Garder seulement si la compression est efficace
                if len(compressed_data) < original_size * 0.8:
                    serialized = compressed_data
                    compressed = True
                    self._metrics['compressions'] += 1
            
            return serialized, compressed
            
        except Exception as e:
            logger.error(f"Data optimization error: {e}")
            return pickle.dumps(data), False
    
    async def deoptimize_data(self, data: bytes, compressed: bool) -> Dict[str, Any]:
        """Déoptimise les données"""
        try:
            if compressed:
                data = zlib.decompress(data)
                self._metrics['decompressions'] += 1
            
            return pickle.loads(data)
            
        except Exception as e:
            logger.error(f"Data deoptimization error: {e}")
            return {}
    
    async def invalidate_cache(self, key: str):
        """Invalide une entrée du cache"""
        with self._lock:
            if key in self._cache:
                entry = self._cache.pop(key)
                self._memory_usage -= entry.size_bytes
                logger.debug(f"Cache entry invalidated: {key}")
    
    async def clear_cache(self):
        """Vide complètement le cache"""
        with self._lock:
            self._cache.clear()
            self._memory_usage = 0
            self._access_stats.clear()
            self._size_stats.clear()
            logger.info("Cache cleared")
    
    async def get_optimization_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques d'optimisation"""
        with self._lock:
            cache_efficiency = 0
            total_requests = self._metrics['cache_hits'] + self._metrics['cache_misses']
            if total_requests > 0:
                cache_efficiency = self._metrics['cache_hits'] / total_requests
            
            return {
                'cache_size': len(self._cache),
                'memory_usage_bytes': self._memory_usage,
                'cache_efficiency': cache_efficiency,
                'metrics': dict(self._metrics),
                'top_accessed': dict(sorted(
                    self._access_stats.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10]),
                'config': {
                    'cache_size': self.config.cache_size,
                    'cache_ttl': self.config.cache_ttl,
                    'compression_enabled': self.config.compression_enabled,
                    'deduplication_enabled': self.config.deduplication_enabled
                }
            }
    
    async def optimize_for_tenant(self, tenant_id: str) -> Dict[str, Any]:
        """Optimise spécifiquement pour un tenant"""
        try:
            # Collecter les données du tenant
            tenant_entries = {}
            with self._lock:
                for key, entry in self._cache.items():
                    if entry.tenant_id == tenant_id:
                        tenant_entries[key] = entry
            
            # Analyser les patterns d'utilisation
            analysis = await self._analyze_tenant_usage(tenant_id, tenant_entries)
            
            # Appliquer les optimisations
            optimizations = await self._apply_tenant_optimizations(tenant_id, analysis)
            
            return {
                'tenant_id': tenant_id,
                'entries_analyzed': len(tenant_entries),
                'optimizations_applied': optimizations,
                'analysis': analysis
            }
            
        except Exception as e:
            logger.error(f"Tenant optimization error for {tenant_id}: {e}")
            return {}
    
    async def _get_from_cache(self, key: str) -> Optional[CacheEntry]:
        """Récupère une entrée du cache"""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # Vérifier l'expiration
                if datetime.now() - entry.created_at > timedelta(seconds=self.config.cache_ttl):
                    self._cache.pop(key)
                    self._memory_usage -= entry.size_bytes
                    return None
                
                # Mettre à jour l'ordre (LRU)
                self._cache.move_to_end(key)
                
                return entry
            
            return None
    
    async def _store_optimized(
        self, 
        key: str, 
        data: Dict[str, Any],
        tenant_id: Optional[str] = None,
        locale_code: Optional[str] = None
    ):
        """Stocke des données optimisées dans le cache"""
        try:
            # Optimiser les données
            optimized_data, compressed = await self.optimize_data(data)
            size_bytes = len(optimized_data)
            
            # Créer l'entrée
            entry = CacheEntry(
                data=data,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                size_bytes=size_bytes,
                compressed=compressed,
                tenant_id=tenant_id,
                locale_code=locale_code
            )
            
            with self._lock:
                # Vérifier la capacité
                while (len(self._cache) >= self.config.cache_size or 
                       self._memory_usage + size_bytes > self._get_memory_limit()):
                    await self._evict_lru()
                
                self._cache[key] = entry
                self._memory_usage += size_bytes
                
                logger.debug(f"Cached optimized data for key: {key}")
            
        except Exception as e:
            logger.error(f"Error storing optimized data: {e}")
    
    async def _update_access_stats(self, key: str, entry: CacheEntry):
        """Met à jour les statistiques d'accès"""
        with self._lock:
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            self._access_stats[key] += 1
    
    async def _evict_lru(self):
        """Évince l'entrée la moins récemment utilisée"""
        if self._cache:
            key, entry = self._cache.popitem(last=False)
            self._memory_usage -= entry.size_bytes
            logger.debug(f"Evicted LRU entry: {key}")
    
    async def _cleanup_loop(self):
        """Boucle de nettoyage automatique"""
        while self._running:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self._cleanup_cache()
                
                if self.config.adaptive_caching:
                    await self._adaptive_optimization()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def _cleanup_cache(self):
        """Nettoie le cache"""
        try:
            expired_keys = []
            now = datetime.now()
            
            with self._lock:
                for key, entry in self._cache.items():
                    if now - entry.created_at > timedelta(seconds=self.config.cache_ttl):
                        expired_keys.append(key)
                
                for key in expired_keys:
                    entry = self._cache.pop(key)
                    self._memory_usage -= entry.size_bytes
            
            if expired_keys:
                self._metrics['cleanups'] += 1
                logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
            
            # Garbage collection si nécessaire
            if self._memory_usage > self._get_memory_limit() * 0.9:
                gc.collect()
                self._metrics['memory_optimizations'] += 1
            
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
    
    async def _adaptive_optimization(self):
        """Optimisation adaptative basée sur les patterns d'usage"""
        try:
            # Analyser les patterns d'accès
            popular_keys = sorted(
                self._access_stats.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            # Précharger les données populaires
            if self.config.preload_popular:
                for key, count in popular_keys:
                    if count >= self.config.popularity_threshold:
                        if key not in self._cache:
                            logger.debug(f"Would preload popular key: {key}")
            
            # Ajuster la compression selon l'utilisation
            if self._metrics['compressions'] > 0:
                compression_ratio = self._metrics['compressions'] / (
                    self._metrics['compressions'] + self._metrics['cache_misses']
                )
                
                if compression_ratio < 0.1:  # Peu utilisée
                    logger.debug("Consider disabling compression for better performance")
            
        except Exception as e:
            logger.error(f"Adaptive optimization error: {e}")
    
    async def _analyze_tenant_usage(
        self, 
        tenant_id: str, 
        entries: Dict[str, CacheEntry]
    ) -> Dict[str, Any]:
        """Analyse les patterns d'usage d'un tenant"""
        try:
            total_size = sum(entry.size_bytes for entry in entries.values())
            total_accesses = sum(entry.access_count for entry in entries.values())
            
            # Localisation la plus utilisée
            locale_usage = defaultdict(int)
            for entry in entries.values():
                if entry.locale_code:
                    locale_usage[entry.locale_code] += entry.access_count
            
            most_used_locale = max(locale_usage.items(), key=lambda x: x[1])[0] if locale_usage else None
            
            return {
                'total_entries': len(entries),
                'total_size_bytes': total_size,
                'total_accesses': total_accesses,
                'average_size': total_size / len(entries) if entries else 0,
                'most_used_locale': most_used_locale,
                'locale_distribution': dict(locale_usage)
            }
            
        except Exception as e:
            logger.error(f"Usage analysis error: {e}")
            return {}
    
    async def _apply_tenant_optimizations(
        self, 
        tenant_id: str, 
        analysis: Dict[str, Any]
    ) -> List[str]:
        """Applique les optimisations pour un tenant"""
        optimizations = []
        
        try:
            # Optimisation 1: Précharger la locale principale
            if analysis.get('most_used_locale'):
                optimizations.append(f"Preload {analysis['most_used_locale']} locale")
            
            # Optimisation 2: Compression sélective
            if analysis.get('average_size', 0) > 10240:  # > 10KB
                optimizations.append("Enable aggressive compression")
            
            # Optimisation 3: Cache étendu pour les tenants actifs
            if analysis.get('total_accesses', 0) > 100:
                optimizations.append("Extend cache TTL")
            
            return optimizations
            
        except Exception as e:
            logger.error(f"Optimization application error: {e}")
            return []
    
    def _get_memory_limit(self) -> int:
        """Calcule la limite mémoire dynamique"""
        # Implémentation basique - peut être améliorée avec psutil
        return 100 * 1024 * 1024  # 100MB par défaut


class CacheOptimizer:
    """Optimiseur de cache avec support Redis"""
    
    def __init__(
        self, 
        redis_client: Optional[Redis] = None,
        config: OptimizationConfig = None
    ):
        self.redis_client = redis_client
        self.config = config or OptimizationConfig()
        self._local_optimizer = LocaleOptimizer(self.config)
        self._distributed_cache = {}
        self._stats = defaultdict(int)
    
    async def start(self):
        """Démarre l'optimiseur de cache"""
        await self._local_optimizer.start()
        logger.info("Cache optimizer started")
    
    async def stop(self):
        """Arrête l'optimiseur de cache"""
        await self._local_optimizer.stop()
        logger.info("Cache optimizer stopped")
    
    async def get_cached_data(
        self, 
        key: str,
        loader_func: callable = None,
        use_distributed: bool = True
    ) -> Optional[Any]:
        """Récupère des données depuis le cache multi-niveaux"""
        try:
            # Niveau 1: Cache local
            local_data = await self._local_optimizer.get_optimized_data(
                key, 
                lambda: None  # Pas de loader pour le cache local
            )
            
            if local_data:
                self._stats['local_hits'] += 1
                return local_data
            
            # Niveau 2: Cache distribué (Redis)
            if use_distributed and self.redis_client:
                distributed_data = await self._get_from_redis(key)
                if distributed_data:
                    self._stats['distributed_hits'] += 1
                    # Stocker dans le cache local
                    await self._local_optimizer._store_optimized(key, distributed_data)
                    return distributed_data
            
            # Niveau 3: Charger depuis la source
            if loader_func:
                data = await loader_func()
                if data:
                    # Stocker dans tous les niveaux
                    await self._store_multilevel(key, data, use_distributed)
                    self._stats['source_loads'] += 1
                    return data
            
            self._stats['total_misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache retrieval error for {key}: {e}")
            return None
    
    async def invalidate_key(self, key: str):
        """Invalide une clé dans tous les niveaux de cache"""
        try:
            # Cache local
            await self._local_optimizer.invalidate_cache(key)
            
            # Cache distribué
            if self.redis_client:
                await self.redis_client.delete(f"locale_cache:{key}")
            
            logger.debug(f"Invalidated key across all cache levels: {key}")
            
        except Exception as e:
            logger.error(f"Cache invalidation error for {key}: {e}")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache"""
        local_stats = await self._local_optimizer.get_optimization_stats()
        
        return {
            'local_cache': local_stats,
            'distributed_stats': dict(self._stats),
            'total_efficiency': self._calculate_total_efficiency(),
            'redis_connected': self.redis_client is not None
        }
    
    async def optimize_cache_hierarchy(self) -> Dict[str, Any]:
        """Optimise la hiérarchie de cache"""
        try:
            results = {
                'local_optimization': await self._local_optimizer.get_optimization_stats(),
                'distributed_sync': await self._sync_distributed_cache(),
                'hierarchy_rebalance': await self._rebalance_cache_hierarchy()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Cache hierarchy optimization error: {e}")
            return {}
    
    async def _get_from_redis(self, key: str) -> Optional[Dict[str, Any]]:
        """Récupère depuis Redis"""
        try:
            if not self.redis_client:
                return None
            
            data = await self.redis_client.get(f"locale_cache:{key}")
            if data:
                return json.loads(data)
            
            return None
            
        except Exception as e:
            logger.error(f"Redis retrieval error: {e}")
            return None
    
    async def _store_multilevel(
        self, 
        key: str, 
        data: Dict[str, Any], 
        use_distributed: bool = True
    ):
        """Stocke dans tous les niveaux de cache"""
        try:
            # Cache local
            await self._local_optimizer._store_optimized(key, data)
            
            # Cache distribué
            if use_distributed and self.redis_client:
                await self.redis_client.setex(
                    f"locale_cache:{key}",
                    self.config.cache_ttl,
                    json.dumps(data, ensure_ascii=False)
                )
            
        except Exception as e:
            logger.error(f"Multi-level storage error: {e}")
    
    async def _sync_distributed_cache(self) -> Dict[str, Any]:
        """Synchronise le cache distribué"""
        try:
            if not self.redis_client:
                return {'status': 'redis_not_available'}
            
            # Obtenir les clés de cache
            keys = await self.redis_client.keys("locale_cache:*")
            
            sync_stats = {
                'keys_found': len(keys),
                'sync_time': datetime.now().isoformat(),
                'status': 'completed'
            }
            
            return sync_stats
            
        except Exception as e:
            logger.error(f"Distributed cache sync error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _rebalance_cache_hierarchy(self) -> Dict[str, Any]:
        """Rééquilibre la hiérarchie de cache"""
        try:
            # Analyser l'efficacité de chaque niveau
            local_efficiency = self._stats.get('local_hits', 0)
            distributed_efficiency = self._stats.get('distributed_hits', 0)
            
            total_requests = (
                self._stats.get('local_hits', 0) + 
                self._stats.get('distributed_hits', 0) + 
                self._stats.get('total_misses', 0)
            )
            
            if total_requests == 0:
                return {'status': 'no_data'}
            
            recommendations = []
            
            # Recommandations basées sur l'efficacité
            if local_efficiency / total_requests < 0.3:
                recommendations.append("Increase local cache size")
            
            if distributed_efficiency / total_requests > 0.5:
                recommendations.append("Consider warming local cache from distributed")
            
            return {
                'local_efficiency': local_efficiency / total_requests,
                'distributed_efficiency': distributed_efficiency / total_requests,
                'recommendations': recommendations,
                'status': 'analyzed'
            }
            
        except Exception as e:
            logger.error(f"Cache rebalancing error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _calculate_total_efficiency(self) -> float:
        """Calcule l'efficacité totale du cache"""
        try:
            total_hits = (
                self._stats.get('local_hits', 0) + 
                self._stats.get('distributed_hits', 0)
            )
            total_requests = total_hits + self._stats.get('total_misses', 0)
            
            if total_requests == 0:
                return 0.0
            
            return total_hits / total_requests
            
        except Exception as e:
            logger.error(f"Efficiency calculation error: {e}")
            return 0.0
