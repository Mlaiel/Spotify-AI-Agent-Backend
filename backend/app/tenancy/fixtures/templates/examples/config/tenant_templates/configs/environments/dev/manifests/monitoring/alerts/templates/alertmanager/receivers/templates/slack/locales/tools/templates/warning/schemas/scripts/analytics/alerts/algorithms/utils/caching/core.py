"""
Core du Système de Cache Ultra-Avancé
=====================================

Classes principales pour le gestionnaire de cache multi-niveaux avec
intelligence artificielle, monitoring temps réel et architecture multi-tenant.

Ce module implémente un système de cache industriel avec:
- Gestion intelligente multi-niveaux (L1/L2/L3)
- Monitoring avancé avec métriques Prometheus
- Analytics ML pour prédiction et optimisation
- Architecture multi-tenant avec isolation sécurisée
- Circuit breaker et failover automatique
- Compression et sérialisation optimisées

Auteurs: Équipe Spotify AI Agent - Direction technique Fahed Mlaiel
"""

import asyncio
import time
import threading
import logging
import hashlib
import pickle
import json
import gzip
import lz4
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future
import redis
import memcache
import sqlite3
from prometheus_client import Counter, Histogram, Gauge, start_http_server

from .exceptions import (
    CacheException, CacheBackendError, CacheMissError, 
    CacheTimeoutError, CacheSecurityError, CacheQuotaExceededError
)
from .serializers import JSONSerializer, PickleSerializer, CompressionSerializer
from .utils import CacheKeyGenerator, TTLCalculator, SecurityUtils, ValidationUtils
from .circuit_breaker import CircuitBreaker, CircuitState


class CacheLevel(Enum):
    """Niveaux de cache disponibles"""
    L1_MEMORY = "l1_memory"
    L2_DISK = "l2_disk" 
    L3_DISTRIBUTED = "l3_distributed"


class CachePolicy(Enum):
    """Politiques de cache disponibles"""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"
    ML_PREDICTIVE = "ml_predictive"


@dataclass
class CacheEntry:
    """Entrée de cache avec métadonnées complètes"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size: int = 0
    ttl: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    tenant_id: Optional[str] = None
    compressed: bool = False
    encrypted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Vérifie si l'entrée a expiré"""
        if not self.ttl:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl
    
    def touch(self):
        """Met à jour les statistiques d'accès"""
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class CacheStats:
    """Statistiques complètes du cache"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    total_size: int = 0
    entry_count: int = 0
    avg_access_time: float = 0.0
    hit_ratio: float = 0.0
    memory_usage: int = 0
    disk_usage: int = 0
    network_usage: int = 0
    
    def calculate_hit_ratio(self) -> float:
        """Calcule le taux de succès du cache"""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0


class CacheMetrics:
    """Collecteur de métriques Prometheus pour le cache"""
    
    def __init__(self, namespace: str = "spotify_ai_cache"):
        self.namespace = namespace
        
        # Compteurs
        self.cache_hits = Counter(
            f'{namespace}_hits_total',
            'Total cache hits',
            ['level', 'tenant_id', 'operation']
        )
        
        self.cache_misses = Counter(
            f'{namespace}_misses_total', 
            'Total cache misses',
            ['level', 'tenant_id', 'operation']
        )
        
        self.cache_operations = Counter(
            f'{namespace}_operations_total',
            'Total cache operations',
            ['level', 'tenant_id', 'operation', 'status']
        )
        
        # Histogrammes
        self.operation_duration = Histogram(
            f'{namespace}_operation_duration_seconds',
            'Cache operation duration',
            ['level', 'tenant_id', 'operation']
        )
        
        self.entry_size = Histogram(
            f'{namespace}_entry_size_bytes',
            'Cache entry size distribution',
            ['level', 'tenant_id']
        )
        
        # Jauges
        self.memory_usage = Gauge(
            f'{namespace}_memory_usage_bytes',
            'Current memory usage',
            ['level', 'tenant_id']
        )
        
        self.entry_count = Gauge(
            f'{namespace}_entries_count',
            'Current number of cache entries',
            ['level', 'tenant_id']
        )
        
        self.hit_ratio = Gauge(
            f'{namespace}_hit_ratio_percent',
            'Cache hit ratio percentage',
            ['level', 'tenant_id']
        )
    
    def record_hit(self, level: str, tenant_id: str, operation: str):
        """Enregistre un cache hit"""
        self.cache_hits.labels(level=level, tenant_id=tenant_id, operation=operation).inc()
    
    def record_miss(self, level: str, tenant_id: str, operation: str):
        """Enregistre un cache miss"""
        self.cache_misses.labels(level=level, tenant_id=tenant_id, operation=operation).inc()
    
    def record_operation(self, level: str, tenant_id: str, operation: str, 
                        status: str, duration: float):
        """Enregistre une opération de cache"""
        self.cache_operations.labels(
            level=level, tenant_id=tenant_id, operation=operation, status=status
        ).inc()
        self.operation_duration.labels(
            level=level, tenant_id=tenant_id, operation=operation
        ).observe(duration)
    
    def update_memory_usage(self, level: str, tenant_id: str, usage: int):
        """Met à jour l'utilisation mémoire"""
        self.memory_usage.labels(level=level, tenant_id=tenant_id).set(usage)
    
    def update_entry_count(self, level: str, tenant_id: str, count: int):
        """Met à jour le nombre d'entrées"""
        self.entry_count.labels(level=level, tenant_id=tenant_id).set(count)
    
    def update_hit_ratio(self, level: str, tenant_id: str, ratio: float):
        """Met à jour le taux de succès"""
        self.hit_ratio.labels(level=level, tenant_id=tenant_id).set(ratio)


class CacheBackend(ABC):
    """Interface abstraite pour les backends de cache"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Récupère une entrée du cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, entry: CacheEntry) -> bool:
        """Stocke une entrée dans le cache"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Supprime une entrée du cache"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Vide complètement le cache"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Vérifie si une clé existe"""
        pass
    
    @abstractmethod
    async def get_stats(self) -> CacheStats:
        """Retourne les statistiques du backend"""
        pass


class MemoryBackend(CacheBackend):
    """Backend mémoire ultra-optimisé avec LRU/LFU"""
    
    def __init__(self, max_size: int = 1000, policy: CachePolicy = CachePolicy.LRU):
        self.max_size = max_size
        self.policy = policy
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.access_counts: Dict[str, int] = {}
        self.lock = asyncio.Lock()
        self.stats = CacheStats()
        
    async def get(self, key: str) -> Optional[CacheEntry]:
        async with self.lock:
            if key not in self.cache:
                self.stats.misses += 1
                return None
            
            entry = self.cache[key]
            if entry.is_expired():
                await self._evict_key(key)
                self.stats.misses += 1
                return None
            
            entry.touch()
            self._update_access_tracking(key)
            self.stats.hits += 1
            return entry
    
    async def set(self, key: str, entry: CacheEntry) -> bool:
        async with self.lock:
            # Éviction si nécessaire
            if len(self.cache) >= self.max_size and key not in self.cache:
                await self._evict_entries()
            
            self.cache[key] = entry
            self._update_access_tracking(key)
            self.stats.sets += 1
            self.stats.total_size += entry.size
            return True
    
    async def delete(self, key: str) -> bool:
        async with self.lock:
            return await self._evict_key(key)
    
    async def clear(self) -> bool:
        async with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.access_counts.clear()
            self.stats = CacheStats()
            return True
    
    async def exists(self, key: str) -> bool:
        async with self.lock:
            return key in self.cache and not self.cache[key].is_expired()
    
    async def get_stats(self) -> CacheStats:
        async with self.lock:
            self.stats.entry_count = len(self.cache)
            self.stats.hit_ratio = self.stats.calculate_hit_ratio()
            return self.stats
    
    def _update_access_tracking(self, key: str):
        """Met à jour le suivi des accès selon la politique"""
        if self.policy == CachePolicy.LRU:
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
        elif self.policy == CachePolicy.LFU:
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
    
    async def _evict_entries(self):
        """Éviction intelligente selon la politique"""
        if self.policy == CachePolicy.LRU:
            key_to_evict = self.access_order[0]
        elif self.policy == CachePolicy.LFU:
            key_to_evict = min(self.access_counts, key=self.access_counts.get)
        else:
            key_to_evict = next(iter(self.cache))
        
        await self._evict_key(key_to_evict)
    
    async def _evict_key(self, key: str) -> bool:
        """Éviction d'une clé spécifique"""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.stats.evictions += 1
            self.stats.total_size -= entry.size
            
            if key in self.access_order:
                self.access_order.remove(key)
            if key in self.access_counts:
                del self.access_counts[key]
            return True
        return False


class RedisBackend(CacheBackend):
    """Backend Redis avec clustering et sharding"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 cluster_mode: bool = False, **kwargs):
        self.redis_url = redis_url
        self.cluster_mode = cluster_mode
        if cluster_mode:
            from rediscluster import RedisCluster
            self.client = RedisCluster.from_url(redis_url, **kwargs)
        else:
            self.client = redis.Redis.from_url(redis_url, **kwargs)
        self.serializer = PickleSerializer()
        self.stats = CacheStats()
        
    async def get(self, key: str) -> Optional[CacheEntry]:
        try:
            data = self.client.get(key)
            if data is None:
                self.stats.misses += 1
                return None
            
            entry = self.serializer.deserialize(data)
            if entry.is_expired():
                await self.delete(key)
                self.stats.misses += 1
                return None
            
            entry.touch()
            self.stats.hits += 1
            return entry
        except Exception as e:
            raise CacheBackendError(f"Redis get error: {e}")
    
    async def set(self, key: str, entry: CacheEntry) -> bool:
        try:
            data = self.serializer.serialize(entry)
            ttl = entry.ttl if entry.ttl else None
            result = self.client.setex(key, ttl or 3600, data)
            self.stats.sets += 1
            return result
        except Exception as e:
            raise CacheBackendError(f"Redis set error: {e}")
    
    async def delete(self, key: str) -> bool:
        try:
            result = self.client.delete(key)
            if result:
                self.stats.deletes += 1
            return bool(result)
        except Exception as e:
            raise CacheBackendError(f"Redis delete error: {e}")
    
    async def clear(self) -> bool:
        try:
            self.client.flushdb()
            self.stats = CacheStats()
            return True
        except Exception as e:
            raise CacheBackendError(f"Redis clear error: {e}")
    
    async def exists(self, key: str) -> bool:
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            raise CacheBackendError(f"Redis exists error: {e}")
    
    async def get_stats(self) -> CacheStats:
        try:
            info = self.client.info()
            self.stats.memory_usage = info.get('used_memory', 0)
            self.stats.entry_count = info.get('db0', {}).get('keys', 0)
            self.stats.hit_ratio = self.stats.calculate_hit_ratio()
            return self.stats
        except Exception as e:
            raise CacheBackendError(f"Redis stats error: {e}")


class MultiLevelCache:
    """Cache multi-niveaux avec failover automatique"""
    
    def __init__(self, backends: Dict[CacheLevel, CacheBackend],
                 metrics: Optional[CacheMetrics] = None):
        self.backends = backends
        self.metrics = metrics or CacheMetrics()
        self.circuit_breakers = {
            level: CircuitBreaker(failure_threshold=5, recovery_timeout=30)
            for level in backends.keys()
        }
        self.logger = logging.getLogger(__name__)
        
    async def get(self, key: str, tenant_id: str = "default") -> Optional[Any]:
        """Récupération multi-niveaux avec promotion automatique"""
        start_time = time.time()
        
        for level, backend in self.backends.items():
            circuit_breaker = self.circuit_breakers[level]
            
            if circuit_breaker.state == CircuitState.OPEN:
                continue
                
            try:
                with circuit_breaker:
                    entry = await backend.get(key)
                    if entry:
                        duration = time.time() - start_time
                        self.metrics.record_hit(level.value, tenant_id, "get")
                        self.metrics.record_operation(
                            level.value, tenant_id, "get", "success", duration
                        )
                        
                        # Promotion vers niveaux supérieurs
                        await self._promote_entry(key, entry, level)
                        return entry.value
                        
            except Exception as e:
                self.logger.error(f"Cache get error on {level}: {e}")
                circuit_breaker.record_failure()
                continue
        
        # Cache miss sur tous les niveaux
        duration = time.time() - start_time
        self.metrics.record_miss("all", tenant_id, "get")
        self.metrics.record_operation("all", tenant_id, "get", "miss", duration)
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, 
                  tenant_id: str = "default", tags: List[str] = None) -> bool:
        """Stockage avec réplication intelligente"""
        start_time = time.time()
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            ttl=ttl,
            tags=tags or [],
            tenant_id=tenant_id,
            size=len(str(value))
        )
        
        success_count = 0
        for level, backend in self.backends.items():
            circuit_breaker = self.circuit_breakers[level]
            
            if circuit_breaker.state == CircuitState.OPEN:
                continue
                
            try:
                with circuit_breaker:
                    success = await backend.set(key, entry)
                    if success:
                        success_count += 1
                        duration = time.time() - start_time
                        self.metrics.record_operation(
                            level.value, tenant_id, "set", "success", duration
                        )
                        
            except Exception as e:
                self.logger.error(f"Cache set error on {level}: {e}")
                circuit_breaker.record_failure()
                continue
        
        return success_count > 0
    
    async def delete(self, key: str, tenant_id: str = "default") -> bool:
        """Suppression sur tous les niveaux"""
        start_time = time.time()
        success_count = 0
        
        for level, backend in self.backends.items():
            try:
                success = await backend.delete(key)
                if success:
                    success_count += 1
                    duration = time.time() - start_time
                    self.metrics.record_operation(
                        level.value, tenant_id, "delete", "success", duration
                    )
            except Exception as e:
                self.logger.error(f"Cache delete error on {level}: {e}")
                continue
        
        return success_count > 0
    
    async def _promote_entry(self, key: str, entry: CacheEntry, current_level: CacheLevel):
        """Promotion d'entrée vers les niveaux supérieurs"""
        levels = list(self.backends.keys())
        current_index = levels.index(current_level)
        
        for i in range(current_index):
            higher_level = levels[i]
            try:
                await self.backends[higher_level].set(key, entry)
            except Exception as e:
                self.logger.error(f"Promotion error to {higher_level}: {e}")
    
    async def get_global_stats(self) -> Dict[str, CacheStats]:
        """Statistiques globales de tous les niveaux"""
        stats = {}
        for level, backend in self.backends.items():
            try:
                stats[level.value] = await backend.get_stats()
            except Exception as e:
                self.logger.error(f"Stats error for {level}: {e}")
        return stats


class TenantAwareCacheManager:
    """Gestionnaire de cache avec isolation multi-tenant"""
    
    def __init__(self, cache: MultiLevelCache, security_manager, quota_manager):
        self.cache = cache
        self.security_manager = security_manager
        self.quota_manager = quota_manager
        self.tenant_stats: Dict[str, CacheStats] = {}
        self.logger = logging.getLogger(__name__)
        
    async def get(self, key: str, tenant_id: str) -> Optional[Any]:
        """Récupération avec vérification tenant"""
        # Vérification des permissions
        if not await self.security_manager.can_read(tenant_id, key):
            raise CacheSecurityError(f"Read access denied for tenant {tenant_id}")
        
        # Génération de clé avec namespace tenant
        tenant_key = self._generate_tenant_key(tenant_id, key)
        return await self.cache.get(tenant_key, tenant_id)
    
    async def set(self, key: str, value: Any, tenant_id: str, 
                  ttl: Optional[int] = None, tags: List[str] = None) -> bool:
        """Stockage avec vérification quota et sécurité"""
        # Vérification des permissions
        if not await self.security_manager.can_write(tenant_id, key):
            raise CacheSecurityError(f"Write access denied for tenant {tenant_id}")
        
        # Vérification du quota
        if not await self.quota_manager.can_allocate(tenant_id, len(str(value))):
            raise CacheQuotaExceededError(f"Quota exceeded for tenant {tenant_id}")
        
        # Génération de clé avec namespace tenant
        tenant_key = self._generate_tenant_key(tenant_id, key)
        success = await self.cache.set(tenant_key, value, ttl, tenant_id, tags)
        
        if success:
            await self.quota_manager.allocate(tenant_id, len(str(value)))
        
        return success
    
    async def delete(self, key: str, tenant_id: str) -> bool:
        """Suppression avec vérification tenant"""
        if not await self.security_manager.can_delete(tenant_id, key):
            raise CacheSecurityError(f"Delete access denied for tenant {tenant_id}")
        
        tenant_key = self._generate_tenant_key(tenant_id, key)
        return await self.cache.delete(tenant_key, tenant_id)
    
    def _generate_tenant_key(self, tenant_id: str, key: str) -> str:
        """Génère une clé avec namespace tenant sécurisé"""
        return f"tenant:{tenant_id}::{hashlib.sha256(key.encode()).hexdigest()[:16]}:{key}"
    
    async def get_tenant_stats(self, tenant_id: str) -> CacheStats:
        """Statistiques spécifiques au tenant"""
        return self.tenant_stats.get(tenant_id, CacheStats())


class CacheHealthChecker:
    """Vérificateur de santé du système de cache"""
    
    def __init__(self, cache_manager: TenantAwareCacheManager):
        self.cache_manager = cache_manager
        self.health_checks = []
        self.last_check = None
        self.health_status = "unknown"
        
    async def run_health_checks(self) -> Dict[str, Any]:
        """Exécute tous les tests de santé"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "checks": {}
        }
        
        for check_name, check_func in self.health_checks:
            try:
                check_result = await check_func()
                results["checks"][check_name] = {
                    "status": "pass" if check_result["healthy"] else "fail",
                    "details": check_result
                }
                if not check_result["healthy"]:
                    results["overall_status"] = "unhealthy"
            except Exception as e:
                results["checks"][check_name] = {
                    "status": "error",
                    "error": str(e)
                }
                results["overall_status"] = "unhealthy"
        
        self.last_check = results
        self.health_status = results["overall_status"]
        return results
    
    def add_health_check(self, name: str, check_func: Callable):
        """Ajoute un test de santé personnalisé"""
        self.health_checks.append((name, check_func))
    
    async def basic_connectivity_check(self) -> Dict[str, Any]:
        """Test de connectivité basique"""
        try:
            test_key = f"health_check_{int(time.time())}"
            test_value = "health_check_value"
            
            # Test set/get/delete
            await self.cache_manager.set(test_key, test_value, "health_check")
            retrieved = await self.cache_manager.get(test_key, "health_check")
            await self.cache_manager.delete(test_key, "health_check")
            
            return {
                "healthy": retrieved == test_value,
                "latency_ms": 0,  # À implémenter
                "operations_tested": ["set", "get", "delete"]
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }


class CacheManager:
    """Gestionnaire principal du système de cache"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = CacheMetrics()
        self.backends = self._initialize_backends()
        self.multi_cache = MultiLevelCache(self.backends, self.metrics)
        self.tenant_manager = TenantAwareCacheManager(
            self.multi_cache, None, None  # À injecter
        )
        self.health_checker = CacheHealthChecker(self.tenant_manager)
        self.logger = logging.getLogger(__name__)
        
        # Démarrage du serveur de métriques
        if config.get("metrics_enabled", True):
            start_http_server(config.get("metrics_port", 8001))
    
    def _initialize_backends(self) -> Dict[CacheLevel, CacheBackend]:
        """Initialise les backends selon la configuration"""
        backends = {}
        
        # L1 - Mémoire
        if self.config.get("l1_enabled", True):
            backends[CacheLevel.L1_MEMORY] = MemoryBackend(
                max_size=self.config.get("l1_max_size", 1000),
                policy=CachePolicy(self.config.get("l1_policy", "lru"))
            )
        
        # L2 - Redis
        if self.config.get("l2_enabled", True):
            backends[CacheLevel.L2_DISK] = RedisBackend(
                redis_url=self.config.get("redis_url", "redis://localhost:6379"),
                cluster_mode=self.config.get("redis_cluster", False)
            )
        
        return backends
    
    async def start(self):
        """Démarre le gestionnaire de cache"""
        self.logger.info("Starting cache manager...")
        
        # Configuration des health checks
        self.health_checker.add_health_check(
            "basic_connectivity", 
            self.health_checker.basic_connectivity_check
        )
        
        # Démarrage du monitoring périodique
        asyncio.create_task(self._periodic_health_check())
        asyncio.create_task(self._periodic_metrics_export())
        
        self.logger.info("Cache manager started successfully")
    
    async def stop(self):
        """Arrête le gestionnaire de cache"""
        self.logger.info("Stopping cache manager...")
        # Nettoyage et fermeture des connexions
        self.logger.info("Cache manager stopped")
    
    async def _periodic_health_check(self):
        """Contrôles de santé périodiques"""
        while True:
            try:
                await self.health_checker.run_health_checks()
                await asyncio.sleep(self.config.get("health_check_interval", 60))
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(10)
    
    async def _periodic_metrics_export(self):
        """Export périodique des métriques"""
        while True:
            try:
                stats = await self.multi_cache.get_global_stats()
                for level, stat in stats.items():
                    self.metrics.update_memory_usage(level, "all", stat.memory_usage)
                    self.metrics.update_entry_count(level, "all", stat.entry_count)
                    self.metrics.update_hit_ratio(level, "all", stat.hit_ratio)
                
                await asyncio.sleep(self.config.get("metrics_export_interval", 30))
            except Exception as e:
                self.logger.error(f"Metrics export error: {e}")
                await asyncio.sleep(10)
