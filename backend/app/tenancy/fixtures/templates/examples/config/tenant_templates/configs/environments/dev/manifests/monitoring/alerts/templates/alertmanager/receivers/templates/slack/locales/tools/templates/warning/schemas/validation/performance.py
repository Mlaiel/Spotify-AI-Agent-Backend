"""
Validateurs de performance optimisés - Spotify AI Agent
Validation haute performance pour environnements critiques
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, List, Optional, Set, Union
import weakref

from pydantic import ValidationError
from pydantic.validators import str_validator

from . import ValidationRules


class PerformanceValidationCache:
    """Cache optimisé pour les validations fréquentes"""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, tuple] = {}  # key -> (value, timestamp)
        self._access_order: List[str] = []
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Récupère une valeur du cache"""
        async with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                
                # Vérification TTL
                if time.time() - timestamp <= self.ttl_seconds:
                    # Mise à jour de l'ordre d'accès (LRU)
                    if key in self._access_order:
                        self._access_order.remove(key)
                    self._access_order.append(key)
                    return value
                else:
                    # Valeur expirée
                    del self._cache[key]
                    if key in self._access_order:
                        self._access_order.remove(key)
            
            return None
    
    async def set(self, key: str, value: Any) -> None:
        """Stocke une valeur dans le cache"""
        async with self._lock:
            # Nettoyage si cache plein
            if len(self._cache) >= self.max_size:
                # Suppression du plus ancien (LRU)
                if self._access_order:
                    oldest_key = self._access_order.pop(0)
                    if oldest_key in self._cache:
                        del self._cache[oldest_key]
            
            # Ajout de la nouvelle valeur
            self._cache[key] = (value, time.time())
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
    
    async def clear(self) -> None:
        """Vide le cache"""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
    
    def stats(self) -> Dict[str, int]:
        """Statistiques du cache"""
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds
        }


# Cache global pour les validations
_validation_cache = PerformanceValidationCache()


def cached_validation(cache_key_func: Optional[Callable] = None, ttl: int = 300):
    """Décorateur pour cache de validation"""
    
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Génération de la clé de cache
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Tentative de récupération du cache
            cached_result = await _validation_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Exécution et mise en cache
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            await _validation_cache.set(cache_key, result)
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Version synchrone simplifiée
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


class BatchValidationProcessor:
    """Processeur de validation par batch pour optimiser les performances"""
    
    def __init__(self, max_workers: int = 4, batch_size: int = 100):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def validate_batch(self, items: List[Any], validator_func: Callable,
                           **validator_kwargs) -> List[Union[Any, Exception]]:
        """Valide un batch d'éléments en parallèle"""
        
        if not items:
            return []
        
        # Division en chunks pour traitement parallèle
        chunks = [items[i:i + self.batch_size] for i in range(0, len(items), self.batch_size)]
        
        # Traitement parallèle des chunks
        loop = asyncio.get_event_loop()
        tasks = []
        
        for chunk in chunks:
            task = loop.run_in_executor(
                self._executor,
                self._validate_chunk,
                chunk,
                validator_func,
                validator_kwargs
            )
            tasks.append(task)
        
        # Attente des résultats
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aplatissement des résultats
        results = []
        for chunk_result in chunk_results:
            if isinstance(chunk_result, Exception):
                # Propagation de l'erreur pour tous les éléments du chunk
                results.extend([chunk_result] * self.batch_size)
            else:
                results.extend(chunk_result)
        
        return results[:len(items)]  # Ajustement à la taille originale
    
    def _validate_chunk(self, chunk: List[Any], validator_func: Callable,
                       validator_kwargs: Dict[str, Any]) -> List[Union[Any, Exception]]:
        """Valide un chunk d'éléments"""
        results = []
        
        for item in chunk:
            try:
                validated_item = validator_func(item, **validator_kwargs)
                results.append(validated_item)
            except Exception as e:
                results.append(e)
        
        return results
    
    def close(self):
        """Ferme l'executor"""
        self._executor.shutdown(wait=True)


class OptimizedValidationRules(ValidationRules):
    """Version optimisée des règles de validation"""
    
    # Cache pour les patterns compilés
    _compiled_patterns = weakref.WeakValueDictionary()
    
    @classmethod
    @lru_cache(maxsize=1000)
    def _get_cached_pattern(cls, pattern_name: str):
        """Récupère un pattern compilé avec cache"""
        if pattern_name not in cls._compiled_patterns:
            pattern = getattr(cls, f"{pattern_name}_PATTERN", None)
            if pattern:
                cls._compiled_patterns[pattern_name] = pattern
        return cls._compiled_patterns.get(pattern_name)
    
    @classmethod
    @cached_validation(lambda cls, value: f"tenant_id:{hash(value)}")
    def validate_tenant_id_fast(cls, value: str) -> str:
        """Version optimisée de la validation tenant_id"""
        if not value:
            raise ValueError("Tenant ID cannot be empty")
        
        # Validation rapide de longueur avant regex
        if len(value) > 255:
            raise ValueError("Tenant ID cannot exceed 255 characters")
        
        value = value.strip().lower()
        
        # Utilisation du pattern mis en cache
        pattern = cls._get_cached_pattern("TENANT_ID")
        if pattern and not pattern.match(value):
            raise ValueError("Invalid tenant ID format")
        
        return value
    
    @classmethod
    @cached_validation(lambda cls, message: f"alert_msg:{hash(message)[:16]}")
    def validate_alert_message_fast(cls, message: str) -> str:
        """Version optimisée de la validation message d'alerte"""
        if not message:
            raise ValueError("Alert message cannot be empty")
        
        # Validation rapide de longueur
        if len(message) > 10000:  # Limite raisonnable
            raise ValueError("Alert message too long")
        
        message = message.strip()
        
        # Validation basique sans regex complexe
        if '<script' in message.lower() or 'javascript:' in message.lower():
            raise ValueError("Alert message contains dangerous content")
        
        return message
    
    @classmethod
    def validate_batch_tenant_ids(cls, tenant_ids: List[str]) -> List[Union[str, Exception]]:
        """Validation par batch des tenant IDs"""
        processor = BatchValidationProcessor()
        
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                processor.validate_batch(tenant_ids, cls.validate_tenant_id_fast)
            )
        finally:
            processor.close()


class MemoryEfficientValidator:
    """Validateur optimisé mémoire pour gros volumes"""
    
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
        self._stats = {
            'processed': 0,
            'errors': 0,
            'memory_peak': 0
        }
    
    async def validate_stream(self, data_stream, validator_func: Callable):
        """Valide un flux de données par chunks"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        chunk = []
        
        async for item in data_stream:
            chunk.append(item)
            
            if len(chunk) >= self.chunk_size:
                await self._process_chunk(chunk, validator_func)
                chunk.clear()
                
                # Monitoring mémoire
                memory_usage = process.memory_info().rss
                self._stats['memory_peak'] = max(self._stats['memory_peak'], memory_usage)
        
        # Traitement du dernier chunk
        if chunk:
            await self._process_chunk(chunk, validator_func)
    
    async def _process_chunk(self, chunk: List[Any], validator_func: Callable):
        """Traite un chunk de données"""
        for item in chunk:
            try:
                validator_func(item)
                self._stats['processed'] += 1
            except Exception:
                self._stats['errors'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de traitement"""
        return {
            **self._stats,
            'error_rate': self._stats['errors'] / max(1, self._stats['processed']),
            'memory_peak_mb': self._stats['memory_peak'] / (1024 * 1024)
        }


class ValidatorProfiler:
    """Profileur de performance pour les validateurs"""
    
    def __init__(self):
        self._timings: Dict[str, List[float]] = {}
        self._call_counts: Dict[str, int] = {}
    
    def profile(self, name: str):
        """Décorateur de profiling"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    duration = end_time - start_time
                    
                    # Enregistrement des métriques
                    if name not in self._timings:
                        self._timings[name] = []
                        self._call_counts[name] = 0
                    
                    self._timings[name].append(duration)
                    self._call_counts[name] += 1
                    
                    # Limitation de l'historique
                    if len(self._timings[name]) > 1000:
                        self._timings[name] = self._timings[name][-500:]
            
            return wrapper
        return decorator
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Retourne les statistiques de performance"""
        stats = {}
        
        for name, timings in self._timings.items():
            if timings:
                stats[name] = {
                    'count': self._call_counts[name],
                    'total_time': sum(timings),
                    'avg_time': sum(timings) / len(timings),
                    'min_time': min(timings),
                    'max_time': max(timings),
                    'p95_time': sorted(timings)[int(len(timings) * 0.95)] if timings else 0
                }
        
        return stats
    
    def reset(self):
        """Remet à zéro les statistiques"""
        self._timings.clear()
        self._call_counts.clear()


# Instance globale du profileur
_validator_profiler = ValidatorProfiler()


def profile_validation(name: str):
    """Décorateur pour profiler une validation"""
    return _validator_profiler.profile(name)


class AdaptiveValidator:
    """Validateur adaptatif qui ajuste ses paramètres selon la charge"""
    
    def __init__(self):
        self.load_factor = 0.0
        self.error_rate = 0.0
        self.performance_mode = "normal"  # normal, fast, strict
        
        self._recent_times: List[float] = []
        self._recent_errors: List[bool] = []
        self._window_size = 100
    
    def validate_adaptive(self, value: Any, validator_func: Callable, **kwargs) -> Any:
        """Validation adaptative selon les conditions actuelles"""
        start_time = time.perf_counter()
        
        try:
            # Ajustement des paramètres selon le mode
            if self.performance_mode == "fast":
                # Mode rapide: validations simplifiées
                kwargs['strict'] = False
                kwargs['cache_enabled'] = True
            elif self.performance_mode == "strict":
                # Mode strict: validations complètes
                kwargs['strict'] = True
                kwargs['cache_enabled'] = False
            
            result = validator_func(value, **kwargs)
            
            # Enregistrement du succès
            self._record_result(time.perf_counter() - start_time, False)
            return result
            
        except Exception as e:
            # Enregistrement de l'erreur
            self._record_result(time.perf_counter() - start_time, True)
            raise e
    
    def _record_result(self, duration: float, is_error: bool):
        """Enregistre un résultat de validation"""
        self._recent_times.append(duration)
        self._recent_errors.append(is_error)
        
        # Limitation de la taille de la fenêtre
        if len(self._recent_times) > self._window_size:
            self._recent_times.pop(0)
            self._recent_errors.pop(0)
        
        # Mise à jour des métriques
        self._update_metrics()
    
    def _update_metrics(self):
        """Met à jour les métriques et ajuste le mode"""
        if not self._recent_times:
            return
        
        # Calcul du facteur de charge (temps moyen)
        avg_time = sum(self._recent_times) / len(self._recent_times)
        self.load_factor = min(1.0, avg_time * 1000)  # Normalisation
        
        # Calcul du taux d'erreur
        self.error_rate = sum(self._recent_errors) / len(self._recent_errors)
        
        # Ajustement du mode de performance
        if self.load_factor > 0.8:
            self.performance_mode = "fast"
        elif self.error_rate > 0.1:
            self.performance_mode = "strict"
        else:
            self.performance_mode = "normal"
    
    def get_status(self) -> Dict[str, Any]:
        """Retourne le statut actuel du validateur"""
        return {
            'performance_mode': self.performance_mode,
            'load_factor': self.load_factor,
            'error_rate': self.error_rate,
            'avg_time_ms': sum(self._recent_times) / len(self._recent_times) * 1000 if self._recent_times else 0
        }


# Instances globales pour utilisation simplifiée
adaptive_validator = AdaptiveValidator()
batch_processor = BatchValidationProcessor()


def get_performance_stats() -> Dict[str, Any]:
    """Retourne les statistiques de performance globales"""
    return {
        'cache_stats': _validation_cache.stats(),
        'profiler_stats': _validator_profiler.get_stats(),
        'adaptive_validator_status': adaptive_validator.get_status()
    }


async def clear_performance_cache():
    """Vide le cache de performance"""
    await _validation_cache.clear()


def reset_performance_stats():
    """Remet à zéro les statistiques de performance"""
    _validator_profiler.reset()
    adaptive_validator.__init__()
