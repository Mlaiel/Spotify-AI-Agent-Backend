"""
🎵 Spotify AI Agent - Performance Utilities
===========================================

Utilitaires enterprise pour le monitoring des performances
avec métriques avancées, profiling et optimisation.

Architecture:
- Monitoring temps d'exécution
- Métriques de performance
- Profiling et benchmarking
- Cache et optimisation mémoire
- Limitation de débit (throttling)
- Détection de goulots d'étranglement

🎖️ Développé par l'équipe d'experts enterprise
"""

import time
import psutil
import functools
import threading
import asyncio
from typing import Dict, Any, Optional, Callable, List, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import cProfile
import pstats
import io
import gc
import tracemalloc
from contextlib import contextmanager


# =============================================================================
# MÉTRIQUES DE PERFORMANCE
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Classe pour stocker les métriques de performance"""
    
    execution_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    peak_memory: float = 0.0
    function_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: int = 0
    success_rate: float = 100.0
    timestamp: datetime = field(default_factory=datetime.now)


class PerformanceMonitor:
    """Moniteur de performance enterprise"""
    
    def __init__(self, max_history: int = 1000):
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.active_timers: Dict[str, float] = {}
        self.function_stats: Dict[str, PerformanceMetrics] = {}
        self._lock = threading.Lock()
    
    def start_timer(self, name: str) -> None:
        """
        Démarre un timer de performance
        
        Args:
            name: Nom du timer
        """
        with self._lock:
            self.active_timers[name] = time.perf_counter()
    
    def stop_timer(self, name: str) -> float:
        """
        Arrête un timer et retourne la durée
        
        Args:
            name: Nom du timer
            
        Returns:
            Durée d'exécution en secondes
        """
        with self._lock:
            if name not in self.active_timers:
                raise ValueError(f"Timer '{name}' n'existe pas")
            
            duration = time.perf_counter() - self.active_timers.pop(name)
            self.metrics_history[name].append(duration)
            return duration
    
    def record_metric(self, name: str, metrics: PerformanceMetrics) -> None:
        """
        Enregistre une métrique de performance
        
        Args:
            name: Nom de la métrique
            metrics: Données de performance
        """
        with self._lock:
            self.function_stats[name] = metrics
            self.metrics_history[f"{name}_execution_time"].append(metrics.execution_time)
            self.metrics_history[f"{name}_memory_usage"].append(metrics.memory_usage)
            self.metrics_history[f"{name}_cpu_usage"].append(metrics.cpu_usage)
    
    def get_stats(self, name: str) -> Dict[str, Any]:
        """
        Obtient les statistiques pour une métrique
        
        Args:
            name: Nom de la métrique
            
        Returns:
            Statistiques agrégées
        """
        with self._lock:
            if name not in self.metrics_history:
                return {}
            
            values = list(self.metrics_history[name])
            if not values:
                return {}
            
            return {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values),
                'latest': values[-1],
                'p95': self._percentile(values, 95),
                'p99': self._percentile(values, 99)
            }
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calcule un percentile"""
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * percentile / 100
        f = int(k)
        c = k - f
        if f == len(sorted_values) - 1:
            return sorted_values[f]
        return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c
    
    def clear_stats(self) -> None:
        """Efface toutes les statistiques"""
        with self._lock:
            self.metrics_history.clear()
            self.function_stats.clear()
            self.active_timers.clear()


# Instance globale du moniteur
performance_monitor = PerformanceMonitor()


# =============================================================================
# DÉCORATEURS DE PERFORMANCE
# =============================================================================

def monitor_performance(monitor: Optional[PerformanceMonitor] = None):
    """
    Décorateur pour monitorer les performances d'une fonction
    
    Args:
        monitor: Instance du moniteur (utilise l'instance globale si None)
    """
    if monitor is None:
        monitor = performance_monitor
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"
            
            # Métriques de début
            process = psutil.Process()
            start_time = time.perf_counter()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            start_cpu = process.cpu_percent()
            
            # Monitoring mémoire détaillé
            tracemalloc.start()
            
            try:
                # Exécution de la fonction
                result = func(*args, **kwargs)
                
                # Métriques de fin
                end_time = time.perf_counter()
                end_memory = process.memory_info().rss / 1024 / 1024  # MB
                end_cpu = process.cpu_percent()
                
                # Pic mémoire
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                # Création des métriques
                metrics = PerformanceMetrics(
                    execution_time=end_time - start_time,
                    memory_usage=end_memory - start_memory,
                    cpu_usage=max(start_cpu, end_cpu),
                    peak_memory=peak / 1024 / 1024,  # MB
                    function_calls=1,
                    errors=0,
                    success_rate=100.0
                )
                
                monitor.record_metric(func_name, metrics)
                return result
                
            except Exception as e:
                # Métriques d'erreur
                end_time = time.perf_counter()
                tracemalloc.stop()
                
                metrics = PerformanceMetrics(
                    execution_time=end_time - start_time,
                    memory_usage=0,
                    cpu_usage=0,
                    peak_memory=0,
                    function_calls=1,
                    errors=1,
                    success_rate=0.0
                )
                
                monitor.record_metric(func_name, metrics)
                raise
        
        return wrapper
    return decorator


def benchmark(iterations: int = 1000):
    """
    Décorateur pour benchmarker une fonction
    
    Args:
        iterations: Nombre d'itérations
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"
            
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                end = time.perf_counter()
                times.append(end - start)
            
            # Statistiques
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"Benchmark {func_name}:")
            print(f"  Iterations: {iterations}")
            print(f"  Average: {avg_time:.6f}s")
            print(f"  Min: {min_time:.6f}s")
            print(f"  Max: {max_time:.6f}s")
            print(f"  Total: {sum(times):.6f}s")
            
            return result
        
        return wrapper
    return decorator


def rate_limit(calls_per_second: float):
    """
    Décorateur pour limiter le débit d'appels
    
    Args:
        calls_per_second: Nombre d'appels autorisés par seconde
    """
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        
        return wrapper
    return decorator


# =============================================================================
# CACHE AVEC MÉTRIQUES
# =============================================================================

class PerformanceCache:
    """Cache avec métriques de performance intégrées"""
    
    def __init__(self, maxsize: int = 128, ttl: Optional[float] = None):
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache: Dict[str, Any] = {}
        self.timestamps: Dict[str, float] = {}
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Récupère une valeur du cache
        
        Args:
            key: Clé de cache
            
        Returns:
            Valeur ou None si absente/expirée
        """
        with self._lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            # Vérifier TTL
            if self.ttl and time.time() - self.timestamps[key] > self.ttl:
                del self.cache[key]
                del self.timestamps[key]
                self.misses += 1
                return None
            
            self.hits += 1
            return self.cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """
        Stocke une valeur dans le cache
        
        Args:
            key: Clé de cache
            value: Valeur à stocker
        """
        with self._lock:
            # Éviction si cache plein
            if len(self.cache) >= self.maxsize and key not in self.cache:
                # LRU simple - supprimer le plus ancien
                oldest_key = min(self.timestamps, key=self.timestamps.get)
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def clear(self) -> None:
        """Vide le cache"""
        with self._lock:
            self.cache.clear()
            self.timestamps.clear()
            self.hits = 0
            self.misses = 0
    
    def stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache"""
        with self._lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100) if total > 0 else 0
            
            return {
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'size': len(self.cache),
                'maxsize': self.maxsize
            }


def memoize(maxsize: int = 128, ttl: Optional[float] = None):
    """
    Décorateur de mémoïzation avec cache de performance
    
    Args:
        maxsize: Taille maximale du cache
        ttl: Time-to-live en secondes
    """
    def decorator(func: Callable) -> Callable:
        cache = PerformanceCache(maxsize, ttl)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Créer une clé de cache
            key = str(hash((args, tuple(sorted(kwargs.items())))))
            
            # Vérifier le cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Calculer et mettre en cache
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result
        
        # Ajouter des méthodes utilitaires
        wrapper.cache_info = cache.stats
        wrapper.cache_clear = cache.clear
        
        return wrapper
    return decorator


# =============================================================================
# PROFILING
# =============================================================================

@contextmanager
def profile_code(sort_by: str = 'cumulative', top_n: int = 20):
    """
    Context manager pour profiler du code
    
    Args:
        sort_by: Critère de tri (cumulative, time, calls)
        top_n: Nombre de lignes à afficher
    """
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        yield profiler
    finally:
        profiler.disable()
        
        # Analyser les résultats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats(sort_by)
        ps.print_stats(top_n)
        
        print("=== PROFILING RESULTS ===")
        print(s.getvalue())


class FunctionProfiler:
    """Profileur de fonctions avancé"""
    
    def __init__(self):
        self.profiles: Dict[str, List[pstats.Stats]] = defaultdict(list)
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Any:
        """
        Profile l'exécution d'une fonction
        
        Args:
            func: Fonction à profiler
            *args, **kwargs: Arguments de la fonction
            
        Returns:
            Résultat de la fonction
        """
        func_name = f"{func.__module__}.{func.__qualname__}"
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()
            
            stats = pstats.Stats(profiler)
            self.profiles[func_name].append(stats)
        
        return result
    
    def get_aggregate_stats(self, func_name: str) -> Optional[pstats.Stats]:
        """
        Obtient les statistiques agrégées pour une fonction
        
        Args:
            func_name: Nom de la fonction
            
        Returns:
            Statistiques agrégées
        """
        if func_name not in self.profiles:
            return None
        
        if len(self.profiles[func_name]) == 1:
            return self.profiles[func_name][0]
        
        # Agréger plusieurs profils
        aggregate = self.profiles[func_name][0]
        for stats in self.profiles[func_name][1:]:
            aggregate.add(stats)
        
        return aggregate
    
    def print_stats(self, func_name: str, sort_by: str = 'cumulative', top_n: int = 20) -> None:
        """
        Affiche les statistiques d'une fonction
        
        Args:
            func_name: Nom de la fonction
            sort_by: Critère de tri
            top_n: Nombre de lignes
        """
        stats = self.get_aggregate_stats(func_name)
        if stats:
            stats.sort_stats(sort_by)
            stats.print_stats(top_n)


# =============================================================================
# MONITORING SYSTÈME
# =============================================================================

class SystemMonitor:
    """Moniteur de ressources système"""
    
    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.is_monitoring = False
        self.metrics: List[Dict[str, Any]] = []
        self._monitor_thread: Optional[threading.Thread] = None
    
    def start_monitoring(self) -> None:
        """Démarre le monitoring système"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Arrête le monitoring système"""
        self.is_monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
    
    def _monitor_loop(self) -> None:
        """Boucle de monitoring"""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics.append(metrics)
                
                # Limiter l'historique
                if len(self.metrics) > 1000:
                    self.metrics = self.metrics[-500:]
                
                time.sleep(self.sampling_interval)
            except Exception:
                pass
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collecte les métriques système"""
        process = psutil.Process()
        
        return {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'process_memory': process.memory_info().rss / 1024 / 1024,  # MB
            'process_cpu': process.cpu_percent(),
            'open_files': len(process.open_files()),
            'connections': len(process.connections())
        }
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Obtient les métriques actuelles"""
        return self._collect_metrics()
    
    def get_average_metrics(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """
        Obtient les métriques moyennes sur une période
        
        Args:
            duration_seconds: Durée en secondes
            
        Returns:
            Métriques moyennes
        """
        cutoff_time = time.time() - duration_seconds
        recent_metrics = [m for m in self.metrics if m['timestamp'] > cutoff_time]
        
        if not recent_metrics:
            return {}
        
        # Calculer les moyennes
        avg_metrics = {}
        for key in recent_metrics[0].keys():
            if key != 'timestamp':
                values = [m[key] for m in recent_metrics if isinstance(m[key], (int, float))]
                if values:
                    avg_metrics[f'avg_{key}'] = sum(values) / len(values)
        
        return avg_metrics


# =============================================================================
# DÉTECTION DE GOULOTS D'ÉTRANGLEMENT
# =============================================================================

class BottleneckDetector:
    """Détecteur de goulots d'étranglement"""
    
    def __init__(self):
        self.slow_functions: Dict[str, List[float]] = defaultdict(list)
        self.memory_hogs: Dict[str, List[float]] = defaultdict(list)
    
    def analyze_function(self, func_name: str, execution_time: float, 
                        memory_usage: float) -> Dict[str, Any]:
        """
        Analyse une fonction pour détecter les problèmes
        
        Args:
            func_name: Nom de la fonction
            execution_time: Temps d'exécution
            memory_usage: Utilisation mémoire
            
        Returns:
            Rapport d'analyse
        """
        # Enregistrer les métriques
        self.slow_functions[func_name].append(execution_time)
        self.memory_hogs[func_name].append(memory_usage)
        
        # Limiter l'historique
        if len(self.slow_functions[func_name]) > 100:
            self.slow_functions[func_name] = self.slow_functions[func_name][-50:]
            self.memory_hogs[func_name] = self.memory_hogs[func_name][-50:]
        
        # Analyser
        issues = []
        suggestions = []
        
        # Détection de lenteur
        avg_time = sum(self.slow_functions[func_name]) / len(self.slow_functions[func_name])
        if avg_time > 1.0:  # Plus d'une seconde
            issues.append("Fonction lente détectée")
            suggestions.append("Considérer l'optimisation ou la mise en cache")
        
        # Détection de forte consommation mémoire
        avg_memory = sum(self.memory_hogs[func_name]) / len(self.memory_hogs[func_name])
        if avg_memory > 100:  # Plus de 100MB
            issues.append("Forte consommation mémoire")
            suggestions.append("Optimiser l'utilisation mémoire ou utiliser des générateurs")
        
        return {
            'function': func_name,
            'avg_execution_time': avg_time,
            'avg_memory_usage': avg_memory,
            'issues': issues,
            'suggestions': suggestions
        }
    
    def get_top_slow_functions(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Retourne les fonctions les plus lentes
        
        Args:
            top_n: Nombre de fonctions à retourner
            
        Returns:
            Liste des fonctions lentes
        """
        function_averages = []
        
        for func_name, times in self.slow_functions.items():
            if times:
                avg_time = sum(times) / len(times)
                function_averages.append({
                    'function': func_name,
                    'avg_time': avg_time,
                    'call_count': len(times)
                })
        
        return sorted(function_averages, key=lambda x: x['avg_time'], reverse=True)[:top_n]


# =============================================================================
# UTILITAIRES D'OPTIMISATION
# =============================================================================

def force_garbage_collection() -> Dict[str, int]:
    """
    Force la collecte des ordures et retourne les statistiques
    
    Returns:
        Statistiques de collecte
    """
    before = gc.get_count()
    collected = gc.collect()
    after = gc.get_count()
    
    return {
        'collected_objects': collected,
        'before_count': before,
        'after_count': after,
        'generations': len(gc.get_stats())
    }


def memory_usage_mb() -> float:
    """
    Retourne l'utilisation mémoire actuelle en MB
    
    Returns:
        Utilisation mémoire en MB
    """
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


@contextmanager
def memory_tracker():
    """Context manager pour traquer l'utilisation mémoire"""
    tracemalloc.start()
    start_memory = memory_usage_mb()
    
    try:
        yield
    finally:
        end_memory = memory_usage_mb()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"Memory usage:")
        print(f"  Start: {start_memory:.2f} MB")
        print(f"  End: {end_memory:.2f} MB")
        print(f"  Difference: {end_memory - start_memory:.2f} MB")
        print(f"  Peak: {peak / 1024 / 1024:.2f} MB")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "PerformanceMetrics",
    "PerformanceMonitor",
    "performance_monitor",
    "monitor_performance",
    "benchmark",
    "rate_limit",
    "PerformanceCache",
    "memoize",
    "profile_code",
    "FunctionProfiler",
    "SystemMonitor",
    "BottleneckDetector",
    "force_garbage_collection",
    "memory_usage_mb",
    "memory_tracker"
]
