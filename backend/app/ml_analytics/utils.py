# 🎵 ML Analytics Utilities
# ========================
# 
# Utilitaires avancés pour ML Analytics
# Optimisations performances et monitoring enterprise
#
# 🎖️ Expert: Ingénieur Machine Learning

"""
🔧 ML Analytics Utilities
==========================

Comprehensive utility functions for ML Analytics:
- Performance optimization and caching
- Data preprocessing and transformation
- Monitoring and metrics collection
- Security and validation helpers
- Async utilities and resource management
"""

import asyncio
import logging
import time
import hashlib
import json
import pickle
import gzip
import base64
import functools
import inspect
from typing import (
    Dict, Any, List, Optional, Union, Callable, 
    TypeVar, Generic, Awaitable, Tuple, Set
)
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import redis
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import weakref
import gc

# Types personnalisés
T = TypeVar('T')
R = TypeVar('R')

logger = logging.getLogger(__name__)


# ===========================
# 🚀 Performance & Caching
# ===========================

class AdvancedCache:
    """Cache avancé avec TTL, compression et persistance"""
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float = 3600,
        compression: bool = True,
        persistent: bool = False,
        cache_dir: Optional[Path] = None
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.compression = compression
        self.persistent = persistent
        self.cache_dir = cache_dir or Path(".cache/ml_analytics")
        
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        
        if self.persistent:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_persistent_cache()
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Génération de clé de cache"""
        content = str(args) + str(sorted(kwargs.items()))
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _serialize_value(self, value: Any) -> bytes:
        """Sérialisation avec compression optionnelle"""
        data = pickle.dumps(value)
        if self.compression:
            data = gzip.compress(data)
        return data
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Désérialisation avec décompression optionnelle"""
        if self.compression:
            data = gzip.decompress(data)
        return pickle.loads(data)
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Vérification d'expiration"""
        if entry['ttl'] is None:
            return False
        return time.time() > entry['created_at'] + entry['ttl']
    
    def _evict_expired(self):
        """Éviction des entrées expirées"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if self._is_expired(entry)
        ]
        
        for key in expired_keys:
            self._remove_entry(key)
    
    def _evict_lru(self):
        """Éviction LRU si nécessaire"""
        if len(self._cache) <= self.max_size:
            return
        
        # Tri par temps d'accès
        sorted_keys = sorted(
            self._access_times.items(),
            key=lambda x: x[1]
        )
        
        # Suppression des plus anciens
        to_remove = len(self._cache) - self.max_size
        for key, _ in sorted_keys[:to_remove]:
            self._remove_entry(key)
    
    def _remove_entry(self, key: str):
        """Suppression d'une entrée"""
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
        
        if self.persistent:
            cache_file = self.cache_dir / f"{key}.cache"
            if cache_file.exists():
                cache_file.unlink()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Récupération d'une valeur"""
        with self._lock:
            if key not in self._cache:
                if self.persistent:
                    self._load_from_disk(key)
                
                if key not in self._cache:
                    return default
            
            entry = self._cache[key]
            if self._is_expired(entry):
                self._remove_entry(key)
                return default
            
            self._access_times[key] = time.time()
            return entry['value']
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[float] = None
    ):
        """Stockage d'une valeur"""
        with self._lock:
            ttl = ttl if ttl is not None else self.default_ttl
            
            entry = {
                'value': value,
                'created_at': time.time(),
                'ttl': ttl
            }
            
            self._cache[key] = entry
            self._access_times[key] = time.time()
            
            if self.persistent:
                self._save_to_disk(key, entry)
            
            self._evict_expired()
            self._evict_lru()
    
    def _save_to_disk(self, key: str, entry: Dict[str, Any]):
        """Sauvegarde sur disque"""
        try:
            cache_file = self.cache_dir / f"{key}.cache"
            serialized = self._serialize_value(entry)
            cache_file.write_bytes(serialized)
        except Exception as e:
            logger.warning(f"Erreur sauvegarde cache {key}: {e}")
    
    def _load_from_disk(self, key: str):
        """Chargement depuis le disque"""
        try:
            cache_file = self.cache_dir / f"{key}.cache"
            if cache_file.exists():
                serialized = cache_file.read_bytes()
                entry = self._deserialize_value(serialized)
                
                if not self._is_expired(entry):
                    self._cache[key] = entry
        except Exception as e:
            logger.warning(f"Erreur chargement cache {key}: {e}")
    
    def _load_persistent_cache(self):
        """Chargement du cache persistant au démarrage"""
        if not self.cache_dir.exists():
            return
        
        for cache_file in self.cache_dir.glob("*.cache"):
            key = cache_file.stem
            self._load_from_disk(key)
    
    def clear(self):
        """Vidage du cache"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            
            if self.persistent:
                for cache_file in self.cache_dir.glob("*.cache"):
                    cache_file.unlink()
    
    def size(self) -> int:
        """Taille du cache"""
        return len(self._cache)
    
    def stats(self) -> Dict[str, Any]:
        """Statistiques du cache"""
        total_size = sum(
            len(self._serialize_value(entry))
            for entry in self._cache.values()
        )
        
        return {
            'entries': len(self._cache),
            'max_size': self.max_size,
            'total_size_bytes': total_size,
            'compression_enabled': self.compression,
            'persistent': self.persistent
        }


# Cache global
_global_cache = AdvancedCache()


def cache_result(
    ttl: Optional[float] = None,
    key_func: Optional[Callable] = None,
    ignore_args: Optional[List[str]] = None
):
    """Décorateur de cache pour les résultats de fonction"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Génération de la clé de cache
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Filtrage des arguments ignorés
                filtered_kwargs = kwargs.copy()
                if ignore_args:
                    for arg in ignore_args:
                        filtered_kwargs.pop(arg, None)
                
                cache_key = _global_cache._generate_key(
                    func.__name__, *args, **filtered_kwargs
                )
            
            # Vérification du cache
            result = _global_cache.get(cache_key)
            if result is not None:
                return result
            
            # Exécution et cache
            result = func(*args, **kwargs)
            _global_cache.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator


async def async_cache_result(
    ttl: Optional[float] = None,
    key_func: Optional[Callable] = None
):
    """Décorateur de cache pour les fonctions async"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Génération de la clé de cache
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = _global_cache._generate_key(
                    func.__name__, *args, **kwargs
                )
            
            # Vérification du cache
            result = _global_cache.get(cache_key)
            if result is not None:
                return result
            
            # Exécution et cache
            result = await func(*args, **kwargs)
            _global_cache.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator


# ===========================
# 📊 Data Processing Utilities
# ===========================

class DataProcessor:
    """Processeur de données avancé"""
    
    @staticmethod
    def normalize_features(
        data: np.ndarray,
        method: str = "standardize"
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Normalisation des features avec statistiques"""
        if method == "standardize":
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            std = np.where(std == 0, 1, std)  # Éviter division par 0
            normalized = (data - mean) / std
            stats = {"mean": mean, "std": std, "method": "standardize"}
        
        elif method == "min_max":
            min_val = np.min(data, axis=0)
            max_val = np.max(data, axis=0)
            range_val = max_val - min_val
            range_val = np.where(range_val == 0, 1, range_val)
            normalized = (data - min_val) / range_val
            stats = {"min": min_val, "max": max_val, "method": "min_max"}
        
        elif method == "robust":
            median = np.median(data, axis=0)
            mad = np.median(np.abs(data - median), axis=0)
            mad = np.where(mad == 0, 1, mad)
            normalized = (data - median) / mad
            stats = {"median": median, "mad": mad, "method": "robust"}
        
        else:
            raise ValueError(f"Méthode de normalisation inconnue: {method}")
        
        return normalized, stats
    
    @staticmethod
    def detect_outliers(
        data: np.ndarray,
        method: str = "iqr",
        threshold: float = 1.5
    ) -> np.ndarray:
        """Détection d'outliers"""
        if method == "iqr":
            Q1 = np.percentile(data, 25, axis=0)
            Q3 = np.percentile(data, 75, axis=0)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (data < lower_bound) | (data > upper_bound)
        
        elif method == "zscore":
            z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
            outliers = z_scores > threshold
        
        elif method == "modified_zscore":
            median = np.median(data, axis=0)
            mad = np.median(np.abs(data - median), axis=0)
            modified_z_scores = 0.6745 * (data - median) / mad
            outliers = np.abs(modified_z_scores) > threshold
        
        else:
            raise ValueError(f"Méthode de détection d'outliers inconnue: {method}")
        
        return outliers.any(axis=1)
    
    @staticmethod
    def handle_missing_values(
        data: pd.DataFrame,
        strategy: str = "mean",
        fill_value: Any = None
    ) -> pd.DataFrame:
        """Gestion des valeurs manquantes"""
        data_copy = data.copy()
        
        if strategy == "mean":
            numeric_columns = data_copy.select_dtypes(include=[np.number]).columns
            data_copy[numeric_columns] = data_copy[numeric_columns].fillna(
                data_copy[numeric_columns].mean()
            )
        
        elif strategy == "median":
            numeric_columns = data_copy.select_dtypes(include=[np.number]).columns
            data_copy[numeric_columns] = data_copy[numeric_columns].fillna(
                data_copy[numeric_columns].median()
            )
        
        elif strategy == "mode":
            for column in data_copy.columns:
                mode_value = data_copy[column].mode()
                if not mode_value.empty:
                    data_copy[column].fillna(mode_value[0], inplace=True)
        
        elif strategy == "forward_fill":
            data_copy = data_copy.fillna(method='ffill')
        
        elif strategy == "backward_fill":
            data_copy = data_copy.fillna(method='bfill')
        
        elif strategy == "constant":
            data_copy = data_copy.fillna(fill_value)
        
        elif strategy == "drop":
            data_copy = data_copy.dropna()
        
        else:
            raise ValueError(f"Stratégie inconnue: {strategy}")
        
        return data_copy
    
    @staticmethod
    def encode_categorical(
        data: pd.DataFrame,
        columns: List[str],
        method: str = "onehot"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Encodage des variables catégorielles"""
        data_copy = data.copy()
        encoding_info = {}
        
        for column in columns:
            if column not in data_copy.columns:
                continue
            
            if method == "onehot":
                dummies = pd.get_dummies(data_copy[column], prefix=column)
                data_copy = pd.concat([data_copy.drop(column, axis=1), dummies], axis=1)
                encoding_info[column] = {
                    "method": "onehot",
                    "categories": list(dummies.columns)
                }
            
            elif method == "label":
                unique_values = data_copy[column].unique()
                mapping = {val: idx for idx, val in enumerate(unique_values)}
                data_copy[column] = data_copy[column].map(mapping)
                encoding_info[column] = {
                    "method": "label",
                    "mapping": mapping
                }
            
            elif method == "target":
                # Encodage par la moyenne de la variable cible (nécessite une colonne target)
                if 'target' in data_copy.columns:
                    target_mean = data_copy.groupby(column)['target'].mean()
                    data_copy[column] = data_copy[column].map(target_mean)
                    encoding_info[column] = {
                        "method": "target",
                        "mapping": target_mean.to_dict()
                    }
        
        return data_copy, encoding_info


# ===========================
# 📈 Monitoring & Metrics
# ===========================

@dataclass
class PerformanceMetrics:
    """Métriques de performance"""
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_usage_percent: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    function_name: str = ""
    args_count: int = 0
    success: bool = True
    error_message: Optional[str] = None


class PerformanceMonitor:
    """Moniteur de performance avancé"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history: List[PerformanceMetrics] = []
        self._lock = threading.Lock()
    
    @asynccontextmanager
    async def monitor_async(self, function_name: str, *args):
        """Context manager pour le monitoring async"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        metrics = PerformanceMetrics(
            function_name=function_name,
            args_count=len(args)
        )
        
        try:
            yield metrics
            metrics.success = True
        except Exception as e:
            metrics.success = False
            metrics.error_message = str(e)
            raise
        finally:
            metrics.execution_time = time.time() - start_time
            metrics.memory_usage_mb = self._get_memory_usage() - start_memory
            metrics.cpu_usage_percent = psutil.cpu_percent()
            
            # GPU usage si disponible
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics.gpu_usage_percent = info.gpu
            except:
                metrics.gpu_usage_percent = 0.0
            
            self._add_metrics(metrics)
    
    def monitor_sync(self, func: Callable) -> Callable:
        """Décorateur pour le monitoring synchrone"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            metrics = PerformanceMetrics(
                function_name=func.__name__,
                args_count=len(args)
            )
            
            try:
                result = func(*args, **kwargs)
                metrics.success = True
                return result
            except Exception as e:
                metrics.success = False
                metrics.error_message = str(e)
                raise
            finally:
                metrics.execution_time = time.time() - start_time
                metrics.memory_usage_mb = self._get_memory_usage() - start_memory
                metrics.cpu_usage_percent = psutil.cpu_percent()
                self._add_metrics(metrics)
        
        return wrapper
    
    def _get_memory_usage(self) -> float:
        """Utilisation mémoire en MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _add_metrics(self, metrics: PerformanceMetrics):
        """Ajout de métriques à l'historique"""
        with self._lock:
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_history:
                self.metrics_history = self.metrics_history[-self.max_history:]
    
    def get_statistics(self, function_name: Optional[str] = None) -> Dict[str, Any]:
        """Statistiques de performance"""
        with self._lock:
            filtered_metrics = self.metrics_history
            if function_name:
                filtered_metrics = [
                    m for m in filtered_metrics
                    if m.function_name == function_name
                ]
            
            if not filtered_metrics:
                return {}
            
            execution_times = [m.execution_time for m in filtered_metrics]
            memory_usages = [m.memory_usage_mb for m in filtered_metrics]
            success_rate = sum(m.success for m in filtered_metrics) / len(filtered_metrics)
            
            return {
                'function_name': function_name,
                'total_calls': len(filtered_metrics),
                'success_rate': success_rate,
                'execution_time': {
                    'mean': np.mean(execution_times),
                    'median': np.median(execution_times),
                    'std': np.std(execution_times),
                    'min': np.min(execution_times),
                    'max': np.max(execution_times),
                    'p95': np.percentile(execution_times, 95),
                    'p99': np.percentile(execution_times, 99)
                },
                'memory_usage': {
                    'mean': np.mean(memory_usages),
                    'median': np.median(memory_usages),
                    'max': np.max(memory_usages)
                }
            }
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Erreurs récentes"""
        with self._lock:
            error_metrics = [
                m for m in self.metrics_history
                if not m.success
            ][-limit:]
            
            return [
                {
                    'function_name': m.function_name,
                    'error_message': m.error_message,
                    'timestamp': m.timestamp.isoformat(),
                    'execution_time': m.execution_time
                }
                for m in error_metrics
            ]


# Instance globale du moniteur
performance_monitor = PerformanceMonitor()


# ===========================
# 🔒 Security & Validation
# ===========================

class SecurityValidator:
    """Validateur de sécurité"""
    
    @staticmethod
    def validate_input_size(data: Any, max_size_mb: float = 100) -> bool:
        """Validation de la taille des données"""
        import sys
        size_bytes = sys.getsizeof(data)
        size_mb = size_bytes / 1024 / 1024
        return size_mb <= max_size_mb
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Assainissement des noms de fichier"""
        import re
        # Suppression des caractères dangereux
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Limitation de la longueur
        sanitized = sanitized[:255]
        # Éviter les noms réservés
        reserved = ['CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 
                   'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 
                   'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9']
        if sanitized.upper() in reserved:
            sanitized = f"_{sanitized}"
        return sanitized
    
    @staticmethod
    def validate_model_path(path: str) -> bool:
        """Validation des chemins de modèles"""
        path_obj = Path(path)
        # Vérification que le chemin ne sort pas du dossier autorisé
        try:
            path_obj.resolve().relative_to(Path.cwd())
            return True
        except ValueError:
            return False
    
    @staticmethod
    def hash_sensitive_data(data: str, salt: str = "") -> str:
        """Hachage des données sensibles"""
        content = data + salt
        return hashlib.sha256(content.encode()).hexdigest()


# ===========================
# ⚡ Async Utilities
# ===========================

class AsyncPool:
    """Pool de tâches asynchrones"""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.semaphore = asyncio.Semaphore(max_workers)
        self.active_tasks: Set[asyncio.Task] = set()
    
    async def submit(self, coro: Awaitable[T]) -> T:
        """Soumission d'une tâche"""
        async with self.semaphore:
            task = asyncio.create_task(coro)
            self.active_tasks.add(task)
            
            try:
                result = await task
                return result
            finally:
                self.active_tasks.discard(task)
    
    async def submit_batch(self, coros: List[Awaitable[T]]) -> List[T]:
        """Soumission de tâches en lot"""
        tasks = [self.submit(coro) for coro in coros]
        return await asyncio.gather(*tasks)
    
    async def shutdown(self, timeout: float = 30):
        """Arrêt propre du pool"""
        if self.active_tasks:
            await asyncio.wait(
                self.active_tasks,
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED
            )


# ===========================
# 🧠 Memory Management
# ===========================

class MemoryManager:
    """Gestionnaire de mémoire intelligent"""
    
    def __init__(self, max_memory_gb: float = 8.0):
        self.max_memory_gb = max_memory_gb
        self.weak_references: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Utilisation mémoire actuelle"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_gb': memory_info.rss / 1024 / 1024 / 1024,
            'vms_gb': memory_info.vms / 1024 / 1024 / 1024,
            'percent': process.memory_percent(),
            'available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024
        }
    
    def check_memory_pressure(self) -> bool:
        """Vérification de la pression mémoire"""
        usage = self.get_memory_usage()
        return usage['rss_gb'] > self.max_memory_gb * 0.8
    
    def force_garbage_collection(self):
        """Garbage collection forcé"""
        collected = gc.collect()
        logger.info(f"Garbage collection: {collected} objets collectés")
        return collected
    
    def register_large_object(self, name: str, obj: Any):
        """Enregistrement d'un objet volumineux"""
        self.weak_references[name] = obj
    
    def cleanup_large_objects(self):
        """Nettoyage des objets volumineux"""
        self.weak_references.clear()
        self.force_garbage_collection()


# ===========================
# 📊 Data Analysis Helpers
# ===========================

def analyze_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyse complète d'un DataFrame"""
    analysis = {
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'duplicated_rows': df.duplicated().sum(),
        'numeric_stats': {},
        'categorical_stats': {}
    }
    
    # Statistiques numériques
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        analysis['numeric_stats'][col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'skewness': df[col].skew(),
            'kurtosis': df[col].kurtosis()
        }
    
    # Statistiques catégorielles
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_columns:
        analysis['categorical_stats'][col] = {
            'unique_count': df[col].nunique(),
            'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
            'value_counts': df[col].value_counts().head(10).to_dict()
        }
    
    return analysis


def create_correlation_matrix(df: pd.DataFrame, method: str = 'pearson') -> np.ndarray:
    """Création d'une matrice de corrélation"""
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr(method=method).values


# Instances globales
memory_manager = MemoryManager()
async_pool = AsyncPool()

# Exports publics
__all__ = [
    # Cache
    'AdvancedCache',
    'cache_result',
    'async_cache_result',
    
    # Data processing
    'DataProcessor',
    'analyze_dataframe',
    'create_correlation_matrix',
    
    # Monitoring
    'PerformanceMetrics',
    'PerformanceMonitor',
    'performance_monitor',
    
    # Security
    'SecurityValidator',
    
    # Async utilities
    'AsyncPool',
    'async_pool',
    
    # Memory management
    'MemoryManager',
    'memory_manager'
]
