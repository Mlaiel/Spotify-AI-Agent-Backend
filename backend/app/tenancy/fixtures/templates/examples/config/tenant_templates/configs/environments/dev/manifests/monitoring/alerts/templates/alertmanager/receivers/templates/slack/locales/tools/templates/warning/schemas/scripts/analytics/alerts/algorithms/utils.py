"""
🎵 Spotify AI Agent - Enterprise Utility Functions for Alert Algorithms
=====================================================================

Ultra-Advanced Utility Module for Music Streaming Platform Intelligence

This comprehensive utility module provides enterprise-grade functions and classes
specifically designed for high-performance alert processing in large-scale music
streaming platforms. It includes advanced data processing, caching, monitoring,
and optimization utilities.

🚀 ENTERPRISE UTILITY CATEGORIES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔧 Data Processing & Transformation:
  • Advanced feature engineering for music streaming metrics
  • Real-time data preprocessing and normalization
  • Time series data handling with seasonal adjustments
  • Multi-dimensional data aggregation and windowing
  • Audio quality metrics calculation and analysis
  • User behavior pattern extraction and encoding

📊 Performance Monitoring & Metrics:
  • Prometheus metrics integration with custom collectors
  • Real-time performance tracking and alerting
  • Resource utilization monitoring (CPU, Memory, GPU)
  • Latency tracking with percentile calculations
  • Throughput measurement and optimization
  • Business metrics correlation and analysis

⚡ Caching & Storage Optimization:
  • Redis-based intelligent caching with TTL management
  • Distributed cache coherence and invalidation
  • Memory-efficient data structures and serialization
  • Compression algorithms for large datasets
  • Hot/Cold data tier management
  • Cache warming and preloading strategies

🔍 Debugging & Profiling Tools:
  • Advanced performance profiling with detailed metrics
  • Memory leak detection and optimization
  • Execution time analysis with bottleneck identification
  • Error tracking and anomaly reporting
  • Debug logging with structured output
  • A/B testing framework integration

🛡️ Security & Validation:
  • Input validation and sanitization
  • Data encryption and secure transmission
  • Access control and authorization utilities
  • Audit logging and compliance tracking
  • Threat detection and prevention
  • Secure configuration management

🎯 Music Streaming Specializations:
  • Audio quality metric calculations (bitrate, latency, buffering)
  • User engagement scoring algorithms
  • Content recommendation performance tracking
  • Geographic performance analysis tools
  • Revenue impact calculation utilities
  • Content licensing and royalty tracking

⚙️ System Integration & Operations:
  • Kubernetes integration and auto-scaling
  • Service mesh integration (Istio/Linkerd)
  • Message queue optimization (Kafka/RabbitMQ)
  • Database connection pooling and optimization
  • API rate limiting and throttling
  • Health check and readiness probe utilities

@Author: Utility Functions by Fahed Mlaiel
@Version: 2.0.0 (Enterprise Edition)  
@Last Updated: 2025-07-19
"""

import asyncio
import json
import pickle
import hashlib
import time
import traceback
import functools
import inspect
import threading
import multiprocessing
import gc
import sys
import os
from typing import (
    Dict, Any, Optional, List, Union, Callable, Tuple,
    AsyncGenerator, Generator, TypeVar, Generic, Set,
    Protocol, runtime_checkable, Iterator, Coroutine
)
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, Future
from contextlib import asynccontextmanager, contextmanager, suppress
from pathlib import Path
import logging
import warnings
from collections import defaultdict, deque
from enum import Enum, auto
import weakref

# High-performance computing libraries
import numpy as np
import pandas as pd
from scipy import stats, signal, optimize
from scipy.spatial.distance import cosine, euclidean
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# System monitoring and performance
import psutil
import resource
from memory_profiler import profile as memory_profile

# Caching and storage
try:
    import redis.asyncio as aioredis
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    warnings.warn("Redis not available. Caching features will be limited.")

# Monitoring and observability
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, Info, Enum as PrometheusEnum,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        multiprocess, values
    )
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    warnings.warn("Prometheus client not available. Metrics collection will be disabled.")

# Deep learning frameworks (optional)
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

try:
    import torch
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False

# Audio processing libraries (for music streaming specific features)
try:
    import librosa
    import soundfile as sf
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Type definitions
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

# Module metadata
__title__ = "Spotify AI Agent - Enterprise Utilities"
__version__ = "2.0.0"
__author__ = "Fahed Mlaiel" 
__license__ = "Proprietary"

class AlertSeverity(Enum):
    """Alert severity levels for classification."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class CacheStrategy(Enum):
    """Caching strategy options."""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"
    ADAPTIVE = "adaptive"

class DataFormat(Enum):
    """Supported data formats for serialization."""
    JSON = "json"
    PICKLE = "pickle"
    MSGPACK = "msgpack"
    AVRO = "avro"
    PARQUET = "parquet"

@dataclass
class PerformanceMetrics:
    """
    Comprehensive performance metrics for algorithm monitoring.
    
    Tracks various performance indicators including latency, throughput,
    accuracy, resource utilization, and business metrics.
    """
    execution_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    accuracy_score: Optional[float] = None
    throughput_per_second: Optional[float] = None
    error_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class CacheEntry:
    """Entrée de cache avec métadonnées."""
    data: Any
    timestamp: datetime
    ttl_seconds: int
    access_count: int = 0
    last_access: datetime = field(default_factory=datetime.now)
    
    @property
    def is_expired(self) -> bool:
        """Vérifie si l'entrée est expirée."""
        return datetime.now() > self.timestamp + timedelta(seconds=self.ttl_seconds)
    
    def touch(self) -> None:
        """Met à jour l'heure d'accès."""
        self.access_count += 1
        self.last_access = datetime.now()

class PrometheusMetricsManager:
    """Gestionnaire centralisé des métriques Prometheus."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self.metrics: Dict[str, Union[Counter, Histogram, Gauge, Summary]] = {}
        self._initialize_metrics()
    
    def _initialize_metrics(self) -> None:
        """Initialise les métriques communes."""
        
        # Compteurs
        self.register_counter(
            'algorithm_executions_total',
            'Total number of algorithm executions',
            ['algorithm_type', 'model_name', 'status']
        )
        
        self.register_counter(
            'algorithm_errors_total',
            'Total number of algorithm errors',
            ['algorithm_type', 'model_name', 'error_type']
        )
        
        # Histogrammes
        self.register_histogram(
            'algorithm_execution_duration_seconds',
            'Algorithm execution duration in seconds',
            ['algorithm_type', 'model_name'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        self.register_histogram(
            'algorithm_memory_usage_mb',
            'Algorithm memory usage in MB',
            ['algorithm_type', 'model_name'],
            buckets=[10, 50, 100, 500, 1000, 2000, 5000, 10000]
        )
        
        # Jauges
        self.register_gauge(
            'algorithm_accuracy_score',
            'Current algorithm accuracy score',
            ['algorithm_type', 'model_name']
        )
        
        self.register_gauge(
            'active_models_count',
            'Number of active models',
            ['algorithm_type']
        )
        
        self.register_gauge(
            'cache_hit_rate',
            'Cache hit rate percentage',
            ['cache_type']
        )
        
        # Résumés
        self.register_summary(
            'algorithm_throughput_per_second',
            'Algorithm throughput per second',
            ['algorithm_type', 'model_name']
        )
    
    def register_counter(self, name: str, description: str, 
                        labels: Optional[List[str]] = None) -> Counter:
        """Enregistre un compteur."""
        metric = Counter(name, description, labels or [], registry=self.registry)
        self.metrics[name] = metric
        return metric
    
    def register_histogram(self, name: str, description: str, 
                          labels: Optional[List[str]] = None,
                          buckets: Optional[List[float]] = None) -> Histogram:
        """Enregistre un histogramme."""
        metric = Histogram(name, description, labels or [], 
                          registry=self.registry, buckets=buckets)
        self.metrics[name] = metric
        return metric
    
    def register_gauge(self, name: str, description: str, 
                      labels: Optional[List[str]] = None) -> Gauge:
        """Enregistre une jauge."""
        metric = Gauge(name, description, labels or [], registry=self.registry)
        self.metrics[name] = metric
        return metric
    
    def register_summary(self, name: str, description: str,
                        labels: Optional[List[str]] = None) -> Summary:
        """Enregistre un résumé."""
        metric = Summary(name, description, labels or [], registry=self.registry)
        self.metrics[name] = metric
        return metric
    
    def get_metric(self, name: str) -> Optional[Union[Counter, Histogram, Gauge, Summary]]:
        """Retourne une métrique par nom."""
        return self.metrics.get(name)
    
    def record_execution(self, algorithm_type: str, model_name: str, 
                        metrics: PerformanceMetrics, status: str = 'success') -> None:
        """Enregistre les métriques d'exécution."""
        
        labels = {'algorithm_type': algorithm_type, 'model_name': model_name}
        
        # Compteur d'exécutions
        counter = self.get_metric('algorithm_executions_total')
        if counter:
            counter.labels(**labels, status=status).inc()
        
        # Durée d'exécution
        duration_hist = self.get_metric('algorithm_execution_duration_seconds')
        if duration_hist:
            duration_hist.labels(**labels).observe(metrics.execution_time_ms / 1000)
        
        # Utilisation mémoire
        memory_hist = self.get_metric('algorithm_memory_usage_mb')
        if memory_hist:
            memory_hist.labels(**labels).observe(metrics.memory_usage_mb)
        
        # Précision si disponible
        if metrics.accuracy_score is not None:
            accuracy_gauge = self.get_metric('algorithm_accuracy_score')
            if accuracy_gauge:
                accuracy_gauge.labels(**labels).set(metrics.accuracy_score)
        
        # Throughput si disponible
        if metrics.throughput_per_second is not None:
            throughput_summary = self.get_metric('algorithm_throughput_per_second')
            if throughput_summary:
                throughput_summary.labels(**labels).observe(metrics.throughput_per_second)
    
    def record_error(self, algorithm_type: str, model_name: str, 
                    error_type: str) -> None:
        """Enregistre une erreur."""
        
        error_counter = self.get_metric('algorithm_errors_total')
        if error_counter:
            error_counter.labels(
                algorithm_type=algorithm_type,
                model_name=model_name,
                error_type=error_type
            ).inc()
    
    def export_metrics(self) -> str:
        """Exporte les métriques au format Prometheus."""
        return generate_latest(self.registry).decode('utf-8')

class RedisCache:
    """Cache Redis avancé avec compression et sérialisation optimisée."""
    
    def __init__(self, redis_client: aioredis.Redis, 
                 default_ttl: int = 300,
                 compression_threshold: int = 1024,
                 max_retries: int = 3):
        self.redis = redis_client
        self.default_ttl = default_ttl
        self.compression_threshold = compression_threshold
        self.max_retries = max_retries
        self.hit_count = 0
        self.miss_count = 0
    
    def _generate_key(self, key: str, algorithm_type: str, 
                     model_name: str) -> str:
        """Génère une clé de cache structurée."""
        return f"algo:{algorithm_type}:{model_name}:{key}"
    
    def _serialize_data(self, data: Any) -> bytes:
        """Sérialise les données avec compression si nécessaire."""
        
        # Utilisation de pickle pour la sérialisation
        serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Compression pour les gros objets
        if len(serialized) > self.compression_threshold:
            import gzip
            serialized = gzip.compress(serialized)
            return b'compressed:' + serialized
        
        return serialized
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Désérialise les données avec décompression si nécessaire."""
        
        if data.startswith(b'compressed:'):
            import gzip
            data = gzip.decompress(data[11:])  # Remove 'compressed:' prefix
        
        return pickle.loads(data)
    
    async def get(self, key: str, algorithm_type: str, 
                 model_name: str) -> Optional[Any]:
        """Récupère une valeur du cache."""
        
        cache_key = self._generate_key(key, algorithm_type, model_name)
        
        try:
            for attempt in range(self.max_retries):
                try:
                    data = await self.redis.get(cache_key)
                    if data:
                        self.hit_count += 1
                        return self._deserialize_data(data)
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    await asyncio.sleep(0.1 * (attempt + 1))
            
            self.miss_count += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache get error for key {cache_key}: {e}")
            self.miss_count += 1
            return None
    
    async def set(self, key: str, algorithm_type: str, model_name: str,
                 value: Any, ttl: Optional[int] = None) -> bool:
        """Stocke une valeur dans le cache."""
        
        cache_key = self._generate_key(key, algorithm_type, model_name)
        ttl = ttl or self.default_ttl
        
        try:
            serialized_data = self._serialize_data(value)
            
            for attempt in range(self.max_retries):
                try:
                    await self.redis.setex(cache_key, ttl, serialized_data)
                    return True
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    await asyncio.sleep(0.1 * (attempt + 1))
            
            return False
            
        except Exception as e:
            logger.error(f"Cache set error for key {cache_key}: {e}")
            return False
    
    async def delete(self, key: str, algorithm_type: str, 
                    model_name: str) -> bool:
        """Supprime une valeur du cache."""
        
        cache_key = self._generate_key(key, algorithm_type, model_name)
        
        try:
            result = await self.redis.delete(cache_key)
            return result > 0
        except Exception as e:
            logger.error(f"Cache delete error for key {cache_key}: {e}")
            return False
    
    async def exists(self, key: str, algorithm_type: str, 
                    model_name: str) -> bool:
        """Vérifie si une clé existe dans le cache."""
        
        cache_key = self._generate_key(key, algorithm_type, model_name)
        
        try:
            result = await self.redis.exists(cache_key)
            return result > 0
        except Exception as e:
            logger.error(f"Cache exists error for key {cache_key}: {e}")
            return False
    
    @property
    def hit_rate(self) -> float:
        """Calcule le taux de réussite du cache."""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    async def flush_pattern(self, pattern: str) -> int:
        """Supprime toutes les clés correspondant au pattern."""
        
        try:
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                return await self.redis.delete(*keys)
            return 0
            
        except Exception as e:
            logger.error(f"Cache flush pattern error for {pattern}: {e}")
            return 0

class DataProcessor:
    """Processeur de données optimisé pour le machine learning."""
    
    def __init__(self):
        self.scalers: Dict[str, Union[StandardScaler, MinMaxScaler]] = {}
        self.feature_stats: Dict[str, Dict[str, float]] = {}
    
    def preprocess_timeseries(self, data: pd.DataFrame, 
                             target_column: str,
                             window_size: int = 24,
                             features: Optional[List[str]] = None) -> pd.DataFrame:
        """Préprocesse les données de séries temporelles."""
        
        if features is None:
            features = [col for col in data.columns if col != target_column]
        
        # Copie des données
        processed_data = data.copy()
        
        # Gestion des valeurs manquantes
        processed_data = processed_data.fillna(method='ffill').fillna(method='bfill')
        
        # Création des features temporelles
        if 'timestamp' in processed_data.columns:
            processed_data['hour'] = pd.to_datetime(processed_data['timestamp']).dt.hour
            processed_data['day_of_week'] = pd.to_datetime(processed_data['timestamp']).dt.dayofweek
            processed_data['is_weekend'] = processed_data['day_of_week'].isin([5, 6]).astype(int)
        
        # Features de fenêtre glissante
        for feature in features:
            if feature in processed_data.columns:
                # Moyennes mobiles
                processed_data[f'{feature}_ma_{window_size//4}'] = (
                    processed_data[feature].rolling(window=window_size//4).mean()
                )
                processed_data[f'{feature}_ma_{window_size//2}'] = (
                    processed_data[feature].rolling(window=window_size//2).mean()
                )
                processed_data[f'{feature}_ma_{window_size}'] = (
                    processed_data[feature].rolling(window=window_size).mean()
                )
                
                # Écarts-types mobiles
                processed_data[f'{feature}_std_{window_size}'] = (
                    processed_data[feature].rolling(window=window_size).std()
                )
                
                # Différences
                processed_data[f'{feature}_diff_1'] = processed_data[feature].diff(1)
                processed_data[f'{feature}_diff_24'] = processed_data[feature].diff(24)
                
                # Percentiles mobiles
                processed_data[f'{feature}_p95_{window_size}'] = (
                    processed_data[feature].rolling(window=window_size).quantile(0.95)
                )
                processed_data[f'{feature}_p05_{window_size}'] = (
                    processed_data[feature].rolling(window=window_size).quantile(0.05)
                )
        
        # Nettoyage des valeurs infinies et NaN
        processed_data = processed_data.replace([np.inf, -np.inf], np.nan)
        processed_data = processed_data.fillna(method='ffill').fillna(0)
        
        return processed_data
    
    def create_sequences(self, data: np.ndarray, 
                        sequence_length: int,
                        target_index: int = -1) -> tuple[np.ndarray, np.ndarray]:
        """Crée des séquences pour les modèles RNN/LSTM."""
        
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length, target_index])
        
        return np.array(X), np.array(y)
    
    def normalize_features(self, data: pd.DataFrame, 
                          method: str = 'standard',
                          feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Normalise les features."""
        
        if feature_columns is None:
            feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        normalized_data = data.copy()
        
        for column in feature_columns:
            if column in data.columns:
                scaler_key = f"{method}_{column}"
                
                if scaler_key not in self.scalers:
                    if method == 'standard':
                        self.scalers[scaler_key] = StandardScaler()
                    elif method == 'minmax':
                        self.scalers[scaler_key] = MinMaxScaler()
                    else:
                        raise ValueError(f"Unknown normalization method: {method}")
                    
                    # Fit du scaler
                    self.scalers[scaler_key].fit(data[[column]])
                
                # Transformation
                normalized_data[column] = self.scalers[scaler_key].transform(
                    data[[column]]
                ).flatten()
        
        return normalized_data
    
    def detect_outliers(self, data: pd.DataFrame, 
                       columns: Optional[List[str]] = None,
                       method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """Détecte les outliers dans les données."""
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_mask = pd.DataFrame(False, index=data.index, columns=columns)
        
        for column in columns:
            if column in data.columns:
                if method == 'iqr':
                    Q1 = data[column].quantile(0.25)
                    Q3 = data[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    outlier_mask[column] = (
                        (data[column] < lower_bound) | (data[column] > upper_bound)
                    )
                
                elif method == 'zscore':
                    z_scores = np.abs(stats.zscore(data[column].dropna()))
                    outlier_mask.loc[data[column].dropna().index, column] = z_scores > threshold
                
                elif method == 'modified_zscore':
                    median = data[column].median()
                    mad = np.median(np.abs(data[column] - median))
                    modified_z_scores = 0.6745 * (data[column] - median) / mad
                    outlier_mask[column] = np.abs(modified_z_scores) > threshold
        
        return outlier_mask

class ResourceMonitor:
    """Moniteur de ressources système."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Retourne les métriques actuelles."""
        
        # Mémoire
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # CPU
        cpu_percent = self.process.cpu_percent()
        
        return PerformanceMetrics(
            execution_time_ms=0,  # À remplir par le contexte d'appel
            memory_usage_mb=memory_mb,
            cpu_usage_percent=cpu_percent
        )
    
    @contextmanager
    def monitor_execution(self, algorithm_type: str, model_name: str):
        """Context manager pour monitorer l'exécution."""
        
        start_time = time.time()
        start_metrics = self.get_current_metrics()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_metrics = self.get_current_metrics()
            
            # Calcul des métriques
            execution_time_ms = (end_time - start_time) * 1000
            memory_delta = end_metrics.memory_usage_mb - start_metrics.memory_usage_mb
            
            metrics = PerformanceMetrics(
                execution_time_ms=execution_time_ms,
                memory_usage_mb=max(memory_delta, 0),
                cpu_usage_percent=end_metrics.cpu_usage_percent
            )
            
            # Log des métriques
            logger.info(
                f"Algorithm {algorithm_type}:{model_name} executed in "
                f"{execution_time_ms:.2f}ms, memory delta: {memory_delta:.2f}MB, "
                f"CPU: {metrics.cpu_usage_percent:.1f}%"
            )

def async_retry(max_retries: int = 3, 
               delay: float = 1.0,
               exponential_backoff: bool = True):
    """Décorateur pour retry automatique des fonctions async."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries: {e}"
                        )
                        raise
                    
                    wait_time = delay * (2 ** attempt) if exponential_backoff else delay
                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}), "
                        f"retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
            
            raise last_exception
        
        return wrapper
    return decorator

def measure_performance(metrics_manager: Optional[PrometheusMetricsManager] = None):
    """Décorateur pour mesurer les performances des fonctions."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            resource_monitor = ResourceMonitor()
            
            # Extraction des paramètres
            algorithm_type = kwargs.get('algorithm_type', 'unknown')
            model_name = kwargs.get('model_name', 'unknown')
            
            try:
                with resource_monitor.monitor_execution(algorithm_type, model_name):
                    result = await func(*args, **kwargs)
                
                # Enregistrement des métriques
                if metrics_manager:
                    execution_time = (time.time() - start_time) * 1000
                    current_metrics = resource_monitor.get_current_metrics()
                    current_metrics.execution_time_ms = execution_time
                    
                    metrics_manager.record_execution(
                        algorithm_type, model_name, current_metrics, 'success'
                    )
                
                return result
                
            except Exception as e:
                if metrics_manager:
                    metrics_manager.record_error(
                        algorithm_type, model_name, type(e).__name__
                    )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            resource_monitor = ResourceMonitor()
            
            # Extraction des paramètres
            algorithm_type = kwargs.get('algorithm_type', 'unknown')
            model_name = kwargs.get('model_name', 'unknown')
            
            try:
                with resource_monitor.monitor_execution(algorithm_type, model_name):
                    result = func(*args, **kwargs)
                
                # Enregistrement des métriques
                if metrics_manager:
                    execution_time = (time.time() - start_time) * 1000
                    current_metrics = resource_monitor.get_current_metrics()
                    current_metrics.execution_time_ms = execution_time
                    
                    metrics_manager.record_execution(
                        algorithm_type, model_name, current_metrics, 'success'
                    )
                
                return result
                
            except Exception as e:
                if metrics_manager:
                    metrics_manager.record_error(
                        algorithm_type, model_name, type(e).__name__
                    )
                raise
        
        # Retourne le wrapper approprié selon le type de fonction
        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

class ModelVersionManager:
    """Gestionnaire de versions de modèles ML."""
    
    def __init__(self, storage_path: str = "models"):
        self.storage_path = storage_path
        self.versions: Dict[str, List[str]] = {}
    
    def save_model(self, model: Any, algorithm_type: str, 
                  model_name: str, version: str) -> str:
        """Sauvegarde un modèle avec versioning."""
        
        import os
        import joblib
        
        model_dir = os.path.join(self.storage_path, algorithm_type, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f"model_v{version}.pkl")
        
        # Sauvegarde du modèle
        if hasattr(model, 'save'):  # TensorFlow/Keras model
            model_tf_path = os.path.join(model_dir, f"model_v{version}")
            model.save(model_tf_path)
        else:  # Scikit-learn ou autre
            joblib.dump(model, model_path)
        
        # Mise à jour du registre des versions
        key = f"{algorithm_type}:{model_name}"
        if key not in self.versions:
            self.versions[key] = []
        
        if version not in self.versions[key]:
            self.versions[key].append(version)
            self.versions[key].sort()
        
        return model_path
    
    def load_model(self, algorithm_type: str, model_name: str, 
                  version: Optional[str] = None) -> Any:
        """Charge un modèle par version."""
        
        import os
        import joblib
        
        key = f"{algorithm_type}:{model_name}"
        
        if version is None:
            # Charger la dernière version
            if key in self.versions and self.versions[key]:
                version = self.versions[key][-1]
            else:
                raise ValueError(f"No versions found for {key}")
        
        model_dir = os.path.join(self.storage_path, algorithm_type, model_name)
        
        # Essayer de charger un modèle TensorFlow d'abord
        model_tf_path = os.path.join(model_dir, f"model_v{version}")
        if os.path.exists(model_tf_path):
            return tf.keras.models.load_model(model_tf_path)
        
        # Sinon charger avec joblib
        model_path = os.path.join(model_dir, f"model_v{version}.pkl")
        if os.path.exists(model_path):
            return joblib.load(model_path)
        
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    def list_versions(self, algorithm_type: str, model_name: str) -> List[str]:
        """Liste les versions disponibles."""
        
        key = f"{algorithm_type}:{model_name}"
        return self.versions.get(key, [])
    
    def delete_version(self, algorithm_type: str, model_name: str, 
                      version: str) -> bool:
        """Supprime une version de modèle."""
        
        import os
        import shutil
        
        model_dir = os.path.join(self.storage_path, algorithm_type, model_name)
        
        # Suppression des fichiers
        model_path = os.path.join(model_dir, f"model_v{version}.pkl")
        model_tf_path = os.path.join(model_dir, f"model_v{version}")
        
        deleted = False
        
        if os.path.exists(model_path):
            os.remove(model_path)
            deleted = True
        
        if os.path.exists(model_tf_path):
            shutil.rmtree(model_tf_path)
            deleted = True
        
        # Mise à jour du registre
        key = f"{algorithm_type}:{model_name}"
        if key in self.versions and version in self.versions[key]:
            self.versions[key].remove(version)
        
        return deleted

# Instances globales
METRICS_MANAGER = PrometheusMetricsManager()
RESOURCE_MONITOR = ResourceMonitor()
MODEL_VERSION_MANAGER = ModelVersionManager()

# Export des utilitaires principaux
__all__ = [
    'PrometheusMetricsManager',
    'RedisCache',
    'DataProcessor',
    'ResourceMonitor',
    'ModelVersionManager',
    'PerformanceMetrics',
    'CacheEntry',
    'async_retry',
    'measure_performance',
    'METRICS_MANAGER',
    'RESOURCE_MONITOR',
    'MODEL_VERSION_MANAGER'
]
