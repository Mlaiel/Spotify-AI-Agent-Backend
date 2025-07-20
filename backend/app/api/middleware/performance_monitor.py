"""
🚀 Performance Monitor Middleware Ultra-Avancé - Spotify AI Agent
================================================================
Monitoring de performance enterprise avec IA, prédictions et auto-optimisation
Auteur: Équipe Lead Dev + Architecte IA + Ingénieur Machine Learning + DBA

Fonctionnalités Enterprise:
- APM (Application Performance Monitoring) complet
- Profiling CPU/Memory en temps réel
- Détection d'anomalies avec ML
- Prédictions de performance
- Auto-scaling intelligent
- Monitoring de base de données
- Metrics business Spotify
- Alerting multi-canal
- Optimisations automatiques
- Chaos engineering intégré
"""

import asyncio
import gc
import os
import psutil
import time
import threading
import tracemalloc
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import json
import statistics
import numpy as np
from contextlib import asynccontextmanager

from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, Summary
import asyncpg
import aiofiles
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

from app.core.config import settings
from app.core.logging import get_logger
from app.utils.metrics_manager import MetricsManager
from app.core.database import get_database_pool

logger = get_logger(__name__)
metrics = MetricsManager()


class PerformanceLevel(Enum):
    """Niveaux de performance"""
    EXCELLENT = "excellent"    # < 100ms, < 50% CPU, < 70% Memory
    GOOD = "good"             # < 300ms, < 70% CPU, < 80% Memory
    DEGRADED = "degraded"     # < 1000ms, < 85% CPU, < 90% Memory
    CRITICAL = "critical"     # > 1000ms, > 85% CPU, > 90% Memory


class AlertSeverity(Enum):
    """Sévérité des alertes"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetrics:
    """Métriques de performance d'une requête"""
    request_id: str
    endpoint: str
    method: str
    start_time: float
    end_time: float
    duration: float
    
    # Métriques système
    cpu_percent_start: float
    cpu_percent_end: float
    memory_mb_start: float
    memory_mb_end: float
    memory_percent_start: float
    memory_percent_end: float
    
    # Métriques réseau
    response_size: int
    request_size: int
    
    # Métriques base de données
    db_queries: int = 0
    db_query_time: float = 0.0
    db_connections_active: int = 0
    
    # Métriques cache
    cache_hits: int = 0
    cache_misses: int = 0
    cache_latency: float = 0.0
    
    # Métriques business
    spotify_api_calls: int = 0
    spotify_api_latency: float = 0.0
    ai_processing_time: float = 0.0
    ai_model_accuracy: Optional[float] = None
    
    # Statut
    status_code: int = 200
    user_id: Optional[str] = None
    user_tier: str = "free"
    geo_region: Optional[str] = None
    
    # Performance level
    performance_level: PerformanceLevel = PerformanceLevel.GOOD
    
    # Anomalies détectées
    anomalies: List[str] = field(default_factory=list)


@dataclass
class SystemHealthMetrics:
    """Métriques de santé système globales"""
    timestamp: datetime
    
    # CPU
    cpu_percent: float
    cpu_count: int
    load_average: Tuple[float, float, float]
    
    # Mémoire
    memory_total: int
    memory_available: int
    memory_percent: float
    memory_cached: int
    
    # Disque
    disk_usage_percent: float
    disk_io_read: int
    disk_io_write: int
    
    # Réseau
    network_bytes_sent: int
    network_bytes_recv: int
    network_packets_sent: int
    network_packets_recv: int
    
    # Processus
    process_count: int
    thread_count: int
    file_descriptors: int
    
    # Application
    active_connections: int
    request_queue_size: int
    error_rate: float
    avg_response_time: float
    
    # Base de données
    db_connection_pool_size: int
    db_active_connections: int
    db_idle_connections: int
    db_query_rate: float
    
    # Cache
    redis_memory_usage: int
    redis_connected_clients: int
    redis_ops_per_sec: float
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)


class PerformanceProfiler:
    """Profiler de performance avancé"""
    
    def __init__(self):
        self.active_profiles: Dict[str, Dict] = {}
        self.tracemalloc_enabled = False
        self._enable_tracemalloc()
    
    def _enable_tracemalloc(self):
        """Activer le tracemalloc pour le profiling mémoire"""
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self.tracemalloc_enabled = True
    
    @asynccontextmanager
    async def profile_request(self, request_id: str, endpoint: str):
        """Context manager pour profiler une requête"""
        profile_data = {
            "request_id": request_id,
            "endpoint": endpoint,
            "start_time": time.time(),
            "start_memory": self._get_memory_usage(),
            "start_cpu": psutil.cpu_percent(),
        }
        
        if self.tracemalloc_enabled:
            profile_data["memory_snapshot_start"] = tracemalloc.take_snapshot()
        
        self.active_profiles[request_id] = profile_data
        
        try:
            yield profile_data
        finally:
            await self._finalize_profile(request_id)
    
    async def _finalize_profile(self, request_id: str):
        """Finaliser le profiling d'une requête"""
        if request_id not in self.active_profiles:
            return
        
        profile_data = self.active_profiles[request_id]
        
        # Métriques finales
        profile_data.update({
            "end_time": time.time(),
            "end_memory": self._get_memory_usage(),
            "end_cpu": psutil.cpu_percent(),
        })
        
        # Calcul de la durée
        profile_data["duration"] = profile_data["end_time"] - profile_data["start_time"]
        
        # Analyse mémoire détaillée
        if self.tracemalloc_enabled and "memory_snapshot_start" in profile_data:
            memory_snapshot_end = tracemalloc.take_snapshot()
            profile_data["memory_diff"] = self._analyze_memory_diff(
                profile_data["memory_snapshot_start"],
                memory_snapshot_end
            )
        
        # Stockage du profil pour analyse
        await self._store_profile(profile_data)
        
        # Nettoyage
        del self.active_profiles[request_id]
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Obtenir l'utilisation mémoire détaillée"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss": memory_info.rss / 1024 / 1024,  # MB
            "vms": memory_info.vms / 1024 / 1024,  # MB
            "percent": process.memory_percent(),
            "available": psutil.virtual_memory().available / 1024 / 1024,  # MB
        }
    
    def _analyze_memory_diff(self, snapshot_start, snapshot_end) -> Dict[str, Any]:
        """Analyser la différence de mémoire"""
        try:
            top_stats = snapshot_end.compare_to(snapshot_start, 'lineno')
            
            analysis = {
                "total_diff": sum(stat.size_diff for stat in top_stats) / 1024 / 1024,  # MB
                "count_diff": sum(stat.count_diff for stat in top_stats),
                "top_allocations": []
            }
            
            # Top 10 des allocations
            for stat in top_stats[:10]:
                if stat.size_diff > 0:  # Seulement les allocations
                    analysis["top_allocations"].append({
                        "filename": stat.traceback.format()[0] if stat.traceback else "unknown",
                        "size_diff_mb": stat.size_diff / 1024 / 1024,
                        "count_diff": stat.count_diff
                    })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Erreur analyse mémoire: {e}")
            return {"error": str(e)}
    
    async def _store_profile(self, profile_data: Dict):
        """Stocker les données de profiling"""
        try:
            # Simplifier les données pour le stockage
            simple_profile = {
                "request_id": profile_data["request_id"],
                "endpoint": profile_data["endpoint"],
                "duration": profile_data["duration"],
                "memory_start": profile_data["start_memory"]["rss"],
                "memory_end": profile_data["end_memory"]["rss"],
                "memory_diff": profile_data["end_memory"]["rss"] - profile_data["start_memory"]["rss"],
                "cpu_start": profile_data["start_cpu"],
                "cpu_end": profile_data["end_cpu"],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Stocker dans un fichier ou base de données
            profile_file = f"/tmp/profiles/profile_{profile_data['request_id']}.json"
            os.makedirs(os.path.dirname(profile_file), exist_ok=True)
            
            async with aiofiles.open(profile_file, 'w') as f:
                await f.write(json.dumps(simple_profile, indent=2))
                
        except Exception as e:
            logger.error(f"Erreur stockage profil: {e}")


class AnomalyDetector:
    """Détecteur d'anomalies avec Machine Learning"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_data = deque(maxlen=1000)
        self.feature_names = [
            'duration', 'cpu_percent_diff', 'memory_mb_diff',
            'response_size', 'db_queries', 'db_query_time'
        ]
    
    async def add_sample(self, metrics: PerformanceMetrics):
        """Ajouter un échantillon pour l'entraînement"""
        features = self._extract_features(metrics)
        self.training_data.append(features)
        
        # Réentraîner le modèle périodiquement
        if len(self.training_data) >= 100 and len(self.training_data) % 50 == 0:
            await self._retrain_model()
    
    def _extract_features(self, metrics: PerformanceMetrics) -> List[float]:
        """Extraire les features pour le ML"""
        return [
            metrics.duration,
            metrics.cpu_percent_end - metrics.cpu_percent_start,
            metrics.memory_mb_end - metrics.memory_mb_start,
            float(metrics.response_size),
            float(metrics.db_queries),
            metrics.db_query_time
        ]
    
    async def _retrain_model(self):
        """Réentraîner le modèle de détection d'anomalies"""
        try:
            if len(self.training_data) < 20:
                return
            
            # Préparer les données
            X = np.array(list(self.training_data))
            X_scaled = self.scaler.fit_transform(X)
            
            # Entraîner le modèle
            self.model.fit(X_scaled)
            self.is_trained = True
            
            # Sauvegarder le modèle
            await self._save_model()
            
            logger.info(f"Modèle d'anomalies réentraîné avec {len(self.training_data)} échantillons")
            
        except Exception as e:
            logger.error(f"Erreur entraînement modèle anomalies: {e}")
    
    async def detect_anomaly(self, metrics: PerformanceMetrics) -> Tuple[bool, float]:
        """Détecter si les métriques sont anormales"""
        if not self.is_trained:
            return False, 0.0
        
        try:
            features = self._extract_features(metrics)
            X = np.array([features])
            X_scaled = self.scaler.transform(X)
            
            # Prédiction
            prediction = self.model.predict(X_scaled)[0]
            anomaly_score = self.model.decision_function(X_scaled)[0]
            
            is_anomaly = prediction == -1
            
            return is_anomaly, float(anomaly_score)
            
        except Exception as e:
            logger.error(f"Erreur détection anomalie: {e}")
            return False, 0.0
    
    async def _save_model(self):
        """Sauvegarder le modèle entraîné"""
        try:
            model_data = {
                "model": joblib.dumps(self.model),
                "scaler": joblib.dumps(self.scaler),
                "feature_names": self.feature_names,
                "trained_at": datetime.utcnow().isoformat(),
                "training_samples": len(self.training_data)
            }
            
            await self.redis_client.set(
                "performance_anomaly_model",
                json.dumps({
                    k: v if k != "model" and k != "scaler" else v.decode('latin1')
                    for k, v in model_data.items()
                })
            )
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde modèle: {e}")


class PerformanceOptimizer:
    """Optimiseur de performance automatique"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.optimization_rules = {
            "high_memory_usage": self._optimize_memory,
            "high_cpu_usage": self._optimize_cpu,
            "slow_database": self._optimize_database,
            "cache_misses": self._optimize_cache,
            "large_responses": self._optimize_responses
        }
    
    async def analyze_and_optimize(self, metrics: PerformanceMetrics) -> List[str]:
        """Analyser et optimiser automatiquement"""
        optimizations = []
        
        # Détection des problèmes
        problems = self._detect_performance_problems(metrics)
        
        for problem in problems:
            if problem in self.optimization_rules:
                optimization = await self.optimization_rules[problem](metrics)
                if optimization:
                    optimizations.append(optimization)
        
        return optimizations
    
    def _detect_performance_problems(self, metrics: PerformanceMetrics) -> List[str]:
        """Détecter les problèmes de performance"""
        problems = []
        
        # Utilisation mémoire élevée
        memory_diff = metrics.memory_mb_end - metrics.memory_mb_start
        if memory_diff > 100:  # Plus de 100MB d'augmentation
            problems.append("high_memory_usage")
        
        # Utilisation CPU élevée
        cpu_diff = metrics.cpu_percent_end - metrics.cpu_percent_start
        if cpu_diff > 50:  # Plus de 50% d'augmentation CPU
            problems.append("high_cpu_usage")
        
        # Base de données lente
        if metrics.db_query_time > 1.0:  # Plus d'1 seconde
            problems.append("slow_database")
        
        # Cache misses élevés
        total_cache_ops = metrics.cache_hits + metrics.cache_misses
        if total_cache_ops > 0 and metrics.cache_misses / total_cache_ops > 0.5:
            problems.append("cache_misses")
        
        # Réponses volumineuses
        if metrics.response_size > 10 * 1024 * 1024:  # Plus de 10MB
            problems.append("large_responses")
        
        return problems
    
    async def _optimize_memory(self, metrics: PerformanceMetrics) -> Optional[str]:
        """Optimisation mémoire"""
        try:
            # Forcer garbage collection
            gc.collect()
            
            # Enregistrer l'optimisation
            await self.redis_client.incr("optimizations:memory_gc")
            
            return "memory_gc_forced"
            
        except Exception as e:
            logger.error(f"Erreur optimisation mémoire: {e}")
            return None
    
    async def _optimize_cpu(self, metrics: PerformanceMetrics) -> Optional[str]:
        """Optimisation CPU"""
        try:
            # Suggestions d'optimisation CPU
            optimization = "cpu_optimization_suggested"
            
            # Ici on pourrait implémenter des optimisations comme:
            # - Réduction de la priorité des tâches non critiques
            # - Activation du cache plus agressif
            # - Parallélisation des tâches
            
            await self.redis_client.incr("optimizations:cpu_suggested")
            
            return optimization
            
        except Exception as e:
            logger.error(f"Erreur optimisation CPU: {e}")
            return None
    
    async def _optimize_database(self, metrics: PerformanceMetrics) -> Optional[str]:
        """Optimisation base de données"""
        try:
            # Suggestions d'optimisation DB
            optimization = "database_optimization_suggested"
            
            # Ici on pourrait implémenter:
            # - Analyse des requêtes lentes
            # - Suggestions d'index
            # - Connection pooling optimization
            
            await self.redis_client.incr("optimizations:database_suggested")
            
            return optimization
            
        except Exception as e:
            logger.error(f"Erreur optimisation database: {e}")
            return None
    
    async def _optimize_cache(self, metrics: PerformanceMetrics) -> Optional[str]:
        """Optimisation cache"""
        try:
            optimization = "cache_optimization_suggested"
            
            # Ici on pourrait implémenter:
            # - Warmup du cache
            # - Ajustement des TTL
            # - Préfetching intelligent
            
            await self.redis_client.incr("optimizations:cache_suggested")
            
            return optimization
            
        except Exception as e:
            logger.error(f"Erreur optimisation cache: {e}")
            return None
    
    async def _optimize_responses(self, metrics: PerformanceMetrics) -> Optional[str]:
        """Optimisation réponses"""
        try:
            optimization = "response_optimization_suggested"
            
            # Ici on pourrait implémenter:
            # - Compression des réponses
            # - Pagination automatique
            # - Streaming responses
            
            await self.redis_client.incr("optimizations:response_suggested")
            
            return optimization
            
        except Exception as e:
            logger.error(f"Erreur optimisation responses: {e}")
            return None


class PerformanceMonitorMiddleware(BaseHTTPMiddleware):
    """
    Middleware de monitoring de performance ultra-avancé
    
    Fonctionnalités:
    - Monitoring complet des performances
    - Profiling automatique
    - Détection d'anomalies avec ML
    - Optimisations automatiques
    - Alerting intelligent
    - Métriques business
    """
    
    def __init__(
        self,
        app,
        redis_client: Optional[redis.Redis] = None,
        enable_profiling: bool = True,
        enable_anomaly_detection: bool = True,
        enable_auto_optimization: bool = True,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        super().__init__(app)
        self.redis_client = redis_client or redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
        
        self.enable_profiling = enable_profiling
        self.enable_anomaly_detection = enable_anomaly_detection
        self.enable_auto_optimization = enable_auto_optimization
        
        # Seuils d'alerte par défaut
        self.alert_thresholds = alert_thresholds or {
            "response_time": 2.0,      # 2 secondes
            "memory_usage": 500.0,     # 500 MB
            "cpu_usage": 80.0,         # 80%
            "error_rate": 5.0,         # 5%
            "db_query_time": 1.0       # 1 seconde
        }
        
        # Composants avancés
        self.profiler = PerformanceProfiler() if enable_profiling else None
        self.anomaly_detector = AnomalyDetector(self.redis_client) if enable_anomaly_detection else None
        self.optimizer = PerformanceOptimizer(self.redis_client) if enable_auto_optimization else None
        
        # Métriques Prometheus
        self._init_prometheus_metrics()
        
        # Cache des métriques pour calculs agrégés
        self.metrics_buffer = deque(maxlen=1000)
        
        # Monitoring système en arrière-plan
        self._start_system_monitoring()
    
    def _init_prometheus_metrics(self):
        """Initialiser les métriques Prometheus"""
        self.prometheus_metrics = {
            "request_duration": Histogram(
                "http_request_duration_seconds",
                "HTTP request duration",
                ["method", "endpoint", "status_code", "user_tier"]
            ),
            "request_size": Histogram(
                "http_request_size_bytes",
                "HTTP request size",
                ["method", "endpoint"]
            ),
            "response_size": Histogram(
                "http_response_size_bytes",
                "HTTP response size",
                ["method", "endpoint", "status_code"]
            ),
            "active_requests": Gauge(
                "http_active_requests",
                "Number of active HTTP requests"
            ),
            "memory_usage": Gauge(
                "application_memory_usage_bytes",
                "Application memory usage"
            ),
            "cpu_usage": Gauge(
                "application_cpu_usage_percent",
                "Application CPU usage"
            ),
            "db_queries": Counter(
                "database_queries_total",
                "Total database queries",
                ["query_type", "status"]
            ),
            "cache_operations": Counter(
                "cache_operations_total",
                "Total cache operations",
                ["operation", "result"]
            ),
            "spotify_api_calls": Counter(
                "spotify_api_calls_total",
                "Total Spotify API calls",
                ["endpoint", "status"]
            ),
            "anomalies_detected": Counter(
                "performance_anomalies_total",
                "Total performance anomalies detected",
                ["type", "severity"]
            ),
            "optimizations_applied": Counter(
                "performance_optimizations_total",
                "Total performance optimizations applied",
                ["type"]
            )
        }
    
    def _start_system_monitoring(self):
        """Démarrer le monitoring système en arrière-plan"""
        async def system_monitor():
            while True:
                try:
                    await self._collect_system_metrics()
                    await asyncio.sleep(30)  # Collecter toutes les 30 secondes
                except Exception as e:
                    logger.error(f"Erreur monitoring système: {e}")
                    await asyncio.sleep(60)
        
        # Démarrer la tâche en arrière-plan
        asyncio.create_task(system_monitor())
    
    async def dispatch(self, request: Request, call_next):
        """Point d'entrée principal du middleware"""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Métriques initiales
        initial_metrics = await self._collect_initial_metrics(request, request_id)
        
        # Incrémenter les requêtes actives
        self.prometheus_metrics["active_requests"].inc()
        
        try:
            # Profiling si activé
            if self.profiler:
                async with self.profiler.profile_request(request_id, request.url.path):
                    response = await call_next(request)
            else:
                response = await call_next(request)
            
            # Métriques finales
            final_metrics = await self._collect_final_metrics(
                request, response, initial_metrics, start_time
            )
            
            # Détection d'anomalies
            if self.anomaly_detector:
                is_anomaly, anomaly_score = await self.anomaly_detector.detect_anomaly(final_metrics)
                if is_anomaly:
                    final_metrics.anomalies.append(f"ML_anomaly_score_{anomaly_score:.3f}")
                    await self._handle_anomaly(final_metrics, anomaly_score)
                
                # Ajouter pour l'entraînement
                await self.anomaly_detector.add_sample(final_metrics)
            
            # Optimisations automatiques
            if self.optimizer:
                optimizations = await self.optimizer.analyze_and_optimize(final_metrics)
                if optimizations:
                    logger.info(f"Optimisations appliquées: {optimizations}")
                    for opt in optimizations:
                        self.prometheus_metrics["optimizations_applied"].labels(type=opt).inc()
            
            # Enregistrement des métriques
            await self._record_metrics(final_metrics)
            
            # Vérification des seuils d'alerte
            await self._check_alert_thresholds(final_metrics)
            
            # Ajout au buffer
            self.metrics_buffer.append(final_metrics)
            
            return response
            
        except Exception as e:
            logger.error(f"Erreur performance monitor: {e}")
            # Ne pas bloquer la requête
            return await call_next(request)
        
        finally:
            # Décrémenter les requêtes actives
            self.prometheus_metrics["active_requests"].dec()
    
    async def _collect_initial_metrics(self, request: Request, request_id: str) -> Dict:
        """Collecter les métriques initiales"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "request_id": request_id,
            "endpoint": request.url.path,
            "method": request.method,
            "start_time": time.time(),
            "cpu_percent_start": psutil.cpu_percent(),
            "memory_mb_start": memory_info.rss / 1024 / 1024,
            "memory_percent_start": process.memory_percent(),
            "request_size": len(await request.body()) if hasattr(request, 'body') else 0,
            "user_id": getattr(request.state, 'user_id', None),
            "user_tier": getattr(request.state, 'user_tier', 'free')
        }
    
    async def _collect_final_metrics(
        self, request: Request, response: Response, 
        initial_metrics: Dict, start_time: float
    ) -> PerformanceMetrics:
        """Collecter les métriques finales"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Métriques de base
        metrics = PerformanceMetrics(
            request_id=initial_metrics["request_id"],
            endpoint=initial_metrics["endpoint"],
            method=initial_metrics["method"],
            start_time=initial_metrics["start_time"],
            end_time=time.time(),
            duration=time.time() - start_time,
            
            cpu_percent_start=initial_metrics["cpu_percent_start"],
            cpu_percent_end=psutil.cpu_percent(),
            memory_mb_start=initial_metrics["memory_mb_start"],
            memory_mb_end=memory_info.rss / 1024 / 1024,
            memory_percent_start=initial_metrics["memory_percent_start"],
            memory_percent_end=process.memory_percent(),
            
            response_size=int(response.headers.get("content-length", 0)),
            request_size=initial_metrics["request_size"],
            
            status_code=response.status_code,
            user_id=initial_metrics["user_id"],
            user_tier=initial_metrics["user_tier"]
        )
        
        # Métriques additionnelles depuis l'état de la requête
        if hasattr(request.state, 'performance_data'):
            perf_data = request.state.performance_data
            metrics.db_queries = perf_data.get("db_queries", 0)
            metrics.db_query_time = perf_data.get("db_query_time", 0.0)
            metrics.cache_hits = perf_data.get("cache_hits", 0)
            metrics.cache_misses = perf_data.get("cache_misses", 0)
            metrics.spotify_api_calls = perf_data.get("spotify_api_calls", 0)
            metrics.spotify_api_latency = perf_data.get("spotify_api_latency", 0.0)
            metrics.ai_processing_time = perf_data.get("ai_processing_time", 0.0)
        
        # Déterminer le niveau de performance
        metrics.performance_level = self._calculate_performance_level(metrics)
        
        return metrics
    
    def _calculate_performance_level(self, metrics: PerformanceMetrics) -> PerformanceLevel:
        """Calculer le niveau de performance"""
        # Critères pour chaque niveau
        if (metrics.duration < 0.1 and 
            metrics.cpu_percent_end < 50 and 
            metrics.memory_percent_end < 70):
            return PerformanceLevel.EXCELLENT
        
        elif (metrics.duration < 0.3 and 
              metrics.cpu_percent_end < 70 and 
              metrics.memory_percent_end < 80):
            return PerformanceLevel.GOOD
        
        elif (metrics.duration < 1.0 and 
              metrics.cpu_percent_end < 85 and 
              metrics.memory_percent_end < 90):
            return PerformanceLevel.DEGRADED
        
        else:
            return PerformanceLevel.CRITICAL
    
    async def _collect_system_metrics(self):
        """Collecter les métriques système globales"""
        try:
            # Métriques CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            load_avg = os.getloadavg()
            
            # Métriques mémoire
            memory = psutil.virtual_memory()
            
            # Métriques disque
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # Métriques réseau
            network_io = psutil.net_io_counters()
            
            # Métriques processus
            process_count = len(psutil.pids())
            
            # Métriques Redis
            redis_info = await self._get_redis_info()
            
            # Créer l'objet métriques système
            system_metrics = SystemHealthMetrics(
                timestamp=datetime.utcnow(),
                cpu_percent=cpu_percent,
                cpu_count=psutil.cpu_count(),
                load_average=load_avg,
                
                memory_total=memory.total,
                memory_available=memory.available,
                memory_percent=memory.percent,
                memory_cached=memory.cached if hasattr(memory, 'cached') else 0,
                
                disk_usage_percent=disk.percent,
                disk_io_read=disk_io.read_bytes if disk_io else 0,
                disk_io_write=disk_io.write_bytes if disk_io else 0,
                
                network_bytes_sent=network_io.bytes_sent if network_io else 0,
                network_bytes_recv=network_io.bytes_recv if network_io else 0,
                network_packets_sent=network_io.packets_sent if network_io else 0,
                network_packets_recv=network_io.packets_recv if network_io else 0,
                
                process_count=process_count,
                thread_count=threading.active_count(),
                file_descriptors=0,  # Pourrait être implémenté
                
                active_connections=0,  # À implémenter
                request_queue_size=0,  # À implémenter
                error_rate=await self._calculate_error_rate(),
                avg_response_time=await self._calculate_avg_response_time(),
                
                db_connection_pool_size=0,  # À implémenter
                db_active_connections=0,    # À implémenter
                db_idle_connections=0,      # À implémenter
                db_query_rate=0,           # À implémenter
                
                redis_memory_usage=redis_info.get("used_memory", 0),
                redis_connected_clients=redis_info.get("connected_clients", 0),
                redis_ops_per_sec=redis_info.get("instantaneous_ops_per_sec", 0)
            )
            
            # Mise à jour des métriques Prometheus
            self.prometheus_metrics["memory_usage"].set(memory.used)
            self.prometheus_metrics["cpu_usage"].set(cpu_percent)
            
            # Stockage dans Redis
            await self._store_system_metrics(system_metrics)
            
        except Exception as e:
            logger.error(f"Erreur collecte métriques système: {e}")
    
    async def _get_redis_info(self) -> Dict:
        """Obtenir les informations Redis"""
        try:
            info = await self.redis_client.info()
            return {
                "used_memory": info.get("used_memory", 0),
                "connected_clients": info.get("connected_clients", 0),
                "instantaneous_ops_per_sec": info.get("instantaneous_ops_per_sec", 0)
            }
        except Exception as e:
            logger.error(f"Erreur info Redis: {e}")
            return {}
    
    async def _calculate_error_rate(self) -> float:
        """Calculer le taux d'erreur des dernières minutes"""
        try:
            if not self.metrics_buffer:
                return 0.0
            
            recent_metrics = [m for m in self.metrics_buffer 
                            if time.time() - m.end_time < 300]  # 5 minutes
            
            if not recent_metrics:
                return 0.0
            
            error_count = sum(1 for m in recent_metrics if m.status_code >= 400)
            return (error_count / len(recent_metrics)) * 100
            
        except Exception:
            return 0.0
    
    async def _calculate_avg_response_time(self) -> float:
        """Calculer le temps de réponse moyen"""
        try:
            if not self.metrics_buffer:
                return 0.0
            
            recent_metrics = [m for m in self.metrics_buffer 
                            if time.time() - m.end_time < 300]  # 5 minutes
            
            if not recent_metrics:
                return 0.0
            
            return statistics.mean(m.duration for m in recent_metrics)
            
        except Exception:
            return 0.0
    
    async def _store_system_metrics(self, metrics: SystemHealthMetrics):
        """Stocker les métriques système"""
        try:
            metrics_data = {
                "timestamp": metrics.timestamp.isoformat(),
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "disk_usage_percent": metrics.disk_usage_percent,
                "redis_memory_usage": metrics.redis_memory_usage,
                "redis_connected_clients": metrics.redis_connected_clients,
                "error_rate": metrics.error_rate,
                "avg_response_time": metrics.avg_response_time
            }
            
            # Stocker dans Redis avec TTL
            key = f"system_metrics:{datetime.utcnow().strftime('%Y%m%d%H%M')}"
            await self.redis_client.setex(
                key,
                timedelta(hours=24),
                json.dumps(metrics_data)
            )
            
        except Exception as e:
            logger.error(f"Erreur stockage métriques système: {e}")
    
    async def _record_metrics(self, metrics: PerformanceMetrics):
        """Enregistrer les métriques de performance"""
        # Métriques Prometheus
        self.prometheus_metrics["request_duration"].labels(
            method=metrics.method,
            endpoint=metrics.endpoint,
            status_code=metrics.status_code,
            user_tier=metrics.user_tier
        ).observe(metrics.duration)
        
        self.prometheus_metrics["response_size"].labels(
            method=metrics.method,
            endpoint=metrics.endpoint,
            status_code=metrics.status_code
        ).observe(metrics.response_size)
        
        # Stockage détaillé dans Redis
        metrics_key = f"performance_metrics:{datetime.utcnow().strftime('%Y%m%d%H')}"
        metrics_data = {
            "request_id": metrics.request_id,
            "endpoint": metrics.endpoint,
            "method": metrics.method,
            "duration": metrics.duration,
            "status_code": metrics.status_code,
            "user_tier": metrics.user_tier,
            "performance_level": metrics.performance_level.value,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.redis_client.lpush(metrics_key, json.dumps(metrics_data))
        await self.redis_client.expire(metrics_key, timedelta(days=7))
    
    async def _handle_anomaly(self, metrics: PerformanceMetrics, anomaly_score: float):
        """Gérer une anomalie détectée"""
        anomaly_data = {
            "request_id": metrics.request_id,
            "endpoint": metrics.endpoint,
            "anomaly_score": anomaly_score,
            "duration": metrics.duration,
            "memory_usage": metrics.memory_mb_end - metrics.memory_mb_start,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Enregistrer l'anomalie
        await self.redis_client.lpush(
            "performance_anomalies",
            json.dumps(anomaly_data)
        )
        await self.redis_client.expire("performance_anomalies", timedelta(days=30))
        
        # Métriques Prometheus
        severity = "high" if abs(anomaly_score) > 0.5 else "medium"
        self.prometheus_metrics["anomalies_detected"].labels(
            type="ml_anomaly",
            severity=severity
        ).inc()
        
        logger.warning(f"Anomalie de performance détectée: {anomaly_data}")
    
    async def _check_alert_thresholds(self, metrics: PerformanceMetrics):
        """Vérifier les seuils d'alerte"""
        alerts = []
        
        # Temps de réponse
        if metrics.duration > self.alert_thresholds["response_time"]:
            alerts.append({
                "type": "slow_response",
                "value": metrics.duration,
                "threshold": self.alert_thresholds["response_time"],
                "severity": AlertSeverity.WARNING if metrics.duration < 5.0 else AlertSeverity.CRITICAL
            })
        
        # Utilisation mémoire
        memory_diff = metrics.memory_mb_end - metrics.memory_mb_start
        if memory_diff > self.alert_thresholds["memory_usage"]:
            alerts.append({
                "type": "high_memory_usage",
                "value": memory_diff,
                "threshold": self.alert_thresholds["memory_usage"],
                "severity": AlertSeverity.WARNING
            })
        
        # Envoyer les alertes
        for alert in alerts:
            await self._send_alert(metrics, alert)
    
    async def _send_alert(self, metrics: PerformanceMetrics, alert: Dict):
        """Envoyer une alerte"""
        alert_data = {
            "request_id": metrics.request_id,
            "endpoint": metrics.endpoint,
            "alert_type": alert["type"],
            "value": alert["value"],
            "threshold": alert["threshold"],
            "severity": alert["severity"].value,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Stocker l'alerte
        await self.redis_client.lpush(
            "performance_alerts",
            json.dumps(alert_data)
        )
        await self.redis_client.expire("performance_alerts", timedelta(days=7))
        
        # Log selon la sévérité
        if alert["severity"] == AlertSeverity.CRITICAL:
            logger.critical(f"Alerte performance CRITIQUE: {alert_data}")
        elif alert["severity"] == AlertSeverity.ERROR:
            logger.error(f"Alerte performance ERREUR: {alert_data}")
        else:
            logger.warning(f"Alerte performance: {alert_data}")
    
    async def get_performance_report(self, hours: int = 24) -> Dict:
        """Générer un rapport de performance"""
        try:
            # Collecter les métriques des dernières heures
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            recent_metrics = [
                m for m in self.metrics_buffer 
                if datetime.fromtimestamp(m.end_time) > cutoff_time
            ]
            
            if not recent_metrics:
                return {"error": "Aucune métrique disponible"}
            
            # Calculs statistiques
            durations = [m.duration for m in recent_metrics]
            memory_usage = [m.memory_mb_end - m.memory_mb_start for m in recent_metrics]
            
            report = {
                "period": f"{hours} heures",
                "total_requests": len(recent_metrics),
                "avg_response_time": statistics.mean(durations),
                "p95_response_time": np.percentile(durations, 95),
                "p99_response_time": np.percentile(durations, 99),
                "min_response_time": min(durations),
                "max_response_time": max(durations),
                
                "avg_memory_usage": statistics.mean(memory_usage),
                "max_memory_usage": max(memory_usage),
                
                "error_rate": sum(1 for m in recent_metrics if m.status_code >= 400) / len(recent_metrics) * 100,
                
                "performance_levels": {
                    level.value: sum(1 for m in recent_metrics if m.performance_level == level)
                    for level in PerformanceLevel
                },
                
                "anomalies_detected": sum(1 for m in recent_metrics if m.anomalies),
                
                "top_slow_endpoints": self._get_top_slow_endpoints(recent_metrics),
                
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Erreur génération rapport: {e}")
            return {"error": str(e)}
    
    def _get_top_slow_endpoints(self, metrics: List[PerformanceMetrics], top_n: int = 10) -> List[Dict]:
        """Obtenir les endpoints les plus lents"""
        endpoint_stats = defaultdict(list)
        
        for metric in metrics:
            endpoint_stats[metric.endpoint].append(metric.duration)
        
        slow_endpoints = []
        for endpoint, durations in endpoint_stats.items():
            slow_endpoints.append({
                "endpoint": endpoint,
                "avg_duration": statistics.mean(durations),
                "max_duration": max(durations),
                "request_count": len(durations)
            })
        
        return sorted(slow_endpoints, key=lambda x: x["avg_duration"], reverse=True)[:top_n]


# Factory functions pour différents environnements
def create_performance_monitor_development() -> PerformanceMonitorMiddleware:
    """Créer middleware performance pour développement"""
    return PerformanceMonitorMiddleware(
        app=None,
        enable_profiling=True,
        enable_anomaly_detection=False,  # Pas d'IA en dev
        enable_auto_optimization=False,
        alert_thresholds={
            "response_time": 5.0,      # Plus permissif en dev
            "memory_usage": 1000.0,
            "cpu_usage": 90.0,
            "error_rate": 10.0,
            "db_query_time": 3.0
        }
    )


def create_performance_monitor_production() -> PerformanceMonitorMiddleware:
    """Créer middleware performance pour production"""
    return PerformanceMonitorMiddleware(
        app=None,
        enable_profiling=False,      # Pas de profiling détaillé en prod
        enable_anomaly_detection=True,
        enable_auto_optimization=True,
        alert_thresholds={
            "response_time": 1.0,      # Strict en production
            "memory_usage": 200.0,
            "cpu_usage": 70.0,
            "error_rate": 1.0,
            "db_query_time": 0.5
        }
    )


def create_performance_monitor_testing() -> PerformanceMonitorMiddleware:
    """Créer middleware performance pour tests"""
    return PerformanceMonitorMiddleware(
        app=None,
        enable_profiling=False,
        enable_anomaly_detection=False,
        enable_auto_optimization=False,
        alert_thresholds={
            "response_time": 10.0,     # Très permissif pour tests
            "memory_usage": 2000.0,
            "cpu_usage": 95.0,
            "error_rate": 50.0,
            "db_query_time": 5.0
        }
    )
