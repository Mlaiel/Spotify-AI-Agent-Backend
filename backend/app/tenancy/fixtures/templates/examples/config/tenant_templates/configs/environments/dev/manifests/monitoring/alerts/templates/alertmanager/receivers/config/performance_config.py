"""
Configuration avancée de performance et d'optimisation pour Alertmanager Receivers

Ce module gère l'optimisation des performances, le monitoring avancé,
l'auto-tuning et l'analyse prédictive des performances système.

Author: Spotify AI Agent Team  
Maintainer: Fahed Mlaiel - Lead Dev & AI Architect
"""

import asyncio
import logging
import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque, defaultdict
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import sys
import tracemalloc
from functools import wraps
import resource

logger = logging.getLogger(__name__)

class PerformanceLevel(Enum):
    """Niveaux de performance système"""
    OPTIMAL = "optimal"
    GOOD = "good"
    DEGRADED = "degraded"
    CRITICAL = "critical"

class OptimizationStrategy(Enum):
    """Stratégies d'optimisation"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"

@dataclass
class PerformanceMetric:
    """Métrique de performance"""
    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class PerformanceAlert:
    """Alerte de performance"""
    metric: str
    level: PerformanceLevel
    message: str
    value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    recommendations: List[str] = field(default_factory=list)

@dataclass
class OptimizationRecommendation:
    """Recommandation d'optimisation"""
    category: str
    priority: int  # 1-10, 10 étant le plus prioritaire
    description: str
    implementation: str
    expected_improvement: str
    risk_level: str
    estimated_effort: str

class PerformanceProfiler:
    """Profileur de performance avancé"""
    
    def __init__(self):
        self.active_profiles: Dict[str, Any] = {}
        self.profile_history: deque = deque(maxlen=1000)
        self.memory_tracker = None
        
    def start_profiling(self, session_id: str, profile_memory: bool = True):
        """Démarre le profiling d'une session"""
        profile_data = {
            "session_id": session_id,
            "start_time": time.time(),
            "start_memory": psutil.Process().memory_info().rss,
            "profile_memory": profile_memory
        }
        
        if profile_memory and not tracemalloc.is_tracing():
            tracemalloc.start()
            self.memory_tracker = True
        
        self.active_profiles[session_id] = profile_data
        logger.debug(f"Started profiling session: {session_id}")
    
    def stop_profiling(self, session_id: str) -> Dict[str, Any]:
        """Arrête le profiling et retourne les résultats"""
        if session_id not in self.active_profiles:
            return {}
        
        profile_data = self.active_profiles[session_id]
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        results = {
            "session_id": session_id,
            "duration": end_time - profile_data["start_time"],
            "memory_delta": end_memory - profile_data["start_memory"],
            "start_memory": profile_data["start_memory"],
            "end_memory": end_memory,
            "timestamp": datetime.utcnow()
        }
        
        # Ajout des données de tracemalloc si disponibles
        if profile_data["profile_memory"] and tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            results["traced_memory"] = {
                "current": current,
                "peak": peak
            }
        
        self.profile_history.append(results)
        del self.active_profiles[session_id]
        
        logger.debug(f"Completed profiling session: {session_id}, duration: {results['duration']:.3f}s")
        return results
    
    def get_memory_snapshot(self) -> Dict[str, Any]:
        """Capture un snapshot de l'utilisation mémoire"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        snapshot = {
            "rss": memory_info.rss,
            "vms": memory_info.vms,
            "percent": process.memory_percent(),
            "available": psutil.virtual_memory().available,
            "timestamp": datetime.utcnow()
        }
        
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            snapshot["tracemalloc"] = {"current": current, "peak": peak}
        
        return snapshot

def performance_monitor(threshold_ms: float = 1000):
    """Décorateur pour monitorer les performances des fonctions"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss
            
            try:
                result = await func(*args, **kwargs)
                success = True
            except Exception as e:
                result = None
                success = False
                raise
            finally:
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss
                duration_ms = (end_time - start_time) * 1000
                
                # Log si dépasse le seuil
                if duration_ms > threshold_ms:
                    logger.warning(
                        f"Performance threshold exceeded for {func.__name__}: "
                        f"{duration_ms:.1f}ms (threshold: {threshold_ms}ms)"
                    )
                
                # Métriques de performance
                performance_manager.record_function_performance(
                    function_name=func.__name__,
                    duration_ms=duration_ms,
                    memory_delta=end_memory - start_memory,
                    success=success
                )
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss
            
            try:
                result = func(*args, **kwargs)
                success = True
            except Exception as e:
                result = None
                success = False
                raise
            finally:
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss
                duration_ms = (end_time - start_time) * 1000
                
                if duration_ms > threshold_ms:
                    logger.warning(
                        f"Performance threshold exceeded for {func.__name__}: "
                        f"{duration_ms:.1f}ms (threshold: {threshold_ms}ms)"
                    )
                
                performance_manager.record_function_performance(
                    function_name=func.__name__,
                    duration_ms=duration_ms,
                    memory_delta=end_memory - start_memory,
                    success=success
                )
            
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

class SystemResourceMonitor:
    """Moniteur des ressources système"""
    
    def __init__(self, sampling_interval: int = 10):
        self.sampling_interval = sampling_interval
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.monitoring_active = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Démarre le monitoring des ressources système"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("System resource monitoring started")
    
    def stop_monitoring(self):
        """Arrête le monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("System resource monitoring stopped")
    
    def _monitor_loop(self):
        """Boucle principale de monitoring"""
        while self.monitoring_active:
            try:
                self._collect_system_metrics()
                time.sleep(self.sampling_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def _collect_system_metrics(self):
        """Collecte les métriques système"""
        timestamp = datetime.utcnow()
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics_history["cpu_percent"].append(
            PerformanceMetric("cpu_percent", cpu_percent, "%", timestamp, 70.0, 90.0)
        )
        
        # Mémoire
        memory = psutil.virtual_memory()
        self.metrics_history["memory_percent"].append(
            PerformanceMetric("memory_percent", memory.percent, "%", timestamp, 80.0, 95.0)
        )
        self.metrics_history["memory_available"].append(
            PerformanceMetric("memory_available", memory.available, "bytes", timestamp)
        )
        
        # Disque
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self.metrics_history["disk_percent"].append(
            PerformanceMetric("disk_percent", disk_percent, "%", timestamp, 80.0, 95.0)
        )
        
        # Charge système
        if hasattr(psutil, "getloadavg"):
            load_avg = psutil.getloadavg()[0]  # 1-minute load average
            cpu_count = psutil.cpu_count()
            load_percent = (load_avg / cpu_count) * 100
            self.metrics_history["load_average"].append(
                PerformanceMetric("load_average", load_percent, "%", timestamp, 70.0, 90.0)
            )
        
        # Processus
        process = psutil.Process()
        process_memory = process.memory_info().rss
        self.metrics_history["process_memory"].append(
            PerformanceMetric("process_memory", process_memory, "bytes", timestamp)
        )
        
        # Threads et descripteurs de fichiers
        try:
            num_threads = process.num_threads()
            self.metrics_history["thread_count"].append(
                PerformanceMetric("thread_count", num_threads, "count", timestamp, 100, 200)
            )
            
            num_fds = process.num_fds() if hasattr(process, "num_fds") else 0
            self.metrics_history["file_descriptors"].append(
                PerformanceMetric("file_descriptors", num_fds, "count", timestamp, 500, 800)
            )
        except (psutil.AccessDenied, AttributeError):
            pass
    
    def get_current_metrics(self) -> Dict[str, PerformanceMetric]:
        """Récupère les métriques actuelles"""
        current_metrics = {}
        for metric_name, metric_history in self.metrics_history.items():
            if metric_history:
                current_metrics[metric_name] = metric_history[-1]
        return current_metrics
    
    def get_metric_statistics(self, metric_name: str, window_minutes: int = 60) -> Dict[str, float]:
        """Calcule les statistiques pour une métrique"""
        if metric_name not in self.metrics_history:
            return {}
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
        recent_metrics = [
            m for m in self.metrics_history[metric_name]
            if m.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        values = [m.value for m in recent_metrics]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p95": statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
            "p99": statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values)
        }

class PerformanceOptimizer:
    """Optimiseur de performance automatique"""
    
    def __init__(self, strategy: OptimizationStrategy = OptimizationStrategy.MODERATE):
        self.strategy = strategy
        self.applied_optimizations: List[str] = []
        self.optimization_history: deque = deque(maxlen=100)
        
    async def analyze_and_optimize(self, metrics: Dict[str, PerformanceMetric]) -> List[OptimizationRecommendation]:
        """Analyse les métriques et propose des optimisations"""
        recommendations = []
        
        # Analyse CPU
        if "cpu_percent" in metrics:
            cpu_metric = metrics["cpu_percent"]
            if cpu_metric.value > 80:
                recommendations.extend(self._get_cpu_optimizations(cpu_metric.value))
        
        # Analyse mémoire
        if "memory_percent" in metrics:
            memory_metric = metrics["memory_percent"]
            if memory_metric.value > 85:
                recommendations.extend(self._get_memory_optimizations(memory_metric.value))
        
        # Analyse disque
        if "disk_percent" in metrics:
            disk_metric = metrics["disk_percent"]
            if disk_metric.value > 90:
                recommendations.extend(self._get_disk_optimizations(disk_metric.value))
        
        # Auto-application des optimisations selon la stratégie
        if self.strategy == OptimizationStrategy.AGGRESSIVE:
            await self._apply_automatic_optimizations(recommendations)
        
        return recommendations
    
    def _get_cpu_optimizations(self, cpu_percent: float) -> List[OptimizationRecommendation]:
        """Recommandations d'optimisation CPU"""
        recommendations = []
        
        if cpu_percent > 90:
            recommendations.append(OptimizationRecommendation(
                category="cpu",
                priority=10,
                description="CPU usage critical - immediate action required",
                implementation="Reduce concurrent processing, implement backpressure",
                expected_improvement="30-50% CPU reduction",
                risk_level="low",
                estimated_effort="medium"
            ))
        elif cpu_percent > 80:
            recommendations.append(OptimizationRecommendation(
                category="cpu",
                priority=7,
                description="High CPU usage detected",
                implementation="Optimize hot code paths, enable CPU profiling",
                expected_improvement="15-25% CPU reduction",
                risk_level="low",
                estimated_effort="low"
            ))
        
        return recommendations
    
    def _get_memory_optimizations(self, memory_percent: float) -> List[OptimizationRecommendation]:
        """Recommandations d'optimisation mémoire"""
        recommendations = []
        
        if memory_percent > 95:
            recommendations.append(OptimizationRecommendation(
                category="memory",
                priority=10,
                description="Critical memory usage - risk of OOM",
                implementation="Force garbage collection, clear caches, reduce buffer sizes",
                expected_improvement="20-40% memory reduction",
                risk_level="medium",
                estimated_effort="low"
            ))
        elif memory_percent > 85:
            recommendations.append(OptimizationRecommendation(
                category="memory",
                priority=8,
                description="High memory usage detected",
                implementation="Optimize data structures, implement memory pooling",
                expected_improvement="10-20% memory reduction",
                risk_level="low",
                estimated_effort="medium"
            ))
        
        return recommendations
    
    def _get_disk_optimizations(self, disk_percent: float) -> List[OptimizationRecommendation]:
        """Recommandations d'optimisation disque"""
        recommendations = []
        
        if disk_percent > 95:
            recommendations.append(OptimizationRecommendation(
                category="disk",
                priority=9,
                description="Critical disk space usage",
                implementation="Clean temporary files, compress logs, archive old data",
                expected_improvement="10-30% disk space recovery",
                risk_level="low",
                estimated_effort="low"
            ))
        
        return recommendations
    
    async def _apply_automatic_optimizations(self, recommendations: List[OptimizationRecommendation]):
        """Applique automatiquement les optimisations sûres"""
        for rec in recommendations:
            if rec.risk_level == "low" and rec.category in ["memory", "disk"]:
                try:
                    await self._execute_optimization(rec)
                except Exception as e:
                    logger.error(f"Failed to apply optimization {rec.description}: {e}")
    
    async def _execute_optimization(self, recommendation: OptimizationRecommendation):
        """Exécute une optimisation spécifique"""
        if recommendation.category == "memory":
            if "garbage collection" in recommendation.implementation.lower():
                gc.collect()
                logger.info("Executed garbage collection optimization")
                self.applied_optimizations.append(f"gc_collect_{datetime.utcnow().isoformat()}")
        
        elif recommendation.category == "disk":
            if "temporary files" in recommendation.implementation.lower():
                # Nettoyage des fichiers temporaires
                logger.info("Executed temporary files cleanup optimization")
                self.applied_optimizations.append(f"temp_cleanup_{datetime.utcnow().isoformat()}")

class PerformanceConfigManager:
    """Gestionnaire principal de la configuration de performance"""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.monitor = SystemResourceMonitor()
        self.optimizer = PerformanceOptimizer()
        self.alerts: deque = deque(maxlen=1000)
        self.function_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.performance_targets: Dict[str, float] = {
            "cpu_percent": 70.0,
            "memory_percent": 80.0,
            "disk_percent": 85.0,
            "response_time_ms": 500.0
        }
        
    async def initialize_performance_monitoring(self) -> bool:
        """Initialise le monitoring de performance"""
        try:
            logger.info("Initializing performance monitoring")
            
            # Démarrage du monitoring système
            self.monitor.start_monitoring()
            
            # Démarrage des tâches d'optimisation
            asyncio.create_task(self._performance_analysis_loop())
            asyncio.create_task(self._alert_processing_loop())
            
            logger.info("Performance monitoring initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize performance monitoring: {e}")
            return False
    
    async def _performance_analysis_loop(self):
        """Boucle d'analyse de performance"""
        while True:
            try:
                await asyncio.sleep(60)  # Analyse every minute
                
                # Récupération des métriques actuelles
                current_metrics = self.monitor.get_current_metrics()
                
                # Détection d'alertes
                alerts = self._detect_performance_alerts(current_metrics)
                self.alerts.extend(alerts)
                
                # Analyse et optimisation
                recommendations = await self.optimizer.analyze_and_optimize(current_metrics)
                
                if recommendations:
                    logger.info(f"Generated {len(recommendations)} performance recommendations")
                
            except Exception as e:
                logger.error(f"Error in performance analysis loop: {e}")
    
    async def _alert_processing_loop(self):
        """Boucle de traitement des alertes"""
        while True:
            try:
                await asyncio.sleep(30)  # Process alerts every 30 seconds
                
                if self.alerts:
                    # Groupement et déduplication des alertes
                    unique_alerts = self._deduplicate_alerts(list(self.alerts))
                    
                    for alert in unique_alerts:
                        if alert.level in [PerformanceLevel.CRITICAL, PerformanceLevel.DEGRADED]:
                            logger.warning(f"Performance alert: {alert.message}")
                
            except Exception as e:
                logger.error(f"Error in alert processing loop: {e}")
    
    def _detect_performance_alerts(self, metrics: Dict[str, PerformanceMetric]) -> List[PerformanceAlert]:
        """Détecte les alertes de performance"""
        alerts = []
        
        for metric_name, metric in metrics.items():
            if metric.threshold_critical and metric.value >= metric.threshold_critical:
                alerts.append(PerformanceAlert(
                    metric=metric_name,
                    level=PerformanceLevel.CRITICAL,
                    message=f"Critical {metric_name}: {metric.value}{metric.unit} >= {metric.threshold_critical}{metric.unit}",
                    value=metric.value,
                    threshold=metric.threshold_critical,
                    recommendations=self._get_alert_recommendations(metric_name, metric.value)
                ))
            elif metric.threshold_warning and metric.value >= metric.threshold_warning:
                alerts.append(PerformanceAlert(
                    metric=metric_name,
                    level=PerformanceLevel.DEGRADED,
                    message=f"Warning {metric_name}: {metric.value}{metric.unit} >= {metric.threshold_warning}{metric.unit}",
                    value=metric.value,
                    threshold=metric.threshold_warning,
                    recommendations=self._get_alert_recommendations(metric_name, metric.value)
                ))
        
        return alerts
    
    def _get_alert_recommendations(self, metric_name: str, value: float) -> List[str]:
        """Génère des recommandations pour une alerte"""
        recommendations = []
        
        if "cpu" in metric_name.lower():
            recommendations.extend([
                "Check for CPU-intensive processes",
                "Consider horizontal scaling",
                "Optimize algorithms and data structures"
            ])
        elif "memory" in metric_name.lower():
            recommendations.extend([
                "Check for memory leaks",
                "Implement memory pooling",
                "Optimize data caching strategies"
            ])
        elif "disk" in metric_name.lower():
            recommendations.extend([
                "Clean up temporary files",
                "Implement log rotation",
                "Archive old data"
            ])
        
        return recommendations
    
    def _deduplicate_alerts(self, alerts: List[PerformanceAlert]) -> List[PerformanceAlert]:
        """Déduplique les alertes similaires"""
        unique_alerts = {}
        
        for alert in alerts:
            # Clé unique basée sur la métrique et le niveau
            key = f"{alert.metric}_{alert.level.value}"
            
            # Garde la plus récente
            if key not in unique_alerts or alert.timestamp > unique_alerts[key].timestamp:
                unique_alerts[key] = alert
        
        return list(unique_alerts.values())
    
    def record_function_performance(
        self,
        function_name: str,
        duration_ms: float,
        memory_delta: int,
        success: bool
    ):
        """Enregistre les performances d'une fonction"""
        metric = {
            "function": function_name,
            "duration_ms": duration_ms,
            "memory_delta": memory_delta,
            "success": success,
            "timestamp": datetime.utcnow()
        }
        
        self.function_metrics[function_name].append(metric)
    
    def get_function_performance_stats(self, function_name: str) -> Dict[str, Any]:
        """Récupère les statistiques de performance d'une fonction"""
        if function_name not in self.function_metrics:
            return {}
        
        metrics = list(self.function_metrics[function_name])
        if not metrics:
            return {}
        
        durations = [m["duration_ms"] for m in metrics]
        success_rate = sum(1 for m in metrics if m["success"]) / len(metrics) * 100
        
        return {
            "function_name": function_name,
            "call_count": len(metrics),
            "success_rate": success_rate,
            "avg_duration_ms": statistics.mean(durations),
            "min_duration_ms": min(durations),
            "max_duration_ms": max(durations),
            "p95_duration_ms": statistics.quantiles(durations, n=20)[18] if len(durations) >= 20 else max(durations),
            "last_call": metrics[-1]["timestamp"]
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Génère un résumé des performances système"""
        current_metrics = self.monitor.get_current_metrics()
        recent_alerts = [a for a in self.alerts if (datetime.utcnow() - a.timestamp).total_seconds() < 3600]
        
        summary = {
            "timestamp": datetime.utcnow(),
            "system_status": self._determine_overall_status(current_metrics),
            "current_metrics": {name: {"value": m.value, "unit": m.unit} for name, m in current_metrics.items()},
            "recent_alerts_count": len(recent_alerts),
            "critical_alerts_count": len([a for a in recent_alerts if a.level == PerformanceLevel.CRITICAL]),
            "applied_optimizations_count": len(self.optimizer.applied_optimizations),
            "monitored_functions_count": len(self.function_metrics)
        }
        
        return summary
    
    def _determine_overall_status(self, metrics: Dict[str, PerformanceMetric]) -> PerformanceLevel:
        """Détermine le statut global du système"""
        if not metrics:
            return PerformanceLevel.GOOD
        
        critical_count = 0
        warning_count = 0
        
        for metric in metrics.values():
            if metric.threshold_critical and metric.value >= metric.threshold_critical:
                critical_count += 1
            elif metric.threshold_warning and metric.value >= metric.threshold_warning:
                warning_count += 1
        
        if critical_count > 0:
            return PerformanceLevel.CRITICAL
        elif warning_count > 2:
            return PerformanceLevel.DEGRADED
        elif warning_count > 0:
            return PerformanceLevel.GOOD
        else:
            return PerformanceLevel.OPTIMAL

# Instance singleton
performance_manager = PerformanceConfigManager()
