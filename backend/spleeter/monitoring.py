"""
🎵 Spotify AI Agent - Spleeter Monitoring
=========================================

Système de monitoring et métriques avancé pour le module Spleeter.
Surveillance de performance, santé système et analytics détaillés.

🎖️ Développé par l'équipe d'experts enterprise
"""

import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import logging
import psutil
import statistics
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import os
import queue
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from .exceptions import MonitoringError

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Métrique de performance individuelle"""
    name: str
    value: float
    timestamp: datetime
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'unit': self.unit,
            'tags': self.tags
        }


@dataclass
class ProcessingStats:
    """Statistiques de traitement"""
    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    total_duration_processed: float = 0.0
    total_processing_time: float = 0.0
    average_processing_ratio: float = 0.0
    peak_memory_usage: float = 0.0
    gpu_utilization: float = 0.0
    cache_hit_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            'total_files': self.total_files,
            'successful_files': self.successful_files,
            'failed_files': self.failed_files,
            'success_rate': self.success_rate,
            'total_duration_processed': self.total_duration_processed,
            'total_processing_time': self.total_processing_time,
            'average_processing_ratio': self.average_processing_ratio,
            'peak_memory_usage': self.peak_memory_usage,
            'gpu_utilization': self.gpu_utilization,
            'cache_hit_rate': self.cache_hit_rate
        }
    
    @property
    def success_rate(self) -> float:
        """Taux de succès en pourcentage"""
        if self.total_files == 0:
            return 0.0
        return (self.successful_files / self.total_files) * 100


@dataclass
class SystemHealth:
    """État de santé du système"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    gpu_usage: float = 0.0
    gpu_memory: float = 0.0
    load_average: float = 0.0
    active_threads: int = 0
    open_files: int = 0
    network_io: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def health_score(self) -> float:
        """Score de santé global (0-100)"""
        scores = []
        
        # CPU (inversé - moins c'est utilisé, mieux c'est)
        cpu_score = max(0, 100 - self.cpu_usage)
        scores.append(cpu_score)
        
        # Mémoire
        memory_score = max(0, 100 - self.memory_usage)
        scores.append(memory_score)
        
        # Disque
        disk_score = max(0, 100 - self.disk_usage)
        scores.append(disk_score)
        
        # GPU si disponible
        if self.gpu_usage > 0:
            gpu_score = max(0, 100 - self.gpu_usage)
            scores.append(gpu_score)
        
        return statistics.mean(scores)
    
    @property
    def status(self) -> str:
        """Status textuel basé sur le score"""
        score = self.health_score
        if score >= 80:
            return "excellent"
        elif score >= 60:
            return "good"
        elif score >= 40:
            return "warning"
        else:
            return "critical"


class MetricsCollector:
    """
    Collecteur de métriques pour les opérations Spleeter
    
    Features:
    - Métriques temps réel
    - Agrégation et statistiques
    - Exportation des données
    - Alertes automatiques
    """
    
    def __init__(self, 
                 buffer_size: int = 10000,
                 retention_hours: int = 24,
                 enable_system_metrics: bool = True):
        self.buffer_size = buffer_size
        self.retention_hours = retention_hours
        self.enable_system_metrics = enable_system_metrics
        
        # Stockage des métriques
        self.metrics_buffer: deque = deque(maxlen=buffer_size)
        self.aggregated_metrics: Dict[str, List[float]] = defaultdict(list)
        self.processing_stats = ProcessingStats()
        
        # Système de callbacks
        self.alert_callbacks: List[Callable] = []
        self.metric_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Thread pour métriques système
        self._system_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Cache pour optimisation
        self._cache_stats = {'hits': 0, 'misses': 0}
        
        # Démarrage automatique
        if enable_system_metrics:
            self.start_system_monitoring()
    
    def start_system_monitoring(self, interval: float = 1.0):
        """
        Démarre la surveillance système
        
        Args:
            interval: Intervalle en secondes
        """
        if self._system_thread and self._system_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._system_thread = threading.Thread(
            target=self._system_monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self._system_thread.start()
        logger.info("Surveillance système démarrée")
    
    def stop_system_monitoring(self):
        """Arrête la surveillance système"""
        self._stop_event.set()
        if self._system_thread:
            self._system_thread.join(timeout=5.0)
        logger.info("Surveillance système arrêtée")
    
    def _system_monitoring_loop(self, interval: float):
        """Boucle de surveillance système"""
        while not self._stop_event.wait(interval):
            try:
                health = self._collect_system_health()
                self.record_system_health(health)
                
                # Alertes si problèmes
                if health.health_score < 40:
                    self._trigger_alert(
                        "system_health_critical",
                        f"Santé système critique: {health.health_score:.1f}%",
                        health.to_dict()
                    )
                
            except Exception as e:
                logger.error(f"Erreur surveillance système: {e}")
    
    def _collect_system_health(self) -> SystemHealth:
        """Collecte l'état de santé système"""
        try:
            # Métriques CPU et mémoire
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Load average (Unix seulement)
            try:
                load_avg = os.getloadavg()[0]
            except (AttributeError, OSError):
                load_avg = 0.0
            
            # Informations processus
            process = psutil.Process()
            threads = process.num_threads()
            
            try:
                open_files = len(process.open_files())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                open_files = 0
            
            # GPU si disponible
            gpu_usage, gpu_memory = self._get_gpu_metrics()
            
            # I/O réseau
            net_io = psutil.net_io_counters()
            network_io = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
            
            return SystemHealth(
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                gpu_usage=gpu_usage,
                gpu_memory=gpu_memory,
                load_average=load_avg,
                active_threads=threads,
                open_files=open_files,
                network_io=network_io
            )
            
        except Exception as e:
            logger.error(f"Erreur collecte métriques système: {e}")
            return SystemHealth(0, 0, 0)
    
    def _get_gpu_metrics(self) -> tuple[float, float]:
        """Récupère les métriques GPU"""
        try:
            # Tentative avec pynvml (NVIDIA)
            try:
                import pynvml
                pynvml.nvmlInit()
                
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                gpu_usage = gpu_util.gpu
                gpu_memory = (memory_info.used / memory_info.total) * 100
                
                return gpu_usage, gpu_memory
                
            except ImportError:
                pass
            
            # Tentative avec TensorFlow
            try:
                import tensorflow as tf
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    # TensorFlow ne donne pas directement l'utilisation
                    # On retourne des valeurs par défaut
                    return 0.0, 0.0
            except ImportError:
                pass
            
            return 0.0, 0.0
            
        except Exception:
            return 0.0, 0.0
    
    def record_metric(self, 
                     name: str, 
                     value: float, 
                     unit: str = "",
                     tags: Optional[Dict[str, str]] = None):
        """
        Enregistre une métrique
        
        Args:
            name: Nom de la métrique
            value: Valeur
            unit: Unité
            tags: Tags additionnels
        """
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            unit=unit,
            tags=tags or {}
        )
        
        self.metrics_buffer.append(metric)
        self.aggregated_metrics[name].append(value)
        
        # Nettoyage des anciennes données
        self._cleanup_old_metrics()
        
        # Callbacks
        for callback in self.metric_callbacks.get(name, []):
            try:
                callback(metric)
            except Exception as e:
                logger.error(f"Erreur callback métrique {name}: {e}")
    
    def record_processing_event(self,
                               event_type: str,
                               audio_duration: float = 0.0,
                               processing_time: float = 0.0,
                               success: bool = True,
                               model_name: str = "",
                               file_size: int = 0):
        """
        Enregistre un événement de traitement
        
        Args:
            event_type: Type d'événement
            audio_duration: Durée audio
            processing_time: Temps de traitement
            success: Succès ou échec
            model_name: Nom du modèle
            file_size: Taille du fichier
        """
        # Mise à jour des statistiques
        self.processing_stats.total_files += 1
        if success:
            self.processing_stats.successful_files += 1
        else:
            self.processing_stats.failed_files += 1
        
        if audio_duration > 0:
            self.processing_stats.total_duration_processed += audio_duration
        
        if processing_time > 0:
            self.processing_stats.total_processing_time += processing_time
            
            # Calcul du ratio de traitement
            if audio_duration > 0:
                ratio = processing_time / audio_duration
                self.record_metric(
                    "processing_ratio",
                    ratio,
                    "x",
                    {"model": model_name, "event_type": event_type}
                )
        
        # Métriques individuelles
        self.record_metric(
            f"processing_{event_type}",
            1,
            "count",
            {"success": str(success), "model": model_name}
        )
        
        if file_size > 0:
            self.record_metric(
                "file_size",
                file_size,
                "bytes",
                {"model": model_name}
            )
    
    def record_cache_event(self, hit: bool, cache_type: str = ""):
        """
        Enregistre un événement de cache
        
        Args:
            hit: True si hit, False si miss
            cache_type: Type de cache
        """
        if hit:
            self._cache_stats['hits'] += 1
        else:
            self._cache_stats['misses'] += 1
        
        # Calcul du taux de hit
        total = self._cache_stats['hits'] + self._cache_stats['misses']
        hit_rate = (self._cache_stats['hits'] / total) * 100 if total > 0 else 0
        
        self.processing_stats.cache_hit_rate = hit_rate
        
        self.record_metric(
            "cache_hit_rate",
            hit_rate,
            "%",
            {"cache_type": cache_type}
        )
    
    def record_system_health(self, health: SystemHealth):
        """
        Enregistre l'état de santé système
        
        Args:
            health: État de santé
        """
        # Métriques individuelles
        self.record_metric("system_cpu_usage", health.cpu_usage, "%")
        self.record_metric("system_memory_usage", health.memory_usage, "%")
        self.record_metric("system_disk_usage", health.disk_usage, "%")
        self.record_metric("system_health_score", health.health_score, "score")
        
        if health.gpu_usage > 0:
            self.record_metric("system_gpu_usage", health.gpu_usage, "%")
            self.record_metric("system_gpu_memory", health.gpu_memory, "%")
        
        self.record_metric("system_load_average", health.load_average, "load")
        self.record_metric("system_active_threads", health.active_threads, "count")
        self.record_metric("system_open_files", health.open_files, "count")
        
        # Mise à jour des pics
        memory_mb = (health.memory_usage / 100) * self._get_total_memory()
        if memory_mb > self.processing_stats.peak_memory_usage:
            self.processing_stats.peak_memory_usage = memory_mb
        
        if health.gpu_usage > self.processing_stats.gpu_utilization:
            self.processing_stats.gpu_utilization = health.gpu_usage
    
    def _get_total_memory(self) -> float:
        """Retourne la mémoire totale en MB"""
        try:
            return psutil.virtual_memory().total / (1024 ** 2)
        except Exception:
            return 8192.0  # 8GB par défaut
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """
        Retourne un résumé des statistiques
        
        Returns:
            Dictionnaire de statistiques
        """
        # Statistiques de traitement
        processing_data = self.processing_stats.to_dict()
        
        # Calcul des moyennes sur les métriques récentes
        recent_metrics = {}
        for name, values in self.aggregated_metrics.items():
            if values:
                recent_values = values[-100:]  # 100 dernières valeurs
                recent_metrics[f"{name}_avg"] = statistics.mean(recent_values)
                recent_metrics[f"{name}_min"] = min(recent_values)
                recent_metrics[f"{name}_max"] = max(recent_values)
                if len(recent_values) > 1:
                    recent_metrics[f"{name}_stddev"] = statistics.stdev(recent_values)
        
        # Métriques système récentes
        system_health = self._collect_system_health()
        
        return {
            'processing_stats': processing_data,
            'recent_metrics': recent_metrics,
            'system_health': {
                'cpu_usage': system_health.cpu_usage,
                'memory_usage': system_health.memory_usage,
                'disk_usage': system_health.disk_usage,
                'gpu_usage': system_health.gpu_usage,
                'health_score': system_health.health_score,
                'status': system_health.status
            },
            'cache_stats': self._cache_stats.copy(),
            'buffer_size': len(self.metrics_buffer),
            'metrics_count': len(self.aggregated_metrics)
        }
    
    def get_metric_history(self, 
                          metric_name: str, 
                          duration_minutes: int = 60) -> List[PerformanceMetric]:
        """
        Retourne l'historique d'une métrique
        
        Args:
            metric_name: Nom de la métrique
            duration_minutes: Durée en minutes
            
        Returns:
            Liste des métriques
        """
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        
        history = [
            metric for metric in self.metrics_buffer
            if metric.name == metric_name and metric.timestamp >= cutoff_time
        ]
        
        return sorted(history, key=lambda x: x.timestamp)
    
    def add_alert_callback(self, callback: Callable):
        """
        Ajoute un callback d'alerte
        
        Args:
            callback: Fonction à appeler pour les alertes
        """
        self.alert_callbacks.append(callback)
    
    def add_metric_callback(self, metric_name: str, callback: Callable):
        """
        Ajoute un callback pour une métrique spécifique
        
        Args:
            metric_name: Nom de la métrique
            callback: Fonction à appeler
        """
        self.metric_callbacks[metric_name].append(callback)
    
    def _trigger_alert(self, alert_type: str, message: str, data: Any = None):
        """
        Déclenche une alerte
        
        Args:
            alert_type: Type d'alerte
            message: Message d'alerte
            data: Données additionnelles
        """
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        logger.warning(f"Alerte {alert_type}: {message}")
        
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Erreur callback alerte: {e}")
    
    def _cleanup_old_metrics(self):
        """Nettoie les anciennes métriques"""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        # Nettoyage du buffer principal
        while (self.metrics_buffer and 
               self.metrics_buffer[0].timestamp < cutoff_time):
            self.metrics_buffer.popleft()
        
        # Nettoyage des métriques agrégées
        for name in list(self.aggregated_metrics.keys()):
            values = self.aggregated_metrics[name]
            # Garde seulement les 1000 dernières valeurs par métrique
            if len(values) > 1000:
                self.aggregated_metrics[name] = values[-1000:]
    
    def export_metrics(self, file_path: Union[str, Path], format: str = "json"):
        """
        Exporte les métriques vers un fichier
        
        Args:
            file_path: Chemin du fichier
            format: Format (json, pickle)
        """
        file_path = Path(file_path)
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'stats_summary': self.get_stats_summary(),
            'recent_metrics': [
                metric.to_dict() for metric in 
                list(self.metrics_buffer)[-1000:]  # 1000 dernières
            ]
        }
        
        try:
            if format.lower() == "json":
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
            elif format.lower() == "pickle":
                with open(file_path, 'wb') as f:
                    pickle.dump(export_data, f)
            else:
                raise ValueError(f"Format non supporté: {format}")
            
            logger.info(f"Métriques exportées vers {file_path}")
            
        except Exception as e:
            logger.error(f"Erreur export métriques: {e}")
            raise MonitoringError(f"Impossible d'exporter les métriques: {e}")


class PerformanceTimer:
    """
    Timer de performance pour mesurer les opérations
    
    Features:
    - Mesure de temps précise
    - Context manager
    - Intégration métriques
    - Statistiques automatiques
    """
    
    def __init__(self, 
                 name: str,
                 collector: Optional[MetricsCollector] = None,
                 auto_record: bool = True):
        self.name = name
        self.collector = collector
        self.auto_record = auto_record
        
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration: Optional[float] = None
    
    def start(self):
        """Démarre le timer"""
        self.start_time = time.perf_counter()
        return self
    
    def stop(self) -> float:
        """
        Arrête le timer
        
        Returns:
            Durée en secondes
        """
        if self.start_time is None:
            raise RuntimeError("Timer non démarré")
        
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        
        if self.auto_record and self.collector:
            self.collector.record_metric(
                f"timer_{self.name}",
                self.duration,
                "seconds"
            )
        
        return self.duration
    
    def __enter__(self):
        """Context manager entry"""
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
        
        if exc_type is not None:
            # Enregistrement d'erreur si exception
            if self.collector:
                self.collector.record_metric(
                    f"timer_{self.name}_error",
                    1,
                    "count",
                    {"error_type": exc_type.__name__}
                )


class ResourceMonitor:
    """
    Moniteur de ressources pour les opérations intensives
    
    Features:
    - Surveillance mémoire
    - Surveillance GPU
    - Alertes de dépassement
    - Historique d'utilisation
    """
    
    def __init__(self, 
                 collector: Optional[MetricsCollector] = None,
                 memory_threshold_mb: float = 8192,
                 gpu_threshold_percent: float = 90):
        self.collector = collector
        self.memory_threshold_mb = memory_threshold_mb
        self.gpu_threshold_percent = gpu_threshold_percent
        
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
    
    @contextmanager
    def monitor_operation(self, operation_name: str):
        """
        Context manager pour surveiller une opération
        
        Args:
            operation_name: Nom de l'opération
        """
        self.start_monitoring(operation_name)
        try:
            yield self
        finally:
            self.stop_monitoring()
    
    def start_monitoring(self, operation_name: str = "operation"):
        """
        Démarre la surveillance
        
        Args:
            operation_name: Nom de l'opération
        """
        if self.monitoring:
            return
        
        self.monitoring = True
        self.stop_event.clear()
        
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(operation_name,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Arrête la surveillance"""
        self.monitoring = False
        self.stop_event.set()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
    
    def _monitoring_loop(self, operation_name: str):
        """Boucle de surveillance"""
        max_memory_mb = 0
        max_gpu_percent = 0
        
        while not self.stop_event.wait(0.5):  # Check toutes les 500ms
            try:
                # Mémoire du processus
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 ** 2)
                max_memory_mb = max(max_memory_mb, memory_mb)
                
                # GPU si disponible
                gpu_usage, _ = self._get_gpu_usage()
                if gpu_usage > 0:
                    max_gpu_percent = max(max_gpu_percent, gpu_usage)
                
                # Enregistrement des métriques
                if self.collector:
                    self.collector.record_metric(
                        f"operation_memory_{operation_name}",
                        memory_mb,
                        "MB"
                    )
                    
                    if gpu_usage > 0:
                        self.collector.record_metric(
                            f"operation_gpu_{operation_name}",
                            gpu_usage,
                            "%"
                        )
                
                # Vérification des seuils
                if memory_mb > self.memory_threshold_mb:
                    logger.warning(
                        f"Seuil mémoire dépassé pour {operation_name}: "
                        f"{memory_mb:.1f}MB > {self.memory_threshold_mb}MB"
                    )
                
                if gpu_usage > self.gpu_threshold_percent:
                    logger.warning(
                        f"Seuil GPU dépassé pour {operation_name}: "
                        f"{gpu_usage:.1f}% > {self.gpu_threshold_percent}%"
                    )
                
            except Exception as e:
                logger.debug(f"Erreur surveillance ressources: {e}")
        
        # Enregistrement des pics
        if self.collector:
            self.collector.record_metric(
                f"operation_peak_memory_{operation_name}",
                max_memory_mb,
                "MB"
            )
            
            if max_gpu_percent > 0:
                self.collector.record_metric(
                    f"operation_peak_gpu_{operation_name}",
                    max_gpu_percent,
                    "%"
                )
    
    def _get_gpu_usage(self) -> tuple[float, float]:
        """Récupère l'usage GPU"""
        try:
            import pynvml
            pynvml.nvmlInit()
            
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            gpu_usage = gpu_util.gpu
            gpu_memory = (memory_info.used / memory_info.total) * 100
            
            return gpu_usage, gpu_memory
            
        except Exception:
            return 0.0, 0.0


# Instance globale pour faciliter l'utilisation
_global_collector: Optional[MetricsCollector] = None


def get_global_collector() -> MetricsCollector:
    """
    Retourne l'instance globale du collecteur
    
    Returns:
        Collecteur de métriques global
    """
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def initialize_monitoring(
    buffer_size: int = 10000,
    retention_hours: int = 24,
    enable_system_metrics: bool = True
) -> MetricsCollector:
    """
    Initialise le système de monitoring global
    
    Args:
        buffer_size: Taille du buffer
        retention_hours: Heures de rétention
        enable_system_metrics: Activer métriques système
        
    Returns:
        Instance du collecteur
    """
    global _global_collector
    
    if _global_collector is not None:
        _global_collector.stop_system_monitoring()
    
    _global_collector = MetricsCollector(
        buffer_size=buffer_size,
        retention_hours=retention_hours,
        enable_system_metrics=enable_system_metrics
    )
    
    logger.info("Système de monitoring initialisé")
    return _global_collector


def shutdown_monitoring():
    """Arrête le système de monitoring global"""
    global _global_collector
    if _global_collector is not None:
        _global_collector.stop_system_monitoring()
        _global_collector = None
    logger.info("Système de monitoring arrêté")


# Fonctions utilitaires pour faciliter l'usage
def record_metric(name: str, value: float, unit: str = "", tags: Optional[Dict[str, str]] = None):
    """Enregistre une métrique via le collecteur global"""
    collector = get_global_collector()
    collector.record_metric(name, value, unit, tags)


def record_processing_event(**kwargs):
    """Enregistre un événement de traitement via le collecteur global"""
    collector = get_global_collector()
    collector.record_processing_event(**kwargs)


def record_cache_event(hit: bool, cache_type: str = ""):
    """Enregistre un événement de cache via le collecteur global"""
    collector = get_global_collector()
    collector.record_cache_event(hit, cache_type)


def get_stats_summary() -> Dict[str, Any]:
    """Retourne le résumé des statistiques globales"""
    collector = get_global_collector()
    return collector.get_stats_summary()


def create_timer(name: str, auto_record: bool = True) -> PerformanceTimer:
    """Crée un timer de performance"""
    collector = get_global_collector()
    return PerformanceTimer(name, collector, auto_record)


def create_resource_monitor(**kwargs) -> ResourceMonitor:
    """Crée un moniteur de ressources"""
    collector = get_global_collector()
    return ResourceMonitor(collector, **kwargs)
