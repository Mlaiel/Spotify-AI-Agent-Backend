# -*- coding: utf-8 -*-
"""
Performance Monitor - Système de Surveillance des Performances Ultra-Avancé
=========================================================================

Moniteur de performance intelligent pour l'agent IA Spotify.
Surveillance complète des performances avec analyse en temps réel,
détection d'anomalies par ML et optimisations automatiques.

Fonctionnalités:
- Monitoring temps réel des performances CPU/RAM/I/O/Réseau
- Analyse des goulots d'étranglement (bottlenecks)
- Profilage automatique des requêtes lentes
- Détection d'anomalies par Machine Learning
- Recommandations d'optimisation automatiques
- Monitoring des performances par tenant
- Alerting prédictif de dégradation

Auteur: Expert Team - Architecte Microservices + Performance Expert - Fahed Mlaiel
Version: 2.0.0
"""

import time
import threading
import asyncio
import logging
import json
import psutil
import gc
import resource
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import redis
import requests
from collections import deque, defaultdict
import cProfile
import pstats
import io

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceLevel(Enum):
    """Niveaux de performance"""
    EXCELLENT = "excellent"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class MetricType(Enum):
    """Types de métriques de performance"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    CONCURRENCY = "concurrency"
    CACHE_HIT_RATE = "cache_hit_rate"

@dataclass
class PerformanceMetric:
    """Métrique de performance"""
    name: str
    value: float
    unit: str
    timestamp: float
    tenant_id: Optional[str] = None
    service: Optional[str] = None
    endpoint: Optional[str] = None
    labels: Optional[Dict[str, str]] = None

@dataclass
class PerformanceThreshold:
    """Seuils de performance"""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    operator: str = ">"  # >, <, ==, !=
    duration_minutes: int = 5  # Durée avant alerte

@dataclass
class BottleneckDetection:
    """Détection de goulot d'étranglement"""
    component: str
    severity: str
    description: str
    impact_score: float
    recommendations: List[str]
    detected_at: float
    metrics: Dict[str, float]

@dataclass
class PerformanceReport:
    """Rapport de performance"""
    start_time: float
    end_time: float
    overall_score: float
    level: str
    summary: str
    metrics: Dict[str, float]
    bottlenecks: List[BottleneckDetection]
    recommendations: List[str]
    tenant_id: Optional[str] = None

class PerformanceMonitor:
    """
    Moniteur de performance ultra-avancé avec IA prédictive
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise le moniteur de performance
        
        Args:
            config: Configuration du moniteur
        """
        self.config = config or self._default_config()
        self.metrics_buffer = deque(maxlen=10000)
        self.thresholds: List[PerformanceThreshold] = []
        self.bottlenecks: List[BottleneckDetection] = []
        self.lock = threading.RLock()
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=6)
        
        # Historique des métriques pour ML
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Profiling data
        self.profiler_data = {}
        self.slow_requests = deque(maxlen=100)
        
        # Storage
        self.db_path = self.config.get('db_path', 'performance.db')
        self.redis_client = self._init_redis()
        
        # Background threads
        self.collection_thread = None
        self.analysis_thread = None
        self.profiling_thread = None
        
        # Statistiques
        self.stats = {
            'total_requests': 0,
            'slow_requests': 0,
            'avg_response_time': 0,
            'current_load': 0,
            'peak_memory_mb': 0,
            'gc_collections': 0
        }
        
        # Initialisation
        self._init_database()
        self._setup_default_thresholds()
        
        logger.info("PerformanceMonitor initialisé avec succès")
    
    def _default_config(self) -> Dict:
        """Configuration par défaut"""
        return {
            'collection_interval': 10,
            'analysis_interval': 60,
            'slow_request_threshold_ms': 1000,
            'memory_warning_threshold_mb': 1024,
            'memory_critical_threshold_mb': 2048,
            'cpu_warning_threshold': 80,
            'cpu_critical_threshold': 95,
            'enable_profiling': True,
            'enable_ml_analysis': True,
            'retention_days': 30,
            'max_concurrent_requests': 1000,
            'redis_host': 'localhost',
            'redis_port': 6379,
            'redis_db': 3,
            'bottleneck_detection_enabled': True,
            'auto_optimization_enabled': False
        }
    
    def _init_redis(self) -> Optional[redis.Redis]:
        """Initialise la connexion Redis"""
        try:
            client = redis.Redis(
                host=self.config['redis_host'],
                port=self.config['redis_port'],
                db=self.config['redis_db'],
                decode_responses=True
            )
            client.ping()
            logger.info("Connexion Redis PerformanceMonitor établie")
            return client
        except Exception as e:
            logger.warning(f"Redis non disponible pour PerformanceMonitor: {e}")
            return None
    
    def _init_database(self):
        """Initialise la base de données SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Table des métriques
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    tenant_id TEXT,
                    service TEXT,
                    endpoint TEXT,
                    labels TEXT
                )
            ''')
            
            # Table des goulots d'étranglement
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS bottlenecks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    description TEXT NOT NULL,
                    impact_score REAL NOT NULL,
                    recommendations TEXT NOT NULL,
                    detected_at REAL NOT NULL,
                    metrics TEXT,
                    resolved_at REAL
                )
            ''')
            
            # Table des rapports de performance
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    overall_score REAL NOT NULL,
                    level TEXT NOT NULL,
                    summary TEXT,
                    metrics TEXT,
                    bottlenecks TEXT,
                    recommendations TEXT,
                    tenant_id TEXT,
                    created_at REAL NOT NULL
                )
            ''')
            
            # Table des requêtes lentes
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS slow_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    endpoint TEXT NOT NULL,
                    method TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    tenant_id TEXT,
                    user_agent TEXT,
                    profiling_data TEXT
                )
            ''')
            
            # Index pour les performances
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp ON performance_metrics(name, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_tenant ON performance_metrics(tenant_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_bottlenecks_detected ON bottlenecks(detected_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_slow_requests_endpoint ON slow_requests(endpoint)')
            
            conn.commit()
            conn.close()
            
            logger.info("Base de données PerformanceMonitor initialisée")
            
        except Exception as e:
            logger.error(f"Erreur initialisation base de données: {e}")
    
    def start(self):
        """Démarre le moniteur de performance"""
        if self.running:
            logger.warning("PerformanceMonitor déjà en cours d'exécution")
            return
        
        self.running = True
        
        # Démarre les threads de surveillance
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        
        if self.config['enable_profiling']:
            self.profiling_thread = threading.Thread(target=self._profiling_loop, daemon=True)
            self.profiling_thread.start()
        
        self.collection_thread.start()
        self.analysis_thread.start()
        
        logger.info("PerformanceMonitor démarré")
    
    def stop(self):
        """Arrête le moniteur de performance"""
        self.running = False
        
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5)
        
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=5)
        
        if self.profiling_thread and self.profiling_thread.is_alive():
            self.profiling_thread.join(timeout=5)
        
        self.executor.shutdown(wait=True)
        logger.info("PerformanceMonitor arrêté")
    
    def record_metric(self, name: str, value: float, unit: str = "",
                     tenant_id: Optional[str] = None,
                     service: Optional[str] = None,
                     endpoint: Optional[str] = None,
                     labels: Optional[Dict[str, str]] = None):
        """
        Enregistre une métrique de performance
        
        Args:
            name: Nom de la métrique
            value: Valeur de la métrique
            unit: Unité de mesure
            tenant_id: ID du tenant
            service: Nom du service
            endpoint: Endpoint concerné
            labels: Labels additionnels
        """
        try:
            metric = PerformanceMetric(
                name=name,
                value=value,
                unit=unit,
                timestamp=time.time(),
                tenant_id=tenant_id,
                service=service,
                endpoint=endpoint,
                labels=labels
            )
            
            with self.lock:
                self.metrics_buffer.append(metric)
                self.metrics_history[name].append((time.time(), value))
            
            # Sauvegarde asynchrone en base
            self.executor.submit(self._save_metric_to_db, metric)
            
            # Vérification des seuils
            self._check_thresholds(metric)
            
        except Exception as e:
            logger.error(f"Erreur enregistrement métrique {name}: {e}")
    
    def track_request_performance(self, endpoint: str, method: str = "GET",
                                duration_ms: float = 0,
                                status_code: int = 200,
                                tenant_id: Optional[str] = None,
                                user_agent: Optional[str] = None):
        """
        Enregistre la performance d'une requête
        
        Args:
            endpoint: Endpoint de la requête
            method: Méthode HTTP
            duration_ms: Durée en millisecondes
            status_code: Code de statut HTTP
            tenant_id: ID du tenant
            user_agent: User agent
        """
        try:
            # Métriques de base
            self.record_metric(
                name="request_duration",
                value=duration_ms,
                unit="ms",
                tenant_id=tenant_id,
                endpoint=endpoint,
                labels={
                    'method': method,
                    'status_code': str(status_code)
                }
            )
            
            # Mise à jour des statistiques
            with self.lock:
                self.stats['total_requests'] += 1
                
                # Calcul de la moyenne mobile
                total_time = self.stats['avg_response_time'] * (self.stats['total_requests'] - 1)
                self.stats['avg_response_time'] = (total_time + duration_ms) / self.stats['total_requests']
                
                # Détection de requête lente
                if duration_ms > self.config['slow_request_threshold_ms']:
                    self.stats['slow_requests'] += 1
                    self._handle_slow_request(endpoint, method, duration_ms, tenant_id, user_agent)
            
        except Exception as e:
            logger.error(f"Erreur tracking requête: {e}")
    
    def track_resource_usage(self, tenant_id: Optional[str] = None):
        """
        Enregistre l'utilisation des ressources système
        
        Args:
            tenant_id: ID du tenant (pour isolation multi-tenant)
        """
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric("cpu_usage", cpu_percent, "percent", tenant_id=tenant_id)
            
            # Mémoire
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            self.record_metric("memory_usage", memory_mb, "MB", tenant_id=tenant_id)
            self.record_metric("memory_percent", memory.percent, "percent", tenant_id=tenant_id)
            
            # Mise à jour du pic mémoire
            if memory_mb > self.stats['peak_memory_mb']:
                self.stats['peak_memory_mb'] = memory_mb
            
            # Disque I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                self.record_metric("disk_read_bytes", disk_io.read_bytes, "bytes", tenant_id=tenant_id)
                self.record_metric("disk_write_bytes", disk_io.write_bytes, "bytes", tenant_id=tenant_id)
                self.record_metric("disk_read_ops", disk_io.read_count, "ops", tenant_id=tenant_id)
                self.record_metric("disk_write_ops", disk_io.write_count, "ops", tenant_id=tenant_id)
            
            # Réseau I/O
            net_io = psutil.net_io_counters()
            if net_io:
                self.record_metric("network_bytes_sent", net_io.bytes_sent, "bytes", tenant_id=tenant_id)
                self.record_metric("network_bytes_recv", net_io.bytes_recv, "bytes", tenant_id=tenant_id)
                self.record_metric("network_packets_sent", net_io.packets_sent, "packets", tenant_id=tenant_id)
                self.record_metric("network_packets_recv", net_io.packets_recv, "packets", tenant_id=tenant_id)
            
            # Load average (Unix uniquement)
            try:
                load_avg = psutil.getloadavg()
                self.record_metric("load_average_1m", load_avg[0], "load", tenant_id=tenant_id)
                self.record_metric("load_average_5m", load_avg[1], "load", tenant_id=tenant_id)
                self.record_metric("load_average_15m", load_avg[2], "load", tenant_id=tenant_id)
                self.stats['current_load'] = load_avg[0]
            except AttributeError:
                pass  # Windows n'a pas getloadavg
            
            # Garbage Collection
            gc_stats = gc.get_stats()
            if gc_stats:
                total_collections = sum(stat.get('collections', 0) for stat in gc_stats)
                self.stats['gc_collections'] = total_collections
                self.record_metric("gc_collections", total_collections, "count", tenant_id=tenant_id)
            
        except Exception as e:
            logger.error(f"Erreur tracking ressources: {e}")
    
    def _handle_slow_request(self, endpoint: str, method: str, duration_ms: float,
                           tenant_id: Optional[str], user_agent: Optional[str]):
        """Gère une requête lente détectée"""
        try:
            slow_request = {
                'endpoint': endpoint,
                'method': method,
                'duration_ms': duration_ms,
                'timestamp': time.time(),
                'tenant_id': tenant_id,
                'user_agent': user_agent
            }
            
            with self.lock:
                self.slow_requests.append(slow_request)
            
            # Sauvegarde en base
            self.executor.submit(self._save_slow_request_to_db, slow_request)
            
            # Déclenchement d'alerte si trop de requêtes lentes
            recent_slow = len([r for r in self.slow_requests 
                              if time.time() - r['timestamp'] < 300])  # 5 minutes
            
            if recent_slow >= 10:  # 10 requêtes lentes en 5 min
                self._trigger_slow_request_alert(endpoint, recent_slow)
            
            logger.warning(f"Requête lente détectée: {endpoint} - {duration_ms:.0f}ms")
            
        except Exception as e:
            logger.error(f"Erreur gestion requête lente: {e}")
    
    def detect_bottlenecks(self, timeframe_minutes: int = 15) -> List[BottleneckDetection]:
        """
        Détecte les goulots d'étranglement système
        
        Args:
            timeframe_minutes: Période d'analyse en minutes
            
        Returns:
            Liste des goulots d'étranglement détectés
        """
        try:
            bottlenecks = []
            cutoff_time = time.time() - (timeframe_minutes * 60)
            
            # Analyse des métriques récentes
            recent_metrics = defaultdict(list)
            
            with self.lock:
                for metric in self.metrics_buffer:
                    if metric.timestamp > cutoff_time:
                        recent_metrics[metric.name].append(metric.value)
            
            # Détection CPU
            if 'cpu_usage' in recent_metrics:
                cpu_values = recent_metrics['cpu_usage']
                avg_cpu = statistics.mean(cpu_values)
                max_cpu = max(cpu_values)
                
                if avg_cpu > 85 or max_cpu > 95:
                    bottlenecks.append(BottleneckDetection(
                        component="CPU",
                        severity="critical" if max_cpu > 95 else "warning",
                        description=f"Utilisation CPU élevée: moyenne {avg_cpu:.1f}%, pic {max_cpu:.1f}%",
                        impact_score=min(avg_cpu / 100, 1.0),
                        recommendations=[
                            "Analyser les processus consommateurs de CPU",
                            "Considérer l'ajout de ressources CPU",
                            "Optimiser les algorithmes critiques",
                            "Implémenter la mise en cache"
                        ],
                        detected_at=time.time(),
                        metrics={'avg_cpu': avg_cpu, 'max_cpu': max_cpu}
                    ))
            
            # Détection Mémoire
            if 'memory_usage' in recent_metrics:
                memory_values = recent_metrics['memory_usage']
                avg_memory = statistics.mean(memory_values)
                max_memory = max(memory_values)
                
                if avg_memory > self.config['memory_warning_threshold_mb']:
                    severity = "critical" if avg_memory > self.config['memory_critical_threshold_mb'] else "warning"
                    bottlenecks.append(BottleneckDetection(
                        component="Memory",
                        severity=severity,
                        description=f"Consommation mémoire élevée: moyenne {avg_memory:.1f}MB, pic {max_memory:.1f}MB",
                        impact_score=min(avg_memory / self.config['memory_critical_threshold_mb'], 1.0),
                        recommendations=[
                            "Analyser les fuites mémoire potentielles",
                            "Optimiser les structures de données",
                            "Implémenter le garbage collection",
                            "Augmenter la RAM disponible"
                        ],
                        detected_at=time.time(),
                        metrics={'avg_memory_mb': avg_memory, 'max_memory_mb': max_memory}
                    ))
            
            # Détection I/O Disque
            if 'disk_read_ops' in recent_metrics and 'disk_write_ops' in recent_metrics:
                read_ops = recent_metrics['disk_read_ops']
                write_ops = recent_metrics['disk_write_ops']
                
                if read_ops or write_ops:
                    avg_read_ops = statistics.mean(read_ops) if read_ops else 0
                    avg_write_ops = statistics.mean(write_ops) if write_ops else 0
                    total_ops = avg_read_ops + avg_write_ops
                    
                    if total_ops > 1000:  # Seuil arbitraire
                        bottlenecks.append(BottleneckDetection(
                            component="Disk I/O",
                            severity="warning",
                            description=f"I/O disque élevé: {total_ops:.0f} ops/s (R: {avg_read_ops:.0f}, W: {avg_write_ops:.0f})",
                            impact_score=min(total_ops / 5000, 1.0),
                            recommendations=[
                                "Optimiser les requêtes de base de données",
                                "Implémenter la mise en cache",
                                "Utiliser des SSD plus rapides",
                                "Réduire les écritures fréquentes"
                            ],
                            detected_at=time.time(),
                            metrics={'read_ops': avg_read_ops, 'write_ops': avg_write_ops}
                        ))
            
            # Détection Réseau
            if 'network_bytes_sent' in recent_metrics and 'network_bytes_recv' in recent_metrics:
                sent_bytes = recent_metrics['network_bytes_sent']
                recv_bytes = recent_metrics['network_bytes_recv']
                
                if sent_bytes or recv_bytes:
                    avg_sent = statistics.mean(sent_bytes) if sent_bytes else 0
                    avg_recv = statistics.mean(recv_bytes) if recv_bytes else 0
                    total_bandwidth = (avg_sent + avg_recv) / (1024 * 1024)  # MB/s
                    
                    if total_bandwidth > 100:  # > 100 MB/s
                        bottlenecks.append(BottleneckDetection(
                            component="Network",
                            severity="warning",
                            description=f"Bande passante réseau élevée: {total_bandwidth:.1f} MB/s",
                            impact_score=min(total_bandwidth / 1000, 1.0),
                            recommendations=[
                                "Optimiser la sérialisation des données",
                                "Implémenter la compression",
                                "Réduire la taille des payloads",
                                "Utiliser une connexion réseau plus rapide"
                            ],
                            detected_at=time.time(),
                            metrics={'sent_mb_per_s': avg_sent / (1024 * 1024), 'recv_mb_per_s': avg_recv / (1024 * 1024)}
                        ))
            
            # Détection requêtes lentes
            recent_slow_requests = [r for r in self.slow_requests 
                                  if time.time() - r['timestamp'] < timeframe_minutes * 60]
            
            if len(recent_slow_requests) > 5:
                avg_duration = statistics.mean([r['duration_ms'] for r in recent_slow_requests])
                bottlenecks.append(BottleneckDetection(
                    component="Application Performance",
                    severity="warning",
                    description=f"{len(recent_slow_requests)} requêtes lentes détectées (avg: {avg_duration:.0f}ms)",
                    impact_score=min(len(recent_slow_requests) / 50, 1.0),
                    recommendations=[
                        "Profiler les endpoints lents",
                        "Optimiser les requêtes de base de données",
                        "Implémenter la mise en cache",
                        "Réduire la complexité algorithmique"
                    ],
                    detected_at=time.time(),
                    metrics={'slow_requests_count': len(recent_slow_requests), 'avg_duration_ms': avg_duration}
                ))
            
            # Sauvegarde des nouveaux goulots d'étranglement
            with self.lock:
                for bottleneck in bottlenecks:
                    if not any(b.component == bottleneck.component and 
                             abs(b.detected_at - bottleneck.detected_at) < 300 
                             for b in self.bottlenecks):
                        self.bottlenecks.append(bottleneck)
                        self.executor.submit(self._save_bottleneck_to_db, bottleneck)
            
            return bottlenecks
            
        except Exception as e:
            logger.error(f"Erreur détection goulots d'étranglement: {e}")
            return []
    
    def generate_performance_report(self, timeframe_minutes: int = 60,
                                  tenant_id: Optional[str] = None) -> PerformanceReport:
        """
        Génère un rapport de performance
        
        Args:
            timeframe_minutes: Période d'analyse en minutes
            tenant_id: ID du tenant (optionnel)
            
        Returns:
            Rapport de performance détaillé
        """
        try:
            end_time = time.time()
            start_time = end_time - (timeframe_minutes * 60)
            
            # Collecte des métriques
            metrics_summary = {}
            cutoff_time = start_time
            
            with self.lock:
                recent_metrics = defaultdict(list)
                for metric in self.metrics_buffer:
                    if (metric.timestamp > cutoff_time and 
                        (not tenant_id or metric.tenant_id == tenant_id)):
                        recent_metrics[metric.name].append(metric.value)
            
            # Calcul des statistiques
            for metric_name, values in recent_metrics.items():
                if values:
                    metrics_summary[f"{metric_name}_avg"] = statistics.mean(values)
                    metrics_summary[f"{metric_name}_max"] = max(values)
                    metrics_summary[f"{metric_name}_min"] = min(values)
                    metrics_summary[f"{metric_name}_p95"] = self._percentile(values, 0.95)
            
            # Détection des goulots d'étranglement
            bottlenecks = self.detect_bottlenecks(timeframe_minutes)
            
            # Calcul du score global de performance
            overall_score = self._calculate_performance_score(metrics_summary, bottlenecks)
            
            # Détermination du niveau
            if overall_score >= 90:
                level = PerformanceLevel.EXCELLENT.value
            elif overall_score >= 80:
                level = PerformanceLevel.GOOD.value
            elif overall_score >= 60:
                level = PerformanceLevel.WARNING.value
            else:
                level = PerformanceLevel.CRITICAL.value
            
            # Génération du résumé
            summary = self._generate_performance_summary(overall_score, level, metrics_summary, bottlenecks)
            
            # Recommandations
            recommendations = self._generate_recommendations(metrics_summary, bottlenecks)
            
            report = PerformanceReport(
                start_time=start_time,
                end_time=end_time,
                overall_score=overall_score,
                level=level,
                summary=summary,
                metrics=metrics_summary,
                bottlenecks=bottlenecks,
                recommendations=recommendations,
                tenant_id=tenant_id
            )
            
            # Sauvegarde du rapport
            self.executor.submit(self._save_report_to_db, report)
            
            return report
            
        except Exception as e:
            logger.error(f"Erreur génération rapport: {e}")
            return PerformanceReport(
                start_time=start_time,
                end_time=time.time(),
                overall_score=0,
                level=PerformanceLevel.UNKNOWN.value,
                summary="Erreur lors de la génération du rapport",
                metrics={},
                bottlenecks=[],
                recommendations=[],
                tenant_id=tenant_id
            )
    
    def _collection_loop(self):
        """Boucle principale de collecte des métriques"""
        while self.running:
            try:
                # Collecte des métriques système
                self.track_resource_usage()
                
                # Nettoyage périodique
                self._cleanup_old_data()
                
                time.sleep(self.config['collection_interval'])
                
            except Exception as e:
                logger.error(f"Erreur dans la boucle de collecte: {e}")
                time.sleep(5)
    
    def _analysis_loop(self):
        """Boucle d'analyse et détection"""
        while self.running:
            try:
                # Détection des goulots d'étranglement
                if self.config['bottleneck_detection_enabled']:
                    self.detect_bottlenecks()
                
                # Analyse ML si activée
                if self.config['enable_ml_analysis']:
                    self._ml_anomaly_detection()
                
                # Auto-optimisation si activée
                if self.config['auto_optimization_enabled']:
                    self._auto_optimization()
                
                time.sleep(self.config['analysis_interval'])
                
            except Exception as e:
                logger.error(f"Erreur dans la boucle d'analyse: {e}")
                time.sleep(30)
    
    def _profiling_loop(self):
        """Boucle de profilage des performances"""
        while self.running:
            try:
                # Profilage périodique du système
                self._profile_system_performance()
                
                time.sleep(300)  # Profiling toutes les 5 minutes
                
            except Exception as e:
                logger.error(f"Erreur dans la boucle de profilage: {e}")
                time.sleep(60)
    
    def _ml_anomaly_detection(self):
        """Détection d'anomalies par Machine Learning"""
        try:
            # TODO: Implémentation ML pour détection d'anomalies
            # - Analyse des patterns de performance
            # - Détection de déviations statistiques
            # - Prédiction de dégradations futures
            # - Alerting prédictif
            
            # Pour l'instant, analyse statistique simple
            for metric_name, history in self.metrics_history.items():
                if len(history) < 30:  # Besoin d'historique
                    continue
                
                values = [value for _, value in history]
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0
                
                # Détection d'outliers (> 3 écarts-types)
                recent_values = values[-5:]  # 5 dernières valeurs
                for value in recent_values:
                    if abs(value - mean_val) > 3 * std_val:
                        logger.warning(f"Anomalie détectée pour {metric_name}: {value} "
                                     f"(moyenne: {mean_val:.2f}, écart-type: {std_val:.2f})")
                        self._trigger_anomaly_alert(metric_name, value, mean_val, std_val)
            
        except Exception as e:
            logger.error(f"Erreur détection anomalies ML: {e}")
    
    def _auto_optimization(self):
        """Optimisation automatique du système"""
        try:
            # TODO: Implémentation d'optimisations automatiques
            # - Ajustement automatique des paramètres
            # - Scaling automatique
            # - Nettoyage de cache
            # - Garbage collection forcé
            
            # Exemple: nettoyage mémoire si nécessaire
            memory = psutil.virtual_memory()
            if memory.percent > 85:
                logger.info("Déclenchement garbage collection pour optimisation mémoire")
                gc.collect()
            
        except Exception as e:
            logger.error(f"Erreur auto-optimisation: {e}")
    
    def _profile_system_performance(self):
        """Profilage des performances système"""
        try:
            profiler = cProfile.Profile()
            profiler.enable()
            
            # Simulation d'activité à profiler
            time.sleep(1)
            
            profiler.disable()
            
            # Analyse des résultats
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(10)  # Top 10 fonctions
            
            profiling_result = s.getvalue()
            
            # Stockage des données de profiling
            self.profiler_data[time.time()] = profiling_result
            
            # Limitation de l'historique
            if len(self.profiler_data) > 100:
                oldest_key = min(self.profiler_data.keys())
                del self.profiler_data[oldest_key]
            
        except Exception as e:
            logger.error(f"Erreur profilage: {e}")
    
    def _check_thresholds(self, metric: PerformanceMetric):
        """Vérifie les seuils de performance"""
        try:
            for threshold in self.thresholds:
                if threshold.metric_name == metric.name:
                    if self._evaluate_threshold(metric.value, threshold):
                        self._trigger_threshold_alert(metric, threshold)
            
        except Exception as e:
            logger.error(f"Erreur vérification seuils: {e}")
    
    def _evaluate_threshold(self, value: float, threshold: PerformanceThreshold) -> bool:
        """Évalue si un seuil est dépassé"""
        if threshold.operator == ">":
            return value > threshold.critical_threshold
        elif threshold.operator == "<":
            return value < threshold.critical_threshold
        elif threshold.operator == "==":
            return value == threshold.critical_threshold
        elif threshold.operator == "!=":
            return value != threshold.critical_threshold
        return False
    
    def _calculate_performance_score(self, metrics: Dict[str, float], 
                                   bottlenecks: List[BottleneckDetection]) -> float:
        """Calcule le score global de performance"""
        try:
            score = 100.0
            
            # Pénalités basées sur les métriques
            if 'cpu_usage_avg' in metrics:
                cpu_penalty = max(0, (metrics['cpu_usage_avg'] - 70) * 0.5)
                score -= cpu_penalty
            
            if 'memory_percent_avg' in metrics:
                memory_penalty = max(0, (metrics['memory_percent_avg'] - 70) * 0.3)
                score -= memory_penalty
            
            if 'request_duration_avg' in metrics:
                latency_penalty = max(0, (metrics['request_duration_avg'] - 200) * 0.1)
                score -= latency_penalty
            
            # Pénalités pour les goulots d'étranglement
            for bottleneck in bottlenecks:
                if bottleneck.severity == "critical":
                    score -= 20 * bottleneck.impact_score
                elif bottleneck.severity == "warning":
                    score -= 10 * bottleneck.impact_score
            
            return max(0, min(100, score))
            
        except Exception as e:
            logger.error(f"Erreur calcul score performance: {e}")
            return 50.0
    
    def _generate_performance_summary(self, score: float, level: str,
                                    metrics: Dict[str, float],
                                    bottlenecks: List[BottleneckDetection]) -> str:
        """Génère un résumé de performance"""
        try:
            summary_parts = [f"Score global: {score:.1f}/100 ({level})"]
            
            if 'cpu_usage_avg' in metrics:
                summary_parts.append(f"CPU moyen: {metrics['cpu_usage_avg']:.1f}%")
            
            if 'memory_percent_avg' in metrics:
                summary_parts.append(f"Mémoire moyenne: {metrics['memory_percent_avg']:.1f}%")
            
            if 'request_duration_avg' in metrics:
                summary_parts.append(f"Latence moyenne: {metrics['request_duration_avg']:.0f}ms")
            
            if bottlenecks:
                critical_bottlenecks = [b for b in bottlenecks if b.severity == "critical"]
                if critical_bottlenecks:
                    summary_parts.append(f"{len(critical_bottlenecks)} goulot(s) critique(s)")
            
            return " | ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Erreur génération résumé: {e}")
            return f"Score: {score:.1f}/100"
    
    def _generate_recommendations(self, metrics: Dict[str, float],
                                bottlenecks: List[BottleneckDetection]) -> List[str]:
        """Génère des recommandations d'optimisation"""
        recommendations = []
        
        try:
            # Recommandations basées sur les métriques
            if 'cpu_usage_avg' in metrics and metrics['cpu_usage_avg'] > 80:
                recommendations.append("Réduire l'utilisation CPU en optimisant les algorithmes")
            
            if 'memory_percent_avg' in metrics and metrics['memory_percent_avg'] > 80:
                recommendations.append("Optimiser l'utilisation mémoire et détecter les fuites")
            
            if 'request_duration_avg' in metrics and metrics['request_duration_avg'] > 1000:
                recommendations.append("Optimiser les temps de réponse des APIs")
            
            # Recommandations des goulots d'étranglement
            for bottleneck in bottlenecks:
                recommendations.extend(bottleneck.recommendations[:2])  # 2 premières recommandations
            
            # Suppression des doublons
            recommendations = list(dict.fromkeys(recommendations))
            
            return recommendations[:10]  # Maximum 10 recommandations
            
        except Exception as e:
            logger.error(f"Erreur génération recommandations: {e}")
            return ["Analyser les performances système"]
    
    def _setup_default_thresholds(self):
        """Configure les seuils par défaut"""
        try:
            default_thresholds = [
                PerformanceThreshold(
                    metric_name="cpu_usage",
                    warning_threshold=self.config['cpu_warning_threshold'],
                    critical_threshold=self.config['cpu_critical_threshold'],
                    operator=">",
                    duration_minutes=5
                ),
                PerformanceThreshold(
                    metric_name="memory_usage",
                    warning_threshold=self.config['memory_warning_threshold_mb'],
                    critical_threshold=self.config['memory_critical_threshold_mb'],
                    operator=">",
                    duration_minutes=5
                ),
                PerformanceThreshold(
                    metric_name="request_duration",
                    warning_threshold=self.config['slow_request_threshold_ms'],
                    critical_threshold=self.config['slow_request_threshold_ms'] * 2,
                    operator=">",
                    duration_minutes=1
                )
            ]
            
            self.thresholds.extend(default_thresholds)
            
        except Exception as e:
            logger.error(f"Erreur configuration seuils: {e}")
    
    def _trigger_threshold_alert(self, metric: PerformanceMetric, threshold: PerformanceThreshold):
        """Déclenche une alerte de seuil"""
        try:
            from .alert_manager import get_alert_manager, AlertSeverity
            
            severity = AlertSeverity.CRITICAL.value if metric.value > threshold.critical_threshold else AlertSeverity.WARNING.value
            
            alert_manager = get_alert_manager()
            alert_manager.trigger_alert(
                name=f"performance_threshold_{metric.name}",
                description=f"Seuil dépassé pour {metric.name}: {metric.value} {metric.unit}",
                severity=severity,
                source="performance_monitor",
                tenant_id=metric.tenant_id,
                labels={
                    'metric_name': metric.name,
                    'threshold_value': str(threshold.critical_threshold),
                    'actual_value': str(metric.value),
                    'service': metric.service or 'unknown',
                    'endpoint': metric.endpoint or 'unknown'
                }
            )
            
        except Exception as e:
            logger.error(f"Erreur alerte seuil: {e}")
    
    def _trigger_slow_request_alert(self, endpoint: str, count: int):
        """Déclenche une alerte pour requêtes lentes"""
        try:
            from .alert_manager import get_alert_manager, AlertSeverity
            
            alert_manager = get_alert_manager()
            alert_manager.trigger_alert(
                name=f"slow_requests_{endpoint.replace('/', '_')}",
                description=f"{count} requêtes lentes détectées sur {endpoint}",
                severity=AlertSeverity.WARNING.value,
                source="performance_monitor",
                labels={
                    'endpoint': endpoint,
                    'slow_request_count': str(count)
                }
            )
            
        except Exception as e:
            logger.error(f"Erreur alerte requêtes lentes: {e}")
    
    def _trigger_anomaly_alert(self, metric_name: str, value: float, mean_val: float, std_val: float):
        """Déclenche une alerte d'anomalie"""
        try:
            from .alert_manager import get_alert_manager, AlertSeverity
            
            alert_manager = get_alert_manager()
            alert_manager.trigger_alert(
                name=f"performance_anomaly_{metric_name}",
                description=f"Anomalie détectée pour {metric_name}: {value:.2f} (moyenne: {mean_val:.2f})",
                severity=AlertSeverity.WARNING.value,
                source="performance_monitor",
                labels={
                    'metric_name': metric_name,
                    'anomaly_value': str(value),
                    'mean_value': str(mean_val),
                    'std_deviation': str(std_val)
                }
            )
            
        except Exception as e:
            logger.error(f"Erreur alerte anomalie: {e}")
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calcule un percentile"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(percentile * len(sorted_values))
        if index >= len(sorted_values):
            index = len(sorted_values) - 1
        
        return sorted_values[index]
    
    def _save_metric_to_db(self, metric: PerformanceMetric):
        """Sauvegarde une métrique en base"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_metrics 
                (name, value, unit, timestamp, tenant_id, service, endpoint, labels)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metric.name, metric.value, metric.unit, metric.timestamp,
                metric.tenant_id, metric.service, metric.endpoint,
                json.dumps(metric.labels) if metric.labels else None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde métrique: {e}")
    
    def _save_bottleneck_to_db(self, bottleneck: BottleneckDetection):
        """Sauvegarde un goulot d'étranglement en base"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO bottlenecks 
                (component, severity, description, impact_score, recommendations, detected_at, metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                bottleneck.component, bottleneck.severity, bottleneck.description,
                bottleneck.impact_score, json.dumps(bottleneck.recommendations),
                bottleneck.detected_at, json.dumps(bottleneck.metrics)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde bottleneck: {e}")
    
    def _save_report_to_db(self, report: PerformanceReport):
        """Sauvegarde un rapport en base"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_reports 
                (start_time, end_time, overall_score, level, summary, metrics, 
                 bottlenecks, recommendations, tenant_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                report.start_time, report.end_time, report.overall_score,
                report.level, report.summary, json.dumps(report.metrics),
                json.dumps([asdict(b) for b in report.bottlenecks]),
                json.dumps(report.recommendations), report.tenant_id, time.time()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde rapport: {e}")
    
    def _save_slow_request_to_db(self, slow_request: Dict):
        """Sauvegarde une requête lente en base"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO slow_requests 
                (endpoint, method, duration_ms, timestamp, tenant_id, user_agent)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                slow_request['endpoint'], slow_request['method'],
                slow_request['duration_ms'], slow_request['timestamp'],
                slow_request['tenant_id'], slow_request['user_agent']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde requête lente: {e}")
    
    def _cleanup_old_data(self):
        """Nettoie les anciennes données"""
        try:
            cutoff_time = time.time() - (self.config['retention_days'] * 24 * 3600)
            
            # Nettoyage mémoire
            with self.lock:
                # Nettoyage des goulots d'étranglement résolus
                self.bottlenecks = [b for b in self.bottlenecks 
                                  if time.time() - b.detected_at < 24 * 3600]  # 24h
                
                # Nettoyage de l'historique ML
                for metric_name in list(self.metrics_history.keys()):
                    history = self.metrics_history[metric_name]
                    # Garde seulement les 1000 dernières valeurs
                    if len(history) > 1000:
                        self.metrics_history[metric_name] = deque(
                            list(history)[-1000:], maxlen=1000
                        )
            
            # Nettoyage base de données
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            tables_to_clean = [
                'performance_metrics',
                'bottlenecks', 
                'performance_reports',
                'slow_requests'
            ]
            
            total_deleted = 0
            for table in tables_to_clean:
                cursor.execute(f'DELETE FROM {table} WHERE timestamp < ? OR detected_at < ? OR created_at < ?', 
                             (cutoff_time, cutoff_time, cutoff_time))
                total_deleted += cursor.rowcount
            
            conn.commit()
            conn.close()
            
            if total_deleted > 0:
                logger.info(f"Nettoyé {total_deleted} enregistrements anciens")
                
        except Exception as e:
            logger.error(f"Erreur nettoyage données: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des performances actuelles"""
        try:
            # Génération d'un rapport sur les 15 dernières minutes
            report = self.generate_performance_report(15)
            
            return {
                'overall_score': report.overall_score,
                'level': report.level,
                'summary': report.summary,
                'stats': self.stats.copy(),
                'active_bottlenecks': len([b for b in self.bottlenecks 
                                         if time.time() - b.detected_at < 3600]),  # 1h
                'slow_requests_last_hour': len([r for r in self.slow_requests 
                                               if time.time() - r['timestamp'] < 3600]),
                'metrics_buffer_size': len(self.metrics_buffer),
                'redis_connected': self.redis_client is not None and self._test_redis(),
                'collection_active': self.collection_thread and self.collection_thread.is_alive(),
                'analysis_active': self.analysis_thread and self.analysis_thread.is_alive(),
                'profiling_active': self.profiling_thread and self.profiling_thread.is_alive() if self.profiling_thread else False,
                'config': self.config
            }
            
        except Exception as e:
            logger.error(f"Erreur résumé performance: {e}")
            return {'error': str(e)}
    
    def _test_redis(self) -> bool:
        """Test la connexion Redis"""
        try:
            self.redis_client.ping()
            return True
        except:
            return False

# Factory pour instance globale
_performance_monitor_instance = None

def get_performance_monitor(config: Optional[Dict] = None) -> PerformanceMonitor:
    """
    Retourne l'instance globale du moniteur de performance
    
    Args:
        config: Configuration optionnelle
        
    Returns:
        Instance de PerformanceMonitor
    """
    global _performance_monitor_instance
    
    if _performance_monitor_instance is None:
        _performance_monitor_instance = PerformanceMonitor(config)
    
    return _performance_monitor_instance

# Fonctions de convenance
def track_request_performance(endpoint: str, method: str = "GET", duration_ms: float = 0,
                            status_code: int = 200, tenant_id: Optional[str] = None):
    """Fonction de convenance pour tracker les performances de requête"""
    monitor = get_performance_monitor()
    monitor.track_request_performance(endpoint, method, duration_ms, status_code, tenant_id)

def record_performance_metric(name: str, value: float, unit: str = "",
                            tenant_id: Optional[str] = None):
    """Fonction de convenance pour enregistrer une métrique de performance"""
    monitor = get_performance_monitor()
    monitor.record_metric(name, value, unit, tenant_id=tenant_id)
