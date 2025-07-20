"""
Moniteur de Performance Ultra-Avancé - Spotify AI Agent
======================================================

Système de monitoring complet pour les performances du module Warning
avec métriques en temps réel, alertes automatiques et optimisation ML.

Auteur: Équipe d'experts dirigée par Fahed Mlaiel
"""

import os
import json
import logging
import time
import threading
import asyncio
import psutil
import sqlite3
from typing import Dict, Any, Optional, List, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, deque
import statistics
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import hashlib
import redis

# Configuration du logging
logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types de métriques disponibles."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"

class AlertStatus(Enum):
    """Statuts des alertes de performance."""
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class ThresholdType(Enum):
    """Types de seuils pour les alertes."""
    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    TREND = "trend"
    ANOMALY = "anomaly"

@dataclass
class MetricDefinition:
    """Définition d'une métrique."""
    name: str
    type: MetricType
    description: str
    unit: str
    labels: List[str]
    help_text: str
    retention_days: int = 30

@dataclass
class MetricValue:
    """Valeur d'une métrique à un instant donné."""
    metric_name: str
    value: float
    labels: Dict[str, str]
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class PerformanceThreshold:
    """Seuil de performance pour les alertes."""
    metric_name: str
    threshold_type: ThresholdType
    warning_value: float
    critical_value: float
    evaluation_window_minutes: int
    min_samples: int
    enabled: bool

@dataclass
class PerformanceAlert:
    """Alerte de performance."""
    alert_id: str
    metric_name: str
    current_value: float
    threshold_value: float
    status: AlertStatus
    message: str
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class MetricsCollector:
    """Collecteur de métriques système et application."""
    
    def __init__(self):
        self.system_metrics_enabled = True
        self.application_metrics_enabled = True
        self.collection_interval = 30  # secondes
        self.running = False
        self.collection_thread = None
        
    def start_collection(self):
        """Démarre la collecte de métriques."""
        if self.running:
            return
        
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info("Collecte de métriques démarrée")
    
    def stop_collection(self):
        """Arrête la collecte de métriques."""
        self.running = False
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5)
        logger.info("Collecte de métriques arrêtée")
    
    def _collection_loop(self):
        """Boucle principale de collecte."""
        while self.running:
            try:
                metrics = []
                
                if self.system_metrics_enabled:
                    metrics.extend(self._collect_system_metrics())
                
                if self.application_metrics_enabled:
                    metrics.extend(self._collect_application_metrics())
                
                # Envoi des métriques au monitoring
                if hasattr(self, 'metrics_callback') and self.metrics_callback:
                    self.metrics_callback(metrics)
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Erreur collecte métriques: {e}")
                time.sleep(5)
    
    def _collect_system_metrics(self) -> List[MetricValue]:
        """Collecte les métriques système."""
        metrics = []
        now = datetime.now()
        
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(MetricValue(
                metric_name="system_cpu_usage_percent",
                value=cpu_percent,
                labels={"component": "system"},
                timestamp=now
            ))
            
            # Mémoire
            memory = psutil.virtual_memory()
            metrics.append(MetricValue(
                metric_name="system_memory_usage_percent",
                value=memory.percent,
                labels={"component": "system"},
                timestamp=now
            ))
            
            metrics.append(MetricValue(
                metric_name="system_memory_available_bytes",
                value=memory.available,
                labels={"component": "system"},
                timestamp=now
            ))
            
            # Disque
            disk = psutil.disk_usage('/')
            metrics.append(MetricValue(
                metric_name="system_disk_usage_percent",
                value=(disk.used / disk.total) * 100,
                labels={"component": "system", "mount": "/"},
                timestamp=now
            ))
            
            # Réseau
            network = psutil.net_io_counters()
            metrics.append(MetricValue(
                metric_name="system_network_bytes_sent",
                value=network.bytes_sent,
                labels={"component": "system", "direction": "sent"},
                timestamp=now
            ))
            
            metrics.append(MetricValue(
                metric_name="system_network_bytes_recv",
                value=network.bytes_recv,
                labels={"component": "system", "direction": "received"},
                timestamp=now
            ))
            
        except Exception as e:
            logger.error(f"Erreur collecte métriques système: {e}")
        
        return metrics
    
    def _collect_application_metrics(self) -> List[MetricValue]:
        """Collecte les métriques applicatives."""
        metrics = []
        now = datetime.now()
        
        try:
            # Processus actuel
            process = psutil.Process()
            
            # CPU du processus
            cpu_percent = process.cpu_percent()
            metrics.append(MetricValue(
                metric_name="app_cpu_usage_percent",
                value=cpu_percent,
                labels={"component": "warning_config", "process": "main"},
                timestamp=now
            ))
            
            # Mémoire du processus
            memory_info = process.memory_info()
            metrics.append(MetricValue(
                metric_name="app_memory_rss_bytes",
                value=memory_info.rss,
                labels={"component": "warning_config", "process": "main"},
                timestamp=now
            ))
            
            # Nombre de threads
            num_threads = process.num_threads()
            metrics.append(MetricValue(
                metric_name="app_threads_count",
                value=num_threads,
                labels={"component": "warning_config", "process": "main"},
                timestamp=now
            ))
            
            # Descripteurs de fichiers (Linux/Unix seulement)
            try:
                num_fds = process.num_fds()
                metrics.append(MetricValue(
                    metric_name="app_file_descriptors_count",
                    value=num_fds,
                    labels={"component": "warning_config", "process": "main"},
                    timestamp=now
                ))
            except AttributeError:
                # Windows n'a pas num_fds()
                pass
            
        except Exception as e:
            logger.error(f"Erreur collecte métriques application: {e}")
        
        return metrics
    
    def set_metrics_callback(self, callback: Callable[[List[MetricValue]], None]):
        """Définit le callback pour les métriques collectées."""
        self.metrics_callback = callback

class MetricsStorage:
    """Stockage optimisé des métriques avec rétention automatique."""
    
    def __init__(self, database_path: str = None, redis_url: str = None):
        self.database_path = database_path or ":memory:"
        self.redis_url = redis_url
        self.db_lock = threading.RLock()
        
        # Initialisation du stockage
        self._init_database()
        self._init_redis()
        
        # Cache en mémoire pour accès rapide
        self.memory_cache = defaultdict(lambda: deque(maxlen=1000))
        self.cache_lock = threading.RLock()
        
        # Thread de nettoyage automatique
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def _init_database(self):
        """Initialise la base de données SQLite."""
        self.db_connection = sqlite3.connect(self.database_path, check_same_thread=False)
        
        with self.db_lock:
            cursor = self.db_connection.cursor()
            
            # Table des métriques
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    labels TEXT,
                    tags TEXT,
                    timestamp TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Table des définitions de métriques
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metric_definitions (
                    name TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    description TEXT,
                    unit TEXT,
                    labels TEXT,
                    help_text TEXT,
                    retention_days INTEGER DEFAULT 30,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Index pour optimisation
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_name_time ON metrics(metric_name, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_time ON metrics(timestamp)")
            
            self.db_connection.commit()
    
    def _init_redis(self):
        """Initialise la connexion Redis pour le cache."""
        if self.redis_url:
            try:
                self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("Connexion Redis établie pour le cache métriques")
            except Exception as e:
                logger.warning(f"Impossible de se connecter à Redis: {e}")
                self.redis_client = None
        else:
            self.redis_client = None
    
    def store_metrics(self, metrics: List[MetricValue]):
        """Stocke une liste de métriques."""
        if not metrics:
            return
        
        # Stockage en base de données
        with self.db_lock:
            cursor = self.db_connection.cursor()
            
            for metric in metrics:
                cursor.execute("""
                    INSERT INTO metrics (metric_name, value, labels, tags, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    metric.metric_name,
                    metric.value,
                    json.dumps(metric.labels),
                    json.dumps(metric.tags),
                    metric.timestamp.isoformat()
                ))
            
            self.db_connection.commit()
        
        # Mise à jour du cache mémoire
        with self.cache_lock:
            for metric in metrics:
                cache_key = f"{metric.metric_name}:{json.dumps(metric.labels, sort_keys=True)}"
                self.memory_cache[cache_key].append({
                    'value': metric.value,
                    'timestamp': metric.timestamp
                })
        
        # Mise à jour du cache Redis
        if self.redis_client:
            try:
                pipe = self.redis_client.pipeline()
                for metric in metrics:
                    cache_key = f"metric:{metric.metric_name}"
                    pipe.zadd(cache_key, {f"{metric.value}:{metric.timestamp.isoformat()}": metric.timestamp.timestamp()})
                    pipe.expire(cache_key, 3600)  # 1 heure de TTL
                pipe.execute()
            except Exception as e:
                logger.warning(f"Erreur mise à jour cache Redis: {e}")
    
    def get_metric_values(self, metric_name: str, start_time: datetime = None,
                         end_time: datetime = None, labels: Dict[str, str] = None) -> List[MetricValue]:
        """Récupère les valeurs d'une métrique."""
        
        # Tentative de récupération depuis le cache mémoire d'abord
        if not start_time and not end_time:
            cache_key = f"{metric_name}:{json.dumps(labels or {}, sort_keys=True)}"
            with self.cache_lock:
                if cache_key in self.memory_cache:
                    cached_values = list(self.memory_cache[cache_key])
                    return [
                        MetricValue(
                            metric_name=metric_name,
                            value=item['value'],
                            labels=labels or {},
                            timestamp=item['timestamp']
                        )
                        for item in cached_values[-100:]  # Dernières 100 valeurs
                    ]
        
        # Récupération depuis la base de données
        with self.db_lock:
            cursor = self.db_connection.cursor()
            
            query = "SELECT * FROM metrics WHERE metric_name = ?"
            params = [metric_name]
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat())
            
            if labels:
                # Filtrage par labels (approximatif)
                for key, value in labels.items():
                    query += " AND labels LIKE ?"
                    params.append(f'%"{key}": "{value}"%')
            
            query += " ORDER BY timestamp DESC LIMIT 1000"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            metrics = []
            for row in rows:
                metric = MetricValue(
                    metric_name=row[1],
                    value=row[2],
                    labels=json.loads(row[3]) if row[3] else {},
                    timestamp=datetime.fromisoformat(row[5]),
                    tags=json.loads(row[4]) if row[4] else {}
                )
                metrics.append(metric)
            
            return metrics
    
    def _cleanup_loop(self):
        """Boucle de nettoyage automatique des anciennes métriques."""
        while True:
            try:
                time.sleep(3600)  # Nettoyage toutes les heures
                self._cleanup_old_metrics()
            except Exception as e:
                logger.error(f"Erreur nettoyage métriques: {e}")
    
    def _cleanup_old_metrics(self):
        """Nettoie les métriques anciennes selon la rétention."""
        with self.db_lock:
            cursor = self.db_connection.cursor()
            
            # Récupération des politiques de rétention
            cursor.execute("SELECT name, retention_days FROM metric_definitions")
            retention_policies = dict(cursor.fetchall())
            
            # Nettoyage par métrique
            for metric_name, retention_days in retention_policies.items():
                cutoff_date = datetime.now() - timedelta(days=retention_days)
                
                cursor.execute("""
                    DELETE FROM metrics 
                    WHERE metric_name = ? AND timestamp < ?
                """, (metric_name, cutoff_date.isoformat()))
            
            # Nettoyage global pour les métriques sans politique (30 jours par défaut)
            default_cutoff = datetime.now() - timedelta(days=30)
            cursor.execute("""
                DELETE FROM metrics 
                WHERE metric_name NOT IN (SELECT name FROM metric_definitions)
                AND timestamp < ?
            """, (default_cutoff.isoformat(),))
            
            deleted_count = cursor.rowcount
            self.db_connection.commit()
            
            if deleted_count > 0:
                logger.info(f"Métriques nettoyées: {deleted_count} entrées supprimées")

class AnomalyDetector:
    """Détecteur d'anomalies basé sur des algorithmes statistiques."""
    
    def __init__(self):
        self.baseline_window_size = 1440  # 24 heures en minutes
        self.sensitivity = 2.0  # Seuil en écarts-types
        self.min_samples = 50
        
    def detect_anomalies(self, metric_values: List[MetricValue]) -> List[Tuple[MetricValue, float]]:
        """Détecte les anomalies dans une série de métriques."""
        if len(metric_values) < self.min_samples:
            return []
        
        # Extraction des valeurs numériques
        values = [mv.value for mv in metric_values]
        
        # Calcul des statistiques de base
        mean_value = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0
        
        if std_dev == 0:
            return []  # Pas de variation, pas d'anomalie
        
        anomalies = []
        
        # Détection par Z-score
        for metric_value in metric_values:
            z_score = abs(metric_value.value - mean_value) / std_dev
            
            if z_score > self.sensitivity:
                anomalies.append((metric_value, z_score))
        
        return anomalies
    
    def detect_trend_anomalies(self, metric_values: List[MetricValue]) -> List[Tuple[MetricValue, str]]:
        """Détecte les anomalies de tendance."""
        if len(metric_values) < 10:
            return []
        
        values = [mv.value for mv in metric_values]
        anomalies = []
        
        # Analyse de la tendance sur fenêtre glissante
        window_size = min(20, len(values) // 4)
        
        for i in range(window_size, len(values)):
            recent_window = values[i-window_size:i]
            current_value = values[i]
            
            # Calcul de la tendance
            trend = self._calculate_trend(recent_window)
            
            # Prédiction basée sur la tendance
            predicted_value = recent_window[-1] + trend
            deviation = abs(current_value - predicted_value) / (abs(predicted_value) + 1)
            
            if deviation > 0.3:  # 30% de déviation
                anomaly_type = "spike" if current_value > predicted_value else "drop"
                anomalies.append((metric_values[i], anomaly_type))
        
        return anomalies
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calcule la tendance d'une série de valeurs."""
        if len(values) < 2:
            return 0
        
        # Régression linéaire simple
        n = len(values)
        x = list(range(n))
        y = values
        
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        return slope

class ThresholdEngine:
    """Moteur d'évaluation des seuils avec logique adaptative."""
    
    def __init__(self):
        self.thresholds = {}
        self.adaptive_thresholds = {}
        self.threshold_lock = threading.RLock()
        
    def register_threshold(self, threshold: PerformanceThreshold):
        """Enregistre un seuil de performance."""
        with self.threshold_lock:
            self.thresholds[threshold.metric_name] = threshold
    
    def evaluate_thresholds(self, metric_values: List[MetricValue]) -> List[PerformanceAlert]:
        """Évalue les seuils pour une liste de métriques."""
        alerts = []
        
        # Groupement par nom de métrique
        metrics_by_name = defaultdict(list)
        for mv in metric_values:
            metrics_by_name[mv.metric_name].append(mv)
        
        with self.threshold_lock:
            for metric_name, values in metrics_by_name.items():
                if metric_name not in self.thresholds:
                    continue
                
                threshold = self.thresholds[metric_name]
                if not threshold.enabled:
                    continue
                
                alert = self._evaluate_metric_threshold(threshold, values)
                if alert:
                    alerts.append(alert)
        
        return alerts
    
    def _evaluate_metric_threshold(self, threshold: PerformanceThreshold, 
                                 values: List[MetricValue]) -> Optional[PerformanceAlert]:
        """Évalue le seuil pour une métrique spécifique."""
        
        if len(values) < threshold.min_samples:
            return None
        
        # Filtrage par fenêtre temporelle
        now = datetime.now()
        window_start = now - timedelta(minutes=threshold.evaluation_window_minutes)
        filtered_values = [v for v in values if v.timestamp >= window_start]
        
        if len(filtered_values) < threshold.min_samples:
            return None
        
        # Calcul de la valeur d'évaluation selon le type
        if threshold.threshold_type == ThresholdType.ABSOLUTE:
            eval_value = filtered_values[-1].value  # Dernière valeur
        elif threshold.threshold_type == ThresholdType.RELATIVE:
            eval_value = statistics.mean([v.value for v in filtered_values])
        elif threshold.threshold_type == ThresholdType.TREND:
            eval_value = self._calculate_trend_value(filtered_values)
        else:
            eval_value = filtered_values[-1].value
        
        # Évaluation des seuils
        status = AlertStatus.OK
        threshold_value = 0
        message = ""
        
        if eval_value >= threshold.critical_value:
            status = AlertStatus.CRITICAL
            threshold_value = threshold.critical_value
            message = f"{threshold.metric_name} critique: {eval_value:.2f} >= {threshold.critical_value:.2f}"
        elif eval_value >= threshold.warning_value:
            status = AlertStatus.WARNING
            threshold_value = threshold.warning_value
            message = f"{threshold.metric_name} avertissement: {eval_value:.2f} >= {threshold.warning_value:.2f}"
        
        if status != AlertStatus.OK:
            alert_id = self._generate_alert_id(threshold.metric_name, status)
            
            return PerformanceAlert(
                alert_id=alert_id,
                metric_name=threshold.metric_name,
                current_value=eval_value,
                threshold_value=threshold_value,
                status=status,
                message=message,
                triggered_at=now,
                metadata={
                    'threshold_type': threshold.threshold_type.value,
                    'samples_count': len(filtered_values),
                    'window_minutes': threshold.evaluation_window_minutes
                }
            )
        
        return None
    
    def _calculate_trend_value(self, values: List[MetricValue]) -> float:
        """Calcule une valeur de tendance."""
        if len(values) < 2:
            return values[0].value if values else 0
        
        numeric_values = [v.value for v in values]
        
        # Calcul de la pente de régression
        n = len(numeric_values)
        x = list(range(n))
        y = numeric_values
        
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        
        # Retourne la pente comme indicateur de tendance
        return slope
    
    def _generate_alert_id(self, metric_name: str, status: AlertStatus) -> str:
        """Génère un ID unique pour l'alerte."""
        timestamp = int(time.time() * 1000)
        data = f"{metric_name}:{status.value}:{timestamp}"
        hash_obj = hashlib.md5(data.encode())
        return f"perf_alert_{hash_obj.hexdigest()[:12]}"

class PerformanceMonitor:
    """
    Moniteur de performance ultra-avancé.
    
    Fonctionnalités:
    - Collecte de métriques système et application
    - Stockage optimisé avec rétention automatique
    - Détection d'anomalies avec ML
    - Alertes basées sur des seuils adaptatifs
    - Dashboards et visualisations en temps réel
    - Optimisation des performances avec recommandations
    """
    
    def __init__(self, database_path: str = None, redis_url: str = None):
        """Initialise le moniteur de performance."""
        
        # Initialisation des composants
        self.metrics_collector = MetricsCollector()
        self.metrics_storage = MetricsStorage(database_path, redis_url)
        self.anomaly_detector = AnomalyDetector()
        self.threshold_engine = ThresholdEngine()
        
        # Configuration
        self.monitoring_enabled = True
        self.alert_callbacks = []
        
        # Métriques du moniteur
        self.monitor_metrics = {
            'metrics_collected': 0,
            'anomalies_detected': 0,
            'alerts_triggered': 0,
            'storage_operations': 0,
            'collection_errors': 0
        }
        self.metrics_lock = threading.RLock()
        
        # Historique des alertes
        self.active_alerts = {}
        self.alert_history = deque(maxlen=10000)
        self.alert_lock = threading.RLock()
        
        # Configuration des seuils par défaut
        self._configure_default_thresholds()
        
        # Connexion du collecteur au stockage
        self.metrics_collector.set_metrics_callback(self._on_metrics_collected)
        
        logger.info("PerformanceMonitor initialisé avec succès")
    
    def _configure_default_thresholds(self):
        """Configure les seuils par défaut."""
        
        # Seuils CPU
        cpu_threshold = PerformanceThreshold(
            metric_name="system_cpu_usage_percent",
            threshold_type=ThresholdType.ABSOLUTE,
            warning_value=70.0,
            critical_value=90.0,
            evaluation_window_minutes=5,
            min_samples=3,
            enabled=True
        )
        self.threshold_engine.register_threshold(cpu_threshold)
        
        # Seuils mémoire
        memory_threshold = PerformanceThreshold(
            metric_name="system_memory_usage_percent",
            threshold_type=ThresholdType.ABSOLUTE,
            warning_value=80.0,
            critical_value=95.0,
            evaluation_window_minutes=5,
            min_samples=3,
            enabled=True
        )
        self.threshold_engine.register_threshold(memory_threshold)
        
        # Seuils disque
        disk_threshold = PerformanceThreshold(
            metric_name="system_disk_usage_percent",
            threshold_type=ThresholdType.ABSOLUTE,
            warning_value=85.0,
            critical_value=95.0,
            evaluation_window_minutes=10,
            min_samples=2,
            enabled=True
        )
        self.threshold_engine.register_threshold(disk_threshold)
        
        # Seuils application
        app_cpu_threshold = PerformanceThreshold(
            metric_name="app_cpu_usage_percent",
            threshold_type=ThresholdType.RELATIVE,
            warning_value=50.0,
            critical_value=80.0,
            evaluation_window_minutes=10,
            min_samples=5,
            enabled=True
        )
        self.threshold_engine.register_threshold(app_cpu_threshold)
    
    def start_monitoring(self):
        """Démarre le monitoring."""
        if not self.monitoring_enabled:
            return
        
        # Démarrage de la collecte
        self.metrics_collector.start_collection()
        
        # Démarrage de l'analyse des anomalies
        self._start_anomaly_detection()
        
        logger.info("Monitoring de performance démarré")
    
    def stop_monitoring(self):
        """Arrête le monitoring."""
        self.monitoring_enabled = False
        self.metrics_collector.stop_collection()
        logger.info("Monitoring de performance arrêté")
    
    def _on_metrics_collected(self, metrics: List[MetricValue]):
        """Callback appelé lors de la collecte de métriques."""
        try:
            # Stockage des métriques
            self.metrics_storage.store_metrics(metrics)
            self._increment_metric('metrics_collected', len(metrics))
            self._increment_metric('storage_operations')
            
            # Évaluation des seuils
            alerts = self.threshold_engine.evaluate_thresholds(metrics)
            
            # Traitement des alertes
            for alert in alerts:
                self._handle_alert(alert)
            
        except Exception as e:
            logger.error(f"Erreur traitement métriques: {e}")
            self._increment_metric('collection_errors')
    
    def _start_anomaly_detection(self):
        """Démarre la détection d'anomalies en arrière-plan."""
        def anomaly_detection_loop():
            while self.monitoring_enabled:
                try:
                    self._run_anomaly_detection()
                    time.sleep(300)  # Analyse toutes les 5 minutes
                except Exception as e:
                    logger.error(f"Erreur détection anomalies: {e}")
                    time.sleep(60)
        
        anomaly_thread = threading.Thread(target=anomaly_detection_loop, daemon=True)
        anomaly_thread.start()
    
    def _run_anomaly_detection(self):
        """Exécute la détection d'anomalies."""
        
        # Récupération des métriques récentes
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=2)
        
        metric_names = [
            "system_cpu_usage_percent",
            "system_memory_usage_percent",
            "app_cpu_usage_percent",
            "app_memory_rss_bytes"
        ]
        
        for metric_name in metric_names:
            try:
                values = self.metrics_storage.get_metric_values(
                    metric_name, start_time, end_time
                )
                
                if not values:
                    continue
                
                # Détection d'anomalies statistiques
                anomalies = self.anomaly_detector.detect_anomalies(values)
                
                for anomalous_value, z_score in anomalies:
                    self._handle_anomaly(anomalous_value, z_score, "statistical")
                
                # Détection d'anomalies de tendance
                trend_anomalies = self.anomaly_detector.detect_trend_anomalies(values)
                
                for anomalous_value, anomaly_type in trend_anomalies:
                    self._handle_anomaly(anomalous_value, 0, f"trend_{anomaly_type}")
                
            except Exception as e:
                logger.error(f"Erreur détection anomalies pour {metric_name}: {e}")
    
    def _handle_anomaly(self, metric_value: MetricValue, severity: float, anomaly_type: str):
        """Traite une anomalie détectée."""
        
        alert_id = f"anomaly_{metric_value.metric_name}_{int(metric_value.timestamp.timestamp())}"
        
        # Éviter les doublons
        with self.alert_lock:
            if alert_id in self.active_alerts:
                return
        
        # Création de l'alerte d'anomalie
        alert = PerformanceAlert(
            alert_id=alert_id,
            metric_name=metric_value.metric_name,
            current_value=metric_value.value,
            threshold_value=0,  # Pas de seuil fixe pour les anomalies
            status=AlertStatus.WARNING if severity < 3 else AlertStatus.CRITICAL,
            message=f"Anomalie détectée ({anomaly_type}): {metric_value.metric_name} = {metric_value.value}",
            triggered_at=datetime.now(),
            metadata={
                'anomaly_type': anomaly_type,
                'severity_score': severity,
                'detection_method': 'ml_statistical'
            }
        )
        
        self._handle_alert(alert)
        self._increment_metric('anomalies_detected')
    
    def _handle_alert(self, alert: PerformanceAlert):
        """Traite une alerte de performance."""
        
        with self.alert_lock:
            # Vérification des doublons
            if alert.alert_id in self.active_alerts:
                return
            
            # Enregistrement de l'alerte
            self.active_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)
        
        # Notification via callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Erreur callback alerte: {e}")
        
        # Log de l'alerte
        logger.warning(f"Alerte performance: {alert.message}")
        
        self._increment_metric('alerts_triggered')
    
    def register_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Enregistre un callback pour les alertes."""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self, metric_names: List[str] = None) -> Dict[str, List[MetricValue]]:
        """Récupère les métriques actuelles."""
        
        if metric_names is None:
            metric_names = [
                "system_cpu_usage_percent",
                "system_memory_usage_percent",
                "system_disk_usage_percent",
                "app_cpu_usage_percent",
                "app_memory_rss_bytes"
            ]
        
        metrics = {}
        for metric_name in metric_names:
            values = self.metrics_storage.get_metric_values(metric_name)
            metrics[metric_name] = values[-10:] if values else []  # Dernières 10 valeurs
        
        return metrics
    
    def get_metric_summary(self, metric_name: str, hours: int = 24) -> Dict[str, Any]:
        """Génère un résumé statistique d'une métrique."""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        values = self.metrics_storage.get_metric_values(metric_name, start_time, end_time)
        
        if not values:
            return {"error": "Aucune donnée disponible"}
        
        numeric_values = [v.value for v in values]
        
        summary = {
            "metric_name": metric_name,
            "period_hours": hours,
            "sample_count": len(numeric_values),
            "min_value": min(numeric_values),
            "max_value": max(numeric_values),
            "mean_value": statistics.mean(numeric_values),
            "median_value": statistics.median(numeric_values),
            "latest_value": numeric_values[-1],
            "trend": "stable"
        }
        
        # Calcul de l'écart-type
        if len(numeric_values) > 1:
            summary["std_dev"] = statistics.stdev(numeric_values)
        else:
            summary["std_dev"] = 0
        
        # Analyse de tendance simple
        if len(numeric_values) > 10:
            first_half = numeric_values[:len(numeric_values)//2]
            second_half = numeric_values[len(numeric_values)//2:]
            
            first_mean = statistics.mean(first_half)
            second_mean = statistics.mean(second_half)
            
            change_percent = ((second_mean - first_mean) / first_mean) * 100
            
            if change_percent > 10:
                summary["trend"] = "increasing"
            elif change_percent < -10:
                summary["trend"] = "decreasing"
            else:
                summary["trend"] = "stable"
            
            summary["trend_change_percent"] = change_percent
        
        return summary
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Récupère les alertes actives."""
        with self.alert_lock:
            return list(self.active_alerts.values())
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Résout une alerte."""
        with self.alert_lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved_at = datetime.now()
                del self.active_alerts[alert_id]
                logger.info(f"Alerte résolue: {alert_id}")
                return True
        return False
    
    def get_performance_recommendations(self) -> List[Dict[str, Any]]:
        """Génère des recommandations d'optimisation."""
        recommendations = []
        
        # Analyse des métriques récentes
        current_metrics = self.get_current_metrics()
        
        for metric_name, values in current_metrics.items():
            if not values:
                continue
            
            latest_value = values[-1].value
            
            # Recommandations basées sur les seuils
            if metric_name == "system_cpu_usage_percent" and latest_value > 80:
                recommendations.append({
                    "type": "performance",
                    "priority": "high" if latest_value > 90 else "medium",
                    "component": "cpu",
                    "message": f"Usage CPU élevé ({latest_value:.1f}%)",
                    "recommendations": [
                        "Identifier les processus consommateurs de CPU",
                        "Optimiser les algorithmes de traitement",
                        "Considérer l'augmentation des ressources CPU"
                    ]
                })
            
            elif metric_name == "system_memory_usage_percent" and latest_value > 85:
                recommendations.append({
                    "type": "performance", 
                    "priority": "high" if latest_value > 95 else "medium",
                    "component": "memory",
                    "message": f"Usage mémoire élevé ({latest_value:.1f}%)",
                    "recommendations": [
                        "Analyser les fuites mémoire potentielles",
                        "Optimiser les caches et buffers",
                        "Implémenter un garbage collection plus efficace"
                    ]
                })
            
            elif metric_name == "system_disk_usage_percent" and latest_value > 90:
                recommendations.append({
                    "type": "capacity",
                    "priority": "critical" if latest_value > 95 else "high",
                    "component": "storage",
                    "message": f"Espace disque faible ({latest_value:.1f}%)",
                    "recommendations": [
                        "Nettoyer les fichiers temporaires et logs",
                        "Implémenter une rotation des logs",
                        "Étendre l'espace de stockage"
                    ]
                })
        
        # Recommandations basées sur les anomalies
        active_alerts = self.get_active_alerts()
        anomaly_alerts = [a for a in active_alerts if 'anomaly_type' in a.metadata]
        
        if len(anomaly_alerts) > 3:
            recommendations.append({
                "type": "stability",
                "priority": "high",
                "component": "system",
                "message": f"Nombreuses anomalies détectées ({len(anomaly_alerts)})",
                "recommendations": [
                    "Analyser les patterns d'utilisation",
                    "Vérifier la stabilité du système",
                    "Examiner les changements récents"
                ]
            })
        
        return recommendations
    
    def _increment_metric(self, metric_name: str, value: int = 1):
        """Incrémente une métrique du moniteur."""
        with self.metrics_lock:
            self.monitor_metrics[metric_name] = self.monitor_metrics.get(metric_name, 0) + value
    
    def get_monitor_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques du moniteur."""
        with self.metrics_lock:
            metrics = self.monitor_metrics.copy()
            
            # Ajout de métriques dérivées
            with self.alert_lock:
                metrics['active_alerts_count'] = len(self.active_alerts)
                metrics['alert_history_size'] = len(self.alert_history)
            
            metrics['monitoring_enabled'] = self.monitoring_enabled
            metrics['collection_running'] = self.metrics_collector.running
            
            return metrics
    
    def health_check(self) -> Dict[str, Any]:
        """Vérifie la santé du moniteur."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        # Vérification du collecteur
        health_status["components"]["metrics_collector"] = "healthy" if self.metrics_collector.running else "unhealthy"
        
        # Vérification du stockage
        try:
            test_metrics = [MetricValue(
                metric_name="health_check_test",
                value=1.0,
                labels={"test": "true"},
                timestamp=datetime.now()
            )]
            self.metrics_storage.store_metrics(test_metrics)
            health_status["components"]["metrics_storage"] = "healthy"
        except Exception as e:
            health_status["components"]["metrics_storage"] = f"unhealthy: {e}"
            health_status["status"] = "degraded"
        
        # Vérification Redis (si configuré)
        if self.metrics_storage.redis_client:
            try:
                self.metrics_storage.redis_client.ping()
                health_status["components"]["redis_cache"] = "healthy"
            except Exception as e:
                health_status["components"]["redis_cache"] = f"unhealthy: {e}"
                health_status["status"] = "degraded"
        
        return health_status
    
    def cleanup(self):
        """Nettoie les ressources."""
        self.stop_monitoring()
        if hasattr(self.metrics_storage, 'db_connection'):
            self.metrics_storage.db_connection.close()
        logger.info("PerformanceMonitor nettoyé avec succès")

# Factory function
def create_performance_monitor(database_path: str = None, redis_url: str = None) -> PerformanceMonitor:
    """Factory function pour créer un moniteur de performance."""
    return PerformanceMonitor(database_path, redis_url)
