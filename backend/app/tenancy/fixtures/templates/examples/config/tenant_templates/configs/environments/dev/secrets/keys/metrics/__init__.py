#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enterprise Metrics Collection & Analytics System
===============================================

Ultra-advanced metrics collection, processing, and analytics system for
cryptographic keys and security infrastructure with real-time monitoring,
AI-powered insights, and comprehensive reporting capabilities.

Expert Development Team:
- Lead Dev + AI Architect
- Senior Backend Developer (Python/FastAPI/Django)
- ML Engineer (TensorFlow/PyTorch/Hugging Face)  
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Backend Security Specialist
- Microservices Architect

Features:
- Real-time metrics collection and processing
- AI-powered anomaly detection and insights
- Multi-dimensional data analysis
- Performance optimization recommendations
- Security metrics and compliance tracking
- Automated alerting and notification system
- Advanced visualization and reporting
- Time-series data management
- Predictive analytics and forecasting
- Custom metrics framework
"""

import asyncio
import json
import logging
import sqlite3
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics
import pickle
import hashlib
import zlib
import gzip
import base64

try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    HAS_ML_LIBS = True
except ImportError:
    HAS_ML_LIBS = False

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    HAS_POSTGRESQL = True
except ImportError:
    HAS_POSTGRESQL = False


class MetricType(Enum):
    """Types de métriques supportés."""
    COUNTER = auto()
    GAUGE = auto()
    HISTOGRAM = auto()
    TIMER = auto()
    RATE = auto()
    RATIO = auto()
    DISTRIBUTION = auto()
    CUMULATIVE = auto()


class MetricSeverity(Enum):
    """Niveaux de sévérité des métriques."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MetricCategory(Enum):
    """Catégories de métriques."""
    SECURITY = "security"
    PERFORMANCE = "performance"
    AVAILABILITY = "availability"
    RELIABILITY = "reliability"
    COMPLIANCE = "compliance"
    BUSINESS = "business"
    SYSTEM = "system"
    NETWORK = "network"
    STORAGE = "storage"
    CUSTOM = "custom"


@dataclass
class MetricDataPoint:
    """Point de données métrique avec métadonnées enrichies."""
    metric_id: str
    timestamp: datetime
    value: Union[int, float, str, bool, Dict, List]
    metric_type: MetricType
    category: MetricCategory
    severity: MetricSeverity = MetricSeverity.LOW
    tags: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    tenant_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    def __post_init__(self):
        """Post-processing après initialisation."""
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
        
        # Génération automatique d'ID si non fourni
        if not hasattr(self, '_id'):
            self._id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['metric_type'] = self.metric_type.name
        data['category'] = self.category.value
        data['severity'] = self.severity.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricDataPoint':
        """Création depuis un dictionnaire."""
        data['metric_type'] = MetricType[data['metric_type']]
        data['category'] = MetricCategory(data['category'])
        data['severity'] = MetricSeverity(data['severity'])
        return cls(**data)


@dataclass
class MetricAggregation:
    """Agrégation de métriques avec statistiques avancées."""
    metric_id: str
    start_time: datetime
    end_time: datetime
    count: int
    sum_value: float
    min_value: float
    max_value: float
    mean: float
    median: float
    std_dev: float
    percentiles: Dict[str, float] = field(default_factory=dict)
    anomalies_detected: int = 0
    trend_direction: str = "stable"
    confidence_score: float = 1.0
    
    def __post_init__(self):
        """Calculs post-initialisation."""
        if self.count > 0:
            self.avg = self.sum_value / self.count
        else:
            self.avg = 0.0


class MetricStorage(ABC):
    """Interface abstraite pour le stockage des métriques."""
    
    @abstractmethod
    async def store_metric(self, metric: MetricDataPoint) -> bool:
        """Stocke une métrique."""
        pass
    
    @abstractmethod
    async def get_metrics(self, query: Dict[str, Any]) -> List[MetricDataPoint]:
        """Récupère des métriques selon une requête."""
        pass
    
    @abstractmethod
    async def aggregate_metrics(self, 
                              metric_id: str, 
                              start_time: datetime, 
                              end_time: datetime) -> Optional[MetricAggregation]:
        """Agrège des métriques sur une période."""
        pass


class SQLiteMetricStorage(MetricStorage):
    """Stockage SQLite optimisé pour les métriques."""
    
    def __init__(self, db_path: str = "metrics.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialise la base de données SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id TEXT PRIMARY KEY,
                    metric_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    value TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    category TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    tags TEXT,
                    labels TEXT,
                    metadata TEXT,
                    source TEXT,
                    tenant_id TEXT,
                    correlation_id TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_id_timestamp 
                ON metrics(metric_id, timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_category_timestamp 
                ON metrics(category, timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_tenant_timestamp 
                ON metrics(tenant_id, timestamp)
            """)
    
    async def store_metric(self, metric: MetricDataPoint) -> bool:
        """Stocke une métrique de manière asynchrone."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO metrics (
                        id, metric_id, timestamp, value, metric_type, 
                        category, severity, tags, labels, metadata, 
                        source, tenant_id, correlation_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric._id, metric.metric_id, metric.timestamp.isoformat(),
                    json.dumps(metric.value), metric.metric_type.name,
                    metric.category.value, metric.severity.value,
                    json.dumps(metric.tags), json.dumps(metric.labels),
                    json.dumps(metric.metadata), metric.source,
                    metric.tenant_id, metric.correlation_id
                ))
            return True
        except Exception as e:
            logging.error(f"Erreur stockage métrique: {e}")
            return False
    
    async def get_metrics(self, query: Dict[str, Any]) -> List[MetricDataPoint]:
        """Récupère des métriques avec requête flexible."""
        conditions = []
        params = []
        
        if 'metric_id' in query:
            conditions.append("metric_id = ?")
            params.append(query['metric_id'])
        
        if 'start_time' in query:
            conditions.append("timestamp >= ?")
            params.append(query['start_time'].isoformat())
        
        if 'end_time' in query:
            conditions.append("timestamp <= ?")
            params.append(query['end_time'].isoformat())
        
        if 'category' in query:
            conditions.append("category = ?")
            params.append(query['category'])
        
        if 'tenant_id' in query:
            conditions.append("tenant_id = ?")
            params.append(query['tenant_id'])
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        sql = f"""
            SELECT * FROM metrics 
            WHERE {where_clause} 
            ORDER BY timestamp DESC
            LIMIT {query.get('limit', 1000)}
        """
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(sql, params)
                rows = cursor.fetchall()
                
                metrics = []
                for row in rows:
                    data = dict(row)
                    data['value'] = json.loads(data['value'])
                    data['tags'] = json.loads(data['tags'] or '{}')
                    data['labels'] = json.loads(data['labels'] or '{}')
                    data['metadata'] = json.loads(data['metadata'] or '{}')
                    data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                    data['metric_type'] = MetricType[data['metric_type']]
                    data['category'] = MetricCategory(data['category'])
                    data['severity'] = MetricSeverity(data['severity'])
                    
                    metrics.append(MetricDataPoint(**data))
                
                return metrics
        except Exception as e:
            logging.error(f"Erreur récupération métriques: {e}")
            return []
    
    async def aggregate_metrics(self, 
                              metric_id: str, 
                              start_time: datetime, 
                              end_time: datetime) -> Optional[MetricAggregation]:
        """Agrège des métriques avec statistiques avancées."""
        try:
            metrics = await self.get_metrics({
                'metric_id': metric_id,
                'start_time': start_time,
                'end_time': end_time
            })
            
            if not metrics:
                return None
            
            values = []
            for metric in metrics:
                if isinstance(metric.value, (int, float)):
                    values.append(float(metric.value))
            
            if not values:
                return None
            
            # Calculs statistiques
            count = len(values)
            sum_value = sum(values)
            min_value = min(values)
            max_value = max(values)
            mean = statistics.mean(values)
            median = statistics.median(values)
            std_dev = statistics.stdev(values) if count > 1 else 0.0
            
            # Percentiles
            percentiles = {}
            if count >= 2:
                percentiles = {
                    'p25': np.percentile(values, 25) if HAS_ML_LIBS else sorted(values)[count//4],
                    'p50': median,
                    'p75': np.percentile(values, 75) if HAS_ML_LIBS else sorted(values)[3*count//4],
                    'p90': np.percentile(values, 90) if HAS_ML_LIBS else sorted(values)[9*count//10],
                    'p95': np.percentile(values, 95) if HAS_ML_LIBS else sorted(values)[19*count//20],
                    'p99': np.percentile(values, 99) if HAS_ML_LIBS else sorted(values)[99*count//100]
                }
            
            return MetricAggregation(
                metric_id=metric_id,
                start_time=start_time,
                end_time=end_time,
                count=count,
                sum_value=sum_value,
                min_value=min_value,
                max_value=max_value,
                mean=mean,
                median=median,
                std_dev=std_dev,
                percentiles=percentiles
            )
        
        except Exception as e:
            logging.error(f"Erreur agrégation métriques: {e}")
            return None


class RedisMetricStorage(MetricStorage):
    """Stockage Redis pour métriques haute performance."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        if not HAS_REDIS:
            raise ImportError("Redis library not available")
        
        self.redis_client = redis.from_url(redis_url)
        self.key_prefix = "metrics:"
        
    async def store_metric(self, metric: MetricDataPoint) -> bool:
        """Stocke une métrique dans Redis."""
        try:
            # Stockage principal
            key = f"{self.key_prefix}{metric.metric_id}:{metric.timestamp.isoformat()}"
            data = metric.to_dict()
            
            pipeline = self.redis_client.pipeline()
            
            # Stockage de la métrique
            pipeline.setex(key, 86400 * 30, json.dumps(data))  # 30 jours
            
            # Index par catégorie
            category_key = f"{self.key_prefix}category:{metric.category.value}"
            pipeline.zadd(category_key, {key: metric.timestamp.timestamp()})
            
            # Index par métrique
            metric_key = f"{self.key_prefix}metric:{metric.metric_id}"
            pipeline.zadd(metric_key, {key: metric.timestamp.timestamp()})
            
            # Statistiques en temps réel
            if isinstance(metric.value, (int, float)):
                stats_key = f"{self.key_prefix}stats:{metric.metric_id}"
                pipeline.zadd(stats_key, {str(metric.value): metric.timestamp.timestamp()})
                pipeline.expire(stats_key, 86400 * 7)  # 7 jours pour les stats
            
            pipeline.execute()
            return True
            
        except Exception as e:
            logging.error(f"Erreur stockage Redis: {e}")
            return False
    
    async def get_metrics(self, query: Dict[str, Any]) -> List[MetricDataPoint]:
        """Récupère des métriques depuis Redis."""
        try:
            keys = []
            
            if 'metric_id' in query:
                # Récupération par métrique
                metric_key = f"{self.key_prefix}metric:{query['metric_id']}"
                start_score = query.get('start_time', datetime.min).timestamp()
                end_score = query.get('end_time', datetime.max).timestamp()
                
                keys = self.redis_client.zrangebyscore(
                    metric_key, start_score, end_score,
                    start=0, num=query.get('limit', 1000)
                )
            elif 'category' in query:
                # Récupération par catégorie
                category_key = f"{self.key_prefix}category:{query['category']}"
                start_score = query.get('start_time', datetime.min).timestamp()
                end_score = query.get('end_time', datetime.max).timestamp()
                
                keys = self.redis_client.zrangebyscore(
                    category_key, start_score, end_score,
                    start=0, num=query.get('limit', 1000)
                )
            
            metrics = []
            if keys:
                pipeline = self.redis_client.pipeline()
                for key in keys:
                    pipeline.get(key)
                
                results = pipeline.execute()
                
                for result in results:
                    if result:
                        data = json.loads(result)
                        metrics.append(MetricDataPoint.from_dict(data))
            
            return metrics
            
        except Exception as e:
            logging.error(f"Erreur récupération Redis: {e}")
            return []
    
    async def aggregate_metrics(self, 
                              metric_id: str, 
                              start_time: datetime, 
                              end_time: datetime) -> Optional[MetricAggregation]:
        """Agrégation rapide avec Redis."""
        try:
            stats_key = f"{self.key_prefix}stats:{metric_id}"
            start_score = start_time.timestamp()
            end_score = end_time.timestamp()
            
            # Récupération des valeurs dans l'intervalle
            values_with_scores = self.redis_client.zrangebyscore(
                stats_key, start_score, end_score, withscores=True
            )
            
            if not values_with_scores:
                return None
            
            values = [float(value) for value, score in values_with_scores]
            
            # Calculs statistiques
            count = len(values)
            sum_value = sum(values)
            min_value = min(values)
            max_value = max(values)
            mean = statistics.mean(values)
            median = statistics.median(values)
            std_dev = statistics.stdev(values) if count > 1 else 0.0
            
            return MetricAggregation(
                metric_id=metric_id,
                start_time=start_time,
                end_time=end_time,
                count=count,
                sum_value=sum_value,
                min_value=min_value,
                max_value=max_value,
                mean=mean,
                median=median,
                std_dev=std_dev
            )
            
        except Exception as e:
            logging.error(f"Erreur agrégation Redis: {e}")
            return None


class MetricProcessor:
    """Processeur de métriques avec transformation et enrichissement."""
    
    def __init__(self):
        self.processors: List[Callable] = []
        self.enrichers: List[Callable] = []
        
    def add_processor(self, processor: Callable):
        """Ajoute un processeur de métrique."""
        self.processors.append(processor)
    
    def add_enricher(self, enricher: Callable):
        """Ajoute un enrichisseur de métrique."""
        self.enrichers.append(enricher)
    
    async def process_metric(self, metric: MetricDataPoint) -> MetricDataPoint:
        """Traite une métrique avec tous les processeurs."""
        processed_metric = metric
        
        # Application des processeurs
        for processor in self.processors:
            try:
                processed_metric = await self._apply_processor(processor, processed_metric)
            except Exception as e:
                logging.error(f"Erreur processeur métrique: {e}")
        
        # Application des enrichisseurs
        for enricher in self.enrichers:
            try:
                processed_metric = await self._apply_enricher(enricher, processed_metric)
            except Exception as e:
                logging.error(f"Erreur enrichisseur métrique: {e}")
        
        return processed_metric
    
    async def _apply_processor(self, processor: Callable, metric: MetricDataPoint) -> MetricDataPoint:
        """Applique un processeur à une métrique."""
        if asyncio.iscoroutinefunction(processor):
            return await processor(metric)
        else:
            return processor(metric)
    
    async def _apply_enricher(self, enricher: Callable, metric: MetricDataPoint) -> MetricDataPoint:
        """Applique un enrichisseur à une métrique."""
        if asyncio.iscoroutinefunction(enricher):
            return await enricher(metric)
        else:
            return enricher(metric)


class AnomalyDetector:
    """Détecteur d'anomalies basé sur machine learning."""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.enabled = HAS_ML_LIBS
        
        if not self.enabled:
            logging.warning("ML libraries not available, anomaly detection disabled")
    
    async def train_model(self, metric_id: str, training_data: List[float]):
        """Entraîne un modèle de détection d'anomalies."""
        if not self.enabled or len(training_data) < 10:
            return False
        
        try:
            # Préparation des données
            X = np.array(training_data).reshape(-1, 1)
            
            # Normalisation
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Modèle Isolation Forest
            model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
            model.fit(X_scaled)
            
            # Stockage du modèle et du scaler
            self.models[metric_id] = model
            self.scalers[metric_id] = scaler
            
            logging.info(f"Modèle d'anomalie entraîné pour {metric_id}")
            return True
            
        except Exception as e:
            logging.error(f"Erreur entraînement modèle anomalie: {e}")
            return False
    
    async def detect_anomaly(self, metric_id: str, value: float) -> Tuple[bool, float]:
        """Détecte si une valeur est une anomalie."""
        if not self.enabled or metric_id not in self.models:
            return False, 0.0
        
        try:
            model = self.models[metric_id]
            scaler = self.scalers[metric_id]
            
            # Normalisation de la valeur
            X = np.array([[value]])
            X_scaled = scaler.transform(X)
            
            # Prédiction
            prediction = model.predict(X_scaled)[0]
            anomaly_score = model.decision_function(X_scaled)[0]
            
            is_anomaly = prediction == -1
            confidence = abs(anomaly_score)
            
            return is_anomaly, confidence
            
        except Exception as e:
            logging.error(f"Erreur détection anomalie: {e}")
            return False, 0.0
    
    async def batch_detect_anomalies(self, 
                                   metric_id: str, 
                                   values: List[float]) -> List[Tuple[bool, float]]:
        """Détection d'anomalies en lot."""
        if not self.enabled or metric_id not in self.models:
            return [(False, 0.0)] * len(values)
        
        try:
            model = self.models[metric_id]
            scaler = self.scalers[metric_id]
            
            # Normalisation
            X = np.array(values).reshape(-1, 1)
            X_scaled = scaler.transform(X)
            
            # Prédictions
            predictions = model.predict(X_scaled)
            anomaly_scores = model.decision_function(X_scaled)
            
            results = []
            for pred, score in zip(predictions, anomaly_scores):
                is_anomaly = pred == -1
                confidence = abs(score)
                results.append((is_anomaly, confidence))
            
            return results
            
        except Exception as e:
            logging.error(f"Erreur détection anomalies batch: {e}")
            return [(False, 0.0)] * len(values)


class MetricAlert:
    """Système d'alertes pour métriques."""
    
    def __init__(self):
        self.rules: Dict[str, Dict] = {}
        self.alert_handlers: List[Callable] = []
        
    def add_rule(self, 
                 rule_id: str,
                 metric_id: str,
                 condition: str,
                 threshold: Union[int, float],
                 severity: MetricSeverity = MetricSeverity.MEDIUM):
        """Ajoute une règle d'alerte."""
        self.rules[rule_id] = {
            'metric_id': metric_id,
            'condition': condition,  # 'gt', 'lt', 'eq', 'ne', 'between'
            'threshold': threshold,
            'severity': severity,
            'enabled': True
        }
    
    def add_alert_handler(self, handler: Callable):
        """Ajoute un gestionnaire d'alerte."""
        self.alert_handlers.append(handler)
    
    async def evaluate_metric(self, metric: MetricDataPoint) -> List[Dict]:
        """Évalue une métrique contre les règles d'alerte."""
        alerts = []
        
        for rule_id, rule in self.rules.items():
            if not rule['enabled'] or rule['metric_id'] != metric.metric_id:
                continue
            
            if not isinstance(metric.value, (int, float)):
                continue
            
            alert_triggered = False
            
            # Évaluation de la condition
            if rule['condition'] == 'gt' and metric.value > rule['threshold']:
                alert_triggered = True
            elif rule['condition'] == 'lt' and metric.value < rule['threshold']:
                alert_triggered = True
            elif rule['condition'] == 'eq' and metric.value == rule['threshold']:
                alert_triggered = True
            elif rule['condition'] == 'ne' and metric.value != rule['threshold']:
                alert_triggered = True
            elif rule['condition'] == 'between':
                if isinstance(rule['threshold'], (list, tuple)) and len(rule['threshold']) == 2:
                    min_val, max_val = rule['threshold']
                    if not (min_val <= metric.value <= max_val):
                        alert_triggered = True
            
            if alert_triggered:
                alert = {
                    'rule_id': rule_id,
                    'metric_id': metric.metric_id,
                    'value': metric.value,
                    'threshold': rule['threshold'],
                    'condition': rule['condition'],
                    'severity': rule['severity'],
                    'timestamp': metric.timestamp,
                    'tags': metric.tags,
                    'metadata': metric.metadata
                }
                alerts.append(alert)
                
                # Notification des gestionnaires
                for handler in self.alert_handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(alert)
                        else:
                            handler(alert)
                    except Exception as e:
                        logging.error(f"Erreur gestionnaire alerte: {e}")
        
        return alerts


class MetricCollector:
    """Collecteur de métriques avec sources multiples."""
    
    def __init__(self, storage: MetricStorage):
        self.storage = storage
        self.processor = MetricProcessor()
        self.anomaly_detector = AnomalyDetector()
        self.alert_system = MetricAlert()
        self.collection_interval = 60  # 1 minute par défaut
        self.running = False
        self._collection_tasks: Set[asyncio.Task] = set()
        
    async def collect_system_metrics(self) -> List[MetricDataPoint]:
        """Collecte les métriques système."""
        metrics = []
        timestamp = datetime.now()
        
        try:
            import psutil
            
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(MetricDataPoint(
                metric_id="system.cpu.usage",
                timestamp=timestamp,
                value=cpu_percent,
                metric_type=MetricType.GAUGE,
                category=MetricCategory.SYSTEM,
                severity=MetricSeverity.LOW if cpu_percent < 80 else MetricSeverity.HIGH,
                tags={"type": "cpu", "unit": "percent"}
            ))
            
            # Mémoire
            memory = psutil.virtual_memory()
            metrics.append(MetricDataPoint(
                metric_id="system.memory.usage",
                timestamp=timestamp,
                value=memory.percent,
                metric_type=MetricType.GAUGE,
                category=MetricCategory.SYSTEM,
                severity=MetricSeverity.LOW if memory.percent < 80 else MetricSeverity.HIGH,
                tags={"type": "memory", "unit": "percent"}
            ))
            
            # Disque
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            metrics.append(MetricDataPoint(
                metric_id="system.disk.usage",
                timestamp=timestamp,
                value=disk_percent,
                metric_type=MetricType.GAUGE,
                category=MetricCategory.STORAGE,
                severity=MetricSeverity.LOW if disk_percent < 80 else MetricSeverity.CRITICAL,
                tags={"type": "disk", "unit": "percent", "mount": "/"}
            ))
            
            # Réseau
            network = psutil.net_io_counters()
            metrics.append(MetricDataPoint(
                metric_id="system.network.bytes_sent",
                timestamp=timestamp,
                value=network.bytes_sent,
                metric_type=MetricType.CUMULATIVE,
                category=MetricCategory.NETWORK,
                tags={"type": "network", "direction": "out", "unit": "bytes"}
            ))
            
            metrics.append(MetricDataPoint(
                metric_id="system.network.bytes_recv",
                timestamp=timestamp,
                value=network.bytes_recv,
                metric_type=MetricType.CUMULATIVE,
                category=MetricCategory.NETWORK,
                tags={"type": "network", "direction": "in", "unit": "bytes"}
            ))
            
        except ImportError:
            logging.warning("psutil not available, skipping system metrics")
        except Exception as e:
            logging.error(f"Erreur collecte métriques système: {e}")
        
        return metrics
    
    async def collect_security_metrics(self) -> List[MetricDataPoint]:
        """Collecte les métriques de sécurité."""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # Simulation de métriques de sécurité
            # En production, ces métriques viendraient de vrais systèmes de sécurité
            
            # Tentatives d'authentification
            failed_auth_attempts = 0  # À remplacer par vraie collecte
            metrics.append(MetricDataPoint(
                metric_id="security.auth.failed_attempts",
                timestamp=timestamp,
                value=failed_auth_attempts,
                metric_type=MetricType.COUNTER,
                category=MetricCategory.SECURITY,
                severity=MetricSeverity.HIGH if failed_auth_attempts > 10 else MetricSeverity.LOW,
                tags={"type": "authentication", "status": "failed"}
            ))
            
            # Accès aux clés
            key_access_count = 0  # À remplacer par vraie collecte
            metrics.append(MetricDataPoint(
                metric_id="security.keys.access_count",
                timestamp=timestamp,
                value=key_access_count,
                metric_type=MetricType.COUNTER,
                category=MetricCategory.SECURITY,
                tags={"type": "key_access", "resource": "cryptographic_keys"}
            ))
            
            # Violations de sécurité détectées
            security_violations = 0  # À remplacer par vraie collecte
            metrics.append(MetricDataPoint(
                metric_id="security.violations.detected",
                timestamp=timestamp,
                value=security_violations,
                metric_type=MetricType.COUNTER,
                category=MetricCategory.SECURITY,
                severity=MetricSeverity.CRITICAL if security_violations > 0 else MetricSeverity.LOW,
                tags={"type": "security_violation", "source": "automated_detection"}
            ))
            
        except Exception as e:
            logging.error(f"Erreur collecte métriques sécurité: {e}")
        
        return metrics
    
    async def collect_performance_metrics(self) -> List[MetricDataPoint]:
        """Collecte les métriques de performance."""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # Métriques de performance applicative
            # En production, ces métriques viendraient de l'application réelle
            
            # Temps de réponse API
            api_response_time = 150.0  # ms - À remplacer par vraie mesure
            metrics.append(MetricDataPoint(
                metric_id="performance.api.response_time",
                timestamp=timestamp,
                value=api_response_time,
                metric_type=MetricType.TIMER,
                category=MetricCategory.PERFORMANCE,
                severity=MetricSeverity.HIGH if api_response_time > 1000 else MetricSeverity.LOW,
                tags={"type": "api", "unit": "milliseconds", "endpoint": "keys"}
            ))
            
            # Débit de requêtes
            request_rate = 25.5  # req/sec - À remplacer par vraie mesure
            metrics.append(MetricDataPoint(
                metric_id="performance.api.request_rate",
                timestamp=timestamp,
                value=request_rate,
                metric_type=MetricType.RATE,
                category=MetricCategory.PERFORMANCE,
                tags={"type": "api", "unit": "requests_per_second"}
            ))
            
            # Taux d'erreur
            error_rate = 2.1  # % - À remplacer par vraie mesure
            metrics.append(MetricDataPoint(
                metric_id="performance.api.error_rate",
                timestamp=timestamp,
                value=error_rate,
                metric_type=MetricType.RATIO,
                category=MetricCategory.RELIABILITY,
                severity=MetricSeverity.CRITICAL if error_rate > 5 else MetricSeverity.LOW,
                tags={"type": "api", "unit": "percent"}
            ))
            
        except Exception as e:
            logging.error(f"Erreur collecte métriques performance: {e}")
        
        return metrics
    
    async def start_collection(self):
        """Démarre la collecte automatique de métriques."""
        if self.running:
            return
        
        self.running = True
        
        # Tâches de collecte
        collection_tasks = [
            asyncio.create_task(self._collect_loop("system", self.collect_system_metrics)),
            asyncio.create_task(self._collect_loop("security", self.collect_security_metrics)),
            asyncio.create_task(self._collect_loop("performance", self.collect_performance_metrics))
        ]
        
        self._collection_tasks.update(collection_tasks)
        
        logging.info("Collecte de métriques démarrée")
    
    async def stop_collection(self):
        """Arrête la collecte automatique de métriques."""
        self.running = False
        
        # Annulation des tâches
        for task in self._collection_tasks:
            task.cancel()
        
        # Attendre l'arrêt des tâches
        await asyncio.gather(*self._collection_tasks, return_exceptions=True)
        
        self._collection_tasks.clear()
        
        logging.info("Collecte de métriques arrêtée")
    
    async def _collect_loop(self, source: str, collector_func: Callable):
        """Boucle de collecte pour une source de métriques."""
        while self.running:
            try:
                metrics = await collector_func()
                
                for metric in metrics:
                    metric.source = source
                    
                    # Traitement de la métrique
                    processed_metric = await self.processor.process_metric(metric)
                    
                    # Détection d'anomalies
                    if isinstance(processed_metric.value, (int, float)):
                        is_anomaly, confidence = await self.anomaly_detector.detect_anomaly(
                            processed_metric.metric_id, processed_metric.value
                        )
                        
                        if is_anomaly:
                            processed_metric.metadata['anomaly_detected'] = True
                            processed_metric.metadata['anomaly_confidence'] = confidence
                            processed_metric.severity = MetricSeverity.HIGH
                    
                    # Stockage
                    await self.storage.store_metric(processed_metric)
                    
                    # Évaluation des alertes
                    alerts = await self.alert_system.evaluate_metric(processed_metric)
                    if alerts:
                        logging.warning(f"Alertes déclenchées pour {processed_metric.metric_id}: {len(alerts)}")
                
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Erreur dans boucle collecte {source}: {e}")
                await asyncio.sleep(5)  # Pause avant retry


class MetricReporter:
    """Générateur de rapports de métriques."""
    
    def __init__(self, storage: MetricStorage):
        self.storage = storage
    
    async def generate_summary_report(self, 
                                    start_time: datetime, 
                                    end_time: datetime,
                                    categories: Optional[List[MetricCategory]] = None) -> Dict[str, Any]:
        """Génère un rapport de synthèse."""
        report = {
            'period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'duration_hours': (end_time - start_time).total_seconds() / 3600
            },
            'categories': {},
            'top_metrics': [],
            'anomalies': [],
            'alerts': []
        }
        
        # Collecte des métriques par catégorie
        categories_to_check = categories or list(MetricCategory)
        
        for category in categories_to_check:
            metrics = await self.storage.get_metrics({
                'category': category.value,
                'start_time': start_time,
                'end_time': end_time,
                'limit': 10000
            })
            
            if metrics:
                # Statistiques de base
                total_metrics = len(metrics)
                unique_metric_ids = len(set(m.metric_id for m in metrics))
                
                # Métriques par sévérité
                severity_counts = defaultdict(int)
                for metric in metrics:
                    severity_counts[metric.severity.value] += 1
                
                # Anomalies détectées
                anomalies = [m for m in metrics if m.metadata.get('anomaly_detected', False)]
                
                report['categories'][category.value] = {
                    'total_metrics': total_metrics,
                    'unique_metrics': unique_metric_ids,
                    'severity_distribution': dict(severity_counts),
                    'anomalies_detected': len(anomalies)
                }
                
                # Ajout des anomalies au rapport global
                for anomaly in anomalies:
                    report['anomalies'].append({
                        'metric_id': anomaly.metric_id,
                        'value': anomaly.value,
                        'timestamp': anomaly.timestamp.isoformat(),
                        'confidence': anomaly.metadata.get('anomaly_confidence', 0.0),
                        'category': anomaly.category.value
                    })
        
        return report
    
    async def generate_performance_report(self, 
                                        metric_ids: List[str],
                                        start_time: datetime, 
                                        end_time: datetime) -> Dict[str, Any]:
        """Génère un rapport de performance détaillé."""
        report = {
            'period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'metrics': {}
        }
        
        for metric_id in metric_ids:
            # Agrégation des métriques
            aggregation = await self.storage.aggregate_metrics(metric_id, start_time, end_time)
            
            if aggregation:
                # Récupération des métriques brutes pour analyse de tendance
                raw_metrics = await self.storage.get_metrics({
                    'metric_id': metric_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'limit': 1000
                })
                
                # Analyse de tendance
                trend_analysis = self._analyze_trend(raw_metrics)
                
                report['metrics'][metric_id] = {
                    'aggregation': asdict(aggregation),
                    'trend': trend_analysis,
                    'data_points': len(raw_metrics)
                }
        
        return report
    
    def _analyze_trend(self, metrics: List[MetricDataPoint]) -> Dict[str, Any]:
        """Analyse la tendance d'une série de métriques."""
        if len(metrics) < 2:
            return {'direction': 'insufficient_data', 'strength': 0.0}
        
        # Tri par timestamp
        sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)
        
        # Extraction des valeurs numériques
        values = []
        for metric in sorted_metrics:
            if isinstance(metric.value, (int, float)):
                values.append(float(metric.value))
        
        if len(values) < 2:
            return {'direction': 'insufficient_numeric_data', 'strength': 0.0}
        
        # Calcul de la tendance simple (régression linéaire basique)
        n = len(values)
        x = list(range(n))
        
        # Moyennes
        mean_x = sum(x) / n
        mean_y = sum(values) / n
        
        # Coefficient de corrélation
        numerator = sum((x[i] - mean_x) * (values[i] - mean_y) for i in range(n))
        denominator_x = sum((x[i] - mean_x) ** 2 for i in range(n))
        denominator_y = sum((values[i] - mean_y) ** 2 for i in range(n))
        
        if denominator_x == 0 or denominator_y == 0:
            return {'direction': 'stable', 'strength': 0.0}
        
        correlation = numerator / (denominator_x * denominator_y) ** 0.5
        
        # Pente
        slope = numerator / denominator_x if denominator_x != 0 else 0
        
        # Direction de la tendance
        if abs(correlation) < 0.1:
            direction = 'stable'
        elif slope > 0:
            direction = 'increasing'
        else:
            direction = 'decreasing'
        
        return {
            'direction': direction,
            'strength': abs(correlation),
            'slope': slope,
            'correlation': correlation
        }


class EnterpriseMetricsSystem:
    """Système de métriques d'entreprise complet."""
    
    def __init__(self, 
                 storage_type: str = "sqlite",
                 storage_config: Optional[Dict[str, Any]] = None):
        
        # Configuration du stockage
        if storage_type == "sqlite":
            db_path = storage_config.get('db_path', 'metrics.db') if storage_config else 'metrics.db'
            self.storage = SQLiteMetricStorage(db_path)
        elif storage_type == "redis":
            redis_url = storage_config.get('redis_url', 'redis://localhost:6379/0') if storage_config else 'redis://localhost:6379/0'
            self.storage = RedisMetricStorage(redis_url)
        else:
            raise ValueError(f"Type de stockage non supporté: {storage_type}")
        
        # Composants du système
        self.collector = MetricCollector(self.storage)
        self.reporter = MetricReporter(self.storage)
        self.processor = self.collector.processor
        self.anomaly_detector = self.collector.anomaly_detector
        self.alert_system = self.collector.alert_system
        
        # Configuration par défaut
        self._setup_default_processors()
        self._setup_default_alerts()
        
        logging.info(f"Système de métriques d'entreprise initialisé avec stockage {storage_type}")
    
    def _setup_default_processors(self):
        """Configure les processeurs par défaut."""
        
        async def add_correlation_id(metric: MetricDataPoint) -> MetricDataPoint:
            """Ajoute un ID de corrélation si manquant."""
            if not metric.correlation_id:
                metric.correlation_id = str(uuid.uuid4())
            return metric
        
        async def enrich_with_hostname(metric: MetricDataPoint) -> MetricDataPoint:
            """Enrichit avec le nom d'hôte."""
            import socket
            metric.metadata['hostname'] = socket.gethostname()
            return metric
        
        async def add_processing_timestamp(metric: MetricDataPoint) -> MetricDataPoint:
            """Ajoute l'horodatage de traitement."""
            metric.metadata['processed_at'] = datetime.now().isoformat()
            return metric
        
        self.processor.add_processor(add_correlation_id)
        self.processor.add_enricher(enrich_with_hostname)
        self.processor.add_enricher(add_processing_timestamp)
    
    def _setup_default_alerts(self):
        """Configure les alertes par défaut."""
        
        # Alertes système
        self.alert_system.add_rule(
            "high_cpu_usage",
            "system.cpu.usage",
            "gt",
            85.0,
            MetricSeverity.HIGH
        )
        
        self.alert_system.add_rule(
            "high_memory_usage",
            "system.memory.usage",
            "gt",
            90.0,
            MetricSeverity.CRITICAL
        )
        
        self.alert_system.add_rule(
            "high_disk_usage",
            "system.disk.usage",
            "gt",
            90.0,
            MetricSeverity.CRITICAL
        )
        
        # Alertes sécurité
        self.alert_system.add_rule(
            "security_violations",
            "security.violations.detected",
            "gt",
            0,
            MetricSeverity.CRITICAL
        )
        
        self.alert_system.add_rule(
            "high_failed_auth",
            "security.auth.failed_attempts",
            "gt",
            10,
            MetricSeverity.HIGH
        )
        
        # Alertes performance
        self.alert_system.add_rule(
            "slow_api_response",
            "performance.api.response_time",
            "gt",
            1000.0,
            MetricSeverity.HIGH
        )
        
        self.alert_system.add_rule(
            "high_error_rate",
            "performance.api.error_rate",
            "gt",
            5.0,
            MetricSeverity.CRITICAL
        )
        
        # Gestionnaire d'alerte par défaut
        async def default_alert_handler(alert: Dict[str, Any]):
            """Gestionnaire d'alerte par défaut."""
            severity = alert['severity'].value if hasattr(alert['severity'], 'value') else alert['severity']
            logging.warning(
                f"ALERTE [{severity.upper()}] {alert['rule_id']}: "
                f"{alert['metric_id']} = {alert['value']} "
                f"(seuil: {alert['threshold']})"
            )
        
        self.alert_system.add_alert_handler(default_alert_handler)
    
    async def start(self):
        """Démarre le système de métriques."""
        await self.collector.start_collection()
        logging.info("Système de métriques démarré")
    
    async def stop(self):
        """Arrête le système de métriques."""
        await self.collector.stop_collection()
        logging.info("Système de métriques arrêté")
    
    async def record_metric(self, 
                          metric_id: str,
                          value: Union[int, float, str, bool, Dict, List],
                          metric_type: MetricType = MetricType.GAUGE,
                          category: MetricCategory = MetricCategory.CUSTOM,
                          severity: MetricSeverity = MetricSeverity.LOW,
                          tags: Optional[Dict[str, str]] = None,
                          metadata: Optional[Dict[str, Any]] = None):
        """Enregistre une métrique personnalisée."""
        
        metric = MetricDataPoint(
            metric_id=metric_id,
            timestamp=datetime.now(),
            value=value,
            metric_type=metric_type,
            category=category,
            severity=severity,
            tags=tags or {},
            metadata=metadata or {},
            source="manual"
        )
        
        # Traitement et stockage
        processed_metric = await self.processor.process_metric(metric)
        await self.storage.store_metric(processed_metric)
        
        # Évaluation des alertes
        alerts = await self.alert_system.evaluate_metric(processed_metric)
        
        return {
            'metric_stored': True,
            'alerts_triggered': len(alerts),
            'metric_id': metric_id
        }
    
    async def get_metrics_summary(self, 
                                hours_back: int = 24) -> Dict[str, Any]:
        """Obtient un résumé des métriques."""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)
        
        return await self.reporter.generate_summary_report(start_time, end_time)
    
    async def get_performance_report(self, 
                                   metric_ids: List[str],
                                   hours_back: int = 24) -> Dict[str, Any]:
        """Obtient un rapport de performance."""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)
        
        return await self.reporter.generate_performance_report(metric_ids, start_time, end_time)
    
    async def train_anomaly_models(self, 
                                 metric_ids: Optional[List[str]] = None,
                                 days_back: int = 30):
        """Entraîne les modèles de détection d'anomalies."""
        if not HAS_ML_LIBS:
            logging.warning("ML libraries not available, skipping anomaly model training")
            return
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        # Si pas de métriques spécifiées, utilise les métriques système courantes
        if not metric_ids:
            metric_ids = [
                "system.cpu.usage",
                "system.memory.usage",
                "system.disk.usage",
                "performance.api.response_time",
                "performance.api.request_rate"
            ]
        
        for metric_id in metric_ids:
            try:
                # Récupération des données d'entraînement
                metrics = await self.storage.get_metrics({
                    'metric_id': metric_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'limit': 10000
                })
                
                # Extraction des valeurs numériques
                values = []
                for metric in metrics:
                    if isinstance(metric.value, (int, float)):
                        values.append(float(metric.value))
                
                if len(values) >= 100:  # Minimum de données pour entraînement
                    success = await self.anomaly_detector.train_model(metric_id, values)
                    if success:
                        logging.info(f"Modèle d'anomalie entraîné pour {metric_id} avec {len(values)} points")
                    else:
                        logging.warning(f"Échec entraînement modèle pour {metric_id}")
                else:
                    logging.warning(f"Pas assez de données pour entraîner le modèle {metric_id} ({len(values)} points)")
                    
            except Exception as e:
                logging.error(f"Erreur entraînement modèle anomalie pour {metric_id}: {e}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Retourne les informations du système de métriques."""
        return {
            'storage_type': type(self.storage).__name__,
            'ml_libraries_available': HAS_ML_LIBS,
            'redis_available': HAS_REDIS,
            'postgresql_available': HAS_POSTGRESQL,
            'collection_running': self.collector.running,
            'collection_interval': self.collector.collection_interval,
            'alert_rules_count': len(self.alert_system.rules),
            'processors_count': len(self.processor.processors),
            'enrichers_count': len(self.processor.enrichers)
        }


# Instance globale pour faciliter l'utilisation
_global_metrics_system: Optional[EnterpriseMetricsSystem] = None


def get_metrics_system(storage_type: str = "sqlite",
                      storage_config: Optional[Dict[str, Any]] = None) -> EnterpriseMetricsSystem:
    """Obtient l'instance globale du système de métriques."""
    global _global_metrics_system
    
    if _global_metrics_system is None:
        _global_metrics_system = EnterpriseMetricsSystem(storage_type, storage_config)
    
    return _global_metrics_system


# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Export des classes principales
__all__ = [
    'MetricType',
    'MetricSeverity', 
    'MetricCategory',
    'MetricDataPoint',
    'MetricAggregation',
    'MetricStorage',
    'SQLiteMetricStorage',
    'RedisMetricStorage',
    'MetricProcessor',
    'AnomalyDetector',
    'MetricAlert',
    'MetricCollector',
    'MetricReporter',
    'EnterpriseMetricsSystem',
    'get_metrics_system'
]


# Exemple d'utilisation si exécuté directement
if __name__ == "__main__":
    async def main_example():
        """Exemple d'utilisation du système de métriques."""
        
        # Initialisation du système
        metrics_system = get_metrics_system("sqlite", {"db_path": "example_metrics.db"})
        
        # Démarrage de la collecte
        await metrics_system.start()
        
        # Enregistrement de métriques personnalisées
        await metrics_system.record_metric(
            "custom.business.revenue",
            1250.75,
            MetricType.GAUGE,
            MetricCategory.BUSINESS,
            MetricSeverity.LOW,
            tags={"currency": "EUR", "department": "sales"}
        )
        
        # Attente pour collecte
        await asyncio.sleep(10)
        
        # Génération de rapport
        summary = await metrics_system.get_metrics_summary(hours_back=1)
        print("Résumé des métriques:")
        print(json.dumps(summary, indent=2, default=str))
        
        # Entraînement des modèles d'anomalie
        await metrics_system.train_anomaly_models(days_back=1)
        
        # Arrêt du système
        await metrics_system.stop()
        
        print("Exemple terminé avec succès")
    
    # Exécution de l'exemple
    asyncio.run(main_example())
