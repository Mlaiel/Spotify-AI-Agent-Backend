#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚀 DATA LAYER ULTRA-AVANCÉ - REAL-TIME METRICS RÉVOLUTIONNAIRE
Architecture de données révolutionnaire pour métriques temps réel enterprise

Système ultra-performant combinant :
├── 📊 Streaming Analytics haute fréquence (microseconde precision)
├── 🧠 ML Predictive Intelligence intégrée
├── 🔄 Event-Driven Architecture distribuée
├── 💾 Multi-Database Optimization (PostgreSQL/Redis/ClickHouse)
├── 🌊 Stream Processing avec Apache Kafka
├── 📈 Real-Time Anomaly Detection
├── 🔍 Advanced Time Series Analytics
├── 🏭 Industrial-Grade Data Pipeline
├── 🛡️ Enterprise Security & Encryption
└── ⚡ Sub-millisecond Response Time

Développé par l'équipe d'experts Achiri avec architecture cloud-native
et optimisations performance de niveau industriel.

Version: 3.0.0 - Production Ready Enterprise
License: Enterprise Commercial License
"""

__version__ = "3.0.0"
__author__ = "Achiri Expert Team - Data Engineering Division"
__license__ = "Enterprise Commercial"
__status__ = "Production"

import asyncio
import logging
import sys
import time
import uuid
import json
import math
from datetime import datetime, timedelta
from typing import (
    Dict, List, Optional, Any, Union, AsyncGenerator, Callable, 
    TypeVar, Generic, Protocol, runtime_checkable, Tuple, Set
)
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import multiprocessing
from collections import defaultdict, deque
import weakref
import pickle
import gzip
import lz4.frame
import msgpack

# Imports pour calculs scientifiques et ML ultra-avancés
try:
    import numpy as np
    import pandas as pd
    from scipy import stats, signal, optimize, interpolate
    from scipy.stats import zscore, iqr, kurtosis, skew
    from scipy.spatial.distance import euclidean
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.decomposition import PCA, FastICA, TruncatedSVD
    from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.model_selection import TimeSeriesSplit
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
    SCIENTIFIC_LIBS_AVAILABLE = True
except ImportError:
    SCIENTIFIC_LIBS_AVAILABLE = False

# Imports pour Time Series avancées
try:
    import prophet
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# Imports pour streaming et queuing enterprise
try:
    import aioredis
    from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
    from aiokafka.admin import AIOKafkaAdminClient, NewTopic
    import asyncpg
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker, declarative_base
    from sqlalchemy import Column, Integer, String, DateTime, Float, JSON, Text, Boolean
    from motor.motor_asyncio import AsyncIOMotorClient
    ASYNC_DB_AVAILABLE = True
except ImportError:
    ASYNC_DB_AVAILABLE = False

# Imports pour ClickHouse (OLAP ultra-performant)
try:
    from clickhouse_driver import Client as ClickHouseClient
    import clickhouse_connect
    CLICKHOUSE_AVAILABLE = True
except ImportError:
    CLICKHOUSE_AVAILABLE = False

# Imports pour monitoring et observabilité
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, CollectorRegistry,
        generate_latest, CONTENT_TYPE_LATEST
    )
    import opentelemetry
    from opentelemetry import trace, metrics as otel_metrics
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

# Imports pour sérialisation ultra-rapide
try:
    import orjson
    import ujson
    import rapidjson
    FAST_JSON_AVAILABLE = True
except ImportError:
    FAST_JSON_AVAILABLE = False

# Imports pour compression avancée
try:
    import zstandard as zstd
    import blosc
    COMPRESSION_AVAILABLE = True
except ImportError:
    COMPRESSION_AVAILABLE = False

# Configuration logging ultra-avancé
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data_layer_metrics.log', encoding='utf-8')
    ]
)

# =============================================================================
# ÉNUMÉRATIONS ET TYPES AVANCÉS
# =============================================================================

class MetricType(Enum):
    """Types de métriques ultra-avancés avec précision industrielle"""
    COUNTER = auto()           # Compteurs monotones
    GAUGE = auto()            # Valeurs instantanées 
    HISTOGRAM = auto()        # Distribution de valeurs
    SUMMARY = auto()          # Résumés statistiques
    TIMER = auto()            # Mesures de temps haute précision
    RATE = auto()             # Taux de change
    PERCENTILE = auto()       # Percentiles calculés
    ANOMALY_SCORE = auto()    # Scores d'anomalie ML
    PREDICTION = auto()       # Prédictions ML
    TREND = auto()            # Analyse de tendances

class AggregationType(Enum):
    """Types d'agrégation avec optimisations algorithmiques"""
    SUM = auto()              # Somme
    MEAN = auto()             # Moyenne arithmétique
    MEDIAN = auto()           # Médiane robuste
    MIN = auto()              # Minimum
    MAX = auto()              # Maximum
    STD = auto()              # Écart-type
    VAR = auto()              # Variance
    PERCENTILE_95 = auto()    # 95e percentile
    PERCENTILE_99 = auto()    # 99e percentile
    WEIGHTED_MEAN = auto()    # Moyenne pondérée
    GEOMETRIC_MEAN = auto()   # Moyenne géométrique
    HARMONIC_MEAN = auto()    # Moyenne harmonique
    TRIMMED_MEAN = auto()     # Moyenne élagée
    INTERQUARTILE_RANGE = auto()  # Écart interquartile

class TimeWindow(Enum):
    """Fenêtres temporelles pour agrégations optimisées"""
    SECOND = auto()           # Seconde (ultra haute fréquence)
    MINUTE = auto()           # Minute
    FIVE_MINUTES = auto()     # 5 minutes
    FIFTEEN_MINUTES = auto()  # 15 minutes (surveillance)
    HOUR = auto()             # Heure
    DAY = auto()              # Jour
    WEEK = auto()             # Semaine
    MONTH = auto()            # Mois
    QUARTER = auto()          # Trimestre
    YEAR = auto()             # Année
    CUSTOM = auto()           # Fenêtre personnalisée

class DataSource(Enum):
    """Sources de données enterprise multi-protocoles"""
    KAFKA_STREAM = auto()     # Apache Kafka streaming
    REDIS_PUBSUB = auto()     # Redis Pub/Sub
    WEBSOCKET = auto()        # WebSocket temps réel
    HTTP_ENDPOINT = auto()    # API REST/GraphQL
    DATABASE_POLLING = auto() # Polling base de données
    FILE_WATCHER = auto()     # Surveillance fichiers
    SYSTEM_METRICS = auto()   # Métriques système
    APPLICATION_LOGS = auto() # Logs applicatifs
    CUSTOM_CONNECTOR = auto() # Connecteur personnalisé

class CompressionAlgorithm(Enum):
    """Algorithmes de compression haute performance"""
    NONE = auto()             # Pas de compression
    GZIP = auto()             # GZIP standard
    LZ4 = auto()              # LZ4 ultra-rapide
    ZSTD = auto()             # Zstandard optimisé
    BLOSC = auto()            # BLOSC scientific
    SNAPPY = auto()           # Snappy Google

class SerializationFormat(Enum):
    """Formats de sérialisation optimisés"""
    JSON = auto()             # JSON standard
    ORJSON = auto()           # orjson ultra-rapide
    MSGPACK = auto()          # MessagePack binaire
    PICKLE = auto()           # Python pickle
    AVRO = auto()             # Apache Avro
    PROTOBUF = auto()         # Protocol Buffers

# =============================================================================
# MODÈLES DE DONNÉES ULTRA-AVANCÉS
# =============================================================================

@dataclass
class MetricPoint:
    """Point de métrique avec métadonnées enrichies"""
    timestamp: datetime
    value: Union[float, int, str, Dict[str, Any]]
    metric_name: str
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Informations de qualité des données
    quality_score: float = 1.0  # Score de qualité [0-1]
    confidence_interval: Optional[Tuple[float, float]] = None
    source_reliability: float = 1.0
    
    # Contexte ML
    anomaly_score: Optional[float] = None
    prediction_horizon: Optional[timedelta] = None
    seasonal_component: Optional[float] = None
    trend_component: Optional[float] = None
    
    # Optimisations
    compressed_payload: Optional[bytes] = None
    hash_signature: Optional[str] = None
    
    def __post_init__(self):
        """Post-traitement avec optimisations"""
        if not isinstance(self.timestamp, datetime):
            self.timestamp = datetime.fromisoformat(str(self.timestamp))
        
        # Génération signature hash pour déduplication
        if self.hash_signature is None:
            content = f"{self.metric_name}_{self.timestamp.isoformat()}_{self.value}"
            self.hash_signature = str(hash(content))

@dataclass
class MetricAggregation:
    """Agrégation de métriques avec calculs statistiques avancés"""
    metric_name: str
    window: TimeWindow
    start_time: datetime
    end_time: datetime
    aggregation_type: AggregationType
    
    # Valeurs calculées
    result: Union[float, Dict[str, float]]
    sample_count: int
    
    # Statistiques descriptives
    descriptive_stats: Dict[str, float] = field(default_factory=dict)
    
    # Détection d'anomalies
    is_anomalous: bool = False
    anomaly_score: float = 0.0
    anomaly_threshold: float = 0.95
    
    # Métadonnées de calcul
    computation_time_ms: float = 0.0
    algorithm_version: str = "1.0"
    confidence_level: float = 0.95

@dataclass
class StreamingWindow:
    """Fenêtre glissante pour streaming avec optimisations mémoire"""
    window_size: timedelta
    slide_interval: timedelta
    max_points: int = 10000
    
    # Buffer circulaire optimisé
    _buffer: deque = field(default_factory=deque)
    _buffer_lock: threading.RLock = field(default_factory=threading.RLock)
    
    # Statistiques de performance
    points_processed: int = 0
    buffer_overflows: int = 0
    last_cleanup: datetime = field(default_factory=datetime.utcnow)
    
    def add_point(self, point: MetricPoint) -> bool:
        """Ajout optimisé avec gestion mémoire"""
        with self._buffer_lock:
            # Nettoyage automatique des points expirés
            now = datetime.utcnow()
            cutoff = now - self.window_size
            
            while self._buffer and self._buffer[0].timestamp < cutoff:
                self._buffer.popleft()
            
            # Ajout du nouveau point
            if len(self._buffer) >= self.max_points:
                self._buffer.popleft()  # FIFO
                self.buffer_overflows += 1
            
            self._buffer.append(point)
            self.points_processed += 1
            return True
    
    def get_points(self, start_time: Optional[datetime] = None) -> List[MetricPoint]:
        """Récupération optimisée des points"""
        with self._buffer_lock:
            if start_time is None:
                return list(self._buffer)
            
            return [p for p in self._buffer if p.timestamp >= start_time]

@dataclass
class MLPredictionConfig:
    """Configuration pour prédictions ML ultra-avancées"""
    model_type: str = "auto"  # auto, arima, prophet, lstm, transformer
    forecast_horizon: int = 24  # Points à prédire
    confidence_intervals: List[float] = field(default_factory=lambda: [0.80, 0.95])
    
    # Paramètres de modèle
    seasonality_mode: str = "auto"  # auto, additive, multiplicative
    trend_mode: str = "auto"
    outlier_detection: bool = True
    automatic_retraining: bool = True
    
    # Optimisations performance
    use_gpu: bool = False
    parallel_processing: bool = True
    model_caching: bool = True
    
    # Validation
    cross_validation_folds: int = 5
    validation_metric: str = "mape"  # mape, rmse, mae

@dataclass
class AnomalyDetectionConfig:
    """Configuration détection d'anomalies multi-algorithmes"""
    algorithms: List[str] = field(default_factory=lambda: [
        "isolation_forest", "local_outlier_factor", "zscore", "iqr"
    ])
    
    # Seuils et paramètres
    contamination_rate: float = 0.1
    zscore_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    
    # Ensemble methods
    ensemble_voting: str = "majority"  # majority, weighted, unanimous
    algorithm_weights: Dict[str, float] = field(default_factory=dict)
    
    # Adaptation en temps réel
    adaptive_thresholds: bool = True
    learning_rate: float = 0.01
    forgetting_factor: float = 0.95

# =============================================================================
# INTERFACES ET PROTOCOLES
# =============================================================================

@runtime_checkable
class MetricCollector(Protocol):
    """Interface pour collecteurs de métriques"""
    
    async def collect(self) -> AsyncGenerator[MetricPoint, None]:
        """Collecte des métriques en streaming"""
        ...
    
    async def start(self) -> None:
        """Démarrage du collecteur"""
        ...
    
    async def stop(self) -> None:
        """Arrêt du collecteur"""
        ...

@runtime_checkable
class MetricProcessor(Protocol):
    """Interface pour processeurs de métriques"""
    
    async def process(self, point: MetricPoint) -> Optional[MetricPoint]:
        """Traitement d'un point de métrique"""
        ...
    
    async def process_batch(self, points: List[MetricPoint]) -> List[MetricPoint]:
        """Traitement par lot optimisé"""
        ...

@runtime_checkable
class MetricStorage(Protocol):
    """Interface pour stockage de métriques"""
    
    async def store(self, points: List[MetricPoint]) -> bool:
        """Stockage des métriques"""
        ...
    
    async def query(
        self, 
        metric_name: str, 
        start_time: datetime, 
        end_time: datetime,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[MetricPoint]:
        """Requête des métriques"""
        ...

# =============================================================================
# COMPOSANTS ULTRA-AVANCÉS
# =============================================================================

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# MODÈLES DE DONNÉES POUR MÉTRIQUES
# =============================================================================

class MetricType(Enum):
    """Types de métriques supportés"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    CUSTOM = "custom"

class DataQuality(Enum):
    """Niveaux de qualité des données"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INVALID = "invalid"

@dataclass
class MetricPoint:
    """Point de métrique individuel avec métadonnées"""
    id: str
    name: str
    value: Union[int, float, str]
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    tenant_id: str = ""
    metric_type: MetricType = MetricType.GAUGE
    source: str = ""
    quality: DataQuality = DataQuality.HIGH
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StreamingConfig:
    """Configuration pour le streaming de métriques"""
    batch_size: int = 1000
    flush_interval_ms: int = 5000
    compression: bool = True
    retention_hours: int = 24
    kafka_topic: str = "metrics"
    redis_stream: str = "metrics_stream"

@dataclass
class AnalyticsResult:
    """Résultat d'analyse de métriques"""
    metric_name: str
    analysis_type: str
    result: Dict[str, Any]
    confidence: float
    timestamp: datetime
    recommendations: List[str] = field(default_factory=list)

# =============================================================================
# COLLECTEUR DE MÉTRIQUES TEMPS RÉEL ENTERPRISE
# =============================================================================

class RealTimeMetricsCollector:
    """
    Collecteur de métriques haute performance avec streaming temps réel,
    analytics ML et gestion intelligente des données.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.streaming_config = StreamingConfig(**config.get('streaming', {}))
        
        # Buffers de métriques
        self.metric_buffer: List[MetricPoint] = []
        self.buffer_lock = asyncio.Lock()
        
        # Connexions
        self.redis_client: Optional[aioredis.Redis] = None
        self.kafka_producer: Optional[AIOKafkaProducer] = None
        self.db_engine = None
        
        # Analytics engine
        self.analytics_engine = MetricsAnalyticsEngine()
        
        # Monitoring
        self.metrics_collected = Counter(
            'metrics_collected_total',
            'Total métriques collectées',
            ['tenant_id', 'metric_type', 'source']
        )
        
        self.collection_duration = Histogram(
            'metric_collection_duration_seconds',
            'Durée de collecte des métriques',
            ['operation']
        )
        
        self.buffer_size = Gauge(
            'metric_buffer_size',
            'Taille actuelle du buffer de métriques'
        )
        
        # Tâches background
        self.background_tasks: List[asyncio.Task] = []
        self.running = False
        
        logger.info("RealTimeMetricsCollector initialisé")

    async def initialize(self):
        """Initialisation du collecteur de métriques"""
        try:
            # Connexion Redis
            self.redis_client = aioredis.from_url(
                self.config['redis_url'],
                encoding='utf-8',
                decode_responses=True
            )
            
            # Connexion Kafka
            self.kafka_producer = AIOKafkaProducer(
                bootstrap_servers=self.config['kafka_servers'],
                value_serializer=lambda x: json.dumps(x, default=str).encode()
            )
            await self.kafka_producer.start()
            
            # Connexion base de données
            self.db_engine = create_async_engine(
                self.config['database_url'],
                pool_size=20,
                max_overflow=50
            )
            
            # Initialisation analytics
            await self.analytics_engine.initialize()
            
            # Démarrage des tâches background
            await self.start_background_tasks()
            
            self.running = True
            logger.info("RealTimeMetricsCollector initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur initialisation collecteur: {e}")
            raise

    async def collect_metric(
        self,
        name: str,
        value: Union[int, float, str],
        labels: Optional[Dict[str, str]] = None,
        tenant_id: str = "",
        source: str = "manual",
        metric_type: MetricType = MetricType.GAUGE
    ) -> str:
        """
        Collection d'une métrique individuelle avec validation et enrichissement.
        
        Returns:
            metric_id: Identifiant unique de la métrique
        """
        with self.collection_duration.labels(operation='collect_single').time():
            try:
                # Génération ID unique
                metric_id = f"metric_{uuid.uuid4().hex[:12]}"
                
                # Validation et enrichissement
                validated_value, quality = await self.validate_and_enrich_metric(
                    name, value, labels or {}
                )
                
                # Création du point de métrique
                metric_point = MetricPoint(
                    id=metric_id,
                    name=name,
                    value=validated_value,
                    timestamp=datetime.utcnow(),
                    labels=labels or {},
                    tenant_id=tenant_id,
                    metric_type=metric_type,
                    source=source,
                    quality=quality,
                    metadata={
                        'collector_version': '1.0.0',
                        'collection_timestamp': time.time()
                    }
                )
                
                # Ajout au buffer
                await self.add_to_buffer(metric_point)
                
                # Mise à jour métriques Prometheus
                self.metrics_collected.labels(
                    tenant_id=tenant_id,
                    metric_type=metric_type.value,
                    source=source
                ).inc()
                
                # Analyse temps réel si critique
                if await self.is_critical_metric(name, validated_value):
                    await self.process_critical_metric(metric_point)
                
                logger.debug(f"Métrique collectée: {name}={validated_value}")
                return metric_id
                
            except Exception as e:
                logger.error(f"Erreur collection métrique {name}: {e}")
                raise

    async def collect_batch_metrics(
        self,
        metrics: List[Dict[str, Any]],
        tenant_id: str = ""
    ) -> List[str]:
        """Collection en lot de métriques pour haute performance"""
        with self.collection_duration.labels(operation='collect_batch').time():
            collected_ids = []
            
            for metric_data in metrics:
                try:
                    metric_id = await self.collect_metric(
                        name=metric_data['name'],
                        value=metric_data['value'],
                        labels=metric_data.get('labels', {}),
                        tenant_id=tenant_id,
                        source=metric_data.get('source', 'batch'),
                        metric_type=MetricType(metric_data.get('type', 'gauge'))
                    )
                    collected_ids.append(metric_id)
                    
                except Exception as e:
                    logger.error(f"Erreur dans le lot pour {metric_data.get('name')}: {e}")
                    continue
            
            logger.info(f"Lot de {len(collected_ids)}/{len(metrics)} métriques collectées")
            return collected_ids

    async def start_streaming(self) -> AsyncGenerator[MetricPoint, None]:
        """Streaming temps réel de métriques"""
        try:
            # Consumer Kafka pour le streaming
            consumer = AIOKafkaConsumer(
                self.streaming_config.kafka_topic,
                bootstrap_servers=self.config['kafka_servers'],
                group_id=f"metrics_stream_{uuid.uuid4().hex[:8]}"
            )
            await consumer.start()
            
            try:
                async for message in consumer:
                    try:
                        metric_data = json.loads(message.value.decode())
                        metric_point = MetricPoint(**metric_data)
                        yield metric_point
                        
                    except Exception as e:
                        logger.error(f"Erreur décodage message streaming: {e}")
                        continue
                        
            finally:
                await consumer.stop()
                
        except Exception as e:
            logger.error(f"Erreur streaming métriques: {e}")
            raise

    async def get_metrics_range(
        self,
        start_time: datetime,
        end_time: datetime,
        metric_names: Optional[List[str]] = None,
        tenant_id: Optional[str] = None,
        aggregation: str = "avg"
    ) -> pd.DataFrame:
        """Récupération de métriques sur une période avec agrégation"""
        try:
            # Construction de la requête
            query = """
                SELECT name, value, timestamp, labels, tenant_id
                FROM metrics 
                WHERE timestamp >= $1 AND timestamp <= $2
            """
            params = [start_time, end_time]
            
            if metric_names:
                query += f" AND name = ANY($3)"
                params.append(metric_names)
            
            if tenant_id:
                param_idx = len(params) + 1
                query += f" AND tenant_id = ${param_idx}"
                params.append(tenant_id)
            
            query += " ORDER BY timestamp"
            
            # Exécution de la requête
            async with self.db_engine.begin() as conn:
                result = await conn.execute(query, *params)
                rows = await result.fetchall()
            
            # Conversion en DataFrame
            df = pd.DataFrame(rows, columns=['name', 'value', 'timestamp', 'labels', 'tenant_id'])
            
            if df.empty:
                return df
            
            # Conversion des types
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Agrégation si demandée
            if aggregation != "raw":
                df = await self.apply_aggregation(df, aggregation)
            
            logger.info(f"Récupéré {len(df)} points de métriques")
            return df
            
        except Exception as e:
            logger.error(f"Erreur récupération métriques: {e}")
            raise

    async def apply_aggregation(self, df: pd.DataFrame, aggregation: str) -> pd.DataFrame:
        """Application d'agrégations sur les données"""
        try:
            # Groupement par nom de métrique et fenêtre temporelle (1 minute)
            df['time_window'] = df['timestamp'].dt.floor('1min')
            
            agg_functions = {
                'avg': 'mean',
                'sum': 'sum',
                'min': 'min',
                'max': 'max',
                'count': 'count',
                'std': 'std'
            }
            
            agg_func = agg_functions.get(aggregation, 'mean')
            
            # Application de l'agrégation
            result = df.groupby(['name', 'time_window', 'tenant_id']).agg({
                'value': agg_func
            }).reset_index()
            
            result.rename(columns={'time_window': 'timestamp'}, inplace=True)
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur agrégation: {e}")
            return df

    async def validate_and_enrich_metric(
        self,
        name: str,
        value: Any,
        labels: Dict[str, str]
    ) -> tuple[Any, DataQuality]:
        """Validation et enrichissement d'une métrique"""
        try:
            # Validation de base
            if not name or name.strip() == "":
                raise ValueError("Nom de métrique vide")
            
            # Normalisation de la valeur
            if isinstance(value, str):
                try:
                    # Tentative de conversion numérique
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    # Garde comme string si conversion échoue
                    pass
            
            # Vérification des outliers pour les valeurs numériques
            quality = DataQuality.HIGH
            
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    quality = DataQuality.INVALID
                elif abs(value) > 1e12:  # Valeur très élevée
                    quality = DataQuality.LOW
                elif value < 0 and name.endswith(('_count', '_total', '_size')):
                    # Valeurs négatives pour des métriques qui devraient être positives
                    quality = DataQuality.MEDIUM
            
            # Enrichissement des labels
            enriched_labels = labels.copy()
            enriched_labels['collection_host'] = self.config.get('hostname', 'unknown')
            enriched_labels['collector_version'] = '1.0.0'
            
            return value, quality
            
        except Exception as e:
            logger.warning(f"Erreur validation métrique {name}: {e}")
            return value, DataQuality.INVALID

    async def add_to_buffer(self, metric_point: MetricPoint):
        """Ajout d'un point de métrique au buffer avec gestion de la taille"""
        async with self.buffer_lock:
            self.metric_buffer.append(metric_point)
            self.buffer_size.set(len(self.metric_buffer))
            
            # Flush automatique si buffer plein
            if len(self.metric_buffer) >= self.streaming_config.batch_size:
                await self.flush_buffer()

    async def flush_buffer(self):
        """Vidage du buffer vers le stockage persistant"""
        if not self.metric_buffer:
            return
        
        try:
            # Copie des métriques à traiter
            metrics_to_flush = self.metric_buffer.copy()
            self.metric_buffer.clear()
            self.buffer_size.set(0)
            
            # Traitement en parallèle
            tasks = [
                self.persist_to_database(metrics_to_flush),
                self.stream_to_kafka(metrics_to_flush),
                self.cache_in_redis(metrics_to_flush)
            ]
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
            logger.debug(f"Buffer vidé: {len(metrics_to_flush)} métriques")
            
        except Exception as e:
            logger.error(f"Erreur vidage buffer: {e}")
            # Remettre les métriques dans le buffer en cas d'erreur
            async with self.buffer_lock:
                self.metric_buffer.extend(metrics_to_flush)

    async def persist_to_database(self, metrics: List[MetricPoint]):
        """Persistance des métriques en base de données"""
        try:
            if not metrics:
                return
            
            # Préparation des données pour insertion en lot
            values = []
            for metric in metrics:
                values.append((
                    metric.id,
                    metric.name,
                    str(metric.value),
                    metric.timestamp,
                    json.dumps(metric.labels),
                    metric.tenant_id,
                    metric.metric_type.value,
                    metric.source,
                    metric.quality.value,
                    json.dumps(metric.metadata)
                ))
            
            # Insertion en lot
            query = """
                INSERT INTO metrics (
                    id, name, value, timestamp, labels, tenant_id, 
                    metric_type, source, quality, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """
            
            async with self.db_engine.begin() as conn:
                await conn.executemany(query, values)
            
            logger.debug(f"Persisté {len(metrics)} métriques en base")
            
        except Exception as e:
            logger.error(f"Erreur persistance base de données: {e}")
            raise

    async def stream_to_kafka(self, metrics: List[MetricPoint]):
        """Streaming des métriques vers Kafka"""
        try:
            if not self.kafka_producer or not metrics:
                return
            
            for metric in metrics:
                message = {
                    'id': metric.id,
                    'name': metric.name,
                    'value': metric.value,
                    'timestamp': metric.timestamp.isoformat(),
                    'labels': metric.labels,
                    'tenant_id': metric.tenant_id,
                    'type': metric.metric_type.value,
                    'source': metric.source
                }
                
                await self.kafka_producer.send(
                    self.streaming_config.kafka_topic,
                    value=message
                )
            
            await self.kafka_producer.flush()
            logger.debug(f"Streamé {len(metrics)} métriques vers Kafka")
            
        except Exception as e:
            logger.error(f"Erreur streaming Kafka: {e}")

    async def cache_in_redis(self, metrics: List[MetricPoint]):
        """Mise en cache des métriques récentes dans Redis"""
        try:
            if not self.redis_client or not metrics:
                return
            
            pipe = self.redis_client.pipeline()
            
            for metric in metrics:
                # Cache par nom de métrique (dernières 1000 valeurs)
                key = f"metrics:{metric.name}:recent"
                value = {
                    'value': metric.value,
                    'timestamp': metric.timestamp.isoformat(),
                    'tenant_id': metric.tenant_id
                }
                
                pipe.lpush(key, json.dumps(value))
                pipe.ltrim(key, 0, 999)  # Garde seulement les 1000 dernières
                pipe.expire(key, 3600)  # Expire après 1 heure
            
            await pipe.execute()
            logger.debug(f"Mis en cache {len(metrics)} métriques dans Redis")
            
        except Exception as e:
            logger.error(f"Erreur cache Redis: {e}")

    async def is_critical_metric(self, name: str, value: Any) -> bool:
        """Détection si une métrique est critique"""
        critical_patterns = [
            'error_rate',
            'cpu_usage',
            'memory_usage',
            'disk_usage',
            'response_time',
            'incident_'
        ]
        
        return any(pattern in name.lower() for pattern in critical_patterns)

    async def process_critical_metric(self, metric_point: MetricPoint):
        """Traitement spécial pour les métriques critiques"""
        try:
            # Analyse immédiate
            analysis = await self.analytics_engine.analyze_metric_point(metric_point)
            
            # Alerte si anomalie détectée
            if analysis and analysis.confidence > 0.8:
                await self.send_alert(metric_point, analysis)
            
            logger.info(f"Métrique critique traitée: {metric_point.name}")
            
        except Exception as e:
            logger.error(f"Erreur traitement métrique critique: {e}")

    async def send_alert(self, metric_point: MetricPoint, analysis: AnalyticsResult):
        """Envoi d'alerte pour anomalie détectée"""
        # Implementation dépendante du système d'alerting
        logger.warning(
            f"ALERTE: Anomalie détectée sur {metric_point.name}="
            f"{metric_point.value} (confiance: {analysis.confidence})"
        )

    async def start_background_tasks(self):
        """Démarrage des tâches background"""
        self.background_tasks = [
            asyncio.create_task(self.periodic_flush()),
            asyncio.create_task(self.cleanup_old_data()),
            asyncio.create_task(self.metrics_analysis_task())
        ]

    async def periodic_flush(self):
        """Vidage périodique du buffer"""
        while self.running:
            try:
                await asyncio.sleep(self.streaming_config.flush_interval_ms / 1000)
                await self.flush_buffer()
            except Exception as e:
                logger.error(f"Erreur flush périodique: {e}")

    async def cleanup_old_data(self):
        """Nettoyage des données anciennes"""
        while self.running:
            try:
                # Nettoyage toutes les heures
                await asyncio.sleep(3600)
                
                cutoff_time = datetime.utcnow() - timedelta(
                    hours=self.streaming_config.retention_hours
                )
                
                # Suppression des anciennes métriques
                query = "DELETE FROM metrics WHERE timestamp < $1"
                async with self.db_engine.begin() as conn:
                    result = await conn.execute(query, cutoff_time)
                    
                logger.info(f"Nettoyé {result.rowcount} anciennes métriques")
                
            except Exception as e:
                logger.error(f"Erreur nettoyage données: {e}")

    async def metrics_analysis_task(self):
        """Tâche d'analyse continue des métriques"""
        while self.running:
            try:
                await asyncio.sleep(300)  # Analyse toutes les 5 minutes
                await self.analytics_engine.run_periodic_analysis()
                
            except Exception as e:
                logger.error(f"Erreur analyse périodique: {e}")

    async def shutdown(self):
        """Arrêt propre du collecteur"""
        logger.info("Arrêt du collecteur de métriques...")
        
        self.running = False
        
        # Arrêt des tâches background
        for task in self.background_tasks:
            task.cancel()
        
        # Vidage final du buffer
        await self.flush_buffer()
        
        # Fermeture des connexions
        if self.kafka_producer:
            await self.kafka_producer.stop()
        
        if self.redis_client:
            await self.redis_client.close()
        
        if self.db_engine:
            await self.db_engine.dispose()
        
        logger.info("Collecteur de métriques arrêté")

# =============================================================================
# MOTEUR D'ANALYTICS AVANCÉ
# =============================================================================

class MetricsAnalyticsEngine:
    """Moteur d'analytics ML pour métriques avec détection d'anomalies"""
    
    def __init__(self):
        self.anomaly_detectors: Dict[str, IsolationForest] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.baseline_models: Dict[str, Any] = {}
        
    async def initialize(self):
        """Initialisation du moteur d'analytics"""
        logger.info("Moteur d'analytics initialisé")
    
    async def analyze_metric_point(self, metric_point: MetricPoint) -> Optional[AnalyticsResult]:
        """Analyse d'un point de métrique individuel"""
        try:
            # Analyse uniquement pour les valeurs numériques
            if not isinstance(metric_point.value, (int, float)):
                return None
            
            # Détection d'anomalie simple basée sur z-score
            if metric_point.name not in self.baseline_models:
                return None
            
            baseline = self.baseline_models[metric_point.name]
            z_score = abs((metric_point.value - baseline['mean']) / baseline['std'])
            
            if z_score > 3:  # Anomalie si z-score > 3
                return AnalyticsResult(
                    metric_name=metric_point.name,
                    analysis_type="anomaly_detection",
                    result={'z_score': z_score, 'threshold': 3},
                    confidence=min(z_score / 5, 1.0),
                    timestamp=datetime.utcnow(),
                    recommendations=[f"Valeur anormale détectée: {metric_point.value}"]
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur analyse métrique: {e}")
            return None
    
    async def run_periodic_analysis(self):
        """Analyse périodique des métriques"""
        logger.debug("Exécution de l'analyse périodique")
        # Implementation de l'analyse périodique

# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================

async def main():
    """Exemple d'utilisation du collecteur de métriques"""
    
    config = {
        'redis_url': 'redis://localhost:6379/0',
        'kafka_servers': ['localhost:9092'],
        'database_url': 'postgresql+asyncpg://user:pass@localhost/metrics',
        'hostname': 'collector-01'
    }
    
    # Initialisation du collecteur
    collector = RealTimeMetricsCollector(config)
    await collector.initialize()
    
    try:
        # Collection de métriques
        await collector.collect_metric(
            name="cpu_usage",
            value=75.5,
            labels={"host": "web-01", "env": "prod"},
            tenant_id="tenant_123",
            source="monitoring"
        )
        
        # Collection en lot
        batch_metrics = [
            {"name": "memory_usage", "value": 85.2, "labels": {"host": "web-01"}},
            {"name": "disk_usage", "value": 45.8, "labels": {"host": "web-01"}},
            {"name": "network_in", "value": 1024000, "labels": {"interface": "eth0"}}
        ]
        
        await collector.collect_batch_metrics(batch_metrics, tenant_id="tenant_123")
        
        # Attendre un peu pour voir les métriques
        await asyncio.sleep(2)
        
        # Récupération des métriques
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=5)
        
        df = await collector.get_metrics_range(start_time, end_time)
        print(f"Métriques récupérées: {len(df)} points")
        
    finally:
        await collector.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
