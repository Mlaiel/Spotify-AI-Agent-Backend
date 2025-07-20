# =============================================================================
# Data Layer - Architecture Enterprise
# =============================================================================
# 
# Module de la couche de données avec métriques temps réel, analytics ML avancés,
# et analyse prédictive d'incidents. Architecture orientée données avec
# traitement de volumes importants et intelligence artificielle.
#
# Auteur: Data Engineering Team + ML Architects
# Direction Technique: Fahed Mlaiel
# Version: 2.0.0 Enterprise
# =============================================================================

"""
Data Layer Enterprise

Ce module fournit la couche de données complète avec:

Fonctionnalités Principales:
- Collection de métriques temps réel multi-sources
- Analytics ML avancés avec modèles prédictifs
- Analyse prédictive d'incidents avec forecasting
- Agrégation et corrélation de données cross-platform
- Data warehousing optimisé pour l'analyse

Composants:
- MetricsCollector: Collection temps réel de métriques
- MLAnalyticsEngine: Moteur d'analytics avec ML
- PredictiveAnalyzer: Analyse prédictive d'incidents
- DataPipeline: Pipeline de traitement ETL
- DataWarehouse: Stockage optimisé pour l'analyse

Architecture:
- Stream Processing pour métriques temps réel
- Batch Processing pour analytics ML
- Feature Store pour modèles ML
- Time Series Database pour métriques
- Graph Database pour relations d'incidents
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import json
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from decimal import Decimal

# ML et analytics
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import DBSCAN, KMeans
import scipy.stats as stats

# Time series analysis
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("statsmodels non disponible - fonctionnalités time series limitées")

# Imports du Core Engine
from . import (
    core_registry, TenantContext, TenantTier, EngineStatus
)

logger = logging.getLogger(__name__)

# ===========================
# Configuration & Constants
# ===========================

# Configuration de collection de métriques
METRICS_CONFIG = {
    "collection_interval_seconds": 30,
    "batch_size": 1000,
    "retention_days": 90,
    "aggregation_intervals": ["1m", "5m", "15m", "1h", "1d"],
    "alert_thresholds": {
        "cpu_usage": 80.0,
        "memory_usage": 85.0,
        "disk_usage": 90.0,
        "error_rate": 5.0,
        "response_time": 2000.0  # ms
    }
}

# Configuration ML
ML_CONFIG = {
    "model_retrain_hours": 24,
    "feature_window_hours": 168,  # 7 jours
    "prediction_horizon_hours": 24,
    "anomaly_threshold": 0.05,
    "cluster_update_hours": 12,
    "feature_importance_threshold": 0.01
}

# Sources de données supportées
DATA_SOURCES = {
    "prometheus": "Prometheus Metrics",
    "elasticsearch": "Elasticsearch Logs",
    "grafana": "Grafana Dashboards",
    "custom_api": "Custom API Endpoints",
    "webhook": "Webhook Events",
    "database": "Database Metrics",
    "application": "Application Metrics"
}

# ===========================
# Enums & Types
# ===========================

class MetricType(Enum):
    """Types de métriques"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"

class DataSourceType(Enum):
    """Types de sources de données"""
    PROMETHEUS = "prometheus"
    ELASTICSEARCH = "elasticsearch"
    GRAFANA = "grafana"
    CUSTOM_API = "custom_api"
    WEBHOOK = "webhook"
    DATABASE = "database"
    APPLICATION = "application"

class AggregationType(Enum):
    """Types d'agrégation"""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    PERCENTILE = "percentile"
    STDDEV = "stddev"

class PredictionType(Enum):
    """Types de prédictions"""
    INCIDENT_PROBABILITY = "incident_probability"
    RESOURCE_USAGE = "resource_usage"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    FAILURE_PREDICTION = "failure_prediction"
    CAPACITY_PLANNING = "capacity_planning"

class AnalysisType(Enum):
    """Types d'analyses"""
    ANOMALY_DETECTION = "anomaly_detection"
    TREND_ANALYSIS = "trend_analysis"
    CORRELATION_ANALYSIS = "correlation_analysis"
    CLUSTERING = "clustering"
    FORECASTING = "forecasting"
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"

# ===========================
# Modèles de Données
# ===========================

@dataclass
class Metric:
    """Métrique individuelle"""
    name: str
    value: Union[int, float]
    timestamp: datetime
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    tenant_id: Optional[str] = None
    source: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire"""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "type": self.metric_type.value,
            "labels": self.labels,
            "tenant_id": self.tenant_id,
            "source": self.source
        }

@dataclass
class MetricSeries:
    """Série temporelle de métriques"""
    metric_name: str
    tenant_id: str
    data_points: List[Tuple[datetime, float]] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_point(self, timestamp: datetime, value: float):
        """Ajoute un point de données"""
        self.data_points.append((timestamp, value))
        # Garde seulement les derniers N points pour la mémoire
        if len(self.data_points) > 10000:
            self.data_points = self.data_points[-5000:]
    
    def get_values(self) -> List[float]:
        """Récupère uniquement les valeurs"""
        return [point[1] for point in self.data_points]
    
    def get_timestamps(self) -> List[datetime]:
        """Récupère uniquement les timestamps"""
        return [point[0] for point in self.data_points]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Conversion en DataFrame pandas"""
        return pd.DataFrame(self.data_points, columns=['timestamp', 'value'])

@dataclass
class AnomalyDetection:
    """Résultat de détection d'anomalie"""
    metric_name: str
    tenant_id: str
    timestamp: datetime
    value: float
    anomaly_score: float
    is_anomaly: bool
    threshold: float
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "tenant_id": self.tenant_id,
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "anomaly_score": self.anomaly_score,
            "is_anomaly": self.is_anomaly,
            "threshold": self.threshold,
            "context": self.context
        }

@dataclass
class PredictionResult:
    """Résultat de prédiction"""
    prediction_type: PredictionType
    tenant_id: str
    target_metric: str
    predicted_value: float
    confidence_interval: Tuple[float, float]
    prediction_horizon: timedelta
    model_accuracy: float
    features_used: List[str]
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_type": self.prediction_type.value,
            "tenant_id": self.tenant_id,
            "target_metric": self.target_metric,
            "predicted_value": self.predicted_value,
            "confidence_interval": list(self.confidence_interval),
            "prediction_horizon_hours": self.prediction_horizon.total_seconds() / 3600,
            "model_accuracy": self.model_accuracy,
            "features_used": self.features_used,
            "created_at": self.created_at.isoformat()
        }

@dataclass
class CorrelationAnalysis:
    """Analyse de corrélation entre métriques"""
    tenant_id: str
    metric_pairs: List[Tuple[str, str]]
    correlation_matrix: Dict[str, Dict[str, float]]
    significant_correlations: List[Dict[str, Any]]
    analyzed_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_strong_correlations(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Récupère les corrélations fortes"""
        strong_corrs = []
        for corr in self.significant_correlations:
            if abs(corr["correlation"]) >= threshold:
                strong_corrs.append(corr)
        return strong_corrs

# ===========================
# Collecteur de Métriques
# ===========================

class MetricsCollector:
    """Collecteur de métriques temps réel"""
    
    def __init__(self):
        self.active_sources: Dict[str, Dict[str, Any]] = {}
        self.metric_series: Dict[str, MetricSeries] = {}
        self.collection_tasks: Dict[str, asyncio.Task] = {}
        self.metrics_buffer: deque = deque(maxlen=10000)
        
        logger.info("Collecteur de métriques initialisé")
    
    def register_source(self, source_id: str, source_config: Dict[str, Any]):
        """Enregistre une source de données"""
        self.active_sources[source_id] = {
            "type": DataSourceType(source_config["type"]),
            "config": source_config,
            "last_collection": None,
            "error_count": 0,
            "metrics_collected": 0
        }
        
        logger.info(f"Source de données enregistrée: {source_id} ({source_config['type']})")
    
    async def start_collection(self, source_id: str):
        """Démarre la collection pour une source"""
        if source_id not in self.active_sources:
            logger.error(f"Source inconnue: {source_id}")
            return
        
        source = self.active_sources[source_id]
        
        if source_id in self.collection_tasks:
            logger.warning(f"Collection déjà active pour {source_id}")
            return
        
        # Création de la tâche de collection
        task = asyncio.create_task(self._collection_loop(source_id))
        self.collection_tasks[source_id] = task
        
        logger.info(f"Collection démarrée pour {source_id}")
    
    async def stop_collection(self, source_id: str):
        """Arrête la collection pour une source"""
        if source_id in self.collection_tasks:
            task = self.collection_tasks[source_id]
            task.cancel()
            del self.collection_tasks[source_id]
            logger.info(f"Collection arrêtée pour {source_id}")
    
    async def _collection_loop(self, source_id: str):
        """Boucle de collection pour une source"""
        source = self.active_sources[source_id]
        source_type = source["type"]
        config = source["config"]
        
        interval = config.get("interval_seconds", METRICS_CONFIG["collection_interval_seconds"])
        
        while True:
            try:
                # Collection selon le type de source
                metrics = await self._collect_from_source(source_type, config)
                
                # Traitement des métriques collectées
                for metric in metrics:
                    await self.process_metric(metric)
                
                # Mise à jour des statistiques
                source["last_collection"] = datetime.utcnow()
                source["metrics_collected"] += len(metrics)
                source["error_count"] = 0  # Reset sur succès
                
                logger.debug(f"Collecté {len(metrics)} métriques de {source_id}")
                
            except Exception as e:
                source["error_count"] += 1
                logger.error(f"Erreur collection {source_id}: {e}")
                
                # Arrêt si trop d'erreurs
                if source["error_count"] > 10:
                    logger.error(f"Trop d'erreurs pour {source_id}, arrêt de la collection")
                    break
            
            await asyncio.sleep(interval)
    
    async def _collect_from_source(self, source_type: DataSourceType, 
                                 config: Dict[str, Any]) -> List[Metric]:
        """Collecte les métriques d'une source spécifique"""
        metrics = []
        
        if source_type == DataSourceType.PROMETHEUS:
            metrics = await self._collect_prometheus(config)
        elif source_type == DataSourceType.APPLICATION:
            metrics = await self._collect_application(config)
        elif source_type == DataSourceType.CUSTOM_API:
            metrics = await self._collect_custom_api(config)
        else:
            logger.warning(f"Type de source non supporté: {source_type}")
        
        return metrics
    
    async def _collect_prometheus(self, config: Dict[str, Any]) -> List[Metric]:
        """Collecte depuis Prometheus"""
        # Simulation de collection Prometheus
        metrics = []
        tenant_id = config.get("tenant_id", "default")
        
        # Métriques système simulées
        system_metrics = [
            ("cpu_usage_percent", np.random.uniform(10, 90), MetricType.GAUGE),
            ("memory_usage_percent", np.random.uniform(20, 80), MetricType.GAUGE),
            ("disk_usage_percent", np.random.uniform(30, 70), MetricType.GAUGE),
            ("network_bytes_total", np.random.randint(1000000, 10000000), MetricType.COUNTER),
            ("request_duration_seconds", np.random.uniform(0.1, 2.0), MetricType.HISTOGRAM)
        ]
        
        for name, value, metric_type in system_metrics:
            metric = Metric(
                name=name,
                value=value,
                timestamp=datetime.utcnow(),
                metric_type=metric_type,
                tenant_id=tenant_id,
                source="prometheus",
                labels={"instance": "web-01", "job": "application"}
            )
            metrics.append(metric)
        
        return metrics
    
    async def _collect_application(self, config: Dict[str, Any]) -> List[Metric]:
        """Collecte depuis l'application"""
        metrics = []
        tenant_id = config.get("tenant_id", "default")
        
        # Métriques application simulées
        app_metrics = [
            ("incidents_per_minute", np.random.poisson(5), MetricType.GAUGE),
            ("response_time_ms", np.random.lognormal(6, 0.5), MetricType.GAUGE),
            ("error_rate_percent", np.random.exponential(2), MetricType.GAUGE),
            ("active_users", np.random.randint(50, 500), MetricType.GAUGE),
            ("throughput_requests_per_second", np.random.gamma(2, 10), MetricType.GAUGE)
        ]
        
        for name, value, metric_type in app_metrics:
            metric = Metric(
                name=name,
                value=float(value),
                timestamp=datetime.utcnow(),
                metric_type=metric_type,
                tenant_id=tenant_id,
                source="application"
            )
            metrics.append(metric)
        
        return metrics
    
    async def _collect_custom_api(self, config: Dict[str, Any]) -> List[Metric]:
        """Collecte depuis API personnalisée"""
        # Simulation - dans un vrai système, ceci ferait des appels HTTP
        metrics = []
        tenant_id = config.get("tenant_id", "default")
        
        # Business metrics simulées
        business_metrics = [
            ("customer_satisfaction", np.random.uniform(3.5, 5.0), MetricType.GAUGE),
            ("revenue_per_hour", np.random.uniform(1000, 5000), MetricType.GAUGE),
            ("conversion_rate", np.random.uniform(0.02, 0.08), MetricType.GAUGE),
            ("churn_rate", np.random.uniform(0.01, 0.05), MetricType.GAUGE)
        ]
        
        for name, value, metric_type in business_metrics:
            metric = Metric(
                name=name,
                value=value,
                timestamp=datetime.utcnow(),
                metric_type=metric_type,
                tenant_id=tenant_id,
                source="custom_api"
            )
            metrics.append(metric)
        
        return metrics
    
    async def process_metric(self, metric: Metric):
        """Traite une métrique collectée"""
        # Ajout au buffer
        self.metrics_buffer.append(metric)
        
        # Création/mise à jour de la série temporelle
        series_key = f"{metric.tenant_id}:{metric.name}"
        if series_key not in self.metric_series:
            self.metric_series[series_key] = MetricSeries(
                metric_name=metric.name,
                tenant_id=metric.tenant_id,
                labels=metric.labels
            )
        
        series = self.metric_series[series_key]
        series.add_point(metric.timestamp, metric.value)
        
        # Vérification des seuils d'alerte
        await self._check_alert_thresholds(metric)
    
    async def _check_alert_thresholds(self, metric: Metric):
        """Vérifie les seuils d'alerte"""
        thresholds = METRICS_CONFIG["alert_thresholds"]
        
        if metric.name in thresholds:
            threshold = thresholds[metric.name]
            if metric.value > threshold:
                logger.warning(f"Seuil d'alerte dépassé: {metric.name} = {metric.value} "
                             f"(seuil: {threshold}) pour tenant {metric.tenant_id}")
    
    def get_metric_series(self, tenant_id: str, metric_name: str) -> Optional[MetricSeries]:
        """Récupère une série temporelle"""
        series_key = f"{tenant_id}:{metric_name}"
        return self.metric_series.get(series_key)
    
    def get_recent_metrics(self, tenant_id: str, hours: int = 1) -> List[Metric]:
        """Récupère les métriques récentes d'un tenant"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_metrics = []
        for metric in self.metrics_buffer:
            if (metric.tenant_id == tenant_id and 
                metric.timestamp > cutoff_time):
                recent_metrics.append(metric)
        
        return recent_metrics

# ===========================
# Moteur Analytics ML
# ===========================

class MLAnalyticsEngine:
    """Moteur d'analytics avancés avec ML"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_importance: Dict[str, Dict[str, float]] = {}
        
        # Modèles ML
        self.anomaly_detectors: Dict[str, IsolationForest] = {}
        self.predictive_models: Dict[str, RandomForestRegressor] = {}
        
        logger.info("Moteur ML Analytics initialisé")
    
    async def train_anomaly_detector(self, tenant_id: str, metric_name: str) -> bool:
        """Entraîne un détecteur d'anomalies pour une métrique"""
        series = self.metrics_collector.get_metric_series(tenant_id, metric_name)
        if not series or len(series.data_points) < 100:
            logger.warning(f"Pas assez de données pour {tenant_id}:{metric_name}")
            return False
        
        # Préparation des données
        df = series.to_dataframe()
        
        # Features engineering
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['rolling_mean'] = df['value'].rolling(window=10).mean()
        df['rolling_std'] = df['value'].rolling(window=10).std()
        
        # Nettoyage
        df = df.dropna()
        
        if len(df) < 50:
            logger.warning(f"Pas assez de données après nettoyage pour {tenant_id}:{metric_name}")
            return False
        
        # Features pour l'entraînement
        features = ['value', 'hour', 'day_of_week', 'rolling_mean', 'rolling_std']
        X = df[features].values
        
        # Normalisation
        scaler_key = f"{tenant_id}:{metric_name}"
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[scaler_key] = scaler
        
        # Entraînement du modèle d'anomalie
        model = IsolationForest(
            contamination=ML_CONFIG["anomaly_threshold"],
            random_state=42,
            n_estimators=100
        )
        model.fit(X_scaled)
        
        self.anomaly_detectors[scaler_key] = model
        
        logger.info(f"Détecteur d'anomalies entraîné pour {tenant_id}:{metric_name}")
        return True
    
    async def detect_anomalies(self, tenant_id: str, metric_name: str) -> List[AnomalyDetection]:
        """Détecte les anomalies dans une métrique"""
        detector_key = f"{tenant_id}:{metric_name}"
        
        if detector_key not in self.anomaly_detectors:
            await self.train_anomaly_detector(tenant_id, metric_name)
        
        if detector_key not in self.anomaly_detectors:
            return []
        
        model = self.anomaly_detectors[detector_key]
        scaler = self.scalers[detector_key]
        
        # Récupération des données récentes
        series = self.metrics_collector.get_metric_series(tenant_id, metric_name)
        if not series:
            return []
        
        # Préparation des données récentes
        df = series.to_dataframe()
        if len(df) < 10:
            return []
        
        # Features engineering
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['rolling_mean'] = df['value'].rolling(window=10).mean()
        df['rolling_std'] = df['value'].rolling(window=10).std()
        df = df.dropna()
        
        if len(df) == 0:
            return []
        
        # Prédiction d'anomalies
        features = ['value', 'hour', 'day_of_week', 'rolling_mean', 'rolling_std']
        X = df[features].values
        X_scaled = scaler.transform(X)
        
        # Scores d'anomalie
        anomaly_scores = model.decision_function(X_scaled)
        anomaly_predictions = model.predict(X_scaled)
        
        # Conversion en résultats
        anomalies = []
        threshold = np.percentile(anomaly_scores, 5)  # 5% plus bas = anomalies
        
        for i, (_, row) in enumerate(df.iterrows()):
            is_anomaly = anomaly_predictions[i] == -1
            score = anomaly_scores[i]
            
            if is_anomaly:
                anomaly = AnomalyDetection(
                    metric_name=metric_name,
                    tenant_id=tenant_id,
                    timestamp=row['timestamp'],
                    value=row['value'],
                    anomaly_score=score,
                    is_anomaly=True,
                    threshold=threshold,
                    context={
                        "rolling_mean": row['rolling_mean'],
                        "rolling_std": row['rolling_std'],
                        "hour": row['hour'],
                        "day_of_week": row['day_of_week']
                    }
                )
                anomalies.append(anomaly)
        
        if anomalies:
            logger.info(f"Détecté {len(anomalies)} anomalies pour {tenant_id}:{metric_name}")
        
        return anomalies
    
    async def train_predictive_model(self, tenant_id: str, target_metric: str, 
                                   feature_metrics: List[str]) -> bool:
        """Entraîne un modèle prédictif"""
        # Collecte des données pour toutes les métriques
        all_data = {}
        
        # Métrique cible
        target_series = self.metrics_collector.get_metric_series(tenant_id, target_metric)
        if not target_series or len(target_series.data_points) < 200:
            logger.warning(f"Pas assez de données cible pour {tenant_id}:{target_metric}")
            return False
        
        all_data[target_metric] = target_series.to_dataframe()
        
        # Features
        for feature_metric in feature_metrics:
            feature_series = self.metrics_collector.get_metric_series(tenant_id, feature_metric)
            if feature_series and len(feature_series.data_points) >= 100:
                all_data[feature_metric] = feature_series.to_dataframe()
        
        if len(all_data) < 2:  # Besoin d'au moins la cible + 1 feature
            logger.warning(f"Pas assez de features pour {tenant_id}:{target_metric}")
            return False
        
        # Alignement temporel des données
        aligned_data = self._align_time_series(all_data)
        
        if len(aligned_data) < 100:
            logger.warning(f"Pas assez de données alignées pour {tenant_id}:{target_metric}")
            return False
        
        # Préparation des features
        X, y = self._prepare_features_for_prediction(aligned_data, target_metric)
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Normalisation
        scaler_key = f"{tenant_id}:{target_metric}:predictive"
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers[scaler_key] = scaler
        
        # Entraînement du modèle
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Évaluation
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Sauvegarde du modèle
        model_key = f"{tenant_id}:{target_metric}:predictive"
        self.predictive_models[model_key] = model
        
        # Feature importance
        feature_names = self._get_feature_names(feature_metrics)
        importances = dict(zip(feature_names, model.feature_importances_))
        self.feature_importance[model_key] = importances
        
        logger.info(f"Modèle prédictif entraîné pour {tenant_id}:{target_metric} "
                   f"(MSE: {mse:.4f}, R²: {r2:.4f})")
        
        return True
    
    def _align_time_series(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Aligne plusieurs séries temporelles"""
        # Conversion de tous les DataFrames avec timestamp comme index
        aligned_dfs = {}
        
        for metric_name, df in data_dict.items():
            df_copy = df.copy()
            df_copy.set_index('timestamp', inplace=True)
            df_copy.columns = [f"{metric_name}_{col}" if col != 'timestamp' else col 
                              for col in df_copy.columns]
            aligned_dfs[metric_name] = df_copy
        
        # Jointure externe de tous les DataFrames
        aligned = pd.concat(aligned_dfs.values(), axis=1, join='outer')
        
        # Interpolation des valeurs manquantes
        aligned = aligned.interpolate(method='linear')
        
        # Suppression des lignes avec trop de valeurs manquantes
        aligned = aligned.dropna()
        
        return aligned
    
    def _prepare_features_for_prediction(self, df: pd.DataFrame, 
                                       target_metric: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prépare les features pour la prédiction"""
        # Colonne cible
        target_col = f"{target_metric}_value"
        y = df[target_col].values
        
        # Features temporelles
        df_features = df.copy()
        df_features['hour'] = df_features.index.hour
        df_features['day_of_week'] = df_features.index.dayofweek
        df_features['month'] = df_features.index.month
        
        # Features de décalage temporel (lag features)
        for col in df_features.columns:
            if col.endswith('_value') and col != target_col:
                for lag in [1, 2, 3, 6, 12]:  # 1-12 périodes de retard
                    df_features[f"{col}_lag_{lag}"] = df_features[col].shift(lag)
        
        # Features de moyennes mobiles
        for col in df_features.columns:
            if col.endswith('_value'):
                df_features[f"{col}_ma_5"] = df_features[col].rolling(5).mean()
                df_features[f"{col}_ma_10"] = df_features[col].rolling(10).mean()
        
        # Suppression de la colonne cible des features
        feature_cols = [col for col in df_features.columns if col != target_col]
        X = df_features[feature_cols].values
        
        # Nettoyage final
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]
        
        return X, y
    
    def _get_feature_names(self, feature_metrics: List[str]) -> List[str]:
        """Génère les noms de features"""
        names = []
        
        # Features de base
        for metric in feature_metrics:
            names.append(f"{metric}_value")
        
        # Features temporelles
        names.extend(['hour', 'day_of_week', 'month'])
        
        # Features de lag
        for metric in feature_metrics:
            for lag in [1, 2, 3, 6, 12]:
                names.append(f"{metric}_value_lag_{lag}")
        
        # Features de moyennes mobiles
        for metric in feature_metrics:
            names.extend([f"{metric}_value_ma_5", f"{metric}_value_ma_10"])
        
        return names
    
    async def perform_correlation_analysis(self, tenant_id: str, 
                                         metrics: List[str]) -> CorrelationAnalysis:
        """Effectue une analyse de corrélation entre métriques"""
        # Collecte des données
        data_dict = {}
        for metric_name in metrics:
            series = self.metrics_collector.get_metric_series(tenant_id, metric_name)
            if series and len(series.data_points) > 50:
                data_dict[metric_name] = series.to_dataframe()
        
        if len(data_dict) < 2:
            logger.warning(f"Pas assez de métriques pour l'analyse de corrélation: {tenant_id}")
            return CorrelationAnalysis(
                tenant_id=tenant_id,
                metric_pairs=[],
                correlation_matrix={},
                significant_correlations=[]
            )
        
        # Alignement des données
        aligned_data = self._align_time_series(data_dict)
        
        # Calcul de la matrice de corrélation
        value_cols = [col for col in aligned_data.columns if col.endswith('_value')]
        corr_matrix = aligned_data[value_cols].corr()
        
        # Conversion en dictionnaire
        corr_dict = {}
        for i, metric1 in enumerate(metrics):
            corr_dict[metric1] = {}
            for j, metric2 in enumerate(metrics):
                col1 = f"{metric1}_value"
                col2 = f"{metric2}_value"
                if col1 in corr_matrix.columns and col2 in corr_matrix.columns:
                    corr_dict[metric1][metric2] = corr_matrix.loc[col1, col2]
        
        # Identification des corrélations significatives
        significant_correlations = []
        for i, metric1 in enumerate(metrics):
            for j, metric2 in enumerate(metrics):
                if i < j:  # Éviter les doublons
                    corr_value = corr_dict.get(metric1, {}).get(metric2, 0)
                    if abs(corr_value) > 0.3:  # Seuil de signification
                        significant_correlations.append({
                            "metric1": metric1,
                            "metric2": metric2,
                            "correlation": corr_value,
                            "strength": self._correlation_strength(abs(corr_value))
                        })
        
        # Génération des paires
        metric_pairs = [(m1, m2) for i, m1 in enumerate(metrics) 
                       for j, m2 in enumerate(metrics) if i < j]
        
        return CorrelationAnalysis(
            tenant_id=tenant_id,
            metric_pairs=metric_pairs,
            correlation_matrix=corr_dict,
            significant_correlations=significant_correlations
        )
    
    def _correlation_strength(self, correlation: float) -> str:
        """Détermine la force de la corrélation"""
        if correlation >= 0.8:
            return "très forte"
        elif correlation >= 0.6:
            return "forte"
        elif correlation >= 0.4:
            return "modérée"
        elif correlation >= 0.2:
            return "faible"
        else:
            return "très faible"

# ===========================
# Analyseur Prédictif
# ===========================

class PredictiveAnalyzer:
    """Analyseur prédictif d'incidents et de performances"""
    
    def __init__(self, metrics_collector: MetricsCollector, 
                 ml_engine: MLAnalyticsEngine):
        self.metrics_collector = metrics_collector
        self.ml_engine = ml_engine
        self.prediction_cache: Dict[str, PredictionResult] = {}
        
        logger.info("Analyseur prédictif initialisé")
    
    async def predict_incident_probability(self, tenant_id: str, 
                                         horizon_hours: int = 24) -> PredictionResult:
        """Prédit la probabilité d'incident"""
        # Métriques clés pour la prédiction d'incident
        key_metrics = [
            "cpu_usage_percent",
            "memory_usage_percent", 
            "error_rate_percent",
            "response_time_ms",
            "throughput_requests_per_second"
        ]
        
        # Collecte des données récentes
        recent_data = {}
        for metric in key_metrics:
            series = self.metrics_collector.get_metric_series(tenant_id, metric)
            if series and len(series.data_points) > 50:
                recent_data[metric] = series.get_values()[-50:]  # 50 derniers points
        
        if len(recent_data) < 3:
            logger.warning(f"Pas assez de métriques pour prédire incidents: {tenant_id}")
            return self._create_default_prediction(
                PredictionType.INCIDENT_PROBABILITY, tenant_id, "incident_risk"
            )
        
        # Calcul de score de risque basé sur plusieurs facteurs
        risk_factors = []
        
        # Facteur 1: Anomalies récentes
        anomaly_count = 0
        for metric in key_metrics:
            anomalies = await self.ml_engine.detect_anomalies(tenant_id, metric)
            recent_anomalies = [a for a in anomalies 
                              if (datetime.utcnow() - a.timestamp).hours < 2]
            anomaly_count += len(recent_anomalies)
        
        anomaly_factor = min(anomaly_count / 10.0, 1.0)  # Normalisé à [0, 1]
        risk_factors.append(("anomalies", anomaly_factor))
        
        # Facteur 2: Tendance des métriques critiques
        trend_factor = 0
        for metric in ["cpu_usage_percent", "memory_usage_percent", "error_rate_percent"]:
            if metric in recent_data:
                values = recent_data[metric]
                if len(values) > 10:
                    # Calcul de tendance (pente de régression linéaire)
                    x = np.arange(len(values))
                    slope, _, _, _, _ = stats.linregress(x, values)
                    
                    # Tendance à la hausse = risque plus élevé
                    if slope > 0:
                        trend_factor += slope / 100.0  # Normalisation approximative
        
        trend_factor = min(trend_factor / 3.0, 1.0)  # Moyenne des 3 métriques
        risk_factors.append(("trends", trend_factor))
        
        # Facteur 3: Valeurs actuelles vs seuils
        threshold_factor = 0
        thresholds = {
            "cpu_usage_percent": 80,
            "memory_usage_percent": 85,
            "error_rate_percent": 5
        }
        
        for metric, threshold in thresholds.items():
            if metric in recent_data:
                current_value = recent_data[metric][-1]
                if current_value > threshold:
                    threshold_factor += (current_value - threshold) / threshold
        
        threshold_factor = min(threshold_factor / len(thresholds), 1.0)
        risk_factors.append(("thresholds", threshold_factor))
        
        # Calcul du score final
        weights = [0.4, 0.3, 0.3]  # Poids pour anomalies, tendances, seuils
        incident_probability = sum(factor * weight for (_, factor), weight 
                                 in zip(risk_factors, weights))
        
        # Intervalle de confiance basé sur la variabilité des données
        confidence_range = 0.1 + (0.2 * (1 - incident_probability))
        confidence_interval = (
            max(0, incident_probability - confidence_range),
            min(1, incident_probability + confidence_range)
        )
        
        # Précision du modèle (simulée)
        model_accuracy = 0.75 + (0.2 * min(len(recent_data) / 5.0, 1.0))
        
        prediction = PredictionResult(
            prediction_type=PredictionType.INCIDENT_PROBABILITY,
            tenant_id=tenant_id,
            target_metric="incident_risk",
            predicted_value=incident_probability,
            confidence_interval=confidence_interval,
            prediction_horizon=timedelta(hours=horizon_hours),
            model_accuracy=model_accuracy,
            features_used=list(recent_data.keys())
        )
        
        # Cache de la prédiction
        cache_key = f"{tenant_id}:incident_probability"
        self.prediction_cache[cache_key] = prediction
        
        logger.info(f"Prédiction d'incident pour {tenant_id}: {incident_probability:.3f} "
                   f"(confiance: {model_accuracy:.3f})")
        
        return prediction
    
    async def predict_resource_usage(self, tenant_id: str, resource_type: str, 
                                   horizon_hours: int = 24) -> PredictionResult:
        """Prédit l'usage des ressources"""
        metric_mapping = {
            "cpu": "cpu_usage_percent",
            "memory": "memory_usage_percent", 
            "disk": "disk_usage_percent",
            "network": "network_bytes_total"
        }
        
        target_metric = metric_mapping.get(resource_type)
        if not target_metric:
            logger.error(f"Type de ressource non supporté: {resource_type}")
            return self._create_default_prediction(
                PredictionType.RESOURCE_USAGE, tenant_id, resource_type
            )
        
        # Récupération des données historiques
        series = self.metrics_collector.get_metric_series(tenant_id, target_metric)
        if not series or len(series.data_points) < 100:
            logger.warning(f"Pas assez de données pour prédiction: {tenant_id}:{target_metric}")
            return self._create_default_prediction(
                PredictionType.RESOURCE_USAGE, tenant_id, target_metric
            )
        
        # Préparation des données
        df = series.to_dataframe()
        df = df.sort_values('timestamp')
        
        # Prédiction avec méthode simple (moyenne mobile + tendance)
        values = df['value'].values
        
        # Décomposition de la série temporelle si statsmodels disponible
        if STATSMODELS_AVAILABLE and len(values) > 24:
            try:
                # Tentative de décomposition saisonnière
                decomposition = seasonal_decompose(
                    values[-min(len(values), 168):],  # Dernière semaine max
                    model='additive',
                    period=24 if len(values) >= 48 else 12
                )
                
                trend = decomposition.trend
                seasonal = decomposition.seasonal
                
                # Prédiction basée sur tendance + saisonnalité
                last_trend = np.nanmean(trend[-5:]) if not np.isnan(trend[-5:]).all() else values[-1]
                seasonal_pattern = seasonal[-24:] if len(seasonal) >= 24 else [0] * 24
                
                # Projection pour horizon_hours
                predictions = []
                for h in range(horizon_hours):
                    seasonal_idx = h % len(seasonal_pattern)
                    predicted_value = last_trend + seasonal_pattern[seasonal_idx]
                    predictions.append(predicted_value)
                
                final_prediction = np.mean(predictions)
                
            except Exception as e:
                logger.warning(f"Échec décomposition saisonnière: {e}, utilisation méthode simple")
                final_prediction = self._simple_trend_prediction(values, horizon_hours)
        else:
            final_prediction = self._simple_trend_prediction(values, horizon_hours)
        
        # Calcul de l'intervalle de confiance
        recent_std = np.std(values[-24:]) if len(values) >= 24 else np.std(values)
        confidence_range = 1.96 * recent_std  # 95% de confiance
        
        confidence_interval = (
            max(0, final_prediction - confidence_range),
            final_prediction + confidence_range
        )
        
        # Précision basée sur la stabilité des données
        cv = recent_std / np.mean(values[-24:]) if len(values) >= 24 else 1.0
        model_accuracy = max(0.3, 1.0 - cv)  # Plus stable = plus précis
        
        prediction = PredictionResult(
            prediction_type=PredictionType.RESOURCE_USAGE,
            tenant_id=tenant_id,
            target_metric=target_metric,
            predicted_value=final_prediction,
            confidence_interval=confidence_interval,
            prediction_horizon=timedelta(hours=horizon_hours),
            model_accuracy=model_accuracy,
            features_used=[target_metric, "temporal_features"]
        )
        
        logger.info(f"Prédiction {resource_type} pour {tenant_id}: {final_prediction:.2f} "
                   f"(précision: {model_accuracy:.3f})")
        
        return prediction
    
    def _simple_trend_prediction(self, values: np.ndarray, horizon_hours: int) -> float:
        """Prédiction simple basée sur la tendance"""
        if len(values) < 5:
            return values[-1] if len(values) > 0 else 0
        
        # Calcul de la tendance avec régression linéaire
        x = np.arange(len(values))
        slope, intercept, _, _, _ = stats.linregress(x, values)
        
        # Projection dans le futur
        future_x = len(values) + horizon_hours / 2  # Point médian de la prédiction
        predicted_value = slope * future_x + intercept
        
        return predicted_value
    
    def _create_default_prediction(self, prediction_type: PredictionType, 
                                 tenant_id: str, target_metric: str) -> PredictionResult:
        """Crée une prédiction par défaut en cas de données insuffisantes"""
        return PredictionResult(
            prediction_type=prediction_type,
            tenant_id=tenant_id,
            target_metric=target_metric,
            predicted_value=0.0,
            confidence_interval=(0.0, 0.0),
            prediction_horizon=timedelta(hours=24),
            model_accuracy=0.0,
            features_used=[]
        )
    
    async def analyze_incident_patterns(self, tenant_id: str) -> Dict[str, Any]:
        """Analyse les patterns d'incidents"""
        # Récupération des métriques d'incidents
        incident_series = self.metrics_collector.get_metric_series(
            tenant_id, "incidents_per_minute"
        )
        
        if not incident_series or len(incident_series.data_points) < 100:
            return {"status": "insufficient_data", "patterns": []}
        
        # Conversion en DataFrame
        df = incident_series.to_dataframe()
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Analyse des patterns temporels
        patterns = {}
        
        # Pattern par heure
        hourly_avg = df.groupby('hour')['value'].mean()
        peak_hours = hourly_avg.nlargest(3).index.tolist()
        patterns['peak_hours'] = peak_hours
        
        # Pattern par jour de semaine
        daily_avg = df.groupby('day_of_week')['value'].mean()
        peak_days = daily_avg.nlargest(2).index.tolist()
        patterns['peak_days'] = peak_days
        
        # Détection de clusters temporels avec DBSCAN
        if len(df) > 50:
            # Features: heure + jour de semaine + valeur
            X = df[['hour', 'day_of_week', 'value']].values
            X_scaled = StandardScaler().fit_transform(X)
            
            clustering = DBSCAN(eps=0.5, min_samples=5)
            clusters = clustering.fit_predict(X_scaled)
            
            # Analyse des clusters
            unique_clusters = np.unique(clusters[clusters != -1])
            cluster_analysis = []
            
            for cluster_id in unique_clusters:
                cluster_mask = clusters == cluster_id
                cluster_data = df[cluster_mask]
                
                cluster_info = {
                    "cluster_id": int(cluster_id),
                    "size": int(np.sum(cluster_mask)),
                    "avg_incidents": float(cluster_data['value'].mean()),
                    "peak_hour": int(cluster_data.groupby('hour')['value'].mean().idxmax()),
                    "peak_day": int(cluster_data.groupby('day_of_week')['value'].mean().idxmax())
                }
                cluster_analysis.append(cluster_info)
            
            patterns['clusters'] = cluster_analysis
        
        return {
            "status": "success",
            "tenant_id": tenant_id,
            "patterns": patterns,
            "analysis_period": {
                "start": df['timestamp'].min().isoformat(),
                "end": df['timestamp'].max().isoformat(),
                "data_points": len(df)
            }
        }

# ===========================
# Gestionnaire Principal Data Layer
# ===========================

class DataLayerManager:
    """Gestionnaire principal de la couche de données"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.ml_engine = MLAnalyticsEngine(self.metrics_collector)
        self.predictive_analyzer = PredictiveAnalyzer(
            self.metrics_collector, self.ml_engine
        )
        
        # Tâches de fond
        self.background_tasks: List[asyncio.Task] = []
        
        logger.info("Gestionnaire Data Layer initialisé")
    
    async def start(self):
        """Démarre tous les services de la couche de données"""
        # Configuration des sources par défaut
        await self._setup_default_sources()
        
        # Démarrage des tâches de fond
        self.background_tasks.append(
            asyncio.create_task(self._periodic_model_training())
        )
        self.background_tasks.append(
            asyncio.create_task(self._periodic_anomaly_detection())
        )
        
        logger.info("Services Data Layer démarrés")
    
    async def stop(self):
        """Arrête tous les services"""
        # Arrêt des tâches de fond
        for task in self.background_tasks:
            task.cancel()
        
        # Arrêt de la collection
        for source_id in list(self.metrics_collector.collection_tasks.keys()):
            await self.metrics_collector.stop_collection(source_id)
        
        logger.info("Services Data Layer arrêtés")
    
    async def _setup_default_sources(self):
        """Configure les sources de données par défaut"""
        default_sources = [
            {
                "id": "prometheus_system",
                "type": "prometheus",
                "config": {
                    "type": "prometheus",
                    "url": "http://prometheus:9090",
                    "tenant_id": "default",
                    "interval_seconds": 30
                }
            },
            {
                "id": "application_metrics",
                "type": "application", 
                "config": {
                    "type": "application",
                    "tenant_id": "default",
                    "interval_seconds": 60
                }
            }
        ]
        
        for source in default_sources:
            self.metrics_collector.register_source(source["id"], source["config"])
            await self.metrics_collector.start_collection(source["id"])
    
    async def _periodic_model_training(self):
        """Entraînement périodique des modèles"""
        while True:
            try:
                await asyncio.sleep(ML_CONFIG["model_retrain_hours"] * 3600)
                
                # Entraînement pour tous les tenants actifs
                tenants = core_registry.list_tenants()
                
                for tenant_context in tenants:
                    if tenant_context.is_active:
                        await self._train_tenant_models(tenant_context.tenant_id)
                
                logger.info("Entraînement périodique des modèles terminé")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erreur entraînement périodique: {e}")
    
    async def _periodic_anomaly_detection(self):
        """Détection périodique d'anomalies"""
        while True:
            try:
                await asyncio.sleep(300)  # Toutes les 5 minutes
                
                # Détection pour tous les tenants actifs
                tenants = core_registry.list_tenants()
                
                for tenant_context in tenants:
                    if tenant_context.is_active:
                        await self._detect_tenant_anomalies(tenant_context.tenant_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erreur détection d'anomalies: {e}")
    
    async def _train_tenant_models(self, tenant_id: str):
        """Entraîne les modèles pour un tenant"""
        # Métriques principales pour les modèles
        key_metrics = [
            "cpu_usage_percent",
            "memory_usage_percent",
            "error_rate_percent",
            "response_time_ms",
            "incidents_per_minute"
        ]
        
        # Entraînement des détecteurs d'anomalies
        for metric in key_metrics:
            await self.ml_engine.train_anomaly_detector(tenant_id, metric)
        
        # Entraînement des modèles prédictifs
        predictive_targets = [
            ("cpu_usage_percent", ["memory_usage_percent", "throughput_requests_per_second"]),
            ("error_rate_percent", ["response_time_ms", "cpu_usage_percent"]),
            ("incidents_per_minute", ["error_rate_percent", "response_time_ms"])
        ]
        
        for target, features in predictive_targets:
            await self.ml_engine.train_predictive_model(tenant_id, target, features)
        
        logger.info(f"Modèles entraînés pour le tenant {tenant_id}")
    
    async def _detect_tenant_anomalies(self, tenant_id: str):
        """Détecte les anomalies pour un tenant"""
        metrics_to_check = [
            "cpu_usage_percent",
            "memory_usage_percent", 
            "error_rate_percent",
            "response_time_ms"
        ]
        
        total_anomalies = 0
        for metric in metrics_to_check:
            anomalies = await self.ml_engine.detect_anomalies(tenant_id, metric)
            total_anomalies += len(anomalies)
            
            # Log des anomalies critiques
            for anomaly in anomalies:
                if anomaly.anomaly_score < -0.5:  # Score très bas = anomalie forte
                    logger.warning(f"Anomalie critique détectée: {tenant_id}:{metric} "
                                 f"(score: {anomaly.anomaly_score:.3f})")
        
        if total_anomalies > 0:
            logger.info(f"Détecté {total_anomalies} anomalies pour le tenant {tenant_id}")
    
    # API publique
    
    async def get_tenant_analytics(self, tenant_id: str) -> Dict[str, Any]:
        """Récupère les analytics d'un tenant"""
        # Métriques récentes
        recent_metrics = self.metrics_collector.get_recent_metrics(tenant_id, hours=24)
        
        # Détection d'anomalies
        anomalies = []
        for metric_name in ["cpu_usage_percent", "memory_usage_percent", "error_rate_percent"]:
            metric_anomalies = await self.ml_engine.detect_anomalies(tenant_id, metric_name)
            anomalies.extend(metric_anomalies)
        
        # Prédictions
        incident_prediction = await self.predictive_analyzer.predict_incident_probability(tenant_id)
        cpu_prediction = await self.predictive_analyzer.predict_resource_usage(
            tenant_id, "cpu", 24
        )
        
        # Analyse de corrélation
        correlation_analysis = await self.ml_engine.perform_correlation_analysis(
            tenant_id, ["cpu_usage_percent", "memory_usage_percent", "error_rate_percent"]
        )
        
        return {
            "tenant_id": tenant_id,
            "metrics_summary": {
                "total_metrics": len(recent_metrics),
                "unique_metric_types": len(set(m.name for m in recent_metrics)),
                "time_range": "24h"
            },
            "anomalies": {
                "count": len(anomalies),
                "recent": [a.to_dict() for a in anomalies[-5:]]  # 5 plus récentes
            },
            "predictions": {
                "incident_probability": incident_prediction.to_dict(),
                "cpu_forecast": cpu_prediction.to_dict()
            },
            "correlations": {
                "strong_correlations": correlation_analysis.get_strong_correlations(),
                "analysis_date": correlation_analysis.analyzed_at.isoformat()
            }
        }

# ===========================
# Exports
# ===========================

__all__ = [
    "DataLayerManager",
    "MetricsCollector", 
    "MLAnalyticsEngine",
    "PredictiveAnalyzer",
    "Metric",
    "MetricSeries",
    "AnomalyDetection",
    "PredictionResult",
    "CorrelationAnalysis",
    "MetricType",
    "DataSourceType",
    "PredictionType",
    "AnalysisType"
]

logger.info("Module Data Layer chargé")
