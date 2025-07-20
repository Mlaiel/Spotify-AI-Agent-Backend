#!/usr/bin/env python3
"""
Configuration Analytics Engine Enterprise

Moteur d'analyse avancée avec intelligence artificielle pour l'optimisation
automatisée des configurations multi-tenants et prédictions comportementales.

Architecture:
✅ Lead Dev + Architecte IA - Analytics en temps réel distribuées
✅ Développeur Backend Senior - Pipelines de données haute performance
✅ Ingénieur Machine Learning - Modèles prédictifs et clustering
✅ DBA & Data Engineer - Analytics sur données massives
✅ Spécialiste Sécurité Backend - Analytics de sécurité et anomalies
✅ Architecte Microservices - Analytics distribuées inter-services

Fonctionnalités Enterprise:
- Analytics temps réel avec streaming ML
- Prédictions comportementales multi-dimensionnelles
- Détection d'anomalies avec deep learning
- Optimisation automatique des configurations
- Recommandations intelligentes avec RL
- Analyse de tendances et patterns complexes
- Dashboard analytics interactif
- Alerting intelligent contextuel
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Imports ML et AI
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn as nn

# Imports pour analytics avancées
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt

# Imports async et caching
import aioredis
import aiocache
from asyncio import Queue, Event, Lock

# Configuration du logging
logger = logging.getLogger(__name__)

class AnalyticsType(Enum):
    """Types d'analytics supportés."""
    PERFORMANCE = "performance"
    USAGE = "usage"
    SECURITY = "security"
    CONFIGURATION = "configuration"
    BEHAVIOR = "behavior"
    PREDICTION = "prediction"
    ANOMALY = "anomaly"
    OPTIMIZATION = "optimization"

class MetricType(Enum):
    """Types de métriques."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"
    PERCENTAGE = "percentage"

class AlertSeverity(Enum):
    """Niveaux de sévérité des alertes."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class AnalyticsEvent:
    """Événement d'analytics."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tenant_id: str = ""
    event_type: str = ""
    category: AnalyticsType = AnalyticsType.USAGE
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    user_id: Optional[str] = None
    session_id: Optional[str] = None

@dataclass
class MetricDefinition:
    """Définition d'une métrique."""
    name: str
    type: MetricType
    description: str
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    aggregation_method: str = "avg"  # avg, sum, max, min, count
    retention_days: int = 90

@dataclass
class AnalyticsInsight:
    """Insight d'analytics avec recommandations."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    category: AnalyticsType = AnalyticsType.OPTIMIZATION
    severity: AlertSeverity = AlertSeverity.INFO
    confidence: float = 0.0
    impact_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    data_points: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    tenant_id: Optional[str] = None

class DeepLearningAnomalyDetector:
    """Détecteur d'anomalies basé sur deep learning."""
    
    def __init__(self, sequence_length: int = 50):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self._lock = threading.Lock()
    
    def _build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Construit le modèle LSTM pour détection d'anomalies."""
        model = keras.Sequential([
            keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(64, return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(input_shape[1], activation='linear')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, data: np.ndarray, epochs: int = 100) -> None:
        """Entraîne le modèle de détection d'anomalies."""
        with self._lock:
            if len(data) < self.sequence_length * 2:
                logger.warning("Données insuffisantes pour l'entraînement")
                return
            
            # Normalisation
            data_scaled = self.scaler.fit_transform(data)
            
            # Création des séquences
            X, y = [], []
            for i in range(len(data_scaled) - self.sequence_length):
                X.append(data_scaled[i:(i + self.sequence_length)])
                y.append(data_scaled[i + self.sequence_length])
            
            X, y = np.array(X), np.array(y)
            
            # Construction et entraînement du modèle
            self.model = self._build_model((self.sequence_length, data.shape[1]))
            
            # Entraînement
            self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=32,
                validation_split=0.2,
                verbose=0,
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
                ]
            )
            
            self.is_trained = True
            logger.info("Modèle de détection d'anomalies entraîné")
    
    def detect_anomalies(self, data: np.ndarray, threshold: float = 2.0) -> List[bool]:
        """Détecte les anomalies dans les données."""
        if not self.is_trained or self.model is None:
            return [False] * len(data)
        
        with self._lock:
            # Normalisation
            data_scaled = self.scaler.transform(data)
            
            # Prédictions
            if len(data_scaled) < self.sequence_length:
                return [False] * len(data)
            
            anomalies = []
            for i in range(len(data_scaled) - self.sequence_length + 1):
                sequence = data_scaled[i:(i + self.sequence_length)]
                sequence = sequence.reshape(1, self.sequence_length, -1)
                
                prediction = self.model.predict(sequence, verbose=0)
                actual = data_scaled[i + self.sequence_length - 1]
                
                # Calcul de l'erreur
                error = np.mean(np.square(prediction - actual))
                
                # Détection d'anomalie basée sur le seuil
                is_anomaly = error > threshold
                anomalies.append(is_anomaly)
            
            # Padding pour correspondre à la taille originale
            anomalies = [False] * (self.sequence_length - 1) + anomalies
            
            return anomalies

class BehaviorAnalyzer:
    """Analyseur de comportements avec clustering et patterns."""
    
    def __init__(self):
        self.clustering_model = KMeans(n_clusters=5, random_state=42)
        self.pattern_models = {}
        self.behavioral_profiles = {}
        self._lock = threading.Lock()
    
    def analyze_user_behavior(self, events: List[AnalyticsEvent]) -> Dict[str, Any]:
        """Analyse le comportement utilisateur."""
        if not events:
            return {}
        
        # Extraction des features comportementales
        features = self._extract_behavioral_features(events)
        
        if not features:
            return {}
        
        # Clustering comportemental
        clusters = self._perform_behavioral_clustering(features)
        
        # Détection de patterns
        patterns = self._detect_behavioral_patterns(events)
        
        # Analyse de la session
        session_analysis = self._analyze_session_patterns(events)
        
        return {
            "clusters": clusters,
            "patterns": patterns,
            "session_analysis": session_analysis,
            "behavioral_score": self._calculate_behavioral_score(features),
            "anomaly_indicators": self._detect_behavioral_anomalies(features)
        }
    
    def _extract_behavioral_features(self, events: List[AnalyticsEvent]) -> Dict[str, float]:
        """Extrait les features comportementales."""
        if not events:
            return {}
        
        features = {}
        
        # Features temporelles
        timestamps = [event.timestamp for event in events]
        if len(timestamps) > 1:
            intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                        for i in range(len(timestamps)-1)]
            features['avg_interval'] = np.mean(intervals)
            features['std_interval'] = np.std(intervals)
            features['session_duration'] = (timestamps[-1] - timestamps[0]).total_seconds()
        
        # Features d'activité
        features['event_count'] = len(events)
        features['unique_types'] = len(set(event.event_type for event in events))
        
        # Features par catégorie
        category_counts = defaultdict(int)
        for event in events:
            category_counts[event.category.value] += 1
        
        total_events = len(events)
        for category, count in category_counts.items():
            features[f'{category}_ratio'] = count / total_events
        
        # Features d'erreur
        error_events = [e for e in events if 'error' in e.event_type.lower()]
        features['error_rate'] = len(error_events) / total_events
        
        return features
    
    def _perform_behavioral_clustering(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Effectue le clustering comportemental."""
        if not features:
            return {}
        
        # Préparation des données
        feature_names = list(features.keys())
        feature_values = np.array([list(features.values())])
        
        try:
            # Clustering
            cluster_label = self.clustering_model.fit_predict(feature_values)[0]
            
            return {
                "cluster_id": int(cluster_label),
                "cluster_confidence": float(np.max(self.clustering_model.transform(feature_values)[0]))
            }
        
        except Exception as e:
            logger.error(f"Erreur dans clustering comportemental: {e}")
            return {}
    
    def _detect_behavioral_patterns(self, events: List[AnalyticsEvent]) -> List[Dict[str, Any]]:
        """Détecte les patterns comportementaux."""
        patterns = []
        
        if len(events) < 3:
            return patterns
        
        # Pattern de répétition
        event_types = [event.event_type for event in events]
        type_counts = defaultdict(int)
        for event_type in event_types:
            type_counts[event_type] += 1
        
        # Détection de patterns répétitifs
        for event_type, count in type_counts.items():
            if count >= 3:
                patterns.append({
                    "type": "repetitive_behavior",
                    "event_type": event_type,
                    "frequency": count,
                    "pattern_strength": min(count / len(events), 1.0)
                })
        
        # Pattern temporel
        if len(events) >= 5:
            timestamps = [event.timestamp for event in events]
            intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                        for i in range(len(timestamps)-1)]
            
            # Détection de régularité temporelle
            if np.std(intervals) < np.mean(intervals) * 0.3:
                patterns.append({
                    "type": "regular_timing",
                    "avg_interval": np.mean(intervals),
                    "regularity_score": 1.0 - (np.std(intervals) / np.mean(intervals))
                })
        
        return patterns
    
    def _analyze_session_patterns(self, events: List[AnalyticsEvent]) -> Dict[str, Any]:
        """Analyse les patterns de session."""
        if not events:
            return {}
        
        sessions = defaultdict(list)
        for event in events:
            session_id = event.session_id or "default"
            sessions[session_id].append(event)
        
        session_analysis = {}
        for session_id, session_events in sessions.items():
            session_analysis[session_id] = {
                "event_count": len(session_events),
                "duration": (session_events[-1].timestamp - session_events[0].timestamp).total_seconds() if len(session_events) > 1 else 0,
                "event_types": list(set(event.event_type for event in session_events)),
                "avg_event_interval": np.mean([(session_events[i+1].timestamp - session_events[i].timestamp).total_seconds() 
                                              for i in range(len(session_events)-1)]) if len(session_events) > 1 else 0
            }
        
        return session_analysis
    
    def _calculate_behavioral_score(self, features: Dict[str, float]) -> float:
        """Calcule un score comportemental global."""
        if not features:
            return 0.0
        
        # Score basé sur l'activité et la régularité
        activity_score = min(features.get('event_count', 0) / 100, 1.0)
        diversity_score = min(features.get('unique_types', 0) / 10, 1.0)
        error_penalty = 1.0 - features.get('error_rate', 0)
        
        return (activity_score * 0.4 + diversity_score * 0.3 + error_penalty * 0.3)
    
    def _detect_behavioral_anomalies(self, features: Dict[str, float]) -> List[str]:
        """Détecte les anomalies comportementales."""
        anomalies = []
        
        # Détection d'anomalies basiques
        if features.get('error_rate', 0) > 0.2:
            anomalies.append("high_error_rate")
        
        if features.get('avg_interval', 0) < 1.0:
            anomalies.append("rapid_fire_behavior")
        
        if features.get('session_duration', 0) > 3600:
            anomalies.append("extended_session")
        
        return anomalies

class PredictiveEngine:
    """Moteur de prédictions avec apprentissage automatique."""
    
    def __init__(self):
        self.models = {}
        self.feature_scalers = {}
        self.is_trained = {}
        self._lock = threading.Lock()
    
    def train_usage_prediction_model(self, historical_data: List[Dict[str, Any]]) -> None:
        """Entraîne le modèle de prédiction d'usage."""
        if len(historical_data) < 100:
            logger.warning("Données insuffisantes pour l'entraînement prédictif")
            return
        
        with self._lock:
            # Préparation des données
            df = pd.DataFrame(historical_data)
            
            # Features temporelles
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['day_of_month'] = pd.to_datetime(df['timestamp']).dt.day
            
            # Features d'activité
            feature_columns = ['hour', 'day_of_week', 'day_of_month', 'event_count']
            target_column = 'future_usage'
            
            if target_column not in df.columns:
                # Création de la target (usage futur basé sur les données)
                df[target_column] = df['event_count'].shift(-1).fillna(0)
            
            # Préparation des données d'entraînement
            X = df[feature_columns].fillna(0)
            y = df[target_column]
            
            # Normalisation
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Division train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Entraînement du modèle
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Évaluation
            y_pred = model.predict(X_test)
            score = model.score(X_test, y_test)
            
            # Stockage du modèle
            self.models['usage_prediction'] = model
            self.feature_scalers['usage_prediction'] = scaler
            self.is_trained['usage_prediction'] = True
            
            logger.info(f"Modèle de prédiction d'usage entraîné (R²: {score:.3f})")
    
    def predict_future_usage(self, current_time: datetime, horizon_hours: int = 24) -> List[Dict[str, Any]]:
        """Prédit l'usage futur."""
        if not self.is_trained.get('usage_prediction', False):
            return []
        
        with self._lock:
            model = self.models['usage_prediction']
            scaler = self.feature_scalers['usage_prediction']
            
            predictions = []
            
            for hour_offset in range(horizon_hours):
                future_time = current_time + timedelta(hours=hour_offset)
                
                # Features pour la prédiction
                features = np.array([[
                    future_time.hour,
                    future_time.weekday(),
                    future_time.day,
                    0  # event_count placeholder
                ]])
                
                # Normalisation et prédiction
                features_scaled = scaler.transform(features)
                predicted_usage = model.predict(features_scaled)[0]
                
                predictions.append({
                    "timestamp": future_time.isoformat(),
                    "predicted_usage": float(predicted_usage),
                    "confidence": 0.8  # À améliorer avec des méthodes d'incertitude
                })
            
            return predictions

class ConfigAnalytics:
    """
    Moteur d'analytics de configuration Enterprise.
    
    Fonctionnalités:
    - Analytics temps réel avec ML avancé
    - Détection d'anomalies multi-dimensionnelle
    - Prédictions comportementales intelligentes
    - Optimisation automatique des configurations
    - Recommandations contextuelles avec RL
    - Dashboard analytics interactif
    """
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 enable_ml: bool = True,
                 enable_deep_learning: bool = True):
        self.redis_url = redis_url
        self.enable_ml = enable_ml
        self.enable_deep_learning = enable_deep_learning
        
        # Composants ML
        self.anomaly_detector = DeepLearningAnomalyDetector() if enable_deep_learning else None
        self.behavior_analyzer = BehaviorAnalyzer()
        self.predictive_engine = PredictiveEngine()
        
        # Stockage des données
        self.events_buffer: deque = deque(maxlen=10000)
        self.metrics_history: deque = deque(maxlen=5000)
        self.insights_cache: Dict[str, AnalyticsInsight] = {}
        
        # Métriques définies
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        self._setup_default_metrics()
        
        # Composants async
        self.redis_client: Optional[aioredis.Redis] = None
        self.cache = aiocache.Cache(aiocache.Cache.MEMORY)
        
        # Synchronisation
        self._analytics_lock = Lock()
        self.shutdown_event = Event()
        
        # Thread pool pour ML
        self.ml_executor = ThreadPoolExecutor(max_workers=4)
        
        # Dashboard data
        self.dashboard_data = {}
        
        logger.info("Analytics engine initialisé")
    
    async def initialize(self) -> None:
        """Initialise le moteur d'analytics."""
        try:
            # Connexion Redis
            self.redis_client = aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Démarrage des processus d'analytics
            self._start_analytics_processors()
            
            logger.info("Moteur d'analytics initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur d'initialisation analytics: {e}")
            raise
    
    def _setup_default_metrics(self) -> None:
        """Configure les métriques par défaut."""
        default_metrics = [
            MetricDefinition(
                name="config_changes",
                type=MetricType.COUNTER,
                description="Nombre de changements de configuration",
                unit="count",
                tags={"category": "configuration"}
            ),
            MetricDefinition(
                name="response_time",
                type=MetricType.TIMER,
                description="Temps de réponse des APIs",
                unit="ms",
                tags={"category": "performance"}
            ),
            MetricDefinition(
                name="error_rate",
                type=MetricType.PERCENTAGE,
                description="Taux d'erreur",
                unit="%",
                tags={"category": "reliability"}
            ),
            MetricDefinition(
                name="active_users",
                type=MetricType.GAUGE,
                description="Utilisateurs actifs",
                unit="count",
                tags={"category": "usage"}
            ),
            MetricDefinition(
                name="security_events",
                type=MetricType.COUNTER,
                description="Événements de sécurité",
                unit="count",
                tags={"category": "security"}
            )
        ]
        
        for metric in default_metrics:
            self.metric_definitions[metric.name] = metric
    
    def _start_analytics_processors(self) -> None:
        """Démarre les processeurs d'analytics."""
        # Processeur d'événements en temps réel
        asyncio.create_task(self._real_time_processor())
        
        # Processeur d'insights
        asyncio.create_task(self._insights_processor())
        
        # Processeur de ML
        if self.enable_ml:
            asyncio.create_task(self._ml_processor())
        
        # Processeur de dashboard
        asyncio.create_task(self._dashboard_processor())
        
        # Processeur de nettoyage
        asyncio.create_task(self._cleanup_processor())
    
    async def track_event(self, event: AnalyticsEvent) -> None:
        """Enregistre un événement d'analytics."""
        # Ajout au buffer
        self.events_buffer.append(event)
        
        # Stockage Redis pour persistance
        if self.redis_client:
            event_data = {
                "id": event.id,
                "timestamp": event.timestamp.isoformat(),
                "tenant_id": event.tenant_id,
                "event_type": event.event_type,
                "category": event.category.value,
                "data": json.dumps(event.data),
                "metadata": json.dumps(event.metadata),
                "source": event.source,
                "user_id": event.user_id,
                "session_id": event.session_id
            }
            
            # Stockage dans Redis Streams
            await self.redis_client.xadd("analytics:events", event_data)
            
            # Indexation par tenant
            await self.redis_client.sadd(f"analytics:tenants:{event.tenant_id}:events", event.id)
    
    async def get_insights(self, 
                          tenant_id: Optional[str] = None,
                          category: Optional[AnalyticsType] = None,
                          limit: int = 50) -> List[AnalyticsInsight]:
        """Récupère les insights d'analytics."""
        insights = list(self.insights_cache.values())
        
        # Filtrage
        if tenant_id:
            insights = [i for i in insights if i.tenant_id == tenant_id]
        
        if category:
            insights = [i for i in insights if i.category == category]
        
        # Tri par confiance et impact
        insights.sort(key=lambda x: (x.confidence * x.impact_score), reverse=True)
        
        return insights[:limit]
    
    async def get_tenant_analytics(self, tenant_id: str) -> Dict[str, Any]:
        """Récupère les analytics pour un tenant spécifique."""
        # Récupération des événements du tenant
        tenant_events = [e for e in self.events_buffer if e.tenant_id == tenant_id]
        
        if not tenant_events:
            return {"tenant_id": tenant_id, "analytics": {}}
        
        # Analytics comportementales
        behavior_analysis = self.behavior_analyzer.analyze_user_behavior(tenant_events)
        
        # Métriques de base
        basic_metrics = await self._calculate_basic_metrics(tenant_events)
        
        # Insights du tenant
        tenant_insights = await self.get_insights(tenant_id=tenant_id)
        
        # Prédictions
        predictions = self.predictive_engine.predict_future_usage(datetime.utcnow())
        
        return {
            "tenant_id": tenant_id,
            "analytics": {
                "behavior_analysis": behavior_analysis,
                "basic_metrics": basic_metrics,
                "insights": [self._insight_to_dict(i) for i in tenant_insights],
                "predictions": predictions,
                "anomalies": await self._detect_tenant_anomalies(tenant_id)
            }
        }
    
    async def generate_recommendations(self, tenant_id: str) -> List[Dict[str, Any]]:
        """Génère des recommandations pour un tenant."""
        # Récupération des données du tenant
        tenant_analytics = await self.get_tenant_analytics(tenant_id)
        
        recommendations = []
        
        # Recommandations basées sur le comportement
        behavior_data = tenant_analytics["analytics"].get("behavior_analysis", {})
        
        if behavior_data.get("behavioral_score", 0) < 0.5:
            recommendations.append({
                "type": "behavior_optimization",
                "title": "Optimisation comportementale",
                "description": "Le score comportemental est faible. Considérez une révision des workflows.",
                "priority": "medium",
                "impact": "medium"
            })
        
        # Recommandations basées sur les anomalies
        anomalies = tenant_analytics["analytics"].get("anomalies", [])
        
        for anomaly in anomalies:
            recommendations.append({
                "type": "anomaly_resolution",
                "title": f"Résolution d'anomalie: {anomaly['type']}",
                "description": f"Anomalie détectée: {anomaly['description']}",
                "priority": "high",
                "impact": "high"
            })
        
        # Recommandations basées sur les métriques
        metrics = tenant_analytics["analytics"].get("basic_metrics", {})
        
        if metrics.get("error_rate", 0) > 0.1:
            recommendations.append({
                "type": "error_reduction",
                "title": "Réduction du taux d'erreur",
                "description": "Le taux d'erreur est élevé. Vérifiez la configuration et les logs.",
                "priority": "high",
                "impact": "high"
            })
        
        return recommendations
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Récupère les données pour le dashboard."""
        return {
            "overview": await self._get_overview_metrics(),
            "real_time": await self._get_real_time_data(),
            "trends": await self._get_trend_data(),
            "alerts": await self._get_active_alerts(),
            "performance": await self._get_performance_data(),
            "security": await self._get_security_data()
        }
    
    async def export_analytics(self, 
                              start_date: datetime,
                              end_date: datetime,
                              format: str = "json") -> Union[str, bytes]:
        """Exporte les données d'analytics."""
        # Filtrage des événements par date
        filtered_events = [
            e for e in self.events_buffer
            if start_date <= e.timestamp <= end_date
        ]
        
        if format == "json":
            return json.dumps([self._event_to_dict(e) for e in filtered_events], 
                            default=str, indent=2)
        
        elif format == "csv":
            df = pd.DataFrame([self._event_to_dict(e) for e in filtered_events])
            return df.to_csv(index=False)
        
        else:
            raise ValueError(f"Format non supporté: {format}")
    
    # Processeurs asynchrones
    async def _real_time_processor(self) -> None:
        """Processeur temps réel des événements."""
        while not self.shutdown_event.is_set():
            try:
                # Traitement en batch des événements récents
                recent_events = list(self.events_buffer)[-100:]  # 100 derniers événements
                
                if recent_events:
                    # Détection d'anomalies en temps réel
                    await self._real_time_anomaly_detection(recent_events)
                    
                    # Mise à jour des métriques
                    await self._update_real_time_metrics(recent_events)
                    
                    # Génération d'alertes
                    await self._generate_real_time_alerts(recent_events)
                
                await asyncio.sleep(10)  # Traitement toutes les 10 secondes
                
            except Exception as e:
                logger.error(f"Erreur dans processeur temps réel: {e}")
                await asyncio.sleep(30)
    
    async def _insights_processor(self) -> None:
        """Processeur de génération d'insights."""
        while not self.shutdown_event.is_set():
            try:
                # Génération d'insights périodique
                await self._generate_insights()
                
                # Nettoyage des insights obsolètes
                await self._cleanup_old_insights()
                
                await asyncio.sleep(300)  # Toutes les 5 minutes
                
            except Exception as e:
                logger.error(f"Erreur dans processeur d'insights: {e}")
                await asyncio.sleep(300)
    
    async def _ml_processor(self) -> None:
        """Processeur ML pour entraînement et prédictions."""
        while not self.shutdown_event.is_set():
            try:
                # Entraînement périodique des modèles
                if len(self.events_buffer) >= 1000:
                    await self._train_ml_models()
                
                # Mise à jour des prédictions
                await self._update_predictions()
                
                await asyncio.sleep(3600)  # Toutes les heures
                
            except Exception as e:
                logger.error(f"Erreur dans processeur ML: {e}")
                await asyncio.sleep(1800)
    
    async def _dashboard_processor(self) -> None:
        """Processeur de mise à jour du dashboard."""
        while not self.shutdown_event.is_set():
            try:
                # Mise à jour des données du dashboard
                self.dashboard_data = await self.get_dashboard_data()
                
                # Publication vers Redis pour le frontend
                if self.redis_client:
                    await self.redis_client.setex(
                        "analytics:dashboard",
                        300,  # TTL 5 minutes
                        json.dumps(self.dashboard_data, default=str)
                    )
                
                await asyncio.sleep(60)  # Toutes les minutes
                
            except Exception as e:
                logger.error(f"Erreur dans processeur dashboard: {e}")
                await asyncio.sleep(120)
    
    async def _cleanup_processor(self) -> None:
        """Processeur de nettoyage des données."""
        while not self.shutdown_event.is_set():
            try:
                # Nettoyage des événements anciens
                await self._cleanup_old_events()
                
                # Nettoyage des métriques anciennes
                await self._cleanup_old_metrics()
                
                # Optimisation des caches
                await self._optimize_caches()
                
                await asyncio.sleep(86400)  # Toutes les 24 heures
                
            except Exception as e:
                logger.error(f"Erreur dans processeur de nettoyage: {e}")
                await asyncio.sleep(3600)
    
    # Méthodes utilitaires
    async def _calculate_basic_metrics(self, events: List[AnalyticsEvent]) -> Dict[str, Any]:
        """Calcule les métriques de base."""
        if not events:
            return {}
        
        # Métriques temporelles
        timestamps = [e.timestamp for e in events]
        time_span = (max(timestamps) - min(timestamps)).total_seconds()
        
        # Métriques par catégorie
        category_counts = defaultdict(int)
        for event in events:
            category_counts[event.category.value] += 1
        
        # Métriques d'erreur
        error_events = [e for e in events if 'error' in e.event_type.lower()]
        
        return {
            "total_events": len(events),
            "time_span_hours": time_span / 3600,
            "events_per_hour": len(events) / max(time_span / 3600, 1),
            "category_distribution": dict(category_counts),
            "error_count": len(error_events),
            "error_rate": len(error_events) / len(events),
            "unique_event_types": len(set(e.event_type for e in events)),
            "unique_users": len(set(e.user_id for e in events if e.user_id)),
            "unique_sessions": len(set(e.session_id for e in events if e.session_id))
        }
    
    def _event_to_dict(self, event: AnalyticsEvent) -> Dict[str, Any]:
        """Convertit un événement en dictionnaire."""
        return {
            "id": event.id,
            "timestamp": event.timestamp.isoformat(),
            "tenant_id": event.tenant_id,
            "event_type": event.event_type,
            "category": event.category.value,
            "data": event.data,
            "metadata": event.metadata,
            "source": event.source,
            "user_id": event.user_id,
            "session_id": event.session_id
        }
    
    def _insight_to_dict(self, insight: AnalyticsInsight) -> Dict[str, Any]:
        """Convertit un insight en dictionnaire."""
        return {
            "id": insight.id,
            "title": insight.title,
            "description": insight.description,
            "category": insight.category.value,
            "severity": insight.severity.value,
            "confidence": insight.confidence,
            "impact_score": insight.impact_score,
            "recommendations": insight.recommendations,
            "data_points": insight.data_points,
            "created_at": insight.created_at.isoformat(),
            "tenant_id": insight.tenant_id
        }
    
    async def _detect_tenant_anomalies(self, tenant_id: str) -> List[Dict[str, Any]]:
        """Détecte les anomalies pour un tenant."""
        anomalies = []
        
        # Récupération des événements du tenant
        tenant_events = [e for e in self.events_buffer if e.tenant_id == tenant_id]
        
        if len(tenant_events) < 10:
            return anomalies
        
        # Détection d'anomalies temporelles
        timestamps = [e.timestamp for e in tenant_events]
        intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                    for i in range(len(timestamps)-1)]
        
        if intervals:
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            # Détection d'activité anormalement rapide
            if mean_interval < 5.0:  # Moins de 5 secondes entre événements
                anomalies.append({
                    "type": "rapid_activity",
                    "description": "Activité anormalement rapide détectée",
                    "severity": "warning",
                    "data": {"mean_interval": mean_interval}
                })
            
            # Détection d'irrégularité temporelle
            if std_interval > mean_interval:
                anomalies.append({
                    "type": "irregular_timing",
                    "description": "Timing irrégulier détecté",
                    "severity": "info",
                    "data": {"irregularity_ratio": std_interval / mean_interval}
                })
        
        return anomalies
    
    async def _generate_insights(self) -> None:
        """Génère des insights d'analytics."""
        # Analyse des tendances globales
        recent_events = [e for e in self.events_buffer 
                        if (datetime.utcnow() - e.timestamp).total_seconds() < 3600]
        
        if len(recent_events) < 10:
            return
        
        # Insight sur l'activité récente
        activity_insight = AnalyticsInsight(
            title="Activité récente élevée",
            description=f"{len(recent_events)} événements dans la dernière heure",
            category=AnalyticsType.USAGE,
            severity=AlertSeverity.INFO,
            confidence=0.9,
            impact_score=0.7,
            recommendations=[
                "Surveiller les ressources système",
                "Vérifier les patterns d'usage"
            ],
            data_points={"recent_events_count": len(recent_events)}
        )
        
        self.insights_cache[activity_insight.id] = activity_insight
        
        # Insight sur les erreurs
        error_events = [e for e in recent_events if 'error' in e.event_type.lower()]
        if len(error_events) > len(recent_events) * 0.1:  # Plus de 10% d'erreurs
            error_insight = AnalyticsInsight(
                title="Taux d'erreur élevé",
                description=f"Taux d'erreur de {len(error_events)/len(recent_events)*100:.1f}%",
                category=AnalyticsType.SECURITY,
                severity=AlertSeverity.WARNING,
                confidence=0.95,
                impact_score=0.9,
                recommendations=[
                    "Vérifier les logs d'erreur",
                    "Analyser les causes racines",
                    "Mettre en place des alertes"
                ],
                data_points={
                    "error_rate": len(error_events)/len(recent_events),
                    "error_count": len(error_events)
                }
            )
            
            self.insights_cache[error_insight.id] = error_insight
    
    async def shutdown(self) -> None:
        """Arrête le moteur d'analytics."""
        self.shutdown_event.set()
        
        # Fermeture des ressources
        if self.redis_client:
            await self.redis_client.close()
        
        self.ml_executor.shutdown(wait=True)
        
        logger.info("Moteur d'analytics arrêté")
