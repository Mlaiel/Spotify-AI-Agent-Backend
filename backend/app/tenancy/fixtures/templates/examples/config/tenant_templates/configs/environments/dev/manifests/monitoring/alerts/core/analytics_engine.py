"""
Ultra-Advanced Analytics Engine - Enterprise-Grade Real-Time Analytics System
===========================================================================

Ce module fournit un moteur d'analytics avancé avec intelligence artificielle,
analyse prédictive, métriques temps réel, machine learning automatique et
visualisation avancée pour des environnements multi-tenant à haute performance.

Fonctionnalités Principales:
- Analytics temps réel avec streaming de données
- Machine learning prédictif avec auto-tuning
- Métriques avancées et KPI intelligents
- Analyse de tendances et détection d'anomalies
- Visualisation interactive et dashboards
- Rapports automatisés et alerting proactif
- Analyse comportementale et patterns
- Optimisation des performances automatique

Architecture Enterprise:
- Pipeline de données distribué avec Apache Kafka
- Moteur ML avec TensorFlow et scikit-learn
- Base de données time-series (InfluxDB/TimescaleDB)
- Cache Redis pour analytics temps réel
- API GraphQL pour queries complexes
- WebSocket pour streaming en temps réel
- Système de backup et archivage
- Sécurité et authentification avancée

Version: 5.0.0
Auteur: Fahed Mlaiel (Lead Dev + Architecte IA)
Architecture: Event-Driven Microservices avec ML Pipeline
"""

import asyncio
import logging
import time
import uuid
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import (
    Dict, List, Optional, Any, Callable, Union, Tuple, Set,
    Protocol, TypeVar, Generic, AsyncIterator, NamedTuple
)
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum, auto
from collections import defaultdict, deque, Counter
import statistics
import redis.asyncio as redis
import asyncpg
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from prometheus_client import Counter as PrometheusCounter, Histogram, Gauge
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configuration du logging structuré
logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types de métriques supportés"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    RATE = "rate"
    CUSTOM = "custom"


class AnalyticsLevel(Enum):
    """Niveaux d'analytics"""
    BASIC = "basic"
    ADVANCED = "advanced"
    PREDICTIVE = "predictive"
    AI_POWERED = "ai_powered"


class TimeGranularity(Enum):
    """Granularité temporelle"""
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


@dataclass
class Metric:
    """Métrique avec métadonnées enrichies"""
    id: str
    name: str
    value: Union[float, int]
    metric_type: MetricType
    labels: Dict[str, str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tenant_id: str = ""
    source: str = "unknown"
    unit: str = ""
    description: str = ""
    tags: Set[str] = field(default_factory=set)


@dataclass
class TimeSeriesData:
    """Données de série temporelle"""
    metric_name: str
    timestamps: List[datetime]
    values: List[float]
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyticsQuery:
    """Requête d'analytics complexe"""
    id: str
    name: str
    query: str
    parameters: Dict[str, Any]
    time_range: Tuple[datetime, datetime]
    granularity: TimeGranularity
    filters: Dict[str, Any] = field(default_factory=dict)
    aggregations: List[str] = field(default_factory=list)
    tenant_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AnalyticsReport:
    """Rapport d'analytics complet"""
    id: str
    title: str
    description: str
    metrics: List[Metric]
    charts: List[Dict[str, Any]]
    insights: List[str]
    recommendations: List[str]
    kpis: Dict[str, float]
    anomalies: List[Dict[str, Any]]
    predictions: List[Dict[str, Any]]
    time_range: Tuple[datetime, datetime]
    generated_at: datetime = field(default_factory=datetime.utcnow)
    tenant_id: str = ""


@dataclass
class MLModel:
    """Modèle de machine learning"""
    id: str
    name: str
    model_type: str
    algorithm: str
    features: List[str]
    target: str
    accuracy: float
    trained_at: datetime
    version: str = "1.0"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TimeSeriesAnalyzer:
    """Analyseur de séries temporelles avec ML"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.anomaly_detectors = {}
        
    async def analyze_trend(self, data: TimeSeriesData) -> Dict[str, Any]:
        """Analyse les tendances dans une série temporelle"""
        try:
            if len(data.values) < 3:
                return {'trend': 'insufficient_data', 'confidence': 0.0}
            
            # Conversion en numpy arrays
            timestamps = np.array([ts.timestamp() for ts in data.timestamps])
            values = np.array(data.values)
            
            # Calcul de la tendance linéaire
            coeffs = np.polyfit(timestamps, values, 1)
            slope = coeffs[0]
            
            # Classification de la tendance
            threshold = np.std(values) * 0.1
            if abs(slope) < threshold:
                trend = 'stable'
            elif slope > 0:
                trend = 'increasing'
            else:
                trend = 'decreasing'
            
            # Calcul du coefficient de corrélation pour la confiance
            correlation = np.corrcoef(timestamps, values)[0, 1]
            confidence = abs(correlation) if not np.isnan(correlation) else 0.0
            
            # Détection de saisonnalité basique
            seasonality = await self.detect_seasonality(data)
            
            return {
                'trend': trend,
                'slope': float(slope),
                'confidence': float(confidence),
                'seasonality': seasonality,
                'volatility': float(np.std(values)),
                'mean': float(np.mean(values)),
                'median': float(np.median(values))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")
            return {'trend': 'error', 'confidence': 0.0, 'error': str(e)}
    
    async def detect_seasonality(self, data: TimeSeriesData) -> Dict[str, Any]:
        """Détecte la saisonnalité dans les données"""
        try:
            if len(data.values) < 10:
                return {'seasonal': False, 'period': None}
            
            values = np.array(data.values)
            
            # Autocorrélation pour détecter les patterns
            max_lag = min(len(values) // 3, 100)
            autocorr = np.correlate(values, values, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Recherche de pics dans l'autocorrélation
            if len(autocorr) > max_lag:
                peaks = []
                for i in range(2, max_lag):
                    if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                        if autocorr[i] > np.mean(autocorr) + np.std(autocorr):
                            peaks.append(i)
                
                if peaks:
                    period = peaks[0]  # Premier pic significatif
                    strength = float(autocorr[period] / autocorr[0])
                    return {
                        'seasonal': True,
                        'period': period,
                        'strength': strength,
                        'detected_periods': peaks[:5]
                    }
            
            return {'seasonal': False, 'period': None}
            
        except Exception as e:
            logger.error(f"Error detecting seasonality: {e}")
            return {'seasonal': False, 'period': None, 'error': str(e)}
    
    async def detect_anomalies(self, data: TimeSeriesData) -> List[Dict[str, Any]]:
        """Détecte les anomalies avec Isolation Forest"""
        try:
            if len(data.values) < 10:
                return []
            
            # Préparation des features
            values = np.array(data.values).reshape(-1, 1)
            
            # Utiliser ou créer un détecteur d'anomalies
            detector_key = f"{data.metric_name}_{data.labels.get('tenant_id', 'default')}"
            
            if detector_key not in self.anomaly_detectors:
                self.anomaly_detectors[detector_key] = IsolationForest(
                    contamination=0.1,
                    random_state=42
                )
                self.anomaly_detectors[detector_key].fit(values)
            
            detector = self.anomaly_detectors[detector_key]
            
            # Détection des anomalies
            anomaly_scores = detector.decision_function(values)
            predictions = detector.predict(values)
            
            anomalies = []
            for i, (score, pred) in enumerate(zip(anomaly_scores, predictions)):
                if pred == -1:  # Anomalie détectée
                    anomalies.append({
                        'timestamp': data.timestamps[i].isoformat(),
                        'value': float(data.values[i]),
                        'anomaly_score': float(score),
                        'severity': 'high' if score < -0.5 else 'medium'
                    })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []
    
    async def predict_values(self, data: TimeSeriesData, horizon: int = 10) -> List[Dict[str, Any]]:
        """Prédit les valeurs futures"""
        try:
            if len(data.values) < 20:
                return []
            
            # Préparation des données pour la prédiction
            values = np.array(data.values)
            timestamps = np.array([ts.timestamp() for ts in data.timestamps])
            
            # Création de features temporelles
            X = []
            y = []
            window_size = 5
            
            for i in range(window_size, len(values)):
                X.append(values[i-window_size:i])
                y.append(values[i])
            
            if len(X) < 10:
                return []
            
            X = np.array(X)
            y = np.array(y)
            
            # Entraînement d'un modèle simple
            model_key = f"{data.metric_name}_{data.labels.get('tenant_id', 'default')}"
            
            if model_key not in self.models:
                self.models[model_key] = RandomForestRegressor(
                    n_estimators=50,
                    random_state=42
                )
                
                # Entraînement
                self.models[model_key].fit(X, y)
            
            model = self.models[model_key]
            
            # Prédiction
            predictions = []
            last_window = values[-window_size:]
            last_timestamp = timestamps[-1]
            
            for i in range(horizon):
                pred_value = model.predict([last_window])[0]
                pred_timestamp = last_timestamp + (i + 1) * (timestamps[-1] - timestamps[-2])
                
                predictions.append({
                    'timestamp': datetime.fromtimestamp(pred_timestamp, tz=timezone.utc).isoformat(),
                    'predicted_value': float(pred_value),
                    'confidence': 0.8 - (i * 0.05)  # Décroissance de confiance
                })
                
                # Mise à jour de la fenêtre
                last_window = np.append(last_window[1:], pred_value)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting values: {e}")
            return []


class KPICalculator:
    """Calculateur de KPI avancé"""
    
    def __init__(self):
        self.kpi_definitions = {
            'availability': self.calculate_availability,
            'performance': self.calculate_performance,
            'error_rate': self.calculate_error_rate,
            'throughput': self.calculate_throughput,
            'latency': self.calculate_latency,
            'satisfaction': self.calculate_satisfaction,
            'efficiency': self.calculate_efficiency,
            'reliability': self.calculate_reliability
        }
    
    async def calculate_kpis(self, metrics: List[Metric], time_range: Tuple[datetime, datetime]) -> Dict[str, float]:
        """Calcule tous les KPI"""
        kpis = {}
        
        for kpi_name, calculator in self.kpi_definitions.items():
            try:
                value = await calculator(metrics, time_range)
                kpis[kpi_name] = value
            except Exception as e:
                logger.error(f"Error calculating KPI {kpi_name}: {e}")
                kpis[kpi_name] = 0.0
        
        # KPI global
        kpis['overall_health'] = np.mean(list(kpis.values()))
        
        return kpis
    
    async def calculate_availability(self, metrics: List[Metric], time_range: Tuple[datetime, datetime]) -> float:
        """Calcule la disponibilité"""
        uptime_metrics = [m for m in metrics if 'uptime' in m.name.lower() or 'availability' in m.name.lower()]
        
        if not uptime_metrics:
            return 100.0
        
        total_uptime = sum(m.value for m in uptime_metrics)
        total_time = len(uptime_metrics)
        
        return (total_uptime / total_time) * 100 if total_time > 0 else 100.0
    
    async def calculate_performance(self, metrics: List[Metric], time_range: Tuple[datetime, datetime]) -> float:
        """Calcule la performance"""
        perf_metrics = [m for m in metrics if 'performance' in m.name.lower() or 'response_time' in m.name.lower()]
        
        if not perf_metrics:
            return 100.0
        
        avg_performance = np.mean([m.value for m in perf_metrics])
        # Normaliser sur 100 (plus c'est bas, mieux c'est pour les temps de réponse)
        return max(0, 100 - avg_performance)
    
    async def calculate_error_rate(self, metrics: List[Metric], time_range: Tuple[datetime, datetime]) -> float:
        """Calcule le taux d'erreur"""
        error_metrics = [m for m in metrics if 'error' in m.name.lower() or 'fail' in m.name.lower()]
        total_metrics = [m for m in metrics if 'total' in m.name.lower() or 'count' in m.name.lower()]
        
        if not error_metrics or not total_metrics:
            return 0.0
        
        total_errors = sum(m.value for m in error_metrics)
        total_requests = sum(m.value for m in total_metrics)
        
        return (total_errors / total_requests) * 100 if total_requests > 0 else 0.0
    
    async def calculate_throughput(self, metrics: List[Metric], time_range: Tuple[datetime, datetime]) -> float:
        """Calcule le débit"""
        throughput_metrics = [m for m in metrics if 'throughput' in m.name.lower() or 'rps' in m.name.lower()]
        
        if not throughput_metrics:
            return 0.0
        
        return np.mean([m.value for m in throughput_metrics])
    
    async def calculate_latency(self, metrics: List[Metric], time_range: Tuple[datetime, datetime]) -> float:
        """Calcule la latence moyenne"""
        latency_metrics = [m for m in metrics if 'latency' in m.name.lower() or 'duration' in m.name.lower()]
        
        if not latency_metrics:
            return 0.0
        
        return np.mean([m.value for m in latency_metrics])
    
    async def calculate_satisfaction(self, metrics: List[Metric], time_range: Tuple[datetime, datetime]) -> float:
        """Calcule l'indice de satisfaction"""
        # Basé sur la performance, erreurs et disponibilité
        availability = await self.calculate_availability(metrics, time_range)
        performance = await self.calculate_performance(metrics, time_range)
        error_rate = await self.calculate_error_rate(metrics, time_range)
        
        # Satisfaction = moyenne pondérée
        satisfaction = (availability * 0.4 + performance * 0.4 + (100 - error_rate) * 0.2)
        return min(100, max(0, satisfaction))
    
    async def calculate_efficiency(self, metrics: List[Metric], time_range: Tuple[datetime, datetime]) -> float:
        """Calcule l'efficacité"""
        resource_metrics = [m for m in metrics if 'cpu' in m.name.lower() or 'memory' in m.name.lower()]
        
        if not resource_metrics:
            return 100.0
        
        avg_utilization = np.mean([m.value for m in resource_metrics])
        # Efficacité optimale autour de 70-80%
        optimal_range = (70, 80)
        
        if optimal_range[0] <= avg_utilization <= optimal_range[1]:
            return 100.0
        elif avg_utilization < optimal_range[0]:
            return avg_utilization / optimal_range[0] * 100
        else:
            return max(0, 100 - (avg_utilization - optimal_range[1]))
    
    async def calculate_reliability(self, metrics: List[Metric], time_range: Tuple[datetime, datetime]) -> float:
        """Calcule la fiabilité"""
        # Basé sur la cohérence des métriques
        all_values = [m.value for m in metrics if isinstance(m.value, (int, float))]
        
        if len(all_values) < 2:
            return 100.0
        
        coefficient_variation = np.std(all_values) / np.mean(all_values) if np.mean(all_values) > 0 else 0
        # Plus la variation est faible, plus c'est fiable
        reliability = max(0, 100 - (coefficient_variation * 100))
        
        return min(100, reliability)


class ChartGenerator:
    """Générateur de graphiques avancés"""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
    
    async def generate_time_series_chart(self, data: TimeSeriesData, title: str = "") -> Dict[str, Any]:
        """Génère un graphique de série temporelle"""
        try:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=data.timestamps,
                y=data.values,
                mode='lines+markers',
                name=data.metric_name,
                line=dict(color=self.color_palette[0]),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Timestamp: %{x}<br>' +
                             'Value: %{y}<br>' +
                             '<extra></extra>'
            ))
            
            fig.update_layout(
                title=title or f"Time Series: {data.metric_name}",
                xaxis_title="Time",
                yaxis_title="Value",
                hovermode='x unified',
                template='plotly_white'
            )
            
            return {
                'type': 'time_series',
                'config': fig.to_dict(),
                'data_points': len(data.values)
            }
            
        except Exception as e:
            logger.error(f"Error generating time series chart: {e}")
            return {'type': 'error', 'error': str(e)}
    
    async def generate_distribution_chart(self, metrics: List[Metric], metric_name: str) -> Dict[str, Any]:
        """Génère un graphique de distribution"""
        try:
            values = [m.value for m in metrics if m.name == metric_name and isinstance(m.value, (int, float))]
            
            if not values:
                return {'type': 'error', 'error': 'No data available'}
            
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=values,
                nbinsx=20,
                name=metric_name,
                marker_color=self.color_palette[1],
                opacity=0.7
            ))
            
            fig.update_layout(
                title=f"Distribution: {metric_name}",
                xaxis_title="Value",
                yaxis_title="Frequency",
                template='plotly_white'
            )
            
            return {
                'type': 'distribution',
                'config': fig.to_dict(),
                'statistics': {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating distribution chart: {e}")
            return {'type': 'error', 'error': str(e)}
    
    async def generate_correlation_heatmap(self, metrics: List[Metric]) -> Dict[str, Any]:
        """Génère une heatmap de corrélation"""
        try:
            # Grouper les métriques par nom
            metric_groups = defaultdict(list)
            for metric in metrics:
                if isinstance(metric.value, (int, float)):
                    metric_groups[metric.name].append(metric.value)
            
            if len(metric_groups) < 2:
                return {'type': 'error', 'error': 'Insufficient metrics for correlation'}
            
            # Créer un DataFrame
            min_length = min(len(values) for values in metric_groups.values())
            data_dict = {name: values[:min_length] for name, values in metric_groups.items()}
            df = pd.DataFrame(data_dict)
            
            # Calculer la corrélation
            correlation_matrix = df.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=correlation_matrix.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate='<b>%{y} vs %{x}</b><br>' +
                             'Correlation: %{z}<br>' +
                             '<extra></extra>'
            ))
            
            fig.update_layout(
                title="Metrics Correlation Heatmap",
                template='plotly_white'
            )
            
            return {
                'type': 'heatmap',
                'config': fig.to_dict(),
                'correlation_matrix': correlation_matrix.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error generating correlation heatmap: {e}")
            return {'type': 'error', 'error': str(e)}
    
    async def generate_kpi_dashboard(self, kpis: Dict[str, float]) -> Dict[str, Any]:
        """Génère un dashboard de KPI"""
        try:
            # Créer des sous-graphiques
            fig = make_subplots(
                rows=2, cols=4,
                subplot_titles=list(kpis.keys()),
                specs=[[{"type": "indicator"} for _ in range(4)] for _ in range(2)]
            )
            
            colors = ['green', 'blue', 'orange', 'red', 'purple', 'brown', 'pink', 'gray']
            
            for i, (kpi_name, value) in enumerate(kpis.items()):
                if i >= 8:  # Maximum 8 KPI
                    break
                
                row = i // 4 + 1
                col = i % 4 + 1
                
                # Déterminer la couleur selon la valeur
                if value >= 80:
                    color = 'green'
                elif value >= 60:
                    color = 'orange'
                else:
                    color = 'red'
                
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=value,
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': color},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        },
                        title={'text': kpi_name.replace('_', ' ').title()}
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(
                title="KPI Dashboard",
                template='plotly_white',
                height=600
            )
            
            return {
                'type': 'kpi_dashboard',
                'config': fig.to_dict(),
                'kpi_summary': kpis
            }
            
        except Exception as e:
            logger.error(f"Error generating KPI dashboard: {e}")
            return {'type': 'error', 'error': str(e)}


class AdvancedAnalyticsEngine:
    """
    Moteur d'analytics avancé avec intelligence artificielle
    
    Fonctionnalités:
    - Analytics temps réel avec streaming
    - Machine learning prédictif
    - Génération de rapports automatisée
    - Visualisation interactive
    - KPI intelligents et alerting
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_buffer = deque(maxlen=config.get('buffer_size', 100000))
        self.time_series_analyzer = TimeSeriesAnalyzer(config.get('ml_config', {}))
        self.kpi_calculator = KPICalculator()
        self.chart_generator = ChartGenerator()
        self.ml_models = {}
        self.performance_metrics = {
            'analytics_processed': PrometheusCounter('analytics_processed_total', 'Total analytics processed'),
            'analytics_time': Histogram('analytics_processing_seconds', 'Time spent processing analytics'),
            'active_queries': Gauge('active_analytics_queries', 'Number of active analytics queries'),
            'ml_predictions': PrometheusCounter('ml_predictions_total', 'Total ML predictions made'),
        }
        
        # Redis pour le cache
        self.redis_client = None
        self.setup_redis()
        
        # Base de données pour persistence
        self.db_pool = None
        self.setup_database()
        
        logger.info("Advanced Analytics Engine initialized")
    
    async def setup_redis(self):
        """Configuration du client Redis"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.get('redis_host', 'localhost'),
                port=self.config.get('redis_port', 6379),
                db=self.config.get('redis_db', 3),
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Redis connection established for analytics engine")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
    
    async def setup_database(self):
        """Configuration de la base de données PostgreSQL"""
        try:
            self.db_pool = await asyncpg.create_pool(
                host=self.config.get('db_host', 'localhost'),
                port=self.config.get('db_port', 5432),
                user=self.config.get('db_user', 'postgres'),
                password=self.config.get('db_password', ''),
                database=self.config.get('db_name', 'alerts'),
                min_size=5,
                max_size=20
            )
            await self.create_tables()
            logger.info("Database connection pool established")
        except Exception as e:
            logger.error(f"Failed to setup database: {e}")
    
    async def create_tables(self):
        """Crée les tables nécessaires"""
        create_metrics_table = """
        CREATE TABLE IF NOT EXISTS analytics_metrics (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(255) NOT NULL,
            value DOUBLE PRECISION NOT NULL,
            metric_type VARCHAR(50) NOT NULL,
            labels JSONB,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            tenant_id VARCHAR(100) NOT NULL,
            source VARCHAR(255),
            unit VARCHAR(50),
            description TEXT,
            tags JSONB,
            INDEX (tenant_id, name, timestamp),
            INDEX (timestamp),
            INDEX USING GIN (labels),
            INDEX USING GIN (tags)
        );
        """
        
        create_reports_table = """
        CREATE TABLE IF NOT EXISTS analytics_reports (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            title VARCHAR(255) NOT NULL,
            description TEXT,
            metrics JSONB NOT NULL,
            charts JSONB,
            insights JSONB,
            recommendations JSONB,
            kpis JSONB,
            anomalies JSONB,
            predictions JSONB,
            time_range_start TIMESTAMP WITH TIME ZONE NOT NULL,
            time_range_end TIMESTAMP WITH TIME ZONE NOT NULL,
            generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            tenant_id VARCHAR(100) NOT NULL,
            INDEX (tenant_id, generated_at),
            INDEX (time_range_start, time_range_end)
        );
        """
        
        create_queries_table = """
        CREATE TABLE IF NOT EXISTS analytics_queries (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(255) NOT NULL,
            query TEXT NOT NULL,
            parameters JSONB,
            time_range_start TIMESTAMP WITH TIME ZONE,
            time_range_end TIMESTAMP WITH TIME ZONE,
            granularity VARCHAR(50),
            filters JSONB,
            aggregations JSONB,
            tenant_id VARCHAR(100) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            INDEX (tenant_id, name),
            INDEX (created_at)
        );
        """
        
        async with self.db_pool.acquire() as conn:
            await conn.execute(create_metrics_table)
            await conn.execute(create_reports_table)
            await conn.execute(create_queries_table)
    
    async def record_metric(self, metric: Metric):
        """Enregistre une métrique"""
        try:
            # Ajouter au buffer
            self.metrics_buffer.append(metric)
            
            # Persister en base de données
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO analytics_metrics 
                        (name, value, metric_type, labels, timestamp, tenant_id, 
                         source, unit, description, tags)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    """, metric.name, metric.value, metric.metric_type.value,
                    json.dumps(metric.labels), metric.timestamp, metric.tenant_id,
                    metric.source, metric.unit, metric.description,
                    json.dumps(list(metric.tags)))
            
            # Cache dans Redis pour accès rapide
            if self.redis_client:
                cache_key = f"metric:{metric.tenant_id}:{metric.name}:{int(metric.timestamp.timestamp())}"
                await self.redis_client.setex(
                    cache_key,
                    timedelta(hours=24),
                    json.dumps(asdict(metric), default=str)
                )
            
            # Mise à jour des métriques de performance
            self.performance_metrics['analytics_processed'].inc()
            
        except Exception as e:
            logger.error(f"Error recording metric: {e}")
    
    async def query_metrics(self, query: AnalyticsQuery) -> List[Metric]:
        """Exécute une requête de métriques"""
        try:
            if not self.db_pool:
                return []
            
            # Construction de la requête SQL
            sql_query = """
                SELECT * FROM analytics_metrics 
                WHERE tenant_id = $1 
                AND timestamp BETWEEN $2 AND $3
            """
            params = [query.tenant_id, query.time_range[0], query.time_range[1]]
            
            # Ajouter des filtres
            filter_conditions = []
            param_index = 4
            
            for key, value in query.filters.items():
                if key == 'name':
                    filter_conditions.append(f"name = ${param_index}")
                    params.append(value)
                    param_index += 1
                elif key == 'source':
                    filter_conditions.append(f"source = ${param_index}")
                    params.append(value)
                    param_index += 1
                elif key == 'metric_type':
                    filter_conditions.append(f"metric_type = ${param_index}")
                    params.append(value)
                    param_index += 1
            
            if filter_conditions:
                sql_query += " AND " + " AND ".join(filter_conditions)
            
            sql_query += " ORDER BY timestamp DESC"
            
            # Exécuter la requête
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(sql_query, *params)
                
                metrics = []
                for row in rows:
                    metric = Metric(
                        id=row['id'],
                        name=row['name'],
                        value=row['value'],
                        metric_type=MetricType(row['metric_type']),
                        labels=json.loads(row['labels']) if row['labels'] else {},
                        timestamp=row['timestamp'],
                        tenant_id=row['tenant_id'],
                        source=row['source'] or 'unknown',
                        unit=row['unit'] or '',
                        description=row['description'] or '',
                        tags=set(json.loads(row['tags'])) if row['tags'] else set()
                    )
                    metrics.append(metric)
                
                return metrics
                
        except Exception as e:
            logger.error(f"Error querying metrics: {e}")
            return []
    
    async def generate_time_series(self, metric_name: str, tenant_id: str, 
                                 time_range: Tuple[datetime, datetime],
                                 granularity: TimeGranularity = TimeGranularity.MINUTE) -> TimeSeriesData:
        """Génère des données de série temporelle"""
        try:
            # Requête pour récupérer les métriques
            query = AnalyticsQuery(
                id=str(uuid.uuid4()),
                name=f"time_series_{metric_name}",
                query="",
                parameters={},
                time_range=time_range,
                granularity=granularity,
                filters={'name': metric_name},
                tenant_id=tenant_id
            )
            
            metrics = await self.query_metrics(query)
            
            if not metrics:
                return TimeSeriesData(
                    metric_name=metric_name,
                    timestamps=[],
                    values=[]
                )
            
            # Grouper par granularité temporelle
            grouped_data = defaultdict(list)
            
            for metric in metrics:
                # Arrondir le timestamp selon la granularité
                if granularity == TimeGranularity.MINUTE:
                    time_key = metric.timestamp.replace(second=0, microsecond=0)
                elif granularity == TimeGranularity.HOUR:
                    time_key = metric.timestamp.replace(minute=0, second=0, microsecond=0)
                elif granularity == TimeGranularity.DAY:
                    time_key = metric.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
                else:
                    time_key = metric.timestamp
                
                grouped_data[time_key].append(metric.value)
            
            # Agréger les valeurs
            timestamps = []
            values = []
            
            for timestamp in sorted(grouped_data.keys()):
                timestamps.append(timestamp)
                # Utiliser la moyenne pour l'agrégation
                values.append(np.mean(grouped_data[timestamp]))
            
            return TimeSeriesData(
                metric_name=metric_name,
                timestamps=timestamps,
                values=values,
                labels={'tenant_id': tenant_id},
                metadata={'granularity': granularity.value, 'aggregation': 'mean'}
            )
            
        except Exception as e:
            logger.error(f"Error generating time series: {e}")
            return TimeSeriesData(metric_name=metric_name, timestamps=[], values=[])
    
    async def analyze_time_series(self, data: TimeSeriesData) -> Dict[str, Any]:
        """Analyse complète d'une série temporelle"""
        try:
            analysis = {}
            
            # Analyse de tendance
            trend_analysis = await self.time_series_analyzer.analyze_trend(data)
            analysis['trend'] = trend_analysis
            
            # Détection d'anomalies
            anomalies = await self.time_series_analyzer.detect_anomalies(data)
            analysis['anomalies'] = anomalies
            
            # Prédictions
            predictions = await self.time_series_analyzer.predict_values(data)
            analysis['predictions'] = predictions
            
            # Statistiques de base
            if data.values:
                analysis['statistics'] = {
                    'count': len(data.values),
                    'mean': float(np.mean(data.values)),
                    'median': float(np.median(data.values)),
                    'std': float(np.std(data.values)),
                    'min': float(np.min(data.values)),
                    'max': float(np.max(data.values)),
                    'range': float(np.max(data.values) - np.min(data.values))
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing time series: {e}")
            return {'error': str(e)}
    
    async def generate_insights(self, metrics: List[Metric], time_range: Tuple[datetime, datetime]) -> List[str]:
        """Génère des insights automatiques"""
        insights = []
        
        try:
            if not metrics:
                return ["No data available for analysis"]
            
            # Analyser les tendances générales
            metric_groups = defaultdict(list)
            for metric in metrics:
                if isinstance(metric.value, (int, float)):
                    metric_groups[metric.name].append(metric.value)
            
            for metric_name, values in metric_groups.items():
                if len(values) < 2:
                    continue
                
                # Tendance
                recent_values = values[-10:] if len(values) >= 10 else values
                older_values = values[:-10] if len(values) >= 20 else values[:len(values)//2]
                
                if older_values and recent_values:
                    recent_avg = np.mean(recent_values)
                    older_avg = np.mean(older_values)
                    
                    change_percent = ((recent_avg - older_avg) / older_avg) * 100 if older_avg != 0 else 0
                    
                    if abs(change_percent) > 10:
                        direction = "increased" if change_percent > 0 else "decreased"
                        insights.append(
                            f"Metric '{metric_name}' has {direction} by {abs(change_percent):.1f}% "
                            f"in recent measurements"
                        )
                
                # Variabilité
                std_dev = np.std(values)
                mean_val = np.mean(values)
                cv = (std_dev / mean_val) * 100 if mean_val != 0 else 0
                
                if cv > 50:
                    insights.append(
                        f"Metric '{metric_name}' shows high variability (CV: {cv:.1f}%), "
                        f"indicating potential instability"
                    )
                elif cv < 5:
                    insights.append(
                        f"Metric '{metric_name}' is very stable (CV: {cv:.1f}%)"
                    )
            
            # Analyser les corrélations
            if len(metric_groups) >= 2:
                correlations = []
                metric_names = list(metric_groups.keys())
                
                for i in range(len(metric_names)):
                    for j in range(i + 1, len(metric_names)):
                        name1, name2 = metric_names[i], metric_names[j]
                        values1, values2 = metric_groups[name1], metric_groups[name2]
                        
                        min_length = min(len(values1), len(values2))
                        if min_length >= 5:
                            corr = np.corrcoef(values1[:min_length], values2[:min_length])[0, 1]
                            
                            if not np.isnan(corr) and abs(corr) > 0.7:
                                relationship = "positively" if corr > 0 else "negatively"
                                insights.append(
                                    f"Strong {relationship} correlation ({corr:.2f}) detected "
                                    f"between '{name1}' and '{name2}'"
                                )
            
            # Analyser les outliers
            for metric_name, values in metric_groups.items():
                if len(values) >= 10:
                    q1, q3 = np.percentile(values, [25, 75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    outliers = [v for v in values if v < lower_bound or v > upper_bound]
                    
                    if len(outliers) > len(values) * 0.05:  # Plus de 5% d'outliers
                        insights.append(
                            f"Metric '{metric_name}' has {len(outliers)} outliers "
                            f"({len(outliers)/len(values)*100:.1f}% of data points)"
                        )
            
            if not insights:
                insights.append("No significant patterns or anomalies detected in the current data")
            
            return insights[:10]  # Limiter à 10 insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return [f"Error generating insights: {str(e)}"]
    
    async def generate_recommendations(self, metrics: List[Metric], kpis: Dict[str, float]) -> List[str]:
        """Génère des recommandations automatiques"""
        recommendations = []
        
        try:
            # Recommandations basées sur les KPI
            if kpis.get('availability', 100) < 95:
                recommendations.append(
                    "Availability is below 95%. Consider implementing redundancy and "
                    "failover mechanisms to improve system reliability."
                )
            
            if kpis.get('error_rate', 0) > 5:
                recommendations.append(
                    "Error rate is above 5%. Review error logs and implement better "
                    "error handling and monitoring."
                )
            
            if kpis.get('performance', 100) < 70:
                recommendations.append(
                    "Performance metrics indicate potential bottlenecks. Consider "
                    "scaling resources or optimizing algorithms."
                )
            
            # Recommandations basées sur les métriques
            resource_metrics = [m for m in metrics if 'cpu' in m.name.lower() or 'memory' in m.name.lower()]
            if resource_metrics:
                avg_utilization = np.mean([m.value for m in resource_metrics])
                
                if avg_utilization > 90:
                    recommendations.append(
                        "Resource utilization is very high (>90%). Consider scaling up "
                        "or optimizing resource usage."
                    )
                elif avg_utilization < 30:
                    recommendations.append(
                        "Resource utilization is low (<30%). Consider scaling down "
                        "to optimize costs."
                    )
            
            # Recommandations sur la monitoring
            unique_metrics = set(m.name for m in metrics)
            if len(unique_metrics) < 5:
                recommendations.append(
                    "Limited metrics variety detected. Consider adding more comprehensive "
                    "monitoring to gain better insights."
                )
            
            # Recommandations temporelles
            now = datetime.utcnow()
            recent_metrics = [m for m in metrics if (now - m.timestamp).total_seconds() < 3600]
            
            if len(recent_metrics) < len(metrics) * 0.5:
                recommendations.append(
                    "More than 50% of metrics are older than 1 hour. Consider increasing "
                    "monitoring frequency for better real-time insights."
                )
            
            if not recommendations:
                recommendations.append("System appears to be operating within normal parameters.")
            
            return recommendations[:8]  # Limiter à 8 recommandations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return [f"Error generating recommendations: {str(e)}"]
    
    async def generate_report(self, tenant_id: str, time_range: Tuple[datetime, datetime], 
                            report_title: str = "Analytics Report") -> AnalyticsReport:
        """Génère un rapport d'analytics complet"""
        start_time = time.time()
        
        try:
            # Récupérer toutes les métriques pour la période
            query = AnalyticsQuery(
                id=str(uuid.uuid4()),
                name="full_report",
                query="",
                parameters={},
                time_range=time_range,
                granularity=TimeGranularity.HOUR,
                tenant_id=tenant_id
            )
            
            metrics = await self.query_metrics(query)
            
            # Calculer les KPI
            kpis = await self.kpi_calculator.calculate_kpis(metrics, time_range)
            
            # Générer des graphiques
            charts = []
            
            # Graphique de distribution pour chaque métrique unique
            unique_metrics = set(m.name for m in metrics)
            for metric_name in list(unique_metrics)[:5]:  # Limiter à 5 graphiques
                chart = await self.chart_generator.generate_distribution_chart(metrics, metric_name)
                charts.append(chart)
            
            # Graphique de corrélation
            if len(unique_metrics) >= 2:
                correlation_chart = await self.chart_generator.generate_correlation_heatmap(metrics)
                charts.append(correlation_chart)
            
            # Dashboard KPI
            kpi_dashboard = await self.chart_generator.generate_kpi_dashboard(kpis)
            charts.append(kpi_dashboard)
            
            # Générer des insights
            insights = await self.generate_insights(metrics, time_range)
            
            # Générer des recommandations
            recommendations = await self.generate_recommendations(metrics, kpis)
            
            # Détecter les anomalies globales
            anomalies = []
            for metric_name in unique_metrics:
                metric_data = [m for m in metrics if m.name == metric_name]
                if len(metric_data) >= 10:
                    # Convertir en TimeSeriesData pour l'analyse
                    ts_data = TimeSeriesData(
                        metric_name=metric_name,
                        timestamps=[m.timestamp for m in metric_data],
                        values=[m.value for m in metric_data if isinstance(m.value, (int, float))]
                    )
                    
                    if ts_data.values:
                        metric_anomalies = await self.time_series_analyzer.detect_anomalies(ts_data)
                        for anomaly in metric_anomalies:
                            anomaly['metric_name'] = metric_name
                        anomalies.extend(metric_anomalies)
            
            # Générer des prédictions
            predictions = []
            for metric_name in list(unique_metrics)[:3]:  # Limiter à 3 prédictions
                metric_data = [m for m in metrics if m.name == metric_name]
                if len(metric_data) >= 20:
                    ts_data = TimeSeriesData(
                        metric_name=metric_name,
                        timestamps=[m.timestamp for m in metric_data],
                        values=[m.value for m in metric_data if isinstance(m.value, (int, float))]
                    )
                    
                    if ts_data.values:
                        metric_predictions = await self.time_series_analyzer.predict_values(ts_data)
                        predictions.append({
                            'metric_name': metric_name,
                            'predictions': metric_predictions
                        })
            
            # Créer le rapport
            report = AnalyticsReport(
                id=str(uuid.uuid4()),
                title=report_title,
                description=f"Comprehensive analytics report for {tenant_id} from {time_range[0]} to {time_range[1]}",
                metrics=metrics,
                charts=charts,
                insights=insights,
                recommendations=recommendations,
                kpis=kpis,
                anomalies=anomalies,
                predictions=predictions,
                time_range=time_range,
                tenant_id=tenant_id
            )
            
            # Persister le rapport
            await self.persist_report(report)
            
            # Mise à jour des métriques
            self.performance_metrics['analytics_time'].observe(time.time() - start_time)
            
            logger.info(f"Generated analytics report for tenant {tenant_id} with {len(metrics)} metrics")
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return AnalyticsReport(
                id=str(uuid.uuid4()),
                title="Error Report",
                description=f"Failed to generate report: {str(e)}",
                metrics=[],
                charts=[],
                insights=[f"Error: {str(e)}"],
                recommendations=["Please check system logs for detailed error information"],
                kpis={},
                anomalies=[],
                predictions=[],
                time_range=time_range,
                tenant_id=tenant_id
            )
    
    async def persist_report(self, report: AnalyticsReport):
        """Persiste un rapport en base de données"""
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO analytics_reports 
                    (id, title, description, metrics, charts, insights, recommendations,
                     kpis, anomalies, predictions, time_range_start, time_range_end,
                     generated_at, tenant_id)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                """, 
                report.id, report.title, report.description,
                json.dumps([asdict(m) for m in report.metrics], default=str),
                json.dumps(report.charts, default=str),
                json.dumps(report.insights),
                json.dumps(report.recommendations),
                json.dumps(report.kpis),
                json.dumps(report.anomalies, default=str),
                json.dumps(report.predictions, default=str),
                report.time_range[0], report.time_range[1],
                report.generated_at, report.tenant_id)
                
            logger.info(f"Persisted analytics report {report.id}")
            
        except Exception as e:
            logger.error(f"Failed to persist report: {e}")
    
    async def get_real_time_metrics(self, tenant_id: str, metric_names: List[str] = None) -> List[Metric]:
        """Récupère les métriques en temps réel"""
        try:
            # Récupérer depuis le buffer en mémoire
            recent_metrics = []
            cutoff_time = datetime.utcnow() - timedelta(minutes=5)
            
            for metric in self.metrics_buffer:
                if metric.tenant_id == tenant_id and metric.timestamp >= cutoff_time:
                    if metric_names is None or metric.name in metric_names:
                        recent_metrics.append(metric)
            
            return recent_metrics
            
        except Exception as e:
            logger.error(f"Error getting real-time metrics: {e}")
            return []
    
    async def shutdown(self):
        """Arrêt propre du moteur d'analytics"""
        if self.redis_client:
            await self.redis_client.close()
        
        if self.db_pool:
            await self.db_pool.close()
        
        logger.info("Analytics engine shutdown complete")


# Factory function
def create_analytics_engine(config: Dict[str, Any]) -> AdvancedAnalyticsEngine:
    """Crée une instance du moteur d'analytics avec la configuration donnée"""
    return AdvancedAnalyticsEngine(config)


# Export des classes principales
__all__ = [
    'AdvancedAnalyticsEngine',
    'MetricType',
    'AnalyticsLevel',
    'TimeGranularity',
    'Metric',
    'TimeSeriesData',
    'AnalyticsQuery',
    'AnalyticsReport',
    'MLModel',
    'TimeSeriesAnalyzer',
    'KPICalculator',
    'ChartGenerator',
    'create_analytics_engine'
]
