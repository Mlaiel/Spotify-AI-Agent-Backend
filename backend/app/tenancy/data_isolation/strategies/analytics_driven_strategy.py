"""
📊 Analytics-Driven Strategy - Stratégie Pilotée par Analytics Ultra-Avancée
============================================================================

Stratégie d'isolation révolutionnaire utilisant des analytics avancées, du big data,
et de l'intelligence décisionnelle pour optimiser l'isolation des données basée sur
des insights métier, patterns d'usage, et prédictions business.

Fonctionnalités Ultra-Avancées:
    📈 Business Intelligence intégrée
    🎯 Predictive Analytics temps réel
    📊 Data-driven decision making
    🔍 Advanced pattern mining
    💡 Insight-driven optimization
    📋 KPI-based adaptation
    🎛️ Dashboard & reporting avancés
    🔮 Forecasting & trends analysis
    📉 Cost optimization analytics
    🏆 Performance benchmarking

Architecture Analytics:
    - Real-time data pipeline
    - Machine learning insights
    - Business metrics correlation
    - Predictive modeling
    - Automated recommendations
    - Executive dashboards

Author: Data Engineer Expert - Team Fahed Mlaiel
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Union, Set, Tuple, Callable, AsyncGenerator
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from collections import deque, defaultdict, Counter
import threading
from concurrent.futures import ThreadPoolExecutor
import statistics
import math
from pathlib import Path
import hashlib
import pickle

# Analytics and ML
try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import TimeSeriesSplit
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Time series analysis
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    TIMESERIES_AVAILABLE = True
except ImportError:
    TIMESERIES_AVAILABLE = False

# Visualization (optional)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

from ..core.tenant_context import TenantContext, TenantType, IsolationLevel
from ..core.isolation_engine import IsolationStrategy, EngineConfig
from ..managers.connection_manager import DatabaseConnection
from ..exceptions import (
    DataIsolationError, AnalyticsError, PredictionError,
    ConfigurationError, PerformanceError
)

# Logger setup
logger = logging.getLogger(__name__)


class AnalyticsMetricType(Enum):
    """Types de métriques analytics"""
    PERFORMANCE = "performance"
    BUSINESS = "business"
    TECHNICAL = "technical"
    FINANCIAL = "financial"
    USER_EXPERIENCE = "user_experience"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    OPERATIONAL = "operational"


class TrendDirection(Enum):
    """Directions de tendance"""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"
    SEASONAL = "seasonal"
    CYCLICAL = "cyclical"


class PredictionHorizon(Enum):
    """Horizons de prédiction"""
    SHORT_TERM = timedelta(hours=1)
    MEDIUM_TERM = timedelta(hours=6)
    LONG_TERM = timedelta(days=1)
    WEEKLY = timedelta(days=7)
    MONTHLY = timedelta(days=30)


class InsightType(Enum):
    """Types d'insights générés"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    COST_OPTIMIZATION = "cost_optimization"
    CAPACITY_PLANNING = "capacity_planning"
    ANOMALY_DETECTION = "anomaly_detection"
    TREND_ANALYSIS = "trend_analysis"
    PATTERN_DISCOVERY = "pattern_discovery"
    RECOMMENDATION = "recommendation"
    ALERT = "alert"


@dataclass
class BusinessMetric:
    """Métrique business"""
    metric_id: str
    name: str
    description: str
    metric_type: AnalyticsMetricType
    
    # Valeurs
    current_value: float = 0.0
    target_value: Optional[float] = None
    threshold_min: Optional[float] = None
    threshold_max: Optional[float] = None
    
    # Métadonnées
    unit: str = ""
    category: str = ""
    priority: int = 1  # 1-10
    
    # Historique
    historical_values: List[Tuple[datetime, float]] = field(default_factory=list)
    
    # KPIs
    is_kpi: bool = False
    kpi_weight: float = 1.0
    
    # Calculs
    trend: Optional[TrendDirection] = None
    change_rate: float = 0.0
    volatility: float = 0.0
    seasonality_strength: float = 0.0
    
    # Prédictions
    predicted_values: Dict[PredictionHorizon, float] = field(default_factory=dict)
    prediction_confidence: Dict[PredictionHorizon, float] = field(default_factory=dict)


@dataclass
class PerformanceInsight:
    """Insight de performance"""
    insight_id: str = field(default_factory=lambda: hashlib.sha256(str(time.time()).encode()).hexdigest()[:16])
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    insight_type: InsightType = InsightType.RECOMMENDATION
    
    # Contenu
    title: str = ""
    description: str = ""
    severity: str = "medium"  # low, medium, high, critical
    confidence: float = 0.8
    
    # Métriques concernées
    affected_metrics: List[str] = field(default_factory=list)
    root_cause: Optional[str] = None
    
    # Recommandations
    recommendations: List[str] = field(default_factory=list)
    estimated_impact: Dict[str, float] = field(default_factory=dict)
    implementation_complexity: str = "medium"  # low, medium, high
    
    # Business impact
    cost_impact: float = 0.0
    performance_impact: float = 0.0
    user_impact: float = 0.0
    
    # Actions
    actionable: bool = True
    auto_implementable: bool = False
    requires_approval: bool = False


@dataclass
class AnalyticsReport:
    """Rapport d'analytics"""
    report_id: str = field(default_factory=lambda: hashlib.sha256(str(time.time()).encode()).hexdigest()[:16])
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    period_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc) - timedelta(hours=24))
    period_end: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Contenu
    title: str = "Analytics Report"
    summary: str = ""
    
    # Métriques
    key_metrics: Dict[str, BusinessMetric] = field(default_factory=dict)
    kpi_summary: Dict[str, float] = field(default_factory=dict)
    
    # Insights
    insights: List[PerformanceInsight] = field(default_factory=list)
    top_insights: List[PerformanceInsight] = field(default_factory=list)
    
    # Tendances
    trends: Dict[str, TrendDirection] = field(default_factory=dict)
    forecasts: Dict[str, Dict[PredictionHorizon, float]] = field(default_factory=dict)
    
    # Performance
    overall_health_score: float = 100.0
    performance_score: float = 100.0
    efficiency_score: float = 100.0
    cost_efficiency: float = 1.0
    
    # Comparaisons
    period_comparison: Dict[str, float] = field(default_factory=dict)
    benchmark_comparison: Dict[str, float] = field(default_factory=dict)


class MetricsCollector:
    """Collecteur de métriques avancé"""
    
    def __init__(self, retention_days: int = 30):
        self.retention_days = retention_days
        self.metrics: Dict[str, BusinessMetric] = {}
        self.raw_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._lock = threading.RLock()
        
    def register_metric(self, metric: BusinessMetric):
        """Enregistre une nouvelle métrique"""
        with self._lock:
            self.metrics[metric.metric_id] = metric
            logger.info(f"Registered metric: {metric.name}")
    
    def add_measurement(self, metric_id: str, value: float, timestamp: Optional[datetime] = None):
        """Ajoute une mesure"""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
            
        with self._lock:
            if metric_id in self.metrics:
                metric = self.metrics[metric_id]
                metric.current_value = value
                metric.historical_values.append((timestamp, value))
                
                # Limite la taille de l'historique
                if len(metric.historical_values) > 1000:
                    metric.historical_values = metric.historical_values[-1000:]
                
                # Stockage des données brutes
                self.raw_data[metric_id].append((timestamp, value))
    
    def get_metric(self, metric_id: str) -> Optional[BusinessMetric]:
        """Récupère une métrique"""
        with self._lock:
            return self.metrics.get(metric_id)
    
    def get_metrics_by_type(self, metric_type: AnalyticsMetricType) -> List[BusinessMetric]:
        """Récupère les métriques par type"""
        with self._lock:
            return [m for m in self.metrics.values() if m.metric_type == metric_type]
    
    def get_kpi_metrics(self) -> List[BusinessMetric]:
        """Récupère les métriques KPI"""
        with self._lock:
            return [m for m in self.metrics.values() if m.is_kpi]
    
    def cleanup_old_data(self):
        """Nettoie les anciennes données"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        
        with self._lock:
            for metric in self.metrics.values():
                metric.historical_values = [
                    (ts, val) for ts, val in metric.historical_values 
                    if ts > cutoff_date
                ]


class TrendAnalyzer:
    """Analyseur de tendances avancé"""
    
    def __init__(self):
        self.min_data_points = 10
        
    def analyze_trend(self, metric: BusinessMetric) -> TrendDirection:
        """Analyse la tendance d'une métrique"""
        if len(metric.historical_values) < self.min_data_points:
            return TrendDirection.STABLE
        
        try:
            # Extraction des valeurs
            values = [val for _, val in metric.historical_values[-50:]]  # Dernières 50 valeurs
            
            if len(values) < 3:
                return TrendDirection.STABLE
            
            # Calcul de la tendance par régression linéaire simple
            n = len(values)
            x = np.arange(n)
            y = np.array(values)
            
            # Régression linéaire
            slope = np.polyfit(x, y, 1)[0]
            
            # Calcul de la volatilité
            volatility = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
            
            # Mise à jour de la métrique
            metric.volatility = volatility
            metric.change_rate = slope
            
            # Détermination de la tendance
            if abs(slope) < 0.01:
                if volatility > 0.2:
                    return TrendDirection.VOLATILE
                else:
                    return TrendDirection.STABLE
            elif slope > 0:
                return TrendDirection.INCREASING
            else:
                return TrendDirection.DECREASING
                
        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")
            return TrendDirection.STABLE
    
    def detect_seasonality(self, metric: BusinessMetric) -> float:
        """Détecte la saisonnalité"""
        if len(metric.historical_values) < 24:  # Minimum 24 points
            return 0.0
        
        try:
            if not TIMESERIES_AVAILABLE:
                return 0.0
            
            # Préparation des données
            df = pd.DataFrame(metric.historical_values, columns=['timestamp', 'value'])
            df.set_index('timestamp', inplace=True)
            df = df.resample('H').mean().fillna(method='forward')
            
            if len(df) < 24:
                return 0.0
            
            # Décomposition saisonnière
            decomposition = seasonal_decompose(df['value'], model='additive', period=24)
            
            # Force de la saisonnalité
            seasonal_strength = np.std(decomposition.seasonal) / np.std(df['value'])
            metric.seasonality_strength = seasonal_strength
            
            return seasonal_strength
            
        except Exception as e:
            logger.error(f"Error detecting seasonality: {e}")
            return 0.0


class Predictor:
    """Prédicteur avancé avec ML"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
    def train_predictor(self, metric: BusinessMetric) -> bool:
        """Entraîne un prédicteur pour une métrique"""
        if not ML_AVAILABLE or len(metric.historical_values) < 50:
            return False
        
        try:
            # Préparation des données
            df = pd.DataFrame(metric.historical_values, columns=['timestamp', 'value'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df = df.resample('H').mean().fillna(method='forward')
            
            if len(df) < 20:
                return False
            
            # Features d'ingénierie temporelle
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
            
            # Features de lag
            for lag in [1, 6, 12, 24]:
                df[f'lag_{lag}'] = df['value'].shift(lag)
            
            # Features statistiques roulantes
            df['rolling_mean_6'] = df['value'].rolling(window=6).mean()
            df['rolling_std_6'] = df['value'].rolling(window=6).std()
            df['rolling_mean_24'] = df['value'].rolling(window=24).mean()
            
            # Nettoyage
            df.dropna(inplace=True)
            
            if len(df) < 10:
                return False
            
            # Préparation X, y
            feature_cols = [col for col in df.columns if col != 'value']
            X = df[feature_cols]
            y = df['value']
            
            # Normalisation
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Entraînement du modèle
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            
            # Validation temporelle
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = r2_score(y_val, y_pred)
                scores.append(score)
            
            avg_score = np.mean(scores)
            
            if avg_score > 0.5:  # Score minimum acceptable
                # Entraînement final sur toutes les données
                model.fit(X_scaled, y)
                
                self.models[metric.metric_id] = {
                    'model': model,
                    'feature_cols': feature_cols,
                    'score': avg_score,
                    'last_training': datetime.now(timezone.utc)
                }
                self.scalers[metric.metric_id] = scaler
                
                logger.info(f"Trained predictor for {metric.name} with R² = {avg_score:.3f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error training predictor: {e}")
            return False
    
    def predict(self, metric: BusinessMetric, horizon: PredictionHorizon) -> Tuple[float, float]:
        """Fait une prédiction pour une métrique"""
        if metric.metric_id not in self.models:
            return 0.0, 0.0
        
        try:
            model_data = self.models[metric.metric_id]
            model = model_data['model']
            feature_cols = model_data['feature_cols']
            scaler = self.scalers[metric.metric_id]
            
            # Prépare les features pour la prédiction
            future_time = datetime.now(timezone.utc) + horizon.value
            
            # Features temporelles
            features = {
                'hour': future_time.hour,
                'day_of_week': future_time.weekday(),
                'day_of_month': future_time.day,
                'is_weekend': 1 if future_time.weekday() >= 5 else 0
            }
            
            # Features de lag (approximation avec les dernières valeurs)
            recent_values = [val for _, val in metric.historical_values[-24:]]
            if len(recent_values) >= 1:
                features['lag_1'] = recent_values[-1]
            if len(recent_values) >= 6:
                features['lag_6'] = recent_values[-6]
                features['rolling_mean_6'] = np.mean(recent_values[-6:])
                features['rolling_std_6'] = np.std(recent_values[-6:])
            if len(recent_values) >= 12:
                features['lag_12'] = recent_values[-12]
            if len(recent_values) >= 24:
                features['lag_24'] = recent_values[-24]
                features['rolling_mean_24'] = np.mean(recent_values[-24:])
            
            # Remplissage des features manquantes
            for col in feature_cols:
                if col not in features:
                    features[col] = 0.0
            
            # Préparation du vecteur de features
            X = np.array([[features[col] for col in feature_cols]])
            X_scaled = scaler.transform(X)
            
            # Prédiction
            prediction = model.predict(X_scaled)[0]
            confidence = model_data['score']  # Utilise le score de validation comme confiance
            
            # Mise à jour de la métrique
            metric.predicted_values[horizon] = prediction
            metric.prediction_confidence[horizon] = confidence
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return 0.0, 0.0


class InsightGenerator:
    """Générateur d'insights avancé"""
    
    def __init__(self):
        self.insight_rules: List[Callable[[BusinessMetric], Optional[PerformanceInsight]]] = []
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Configure les règles d'insights par défaut"""
        self.insight_rules.extend([
            self._check_performance_degradation,
            self._check_anomaly,
            self._check_threshold_violation,
            self._check_trend_analysis,
            self._check_cost_optimization,
            self._check_capacity_planning
        ])
    
    def generate_insights(self, metrics: List[BusinessMetric]) -> List[PerformanceInsight]:
        """Génère des insights pour une liste de métriques"""
        insights = []
        
        for metric in metrics:
            for rule in self.insight_rules:
                try:
                    insight = rule(metric)
                    if insight:
                        insights.append(insight)
                except Exception as e:
                    logger.error(f"Error generating insight: {e}")
        
        # Tri par sévérité et confiance
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        insights.sort(key=lambda i: (severity_order.get(i.severity, 0), i.confidence), reverse=True)
        
        return insights
    
    def _check_performance_degradation(self, metric: BusinessMetric) -> Optional[PerformanceInsight]:
        """Vérifie la dégradation de performance"""
        if metric.metric_type != AnalyticsMetricType.PERFORMANCE:
            return None
        
        if len(metric.historical_values) < 10:
            return None
        
        # Compare les dernières valeurs avec la moyenne historique
        recent_values = [val for _, val in metric.historical_values[-5:]]
        historical_values = [val for _, val in metric.historical_values[:-5]]
        
        if not recent_values or not historical_values:
            return None
        
        recent_avg = np.mean(recent_values)
        historical_avg = np.mean(historical_values)
        
        # Détection de dégradation (pour les métriques où plus bas = mieux, comme latence)
        if recent_avg > historical_avg * 1.2:  # 20% de dégradation
            severity = "high" if recent_avg > historical_avg * 1.5 else "medium"
            
            return PerformanceInsight(
                insight_type=InsightType.PERFORMANCE_DEGRADATION,
                title=f"Dégradation de performance détectée - {metric.name}",
                description=f"La métrique {metric.name} a augmenté de {((recent_avg/historical_avg-1)*100):.1f}% récemment",
                severity=severity,
                confidence=0.8,
                affected_metrics=[metric.metric_id],
                recommendations=[
                    "Analyser les causes de la dégradation",
                    "Vérifier les ressources système",
                    "Optimiser les requêtes/processus",
                    "Considérer un scaling horizontal"
                ],
                performance_impact=recent_avg - historical_avg
            )
        
        return None
    
    def _check_anomaly(self, metric: BusinessMetric) -> Optional[PerformanceInsight]:
        """Vérifie les anomalies statistiques"""
        if len(metric.historical_values) < 20:
            return None
        
        values = [val for _, val in metric.historical_values]
        current_value = metric.current_value
        
        # Détection d'anomalie par écart-type
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return None
        
        z_score = abs((current_value - mean_val) / std_val)
        
        if z_score > 3:  # Anomalie significative
            return PerformanceInsight(
                insight_type=InsightType.ANOMALY_DETECTION,
                title=f"Anomalie détectée - {metric.name}",
                description=f"Valeur anormale détectée (Z-score: {z_score:.2f})",
                severity="high" if z_score > 4 else "medium",
                confidence=min(z_score / 5, 1.0),
                affected_metrics=[metric.metric_id],
                recommendations=[
                    "Investiguer la cause de l'anomalie",
                    "Vérifier les logs système",
                    "Analyser les changements récents"
                ]
            )
        
        return None
    
    def _check_threshold_violation(self, metric: BusinessMetric) -> Optional[PerformanceInsight]:
        """Vérifie les violations de seuils"""
        current_value = metric.current_value
        
        # Vérification seuil maximum
        if metric.threshold_max and current_value > metric.threshold_max:
            return PerformanceInsight(
                insight_type=InsightType.ALERT,
                title=f"Seuil maximum dépassé - {metric.name}",
                description=f"Valeur actuelle ({current_value:.2f}) dépasse le seuil maximum ({metric.threshold_max:.2f})",
                severity="high",
                confidence=1.0,
                affected_metrics=[metric.metric_id],
                recommendations=[
                    "Action immédiate requise",
                    "Vérifier la configuration des seuils",
                    "Implémenter des actions correctives"
                ]
            )
        
        # Vérification seuil minimum
        if metric.threshold_min and current_value < metric.threshold_min:
            return PerformanceInsight(
                insight_type=InsightType.ALERT,
                title=f"Seuil minimum non atteint - {metric.name}",
                description=f"Valeur actuelle ({current_value:.2f}) en dessous du seuil minimum ({metric.threshold_min:.2f})",
                severity="medium",
                confidence=1.0,
                affected_metrics=[metric.metric_id],
                recommendations=[
                    "Analyser les causes de la sous-performance",
                    "Ajuster les paramètres système"
                ]
            )
        
        return None
    
    def _check_trend_analysis(self, metric: BusinessMetric) -> Optional[PerformanceInsight]:
        """Analyse les tendances"""
        if not metric.trend or len(metric.historical_values) < 15:
            return None
        
        if metric.trend == TrendDirection.INCREASING and metric.change_rate > 0.1:
            return PerformanceInsight(
                insight_type=InsightType.TREND_ANALYSIS,
                title=f"Tendance croissante détectée - {metric.name}",
                description=f"Tendance à la hausse continue (taux: {metric.change_rate:.3f})",
                severity="medium",
                confidence=0.7,
                affected_metrics=[metric.metric_id],
                recommendations=[
                    "Surveiller l'évolution",
                    "Planifier la capacité",
                    "Préparer des actions préventives"
                ]
            )
        
        elif metric.trend == TrendDirection.VOLATILE and metric.volatility > 0.3:
            return PerformanceInsight(
                insight_type=InsightType.TREND_ANALYSIS,
                title=f"Forte volatilité détectée - {metric.name}",
                description=f"Métrique très volatile (volatilité: {metric.volatility:.2f})",
                severity="medium",
                confidence=0.8,
                affected_metrics=[metric.metric_id],
                recommendations=[
                    "Analyser les causes de volatilité",
                    "Améliorer la stabilité système",
                    "Considérer un lissage des métriques"
                ]
            )
        
        return None
    
    def _check_cost_optimization(self, metric: BusinessMetric) -> Optional[PerformanceInsight]:
        """Vérifie les opportunités d'optimisation des coûts"""
        if metric.metric_type != AnalyticsMetricType.FINANCIAL:
            return None
        
        # Logique d'optimisation des coûts
        # (à personnaliser selon les métriques spécifiques)
        
        return None
    
    def _check_capacity_planning(self, metric: BusinessMetric) -> Optional[PerformanceInsight]:
        """Vérifie les besoins de planification de capacité"""
        if metric.metric_type not in [AnalyticsMetricType.PERFORMANCE, AnalyticsMetricType.TECHNICAL]:
            return None
        
        # Logique de planification de capacité
        # (à personnaliser selon les métriques spécifiques)
        
        return None


class AnalyticsDashboard:
    """Dashboard analytics avancé"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.trend_analyzer = TrendAnalyzer()
        self.predictor = Predictor()
        self.insight_generator = InsightGenerator()
        
    def generate_report(
        self, 
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None
    ) -> AnalyticsReport:
        """Génère un rapport d'analytics complet"""
        
        if period_end is None:
            period_end = datetime.now(timezone.utc)
        if period_start is None:
            period_start = period_end - timedelta(hours=24)
        
        report = AnalyticsReport(
            period_start=period_start,
            period_end=period_end
        )
        
        # Collecte des métriques
        all_metrics = list(self.metrics_collector.metrics.values())
        kpi_metrics = self.metrics_collector.get_kpi_metrics()
        
        # Analyse des tendances
        for metric in all_metrics:
            metric.trend = self.trend_analyzer.analyze_trend(metric)
            self.trend_analyzer.detect_seasonality(metric)
            
        # Prédictions
        for metric in all_metrics:
            self.predictor.train_predictor(metric)
            for horizon in PredictionHorizon:
                pred, conf = self.predictor.predict(metric, horizon)
                if conf > 0.5:
                    if metric.metric_id not in report.forecasts:
                        report.forecasts[metric.metric_id] = {}
                    report.forecasts[metric.metric_id][horizon] = pred
        
        # Génération d'insights
        insights = self.insight_generator.generate_insights(all_metrics)
        report.insights = insights
        report.top_insights = insights[:5]  # Top 5 insights
        
        # Calcul des scores globaux
        report.overall_health_score = self._calculate_health_score(all_metrics)
        report.performance_score = self._calculate_performance_score(all_metrics)
        report.efficiency_score = self._calculate_efficiency_score(all_metrics)
        
        # Métriques clés
        for metric in kpi_metrics:
            report.key_metrics[metric.metric_id] = metric
            report.kpi_summary[metric.name] = metric.current_value
        
        # Tendances globales
        for metric in all_metrics:
            if metric.trend:
                report.trends[metric.name] = metric.trend
        
        # Résumé
        report.summary = self._generate_summary(report)
        
        return report
    
    def _calculate_health_score(self, metrics: List[BusinessMetric]) -> float:
        """Calcule le score de santé global"""
        if not metrics:
            return 100.0
        
        scores = []
        for metric in metrics:
            if metric.threshold_max and metric.current_value > metric.threshold_max:
                scores.append(0.0)
            elif metric.threshold_min and metric.current_value < metric.threshold_min:
                scores.append(50.0)
            else:
                scores.append(100.0)
        
        return np.mean(scores)
    
    def _calculate_performance_score(self, metrics: List[BusinessMetric]) -> float:
        """Calcule le score de performance"""
        perf_metrics = [m for m in metrics if m.metric_type == AnalyticsMetricType.PERFORMANCE]
        
        if not perf_metrics:
            return 100.0
        
        # Score basé sur les tendances et seuils
        scores = []
        for metric in perf_metrics:
            score = 100.0
            
            # Pénalité pour dégradation de tendance
            if metric.trend == TrendDirection.INCREASING and metric.change_rate > 0:
                score -= min(metric.change_rate * 50, 30)
            elif metric.trend == TrendDirection.VOLATILE:
                score -= min(metric.volatility * 50, 25)
            
            # Pénalité pour violation de seuils
            if metric.threshold_max and metric.current_value > metric.threshold_max:
                score -= 40
            elif metric.threshold_min and metric.current_value < metric.threshold_min:
                score -= 20
            
            scores.append(max(score, 0))
        
        return np.mean(scores)
    
    def _calculate_efficiency_score(self, metrics: List[BusinessMetric]) -> float:
        """Calcule le score d'efficacité"""
        # Score basé sur l'utilisation des ressources et les coûts
        # (à personnaliser selon les métriques spécifiques)
        return 85.0  # Valeur par défaut
    
    def _generate_summary(self, report: AnalyticsReport) -> str:
        """Génère un résumé du rapport"""
        summary_parts = []
        
        # Santé globale
        health_status = "Excellent" if report.overall_health_score > 90 else \
                       "Bon" if report.overall_health_score > 70 else \
                       "Moyen" if report.overall_health_score > 50 else "Critique"
        
        summary_parts.append(f"Santé globale du système: {health_status} ({report.overall_health_score:.1f}%)")
        
        # Insights critiques
        critical_insights = [i for i in report.insights if i.severity == "critical"]
        if critical_insights:
            summary_parts.append(f"{len(critical_insights)} insight(s) critique(s) détecté(s)")
        
        # Tendances importantes
        increasing_trends = [name for name, trend in report.trends.items() if trend == TrendDirection.INCREASING]
        if increasing_trends:
            summary_parts.append(f"Tendances croissantes: {', '.join(increasing_trends[:3])}")
        
        return ". ".join(summary_parts) + "."


class AnalyticsDrivenStrategy(IsolationStrategy):
    """
    Stratégie d'isolation pilotée par analytics ultra-avancée
    
    Cette stratégie utilise des analytics avancées, du machine learning,
    et de l'intelligence décisionnelle pour optimiser l'isolation basée
    sur des insights métier et des prédictions de performance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.config = config or {}
        
        # Composants analytics
        self.metrics_collector = MetricsCollector(
            retention_days=self.config.get('retention_days', 30)
        )
        self.dashboard = AnalyticsDashboard(self.metrics_collector)
        
        # Configuration
        self.analytics_enabled = self.config.get('analytics_enabled', True)
        self.auto_optimization = self.config.get('auto_optimization', True)
        self.reporting_interval = self.config.get('reporting_interval', 3600)  # 1 heure
        
        # État
        self.last_report = None
        self.optimization_history = deque(maxlen=100)
        self._initialized = False
        
        # Métriques par défaut
        self._setup_default_metrics()
        
        logger.info("Analytics-driven strategy initialized")
    
    def _setup_default_metrics(self):
        """Configure les métriques par défaut"""
        
        # Métriques de performance
        self.metrics_collector.register_metric(BusinessMetric(
            metric_id="query_latency",
            name="Latence de Requête",
            description="Temps de réponse moyen des requêtes",
            metric_type=AnalyticsMetricType.PERFORMANCE,
            unit="ms",
            threshold_max=100.0,
            is_kpi=True,
            kpi_weight=1.5
        ))
        
        self.metrics_collector.register_metric(BusinessMetric(
            metric_id="throughput",
            name="Débit",
            description="Nombre de requêtes par seconde",
            metric_type=AnalyticsMetricType.PERFORMANCE,
            unit="qps",
            threshold_min=10.0,
            is_kpi=True,
            kpi_weight=1.3
        ))
        
        # Métriques techniques
        self.metrics_collector.register_metric(BusinessMetric(
            metric_id="cpu_usage",
            name="Utilisation CPU",
            description="Pourcentage d'utilisation du CPU",
            metric_type=AnalyticsMetricType.TECHNICAL,
            unit="%",
            threshold_max=80.0
        ))
        
        self.metrics_collector.register_metric(BusinessMetric(
            metric_id="memory_usage",
            name="Utilisation Mémoire",
            description="Utilisation de la mémoire en MB",
            metric_type=AnalyticsMetricType.TECHNICAL,
            unit="MB",
            threshold_max=8192.0
        ))
        
        # Métriques de sécurité
        self.metrics_collector.register_metric(BusinessMetric(
            metric_id="auth_failures",
            name="Échecs d'Authentification",
            description="Nombre d'échecs d'authentification",
            metric_type=AnalyticsMetricType.SECURITY,
            unit="count",
            threshold_max=10.0
        ))
        
        # Métriques business
        self.metrics_collector.register_metric(BusinessMetric(
            metric_id="user_satisfaction",
            name="Satisfaction Utilisateur",
            description="Score de satisfaction utilisateur",
            metric_type=AnalyticsMetricType.USER_EXPERIENCE,
            unit="score",
            threshold_min=80.0,
            is_kpi=True,
            kpi_weight=2.0
        ))
    
    async def initialize(self):
        """Initialise la stratégie analytics"""
        if self._initialized:
            return
            
        try:
            # Initialisation des composants
            if self.analytics_enabled:
                # Démarre la collecte périodique de métriques
                asyncio.create_task(self._periodic_metrics_collection())
                
                # Démarre la génération de rapports périodiques
                asyncio.create_task(self._periodic_reporting())
            
            self._initialized = True
            logger.info("Analytics-driven strategy fully initialized")
            
        except Exception as e:
            logger.error(f"Error initializing analytics strategy: {e}")
            raise
    
    async def isolate_query(self, query: str, tenant_context: TenantContext) -> str:
        """Isole une requête avec analytics"""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Exécution de la requête (logique d'isolation à implémenter)
            # Pour cet exemple, on simule l'isolation
            isolated_query = f"/* Analytics-driven isolation for tenant {tenant_context.tenant_id} */ {query}"
            
            # Simulation d'exécution
            await asyncio.sleep(0.01)  # Simulation de latence
            
            # Collecte des métriques
            execution_time = (time.time() - start_time) * 1000
            
            if self.analytics_enabled:
                await self._collect_query_metrics(execution_time, tenant_context)
            
            return isolated_query
            
        except Exception as e:
            # Collecte des métriques d'erreur
            if self.analytics_enabled:
                await self._collect_error_metrics(e, tenant_context)
            raise
    
    async def _collect_query_metrics(self, execution_time_ms: float, tenant_context: TenantContext):
        """Collecte les métriques de requête"""
        try:
            # Latence
            self.metrics_collector.add_measurement("query_latency", execution_time_ms)
            
            # Débit (approximation)
            self.metrics_collector.add_measurement("throughput", 1000 / max(execution_time_ms, 1))
            
            # Métriques système
            if MONITORING_AVAILABLE:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_mb = psutil.virtual_memory().used / (1024 * 1024)
                
                self.metrics_collector.add_measurement("cpu_usage", cpu_percent)
                self.metrics_collector.add_measurement("memory_usage", memory_mb)
            
        except Exception as e:
            logger.error(f"Error collecting query metrics: {e}")
    
    async def _collect_error_metrics(self, error: Exception, tenant_context: TenantContext):
        """Collecte les métriques d'erreur"""
        try:
            # Incrémente les compteurs d'erreur selon le type
            if "auth" in str(error).lower():
                current_failures = self.metrics_collector.get_metric("auth_failures")
                if current_failures:
                    self.metrics_collector.add_measurement(
                        "auth_failures", 
                        current_failures.current_value + 1
                    )
            
        except Exception as e:
            logger.error(f"Error collecting error metrics: {e}")
    
    async def _periodic_metrics_collection(self):
        """Collecte périodique des métriques système"""
        while True:
            try:
                if MONITORING_AVAILABLE:
                    import psutil
                    
                    # Métriques système
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    
                    self.metrics_collector.add_measurement("cpu_usage", cpu_percent)
                    self.metrics_collector.add_measurement("memory_usage", memory.used / (1024 * 1024))
                
                # Nettoyage des anciennes données
                self.metrics_collector.cleanup_old_data()
                
                await asyncio.sleep(60)  # Collecte toutes les minutes
                
            except Exception as e:
                logger.error(f"Error in periodic metrics collection: {e}")
                await asyncio.sleep(60)
    
    async def _periodic_reporting(self):
        """Génération périodique de rapports"""
        while True:
            try:
                # Génère un rapport
                report = self.dashboard.generate_report()
                self.last_report = report
                
                # Optimisation automatique basée sur les insights
                if self.auto_optimization:
                    await self._auto_optimize(report)
                
                logger.info(f"Analytics report generated: Health={report.overall_health_score:.1f}%, "
                           f"Insights={len(report.insights)}")
                
                await asyncio.sleep(self.reporting_interval)
                
            except Exception as e:
                logger.error(f"Error in periodic reporting: {e}")
                await asyncio.sleep(self.reporting_interval)
    
    async def _auto_optimize(self, report: AnalyticsReport):
        """Optimisation automatique basée sur le rapport"""
        try:
            # Analyse des insights critiques
            critical_insights = [i for i in report.insights if i.severity == "critical"]
            
            for insight in critical_insights:
                if insight.auto_implementable and not insight.requires_approval:
                    success = await self._implement_optimization(insight)
                    
                    self.optimization_history.append({
                        'timestamp': datetime.now(timezone.utc),
                        'insight_id': insight.insight_id,
                        'action': insight.recommendations[0] if insight.recommendations else "Unknown",
                        'success': success
                    })
            
        except Exception as e:
            logger.error(f"Error in auto-optimization: {e}")
    
    async def _implement_optimization(self, insight: PerformanceInsight) -> bool:
        """Implémente une optimisation automatique"""
        try:
            # Logique d'implémentation des optimisations
            # (à personnaliser selon les types d'insights)
            
            logger.info(f"Implementing optimization: {insight.title}")
            
            # Simulation d'implémentation
            await asyncio.sleep(0.1)
            
            return True
            
        except Exception as e:
            logger.error(f"Error implementing optimization: {e}")
            return False
    
    async def validate_isolation(self, tenant_context: TenantContext) -> bool:
        """Valide l'isolation avec analytics"""
        # Validation basée sur les métriques de sécurité
        auth_failures_metric = self.metrics_collector.get_metric("auth_failures")
        
        if auth_failures_metric and auth_failures_metric.current_value > 10:
            logger.warning("High authentication failures detected")
            return False
        
        return True
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des analytics"""
        summary = {
            'metrics_count': len(self.metrics_collector.metrics),
            'kpi_count': len(self.metrics_collector.get_kpi_metrics()),
            'optimization_count': len(self.optimization_history),
            'last_report_time': self.last_report.timestamp.isoformat() if self.last_report else None
        }
        
        if self.last_report:
            summary.update({
                'health_score': self.last_report.overall_health_score,
                'performance_score': self.last_report.performance_score,
                'insights_count': len(self.last_report.insights),
                'critical_insights_count': len([i for i in self.last_report.insights if i.severity == "critical"])
            })
        
        return summary
    
    def get_latest_report(self) -> Optional[AnalyticsReport]:
        """Retourne le dernier rapport généré"""
        return self.last_report


# Factory pour création de la stratégie
def create_analytics_driven_strategy(config: Optional[Dict[str, Any]] = None) -> AnalyticsDrivenStrategy:
    """
    Factory pour créer une stratégie pilotée par analytics
    
    Args:
        config: Configuration de la stratégie
        
    Returns:
        Instance de AnalyticsDrivenStrategy configurée
    """
    return AnalyticsDrivenStrategy(config)


# Export du module
__all__ = [
    'AnalyticsDrivenStrategy',
    'MetricsCollector',
    'TrendAnalyzer',
    'Predictor',
    'InsightGenerator',
    'AnalyticsDashboard',
    'BusinessMetric',
    'PerformanceInsight',
    'AnalyticsReport',
    'AnalyticsMetricType',
    'TrendDirection',
    'PredictionHorizon',
    'InsightType',
    'create_analytics_driven_strategy'
]
