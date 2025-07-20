"""
Analytics Algorithms - Algorithmes d'Analyse Avancés
====================================================

Algorithmes d'intelligence artificielle et d'analyse de données avancés
pour le système de monitoring Spotify AI Agent.

Features:
    - Détection d'anomalies par Machine Learning
    - Prédiction de tendances et forecasting
    - Analyse comportementale intelligente
    - Clustering et segmentation automatique
    - Algorithmes d'optimisation en temps réel

Author: Expert Data Science Team + Senior Machine Learning Engineer + Analytics Architect
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import asyncio
import logging
import json
import warnings
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import math
import statistics
from collections import defaultdict, deque
import pickle
import joblib

# Machine Learning libraries
try:
    from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingClassifier
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import mean_squared_error, accuracy_score, silhouette_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from scipy import stats
    from scipy.stats import zscore, pearsonr
    from scipy.signal import find_peaks, savgol_filter
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("Librairies ML non disponibles, fonctionnalités limitées")

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


# =============================================================================
# TYPES ET ÉNUMÉRATIONS
# =============================================================================

class AnomalyType(Enum):
    """Types d'anomalies détectables."""
    POINT = "point"  # Anomalie ponctuelle
    CONTEXTUAL = "contextual"  # Anomalie contextuelle
    COLLECTIVE = "collective"  # Anomalie collective
    TREND = "trend"  # Anomalie de tendance
    SEASONAL = "seasonal"  # Anomalie saisonnière


class ForecastMethod(Enum):
    """Méthodes de prédiction."""
    LINEAR = "linear"
    ARIMA = "arima"
    EXPONENTIAL = "exponential"
    LSTM = "lstm"
    PROPHET = "prophet"
    ENSEMBLE = "ensemble"


class AnalysisType(Enum):
    """Types d'analyses."""
    ANOMALY_DETECTION = "anomaly_detection"
    TREND_ANALYSIS = "trend_analysis"
    FORECASTING = "forecasting"
    CLUSTERING = "clustering"
    CORRELATION = "correlation"
    BEHAVIORAL = "behavioral"
    PERFORMANCE = "performance"
    BUSINESS_INTELLIGENCE = "business_intelligence"


@dataclass
class AnalysisResult:
    """Résultat d'une analyse."""
    analysis_type: AnalysisType
    timestamp: datetime
    tenant_id: str
    confidence: float
    data: Dict[str, Any]
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimeSeriesData:
    """Données de série temporelle."""
    timestamps: List[datetime]
    values: List[float]
    metric_name: str
    tenant_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convertit en DataFrame pandas."""
        return pd.DataFrame({
            'timestamp': self.timestamps,
            'value': self.values
        }).set_index('timestamp')


# =============================================================================
# CLASSE BASE POUR LES ALGORITHMES
# =============================================================================

class BaseAnalyticsAlgorithm(ABC):
    """Classe de base pour tous les algorithmes d'analyse."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.is_trained = False
        self.model = None
        self.scaler = None
        self.last_training = None
        
    @abstractmethod
    async def analyze(self, data: TimeSeriesData) -> AnalysisResult:
        """Analyse les données et retourne le résultat."""
        pass
    
    @abstractmethod
    def train(self, training_data: List[TimeSeriesData]) -> bool:
        """Entraîne l'algorithme avec des données d'entraînement."""
        pass
    
    def save_model(self, path: str) -> bool:
        """Sauvegarde le modèle entraîné."""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'config': self.config,
                'last_training': self.last_training
            }
            joblib.dump(model_data, path)
            return True
        except Exception as e:
            logger.error(f"Erreur sauvegarde modèle: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """Charge un modèle sauvegardé."""
        try:
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.config.update(model_data.get('config', {}))
            self.last_training = model_data.get('last_training')
            self.is_trained = True
            return True
        except Exception as e:
            logger.error(f"Erreur chargement modèle: {e}")
            return False


# =============================================================================
# DÉTECTION D'ANOMALIES
# =============================================================================

class IsolationForestAnomalyDetector(BaseAnalyticsAlgorithm):
    """Détecteur d'anomalies basé sur Isolation Forest."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.contamination = self.config.get('contamination', 0.1)
        self.n_estimators = self.config.get('n_estimators', 100)
        self.window_size = self.config.get('window_size', 50)
        
    def train(self, training_data: List[TimeSeriesData]) -> bool:
        """Entraîne le détecteur d'anomalies."""
        if not ML_AVAILABLE:
            logger.error("Sklearn non disponible")
            return False
        
        try:
            # Préparer les données d'entraînement
            features = []
            for ts_data in training_data:
                df = ts_data.to_dataframe()
                if len(df) < self.window_size:
                    continue
                
                # Extraire des features statistiques par fenêtre glissante
                for i in range(self.window_size, len(df)):
                    window = df.iloc[i-self.window_size:i]['value']
                    feature_vector = self._extract_features(window)
                    features.append(feature_vector)
            
            if not features:
                logger.error("Pas assez de données pour l'entraînement")
                return False
            
            features_array = np.array(features)
            
            # Normalisation
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features_array)
            
            # Entraînement du modèle
            self.model = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=42
            )
            self.model.fit(features_scaled)
            
            self.is_trained = True
            self.last_training = datetime.utcnow()
            
            logger.info(f"Modèle d'anomalies entraîné avec {len(features)} échantillons")
            return True
            
        except Exception as e:
            logger.error(f"Erreur entraînement détecteur anomalies: {e}")
            return False
    
    async def analyze(self, data: TimeSeriesData) -> AnalysisResult:
        """Détecte les anomalies dans les données."""
        if not self.is_trained:
            return AnalysisResult(
                analysis_type=AnalysisType.ANOMALY_DETECTION,
                timestamp=datetime.utcnow(),
                tenant_id=data.tenant_id,
                confidence=0.0,
                data={},
                insights=["Modèle non entraîné"]
            )
        
        try:
            df = data.to_dataframe()
            if len(df) < self.window_size:
                return AnalysisResult(
                    analysis_type=AnalysisType.ANOMALY_DETECTION,
                    timestamp=datetime.utcnow(),
                    tenant_id=data.tenant_id,
                    confidence=0.0,
                    data={},
                    insights=["Pas assez de données pour l'analyse"]
                )
            
            anomalies = []
            anomaly_scores = []
            
            # Analyser par fenêtre glissante
            for i in range(self.window_size, len(df)):
                window = df.iloc[i-self.window_size:i]['value']
                feature_vector = self._extract_features(window)
                
                # Normaliser et prédire
                feature_scaled = self.scaler.transform([feature_vector])
                is_anomaly = self.model.predict(feature_scaled)[0] == -1
                anomaly_score = self.model.decision_function(feature_scaled)[0]
                
                if is_anomaly:
                    anomalies.append({
                        'timestamp': df.index[i],
                        'value': df.iloc[i]['value'],
                        'anomaly_score': anomaly_score,
                        'type': self._classify_anomaly_type(window, df.iloc[i]['value'])
                    })
                
                anomaly_scores.append(abs(anomaly_score))
            
            # Calculer la confiance
            confidence = 1.0 - (len(anomalies) / max(len(anomaly_scores), 1))
            
            # Générer des insights
            insights = self._generate_anomaly_insights(anomalies, df)
            recommendations = self._generate_anomaly_recommendations(anomalies)
            
            return AnalysisResult(
                analysis_type=AnalysisType.ANOMALY_DETECTION,
                timestamp=datetime.utcnow(),
                tenant_id=data.tenant_id,
                confidence=confidence,
                data={
                    'anomalies': anomalies,
                    'total_anomalies': len(anomalies),
                    'anomaly_rate': len(anomalies) / len(anomaly_scores),
                    'avg_anomaly_score': np.mean(anomaly_scores),
                    'max_anomaly_score': np.max(anomaly_scores)
                },
                insights=insights,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Erreur détection anomalies: {e}")
            return AnalysisResult(
                analysis_type=AnalysisType.ANOMALY_DETECTION,
                timestamp=datetime.utcnow(),
                tenant_id=data.tenant_id,
                confidence=0.0,
                data={},
                insights=[f"Erreur analyse: {str(e)}"]
            )
    
    def _extract_features(self, window: pd.Series) -> List[float]:
        """Extrait des features statistiques d'une fenêtre."""
        features = [
            window.mean(),  # Moyenne
            window.std(),   # Écart-type
            window.min(),   # Minimum
            window.max(),   # Maximum
            window.median(), # Médiane
            stats.skew(window),  # Asymétrie
            stats.kurtosis(window),  # Kurtosis
            len(find_peaks(window)[0]),  # Nombre de pics
            window.diff().mean(),  # Tendance moyenne
            window.rolling(5).std().mean()  # Volatilité moyenne
        ]
        
        # Remplacer les NaN par 0
        return [f if not math.isnan(f) else 0.0 for f in features]
    
    def _classify_anomaly_type(self, window: pd.Series, current_value: float) -> str:
        """Classifie le type d'anomalie."""
        window_mean = window.mean()
        window_std = window.std()
        
        # Anomalie de valeur extrême
        if abs(current_value - window_mean) > 3 * window_std:
            return "extreme_value"
        
        # Anomalie de tendance
        if window.diff().mean() > 2 * window.diff().std():
            return "trend_change"
        
        # Anomalie de volatilité
        if window.rolling(5).std().iloc[-1] > 2 * window.rolling(5).std().mean():
            return "volatility_spike"
        
        return "general"
    
    def _generate_anomaly_insights(self, anomalies: List[Dict], df: pd.DataFrame) -> List[str]:
        """Génère des insights sur les anomalies détectées."""
        insights = []
        
        if not anomalies:
            insights.append("Aucune anomalie détectée dans la période analysée")
            return insights
        
        # Analyse temporelle
        anomaly_times = [a['timestamp'] for a in anomalies]
        if len(set(t.hour for t in anomaly_times)) == 1:
            hour = anomaly_times[0].hour
            insights.append(f"Anomalies concentrées à {hour}h - possible pattern récurrent")
        
        # Analyse des types
        type_counts = defaultdict(int)
        for a in anomalies:
            type_counts[a['type']] += 1
        
        if type_counts:
            dominant_type = max(type_counts, key=type_counts.get)
            insights.append(f"Type d'anomalie dominant: {dominant_type} ({type_counts[dominant_type]} occurrences)")
        
        # Analyse de la sévérité
        severe_anomalies = [a for a in anomalies if a['anomaly_score'] < -0.5]
        if severe_anomalies:
            insights.append(f"{len(severe_anomalies)} anomalies sévères détectées")
        
        return insights
    
    def _generate_anomaly_recommendations(self, anomalies: List[Dict]) -> List[str]:
        """Génère des recommandations basées sur les anomalies."""
        recommendations = []
        
        if not anomalies:
            recommendations.append("Maintenir la surveillance actuelle")
            return recommendations
        
        if len(anomalies) > 5:
            recommendations.append("Augmenter la fréquence de surveillance")
            recommendations.append("Analyser les causes sous-jacentes des anomalies fréquentes")
        
        severe_count = len([a for a in anomalies if a['anomaly_score'] < -0.5])
        if severe_count > 0:
            recommendations.append("Investigation immédiate requise pour les anomalies sévères")
        
        # Recommandations par type
        type_counts = defaultdict(int)
        for a in anomalies:
            type_counts[a['type']] += 1
        
        if type_counts.get('trend_change', 0) > 2:
            recommendations.append("Revoir les modèles de prédiction de tendances")
        
        if type_counts.get('volatility_spike', 0) > 2:
            recommendations.append("Analyser les facteurs de volatilité du système")
        
        return recommendations


# =============================================================================
# PRÉDICTION ET FORECASTING
# =============================================================================

class TimeSeriesForecaster(BaseAnalyticsAlgorithm):
    """Système de prédiction de séries temporelles."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.method = ForecastMethod(self.config.get('method', 'arima'))
        self.forecast_horizon = self.config.get('forecast_horizon', 24)
        self.seasonal_periods = self.config.get('seasonal_periods', 24)
        
    def train(self, training_data: List[TimeSeriesData]) -> bool:
        """Entraîne le modèle de prédiction."""
        try:
            # Combiner toutes les données d'entraînement
            all_values = []
            all_timestamps = []
            
            for ts_data in training_data:
                all_values.extend(ts_data.values)
                all_timestamps.extend(ts_data.timestamps)
            
            # Créer un DataFrame ordonné
            df = pd.DataFrame({
                'timestamp': all_timestamps,
                'value': all_values
            }).sort_values('timestamp').set_index('timestamp')
            
            # Rééchantillonner si nécessaire
            df = df.resample('H').mean().fillna(method='ffill')
            
            if len(df) < 48:  # Minimum 48 heures de données
                logger.error("Pas assez de données pour l'entraînement du forecaster")
                return False
            
            # Entraîner selon la méthode choisie
            if self.method == ForecastMethod.ARIMA:
                self.model = self._train_arima(df['value'])
            elif self.method == ForecastMethod.EXPONENTIAL:
                self.model = self._train_exponential_smoothing(df['value'])
            elif self.method == ForecastMethod.LINEAR:
                self.model = self._train_linear_regression(df['value'])
            else:
                logger.error(f"Méthode de prédiction non supportée: {self.method}")
                return False
            
            self.is_trained = True
            self.last_training = datetime.utcnow()
            
            logger.info(f"Modèle de prédiction {self.method.value} entraîné")
            return True
            
        except Exception as e:
            logger.error(f"Erreur entraînement forecaster: {e}")
            return False
    
    async def analyze(self, data: TimeSeriesData) -> AnalysisResult:
        """Génère des prédictions basées sur les données."""
        if not self.is_trained:
            return AnalysisResult(
                analysis_type=AnalysisType.FORECASTING,
                timestamp=datetime.utcnow(),
                tenant_id=data.tenant_id,
                confidence=0.0,
                data={},
                insights=["Modèle non entraîné"]
            )
        
        try:
            df = data.to_dataframe()
            if len(df) < 24:  # Minimum 24 points pour une prédiction fiable
                return AnalysisResult(
                    analysis_type=AnalysisType.FORECASTING,
                    timestamp=datetime.utcnow(),
                    tenant_id=data.tenant_id,
                    confidence=0.0,
                    data={},
                    insights=["Pas assez de données historiques pour la prédiction"]
                )
            
            # Générer les prédictions
            forecast_values, confidence_intervals = self._generate_forecast(df['value'])
            
            # Créer les timestamps futurs
            last_timestamp = df.index[-1]
            future_timestamps = [
                last_timestamp + timedelta(hours=i+1) 
                for i in range(self.forecast_horizon)
            ]
            
            # Analyser les tendances
            trend_analysis = self._analyze_trend(df['value'], forecast_values)
            
            # Calculer la confiance
            confidence = self._calculate_forecast_confidence(df['value'], forecast_values)
            
            # Générer insights et recommandations
            insights = self._generate_forecast_insights(df['value'], forecast_values, trend_analysis)
            recommendations = self._generate_forecast_recommendations(trend_analysis)
            
            return AnalysisResult(
                analysis_type=AnalysisType.FORECASTING,
                timestamp=datetime.utcnow(),
                tenant_id=data.tenant_id,
                confidence=confidence,
                data={
                    'forecast_timestamps': [ts.isoformat() for ts in future_timestamps],
                    'forecast_values': forecast_values.tolist(),
                    'confidence_intervals': {
                        'lower': confidence_intervals[0].tolist(),
                        'upper': confidence_intervals[1].tolist()
                    },
                    'trend_analysis': trend_analysis,
                    'forecast_horizon_hours': self.forecast_horizon,
                    'method_used': self.method.value
                },
                insights=insights,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Erreur génération prédictions: {e}")
            return AnalysisResult(
                analysis_type=AnalysisType.FORECASTING,
                timestamp=datetime.utcnow(),
                tenant_id=data.tenant_id,
                confidence=0.0,
                data={},
                insights=[f"Erreur prédiction: {str(e)}"]
            )
    
    def _train_arima(self, series: pd.Series):
        """Entraîne un modèle ARIMA."""
        if not ML_AVAILABLE:
            raise ImportError("Statsmodels requis pour ARIMA")
        
        # Auto-détection des paramètres ARIMA
        best_aic = float('inf')
        best_model = None
        
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_model = fitted_model
                    except:
                        continue
        
        if best_model is None:
            # Fallback vers modèle simple
            model = ARIMA(series, order=(1, 1, 1))
            best_model = model.fit()
        
        return best_model
    
    def _train_exponential_smoothing(self, series: pd.Series):
        """Entraîne un modèle de lissage exponentiel."""
        if not ML_AVAILABLE:
            raise ImportError("Statsmodels requis pour Exponential Smoothing")
        
        try:
            model = ExponentialSmoothing(
                series,
                trend='add',
                seasonal='add',
                seasonal_periods=min(self.seasonal_periods, len(series) // 2)
            )
            return model.fit()
        except:
            # Fallback vers modèle simple
            model = ExponentialSmoothing(series, trend='add')
            return model.fit()
    
    def _train_linear_regression(self, series: pd.Series):
        """Entraîne un modèle de régression linéaire."""
        if not ML_AVAILABLE:
            raise ImportError("Sklearn requis pour Linear Regression")
        
        # Préparer les features temporelles
        X = np.arange(len(series)).reshape(-1, 1)
        y = series.values
        
        model = LinearRegression()
        model.fit(X, y)
        
        return model
    
    def _generate_forecast(self, series: pd.Series) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Génère les prédictions."""
        if self.method == ForecastMethod.ARIMA:
            forecast = self.model.forecast(steps=self.forecast_horizon)
            conf_int = self.model.get_forecast(steps=self.forecast_horizon).conf_int()
            return forecast, (conf_int.iloc[:, 0].values, conf_int.iloc[:, 1].values)
        
        elif self.method == ForecastMethod.EXPONENTIAL:
            forecast = self.model.forecast(steps=self.forecast_horizon)
            # Estimation simple des intervalles de confiance
            std_error = np.std(series.values[-24:])  # Erreur basée sur les 24 dernières valeurs
            lower = forecast - 1.96 * std_error
            upper = forecast + 1.96 * std_error
            return forecast, (lower, upper)
        
        elif self.method == ForecastMethod.LINEAR:
            last_index = len(series)
            future_X = np.arange(last_index, last_index + self.forecast_horizon).reshape(-1, 1)
            forecast = self.model.predict(future_X)
            
            # Estimation des intervalles de confiance
            residuals = series.values - self.model.predict(np.arange(len(series)).reshape(-1, 1))
            std_error = np.std(residuals)
            lower = forecast - 1.96 * std_error
            upper = forecast + 1.96 * std_error
            
            return forecast, (lower, upper)
        
        else:
            raise ValueError(f"Méthode non supportée: {self.method}")
    
    def _analyze_trend(self, historical: pd.Series, forecast: np.ndarray) -> Dict[str, Any]:
        """Analyse les tendances dans les données."""
        # Tendance historique
        historical_trend = np.polyfit(range(len(historical)), historical.values, 1)[0]
        
        # Tendance prévue
        forecast_trend = np.polyfit(range(len(forecast)), forecast, 1)[0]
        
        # Classification de la tendance
        def classify_trend(slope):
            if abs(slope) < 0.1:
                return "stable"
            elif slope > 0:
                return "croissante"
            else:
                return "décroissante"
        
        return {
            'historical_trend_slope': float(historical_trend),
            'forecast_trend_slope': float(forecast_trend),
            'historical_trend_type': classify_trend(historical_trend),
            'forecast_trend_type': classify_trend(forecast_trend),
            'trend_change': abs(forecast_trend - historical_trend) > 0.5,
            'forecast_min': float(np.min(forecast)),
            'forecast_max': float(np.max(forecast)),
            'forecast_mean': float(np.mean(forecast)),
            'volatility': float(np.std(forecast))
        }
    
    def _calculate_forecast_confidence(self, historical: pd.Series, forecast: np.ndarray) -> float:
        """Calcule la confiance dans les prédictions."""
        # Basé sur la stabilité des données historiques et la cohérence des prédictions
        historical_stability = 1.0 / (1.0 + np.std(historical.values) / np.mean(historical.values))
        forecast_consistency = 1.0 / (1.0 + np.std(forecast) / np.mean(forecast))
        
        # Confiance basée sur la longueur des données historiques
        data_confidence = min(1.0, len(historical) / 168)  # 168 heures = 1 semaine
        
        return float(np.mean([historical_stability, forecast_consistency, data_confidence]))
    
    def _generate_forecast_insights(self, historical: pd.Series, forecast: np.ndarray, 
                                  trend_analysis: Dict[str, Any]) -> List[str]:
        """Génère des insights sur les prédictions."""
        insights = []
        
        # Insights sur la tendance
        if trend_analysis['trend_change']:
            insights.append(f"Changement de tendance détecté: de {trend_analysis['historical_trend_type']} "
                          f"à {trend_analysis['forecast_trend_type']}")
        else:
            insights.append(f"Tendance stable maintenue: {trend_analysis['forecast_trend_type']}")
        
        # Insights sur les valeurs extrêmes
        current_mean = historical.mean()
        forecast_mean = trend_analysis['forecast_mean']
        
        if abs(forecast_mean - current_mean) > current_mean * 0.2:
            change_pct = ((forecast_mean - current_mean) / current_mean) * 100
            insights.append(f"Changement significatif prévu: {change_pct:+.1f}% par rapport à la moyenne actuelle")
        
        # Insights sur la volatilité
        historical_volatility = historical.std()
        forecast_volatility = trend_analysis['volatility']
        
        if forecast_volatility > historical_volatility * 1.5:
            insights.append("Augmentation de la volatilité prévue")
        elif forecast_volatility < historical_volatility * 0.5:
            insights.append("Stabilisation prévue (diminution de la volatilité)")
        
        # Insights sur les pics et creux
        if trend_analysis['forecast_max'] > historical.max() * 1.1:
            insights.append("Nouveau pic maximum prévu")
        
        if trend_analysis['forecast_min'] < historical.min() * 0.9:
            insights.append("Nouveau minimum prévu")
        
        return insights
    
    def _generate_forecast_recommendations(self, trend_analysis: Dict[str, Any]) -> List[str]:
        """Génère des recommandations basées sur les prédictions."""
        recommendations = []
        
        # Recommandations basées sur la tendance
        if trend_analysis['forecast_trend_type'] == 'croissante':
            recommendations.append("Préparer la montée en charge des ressources")
            recommendations.append("Surveiller les limites de capacité")
        elif trend_analysis['forecast_trend_type'] == 'décroissante':
            recommendations.append("Analyser les causes de la baisse")
            recommendations.append("Envisager l'optimisation des ressources")
        
        # Recommandations basées sur la volatilité
        if trend_analysis['volatility'] > trend_analysis['forecast_mean'] * 0.3:
            recommendations.append("Augmenter la fréquence de monitoring")
            recommendations.append("Mettre en place des alertes proactives")
        
        # Recommandations basées sur les changements
        if trend_analysis['trend_change']:
            recommendations.append("Revoir les modèles de prédiction existants")
            recommendations.append("Analyser les facteurs causant le changement de tendance")
        
        return recommendations


# =============================================================================
# ANALYSE DE CLUSTERING
# =============================================================================

class BehavioralClusteringAnalyzer(BaseAnalyticsAlgorithm):
    """Analyseur de clustering comportemental."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.n_clusters = self.config.get('n_clusters', 5)
        self.algorithm = self.config.get('algorithm', 'kmeans')  # kmeans, dbscan
        self.feature_columns = self.config.get('feature_columns', [])
        
    def train(self, training_data: List[TimeSeriesData]) -> bool:
        """Entraîne le modèle de clustering."""
        if not ML_AVAILABLE:
            logger.error("Sklearn non disponible pour le clustering")
            return False
        
        try:
            # Préparer les features pour le clustering
            features_list = []
            tenant_ids = []
            
            for ts_data in training_data:
                features = self._extract_behavioral_features(ts_data)
                if features:
                    features_list.append(features)
                    tenant_ids.append(ts_data.tenant_id)
            
            if len(features_list) < 3:
                logger.error("Pas assez de données pour le clustering")
                return False
            
            features_array = np.array(features_list)
            
            # Normalisation
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features_array)
            
            # Entraînement du modèle de clustering
            if self.algorithm == 'kmeans':
                self.model = KMeans(n_clusters=self.n_clusters, random_state=42)
            elif self.algorithm == 'dbscan':
                self.model = DBSCAN(eps=0.5, min_samples=2)
            else:
                logger.error(f"Algorithme de clustering non supporté: {self.algorithm}")
                return False
            
            cluster_labels = self.model.fit_predict(features_scaled)
            
            # Analyser la qualité du clustering
            if len(set(cluster_labels)) > 1:
                silhouette_avg = silhouette_score(features_scaled, cluster_labels)
                logger.info(f"Score de silhouette: {silhouette_avg:.3f}")
            
            self.is_trained = True
            self.last_training = datetime.utcnow()
            
            logger.info(f"Modèle de clustering {self.algorithm} entraîné avec {len(features_list)} échantillons")
            return True
            
        except Exception as e:
            logger.error(f"Erreur entraînement clustering: {e}")
            return False
    
    async def analyze(self, data: TimeSeriesData) -> AnalysisResult:
        """Analyse les clusters comportementaux."""
        if not self.is_trained:
            return AnalysisResult(
                analysis_type=AnalysisType.CLUSTERING,
                timestamp=datetime.utcnow(),
                tenant_id=data.tenant_id,
                confidence=0.0,
                data={},
                insights=["Modèle non entraîné"]
            )
        
        try:
            # Extraire les features comportementales
            features = self._extract_behavioral_features(data)
            if not features:
                return AnalysisResult(
                    analysis_type=AnalysisType.CLUSTERING,
                    timestamp=datetime.utcnow(),
                    tenant_id=data.tenant_id,
                    confidence=0.0,
                    data={},
                    insights=["Impossible d'extraire les features comportementales"]
                )
            
            # Normaliser et prédire le cluster
            features_scaled = self.scaler.transform([features])
            cluster_label = self.model.predict(features_scaled)[0]
            
            # Analyser les caractéristiques du cluster
            cluster_analysis = self._analyze_cluster_characteristics(cluster_label, features)
            
            # Calculer la confiance
            confidence = self._calculate_clustering_confidence(features_scaled, cluster_label)
            
            # Générer insights et recommandations
            insights = self._generate_clustering_insights(cluster_label, cluster_analysis)
            recommendations = self._generate_clustering_recommendations(cluster_label, cluster_analysis)
            
            return AnalysisResult(
                analysis_type=AnalysisType.CLUSTERING,
                timestamp=datetime.utcnow(),
                tenant_id=data.tenant_id,
                confidence=confidence,
                data={
                    'cluster_label': int(cluster_label),
                    'cluster_analysis': cluster_analysis,
                    'behavioral_features': dict(zip(self._get_feature_names(), features)),
                    'algorithm_used': self.algorithm
                },
                insights=insights,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Erreur analyse clustering: {e}")
            return AnalysisResult(
                analysis_type=AnalysisType.CLUSTERING,
                timestamp=datetime.utcnow(),
                tenant_id=data.tenant_id,
                confidence=0.0,
                data={},
                insights=[f"Erreur clustering: {str(e)}"]
            )
    
    def _extract_behavioral_features(self, data: TimeSeriesData) -> List[float]:
        """Extrait les features comportementales."""
        try:
            df = data.to_dataframe()
            if len(df) < 24:  # Minimum 24 points
                return None
            
            values = df['value'].values
            
            # Features statistiques de base
            features = [
                np.mean(values),  # Moyenne
                np.std(values),   # Écart-type
                np.median(values), # Médiane
                stats.skew(values), # Asymétrie
                stats.kurtosis(values), # Kurtosis
            ]
            
            # Features temporelles
            if len(df) >= 24:
                # Analyse par heure de la journée
                df_hourly = df.groupby(df.index.hour)['value'].mean()
                features.extend([
                    df_hourly.std(),  # Variabilité horaire
                    df_hourly.max() - df_hourly.min(),  # Amplitude horaire
                ])
                
                # Pattern jour/nuit (en supposant que les heures 22-6 sont la nuit)
                night_hours = [22, 23, 0, 1, 2, 3, 4, 5, 6]
                night_values = df[df.index.hour.isin(night_hours)]['value']
                day_values = df[~df.index.hour.isin(night_hours)]['value']
                
                if len(night_values) > 0 and len(day_values) > 0:
                    features.append(night_values.mean() / day_values.mean())  # Ratio nuit/jour
                else:
                    features.append(1.0)
            else:
                features.extend([0.0, 0.0, 1.0])
            
            # Features de tendance
            if len(values) > 1:
                trend_slope = np.polyfit(range(len(values)), values, 1)[0]
                features.append(trend_slope)
            else:
                features.append(0.0)
            
            # Features de volatilité
            if len(values) > 2:
                rolling_volatility = pd.Series(values).rolling(window=min(6, len(values)//2)).std()
                features.extend([
                    rolling_volatility.mean(),  # Volatilité moyenne
                    rolling_volatility.std(),   # Volatilité de la volatilité
                ])
            else:
                features.extend([0.0, 0.0])
            
            # Features de périodicité
            if len(values) >= 12:
                # Décomposition simple pour détecter la saisonnalité
                try:
                    decomposition = seasonal_decompose(pd.Series(values), period=min(12, len(values)//2), model='additive')
                    seasonal_strength = decomposition.seasonal.std() / decomposition.observed.std()
                    features.append(seasonal_strength)
                except:
                    features.append(0.0)
            else:
                features.append(0.0)
            
            # Remplacer les NaN et inf par des valeurs par défaut
            features = [f if np.isfinite(f) else 0.0 for f in features]
            
            return features
            
        except Exception as e:
            logger.error(f"Erreur extraction features comportementales: {e}")
            return None
    
    def _get_feature_names(self) -> List[str]:
        """Retourne les noms des features."""
        return [
            'mean', 'std', 'median', 'skewness', 'kurtosis',
            'hourly_variability', 'hourly_amplitude', 'night_day_ratio',
            'trend_slope', 'avg_volatility', 'volatility_volatility',
            'seasonal_strength'
        ]
    
    def _analyze_cluster_characteristics(self, cluster_label: int, features: List[float]) -> Dict[str, Any]:
        """Analyse les caractéristiques du cluster."""
        feature_names = self._get_feature_names()
        
        # Caractéristiques de base
        characteristics = {
            'cluster_id': cluster_label,
            'activity_level': 'high' if features[0] > np.median(features[:5]) else 'low',
            'volatility_level': 'high' if features[1] > np.median(features[:5]) else 'low',
            'trend_direction': 'increasing' if features[8] > 0.1 else 'decreasing' if features[8] < -0.1 else 'stable',
            'pattern_type': 'seasonal' if features[-1] > 0.3 else 'irregular'
        }
        
        # Classification comportementale
        if features[0] > 0.8 and features[1] < 0.2:  # Haute moyenne, faible volatilité
            characteristics['behavior_type'] = 'stable_high_usage'
        elif features[0] < 0.2 and features[1] < 0.2:  # Faible moyenne, faible volatilité
            characteristics['behavior_type'] = 'stable_low_usage'
        elif features[1] > 0.5:  # Haute volatilité
            characteristics['behavior_type'] = 'volatile_usage'
        elif features[-1] > 0.4:  # Forte saisonnalité
            characteristics['behavior_type'] = 'seasonal_pattern'
        else:
            characteristics['behavior_type'] = 'irregular_pattern'
        
        return characteristics
    
    def _calculate_clustering_confidence(self, features_scaled: np.ndarray, cluster_label: int) -> float:
        """Calcule la confiance dans l'assignation de cluster."""
        try:
            # Distance au centre du cluster
            if hasattr(self.model, 'cluster_centers_'):
                center = self.model.cluster_centers_[cluster_label]
                distance_to_center = np.linalg.norm(features_scaled[0] - center)
                
                # Normaliser la distance (confiance inversement proportionnelle)
                max_distance = 3.0  # Distance maximale raisonnable
                confidence = max(0.0, 1.0 - (distance_to_center / max_distance))
            else:
                # Pour DBSCAN, utiliser une méthode différente
                confidence = 0.8 if cluster_label != -1 else 0.2  # -1 = bruit
            
            return float(confidence)
            
        except Exception:
            return 0.5  # Confiance moyenne par défaut
    
    def _generate_clustering_insights(self, cluster_label: int, 
                                    cluster_analysis: Dict[str, Any]) -> List[str]:
        """Génère des insights sur le clustering."""
        insights = []
        
        behavior_type = cluster_analysis['behavior_type']
        
        if behavior_type == 'stable_high_usage':
            insights.append("Utilisateur à usage stable et élevé - profil premium potentiel")
        elif behavior_type == 'stable_low_usage':
            insights.append("Utilisateur à usage stable mais faible - risque de churn")
        elif behavior_type == 'volatile_usage':
            insights.append("Comportement d'usage volatil - nécessite analyse approfondie")
        elif behavior_type == 'seasonal_pattern':
            insights.append("Pattern saisonnier détecté - prévisibilité élevée")
        else:
            insights.append("Pattern d'usage irrégulier - difficile à prévoir")
        
        # Insights sur la tendance
        trend = cluster_analysis['trend_direction']
        if trend == 'increasing':
            insights.append("Tendance d'usage croissante - engagement en hausse")
        elif trend == 'decreasing':
            insights.append("Tendance d'usage décroissante - risque de désengagement")
        
        # Insights sur l'activité
        if cluster_analysis['activity_level'] == 'high':
            insights.append("Niveau d'activité élevé - utilisateur engagé")
        else:
            insights.append("Niveau d'activité modéré - potentiel d'amélioration")
        
        return insights
    
    def _generate_clustering_recommendations(self, cluster_label: int, 
                                           cluster_analysis: Dict[str, Any]) -> List[str]:
        """Génère des recommandations basées sur le clustering."""
        recommendations = []
        
        behavior_type = cluster_analysis['behavior_type']
        
        if behavior_type == 'stable_high_usage':
            recommendations.append("Proposer des fonctionnalités premium")
            recommendations.append("Utiliser comme ambassadeur pour le référencement")
        elif behavior_type == 'stable_low_usage':
            recommendations.append("Campagne de réengagement ciblée")
            recommendations.append("Analyse des obstacles à l'usage")
        elif behavior_type == 'volatile_usage':
            recommendations.append("Monitoring renforcé des patterns d'usage")
            recommendations.append("Personnalisation adaptative du contenu")
        elif behavior_type == 'seasonal_pattern':
            recommendations.append("Planification marketing basée sur la saisonnalité")
            recommendations.append("Optimisation des ressources selon les cycles")
        
        # Recommandations basées sur la tendance
        if cluster_analysis['trend_direction'] == 'decreasing':
            recommendations.append("Intervention préventive pour éviter le churn")
            recommendations.append("Analyse des facteurs de désengagement")
        
        return recommendations


# =============================================================================
# GESTIONNAIRE D'ALGORITHMES
# =============================================================================

class AnalyticsManager:
    """Gestionnaire centralisé des algorithmes d'analyse."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.algorithms = {}
        self.model_cache = {}
        self._initialize_algorithms()
    
    def _initialize_algorithms(self):
        """Initialise tous les algorithmes disponibles."""
        try:
            # Détecteur d'anomalies
            self.algorithms['anomaly_detector'] = IsolationForestAnomalyDetector(
                self.config.get('anomaly_detection', {})
            )
            
            # Système de prédiction
            self.algorithms['forecaster'] = TimeSeriesForecaster(
                self.config.get('forecasting', {})
            )
            
            # Analyseur de clustering
            self.algorithms['clustering_analyzer'] = BehavioralClusteringAnalyzer(
                self.config.get('clustering', {})
            )
            
            logger.info("Algorithmes d'analyse initialisés")
            
        except Exception as e:
            logger.error(f"Erreur initialisation algorithmes: {e}")
    
    async def run_analysis(self, analysis_type: AnalysisType, 
                          data: TimeSeriesData) -> AnalysisResult:
        """Lance une analyse spécifique."""
        try:
            if analysis_type == AnalysisType.ANOMALY_DETECTION:
                return await self.algorithms['anomaly_detector'].analyze(data)
            elif analysis_type == AnalysisType.FORECASTING:
                return await self.algorithms['forecaster'].analyze(data)
            elif analysis_type == AnalysisType.CLUSTERING:
                return await self.algorithms['clustering_analyzer'].analyze(data)
            else:
                raise ValueError(f"Type d'analyse non supporté: {analysis_type}")
                
        except Exception as e:
            logger.error(f"Erreur exécution analyse {analysis_type}: {e}")
            return AnalysisResult(
                analysis_type=analysis_type,
                timestamp=datetime.utcnow(),
                tenant_id=data.tenant_id,
                confidence=0.0,
                data={},
                insights=[f"Erreur: {str(e)}"]
            )
    
    async def run_comprehensive_analysis(self, data: TimeSeriesData) -> List[AnalysisResult]:
        """Lance une analyse complète avec tous les algorithmes."""
        analyses = []
        
        analysis_types = [
            AnalysisType.ANOMALY_DETECTION,
            AnalysisType.FORECASTING,
            AnalysisType.CLUSTERING
        ]
        
        for analysis_type in analysis_types:
            try:
                result = await self.run_analysis(analysis_type, data)
                analyses.append(result)
            except Exception as e:
                logger.error(f"Erreur analyse {analysis_type}: {e}")
        
        return analyses
    
    def train_all_algorithms(self, training_data: List[TimeSeriesData]) -> Dict[str, bool]:
        """Entraîne tous les algorithmes."""
        results = {}
        
        for name, algorithm in self.algorithms.items():
            try:
                success = algorithm.train(training_data)
                results[name] = success
                logger.info(f"Entraînement {name}: {'Succès' if success else 'Échec'}")
            except Exception as e:
                logger.error(f"Erreur entraînement {name}: {e}")
                results[name] = False
        
        return results
    
    def save_models(self, models_dir: str) -> Dict[str, bool]:
        """Sauvegarde tous les modèles entraînés."""
        import os
        os.makedirs(models_dir, exist_ok=True)
        
        results = {}
        
        for name, algorithm in self.algorithms.items():
            try:
                model_path = os.path.join(models_dir, f"{name}_model.pkl")
                success = algorithm.save_model(model_path)
                results[name] = success
            except Exception as e:
                logger.error(f"Erreur sauvegarde modèle {name}: {e}")
                results[name] = False
        
        return results
    
    def load_models(self, models_dir: str) -> Dict[str, bool]:
        """Charge tous les modèles sauvegardés."""
        import os
        results = {}
        
        for name, algorithm in self.algorithms.items():
            try:
                model_path = os.path.join(models_dir, f"{name}_model.pkl")
                if os.path.exists(model_path):
                    success = algorithm.load_model(model_path)
                    results[name] = success
                else:
                    results[name] = False
            except Exception as e:
                logger.error(f"Erreur chargement modèle {name}: {e}")
                results[name] = False
        
        return results


# =============================================================================
# UTILITAIRES D'EXPORT
# =============================================================================

def create_sample_data(tenant_id: str = "sample_tenant", 
                      duration_hours: int = 168) -> TimeSeriesData:
    """Crée des données d'exemple pour les tests."""
    start_time = datetime.utcnow() - timedelta(hours=duration_hours)
    
    timestamps = []
    values = []
    
    for i in range(duration_hours):
        timestamp = start_time + timedelta(hours=i)
        
        # Simulation d'une métrique avec pattern quotidien + bruit + tendance
        hour_of_day = timestamp.hour
        daily_pattern = 50 + 30 * math.sin(2 * math.pi * hour_of_day / 24)
        trend = 0.1 * i  # Tendance légèrement croissante
        noise = np.random.normal(0, 5)
        
        # Ajouter quelques anomalies
        if i % 50 == 0:  # Anomalie tous les ~2 jours
            noise += np.random.normal(0, 20)
        
        value = max(0, daily_pattern + trend + noise)
        
        timestamps.append(timestamp)
        values.append(value)
    
    return TimeSeriesData(
        timestamps=timestamps,
        values=values,
        metric_name="sample_metric",
        tenant_id=tenant_id,
        metadata={"generated": True, "duration_hours": duration_hours}
    )


async def main():
    """Fonction principale pour les tests."""
    print("Initialisation du système d'analyse...")
    
    # Configuration
    config = {
        'anomaly_detection': {
            'contamination': 0.1,
            'n_estimators': 100
        },
        'forecasting': {
            'method': 'arima',
            'forecast_horizon': 24
        },
        'clustering': {
            'n_clusters': 5,
            'algorithm': 'kmeans'
        }
    }
    
    # Créer le gestionnaire
    manager = AnalyticsManager(config)
    
    # Créer des données d'exemple
    print("Génération des données d'exemple...")
    training_data = [
        create_sample_data(f"tenant_{i:03d}", 168) 
        for i in range(5)
    ]
    
    # Entraîner les algorithmes
    print("Entraînement des algorithmes...")
    training_results = manager.train_all_algorithms(training_data)
    print(f"Résultats d'entraînement: {training_results}")
    
    # Tester l'analyse
    print("Test d'analyse complète...")
    test_data = create_sample_data("test_tenant", 72)
    
    results = await manager.run_comprehensive_analysis(test_data)
    
    for result in results:
        print(f"\n=== {result.analysis_type.value.upper()} ===")
        print(f"Confiance: {result.confidence:.2f}")
        print(f"Insights: {result.insights}")
        print(f"Recommandations: {result.recommendations}")
    
    print("\nAnalyse terminée avec succès!")


if __name__ == "__main__":
    asyncio.run(main())
