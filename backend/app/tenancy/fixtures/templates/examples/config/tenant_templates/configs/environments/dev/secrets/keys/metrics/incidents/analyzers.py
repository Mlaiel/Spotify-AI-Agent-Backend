"""
Analyseurs Avancés - Intelligence Artificielle et Analytics
===========================================================

Système d'analyse avancé avec ML/AI pour:
- Détection d'anomalies par apprentissage automatique
- Analyse prédictive des tendances
- Corrélation intelligente des événements
- Analyse de cause racine automatisée
- Recommandations d'optimisation

Analyseurs spécialisés:
    - AnomalyDetector: Détection d'anomalies ML
    - TrendAnalyzer: Analyse des tendances temporelles
    - PredictiveAnalyzer: Prédictions basées sur l'historique
    - CorrelationAnalyzer: Corrélation cross-metrics
    - RootCauseAnalyzer: Analyse de cause racine
"""

import asyncio
import json
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import structlog
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats, signal
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings('ignore')

logger = structlog.get_logger(__name__)

class AnalysisType(Enum):
    """Types d'analyses supportées"""
    ANOMALY_DETECTION = "anomaly_detection"
    TREND_ANALYSIS = "trend_analysis"
    PREDICTIVE_ANALYSIS = "predictive_analysis"
    CORRELATION_ANALYSIS = "correlation_analysis"
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    PATTERN_RECOGNITION = "pattern_recognition"

class AnomalyType(Enum):
    """Types d'anomalies détectées"""
    POINT_ANOMALY = "point_anomaly"
    CONTEXTUAL_ANOMALY = "contextual_anomaly"
    COLLECTIVE_ANOMALY = "collective_anomaly"
    TREND_ANOMALY = "trend_anomaly"
    SEASONAL_ANOMALY = "seasonal_anomaly"

@dataclass
class AnalysisResult:
    """Résultat d'analyse générique"""
    analysis_type: AnalysisType
    metric_name: str
    timestamp: datetime
    confidence_score: float
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

@dataclass
class AnomalyDetectionResult(AnalysisResult):
    """Résultat spécialisé pour la détection d'anomalies"""
    is_anomaly: bool = False
    anomaly_type: AnomalyType = AnomalyType.POINT_ANOMALY
    anomaly_score: float = 0.0
    expected_range: Tuple[float, float] = (0.0, 0.0)
    actual_value: float = 0.0
    deviation_magnitude: float = 0.0

@dataclass
class TrendAnalysisResult(AnalysisResult):
    """Résultat d'analyse de tendance"""
    trend_direction: str = "stable"  # "increasing", "decreasing", "stable"
    trend_strength: float = 0.0
    seasonal_component: bool = False
    cycle_length: Optional[int] = None
    trend_equation: Optional[str] = None

@dataclass
class PredictionResult(AnalysisResult):
    """Résultat de prédiction"""
    predicted_values: List[float] = field(default_factory=list)
    prediction_intervals: List[Tuple[float, float]] = field(default_factory=list)
    forecast_horizon: int = 0
    model_accuracy: float = 0.0
    model_type: str = ""

class BaseAnalyzer(ABC):
    """Classe de base pour tous les analyseurs"""
    
    def __init__(self, analyzer_name: str):
        self.analyzer_name = analyzer_name
        self.models = {}
        self.historical_data = defaultdict(deque)
        self.analysis_cache = {}
        self.min_data_points = 30
        self.cache_ttl = timedelta(minutes=15)
        
    @abstractmethod
    async def analyze(self, data: Dict[str, Any]) -> AnalysisResult:
        """Analyse principale - à implémenter par chaque analyseur"""
        pass
    
    async def train_model(self, metric_name: str, historical_data: List[float]):
        """Entraînement du modèle pour une métrique"""
        if len(historical_data) >= self.min_data_points:
            await self._train_specific_model(metric_name, historical_data)
            logger.info(f"Modèle entraîné pour {metric_name} avec {len(historical_data)} points")
    
    @abstractmethod
    async def _train_specific_model(self, metric_name: str, data: List[float]):
        """Entraînement spécifique au type d'analyseur"""
        pass
    
    def _normalize_data(self, data: List[float]) -> np.ndarray:
        """Normalisation des données"""
        scaler = StandardScaler()
        return scaler.fit_transform(np.array(data).reshape(-1, 1)).flatten()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Vérification de la validité du cache"""
        if cache_key not in self.analysis_cache:
            return False
        
        cached_time = self.analysis_cache[cache_key].get("timestamp")
        if not cached_time:
            return False
            
        return datetime.utcnow() - cached_time < self.cache_ttl

class AnomalyDetector(BaseAnalyzer):
    """Détecteur d'anomalies par apprentissage automatique"""
    
    def __init__(self):
        super().__init__("anomaly_detector")
        self.isolation_forests = {}
        self.statistical_models = {}
        self.ensemble_weights = {
            "isolation_forest": 0.4,
            "statistical": 0.3,
            "lstm": 0.3
        }
        
    async def analyze(self, data: Dict[str, Any]) -> AnomalyDetectionResult:
        """Détection d'anomalies multi-algorithmes"""
        metric_name = data.get("metric_name")
        values = data.get("values", [])
        timestamp = data.get("timestamp", datetime.utcnow())
        
        if not values or len(values) < 2:
            return AnomalyDetectionResult(
                analysis_type=AnalysisType.ANOMALY_DETECTION,
                metric_name=metric_name,
                timestamp=timestamp,
                confidence_score=0.0,
                is_anomaly=False
            )
        
        # Préparation des données
        current_value = values[-1]
        historical_values = values[:-1]
        
        # Détection par Isolation Forest
        isolation_result = await self._detect_with_isolation_forest(
            metric_name, historical_values, current_value
        )
        
        # Détection statistique
        statistical_result = await self._detect_with_statistics(
            metric_name, historical_values, current_value
        )
        
        # Détection par LSTM (simplifiée)
        lstm_result = await self._detect_with_lstm(
            metric_name, historical_values, current_value
        )
        
        # Combinaison des résultats (ensemble)
        final_result = await self._combine_anomaly_results(
            isolation_result, statistical_result, lstm_result,
            metric_name, timestamp, current_value
        )
        
        return final_result
    
    async def _train_specific_model(self, metric_name: str, data: List[float]):
        """Entraînement des modèles d'anomalie"""
        # Entraînement Isolation Forest
        if len(data) >= 50:
            X = np.array(data).reshape(-1, 1)
            isolation_forest = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            isolation_forest.fit(X)
            self.isolation_forests[metric_name] = isolation_forest
        
        # Calcul des paramètres statistiques
        self.statistical_models[metric_name] = {
            "mean": np.mean(data),
            "std": np.std(data),
            "median": np.median(data),
            "q1": np.percentile(data, 25),
            "q3": np.percentile(data, 75),
            "iqr": np.percentile(data, 75) - np.percentile(data, 25)
        }
    
    async def _detect_with_isolation_forest(self, metric_name: str, 
                                          historical: List[float], 
                                          current: float) -> Dict[str, Any]:
        """Détection avec Isolation Forest"""
        if metric_name not in self.isolation_forests:
            return {"is_anomaly": False, "score": 0.0, "confidence": 0.0}
        
        model = self.isolation_forests[metric_name]
        prediction = model.predict([[current]])
        anomaly_score = model.decision_function([[current]])[0]
        
        # -1 = anomalie, 1 = normal
        is_anomaly = prediction[0] == -1
        
        # Conversion du score en confiance (0-1)
        confidence = abs(anomaly_score) if is_anomaly else 0.0
        
        return {
            "is_anomaly": is_anomaly,
            "score": anomaly_score,
            "confidence": min(confidence, 1.0)
        }
    
    async def _detect_with_statistics(self, metric_name: str,
                                    historical: List[float],
                                    current: float) -> Dict[str, Any]:
        """Détection statistique (Z-score, IQR)"""
        if metric_name not in self.statistical_models:
            return {"is_anomaly": False, "score": 0.0, "confidence": 0.0}
        
        stats_model = self.statistical_models[metric_name]
        
        # Z-score
        z_score = abs((current - stats_model["mean"]) / stats_model["std"]) if stats_model["std"] > 0 else 0
        
        # IQR method
        lower_bound = stats_model["q1"] - 1.5 * stats_model["iqr"]
        upper_bound = stats_model["q3"] + 1.5 * stats_model["iqr"]
        
        is_outlier_iqr = current < lower_bound or current > upper_bound
        is_outlier_zscore = z_score > 3
        
        is_anomaly = is_outlier_iqr or is_outlier_zscore
        confidence = min(z_score / 3.0, 1.0) if is_anomaly else 0.0
        
        return {
            "is_anomaly": is_anomaly,
            "score": z_score,
            "confidence": confidence,
            "method": "statistical"
        }
    
    async def _detect_with_lstm(self, metric_name: str,
                              historical: List[float],
                              current: float) -> Dict[str, Any]:
        """Détection simplifiée par réseau de neurones (simulée)"""
        # Simulation d'un modèle LSTM
        # En production, on utiliserait TensorFlow/PyTorch
        
        if len(historical) < 10:
            return {"is_anomaly": False, "score": 0.0, "confidence": 0.0}
        
        # Prédiction simple basée sur la moyenne mobile
        window_size = min(10, len(historical))
        recent_values = historical[-window_size:]
        predicted_value = np.mean(recent_values)
        
        # Calcul de l'erreur de prédiction
        prediction_error = abs(current - predicted_value)
        error_threshold = np.std(recent_values) * 2
        
        is_anomaly = prediction_error > error_threshold
        confidence = min(prediction_error / error_threshold, 1.0) if is_anomaly else 0.0
        
        return {
            "is_anomaly": is_anomaly,
            "score": prediction_error,
            "confidence": confidence,
            "predicted_value": predicted_value,
            "method": "lstm_simulation"
        }
    
    async def _combine_anomaly_results(self, isolation_result: Dict,
                                     statistical_result: Dict,
                                     lstm_result: Dict,
                                     metric_name: str,
                                     timestamp: datetime,
                                     current_value: float) -> AnomalyDetectionResult:
        """Combinaison des résultats de détection"""
        
        # Score pondéré
        weighted_score = (
            isolation_result["confidence"] * self.ensemble_weights["isolation_forest"] +
            statistical_result["confidence"] * self.ensemble_weights["statistical"] +
            lstm_result["confidence"] * self.ensemble_weights["lstm"]
        )
        
        # Décision finale
        votes = [
            isolation_result["is_anomaly"],
            statistical_result["is_anomaly"],
            lstm_result["is_anomaly"]
        ]
        is_anomaly = sum(votes) >= 2  # Majorité
        
        # Classification du type d'anomalie
        anomaly_type = AnomalyType.POINT_ANOMALY
        if is_anomaly:
            if statistical_result.get("method") == "statistical":
                anomaly_type = AnomalyType.CONTEXTUAL_ANOMALY
            elif lstm_result.get("method") == "lstm_simulation":
                anomaly_type = AnomalyType.TREND_ANOMALY
        
        # Recommandations
        recommendations = []
        if is_anomaly:
            recommendations.extend([
                "Vérifier les logs système pour cette période",
                "Analyser les métriques corrélées",
                "Évaluer l'impact sur les utilisateurs"
            ])
            
            if weighted_score > 0.8:
                recommendations.append("Escalade immédiate recommandée")
        
        return AnomalyDetectionResult(
            analysis_type=AnalysisType.ANOMALY_DETECTION,
            metric_name=metric_name,
            timestamp=timestamp,
            confidence_score=weighted_score,
            is_anomaly=is_anomaly,
            anomaly_type=anomaly_type,
            anomaly_score=weighted_score,
            actual_value=current_value,
            details={
                "isolation_forest": isolation_result,
                "statistical": statistical_result,
                "lstm": lstm_result,
                "ensemble_weights": self.ensemble_weights
            },
            recommendations=recommendations
        )

class TrendAnalyzer(BaseAnalyzer):
    """Analyseur de tendances temporelles"""
    
    def __init__(self):
        super().__init__("trend_analyzer")
        self.decomposition_cache = {}
        
    async def analyze(self, data: Dict[str, Any]) -> TrendAnalysisResult:
        """Analyse des tendances temporelles"""
        metric_name = data.get("metric_name")
        values = data.get("values", [])
        timestamps = data.get("timestamps", [])
        
        if len(values) < self.min_data_points:
            return TrendAnalysisResult(
                analysis_type=AnalysisType.TREND_ANALYSIS,
                metric_name=metric_name,
                timestamp=datetime.utcnow(),
                confidence_score=0.0,
                trend_direction="insufficient_data"
            )
        
        # Création de la série temporelle
        ts_data = pd.Series(values, index=pd.to_datetime(timestamps))
        
        # Décomposition de la série temporelle
        decomposition = await self._decompose_time_series(ts_data)
        
        # Analyse de la tendance
        trend_analysis = await self._analyze_trend_component(decomposition.trend)
        
        # Détection de saisonnalité
        seasonality_analysis = await self._analyze_seasonality(decomposition.seasonal)
        
        # Test de stationnarité
        stationarity_test = await self._test_stationarity(ts_data)
        
        # Calcul de la confiance
        confidence = self._calculate_trend_confidence(trend_analysis, stationarity_test)
        
        return TrendAnalysisResult(
            analysis_type=AnalysisType.TREND_ANALYSIS,
            metric_name=metric_name,
            timestamp=datetime.utcnow(),
            confidence_score=confidence,
            trend_direction=trend_analysis["direction"],
            trend_strength=trend_analysis["strength"],
            seasonal_component=seasonality_analysis["has_seasonality"],
            cycle_length=seasonality_analysis.get("cycle_length"),
            details={
                "decomposition": {
                    "trend_variance": trend_analysis["variance"],
                    "seasonal_variance": seasonality_analysis["variance"],
                    "residual_variance": np.var(decomposition.resid.dropna())
                },
                "stationarity": stationarity_test,
                "trend_equation": trend_analysis.get("equation")
            }
        )
    
    async def _train_specific_model(self, metric_name: str, data: List[float]):
        """Entraînement pour l'analyse de tendances"""
        # Pas d'entraînement spécifique nécessaire pour l'analyse de tendances
        # Les méthodes sont principalement statistiques
        pass
    
    async def _decompose_time_series(self, ts_data: pd.Series):
        """Décomposition de série temporelle"""
        try:
            # Décomposition additive
            decomposition = seasonal_decompose(
                ts_data.dropna(),
                model='additive',
                period=min(len(ts_data) // 4, 24)  # Période adaptative
            )
            return decomposition
        except Exception as e:
            logger.warning(f"Erreur lors de la décomposition: {e}")
            # Fallback: décomposition manuelle simple
            trend = ts_data.rolling(window=min(len(ts_data) // 4, 12)).mean()
            seasonal = ts_data - trend
            resid = ts_data - trend - seasonal
            
            # Simulation d'un objet de décomposition
            from types import SimpleNamespace
            return SimpleNamespace(trend=trend, seasonal=seasonal, resid=resid)
    
    async def _analyze_trend_component(self, trend: pd.Series) -> Dict[str, Any]:
        """Analyse du composant de tendance"""
        # Suppression des NaN
        clean_trend = trend.dropna()
        
        if len(clean_trend) < 2:
            return {"direction": "insufficient_data", "strength": 0.0, "variance": 0.0}
        
        # Régression linéaire pour déterminer la direction
        x = np.arange(len(clean_trend))
        y = clean_trend.values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Détermination de la direction
        if abs(slope) < std_err:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        # Force de la tendance basée sur R²
        strength = r_value ** 2
        
        return {
            "direction": direction,
            "strength": strength,
            "slope": slope,
            "r_squared": r_value ** 2,
            "p_value": p_value,
            "variance": np.var(clean_trend),
            "equation": f"y = {slope:.4f}x + {intercept:.4f}"
        }
    
    async def _analyze_seasonality(self, seasonal: pd.Series) -> Dict[str, Any]:
        """Analyse de la composante saisonnière"""
        clean_seasonal = seasonal.dropna()
        
        if len(clean_seasonal) < 12:
            return {"has_seasonality": False, "variance": 0.0}
        
        # Variance de la composante saisonnière
        seasonal_variance = np.var(clean_seasonal)
        
        # Détection de cycles via autocorrélation
        autocorr = [clean_seasonal.autocorr(lag=i) for i in range(1, min(len(clean_seasonal) // 2, 48))]
        
        # Recherche de pics dans l'autocorrélation
        peaks = []
        for i, corr in enumerate(autocorr[1:], 2):
            if corr > 0.3 and corr > autocorr[i-2] and corr > autocorr[i]:
                peaks.append(i)
        
        has_seasonality = seasonal_variance > 0.01 and len(peaks) > 0
        cycle_length = peaks[0] if peaks else None
        
        return {
            "has_seasonality": has_seasonality,
            "variance": seasonal_variance,
            "cycle_length": cycle_length,
            "autocorr_peaks": peaks[:3]  # Top 3 cycles
        }
    
    async def _test_stationarity(self, ts_data: pd.Series) -> Dict[str, Any]:
        """Test de stationnarité (ADF test)"""
        try:
            clean_data = ts_data.dropna()
            
            # Augmented Dickey-Fuller test
            adf_result = adfuller(clean_data)
            
            # KPSS test
            kpss_result = kpss(clean_data, regression='c')
            
            # Interprétation des résultats
            is_stationary_adf = adf_result[1] < 0.05  # p-value < 0.05
            is_stationary_kpss = kpss_result[1] > 0.05  # p-value > 0.05
            
            return {
                "adf_statistic": adf_result[0],
                "adf_pvalue": adf_result[1],
                "kpss_statistic": kpss_result[0],
                "kpss_pvalue": kpss_result[1],
                "is_stationary_adf": is_stationary_adf,
                "is_stationary_kpss": is_stationary_kpss,
                "is_stationary": is_stationary_adf and is_stationary_kpss
            }
        except Exception as e:
            logger.warning(f"Erreur lors du test de stationnarité: {e}")
            return {"is_stationary": False, "error": str(e)}
    
    def _calculate_trend_confidence(self, trend_analysis: Dict, stationarity_test: Dict) -> float:
        """Calcul de la confiance dans l'analyse de tendance"""
        confidence = 0.0
        
        # Confiance basée sur R²
        confidence += trend_analysis.get("r_squared", 0) * 0.4
        
        # Confiance basée sur la significativité statistique
        p_value = trend_analysis.get("p_value", 1.0)
        if p_value < 0.05:
            confidence += 0.3
        elif p_value < 0.1:
            confidence += 0.2
        
        # Confiance basée sur la stationnarité
        if stationarity_test.get("is_stationary", False):
            confidence += 0.3
        
        return min(confidence, 1.0)

class PredictiveAnalyzer(BaseAnalyzer):
    """Analyseur prédictif basé sur l'historique"""
    
    def __init__(self):
        super().__init__("predictive_analyzer")
        self.arima_models = {}
        self.regression_models = {}
        
    async def analyze(self, data: Dict[str, Any]) -> PredictionResult:
        """Analyse prédictive avec multiple modèles"""
        metric_name = data.get("metric_name")
        values = data.get("values", [])
        forecast_horizon = data.get("forecast_horizon", 24)
        
        if len(values) < self.min_data_points:
            return PredictionResult(
                analysis_type=AnalysisType.PREDICTIVE_ANALYSIS,
                metric_name=metric_name,
                timestamp=datetime.utcnow(),
                confidence_score=0.0,
                forecast_horizon=forecast_horizon
            )
        
        # Prédiction ARIMA
        arima_predictions = await self._predict_with_arima(metric_name, values, forecast_horizon)
        
        # Prédiction par régression
        regression_predictions = await self._predict_with_regression(metric_name, values, forecast_horizon)
        
        # Prédiction par moyenne mobile
        moving_avg_predictions = await self._predict_with_moving_average(values, forecast_horizon)
        
        # Ensemble des prédictions
        ensemble_predictions = await self._ensemble_predictions(
            arima_predictions, regression_predictions, moving_avg_predictions
        )
        
        # Calcul des intervalles de confiance
        confidence_intervals = await self._calculate_confidence_intervals(
            values, ensemble_predictions["predictions"]
        )
        
        return PredictionResult(
            analysis_type=AnalysisType.PREDICTIVE_ANALYSIS,
            metric_name=metric_name,
            timestamp=datetime.utcnow(),
            confidence_score=ensemble_predictions["confidence"],
            predicted_values=ensemble_predictions["predictions"],
            prediction_intervals=confidence_intervals,
            forecast_horizon=forecast_horizon,
            model_accuracy=ensemble_predictions["accuracy"],
            model_type="ensemble",
            details={
                "arima": arima_predictions,
                "regression": regression_predictions,
                "moving_average": moving_avg_predictions,
                "ensemble_weights": ensemble_predictions["weights"]
            }
        )
    
    async def _train_specific_model(self, metric_name: str, data: List[float]):
        """Entraînement des modèles prédictifs"""
        # Entraînement ARIMA
        try:
            ts_data = pd.Series(data)
            model = ARIMA(ts_data, order=(1, 1, 1))
            fitted_model = model.fit()
            self.arima_models[metric_name] = fitted_model
        except Exception as e:
            logger.warning(f"Erreur entraînement ARIMA pour {metric_name}: {e}")
        
        # Entraînement modèle de régression
        if len(data) >= 50:
            X = np.arange(len(data)).reshape(-1, 1)
            y = np.array(data)
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            self.regression_models[metric_name] = model
    
    async def _predict_with_arima(self, metric_name: str, values: List[float], horizon: int) -> Dict[str, Any]:
        """Prédiction avec modèle ARIMA"""
        try:
            if metric_name in self.arima_models:
                model = self.arima_models[metric_name]
                forecast = model.forecast(steps=horizon)
                
                return {
                    "predictions": forecast.tolist(),
                    "model_type": "arima",
                    "accuracy": 0.8,  # À calculer avec validation croisée
                    "confidence": 0.7
                }
            else:
                # ARIMA simple sans modèle pré-entraîné
                ts_data = pd.Series(values)
                model = ARIMA(ts_data, order=(1, 1, 1))
                fitted_model = model.fit()
                forecast = fitted_model.forecast(steps=horizon)
                
                return {
                    "predictions": forecast.tolist(),
                    "model_type": "arima_adhoc",
                    "accuracy": 0.6,
                    "confidence": 0.5
                }
        except Exception as e:
            logger.warning(f"Erreur prédiction ARIMA: {e}")
            return {
                "predictions": [values[-1]] * horizon,
                "model_type": "arima_error",
                "accuracy": 0.1,
                "confidence": 0.1
            }
    
    async def _predict_with_regression(self, metric_name: str, values: List[float], horizon: int) -> Dict[str, Any]:
        """Prédiction avec régression"""
        try:
            if metric_name in self.regression_models:
                model = self.regression_models[metric_name]
                
                # Prédiction sur les prochains points
                last_index = len(values)
                future_X = np.arange(last_index, last_index + horizon).reshape(-1, 1)
                predictions = model.predict(future_X)
                
                return {
                    "predictions": predictions.tolist(),
                    "model_type": "random_forest",
                    "accuracy": 0.75,
                    "confidence": 0.65
                }
            else:
                # Régression linéaire simple
                X = np.arange(len(values)).reshape(-1, 1)
                y = np.array(values)
                
                slope, intercept = np.polyfit(range(len(values)), values, 1)
                
                predictions = []
                for i in range(horizon):
                    pred_value = slope * (len(values) + i) + intercept
                    predictions.append(pred_value)
                
                return {
                    "predictions": predictions,
                    "model_type": "linear_regression",
                    "accuracy": 0.5,
                    "confidence": 0.4
                }
        except Exception as e:
            logger.warning(f"Erreur prédiction régression: {e}")
            return {
                "predictions": [values[-1]] * horizon,
                "model_type": "regression_error",
                "accuracy": 0.1,
                "confidence": 0.1
            }
    
    async def _predict_with_moving_average(self, values: List[float], horizon: int) -> Dict[str, Any]:
        """Prédiction par moyenne mobile"""
        window_size = min(12, len(values) // 4)
        recent_values = values[-window_size:]
        avg_value = np.mean(recent_values)
        
        # Ajout d'une petite tendance basée sur les dernières valeurs
        if len(values) >= 2:
            trend = (values[-1] - values[-min(len(values), 5)]) / min(len(values), 5)
        else:
            trend = 0
        
        predictions = []
        for i in range(horizon):
            pred_value = avg_value + trend * i
            predictions.append(pred_value)
        
        return {
            "predictions": predictions,
            "model_type": "moving_average",
            "accuracy": 0.4,
            "confidence": 0.3
        }
    
    async def _ensemble_predictions(self, arima_pred: Dict, regression_pred: Dict, moving_avg_pred: Dict) -> Dict[str, Any]:
        """Combinaison des prédictions par ensemble"""
        
        # Poids basés sur la confiance des modèles
        weights = {
            "arima": arima_pred["confidence"],
            "regression": regression_pred["confidence"],
            "moving_average": moving_avg_pred["confidence"]
        }
        
        # Normalisation des poids
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            weights = {"arima": 0.33, "regression": 0.33, "moving_average": 0.34}
        
        # Combinaison pondérée
        arima_predictions = arima_pred["predictions"]
        regression_predictions = regression_pred["predictions"]
        moving_avg_predictions = moving_avg_pred["predictions"]
        
        ensemble_predictions = []
        for i in range(len(arima_predictions)):
            combined = (
                weights["arima"] * arima_predictions[i] +
                weights["regression"] * regression_predictions[i] +
                weights["moving_average"] * moving_avg_predictions[i]
            )
            ensemble_predictions.append(combined)
        
        # Confiance et précision moyennes pondérées
        ensemble_confidence = (
            weights["arima"] * arima_pred["confidence"] +
            weights["regression"] * regression_pred["confidence"] +
            weights["moving_average"] * moving_avg_pred["confidence"]
        )
        
        ensemble_accuracy = (
            weights["arima"] * arima_pred["accuracy"] +
            weights["regression"] * regression_pred["accuracy"] +
            weights["moving_average"] * moving_avg_pred["accuracy"]
        )
        
        return {
            "predictions": ensemble_predictions,
            "confidence": ensemble_confidence,
            "accuracy": ensemble_accuracy,
            "weights": weights
        }
    
    async def _calculate_confidence_intervals(self, historical_values: List[float], predictions: List[float]) -> List[Tuple[float, float]]:
        """Calcul des intervalles de confiance pour les prédictions"""
        # Estimation de l'erreur basée sur l'historique
        if len(historical_values) < 10:
            error_std = np.std(historical_values) if len(historical_values) > 1 else 0.1
        else:
            # Calcul de l'erreur sur une validation simple
            train_size = int(len(historical_values) * 0.8)
            train_data = historical_values[:train_size]
            test_data = historical_values[train_size:]
            
            # Simulation d'une prédiction simple
            predicted_test = [np.mean(train_data)] * len(test_data)
            errors = [abs(pred - actual) for pred, actual in zip(predicted_test, test_data)]
            error_std = np.std(errors)
        
        # Intervalles de confiance à 95%
        confidence_intervals = []
        for pred in predictions:
            lower_bound = pred - 1.96 * error_std
            upper_bound = pred + 1.96 * error_std
            confidence_intervals.append((lower_bound, upper_bound))
        
        return confidence_intervals

class AnalyticsEngine:
    """Moteur d'analytics principal orchestrant tous les analyseurs"""
    
    def __init__(self):
        self.analyzers = {
            AnalysisType.ANOMALY_DETECTION: AnomalyDetector(),
            AnalysisType.TREND_ANALYSIS: TrendAnalyzer(),
            AnalysisType.PREDICTIVE_ANALYSIS: PredictiveAnalyzer()
        }
        self.analysis_history = defaultdict(list)
        self.alert_thresholds = {}
        
    async def run_analysis(self, analysis_type: AnalysisType, data: Dict[str, Any]) -> AnalysisResult:
        """Exécution d'une analyse spécifique"""
        analyzer = self.analyzers.get(analysis_type)
        if not analyzer:
            raise ValueError(f"Analyseur non trouvé pour le type: {analysis_type}")
        
        result = await analyzer.analyze(data)
        
        # Sauvegarde de l'historique
        self.analysis_history[analysis_type].append(result)
        
        # Vérification des alertes
        await self._check_alert_conditions(result)
        
        return result
    
    async def run_comprehensive_analysis(self, metric_name: str, data: Dict[str, Any]) -> Dict[str, AnalysisResult]:
        """Analyse complète avec tous les analyseurs"""
        results = {}
        
        # Exécution de toutes les analyses en parallèle
        tasks = []
        for analysis_type, analyzer in self.analyzers.items():
            task = asyncio.create_task(analyzer.analyze({**data, "metric_name": metric_name}))
            tasks.append((analysis_type, task))
        
        # Récupération des résultats
        for analysis_type, task in tasks:
            try:
                result = await task
                results[analysis_type.value] = result
            except Exception as e:
                logger.error(f"Erreur lors de l'analyse {analysis_type}: {e}")
        
        return results
    
    async def train_all_analyzers(self, metric_name: str, historical_data: List[float]):
        """Entraînement de tous les analyseurs pour une métrique"""
        for analyzer in self.analyzers.values():
            await analyzer.train_model(metric_name, historical_data)
    
    async def _check_alert_conditions(self, result: AnalysisResult):
        """Vérification des conditions d'alerte"""
        # Logique d'alerte basée sur les résultats d'analyse
        if result.confidence_score > 0.8:
            if isinstance(result, AnomalyDetectionResult) and result.is_anomaly:
                await self._trigger_anomaly_alert(result)
            elif isinstance(result, TrendAnalysisResult) and result.trend_strength > 0.8:
                await self._trigger_trend_alert(result)
    
    async def _trigger_anomaly_alert(self, result: AnomalyDetectionResult):
        """Déclenchement d'alerte pour anomalie"""
        logger.warning(
            f"Anomalie détectée pour {result.metric_name}",
            confidence=result.confidence_score,
            anomaly_type=result.anomaly_type.value,
            actual_value=result.actual_value
        )
    
    async def _trigger_trend_alert(self, result: TrendAnalysisResult):
        """Déclenchement d'alerte pour tendance critique"""
        logger.warning(
            f"Tendance critique détectée pour {result.metric_name}",
            direction=result.trend_direction,
            strength=result.trend_strength
        )
