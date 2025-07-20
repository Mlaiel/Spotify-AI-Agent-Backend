"""
Modèles prédictifs avancés pour la prévision d'incidents et l'analyse de capacité.

Ce module implémente des modèles ML/DL sophistiqués pour :
- Prédiction proactive d'incidents
- Prévision de capacité et tendances
- Analyse de saisonnalité et patterns temporels
- Détection précoce de dégradations
- Modélisation de charge et performance

Utilise TensorFlow, scikit-learn et des techniques de time series avancées.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
import pickle
import json
import asyncio
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import joblib

# Time Series Analysis
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
from scipy.signal import find_peaks

# Deep Learning
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Monitoring et métriques
from prometheus_client import Counter, Histogram, Gauge, Summary
import redis

logger = logging.getLogger(__name__)

# Métriques Prometheus
PREDICTIONS_MADE = Counter('predictions_made_total', 'Total predictions made', ['model_type', 'prediction_type'])
PREDICTION_LATENCY = Histogram('prediction_duration_seconds', 'Time spent making predictions', ['model_type'])
PREDICTION_ACCURACY = Gauge('prediction_accuracy', 'Current prediction accuracy', ['model_type'])
CAPACITY_FORECASTS = Summary('capacity_forecasts', 'Capacity forecasting scores')

class PredictionType(Enum):
    """Types de prédictions supportées."""
    INCIDENT_PROBABILITY = "incident_probability"
    CAPACITY_USAGE = "capacity_usage"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    TREND_ANALYSIS = "trend_analysis"
    ANOMALY_LIKELIHOOD = "anomaly_likelihood"

class TimeHorizon(Enum):
    """Horizons temporels de prédiction."""
    SHORT_TERM = "short_term"      # 1-6 heures
    MEDIUM_TERM = "medium_term"    # 6-24 heures
    LONG_TERM = "long_term"        # 1-7 jours
    EXTENDED = "extended"          # 1-4 semaines

@dataclass
class PredictionResult:
    """Résultat d'une prédiction."""
    prediction_id: str
    model_name: str
    prediction_type: PredictionType
    time_horizon: TimeHorizon
    prediction_time: datetime
    target_time: datetime
    predicted_value: float
    confidence_interval: Tuple[float, float]
    confidence_score: float
    feature_importance: Dict[str, float] = field(default_factory=dict)
    model_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IncidentPrediction:
    """Prédiction spécialisée pour les incidents."""
    incident_probability: float
    predicted_severity: str
    predicted_components: List[str]
    risk_factors: Dict[str, float]
    preventive_actions: List[str]
    estimated_impact: Dict[str, Any]

@dataclass
class CapacityForecast:
    """Prévision de capacité."""
    resource_type: str
    current_usage: float
    predicted_usage: float
    capacity_limit: float
    time_to_saturation: Optional[timedelta]
    recommended_actions: List[str]
    growth_rate: float

class BasePredictiveModel(ABC):
    """Classe de base pour tous les modèles prédictifs."""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.training_history = []
        self.performance_metrics = {}
        
    @abstractmethod
    def train(self, training_data: pd.DataFrame, target_column: str) -> None:
        """Entraîne le modèle."""
        pass
    
    @abstractmethod
    def predict(self, features: np.ndarray, prediction_time: datetime) -> PredictionResult:
        """Fait une prédiction."""
        pass
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prépare les features pour l'entraînement/prédiction."""
        
        # Features temporelles
        if 'timestamp' in data.columns:
            data['hour'] = data['timestamp'].dt.hour
            data['day_of_week'] = data['timestamp'].dt.dayofweek
            data['day_of_month'] = data['timestamp'].dt.day
            data['month'] = data['timestamp'].dt.month
            data['is_weekend'] = (data['timestamp'].dt.dayofweek >= 5).astype(int)
            data['is_business_hours'] = ((data['timestamp'].dt.hour >= 9) & 
                                        (data['timestamp'].dt.hour <= 17)).astype(int)
        
        # Features de lag pour séries temporelles
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != 'timestamp':
                data[f'{col}_lag_1'] = data[col].shift(1)
                data[f'{col}_lag_24'] = data[col].shift(24)  # 24h lag
                data[f'{col}_rolling_mean_6'] = data[col].rolling(window=6).mean()
                data[f'{col}_rolling_std_6'] = data[col].rolling(window=6).std()
        
        return data.dropna()
    
    def save_model(self, path: str) -> None:
        """Sauvegarde le modèle."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'performance_metrics': self.performance_metrics
        }
        joblib.dump(model_data, path)
        
    def load_model(self, path: str) -> None:
        """Charge un modèle sauvegardé."""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.config = model_data['config']
        self.model_name = model_data['model_name']
        self.is_trained = model_data['is_trained']
        self.feature_names = model_data['feature_names']
        self.performance_metrics = model_data.get('performance_metrics', {})

class IncidentPredictor(BasePredictiveModel):
    """Prédicteur d'incidents utilisant des modèles d'ensemble."""
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            'models': ['random_forest', 'gradient_boosting', 'neural_network'],
            'ensemble_method': 'weighted_average',
            'feature_selection': True,
            'cross_validation_folds': 5,
            'time_horizons': ['1h', '6h', '24h'],
            'incident_threshold': 0.7
        }
        config = config or default_config
        super().__init__("incident_predictor", config)
        
        self.sub_models = {}
        self.feature_selector = None
        self.incident_history = []
        
    def train(self, training_data: pd.DataFrame, target_column: str = 'incident_occurred') -> None:
        """Entraîne le prédicteur d'incidents."""
        
        logger.info(f"Training incident predictor with {len(training_data)} samples")
        
        try:
            # Préparation des données
            prepared_data = self.prepare_features(training_data)
            
            # Séparation features/target
            feature_columns = [col for col in prepared_data.columns 
                             if col not in [target_column, 'timestamp', 'incident_id']]
            
            X = prepared_data[feature_columns]
            y = prepared_data[target_column]
            
            self.feature_names = feature_columns
            
            # Sélection de features si activée
            if self.config['feature_selection']:
                X = self._select_features(X, y)
            
            # Normalisation
            X_scaled = self.scaler.fit_transform(X)
            
            # Division train/test temporelle
            split_index = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
            y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
            
            # Entraînement des sous-modèles
            for model_type in self.config['models']:
                logger.info(f"Training {model_type} sub-model...")
                
                if model_type == 'random_forest':
                    model = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        min_samples_split=5,
                        random_state=42,
                        n_jobs=-1
                    )
                elif model_type == 'gradient_boosting':
                    model = GradientBoostingRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=6,
                        random_state=42
                    )
                elif model_type == 'neural_network':
                    model = self._create_neural_network(X_train.shape[1])
                else:
                    continue
                
                # Entraînement
                if model_type == 'neural_network':
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=50,
                        batch_size=32,
                        verbose=0,
                        callbacks=[
                            EarlyStopping(patience=10, restore_best_weights=True),
                            ReduceLROnPlateau(patience=5, factor=0.5)
                        ]
                    )
                    self.training_history.append(history.history)
                else:
                    model.fit(X_train, y_train)
                
                # Évaluation
                if model_type == 'neural_network':
                    train_pred = model.predict(X_train).flatten()
                    test_pred = model.predict(X_test).flatten()
                else:
                    train_pred = model.predict(X_train)
                    test_pred = model.predict(X_test)
                
                train_mse = mean_squared_error(y_train, train_pred)
                test_mse = mean_squared_error(y_test, test_pred)
                test_r2 = r2_score(y_test, test_pred)
                
                logger.info(f"{model_type} - Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}, R²: {test_r2:.4f}")
                
                self.sub_models[model_type] = {
                    'model': model,
                    'performance': {
                        'train_mse': train_mse,
                        'test_mse': test_mse,
                        'test_r2': test_r2
                    }
                }
            
            self.is_trained = True
            logger.info("Incident predictor training completed")
            
        except Exception as e:
            logger.error(f"Error training incident predictor: {e}")
            raise
    
    def _create_neural_network(self, input_dim: int) -> Model:
        """Crée un réseau de neurones pour la prédiction d'incidents."""
        
        model = Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Sélectionne les features les plus importantes."""
        
        from sklearn.feature_selection import SelectKBest, f_regression
        
        # Sélection des K meilleures features
        k = min(20, len(X.columns))
        selector = SelectKBest(score_func=f_regression, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Mise à jour des noms de features
        selected_features = X.columns[selector.get_support()].tolist()
        self.feature_names = selected_features
        self.feature_selector = selector
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def predict(self, features: np.ndarray, prediction_time: datetime) -> PredictionResult:
        """Prédit la probabilité d'incident."""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        with PREDICTION_LATENCY.labels(model_type='incident_predictor').time():
            try:
                # Préparation des features
                if self.feature_selector:
                    features = self.feature_selector.transform(features)
                
                features_scaled = self.scaler.transform(features)
                
                # Prédictions des sous-modèles
                predictions = {}
                confidences = {}
                
                for model_type, model_data in self.sub_models.items():
                    model = model_data['model']
                    
                    if model_type == 'neural_network':
                        pred = model.predict(features_scaled).flatten()[0]
                    else:
                        pred = model.predict(features_scaled)[0]
                    
                    predictions[model_type] = pred
                    
                    # Calcul de confiance basé sur les performances
                    test_r2 = model_data['performance']['test_r2']
                    confidences[model_type] = max(0, test_r2)
                
                # Ensemble des prédictions
                ensemble_prediction = self._ensemble_predict(predictions, confidences)
                ensemble_confidence = np.mean(list(confidences.values()))
                
                # Calcul de l'intervalle de confiance
                pred_std = np.std(list(predictions.values()))
                confidence_interval = (
                    ensemble_prediction - 1.96 * pred_std,
                    ensemble_prediction + 1.96 * pred_std
                )
                
                # Features importance (moyenne des modèles)
                feature_importance = self._calculate_feature_importance()
                
                result = PredictionResult(
                    prediction_id=f"incident_{prediction_time.strftime('%Y%m%d_%H%M%S')}",
                    model_name=self.model_name,
                    prediction_type=PredictionType.INCIDENT_PROBABILITY,
                    time_horizon=TimeHorizon.SHORT_TERM,
                    prediction_time=prediction_time,
                    target_time=prediction_time + timedelta(hours=1),
                    predicted_value=ensemble_prediction,
                    confidence_interval=confidence_interval,
                    confidence_score=ensemble_confidence,
                    feature_importance=feature_importance,
                    model_metadata={
                        'sub_models_used': list(self.sub_models.keys()),
                        'ensemble_method': self.config['ensemble_method']
                    }
                )
                
                PREDICTIONS_MADE.labels(
                    model_type='incident_predictor',
                    prediction_type='incident_probability'
                ).inc()
                
                return result
                
            except Exception as e:
                logger.error(f"Error in incident prediction: {e}")
                raise
    
    def _ensemble_predict(self, predictions: Dict[str, float], 
                         confidences: Dict[str, float]) -> float:
        """Combine les prédictions des sous-modèles."""
        
        if self.config['ensemble_method'] == 'weighted_average':
            total_weight = sum(confidences.values())
            if total_weight == 0:
                return np.mean(list(predictions.values()))
            
            weighted_sum = sum(pred * confidences[model] 
                             for model, pred in predictions.items())
            return weighted_sum / total_weight
        
        elif self.config['ensemble_method'] == 'simple_average':
            return np.mean(list(predictions.values()))
        
        elif self.config['ensemble_method'] == 'max':
            return max(predictions.values())
        
        else:
            return np.mean(list(predictions.values()))
    
    def _calculate_feature_importance(self) -> Dict[str, float]:
        """Calcule l'importance moyenne des features."""
        
        importance_dict = {}
        
        for model_type, model_data in self.sub_models.items():
            if model_type in ['random_forest', 'gradient_boosting']:
                model = model_data['model']
                importances = model.feature_importances_
                
                for i, feature in enumerate(self.feature_names):
                    if feature not in importance_dict:
                        importance_dict[feature] = []
                    importance_dict[feature].append(importances[i])
        
        # Moyenne des importances
        avg_importance = {
            feature: np.mean(values) 
            for feature, values in importance_dict.items()
        }
        
        return avg_importance
    
    def predict_incident_details(self, features: np.ndarray) -> IncidentPrediction:
        """Prédit les détails d'un incident potentiel."""
        
        # Prédiction de base
        base_prediction = self.predict(features, datetime.now())
        
        # Analyse des composants à risque
        risk_factors = self._analyze_risk_factors(features)
        predicted_components = self._predict_affected_components(features, risk_factors)
        
        # Estimation de la sévérité
        severity_score = base_prediction.predicted_value
        if severity_score > 0.8:
            predicted_severity = "critical"
        elif severity_score > 0.6:
            predicted_severity = "high"
        elif severity_score > 0.4:
            predicted_severity = "medium"
        else:
            predicted_severity = "low"
        
        # Actions préventives
        preventive_actions = self._suggest_preventive_actions(risk_factors, predicted_severity)
        
        # Estimation d'impact
        estimated_impact = self._estimate_incident_impact(predicted_severity, predicted_components)
        
        return IncidentPrediction(
            incident_probability=base_prediction.predicted_value,
            predicted_severity=predicted_severity,
            predicted_components=predicted_components,
            risk_factors=risk_factors,
            preventive_actions=preventive_actions,
            estimated_impact=estimated_impact
        )
    
    def _analyze_risk_factors(self, features: np.ndarray) -> Dict[str, float]:
        """Analyse les facteurs de risque basés sur les features."""
        
        risk_factors = {}
        
        # Simulation d'analyse de facteurs de risque
        feature_values = features[0] if len(features.shape) > 1 else features
        
        for i, feature_name in enumerate(self.feature_names[:len(feature_values)]):
            value = feature_values[i]
            
            # Analyse basée sur le nom de la feature
            if 'cpu' in feature_name.lower():
                risk_factors['cpu_overload'] = min(1.0, value / 100)
            elif 'memory' in feature_name.lower():
                risk_factors['memory_pressure'] = min(1.0, value / 100)
            elif 'disk' in feature_name.lower():
                risk_factors['disk_saturation'] = min(1.0, value / 100)
            elif 'network' in feature_name.lower():
                risk_factors['network_congestion'] = min(1.0, value / 1000)
            elif 'error' in feature_name.lower():
                risk_factors['error_rate_spike'] = min(1.0, value / 10)
        
        return risk_factors
    
    def _predict_affected_components(self, features: np.ndarray, 
                                   risk_factors: Dict[str, float]) -> List[str]:
        """Prédit les composants qui seront affectés."""
        
        affected_components = []
        
        # Logique de prédiction basée sur les facteurs de risque
        for factor, score in risk_factors.items():
            if score > 0.6:
                if 'cpu' in factor:
                    affected_components.extend(['api-server', 'worker-nodes'])
                elif 'memory' in factor:
                    affected_components.extend(['database', 'cache-layer'])
                elif 'disk' in factor:
                    affected_components.extend(['storage-service', 'logging-system'])
                elif 'network' in factor:
                    affected_components.extend(['load-balancer', 'api-gateway'])
                elif 'error' in factor:
                    affected_components.extend(['user-service', 'payment-service'])
        
        return list(set(affected_components))
    
    def _suggest_preventive_actions(self, risk_factors: Dict[str, float], 
                                   severity: str) -> List[str]:
        """Suggère des actions préventives."""
        
        actions = []
        
        # Actions basées sur les facteurs de risque
        for factor, score in risk_factors.items():
            if score > 0.5:
                if 'cpu' in factor:
                    actions.append("Scale up compute resources")
                    actions.append("Review CPU-intensive processes")
                elif 'memory' in factor:
                    actions.append("Increase memory allocation")
                    actions.append("Check for memory leaks")
                elif 'disk' in factor:
                    actions.append("Clean up disk space")
                    actions.append("Archive old logs")
                elif 'network' in factor:
                    actions.append("Check network bandwidth")
                    actions.append("Review traffic patterns")
        
        # Actions basées sur la sévérité
        if severity in ['critical', 'high']:
            actions.append("Notify on-call team")
            actions.append("Prepare rollback plan")
            actions.append("Activate monitoring dashboard")
        
        return actions
    
    def _estimate_incident_impact(self, severity: str, 
                                 components: List[str]) -> Dict[str, Any]:
        """Estime l'impact d'un incident."""
        
        # Mapping de criticité des composants
        component_criticality = {
            'payment-service': 0.9,
            'user-service': 0.8,
            'api-gateway': 0.8,
            'database': 0.9,
            'api-server': 0.7,
            'storage-service': 0.6,
            'logging-system': 0.3
        }
        
        # Calcul de l'impact
        max_criticality = max(component_criticality.get(comp, 0.5) for comp in components) if components else 0.5
        
        severity_multiplier = {
            'critical': 1.0,
            'high': 0.7,
            'medium': 0.4,
            'low': 0.2
        }.get(severity, 0.5)
        
        impact_score = max_criticality * severity_multiplier
        
        # Estimation des utilisateurs affectés
        if impact_score > 0.8:
            estimated_users = 10000
        elif impact_score > 0.6:
            estimated_users = 1000
        elif impact_score > 0.4:
            estimated_users = 100
        else:
            estimated_users = 10
        
        # Estimation du coût
        estimated_cost_per_hour = estimated_users * 10  # $10 par utilisateur par heure
        
        return {
            'impact_score': impact_score,
            'estimated_affected_users': estimated_users,
            'estimated_cost_per_hour': estimated_cost_per_hour,
            'affected_components': components,
            'business_functions_impacted': self._get_business_functions(components)
        }
    
    def _get_business_functions(self, components: List[str]) -> List[str]:
        """Identifie les fonctions business impactées."""
        
        function_mapping = {
            'payment-service': ['payments', 'billing'],
            'user-service': ['authentication', 'user_management'],
            'api-gateway': ['all_api_services'],
            'database': ['data_persistence', 'analytics'],
            'storage-service': ['file_storage', 'backups']
        }
        
        impacted_functions = set()
        for component in components:
            if component in function_mapping:
                impacted_functions.update(function_mapping[component])
        
        return list(impacted_functions)

class CapacityForecaster(BasePredictiveModel):
    """Prédicteur de capacité utilisant des modèles de séries temporelles."""
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            'forecast_horizons': [24, 72, 168],  # heures
            'seasonality_periods': [24, 168],     # heures et semaines
            'trend_detection': True,
            'capacity_thresholds': {
                'warning': 0.7,
                'critical': 0.85,
                'emergency': 0.95
            }
        }
        config = config or default_config
        super().__init__("capacity_forecaster", config)
        
        self.time_series_models = {}
        self.seasonality_components = {}
        
    def train(self, training_data: pd.DataFrame, target_column: str = 'usage_percent') -> None:
        """Entraîne le modèle de prévision de capacité."""
        
        logger.info(f"Training capacity forecaster with {len(training_data)} samples")
        
        try:
            # Préparation des données de série temporelle
            if 'timestamp' not in training_data.columns:
                raise ValueError("Timestamp column required for capacity forecasting")
            
            # Tri par timestamp
            data = training_data.sort_values('timestamp')
            
            # Création de l'index temporel
            data.set_index('timestamp', inplace=True)
            
            # Rééchantillonnage horaire si nécessaire
            if len(data) > 1000:
                data = data.resample('1H').mean()
            
            # Analyse de saisonnalité
            self._analyze_seasonality(data[target_column])
            
            # Entraînement de différents modèles
            self._train_arima_model(data[target_column])
            self._train_exponential_smoothing(data[target_column])
            self._train_lstm_model(data[target_column])
            
            self.is_trained = True
            logger.info("Capacity forecaster training completed")
            
        except Exception as e:
            logger.error(f"Error training capacity forecaster: {e}")
            raise
    
    def _analyze_seasonality(self, series: pd.Series) -> None:
        """Analyse la saisonnalité de la série temporelle."""
        
        try:
            # Décomposition saisonnière
            decomposition = seasonal_decompose(
                series.dropna(), 
                model='additive', 
                period=24  # Saisonnalité quotidienne
            )
            
            self.seasonality_components = {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid
            }
            
            # Test de stationnarité
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(series.dropna())
            
            self.performance_metrics['seasonality'] = {
                'is_stationary': adf_result[1] < 0.05,
                'adf_statistic': adf_result[0],
                'adf_pvalue': adf_result[1],
                'seasonal_strength': abs(decomposition.seasonal.std() / series.std())
            }
            
        except Exception as e:
            logger.warning(f"Seasonality analysis failed: {e}")
    
    def _train_arima_model(self, series: pd.Series) -> None:
        """Entraîne un modèle ARIMA."""
        
        try:
            # Auto-ARIMA simplifié
            best_aic = float('inf')
            best_params = None
            
            for p in range(3):
                for d in range(2):
                    for q in range(3):
                        try:
                            model = ARIMA(series.dropna(), order=(p, d, q))
                            fitted_model = model.fit()
                            
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_params = (p, d, q)
                                
                        except:
                            continue
            
            if best_params:
                final_model = ARIMA(series.dropna(), order=best_params)
                self.time_series_models['arima'] = final_model.fit()
                
                logger.info(f"ARIMA model trained with order {best_params}, AIC: {best_aic:.2f}")
                
        except Exception as e:
            logger.warning(f"ARIMA training failed: {e}")
    
    def _train_exponential_smoothing(self, series: pd.Series) -> None:
        """Entraîne un modèle de lissage exponentiel."""
        
        try:
            model = ExponentialSmoothing(
                series.dropna(),
                trend='add',
                seasonal='add',
                seasonal_periods=24
            )
            
            self.time_series_models['exponential_smoothing'] = model.fit()
            
            logger.info("Exponential smoothing model trained")
            
        except Exception as e:
            logger.warning(f"Exponential smoothing training failed: {e}")
    
    def _train_lstm_model(self, series: pd.Series) -> None:
        """Entraîne un modèle LSTM pour prédiction de séries temporelles."""
        
        try:
            # Préparation des données LSTM
            sequence_length = 24  # 24 heures de lookback
            
            data = series.dropna().values
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data.reshape(-1, 1))
            
            # Création des séquences
            X, y = [], []
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i-sequence_length:i, 0])
                y.append(scaled_data[i, 0])
            
            X, y = np.array(X), np.array(y)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            if len(X) < 50:  # Pas assez de données
                return
            
            # Construction du modèle LSTM
            model = Sequential([
                layers.LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
                layers.Dropout(0.2),
                layers.LSTM(50, return_sequences=False),
                layers.Dropout(0.2),
                layers.Dense(25),
                layers.Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Entraînement
            history = model.fit(
                X, y,
                batch_size=32,
                epochs=50,
                validation_split=0.2,
                verbose=0,
                callbacks=[EarlyStopping(patience=10)]
            )
            
            self.time_series_models['lstm'] = {
                'model': model,
                'scaler': scaler,
                'sequence_length': sequence_length
            }
            
            logger.info("LSTM model trained for capacity forecasting")
            
        except Exception as e:
            logger.warning(f"LSTM training failed: {e}")
    
    def predict(self, features: np.ndarray, prediction_time: datetime) -> PredictionResult:
        """Prédit l'usage futur de capacité."""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        with PREDICTION_LATENCY.labels(model_type='capacity_forecaster').time():
            try:
                forecasts = {}
                
                # Prédictions des différents modèles
                if 'arima' in self.time_series_models:
                    arima_forecast = self._predict_arima(steps=24)
                    forecasts['arima'] = arima_forecast
                
                if 'exponential_smoothing' in self.time_series_models:
                    es_forecast = self._predict_exponential_smoothing(steps=24)
                    forecasts['exponential_smoothing'] = es_forecast
                
                if 'lstm' in self.time_series_models:
                    lstm_forecast = self._predict_lstm(current_data=features, steps=24)
                    forecasts['lstm'] = lstm_forecast
                
                # Ensemble des prédictions
                if forecasts:
                    ensemble_forecast = np.mean([f for f in forecasts.values() if f is not None])
                    forecast_std = np.std([f for f in forecasts.values() if f is not None])
                else:
                    ensemble_forecast = 0.5  # Valeur par défaut
                    forecast_std = 0.1
                
                # Intervalle de confiance
                confidence_interval = (
                    max(0, ensemble_forecast - 1.96 * forecast_std),
                    min(1, ensemble_forecast + 1.96 * forecast_std)
                )
                
                result = PredictionResult(
                    prediction_id=f"capacity_{prediction_time.strftime('%Y%m%d_%H%M%S')}",
                    model_name=self.model_name,
                    prediction_type=PredictionType.CAPACITY_USAGE,
                    time_horizon=TimeHorizon.MEDIUM_TERM,
                    prediction_time=prediction_time,
                    target_time=prediction_time + timedelta(hours=24),
                    predicted_value=ensemble_forecast,
                    confidence_interval=confidence_interval,
                    confidence_score=1 - forecast_std,
                    model_metadata={
                        'models_used': list(forecasts.keys()),
                        'seasonality_detected': bool(self.seasonality_components)
                    }
                )
                
                PREDICTIONS_MADE.labels(
                    model_type='capacity_forecaster',
                    prediction_type='capacity_usage'
                ).inc()
                
                CAPACITY_FORECASTS.observe(ensemble_forecast)
                
                return result
                
            except Exception as e:
                logger.error(f"Error in capacity prediction: {e}")
                raise
    
    def _predict_arima(self, steps: int) -> Optional[float]:
        """Prédiction ARIMA."""
        try:
            model = self.time_series_models['arima']
            forecast = model.forecast(steps=steps)
            return float(forecast[steps-1])  # Dernière prédiction
        except:
            return None
    
    def _predict_exponential_smoothing(self, steps: int) -> Optional[float]:
        """Prédiction lissage exponentiel."""
        try:
            model = self.time_series_models['exponential_smoothing']
            forecast = model.forecast(steps=steps)
            return float(forecast[steps-1])
        except:
            return None
    
    def _predict_lstm(self, current_data: np.ndarray, steps: int) -> Optional[float]:
        """Prédiction LSTM."""
        try:
            lstm_data = self.time_series_models['lstm']
            model = lstm_data['model']
            scaler = lstm_data['scaler']
            sequence_length = lstm_data['sequence_length']
            
            # Préparer les données d'entrée
            if len(current_data) >= sequence_length:
                input_data = current_data[-sequence_length:].reshape(1, sequence_length, 1)
                scaled_input = scaler.transform(input_data.reshape(-1, 1))
                scaled_input = scaled_input.reshape(1, sequence_length, 1)
                
                prediction = model.predict(scaled_input)
                prediction = scaler.inverse_transform(prediction)
                
                return float(prediction[0, 0])
            
            return None
            
        except:
            return None
    
    def forecast_capacity(self, current_usage: float, capacity_limit: float, 
                         growth_data: List[float]) -> CapacityForecast:
        """Génère une prévision complète de capacité."""
        
        # Calcul du taux de croissance
        if len(growth_data) >= 2:
            growth_rate = (growth_data[-1] - growth_data[0]) / len(growth_data)
        else:
            growth_rate = 0.0
        
        # Prédiction de l'usage futur
        predicted_usage = current_usage + (growth_rate * 24)  # 24h prédiction
        
        # Temps jusqu'à saturation
        time_to_saturation = None
        if growth_rate > 0:
            remaining_capacity = capacity_limit - current_usage
            hours_to_saturation = remaining_capacity / growth_rate
            
            if hours_to_saturation > 0:
                time_to_saturation = timedelta(hours=hours_to_saturation)
        
        # Actions recommandées
        recommended_actions = []
        usage_percent = predicted_usage / capacity_limit
        
        if usage_percent >= self.config['capacity_thresholds']['emergency']:
            recommended_actions.extend([
                "IMMEDIATE: Scale resources urgently",
                "Activate emergency capacity procedures",
                "Notify infrastructure team immediately"
            ])
        elif usage_percent >= self.config['capacity_thresholds']['critical']:
            recommended_actions.extend([
                "Scale resources within 2 hours",
                "Review capacity planning",
                "Monitor closely"
            ])
        elif usage_percent >= self.config['capacity_thresholds']['warning']:
            recommended_actions.extend([
                "Plan resource scaling",
                "Review usage trends",
                "Consider optimization"
            ])
        
        return CapacityForecast(
            resource_type="system_capacity",
            current_usage=current_usage,
            predicted_usage=predicted_usage,
            capacity_limit=capacity_limit,
            time_to_saturation=time_to_saturation,
            recommended_actions=recommended_actions,
            growth_rate=growth_rate
        )

class TrendAnalyzer(BasePredictiveModel):
    """Analyseur de tendances pour identifier les patterns à long terme."""
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            'trend_window': 30,  # jours
            'change_point_detection': True,
            'anomaly_detection': True,
            'seasonal_analysis': True
        }
        config = config or default_config
        super().__init__("trend_analyzer", config)
        
        self.trend_models = {}
        self.change_points = []
        
    def train(self, training_data: pd.DataFrame, target_column: str) -> None:
        """Entraîne l'analyseur de tendances."""
        
        logger.info("Training trend analyzer...")
        
        try:
            # Préparation des données
            data = training_data.copy()
            if 'timestamp' in data.columns:
                data.set_index('timestamp', inplace=True)
            
            # Détection des points de changement
            if self.config['change_point_detection']:
                self._detect_change_points(data[target_column])
            
            # Analyse de tendance
            self._analyze_long_term_trend(data[target_column])
            
            # Analyse saisonnière avancée
            if self.config['seasonal_analysis']:
                self._advanced_seasonal_analysis(data[target_column])
            
            self.is_trained = True
            
        except Exception as e:
            logger.error(f"Error training trend analyzer: {e}")
            raise
    
    def _detect_change_points(self, series: pd.Series) -> None:
        """Détecte les points de changement de tendance."""
        
        try:
            # Algorithme simple de détection de changement
            values = series.dropna().values
            
            # Calcul de la dérivée seconde
            if len(values) < 10:
                return
            
            # Lissage des données
            window = min(7, len(values) // 10)
            smoothed = pd.Series(values).rolling(window=window).mean().dropna()
            
            # Calcul des gradients
            gradients = np.gradient(smoothed)
            gradient_changes = np.gradient(gradients)
            
            # Détection des pics dans les changements de gradient
            peaks, _ = find_peaks(np.abs(gradient_changes), height=np.std(gradient_changes))
            
            # Conversion en timestamps
            timestamps = series.dropna().index[window-1:]  # Ajustement pour le lissage
            self.change_points = [timestamps[i] for i in peaks if i < len(timestamps)]
            
            logger.info(f"Detected {len(self.change_points)} change points")
            
        except Exception as e:
            logger.warning(f"Change point detection failed: {e}")
    
    def _analyze_long_term_trend(self, series: pd.Series) -> None:
        """Analyse la tendance à long terme."""
        
        try:
            # Régression linéaire pour tendance globale
            values = series.dropna()
            x = np.arange(len(values))
            y = values.values
            
            # Régression linéaire
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Modèle polynomial pour tendances non-linéaires
            poly_coeffs = np.polyfit(x, y, degree=3)
            
            self.trend_models['linear'] = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value
            }
            
            self.trend_models['polynomial'] = {
                'coefficients': poly_coeffs,
                'degree': 3
            }
            
            # Classification de la tendance
            if abs(slope) < 0.001:
                trend_type = "stable"
            elif slope > 0:
                trend_type = "increasing"
            else:
                trend_type = "decreasing"
            
            self.performance_metrics['trend_analysis'] = {
                'trend_type': trend_type,
                'trend_strength': abs(r_value),
                'trend_significance': p_value < 0.05
            }
            
        except Exception as e:
            logger.warning(f"Trend analysis failed: {e}")
    
    def _advanced_seasonal_analysis(self, series: pd.Series) -> None:
        """Analyse saisonnière avancée."""
        
        try:
            # Analyse spectrale pour détecter les périodicités
            from scipy.fft import fft, fftfreq
            
            values = series.dropna().values
            
            # FFT
            fft_values = fft(values)
            frequencies = fftfreq(len(values))
            
            # Identification des fréquences dominantes
            power_spectrum = np.abs(fft_values)**2
            dominant_freq_indices = np.argsort(power_spectrum)[-5:]  # Top 5
            
            dominant_periods = []
            for idx in dominant_freq_indices:
                if frequencies[idx] != 0:
                    period = 1 / abs(frequencies[idx])
                    if 2 <= period <= len(values) // 2:  # Périodes raisonnables
                        dominant_periods.append(period)
            
            self.performance_metrics['seasonal_analysis'] = {
                'dominant_periods': dominant_periods,
                'spectral_analysis_completed': True
            }
            
        except Exception as e:
            logger.warning(f"Advanced seasonal analysis failed: {e}")
    
    def predict(self, features: np.ndarray, prediction_time: datetime) -> PredictionResult:
        """Prédit les tendances futures."""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        try:
            # Prédiction basée sur la tendance linéaire
            linear_model = self.trend_models.get('linear', {})
            
            if linear_model:
                # Projection linéaire
                future_steps = 24  # 24 heures dans le futur
                current_step = len(features) if hasattr(features, '__len__') else 1000
                
                predicted_value = (linear_model['slope'] * (current_step + future_steps) + 
                                 linear_model['intercept'])
                
                confidence = linear_model.get('r_squared', 0.5)
            else:
                predicted_value = 0.5
                confidence = 0.3
            
            result = PredictionResult(
                prediction_id=f"trend_{prediction_time.strftime('%Y%m%d_%H%M%S')}",
                model_name=self.model_name,
                prediction_type=PredictionType.TREND_ANALYSIS,
                time_horizon=TimeHorizon.LONG_TERM,
                prediction_time=prediction_time,
                target_time=prediction_time + timedelta(hours=24),
                predicted_value=predicted_value,
                confidence_interval=(predicted_value * 0.9, predicted_value * 1.1),
                confidence_score=confidence,
                model_metadata={
                    'trend_type': self.performance_metrics.get('trend_analysis', {}).get('trend_type', 'unknown'),
                    'change_points_detected': len(self.change_points),
                    'seasonal_periods': self.performance_metrics.get('seasonal_analysis', {}).get('dominant_periods', [])
                }
            )
            
            PREDICTIONS_MADE.labels(
                model_type='trend_analyzer',
                prediction_type='trend_analysis'
            ).inc()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in trend prediction: {e}")
            raise

# Factory et orchestrateur

class PredictiveModelFactory:
    """Factory pour créer des modèles prédictifs."""
    
    @staticmethod
    def create_model(model_type: str, config: Dict[str, Any] = None) -> BasePredictiveModel:
        """Crée un modèle du type spécifié."""
        
        if model_type == "incident_predictor":
            return IncidentPredictor(config)
        elif model_type == "capacity_forecaster":
            return CapacityForecaster(config)
        elif model_type == "trend_analyzer":
            return TrendAnalyzer(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

class PredictionOrchestrator:
    """Orchestrateur pour coordonner les prédictions de multiples modèles."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.models = {}
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
    def add_model(self, model_name: str, model: BasePredictiveModel) -> None:
        """Ajoute un modèle à l'orchestrateur."""
        self.models[model_name] = model
        
    def predict_all(self, features: np.ndarray, prediction_time: datetime) -> Dict[str, PredictionResult]:
        """Fait des prédictions avec tous les modèles."""
        
        results = {}
        
        for model_name, model in self.models.items():
            try:
                if model.is_trained:
                    result = model.predict(features, prediction_time)
                    results[model_name] = result
                    
                    # Cache des résultats
                    self._cache_result(result)
                    
            except Exception as e:
                logger.error(f"Prediction failed for model {model_name}: {e}")
                continue
        
        return results
    
    def _cache_result(self, result: PredictionResult) -> None:
        """Met en cache le résultat de prédiction."""
        try:
            cache_key = f"prediction:{result.prediction_id}"
            cache_data = {
                'model_name': result.model_name,
                'prediction_type': result.prediction_type.value,
                'predicted_value': result.predicted_value,
                'confidence_score': result.confidence_score,
                'prediction_time': result.prediction_time.isoformat(),
                'target_time': result.target_time.isoformat()
            }
            
            self.redis_client.setex(
                cache_key,
                timedelta(hours=24),
                json.dumps(cache_data)
            )
            
        except Exception as e:
            logger.warning(f"Failed to cache prediction result: {e}")
    
    async def predict_all_async(self, features: np.ndarray, 
                               prediction_time: datetime) -> Dict[str, PredictionResult]:
        """Version asynchrone des prédictions."""
        
        tasks = []
        
        for model_name, model in self.models.items():
            if model.is_trained:
                task = asyncio.create_task(
                    asyncio.to_thread(model.predict, features, prediction_time),
                    name=model_name
                )
                tasks.append(task)
        
        results = {}
        
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                model_name = task.get_name()
                results[model_name] = result
                
                # Cache asynchrone
                await asyncio.to_thread(self._cache_result, result)
                
            except Exception as e:
                logger.error(f"Async prediction failed: {e}")
                continue
        
        return results
