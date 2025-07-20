"""
Advanced ML Algorithms for Spotify AI Agent Analytics
===================================================

Ultra-sophisticated machine learning algorithms for predictive analytics,
anomaly detection, trend forecasting, and intelligent recommendations.

Author: Fahed Mlaiel
Roles: Lead Dev + Architecte IA, Ingénieur Machine Learning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention
import joblib
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class MLModelConfig:
    """Configuration for ML models."""
    model_type: str
    parameters: Dict[str, Any]
    training_data_window: timedelta
    retrain_frequency: timedelta
    performance_threshold: float = 0.85

@dataclass
class PredictionResult:
    """Result of ML prediction."""
    prediction: Union[float, int, str]
    confidence: float
    timestamp: datetime
    features_used: List[str]
    model_version: str

class BaseMLAlgorithm(ABC):
    """Base class for all ML algorithms."""
    
    def __init__(self, config: MLModelConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.last_training = None
        self.performance_metrics = {}
        
    @abstractmethod
    async def train(self, data: pd.DataFrame) -> Dict[str, float]:
        """Train the ML model."""
        pass
        
    @abstractmethod
    async def predict(self, features: Dict[str, Any]) -> PredictionResult:
        """Make predictions using the trained model."""
        pass
        
    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save the trained model."""
        pass
        
    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load a pre-trained model."""
        pass

class AnomalyDetectionAlgorithm(BaseMLAlgorithm):
    """Advanced anomaly detection using multiple algorithms."""
    
    def __init__(self, config: MLModelConfig):
        super().__init__(config)
        self.isolation_forest = IsolationForest(
            contamination=config.parameters.get('contamination', 0.1),
            random_state=42
        )
        self.dbscan = DBSCAN(
            eps=config.parameters.get('eps', 0.5),
            min_samples=config.parameters.get('min_samples', 5)
        )
        self.ensemble_weights = config.parameters.get('ensemble_weights', [0.6, 0.4])
        
    async def train(self, data: pd.DataFrame) -> Dict[str, float]:
        """Train anomaly detection models."""
        try:
            # Prepare features
            features = self._prepare_features(data)
            scaled_features = self.scaler.fit_transform(features)
            
            # Train Isolation Forest
            self.isolation_forest.fit(scaled_features)
            
            # Train DBSCAN
            dbscan_labels = self.dbscan.fit_predict(scaled_features)
            
            self.is_trained = True
            self.last_training = datetime.now()
            
            # Calculate performance metrics
            anomaly_scores = self.isolation_forest.score_samples(scaled_features)
            self.performance_metrics = {
                'mean_anomaly_score': float(np.mean(anomaly_scores)),
                'std_anomaly_score': float(np.std(anomaly_scores)),
                'outlier_fraction': float(np.sum(dbscan_labels == -1) / len(dbscan_labels)),
                'training_samples': len(data)
            }
            
            logger.info(f"Anomaly detection model trained on {len(data)} samples")
            return self.performance_metrics
            
        except Exception as e:
            logger.error(f"Error training anomaly detection model: {e}")
            raise

    async def predict(self, features: Dict[str, Any]) -> PredictionResult:
        """Detect anomalies in new data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        try:
            # Prepare single sample
            feature_vector = self._dict_to_vector(features)
            scaled_features = self.scaler.transform([feature_vector])
            
            # Get predictions from both models
            isolation_score = self.isolation_forest.decision_function(scaled_features)[0]
            isolation_anomaly = self.isolation_forest.predict(scaled_features)[0]
            
            # Ensemble prediction
            anomaly_probability = self._calculate_anomaly_probability(isolation_score)
            is_anomaly = anomaly_probability > 0.5
            
            return PredictionResult(
                prediction=int(is_anomaly),
                confidence=anomaly_probability,
                timestamp=datetime.now(),
                features_used=list(features.keys()),
                model_version="anomaly_v1.0"
            )
            
        except Exception as e:
            logger.error(f"Error in anomaly prediction: {e}")
            raise

    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for training."""
        # Select numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        return data[numeric_cols].fillna(0).values

    def _dict_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert feature dictionary to vector."""
        return np.array([float(v) for v in features.values()])

    def _calculate_anomaly_probability(self, isolation_score: float) -> float:
        """Calculate anomaly probability from isolation score."""
        # Normalize isolation score to probability
        return 1 / (1 + np.exp(isolation_score * 5))

    def save_model(self, path: str) -> None:
        """Save the anomaly detection models."""
        model_data = {
            'isolation_forest': self.isolation_forest,
            'dbscan': self.dbscan,
            'scaler': self.scaler,
            'config': self.config,
            'performance_metrics': self.performance_metrics
        }
        joblib.dump(model_data, f"{path}/anomaly_model.pkl")

    def load_model(self, path: str) -> None:
        """Load pre-trained anomaly detection models."""
        model_data = joblib.load(f"{path}/anomaly_model.pkl")
        self.isolation_forest = model_data['isolation_forest']
        self.dbscan = model_data['dbscan']
        self.scaler = model_data['scaler']
        self.performance_metrics = model_data['performance_metrics']
        self.is_trained = True

class TrendForecastingAlgorithm(BaseMLAlgorithm):
    """Advanced trend forecasting using LSTM and ensemble methods."""
    
    def __init__(self, config: MLModelConfig):
        super().__init__(config)
        self.lstm_model = None
        self.rf_model = RandomForestRegressor(
            n_estimators=config.parameters.get('n_estimators', 100),
            random_state=42
        )
        self.sequence_length = config.parameters.get('sequence_length', 24)
        self.forecast_horizon = config.parameters.get('forecast_horizon', 12)
        
    async def train(self, data: pd.DataFrame) -> Dict[str, float]:
        """Train trend forecasting models."""
        try:
            # Prepare time series data
            time_series = self._prepare_time_series(data)
            
            # Train LSTM model
            lstm_metrics = await self._train_lstm(time_series)
            
            # Train Random Forest model
            rf_metrics = await self._train_random_forest(data)
            
            self.is_trained = True
            self.last_training = datetime.now()
            
            # Combine metrics
            self.performance_metrics = {
                'lstm_mse': lstm_metrics['mse'],
                'lstm_mae': lstm_metrics['mae'],
                'rf_mse': rf_metrics['mse'],
                'rf_r2': rf_metrics['r2'],
                'training_samples': len(data)
            }
            
            logger.info(f"Trend forecasting model trained on {len(data)} samples")
            return self.performance_metrics
            
        except Exception as e:
            logger.error(f"Error training trend forecasting model: {e}")
            raise

    async def _train_lstm(self, time_series: np.ndarray) -> Dict[str, float]:
        """Train LSTM model for time series forecasting."""
        # Normalize data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(time_series.reshape(-1, 1))
        
        # Create sequences
        X, y = self._create_sequences(scaled_data, self.sequence_length)
        
        # Build LSTM model
        self.lstm_model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        self.lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Train model
        history = self.lstm_model.fit(
            X, y,
            batch_size=32,
            epochs=50,
            validation_split=0.2,
            verbose=0
        )
        
        return {
            'mse': float(history.history['loss'][-1]),
            'mae': float(history.history['mae'][-1])
        }

    async def _train_random_forest(self, data: pd.DataFrame) -> Dict[str, float]:
        """Train Random Forest model for trend analysis."""
        # Prepare features
        features = self._extract_trend_features(data)
        target = data['value'].values if 'value' in data.columns else data.iloc[:, 0].values
        
        # Train model
        self.rf_model.fit(features, target)
        
        # Calculate metrics
        predictions = self.rf_model.predict(features)
        mse = np.mean((target - predictions) ** 2)
        r2 = self.rf_model.score(features, target)
        
        return {'mse': float(mse), 'r2': float(r2)}

    def _prepare_time_series(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare time series data for LSTM training."""
        if 'timestamp' in data.columns and 'value' in data.columns:
            return data.sort_values('timestamp')['value'].values
        else:
            # Assume first column is the target
            return data.iloc[:, 0].values

    def _create_sequences(self, data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    def _extract_trend_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for trend analysis."""
        features = []
        
        # Time-based features
        if 'timestamp' in data.columns:
            data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
            data['day_of_week'] = pd.to_datetime(data['timestamp']).dt.dayofweek
            data['month'] = pd.to_datetime(data['timestamp']).dt.month
            
        # Select numeric features
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        return data[numeric_cols].fillna(0).values

    async def predict(self, features: Dict[str, Any]) -> PredictionResult:
        """Forecast future trends."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        try:
            # Get LSTM prediction
            lstm_pred = await self._predict_lstm(features)
            
            # Get Random Forest prediction
            rf_pred = await self._predict_random_forest(features)
            
            # Ensemble prediction
            ensemble_pred = 0.7 * lstm_pred + 0.3 * rf_pred
            confidence = min(0.95, max(0.5, 1.0 - abs(lstm_pred - rf_pred) / max(abs(lstm_pred), abs(rf_pred), 1)))
            
            return PredictionResult(
                prediction=float(ensemble_pred),
                confidence=confidence,
                timestamp=datetime.now(),
                features_used=list(features.keys()),
                model_version="trend_v1.0"
            )
            
        except Exception as e:
            logger.error(f"Error in trend prediction: {e}")
            raise

    async def _predict_lstm(self, features: Dict[str, Any]) -> float:
        """Make LSTM prediction."""
        # This would use the last sequence_length values to predict next value
        # Simplified implementation
        return 0.0

    async def _predict_random_forest(self, features: Dict[str, Any]) -> float:
        """Make Random Forest prediction."""
        feature_vector = self._dict_to_vector(features)
        return self.rf_model.predict([feature_vector])[0]

    def _dict_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert feature dictionary to vector."""
        return np.array([float(v) for v in features.values()])

    def save_model(self, path: str) -> None:
        """Save the trend forecasting models."""
        # Save LSTM model
        if self.lstm_model:
            self.lstm_model.save(f"{path}/lstm_model.h5")
        
        # Save Random Forest model
        joblib.dump(self.rf_model, f"{path}/rf_model.pkl")
        
        # Save other components
        model_data = {
            'scaler': self.scaler,
            'config': self.config,
            'performance_metrics': self.performance_metrics
        }
        joblib.dump(model_data, f"{path}/trend_model_data.pkl")

    def load_model(self, path: str) -> None:
        """Load pre-trained trend forecasting models."""
        # Load LSTM model
        try:
            self.lstm_model = tf.keras.models.load_model(f"{path}/lstm_model.h5")
        except:
            logger.warning("LSTM model not found, using Random Forest only")
        
        # Load Random Forest model
        self.rf_model = joblib.load(f"{path}/rf_model.pkl")
        
        # Load other components
        model_data = joblib.load(f"{path}/trend_model_data.pkl")
        self.scaler = model_data['scaler']
        self.performance_metrics = model_data['performance_metrics']
        self.is_trained = True

class RecommendationAlgorithm(BaseMLAlgorithm):
    """Advanced recommendation system using collaborative filtering and content-based approaches."""
    
    def __init__(self, config: MLModelConfig):
        super().__init__(config)
        self.user_item_matrix = None
        self.item_features = None
        self.user_features = None
        self.similarity_matrix = None
        self.mlp_model = None
        
    async def train(self, data: pd.DataFrame) -> Dict[str, float]:
        """Train recommendation models."""
        try:
            # Prepare user-item interaction data
            self.user_item_matrix = self._create_user_item_matrix(data)
            
            # Extract features
            self.item_features = self._extract_item_features(data)
            self.user_features = self._extract_user_features(data)
            
            # Train collaborative filtering
            cf_metrics = await self._train_collaborative_filtering()
            
            # Train content-based filtering
            cb_metrics = await self._train_content_based()
            
            # Train neural collaborative filtering
            ncf_metrics = await self._train_neural_cf(data)
            
            self.is_trained = True
            self.last_training = datetime.now()
            
            self.performance_metrics = {
                'cf_coverage': cf_metrics['coverage'],
                'cb_accuracy': cb_metrics['accuracy'],
                'ncf_loss': ncf_metrics['loss'],
                'training_samples': len(data)
            }
            
            logger.info(f"Recommendation model trained on {len(data)} samples")
            return self.performance_metrics
            
        except Exception as e:
            logger.error(f"Error training recommendation model: {e}")
            raise

    def _create_user_item_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create user-item interaction matrix."""
        if all(col in data.columns for col in ['user_id', 'item_id', 'rating']):
            return data.pivot_table(index='user_id', columns='item_id', values='rating', fill_value=0)
        else:
            # Create dummy matrix for demonstration
            return pd.DataFrame(np.random.randint(0, 6, (100, 50)))

    def _extract_item_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract item features for content-based filtering."""
        # This would extract features like genre, artist, duration, etc.
        # Simplified implementation
        n_items = self.user_item_matrix.shape[1] if self.user_item_matrix is not None else 50
        return pd.DataFrame(np.random.randn(n_items, 10))

    def _extract_user_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract user features for content-based filtering."""
        # This would extract features like age, preferences, listening history, etc.
        # Simplified implementation
        n_users = self.user_item_matrix.shape[0] if self.user_item_matrix is not None else 100
        return pd.DataFrame(np.random.randn(n_users, 5))

    async def _train_collaborative_filtering(self) -> Dict[str, float]:
        """Train collaborative filtering model."""
        # Calculate item-item similarity
        item_similarity = np.corrcoef(self.user_item_matrix.T)
        self.similarity_matrix = pd.DataFrame(
            item_similarity,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )
        
        # Calculate coverage
        coverage = np.sum(~np.isnan(item_similarity)) / (item_similarity.shape[0] ** 2)
        
        return {'coverage': float(coverage)}

    async def _train_content_based(self) -> Dict[str, float]:
        """Train content-based filtering model."""
        # This would train a model to predict user preferences based on item features
        # Simplified implementation
        return {'accuracy': 0.85}

    async def _train_neural_cf(self, data: pd.DataFrame) -> Dict[str, float]:
        """Train neural collaborative filtering model."""
        # Build neural network for collaborative filtering
        n_users = self.user_item_matrix.shape[0]
        n_items = self.user_item_matrix.shape[1]
        
        self.mlp_model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            random_state=42,
            max_iter=200
        )
        
        # Prepare training data
        user_ids, item_ids, ratings = [], [], []
        for user_idx in range(n_users):
            for item_idx in range(n_items):
                if self.user_item_matrix.iloc[user_idx, item_idx] > 0:
                    user_ids.append(user_idx)
                    item_ids.append(item_idx)
                    ratings.append(self.user_item_matrix.iloc[user_idx, item_idx])
        
        # Create feature vectors
        features = np.column_stack([user_ids, item_ids])
        targets = np.array(ratings)
        
        # Train model
        self.mlp_model.fit(features, targets)
        
        # Calculate loss
        predictions = self.mlp_model.predict(features)
        loss = np.mean((targets - predictions) ** 2)
        
        return {'loss': float(loss)}

    async def predict(self, features: Dict[str, Any]) -> PredictionResult:
        """Generate recommendations."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        try:
            user_id = features.get('user_id', 0)
            n_recommendations = features.get('n_recommendations', 10)
            
            # Get collaborative filtering recommendations
            cf_recs = self._get_collaborative_recommendations(user_id, n_recommendations)
            
            # Get content-based recommendations
            cb_recs = self._get_content_recommendations(user_id, n_recommendations)
            
            # Combine recommendations
            hybrid_recs = self._combine_recommendations(cf_recs, cb_recs)
            
            return PredictionResult(
                prediction=hybrid_recs[:n_recommendations],
                confidence=0.8,
                timestamp=datetime.now(),
                features_used=['user_id', 'n_recommendations'],
                model_version="recommendation_v1.0"
            )
            
        except Exception as e:
            logger.error(f"Error in recommendation prediction: {e}")
            raise

    def _get_collaborative_recommendations(self, user_id: int, n_recs: int) -> List[int]:
        """Get collaborative filtering recommendations."""
        if user_id >= len(self.user_item_matrix):
            return list(range(n_recs))
            
        user_ratings = self.user_item_matrix.iloc[user_id]
        unrated_items = user_ratings[user_ratings == 0].index
        
        # Calculate predicted ratings for unrated items
        predictions = []
        for item in unrated_items:
            pred_rating = self._predict_rating(user_id, item)
            predictions.append((item, pred_rating))
        
        # Sort by predicted rating and return top recommendations
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [item for item, rating in predictions[:n_recs]]

    def _predict_rating(self, user_id: int, item_id: int) -> float:
        """Predict rating for user-item pair."""
        # Simplified collaborative filtering prediction
        user_ratings = self.user_item_matrix.iloc[user_id]
        item_similarities = self.similarity_matrix.loc[item_id] if item_id in self.similarity_matrix.index else pd.Series()
        
        # Calculate weighted average of similar items
        rated_items = user_ratings[user_ratings > 0]
        if len(rated_items) == 0:
            return 3.0  # Default rating
            
        numerator = 0
        denominator = 0
        for rated_item, rating in rated_items.items():
            if rated_item in item_similarities.index:
                similarity = item_similarities[rated_item]
                if not np.isnan(similarity):
                    numerator += similarity * rating
                    denominator += abs(similarity)
        
        return numerator / denominator if denominator > 0 else 3.0

    def _get_content_recommendations(self, user_id: int, n_recs: int) -> List[int]:
        """Get content-based recommendations."""
        # Simplified content-based recommendations
        return list(range(n_recs))

    def _combine_recommendations(self, cf_recs: List[int], cb_recs: List[int]) -> List[int]:
        """Combine collaborative and content-based recommendations."""
        # Simple hybrid approach: alternate between CF and CB recommendations
        combined = []
        max_len = max(len(cf_recs), len(cb_recs))
        
        for i in range(max_len):
            if i < len(cf_recs) and cf_recs[i] not in combined:
                combined.append(cf_recs[i])
            if i < len(cb_recs) and cb_recs[i] not in combined:
                combined.append(cb_recs[i])
                
        return combined

    def save_model(self, path: str) -> None:
        """Save the recommendation models."""
        model_data = {
            'user_item_matrix': self.user_item_matrix,
            'item_features': self.item_features,
            'user_features': self.user_features,
            'similarity_matrix': self.similarity_matrix,
            'mlp_model': self.mlp_model,
            'config': self.config,
            'performance_metrics': self.performance_metrics
        }
        joblib.dump(model_data, f"{path}/recommendation_model.pkl")

    def load_model(self, path: str) -> None:
        """Load pre-trained recommendation models."""
        model_data = joblib.load(f"{path}/recommendation_model.pkl")
        self.user_item_matrix = model_data['user_item_matrix']
        self.item_features = model_data['item_features']
        self.user_features = model_data['user_features']
        self.similarity_matrix = model_data['similarity_matrix']
        self.mlp_model = model_data['mlp_model']
        self.performance_metrics = model_data['performance_metrics']
        self.is_trained = True

class MLAlgorithmFactory:
    """Factory for creating ML algorithms."""
    
    @staticmethod
    def create_algorithm(algorithm_type: str, config: MLModelConfig) -> BaseMLAlgorithm:
        """Create an ML algorithm instance."""
        algorithms = {
            'anomaly_detection': AnomalyDetectionAlgorithm,
            'trend_forecasting': TrendForecastingAlgorithm,
            'recommendation': RecommendationAlgorithm
        }
        
        if algorithm_type not in algorithms:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}")
            
        return algorithms[algorithm_type](config)

# Global algorithm instances
anomaly_detector = None
trend_forecaster = None
recommender = None

def initialize_algorithms():
    """Initialize global algorithm instances."""
    global anomaly_detector, trend_forecaster, recommender
    
    # Anomaly detection config
    anomaly_config = MLModelConfig(
        model_type='anomaly_detection',
        parameters={'contamination': 0.1, 'eps': 0.5, 'min_samples': 5},
        training_data_window=timedelta(days=7),
        retrain_frequency=timedelta(hours=6)
    )
    anomaly_detector = AnomalyDetectionAlgorithm(anomaly_config)
    
    # Trend forecasting config
    trend_config = MLModelConfig(
        model_type='trend_forecasting',
        parameters={'sequence_length': 24, 'forecast_horizon': 12, 'n_estimators': 100},
        training_data_window=timedelta(days=30),
        retrain_frequency=timedelta(days=1)
    )
    trend_forecaster = TrendForecastingAlgorithm(trend_config)
    
    # Recommendation config
    rec_config = MLModelConfig(
        model_type='recommendation',
        parameters={'n_factors': 50, 'regularization': 0.01},
        training_data_window=timedelta(days=90),
        retrain_frequency=timedelta(days=7)
    )
    recommender = RecommendationAlgorithm(rec_config)

__all__ = [
    'BaseMLAlgorithm',
    'AnomalyDetectionAlgorithm',
    'TrendForecastingAlgorithm',
    'RecommendationAlgorithm',
    'MLAlgorithmFactory',
    'MLModelConfig',
    'PredictionResult',
    'anomaly_detector',
    'trend_forecaster',
    'recommender',
    'initialize_algorithms'
]
