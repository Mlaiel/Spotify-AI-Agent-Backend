"""
Predictive Alerting System for Spotify AI Agent
===============================================

Advanced predictive alerting using machine learning forecasting models
to proactively detect potential issues before they occur.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import logging

@dataclass
class PredictiveAlert:
    """Predictive alert with forecast data and confidence."""
    alert_id: str
    tenant_id: str
    metric_name: str
    predicted_value: float
    confidence_interval: Tuple[float, float]
    prediction_horizon: timedelta
    severity: str
    probability: float
    timestamp: datetime
    recommendations: List[str]
    metadata: Dict[str, Any]

class PredictiveAlerting:
    """
    Machine learning-based predictive alerting system.
    
    Features:
    - Time series forecasting with LSTM
    - Trend analysis and extrapolation
    - Capacity planning predictions
    - Resource utilization forecasting
    - Multi-step ahead predictions
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.logger = logging.getLogger(__name__)
        self.prediction_horizon = config.get('prediction_horizon_hours', 24)
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize predictive models."""
        # LSTM for time series forecasting
        self._build_forecasting_lstm()
        
        # Random Forest for feature-based prediction
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
    def _build_forecasting_lstm(self):
        """Build LSTM model for time series forecasting."""
        sequence_length = self.config.get('sequence_length', 60)
        n_features = self.config.get('n_features', 5)
        
        model = keras.Sequential([
            keras.layers.LSTM(128, return_sequences=True, 
                            input_shape=(sequence_length, n_features)),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(64, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32, return_sequences=False),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.models['forecasting_lstm'] = model
        
    def generate_predictions(self, 
                           historical_data: pd.DataFrame,
                           tenant_id: str) -> List[PredictiveAlert]:
        """
        Generate predictive alerts based on historical data.
        
        Args:
            historical_data: Time series data for prediction
            tenant_id: Tenant identifier
            
        Returns:
            List of predictive alerts
        """
        alerts = []
        
        # Prepare data for prediction
        prepared_data = self._prepare_time_series_data(historical_data)
        
        # Generate forecasts for each metric
        for metric_name in prepared_data.columns:
            if metric_name != 'timestamp':
                metric_alerts = self._predict_metric_alerts(
                    prepared_data, metric_name, tenant_id
                )
                alerts.extend(metric_alerts)
                
        return alerts
        
    def _prepare_time_series_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare time series data for forecasting."""
        # Ensure timestamp column exists
        if 'timestamp' not in data.columns:
            data['timestamp'] = pd.date_range(
                start=datetime.now() - timedelta(hours=len(data)),
                periods=len(data),
                freq='H'
            )
            
        # Sort by timestamp
        data = data.sort_values('timestamp')
        
        # Handle missing values
        data = data.fillna(method='forward').fillna(method='backward')
        
        # Add time-based features
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['month'] = data['timestamp'].dt.month
        
        return data
        
    def _predict_metric_alerts(self, 
                             data: pd.DataFrame,
                             metric_name: str,
                             tenant_id: str) -> List[PredictiveAlert]:
        """Predict alerts for a specific metric."""
        alerts = []
        
        if metric_name not in data.columns:
            return alerts
            
        # Create sequences for LSTM
        sequences, targets = self._create_forecasting_sequences(
            data[metric_name].values
        )
        
        if len(sequences) == 0:
            return alerts
            
        # Train model if not trained
        if not hasattr(self.models['forecasting_lstm'], 'trained'):
            self._train_forecasting_model(sequences, targets)
            
        # Generate predictions
        last_sequence = sequences[-1:].reshape(1, -1, 1)
        predictions = []
        
        # Multi-step prediction
        current_seq = last_sequence.copy()
        for step in range(self.prediction_horizon):
            pred = self.models['forecasting_lstm'].predict(current_seq, verbose=0)
            predictions.append(pred[0, 0])
            
            # Update sequence for next prediction
            current_seq = np.roll(current_seq, -1, axis=1)
            current_seq[0, -1, 0] = pred[0, 0]
            
        # Analyze predictions for potential issues
        alert_predictions = self._analyze_predictions(
            predictions, data[metric_name], metric_name, tenant_id
        )
        
        alerts.extend(alert_predictions)
        return alerts
        
    def _create_forecasting_sequences(self, 
                                    data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for forecasting model."""
        sequence_length = self.config.get('sequence_length', 60)
        
        if len(data) < sequence_length + 1:
            return np.array([]), np.array([])
            
        sequences = []
        targets = []
        
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:i + sequence_length])
            targets.append(data[i + sequence_length])
            
        return np.array(sequences), np.array(targets)
        
    def _train_forecasting_model(self, sequences: np.ndarray, targets: np.ndarray):
        """Train the forecasting LSTM model."""
        if len(sequences) == 0:
            return
            
        # Reshape for LSTM input
        X = sequences.reshape(sequences.shape[0], sequences.shape[1], 1)
        y = targets
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train model
        early_stopping = keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True
        )
        
        self.models['forecasting_lstm'].fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Mark as trained
        self.models['forecasting_lstm'].trained = True
        
    def _analyze_predictions(self, 
                           predictions: List[float],
                           historical_data: pd.Series,
                           metric_name: str,
                           tenant_id: str) -> List[PredictiveAlert]:
        """Analyze predictions to generate alerts."""
        alerts = []
        
        # Calculate statistical thresholds
        mean_val = historical_data.mean()
        std_val = historical_data.std()
        
        # Define thresholds
        critical_threshold = mean_val + 3 * std_val
        warning_threshold = mean_val + 2 * std_val
        
        # Analyze each prediction
        for i, pred_value in enumerate(predictions):
            prediction_time = datetime.now() + timedelta(hours=i+1)
            
            # Determine severity
            severity = 'info'
            probability = 0.0
            recommendations = []
            
            if pred_value > critical_threshold:
                severity = 'critical'
                probability = min((pred_value - critical_threshold) / std_val, 1.0)
                recommendations = self._get_critical_recommendations(metric_name)
            elif pred_value > warning_threshold:
                severity = 'warning'
                probability = (pred_value - warning_threshold) / std_val
                recommendations = self._get_warning_recommendations(metric_name)
            
            if severity != 'info':
                # Calculate confidence interval
                confidence_range = std_val * 1.96  # 95% confidence
                confidence_interval = (
                    pred_value - confidence_range,
                    pred_value + confidence_range
                )
                
                alert = PredictiveAlert(
                    alert_id=f"pred_{tenant_id}_{metric_name}_{i}_{int(prediction_time.timestamp())}",
                    tenant_id=tenant_id,
                    metric_name=metric_name,
                    predicted_value=pred_value,
                    confidence_interval=confidence_interval,
                    prediction_horizon=timedelta(hours=i+1),
                    severity=severity,
                    probability=probability,
                    timestamp=prediction_time,
                    recommendations=recommendations,
                    metadata={
                        'historical_mean': mean_val,
                        'historical_std': std_val,
                        'threshold_used': warning_threshold if severity == 'warning' else critical_threshold,
                        'prediction_step': i+1
                    }
                )
                alerts.append(alert)
                
        return alerts
        
    def _get_critical_recommendations(self, metric_name: str) -> List[str]:
        """Get recommendations for critical alerts."""
        recommendations = {
            'cpu_usage': [
                'Scale horizontally by adding more instances',
                'Optimize CPU-intensive operations',
                'Review and optimize database queries',
                'Consider implementing caching strategies'
            ],
            'memory_usage': [
                'Increase memory allocation',
                'Identify and fix memory leaks',
                'Optimize data structures',
                'Implement memory pooling'
            ],
            'disk_usage': [
                'Clean up old logs and temporary files',
                'Archive historical data',
                'Add more storage capacity',
                'Implement data compression'
            ],
            'response_time': [
                'Scale the application tier',
                'Optimize database queries',
                'Add caching layers',
                'Review network configuration'
            ]
        }
        
        return recommendations.get(metric_name, [
            'Monitor the situation closely',
            'Prepare scaling procedures',
            'Review system resources'
        ])
        
    def _get_warning_recommendations(self, metric_name: str) -> List[str]:
        """Get recommendations for warning alerts."""
        recommendations = {
            'cpu_usage': [
                'Monitor CPU trends',
                'Prepare for potential scaling',
                'Review recent deployments'
            ],
            'memory_usage': [
                'Monitor memory consumption',
                'Check for gradual memory increase',
                'Review application behavior'
            ],
            'disk_usage': [
                'Plan disk cleanup activities',
                'Monitor disk growth rate',
                'Review log retention policies'
            ],
            'response_time': [
                'Monitor response time trends',
                'Check for performance regressions',
                'Review recent changes'
            ]
        }
        
        return recommendations.get(metric_name, [
            'Monitor the trend',
            'Prepare mitigation strategies'
        ])
        
    def get_forecast_accuracy(self, 
                            test_data: pd.DataFrame,
                            metric_name: str) -> Dict[str, float]:
        """Calculate forecast accuracy metrics."""
        if metric_name not in test_data.columns:
            return {}
            
        # Generate predictions for test data
        sequences, actual = self._create_forecasting_sequences(
            test_data[metric_name].values
        )
        
        if len(sequences) == 0:
            return {}
            
        X_test = sequences.reshape(sequences.shape[0], sequences.shape[1], 1)
        predictions = self.models['forecasting_lstm'].predict(X_test, verbose=0)
        
        # Calculate metrics
        mae = mean_absolute_error(actual, predictions.flatten())
        mse = mean_squared_error(actual, predictions.flatten())
        rmse = np.sqrt(mse)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((actual - predictions.flatten()) / actual)) * 100
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'samples_tested': len(actual)
        }
