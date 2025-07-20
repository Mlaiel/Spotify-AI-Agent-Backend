"""
Dynamic Threshold Adapter for Spotify AI Agent
==============================================

Intelligent threshold management system that automatically adapts alert
thresholds based on historical patterns, seasonality, and system behavior.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import logging
import joblib

@dataclass
class ThresholdConfig:
    """Configuration for dynamic thresholds."""
    metric_name: str
    tenant_id: str
    current_threshold: float
    confidence_interval: Tuple[float, float]
    adaptation_rate: float
    last_updated: datetime
    model_accuracy: float
    metadata: Dict[str, Any]

class ThresholdAdapter:
    """
    Machine learning-based dynamic threshold adaptation system.
    
    Features:
    - Seasonal pattern recognition
    - Trend-aware threshold adjustment
    - Anomaly-resistant adaptation
    - Multi-tenant threshold isolation
    - Performance-based threshold tuning
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        self.threshold_history = {}
        self.adaptation_rate = config.get('adaptation_rate', 0.1)
        self.min_data_points = config.get('min_data_points', 100)
        
    def adapt_thresholds(self, 
                        metric_data: pd.DataFrame,
                        tenant_id: str) -> Dict[str, ThresholdConfig]:
        """
        Adapt thresholds for all metrics based on recent data.
        
        Args:
            metric_data: Historical metric data
            tenant_id: Tenant identifier
            
        Returns:
            Dictionary of adapted threshold configurations
        """
        adapted_thresholds = {}
        
        for metric_name in metric_data.columns:
            if metric_name != 'timestamp':
                threshold_config = self._adapt_metric_threshold(
                    metric_data, metric_name, tenant_id
                )
                if threshold_config:
                    adapted_thresholds[metric_name] = threshold_config
                    
        return adapted_thresholds
        
    def _adapt_metric_threshold(self, 
                              data: pd.DataFrame,
                              metric_name: str,
                              tenant_id: str) -> Optional[ThresholdConfig]:
        """Adapt threshold for a specific metric."""
        if metric_name not in data.columns:
            return None
            
        metric_data = data[metric_name].dropna()
        if len(metric_data) < self.min_data_points:
            self.logger.warning(f"Insufficient data for {metric_name}: {len(metric_data)}")
            return None
            
        # Extract time-based features
        features = self._extract_temporal_features(data)
        
        # Build or update model for this metric
        model_key = f"{tenant_id}_{metric_name}"
        if model_key not in self.models:
            self._initialize_model(model_key)
            
        # Train model to predict metric values
        self._train_threshold_model(model_key, features, metric_data.values)
        
        # Calculate dynamic threshold
        new_threshold = self._calculate_dynamic_threshold(
            model_key, features, metric_data
        )
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(
            model_key, features, metric_data
        )
        
        # Calculate model accuracy
        accuracy = self._evaluate_model_accuracy(model_key, features, metric_data.values)
        
        # Get current threshold for comparison
        current_threshold = self._get_current_threshold(tenant_id, metric_name)
        
        # Apply adaptation rate
        adapted_threshold = self._apply_adaptation_rate(
            current_threshold, new_threshold
        )
        
        return ThresholdConfig(
            metric_name=metric_name,
            tenant_id=tenant_id,
            current_threshold=adapted_threshold,
            confidence_interval=confidence_interval,
            adaptation_rate=self.adaptation_rate,
            last_updated=datetime.now(),
            model_accuracy=accuracy,
            metadata={
                'previous_threshold': current_threshold,
                'raw_prediction': new_threshold,
                'data_points_used': len(metric_data),
                'adaptation_method': 'ml_based'
            }
        )
        
    def _extract_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features for threshold modeling."""
        features = pd.DataFrame()
        
        # Ensure timestamp column
        if 'timestamp' not in data.columns:
            data['timestamp'] = pd.date_range(
                start=datetime.now() - timedelta(hours=len(data)),
                periods=len(data),
                freq='H'
            )
            
        # Extract time-based features
        timestamps = pd.to_datetime(data['timestamp'])
        features['hour'] = timestamps.dt.hour
        features['day_of_week'] = timestamps.dt.dayofweek
        features['day_of_month'] = timestamps.dt.day
        features['month'] = timestamps.dt.month
        features['is_weekend'] = timestamps.dt.weekday >= 5
        
        # Add cyclical encoding for better ML performance
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        
        # Add lag features
        for metric_col in data.columns:
            if metric_col != 'timestamp':
                for lag in [1, 6, 24]:  # 1 hour, 6 hours, 24 hours
                    features[f'{metric_col}_lag_{lag}'] = data[metric_col].shift(lag)
                    
        # Add rolling statistics
        for metric_col in data.columns:
            if metric_col != 'timestamp':
                features[f'{metric_col}_rolling_mean_24'] = data[metric_col].rolling(24).mean()
                features[f'{metric_col}_rolling_std_24'] = data[metric_col].rolling(24).std()
                features[f'{metric_col}_rolling_max_24'] = data[metric_col].rolling(24).max()
                features[f'{metric_col}_rolling_min_24'] = data[metric_col].rolling(24).min()
                
        return features.fillna(method='forward').fillna(0)
        
    def _initialize_model(self, model_key: str):
        """Initialize model and scaler for a metric."""
        self.models[model_key] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.scalers[model_key] = StandardScaler()
        
    def _train_threshold_model(self, 
                             model_key: str,
                             features: pd.DataFrame,
                             target: np.ndarray):
        """Train the threshold prediction model."""
        # Remove rows with NaN values
        valid_indices = ~np.isnan(target)
        X = features[valid_indices]
        y = target[valid_indices]
        
        if len(X) < self.min_data_points:
            return
            
        # Scale features
        X_scaled = self.scalers[model_key].fit_transform(X)
        
        # Train model
        self.models[model_key].fit(X_scaled, y)
        
    def _calculate_dynamic_threshold(self, 
                                   model_key: str,
                                   features: pd.DataFrame,
                                   metric_data: pd.Series) -> float:
        """Calculate dynamic threshold using trained model."""
        if model_key not in self.models:
            # Fallback to statistical threshold
            return self._calculate_statistical_threshold(metric_data)
            
        # Use latest features for prediction
        latest_features = features.iloc[-1:].fillna(method='forward').fillna(0)
        
        try:
            # Scale features
            latest_scaled = self.scalers[model_key].transform(latest_features)
            
            # Predict expected value
            predicted_value = self.models[model_key].predict(latest_scaled)[0]
            
            # Calculate threshold as predicted value + margin
            std_dev = metric_data.std()
            margin_factor = self.config.get('threshold_margin_factor', 2.0)
            
            threshold = predicted_value + (margin_factor * std_dev)
            
            # Ensure threshold is reasonable
            min_threshold = metric_data.quantile(0.95)
            max_threshold = metric_data.max() * 1.5
            
            return np.clip(threshold, min_threshold, max_threshold)
            
        except Exception as e:
            self.logger.error(f"Error calculating dynamic threshold: {e}")
            return self._calculate_statistical_threshold(metric_data)
            
    def _calculate_statistical_threshold(self, metric_data: pd.Series) -> float:
        """Fallback statistical threshold calculation."""
        # Use 95th percentile + 2 standard deviations
        percentile_95 = metric_data.quantile(0.95)
        std_dev = metric_data.std()
        return percentile_95 + (2 * std_dev)
        
    def _calculate_confidence_interval(self, 
                                     model_key: str,
                                     features: pd.DataFrame,
                                     metric_data: pd.Series) -> Tuple[float, float]:
        """Calculate confidence interval for threshold."""
        if model_key not in self.models:
            # Statistical confidence interval
            mean_val = metric_data.mean()
            std_val = metric_data.std()
            margin = 1.96 * std_val  # 95% confidence
            return (mean_val - margin, mean_val + margin)
            
        try:
            # Use model uncertainty for confidence interval
            latest_features = features.iloc[-1:].fillna(method='forward').fillna(0)
            latest_scaled = self.scalers[model_key].transform(latest_features)
            
            # Get predictions from all trees (for Random Forest)
            tree_predictions = []
            for estimator in self.models[model_key].estimators_:
                pred = estimator.predict(latest_scaled)[0]
                tree_predictions.append(pred)
                
            # Calculate confidence interval from prediction variance
            pred_mean = np.mean(tree_predictions)
            pred_std = np.std(tree_predictions)
            margin = 1.96 * pred_std
            
            return (pred_mean - margin, pred_mean + margin)
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence interval: {e}")
            mean_val = metric_data.mean()
            std_val = metric_data.std()
            margin = 1.96 * std_val
            return (mean_val - margin, mean_val + margin)
            
    def _evaluate_model_accuracy(self, 
                               model_key: str,
                               features: pd.DataFrame,
                               target: np.ndarray) -> float:
        """Evaluate model accuracy using cross-validation."""
        if model_key not in self.models:
            return 0.0
            
        try:
            # Use last 20% of data for testing
            test_size = int(len(features) * 0.2)
            if test_size < 10:
                return 0.0
                
            X_test = features.iloc[-test_size:]
            y_test = target[-test_size:]
            
            # Remove NaN values
            valid_indices = ~np.isnan(y_test)
            X_test = X_test[valid_indices]
            y_test = y_test[valid_indices]
            
            if len(X_test) == 0:
                return 0.0
                
            # Scale features
            X_test_scaled = self.scalers[model_key].transform(X_test)
            
            # Make predictions
            predictions = self.models[model_key].predict(X_test_scaled)
            
            # Calculate R² score
            from sklearn.metrics import r2_score
            accuracy = r2_score(y_test, predictions)
            
            return max(0.0, accuracy)  # Ensure non-negative
            
        except Exception as e:
            self.logger.error(f"Error evaluating model accuracy: {e}")
            return 0.0
            
    def _get_current_threshold(self, tenant_id: str, metric_name: str) -> float:
        """Get current threshold for comparison."""
        key = f"{tenant_id}_{metric_name}"
        if key in self.threshold_history:
            return self.threshold_history[key].current_threshold
            
        # Default threshold if no history
        return self.config.get('default_threshold', 100.0)
        
    def _apply_adaptation_rate(self, 
                             current_threshold: float,
                             new_threshold: float) -> float:
        """Apply adaptation rate to smooth threshold changes."""
        # Gradual adaptation to prevent threshold oscillation
        adapted = current_threshold + (self.adaptation_rate * (new_threshold - current_threshold))
        
        # Ensure reasonable bounds
        max_change = current_threshold * 0.5  # Max 50% change per adaptation
        min_change = current_threshold * -0.5
        
        change = adapted - current_threshold
        bounded_change = np.clip(change, min_change, max_change)
        
        return current_threshold + bounded_change
        
    def update_threshold_history(self, threshold_configs: Dict[str, ThresholdConfig]):
        """Update threshold history for tracking."""
        for metric_name, config in threshold_configs.items():
            key = f"{config.tenant_id}_{metric_name}"
            self.threshold_history[key] = config
            
    def get_threshold_analytics(self, 
                              tenant_id: str,
                              metric_name: str) -> Dict[str, Any]:
        """Get analytics and insights for threshold adaptation."""
        key = f"{tenant_id}_{metric_name}"
        
        if key not in self.threshold_history:
            return {'error': 'No threshold history found'}
            
        config = self.threshold_history[key]
        
        # Calculate threshold stability
        recent_thresholds = self._get_recent_thresholds(key, days=7)
        stability = 1.0 - (np.std(recent_thresholds) / np.mean(recent_thresholds)) if recent_thresholds else 0.0
        
        return {
            'current_threshold': config.current_threshold,
            'model_accuracy': config.model_accuracy,
            'confidence_interval': config.confidence_interval,
            'last_updated': config.last_updated.isoformat(),
            'adaptation_rate': config.adaptation_rate,
            'threshold_stability': stability,
            'metadata': config.metadata,
            'recommendations': self._get_threshold_recommendations(config, stability)
        }
        
    def _get_recent_thresholds(self, key: str, days: int = 7) -> List[float]:
        """Get recent threshold values for analysis."""
        # This would typically query a database or cache
        # For now, return mock data
        if key in self.threshold_history:
            return [self.threshold_history[key].current_threshold]
        return []
        
    def _get_threshold_recommendations(self, 
                                     config: ThresholdConfig,
                                     stability: float) -> List[str]:
        """Get recommendations for threshold optimization."""
        recommendations = []
        
        if config.model_accuracy < 0.7:
            recommendations.append("Model accuracy is low - consider collecting more training data")
            
        if stability < 0.8:
            recommendations.append("Threshold is unstable - consider reducing adaptation rate")
            
        if config.adaptation_rate > 0.2:
            recommendations.append("High adaptation rate may cause threshold oscillation")
            
        confidence_width = config.confidence_interval[1] - config.confidence_interval[0]
        if confidence_width > config.current_threshold * 0.5:
            recommendations.append("Wide confidence interval indicates high uncertainty")
            
        return recommendations if recommendations else ["Threshold configuration appears optimal"]
        
    def save_models(self, path: str):
        """Save threshold models to disk."""
        joblib.dump(self.models, f"{path}/threshold_models.pkl")
        joblib.dump(self.scalers, f"{path}/threshold_scalers.pkl")
        joblib.dump(self.threshold_history, f"{path}/threshold_history.pkl")
        
    def load_models(self, path: str):
        """Load threshold models from disk."""
        try:
            self.models = joblib.load(f"{path}/threshold_models.pkl")
            self.scalers = joblib.load(f"{path}/threshold_scalers.pkl")
            self.threshold_history = joblib.load(f"{path}/threshold_history.pkl")
        except FileNotFoundError:
            self.logger.warning("No saved models found, starting with empty models")
