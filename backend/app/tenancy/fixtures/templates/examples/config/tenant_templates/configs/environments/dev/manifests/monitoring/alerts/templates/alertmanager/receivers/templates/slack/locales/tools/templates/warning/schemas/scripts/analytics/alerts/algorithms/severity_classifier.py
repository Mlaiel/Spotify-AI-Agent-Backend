"""
Intelligent Severity Classifier for Spotify AI Agent
====================================================

Advanced machine learning-based severity classification system that dynamically
determines alert severity based on context, impact, and business rules.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging

@dataclass
class SeverityClassificationResult:
    """Result of severity classification."""
    original_severity: str
    predicted_severity: str
    confidence: float
    severity_probabilities: Dict[str, float]
    contributing_factors: List[str]
    business_impact_score: float
    recommendation: str

@dataclass
class BusinessImpactContext:
    """Business impact context for severity calculation."""
    tenant_tier: str  # premium, standard, basic
    service_criticality: str  # critical, important, standard
    affected_users: int
    revenue_impact: float
    sla_threshold: float
    time_of_day: str
    day_of_week: str

class SeverityClassifier:
    """
    ML-based intelligent severity classification system.
    
    Features:
    - Multi-model ensemble classification
    - Business impact analysis
    - Contextual severity adjustment
    - Temporal pattern consideration
    - User impact assessment
    - SLA impact evaluation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_importance = {}
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize classification models."""
        # Random Forest for robust classification
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            random_state=42,
            class_weight='balanced'
        )
        
        # Gradient Boosting for complex patterns
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # Standard scaler for numerical features
        self.scalers['standard'] = StandardScaler()
        
        # Label encoder for severity levels
        self.label_encoders['severity'] = LabelEncoder()
        
    def classify_severity(self, 
                         alert_data: Dict[str, Any],
                         business_context: BusinessImpactContext,
                         historical_data: Optional[pd.DataFrame] = None) -> SeverityClassificationResult:
        """
        Classify alert severity using ML models and business context.
        
        Args:
            alert_data: Alert information
            business_context: Business impact context
            historical_data: Historical alert data for context
            
        Returns:
            Severity classification result
        """
        # Extract features from alert and context
        features = self._extract_classification_features(alert_data, business_context, historical_data)
        
        # Get predictions from all models
        predictions = self._get_model_predictions(features)
        
        # Calculate business impact score
        business_impact = self._calculate_business_impact(alert_data, business_context)
        
        # Ensemble prediction
        final_prediction, confidence, probabilities = self._ensemble_prediction(predictions, business_impact)
        
        # Get contributing factors
        contributing_factors = self._identify_contributing_factors(features, final_prediction)
        
        # Generate recommendation
        recommendation = self._generate_severity_recommendation(
            final_prediction, confidence, business_impact, alert_data
        )
        
        return SeverityClassificationResult(
            original_severity=alert_data.get('severity', 'unknown'),
            predicted_severity=final_prediction,
            confidence=confidence,
            severity_probabilities=probabilities,
            contributing_factors=contributing_factors,
            business_impact_score=business_impact,
            recommendation=recommendation
        )
        
    def _extract_classification_features(self, 
                                       alert_data: Dict[str, Any],
                                       business_context: BusinessImpactContext,
                                       historical_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Extract features for severity classification."""
        features = {}
        
        # Basic alert features
        features['metric_value'] = float(alert_data.get('value', 0))
        features['threshold'] = float(alert_data.get('threshold', 0))
        features['threshold_ratio'] = (
            features['metric_value'] / max(features['threshold'], 0.1)
        )
        
        # Temporal features
        timestamp = pd.to_datetime(alert_data.get('timestamp', datetime.now()))
        features['hour'] = timestamp.hour
        features['day_of_week'] = timestamp.weekday()
        features['month'] = timestamp.month
        features['is_weekend'] = timestamp.weekday() >= 5
        features['is_business_hours'] = 9 <= timestamp.hour <= 17
        
        # Business context features
        features['tenant_tier_score'] = self._encode_tenant_tier(business_context.tenant_tier)
        features['service_criticality_score'] = self._encode_service_criticality(
            business_context.service_criticality
        )
        features['affected_users'] = business_context.affected_users
        features['revenue_impact'] = business_context.revenue_impact
        features['sla_threshold'] = business_context.sla_threshold
        
        # Metric type features
        metric_name = alert_data.get('metric_name', '').lower()
        features['is_cpu_metric'] = 'cpu' in metric_name
        features['is_memory_metric'] = 'memory' in metric_name
        features['is_disk_metric'] = 'disk' in metric_name
        features['is_network_metric'] = 'network' in metric_name
        features['is_response_time_metric'] = any(term in metric_name for term in ['response', 'latency'])
        features['is_error_rate_metric'] = 'error' in metric_name
        
        # Historical context features
        if historical_data is not None and not historical_data.empty:
            hist_features = self._extract_historical_features(alert_data, historical_data)
            features.update(hist_features)
        else:
            # Default values when no historical data
            features.update({
                'recent_alert_frequency': 0,
                'historical_severity_trend': 0,
                'metric_volatility': 0,
                'recent_escalations': 0
            })
            
        return features
        
    def _encode_tenant_tier(self, tier: str) -> float:
        """Encode tenant tier as numerical score."""
        tier_scores = {'premium': 1.0, 'standard': 0.7, 'basic': 0.4}
        return tier_scores.get(tier.lower(), 0.5)
        
    def _encode_service_criticality(self, criticality: str) -> float:
        """Encode service criticality as numerical score."""
        criticality_scores = {'critical': 1.0, 'important': 0.7, 'standard': 0.4}
        return criticality_scores.get(criticality.lower(), 0.5)
        
    def _extract_historical_features(self, 
                                   alert_data: Dict[str, Any],
                                   historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Extract features from historical alert data."""
        features = {}
        
        metric_name = alert_data.get('metric_name', '')
        tenant_id = alert_data.get('tenant_id', '')
        
        # Filter historical data for this metric and tenant
        metric_history = historical_data[
            (historical_data['metric_name'] == metric_name) &
            (historical_data['tenant_id'] == tenant_id)
        ]
        
        if len(metric_history) > 0:
            # Recent alert frequency (last 24 hours)
            recent_cutoff = datetime.now() - timedelta(hours=24)
            recent_alerts = metric_history[
                pd.to_datetime(metric_history['timestamp']) >= recent_cutoff
            ]
            features['recent_alert_frequency'] = len(recent_alerts)
            
            # Severity trend analysis
            if 'severity' in metric_history.columns:
                severity_values = {'critical': 3, 'warning': 2, 'info': 1}
                severity_scores = metric_history['severity'].map(severity_values).fillna(1)
                features['historical_severity_trend'] = severity_scores.mean()
            else:
                features['historical_severity_trend'] = 1.0
                
            # Metric volatility
            if 'value' in metric_history.columns:
                values = pd.to_numeric(metric_history['value'], errors='coerce').dropna()
                if len(values) > 1:
                    features['metric_volatility'] = values.std() / max(values.mean(), 0.1)
                else:
                    features['metric_volatility'] = 0
            else:
                features['metric_volatility'] = 0
                
            # Recent escalations
            if 'severity' in metric_history.columns:
                critical_alerts = recent_alerts[recent_alerts['severity'] == 'critical']
                features['recent_escalations'] = len(critical_alerts)
            else:
                features['recent_escalations'] = 0
        else:
            # No historical data available
            features.update({
                'recent_alert_frequency': 0,
                'historical_severity_trend': 1.0,
                'metric_volatility': 0,
                'recent_escalations': 0
            })
            
        return features
        
    def _get_model_predictions(self, features: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Get predictions from all trained models."""
        predictions = {}
        
        # Convert features to array
        feature_array = np.array([list(features.values())]).reshape(1, -1)
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                try:
                    # Scale features if scaler is available
                    if model_name in self.scalers:
                        feature_array_scaled = self.scalers[model_name].transform(feature_array)
                    else:
                        feature_array_scaled = feature_array
                        
                    # Get probability predictions
                    probabilities = model.predict_proba(feature_array_scaled)[0]
                    predicted_class = model.predict(feature_array_scaled)[0]
                    
                    # Map back to severity labels
                    if 'severity' in self.label_encoders:
                        severity_labels = self.label_encoders['severity'].classes_
                        prob_dict = dict(zip(severity_labels, probabilities))
                        predicted_severity = self.label_encoders['severity'].inverse_transform([predicted_class])[0]
                    else:
                        # Default mapping if no encoder trained
                        severity_labels = ['info', 'warning', 'critical']
                        if len(probabilities) == len(severity_labels):
                            prob_dict = dict(zip(severity_labels, probabilities))
                            predicted_severity = severity_labels[np.argmax(probabilities)]
                        else:
                            prob_dict = {'info': 0.33, 'warning': 0.33, 'critical': 0.34}
                            predicted_severity = 'warning'
                    
                    predictions[model_name] = {
                        'predicted_severity': predicted_severity,
                        'probabilities': prob_dict,
                        'confidence': max(probabilities)
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error getting prediction from {model_name}: {e}")
                    # Fallback prediction
                    predictions[model_name] = {
                        'predicted_severity': 'warning',
                        'probabilities': {'info': 0.3, 'warning': 0.4, 'critical': 0.3},
                        'confidence': 0.4
                    }
            else:
                # Model not trained or available
                predictions[model_name] = {
                    'predicted_severity': 'warning',
                    'probabilities': {'info': 0.3, 'warning': 0.4, 'critical': 0.3},
                    'confidence': 0.3
                }
                
        return predictions
        
    def _calculate_business_impact(self, 
                                 alert_data: Dict[str, Any],
                                 business_context: BusinessImpactContext) -> float:
        """Calculate business impact score."""
        impact_score = 0.0
        
        # Tenant tier impact
        tier_impact = {'premium': 0.4, 'standard': 0.25, 'basic': 0.1}
        impact_score += tier_impact.get(business_context.tenant_tier.lower(), 0.2)
        
        # Service criticality impact
        criticality_impact = {'critical': 0.3, 'important': 0.2, 'standard': 0.1}
        impact_score += criticality_impact.get(business_context.service_criticality.lower(), 0.15)
        
        # User impact (normalize to 0-0.2 range)
        max_users = self.config.get('max_expected_users', 10000)
        user_impact = min(business_context.affected_users / max_users, 1.0) * 0.2
        impact_score += user_impact
        
        # Revenue impact (normalize to 0-0.1 range)
        max_revenue = self.config.get('max_revenue_impact', 100000)
        revenue_impact = min(business_context.revenue_impact / max_revenue, 1.0) * 0.1
        impact_score += revenue_impact
        
        # Time-based impact
        if business_context.time_of_day in ['business_hours']:
            impact_score += 0.05
        if business_context.day_of_week in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']:
            impact_score += 0.05
            
        return min(impact_score, 1.0)  # Cap at 1.0
        
    def _ensemble_prediction(self, 
                           predictions: Dict[str, Dict[str, Any]],
                           business_impact: float) -> Tuple[str, float, Dict[str, float]]:
        """Combine predictions from multiple models with business impact."""
        if not predictions:
            return 'warning', 0.5, {'info': 0.3, 'warning': 0.4, 'critical': 0.3}
            
        # Aggregate probabilities from all models
        all_probabilities = {}
        all_confidences = []
        
        for model_name, pred_data in predictions.items():
            model_weight = self.config.get('model_weights', {}).get(model_name, 1.0)
            
            for severity, prob in pred_data['probabilities'].items():
                if severity not in all_probabilities:
                    all_probabilities[severity] = []
                all_probabilities[severity].append(prob * model_weight)
                
            all_confidences.append(pred_data['confidence'])
            
        # Calculate average probabilities
        avg_probabilities = {}
        for severity, probs in all_probabilities.items():
            avg_probabilities[severity] = np.mean(probs)
            
        # Adjust probabilities based on business impact
        if business_impact > 0.7:
            # High business impact - increase critical probability
            avg_probabilities['critical'] = min(1.0, avg_probabilities.get('critical', 0) * 1.5)
            avg_probabilities['warning'] = avg_probabilities.get('warning', 0) * 0.8
            avg_probabilities['info'] = avg_probabilities.get('info', 0) * 0.5
        elif business_impact < 0.3:
            # Low business impact - decrease critical probability
            avg_probabilities['critical'] = avg_probabilities.get('critical', 0) * 0.5
            avg_probabilities['warning'] = min(1.0, avg_probabilities.get('warning', 0) * 1.2)
            avg_probabilities['info'] = min(1.0, avg_probabilities.get('info', 0) * 1.3)
            
        # Normalize probabilities
        total_prob = sum(avg_probabilities.values())
        if total_prob > 0:
            normalized_probabilities = {
                severity: prob / total_prob 
                for severity, prob in avg_probabilities.items()
            }
        else:
            normalized_probabilities = {'info': 0.3, 'warning': 0.4, 'critical': 0.3}
            
        # Get final prediction
        final_severity = max(normalized_probabilities, key=normalized_probabilities.get)
        confidence = np.mean(all_confidences) if all_confidences else 0.5
        
        # Adjust confidence based on business impact
        if business_impact > 0.8:
            confidence = min(1.0, confidence * 1.2)
        elif business_impact < 0.2:
            confidence = confidence * 0.8
            
        return final_severity, confidence, normalized_probabilities
        
    def _identify_contributing_factors(self, 
                                     features: Dict[str, Any],
                                     predicted_severity: str) -> List[str]:
        """Identify factors that contributed to the severity classification."""
        factors = []
        
        # High threshold ratio
        if features.get('threshold_ratio', 0) > 2.0:
            factors.append(f"High threshold breach ratio: {features['threshold_ratio']:.2f}")
            
        # Business impact factors
        if features.get('tenant_tier_score', 0) > 0.8:
            factors.append("Premium tenant affected")
            
        if features.get('service_criticality_score', 0) > 0.8:
            factors.append("Critical service affected")
            
        if features.get('affected_users', 0) > 1000:
            factors.append(f"High user impact: {features['affected_users']} users")
            
        # Temporal factors
        if features.get('is_business_hours', False):
            factors.append("Occurred during business hours")
            
        if not features.get('is_weekend', True):
            factors.append("Occurred on weekday")
            
        # Historical factors
        if features.get('recent_alert_frequency', 0) > 5:
            factors.append("High recent alert frequency")
            
        if features.get('recent_escalations', 0) > 0:
            factors.append("Recent critical alerts for this metric")
            
        # Metric type factors
        critical_metrics = ['is_cpu_metric', 'is_memory_metric', 'is_response_time_metric']
        for metric_type in critical_metrics:
            if features.get(metric_type, False):
                metric_name = metric_type.replace('is_', '').replace('_metric', '')
                factors.append(f"Critical {metric_name} metric affected")
                
        return factors[:5]  # Return top 5 factors
        
    def _generate_severity_recommendation(self, 
                                        predicted_severity: str,
                                        confidence: float,
                                        business_impact: float,
                                        alert_data: Dict[str, Any]) -> str:
        """Generate recommendation based on severity classification."""
        if confidence < 0.6:
            return "Low confidence prediction - manual review recommended"
            
        if predicted_severity == 'critical':
            if business_impact > 0.7:
                return "Immediate escalation required - high business impact"
            else:
                return "Escalate to on-call engineer"
        elif predicted_severity == 'warning':
            if business_impact > 0.5:
                return "Monitor closely - potential for escalation"
            else:
                return "Standard monitoring and investigation"
        else:  # info
            return "Log for tracking - no immediate action required"
            
    def train_models(self, training_data: pd.DataFrame):
        """Train severity classification models."""
        if len(training_data) < 50:
            self.logger.warning("Insufficient training data for model training")
            return
            
        # Prepare features and targets
        X = self._prepare_training_features(training_data)
        y = training_data['severity']
        
        # Encode labels
        y_encoded = self.label_encoders['severity'].fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        # Train models
        for model_name, model in self.models.items():
            self.logger.info(f"Training {model_name} model")
            
            if model_name in ['random_forest', 'gradient_boosting']:
                model.fit(X_train_scaled, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test_scaled)
                accuracy = (y_pred == y_test).mean()
                
                self.logger.info(f"{model_name} accuracy: {accuracy:.3f}")
                
                # Store feature importance
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[model_name] = model.feature_importances_
                    
        self.logger.info("Model training completed")
        
    def _prepare_training_features(self, training_data: pd.DataFrame) -> np.ndarray:
        """Prepare features for model training."""
        # This is a simplified version - in practice, you'd extract
        # the same features as in _extract_classification_features
        feature_columns = [
            'threshold_ratio', 'hour', 'day_of_week', 'affected_users',
            'is_business_hours', 'recent_alert_frequency'
        ]
        
        # Fill missing features with defaults
        for col in feature_columns:
            if col not in training_data.columns:
                training_data[col] = 0
                
        return training_data[feature_columns].fillna(0).values
        
    def save_models(self, path: str):
        """Save trained models to disk."""
        joblib.dump(self.models, f"{path}/severity_models.pkl")
        joblib.dump(self.scalers, f"{path}/severity_scalers.pkl")
        joblib.dump(self.label_encoders, f"{path}/severity_encoders.pkl")
        joblib.dump(self.feature_importance, f"{path}/feature_importance.pkl")
        
    def load_models(self, path: str):
        """Load trained models from disk."""
        try:
            self.models = joblib.load(f"{path}/severity_models.pkl")
            self.scalers = joblib.load(f"{path}/severity_scalers.pkl")
            self.label_encoders = joblib.load(f"{path}/severity_encoders.pkl")
            self.feature_importance = joblib.load(f"{path}/feature_importance.pkl")
        except FileNotFoundError:
            self.logger.warning("No saved models found, using default configuration")
            
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for trained models."""
        return {
            'models_trained': list(self.models.keys()),
            'feature_importance': self.feature_importance,
            'config': self.config
        }
