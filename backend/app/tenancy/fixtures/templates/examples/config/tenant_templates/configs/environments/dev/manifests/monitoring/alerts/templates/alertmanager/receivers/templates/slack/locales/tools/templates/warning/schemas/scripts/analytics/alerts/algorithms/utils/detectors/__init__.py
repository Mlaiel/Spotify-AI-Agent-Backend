"""
🔍 Enterprise Anomaly Detection Module for Spotify AI Agent

This module provides advanced ML-powered anomaly detection capabilities including
statistical analysis, pattern recognition, and predictive alerting for
large-scale music streaming platform operations.

Features:
- ML-powered anomaly detection algorithms
- Statistical outlier detection
- Pattern recognition and drift detection
- Real-time streaming anomaly detection
- Multi-dimensional anomaly analysis
- Threshold-based detection with adaptive learning
- Time series anomaly detection
- User behavior anomaly detection
- Content performance anomaly detection
- System performance anomaly detection

Author: Fahed Mlaiel (Lead Developer & AI Architect)
Version: 2.0.0 (Enterprise Edition)
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Set
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.stats as stats
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies"""
    STATISTICAL = "statistical"      # Statistical outliers
    PATTERN = "pattern"              # Pattern anomalies
    THRESHOLD = "threshold"          # Threshold violations
    TEMPORAL = "temporal"            # Time-based anomalies
    BEHAVIORAL = "behavioral"        # Behavioral anomalies
    PERFORMANCE = "performance"      # Performance anomalies
    SECURITY = "security"            # Security anomalies


class DetectionMethod(Enum):
    """Detection methods"""
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    STATISTICAL_Z_SCORE = "statistical_z_score"
    STATISTICAL_IQR = "statistical_iqr"
    MAHALANOBIS = "mahalanobis"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    LSTM_AUTOENCODER = "lstm_autoencoder"
    ARIMA_RESIDUALS = "arima_residuals"


class Severity(Enum):
    """Anomaly severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DetectionConfig:
    """Configuration for anomaly detection"""
    method: DetectionMethod = DetectionMethod.ISOLATION_FOREST
    sensitivity: float = 0.1  # Contamination rate (0.0 to 1.0)
    window_size: int = 100    # Data points to consider
    min_samples: int = 50     # Minimum samples required
    threshold_std_dev: float = 2.0  # Standard deviations for threshold
    enable_adaptive_learning: bool = True
    enable_ensemble_detection: bool = True
    confidence_threshold: float = 0.7
    update_frequency_minutes: int = 15


@dataclass
class AnomalyResult:
    """Result of anomaly detection"""
    is_anomaly: bool
    anomaly_score: float
    confidence: float
    severity: Severity
    detected_at: datetime
    method_used: DetectionMethod
    features_analyzed: List[str] = field(default_factory=list)
    explanation: str = ""
    recommendation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataPoint:
    """Data point for analysis"""
    timestamp: datetime
    features: Dict[str, float]
    labels: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAnomalyDetector(ABC):
    """Base anomaly detector"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._metrics = {
            'detections': Counter('anomaly_detections_total', 'Total anomaly detections'),
            'processing_time': Histogram('anomaly_detection_seconds', 'Detection processing time'),
            'accuracy': Gauge('anomaly_detection_accuracy', 'Detection accuracy'),
            'false_positives': Counter('anomaly_false_positives_total', 'False positive detections')
        }
        self._model = None
        self._scaler = StandardScaler()
        self._training_data = []
    
    @abstractmethod
    async def detect(self, data_point: DataPoint) -> AnomalyResult:
        """Detect anomalies in data point"""
        pass
    
    @abstractmethod
    async def train(self, training_data: List[DataPoint]) -> bool:
        """Train the detector with historical data"""
        pass
    
    async def batch_detect(self, data_points: List[DataPoint]) -> List[AnomalyResult]:
        """Detect anomalies in batch"""
        results = []
        for data_point in data_points:
            result = await self.detect(data_point)
            results.append(result)
        return results


class AnomalyDetector(BaseAnomalyDetector):
    """Main anomaly detector with multiple algorithms"""
    
    def __init__(self, config: DetectionConfig):
        super().__init__(config)
        self.detectors = {
            DetectionMethod.ISOLATION_FOREST: IsolationForestDetector(config),
            DetectionMethod.ONE_CLASS_SVM: OneClassSVMDetector(config),
            DetectionMethod.STATISTICAL_Z_SCORE: StatisticalDetector(config),
            DetectionMethod.STATISTICAL_IQR: IQRDetector(config),
            DetectionMethod.MAHALANOBIS: MahalanobisDetector(config)
        }
        self._ensemble_weights = {}
        self._performance_history = {}
        self._initialize_ensemble()
    
    def _initialize_ensemble(self):
        """Initialize ensemble weights"""
        num_detectors = len(self.detectors)
        initial_weight = 1.0 / num_detectors
        
        for method in self.detectors:
            self._ensemble_weights[method] = initial_weight
            self._performance_history[method] = []
    
    async def detect(self, data_point: DataPoint) -> AnomalyResult:
        """Detect anomalies using ensemble of methods"""
        start_time = datetime.now()
        
        try:
            if self.config.enable_ensemble_detection:
                result = await self._ensemble_detect(data_point)
            else:
                detector = self.detectors[self.config.method]
                result = await detector.detect(data_point)
            
            # Update metrics
            self._metrics['detections'].inc()
            processing_time = (datetime.now() - start_time).total_seconds()
            self._metrics['processing_time'].observe(processing_time)
            
            # Add explanation and recommendation
            result.explanation = await self._generate_explanation(result, data_point)
            result.recommendation = await self._generate_recommendation(result, data_point)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Anomaly detection error: {e}")
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                confidence=0.0,
                severity=Severity.LOW,
                detected_at=datetime.now(),
                method_used=self.config.method,
                explanation=f"Detection error: {str(e)}"
            )
    
    async def _ensemble_detect(self, data_point: DataPoint) -> AnomalyResult:
        """Ensemble detection using multiple methods"""
        detection_results = []
        
        # Run all detectors
        for method, detector in self.detectors.items():
            try:
                result = await detector.detect(data_point)
                result.method_used = method
                detection_results.append(result)
            except Exception as e:
                self.logger.warning(f"Detector {method} failed: {e}")
        
        if not detection_results:
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                confidence=0.0,
                severity=Severity.LOW,
                detected_at=datetime.now(),
                method_used=DetectionMethod.ISOLATION_FOREST,
                explanation="All detectors failed"
            )
        
        # Combine results using weighted voting
        weighted_score = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0
        anomaly_votes = 0
        
        for result in detection_results:
            weight = self._ensemble_weights.get(result.method_used, 1.0)
            weighted_score += result.anomaly_score * weight
            weighted_confidence += result.confidence * weight
            total_weight += weight
            
            if result.is_anomaly:
                anomaly_votes += weight
        
        # Normalize scores
        final_score = weighted_score / total_weight if total_weight > 0 else 0.0
        final_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.0
        
        # Determine if anomaly based on weighted votes
        is_anomaly = (anomaly_votes / total_weight) > 0.5 if total_weight > 0 else False
        
        # Determine severity
        severity = Severity.LOW
        if final_score > 0.9:
            severity = Severity.CRITICAL
        elif final_score > 0.7:
            severity = Severity.HIGH
        elif final_score > 0.5:
            severity = Severity.MEDIUM
        
        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=final_score,
            confidence=final_confidence,
            severity=severity,
            detected_at=datetime.now(),
            method_used=DetectionMethod.ISOLATION_FOREST,  # Ensemble method
            features_analyzed=list(data_point.features.keys()),
            metadata={
                'ensemble_results': [
                    {
                        'method': r.method_used.value,
                        'score': r.anomaly_score,
                        'is_anomaly': r.is_anomaly
                    } for r in detection_results
                ],
                'weights_used': self._ensemble_weights
            }
        )
    
    async def train(self, training_data: List[DataPoint]) -> bool:
        """Train all detectors"""
        if len(training_data) < self.config.min_samples:
            self.logger.warning(f"Insufficient training data: {len(training_data)} < {self.config.min_samples}")
            return False
        
        success_count = 0
        
        for method, detector in self.detectors.items():
            try:
                success = await detector.train(training_data)
                if success:
                    success_count += 1
                    self.logger.info(f"Successfully trained {method}")
                else:
                    self.logger.warning(f"Failed to train {method}")
            except Exception as e:
                self.logger.error(f"Training error for {method}: {e}")
        
        # Update ensemble weights based on training success
        if success_count > 0:
            await self._update_ensemble_weights(training_data)
        
        return success_count > 0
    
    async def _update_ensemble_weights(self, validation_data: List[DataPoint]):
        """Update ensemble weights based on performance"""
        if not self.config.enable_adaptive_learning:
            return
        
        # Evaluate each detector on validation data
        for method, detector in self.detectors.items():
            try:
                # Calculate performance metrics
                accuracy = await self._calculate_detector_accuracy(detector, validation_data)
                
                # Update performance history
                self._performance_history[method].append(accuracy)
                
                # Keep only recent performance data
                if len(self._performance_history[method]) > 10:
                    self._performance_history[method] = self._performance_history[method][-10:]
                
                # Update weight based on recent performance
                recent_performance = np.mean(self._performance_history[method])
                self._ensemble_weights[method] = max(0.1, recent_performance)
                
            except Exception as e:
                self.logger.warning(f"Failed to update weight for {method}: {e}")
        
        # Normalize weights
        total_weight = sum(self._ensemble_weights.values())
        if total_weight > 0:
            for method in self._ensemble_weights:
                self._ensemble_weights[method] /= total_weight
    
    async def _calculate_detector_accuracy(self, detector: BaseAnomalyDetector, validation_data: List[DataPoint]) -> float:
        """Calculate detector accuracy on validation data"""
        # Simplified accuracy calculation
        # In production, use proper validation with ground truth labels
        
        correct_predictions = 0
        total_predictions = len(validation_data)
        
        for data_point in validation_data:
            try:
                result = await detector.detect(data_point)
                # Assume non-anomalies are correct (placeholder logic)
                if not result.is_anomaly:
                    correct_predictions += 1
            except Exception:
                pass
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    async def _generate_explanation(self, result: AnomalyResult, data_point: DataPoint) -> str:
        """Generate explanation for anomaly result"""
        if not result.is_anomaly:
            return "No anomaly detected - data point falls within normal patterns"
        
        explanations = []
        
        # Analyze which features contributed to anomaly
        feature_values = data_point.features
        
        # Check for obvious outliers
        for feature, value in feature_values.items():
            if abs(value) > 3:  # More than 3 standard deviations (assuming normalized)
                explanations.append(f"Feature '{feature}' has extreme value: {value:.2f}")
        
        # Add severity-based explanation
        if result.severity == Severity.CRITICAL:
            explanations.append("Critical anomaly detected - immediate attention required")
        elif result.severity == Severity.HIGH:
            explanations.append("High-priority anomaly requiring investigation")
        elif result.severity == Severity.MEDIUM:
            explanations.append("Moderate anomaly detected - monitor closely")
        else:
            explanations.append("Low-priority anomaly detected")
        
        return "; ".join(explanations) if explanations else "Anomaly detected by ML algorithms"
    
    async def _generate_recommendation(self, result: AnomalyResult, data_point: DataPoint) -> str:
        """Generate recommendation for anomaly result"""
        if not result.is_anomaly:
            return "Continue normal monitoring"
        
        recommendations = []
        
        # Severity-based recommendations
        if result.severity == Severity.CRITICAL:
            recommendations.extend([
                "Immediate investigation required",
                "Consider triggering incident response",
                "Alert on-call team"
            ])
        elif result.severity == Severity.HIGH:
            recommendations.extend([
                "Investigate within 1 hour",
                "Check related systems",
                "Monitor for escalation"
            ])
        elif result.severity == Severity.MEDIUM:
            recommendations.extend([
                "Investigate within 4 hours",
                "Monitor trend development"
            ])
        else:
            recommendations.append("Monitor and log for trend analysis")
        
        # Feature-specific recommendations
        feature_values = data_point.features
        
        if 'cpu_usage' in feature_values and feature_values['cpu_usage'] > 0.9:
            recommendations.append("Check for CPU-intensive processes")
        
        if 'memory_usage' in feature_values and feature_values['memory_usage'] > 0.9:
            recommendations.append("Investigate memory leaks or high memory usage")
        
        if 'error_rate' in feature_values and feature_values['error_rate'] > 0.1:
            recommendations.append("Check application logs for error patterns")
        
        if 'latency' in feature_values and feature_values['latency'] > 1000:
            recommendations.append("Investigate performance bottlenecks")
        
        return "; ".join(recommendations) if recommendations else "General investigation recommended"


class IsolationForestDetector(BaseAnomalyDetector):
    """Isolation Forest anomaly detector"""
    
    async def detect(self, data_point: DataPoint) -> AnomalyResult:
        """Detect anomalies using Isolation Forest"""
        if self._model is None:
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                confidence=0.0,
                severity=Severity.LOW,
                detected_at=datetime.now(),
                method_used=DetectionMethod.ISOLATION_FOREST,
                explanation="Model not trained"
            )
        
        try:
            # Prepare features
            features = np.array(list(data_point.features.values())).reshape(1, -1)
            features_scaled = self._scaler.transform(features)
            
            # Predict anomaly
            prediction = self._model.predict(features_scaled)[0]
            anomaly_score = abs(self._model.decision_function(features_scaled)[0])
            
            # Normalize anomaly score to [0, 1]
            normalized_score = min(max((anomaly_score + 0.5) / 1.0, 0.0), 1.0)
            
            is_anomaly = prediction == -1
            confidence = normalized_score if is_anomaly else 1.0 - normalized_score
            
            # Determine severity
            severity = Severity.LOW
            if normalized_score > 0.8:
                severity = Severity.CRITICAL
            elif normalized_score > 0.6:
                severity = Severity.HIGH
            elif normalized_score > 0.4:
                severity = Severity.MEDIUM
            
            return AnomalyResult(
                is_anomaly=is_anomaly,
                anomaly_score=normalized_score,
                confidence=confidence,
                severity=severity,
                detected_at=datetime.now(),
                method_used=DetectionMethod.ISOLATION_FOREST,
                features_analyzed=list(data_point.features.keys()),
                metadata={'raw_score': anomaly_score}
            )
            
        except Exception as e:
            self.logger.error(f"Isolation Forest detection error: {e}")
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                confidence=0.0,
                severity=Severity.LOW,
                detected_at=datetime.now(),
                method_used=DetectionMethod.ISOLATION_FOREST,
                explanation=f"Detection error: {str(e)}"
            )
    
    async def train(self, training_data: List[DataPoint]) -> bool:
        """Train Isolation Forest model"""
        try:
            # Prepare training data
            features_list = []
            for data_point in training_data:
                features_list.append(list(data_point.features.values()))
            
            features_array = np.array(features_list)
            
            # Fit scaler
            self._scaler.fit(features_array)
            features_scaled = self._scaler.transform(features_array)
            
            # Train Isolation Forest
            self._model = IsolationForest(
                contamination=self.config.sensitivity,
                random_state=42,
                n_estimators=100
            )
            self._model.fit(features_scaled)
            
            self.logger.info(f"Trained Isolation Forest with {len(training_data)} samples")
            return True
            
        except Exception as e:
            self.logger.error(f"Isolation Forest training error: {e}")
            return False


class OneClassSVMDetector(BaseAnomalyDetector):
    """One-Class SVM anomaly detector"""
    
    async def detect(self, data_point: DataPoint) -> AnomalyResult:
        """Detect anomalies using One-Class SVM"""
        if self._model is None:
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                confidence=0.0,
                severity=Severity.LOW,
                detected_at=datetime.now(),
                method_used=DetectionMethod.ONE_CLASS_SVM,
                explanation="Model not trained"
            )
        
        try:
            # Prepare features
            features = np.array(list(data_point.features.values())).reshape(1, -1)
            features_scaled = self._scaler.transform(features)
            
            # Predict anomaly
            prediction = self._model.predict(features_scaled)[0]
            decision_score = self._model.decision_function(features_scaled)[0]
            
            # Normalize score
            normalized_score = max(0.0, -decision_score / 2.0)  # SVM scores are typically negative for outliers
            
            is_anomaly = prediction == -1
            confidence = normalized_score if is_anomaly else 1.0 - normalized_score
            
            # Determine severity
            severity = Severity.LOW
            if normalized_score > 0.8:
                severity = Severity.CRITICAL
            elif normalized_score > 0.6:
                severity = Severity.HIGH
            elif normalized_score > 0.4:
                severity = Severity.MEDIUM
            
            return AnomalyResult(
                is_anomaly=is_anomaly,
                anomaly_score=normalized_score,
                confidence=confidence,
                severity=severity,
                detected_at=datetime.now(),
                method_used=DetectionMethod.ONE_CLASS_SVM,
                features_analyzed=list(data_point.features.keys()),
                metadata={'decision_score': decision_score}
            )
            
        except Exception as e:
            self.logger.error(f"One-Class SVM detection error: {e}")
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                confidence=0.0,
                severity=Severity.LOW,
                detected_at=datetime.now(),
                method_used=DetectionMethod.ONE_CLASS_SVM,
                explanation=f"Detection error: {str(e)}"
            )
    
    async def train(self, training_data: List[DataPoint]) -> bool:
        """Train One-Class SVM model"""
        try:
            # Prepare training data
            features_list = []
            for data_point in training_data:
                features_list.append(list(data_point.features.values()))
            
            features_array = np.array(features_list)
            
            # Fit scaler
            self._scaler.fit(features_array)
            features_scaled = self._scaler.transform(features_array)
            
            # Train One-Class SVM
            self._model = OneClassSVM(
                nu=self.config.sensitivity,
                kernel='rbf',
                gamma='scale'
            )
            self._model.fit(features_scaled)
            
            self.logger.info(f"Trained One-Class SVM with {len(training_data)} samples")
            return True
            
        except Exception as e:
            self.logger.error(f"One-Class SVM training error: {e}")
            return False


class StatisticalDetector(BaseAnomalyDetector):
    """Statistical Z-score based anomaly detector"""
    
    def __init__(self, config: DetectionConfig):
        super().__init__(config)
        self._feature_stats = {}
    
    async def detect(self, data_point: DataPoint) -> AnomalyResult:
        """Detect anomalies using statistical Z-score"""
        if not self._feature_stats:
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                confidence=0.0,
                severity=Severity.LOW,
                detected_at=datetime.now(),
                method_used=DetectionMethod.STATISTICAL_Z_SCORE,
                explanation="Model not trained"
            )
        
        try:
            anomalous_features = []
            max_z_score = 0.0
            
            # Check each feature
            for feature, value in data_point.features.items():
                if feature in self._feature_stats:
                    mean = self._feature_stats[feature]['mean']
                    std = self._feature_stats[feature]['std']
                    
                    if std > 0:
                        z_score = abs((value - mean) / std)
                        
                        if z_score > self.config.threshold_std_dev:
                            anomalous_features.append({
                                'feature': feature,
                                'value': value,
                                'z_score': z_score,
                                'threshold': self.config.threshold_std_dev
                            })
                        
                        max_z_score = max(max_z_score, z_score)
            
            # Normalize score
            normalized_score = min(max_z_score / 5.0, 1.0)  # Normalize to [0, 1]
            
            is_anomaly = len(anomalous_features) > 0
            confidence = normalized_score if is_anomaly else 1.0 - normalized_score
            
            # Determine severity
            severity = Severity.LOW
            if max_z_score > 5:
                severity = Severity.CRITICAL
            elif max_z_score > 4:
                severity = Severity.HIGH
            elif max_z_score > 3:
                severity = Severity.MEDIUM
            
            return AnomalyResult(
                is_anomaly=is_anomaly,
                anomaly_score=normalized_score,
                confidence=confidence,
                severity=severity,
                detected_at=datetime.now(),
                method_used=DetectionMethod.STATISTICAL_Z_SCORE,
                features_analyzed=list(data_point.features.keys()),
                metadata={
                    'anomalous_features': anomalous_features,
                    'max_z_score': max_z_score
                }
            )
            
        except Exception as e:
            self.logger.error(f"Statistical detection error: {e}")
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                confidence=0.0,
                severity=Severity.LOW,
                detected_at=datetime.now(),
                method_used=DetectionMethod.STATISTICAL_Z_SCORE,
                explanation=f"Detection error: {str(e)}"
            )
    
    async def train(self, training_data: List[DataPoint]) -> bool:
        """Train statistical model by calculating feature statistics"""
        try:
            # Collect all feature values
            feature_values = {}
            
            for data_point in training_data:
                for feature, value in data_point.features.items():
                    if feature not in feature_values:
                        feature_values[feature] = []
                    feature_values[feature].append(value)
            
            # Calculate statistics for each feature
            self._feature_stats = {}
            for feature, values in feature_values.items():
                if len(values) > 1:
                    self._feature_stats[feature] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }
            
            self.logger.info(f"Trained statistical detector with {len(training_data)} samples")
            return True
            
        except Exception as e:
            self.logger.error(f"Statistical training error: {e}")
            return False


class IQRDetector(BaseAnomalyDetector):
    """Interquartile Range (IQR) based anomaly detector"""
    
    def __init__(self, config: DetectionConfig):
        super().__init__(config)
        self._feature_stats = {}
    
    async def detect(self, data_point: DataPoint) -> AnomalyResult:
        """Detect anomalies using IQR method"""
        if not self._feature_stats:
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                confidence=0.0,
                severity=Severity.LOW,
                detected_at=datetime.now(),
                method_used=DetectionMethod.STATISTICAL_IQR,
                explanation="Model not trained"
            )
        
        try:
            anomalous_features = []
            max_outlier_score = 0.0
            
            # Check each feature
            for feature, value in data_point.features.items():
                if feature in self._feature_stats:
                    q1 = self._feature_stats[feature]['q1']
                    q3 = self._feature_stats[feature]['q3']
                    iqr = q3 - q1
                    
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    if value < lower_bound or value > upper_bound:
                        # Calculate outlier score
                        if value < lower_bound:
                            outlier_score = (lower_bound - value) / iqr if iqr > 0 else 0
                        else:
                            outlier_score = (value - upper_bound) / iqr if iqr > 0 else 0
                        
                        anomalous_features.append({
                            'feature': feature,
                            'value': value,
                            'outlier_score': outlier_score,
                            'bounds': [lower_bound, upper_bound]
                        })
                        
                        max_outlier_score = max(max_outlier_score, outlier_score)
            
            # Normalize score
            normalized_score = min(max_outlier_score / 3.0, 1.0)  # Normalize to [0, 1]
            
            is_anomaly = len(anomalous_features) > 0
            confidence = normalized_score if is_anomaly else 1.0 - normalized_score
            
            # Determine severity
            severity = Severity.LOW
            if max_outlier_score > 3:
                severity = Severity.CRITICAL
            elif max_outlier_score > 2:
                severity = Severity.HIGH
            elif max_outlier_score > 1:
                severity = Severity.MEDIUM
            
            return AnomalyResult(
                is_anomaly=is_anomaly,
                anomaly_score=normalized_score,
                confidence=confidence,
                severity=severity,
                detected_at=datetime.now(),
                method_used=DetectionMethod.STATISTICAL_IQR,
                features_analyzed=list(data_point.features.keys()),
                metadata={
                    'anomalous_features': anomalous_features,
                    'max_outlier_score': max_outlier_score
                }
            )
            
        except Exception as e:
            self.logger.error(f"IQR detection error: {e}")
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                confidence=0.0,
                severity=Severity.LOW,
                detected_at=datetime.now(),
                method_used=DetectionMethod.STATISTICAL_IQR,
                explanation=f"Detection error: {str(e)}"
            )
    
    async def train(self, training_data: List[DataPoint]) -> bool:
        """Train IQR model by calculating quartiles"""
        try:
            # Collect all feature values
            feature_values = {}
            
            for data_point in training_data:
                for feature, value in data_point.features.items():
                    if feature not in feature_values:
                        feature_values[feature] = []
                    feature_values[feature].append(value)
            
            # Calculate quartiles for each feature
            self._feature_stats = {}
            for feature, values in feature_values.items():
                if len(values) > 4:  # Need at least 4 values for quartiles
                    self._feature_stats[feature] = {
                        'q1': np.percentile(values, 25),
                        'q3': np.percentile(values, 75),
                        'median': np.median(values),
                        'count': len(values)
                    }
            
            self.logger.info(f"Trained IQR detector with {len(training_data)} samples")
            return True
            
        except Exception as e:
            self.logger.error(f"IQR training error: {e}")
            return False


class MahalanobisDetector(BaseAnomalyDetector):
    """Mahalanobis distance based anomaly detector"""
    
    def __init__(self, config: DetectionConfig):
        super().__init__(config)
        self._mean_vector = None
        self._inv_covariance = None
    
    async def detect(self, data_point: DataPoint) -> AnomalyResult:
        """Detect anomalies using Mahalanobis distance"""
        if self._mean_vector is None or self._inv_covariance is None:
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                confidence=0.0,
                severity=Severity.LOW,
                detected_at=datetime.now(),
                method_used=DetectionMethod.MAHALANOBIS,
                explanation="Model not trained"
            )
        
        try:
            # Prepare features
            features = np.array(list(data_point.features.values()))
            
            # Calculate Mahalanobis distance
            diff = features - self._mean_vector
            mahalanobis_dist = np.sqrt(diff.dot(self._inv_covariance).dot(diff))
            
            # Normalize score using chi-square distribution
            # For n dimensions, threshold is approximately sqrt(chi2.ppf(0.95, n))
            n_features = len(features)
            threshold = np.sqrt(stats.chi2.ppf(0.95, n_features))
            
            normalized_score = min(mahalanobis_dist / (threshold * 2), 1.0)
            
            is_anomaly = mahalanobis_dist > threshold
            confidence = normalized_score if is_anomaly else 1.0 - normalized_score
            
            # Determine severity
            severity = Severity.LOW
            if mahalanobis_dist > threshold * 2:
                severity = Severity.CRITICAL
            elif mahalanobis_dist > threshold * 1.5:
                severity = Severity.HIGH
            elif mahalanobis_dist > threshold:
                severity = Severity.MEDIUM
            
            return AnomalyResult(
                is_anomaly=is_anomaly,
                anomaly_score=normalized_score,
                confidence=confidence,
                severity=severity,
                detected_at=datetime.now(),
                method_used=DetectionMethod.MAHALANOBIS,
                features_analyzed=list(data_point.features.keys()),
                metadata={
                    'mahalanobis_distance': mahalanobis_dist,
                    'threshold': threshold
                }
            )
            
        except Exception as e:
            self.logger.error(f"Mahalanobis detection error: {e}")
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                confidence=0.0,
                severity=Severity.LOW,
                detected_at=datetime.now(),
                method_used=DetectionMethod.MAHALANOBIS,
                explanation=f"Detection error: {str(e)}"
            )
    
    async def train(self, training_data: List[DataPoint]) -> bool:
        """Train Mahalanobis detector by calculating mean and covariance"""
        try:
            # Prepare training data
            features_list = []
            for data_point in training_data:
                features_list.append(list(data_point.features.values()))
            
            features_array = np.array(features_list)
            
            # Calculate mean vector
            self._mean_vector = np.mean(features_array, axis=0)
            
            # Calculate covariance matrix
            covariance_matrix = np.cov(features_array.T)
            
            # Add regularization to avoid singular matrix
            reg_param = 1e-6
            covariance_matrix += reg_param * np.eye(covariance_matrix.shape[0])
            
            # Calculate inverse covariance matrix
            self._inv_covariance = np.linalg.inv(covariance_matrix)
            
            self.logger.info(f"Trained Mahalanobis detector with {len(training_data)} samples")
            return True
            
        except Exception as e:
            self.logger.error(f"Mahalanobis training error: {e}")
            return False


class ThresholdDetector(BaseAnomalyDetector):
    """Simple threshold-based anomaly detector"""
    
    def __init__(self, config: DetectionConfig, thresholds: Dict[str, Dict[str, float]] = None):
        super().__init__(config)
        self._thresholds = thresholds or {}
    
    def set_threshold(self, feature: str, min_value: float = None, max_value: float = None):
        """Set threshold for a feature"""
        if feature not in self._thresholds:
            self._thresholds[feature] = {}
        
        if min_value is not None:
            self._thresholds[feature]['min'] = min_value
        if max_value is not None:
            self._thresholds[feature]['max'] = max_value
    
    async def detect(self, data_point: DataPoint) -> AnomalyResult:
        """Detect anomalies using threshold violations"""
        violations = []
        max_violation_score = 0.0
        
        try:
            # Check each feature against thresholds
            for feature, value in data_point.features.items():
                if feature in self._thresholds:
                    thresholds = self._thresholds[feature]
                    
                    # Check minimum threshold
                    if 'min' in thresholds and value < thresholds['min']:
                        violation_score = (thresholds['min'] - value) / abs(thresholds['min']) if thresholds['min'] != 0 else 1.0
                        violations.append({
                            'feature': feature,
                            'value': value,
                            'threshold': thresholds['min'],
                            'type': 'below_minimum',
                            'violation_score': violation_score
                        })
                        max_violation_score = max(max_violation_score, violation_score)
                    
                    # Check maximum threshold
                    if 'max' in thresholds and value > thresholds['max']:
                        violation_score = (value - thresholds['max']) / abs(thresholds['max']) if thresholds['max'] != 0 else 1.0
                        violations.append({
                            'feature': feature,
                            'value': value,
                            'threshold': thresholds['max'],
                            'type': 'above_maximum',
                            'violation_score': violation_score
                        })
                        max_violation_score = max(max_violation_score, violation_score)
            
            # Normalize score
            normalized_score = min(max_violation_score, 1.0)
            
            is_anomaly = len(violations) > 0
            confidence = 1.0 if is_anomaly else 0.9  # High confidence for threshold-based detection
            
            # Determine severity
            severity = Severity.LOW
            if max_violation_score > 2.0:
                severity = Severity.CRITICAL
            elif max_violation_score > 1.5:
                severity = Severity.HIGH
            elif max_violation_score > 1.0:
                severity = Severity.MEDIUM
            
            return AnomalyResult(
                is_anomaly=is_anomaly,
                anomaly_score=normalized_score,
                confidence=confidence,
                severity=severity,
                detected_at=datetime.now(),
                method_used=DetectionMethod.STATISTICAL_Z_SCORE,  # Use as proxy
                features_analyzed=list(data_point.features.keys()),
                metadata={
                    'threshold_violations': violations,
                    'max_violation_score': max_violation_score
                }
            )
            
        except Exception as e:
            self.logger.error(f"Threshold detection error: {e}")
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                confidence=0.0,
                severity=Severity.LOW,
                detected_at=datetime.now(),
                method_used=DetectionMethod.STATISTICAL_Z_SCORE,
                explanation=f"Detection error: {str(e)}"
            )
    
    async def train(self, training_data: List[DataPoint]) -> bool:
        """Training not required for threshold detector"""
        self.logger.info("Threshold detector does not require training")
        return True


class PatternDetector(BaseAnomalyDetector):
    """Pattern-based anomaly detector for sequential data"""
    
    def __init__(self, config: DetectionConfig):
        super().__init__(config)
        self._pattern_history = []
        self._normal_patterns = set()
        self._pattern_frequencies = {}
    
    async def detect(self, data_point: DataPoint) -> AnomalyResult:
        """Detect pattern anomalies"""
        try:
            # Convert data point to pattern
            pattern = self._extract_pattern(data_point)
            
            # Add to history
            self._pattern_history.append(pattern)
            
            # Keep only recent patterns
            if len(self._pattern_history) > self.config.window_size:
                self._pattern_history = self._pattern_history[-self.config.window_size:]
            
            # Check if pattern is normal
            is_normal_pattern = pattern in self._normal_patterns
            
            # Calculate pattern frequency score
            frequency_score = self._pattern_frequencies.get(pattern, 0) / len(self._pattern_history)
            
            # Detect sequence anomalies
            sequence_anomaly_score = await self._detect_sequence_anomaly()
            
            # Combine scores
            pattern_score = 0.0 if is_normal_pattern else 1.0
            combined_score = (pattern_score * 0.5 + 
                            (1.0 - frequency_score) * 0.3 + 
                            sequence_anomaly_score * 0.2)
            
            is_anomaly = combined_score > 0.6
            confidence = combined_score
            
            # Determine severity
            severity = Severity.LOW
            if combined_score > 0.9:
                severity = Severity.CRITICAL
            elif combined_score > 0.8:
                severity = Severity.HIGH
            elif combined_score > 0.7:
                severity = Severity.MEDIUM
            
            return AnomalyResult(
                is_anomaly=is_anomaly,
                anomaly_score=combined_score,
                confidence=confidence,
                severity=severity,
                detected_at=datetime.now(),
                method_used=DetectionMethod.STATISTICAL_Z_SCORE,  # Use as proxy
                features_analyzed=list(data_point.features.keys()),
                metadata={
                    'pattern': pattern,
                    'is_normal_pattern': is_normal_pattern,
                    'frequency_score': frequency_score,
                    'sequence_anomaly_score': sequence_anomaly_score
                }
            )
            
        except Exception as e:
            self.logger.error(f"Pattern detection error: {e}")
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                confidence=0.0,
                severity=Severity.LOW,
                detected_at=datetime.now(),
                method_used=DetectionMethod.STATISTICAL_Z_SCORE,
                explanation=f"Detection error: {str(e)}"
            )
    
    def _extract_pattern(self, data_point: DataPoint) -> str:
        """Extract pattern from data point"""
        # Simple pattern extraction based on feature ranges
        pattern_parts = []
        
        for feature, value in data_point.features.items():
            if value < -1:
                pattern_parts.append(f"{feature}:low")
            elif value > 1:
                pattern_parts.append(f"{feature}:high")
            else:
                pattern_parts.append(f"{feature}:normal")
        
        return "|".join(sorted(pattern_parts))
    
    async def _detect_sequence_anomaly(self) -> float:
        """Detect anomalies in pattern sequence"""
        if len(self._pattern_history) < 3:
            return 0.0
        
        # Look for unusual pattern transitions
        recent_patterns = self._pattern_history[-3:]
        
        # Check for rapid pattern changes
        unique_patterns = len(set(recent_patterns))
        if unique_patterns == len(recent_patterns):  # All different patterns
            return 0.8
        
        # Check for pattern repetition anomalies
        if len(set(recent_patterns)) == 1:  # All same patterns
            pattern = recent_patterns[0]
            if self._pattern_frequencies.get(pattern, 0) < 2:  # Rare pattern repeating
                return 0.6
        
        return 0.0
    
    async def train(self, training_data: List[DataPoint]) -> bool:
        """Train pattern detector with normal patterns"""
        try:
            # Extract all patterns from training data
            patterns = []
            for data_point in training_data:
                pattern = self._extract_pattern(data_point)
                patterns.append(pattern)
            
            # Identify normal patterns (frequent patterns)
            pattern_counts = {}
            for pattern in patterns:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            # Consider patterns that appear more than 5% of the time as normal
            min_frequency = len(patterns) * 0.05
            self._normal_patterns = set(
                pattern for pattern, count in pattern_counts.items()
                if count >= min_frequency
            )
            
            # Store pattern frequencies
            self._pattern_frequencies = pattern_counts
            
            self.logger.info(f"Trained pattern detector with {len(self._normal_patterns)} normal patterns")
            return True
            
        except Exception as e:
            self.logger.error(f"Pattern training error: {e}")
            return False


class OutlierDetector(BaseAnomalyDetector):
    """Simple outlier detector for quick detection"""
    
    async def detect(self, data_point: DataPoint) -> AnomalyResult:
        """Detect simple outliers"""
        try:
            outliers = []
            max_outlier_score = 0.0
            
            # Simple outlier detection based on extreme values
            for feature, value in data_point.features.items():
                outlier_score = 0.0
                
                # Check for extreme values
                if abs(value) > 5:  # More than 5 standard deviations (assuming normalized)
                    outlier_score = min(abs(value) / 10, 1.0)
                    outliers.append({
                        'feature': feature,
                        'value': value,
                        'outlier_score': outlier_score
                    })
                
                max_outlier_score = max(max_outlier_score, outlier_score)
            
            is_anomaly = len(outliers) > 0
            confidence = max_outlier_score if is_anomaly else 0.9
            
            # Determine severity
            severity = Severity.LOW
            if max_outlier_score > 0.8:
                severity = Severity.CRITICAL
            elif max_outlier_score > 0.6:
                severity = Severity.HIGH
            elif max_outlier_score > 0.4:
                severity = Severity.MEDIUM
            
            return AnomalyResult(
                is_anomaly=is_anomaly,
                anomaly_score=max_outlier_score,
                confidence=confidence,
                severity=severity,
                detected_at=datetime.now(),
                method_used=DetectionMethod.STATISTICAL_Z_SCORE,  # Use as proxy
                features_analyzed=list(data_point.features.keys()),
                metadata={'outliers': outliers}
            )
            
        except Exception as e:
            self.logger.error(f"Outlier detection error: {e}")
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                confidence=0.0,
                severity=Severity.LOW,
                detected_at=datetime.now(),
                method_used=DetectionMethod.STATISTICAL_Z_SCORE,
                explanation=f"Detection error: {str(e)}"
            )
    
    async def train(self, training_data: List[DataPoint]) -> bool:
        """Training not required for simple outlier detector"""
        self.logger.info("Simple outlier detector does not require training")
        return True


class DriftDetector:
    """Concept drift detector for model performance monitoring"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._reference_data = []
        self._current_window = []
        self._drift_threshold = 0.05  # 5% significance level
    
    async def detect_drift(self, data_points: List[DataPoint]) -> Dict[str, Any]:
        """Detect concept drift in data distribution"""
        try:
            if not self._reference_data:
                self.logger.warning("No reference data for drift detection")
                return {'drift_detected': False, 'reason': 'No reference data'}
            
            # Update current window
            self._current_window.extend(data_points)
            
            # Keep only recent data in window
            if len(self._current_window) > self.config.window_size:
                self._current_window = self._current_window[-self.config.window_size:]
            
            if len(self._current_window) < self.config.min_samples:
                return {'drift_detected': False, 'reason': 'Insufficient current data'}
            
            # Perform statistical tests for drift detection
            drift_results = {}
            
            # Feature-wise drift detection
            for feature in self._get_common_features():
                drift_score = await self._detect_feature_drift(feature)
                drift_results[feature] = drift_score
            
            # Overall drift assessment
            max_drift_score = max(drift_results.values()) if drift_results else 0.0
            drift_detected = max_drift_score > self._drift_threshold
            
            return {
                'drift_detected': drift_detected,
                'max_drift_score': max_drift_score,
                'feature_drift_scores': drift_results,
                'threshold': self._drift_threshold,
                'window_size': len(self._current_window),
                'reference_size': len(self._reference_data)
            }
            
        except Exception as e:
            self.logger.error(f"Drift detection error: {e}")
            return {'drift_detected': False, 'error': str(e)}
    
    def set_reference_data(self, reference_data: List[DataPoint]):
        """Set reference data for drift comparison"""
        self._reference_data = reference_data[-self.config.window_size:]  # Keep only recent reference data
        self.logger.info(f"Set reference data with {len(self._reference_data)} samples")
    
    def _get_common_features(self) -> List[str]:
        """Get features common to both reference and current data"""
        if not self._reference_data or not self._current_window:
            return []
        
        ref_features = set(self._reference_data[0].features.keys())
        current_features = set(self._current_window[0].features.keys())
        
        return list(ref_features.intersection(current_features))
    
    async def _detect_feature_drift(self, feature: str) -> float:
        """Detect drift for a specific feature using KS test"""
        try:
            # Extract feature values
            ref_values = [dp.features[feature] for dp in self._reference_data if feature in dp.features]
            current_values = [dp.features[feature] for dp in self._current_window if feature in dp.features]
            
            if len(ref_values) < 10 or len(current_values) < 10:
                return 0.0
            
            # Perform Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(ref_values, current_values)
            
            # Return the test statistic (higher values indicate more drift)
            return statistic
            
        except Exception as e:
            self.logger.warning(f"Feature drift detection error for {feature}: {e}")
            return 0.0


# Factory functions
def create_anomaly_detector(config: DetectionConfig = None) -> AnomalyDetector:
    """Create anomaly detector with configuration"""
    if config is None:
        config = DetectionConfig()
    
    return AnomalyDetector(config)


def create_threshold_detector(thresholds: Dict[str, Dict[str, float]] = None, 
                            config: DetectionConfig = None) -> ThresholdDetector:
    """Create threshold detector"""
    if config is None:
        config = DetectionConfig()
    
    return ThresholdDetector(config, thresholds)


def create_pattern_detector(config: DetectionConfig = None) -> PatternDetector:
    """Create pattern detector"""
    if config is None:
        config = DetectionConfig()
    
    return PatternDetector(config)


def create_outlier_detector(config: DetectionConfig = None) -> OutlierDetector:
    """Create outlier detector"""
    if config is None:
        config = DetectionConfig()
    
    return OutlierDetector(config)


def create_drift_detector(config: DetectionConfig = None) -> DriftDetector:
    """Create drift detector"""
    if config is None:
        config = DetectionConfig()
    
    return DriftDetector(config)


# Export all classes and functions
__all__ = [
    'AnomalyType',
    'DetectionMethod',
    'Severity',
    'DetectionConfig',
    'AnomalyResult',
    'DataPoint',
    'AnomalyDetector',
    'IsolationForestDetector',
    'OneClassSVMDetector',
    'StatisticalDetector',
    'IQRDetector',
    'MahalanobisDetector',
    'ThresholdDetector',
    'PatternDetector',
    'OutlierDetector',
    'DriftDetector',
    'create_anomaly_detector',
    'create_threshold_detector',
    'create_pattern_detector',
    'create_outlier_detector',
    'create_drift_detector'
]
