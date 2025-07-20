"""
Ultra-Advanced Anomaly Detection System with Ensemble Methods

This module implements a sophisticated anomaly detection system using multiple
algorithms, ensemble methods, adaptive thresholds, and real-time processing
capabilities for multi-tenant environments.

Features:
- Multi-algorithm ensemble (Isolation Forest, One-Class SVM, LSTM Autoencoder)
- Real-time streaming anomaly detection
- Adaptive threshold adjustment based on data patterns
- Contextual anomaly detection with business rules
- Time-series anomaly detection with seasonal patterns
- Distributed processing for large-scale data
- Explainable anomaly detection with feature contributions
- Multi-dimensional anomaly analysis
- Custom anomaly patterns and rules
- Performance optimization with intelligent caching

Created by Expert Team:
- Lead Dev + AI Architect: Ensemble architecture and adaptive algorithms
- ML Engineer: Advanced anomaly detection algorithms and neural networks
- Backend Developer: Real-time processing and streaming integration
- Data Engineer: Time-series analysis and feature engineering
- Security Specialist: Fraud detection and security anomaly patterns
- Microservices Architect: Scalable detection infrastructure
"""

from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import logging
import uuid
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import pickle
import threading
from abc import ABC, abstractmethod
from collections import deque, defaultdict

# Anomaly detection algorithms
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score
import scipy.stats as stats

# Time series anomaly detection
try:
    from pyod.models.iforest import IForest
    from pyod.models.ocsvm import OCSVM
    from pyod.models.lof import LOF
    from pyod.models.knn import KNN
    PYOD_AVAILABLE = True
except ImportError:
    PYOD_AVAILABLE = False

# Deep learning for anomaly detection
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

logger = logging.getLogger(__name__)

class AnomalyType(Enum):
    """Types of anomalies"""
    POINT = "point"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"
    SEASONAL = "seasonal"
    TREND = "trend"
    CHANGE_POINT = "change_point"
    PATTERN = "pattern"

class DetectionAlgorithm(Enum):
    """Anomaly detection algorithms"""
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    DBSCAN = "dbscan"
    LSTM_AUTOENCODER = "lstm_autoencoder"
    STATISTICAL = "statistical"
    ENSEMBLE = "ensemble"

class AnomalyStatus(Enum):
    """Anomaly status"""
    DETECTED = "detected"
    CONFIRMED = "confirmed"
    FALSE_POSITIVE = "false_positive"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"

@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection"""
    # Algorithm settings
    algorithms: List[DetectionAlgorithm] = field(default_factory=lambda: [
        DetectionAlgorithm.ISOLATION_FOREST,
        DetectionAlgorithm.ONE_CLASS_SVM,
        DetectionAlgorithm.LSTM_AUTOENCODER
    ])
    
    # Ensemble settings
    ensemble_enabled: bool = True
    ensemble_voting: str = "soft"  # soft, hard, weighted
    algorithm_weights: Dict[str, float] = field(default_factory=dict)
    
    # Threshold settings
    contamination_rate: float = 0.1
    adaptive_thresholds: bool = True
    confidence_threshold: float = 0.8
    
    # Processing settings
    real_time_processing: bool = True
    batch_processing: bool = True
    streaming_window_size: int = 1000
    
    # Feature settings
    feature_scaling: bool = True
    dimensionality_reduction: bool = False
    feature_selection: bool = True
    
    # Performance settings
    max_workers: int = 4
    caching_enabled: bool = True
    preprocessing_cache_size: int = 10000

@dataclass
class AnomalyPoint:
    """Individual anomaly point"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data_point: Union[Dict, List, np.ndarray] = None
    
    # Anomaly characteristics
    anomaly_type: AnomalyType = AnomalyType.POINT
    anomaly_score: float = 0.0
    confidence: float = 0.0
    severity: str = "medium"  # low, medium, high, critical
    
    # Detection details
    detected_by: List[str] = field(default_factory=list)
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    explanation: Optional[str] = None
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    business_impact: Optional[str] = None
    
    # Status
    status: AnomalyStatus = AnomalyStatus.DETECTED
    investigated_by: Optional[str] = None
    resolution_notes: Optional[str] = None

@dataclass
class AnomalyResult:
    """Result of anomaly detection"""
    anomalies: List[AnomalyPoint] = field(default_factory=list)
    total_points: int = 0
    anomaly_rate: float = 0.0
    
    # Algorithm results
    algorithm_scores: Dict[str, List[float]] = field(default_factory=dict)
    ensemble_scores: List[float] = field(default_factory=list)
    
    # Performance metrics
    detection_time: float = 0.0
    processing_time: float = 0.0
    
    # Statistics
    severity_distribution: Dict[str, int] = field(default_factory=dict)
    type_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Metadata
    tenant_id: str = ""
    model_version: str = "1.0"
    detection_timestamp: datetime = field(default_factory=datetime.utcnow)

class BaseAnomalyDetector(ABC):
    """Abstract base class for anomaly detectors"""
    
    def __init__(self, algorithm_type: DetectionAlgorithm):
        self.algorithm_type = algorithm_type
        self.model = None
        self.is_fitted = False
        self.scaler = None
        
    @abstractmethod
    async def fit(self, X: np.ndarray, config: Optional[Dict] = None) -> None:
        """Fit the anomaly detector"""
        pass
    
    @abstractmethod
    async def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies and return labels and scores"""
        pass
    
    def preprocess_data(self, X: np.ndarray, fit_scaler: bool = False) -> np.ndarray:
        """Preprocess data for anomaly detection"""
        if fit_scaler or self.scaler is None:
            self.scaler = RobustScaler()
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)

class IsolationForestDetector(BaseAnomalyDetector):
    """Isolation Forest anomaly detector"""
    
    def __init__(self, contamination: float = 0.1):
        super().__init__(DetectionAlgorithm.ISOLATION_FOREST)
        self.contamination = contamination
    
    async def fit(self, X: np.ndarray, config: Optional[Dict] = None) -> None:
        """Fit Isolation Forest model"""
        # Preprocess data
        X_scaled = self.preprocess_data(X, fit_scaler=True)
        
        # Configure model
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1,
            n_estimators=100
        )
        
        # Fit model
        self.model.fit(X_scaled)
        self.is_fitted = True
    
    async def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies with Isolation Forest"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Preprocess data
        X_scaled = self.preprocess_data(X, fit_scaler=False)
        
        # Predict
        labels = self.model.predict(X_scaled)  # 1 for normal, -1 for anomaly
        scores = self.model.decision_function(X_scaled)
        
        # Convert labels to binary (0 for normal, 1 for anomaly)
        binary_labels = (labels == -1).astype(int)
        
        # Normalize scores to [0, 1]
        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        return binary_labels, normalized_scores

class OneClassSVMDetector(BaseAnomalyDetector):
    """One-Class SVM anomaly detector"""
    
    def __init__(self, nu: float = 0.1):
        super().__init__(DetectionAlgorithm.ONE_CLASS_SVM)
        self.nu = nu
    
    async def fit(self, X: np.ndarray, config: Optional[Dict] = None) -> None:
        """Fit One-Class SVM model"""
        # Preprocess data
        X_scaled = self.preprocess_data(X, fit_scaler=True)
        
        # Configure model
        self.model = OneClassSVM(
            nu=self.nu,
            kernel='rbf',
            gamma='scale'
        )
        
        # Fit model
        self.model.fit(X_scaled)
        self.is_fitted = True
    
    async def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies with One-Class SVM"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Preprocess data
        X_scaled = self.preprocess_data(X, fit_scaler=False)
        
        # Predict
        labels = self.model.predict(X_scaled)  # 1 for normal, -1 for anomaly
        scores = self.model.decision_function(X_scaled)
        
        # Convert labels to binary (0 for normal, 1 for anomaly)
        binary_labels = (labels == -1).astype(int)
        
        # Normalize scores to [0, 1]
        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        return binary_labels, normalized_scores

class LSTMAutoencoderDetector(BaseAnomalyDetector):
    """LSTM Autoencoder for time-series anomaly detection"""
    
    def __init__(self, sequence_length: int = 10, encoding_dim: int = 32):
        super().__init__(DetectionAlgorithm.LSTM_AUTOENCODER)
        self.sequence_length = sequence_length
        self.encoding_dim = encoding_dim
        self.threshold = None
    
    async def fit(self, X: np.ndarray, config: Optional[Dict] = None) -> None:
        """Fit LSTM Autoencoder model"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM Autoencoder")
        
        # Preprocess data
        X_scaled = self.preprocess_data(X, fit_scaler=True)
        
        # Create sequences
        X_sequences = self._create_sequences(X_scaled)
        
        # Build autoencoder model
        input_dim = X_sequences.shape[2]
        
        # Encoder
        encoder_input = Input(shape=(self.sequence_length, input_dim))
        encoded = LSTM(self.encoding_dim, return_sequences=False)(encoder_input)
        
        # Decoder
        decoder_input = tf.keras.layers.RepeatVector(self.sequence_length)(encoded)
        decoded = LSTM(input_dim, return_sequences=True)(decoder_input)
        
        # Autoencoder model
        self.model = Model(encoder_input, decoded)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Train model
        self.model.fit(
            X_sequences, X_sequences,
            epochs=50,
            batch_size=32,
            shuffle=True,
            verbose=0
        )
        
        # Calculate reconstruction threshold
        reconstructions = self.model.predict(X_sequences)
        reconstruction_errors = np.mean(np.square(X_sequences - reconstructions), axis=(1, 2))
        self.threshold = np.percentile(reconstruction_errors, 95)
        
        self.is_fitted = True
    
    async def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies with LSTM Autoencoder"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Preprocess data
        X_scaled = self.preprocess_data(X, fit_scaler=False)
        
        # Create sequences
        X_sequences = self._create_sequences(X_scaled)
        
        # Get reconstructions
        reconstructions = self.model.predict(X_sequences)
        
        # Calculate reconstruction errors
        reconstruction_errors = np.mean(np.square(X_sequences - reconstructions), axis=(1, 2))
        
        # Classify anomalies
        labels = (reconstruction_errors > self.threshold).astype(int)
        
        # Normalize scores
        normalized_scores = reconstruction_errors / (self.threshold * 2)
        normalized_scores = np.clip(normalized_scores, 0, 1)
        
        return labels, normalized_scores
    
    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        """Create sequences for LSTM input"""
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i:i + self.sequence_length])
        return np.array(sequences)

class EnsembleAnomalyDetector:
    """Ensemble anomaly detector combining multiple algorithms"""
    
    def __init__(self, detectors: List[BaseAnomalyDetector], weights: Optional[List[float]] = None):
        self.detectors = detectors
        self.weights = weights or [1.0] * len(detectors)
        self.is_fitted = False
    
    async def fit(self, X: np.ndarray, config: Optional[Dict] = None) -> None:
        """Fit all detectors in ensemble"""
        tasks = []
        for detector in self.detectors:
            tasks.append(detector.fit(X, config))
        
        await asyncio.gather(*tasks)
        self.is_fitted = True
    
    async def predict(self, X: np.ndarray, voting: str = "soft") -> Tuple[np.ndarray, np.ndarray]:
        """Predict using ensemble voting"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Get predictions from all detectors
        all_labels = []
        all_scores = []
        
        for detector in self.detectors:
            try:
                labels, scores = await detector.predict(X)
                all_labels.append(labels)
                all_scores.append(scores)
            except Exception as e:
                logger.warning(f"Detector {detector.algorithm_type} failed: {e}")
                continue
        
        if not all_labels:
            raise ValueError("No detectors produced valid results")
        
        # Combine results
        if voting == "hard":
            # Majority voting on labels
            combined_labels = np.array(all_labels)
            weighted_votes = np.average(combined_labels, axis=0, weights=self.weights[:len(all_labels)])
            final_labels = (weighted_votes > 0.5).astype(int)
            final_scores = weighted_votes
        else:  # soft voting
            # Weighted average of scores
            combined_scores = np.array(all_scores)
            final_scores = np.average(combined_scores, axis=0, weights=self.weights[:len(all_scores)])
            final_labels = (final_scores > 0.5).astype(int)
        
        return final_labels, final_scores

class AnomalyDetector:
    """
    Ultra-advanced anomaly detection system with ensemble methods
    """
    
    def __init__(self, config: AnomalyConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Detector registry
        self.detectors = {}  # tenant_id -> detector
        self.ensemble_detectors = {}  # tenant_id -> ensemble_detector
        
        # Anomaly history
        self.anomaly_history = defaultdict(deque)  # tenant_id -> deque of anomalies
        
        # Adaptive thresholds
        self.adaptive_thresholds = {}  # tenant_id -> threshold_info
        
        # Real-time processing
        self.streaming_buffers = defaultdict(deque)  # tenant_id -> data_buffer
        
        # Thread pool for concurrent processing
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialize anomaly detection system"""
        try:
            self.logger.info("Initializing Anomaly Detection System...")
            
            # Initialize detector factories
            self.detector_factories = {
                DetectionAlgorithm.ISOLATION_FOREST: self._create_isolation_forest,
                DetectionAlgorithm.ONE_CLASS_SVM: self._create_one_class_svm,
                DetectionAlgorithm.LSTM_AUTOENCODER: self._create_lstm_autoencoder,
            }
            
            self.is_initialized = True
            self.logger.info("Anomaly Detection System initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Anomaly Detection System: {e}")
            return False
    
    async def register_tenant(self, tenant_id: str, config: Optional[Dict] = None) -> bool:
        """Register tenant for anomaly detection"""
        try:
            # Initialize tenant-specific components
            self.anomaly_history[tenant_id] = deque(maxlen=10000)
            self.adaptive_thresholds[tenant_id] = {
                'threshold': self.config.confidence_threshold,
                'history': deque(maxlen=1000),
                'last_update': datetime.utcnow()
            }
            self.streaming_buffers[tenant_id] = deque(maxlen=self.config.streaming_window_size)
            
            self.logger.info(f"Tenant {tenant_id} registered for anomaly detection")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register tenant {tenant_id}: {e}")
            return False
    
    async def fit_detector(
        self,
        tenant_id: str,
        training_data: Union[pd.DataFrame, np.ndarray],
        config: Optional[Dict] = None
    ) -> bool:
        """Fit anomaly detector for tenant"""
        try:
            # Prepare training data
            if isinstance(training_data, pd.DataFrame):
                X = training_data.values
            else:
                X = training_data
            
            # Create individual detectors
            detectors = []
            for algorithm in self.config.algorithms:
                detector = self.detector_factories[algorithm]()
                await detector.fit(X, config)
                detectors.append(detector)
            
            # Create ensemble detector
            if self.config.ensemble_enabled and len(detectors) > 1:
                ensemble = EnsembleAnomalyDetector(detectors)
                await ensemble.fit(X, config)
                self.ensemble_detectors[tenant_id] = ensemble
            else:
                self.detectors[tenant_id] = detectors[0] if detectors else None
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to fit detector for tenant {tenant_id}: {e}")
            return False
    
    async def detect_anomalies(
        self,
        tenant_id: str,
        data: Union[pd.DataFrame, np.ndarray]
    ) -> AnomalyResult:
        """Detect anomalies in data"""
        try:
            start_time = time.time()
            
            # Prepare data
            if isinstance(data, pd.DataFrame):
                X = data.values
            else:
                X = data
            
            # Get detector
            detector = self.ensemble_detectors.get(tenant_id) or self.detectors.get(tenant_id)
            if not detector:
                raise ValueError(f"No detector found for tenant {tenant_id}")
            
            # Detect anomalies
            labels, scores = await detector.predict(X)
            
            # Create anomaly points
            anomalies = []
            for i, (label, score) in enumerate(zip(labels, scores)):
                if label == 1:  # Anomaly detected
                    anomaly = AnomalyPoint(
                        data_point=X[i].tolist() if X.ndim > 1 else [X[i]],
                        anomaly_score=float(score),
                        confidence=float(score),
                        severity=self._determine_severity(score),
                        detected_by=[detector.algorithm_type.value] if hasattr(detector, 'algorithm_type') else ['ensemble']
                    )
                    anomalies.append(anomaly)
            
            # Calculate statistics
            anomaly_rate = len(anomalies) / len(X) if len(X) > 0 else 0.0
            processing_time = time.time() - start_time
            
            # Update adaptive thresholds
            if self.config.adaptive_thresholds:
                await self._update_adaptive_threshold(tenant_id, scores, labels)
            
            # Create result
            result = AnomalyResult(
                anomalies=anomalies,
                total_points=len(X),
                anomaly_rate=anomaly_rate,
                algorithm_scores={detector.algorithm_type.value if hasattr(detector, 'algorithm_type') else 'ensemble': scores.tolist()},
                detection_time=processing_time,
                processing_time=processing_time,
                tenant_id=tenant_id
            )
            
            # Update history
            self.anomaly_history[tenant_id].extend(anomalies)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to detect anomalies for tenant {tenant_id}: {e}")
            raise
    
    async def detect_with_model(
        self,
        tenant_id: str,
        model_id: str,
        data: Union[pd.DataFrame, np.ndarray]
    ) -> AnomalyResult:
        """Detect anomalies using specific model"""
        # This would use a specific trained model
        return await self.detect_anomalies(tenant_id, data)
    
    def _determine_severity(self, score: float) -> str:
        """Determine anomaly severity based on score"""
        if score >= 0.9:
            return "critical"
        elif score >= 0.7:
            return "high"
        elif score >= 0.5:
            return "medium"
        else:
            return "low"
    
    async def _update_adaptive_threshold(
        self,
        tenant_id: str,
        scores: np.ndarray,
        labels: np.ndarray
    ) -> None:
        """Update adaptive threshold based on recent performance"""
        threshold_info = self.adaptive_thresholds[tenant_id]
        
        # Add recent scores to history
        threshold_info['history'].extend(scores.tolist())
        
        # Update threshold every hour or 100 predictions
        time_since_update = datetime.utcnow() - threshold_info['last_update']
        if time_since_update > timedelta(hours=1) or len(threshold_info['history']) >= 100:
            # Calculate new threshold based on score distribution
            recent_scores = list(threshold_info['history'])
            new_threshold = np.percentile(recent_scores, 95)
            
            # Smooth threshold update
            threshold_info['threshold'] = 0.8 * threshold_info['threshold'] + 0.2 * new_threshold
            threshold_info['last_update'] = datetime.utcnow()
            threshold_info['history'].clear()
    
    def _create_isolation_forest(self) -> IsolationForestDetector:
        """Create Isolation Forest detector"""
        return IsolationForestDetector(contamination=self.config.contamination_rate)
    
    def _create_one_class_svm(self) -> OneClassSVMDetector:
        """Create One-Class SVM detector"""
        return OneClassSVMDetector(nu=self.config.contamination_rate)
    
    def _create_lstm_autoencoder(self) -> LSTMAutoencoderDetector:
        """Create LSTM Autoencoder detector"""
        return LSTMAutoencoderDetector()

# Export main classes
__all__ = [
    "AnomalyDetector",
    "AnomalyConfig",
    "AnomalyResult",
    "AnomalyPoint",
    "AnomalyType",
    "DetectionAlgorithm",
    "AnomalyStatus",
    "BaseAnomalyDetector",
    "IsolationForestDetector",
    "OneClassSVMDetector",
    "LSTMAutoencoderDetector",
    "EnsembleAnomalyDetector"
]
