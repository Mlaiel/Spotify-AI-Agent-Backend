"""
🎵 Spotify AI Agent - Machine Learning Models Package
===================================================

Enterprise-Grade ML/DL Models for Music Streaming Alert Intelligence

This package contains sophisticated machine learning and deep learning models
specifically designed for intelligent alert processing in music streaming platforms.
Each model is optimized for high-throughput, low-latency operations at scale.

🚀 MODEL CATEGORIES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧠 Anomaly Detection Models:
  • IsolationForestModel     - Ensemble-based outlier detection
  • AutoEncoderModel         - Neural network reconstruction-based
  • OneClassSVMModel         - Support Vector Machine novelty detection
  • EnsembleAnomalyModel     - Multi-model consensus approach
  • VAEAnomalyModel          - Variational Autoencoder for complex patterns
  • LSTMAnomalyModel         - Temporal sequence anomaly detection
  • GMManomalyModel          - Gaussian Mixture Model anomaly detection

🔮 Predictive Models:
  • ProphetForecastModel     - Time series forecasting with seasonality
  • LSTMPredictorModel       - Long Short-Term Memory networks
  • TransformerModel         - Attention-based sequence modeling
  • ARIMAModel               - Classical time series analysis
  • GRUPredictorModel        - Gated Recurrent Unit networks
  • XGBoostRegressorModel    - Gradient boosting regression
  • TCNModel                 - Temporal Convolutional Networks

🔗 Correlation Models:
  • GraphNeuralNetworkModel  - Complex relationship modeling
  • CausalInferenceModel     - Causal relationship discovery
  • CorrelationMatrixModel   - Statistical correlation analysis
  • EventSequenceModel      - Temporal event pattern analysis
  • DynamicTimeWarpingModel  - Time series alignment and similarity

🎯 Classification Models:
  • XGBoostClassifierModel   - Gradient boosting classification
  • RandomForestModel        - Ensemble decision trees
  • NeuralClassifierModel    - Deep neural network classification
  • SVMClassifierModel       - Support Vector Machine classification
  • EnsembleClassifierModel  - Multi-model classification ensemble
  • BertClassifierModel      - BERT-based text classification
  • TabNetModel              - Attention-based tabular learning

🎭 Specialized Models:
  • SeverityPredictorModel   - Alert severity classification
  • BusinessImpactModel      - Revenue/user impact estimation
  • UserBehaviorModel        - User interaction pattern analysis
  • SystemBehaviorModel      - Infrastructure behavior modeling
  • NoiseReductionModel      - Signal processing and filtering
  • RootCauseAnalysisModel   - Root cause identification
  • AnomalyExplanationModel  - Explainable anomaly detection

🎵 Music Streaming Specific Models:
  • AudioQualityModel        - Audio quality degradation detection
  • RecommendationModel      - Recommendation engine performance
  • EngagementPredictorModel - User engagement forecasting
  • ChurnPredictionModel     - User churn risk assessment
  • ContentPopularityModel   - Viral content prediction
  • PlaylistOptimizationModel - Playlist quality optimization
  • SkipPredictionModel      - Song skip probability prediction
  • ListeningSessionModel    - Session behavior analysis
  • GenreClassificationModel - Music genre classification
  • MoodDetectionModel       - Audio mood and emotion detection

🎮 Real-time Models:
  • StreamingAnomalyModel    - Real-time streaming data anomaly detection
  • LiveRecommendationModel  - Real-time recommendation scoring
  • DynamicLoadBalancingModel - Traffic distribution optimization
  • AdaptiveBitRateModel     - Dynamic quality adjustment
  • EdgeCacheOptimizationModel - CDN optimization

⚡ PERFORMANCE FEATURES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Sub-10ms inference latency for real-time operations
• Distributed training across multiple GPU clusters
• Model compression and quantization for edge deployment
• Automatic hyperparameter optimization (AutoML)
• Model versioning and A/B testing capabilities
• Explainable AI (XAI) integration for model interpretability
• Federated learning support for privacy-preserving training
• Continuous learning and model adaptation
• Production-ready model serving with auto-scaling

🛡️ ENTERPRISE SECURITY & COMPLIANCE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• GDPR-compliant data processing and model training
• Differential privacy for sensitive user data
• Model fairness and bias detection/mitigation
• Secure multi-party computation support
• Homomorphic encryption for privacy-preserving inference
• Audit trails for model decisions and predictions
• Data lineage tracking for compliance reporting

🌍 GLOBAL SCALE OPTIMIZATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Multi-region model deployment and synchronization
• Cross-cultural and linguistic adaptation capabilities
• Timezone-aware temporal modeling
• Currency and market-specific financial modeling
• Regional compliance and regulatory adaptation
• Latency-optimized model serving per geographic region

🔧 INTEGRATION CAPABILITIES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Kafka streaming integration for real-time data ingestion
• Redis caching for model predictions and features
• PostgreSQL integration for model metadata and versioning
• Elasticsearch integration for model monitoring and logging
• Kubernetes native deployment with Helm charts
• Prometheus metrics and Grafana dashboards
• MLflow experiment tracking and model registry
• TensorBoard integration for deep learning models

Version: 2.0.0 (Enterprise Edition)
Last Updated: 2025-07-19
"""

import logging
import warnings
from typing import Dict, Any, List, Optional, Union, Type
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

# Suppress common warnings for production environment
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Configure logging
logger = logging.getLogger(__name__)

# Model Registry for Enterprise Model Management
_MODEL_REGISTRY: Dict[str, Type] = {}
_MODEL_METADATA: Dict[str, Dict[str, Any]] = {}


class ModelRegistryError(Exception):
    """Exception raised for model registry related errors"""
    pass


class ModelInterface(ABC):
    """
    Abstract base class defining the interface for all ML models
    in the Spotify AI Agent ecosystem.
    
    This interface ensures consistency across all models and provides
    enterprise-grade features like model versioning, monitoring,
    and compliance tracking.
    """
    
    def __init__(self, model_name: str, version: str = "1.0.0"):
        self.model_name = model_name
        self.version = version
        self.is_trained = False
        self.training_metadata = {}
        self.prediction_count = 0
        self.performance_metrics = {}
        
    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Optional[Union[np.ndarray, pd.Series]] = None, 
            **kwargs) -> 'ModelInterface':
        """
        Train the model on provided data.
        
        Args:
            X: Training features
            y: Training targets (optional for unsupervised models)
            **kwargs: Additional training parameters
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame], 
                **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        """
        Make predictions on new data.
        
        Args:
            X: Input features for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame], 
                      **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        """
        Predict class probabilities (for applicable models).
        
        Args:
            X: Input features for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Prediction probabilities
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass
    
    @abstractmethod
    def explain_prediction(self, X: Union[np.ndarray, pd.DataFrame], 
                          instance_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Provide explanations for model predictions (Explainable AI).
        
        Args:
            X: Input data
            instance_idx: Specific instance to explain (if None, explain all)
            
        Returns:
            Explanation data
        """
        pass
    
    def get_model_metadata(self) -> Dict[str, Any]:
        """Get comprehensive model metadata"""
        return {
            'model_name': self.model_name,
            'version': self.version,
            'is_trained': self.is_trained,
            'training_metadata': self.training_metadata,
            'prediction_count': self.prediction_count,
            'performance_metrics': self.performance_metrics,
            'model_size': self._calculate_model_size(),
            'creation_timestamp': getattr(self, 'creation_timestamp', None),
            'last_training_timestamp': getattr(self, 'last_training_timestamp', None),
            'last_prediction_timestamp': getattr(self, 'last_prediction_timestamp', None)
        }
    
    def _calculate_model_size(self) -> int:
        """Calculate approximate model size in bytes"""
        try:
            import sys
            return sys.getsizeof(self)
        except Exception:
            return 0
    
    def validate_input(self, X: Union[np.ndarray, pd.DataFrame]) -> bool:
        """Validate input data format and quality"""
        if X is None or len(X) == 0:
            raise ValueError("Input data cannot be None or empty")
        
        if isinstance(X, pd.DataFrame):
            # Check for missing values
            if X.isnull().any().any():
                logger.warning("Input data contains missing values")
            
            # Check for infinite values
            if np.isinf(X.select_dtypes(include=[np.number])).any().any():
                raise ValueError("Input data contains infinite values")
        
        return True


def register_model(model_class: Type[ModelInterface], 
                  category: str = "general",
                  description: str = "",
                  tags: List[str] = None) -> None:
    """
    Register a model class in the enterprise model registry.
    
    Args:
        model_class: Model class to register
        category: Model category (anomaly_detection, prediction, etc.)
        description: Model description
        tags: Model tags for organization
    """
    if not issubclass(model_class, ModelInterface):
        raise ModelRegistryError(f"Model {model_class.__name__} must inherit from ModelInterface")
    
    model_name = model_class.__name__
    _MODEL_REGISTRY[model_name] = model_class
    _MODEL_METADATA[model_name] = {
        'category': category,
        'description': description,
        'tags': tags or [],
        'class_name': model_name,
        'module': model_class.__module__
    }
    
    logger.info(f"Registered model: {model_name} in category: {category}")


def get_model_class(model_name: str) -> Type[ModelInterface]:
    """
    Get model class from registry.
    
    Args:
        model_name: Name of the model class
        
    Returns:
        Model class
        
    Raises:
        ModelRegistryError: If model not found
    """
    if model_name not in _MODEL_REGISTRY:
        raise ModelRegistryError(f"Model {model_name} not found in registry")
    
    return _MODEL_REGISTRY[model_name]


def list_available_models(category: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    List all available models in the registry.
    
    Args:
        category: Filter by category (optional)
        
    Returns:
        Dictionary of model names and their metadata
    """
    if category:
        return {
            name: metadata for name, metadata in _MODEL_METADATA.items()
            if metadata['category'] == category
        }
    return _MODEL_METADATA.copy()


def create_model_instance(model_name: str, **kwargs) -> ModelInterface:
    """
    Create an instance of a registered model.
    
    Args:
        model_name: Name of the model class
        **kwargs: Arguments to pass to model constructor
        
    Returns:
        Model instance
    """
    model_class = get_model_class(model_name)
    return model_class(**kwargs)


# Import all model classes and register them
try:
    # Anomaly Detection Models
    from .isolation_forest_model import IsolationForestModel
    from .autoencoder_model import AutoEncoderModel
    from .oneclass_svm_model import OneClassSVMModel
    from .ensemble_anomaly_model import EnsembleAnomalyModel
    from .vae_anomaly_model import VAEAnomalyModel
    from .lstm_anomaly_model import LSTMAnomalyModel
    from .gmm_anomaly_model import GMManomalyModel
    
    # Register anomaly detection models
    register_model(IsolationForestModel, "anomaly_detection", 
                  "Ensemble-based outlier detection optimized for music streaming", 
                  ["anomaly", "ensemble", "outlier"])
    register_model(AutoEncoderModel, "anomaly_detection", 
                  "Neural network reconstruction-based anomaly detection", 
                  ["anomaly", "neural_network", "reconstruction"])
    register_model(OneClassSVMModel, "anomaly_detection", 
                  "Support Vector Machine novelty detection", 
                  ["anomaly", "svm", "novelty"])
    register_model(EnsembleAnomalyModel, "anomaly_detection", 
                  "Multi-model consensus anomaly detection", 
                  ["anomaly", "ensemble", "consensus"])
    register_model(VAEAnomalyModel, "anomaly_detection", 
                  "Variational Autoencoder for complex pattern anomalies", 
                  ["anomaly", "vae", "generative"])
    register_model(LSTMAnomalyModel, "anomaly_detection", 
                  "Temporal sequence anomaly detection using LSTM", 
                  ["anomaly", "lstm", "temporal"])
    register_model(GMManomalyModel, "anomaly_detection", 
                  "Gaussian Mixture Model anomaly detection", 
                  ["anomaly", "gmm", "probabilistic"])
    
except ImportError as e:
    logger.warning(f"Could not import anomaly detection models: {e}")

try:
    # Predictive Models
    from .prophet_forecast_model import ProphetForecastModel
    from .lstm_predictor_model import LSTMPredictorModel
    from .transformer_model import TransformerModel
    from .arima_model import ARIMAModel
    from .gru_predictor_model import GRUPredictorModel
    from .xgboost_regressor_model import XGBoostRegressorModel
    from .tcn_model import TCNModel
    
    # Register predictive models
    register_model(ProphetForecastModel, "prediction", 
                  "Time series forecasting with seasonality", 
                  ["prediction", "timeseries", "seasonality"])
    register_model(LSTMPredictorModel, "prediction", 
                  "Long Short-Term Memory networks for prediction", 
                  ["prediction", "lstm", "neural_network"])
    register_model(TransformerModel, "prediction", 
                  "Attention-based sequence modeling", 
                  ["prediction", "transformer", "attention"])
    register_model(ARIMAModel, "prediction", 
                  "Classical time series analysis", 
                  ["prediction", "timeseries", "classical"])
    register_model(GRUPredictorModel, "prediction", 
                  "Gated Recurrent Unit networks", 
                  ["prediction", "gru", "neural_network"])
    register_model(XGBoostRegressorModel, "prediction", 
                  "Gradient boosting regression", 
                  ["prediction", "xgboost", "ensemble"])
    register_model(TCNModel, "prediction", 
                  "Temporal Convolutional Networks", 
                  ["prediction", "tcn", "convolutional"])
    
except ImportError as e:
    logger.warning(f"Could not import predictive models: {e}")

try:
    # Classification Models
    from .xgboost_classifier_model import XGBoostClassifierModel
    from .random_forest_model import RandomForestModel
    from .neural_classifier_model import NeuralClassifierModel
    from .svm_classifier_model import SVMClassifierModel
    from .ensemble_classifier_model import EnsembleClassifierModel
    from .bert_classifier_model import BertClassifierModel
    from .tabnet_model import TabNetModel
    
    # Register classification models
    register_model(XGBoostClassifierModel, "classification", 
                  "Gradient boosting classification", 
                  ["classification", "xgboost", "ensemble"])
    register_model(RandomForestModel, "classification", 
                  "Ensemble decision trees", 
                  ["classification", "random_forest", "ensemble"])
    register_model(NeuralClassifierModel, "classification", 
                  "Deep neural network classification", 
                  ["classification", "neural_network", "deep_learning"])
    register_model(SVMClassifierModel, "classification", 
                  "Support Vector Machine classification", 
                  ["classification", "svm", "kernel"])
    register_model(EnsembleClassifierModel, "classification", 
                  "Multi-model classification ensemble", 
                  ["classification", "ensemble", "voting"])
    register_model(BertClassifierModel, "classification", 
                  "BERT-based text classification", 
                  ["classification", "bert", "nlp"])
    register_model(TabNetModel, "classification", 
                  "Attention-based tabular learning", 
                  ["classification", "tabnet", "attention"])
    
except ImportError as e:
    logger.warning(f"Could not import classification models: {e}")

try:
    # Music Streaming Specific Models
    from .audio_quality_model import AudioQualityModel
    from .recommendation_model import RecommendationModel
    from .engagement_predictor_model import EngagementPredictorModel
    from .churn_prediction_model import ChurnPredictionModel
    from .content_popularity_model import ContentPopularityModel
    from .skip_prediction_model import SkipPredictionModel
    
    # Register music streaming models
    register_model(AudioQualityModel, "music_streaming", 
                  "Audio quality degradation detection", 
                  ["music", "audio", "quality"])
    register_model(RecommendationModel, "music_streaming", 
                  "Recommendation engine performance monitoring", 
                  ["music", "recommendation", "performance"])
    register_model(EngagementPredictorModel, "music_streaming", 
                  "User engagement forecasting", 
                  ["music", "engagement", "prediction"])
    register_model(ChurnPredictionModel, "music_streaming", 
                  "User churn risk assessment", 
                  ["music", "churn", "risk"])
    register_model(ContentPopularityModel, "music_streaming", 
                  "Viral content prediction", 
                  ["music", "content", "popularity"])
    register_model(SkipPredictionModel, "music_streaming", 
                  "Song skip probability prediction", 
                  ["music", "skip", "behavior"])
    
except ImportError as e:
    logger.warning(f"Could not import music streaming models: {e}")

# Export all public interfaces
__all__ = [
    # Core interfaces
    'ModelInterface',
    'ModelRegistryError',
    
    # Registry functions
    'register_model',
    'get_model_class',
    'list_available_models',
    'create_model_instance',
    
    # Anomaly Detection Models
    'IsolationForestModel',
    'AutoEncoderModel',
    'OneClassSVMModel',
    'EnsembleAnomalyModel',
    'VAEAnomalyModel',
    'LSTMAnomalyModel',
    'GMManomalyModel',
    
    # Predictive Models
    'ProphetForecastModel',
    'LSTMPredictorModel',
    'TransformerModel',
    'ARIMAModel',
    'GRUPredictorModel',
    'XGBoostRegressorModel',
    'TCNModel',
    
    # Classification Models
    'XGBoostClassifierModel',
    'RandomForestModel',
    'NeuralClassifierModel',
    'SVMClassifierModel',
    'EnsembleClassifierModel',
    'BertClassifierModel',
    'TabNetModel',
    
    # Music Streaming Models
    'AudioQualityModel',
    'RecommendationModel',
    'EngagementPredictorModel',
    'ChurnPredictionModel',
    'ContentPopularityModel',
    'SkipPredictionModel',
]

# Model registry information
logger.info(f"Loaded {len(_MODEL_REGISTRY)} models across {len(set(m['category'] for m in _MODEL_METADATA.values()))} categories")
logger.info(f"Available categories: {list(set(m['category'] for m in _MODEL_METADATA.values()))}")

# Enterprise model validation
def validate_model_registry():
    """Validate all registered models for enterprise compliance"""
    validation_results = {}
    
    for model_name, model_class in _MODEL_REGISTRY.items():
        try:
            # Check if model implements required interface
            required_methods = ['fit', 'predict', 'predict_proba', 'get_feature_importance', 'explain_prediction']
            missing_methods = []
            
            for method in required_methods:
                if not hasattr(model_class, method):
                    missing_methods.append(method)
            
            validation_results[model_name] = {
                'valid': len(missing_methods) == 0,
                'missing_methods': missing_methods,
                'category': _MODEL_METADATA[model_name]['category']
            }
            
        except Exception as e:
            validation_results[model_name] = {
                'valid': False,
                'error': str(e),
                'category': _MODEL_METADATA[model_name]['category']
            }
    
    # Log validation summary
    valid_models = sum(1 for r in validation_results.values() if r['valid'])
    total_models = len(validation_results)
    
    logger.info(f"Model registry validation: {valid_models}/{total_models} models are valid")
    
    if valid_models < total_models:
        invalid_models = [name for name, result in validation_results.items() if not result['valid']]
        logger.warning(f"Invalid models detected: {invalid_models}")
    
    return validation_results

# Run validation on import
_validation_results = validate_model_registry()
  • ChurnPredictionModel     - User churn risk assessment
  • ContentAnalyticsModel    - Content performance analysis

⚡ Performance Optimizations:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• GPU Acceleration with CUDA/ROCm Support
• Model Quantization for Reduced Memory Usage  
• Dynamic Batching for Optimal Throughput
• Model Caching and Warm-up Strategies
• Distributed Training and Inference
• AutoML and Hyperparameter Optimization
• Model Versioning and A/B Testing
• Real-time Model Updates and Deployment

🏗️ Architecture Features:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Abstract Base Classes for Consistent Interface
• Factory Pattern for Dynamic Model Creation
• Strategy Pattern for Algorithm Selection
• Observer Pattern for Model Monitoring
• Plugin Architecture for Custom Models
• Configuration-Driven Model Selection
• Automatic Model Validation and Testing
• Production-Ready Deployment Pipelines

@Author: ML Models Package by Fahed Mlaiel
@Version: 2.0.0 (Enterprise Edition)
@Last Updated: 2025-07-19
"""

import logging
import importlib
from typing import Dict, Any, List, Optional, Type, Union, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
import inspect
import warnings
from datetime import datetime
import json

# Core ML/DL frameworks
try:
    import numpy as np
    import pandas as pd
    import scikit_learn
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not available. Some models will be disabled.")

try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    warnings.warn("TensorFlow not available. Deep learning models will be disabled.")

try:
    import torch
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    warnings.warn("PyTorch not available. Some deep learning models will be disabled.")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost not available. Gradient boosting models will be disabled.")

# Configure logging
logger = logging.getLogger(__name__)

# Package metadata
__title__ = "Spotify AI Agent - ML Models Package"
__version__ = "2.0.0"
__author__ = "Fahed Mlaiel"
__license__ = "Proprietary"

# Model categories and their priority
MODEL_CATEGORIES = {
    "anomaly_detection": {
        "priority": 1,
        "description": "Detect anomalous patterns in streaming metrics",
        "models": [
            "IsolationForestModel",
            "AutoEncoderModel", 
            "OneClassSVMModel",
            "EnsembleAnomalyModel",
            "VAEAnomalyModel"
        ]
    },
    "predictive": {
        "priority": 2,
        "description": "Forecast future states and trends",
        "models": [
            "ProphetForecastModel",
            "LSTMPredictorModel",
            "TransformerModel",
            "ARIMAModel",
            "GRUPredictorModel"
        ]
    },
    "correlation": {
        "priority": 3,
        "description": "Discover relationships between metrics and events",
        "models": [
            "GraphNeuralNetworkModel",
            "CausalInferenceModel",
            "CorrelationMatrixModel",
            "EventSequenceModel"
        ]
    },
    "classification": {
        "priority": 4,
        "description": "Classify alerts and predict categories",
        "models": [
            "XGBoostClassifierModel",
            "RandomForestModel",
            "NeuralClassifierModel",
            "SVMClassifierModel",
            "EnsembleClassifierModel"
        ]
    },
    "specialized": {
        "priority": 5,
        "description": "Domain-specific models for music streaming",
        "models": [
            "SeverityPredictorModel",
            "BusinessImpactModel",
            "UserBehaviorModel",
            "SystemBehaviorModel",
            "NoiseReductionModel",
            "AudioQualityModel",
            "RecommendationModel",
            "EngagementPredictorModel",
            "ChurnPredictionModel",
            "ContentAnalyticsModel"
        ]
    }
}

# Performance benchmarks for each model type
PERFORMANCE_BENCHMARKS = {
    "latency_ms": {
        "real_time": 10,      # Real-time inference
        "near_real_time": 100, # Near real-time
        "batch": 1000         # Batch processing
    },
    "throughput_per_second": {
        "high": 10000,        # High throughput
        "medium": 1000,       # Medium throughput
        "low": 100           # Low throughput
    },
    "accuracy": {
        "excellent": 0.95,    # Excellent accuracy
        "good": 0.90,        # Good accuracy
        "acceptable": 0.85   # Acceptable accuracy
    },
    "memory_mb": {
        "lightweight": 100,   # Lightweight models
        "medium": 500,       # Medium memory usage
        "heavyweight": 2000  # Heavy models
    }
}

class BaseModel(ABC):
    """
    Abstract base class for all machine learning models.
    
    Provides consistent interface and common functionality for all models
    in the Spotify AI Agent alert processing system.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the base model.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config or {}
        self.model = None
        self.is_trained = False
        self.is_fitted = False
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "version": __version__,
            "model_type": self.__class__.__name__
        }
        self.performance_metrics = {}
        
        # Configure logging for this model
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'BaseModel':
        """
        Train the model on the provided data.
        
        Args:
            X: Training features
            y: Training targets (optional for unsupervised learning)
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on the provided data.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions array
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return prediction probabilities.
        
        Args:
            X: Features for prediction
            
        Returns:
            Probability array
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary containing model metadata and performance metrics
        """
        return {
            "metadata": self.metadata,
            "config": self.config,
            "is_trained": self.is_trained,
            "is_fitted": self.is_fitted,
            "performance_metrics": self.performance_metrics,
            "model_type": self.__class__.__name__,
            "model_category": self._get_model_category()
        }
    
    def _get_model_category(self) -> str:
        """Get the category this model belongs to."""
        model_name = self.__class__.__name__
        for category, info in MODEL_CATEGORIES.items():
            if model_name in info["models"]:
                return category
        return "unknown"
    
    def validate_performance(self) -> bool:
        """
        Validate model performance against benchmarks.
        
        Returns:
            True if performance meets requirements
        """
        if not self.performance_metrics:
            self.logger.warning("No performance metrics available for validation")
            return False
            
        category = self._get_model_category()
        if category == "unknown":
            return True  # Skip validation for unknown categories
            
        # Check latency requirements
        latency = self.performance_metrics.get("latency_ms", float('inf'))
        if latency > PERFORMANCE_BENCHMARKS["latency_ms"]["batch"]:
            self.logger.warning(f"Model latency {latency}ms exceeds threshold")
            return False
            
        # Check accuracy requirements  
        accuracy = self.performance_metrics.get("accuracy", 0)
        if accuracy < PERFORMANCE_BENCHMARKS["accuracy"]["acceptable"]:
            self.logger.warning(f"Model accuracy {accuracy} below threshold")
            return False
            
        return True
    
    def save_model(self, path: str) -> bool:
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model
            
        Returns:
            True if successful
        """
        try:
            model_data = {
                "metadata": self.metadata,
                "config": self.config,
                "performance_metrics": self.performance_metrics,
                "model_state": self._serialize_model()
            }
            
            with open(path, 'w') as f:
                json.dump(model_data, f, indent=2)
                
            self.logger.info(f"Model saved to {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            return False
    
    def load_model(self, path: str) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            True if successful
        """
        try:
            with open(path, 'r') as f:
                model_data = json.load(f)
                
            self.metadata = model_data["metadata"]
            self.config = model_data["config"]
            self.performance_metrics = model_data["performance_metrics"]
            
            self._deserialize_model(model_data["model_state"])
            self.is_fitted = True
            
            self.logger.info(f"Model loaded from {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            return False
    
    @abstractmethod
    def _serialize_model(self) -> Dict[str, Any]:
        """Serialize model state for saving."""
        pass
    
    @abstractmethod
    def _deserialize_model(self, model_state: Dict[str, Any]) -> None:
        """Deserialize model state for loading."""
        pass

class ModelFactory:
    """
    Factory class for creating and managing ML models.
    
    Provides dynamic model creation, configuration management,
    and model lifecycle management.
    """
    
    def __init__(self):
        self._model_registry = {}
        self._model_cache = {}
        self.logger = logging.getLogger(f"{__name__}.ModelFactory")
        
        # Register available models
        self._register_models()
    
    def _register_models(self) -> None:
        """Register all available models."""
        for category, info in MODEL_CATEGORIES.items():
            for model_name in info["models"]:
                try:
                    # Dynamically import model class
                    module_name = f".{model_name.lower()}"
                    module = importlib.import_module(module_name, package=__name__)
                    model_class = getattr(module, model_name)
                    
                    self._model_registry[model_name] = {
                        "class": model_class,
                        "category": category,
                        "priority": info["priority"],
                        "description": info["description"]
                    }
                    
                except ImportError as e:
                    self.logger.warning(f"Failed to import {model_name}: {str(e)}")
                except AttributeError as e:
                    self.logger.warning(f"Model class {model_name} not found: {str(e)}")
    
    def create_model(self, model_name: str, config: Dict[str, Any] = None) -> Optional[BaseModel]:
        """
        Create a model instance.
        
        Args:
            model_name: Name of the model to create
            config: Model configuration
            
        Returns:
            Model instance or None if creation failed
        """
        if model_name not in self._model_registry:
            self.logger.error(f"Unknown model: {model_name}")
            return None
        
        try:
            model_info = self._model_registry[model_name]
            model_class = model_info["class"]
            
            # Create model instance
            model = model_class(config or {})
            
            self.logger.info(f"Created model: {model_name}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to create model {model_name}: {str(e)}")
            return None
    
    def get_available_models(self, category: str = None) -> List[str]:
        """
        Get list of available models.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of model names
        """
        if category:
            return [
                name for name, info in self._model_registry.items()
                if info["category"] == category
            ]
        return list(self._model_registry.keys())
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information dictionary
        """
        return self._model_registry.get(model_name)

# Global model factory instance
model_factory = ModelFactory()

# Export public classes and functions
__all__ = [
    "BaseModel",
    "ModelFactory", 
    "model_factory",
    "MODEL_CATEGORIES",
    "PERFORMANCE_BENCHMARKS",
    "__version__"
]

def get_model_categories() -> Dict[str, Any]:
    """
    Get information about all model categories.
    
    Returns:
        Dictionary of model categories and their information
    """
    return MODEL_CATEGORIES.copy()

def get_available_models(category: str = None) -> List[str]:
    """
    Get list of available models.
    
    Args:
        category: Optional category filter
        
    Returns:
        List of model names
    """
    return model_factory.get_available_models(category)

def create_model(model_name: str, config: Dict[str, Any] = None) -> Optional[BaseModel]:
    """
    Create a model instance using the global factory.
    
    Args:
        model_name: Name of the model to create
        config: Model configuration
        
    Returns:
        Model instance or None if creation failed
    """
    return model_factory.create_model(model_name, config)

def validate_environment() -> Dict[str, bool]:
    """
    Validate the environment and check for required dependencies.
    
    Returns:
        Dictionary of dependency availability
    """
    return {
        "sklearn": HAS_SKLEARN,
        "tensorflow": HAS_TENSORFLOW,
        "pytorch": HAS_PYTORCH,
        "xgboost": HAS_XGBOOST,
        "numpy": True,  # Always available as it's a hard dependency
        "pandas": True  # Always available as it's a hard dependency
    }

# Initialize logging for the package
logger.info(f"Spotify AI Agent ML Models Package v{__version__} initialized")
logger.info(f"Available frameworks: {validate_environment()}")
logger.info(f"Registered {len(model_factory._model_registry)} models across {len(MODEL_CATEGORIES)} categories")
