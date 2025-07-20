"""
Types et énumérations ML - Spotify AI Agent
Définitions des types pour Machine Learning et Intelligence Artificielle
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple, Literal
from uuid import UUID
from enum import Enum
from decimal import Decimal
import numpy as np

from pydantic import BaseModel, Field, validator
from ..base import BaseSchema, StrictEmail, StrictUrl


class ModelType(str, Enum):
    """Types de modèles ML supportés"""
    # Apprentissage supervisé
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    ORDINAL_REGRESSION = "ordinal_regression"
    
    # Apprentissage non supervisé
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    ANOMALY_DETECTION = "anomaly_detection"
    DENSITY_ESTIMATION = "density_estimation"
    
    # Apprentissage par renforcement
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    DEEP_Q_LEARNING = "deep_q_learning"
    POLICY_GRADIENT = "policy_gradient"
    ACTOR_CRITIC = "actor_critic"
    
    # Séries temporelles
    TIME_SERIES_FORECASTING = "time_series_forecasting"
    SEQUENCE_PREDICTION = "sequence_prediction"
    SEQUENCE_TO_SEQUENCE = "sequence_to_sequence"
    
    # Traitement du langage naturel
    NATURAL_LANGUAGE_PROCESSING = "nlp"
    TEXT_CLASSIFICATION = "text_classification"
    NAMED_ENTITY_RECOGNITION = "ner"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    MACHINE_TRANSLATION = "machine_translation"
    QUESTION_ANSWERING = "question_answering"
    
    # Vision par ordinateur
    COMPUTER_VISION = "computer_vision"
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    FACE_RECOGNITION = "face_recognition"
    
    # Audio et signal
    AUDIO_PROCESSING = "audio_processing"
    SPEECH_RECOGNITION = "speech_recognition"
    MUSIC_ANALYSIS = "music_analysis"
    AUDIO_CLASSIFICATION = "audio_classification"
    AUDIO_GENERATION = "audio_generation"
    
    # Recommandation
    RECOMMENDATION = "recommendation"
    COLLABORATIVE_FILTERING = "collaborative_filtering"
    CONTENT_BASED = "content_based"
    HYBRID_RECOMMENDATION = "hybrid_recommendation"
    
    # Modèles avancés
    ENSEMBLE = "ensemble"
    DEEP_LEARNING = "deep_learning"
    TRANSFER_LEARNING = "transfer_learning"
    FEW_SHOT_LEARNING = "few_shot_learning"
    META_LEARNING = "meta_learning"
    FEDERATED_LEARNING = "federated_learning"
    
    # AutoML
    AUTOML = "automl"
    NEURAL_ARCHITECTURE_SEARCH = "neural_architecture_search"
    HYPERPARAMETER_OPTIMIZATION = "hyperparameter_optimization"


class FrameworkType(str, Enum):
    """Frameworks ML/IA supportés"""
    # Deep Learning
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    KERAS = "keras"
    JAX = "jax"
    MXNET = "mxnet"
    CAFFE = "caffe"
    ONNX = "onnx"
    
    # Machine Learning classique
    SCIKIT_LEARN = "scikit_learn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    H2O = "h2o"
    WEKA = "weka"
    
    # NLP spécialisé
    HUGGING_FACE = "hugging_face"
    SPACY = "spacy"
    NLTK = "nltk"
    TRANSFORMERS = "transformers"
    BERT = "bert"
    GPT = "gpt"
    
    # Audio/Signal
    LIBROSA = "librosa"
    SCIPY_SIGNAL = "scipy_signal"
    ESSENTIA = "essentia"
    MADMOM = "madmom"
    SPLEETER = "spleeter"
    
    # Apprentissage par renforcement
    STABLE_BASELINES = "stable_baselines"
    RAY_RLLIB = "ray_rllib"
    OPENAI_GYM = "openai_gym"
    
    # AutoML
    AUTO_SKLEARN = "auto_sklearn"
    AUTO_KERAS = "auto_keras"
    H2O_AUTOML = "h2o_automl"
    TPOT = "tpot"
    
    # Distributed ML
    SPARK_ML = "spark_ml"
    DASK_ML = "dask_ml"
    RAY = "ray"
    HOROVOD = "horovod"


class ModelStatus(str, Enum):
    """Statuts du cycle de vie des modèles"""
    DRAFT = "draft"
    PENDING = "pending"
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATING = "validating"
    VALIDATED = "validated"
    TESTING = "testing"
    TESTED = "tested"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    SERVING = "serving"
    MONITORING = "monitoring"
    RETRAINING = "retraining"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingStatus(str, Enum):
    """Statuts d'entraînement des modèles"""
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    VALIDATING = "validating"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    EARLY_STOPPING = "early_stopping"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    RESUMED = "resumed"


class InferenceMode(str, Enum):
    """Modes d'inférence des modèles"""
    BATCH = "batch"
    REAL_TIME = "real_time"
    STREAMING = "streaming"
    EDGE = "edge"
    OFFLINE = "offline"
    ONLINE = "online"
    ASYNC = "async"
    SYNC = "sync"


class DeploymentTarget(str, Enum):
    """Cibles de déploiement des modèles"""
    KUBERNETES = "kubernetes"
    DOCKER = "docker"
    CLOUD_RUN = "cloud_run"
    AWS_LAMBDA = "aws_lambda"
    AWS_SAGEMAKER = "aws_sagemaker"
    AZURE_ML = "azure_ml"
    GCP_AI_PLATFORM = "gcp_ai_platform"
    EDGE_DEVICE = "edge_device"
    MOBILE = "mobile"
    WEB_BROWSER = "web_browser"
    IOT = "iot"
    EMBEDDED = "embedded"


class OptimizationObjective(str, Enum):
    """Objectifs d'optimisation"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC_ROC = "auc_roc"
    AUC_PR = "auc_pr"
    RMSE = "rmse"
    MAE = "mae"
    MAPE = "mape"
    R2_SCORE = "r2_score"
    LOG_LOSS = "log_loss"
    CROSS_ENTROPY = "cross_entropy"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    ENERGY_CONSUMPTION = "energy_consumption"
    MODEL_SIZE = "model_size"
    INFERENCE_COST = "inference_cost"
    TRAINING_TIME = "training_time"
    CUSTOM = "custom"


class DataType(str, Enum):
    """Types de données ML"""
    TABULAR = "tabular"
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TIME_SERIES = "time_series"
    GRAPH = "graph"
    GEOSPATIAL = "geospatial"
    MULTI_MODAL = "multi_modal"


class FeatureType(str, Enum):
    """Types de features"""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    BINARY = "binary"
    TEXT = "text"
    DATETIME = "datetime"
    GEOSPATIAL = "geospatial"
    EMBEDDING = "embedding"
    IMAGE = "image"
    AUDIO = "audio"
    DERIVED = "derived"
    ENGINEERED = "engineered"


class ModelArchitecture(str, Enum):
    """Architectures de modèles"""
    # ML classique
    LINEAR_REGRESSION = "linear_regression"
    LOGISTIC_REGRESSION = "logistic_regression"
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    SVM = "svm"
    KNN = "knn"
    NAIVE_BAYES = "naive_bayes"
    
    # Deep Learning
    FEEDFORWARD = "feedforward"
    CNN = "cnn"
    RNN = "rnn"
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    AUTOENCODER = "autoencoder"
    GAN = "gan"
    VAE = "vae"
    
    # Modèles spécialisés
    ATTENTION = "attention"
    RESNET = "resnet"
    BERT = "bert"
    GPT = "gpt"
    UNET = "unet"
    YOLO = "yolo"


class HyperparameterType(str, Enum):
    """Types d'hyperparamètres"""
    INTEGER = "integer"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    CHOICE = "choice"
    UNIFORM = "uniform"
    LOG_UNIFORM = "log_uniform"
    NORMAL = "normal"
    LOG_NORMAL = "log_normal"


class ExperimentStatus(str, Enum):
    """Statuts d'expérimentation"""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ModelFormat(str, Enum):
    """Formats de modèles"""
    PICKLE = "pickle"
    JOBLIB = "joblib"
    ONNX = "onnx"
    TENSORFLOW_SAVED_MODEL = "tensorflow_saved_model"
    PYTORCH_STATE_DICT = "pytorch_state_dict"
    PYTORCH_JIT = "pytorch_jit"
    KERAS_H5 = "keras_h5"
    SCIKIT_LEARN = "scikit_learn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    HUGGING_FACE = "hugging_face"
    MLFLOW = "mlflow"
    PMML = "pmml"
    COREML = "coreml"
    TENSORFLOW_LITE = "tensorflow_lite"


# Configurations de modèles
class ModelConfig(BaseSchema):
    """Configuration générale d'un modèle ML"""
    model_type: ModelType
    framework: FrameworkType
    architecture: Optional[ModelArchitecture] = None
    version: str = Field(default="1.0.0", description="Version du modèle")
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
    # Paramètres généraux
    random_seed: Optional[int] = Field(default=42, description="Graine aléatoire")
    debug_mode: bool = Field(default=False, description="Mode debug")
    
    # Ressources
    cpu_cores: Optional[int] = Field(default=None, ge=1, le=64)
    memory_gb: Optional[float] = Field(default=None, ge=0.5, le=512.0)
    gpu_count: Optional[int] = Field(default=0, ge=0, le=8)
    gpu_memory_gb: Optional[float] = Field(default=None, ge=1.0, le=80.0)
    
    # Configuration avancée
    distributed_training: bool = Field(default=False)
    mixed_precision: bool = Field(default=False)
    quantization: bool = Field(default=False)
    pruning: bool = Field(default=False)
    
    class Config:
        use_enum_values = True


class TrainingConfig(BaseSchema):
    """Configuration d'entraînement"""
    # Dataset
    train_data_path: str
    validation_data_path: Optional[str] = None
    test_data_path: Optional[str] = None
    
    # Hyperparamètres d'entraînement
    batch_size: int = Field(default=32, ge=1, le=1024)
    learning_rate: float = Field(default=0.001, gt=0.0, le=1.0)
    epochs: int = Field(default=100, ge=1, le=10000)
    
    # Optimisation
    optimizer: str = Field(default="adam")
    loss_function: str
    metrics: List[str] = Field(default_factory=list)
    
    # Régularisation
    dropout_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    l1_regularization: Optional[float] = Field(default=None, ge=0.0)
    l2_regularization: Optional[float] = Field(default=None, ge=0.0)
    
    # Early stopping
    early_stopping: bool = Field(default=True)
    patience: int = Field(default=10, ge=1)
    min_delta: float = Field(default=0.001, ge=0.0)
    
    # Validation
    validation_split: Optional[float] = Field(default=0.2, ge=0.0, le=1.0)
    cross_validation_folds: Optional[int] = Field(default=None, ge=2, le=20)
    
    # Checkpointing
    save_checkpoints: bool = Field(default=True)
    checkpoint_frequency: int = Field(default=10, ge=1)
    
    class Config:
        use_enum_values = True


class InferenceConfig(BaseSchema):
    """Configuration d'inférence"""
    # Mode d'inférence
    inference_mode: InferenceMode
    batch_size: int = Field(default=1, ge=1, le=1024)
    
    # Performance
    max_latency_ms: Optional[float] = Field(default=None, gt=0.0)
    min_throughput: Optional[float] = Field(default=None, gt=0.0)
    
    # Preprocessing
    preprocessing_steps: List[str] = Field(default_factory=list)
    postprocessing_steps: List[str] = Field(default_factory=list)
    
    # Cache
    enable_caching: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=3600, ge=0)
    
    # Monitoring
    enable_monitoring: bool = Field(default=True)
    log_predictions: bool = Field(default=False)
    sample_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    
    class Config:
        use_enum_values = True


class DeploymentConfig(BaseSchema):
    """Configuration de déploiement"""
    # Cible de déploiement
    deployment_target: DeploymentTarget
    environment: str = Field(default="production")
    
    # Scaling
    min_replicas: int = Field(default=1, ge=1)
    max_replicas: int = Field(default=10, ge=1)
    auto_scaling: bool = Field(default=True)
    
    # Ressources par réplique
    cpu_request: str = Field(default="500m")
    cpu_limit: str = Field(default="1000m")
    memory_request: str = Field(default="512Mi")
    memory_limit: str = Field(default="1Gi")
    
    # Health checks
    health_check_path: str = Field(default="/health")
    readiness_probe_delay: int = Field(default=30, ge=0)
    liveness_probe_delay: int = Field(default=60, ge=0)
    
    # Réseau
    port: int = Field(default=8080, ge=1, le=65535)
    enable_https: bool = Field(default=True)
    
    # Blue-green deployment
    blue_green_deployment: bool = Field(default=False)
    canary_deployment: bool = Field(default=False)
    canary_percentage: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    
    class Config:
        use_enum_values = True


class FeatureConfig(BaseSchema):
    """Configuration de features"""
    # Feature engineering
    feature_selection: bool = Field(default=True)
    feature_scaling: bool = Field(default=True)
    feature_encoding: bool = Field(default=True)
    
    # Feature store
    use_feature_store: bool = Field(default=True)
    feature_store_url: Optional[StrictUrl] = None
    
    # Feature monitoring
    enable_drift_detection: bool = Field(default=True)
    drift_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    
    class Config:
        use_enum_values = True


class PipelineConfig(BaseSchema):
    """Configuration de pipeline ML"""
    # Pipeline steps
    data_validation: bool = Field(default=True)
    feature_engineering: bool = Field(default=True)
    model_training: bool = Field(default=True)
    model_evaluation: bool = Field(default=True)
    model_deployment: bool = Field(default=True)
    
    # Orchestration
    orchestrator: str = Field(default="kubeflow")
    schedule: Optional[str] = None  # Cron expression
    
    # Retry policy
    max_retries: int = Field(default=3, ge=0)
    retry_delay_seconds: int = Field(default=60, ge=0)
    
    class Config:
        use_enum_values = True


class ExperimentConfig(BaseSchema):
    """Configuration d'expérimentation"""
    # Expérimentation
    experiment_name: str
    run_name: Optional[str] = None
    
    # Hyperparameter tuning
    hyperparameter_tuning: bool = Field(default=False)
    tuning_algorithm: str = Field(default="random_search")
    max_trials: int = Field(default=100, ge=1)
    
    # Tracking
    tracking_uri: Optional[StrictUrl] = None
    artifact_location: Optional[str] = None
    
    # A/B Testing
    ab_testing: bool = Field(default=False)
    control_group_percentage: float = Field(default=50.0, ge=0.0, le=100.0)
    
    class Config:
        use_enum_values = True


# Types utilitaires
ModelID = UUID
ExperimentID = UUID
RunID = UUID
FeatureID = UUID

# Types de métriques
MetricValue = Union[float, int, str, bool]
MetricDict = Dict[str, MetricValue]
HyperparameterDict = Dict[str, Any]

# Types de données
DataArray = Union[np.ndarray, List[Any]]
FeatureVector = List[float]
PredictionResult = Union[float, int, str, List[Any], Dict[str, Any]]

__all__ = [
    # Énumérations principales
    'ModelType', 'FrameworkType', 'ModelStatus', 'TrainingStatus',
    'InferenceMode', 'DeploymentTarget', 'OptimizationObjective',
    'DataType', 'FeatureType', 'ModelArchitecture', 'HyperparameterType',
    'ExperimentStatus', 'ModelFormat',
    
    # Configurations
    'ModelConfig', 'TrainingConfig', 'InferenceConfig', 'DeploymentConfig',
    'FeatureConfig', 'PipelineConfig', 'ExperimentConfig',
    
    # Types utilitaires
    'ModelID', 'ExperimentID', 'RunID', 'FeatureID',
    'MetricValue', 'MetricDict', 'HyperparameterDict',
    'DataArray', 'FeatureVector', 'PredictionResult'
]
