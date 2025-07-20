"""
Schémas de modèles ML pour alertes - Spotify AI Agent
Intégration avancée de l'intelligence artificielle et du machine learning
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Union, Tuple
from uuid import UUID, uuid4
from enum import Enum
import json
import numpy as np
from decimal import Decimal

from pydantic import BaseModel, Field, validator, computed_field, ConfigDict

from . import (
    BaseSchema, TimestampMixin, TenantMixin, MetadataMixin,
    AlertLevel, AlertStatus, WarningCategory, Priority, Environment
)


class ModelType(str, Enum):
    """Types de modèles ML"""
    ANOMALY_DETECTION = "anomaly_detection"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TIME_SERIES_FORECASTING = "time_series_forecasting"
    CLUSTERING = "clustering"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    DEEP_LEARNING = "deep_learning"
    ENSEMBLE = "ensemble"
    NATURAL_LANGUAGE_PROCESSING = "nlp"
    COMPUTER_VISION = "computer_vision"


class ModelFramework(str, Enum):
    """Frameworks ML supportés"""
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    SCIKIT_LEARN = "scikit_learn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    PROPHET = "prophet"
    STATSMODELS = "statsmodels"
    HUGGING_FACE = "hugging_face"
    ONNX = "onnx"
    CUSTOM = "custom"


class ModelStatus(str, Enum):
    """États des modèles"""
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATING = "validating"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    FAILED = "failed"
    ARCHIVED = "archived"


class TrainingStatus(str, Enum):
    """États d'entraînement"""
    QUEUED = "queued"
    PREPARING_DATA = "preparing_data"
    TRAINING = "training"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FeatureType(str, Enum):
    """Types de features"""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    TEXT = "text"
    BOOLEAN = "boolean"
    GEOSPATIAL = "geospatial"
    IMAGE = "image"
    AUDIO = "audio"
    DERIVED = "derived"


class ModelMetric(BaseModel):
    """Métrique de modèle ML"""
    
    name: str = Field(..., min_length=1, max_length=100)
    value: Union[float, int, str] = Field(...)
    metric_type: str = Field(..., min_length=1, max_length=50)  # accuracy, precision, recall, f1, auc, etc.
    
    # Contexte
    dataset: str = Field("validation")  # train, validation, test
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Métadonnées
    higher_is_better: bool = Field(True)
    threshold: Optional[float] = Field(None)
    confidence_interval: Optional[Tuple[float, float]] = Field(None)
    
    # Comparaison
    baseline_value: Optional[Union[float, int]] = Field(None)
    improvement: Optional[float] = Field(None)


class FeatureDefinition(BaseModel):
    """Définition d'une feature"""
    
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    feature_type: FeatureType = Field(...)
    
    # Configuration
    source_field: Optional[str] = Field(None, max_length=100)
    transformation: Optional[str] = Field(None, max_length=500)
    aggregation_window: Optional[str] = Field(None)  # 5m, 1h, 1d, etc.
    
    # Validation
    required: bool = Field(True)
    default_value: Optional[Any] = Field(None)
    min_value: Optional[float] = Field(None)
    max_value: Optional[float] = Field(None)
    allowed_values: Optional[List[Any]] = Field(None)
    
    # Qualité des données
    missing_value_strategy: str = Field("fail")  # fail, drop, mean, median, mode, constant
    outlier_detection: bool = Field(False)
    normalization: Optional[str] = Field(None)  # standardize, normalize, min_max
    
    # Importance
    importance_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    correlation_with_target: Optional[float] = Field(None, ge=-1.0, le=1.0)
    
    # Métadonnées
    tags: Set[str] = Field(default_factory=set)


class MLModel(BaseSchema, TimestampMixin, TenantMixin, MetadataMixin):
    """Modèle de machine learning pour alertes"""
    
    # Informations de base
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    version: str = Field("1.0.0")
    
    # Configuration du modèle
    model_type: ModelType = Field(...)
    framework: ModelFramework = Field(...)
    algorithm: str = Field(..., min_length=1, max_length=100)
    
    # Architecture et paramètres
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    architecture: Optional[Dict[str, Any]] = Field(None)
    model_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Features et données
    features: List[FeatureDefinition] = Field(default_factory=list)
    target_variable: str = Field(..., min_length=1, max_length=100)
    feature_engineering_pipeline: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Données d'entraînement
    training_data_source: str = Field(..., min_length=1, max_length=255)
    training_data_size: Optional[int] = Field(None, ge=0)
    training_period_start: Optional[datetime] = Field(None)
    training_period_end: Optional[datetime] = Field(None)
    
    # Division des données
    train_split: float = Field(0.7, ge=0.1, le=0.9)
    validation_split: float = Field(0.15, ge=0.05, le=0.4)
    test_split: float = Field(0.15, ge=0.05, le=0.4)
    
    # État et performance
    status: ModelStatus = Field(ModelStatus.TRAINING)
    training_status: Optional[TrainingStatus] = Field(None)
    
    # Métriques de performance
    metrics: List[ModelMetric] = Field(default_factory=list)
    best_metric_value: Optional[float] = Field(None)
    baseline_metrics: Dict[str, float] = Field(default_factory=dict)
    
    # Déploiement
    deployed_at: Optional[datetime] = Field(None)
    deployment_config: Dict[str, Any] = Field(default_factory=dict)
    serving_endpoint: Optional[str] = Field(None)
    
    # Monitoring et drift
    drift_detection_enabled: bool = Field(True)
    drift_threshold: float = Field(0.1, ge=0.0, le=1.0)
    last_drift_check: Optional[datetime] = Field(None)
    drift_detected: bool = Field(False)
    
    # Retraining
    auto_retrain: bool = Field(False)
    retrain_frequency_days: Optional[int] = Field(None, ge=1, le=365)
    retrain_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    last_retrain: Optional[datetime] = Field(None)
    
    # Fichiers et artefacts
    model_file_path: Optional[str] = Field(None)
    model_size_bytes: Optional[int] = Field(None, ge=0)
    preprocessing_pipeline_path: Optional[str] = Field(None)
    
    # Métadonnées d'entraînement
    training_duration_seconds: Optional[float] = Field(None, ge=0)
    training_epochs: Optional[int] = Field(None, ge=1)
    early_stopping_epoch: Optional[int] = Field(None, ge=1)
    
    # Explicabilité
    interpretable: bool = Field(False)
    feature_importance: Dict[str, float] = Field(default_factory=dict)
    explanation_method: Optional[str] = Field(None)
    
    # Usage et performance en production
    prediction_count: int = Field(0, ge=0)
    avg_prediction_latency_ms: Optional[float] = Field(None, ge=0)
    last_prediction: Optional[datetime] = Field(None)
    
    # Validation et tests
    validation_results: Dict[str, Any] = Field(default_factory=dict)
    a_b_test_id: Optional[UUID] = Field(None)
    champion_challenger_test: bool = Field(False)
    
    # Audit et compliance
    data_lineage: Dict[str, Any] = Field(default_factory=dict)
    bias_assessment: Optional[Dict[str, Any]] = Field(None)
    fairness_metrics: Dict[str, float] = Field(default_factory=dict)
    
    # Tags et organisation
    tags: Set[str] = Field(default_factory=set)
    labels: Dict[str, str] = Field(default_factory=dict)
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )

    @validator('train_split', 'validation_split', 'test_split')
    def validate_data_splits(cls, v, values):
        """Valide que les splits totalisent ~1.0"""
        if 'train_split' in values and 'validation_split' in values:
            total = values['train_split'] + values['validation_split'] + v
            if not (0.95 <= total <= 1.05):  # Tolérance de 5%
                raise ValueError('Data splits must sum to approximately 1.0')
        return v

    @computed_field
    @property
    def total_features(self) -> int:
        """Nombre total de features"""
        return len(self.features)

    @computed_field
    @property
    def model_age_days(self) -> Optional[int]:
        """Âge du modèle en jours"""
        if not self.deployed_at:
            return None
        
        age = datetime.now(timezone.utc) - self.deployed_at
        return age.days

    @computed_field
    @property
    def is_production_ready(self) -> bool:
        """Indique si le modèle est prêt pour la production"""
        return (
            self.status == ModelStatus.VALIDATED and
            self.best_metric_value is not None and
            len(self.metrics) > 0 and
            self.model_file_path is not None
        )

    def add_metric(self, name: str, value: Union[float, int, str], 
                   metric_type: str, dataset: str = "validation"):
        """Ajoute une métrique au modèle"""
        metric = ModelMetric(
            name=name,
            value=value,
            metric_type=metric_type,
            dataset=dataset
        )
        self.metrics.append(metric)
        
        # Mettre à jour la meilleure métrique si applicable
        if isinstance(value, (int, float)) and metric_type in ['accuracy', 'f1', 'auc']:
            if self.best_metric_value is None or value > self.best_metric_value:
                self.best_metric_value = value

    def get_latest_metrics(self, dataset: str = "validation") -> Dict[str, Any]:
        """Obtient les dernières métriques pour un dataset"""
        latest_metrics = {}
        for metric in self.metrics:
            if metric.dataset == dataset:
                if metric.name not in latest_metrics or metric.timestamp > latest_metrics[metric.name]['timestamp']:
                    latest_metrics[metric.name] = {
                        'value': metric.value,
                        'timestamp': metric.timestamp,
                        'type': metric.metric_type
                    }
        return latest_metrics

    def calculate_feature_importance(self) -> Dict[str, float]:
        """Calcule l'importance des features"""
        # Implémentation basique - à étendre selon le framework
        importance = {}
        total_features = len(self.features)
        
        for i, feature in enumerate(self.features):
            if feature.importance_score is not None:
                importance[feature.name] = feature.importance_score
            else:
                # Score par défaut basé sur la corrélation
                if feature.correlation_with_target is not None:
                    importance[feature.name] = abs(feature.correlation_with_target)
                else:
                    importance[feature.name] = 1.0 / total_features
        
        return importance

    def check_drift(self, current_data_stats: Dict[str, Any]) -> bool:
        """Vérifie la dérive du modèle"""
        # Implémentation simplifiée - à étendre avec des méthodes statistiques
        drift_score = 0.0
        
        for feature in self.features:
            if feature.name in current_data_stats:
                # Exemple: comparaison des moyennes (à améliorer)
                current_mean = current_data_stats[feature.name].get('mean', 0)
                historical_mean = self.data_lineage.get(f'{feature.name}_mean', 0)
                
                if historical_mean != 0:
                    relative_change = abs(current_mean - historical_mean) / abs(historical_mean)
                    drift_score += relative_change
        
        # Moyenne des changements relatifs
        if len(self.features) > 0:
            drift_score /= len(self.features)
        
        self.drift_detected = drift_score > self.drift_threshold
        self.last_drift_check = datetime.now(timezone.utc)
        
        return self.drift_detected


class AnomalyDetectionConfig(BaseModel):
    """Configuration pour la détection d'anomalies"""
    
    algorithm: str = Field("isolation_forest")  # isolation_forest, one_class_svm, local_outlier_factor
    contamination: float = Field(0.1, ge=0.001, le=0.5)
    window_size: int = Field(100, ge=10, le=10000)
    
    # Seuils
    anomaly_threshold: float = Field(0.5, ge=0.0, le=1.0)
    severity_thresholds: Dict[str, float] = Field(default_factory=dict)
    
    # Features temporelles
    temporal_features: bool = Field(True)
    seasonal_decomposition: bool = Field(False)
    trend_analysis: bool = Field(False)
    
    # Post-traitement
    smoothing_window: Optional[int] = Field(None, ge=1, le=100)
    min_anomaly_duration: Optional[int] = Field(None, ge=1)  # En secondes


class ModelPrediction(BaseSchema, TimestampMixin, TenantMixin):
    """Prédiction d'un modèle ML"""
    
    prediction_id: UUID = Field(default_factory=uuid4)
    model_id: UUID = Field(...)
    
    # Données d'entrée
    input_features: Dict[str, Any] = Field(...)
    feature_vector: Optional[List[float]] = Field(None)
    
    # Résultats de prédiction
    prediction: Any = Field(...)  # Classification, regression, anomaly score, etc.
    probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Prédictions multiples (pour ensembles)
    class_probabilities: Optional[Dict[str, float]] = Field(None)
    ensemble_predictions: Optional[List[Any]] = Field(None)
    
    # Métriques de qualité
    prediction_uncertainty: Optional[float] = Field(None, ge=0.0)
    feature_importance_scores: Optional[Dict[str, float]] = Field(None)
    
    # Contexte d'exécution
    model_version: str = Field(...)
    prediction_latency_ms: Optional[float] = Field(None, ge=0)
    
    # Validation et feedback
    actual_value: Optional[Any] = Field(None)  # Pour évaluation ultérieure
    feedback_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    human_validated: bool = Field(False)
    
    # Explicabilité
    explanation: Optional[Dict[str, Any]] = Field(None)
    shap_values: Optional[List[float]] = Field(None)
    
    # Métadonnées
    alert_id: Optional[UUID] = Field(None)
    batch_id: Optional[UUID] = Field(None)
    
    @computed_field
    @property
    def is_anomaly(self) -> Optional[bool]:
        """Indique si la prédiction est une anomalie"""
        if isinstance(self.prediction, (int, float)):
            # Pour les scores d'anomalie (plus élevé = plus anormal)
            return self.prediction > 0.5
        elif isinstance(self.prediction, bool):
            return self.prediction
        return None

    @computed_field
    @property
    def prediction_quality(self) -> str:
        """Qualité de la prédiction"""
        if self.confidence is None:
            return "unknown"
        elif self.confidence >= 0.9:
            return "high"
        elif self.confidence >= 0.7:
            return "medium"
        else:
            return "low"


class ModelTrainingJob(BaseSchema, TimestampMixin, TenantMixin):
    """Job d'entraînement de modèle"""
    
    job_id: UUID = Field(default_factory=uuid4)
    model_id: UUID = Field(...)
    job_name: str = Field(..., min_length=1, max_length=255)
    
    # Configuration d'entraînement
    training_config: Dict[str, Any] = Field(...)
    dataset_config: Dict[str, Any] = Field(...)
    
    # État d'exécution
    status: TrainingStatus = Field(TrainingStatus.QUEUED)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    
    # Progression
    current_epoch: Optional[int] = Field(None, ge=0)
    total_epochs: Optional[int] = Field(None, ge=1)
    progress_percentage: float = Field(0.0, ge=0.0, le=100.0)
    
    # Métriques d'entraînement
    training_metrics: List[Dict[str, Any]] = Field(default_factory=list)
    validation_metrics: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Ressources utilisées
    compute_resources: Dict[str, Any] = Field(default_factory=dict)
    estimated_duration_minutes: Optional[int] = Field(None, ge=1)
    actual_duration_minutes: Optional[float] = Field(None, ge=0)
    
    # Résultats
    final_model_path: Optional[str] = Field(None)
    best_checkpoint_path: Optional[str] = Field(None)
    training_logs_path: Optional[str] = Field(None)
    
    # Erreurs et debugging
    error_message: Optional[str] = Field(None)
    error_traceback: Optional[str] = Field(None)
    debug_info: Dict[str, Any] = Field(default_factory=dict)
    
    @computed_field
    @property
    def is_completed(self) -> bool:
        """Indique si l'entraînement est terminé"""
        return self.status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED]

    @computed_field
    @property
    def duration_minutes(self) -> Optional[float]:
        """Durée d'entraînement en minutes"""
        if not self.started_at:
            return None
        
        end_time = self.completed_at or datetime.now(timezone.utc)
        duration = end_time - self.started_at
        return duration.total_seconds() / 60

    def add_training_metric(self, epoch: int, metrics: Dict[str, float], 
                           validation: bool = False):
        """Ajoute des métriques d'entraînement"""
        metric_entry = {
            'epoch': epoch,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'metrics': metrics
        }
        
        if validation:
            self.validation_metrics.append(metric_entry)
        else:
            self.training_metrics.append(metric_entry)
        
        # Mettre à jour la progression
        if self.total_epochs:
            self.progress_percentage = min((epoch / self.total_epochs) * 100, 100.0)
        
        self.current_epoch = epoch


class ModelExperiment(BaseSchema, TimestampMixin, TenantMixin, MetadataMixin):
    """Expérience de modèle ML"""
    
    experiment_id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    
    # Configuration de l'expérience
    objective: str = Field(..., min_length=1, max_length=500)
    hypothesis: Optional[str] = Field(None, max_length=1000)
    
    # Modèles testés
    baseline_model_id: Optional[UUID] = Field(None)
    experimental_models: List[UUID] = Field(default_factory=list)
    
    # Paramètres expérimentaux
    experimental_config: Dict[str, Any] = Field(default_factory=dict)
    hyperparameter_search_space: Dict[str, Any] = Field(default_factory=dict)
    
    # Résultats
    results: Dict[str, Any] = Field(default_factory=dict)
    best_model_id: Optional[UUID] = Field(None)
    best_metrics: Dict[str, float] = Field(default_factory=dict)
    
    # État
    status: str = Field("planning")  # planning, running, completed, failed
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    
    # Conclusions
    conclusions: Optional[str] = Field(None, max_length=2000)
    next_steps: List[str] = Field(default_factory=list)
    
    # Collaboration
    participants: List[UUID] = Field(default_factory=list)
    
    @computed_field
    @property
    def experiment_duration_days(self) -> Optional[int]:
        """Durée de l'expérience en jours"""
        if not self.started_at or not self.completed_at:
            return None
        
        duration = self.completed_at - self.started_at
        return duration.days


__all__ = [
    'ModelType', 'ModelFramework', 'ModelStatus', 'TrainingStatus', 'FeatureType',
    'ModelMetric', 'FeatureDefinition', 'MLModel', 'AnomalyDetectionConfig',
    'ModelPrediction', 'ModelTrainingJob', 'ModelExperiment'
]
