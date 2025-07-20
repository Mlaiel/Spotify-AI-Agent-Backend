"""
Machine Learning Schemas - Ultra-Advanced Edition
===============================================

Schémas ultra-avancés pour les modèles ML, prédictions, training et métriques
avec support pour TensorFlow, PyTorch, Hugging Face et frameworks custom.

Features:
- Support multi-frameworks (TF, PyTorch, HF, Scikit-learn)
- Validation des hyperparamètres
- Métriques de performance ML avancées
- Gestion des modèles versionnés
- A/B testing et déploiement canary
- Monitoring de drift et performance
"""

from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Literal, Set, Tuple
from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import UUID4, PositiveInt, NonNegativeFloat, PositiveFloat
import numpy as np
from dataclasses import dataclass


class MLFramework(str, Enum):
    """Frameworks ML supportés."""
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    HUGGING_FACE = "hugging_face"
    SCIKIT_LEARN = "scikit_learn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    ONNX = "onnx"
    CUSTOM = "custom"


class MLModelType(str, Enum):
    """Types de modèles ML."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    RECOMMENDATION = "recommendation"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    TIME_SERIES = "time_series"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    GENERATIVE = "generative"
    ANOMALY_DETECTION = "anomaly_detection"


class MLModelStatus(str, Enum):
    """Statuts des modèles ML."""
    TRAINING = "training"
    VALIDATING = "validating"
    TESTING = "testing"
    READY = "ready"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"
    ARCHIVED = "archived"


class TrainingPhase(str, Enum):
    """Phases d'entraînement."""
    DATA_PREPARATION = "data_preparation"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    VALIDATION = "validation"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"


class MLDataset(BaseModel):
    """Dataset pour l'entraînement ML."""
    
    dataset_id: UUID4 = Field(default_factory=lambda: UUID4())
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    
    # Métadonnées du dataset
    version: str = Field(..., description="Version du dataset")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Caractéristiques des données
    total_samples: PositiveInt = Field(..., description="Nombre total d'échantillons")
    feature_count: PositiveInt = Field(..., description="Nombre de features")
    target_columns: List[str] = Field(..., description="Colonnes cibles")
    feature_columns: List[str] = Field(..., description="Colonnes de features")
    
    # Splits de données
    train_samples: PositiveInt = Field(..., description="Échantillons d'entraînement")
    validation_samples: PositiveInt = Field(..., description="Échantillons de validation")
    test_samples: PositiveInt = Field(..., description="Échantillons de test")
    
    # Qualité des données
    missing_values_percent: NonNegativeFloat = Field(default=0.0, le=100.0)
    duplicate_rows_percent: NonNegativeFloat = Field(default=0.0, le=100.0)
    outliers_percent: NonNegativeFloat = Field(default=0.0, le=100.0)
    
    # Métadonnées techniques
    file_size_mb: PositiveFloat = Field(..., description="Taille du fichier en MB")
    data_format: str = Field(..., description="Format des données (CSV, Parquet, etc.)")
    storage_location: str = Field(..., description="Emplacement de stockage")
    
    # Checksums et validation
    checksum: str = Field(..., description="Checksum MD5/SHA256 du dataset")
    schema_version: str = Field(default="1.0.0", description="Version du schéma")
    
    @validator('train_samples', 'validation_samples', 'test_samples')
    def validate_sample_distribution(cls, v, values):
        if 'total_samples' in values:
            total = values.get('train_samples', 0) + values.get('validation_samples', 0) + v
            if total > values['total_samples']:
                raise ValueError("Sum of splits cannot exceed total samples")
        return v


class MLHyperparameters(BaseModel):
    """Hyperparamètres pour l'entraînement ML."""
    
    # Paramètres généraux
    learning_rate: PositiveFloat = Field(default=0.001, le=1.0)
    batch_size: PositiveInt = Field(default=32)
    epochs: PositiveInt = Field(default=100, le=10000)
    
    # Régularisation
    l1_regularization: NonNegativeFloat = Field(default=0.0, le=1.0)
    l2_regularization: NonNegativeFloat = Field(default=0.0, le=1.0)
    dropout_rate: NonNegativeFloat = Field(default=0.0, le=0.9)
    
    # Architecture (pour réseaux de neurones)
    hidden_layers: List[int] = Field(default_factory=list)
    activation_function: str = Field(default="relu")
    optimizer: str = Field(default="adam")
    
    # Paramètres spécifiques au framework
    framework_params: Dict[str, Any] = Field(default_factory=dict)
    
    # Early stopping
    early_stopping: bool = Field(default=True)
    patience: PositiveInt = Field(default=10)
    monitor_metric: str = Field(default="val_loss")
    
    # Scheduler
    use_scheduler: bool = Field(default=False)
    scheduler_type: Optional[str] = Field(None)
    scheduler_params: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('hidden_layers')
    def validate_hidden_layers(cls, v):
        if len(v) > 20:  # Limite raisonnable
            raise ValueError("Too many hidden layers")
        if any(layer < 1 for layer in v):
            raise ValueError("Hidden layer size must be positive")
        return v


class MLMetrics(BaseModel):
    """Métriques d'évaluation ML avancées."""
    
    # Métriques générales
    loss: float = Field(..., description="Fonction de perte")
    accuracy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Précision")
    
    # Métriques de classification
    precision: Optional[float] = Field(None, ge=0.0, le=1.0)
    recall: Optional[float] = Field(None, ge=0.0, le=1.0)
    f1_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    auc_roc: Optional[float] = Field(None, ge=0.0, le=1.0)
    auc_pr: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Métriques de régression
    mae: Optional[float] = Field(None, ge=0.0, description="Mean Absolute Error")
    mse: Optional[float] = Field(None, ge=0.0, description="Mean Squared Error")
    rmse: Optional[float] = Field(None, ge=0.0, description="Root Mean Squared Error")
    r2_score: Optional[float] = Field(None, le=1.0, description="R² Score")
    
    # Métriques de ranking/recommandation
    ndcg: Optional[float] = Field(None, ge=0.0, le=1.0, description="NDCG Score")
    map_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Mean Average Precision")
    mrr: Optional[float] = Field(None, ge=0.0, le=1.0, description="Mean Reciprocal Rank")
    
    # Métriques personnalisées
    custom_metrics: Dict[str, float] = Field(default_factory=dict)
    
    # Métriques de confiance
    confidence_intervals: Dict[str, Tuple[float, float]] = Field(default_factory=dict)
    statistical_significance: Dict[str, float] = Field(default_factory=dict)
    
    # Métriques de robustesse
    adversarial_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    noise_robustness: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    @validator('custom_metrics')
    def validate_custom_metrics(cls, v):
        for metric_name, value in v.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Custom metric {metric_name} must be numeric")
        return v


class MLModel(BaseModel):
    """Modèle ML ultra-avancé avec versioning et déploiement."""
    
    # Identification
    model_id: UUID4 = Field(default_factory=lambda: UUID4())
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    version: str = Field(..., description="Version du modèle")
    
    # Métadonnées
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: UUID4 = Field(..., description="ID de l'utilisateur créateur")
    
    # Configuration technique
    framework: MLFramework = Field(..., description="Framework utilisé")
    model_type: MLModelType = Field(..., description="Type de modèle")
    status: MLModelStatus = Field(default=MLModelStatus.TRAINING)
    
    # Architecture et paramètres
    architecture: Dict[str, Any] = Field(..., description="Architecture du modèle")
    hyperparameters: MLHyperparameters = Field(..., description="Hyperparamètres")
    
    # Données d'entraînement
    training_dataset: MLDataset = Field(..., description="Dataset d'entraînement")
    validation_dataset: Optional[MLDataset] = Field(None, description="Dataset de validation")
    test_dataset: Optional[MLDataset] = Field(None, description="Dataset de test")
    
    # Métriques de performance
    training_metrics: MLMetrics = Field(..., description="Métriques d'entraînement")
    validation_metrics: Optional[MLMetrics] = Field(None, description="Métriques de validation")
    test_metrics: Optional[MLMetrics] = Field(None, description="Métriques de test")
    
    # Déploiement
    deployment_config: Dict[str, Any] = Field(default_factory=dict)
    inference_url: Optional[str] = Field(None, description="URL d'inférence")
    model_size_mb: Optional[PositiveFloat] = Field(None, description="Taille du modèle en MB")
    
    # Performance en production
    avg_inference_time_ms: Optional[PositiveFloat] = Field(None, description="Temps d'inférence moyen")
    throughput_qps: Optional[PositiveFloat] = Field(None, description="Débit en QPS")
    memory_usage_mb: Optional[PositiveFloat] = Field(None, description="Usage mémoire")
    
    # Monitoring et drift
    data_drift_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    concept_drift_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    performance_drift_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Tags et métadonnées
    tags: List[str] = Field(default_factory=list, description="Tags du modèle")
    business_impact: Optional[str] = Field(None, description="Impact business")
    
    # A/B Testing
    experiment_id: Optional[UUID4] = Field(None, description="ID d'expérience A/B")
    traffic_percentage: Optional[float] = Field(None, ge=0.0, le=100.0)
    
    @validator('version')
    def validate_version(cls, v):
        import re
        if not re.match(r'^\d+\.\d+\.\d+$', v):
            raise ValueError("Version must follow semantic versioning (x.y.z)")
        return v


class MLPrediction(BaseModel):
    """Prédiction ML avec métadonnées complètes."""
    
    # Identification
    prediction_id: UUID4 = Field(default_factory=lambda: UUID4())
    model_id: UUID4 = Field(..., description="ID du modèle utilisé")
    
    # Contexte de la prédiction
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tenant_id: UUID4 = Field(..., description="ID du tenant")
    user_id: Optional[UUID4] = Field(None, description="ID de l'utilisateur")
    session_id: Optional[str] = Field(None, description="ID de session")
    
    # Données d'entrée
    input_features: Dict[str, Any] = Field(..., description="Features d'entrée")
    preprocessed_features: Optional[Dict[str, Any]] = Field(None, description="Features préprocessées")
    
    # Résultats de prédiction
    prediction: Union[float, int, str, List[Any]] = Field(..., description="Prédiction principale")
    prediction_probabilities: Optional[Dict[str, float]] = Field(None, description="Probabilités par classe")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Score de confiance")
    
    # Métriques de qualité
    uncertainty_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    outlier_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Performance
    inference_time_ms: PositiveFloat = Field(..., description="Temps d'inférence en ms")
    preprocessing_time_ms: Optional[PositiveFloat] = Field(None, description="Temps de préprocessing")
    postprocessing_time_ms: Optional[PositiveFloat] = Field(None, description="Temps de postprocessing")
    
    # Contexte technique
    model_version: str = Field(..., description="Version du modèle")
    framework_version: str = Field(..., description="Version du framework")
    
    # Explicabilité (XAI)
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Importance des features")
    shap_values: Optional[Dict[str, float]] = Field(None, description="Valeurs SHAP")
    lime_explanation: Optional[Dict[str, Any]] = Field(None, description="Explication LIME")
    
    # Feedback et validation
    ground_truth: Optional[Union[float, int, str]] = Field(None, description="Vérité terrain")
    feedback_score: Optional[float] = Field(None, ge=0.0, le=5.0, description="Score de feedback utilisateur")
    is_correct: Optional[bool] = Field(None, description="Prédiction correcte")
    
    # Métadonnées business
    business_value: Optional[Decimal] = Field(None, description="Valeur business de la prédiction")
    impact_category: Optional[str] = Field(None, description="Catégorie d'impact")
    
    @validator('prediction_probabilities')
    def validate_probabilities(cls, v):
        if v is not None:
            total = sum(v.values())
            if not 0.99 <= total <= 1.01:  # Tolérance pour erreurs de précision
                raise ValueError("Probabilities must sum to 1.0")
        return v


class MLTraining(BaseModel):
    """Session d'entraînement ML avec suivi complet."""
    
    # Identification
    training_id: UUID4 = Field(default_factory=lambda: UUID4())
    model_id: UUID4 = Field(..., description="ID du modèle")
    experiment_id: Optional[UUID4] = Field(None, description="ID d'expérience")
    
    # Métadonnées
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None)
    status: str = Field(default="running")
    
    # Configuration
    phase: TrainingPhase = Field(..., description="Phase d'entraînement")
    hyperparameters: MLHyperparameters = Field(..., description="Hyperparamètres utilisés")
    
    # Ressources utilisées
    compute_instance_type: str = Field(..., description="Type d'instance de calcul")
    gpu_count: int = Field(default=0, ge=0, description="Nombre de GPUs")
    memory_gb: PositiveFloat = Field(..., description="Mémoire allouée en GB")
    cpu_cores: PositiveInt = Field(..., description="Nombre de cœurs CPU")
    
    # Métriques d'entraînement par époque
    epoch_metrics: List[Dict[str, float]] = Field(default_factory=list)
    best_epoch: Optional[int] = Field(None, description="Meilleure époque")
    
    # Métriques finales
    final_metrics: Optional[MLMetrics] = Field(None, description="Métriques finales")
    
    # Coûts et ressources
    training_cost_usd: Optional[Decimal] = Field(None, ge=0, description="Coût d'entraînement en USD")
    compute_hours: Optional[PositiveFloat] = Field(None, description="Heures de calcul")
    energy_consumption_kwh: Optional[PositiveFloat] = Field(None, description="Consommation énergétique")
    
    # Checkpoints et artifacts
    checkpoint_path: Optional[str] = Field(None, description="Chemin des checkpoints")
    model_artifacts_path: Optional[str] = Field(None, description="Chemin des artifacts")
    tensorboard_logs_path: Optional[str] = Field(None, description="Chemin des logs TensorBoard")
    
    # Optimisation automatique
    auto_hyperparameter_tuning: bool = Field(default=False)
    hyperparameter_search_space: Optional[Dict[str, Any]] = Field(None)
    best_hyperparameters: Optional[Dict[str, Any]] = Field(None)
    
    # Erreurs et warnings
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    @validator('completed_at')
    def validate_completion_time(cls, v, values):
        if v is not None and 'started_at' in values:
            if v <= values['started_at']:
                raise ValueError("Completion time must be after start time")
        return v
    
    @property
    def duration_hours(self) -> Optional[float]:
        """Calcule la durée d'entraînement en heures."""
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds() / 3600
        return None


class MLExperiment(BaseModel):
    """Expérience ML avec A/B testing et comparaisons."""
    
    # Identification
    experiment_id: UUID4 = Field(default_factory=lambda: UUID4())
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    
    # Métadonnées
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(None)
    ended_at: Optional[datetime] = Field(None)
    created_by: UUID4 = Field(..., description="ID de l'utilisateur créateur")
    
    # Configuration de l'expérience
    objective: str = Field(..., description="Objectif de l'expérience")
    hypothesis: str = Field(..., description="Hypothèse testée")
    success_criteria: Dict[str, Any] = Field(..., description="Critères de succès")
    
    # Modèles comparés
    baseline_model_id: UUID4 = Field(..., description="Modèle de référence")
    candidate_model_ids: List[UUID4] = Field(..., description="Modèles candidats")
    
    # Configuration A/B
    traffic_split: Dict[str, float] = Field(..., description="Répartition du trafic")
    control_group_size: float = Field(..., ge=0.0, le=1.0, description="Taille du groupe de contrôle")
    
    # Métriques de comparaison
    primary_metric: str = Field(..., description="Métrique principale")
    secondary_metrics: List[str] = Field(default_factory=list)
    
    # Résultats
    results: Optional[Dict[str, Any]] = Field(None, description="Résultats de l'expérience")
    statistical_significance: Optional[Dict[str, float]] = Field(None)
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99)
    
    # Status et décisions
    status: str = Field(default="planned", description="Statut de l'expérience")
    decision: Optional[str] = Field(None, description="Décision finale")
    winner_model_id: Optional[UUID4] = Field(None, description="Modèle gagnant")
    
    # Métadonnées business
    business_impact: Optional[str] = Field(None, description="Impact business observé")
    estimated_revenue_impact: Optional[Decimal] = Field(None, description="Impact revenus estimé")
    
    @validator('traffic_split')
    def validate_traffic_split(cls, v):
        total = sum(v.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError("Traffic split must sum to 1.0")
        return v


# Export des classes principales
__all__ = [
    "MLFramework",
    "MLModelType",
    "MLModelStatus",
    "TrainingPhase",
    "MLDataset",
    "MLHyperparameters",
    "MLMetrics",
    "MLModel",
    "MLPrediction",
    "MLTraining",
    "MLExperiment"
]
