"""
Schémas ML/IA - Spotify AI Agent
Architecture complète de Machine Learning pour analyse prédictive et optimisation intelligente

Ce module fournit:
- Modèles d'IA avancés pour analyse audio
- Pipelines ML industriels et automatisés
- Moteurs de recommandation intelligents
- Détection d'anomalies et prédiction
- Infrastructure MLOps enterprise
- Gouvernance et sécurité ML
"""

from .models import (
    # Modèles de base ML
    BaseMLModel, AudioMLModel, RecommendationModel, AnomalyDetectionModel,
    TimeSeriesModel, NLPModel, EnsembleModel, AutoMLModel,
    
    # Modèles spécialisés
    SpectralAnalysisModel, BeatDetectionModel, GenreClassificationModel,
    MoodAnalysisModel, InstrumentRecognitionModel, StemSeparationModel,
    
    # Recommandations
    CollaborativeFilteringModel, ContentBasedModel, HybridRecommendationModel,
    RealTimeRecommendationModel, ColdStartModel, ContextualRecommendationModel,
    
    # Détection anomalies
    SystemAnomalyDetector, UserBehaviorAnomalyDetector, AudioQualityAnomalyDetector,
    PerformanceAnomalyDetector, SecurityAnomalyDetector
)

from .pipelines import (
    # Pipelines ML
    MLPipeline, DataPipeline, FeaturePipeline, TrainingPipeline,
    InferencePipeline, EvaluationPipeline, DeploymentPipeline,
    
    # Pipelines spécialisés
    AudioProcessingPipeline, RecommendationPipeline, AnomalyDetectionPipeline,
    ModelOptimizationPipeline, AutoMLPipeline, ExperimentPipeline
)

from .features import (
    # Feature engineering
    FeatureExtractor, AudioFeatureExtractor, UserFeatureExtractor,
    ContextFeatureExtractor, TemporalFeatureExtractor,
    
    # Feature store
    FeatureStore, FeatureRegistry, FeatureValidator, FeatureTransformer,
    FeatureMonitor, FeatureLineage
)

from .metrics import (
    # Métriques ML
    MLMetrics, ClassificationMetrics, RegressionMetrics, ClusteringMetrics,
    RecommendationMetrics, AnomalyDetectionMetrics, AudioMetrics,
    
    # Monitoring
    ModelMonitor, DataDriftDetector, ConceptDriftDetector, PerformanceMonitor,
    BiasDetector, FairnessMetrics, ExplainabilityMetrics
)

from .serving import (
    # Serving de modèles
    ModelServer, ModelEndpoint, ModelGateway, ModelCache,
    ModelVersionManager, ModelRouter, ModelScaler,
    
    # Inference
    InferenceEngine, BatchInferenceEngine, RealTimeInferenceEngine,
    StreamingInferenceEngine, EdgeInferenceEngine
)

from .registry import (
    # Registre de modèles
    ModelRegistry, ModelVersioning, ModelMetadata, ModelArtifacts,
    ModelLineage, ModelGovernance, ModelApproval,
    
    # Gestion expérimentations
    ExperimentTracker, ExperimentManager, HyperparameterOptimizer,
    TrialManager, MetricLogger, ArtifactManager
)

from .optimization import (
    # Optimisation de modèles
    HyperparameterOptimizer, AutoMLOptimizer, ArchitectureOptimizer,
    ModelPruner, ModelQuantizer, ModelDistiller,
    
    # Optimisation ressources
    ResourceOptimizer, ComputeOptimizer, MemoryOptimizer,
    LatencyOptimizer, ThroughputOptimizer, CostOptimizer
)

from .security import (
    # Sécurité ML
    ModelSecurity, DataPrivacy, FederatedLearning, DifferentialPrivacy,
    ModelEncryption, AdversarialDefense, InputValidation,
    
    # Gouvernance
    MLGovernance, ModelAudit, ComplianceChecker, BiasAuditor,
    ExplainabilityFramework, FairnessFramework
)

from .types import (
    # Types ML
    ModelType, FrameworkType, ModelStatus, TrainingStatus,
    InferenceMode, DeploymentTarget, OptimizationObjective,
    
    # Configurations
    ModelConfig, TrainingConfig, InferenceConfig, DeploymentConfig,
    FeatureConfig, PipelineConfig, ExperimentConfig
)

from .exceptions import (
    # Exceptions ML
    MLException, ModelException, TrainingException, InferenceException,
    FeatureException, PipelineException, OptimizationException,
    SecurityException, GovernanceException
)

from .utils import (
    # Utilitaires ML
    ModelUtils, DataUtils, FeatureUtils, MetricUtils,
    VisualizationUtils, ExperimentUtils, DeploymentUtils,
    
    # Helpers
    ModelConverter, DataConverter, FormatConverter,
    ModelValidator, DataValidator, ConfigValidator
)

# Configuration par défaut pour le module ML
DEFAULT_ML_CONFIG = {
    'auto_scaling': True,
    'monitoring_enabled': True,
    'security_enabled': True,
    'governance_enabled': True,
    'optimization_enabled': True,
    'explainability_enabled': True,
    'fairness_enabled': True,
    'privacy_enabled': True,
    'caching_enabled': True,
    'logging_enabled': True
}

# Métadonnées du module
__version__ = "2.0.0"
__description__ = "Module ML/IA complet pour Spotify AI Agent"

__all__ = [
    # Modèles ML
    'BaseMLModel', 'AudioMLModel', 'RecommendationModel', 'AnomalyDetectionModel',
    'TimeSeriesModel', 'NLPModel', 'EnsembleModel', 'AutoMLModel',
    
    # Modèles spécialisés
    'SpectralAnalysisModel', 'BeatDetectionModel', 'GenreClassificationModel',
    'MoodAnalysisModel', 'InstrumentRecognitionModel', 'StemSeparationModel',
    
    # Recommandations
    'CollaborativeFilteringModel', 'ContentBasedModel', 'HybridRecommendationModel',
    'RealTimeRecommendationModel', 'ColdStartModel', 'ContextualRecommendationModel',
    
    # Détection anomalies
    'SystemAnomalyDetector', 'UserBehaviorAnomalyDetector', 'AudioQualityAnomalyDetector',
    'PerformanceAnomalyDetector', 'SecurityAnomalyDetector',
    
    # Pipelines
    'MLPipeline', 'DataPipeline', 'FeaturePipeline', 'TrainingPipeline',
    'InferencePipeline', 'EvaluationPipeline', 'DeploymentPipeline',
    'AudioProcessingPipeline', 'RecommendationPipeline', 'AnomalyDetectionPipeline',
    
    # Features
    'FeatureExtractor', 'AudioFeatureExtractor', 'UserFeatureExtractor',
    'FeatureStore', 'FeatureRegistry', 'FeatureValidator', 'FeatureTransformer',
    
    # Métriques et monitoring
    'MLMetrics', 'ClassificationMetrics', 'RegressionMetrics', 'ClusteringMetrics',
    'ModelMonitor', 'DataDriftDetector', 'ConceptDriftDetector', 'PerformanceMonitor',
    
    # Serving
    'ModelServer', 'ModelEndpoint', 'ModelGateway', 'ModelCache',
    'InferenceEngine', 'BatchInferenceEngine', 'RealTimeInferenceEngine',
    
    # Registry et expérimentations
    'ModelRegistry', 'ModelVersioning', 'ModelMetadata', 'ModelArtifacts',
    'ExperimentTracker', 'ExperimentManager', 'HyperparameterOptimizer',
    
    # Optimisation
    'HyperparameterOptimizer', 'AutoMLOptimizer', 'ArchitectureOptimizer',
    'ResourceOptimizer', 'ComputeOptimizer', 'MemoryOptimizer',
    
    # Sécurité et gouvernance
    'ModelSecurity', 'DataPrivacy', 'FederatedLearning', 'DifferentialPrivacy',
    'MLGovernance', 'ModelAudit', 'ComplianceChecker', 'BiasAuditor',
    
    # Types et configurations
    'ModelType', 'FrameworkType', 'ModelStatus', 'TrainingStatus',
    'ModelConfig', 'TrainingConfig', 'InferenceConfig', 'DeploymentConfig',
    
    # Exceptions
    'MLException', 'ModelException', 'TrainingException', 'InferenceException',
    
    # Utilitaires
    'ModelUtils', 'DataUtils', 'FeatureUtils', 'MetricUtils',
    'ModelConverter', 'DataConverter', 'ModelValidator', 'DataValidator',
    
    # Configuration
    'DEFAULT_ML_CONFIG'
]
    EVALUATING = "evaluating"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"
    ARCHIVED = "archived"


class DataDriftStatus(str, Enum):
    """Statuts de dérive des données"""
    STABLE = "stable"
    SLIGHT_DRIFT = "slight_drift"
    MODERATE_DRIFT = "moderate_drift"
    SEVERE_DRIFT = "severe_drift"
    UNKNOWN = "unknown"


class MLModel(BaseSchema):
    """Modèle d'intelligence artificielle"""
    name: StrictStr = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    model_type: MLModelType = Field(...)
    framework: MLFramework = Field(...)
    
    # Version et déploiement
    version: str = Field(..., regex=r"^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$")
    is_production: bool = Field(False)
    deployment_environment: Environment = Field(Environment.DEVELOPMENT)
    
    # Configuration du modèle
    model_config: Dict[str, Any] = Field(default_factory=dict)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    training_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Métriques de performance
    accuracy: Optional[float] = Field(None, ge=0, le=1)
    precision: Optional[float] = Field(None, ge=0, le=1)
    recall: Optional[float] = Field(None, ge=0, le=1)
    f1_score: Optional[float] = Field(None, ge=0, le=1)
    auc_roc: Optional[float] = Field(None, ge=0, le=1)
    mae: Optional[float] = Field(None, ge=0)  # Mean Absolute Error
    mse: Optional[float] = Field(None, ge=0)  # Mean Squared Error
    rmse: Optional[float] = Field(None, ge=0)  # Root Mean Squared Error
    
    # Données d'entraînement
    training_dataset_size: Optional[int] = Field(None, ge=0)
    validation_dataset_size: Optional[int] = Field(None, ge=0)
    test_dataset_size: Optional[int] = Field(None, ge=0)
    feature_count: Optional[int] = Field(None, ge=0)
    
    # Timestamps d'entraînement
    training_started_at: Optional[datetime] = Field(None)
    training_completed_at: Optional[datetime] = Field(None)
    last_evaluation_at: Optional[datetime] = Field(None)
    
    # État et monitoring
    status: ModelStatus = Field(ModelStatus.TRAINING)
    health_score: Optional[float] = Field(None, ge=0, le=100)
    prediction_count: int = Field(0, ge=0)
    average_inference_time_ms: Optional[float] = Field(None, ge=0)
    
    # Dérive des données
    data_drift_status: DataDriftStatus = Field(DataDriftStatus.UNKNOWN)
    last_drift_check_at: Optional[datetime] = Field(None)
    drift_score: Optional[float] = Field(None, ge=0, le=1)
    
    # Artéfacts et chemins
    model_path: Optional[str] = Field(None)
    artifacts_path: Optional[str] = Field(None)
    checkpoint_path: Optional[str] = Field(None)
    
    # Classes prédites (pour classification)
    target_classes: List[str] = Field(default_factory=list)
    
    @computed_field
    @property
    def training_duration_hours(self) -> Optional[float]:
        """Durée d'entraînement en heures"""
        if not self.training_started_at or not self.training_completed_at:
            return None
        delta = self.training_completed_at - self.training_started_at
        return delta.total_seconds() / 3600
    
    @computed_field
    @property
    def is_healthy(self) -> bool:
        """Indique si le modèle est en bonne santé"""
        if self.health_score is None:
            return self.status == ModelStatus.DEPLOYED
        return self.health_score >= 70.0
    
    @computed_field
    @property
    def performance_summary(self) -> Dict[str, Optional[float]]:
        """Résumé des performances"""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'auc_roc': self.auc_roc
        }


class AlertPrediction(BaseSchema):
    """Prédiction d'alerte générée par IA"""
    prediction_id: UUID = Field(default_factory=uuid4)
    model_id: UUID = Field(...)
    
    # Données d'entrée
    input_features: Dict[str, Any] = Field(...)
    raw_data: Optional[Dict[str, Any]] = Field(None)
    
    # Prédiction
    predicted_category: WarningCategory = Field(...)
    predicted_level: AlertLevel = Field(...)
    confidence_score: float = Field(..., ge=0, le=1)
    severity_score: float = Field(..., ge=0, le=1)
    
    # Probabilités par classe (pour classification)
    class_probabilities: Dict[str, float] = Field(default_factory=dict)
    
    # Analyse temporelle
    predicted_occurrence_time: Optional[datetime] = Field(None)
    time_to_occurrence_hours: Optional[float] = Field(None, ge=0)
    
    # Contexte et recommandations
    contributing_factors: List[str] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    similar_historical_alerts: List[UUID] = Field(default_factory=list)
    
    # Métadonnées de prédiction
    prediction_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    inference_time_ms: Optional[float] = Field(None, ge=0)
    model_version: Optional[str] = Field(None)
    
    # Validation et feedback
    actual_outcome: Optional[bool] = Field(None, description="Réalisation effective")
    feedback_score: Optional[float] = Field(None, ge=0, le=1)
    validation_timestamp: Optional[datetime] = Field(None)
    
    @computed_field
    @property
    def risk_level(self) -> str:
        """Niveau de risque basé sur la confiance et sévérité"""
        combined_score = (self.confidence_score + self.severity_score) / 2
        
        if combined_score >= 0.8:
            return "high"
        elif combined_score >= 0.6:
            return "medium"
        elif combined_score >= 0.4:
            return "low"
        else:
            return "very_low"
    
    def add_feedback(self, actual_outcome: bool, feedback_score: Optional[float] = None):
        """Ajoute un feedback sur la prédiction"""
        self.actual_outcome = actual_outcome
        self.feedback_score = feedback_score
        self.validation_timestamp = datetime.now(timezone.utc)


class AnomalyDetectionResult(BaseSchema):
    """Résultat de détection d'anomalie"""
    detection_id: UUID = Field(default_factory=uuid4)
    model_id: UUID = Field(...)
    
    # Données analysées
    data_point: Dict[str, Any] = Field(...)
    timestamp: datetime = Field(...)
    
    # Résultat de détection
    is_anomaly: bool = Field(...)
    anomaly_score: float = Field(..., ge=0, le=1)
    severity: AlertLevel = Field(...)
    
    # Analyse détaillée
    anomalous_features: List[str] = Field(default_factory=list)
    feature_contributions: Dict[str, float] = Field(default_factory=dict)
    baseline_comparison: Dict[str, Any] = Field(default_factory=dict)
    
    # Contexte historique
    historical_pattern_match: Optional[str] = Field(None)
    similar_anomalies: List[UUID] = Field(default_factory=list)
    frequency_analysis: Optional[Dict[str, Any]] = Field(None)
    
    # Actions recommandées
    immediate_actions: List[str] = Field(default_factory=list)
    investigation_steps: List[str] = Field(default_factory=list)
    escalation_recommended: bool = Field(False)
    
    @computed_field
    @property
    def risk_category(self) -> str:
        """Catégorie de risque de l'anomalie"""
        if self.anomaly_score >= 0.9:
            return "critical"
        elif self.anomaly_score >= 0.7:
            return "high"
        elif self.anomaly_score >= 0.5:
            return "medium"
        else:
            return "low"


class PatternAnalysis(BaseSchema):
    """Analyse de patterns dans les alertes"""
    analysis_id: UUID = Field(default_factory=uuid4)
    
    # Période d'analyse
    analysis_period_start: datetime = Field(...)
    analysis_period_end: datetime = Field(...)
    
    # Patterns détectés
    recurring_patterns: List[Dict[str, Any]] = Field(default_factory=list)
    seasonal_patterns: List[Dict[str, Any]] = Field(default_factory=list)
    correlation_patterns: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Insights automatiques
    top_root_causes: List[str] = Field(default_factory=list)
    trend_analysis: Dict[str, Any] = Field(default_factory=dict)
    prediction_confidence: float = Field(0.0, ge=0, le=1)
    
    # Recommandations d'optimisation
    optimization_suggestions: List[str] = Field(default_factory=list)
    prevention_strategies: List[str] = Field(default_factory=list)
    monitoring_improvements: List[str] = Field(default_factory=list)
    
    # Métriques de qualité
    pattern_coverage: float = Field(0.0, ge=0, le=1, description="% d'alertes expliquées")
    false_positive_reduction: Optional[float] = Field(None, ge=0, le=1)
    alert_grouping_efficiency: Optional[float] = Field(None, ge=0, le=1)


class MLPipeline(BaseSchema):
    """Pipeline de machine learning"""
    pipeline_id: UUID = Field(default_factory=uuid4)
    name: StrictStr = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    
    # Configuration du pipeline
    stages: List[Dict[str, Any]] = Field(..., min_items=1)
    input_schema: Dict[str, Any] = Field(...)
    output_schema: Dict[str, Any] = Field(...)
    
    # Modèles inclus
    model_ids: List[UUID] = Field(default_factory=list)
    primary_model_id: Optional[UUID] = Field(None)
    
    # État d'exécution
    status: str = Field("inactive", regex=r"^(inactive|running|completed|failed|paused)$")
    current_stage: Optional[str] = Field(None)
    progress_percentage: float = Field(0.0, ge=0, le=100)
    
    # Métriques d'exécution
    total_executions: int = Field(0, ge=0)
    successful_executions: int = Field(0, ge=0)
    failed_executions: int = Field(0, ge=0)
    average_execution_time_minutes: Optional[float] = Field(None, ge=0)
    
    # Scheduling
    is_scheduled: bool = Field(False)
    schedule_cron: Optional[str] = Field(None)
    next_execution: Optional[datetime] = Field(None)
    last_execution: Optional[datetime] = Field(None)
    
    # Ressources
    resource_requirements: Dict[str, Any] = Field(default_factory=dict)
    allocated_resources: Dict[str, Any] = Field(default_factory=dict)
    
    @computed_field
    @property
    def success_rate(self) -> float:
        """Taux de succès du pipeline"""
        if self.total_executions == 0:
            return 1.0
        return self.successful_executions / self.total_executions


class FeatureStore(BaseSchema):
    """Store de features pour ML"""
    feature_store_id: UUID = Field(default_factory=uuid4)
    name: StrictStr = Field(..., min_length=1, max_length=255)
    
    # Features disponibles
    features: List[Dict[str, Any]] = Field(default_factory=list)
    feature_groups: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Métadonnées des features
    feature_descriptions: Dict[str, str] = Field(default_factory=dict)
    feature_types: Dict[str, str] = Field(default_factory=dict)
    feature_statistics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # Versioning
    schema_version: str = Field("1.0.0")
    feature_versions: Dict[str, str] = Field(default_factory=dict)
    
    # Qualité des données
    data_quality_score: Optional[float] = Field(None, ge=0, le=1)
    missing_values_percentage: Dict[str, float] = Field(default_factory=dict)
    outlier_percentage: Dict[str, float] = Field(default_factory=dict)
    
    # Lineage et traçabilité
    data_sources: List[str] = Field(default_factory=list)
    transformation_pipeline: List[Dict[str, Any]] = Field(default_factory=list)
    
    @computed_field
    @property
    def feature_count(self) -> int:
        """Nombre total de features"""
        return len(self.features)


class ModelMonitoring(BaseSchema):
    """Monitoring de modèle ML en production"""
    monitoring_id: UUID = Field(default_factory=uuid4)
    model_id: UUID = Field(...)
    
    # Métriques de performance en temps réel
    current_accuracy: Optional[float] = Field(None, ge=0, le=1)
    accuracy_trend: Optional[str] = Field(None, regex=r"^(improving|stable|degrading)$")
    latency_p50_ms: Optional[float] = Field(None, ge=0)
    latency_p95_ms: Optional[float] = Field(None, ge=0)
    latency_p99_ms: Optional[float] = Field(None, ge=0)
    
    # Utilisation des ressources
    cpu_usage_percent: Optional[float] = Field(None, ge=0, le=100)
    memory_usage_mb: Optional[float] = Field(None, ge=0)
    gpu_usage_percent: Optional[float] = Field(None, ge=0, le=100)
    
    # Métriques business
    predictions_per_minute: Optional[float] = Field(None, ge=0)
    error_rate: Optional[float] = Field(None, ge=0, le=1)
    
    # Alertes de dérive
    feature_drift_alerts: List[Dict[str, Any]] = Field(default_factory=list)
    concept_drift_alerts: List[Dict[str, Any]] = Field(default_factory=list)
    data_quality_alerts: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Seuils d'alerte
    accuracy_threshold: float = Field(0.7, ge=0, le=1)
    latency_threshold_ms: float = Field(1000.0, ge=0)
    error_rate_threshold: float = Field(0.1, ge=0, le=1)
    
    # État du monitoring
    is_active: bool = Field(True)
    last_check_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    next_check_timestamp: Optional[datetime] = Field(None)
    
    @computed_field
    @property
    def overall_health_score(self) -> float:
        """Score de santé global du modèle"""
        score = 100.0
        
        # Pénalité pour accuracy faible
        if self.current_accuracy and self.current_accuracy < self.accuracy_threshold:
            score -= 30 * (1 - self.current_accuracy / self.accuracy_threshold)
        
        # Pénalité pour latence élevée
        if self.latency_p95_ms and self.latency_p95_ms > self.latency_threshold_ms:
            score -= 25 * (self.latency_p95_ms / self.latency_threshold_ms - 1)
        
        # Pénalité pour taux d'erreur élevé
        if self.error_rate and self.error_rate > self.error_rate_threshold:
            score -= 25 * (self.error_rate / self.error_rate_threshold)
        
        # Pénalité pour alertes de dérive
        total_drift_alerts = (
            len(self.feature_drift_alerts) + 
            len(self.concept_drift_alerts) + 
            len(self.data_quality_alerts)
        )
        score -= min(total_drift_alerts * 5, 20)
        
        return max(score, 0.0)


class AutoMLExperiment(BaseSchema):
    """Expérience AutoML"""
    experiment_id: UUID = Field(default_factory=uuid4)
    name: StrictStr = Field(..., min_length=1, max_length=255)
    
    # Configuration de l'expérience
    problem_type: MLModelType = Field(...)
    target_metric: str = Field(...)
    optimization_direction: str = Field("maximize", regex=r"^(maximize|minimize)$")
    
    # Espace de recherche
    search_space: Dict[str, Any] = Field(...)
    algorithms_to_try: List[str] = Field(default_factory=list)
    max_trials: int = Field(100, ge=1, le=1000)
    max_duration_hours: int = Field(24, ge=1, le=168)  # Max 1 semaine
    
    # État de l'expérience
    status: str = Field("pending", regex=r"^(pending|running|completed|failed|stopped)$")
    trials_completed: int = Field(0, ge=0)
    best_score: Optional[float] = Field(None)
    best_model_id: Optional[UUID] = Field(None)
    
    # Résultats
    trial_history: List[Dict[str, Any]] = Field(default_factory=list)
    best_hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    feature_importance: Dict[str, float] = Field(default_factory=dict)
    
    # Timing
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    estimated_remaining_time_hours: Optional[float] = Field(None, ge=0)
    
    @computed_field
    @property
    def progress_percentage(self) -> float:
        """Pourcentage de progression"""
        return (self.trials_completed / self.max_trials) * 100
    
    @computed_field
    @property
    def experiment_duration_hours(self) -> Optional[float]:
        """Durée de l'expérience en heures"""
        if not self.started_at:
            return None
        
        end_time = self.completed_at or datetime.now(timezone.utc)
        return (end_time - self.started_at).total_seconds() / 3600
