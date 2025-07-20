"""
Advanced ML Monitoring - Industrial Grade AI/ML Observability System
===================================================================

Ce module fournit une architecture de monitoring ML/IA ultra-avancée pour
surveillance complète des modèles, pipelines et performances d'intelligence artificielle.

Features:
- Model performance tracking and drift detection
- Data quality monitoring and validation
- Pipeline observability and lineage tracking
- A/B testing and experimentation monitoring
- Feature store monitoring and validation
- Model explainability and fairness tracking
- Resource utilization and cost optimization
- MLOps pipeline monitoring
"""

from typing import Dict, List, Optional, Union, Any, Tuple
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime, timedelta
import json
from uuid import uuid4


class ModelType(str, Enum):
    """Types de modèles ML"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    RECOMMENDATION = "recommendation"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    TIME_SERIES = "time_series"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    GENERATIVE = "generative"
    ENSEMBLE = "ensemble"
    DEEP_LEARNING = "deep_learning"
    TRANSFORMER = "transformer"


class ModelStage(str, Enum):
    """Étapes du cycle de vie du modèle"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    SHADOW = "shadow"
    CANARY = "canary"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class ModelStatus(str, Enum):
    """Statuts de modèle"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DEGRADED = "degraded"
    FAILING = "failing"
    UNKNOWN = "unknown"


class DriftType(str, Enum):
    """Types de drift"""
    DATA_DRIFT = "data_drift"           # Dérive des données d'entrée
    CONCEPT_DRIFT = "concept_drift"     # Dérive du concept cible
    PREDICTION_DRIFT = "prediction_drift" # Dérive des prédictions
    FEATURE_DRIFT = "feature_drift"     # Dérive des features
    PERFORMANCE_DRIFT = "performance_drift" # Dérive des performances


class DataQualityIssue(str, Enum):
    """Types de problèmes qualité données"""
    MISSING_VALUES = "missing_values"
    OUTLIERS = "outliers"
    DUPLICATES = "duplicates"
    SCHEMA_MISMATCH = "schema_mismatch"
    DISTRIBUTION_SHIFT = "distribution_shift"
    CORRELATION_CHANGE = "correlation_change"
    CARDINALITY_CHANGE = "cardinality_change"
    FRESHNESS_ISSUE = "freshness_issue"


class ExperimentStatus(str, Enum):
    """Statuts d'expérimentation"""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ModelMetrics(BaseModel):
    """Métriques de performance d'un modèle"""
    
    # Identifiants
    model_id: str = Field(..., description="ID du modèle")
    model_version: str = Field(..., description="Version du modèle")
    
    # Métriques de performance génériques
    accuracy: Optional[float] = Field(None, description="Précision")
    precision: Optional[float] = Field(None, description="Précision")
    recall: Optional[float] = Field(None, description="Rappel")
    f1_score: Optional[float] = Field(None, description="Score F1")
    auc_roc: Optional[float] = Field(None, description="AUC-ROC")
    
    # Métriques de régression
    mse: Optional[float] = Field(None, description="Mean Squared Error")
    rmse: Optional[float] = Field(None, description="Root Mean Squared Error")
    mae: Optional[float] = Field(None, description="Mean Absolute Error")
    r2_score: Optional[float] = Field(None, description="R² Score")
    
    # Métriques de recommandation
    ndcg: Optional[float] = Field(None, description="Normalized DCG")
    map_score: Optional[float] = Field(None, description="Mean Average Precision")
    mrr: Optional[float] = Field(None, description="Mean Reciprocal Rank")
    
    # Métriques custom
    custom_metrics: Dict[str, float] = Field(default_factory=dict, description="Métriques personnalisées")
    
    # Métriques business
    business_impact: Optional[float] = Field(None, description="Impact business")
    revenue_impact: Optional[float] = Field(None, description="Impact revenus")
    cost_savings: Optional[float] = Field(None, description="Économies coûts")
    
    # Métriques de latence et throughput
    avg_inference_time_ms: Optional[float] = Field(None, description="Temps inférence moyen (ms)")
    p95_inference_time_ms: Optional[float] = Field(None, description="Temps inférence P95 (ms)")
    throughput_rps: Optional[float] = Field(None, description="Débit (req/s)")
    
    # Fairness et explicabilité
    fairness_score: Optional[float] = Field(None, description="Score équité")
    bias_metrics: Dict[str, float] = Field(default_factory=dict, description="Métriques de biais")
    explainability_score: Optional[float] = Field(None, description="Score explicabilité")
    
    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'r2_score')
    def validate_percentage_metrics(cls, v):
        if v is not None and not 0 <= v <= 1:
            raise ValueError('Metric must be between 0 and 1')
        return v


class DriftDetection(BaseModel):
    """Détection de dérive (drift)"""
    
    # Identifiants
    drift_id: str = Field(default_factory=lambda: str(uuid4()), description="ID drift")
    model_id: str = Field(..., description="ID modèle")
    
    # Type et sévérité
    drift_type: DriftType = Field(..., description="Type de drift")
    severity: str = Field(..., description="Sévérité (low/medium/high/critical)")
    
    # Détection
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    detection_method: str = Field(..., description="Méthode de détection")
    
    # Données statistiques
    drift_score: float = Field(..., description="Score de drift")
    confidence: float = Field(..., description="Confiance détection")
    threshold_used: float = Field(..., description="Seuil utilisé")
    
    # Features affectées
    affected_features: List[str] = Field(default_factory=list, description="Features affectées")
    feature_importance_changes: Dict[str, float] = Field(
        default_factory=dict, description="Changements importance features"
    )
    
    # Comparaison
    reference_period_start: datetime = Field(..., description="Début période référence")
    reference_period_end: datetime = Field(..., description="Fin période référence")
    current_period_start: datetime = Field(..., description="Début période actuelle")
    current_period_end: datetime = Field(..., description="Fin période actuelle")
    
    # Détails statistiques
    statistical_tests: Dict[str, Any] = Field(default_factory=dict, description="Tests statistiques")
    distribution_changes: Dict[str, Any] = Field(default_factory=dict, description="Changements distribution")
    
    # Impact
    predicted_impact: Optional[str] = Field(None, description="Impact prédit")
    recommended_actions: List[str] = Field(default_factory=list, description="Actions recommandées")
    
    # Statut
    acknowledged: bool = Field(False, description="Accusé réception")
    resolved: bool = Field(False, description="Résolu")
    
    @validator('drift_score', 'confidence', 'threshold_used')
    def validate_score_range(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Score must be between 0 and 1')
        return v


class DataQualityCheck(BaseModel):
    """Vérification qualité des données"""
    
    # Identifiants
    check_id: str = Field(default_factory=lambda: str(uuid4()), description="ID vérification")
    dataset_id: str = Field(..., description="ID dataset")
    
    # Configuration
    check_type: DataQualityIssue = Field(..., description="Type de vérification")
    check_name: str = Field(..., description="Nom de la vérification")
    
    # Résultats
    passed: bool = Field(..., description="Vérification passée")
    score: float = Field(..., description="Score qualité")
    
    # Détails
    total_records: int = Field(..., description="Nombre total enregistrements")
    affected_records: int = Field(0, description="Enregistrements affectés")
    affected_columns: List[str] = Field(default_factory=list, description="Colonnes affectées")
    
    # Statistiques
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Statistiques")
    thresholds: Dict[str, float] = Field(default_factory=dict, description="Seuils")
    
    # Détails spécifiques
    details: Dict[str, Any] = Field(default_factory=dict, description="Détails spécifiques")
    
    # Recommandations
    recommendations: List[str] = Field(default_factory=list, description="Recommandations")
    
    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('score')
    def validate_score(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Score must be between 0 and 1')
        return v


class ModelExperiment(BaseModel):
    """Expérimentation de modèle"""
    
    # Identifiants
    experiment_id: str = Field(..., description="ID expérimentation")
    experiment_name: str = Field(..., description="Nom expérimentation")
    model_id: str = Field(..., description="ID modèle")
    
    # Configuration
    experiment_type: str = Field(..., description="Type expérimentation (A/B, shadow, canary)")
    baseline_model_id: str = Field(..., description="ID modèle baseline")
    candidate_model_id: str = Field(..., description="ID modèle candidat")
    
    # Statut
    status: ExperimentStatus = Field(..., description="Statut expérimentation")
    
    # Configuration trafic
    traffic_split: Dict[str, float] = Field(..., description="Répartition trafic")
    target_population: Optional[str] = Field(None, description="Population cible")
    
    # Durée
    start_time: datetime = Field(..., description="Début expérimentation")
    end_time: Optional[datetime] = Field(None, description="Fin expérimentation")
    duration_planned: str = Field(..., description="Durée planifiée")
    
    # Métriques de succès
    success_metrics: List[str] = Field(..., description="Métriques de succès")
    statistical_significance_threshold: float = Field(0.05, description="Seuil significativité")
    
    # Résultats
    baseline_metrics: Dict[str, float] = Field(default_factory=dict, description="Métriques baseline")
    candidate_metrics: Dict[str, float] = Field(default_factory=dict, description="Métriques candidat")
    statistical_results: Dict[str, Any] = Field(default_factory=dict, description="Résultats statistiques")
    
    # Décision
    decision: Optional[str] = Field(None, description="Décision (deploy/rollback/continue)")
    decision_rationale: Optional[str] = Field(None, description="Justification décision")
    
    # Métadonnées
    created_by: str = Field(..., description="Créé par")
    tags: List[str] = Field(default_factory=list, description="Tags")


class FeatureStoreMetrics(BaseModel):
    """Métriques du feature store"""
    
    # Identifiants
    feature_group_id: str = Field(..., description="ID groupe features")
    feature_name: str = Field(..., description="Nom feature")
    
    # Qualité données
    completeness: float = Field(..., description="Complétude")
    uniqueness: float = Field(..., description="Unicité")
    validity: float = Field(..., description="Validité")
    consistency: float = Field(..., description="Cohérence")
    
    # Freshness
    last_updated: datetime = Field(..., description="Dernière mise à jour")
    staleness_hours: float = Field(..., description="Ancienneté (heures)")
    
    # Usage
    consumption_rate: float = Field(..., description="Taux consommation")
    unique_consumers: int = Field(..., description="Consommateurs uniques")
    
    # Performance
    latency_p95_ms: float = Field(..., description="Latence P95 (ms)")
    throughput_rps: float = Field(..., description="Débit (req/s)")
    
    # Drift
    drift_detected: bool = Field(False, description="Drift détecté")
    drift_score: Optional[float] = Field(None, description="Score drift")
    
    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MLModelConfig(BaseModel):
    """Configuration complète d'un modèle ML"""
    
    # Identifiants
    model_id: str = Field(..., description="ID unique du modèle")
    model_name: str = Field(..., description="Nom du modèle")
    model_version: str = Field(..., description="Version du modèle")
    
    # Classification
    model_type: ModelType = Field(..., description="Type de modèle")
    model_stage: ModelStage = Field(..., description="Étape du modèle")
    model_status: ModelStatus = Field(ModelStatus.UNKNOWN, description="Statut du modèle")
    
    # Métadonnées
    description: str = Field("", description="Description")
    use_case: str = Field(..., description="Cas d'usage")
    business_objective: str = Field(..., description="Objectif business")
    
    # Propriétaires
    owner: str = Field(..., description="Propriétaire")
    team: str = Field(..., description="Équipe")
    data_scientist: str = Field(..., description="Data scientist")
    
    # Framework et infrastructure
    framework: str = Field(..., description="Framework ML (TensorFlow, PyTorch, etc.)")
    framework_version: str = Field(..., description="Version framework")
    python_version: str = Field(..., description="Version Python")
    
    # Déploiement
    deployment_platform: str = Field(..., description="Plateforme déploiement")
    serving_infrastructure: str = Field(..., description="Infrastructure serving")
    
    # Features et données
    input_features: List[str] = Field(..., description="Features d'entrée")
    output_schema: Dict[str, str] = Field(..., description="Schéma de sortie")
    training_dataset_id: str = Field(..., description="ID dataset entraînement")
    
    # Configuration monitoring
    monitoring_enabled: bool = Field(True, description="Monitoring activé")
    drift_detection_enabled: bool = Field(True, description="Détection drift activée")
    data_quality_checks_enabled: bool = Field(True, description="Vérifications qualité activées")
    
    # Seuils d'alerte
    performance_thresholds: Dict[str, float] = Field(
        default_factory=dict, description="Seuils performance"
    )
    drift_thresholds: Dict[str, float] = Field(
        default_factory=dict, description="Seuils drift"
    )
    
    # SLA
    latency_sla_ms: float = Field(1000.0, description="SLA latence (ms)")
    throughput_sla_rps: float = Field(100.0, description="SLA débit (req/s)")
    availability_sla: float = Field(99.9, description="SLA disponibilité (%)")
    
    # Retention et archivage
    model_retention_days: int = Field(90, description="Rétention modèle (jours)")
    metrics_retention_days: int = Field(365, description="Rétention métriques (jours)")
    
    # Tags et métadonnées
    tags: List[str] = Field(default_factory=list, description="Tags")
    
    # Audit
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "model_id": "spotify-track-recommender-v2",
                "model_name": "Spotify Track Recommender",
                "model_version": "2.1.0",
                "model_type": "recommendation",
                "model_stage": "production",
                "use_case": "Music recommendation for Spotify users",
                "framework": "TensorFlow",
                "input_features": ["user_id", "listening_history", "demographics"],
                "owner": "Data Science Team"
            }
        }


class MLMonitoringService(BaseModel):
    """Service de monitoring ML ultra-avancé"""
    
    # Configuration
    service_name: str = Field("ml-monitoring", description="Nom du service")
    version: str = Field("1.0.0", description="Version")
    
    # Modèles surveillés
    monitored_models: Dict[str, MLModelConfig] = Field(
        default_factory=dict, description="Modèles surveillés"
    )
    
    # Configuration globale
    global_monitoring_enabled: bool = Field(True, description="Monitoring global activé")
    drift_detection_frequency: str = Field("1h", description="Fréquence détection drift")
    data_quality_check_frequency: str = Field("30m", description="Fréquence vérification qualité")
    
    # Machine Learning Operations
    mlops_integration: bool = Field(True, description="Intégration MLOps")
    auto_retrain_enabled: bool = Field(False, description="Réentraînement auto")
    auto_scaling_enabled: bool = Field(True, description="Auto-scaling activé")
    
    # Alertes et notifications
    alert_channels: List[str] = Field(default_factory=list, description="Canaux d'alerte")
    escalation_enabled: bool = Field(True, description="Escalade activée")
    
    # Feature Store
    feature_store_monitoring: bool = Field(True, description="Monitoring feature store")
    
    # Expérimentation
    experiment_tracking: bool = Field(True, description="Suivi expérimentations")
    
    def add_model(self, model_config: MLModelConfig) -> None:
        """Ajouter un modèle au monitoring"""
        if model_config.model_id in self.monitored_models:
            raise ValueError(f"Model {model_config.model_id} already exists")
        self.monitored_models[model_config.model_id] = model_config
    
    def get_model(self, model_id: str) -> Optional[MLModelConfig]:
        """Récupérer configuration d'un modèle"""
        return self.monitored_models.get(model_id)
    
    def get_production_models(self) -> List[MLModelConfig]:
        """Récupérer modèles en production"""
        return [
            m for m in self.monitored_models.values() 
            if m.model_stage == ModelStage.PRODUCTION
        ]
    
    def get_models_by_type(self, model_type: ModelType) -> List[MLModelConfig]:
        """Récupérer modèles par type"""
        return [
            m for m in self.monitored_models.values() 
            if m.model_type == model_type
        ]


# Modèles prédéfinis pour Spotify AI Agent
SPOTIFY_ML_MODELS = [
    MLModelConfig(
        model_id="spotify_track_recommender",
        model_name="Spotify Track Recommender",
        model_version="3.2.1",
        model_type=ModelType.RECOMMENDATION,
        model_stage=ModelStage.PRODUCTION,
        model_status=ModelStatus.HEALTHY,
        use_case="Personalized music recommendations",
        business_objective="Increase user engagement and listening time",
        owner="ML Platform Team",
        team="Data Science",
        data_scientist="Senior ML Engineer",
        framework="TensorFlow",
        framework_version="2.12.0",
        python_version="3.9.16",
        deployment_platform="Kubernetes",
        serving_infrastructure="TensorFlow Serving",
        input_features=[
            "user_id", "user_listening_history", "user_demographics",
            "track_features", "contextual_features", "temporal_features"
        ],
        output_schema={
            "track_ids": "List[str]",
            "scores": "List[float]",
            "explanations": "List[str]"
        },
        training_dataset_id="spotify_user_interactions_v3",
        performance_thresholds={
            "ndcg": 0.85,
            "map_score": 0.80,
            "engagement_rate": 0.75
        },
        latency_sla_ms=150.0,
        throughput_sla_rps=500.0,
        tags=["recommendation", "personalization", "production"]
    ),
    
    MLModelConfig(
        model_id="spotify_content_moderation",
        model_name="Content Moderation Classifier",
        model_version="1.5.0",
        model_type=ModelType.CLASSIFICATION,
        model_stage=ModelStage.PRODUCTION,
        use_case="Automatic content moderation",
        business_objective="Ensure platform safety and compliance",
        owner="Trust & Safety Team",
        team="ML Engineering",
        framework="PyTorch",
        input_features=["audio_features", "metadata", "user_reports"],
        performance_thresholds={
            "precision": 0.95,
            "recall": 0.90,
            "f1_score": 0.92
        },
        latency_sla_ms=500.0,
        tags=["classification", "safety", "compliance"]
    )
]


def create_default_ml_monitoring_service() -> MLMonitoringService:
    """Créer service de monitoring ML avec modèles par défaut"""
    service = MLMonitoringService(
        alert_channels=["slack", "pagerduty", "email"],
        mlops_integration=True,
        auto_scaling_enabled=True,
        experiment_tracking=True
    )
    
    # Ajouter modèles prédéfinis
    for model in SPOTIFY_ML_MODELS:
        service.add_model(model)
    
    return service


# Export des classes principales
__all__ = [
    "ModelType",
    "ModelStage", 
    "ModelStatus",
    "DriftType",
    "DataQualityIssue",
    "ExperimentStatus",
    "ModelMetrics",
    "DriftDetection",
    "DataQualityCheck",
    "ModelExperiment",
    "FeatureStoreMetrics",
    "MLModelConfig",
    "MLMonitoringService",
    "SPOTIFY_ML_MODELS",
    "create_default_ml_monitoring_service"
]
