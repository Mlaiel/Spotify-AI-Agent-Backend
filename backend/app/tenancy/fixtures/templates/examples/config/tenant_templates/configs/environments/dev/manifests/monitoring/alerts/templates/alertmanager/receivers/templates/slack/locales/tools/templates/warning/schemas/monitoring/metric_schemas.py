"""
Advanced Metric Schemas - Industrial Grade Monitoring
=====================================================

Ce module définit des schémas de métriques ultra-avancés pour un monitoring
industriel complet avec support multi-dimensionnel et intelligence artificielle.

Features:
- Multi-dimensional metrics with automatic tagging
- Real-time aggregation and rollups
- ML-based anomaly detection
- Auto-scaling triggers
- Cost optimization metrics
- Security & compliance tracking
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime, timedelta
import json


class MetricType(str, Enum):
    """Types de métriques supportés"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    RATE = "rate"
    PERCENTAGE = "percentage"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    AVAILABILITY = "availability"
    SATURATION = "saturation"
    BUSINESS_KPI = "business_kpi"
    SECURITY_EVENT = "security_event"
    COMPLIANCE_SCORE = "compliance_score"
    ML_INFERENCE = "ml_inference"
    COST_METRIC = "cost_metric"


class MetricSeverity(str, Enum):
    """Niveaux de sévérité pour les métriques"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class MetricCategory(str, Enum):
    """Catégories de métriques"""
    INFRASTRUCTURE = "infrastructure"
    APPLICATION = "application"
    BUSINESS = "business"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    ML_AI = "ml_ai"
    USER_EXPERIENCE = "user_experience"
    FINANCIAL = "financial"
    OPERATIONAL = "operational"


class AggregationMethod(str, Enum):
    """Méthodes d'agrégation"""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    PERCENTILE_50 = "p50"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"
    RATE = "rate"
    INCREASE = "increase"
    STDDEV = "stddev"
    VARIANCE = "variance"


class MetricDimension(BaseModel):
    """Dimension de métrique pour multi-dimensionnalité"""
    name: str = Field(..., description="Nom de la dimension")
    value: str = Field(..., description="Valeur de la dimension")
    cardinality: Optional[int] = Field(None, description="Cardinalité attendue")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "service",
                "value": "spotify-api",
                "cardinality": 50
            }
        }


class MetricThreshold(BaseModel):
    """Seuils pour déclenchement d'alertes"""
    warning: Optional[float] = Field(None, description="Seuil d'avertissement")
    critical: Optional[float] = Field(None, description="Seuil critique")
    operator: str = Field("gt", description="Opérateur de comparaison (gt, lt, eq, gte, lte)")
    duration: str = Field("5m", description="Durée de dépassement")
    
    @validator('operator')
    def validate_operator(cls, v):
        allowed = ['gt', 'lt', 'eq', 'gte', 'lte', 'ne']
        if v not in allowed:
            raise ValueError(f'Operator must be one of {allowed}')
        return v


class MLAnomalyConfig(BaseModel):
    """Configuration pour détection d'anomalies ML"""
    enabled: bool = Field(True, description="Activer la détection ML")
    model_type: str = Field("isolation_forest", description="Type de modèle ML")
    sensitivity: float = Field(0.8, description="Sensibilité de détection (0-1)")
    training_window: str = Field("7d", description="Fenêtre d'entraînement")
    prediction_window: str = Field("1h", description="Fenêtre de prédiction")
    features: List[str] = Field(default_factory=list, description="Features pour ML")
    
    @validator('sensitivity')
    def validate_sensitivity(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Sensitivity must be between 0 and 1')
        return v


class MetricRetention(BaseModel):
    """Configuration de rétention des métriques"""
    raw_retention: str = Field("7d", description="Rétention données brutes")
    aggregated_retention: str = Field("90d", description="Rétention données agrégées")
    compressed_retention: str = Field("2y", description="Rétention données compressées")
    sampling_rate: float = Field(1.0, description="Taux d'échantillonnage")
    
    @validator('sampling_rate')
    def validate_sampling_rate(cls, v):
        if not 0 < v <= 1:
            raise ValueError('Sampling rate must be between 0 and 1')
        return v


class MetricSchema(BaseModel):
    """Schéma principal pour définition de métrique avancée"""
    
    # Identifiants
    name: str = Field(..., description="Nom unique de la métrique")
    display_name: str = Field(..., description="Nom d'affichage")
    description: str = Field(..., description="Description détaillée")
    
    # Type et catégorie
    metric_type: MetricType = Field(..., description="Type de métrique")
    category: MetricCategory = Field(..., description="Catégorie de métrique")
    severity: MetricSeverity = Field(MetricSeverity.MEDIUM, description="Sévérité")
    
    # Dimensions et labels
    dimensions: List[MetricDimension] = Field(default_factory=list, description="Dimensions")
    labels: Dict[str, str] = Field(default_factory=dict, description="Labels statiques")
    
    # Unité et format
    unit: str = Field("", description="Unité de mesure")
    format: str = Field("number", description="Format d'affichage")
    
    # Configuration d'agrégation
    aggregation_method: AggregationMethod = Field(AggregationMethod.AVG, description="Méthode d'agrégation")
    aggregation_interval: str = Field("1m", description="Intervalle d'agrégation")
    
    # Seuils et alertes
    thresholds: Optional[MetricThreshold] = Field(None, description="Seuils d'alerte")
    
    # ML et IA
    ml_config: Optional[MLAnomalyConfig] = Field(None, description="Configuration ML")
    
    # Rétention
    retention: MetricRetention = Field(default_factory=MetricRetention, description="Configuration rétention")
    
    # Métadonnées
    tags: List[str] = Field(default_factory=list, description="Tags pour classification")
    owner: str = Field("", description="Propriétaire de la métrique")
    team: str = Field("", description="Équipe responsable")
    
    # Configuration technique
    collection_interval: str = Field("30s", description="Intervalle de collecte")
    enabled: bool = Field(True, description="Métrique activée")
    
    # SLI/SLO
    is_sli: bool = Field(False, description="Utilisé comme SLI")
    slo_target: Optional[float] = Field(None, description="Objectif SLO")
    
    # Contexte business
    business_impact: str = Field("", description="Impact business")
    cost_center: str = Field("", description="Centre de coût")
    
    # Version et audit
    version: str = Field("1.0.0", description="Version du schéma")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Date de création")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Dernière modification")
    
    class Config:
        use_enum_values = True
        schema_extra = {
            "example": {
                "name": "spotify_api_response_time",
                "display_name": "Spotify API Response Time",
                "description": "Temps de réponse des APIs Spotify avec détection d'anomalies ML",
                "metric_type": "latency",
                "category": "application",
                "severity": "high",
                "dimensions": [
                    {"name": "endpoint", "value": "/api/v1/tracks", "cardinality": 20},
                    {"name": "method", "value": "GET", "cardinality": 5}
                ],
                "unit": "ms",
                "thresholds": {
                    "warning": 500,
                    "critical": 1000,
                    "operator": "gt",
                    "duration": "5m"
                },
                "ml_config": {
                    "enabled": True,
                    "model_type": "isolation_forest",
                    "sensitivity": 0.8
                },
                "is_sli": True,
                "slo_target": 95.0
            }
        }


class MetricCollectionRule(BaseModel):
    """Règle de collecte de métrique"""
    metric_name: str = Field(..., description="Nom de la métrique")
    source_query: str = Field(..., description="Requête source (PromQL, etc.)")
    collection_method: str = Field("pull", description="Méthode de collecte")
    scrape_interval: str = Field("30s", description="Intervalle de scraping")
    timeout: str = Field("10s", description="Timeout de collecte")
    retries: int = Field(3, description="Nombre de tentatives")
    
    # Filtres et transformations
    filters: Dict[str, Any] = Field(default_factory=dict, description="Filtres à appliquer")
    transformations: List[str] = Field(default_factory=list, description="Transformations")
    
    # Conditions de collecte
    conditions: List[str] = Field(default_factory=list, description="Conditions de collecte")
    dependencies: List[str] = Field(default_factory=list, description="Dépendances")


class MetricRegistry(BaseModel):
    """Registre centralisé des métriques"""
    metrics: List[MetricSchema] = Field(default_factory=list, description="Liste des métriques")
    collection_rules: List[MetricCollectionRule] = Field(default_factory=list, description="Règles de collecte")
    global_labels: Dict[str, str] = Field(default_factory=dict, description="Labels globaux")
    
    def add_metric(self, metric: MetricSchema) -> None:
        """Ajouter une métrique au registre"""
        if any(m.name == metric.name for m in self.metrics):
            raise ValueError(f"Metric {metric.name} already exists")
        self.metrics.append(metric)
    
    def get_metric(self, name: str) -> Optional[MetricSchema]:
        """Récupérer une métrique par nom"""
        return next((m for m in self.metrics if m.name == name), None)
    
    def get_metrics_by_category(self, category: MetricCategory) -> List[MetricSchema]:
        """Récupérer les métriques par catégorie"""
        return [m for m in self.metrics if m.category == category]
    
    def get_sli_metrics(self) -> List[MetricSchema]:
        """Récupérer les métriques SLI"""
        return [m for m in self.metrics if m.is_sli]


# Métriques prédéfinies pour Spotify AI Agent
SPOTIFY_CORE_METRICS = [
    MetricSchema(
        name="spotify_track_recommendation_accuracy",
        display_name="Track Recommendation Accuracy",
        description="Précision des recommandations de tracks avec ML",
        metric_type=MetricType.PERCENTAGE,
        category=MetricCategory.ML_AI,
        severity=MetricSeverity.HIGH,
        unit="%",
        thresholds=MetricThreshold(warning=85.0, critical=75.0, operator="lt"),
        ml_config=MLAnomalyConfig(enabled=True, model_type="lstm"),
        is_sli=True,
        slo_target=90.0
    ),
    
    MetricSchema(
        name="spotify_api_request_rate",
        display_name="Spotify API Request Rate",
        description="Taux de requêtes vers les APIs Spotify",
        metric_type=MetricType.RATE,
        category=MetricCategory.APPLICATION,
        severity=MetricSeverity.MEDIUM,
        unit="req/s",
        aggregation_method=AggregationMethod.SUM,
        thresholds=MetricThreshold(warning=1000, critical=1500, operator="gt")
    ),
    
    MetricSchema(
        name="spotify_user_engagement_score",
        display_name="User Engagement Score",
        description="Score d'engagement utilisateur multi-dimensionnel",
        metric_type=MetricType.BUSINESS_KPI,
        category=MetricCategory.BUSINESS,
        severity=MetricSeverity.HIGH,
        unit="score",
        ml_config=MLAnomalyConfig(enabled=True, sensitivity=0.9),
        is_sli=True,
        slo_target=8.5
    ),
    
    MetricSchema(
        name="spotify_security_threat_score",
        display_name="Security Threat Score",
        description="Score de menace sécurité en temps réel",
        metric_type=MetricType.SECURITY_EVENT,
        category=MetricCategory.SECURITY,
        severity=MetricSeverity.CRITICAL,
        unit="score",
        thresholds=MetricThreshold(warning=7.0, critical=9.0, operator="gt", duration="1m")
    ),
    
    MetricSchema(
        name="spotify_infrastructure_cost",
        display_name="Infrastructure Cost",
        description="Coût d'infrastructure en temps réel avec optimisation IA",
        metric_type=MetricType.COST_METRIC,
        category=MetricCategory.FINANCIAL,
        severity=MetricSeverity.MEDIUM,
        unit="USD",
        ml_config=MLAnomalyConfig(enabled=True, model_type="prophet"),
        business_impact="Direct impact on operational costs"
    )
]


def create_default_registry() -> MetricRegistry:
    """Créer un registre avec les métriques par défaut"""
    registry = MetricRegistry(
        global_labels={
            "environment": "production",
            "project": "spotify-ai-agent",
            "region": "global"
        }
    )
    
    for metric in SPOTIFY_CORE_METRICS:
        registry.add_metric(metric)
    
    return registry


# Export des classes principales
__all__ = [
    "MetricType",
    "MetricSeverity", 
    "MetricCategory",
    "AggregationMethod",
    "MetricSchema",
    "MetricDimension",
    "MetricThreshold",
    "MLAnomalyConfig",
    "MetricRetention",
    "MetricCollectionRule",
    "MetricRegistry",
    "SPOTIFY_CORE_METRICS",
    "create_default_registry"
]
