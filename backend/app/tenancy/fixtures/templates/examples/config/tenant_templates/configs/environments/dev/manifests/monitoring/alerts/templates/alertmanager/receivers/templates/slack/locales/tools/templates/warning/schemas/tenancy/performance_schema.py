"""
Performance Metrics Schema Module
=================================

Ce module définit les schémas pour les métriques de performance multi-tenant
avec analytics avancés, prédiction ML et optimisation automatique des
ressources dans un environnement cloud-native scalable.
"""

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.networks import HttpUrl


class MetricCategory(str, Enum):
    """Catégories de métriques de performance."""
    SYSTEM = "system"                    # CPU, RAM, Disk, Network
    APPLICATION = "application"          # Response time, throughput, errors
    DATABASE = "database"                # Query time, connections, locks
    NETWORK = "network"                  # Latency, bandwidth, packet loss
    BUSINESS = "business"                # Conversions, revenue, user engagement
    USER_EXPERIENCE = "user_experience"  # Page load, satisfaction scores
    SECURITY = "security"                # Failed logins, intrusions
    COST = "cost"                        # Cloud costs, resource utilization


class MetricUnit(str, Enum):
    """Unités de mesure."""
    PERCENTAGE = "percentage"
    MILLISECONDS = "milliseconds"
    SECONDS = "seconds"
    BYTES = "bytes"
    KILOBYTES = "kilobytes"
    MEGABYTES = "megabytes"
    GIGABYTES = "gigabytes"
    COUNT = "count"
    RATE_PER_SECOND = "rate_per_second"
    CURRENCY_USD = "currency_usd"
    CURRENCY_EUR = "currency_eur"


class TrendDirection(str, Enum):
    """Direction de la tendance."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class PerformanceStatus(str, Enum):
    """État de performance."""
    EXCELLENT = "excellent"      # > 95% SLA
    GOOD = "good"                # 90-95% SLA
    ACCEPTABLE = "acceptable"     # 80-90% SLA
    POOR = "poor"                # 70-80% SLA
    CRITICAL = "critical"        # < 70% SLA


class AnomalyType(str, Enum):
    """Types d'anomalies détectées."""
    SPIKE = "spike"              # Pic soudain
    DROP = "drop"                # Chute soudaine
    TREND_CHANGE = "trend_change" # Changement de tendance
    SEASONAL = "seasonal"        # Variation saisonnière
    OUTLIER = "outlier"          # Valeur aberrante
    PATTERN_BREAK = "pattern_break" # Rupture de pattern


class MetricThreshold(BaseModel):
    """Seuil pour une métrique."""
    threshold_id: str = Field(..., description="ID du seuil")
    name: str = Field(..., description="Nom du seuil")
    operator: str = Field(..., regex="^(gt|gte|lt|lte|eq|between)$")
    warning_value: Optional[float] = Field(None, description="Valeur d'avertissement")
    critical_value: Optional[float] = Field(None, description="Valeur critique")
    duration_minutes: int = Field(5, ge=1, le=1440, description="Durée avant déclenchement")
    enabled: bool = Field(True, description="Seuil activé")
    
    class Config:
        schema_extra = {
            "example": {
                "threshold_id": "cpu_usage_threshold",
                "name": "CPU Usage Threshold",
                "operator": "gt",
                "warning_value": 75.0,
                "critical_value": 90.0,
                "duration_minutes": 5
            }
        }


class PerformanceBaseline(BaseModel):
    """Ligne de base pour une métrique."""
    baseline_id: str = Field(..., description="ID de la baseline")
    metric_name: str = Field(..., description="Nom de la métrique")
    
    # Valeurs statistiques
    mean_value: float = Field(..., description="Valeur moyenne")
    median_value: float = Field(..., description="Valeur médiane")
    std_deviation: float = Field(..., ge=0, description="Écart-type")
    min_value: float = Field(..., description="Valeur minimale")
    max_value: float = Field(..., description="Valeur maximale")
    
    # Percentiles
    p95_value: float = Field(..., description="95e percentile")
    p99_value: float = Field(..., description="99e percentile")
    
    # Période d'observation
    observation_start: datetime = Field(..., description="Début observation")
    observation_end: datetime = Field(..., description="Fin observation")
    sample_count: int = Field(..., ge=1, description="Nombre d'échantillons")
    
    # Contexte
    conditions: Dict[str, Any] = Field(default_factory=dict, description="Conditions d'observation")
    tags: Dict[str, str] = Field(default_factory=dict, description="Tags")
    
    # Validité
    confidence_level: float = Field(..., ge=0, le=100, description="Niveau de confiance %")
    valid_until: datetime = Field(..., description="Valide jusqu'à")
    
    class Config:
        schema_extra = {
            "example": {
                "baseline_id": "cpu_usage_baseline_q4_2024",
                "metric_name": "cpu_usage_percent",
                "mean_value": 45.2,
                "median_value": 42.1,
                "std_deviation": 12.8,
                "p95_value": 68.5,
                "p99_value": 82.3,
                "confidence_level": 95.0
            }
        }


class AnomalyDetection(BaseModel):
    """Détection d'anomalie."""
    anomaly_id: str = Field(default_factory=lambda: str(uuid4()))
    metric_name: str = Field(..., description="Nom de la métrique")
    detection_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Type et sévérité
    anomaly_type: AnomalyType = Field(..., description="Type d'anomalie")
    severity_score: float = Field(..., ge=0, le=100, description="Score de sévérité")
    confidence_score: float = Field(..., ge=0, le=100, description="Score de confiance")
    
    # Valeurs
    observed_value: float = Field(..., description="Valeur observée")
    expected_value: float = Field(..., description="Valeur attendue")
    deviation_percentage: float = Field(..., description="Déviation en %")
    
    # Contexte
    baseline_id: Optional[str] = Field(None, description="ID baseline de référence")
    contributing_factors: List[str] = Field(default_factory=list, description="Facteurs contributifs")
    
    # ML et prédiction
    model_used: Optional[str] = Field(None, description="Modèle ML utilisé")
    feature_importance: Dict[str, float] = Field(default_factory=dict, description="Importance des features")
    
    # État
    acknowledged: bool = Field(False, description="Anomalie reconnue")
    false_positive: Optional[bool] = Field(None, description="Faux positif")
    
    class Config:
        schema_extra = {
            "example": {
                "metric_name": "response_time_ms",
                "anomaly_type": "spike",
                "severity_score": 85.2,
                "confidence_score": 92.1,
                "observed_value": 2500.0,
                "expected_value": 150.0,
                "deviation_percentage": 1566.7
            }
        }


class PerformanceTrend(BaseModel):
    """Tendance de performance."""
    trend_id: str = Field(default_factory=lambda: str(uuid4()))
    metric_name: str = Field(..., description="Nom de la métrique")
    
    # Période d'analyse
    start_date: datetime = Field(..., description="Date de début")
    end_date: datetime = Field(..., description="Date de fin")
    
    # Tendance
    direction: TrendDirection = Field(..., description="Direction de la tendance")
    slope: float = Field(..., description="Pente de la tendance")
    correlation_coefficient: float = Field(..., ge=-1, le=1, description="Coefficient de corrélation")
    
    # Prédiction
    projected_value_7d: Optional[float] = Field(None, description="Valeur prédite 7j")
    projected_value_30d: Optional[float] = Field(None, description="Valeur prédite 30j")
    prediction_confidence: Optional[float] = Field(None, ge=0, le=100, description="Confiance prédiction")
    
    # Analyse
    seasonal_component: bool = Field(False, description="Composante saisonnière détectée")
    change_points: List[datetime] = Field(default_factory=list, description="Points de changement")
    
    class Config:
        schema_extra = {
            "example": {
                "metric_name": "database_response_time",
                "start_date": "2024-01-01T00:00:00Z",
                "end_date": "2024-01-31T23:59:59Z",
                "direction": "increasing",
                "slope": 0.15,
                "correlation_coefficient": 0.78,
                "projected_value_7d": 125.5,
                "prediction_confidence": 87.3
            }
        }


class PerformanceOptimization(BaseModel):
    """Recommandation d'optimisation."""
    optimization_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = Field(..., description="Titre de l'optimisation")
    description: str = Field(..., description="Description détaillée")
    
    # Contexte
    affected_metrics: List[str] = Field(..., description="Métriques affectées")
    category: MetricCategory = Field(..., description="Catégorie")
    priority: str = Field(..., regex="^(low|medium|high|critical)$")
    
    # Estimation d'impact
    estimated_improvement: Dict[str, float] = Field(..., description="Amélioration estimée")
    implementation_effort: str = Field(..., regex="^(low|medium|high)$")
    cost_impact: Optional[float] = Field(None, description="Impact coût")
    
    # Actions
    recommended_actions: List[str] = Field(..., description="Actions recommandées")
    automation_possible: bool = Field(False, description="Automatisation possible")
    
    # Validation
    confidence_score: float = Field(..., ge=0, le=100, description="Score de confiance")
    historical_effectiveness: Optional[float] = Field(None, ge=0, le=100, description="Efficacité historique")
    
    class Config:
        schema_extra = {
            "example": {
                "title": "Database Query Optimization",
                "description": "Optimize slow-running database queries",
                "affected_metrics": ["database_response_time", "query_duration"],
                "category": "database",
                "priority": "high",
                "estimated_improvement": {
                    "response_time_reduction": 35.0,
                    "throughput_increase": 20.0
                },
                "implementation_effort": "medium",
                "confidence_score": 89.5
            }
        }


class SLAMetrics(BaseModel):
    """Métriques SLA."""
    sla_id: str = Field(..., description="ID SLA")
    name: str = Field(..., description="Nom du SLA")
    target_value: float = Field(..., description="Valeur cible")
    current_value: float = Field(..., description="Valeur actuelle")
    
    # Performance
    achievement_percentage: float = Field(..., ge=0, le=100, description="Taux d'atteinte %")
    status: PerformanceStatus = Field(..., description="État de performance")
    
    # Historique
    rolling_7d_average: Optional[float] = Field(None, description="Moyenne 7j glissants")
    rolling_30d_average: Optional[float] = Field(None, description="Moyenne 30j glissants")
    
    # Violations
    violations_count: int = Field(0, ge=0, description="Nombre de violations")
    last_violation: Optional[datetime] = Field(None, description="Dernière violation")
    
    # Impact
    business_impact: Optional[str] = Field(None, description="Impact business")
    penalty_amount: Optional[float] = Field(None, ge=0, description="Montant pénalité")
    
    class Config:
        schema_extra = {
            "example": {
                "sla_id": "api_response_time_sla",
                "name": "API Response Time SLA",
                "target_value": 200.0,
                "current_value": 185.2,
                "achievement_percentage": 95.8,
                "status": "good",
                "violations_count": 2
            }
        }


class PerformanceMetricsSchema(BaseModel):
    """
    Schéma principal des métriques de performance avec analytics avancés,
    ML intégré et optimisation automatique pour environnement multi-tenant.
    """
    # Identifiants
    metrics_id: str = Field(default_factory=lambda: str(uuid4()))
    tenant_id: str = Field(..., description="ID du tenant")
    name: str = Field(..., description="Nom du set de métriques")
    description: Optional[str] = Field(None, description="Description")
    
    # Configuration temporelle
    collection_interval_seconds: int = Field(60, ge=5, le=3600, description="Intervalle collecte")
    retention_days: int = Field(90, ge=1, le=365, description="Rétention données")
    aggregation_intervals: List[str] = Field(
        default_factory=lambda: ["1m", "5m", "1h", "1d"],
        description="Intervalles d'agrégation"
    )
    
    # Métriques système
    system_metrics: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Métriques système (CPU, RAM, etc.)"
    )
    
    # Métriques application
    application_metrics: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Métriques applicatives"
    )
    
    # Métriques business
    business_metrics: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Métriques business"
    )
    
    # Seuils et alertes
    thresholds: List[MetricThreshold] = Field(
        default_factory=list,
        description="Seuils configurés"
    )
    
    # Baselines et références
    baselines: List[PerformanceBaseline] = Field(
        default_factory=list,
        description="Lignes de base"
    )
    
    # Détection d'anomalies
    anomaly_detection_enabled: bool = Field(True, description="Détection d'anomalies activée")
    anomalies: List[AnomalyDetection] = Field(
        default_factory=list,
        description="Anomalies détectées"
    )
    ml_models_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration modèles ML"
    )
    
    # Tendances et prédictions
    trends: List[PerformanceTrend] = Field(
        default_factory=list,
        description="Tendances analysées"
    )
    trend_analysis_enabled: bool = Field(True, description="Analyse de tendances activée")
    
    # Optimisations
    optimizations: List[PerformanceOptimization] = Field(
        default_factory=list,
        description="Recommandations d'optimisation"
    )
    auto_optimization_enabled: bool = Field(False, description="Optimisation automatique")
    
    # SLA et objectifs
    sla_metrics: List[SLAMetrics] = Field(
        default_factory=list,
        description="Métriques SLA"
    )
    performance_targets: Dict[str, float] = Field(
        default_factory=dict,
        description="Objectifs de performance"
    )
    
    # Reporting et dashboards
    reporting_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration reporting"
    )
    dashboard_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration dashboards"
    )
    
    # Intégrations
    external_metrics_sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sources métriques externes"
    )
    export_endpoints: List[HttpUrl] = Field(
        default_factory=list,
        description="Endpoints d'export"
    )
    
    # Performance globale
    overall_performance_score: float = Field(0, ge=0, le=100, description="Score performance global")
    health_status: PerformanceStatus = Field(PerformanceStatus.GOOD, description="État de santé")
    
    # Métadonnées
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None
    last_analysis: Optional[datetime] = None
    
    # Tags et classification
    tags: Dict[str, str] = Field(default_factory=dict, description="Tags")
    environment: str = Field("production", description="Environnement")
    
    class Config:
        use_enum_values = True
        validate_assignment = True
        extra = "forbid"
        schema_extra = {
            "example": {
                "tenant_id": "enterprise_001",
                "name": "Production Performance Metrics",
                "description": "Complete performance monitoring for production environment",
                "collection_interval_seconds": 30,
                "retention_days": 90,
                "anomaly_detection_enabled": True,
                "trend_analysis_enabled": True,
                "overall_performance_score": 94.2,
                "health_status": "excellent",
                "system_metrics": {
                    "cpu_usage_percent": {
                        "current": 45.2,
                        "average_24h": 42.8,
                        "peak_24h": 78.5
                    }
                }
            }
        }
    
    @validator('updated_at', always=True)
    def set_updated_at(cls, v):
        """Met à jour automatiquement le timestamp."""
        return v or datetime.now(timezone.utc)
    
    @validator('overall_performance_score')
    def validate_performance_score(cls, v):
        """Valide le score de performance."""
        if not 0 <= v <= 100:
            raise ValueError("Performance score must be between 0 and 100")
        return v
    
    @validator('collection_interval_seconds')
    def validate_collection_interval(cls, v):
        """Valide l'intervalle de collecte."""
        if v < 5:
            raise ValueError("Collection interval must be at least 5 seconds")
        return v
    
    @root_validator
    def validate_sla_targets_consistency(cls, values):
        """Valide la cohérence entre SLA et objectifs."""
        sla_metrics = values.get('sla_metrics', [])
        targets = values.get('performance_targets', {})
        
        for sla in sla_metrics:
            metric_name = sla.name.lower().replace(' ', '_')
            if metric_name in targets and targets[metric_name] != sla.target_value:
                raise ValueError(f"Inconsistent targets for {metric_name}")
        
        return values
    
    def calculate_overall_score(self) -> float:
        """Calcule le score de performance global."""
        if not self.sla_metrics:
            return 0.0
        
        total_weight = 0
        weighted_score = 0
        
        for sla in self.sla_metrics:
            # Poids basé sur l'impact business
            weight = 3 if sla.business_impact else 1
            total_weight += weight
            weighted_score += weight * sla.achievement_percentage
        
        return (weighted_score / total_weight) if total_weight > 0 else 0.0
    
    def get_critical_anomalies(self) -> List[AnomalyDetection]:
        """Retourne les anomalies critiques."""
        return [a for a in self.anomalies if a.severity_score >= 80]
    
    def get_sla_violations(self) -> List[SLAMetrics]:
        """Retourne les SLA en violation."""
        return [s for s in self.sla_metrics 
                if s.achievement_percentage < 90 or s.status in [PerformanceStatus.POOR, PerformanceStatus.CRITICAL]]
    
    def get_optimization_priorities(self) -> List[PerformanceOptimization]:
        """Retourne les optimisations prioritaires."""
        return sorted([o for o in self.optimizations if o.priority in ["high", "critical"]],
                     key=lambda x: (x.priority == "critical", x.confidence_score),
                     reverse=True)
    
    def generate_performance_summary(self) -> Dict[str, Any]:
        """Génère un résumé de performance."""
        return {
            "tenant_id": self.tenant_id,
            "report_timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_score": self.overall_performance_score,
            "health_status": self.health_status.value,
            "sla_summary": {
                "total_slas": len(self.sla_metrics),
                "meeting_targets": len([s for s in self.sla_metrics if s.achievement_percentage >= 95]),
                "violations": len(self.get_sla_violations()),
                "average_achievement": sum(s.achievement_percentage for s in self.sla_metrics) / len(self.sla_metrics) if self.sla_metrics else 0
            },
            "anomalies_summary": {
                "total": len(self.anomalies),
                "critical": len(self.get_critical_anomalies()),
                "last_24h": len([a for a in self.anomalies 
                               if a.detection_timestamp > datetime.now(timezone.utc) - timedelta(days=1)])
            },
            "optimization_summary": {
                "total_recommendations": len(self.optimizations),
                "high_priority": len([o for o in self.optimizations if o.priority == "high"]),
                "critical_priority": len([o for o in self.optimizations if o.priority == "critical"]),
                "automation_candidates": len([o for o in self.optimizations if o.automation_possible])
            },
            "trends_summary": {
                "improving_metrics": len([t for t in self.trends if t.direction == TrendDirection.DECREASING and "response_time" in t.metric_name.lower()]),
                "degrading_metrics": len([t for t in self.trends if t.direction == TrendDirection.INCREASING and "response_time" in t.metric_name.lower()]),
                "stable_metrics": len([t for t in self.trends if t.direction == TrendDirection.STABLE])
            }
        }
    
    def predict_future_performance(self, days_ahead: int = 7) -> Dict[str, Any]:
        """Prédit la performance future basée sur les tendances."""
        predictions = {}
        
        for trend in self.trends:
            if trend.direction != TrendDirection.STABLE and trend.prediction_confidence and trend.prediction_confidence > 70:
                if days_ahead <= 7 and trend.projected_value_7d:
                    predictions[trend.metric_name] = {
                        "predicted_value": trend.projected_value_7d,
                        "confidence": trend.prediction_confidence,
                        "trend_direction": trend.direction.value
                    }
                elif days_ahead <= 30 and trend.projected_value_30d:
                    predictions[trend.metric_name] = {
                        "predicted_value": trend.projected_value_30d,
                        "confidence": trend.prediction_confidence,
                        "trend_direction": trend.direction.value
                    }
        
        return {
            "prediction_date": datetime.now(timezone.utc).isoformat(),
            "days_ahead": days_ahead,
            "predictions": predictions,
            "risk_assessment": self._assess_future_risks(predictions)
        }
    
    def _assess_future_risks(self, predictions: Dict[str, Any]) -> List[str]:
        """Évalue les risques futurs basés sur les prédictions."""
        risks = []
        
        for metric_name, prediction in predictions.items():
            if prediction["trend_direction"] == "increasing" and "response_time" in metric_name.lower():
                if prediction["confidence"] > 80:
                    risks.append(f"High risk of performance degradation in {metric_name}")
            elif prediction["trend_direction"] == "increasing" and "error" in metric_name.lower():
                if prediction["confidence"] > 75:
                    risks.append(f"Increasing error rate predicted for {metric_name}")
        
        return risks
