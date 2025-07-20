"""
Warning Schema Module
====================

Ce module définit les schémas pour le système d'avertissements intelligents
avec classification automatique, prédiction d'escalation et intégration ML
pour la détection proactive d'anomalies dans un environnement multi-tenant.
"""

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.networks import HttpUrl


class WarningSeverity(str, Enum):
    """Niveaux de sévérité des avertissements."""
    URGENT = "urgent"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ADVISORY = "advisory"


class WarningType(str, Enum):
    """Types d'avertissements."""
    THRESHOLD_APPROACHING = "threshold_approaching"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    CAPACITY_WARNING = "capacity_warning"
    SECURITY_CONCERN = "security_concern"
    MAINTENANCE_REQUIRED = "maintenance_required"
    COMPLIANCE_ISSUE = "compliance_issue"
    QUOTA_WARNING = "quota_warning"
    ANOMALY_DETECTED = "anomaly_detected"
    PREDICTIVE_ALERT = "predictive_alert"


class WarningCategory(str, Enum):
    """Catégories d'avertissements."""
    SYSTEM = "system"
    APPLICATION = "application"
    INFRASTRUCTURE = "infrastructure"
    SECURITY = "security"
    BUSINESS = "business"
    USER_EXPERIENCE = "user_experience"
    COMPLIANCE = "compliance"
    FINANCIAL = "financial"


class WarningStatus(str, Enum):
    """États des avertissements."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    MITIGATED = "mitigated"
    RESOLVED = "resolved"
    EXPIRED = "expired"
    SUPPRESSED = "suppressed"


class EscalationProbability(str, Enum):
    """Probabilité d'escalation vers une alerte."""
    VERY_LOW = "very_low"      # < 10%
    LOW = "low"                # 10-25%
    MEDIUM = "medium"          # 25-50%
    HIGH = "high"              # 50-75%
    VERY_HIGH = "very_high"    # > 75%


class WarningSource(str, Enum):
    """Sources des avertissements."""
    MONITORING_SYSTEM = "monitoring_system"
    ML_PREDICTION = "ml_prediction"
    ANOMALY_DETECTION = "anomaly_detection"
    USER_REPORT = "user_report"
    AUTOMATED_CHECK = "automated_check"
    EXTERNAL_FEED = "external_feed"
    SCHEDULED_ANALYSIS = "scheduled_analysis"


class WarningMetric(BaseModel):
    """Métrique associée à un avertissement."""
    metric_name: str = Field(..., description="Nom de la métrique")
    current_value: Union[float, int, str] = Field(..., description="Valeur actuelle")
    baseline_value: Optional[Union[float, int, str]] = Field(None, description="Valeur de référence")
    threshold_value: Optional[Union[float, int, str]] = Field(None, description="Seuil d'alerte")
    unit: Optional[str] = Field(None, description="Unité de mesure")
    trend: Optional[str] = Field(None, regex="^(increasing|decreasing|stable|volatile)$")
    percentage_change: Optional[float] = Field(None, description="Changement en %")
    
    class Config:
        schema_extra = {
            "example": {
                "metric_name": "cpu_usage_percent",
                "current_value": 75.5,
                "baseline_value": 45.0,
                "threshold_value": 80.0,
                "unit": "percent",
                "trend": "increasing",
                "percentage_change": 67.8
            }
        }


class WarningPrediction(BaseModel):
    """Prédiction ML associée à un avertissement."""
    model_name: str = Field(..., description="Nom du modèle ML utilisé")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Score de confiance")
    prediction_window_hours: int = Field(..., ge=1, le=168, description="Fenêtre de prédiction")
    escalation_probability: EscalationProbability = Field(..., description="Probabilité d'escalation")
    predicted_impact: Optional[str] = Field(None, description="Impact prédit")
    recommended_actions: List[str] = Field(default_factory=list, description="Actions recommandées")
    feature_importance: Dict[str, float] = Field(default_factory=dict, description="Importance des features")
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "gradient_boosting_v2",
                "confidence_score": 0.87,
                "prediction_window_hours": 24,
                "escalation_probability": "high",
                "predicted_impact": "Service degradation expected within 24h",
                "recommended_actions": [
                    "Scale up resources",
                    "Review recent deployments",
                    "Check dependency health"
                ]
            }
        }


class WarningMitigation(BaseModel):
    """Actions de mitigation pour un avertissement."""
    action_type: str = Field(..., description="Type d'action")
    description: str = Field(..., description="Description de l'action")
    automated: bool = Field(False, description="Action automatisée")
    executed: bool = Field(False, description="Action exécutée")
    executed_at: Optional[datetime] = None
    success: Optional[bool] = None
    result_description: Optional[str] = None
    rollback_possible: bool = Field(False, description="Rollback possible")
    
    class Config:
        schema_extra = {
            "example": {
                "action_type": "auto_scale",
                "description": "Automatically scale up compute resources",
                "automated": True,
                "executed": True,
                "success": True,
                "result_description": "Successfully scaled from 2 to 4 instances"
            }
        }


class WarningImpact(BaseModel):
    """Impact d'un avertissement."""
    business_impact: Optional[str] = Field(None, description="Impact business")
    technical_impact: Optional[str] = Field(None, description="Impact technique")
    user_impact: Optional[str] = Field(None, description="Impact utilisateur")
    financial_impact: Optional[float] = Field(None, description="Impact financier estimé")
    affected_services: List[str] = Field(default_factory=list, description="Services affectés")
    affected_users: Optional[int] = Field(None, ge=0, description="Nombre d'utilisateurs affectés")
    sla_at_risk: bool = Field(False, description="SLA en risque")
    compliance_risk: bool = Field(False, description="Risque de compliance")


class WarningSchema(BaseModel):
    """
    Schéma principal pour les avertissements avec intelligence artificielle,
    prédiction d'escalation et gestion automatisée des mitigations.
    """
    # Identifiants et métadonnées
    warning_id: str = Field(default_factory=lambda: str(uuid4()), description="ID unique")
    tenant_id: str = Field(..., description="ID du tenant")
    correlation_id: Optional[str] = Field(None, description="ID de corrélation")
    parent_warning_id: Optional[str] = Field(None, description="Avertissement parent")
    
    # Classification
    title: str = Field(..., min_length=5, max_length=200, description="Titre")
    description: str = Field(..., min_length=10, max_length=2000, description="Description détaillée")
    severity: WarningSeverity = Field(..., description="Niveau de sévérité")
    warning_type: WarningType = Field(..., description="Type d'avertissement")
    category: WarningCategory = Field(..., description="Catégorie")
    status: WarningStatus = Field(WarningStatus.ACTIVE, description="État")
    
    # Source et contexte
    source: WarningSource = Field(..., description="Source de l'avertissement")
    source_system: Optional[str] = Field(None, description="Système source")
    environment: str = Field("production", description="Environnement")
    service: Optional[str] = Field(None, description="Service concerné")
    component: Optional[str] = Field(None, description="Composant")
    
    # Timing
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None
    detected_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # Métriques et données
    metrics: List[WarningMetric] = Field(default_factory=list, description="Métriques associées")
    prediction: Optional[WarningPrediction] = Field(None, description="Prédiction ML")
    impact: WarningImpact = Field(default_factory=WarningImpact, description="Analyse d'impact")
    
    # Actions et mitigation
    mitigations: List[WarningMitigation] = Field(default_factory=list, description="Actions de mitigation")
    auto_mitigation_enabled: bool = Field(False, description="Mitigation automatique activée")
    escalation_threshold_hours: int = Field(24, ge=1, le=168, description="Seuil d'escalation (h)")
    
    # Notifications et escalation
    notification_sent: bool = Field(False, description="Notification envoyée")
    escalated_to_alert: bool = Field(False, description="Escaladé en alerte")
    escalated_at: Optional[datetime] = None
    alert_id: Optional[str] = Field(None, description="ID alerte si escaladé")
    
    # Configuration avancée
    suppression_rules: Dict[str, Any] = Field(default_factory=dict, description="Règles de suppression")
    correlation_rules: Dict[str, Any] = Field(default_factory=dict, description="Règles de corrélation")
    learning_enabled: bool = Field(True, description="Apprentissage ML activé")
    
    # Métadonnées et contexte
    tags: Dict[str, str] = Field(default_factory=dict, description="Tags")
    labels: Dict[str, str] = Field(default_factory=dict, description="Labels")
    context: Dict[str, Any] = Field(default_factory=dict, description="Contexte additionnel")
    evidence: List[Dict[str, Any]] = Field(default_factory=list, description="Preuves/Evidence")
    
    # URLs et références
    runbook_url: Optional[HttpUrl] = Field(None, description="URL du runbook")
    dashboard_url: Optional[HttpUrl] = Field(None, description="URL dashboard")
    investigation_url: Optional[HttpUrl] = Field(None, description="URL investigation")
    
    class Config:
        use_enum_values = True
        validate_assignment = True
        extra = "forbid"
        schema_extra = {
            "example": {
                "tenant_id": "enterprise_001",
                "title": "CPU Usage Approaching Threshold",
                "description": "CPU usage has been steadily increasing and is approaching the critical threshold",
                "severity": "medium",
                "warning_type": "threshold_approaching",
                "category": "infrastructure",
                "source": "monitoring_system",
                "metrics": [
                    {
                        "metric_name": "cpu_usage_percent",
                        "current_value": 75.5,
                        "threshold_value": 80.0,
                        "trend": "increasing"
                    }
                ],
                "auto_mitigation_enabled": True
            }
        }
    
    @validator('updated_at', always=True)
    def set_updated_at(cls, v):
        """Met à jour automatiquement le timestamp."""
        return v or datetime.now(timezone.utc)
    
    @validator('expires_at')
    def validate_expiry(cls, v, values):
        """Valide la date d'expiration."""
        if v and 'created_at' in values:
            if v <= values['created_at']:
                raise ValueError("expires_at must be after created_at")
        return v
    
    @root_validator
    def validate_escalation_config(cls, values):
        """Valide la configuration d'escalation."""
        severity = values.get('severity')
        escalation_threshold = values.get('escalation_threshold_hours', 24)
        
        # Avertissements urgents doivent avoir un seuil d'escalation court
        if severity == WarningSeverity.URGENT and escalation_threshold > 4:
            raise ValueError("Urgent warnings must have escalation threshold <= 4 hours")
        
        return values
    
    @root_validator
    def validate_prediction_consistency(cls, values):
        """Valide la cohérence de la prédiction."""
        prediction = values.get('prediction')
        severity = values.get('severity')
        
        if prediction and severity:
            # Cohérence entre sévérité et probabilité d'escalation
            high_severity = severity in [WarningSeverity.URGENT, WarningSeverity.HIGH]
            high_escalation = prediction.escalation_probability in [
                EscalationProbability.HIGH, EscalationProbability.VERY_HIGH
            ]
            
            if high_severity and prediction.escalation_probability == EscalationProbability.VERY_LOW:
                raise ValueError("High severity warnings should not have very low escalation probability")
        
        return values
    
    def calculate_urgency_score(self) -> float:
        """Calcule un score d'urgence pour le tri."""
        severity_weights = {
            WarningSeverity.URGENT: 100,
            WarningSeverity.HIGH: 80,
            WarningSeverity.MEDIUM: 60,
            WarningSeverity.LOW: 40,
            WarningSeverity.ADVISORY: 20
        }
        
        base_score = severity_weights.get(self.severity, 0)
        
        # Bonus pour prédiction d'escalation élevée
        if self.prediction:
            escalation_bonus = {
                EscalationProbability.VERY_HIGH: 20,
                EscalationProbability.HIGH: 15,
                EscalationProbability.MEDIUM: 10,
                EscalationProbability.LOW: 5,
                EscalationProbability.VERY_LOW: 0
            }
            base_score += escalation_bonus.get(self.prediction.escalation_probability, 0)
        
        # Bonus pour impact SLA
        if self.impact.sla_at_risk:
            base_score += 15
        
        # Bonus pour nombre d'utilisateurs affectés
        if self.impact.affected_users:
            user_bonus = min(self.impact.affected_users / 100, 10)  # Max 10 points
            base_score += user_bonus
        
        return base_score
    
    def should_escalate(self) -> bool:
        """Détermine si l'avertissement doit être escaladé."""
        # Vérifier le seuil de temps
        if self.created_at:
            hours_since_creation = (datetime.now(timezone.utc) - self.created_at).total_seconds() / 3600
            if hours_since_creation >= self.escalation_threshold_hours:
                return True
        
        # Vérifier la prédiction ML
        if self.prediction:
            if self.prediction.escalation_probability in [EscalationProbability.HIGH, EscalationProbability.VERY_HIGH]:
                return True
        
        # Vérifier l'impact critique
        if self.impact.sla_at_risk or self.impact.compliance_risk:
            return True
        
        return False
    
    def get_recommended_actions(self) -> List[str]:
        """Retourne les actions recommandées."""
        actions = []
        
        # Actions de la prédiction ML
        if self.prediction:
            actions.extend(self.prediction.recommended_actions)
        
        # Actions basées sur le type d'avertissement
        type_actions = {
            WarningType.THRESHOLD_APPROACHING: [
                "Monitor metric closely",
                "Prepare scaling resources",
                "Review recent changes"
            ],
            WarningType.PERFORMANCE_DEGRADATION: [
                "Analyze performance metrics",
                "Check resource utilization",
                "Review error logs"
            ],
            WarningType.CAPACITY_WARNING: [
                "Plan capacity increase",
                "Analyze usage patterns",
                "Consider auto-scaling"
            ]
        }
        
        actions.extend(type_actions.get(self.warning_type, []))
        
        return list(set(actions))  # Supprimer les doublons
    
    def get_correlation_key(self) -> str:
        """Génère une clé de corrélation pour regrouper les avertissements similaires."""
        key_parts = [
            self.tenant_id,
            self.category.value,
            self.service or "unknown",
            self.component or "unknown"
        ]
        
        # Ajouter les métriques principales
        if self.metrics:
            main_metric = self.metrics[0].metric_name
            key_parts.append(main_metric)
        
        return ":".join(key_parts)


class TenantWarningSchema(WarningSchema):
    """
    Extension du schéma d'avertissement avec fonctionnalités
    spécifiques au multi-tenant et à la personnalisation.
    """
    tenant_config: Dict[str, Any] = Field(default_factory=dict, description="Config tenant")
    custom_thresholds: Dict[str, float] = Field(default_factory=dict, description="Seuils personnalisés")
    tenant_specific_actions: List[str] = Field(default_factory=list, description="Actions spécifiques")
    cost_impact: Optional[float] = Field(None, ge=0, description="Impact coût estimé")
    business_hours_only: bool = Field(False, description="Heures de bureau seulement")
    custom_escalation_rules: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "tenant_id": "enterprise_001",
                "title": "Storage Quota at 85%",
                "description": "Tenant storage usage is approaching the allocated quota limit",
                "severity": "medium",
                "warning_type": "quota_warning",
                "category": "business",
                "tenant_config": {
                    "quota_limit_gb": 1000,
                    "current_usage_gb": 850,
                    "warning_threshold": 0.85
                },
                "cost_impact": 250.0,
                "custom_thresholds": {
                    "warning_level": 0.85,
                    "critical_level": 0.95
                }
            }
        }


class WarningAggregationSchema(BaseModel):
    """Agrégation d'avertissements pour les rapports et dashboards."""
    tenant_id: str = Field(..., description="ID du tenant")
    period_start: datetime = Field(..., description="Début période")
    period_end: datetime = Field(..., description="Fin période")
    total_warnings: int = Field(0, ge=0)
    warnings_by_severity: Dict[WarningSeverity, int] = Field(default_factory=dict)
    warnings_by_type: Dict[WarningType, int] = Field(default_factory=dict)
    warnings_by_category: Dict[WarningCategory, int] = Field(default_factory=dict)
    escalation_rate: float = Field(0, ge=0, le=100, description="Taux d'escalation %")
    false_positive_rate: float = Field(0, ge=0, le=100, description="Taux faux positifs %")
    auto_mitigation_success_rate: float = Field(0, ge=0, le=100, description="Taux succès mitigation auto %")
    average_resolution_time_hours: Optional[float] = Field(None, ge=0)
    top_warning_sources: List[Dict[str, Any]] = Field(default_factory=list)
    prediction_accuracy: Optional[float] = Field(None, ge=0, le=100, description="Précision prédictions %")
    
    class Config:
        schema_extra = {
            "example": {
                "tenant_id": "enterprise_001",
                "period_start": "2024-01-01T00:00:00Z",
                "period_end": "2024-01-31T23:59:59Z",
                "total_warnings": 156,
                "escalation_rate": 12.8,
                "auto_mitigation_success_rate": 87.5,
                "prediction_accuracy": 92.3
            }
        }
