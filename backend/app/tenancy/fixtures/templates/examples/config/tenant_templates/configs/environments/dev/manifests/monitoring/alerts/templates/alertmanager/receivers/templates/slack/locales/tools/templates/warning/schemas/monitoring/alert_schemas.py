"""
Advanced Alert Schemas - Intelligent Industrial Alerting System
==============================================================

Ce module définit des schémas d'alertes ultra-avancés avec intelligence artificielle,
routing intelligent, et intégration multi-canal pour monitoring industriel.

Features:
- ML-based alert correlation and deduplication
- Smart routing with escalation policies
- Multi-channel notifications (Slack, Teams, PagerDuty, SMS)
- Auto-remediation triggers
- Compliance & audit trail
- Cost-aware alerting
"""

from typing import Dict, List, Optional, Union, Any, Callable
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime, timedelta
import json
from .metric_schemas import MetricSeverity, MetricCategory


class AlertStatus(str, Enum):
    """Statuts d'alerte"""
    FIRING = "firing"
    PENDING = "pending"
    RESOLVED = "resolved"
    SILENCED = "silenced"
    INHIBITED = "inhibited"
    SUPPRESSED = "suppressed"
    ACKNOWLEDGED = "acknowledged"
    ESCALATED = "escalated"


class AlertPriority(str, Enum):
    """Priorités d'alerte selon matrice RICE"""
    P0_CRITICAL = "p0_critical"      # Service down, revenue impact
    P1_HIGH = "p1_high"              # Performance degraded, user impact
    P2_MEDIUM = "p2_medium"          # Warning thresholds, preventive
    P3_LOW = "p3_low"                # Informational, maintenance
    P4_INFO = "p4_info"              # Metrics, trends, reports


class NotificationChannel(str, Enum):
    """Canaux de notification"""
    SLACK = "slack"
    TEAMS = "teams"
    EMAIL = "email"
    SMS = "sms"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    OPSGENIE = "opsgenie"
    JIRA = "jira"
    SERVICENOW = "servicenow"
    CUSTOM = "custom"


class EscalationPolicy(str, Enum):
    """Politiques d'escalade"""
    IMMEDIATE = "immediate"          # Escalade immédiate
    PROGRESSIVE = "progressive"      # Escalade progressive temporelle
    CONDITIONAL = "conditional"      # Escalade conditionnelle
    ML_BASED = "ml_based"           # Escalade basée sur ML
    BUSINESS_HOURS = "business_hours" # Escalade selon heures ouvrables


class AlertCorrelationMethod(str, Enum):
    """Méthodes de corrélation d'alertes"""
    TIME_WINDOW = "time_window"
    SERVICE_DEPENDENCY = "service_dependency"
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    ML_CLUSTERING = "ml_clustering"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    TOPOLOGY_BASED = "topology_based"


class RemediationAction(str, Enum):
    """Actions de remédiation automatique"""
    RESTART_SERVICE = "restart_service"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    CIRCUIT_BREAKER = "circuit_breaker"
    FAILOVER = "failover"
    CACHE_CLEAR = "cache_clear"
    ROLLBACK = "rollback"
    CUSTOM_SCRIPT = "custom_script"
    NONE = "none"


class NotificationTemplate(BaseModel):
    """Template de notification personnalisable"""
    channel: NotificationChannel = Field(..., description="Canal de notification")
    template_id: str = Field(..., description="ID du template")
    subject_template: str = Field(..., description="Template du sujet")
    body_template: str = Field(..., description="Template du corps")
    format: str = Field("markdown", description="Format (markdown, html, text)")
    
    # Variables contextuelles
    variables: Dict[str, Any] = Field(default_factory=dict, description="Variables du template")
    
    # Configuration canal spécifique
    channel_config: Dict[str, Any] = Field(default_factory=dict, description="Config canal")
    
    class Config:
        schema_extra = {
            "example": {
                "channel": "slack",
                "template_id": "spotify_critical_alert",
                "subject_template": "🚨 CRITICAL: {{alert_name}} - {{service}}",
                "body_template": """
*Alert*: {{alert_name}}
*Service*: {{service}}
*Severity*: {{severity}}
*Description*: {{description}}
*Runbook*: {{runbook_url}}
*Dashboard*: {{dashboard_url}}
""",
                "channel_config": {
                    "channel": "#alerts-critical",
                    "username": "AlertBot",
                    "icon_emoji": ":warning:"
                }
            }
        }


class EscalationRule(BaseModel):
    """Règle d'escalade intelligente"""
    name: str = Field(..., description="Nom de la règle")
    condition: str = Field(..., description="Condition d'escalade")
    delay: str = Field("15m", description="Délai avant escalade")
    
    # Destinataires par niveau
    level_1: List[str] = Field(default_factory=list, description="Escalade niveau 1")
    level_2: List[str] = Field(default_factory=list, description="Escalade niveau 2")
    level_3: List[str] = Field(default_factory=list, description="Escalade niveau 3")
    
    # Configuration avancée
    max_escalations: int = Field(3, description="Nombre max d'escalades")
    business_hours_only: bool = Field(False, description="Escalade heures ouvrables uniquement")
    
    # ML et IA
    use_ml_prediction: bool = Field(False, description="Utiliser prédiction ML")
    correlation_window: str = Field("30m", description="Fenêtre de corrélation")


class CorrelationRule(BaseModel):
    """Règle de corrélation d'alertes ML-based"""
    name: str = Field(..., description="Nom de la règle")
    method: AlertCorrelationMethod = Field(..., description="Méthode de corrélation")
    
    # Configuration temporelle
    time_window: str = Field("10m", description="Fenêtre temporelle")
    correlation_threshold: float = Field(0.8, description="Seuil de corrélation")
    
    # Filtres
    include_labels: Dict[str, str] = Field(default_factory=dict, description="Labels à inclure")
    exclude_labels: Dict[str, str] = Field(default_factory=dict, description="Labels à exclure")
    
    # Actions
    suppress_duplicates: bool = Field(True, description="Supprimer doublons")
    create_incident: bool = Field(False, description="Créer incident")
    
    # ML Configuration
    ml_model_config: Dict[str, Any] = Field(default_factory=dict, description="Config modèle ML")


class AutoRemediationRule(BaseModel):
    """Règle de remédiation automatique"""
    name: str = Field(..., description="Nom de la règle")
    condition: str = Field(..., description="Condition de déclenchement")
    action: RemediationAction = Field(..., description="Action à exécuter")
    
    # Configuration sécurité
    requires_approval: bool = Field(True, description="Nécessite approbation")
    max_attempts: int = Field(3, description="Nombre max tentatives")
    cooldown_period: str = Field("30m", description="Période de refroidissement")
    
    # Paramètres d'action
    action_params: Dict[str, Any] = Field(default_factory=dict, description="Paramètres action")
    
    # Rollback
    rollback_action: Optional[str] = Field(None, description="Action de rollback")
    rollback_timeout: str = Field("5m", description="Timeout rollback")
    
    # Audit et conformité
    audit_trail: bool = Field(True, description="Traçabilité audit")
    compliance_check: bool = Field(True, description="Vérification conformité")


class AlertRule(BaseModel):
    """Règle d'alerte avancée avec IA"""
    
    # Identifiants
    name: str = Field(..., description="Nom unique de la règle")
    display_name: str = Field(..., description="Nom d'affichage")
    description: str = Field(..., description="Description détaillée")
    
    # Configuration métrique
    metric_name: str = Field(..., description="Nom de la métrique")
    query: str = Field(..., description="Requête de métrique (PromQL, etc.)")
    
    # Conditions
    condition: str = Field(..., description="Condition d'alerte")
    threshold: float = Field(..., description="Seuil de déclenchement")
    comparison_operator: str = Field("gt", description="Opérateur de comparaison")
    
    # Temporisation
    evaluation_interval: str = Field("1m", description="Intervalle d'évaluation")
    for_duration: str = Field("5m", description="Durée de dépassement")
    
    # Priorité et sévérité
    priority: AlertPriority = Field(..., description="Priorité de l'alerte")
    severity: MetricSeverity = Field(..., description="Sévérité")
    category: MetricCategory = Field(..., description="Catégorie")
    
    # Labels et annotations
    labels: Dict[str, str] = Field(default_factory=dict, description="Labels")
    annotations: Dict[str, str] = Field(default_factory=dict, description="Annotations")
    
    # Notifications
    notification_templates: List[NotificationTemplate] = Field(
        default_factory=list, description="Templates de notification"
    )
    
    # Escalade
    escalation_policy: EscalationPolicy = Field(
        EscalationPolicy.PROGRESSIVE, description="Politique d'escalade"
    )
    escalation_rules: List[EscalationRule] = Field(
        default_factory=list, description="Règles d'escalade"
    )
    
    # Corrélation
    correlation_rules: List[CorrelationRule] = Field(
        default_factory=list, description="Règles de corrélation"
    )
    
    # Remédiation automatique
    auto_remediation: Optional[AutoRemediationRule] = Field(
        None, description="Règle de remédiation auto"
    )
    
    # Configuration avancée
    inhibit_rules: List[str] = Field(default_factory=list, description="Règles d'inhibition")
    silence_matchers: List[Dict[str, str]] = Field(
        default_factory=list, description="Matchers de silence"
    )
    
    # ML et IA
    use_ml_prediction: bool = Field(False, description="Utiliser prédiction ML")
    anomaly_detection: bool = Field(False, description="Détection d'anomalies")
    trend_analysis: bool = Field(False, description="Analyse de tendances")
    
    # SLA et business
    sla_impact: bool = Field(False, description="Impact SLA")
    business_impact_score: float = Field(0.0, description="Score impact business")
    cost_impact: Optional[float] = Field(None, description="Impact coût estimé")
    
    # Runbooks et documentation
    runbook_url: Optional[str] = Field(None, description="URL runbook")
    dashboard_url: Optional[str] = Field(None, description="URL dashboard")
    documentation_url: Optional[str] = Field(None, description="URL documentation")
    
    # Métadonnées
    owner: str = Field("", description="Propriétaire")
    team: str = Field("", description="Équipe responsable")
    environment: str = Field("production", description="Environnement")
    
    # État et activation
    enabled: bool = Field(True, description="Règle activée")
    test_mode: bool = Field(False, description="Mode test")
    
    # Audit
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field("1.0.0", description="Version de la règle")
    
    @validator('comparison_operator')
    def validate_operator(cls, v):
        allowed = ['gt', 'lt', 'eq', 'gte', 'lte', 'ne']
        if v not in allowed:
            raise ValueError(f'Operator must be one of {allowed}')
        return v
    
    @validator('business_impact_score')
    def validate_business_impact(cls, v):
        if not 0 <= v <= 10:
            raise ValueError('Business impact score must be between 0 and 10')
        return v
    
    class Config:
        use_enum_values = True
        schema_extra = {
            "example": {
                "name": "spotify_api_high_latency",
                "display_name": "Spotify API High Latency",
                "description": "Alerte pour latence élevée des APIs Spotify avec ML",
                "metric_name": "spotify_api_response_time",
                "query": "avg(spotify_api_response_time) by (endpoint)",
                "condition": "avg(spotify_api_response_time) > 500",
                "threshold": 500,
                "priority": "p1_high",
                "severity": "high",
                "use_ml_prediction": True,
                "anomaly_detection": True,
                "sla_impact": True,
                "business_impact_score": 8.5
            }
        }


class AlertManagerConfig(BaseModel):
    """Configuration globale AlertManager"""
    
    # Configuration générale
    global_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Templates
    templates: List[str] = Field(default_factory=list, description="Fichiers templates")
    
    # Routes et groupement
    route_config: Dict[str, Any] = Field(default_factory=dict)
    group_by: List[str] = Field(default_factory=list)
    group_wait: str = Field("10s")
    group_interval: str = Field("5m")
    repeat_interval: str = Field("12h")
    
    # Receivers
    receivers: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Inhibition
    inhibit_rules: List[Dict[str, Any]] = Field(default_factory=list)
    
    # ML et intelligence
    ml_correlation_enabled: bool = Field(True)
    auto_grouping_enabled: bool = Field(True)
    smart_routing_enabled: bool = Field(True)


class AlertRegistry(BaseModel):
    """Registre centralisé des alertes"""
    
    alert_rules: List[AlertRule] = Field(default_factory=list)
    notification_templates: List[NotificationTemplate] = Field(default_factory=list)
    escalation_rules: List[EscalationRule] = Field(default_factory=list)
    correlation_rules: List[CorrelationRule] = Field(default_factory=list)
    auto_remediation_rules: List[AutoRemediationRule] = Field(default_factory=list)
    
    # Configuration globale
    alertmanager_config: AlertManagerConfig = Field(default_factory=AlertManagerConfig)
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Ajouter une règle d'alerte"""
        if any(r.name == rule.name for r in self.alert_rules):
            raise ValueError(f"Alert rule {rule.name} already exists")
        self.alert_rules.append(rule)
    
    def get_alert_rule(self, name: str) -> Optional[AlertRule]:
        """Récupérer une règle par nom"""
        return next((r for r in self.alert_rules if r.name == name), None)
    
    def get_rules_by_priority(self, priority: AlertPriority) -> List[AlertRule]:
        """Récupérer les règles par priorité"""
        return [r for r in self.alert_rules if r.priority == priority]
    
    def get_critical_rules(self) -> List[AlertRule]:
        """Récupérer les règles critiques"""
        return [r for r in self.alert_rules if r.priority == AlertPriority.P0_CRITICAL]


# Règles d'alerte prédéfinies pour Spotify AI Agent
SPOTIFY_CORE_ALERT_RULES = [
    AlertRule(
        name="spotify_api_critical_latency",
        display_name="Spotify API Critical Latency",
        description="Latence critique des APIs Spotify avec impact SLA",
        metric_name="spotify_api_response_time",
        query="avg(spotify_api_response_time) by (endpoint)",
        condition="avg(spotify_api_response_time) > 1000",
        threshold=1000,
        priority=AlertPriority.P0_CRITICAL,
        severity=MetricSeverity.CRITICAL,
        category=MetricCategory.APPLICATION,
        for_duration="2m",
        sla_impact=True,
        business_impact_score=9.5,
        use_ml_prediction=True,
        anomaly_detection=True,
        labels={
            "service": "spotify-api",
            "impact": "sla",
            "team": "platform"
        },
        annotations={
            "summary": "Spotify API latency is critically high",
            "description": "API response time exceeded 1000ms for more than 2 minutes",
            "runbook": "https://runbooks.spotify.internal/api-latency"
        }
    ),
    
    AlertRule(
        name="spotify_ml_model_accuracy_degradation",
        display_name="ML Model Accuracy Degradation",
        description="Dégradation de précision des modèles ML de recommandation",
        metric_name="spotify_track_recommendation_accuracy",
        query="avg(spotify_track_recommendation_accuracy)",
        condition="avg(spotify_track_recommendation_accuracy) < 80",
        threshold=80,
        comparison_operator="lt",
        priority=AlertPriority.P1_HIGH,
        severity=MetricSeverity.HIGH,
        category=MetricCategory.ML_AI,
        for_duration="10m",
        business_impact_score=8.0,
        use_ml_prediction=True,
        trend_analysis=True,
        labels={
            "service": "ml-recommendations",
            "model_type": "collaborative_filtering",
            "impact": "user_experience"
        }
    ),
    
    AlertRule(
        name="spotify_security_threat_detected",
        display_name="Security Threat Detected",
        description="Menace de sécurité détectée par ML",
        metric_name="spotify_security_threat_score",
        query="max(spotify_security_threat_score)",
        condition="max(spotify_security_threat_score) > 8",
        threshold=8,
        priority=AlertPriority.P0_CRITICAL,
        severity=MetricSeverity.CRITICAL,
        category=MetricCategory.SECURITY,
        for_duration="1m",
        business_impact_score=10.0,
        anomaly_detection=True,
        labels={
            "service": "security-monitoring",
            "threat_type": "ml_detected",
            "compliance": "required"
        }
    ),
    
    AlertRule(
        name="spotify_cost_anomaly_detected",
        display_name="Infrastructure Cost Anomaly",
        description="Anomalie de coût d'infrastructure détectée",
        metric_name="spotify_infrastructure_cost",
        query="rate(spotify_infrastructure_cost[1h])",
        condition="rate(spotify_infrastructure_cost[1h]) > 100",
        threshold=100,
        priority=AlertPriority.P2_MEDIUM,
        severity=MetricSeverity.MEDIUM,
        category=MetricCategory.FINANCIAL,
        for_duration="15m",
        business_impact_score=7.0,
        use_ml_prediction=True,
        trend_analysis=True,
        labels={
            "service": "cost-optimization",
            "type": "anomaly",
            "impact": "financial"
        }
    )
]


def create_default_alert_registry() -> AlertRegistry:
    """Créer un registre avec les règles d'alerte par défaut"""
    registry = AlertRegistry()
    
    # Ajouter les règles prédéfinies
    for rule in SPOTIFY_CORE_ALERT_RULES:
        registry.add_alert_rule(rule)
    
    # Configuration AlertManager par défaut
    registry.alertmanager_config = AlertManagerConfig(
        global_config={
            "smtp_smarthost": "localhost:587",
            "smtp_from": "alertmanager@spotify.internal"
        },
        group_by=["alertname", "cluster", "service"],
        group_wait="10s",
        group_interval="5m",
        repeat_interval="12h",
        ml_correlation_enabled=True,
        auto_grouping_enabled=True,
        smart_routing_enabled=True
    )
    
    return registry


# Export des classes principales
__all__ = [
    "AlertStatus",
    "AlertPriority",
    "NotificationChannel",
    "EscalationPolicy",
    "AlertCorrelationMethod",
    "RemediationAction",
    "NotificationTemplate",
    "EscalationRule",
    "CorrelationRule",
    "AutoRemediationRule",
    "AlertRule",
    "AlertManagerConfig",
    "AlertRegistry",
    "SPOTIFY_CORE_ALERT_RULES",
    "create_default_alert_registry"
]
