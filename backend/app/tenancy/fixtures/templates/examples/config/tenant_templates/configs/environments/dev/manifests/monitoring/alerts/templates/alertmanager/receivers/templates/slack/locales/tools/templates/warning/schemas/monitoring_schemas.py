"""
Monitoring Schemas - Schémas de Données de Monitoring
====================================================

Schémas Pydantic pour la validation et la sérialisation des données
de monitoring du système Spotify AI Agent.

Features:
    - Validation stricte des données de monitoring
    - Sérialisation JSON optimisée
    - Support type hints complet
    - Validation des métriques business
    - Schémas pour alertes et rapports

Author: Expert Data Architecture + Schema Design Team
"""

from typing import Dict, List, Optional, Union, Any, Literal
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator
from decimal import Decimal
import json

# =============================================================================
# ENUMS ET TYPES DE BASE
# =============================================================================

class AlertSeverity(str, Enum):
    """Niveaux de sévérité des alertes."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class MetricType(str, Enum):
    """Types de métriques."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"


class DataSourceType(str, Enum):
    """Types de sources de données."""
    DATABASE = "database"
    API = "api"
    CACHE = "cache"
    QUEUE = "queue"
    FILE_SYSTEM = "file_system"
    EXTERNAL_SERVICE = "external_service"


class TenantTier(str, Enum):
    """Niveaux de tenant."""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


# =============================================================================
# MODÈLES DE BASE
# =============================================================================

class BaseMetric(BaseModel):
    """Modèle de base pour toutes les métriques."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tenant_id: str = Field(..., min_length=1, max_length=100)
    metric_name: str = Field(..., min_length=1, max_length=200)
    metric_type: MetricType
    value: Union[int, float, str]
    unit: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('tenant_id')
    def validate_tenant_id(cls, v):
        """Valide le format du tenant ID."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Tenant ID must be alphanumeric with - or _')
        return v
    
    @validator('value')
    def validate_value(cls, v, values):
        """Valide la valeur selon le type de métrique."""
        metric_type = values.get('metric_type')
        if metric_type in [MetricType.COUNTER, MetricType.GAUGE]:
            if not isinstance(v, (int, float)):
                raise ValueError(f'Value must be numeric for {metric_type}')
            if metric_type == MetricType.COUNTER and v < 0:
                raise ValueError('Counter values must be non-negative')
        return v


class Alert(BaseModel):
    """Modèle d'alerte de monitoring."""
    alert_id: str = Field(..., min_length=1)
    tenant_id: str
    severity: AlertSeverity
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., max_length=1000)
    source_metric: str
    threshold_value: Union[int, float]
    current_value: Union[int, float]
    triggered_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    is_active: bool = True
    affected_components: List[str] = Field(default_factory=list)
    remediation_steps: List[str] = Field(default_factory=list)
    tags: Dict[str, str] = Field(default_factory=dict)
    
    @validator('resolved_at')
    def validate_resolution(cls, v, values):
        """Valide que la résolution est après le déclenchement."""
        if v and v < values.get('triggered_at', datetime.utcnow()):
            raise ValueError('Resolution time cannot be before trigger time')
        return v


class HealthScore(BaseModel):
    """Score de santé d'un composant."""
    component_name: str
    score: float = Field(..., ge=0, le=100)
    weight: float = Field(default=1.0, ge=0, le=1.0)
    status: Literal["healthy", "warning", "critical", "unknown"] = "unknown"
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    contributing_factors: Dict[str, float] = Field(default_factory=dict)
    
    @validator('score')
    def validate_score_range(cls, v):
        """Valide que le score est dans la plage correcte."""
        return round(v, 2)
    
    @root_validator
    def set_status_from_score(cls, values):
        """Définit le statut basé sur le score."""
        score = values.get('score', 0)
        if score >= 85:
            values['status'] = "healthy"
        elif score >= 70:
            values['status'] = "warning"
        elif score >= 50:
            values['status'] = "critical"
        else:
            values['status'] = "unknown"
        return values


# =============================================================================
# SCHÉMAS DE PERFORMANCE
# =============================================================================

class SystemPerformanceMetrics(BaseModel):
    """Métriques de performance système."""
    cpu_usage_percent: float = Field(..., ge=0, le=100)
    memory_usage_percent: float = Field(..., ge=0, le=100)
    disk_usage_percent: float = Field(..., ge=0, le=100)
    network_io_bytes: Dict[str, int] = Field(default_factory=dict)
    disk_io_bytes: Dict[str, int] = Field(default_factory=dict)
    load_average: Dict[str, float] = Field(default_factory=dict)
    active_connections: int = Field(..., ge=0)
    response_time_ms: float = Field(..., ge=0)
    throughput_requests_per_second: float = Field(..., ge=0)
    error_rate: float = Field(..., ge=0, le=1)
    uptime_seconds: int = Field(..., ge=0)
    
    class Config:
        schema_extra = {
            "example": {
                "cpu_usage_percent": 45.7,
                "memory_usage_percent": 67.2,
                "disk_usage_percent": 23.1,
                "network_io_bytes": {"received": 1234567, "transmitted": 987654},
                "disk_io_bytes": {"read": 2345678, "written": 1876543},
                "load_average": {"1min": 1.2, "5min": 0.9, "15min": 0.7},
                "active_connections": 156,
                "response_time_ms": 89.3,
                "throughput_requests_per_second": 234.5,
                "error_rate": 0.023,
                "uptime_seconds": 86400
            }
        }


class DatabasePerformanceMetrics(BaseModel):
    """Métriques de performance base de données."""
    connection_pool_size: int = Field(..., ge=0)
    active_connections: int = Field(..., ge=0)
    query_execution_time_ms: Dict[str, float] = Field(default_factory=dict)
    slow_queries_count: int = Field(..., ge=0)
    cache_hit_ratio: float = Field(..., ge=0, le=1)
    index_usage_ratio: float = Field(..., ge=0, le=1)
    deadlock_count: int = Field(..., ge=0)
    replication_lag_ms: Optional[float] = Field(None, ge=0)
    disk_space_used_percent: float = Field(..., ge=0, le=100)
    backup_status: Literal["success", "failed", "in_progress", "not_configured"]
    last_backup_timestamp: Optional[datetime] = None
    
    @validator('active_connections')
    def validate_active_vs_pool(cls, v, values):
        """Valide que les connexions actives <= pool size."""
        pool_size = values.get('connection_pool_size', 0)
        if v > pool_size:
            raise ValueError('Active connections cannot exceed pool size')
        return v


class APIPerformanceMetrics(BaseModel):
    """Métriques de performance API."""
    total_requests: int = Field(..., ge=0)
    successful_requests: int = Field(..., ge=0)
    failed_requests: int = Field(..., ge=0)
    average_response_time_ms: float = Field(..., ge=0)
    p95_response_time_ms: float = Field(..., ge=0)
    p99_response_time_ms: float = Field(..., ge=0)
    rate_limit_hits: int = Field(..., ge=0)
    error_breakdown: Dict[str, int] = Field(default_factory=dict)
    endpoint_performance: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    cache_hit_rate: float = Field(..., ge=0, le=1)
    data_transferred_mb: float = Field(..., ge=0)
    
    @validator('successful_requests', 'failed_requests')
    def validate_request_totals(cls, v, values):
        """Valide que la somme des requêtes correspond au total."""
        total = values.get('total_requests', 0)
        if 'successful_requests' in values and 'failed_requests' in values:
            if values['successful_requests'] + values['failed_requests'] != total:
                raise ValueError('Sum of successful and failed requests must equal total')
        return v


# =============================================================================
# SCHÉMAS BUSINESS
# =============================================================================

class TenantMetrics(BaseModel):
    """Métriques spécifiques au tenant."""
    tenant_id: str
    tenant_tier: TenantTier
    active_users_count: int = Field(..., ge=0)
    total_users_count: int = Field(..., ge=0)
    monthly_active_users: int = Field(..., ge=0)
    daily_active_users: int = Field(..., ge=0)
    session_duration_avg_minutes: float = Field(..., ge=0)
    feature_usage_stats: Dict[str, int] = Field(default_factory=dict)
    storage_used_gb: float = Field(..., ge=0)
    storage_limit_gb: float = Field(..., ge=0)
    api_calls_count: int = Field(..., ge=0)
    api_calls_limit: Optional[int] = None
    billing_amount_current_month: Decimal = Field(..., ge=0)
    churn_risk_score: float = Field(..., ge=0, le=1)
    satisfaction_score: float = Field(..., ge=1, le=5)
    
    @validator('monthly_active_users', 'daily_active_users')
    def validate_active_users(cls, v, values):
        """Valide que les utilisateurs actifs <= total."""
        total = values.get('total_users_count', 0)
        if v > total:
            raise ValueError('Active users cannot exceed total users')
        return v
    
    @validator('storage_used_gb')
    def validate_storage_usage(cls, v, values):
        """Valide que l'usage storage <= limite."""
        limit = values.get('storage_limit_gb', float('inf'))
        if v > limit:
            raise ValueError('Storage used cannot exceed limit')
        return v


class BusinessKPIMetrics(BaseModel):
    """Métriques KPI business."""
    monthly_recurring_revenue: Decimal = Field(..., ge=0)
    annual_recurring_revenue: Decimal = Field(..., ge=0)
    customer_acquisition_cost: Decimal = Field(..., ge=0)
    customer_lifetime_value: Decimal = Field(..., ge=0)
    churn_rate_monthly: float = Field(..., ge=0, le=1)
    conversion_rate: float = Field(..., ge=0, le=1)
    net_promoter_score: float = Field(..., ge=-100, le=100)
    user_engagement_score: float = Field(..., ge=0, le=100)
    feature_adoption_rates: Dict[str, float] = Field(default_factory=dict)
    support_ticket_volume: int = Field(..., ge=0)
    support_resolution_time_hours: float = Field(..., ge=0)
    trial_to_paid_conversion: float = Field(..., ge=0, le=1)
    upsell_rate: float = Field(..., ge=0, le=1)
    
    @validator('annual_recurring_revenue')
    def validate_arr_vs_mrr(cls, v, values):
        """Valide la cohérence ARR vs MRR."""
        mrr = values.get('monthly_recurring_revenue', Decimal(0))
        expected_arr = mrr * 12
        if abs(v - expected_arr) > expected_arr * Decimal('0.1'):  # 10% tolerance
            raise ValueError('ARR should be approximately 12x MRR')
        return v


class UserBehaviorMetrics(BaseModel):
    """Métriques de comportement utilisateur."""
    page_views_total: int = Field(..., ge=0)
    unique_page_views: int = Field(..., ge=0)
    session_count: int = Field(..., ge=0)
    bounce_rate: float = Field(..., ge=0, le=1)
    average_session_duration_seconds: float = Field(..., ge=0)
    pages_per_session: float = Field(..., ge=0)
    conversion_funnel_metrics: Dict[str, float] = Field(default_factory=dict)
    user_journey_completion_rate: float = Field(..., ge=0, le=1)
    feature_interaction_counts: Dict[str, int] = Field(default_factory=dict)
    error_encounter_rate: float = Field(..., ge=0, le=1)
    help_seeking_behavior_rate: float = Field(..., ge=0, le=1)
    social_sharing_rate: float = Field(..., ge=0, le=1)
    content_creation_rate: float = Field(..., ge=0, le=1)
    
    @validator('unique_page_views')
    def validate_unique_vs_total_views(cls, v, values):
        """Valide que vues uniques <= vues totales."""
        total = values.get('page_views_total', 0)
        if v > total:
            raise ValueError('Unique page views cannot exceed total page views')
        return v


# =============================================================================
# SCHÉMAS ML ET IA
# =============================================================================

class MLModelMetrics(BaseModel):
    """Métriques de modèle ML."""
    model_name: str = Field(..., min_length=1)
    model_version: str = Field(..., min_length=1)
    accuracy: float = Field(..., ge=0, le=1)
    precision: float = Field(..., ge=0, le=1)
    recall: float = Field(..., ge=0, le=1)
    f1_score: float = Field(..., ge=0, le=1)
    auc_roc: Optional[float] = Field(None, ge=0, le=1)
    prediction_latency_ms: float = Field(..., ge=0)
    training_time_hours: Optional[float] = Field(None, ge=0)
    data_drift_score: float = Field(..., ge=0, le=1)
    model_drift_score: float = Field(..., ge=0, le=1)
    inference_count_24h: int = Field(..., ge=0)
    error_rate: float = Field(..., ge=0, le=1)
    resource_usage: Dict[str, float] = Field(default_factory=dict)
    feature_importance: Dict[str, float] = Field(default_factory=dict)
    
    @validator('f1_score')
    def validate_f1_score(cls, v, values):
        """Valide la cohérence du F1 score avec precision/recall."""
        precision = values.get('precision')
        recall = values.get('recall')
        if precision is not None and recall is not None:
            if precision + recall > 0:
                expected_f1 = 2 * (precision * recall) / (precision + recall)
                if abs(v - expected_f1) > 0.01:  # Tolérance de 1%
                    raise ValueError('F1 score inconsistent with precision and recall')
        return v


class AIAgentMetrics(BaseModel):
    """Métriques d'agent IA."""
    agent_name: str = Field(..., min_length=1)
    task_completion_rate: float = Field(..., ge=0, le=1)
    task_success_rate: float = Field(..., ge=0, le=1)
    average_task_duration_seconds: float = Field(..., ge=0)
    user_satisfaction_score: float = Field(..., ge=1, le=5)
    escalation_rate: float = Field(..., ge=0, le=1)
    conversation_length_avg: float = Field(..., ge=0)
    intent_recognition_accuracy: float = Field(..., ge=0, le=1)
    response_coherence_score: float = Field(..., ge=0, le=1)
    knowledge_base_coverage: float = Field(..., ge=0, le=1)
    learning_improvement_rate: float = Field(..., ge=0)
    context_retention_score: float = Field(..., ge=0, le=1)
    multilingual_capability_score: Optional[float] = Field(None, ge=0, le=1)


# =============================================================================
# SCHÉMAS AUDIO ET SPOTIFY
# =============================================================================

class AudioQualityMetrics(BaseModel):
    """Métriques de qualité audio."""
    sample_rate_hz: int = Field(..., ge=8000, le=192000)
    bit_depth: int = Field(..., ge=8, le=32)
    channels: int = Field(..., ge=1, le=8)
    bitrate_kbps: Optional[int] = Field(None, ge=32, le=2000)
    dynamic_range_db: float = Field(..., ge=0, le=144)
    signal_to_noise_ratio_db: float = Field(..., ge=0)
    total_harmonic_distortion: float = Field(..., ge=0, le=1)
    frequency_response_flatness: float = Field(..., ge=0, le=1)
    stereo_imaging_score: float = Field(..., ge=0, le=1)
    codec_efficiency_score: float = Field(..., ge=0, le=1)
    perceived_quality_score: float = Field(..., ge=0, le=5)
    streaming_interruptions_count: int = Field(..., ge=0)
    buffer_health_percentage: float = Field(..., ge=0, le=100)
    latency_ms: float = Field(..., ge=0)
    jitter_ms: float = Field(..., ge=0)


class SpotifyIntegrationMetrics(BaseModel):
    """Métriques d'intégration Spotify."""
    api_calls_total: int = Field(..., ge=0)
    api_calls_successful: int = Field(..., ge=0)
    api_calls_failed: int = Field(..., ge=0)
    average_response_time_ms: float = Field(..., ge=0)
    rate_limit_usage_percentage: float = Field(..., ge=0, le=100)
    data_sync_latency_minutes: float = Field(..., ge=0)
    cache_hit_rate: float = Field(..., ge=0, le=1)
    tracks_synchronized: int = Field(..., ge=0)
    playlists_synchronized: int = Field(..., ge=0)
    user_profiles_synchronized: int = Field(..., ge=0)
    data_freshness_score: float = Field(..., ge=0, le=1)
    integration_health_score: float = Field(..., ge=0, le=100)
    quota_utilization: Dict[str, float] = Field(default_factory=dict)
    
    @validator('api_calls_successful', 'api_calls_failed')
    def validate_api_call_totals(cls, v, values):
        """Valide que la somme des appels correspond au total."""
        total = values.get('api_calls_total', 0)
        if 'api_calls_successful' in values and 'api_calls_failed' in values:
            success = values['api_calls_successful']
            failed = values['api_calls_failed']
            if success + failed != total:
                raise ValueError('Sum of successful and failed calls must equal total')
        return v


class PlaylistAnalytics(BaseModel):
    """Analytics de playlist."""
    playlist_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1, max_length=200)
    tracks_count: int = Field(..., ge=0)
    followers_count: int = Field(..., ge=0)
    total_plays: int = Field(..., ge=0)
    unique_listeners: int = Field(..., ge=0)
    average_completion_rate: float = Field(..., ge=0, le=1)
    skip_rate: float = Field(..., ge=0, le=1)
    save_rate: float = Field(..., ge=0, le=1)
    share_count: int = Field(..., ge=0)
    genre_distribution: Dict[str, float] = Field(default_factory=dict)
    popularity_score: float = Field(..., ge=0, le=100)
    engagement_score: float = Field(..., ge=0, le=100)
    viral_potential_score: float = Field(..., ge=0, le=1)
    collaborative_activity_score: Optional[float] = Field(None, ge=0, le=1)


# =============================================================================
# SCHÉMAS DE SÉCURITÉ
# =============================================================================

class SecurityMetrics(BaseModel):
    """Métriques de sécurité."""
    failed_login_attempts: int = Field(..., ge=0)
    successful_logins: int = Field(..., ge=0)
    blocked_ips_count: int = Field(..., ge=0)
    suspicious_activities_detected: int = Field(..., ge=0)
    security_alerts_triggered: int = Field(..., ge=0)
    vulnerability_scan_score: float = Field(..., ge=0, le=100)
    compliance_score: float = Field(..., ge=0, le=100)
    data_breach_incidents: int = Field(..., ge=0)
    encryption_coverage_percentage: float = Field(..., ge=0, le=100)
    access_control_violations: int = Field(..., ge=0)
    audit_trail_completeness: float = Field(..., ge=0, le=1)
    threat_intelligence_score: float = Field(..., ge=0, le=100)
    incident_response_time_minutes: Optional[float] = Field(None, ge=0)


class ComplianceMetrics(BaseModel):
    """Métriques de conformité."""
    gdpr_compliance_score: float = Field(..., ge=0, le=100)
    sox_compliance_score: float = Field(..., ge=0, le=100)
    pci_dss_compliance_score: float = Field(..., ge=0, le=100)
    data_retention_compliance: float = Field(..., ge=0, le=1)
    user_consent_rate: float = Field(..., ge=0, le=1)
    data_subject_requests_count: int = Field(..., ge=0)
    data_processing_transparency_score: float = Field(..., ge=0, le=100)
    privacy_policy_acceptance_rate: float = Field(..., ge=0, le=1)
    breach_notification_compliance: bool = True
    audit_readiness_score: float = Field(..., ge=0, le=100)


# =============================================================================
# SCHÉMAS DE RAPPORT
# =============================================================================

class MonitoringReport(BaseModel):
    """Rapport de monitoring complet."""
    report_id: str = Field(..., min_length=1)
    tenant_id: str
    report_type: Literal["daily", "weekly", "monthly", "quarterly", "annual"]
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    period_start: datetime
    period_end: datetime
    executive_summary: str = Field(..., max_length=2000)
    overall_health_score: HealthScore
    system_performance: SystemPerformanceMetrics
    business_metrics: BusinessKPIMetrics
    user_behavior: UserBehaviorMetrics
    ml_performance: Optional[List[MLModelMetrics]] = None
    audio_quality: Optional[AudioQualityMetrics] = None
    spotify_integration: Optional[SpotifyIntegrationMetrics] = None
    security_overview: SecurityMetrics
    compliance_status: ComplianceMetrics
    active_alerts: List[Alert] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    trends: Dict[str, str] = Field(default_factory=dict)
    
    @validator('period_end')
    def validate_period_dates(cls, v, values):
        """Valide que la période de fin est après le début."""
        start = values.get('period_start')
        if start and v <= start:
            raise ValueError('Period end must be after period start')
        return v
    
    @validator('generated_at')
    def validate_generation_time(cls, v, values):
        """Valide que le rapport est généré après la période."""
        period_end = values.get('period_end')
        if period_end and v < period_end:
            raise ValueError('Report cannot be generated before period end')
        return v


class DashboardData(BaseModel):
    """Données pour tableau de bord."""
    tenant_id: str
    dashboard_type: Literal["executive", "operational", "technical", "business"]
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    refresh_interval_seconds: int = Field(default=300, ge=30, le=3600)
    key_metrics: Dict[str, Union[int, float, str]]
    health_indicators: List[HealthScore]
    recent_alerts: List[Alert] = Field(default_factory=list)
    trend_data: Dict[str, List[Dict[str, Union[str, float]]]] = Field(default_factory=dict)
    performance_indicators: Dict[str, float] = Field(default_factory=dict)
    status_indicators: Dict[str, str] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "tenant_id": "tenant_001",
                "dashboard_type": "executive",
                "key_metrics": {
                    "total_users": 15678,
                    "monthly_revenue": 234567.89,
                    "system_uptime": 99.97,
                    "user_satisfaction": 4.2
                },
                "health_indicators": [
                    {
                        "component_name": "API Gateway",
                        "score": 94.5,
                        "status": "healthy"
                    }
                ]
            }
        }


# =============================================================================
# CONFIGURATION ET VALIDATION
# =============================================================================

class MonitoringConfiguration(BaseModel):
    """Configuration du système de monitoring."""
    tenant_id: str
    monitoring_enabled: bool = True
    collection_interval_seconds: int = Field(default=60, ge=10, le=3600)
    retention_days: int = Field(default=90, ge=1, le=365)
    alert_thresholds: Dict[str, Union[int, float]] = Field(default_factory=dict)
    notification_channels: List[str] = Field(default_factory=list)
    custom_metrics: List[str] = Field(default_factory=list)
    dashboard_preferences: Dict[str, Any] = Field(default_factory=dict)
    data_sources: List[DataSourceType] = Field(default_factory=list)
    compliance_requirements: List[str] = Field(default_factory=list)
    
    @validator('alert_thresholds')
    def validate_thresholds(cls, v):
        """Valide que les seuils sont dans des plages raisonnables."""
        for metric_name, threshold in v.items():
            if isinstance(threshold, (int, float)):
                if threshold < 0:
                    raise ValueError(f'Threshold for {metric_name} must be non-negative')
        return v


# =============================================================================
# MODÈLES D'EXPORT
# =============================================================================

class ExportRequest(BaseModel):
    """Requête d'export de données."""
    export_id: str = Field(..., min_length=1)
    tenant_id: str
    export_type: Literal["csv", "json", "pdf", "excel"]
    data_types: List[str] = Field(..., min_items=1)
    date_range_start: datetime
    date_range_end: datetime
    filters: Dict[str, Any] = Field(default_factory=dict)
    aggregation_level: Literal["raw", "hourly", "daily", "weekly", "monthly"] = "daily"
    include_metadata: bool = True
    compression: bool = False
    
    @validator('date_range_end')
    def validate_date_range(cls, v, values):
        """Valide que la plage de dates est correcte."""
        start = values.get('date_range_start')
        if start and v <= start:
            raise ValueError('End date must be after start date')
        if start and (v - start).days > 365:
            raise ValueError('Date range cannot exceed 365 days')
        return v


# =============================================================================
# CONFIGURATION GLOBALE
# =============================================================================

# Configuration Pydantic globale
class Config:
    use_enum_values = True
    validate_assignment = True
    extra = "forbid"
    json_encoders = {
        datetime: lambda dt: dt.isoformat(),
        Decimal: lambda d: float(d)
    }


# Application de la configuration à tous les modèles
for model_class in [
    BaseMetric, Alert, HealthScore, SystemPerformanceMetrics,
    DatabasePerformanceMetrics, APIPerformanceMetrics, TenantMetrics,
    BusinessKPIMetrics, UserBehaviorMetrics, MLModelMetrics,
    AIAgentMetrics, AudioQualityMetrics, SpotifyIntegrationMetrics,
    PlaylistAnalytics, SecurityMetrics, ComplianceMetrics,
    MonitoringReport, DashboardData, MonitoringConfiguration,
    ExportRequest
]:
    if hasattr(model_class, 'Config'):
        for attr, value in Config.__dict__.items():
            if not attr.startswith('_'):
                setattr(model_class.Config, attr, value)
    else:
        model_class.Config = Config


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def validate_monitoring_data(data: Dict[str, Any], schema_class: BaseModel) -> BaseModel:
    """
    Valide des données de monitoring contre un schéma.
    
    Args:
        data: Données à valider
        schema_class: Classe de schéma Pydantic
        
    Returns:
        Instance validée du modèle
        
    Raises:
        ValueError: Si les données ne sont pas valides
    """
    try:
        return schema_class(**data)
    except Exception as e:
        raise ValueError(f"Data validation failed: {str(e)}")


def create_health_score(component: str, metrics: Dict[str, float], 
                       weights: Optional[Dict[str, float]] = None) -> HealthScore:
    """
    Crée un score de santé basé sur des métriques.
    
    Args:
        component: Nom du composant
        metrics: Métriques avec leurs valeurs (0-100)
        weights: Poids optionnels pour chaque métrique
        
    Returns:
        Score de santé calculé
    """
    if not metrics:
        return HealthScore(component_name=component, score=0.0)
    
    weights = weights or {metric: 1.0 for metric in metrics}
    
    weighted_sum = sum(
        metrics[metric] * weights.get(metric, 1.0)
        for metric in metrics
    )
    total_weight = sum(weights.get(metric, 1.0) for metric in metrics)
    
    score = weighted_sum / total_weight if total_weight > 0 else 0.0
    
    return HealthScore(
        component_name=component,
        score=score,
        contributing_factors=metrics
    )


def serialize_for_api(model_instance: BaseModel) -> Dict[str, Any]:
    """
    Sérialise un modèle pour l'API en gérant les types spéciaux.
    
    Args:
        model_instance: Instance du modèle Pydantic
        
    Returns:
        Dictionnaire sérialisé
    """
    return json.loads(model_instance.json())


__all__ = [
    # Enums
    'AlertSeverity', 'MetricType', 'DataSourceType', 'TenantTier',
    
    # Base Models
    'BaseMetric', 'Alert', 'HealthScore',
    
    # Performance Schemas
    'SystemPerformanceMetrics', 'DatabasePerformanceMetrics', 'APIPerformanceMetrics',
    
    # Business Schemas
    'TenantMetrics', 'BusinessKPIMetrics', 'UserBehaviorMetrics',
    
    # ML/AI Schemas
    'MLModelMetrics', 'AIAgentMetrics',
    
    # Audio/Spotify Schemas
    'AudioQualityMetrics', 'SpotifyIntegrationMetrics', 'PlaylistAnalytics',
    
    # Security Schemas
    'SecurityMetrics', 'ComplianceMetrics',
    
    # Report Schemas
    'MonitoringReport', 'DashboardData', 'MonitoringConfiguration', 'ExportRequest',
    
    # Utilities
    'validate_monitoring_data', 'create_health_score', 'serialize_for_api'
]
