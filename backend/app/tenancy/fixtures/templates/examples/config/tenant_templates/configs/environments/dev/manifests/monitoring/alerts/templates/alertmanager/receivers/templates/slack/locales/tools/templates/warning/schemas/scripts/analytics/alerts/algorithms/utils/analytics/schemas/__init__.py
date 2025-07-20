"""
Analytics Schemas Module - Ultra-Advanced Edition
================================================

Module ultra-avancé pour la gestion des schémas d'analytics et de monitoring
dans l'écosystème Spotify AI Agent.

Ce module fournit:
- Schémas de validation Pydantic avancés
- Modèles de données pour analytics en temps réel
- Structures de données pour ML et AI
- Schémas de monitoring et alerting
- Intégration avec systèmes distribués

Développé par l'équipe:
- Lead Dev + Architecte IA: Fahed Mlaiel
- Développeur Backend Senior (Python/FastAPI/Django): Fahed Mlaiel
- Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face): Fahed Mlaiel
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB): Fahed Mlaiel
- Spécialiste Sécurité Backend: Fahed Mlaiel
- Architecte Microservices: Fahed Mlaiel

Version: 2.0.0
Date: 2025-07-19
License: MIT
"""

from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Literal
from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import UUID4, EmailStr, PositiveInt, NonNegativeFloat
import re

class TenantTier(str, Enum):
    """Tenant subscription tiers."""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"

class UserTier(str, Enum):
    """User subscription tiers."""
    FREE = "free"
    PREMIUM = "premium"
    FAMILY = "family"
    STUDENT = "student"

class EventType(str, Enum):
    """Supported event types."""
    TRACK_PLAY = "track_play"
    TRACK_SKIP = "track_skip"
    TRACK_LIKE = "track_like"
    TRACK_SHARE = "track_share"
    PLAYLIST_CREATE = "playlist_create"
    PLAYLIST_UPDATE = "playlist_update"
    USER_SIGNUP = "user_signup"
    USER_LOGIN = "user_login"
    SUBSCRIPTION_CHANGE = "subscription_change"
    SEARCH_QUERY = "search_query"
    RECOMMENDATION_CLICK = "recommendation_click"
    AD_IMPRESSION = "ad_impression"
    AD_CLICK = "ad_click"
    PAYMENT_PROCESSED = "payment_processed"

class DeviceType(str, Enum):
    """Supported device types."""
    MOBILE = "mobile"
    DESKTOP = "desktop"
    TABLET = "tablet"
    SMART_SPEAKER = "smart_speaker"
    TV = "tv"
    CAR = "car"
    GAMING_CONSOLE = "gaming_console"

class Platform(str, Enum):
    """Supported platforms."""
    IOS = "ios"
    ANDROID = "android"
    WEB = "web"
    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"

class MetricType(str, Enum):
    """Types of analytics metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertSeverity(str, Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class ContentRating(str, Enum):
    """Content rating classifications."""
    CLEAN = "clean"
    EXPLICIT = "explicit"
    UNKNOWN = "unknown"

# Base Models

class TenantInfo(BaseModel):
    """Tenant information schema."""
    tenant_id: str = Field(..., regex=r"^tenant_[a-zA-Z0-9_-]+$")
    tenant_name: str = Field(..., min_length=1, max_length=100)
    tier: TenantTier
    region: str = Field(..., regex=r"^[a-z]{2}-[a-z]+$")
    created_at: datetime
    settings: Dict[str, Any] = Field(default_factory=dict)

    @validator('settings')
    def validate_settings(cls, v):
        """Validate tenant settings."""
        allowed_keys = {
            'data_retention_days', 'rate_limit', 'analytics_enabled',
            'custom_metrics', 'alert_webhooks', 'feature_flags'
        }
        for key in v.keys():
            if key not in allowed_keys:
                raise ValueError(f"Invalid setting key: {key}")
        return v

class UserProfile(BaseModel):
    """User profile schema."""
    user_id: str = Field(..., regex=r"^user_[a-zA-Z0-9_-]+$")
    tenant_id: str = Field(..., regex=r"^tenant_[a-zA-Z0-9_-]+$")
    email: Optional[EmailStr] = None
    user_tier: UserTier
    registration_date: datetime
    last_active: Optional[datetime] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)
    demographics: Optional[Dict[str, Any]] = None

    @validator('preferences')
    def validate_preferences(cls, v):
        """Validate user preferences."""
        allowed_keys = {
            'preferred_genres', 'language', 'explicit_content',
            'notifications', 'privacy_settings', 'audio_quality'
        }
        for key in v.keys():
            if key not in allowed_keys:
                raise ValueError(f"Invalid preference key: {key}")
        return v

class TrackMetadata(BaseModel):
    """Track metadata schema."""
    track_id: str = Field(..., regex=r"^track_[a-zA-Z0-9_-]+$")
    title: str = Field(..., min_length=1, max_length=200)
    artist_id: str = Field(..., regex=r"^artist_[a-zA-Z0-9_-]+$")
    artist_name: str = Field(..., min_length=1, max_length=100)
    album_id: Optional[str] = Field(None, regex=r"^album_[a-zA-Z0-9_-]+$")
    duration_ms: PositiveInt
    genre: List[str] = Field(..., min_items=1, max_items=5)
    release_date: datetime
    content_rating: ContentRating
    popularity_score: NonNegativeFloat = Field(..., le=1.0)
    audio_features: Optional[Dict[str, float]] = None

    @validator('audio_features')
    def validate_audio_features(cls, v):
        """Validate audio features."""
        if v is None:
            return v
        
        required_features = {
            'danceability', 'energy', 'valence', 'acousticness',
            'instrumentalness', 'liveness', 'speechiness'
        }
        
        for feature in required_features:
            if feature not in v:
                raise ValueError(f"Missing audio feature: {feature}")
            if not 0 <= v[feature] <= 1:
                raise ValueError(f"Audio feature {feature} must be between 0 and 1")
        
        return v

# Event Schemas

class BaseEventSchema(BaseModel):
    """Base schema for all events."""
    event_id: str = Field(..., regex=r"^event_[a-zA-Z0-9_-]+$")
    tenant_id: str = Field(..., regex=r"^tenant_[a-zA-Z0-9_-]+$")
    user_id: str = Field(..., regex=r"^user_[a-zA-Z0-9_-]+$")
    event_type: EventType
    timestamp: datetime
    session_id: Optional[str] = Field(None, regex=r"^session_[a-zA-Z0-9_-]+$")
    device_info: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    @validator('timestamp')
    def validate_timestamp(cls, v):
        """Validate event timestamp."""
        now = datetime.now()
        # Allow events up to 5 minutes in the future (clock skew)
        if v > now + timedelta(minutes=5):
            raise ValueError("Event timestamp cannot be in the future")
        # Reject events older than 7 days
        if v < now - timedelta(days=7):
            raise ValueError("Event timestamp is too old")
        return v

    @validator('device_info')
    def validate_device_info(cls, v):
        """Validate device information."""
        if v is None:
            return v
        
        allowed_keys = {
            'device_type', 'platform', 'os_version', 'app_version',
            'screen_resolution', 'timezone', 'locale'
        }
        
        for key in v.keys():
            if key not in allowed_keys:
                raise ValueError(f"Invalid device info key: {key}")
        
        return v

class TrackPlayEvent(BaseEventSchema):
    """Track play event schema."""
    event_type: Literal[EventType.TRACK_PLAY] = EventType.TRACK_PLAY
    track_id: str = Field(..., regex=r"^track_[a-zA-Z0-9_-]+$")
    artist_id: str = Field(..., regex=r"^artist_[a-zA-Z0-9_-]+$")
    playlist_id: Optional[str] = Field(None, regex=r"^playlist_[a-zA-Z0-9_-]+$")
    listen_duration_ms: NonNegativeFloat
    track_duration_ms: PositiveInt
    play_source: str = Field(..., max_length=50)
    audio_quality: str = Field(..., regex=r"^(low|medium|high|lossless)$")
    shuffle_mode: bool = False
    repeat_mode: str = Field("off", regex=r"^(off|track|playlist)$")
    volume_level: Optional[float] = Field(None, ge=0.0, le=1.0)

    @validator('listen_duration_ms')
    def validate_listen_duration(cls, v, values):
        """Validate listen duration."""
        if 'track_duration_ms' in values and v > values['track_duration_ms']:
            raise ValueError("Listen duration cannot exceed track duration")
        return v

    @root_validator
    def validate_completion_rate(cls, values):
        """Calculate and validate completion rate."""
        listen_duration = values.get('listen_duration_ms', 0)
        track_duration = values.get('track_duration_ms', 1)
        
        completion_rate = listen_duration / track_duration
        if completion_rate > 1.1:  # Allow 10% tolerance for clock skew
            raise ValueError("Invalid completion rate")
        
        values['completion_rate'] = min(completion_rate, 1.0)
        return values

class SearchQueryEvent(BaseEventSchema):
    """Search query event schema."""
    event_type: Literal[EventType.SEARCH_QUERY] = EventType.SEARCH_QUERY
    query: str = Field(..., min_length=1, max_length=200)
    query_type: str = Field(..., regex=r"^(track|artist|album|playlist|genre)$")
    results_count: NonNegativeFloat
    selected_result_index: Optional[int] = Field(None, ge=0)
    search_context: Optional[str] = Field(None, max_length=100)

class SubscriptionChangeEvent(BaseEventSchema):
    """Subscription change event schema."""
    event_type: Literal[EventType.SUBSCRIPTION_CHANGE] = EventType.SUBSCRIPTION_CHANGE
    old_tier: UserTier
    new_tier: UserTier
    change_reason: str = Field(..., max_length=100)
    payment_method: Optional[str] = Field(None, max_length=50)
    amount: Optional[Decimal] = Field(None, decimal_places=2)
    currency: Optional[str] = Field(None, regex=r"^[A-Z]{3}$")

# Analytics Schemas

class AnalyticsMetricSchema(BaseModel):
    """Analytics metric schema."""
    metric_id: str = Field(..., regex=r"^metric_[a-zA-Z0-9_-]+$")
    tenant_id: str = Field(..., regex=r"^tenant_[a-zA-Z0-9_-]+$")
    metric_name: str = Field(..., regex=r"^[a-zA-Z][a-zA-Z0-9_]*$")
    metric_type: MetricType
    value: Union[int, float, Decimal]
    timestamp: datetime
    labels: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('metric_name')
    def validate_metric_name(cls, v):
        """Validate metric name follows conventions."""
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", v):
            raise ValueError("Metric name must start with letter and contain only alphanumeric and underscore")
        if len(v) > 100:
            raise ValueError("Metric name too long")
        return v

    @validator('labels')
    def validate_labels(cls, v):
        """Validate metric labels."""
        if len(v) > 20:
            raise ValueError("Too many labels")
        
        for key, value in v.items():
            if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", key):
                raise ValueError(f"Invalid label key: {key}")
            if len(str(value)) > 100:
                raise ValueError(f"Label value too long: {key}")
        
        return v

class AlertRuleSchema(BaseModel):
    """Alert rule schema."""
    rule_id: str = Field(..., regex=r"^alert_rule_[a-zA-Z0-9_-]+$")
    tenant_id: str = Field(..., regex=r"^tenant_[a-zA-Z0-9_-]+$")
    rule_name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1, max_length=500)
    metric_query: str = Field(..., min_length=1, max_length=1000)
    condition: str = Field(..., regex=r"^(gt|gte|lt|lte|eq|ne)$")
    threshold: Union[int, float, Decimal]
    severity: AlertSeverity
    evaluation_interval_seconds: PositiveInt = Field(..., ge=30, le=3600)
    for_duration_seconds: PositiveInt = Field(..., ge=0, le=86400)
    labels: Dict[str, str] = Field(default_factory=dict)
    annotations: Dict[str, str] = Field(default_factory=dict)
    enabled: bool = True

class DataQualityReportSchema(BaseModel):
    """Data quality report schema."""
    report_id: str = Field(..., regex=r"^dq_report_[a-zA-Z0-9_-]+$")
    tenant_id: str = Field(..., regex=r"^tenant_[a-zA-Z0-9_-]+$")
    dataset_name: str = Field(..., min_length=1, max_length=100)
    generated_at: datetime
    total_records: NonNegativeFloat
    quality_score: float = Field(..., ge=0.0, le=1.0)
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)

    @validator('issues')
    def validate_issues(cls, v):
        """Validate data quality issues."""
        for issue in v:
            required_keys = {'issue_type', 'severity', 'description', 'affected_fields'}
            if not all(key in issue for key in required_keys):
                raise ValueError("Issue missing required keys")
        return v

# ML Model Schemas

class MLModelConfigSchema(BaseModel):
    """ML model configuration schema."""
    model_id: str = Field(..., regex=r"^model_[a-zA-Z0-9_-]+$")
    tenant_id: str = Field(..., regex=r"^tenant_[a-zA-Z0-9_-]+$")
    model_type: str = Field(..., regex=r"^(anomaly_detection|trend_forecasting|recommendation|classification|regression)$")
    algorithm: str = Field(..., min_length=1, max_length=50)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    training_config: Dict[str, Any] = Field(default_factory=dict)
    validation_config: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

class MLPredictionSchema(BaseModel):
    """ML prediction result schema."""
    prediction_id: str = Field(..., regex=r"^prediction_[a-zA-Z0-9_-]+$")
    model_id: str = Field(..., regex=r"^model_[a-zA-Z0-9_-]+$")
    tenant_id: str = Field(..., regex=r"^tenant_[a-zA-Z0-9_-]+$")
    input_features: Dict[str, Any]
    prediction: Union[int, float, str, List[Any]]
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    model_version: str = Field(..., min_length=1, max_length=20)
    predicted_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

# Performance Monitoring Schemas

class PerformanceMetricSchema(BaseModel):
    """Performance monitoring metric schema."""
    metric_id: str = Field(..., regex=r"^perf_metric_[a-zA-Z0-9_-]+$")
    tenant_id: str = Field(..., regex=r"^tenant_[a-zA-Z0-9_-]+$")
    operation_name: str = Field(..., min_length=1, max_length=100)
    duration_ms: NonNegativeFloat
    memory_usage_mb: Optional[NonNegativeFloat] = None
    cpu_usage_percent: Optional[float] = Field(None, ge=0.0, le=100.0)
    success: bool
    error_message: Optional[str] = Field(None, max_length=1000)
    timestamp: datetime
    labels: Dict[str, str] = Field(default_factory=dict)

# Aggregation Schemas

class DashboardDataSchema(BaseModel):
    """Dashboard data schema."""
    dashboard_id: str = Field(..., regex=r"^dashboard_[a-zA-Z0-9_-]+$")
    tenant_id: str = Field(..., regex=r"^tenant_[a-zA-Z0-9_-]+$")
    generated_at: datetime
    time_range_start: datetime
    time_range_end: datetime
    metrics: Dict[str, Any] = Field(default_factory=dict)
    charts: List[Dict[str, Any]] = Field(default_factory=list)
    alerts: List[Dict[str, Any]] = Field(default_factory=list)
    insights: List[str] = Field(default_factory=list)

# Validation Utilities

def validate_tenant_access(tenant_id: str, user_tenant_id: str) -> bool:
    """Validate tenant access permissions."""
    return tenant_id == user_tenant_id

def validate_data_retention(timestamp: datetime, retention_days: int) -> bool:
    """Validate data against retention policy."""
    retention_threshold = datetime.now() - timedelta(days=retention_days)
    return timestamp >= retention_threshold

def sanitize_user_input(input_str: str) -> str:
    """Sanitize user input for security."""
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\'\&]', '', input_str)
    return sanitized.strip()

# Schema Registry

SCHEMA_REGISTRY = {
    'base_event': BaseEventSchema,
    'track_play': TrackPlayEvent,
    'search_query': SearchQueryEvent,
    'subscription_change': SubscriptionChangeEvent,
    'analytics_metric': AnalyticsMetricSchema,
    'alert_rule': AlertRuleSchema,
    'data_quality_report': DataQualityReportSchema,
    'ml_model_config': MLModelConfigSchema,
    'ml_prediction': MLPredictionSchema,
    'performance_metric': PerformanceMetricSchema,
    'dashboard_data': DashboardDataSchema,
    'tenant_info': TenantInfo,
    'user_profile': UserProfile,
    'track_metadata': TrackMetadata
}

def get_schema(schema_name: str) -> BaseModel:
    """Get schema by name."""
    if schema_name not in SCHEMA_REGISTRY:
        raise ValueError(f"Unknown schema: {schema_name}")
    return SCHEMA_REGISTRY[schema_name]

def validate_data(data: Dict[str, Any], schema_name: str) -> BaseModel:
    """Validate data against schema."""
    schema_class = get_schema(schema_name)
    return schema_class(**data)

# Export all schemas
__all__ = [
    'TenantTier', 'UserTier', 'EventType', 'DeviceType', 'Platform',
    'MetricType', 'AlertSeverity', 'ContentRating',
    'TenantInfo', 'UserProfile', 'TrackMetadata',
    'BaseEventSchema', 'TrackPlayEvent', 'SearchQueryEvent', 'SubscriptionChangeEvent',
    'AnalyticsMetricSchema', 'AlertRuleSchema', 'DataQualityReportSchema',
    'MLModelConfigSchema', 'MLPredictionSchema', 'PerformanceMetricSchema',
    'DashboardDataSchema', 'SCHEMA_REGISTRY', 'get_schema', 'validate_data',
    'validate_tenant_access', 'validate_data_retention', 'sanitize_user_input'
]
