"""
Advanced observability and metrics schemas for enterprise monitoring.

This module provides comprehensive schemas for metrics collection, alerting,
tracing, logging, and advanced observability features with AI-powered insights.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import EmailStr


class MetricType(str, Enum):
    """Types of metrics supported."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"
    RATE = "rate"
    PERCENTILE = "percentile"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


class DataRetentionPolicy(str, Enum):
    """Data retention policies."""
    SHORT_TERM = "short_term"    # 7 days
    MEDIUM_TERM = "medium_term"  # 30 days
    LONG_TERM = "long_term"      # 90 days
    ARCHIVE = "archive"          # 1 year
    PERMANENT = "permanent"      # Indefinite


class SamplingStrategy(str, Enum):
    """Sampling strategies for high-volume metrics."""
    NONE = "none"
    UNIFORM = "uniform"
    ADAPTIVE = "adaptive"
    PROBABILISTIC = "probabilistic"
    RATE_LIMITING = "rate_limiting"


class MetricDefinitionSchema(BaseModel):
    """Schema for defining custom metrics."""
    metric_id: UUID = Field(..., description="Unique metric identifier")
    name: str = Field(..., description="Metric name")
    description: str = Field(..., description="Metric description")
    type: MetricType = Field(..., description="Metric type")
    
    # Metric configuration
    unit: str = Field(..., description="Metric unit (e.g., 'bytes', 'seconds', 'count')")
    labels: List[str] = Field([], description="Available metric labels")
    help_text: str = Field(..., description="Help text for the metric")
    
    # Collection settings
    collection_interval: int = Field(60, ge=1, description="Collection interval in seconds")
    sampling_strategy: SamplingStrategy = Field(SamplingStrategy.NONE, description="Sampling strategy")
    sampling_rate: float = Field(1.0, ge=0.0, le=1.0, description="Sampling rate (0.0-1.0)")
    
    # Retention
    retention_policy: DataRetentionPolicy = Field(DataRetentionPolicy.MEDIUM_TERM, description="Data retention policy")
    custom_retention_days: Optional[int] = Field(None, description="Custom retention period in days")
    
    # Aggregation
    aggregation_functions: List[str] = Field(["avg", "sum", "min", "max"], description="Supported aggregation functions")
    rollup_intervals: List[int] = Field([300, 3600, 86400], description="Rollup intervals in seconds")
    
    # Alert configuration
    alertable: bool = Field(True, description="Metric can trigger alerts")
    default_thresholds: Dict[str, float] = Field({}, description="Default alert thresholds")
    
    # Metadata
    tags: Dict[str, str] = Field({}, description="Metric tags")
    owner: EmailStr = Field(..., description="Metric owner")
    category: str = Field("custom", description="Metric category")
    
    class Config:
        schema_extra = {
            "example": {
                "metric_id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "api_request_duration",
                "description": "Duration of API requests in seconds",
                "type": "histogram",
                "unit": "seconds",
                "labels": ["method", "endpoint", "status_code", "tenant_id"],
                "help_text": "Tracks the duration of API requests across all endpoints",
                "collection_interval": 30,
                "sampling_strategy": "adaptive",
                "retention_policy": "long_term",
                "aggregation_functions": ["avg", "p95", "p99", "max"],
                "alertable": True,
                "default_thresholds": {
                    "warning": 2.0,
                    "critical": 5.0
                },
                "owner": "api-team@spotify.com",
                "category": "performance"
            }
        }


class AlertRuleSchema(BaseModel):
    """Schema for alert rule configuration."""
    rule_id: UUID = Field(..., description="Unique rule identifier")
    name: str = Field(..., description="Alert rule name")
    description: str = Field(..., description="Alert rule description")
    metric_id: UUID = Field(..., description="Associated metric ID")
    
    # Alert conditions
    query: str = Field(..., description="PromQL or similar query")
    threshold: float = Field(..., description="Alert threshold value")
    operator: str = Field(..., description="Comparison operator (>, <, >=, <=, ==, !=)")
    severity: AlertSeverity = Field(..., description="Alert severity")
    
    # Timing configuration
    evaluation_interval: int = Field(60, ge=10, description="Evaluation interval in seconds")
    for_duration: int = Field(300, ge=0, description="Alert must be active for this duration")
    
    # Grouping and routing
    group_by: List[str] = Field([], description="Labels to group alerts by")
    route_to: List[str] = Field([], description="Alert routing destinations")
    
    # Suppression
    inhibit_rules: List[str] = Field([], description="Rules that can inhibit this alert")
    silence_patterns: List[str] = Field([], description="Patterns for auto-silencing")
    
    # Notification settings
    notification_channels: List[str] = Field([], description="Notification channels")
    escalation_rules: List[Dict[str, Any]] = Field([], description="Escalation configuration")
    
    # Dependencies
    depends_on: List[UUID] = Field([], description="Alert dependencies")
    
    # Status
    enabled: bool = Field(True, description="Rule is enabled")
    
    # Metadata
    tags: Dict[str, str] = Field({}, description="Alert rule tags")
    owner: EmailStr = Field(..., description="Rule owner")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "rule_id": "660e8400-e29b-41d4-a716-446655440001",
                "name": "High API Response Time",
                "description": "Alert when API response time exceeds threshold",
                "metric_id": "550e8400-e29b-41d4-a716-446655440000",
                "query": "histogram_quantile(0.95, api_request_duration) > 2.0",
                "threshold": 2.0,
                "operator": ">",
                "severity": "warning",
                "evaluation_interval": 60,
                "for_duration": 300,
                "group_by": ["endpoint", "tenant_id"],
                "notification_channels": ["#api-alerts", "api-team@spotify.com"],
                "escalation_rules": [
                    {
                        "after_minutes": 15,
                        "severity": "critical",
                        "notify": ["#ops-critical"]
                    }
                ],
                "enabled": True,
                "owner": "api-team@spotify.com"
            }
        }


class DashboardConfigSchema(BaseModel):
    """Schema for dashboard configuration."""
    dashboard_id: UUID = Field(..., description="Unique dashboard identifier")
    name: str = Field(..., description="Dashboard name")
    description: str = Field(..., description="Dashboard description")
    
    # Layout configuration
    layout: Dict[str, Any] = Field(..., description="Dashboard layout configuration")
    panels: List[Dict[str, Any]] = Field([], description="Dashboard panels")
    
    # Data sources
    data_sources: List[str] = Field([], description="Connected data sources")
    metrics: List[UUID] = Field([], description="Used metrics")
    
    # Refresh settings
    refresh_interval: int = Field(300, ge=10, description="Refresh interval in seconds")
    auto_refresh: bool = Field(True, description="Enable auto refresh")
    
    # Time range
    default_time_range: str = Field("1h", description="Default time range")
    max_time_range: str = Field("30d", description="Maximum time range")
    
    # Access control
    visibility: str = Field("private", description="Dashboard visibility (private, team, public)")
    allowed_users: List[EmailStr] = Field([], description="Allowed users")
    allowed_teams: List[str] = Field([], description="Allowed teams")
    
    # Features
    annotations_enabled: bool = Field(True, description="Enable annotations")
    templating_enabled: bool = Field(True, description="Enable templating")
    alerting_enabled: bool = Field(True, description="Enable dashboard alerting")
    
    # Metadata
    tags: List[str] = Field([], description="Dashboard tags")
    owner: EmailStr = Field(..., description="Dashboard owner")
    
    class Config:
        schema_extra = {
            "example": {
                "dashboard_id": "770e8400-e29b-41d4-a716-446655440002",
                "name": "API Performance Overview",
                "description": "Real-time overview of API performance metrics",
                "layout": {
                    "grid_size": 24,
                    "height": 800,
                    "panels_per_row": 3
                },
                "panels": [
                    {
                        "id": "panel_1",
                        "title": "Request Rate",
                        "type": "graph",
                        "metric_id": "550e8400-e29b-41d4-a716-446655440000"
                    }
                ],
                "refresh_interval": 30,
                "default_time_range": "1h",
                "visibility": "team",
                "allowed_teams": ["api-team", "ops-team"],
                "owner": "api-team@spotify.com"
            }
        }


class TracingConfigSchema(BaseModel):
    """Schema for distributed tracing configuration."""
    tracing_id: UUID = Field(..., description="Unique tracing configuration identifier")
    name: str = Field(..., description="Tracing configuration name")
    
    # Tracing backend
    backend_type: str = Field("jaeger", description="Tracing backend (jaeger, zipkin, otlp)")
    endpoint: str = Field(..., description="Tracing backend endpoint")
    
    # Sampling configuration
    sampling_strategy: SamplingStrategy = Field(SamplingStrategy.PROBABILISTIC, description="Sampling strategy")
    sampling_rate: float = Field(0.1, ge=0.0, le=1.0, description="Base sampling rate")
    adaptive_sampling: bool = Field(True, description="Enable adaptive sampling")
    
    # Service configuration
    service_name: str = Field(..., description="Service name for traces")
    service_version: str = Field(..., description="Service version")
    environment: str = Field(..., description="Environment name")
    
    # Span configuration
    max_span_attributes: int = Field(64, ge=1, description="Maximum span attributes")
    max_span_events: int = Field(128, ge=1, description="Maximum span events")
    max_span_links: int = Field(32, ge=1, description="Maximum span links")
    
    # Performance settings
    batch_export: bool = Field(True, description="Enable batch export")
    batch_size: int = Field(512, ge=1, description="Batch size for export")
    export_timeout: int = Field(30, ge=1, description="Export timeout in seconds")
    
    # Security
    authentication: Dict[str, Any] = Field({}, description="Authentication configuration")
    encryption_enabled: bool = Field(True, description="Enable encryption")
    
    # Retention
    retention_days: int = Field(7, ge=1, description="Trace retention in days")
    
    class Config:
        schema_extra = {
            "example": {
                "tracing_id": "880e8400-e29b-41d4-a716-446655440003",
                "name": "Production Tracing",
                "backend_type": "jaeger",
                "endpoint": "http://jaeger-collector:14268/api/traces",
                "sampling_strategy": "adaptive",
                "sampling_rate": 0.1,
                "service_name": "spotify-ai-agent",
                "service_version": "2.1.0",
                "environment": "production",
                "batch_export": True,
                "batch_size": 512,
                "retention_days": 7
            }
        }


class LoggingConfigSchema(BaseModel):
    """Schema for logging configuration."""
    logging_id: UUID = Field(..., description="Unique logging configuration identifier")
    name: str = Field(..., description="Logging configuration name")
    
    # Log levels
    root_level: str = Field("INFO", description="Root logging level")
    logger_levels: Dict[str, str] = Field({}, description="Per-logger level configuration")
    
    # Formatters
    format_string: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    structured_logging: bool = Field(True, description="Enable structured JSON logging")
    
    # Handlers
    console_enabled: bool = Field(True, description="Enable console output")
    file_enabled: bool = Field(True, description="Enable file output")
    syslog_enabled: bool = Field(False, description="Enable syslog output")
    remote_enabled: bool = Field(True, description="Enable remote logging")
    
    # File configuration
    log_file_path: str = Field("/var/log/app.log", description="Log file path")
    max_file_size: str = Field("10MB", description="Maximum log file size")
    backup_count: int = Field(5, ge=1, description="Number of backup files")
    
    # Remote logging
    remote_endpoint: Optional[str] = Field(None, description="Remote logging endpoint")
    remote_format: str = Field("json", description="Remote logging format")
    
    # Filtering
    filters: List[Dict[str, Any]] = Field([], description="Log filters")
    sensitive_data_masking: bool = Field(True, description="Enable sensitive data masking")
    
    # Performance
    async_logging: bool = Field(True, description="Enable asynchronous logging")
    buffer_size: int = Field(1024, ge=1, description="Log buffer size")
    
    # Security
    log_encryption: bool = Field(False, description="Enable log encryption")
    audit_logging: bool = Field(True, description="Enable audit logging")
    
    class Config:
        schema_extra = {
            "example": {
                "logging_id": "990e8400-e29b-41d4-a716-446655440004",
                "name": "Production Logging",
                "root_level": "INFO",
                "logger_levels": {
                    "spotify.ai": "DEBUG",
                    "sqlalchemy": "WARNING"
                },
                "structured_logging": True,
                "console_enabled": False,
                "file_enabled": True,
                "remote_enabled": True,
                "log_file_path": "/var/log/spotify-ai.log",
                "max_file_size": "100MB",
                "backup_count": 10,
                "remote_endpoint": "https://logs.spotify.com/api/v1/logs",
                "async_logging": True,
                "sensitive_data_masking": True
            }
        }


class AIInsightsConfigSchema(BaseModel):
    """Schema for AI-powered observability insights."""
    insights_id: UUID = Field(..., description="Unique insights configuration identifier")
    name: str = Field(..., description="AI insights configuration name")
    
    # AI model configuration
    model_type: str = Field("gpt-4", description="AI model for insights")
    model_config: Dict[str, Any] = Field({}, description="Model configuration")
    
    # Analysis configuration
    anomaly_detection: bool = Field(True, description="Enable anomaly detection")
    trend_analysis: bool = Field(True, description="Enable trend analysis")
    correlation_analysis: bool = Field(True, description="Enable correlation analysis")
    predictive_analysis: bool = Field(True, description="Enable predictive analysis")
    
    # Thresholds
    anomaly_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Anomaly detection threshold")
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum confidence threshold")
    
    # Analysis scope
    metrics_scope: List[UUID] = Field([], description="Metrics to analyze")
    time_window: str = Field("24h", description="Analysis time window")
    analysis_frequency: int = Field(3600, ge=300, description="Analysis frequency in seconds")
    
    # Output configuration
    generate_reports: bool = Field(True, description="Generate insight reports")
    auto_alerts: bool = Field(True, description="Generate automatic alerts")
    recommendations: bool = Field(True, description="Generate recommendations")
    
    # Notification
    notification_channels: List[str] = Field([], description="Insight notification channels")
    
    class Config:
        schema_extra = {
            "example": {
                "insights_id": "aa0e8400-e29b-41d4-a716-446655440005",
                "name": "Production AI Insights",
                "model_type": "gpt-4",
                "anomaly_detection": True,
                "trend_analysis": True,
                "correlation_analysis": True,
                "anomaly_threshold": 0.85,
                "confidence_threshold": 0.75,
                "time_window": "24h",
                "analysis_frequency": 1800,
                "generate_reports": True,
                "auto_alerts": True,
                "notification_channels": ["#ai-insights"]
            }
        }


class ObservabilityConfigSchema(BaseModel):
    """Master schema for observability configuration."""
    config_id: UUID = Field(..., description="Configuration identifier")
    name: str = Field(..., description="Configuration name")
    description: str = Field(..., description="Configuration description")
    environment: str = Field(..., description="Target environment")
    
    # Core components
    metrics: List[MetricDefinitionSchema] = Field([], description="Metric definitions")
    alert_rules: List[AlertRuleSchema] = Field([], description="Alert rules")
    dashboards: List[DashboardConfigSchema] = Field([], description="Dashboard configurations")
    tracing: TracingConfigSchema = Field(..., description="Tracing configuration")
    logging: LoggingConfigSchema = Field(..., description="Logging configuration")
    ai_insights: AIInsightsConfigSchema = Field(..., description="AI insights configuration")
    
    # Global settings
    data_retention_policy: DataRetentionPolicy = Field(DataRetentionPolicy.MEDIUM_TERM, description="Default retention policy")
    high_availability: bool = Field(True, description="Enable high availability")
    disaster_recovery: bool = Field(True, description="Enable disaster recovery")
    
    # Performance
    max_metrics_per_second: int = Field(10000, ge=1, description="Maximum metrics ingestion rate")
    query_timeout_seconds: int = Field(300, ge=1, description="Query timeout")
    cache_enabled: bool = Field(True, description="Enable caching")
    
    # Security
    encryption_enabled: bool = Field(True, description="Enable encryption")
    authentication_required: bool = Field(True, description="Require authentication")
    rbac_enabled: bool = Field(True, description="Enable role-based access control")
    
    # Compliance
    audit_logging: bool = Field(True, description="Enable audit logging")
    data_anonymization: bool = Field(True, description="Enable data anonymization")
    gdpr_compliance: bool = Field(True, description="GDPR compliance mode")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "config_id": "bb0e8400-e29b-41d4-a716-446655440006",
                "name": "Production Observability",
                "description": "Complete observability configuration for production",
                "environment": "production",
                "data_retention_policy": "long_term",
                "high_availability": True,
                "max_metrics_per_second": 50000,
                "encryption_enabled": True,
                "audit_logging": True,
                "gdpr_compliance": True
            }
        }


# Export all schemas
__all__ = [
    "MetricType",
    "AlertSeverity",
    "DataRetentionPolicy",
    "SamplingStrategy",
    "MetricDefinitionSchema",
    "AlertRuleSchema",
    "DashboardConfigSchema",
    "TracingConfigSchema",
    "LoggingConfigSchema",
    "AIInsightsConfigSchema",
    "ObservabilityConfigSchema"
]
