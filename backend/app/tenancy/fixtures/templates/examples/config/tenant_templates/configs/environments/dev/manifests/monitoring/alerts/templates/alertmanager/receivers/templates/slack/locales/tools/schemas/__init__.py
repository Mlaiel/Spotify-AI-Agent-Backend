"""
Advanced schemas for enterprise-grade configuration management.

This module provides comprehensive Pydantic schemas for validation, serialization,
and deserialization of alerting, monitoring, localization, automation, and observability configurations.
"""

# Core alert schemas
from .alert_schemas import (
    AlertRuleSchema,
    AlertGroupSchema,
    AlertManagerConfigSchema,
    AlertReceiverSchema,
    AlertRoutingSchema,
    AlertInhibitRuleSchema,
    AlertSilenceSchema,
    AlertTemplateSchema,
    AlertThresholdSchema,
    AlertConditionSchema,
    AlertMetricSchema,
    AlertTimeWindowSchema,
    AlertEscalationSchema,
    AlertChannelSchema
)

# Monitoring schemas
from .monitoring_schemas import (
    MonitoringConfigSchema,
    MetricConfigSchema,
    DashboardConfigSchema,
    PrometheusConfigSchema,
    GrafanaConfigSchema,
    LoggingConfigSchema,
    TracingConfigSchema,
    HealthCheckConfigSchema,
    PerformanceMetricSchema,
    SystemMetricSchema,
    BusinessMetricSchema,
    AlertingMetricSchema
)

# Slack integration schemas
from .slack_schemas import (
    SlackConfigSchema,
    SlackChannelSchema,
    SlackMessageSchema,
    SlackTemplateSchema,
    SlackWebhookSchema,
    SlackUserGroupSchema,
    SlackAttachmentSchema,
    SlackFieldSchema,
    SlackActionSchema,
    SlackButtonSchema,
    SlackSelectSchema,
    SlackModalSchema,
    SlackBlockSchema,
    SlackElementSchema
)

# Tenant management schemas
from .tenant_schemas import (
    TenantConfigSchema,
    TenantEnvironmentSchema,
    TenantResourceSchema,
    TenantLimitSchema,
    TenantQuotaSchema,
    TenantSecuritySchema,
    TenantNetworkSchema,
    TenantStorageSchema,
    TenantComputeSchema,
    TenantMonitoringSchema,
    TenantLoggingSchema,
    TenantBackupSchema
)

# Validation schemas
from .validation_schemas import (
    ValidationRuleSchema,
    ValidationResultSchema,
    ValidationErrorSchema,
    ValidationContextSchema,
    SchemaValidatorSchema,
    DataValidatorSchema,
    ConfigValidatorSchema,
    SecurityValidatorSchema,
    ComplianceValidatorSchema,
    PerformanceValidatorSchema
)

# Tool configuration schemas
from .tool_schemas import (
    ToolConfigSchema,
    ToolExecutionSchema,
    ToolResultSchema,
    ToolParameterSchema,
    ToolMetadataSchema,
    ToolSecuritySchema,
    ToolLoggingSchema,
    ToolMonitoringSchema,
    AutomationToolSchema,
    DeploymentToolSchema,
    MonitoringToolSchema,
    SecurityToolSchema
)

# Enterprise configuration schemas
from .enterprise_config import (
    ConfigPriorityLevel,
    ComplianceStandard,
    EncryptionAlgorithm,
    ConfigurationSourceSchema,
    ConfigurationEncryptionSchema,
    ConfigurationComplianceSchema,
    ConfigurationBackupSchema,
    ConfigurationValidationSchema,
    ConfigurationDeploymentSchema,
    EnterpriseConfigurationSchema,
    ConfigurationChangelogSchema
)

# Localization and internationalization schemas
from .localization_schemas import (
    SupportedLocale,
    TextDirection,
    DateFormat,
    TimeFormat,
    NumberFormat,
    CurrencyPosition,
    LocaleConfigurationSchema,
    TranslationSchema,
    MessageLocalizationSchema,
    CulturalAdaptationSchema,
    AITranslationConfigSchema,
    LocalizationConfigurationSchema
)

# Automation and tooling schemas
from .automation_tools import (
    ToolCategory,
    ExecutionStrategy,
    ToolStatus,
    Priority,
    ToolConfigurationSchema,
    WorkflowStepSchema,
    WorkflowSchema,
    ExecutionContextSchema,
    ExecutionResultSchema,
    MaintenanceWindowSchema,
    AutomationToolsConfigSchema
)

# Observability and metrics schemas
from .observability_schemas import (
    MetricType,
    AlertSeverity,
    DataRetentionPolicy,
    SamplingStrategy,
    MetricDefinitionSchema,
    AlertRuleSchema as ObservabilityAlertRuleSchema,
    DashboardConfigSchema as ObservabilityDashboardConfigSchema,
    TracingConfigSchema,
    LoggingConfigSchema as ObservabilityLoggingConfigSchema,
    AIInsightsConfigSchema,
    ObservabilityConfigSchema
)

__all__ = [
    # Alert Schemas
    "AlertRuleSchema",
    "AlertGroupSchema", 
    "AlertManagerConfigSchema",
    "AlertReceiverSchema",
    "AlertRoutingSchema",
    "AlertInhibitRuleSchema",
    "AlertSilenceSchema",
    "AlertTemplateSchema",
    "AlertThresholdSchema",
    "AlertConditionSchema",
    "AlertMetricSchema",
    "AlertTimeWindowSchema",
    "AlertEscalationSchema",
    "AlertChannelSchema",
    
    # Monitoring Schemas
    "MonitoringConfigSchema",
    "MetricConfigSchema",
    "DashboardConfigSchema",
    "PrometheusConfigSchema",
    "GrafanaConfigSchema",
    "LoggingConfigSchema",
    "TracingConfigSchema",
    "HealthCheckConfigSchema",
    "PerformanceMetricSchema",
    "SystemMetricSchema",
    "BusinessMetricSchema",
    "AlertingMetricSchema",
    
    # Slack Schemas
    "SlackConfigSchema",
    "SlackChannelSchema",
    "SlackMessageSchema",
    "SlackTemplateSchema",
    "SlackWebhookSchema",
    "SlackUserGroupSchema",
    "SlackAttachmentSchema",
    "SlackFieldSchema",
    "SlackActionSchema",
    "SlackButtonSchema",
    "SlackSelectSchema",
    "SlackModalSchema",
    "SlackBlockSchema",
    "SlackElementSchema",
    
    # Tenant Schemas
    "TenantConfigSchema",
    "TenantEnvironmentSchema",
    "TenantResourceSchema",
    "TenantLimitSchema",
    "TenantQuotaSchema",
    "TenantSecuritySchema",
    "TenantNetworkSchema",
    "TenantStorageSchema",
    "TenantComputeSchema",
    "TenantMonitoringSchema",
    "TenantLoggingSchema",
    "TenantBackupSchema",
    
    # Validation Schemas
    "ValidationRuleSchema",
    "ValidationResultSchema",
    "ValidationErrorSchema",
    "ValidationContextSchema",
    "SchemaValidatorSchema",
    "DataValidatorSchema",
    "ConfigValidatorSchema",
    "SecurityValidatorSchema",
    "ComplianceValidatorSchema",
    "PerformanceValidatorSchema",
    
    # Tool Schemas
    "ToolConfigSchema",
    "ToolExecutionSchema",
    "ToolResultSchema",
    "ToolParameterSchema",
    "ToolMetadataSchema",
    "ToolSecuritySchema",
    "ToolLoggingSchema",
    "ToolMonitoringSchema",
    "AutomationToolSchema",
    "DeploymentToolSchema",
    "MonitoringToolSchema",
    "SecurityToolSchema",
    
    # Enterprise Configuration Schemas
    "ConfigPriorityLevel",
    "ComplianceStandard",
    "EncryptionAlgorithm",
    "ConfigurationSourceSchema",
    "ConfigurationEncryptionSchema",
    "ConfigurationComplianceSchema",
    "ConfigurationBackupSchema",
    "ConfigurationValidationSchema",
    "ConfigurationDeploymentSchema",
    "EnterpriseConfigurationSchema",
    "ConfigurationChangelogSchema",
    
    # Localization Schemas
    "SupportedLocale",
    "TextDirection",
    "DateFormat",
    "TimeFormat",
    "NumberFormat",
    "CurrencyPosition",
    "LocaleConfigurationSchema",
    "TranslationSchema",
    "MessageLocalizationSchema",
    "CulturalAdaptationSchema",
    "AITranslationConfigSchema",
    "LocalizationConfigurationSchema",
    
    # Automation Schemas
    "ToolCategory",
    "ExecutionStrategy",
    "ToolStatus",
    "Priority",
    "ToolConfigurationSchema",
    "WorkflowStepSchema",
    "WorkflowSchema",
    "ExecutionContextSchema",
    "ExecutionResultSchema",
    "MaintenanceWindowSchema",
    "AutomationToolsConfigSchema",
    
    # Observability Schemas
    "MetricType",
    "AlertSeverity",
    "DataRetentionPolicy",
    "SamplingStrategy",
    "MetricDefinitionSchema",
    "ObservabilityAlertRuleSchema",
    "ObservabilityDashboardConfigSchema",
    "TracingConfigSchema",
    "ObservabilityLoggingConfigSchema",
    "AIInsightsConfigSchema",
    "ObservabilityConfigSchema"
]

# Version and metadata
__version__ = "2.1.0"
__author__ = "Fahed Mlaiel"
__description__ = "Enterprise-grade schemas for multi-tenant configuration management"

# Schema registry for dynamic loading and validation
SCHEMA_REGISTRY = {
    # Alert schemas
    "alert_rule": AlertRuleSchema,
    "alert_group": AlertGroupSchema,
    "alert_manager_config": AlertManagerConfigSchema,
    
    # Monitoring schemas  
    "monitoring_config": MonitoringConfigSchema,
    "metric_config": MetricConfigSchema,
    "dashboard_config": DashboardConfigSchema,
    
    # Slack schemas
    "slack_config": SlackConfigSchema,
    "slack_message": SlackMessageSchema,
    "slack_template": SlackTemplateSchema,
    
    # Tenant schemas
    "tenant_config": TenantConfigSchema,
    "tenant_environment": TenantEnvironmentSchema,
    "tenant_resource": TenantResourceSchema,
    
    # Enterprise schemas
    "enterprise_config": EnterpriseConfigurationSchema,
    "configuration_changelog": ConfigurationChangelogSchema,
    
    # Localization schemas
    "locale_config": LocaleConfigurationSchema,
    "localization_config": LocalizationConfigurationSchema,
    "translation": TranslationSchema,
    
    # Automation schemas
    "tool_config": ToolConfigurationSchema,
    "workflow": WorkflowSchema,
    "execution_context": ExecutionContextSchema,
    "execution_result": ExecutionResultSchema,
    
    # Observability schemas
    "metric_definition": MetricDefinitionSchema,
    "observability_config": ObservabilityConfigSchema,
    "tracing_config": TracingConfigSchema,
    "ai_insights_config": AIInsightsConfigSchema
}

def get_schema_by_name(schema_name: str):
    """Get schema class by name from registry."""
    return SCHEMA_REGISTRY.get(schema_name)

def list_available_schemas():
    """List all available schema names."""
    return list(SCHEMA_REGISTRY.keys())

def validate_with_schema(schema_name: str, data: dict):
    """Validate data with specified schema."""
    schema_class = get_schema_by_name(schema_name)
    if not schema_class:
        raise ValueError(f"Schema '{schema_name}' not found in registry")
    return schema_class(**data)
    "SecurityToolSchema"
]
