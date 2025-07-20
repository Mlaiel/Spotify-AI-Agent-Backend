"""
🎯 Schémas Pydantic Avancés - Spotify AI Agent
═══════════════════════════════════════════════════════

Module principal des schémas de validation et sérialisation
pour le système d'alerting et monitoring IA de Spotify AI Agent.

Architecture modulaire avec support:
✅ Validation stricte type-safe
✅ Multi-tenant avec isolation
✅ ML/IA pour prédictions d'alertes
✅ Notifications multi-canal
✅ Métriques et monitoring avancés
✅ API REST avec pagination
✅ Sécurité et audit intégrés

Développé par Fahed Mlaiel - Lead Developer & AI Architect
"""

# Imports de base
from .base import (
    BaseSchema,
    BaseResponse,
    BaseError,
    ValidationError,
    BusinessError,
    SystemError,
    TimestampMixin,
    TenantMixin,
    MetadataMixin,
    AuditMixin,
    SoftDeleteMixin,
    SecurityMixin,
    GeolocationMixin,
    CacheableMixin,
    PerformanceMixin
)

# Imports des énumérations
from .base.enums import (
    AlertLevel,
    AlertStatus,
    WarningCategory,
    NotificationChannel,
    TenantStatus,
    SecurityLevel,
    Environment,
    MetricType,
    TimeUnit,
    DataSize,
    ProcessingStatus,
    Priority,
    SYSTEM_CONSTANTS,
    VALIDATION_PATTERNS,
    DEFAULT_ERROR_MESSAGES
)

# Imports des alertes
from .alerts import (
    AlertRule,
    AlertInstance,
    AlertSummary,
    AlertGroup,
    AlertFilter,
    AlertMetrics,
    SystemMetrics,
    ApplicationMetrics,
    MetricThreshold,
    MetricAggregation,
    MetricValue,
    MetricSeries
)

# Imports des notifications
from .notifications import (
    NotificationTemplate,
    NotificationMessage,
    NotificationStatus,
    SlackNotificationConfig,
    EmailNotificationConfig,
    WebhookNotificationConfig,
    NotificationChannel as NotificationChannelConfig,
    NotificationBatch
)

# Imports ML/IA
from .ml import (
    MLModel,
    MLModelType,
    MLFramework,
    ModelStatus,
    DataDriftStatus,
    AlertPrediction,
    AnomalyDetectionResult,
    PatternAnalysis,
    MLPipeline,
    FeatureStore,
    ModelMonitoring,
    AutoMLExperiment
)

# Imports validation
from .validation import (
    ValidationRules,
    DataSanitizer,
    SpotifyDomainValidators,
    AudioProcessingValidators,
    PerformanceValidators,
    SecurityValidationRules,
    ComplianceValidators,
    DatabaseConfig,
    CacheConfig,
    SecurityConfig,
    LoggingConfig,
    MonitoringConfig,
    PerformanceConfig,
    EnvironmentConfig,
    ConfigurationTemplate,
    DeploymentConfig,
    validate_tenant_id_field,
    validate_alert_message_field,
    validate_metadata_field,
    validate_tags_field,
    validate_severity_score_field,
    validate_spotify_track_id,
    validate_spotify_artist_id,
    validate_audio_features_field,
    validate_latency_metrics_field,
    validate_throughput_metrics_field,
    validate_password_field,
    validate_api_key_field,
    validate_ip_address_field
)

# Version et métadonnées
__version__ = "2.0.0"
__author__ = "Fahed Mlaiel"
__email__ = "fahed.mlaiel@spotify-ai-agent.com"
__description__ = "Schémas Pydantic avancés pour Spotify AI Agent"

# Schémas principaux exportés
__all__ = [
    # Base
    "BaseSchema",
    "BaseResponse", 
    "BaseError",
    "ValidationError",
    "BusinessError",
    "SystemError",
    
    # Mixins
    "TimestampMixin",
    "TenantMixin", 
    "MetadataMixin",
    "AuditMixin",
    "SoftDeleteMixin",
    "SecurityMixin",
    "GeolocationMixin",
    "CacheableMixin",
    "PerformanceMixin",
    
    # Énumérations
    "AlertLevel",
    "AlertStatus",
    "WarningCategory", 
    "NotificationChannel",
    "TenantStatus",
    "SecurityLevel",
    "Environment",
    "MetricType",
    "TimeUnit",
    "DataSize",
    "ProcessingStatus",
    "Priority",
    
    # Alertes
    "AlertRule",
    "AlertInstance",
    "AlertSummary",
    "AlertGroup", 
    "AlertFilter",
    "AlertMetrics",
    "SystemMetrics",
    "ApplicationMetrics",
    "MetricThreshold",
    "MetricAggregation",
    "MetricValue",
    "MetricSeries",
    
    # Notifications
    "NotificationTemplate",
    "NotificationMessage",
    "NotificationStatus",
    "SlackNotificationConfig",
    "EmailNotificationConfig", 
    "WebhookNotificationConfig",
    "NotificationChannelConfig",
    "NotificationBatch",
    
    # ML/IA
    "MLModel",
    "MLModelType",
    "MLFramework",
    "ModelStatus",
    "DataDriftStatus",
    "AlertPrediction",
    "AnomalyDetectionResult",
    "PatternAnalysis",
    "MLPipeline",
    "FeatureStore",
    "ModelMonitoring",
    "AutoMLExperiment",
    
    # Validation
    "ValidationRules",
    "DataSanitizer",
    "SpotifyDomainValidators",
    "AudioProcessingValidators",
    "PerformanceValidators",
    "SecurityValidationRules",
    "ComplianceValidators",
    "DatabaseConfig",
    "CacheConfig",
    "SecurityConfig",
    "LoggingConfig",
    "MonitoringConfig",
    "PerformanceConfig",
    "EnvironmentConfig",
    "ConfigurationTemplate",
    "DeploymentConfig",
    "validate_tenant_id_field",
    "validate_alert_message_field",
    "validate_metadata_field",
    "validate_tags_field",
    "validate_severity_score_field",
    "validate_spotify_track_id",
    "validate_spotify_artist_id",
    "validate_audio_features_field",
    "validate_latency_metrics_field",
    "validate_throughput_metrics_field",
    "validate_password_field",
    "validate_api_key_field",
    "validate_ip_address_field",    # Constantes
    "SYSTEM_CONSTANTS",
    "VALIDATION_PATTERNS", 
    "DEFAULT_ERROR_MESSAGES"
]

# Configuration globale
SCHEMA_CONFIG = {
    "version": __version__,
    "strict_validation": True,
    "auto_generate_docs": True,
    "export_openapi": True,
    "enable_caching": True,
    "tenant_isolation": True,
    "audit_enabled": True,
    "performance_monitoring": True
}


class SchemaRegistry:
    """Registre central des schémas"""
    
    _schemas = {}
    _initialized = False
    
    @classmethod
    def register_schema(cls, name: str, schema_class):
        """Enregistre un schéma"""
        cls._schemas[name] = schema_class
    
    @classmethod
    def get_schema(cls, name: str):
        """Récupère un schéma"""
        return cls._schemas.get(name)
    
    @classmethod
    def list_schemas(cls):
        """Liste tous les schémas"""
        return list(cls._schemas.keys())
    
    @classmethod
    def initialize(cls):
        """Initialise le registre"""
        if cls._initialized:
            return
        
        # Enregistrement automatique des schémas principaux
        schemas_to_register = [
            ("AlertInstance", AlertInstance),
            ("AlertRule", AlertRule),
            ("AlertGroup", AlertGroup),
            ("NotificationMessage", NotificationMessage),
            ("NotificationTemplate", NotificationTemplate),
            ("MLModel", MLModel),
            ("AlertPrediction", AlertPrediction),
            ("SystemMetrics", SystemMetrics),
            ("ApplicationMetrics", ApplicationMetrics)
        ]
        
        for name, schema_class in schemas_to_register:
            cls.register_schema(name, schema_class)
        
        cls._initialized = True


# Initialisation automatique
SchemaRegistry.initialize()


def get_schema_info():
    """Retourne les informations sur les schémas"""
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "total_schemas": len(__all__),
        "config": SCHEMA_CONFIG,
        "registered_schemas": SchemaRegistry.list_schemas()
    }


def validate_schema_compatibility():
    """Valide la compatibilité des schémas"""
    try:
        # Test de création d'instances basiques
        test_alert = AlertInstance(
            rule_id="123e4567-e89b-12d3-a456-426614174000",
            alert_id="test_alert",
            title="Test Alert",
            message="Test message",
            level=AlertLevel.WARNING,
            category=WarningCategory.PERFORMANCE,
            tenant_id="test_tenant"
        )
        
        test_notification = NotificationMessage(
            channel=NotificationChannel.SLACK,
            recipients=["test@example.com"],
            body="Test notification",
            tenant_id="test_tenant"
        )
        
        test_model = MLModel(
            name="test_model",
            model_type=MLModelType.CLASSIFICATION,
            framework=MLFramework.TENSORFLOW,
            version="1.0.0",
            tenant_id="test_tenant"
        )
        
        return True, "All schemas are compatible"
    
    except Exception as e:
        return False, f"Schema compatibility error: {e}"


# Auto-validation au chargement du module
_is_compatible, _compatibility_message = validate_schema_compatibility()
if not _is_compatible:
    import warnings
    warnings.warn(f"Schema compatibility issue: {_compatibility_message}")


# Utilitaires pour développeurs
def create_sample_alert() -> AlertInstance:
    """Crée un exemple d'alerte pour les tests"""
    return AlertInstance(
        rule_id="123e4567-e89b-12d3-a456-426614174000",
        alert_id="sample_alert_001",
        title="Exemple d'Alerte Critique",
        message="Utilisation CPU supérieure à 90% détectée sur le serveur de production",
        level=AlertLevel.CRITICAL,
        category=WarningCategory.PERFORMANCE,
        tenant_id="spotify_production",
        service_name="api_gateway",
        environment=Environment.PRODUCTION,
        severity_score=0.9,
        confidence_score=0.95
    )


def create_sample_notification() -> NotificationMessage:
    """Crée un exemple de notification pour les tests"""
    return NotificationMessage(
        channel=NotificationChannel.SLACK,
        recipients=["devops@spotify.com"],
        subject="🚨 Alerte Critique - Serveur Production",
        body="Une alerte critique a été détectée. Intervention immédiate requise.",
        tenant_id="spotify_production",
        priority=Priority.CRITICAL
    )


def create_sample_ml_model() -> MLModel:
    """Crée un exemple de modèle ML pour les tests"""
    return MLModel(
        name="alert_classifier_v2",
        description="Modèle de classification automatique des alertes",
        model_type=MLModelType.CLASSIFICATION,
        framework=MLFramework.TENSORFLOW,
        version="2.1.0",
        tenant_id="spotify_ai",
        is_production=True,
        deployment_environment=Environment.PRODUCTION,
        accuracy=0.94,
        precision=0.92,
        recall=0.96,
        f1_score=0.94,
        target_classes=["critical", "high", "warning", "info", "debug"]
    )

