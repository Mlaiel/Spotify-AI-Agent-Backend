"""
🎵 Spotify AI Agent - Advanced Middleware System
===============================================

Module d'importation centralisé pour tous les middleware avancés.
Système complet de middleware pour authentification, sécurité, 
monitoring, I18N, rate limiting et gestion d'erreurs.

Architecture:
- Middleware d'authentification OAuth2 + JWT
- Middleware I18N multilingue (25+ langues)
- Middleware de sécurité avancé
- Middleware de monitoring et logging
- Middleware de rate limiting intelligent
- Middleware de gestion d'erreurs contextualisées
- Middleware de performance et métriques

Author: Lead Dev + Architecte IA
License: Proprietary - Spotify AI Agent
"""

from .auth_middleware import (
    AuthTokenData,
    SpotifyAuthData,
    AuthenticationMiddleware,
    SpotifyAuthMiddleware,
    JWTAuthMiddleware,
    RoleBasedAuthMiddleware,
    APIKeyAuthMiddleware,
)

from .i18n_middleware import (
    InternationalizationMiddleware,
    LanguageDetectionMiddleware,
    TranslationCacheMiddleware,
    RTLSupportMiddleware
)

from .rate_limiting import (
    RateLimitingMiddleware,
    AdaptiveRateLimitMiddleware,
    UserTierRateLimitMiddleware,
    APIEndpointRateLimitMiddleware,
    SpotifyAPIRateLimitMiddleware
)

from .logging_middleware import (
    AdvancedLoggingMiddleware,
    RequestTracingMiddleware,
    PerformanceLoggingMiddleware,
    SecurityAuditMiddleware,
    BusinessMetricsMiddleware
)

from .error_handler import (
    AdvancedErrorHandler,
    ErrorClassifier,
    CircuitBreaker,
    ErrorSeverity,
    ErrorCategory,
    ErrorPattern,
    RecoveryStrategy,
    create_error_handler,
    setup_error_handlers
)

from .security_headers import (
    SecurityHeadersMiddleware,
    SecurityLevel,
    CSPBuilder,
    ThreatIntelligence,
    create_security_middleware,
    create_development_security,
    create_production_security
)

from .cors_middleware import (
    AdvancedCORSMiddleware,
    CORSSecurityLevel,
    OriginType,
    OriginConfig,
    CORSMetrics,
    CORSViolation,
    ContextPropagator,
    context_propagator,
    CORSConfigManager,
    create_development_cors_middleware,
    create_production_cors_middleware,
    create_testing_cors_middleware
)

from .request_id_middleware import (
    RequestIDMiddleware,
    IDFormat,
    ContextType,
    TraceContext,
    RequestJourney,
    SnowflakeIDGenerator,
    NanoIDGenerator,
    ULIDGenerator,
    RequestIDGenerator,
    JourneyTracker,
    create_request_id_middleware_development,
    create_request_id_middleware_production,
    create_request_id_middleware_testing
)

from .performance_monitor import (
    PerformanceMonitorMiddleware,
    PerformanceLevel,
    AlertSeverity,
    PerformanceMetrics,
    SystemHealthMetrics,
    PerformanceProfiler,
    AnomalyDetector,
    PerformanceOptimizer,
    create_performance_monitor_development,
    create_performance_monitor_production,
    create_performance_monitor_testing
)

from .cache_middleware import (
    AdvancedCacheMiddleware,
    CacheLevel,
    CacheStrategy,
    CachePattern,
    CacheConfig,
    CacheKey,
    CacheItem,
    CacheMetrics,
    AdvancedCacheManager,
    create_cache_middleware,
    create_production_cache_config,
    create_development_cache_config
)

from .monitoring_middleware import (
    AdvancedMonitoringMiddleware,
    MetricType,
    AlertSeverity,
    HealthStatus,
    MetricDefinition,
    AlertRule,
    TraceSpan,
    PerformanceMetrics as MonitoringPerformanceMetrics,
    RequestMetrics,
    MetricsRegistry,
    DistributedTracer,
    SystemMonitor,
    AlertManager,
    create_monitoring_middleware,
    create_production_monitoring,
    create_development_monitoring
)

from .security_audit_middleware import (
    AdvancedSecurityAuditMiddleware,
    ThreatLevel,
    SecurityEventType,
    ComplianceStandard,
    RiskScore,
    SecurityEvent,
    ThreatIndicator,
    UserBehaviorProfile,
    ComplianceRule,
    ThreatDetectionEngine,
    ComplianceMonitor,
    SecurityAuditLogger,
    create_security_audit_middleware,
    create_production_security_middleware,
    create_development_security_middleware
)

from .data_pipeline_middleware import (
    AdvancedDataPipelineMiddleware,
    DataFormat,
    ProcessingMode,
    DataQuality,
    PipelineStage,
    DataSchema,
    DataRecord,
    PipelineMetrics,
    DataTransformer,
    MessageQueueManager,
    DataPipelineProcessor,
    create_data_pipeline_middleware,
    create_production_pipeline,
    create_development_pipeline
)

from .security_headers import (
    SecurityHeadersMiddleware,
    SecurityLevel,
    ViolationType,
    ThreatLevel,
    ThreatIntelligence,
    GeoValidator,
    RateLimiter,
    CSPBuilder,
    create_security_middleware,
    create_development_security,
    create_production_security,
    create_paranoid_security
)

from .error_handler import (
    AdvancedErrorHandler,
    ErrorCategory,
    ErrorSeverity,
    ErrorContext,
    ErrorClassifier,
    ErrorAlerting,
    ErrorRecovery,
    CircuitBreaker,
    create_error_handler,
    setup_error_handlers,
    error_handler_decorator
)

from .performance_monitor import (
    AdvancedPerformanceMonitor,
    PerformanceMetrics,
    PerformanceLevel,
    PerformanceThresholds,
    PerformanceAnalyzer,
    ResourceMonitor,
    PerformanceProfiler,
    PrometheusMetrics,
    create_performance_monitor,
    performance_monitor_decorator
)

from .request_id_middleware import (
    AdvancedRequestIdMiddleware,
    RequestContext,
    RequestIdFormat,
    RequestIdGenerator,
    RequestTracker,
    RequestCorrelation,
    RequestMetrics,
    create_request_id_middleware,
    request_context_decorator,
    get_current_request_id
)

from .cors_middleware import (
    AdvancedCorsMiddleware,
    CorsConfig,
    CorsPolicy,
    OriginRule,
    OriginType,
    OriginValidator,
    SecurityValidator,
    CorsAnalytics,
    create_cors_middleware,
    create_development_cors,
    create_production_cors
)

# Ordre d'exécution optimisé des middleware
ENTERPRISE_MIDDLEWARE_STACK = [
    # 1. Identification et traçage des requêtes
    (RequestIDMiddleware, {"id_format": IDFormat.SNOWFLAKE, "enable_journey_tracking": True}),
    
    # 2. Sécurité de base et CORS
    (SecurityHeadersMiddleware, {"security_level": "production"}),
    (AdvancedCORSMiddleware, {"default_security_level": CORSSecurityLevel.STRICT}),
    
    # 3. Monitoring et performance
    (PerformanceMonitorMiddleware, {"enable_profiling": True, "enable_anomaly_detection": True}),
    (AdvancedMonitoringMiddleware, {"enable_distributed_tracing": True}),
    
    # 4. Cache multi-niveaux
    (AdvancedCacheMiddleware, {"enable_l1": True, "enable_l2": True, "enable_l3": True}),
    
    # 5. Pipeline de données
    (AdvancedDataPipelineMiddleware, {"enable_streaming": True, "enable_batch": True}),
    
    # 6. Audit de sécurité
    (AdvancedSecurityAuditMiddleware, {"enable_threat_detection": True, "compliance_frameworks": ["GDPR", "SOX"]}),
    
    # 7. Rate limiting et protection
    (RateLimitingMiddleware, {"algorithm": "adaptive"}),
    (AdaptiveRateLimitMiddleware, {"enable_ml": True}),
    
    # 8. Authentification et autorisation
    (AuthenticationMiddleware, {"enable_session_management": True}),
    (SpotifyAuthMiddleware, {"enable_token_refresh": True}),
    
    # 9. Localisation et internationalisation
    (InternationalizationMiddleware, {"supported_languages": 25, "enable_rtl": True}),
    
    # 10. Logging avancé
    (AdvancedLoggingMiddleware, {"enable_structured_logging": True, "enable_audit": True}),
    
    # 11. Gestion d'erreurs (doit être en dernier)
    (AdvancedErrorHandler, {"enable_circuit_breaker": True, "enable_recovery": True}),
]

# Stack de middleware par environnement
MIDDLEWARE_STACKS = {
    "development": [
        (RequestIDMiddleware, create_request_id_middleware_development),
        (AdvancedCORSMiddleware, create_development_cors_middleware),
        (PerformanceMonitorMiddleware, create_performance_monitor_development),
        (AdvancedCacheMiddleware, create_development_cache_config),
        (AdvancedMonitoringMiddleware, create_development_monitoring),
        (AdvancedSecurityAuditMiddleware, create_development_security_middleware),
        (AdvancedDataPipelineMiddleware, create_development_pipeline),
    ],
    
    "production": [
        (RequestIDMiddleware, create_request_id_middleware_production),
        (AdvancedCORSMiddleware, create_production_cors_middleware),
        (PerformanceMonitorMiddleware, create_performance_monitor_production),
        (AdvancedCacheMiddleware, create_production_cache_config),
        (AdvancedMonitoringMiddleware, create_production_monitoring),
        (AdvancedSecurityAuditMiddleware, create_production_security_middleware),
        (AdvancedDataPipelineMiddleware, create_production_pipeline),
    ],
    
    "testing": [
        (RequestIDMiddleware, create_request_id_middleware_testing),
        (AdvancedCORSMiddleware, create_testing_cors_middleware),
        (PerformanceMonitorMiddleware, create_performance_monitor_testing),
    ]
}

# Configuration des middleware par environnement
ENTERPRISE_MIDDLEWARE_CONFIG = {
    "development": {
        "request_id": {
            "id_format": IDFormat.UUID4,
            "enable_journey_tracking": True,
            "enable_profiling": True
        },
        "cors": {
            "security_level": CORSSecurityLevel.DEVELOPMENT,
            "enable_analytics": True,
            "enable_caching": False
        },
        "performance": {
            "enable_profiling": True,
            "enable_anomaly_detection": False,
            "alert_thresholds": {"response_time": 5.0, "memory_usage": 1000.0}
        },
        "cache": {
            "l1_size": 100,
            "l2_ttl": 300,
            "enable_compression": False,
            "enable_encryption": False
        },
        "monitoring": {
            "enable_distributed_tracing": True,
            "sampling_rate": 1.0,
            "enable_profiling": True
        },
        "security": {
            "audit_level": "INFO",
            "enable_threat_detection": False,
            "compliance_frameworks": []
        },
        "data_pipeline": {
            "enable_streaming": True,
            "buffer_size": 100,
            "enable_validation": True
        }
    },
    
    "production": {
        "request_id": {
            "id_format": IDFormat.SNOWFLAKE,
            "enable_journey_tracking": True,
            "enable_profiling": False
        },
        "cors": {
            "security_level": CORSSecurityLevel.STRICT,
            "enable_analytics": True,
            "enable_caching": True
        },
        "performance": {
            "enable_profiling": False,
            "enable_anomaly_detection": True,
            "alert_thresholds": {"response_time": 1.0, "memory_usage": 200.0}
        },
        "cache": {
            "l1_size": 1000,
            "l2_ttl": 3600,
            "enable_compression": True,
            "enable_encryption": True
        },
        "monitoring": {
            "enable_distributed_tracing": True,
            "sampling_rate": 0.1,
            "enable_profiling": False
        },
        "security": {
            "audit_level": "HIGH",
            "enable_threat_detection": True,
            "compliance_frameworks": ["GDPR", "SOX", "HIPAA"]
        },
        "data_pipeline": {
            "enable_streaming": True,
            "buffer_size": 10000,
            "enable_validation": True
        }
    }
}

# Métriques et KPIs des middleware enterprise
ENTERPRISE_MIDDLEWARE_METRICS = {
    "performance": [
        "request_duration_seconds",
        "middleware_processing_time",
        "memory_usage_mb",
        "cpu_usage_percent",
        "cache_hit_ratio",
        "database_query_time",
        "ai_processing_time",
        "anomalies_detected"
    ],
    "security": [
        "failed_auth_attempts",
        "rate_limit_violations", 
        "security_header_violations",
        "suspicious_requests",
        "cors_violations",
        "threat_detections",
        "compliance_violations",
        "blocked_ips"
    ],
    "business": [
        "api_requests_by_endpoint",
        "user_activity_patterns",
        "feature_usage_stats",
        "error_rates_by_type",
        "spotify_api_calls",
        "ai_model_accuracy",
        "user_conversion_rate",
        "revenue_per_user"
    ],
    "reliability": [
        "circuit_breaker_trips",
        "error_recovery_success",
        "auto_healing_events",
        "sla_compliance",
        "availability_percentage",
        "mttr_minutes",
        "mtbf_hours"
    ]
}

# Factory functions pour créer les stacks middleware
def create_middleware_stack(environment: str = "production"):
    """Créer un stack de middleware pour un environnement donné"""
    if environment not in MIDDLEWARE_STACKS:
        raise ValueError(f"Environnement non supporté: {environment}")
    
    middleware_stack = []
    for middleware_class, factory_func in MIDDLEWARE_STACKS[environment]:
        if callable(factory_func):
            middleware_instance = factory_func()
        else:
            middleware_instance = middleware_class(**factory_func)
        middleware_stack.append(middleware_instance)
    
    return middleware_stack


def apply_middleware_to_app(app, environment: str = "production"):
    """Appliquer tous les middleware à une application FastAPI"""
    middleware_stack = create_middleware_stack(environment)
    
    # Appliquer les middleware dans l'ordre inverse (FastAPI les exécute en ordre LIFO)
    for middleware in reversed(middleware_stack):
        app.add_middleware(type(middleware), **middleware.__dict__)


# Utilitaires de configuration
def get_middleware_config(environment: str, middleware_name: str):
    """Obtenir la configuration d'un middleware spécifique"""
    config = ENTERPRISE_MIDDLEWARE_CONFIG.get(environment, {})
    return config.get(middleware_name, {})


def validate_middleware_config(config: dict) -> bool:
    """Valider la configuration des middleware"""
    required_sections = [
        "request_id", "cors", "performance", "cache", 
        "monitoring", "security", "data_pipeline"
    ]
    
    for section in required_sections:
        if section not in config:
            return False
    
    return True

# Exports globaux enterprise
__all__ = [
    # Core Advanced Middleware Classes
    "AuthenticationMiddleware",
    "InternationalizationMiddleware", 
    "RateLimitingMiddleware",
    "AdvancedLoggingMiddleware",
    "SecurityHeadersMiddleware",
    "AdvancedErrorHandler",
    "RequestIDMiddleware",
    "AdvancedCORSMiddleware",
    "PerformanceMonitorMiddleware",
    "AdvancedCacheMiddleware",
    "AdvancedMonitoringMiddleware",
    "AdvancedSecurityAuditMiddleware",
    "AdvancedDataPipelineMiddleware",
    
    # Specialized Authentication Middleware
    "SpotifyAuthMiddleware",
    "JWTAuthMiddleware", 
    "RoleBasedAuthMiddleware",
    "APIKeyAuthMiddleware",
    
    # I18N Middleware Components
    "LanguageDetectionMiddleware",
    "TranslationCacheMiddleware",
    "RTLSupportMiddleware",
    
    # Rate Limiting Variants
    "AdaptiveRateLimitMiddleware",
    "UserTierRateLimitMiddleware",
    "APIEndpointRateLimitMiddleware",
    "SpotifyAPIRateLimitMiddleware",
    
    # Logging and Monitoring
    "RequestTracingMiddleware",
    "PerformanceLoggingMiddleware", 
    "SecurityAuditMiddleware",
    "BusinessMetricsMiddleware",
    
    # Security Components
    "CSPBuilder",
    "ThreatIntelligence",
    "SecurityLevel",
    "ViolationType",
    "ThreatLevel",
    
    # CORS Components
    "CORSSecurityLevel",
    "OriginType",
    "OriginConfig",
    "CORSMetrics",
    "CORSViolation",
    "CORSConfigManager",
    
    # Request ID Components
    "IDFormat",
    "ContextType",
    "TraceContext",
    "RequestJourney",
    "SnowflakeIDGenerator",
    "NanoIDGenerator",
    "ULIDGenerator",
    "RequestIDGenerator",
    "JourneyTracker",
    
    # Performance Components
    "PerformanceLevel",
    "AlertSeverity",
    "PerformanceMetrics",
    "SystemHealthMetrics",
    "PerformanceProfiler",
    "AnomalyDetector",
    "PerformanceOptimizer",
    
    # Cache Components
    "CacheLevel",
    "CacheStrategy",
    "CachePattern", 
    "CacheConfig",
    "CacheKey",
    "CacheItem",
    "CacheMetrics",
    "AdvancedCacheManager",
    
    # Monitoring Components
    "MetricType",
    "HealthStatus",
    "MetricDefinition",
    "AlertRule",
    "TraceSpan",
    "RequestMetrics",
    "MetricsRegistry",
    "DistributedTracer",
    "SystemMonitor",
    "AlertManager",
    
    # Security Audit Components
    "ThreatLevel",
    "SecurityEventType",
    "ComplianceStandard",
    "RiskScore",
    "SecurityEvent",
    "ThreatIndicator",
    "UserBehaviorProfile",
    "ComplianceRule",
    "ThreatDetectionEngine",
    "ComplianceMonitor",
    "SecurityAuditLogger",
    
    # Data Pipeline Components
    "DataFormat",
    "ProcessingMode",
    "DataQuality",
    "PipelineStage",
    "DataSchema",
    "DataRecord",
    "PipelineMetrics",
    "DataTransformer",
    "MessageQueueManager",
    "DataPipelineProcessor",
    
    # Error Handling Components
    "ErrorCategory",
    "ErrorSeverity",
    "ErrorContext",
    "ErrorClassifier",
    "ErrorAlerting",
    "ErrorRecovery",
    "CircuitBreaker",
    
    # Factory Functions
    "create_development_cors_middleware",
    "create_production_cors_middleware",
    "create_testing_cors_middleware",
    "create_request_id_middleware_development",
    "create_request_id_middleware_production", 
    "create_request_id_middleware_testing",
    "create_performance_monitor_development",
    "create_performance_monitor_production",
    "create_performance_monitor_testing",
    "create_cache_middleware",
    "create_production_cache_config",
    "create_development_cache_config",
    "create_monitoring_middleware",
    "create_production_monitoring",
    "create_development_monitoring",
    "create_security_audit_middleware",
    "create_production_security_middleware",
    "create_development_security_middleware",
    "create_data_pipeline_middleware",
    "create_production_pipeline",
    "create_development_pipeline",
    "create_security_middleware",
    "create_development_security",
    "create_production_security",
    "create_error_handler",
    "setup_error_handlers",
    
    # Configuration and Utils
    "ENTERPRISE_MIDDLEWARE_STACK",
    "MIDDLEWARE_STACKS",
    "ENTERPRISE_MIDDLEWARE_CONFIG",
    "ENTERPRISE_MIDDLEWARE_METRICS",
    "create_middleware_stack",
    "apply_middleware_to_app",
    "get_middleware_config",
    "validate_middleware_config"
]
