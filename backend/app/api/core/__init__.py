"""
🎵 Spotify AI Agent - API Core Module
=====================================

Module central de l'API contenant les composants fondamentaux et la configuration
core de l'architecture enterprise. Ce module fournit une abstraction robuste
pour la gestion des requêtes, responses, middlewares et configuration API.

Architecture:
- Configuration API centralisée
- Gestionnaire de contexte de requête
- Factory patterns pour les composants
- Abstractions pour les middlewares
- Système de métriques et monitoring
- Gestion d'erreurs centralisée

Développé par Fahed Mlaiel - Enterprise API Architecture Expert
"""

from .config import (
    APIConfig,
    APISettings, 
    SecurityConfig,
    CacheConfig,
    DatabaseConfig,
    RedisConfig,
    MonitoringConfig,
    get_api_config,
    get_security_config,
    get_settings,
    api_config
)

from .context import (
    RequestContext,
    APIContext,
    get_request_context,
    set_request_context,
    request_context_middleware
)

from .factory import (
    ComponentFactory,
    MiddlewareFactory,
    ServiceFactory,
    create_api_component,
    create_middleware_stack
)

from .exceptions import (
    APIException,
    ValidationException,
    AuthenticationException,
    AuthorizationException,
    RateLimitException,
    CacheException,
    DatabaseException,
    ExternalServiceException,
    api_exception_handler
)

from .response import (
    APIResponse,
    SuccessResponse,
    ErrorResponse,
    PaginatedResponse,
    create_success_response,
    create_error_response,
    create_paginated_response
)

from .monitoring import (
    APIMetrics,
    PerformanceMonitor,
    HealthChecker,
    get_api_metrics,
    monitor_api_call,
    health_check
)

__version__ = "2.0.0"
__author__ = "Fahed Mlaiel"

__all__ = [
    # Configuration
    "APIConfig",
    "APISettings", 
    "SecurityConfig",
    "CacheConfig",
    "DatabaseConfig",
    "RedisConfig",
    "MonitoringConfig",
    "get_api_config",
    "get_security_config",
    "get_settings",
    "api_config",
    
    # Context Management
    "RequestContext",
    "APIContext",
    "get_request_context",
    "set_request_context",
    "request_context_middleware",
    
    # Factory Patterns
    "ComponentFactory",
    "MiddlewareFactory", 
    "ServiceFactory",
    "create_api_component",
    "create_middleware_stack",
    
    # Exception Handling
    "APIException",
    "ValidationException",
    "AuthenticationException",
    "AuthorizationException",
    "RateLimitException",
    "CacheException",
    "DatabaseException",
    "ExternalServiceException",
    "api_exception_handler",
    
    # Response Management
    "APIResponse",
    "SuccessResponse", 
    "ErrorResponse",
    "PaginatedResponse",
    "create_success_response",
    "create_error_response",
    "create_paginated_response",
    
    # Monitoring & Metrics
    "APIMetrics",
    "PerformanceMonitor",
    "HealthChecker",
    "get_api_metrics",
    "monitor_api_call",
    "health_check"
]
