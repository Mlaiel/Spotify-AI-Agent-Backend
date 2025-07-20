"""
🎵 Spotify AI Agent - API Exception Management
=============================================

Système de gestion d'exceptions enterprise avec hiérarchie complète,
codes d'erreur structurés, et handlers automatiques.

Architecture:
- Hiérarchie d'exceptions métier
- Codes d'erreur standardisés
- Context d'erreur enrichi
- Handlers automatiques
- Logging structuré
- Traçabilité complète

Développé par Fahed Mlaiel - Enterprise Exception Management Expert
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.status import *


class ErrorCode(str, Enum):
    """Codes d'erreur standardisés"""
    
    # Erreurs génériques
    INTERNAL_ERROR = "INTERNAL_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    
    # Erreurs de validation
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_INPUT = "INVALID_INPUT"
    MISSING_FIELD = "MISSING_FIELD"
    INVALID_FORMAT = "INVALID_FORMAT"
    
    # Erreurs d'authentification
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    INVALID_TOKEN = "INVALID_TOKEN"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    INVALID_CREDENTIALS = "INVALID_CREDENTIALS"
    
    # Erreurs d'autorisation
    AUTHORIZATION_FAILED = "AUTHORIZATION_FAILED"
    INSUFFICIENT_PERMISSIONS = "INSUFFICIENT_PERMISSIONS"
    ACCESS_DENIED = "ACCESS_DENIED"
    FORBIDDEN_RESOURCE = "FORBIDDEN_RESOURCE"
    
    # Erreurs de ressources
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    RESOURCE_ALREADY_EXISTS = "RESOURCE_ALREADY_EXISTS"
    RESOURCE_CONFLICT = "RESOURCE_CONFLICT"
    RESOURCE_LOCKED = "RESOURCE_LOCKED"
    
    # Erreurs de rate limiting
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    
    # Erreurs de cache
    CACHE_ERROR = "CACHE_ERROR"
    CACHE_MISS = "CACHE_MISS"
    CACHE_TIMEOUT = "CACHE_TIMEOUT"
    
    # Erreurs de base de données
    DATABASE_ERROR = "DATABASE_ERROR"
    DATABASE_CONNECTION_ERROR = "DATABASE_CONNECTION_ERROR"
    DATABASE_TIMEOUT = "DATABASE_TIMEOUT"
    TRANSACTION_ERROR = "TRANSACTION_ERROR"
    
    # Erreurs de services externes
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"
    SPOTIFY_API_ERROR = "SPOTIFY_API_ERROR"
    OPENAI_API_ERROR = "OPENAI_API_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    
    # Erreurs métier Spotify
    PLAYLIST_NOT_FOUND = "PLAYLIST_NOT_FOUND"
    TRACK_NOT_FOUND = "TRACK_NOT_FOUND"
    USER_NOT_FOUND = "USER_NOT_FOUND"
    INVALID_PLAYLIST = "INVALID_PLAYLIST"
    
    # Erreurs ML/IA
    MODEL_ERROR = "MODEL_ERROR"
    PREDICTION_FAILED = "PREDICTION_FAILED"
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
    INFERENCE_TIMEOUT = "INFERENCE_TIMEOUT"


class ErrorSeverity(str, Enum):
    """Niveaux de sévérité des erreurs"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class APIException(Exception):
    """Exception de base pour l'API"""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        status_code: int = HTTP_500_INTERNAL_SERVER_ERROR,
        details: Dict[str, Any] = None,
        user_message: str = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        is_retryable: bool = False,
        context: Dict[str, Any] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        self.user_message = user_message or self._get_default_user_message()
        self.severity = severity
        self.is_retryable = is_retryable
        self.context = context or {}
        
        # Métadonnées d'erreur
        self.error_id = str(uuid.uuid4())
        self.timestamp = datetime.now(timezone.utc)
        
        super().__init__(self.message)
    
    def _get_default_user_message(self) -> str:
        """Message par défaut pour l'utilisateur"""
        user_messages = {
            ErrorCode.INTERNAL_ERROR: "Une erreur interne est survenue. Veuillez réessayer.",
            ErrorCode.VALIDATION_ERROR: "Les données fournies ne sont pas valides.",
            ErrorCode.AUTHENTICATION_FAILED: "Authentification échouée. Vérifiez vos identifiants.",
            ErrorCode.AUTHORIZATION_FAILED: "Vous n'avez pas les permissions nécessaires.",
            ErrorCode.RESOURCE_NOT_FOUND: "La ressource demandée n'a pas été trouvée.",
            ErrorCode.RATE_LIMIT_EXCEEDED: "Trop de requêtes. Veuillez attendre avant de réessayer.",
            ErrorCode.SERVICE_UNAVAILABLE: "Le service est temporairement indisponible."
        }
        return user_messages.get(self.error_code, "Une erreur est survenue.")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'exception en dictionnaire"""
        return {
            "error_id": self.error_id,
            "error_code": self.error_code,
            "message": self.message,
            "user_message": self.user_message,
            "status_code": self.status_code,
            "severity": self.severity,
            "is_retryable": self.is_retryable,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "context": self.context
        }


class ValidationException(APIException):
    """Exception de validation"""
    
    def __init__(
        self,
        message: str,
        field: str = None,
        value: Any = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if field:
            details['field'] = field
        if value is not None:
            details['value'] = str(value)
        
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
            details=details,
            severity=ErrorSeverity.LOW,
            **kwargs
        )


class AuthenticationException(APIException):
    """Exception d'authentification"""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(
            message=message,
            error_code=ErrorCode.AUTHENTICATION_FAILED,
            status_code=HTTP_401_UNAUTHORIZED,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )


class AuthorizationException(APIException):
    """Exception d'autorisation"""
    
    def __init__(self, message: str = "Authorization failed", **kwargs):
        super().__init__(
            message=message,
            error_code=ErrorCode.AUTHORIZATION_FAILED,
            status_code=HTTP_403_FORBIDDEN,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )


class ResourceNotFoundException(APIException):
    """Exception ressource non trouvée"""
    
    def __init__(
        self,
        resource_type: str,
        resource_id: str = None,
        **kwargs
    ):
        message = f"{resource_type} not found"
        if resource_id:
            message += f" (ID: {resource_id})"
        
        details = {'resource_type': resource_type}
        if resource_id:
            details['resource_id'] = resource_id
        
        super().__init__(
            message=message,
            error_code=ErrorCode.RESOURCE_NOT_FOUND,
            status_code=HTTP_404_NOT_FOUND,
            details=details,
            severity=ErrorSeverity.LOW,
            **kwargs
        )


class RateLimitException(APIException):
    """Exception de rate limiting"""
    
    def __init__(
        self,
        limit: int = None,
        window: str = None,
        retry_after: int = None,
        **kwargs
    ):
        message = "Rate limit exceeded"
        details = {}
        
        if limit:
            details['limit'] = limit
            message += f" (limit: {limit}"
        if window:
            details['window'] = window
            message += f"/{window}"
        if limit:
            message += ")"
        if retry_after:
            details['retry_after'] = retry_after
        
        super().__init__(
            message=message,
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            details=details,
            severity=ErrorSeverity.MEDIUM,
            is_retryable=True,
            **kwargs
        )


class CacheException(APIException):
    """Exception de cache"""
    
    def __init__(self, message: str = "Cache error", **kwargs):
        super().__init__(
            message=message,
            error_code=ErrorCode.CACHE_ERROR,
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            severity=ErrorSeverity.LOW,
            is_retryable=True,
            **kwargs
        )


class DatabaseException(APIException):
    """Exception de base de données"""
    
    def __init__(self, message: str = "Database error", **kwargs):
        super().__init__(
            message=message,
            error_code=ErrorCode.DATABASE_ERROR,
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            severity=ErrorSeverity.HIGH,
            is_retryable=True,
            **kwargs
        )


class ExternalServiceException(APIException):
    """Exception de service externe"""
    
    def __init__(
        self,
        service_name: str,
        message: str = None,
        upstream_status: int = None,
        **kwargs
    ):
        message = message or f"{service_name} service error"
        details = {'service_name': service_name}
        
        if upstream_status:
            details['upstream_status'] = upstream_status
        
        super().__init__(
            message=message,
            error_code=ErrorCode.EXTERNAL_SERVICE_ERROR,
            status_code=HTTP_502_BAD_GATEWAY,
            details=details,
            severity=ErrorSeverity.MEDIUM,
            is_retryable=True,
            **kwargs
        )


class SpotifyAPIException(ExternalServiceException):
    """Exception spécifique à l'API Spotify"""
    
    def __init__(self, message: str = None, **kwargs):
        super().__init__(
            service_name="Spotify",
            message=message or "Spotify API error",
            **kwargs
        )
        # Surcharger l'error_code après l'init parent
        self.error_code = ErrorCode.SPOTIFY_API_ERROR


class ModelException(APIException):
    """Exception liée aux modèles ML"""
    
    def __init__(
        self,
        model_name: str = None,
        message: str = "Model error",
        **kwargs
    ):
        details = {}
        if model_name:
            details['model_name'] = model_name
            message = f"Model '{model_name}' error"
        
        super().__init__(
            message=message,
            error_code=ErrorCode.MODEL_ERROR,
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


# =============================================================================
# GESTIONNAIRES D'EXCEPTIONS
# =============================================================================

async def api_exception_handler(request: Request, exc: APIException) -> JSONResponse:
    """Gestionnaire pour les exceptions API personnalisées"""
    from .context import get_request_context
    
    # Enrichir le contexte d'erreur
    context = get_request_context()
    if context:
        context.set_error(exc, exc.user_message)
        exc.context.update({
            'request_id': context.request_id,
            'correlation_id': context.correlation_id,
            'user_id': context.user.user_id if context.user else None
        })
    else:
        # Si pas de contexte (ex: dans les tests), utiliser des valeurs par défaut
        exc.context.update({
            'request_id': 'test-request-id',
            'correlation_id': 'test-correlation-id',
            'user_id': None
        })
    
    # Log selon la sévérité
    from app.core.logging import get_logger
    logger = get_logger(__name__)
    
    # Préparer les données de log sans conflit de clés
    log_data = {k: v for k, v in exc.to_dict().items() if k != 'message'}
    
    if exc.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
        logger.error(f"API Exception: {exc.message}", extra=log_data)
    elif exc.severity == ErrorSeverity.MEDIUM:
        logger.warning(f"API Exception: {exc.message}", extra=log_data)
    else:
        logger.info(f"API Exception: {exc.message}", extra=log_data)
    
    # Réponse JSON
    response_data = {
        "error": {
            "code": exc.error_code,
            "message": exc.user_message,
            "error_id": exc.error_id,
            "timestamp": exc.timestamp.isoformat()
        }
    }
    
    # Ajouter les détails en mode debug
    try:
        from .config import get_api_config
        debug_mode = get_api_config().debug
    except Exception:
        # Si impossible d'obtenir la config (ex: dans les tests), assumer debug=False
        debug_mode = False
    
    if debug_mode:
        response_data["error"]["details"] = exc.details
        response_data["error"]["technical_message"] = exc.message
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response_data,
        headers={
            "X-Error-ID": exc.error_id,
            "X-Request-ID": exc.context.get('request_id', ''),
            "X-Correlation-ID": exc.context.get('correlation_id', '')
        }
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Gestionnaire pour les exceptions HTTP FastAPI"""
    # Convertir en APIException pour traitement uniforme
    api_exc = APIException(
        message=str(exc.detail),
        status_code=exc.status_code,
        user_message=str(exc.detail)
    )
    
    return await api_exception_handler(request, api_exc)


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Gestionnaire pour toutes les autres exceptions"""
    # Convertir en APIException
    api_exc = APIException(
        message=str(exc),
        error_code=ErrorCode.UNKNOWN_ERROR,
        severity=ErrorSeverity.CRITICAL
    )
    
    return await api_exception_handler(request, api_exc)


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def register_exception_handlers(app):
    """Enregistre tous les gestionnaires d'exceptions"""
    app.add_exception_handler(APIException, api_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)


def raise_not_found(resource_type: str, resource_id: str = None):
    """Helper pour lever une exception ressource non trouvée"""
    raise ResourceNotFoundException(resource_type, resource_id)


def raise_validation_error(message: str, field: str = None, value: Any = None):
    """Helper pour lever une exception de validation"""
    raise ValidationException(message, field, value)


def raise_auth_error(message: str = None):
    """Helper pour lever une exception d'authentification"""
    raise AuthenticationException(message)


def raise_permission_error(message: str = None):
    """Helper pour lever une exception d'autorisation"""
    raise AuthorizationException(message)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ErrorCode",
    "ErrorSeverity",
    "APIException",
    "ValidationException",
    "AuthenticationException",
    "AuthorizationException",
    "ResourceNotFoundException",
    "RateLimitException",
    "CacheException",
    "DatabaseException",
    "ExternalServiceException",
    "SpotifyAPIException",
    "ModelException",
    "api_exception_handler",
    "http_exception_handler",
    "general_exception_handler",
    "register_exception_handlers",
    "raise_not_found",
    "raise_validation_error",
    "raise_auth_error",
    "raise_permission_error"
]
