"""
Exceptions personnalisées - Spotify AI Agent
Hiérarchie d'exceptions structurée avec contexte et traçabilité
"""

import traceback
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Type, Callable
from uuid import uuid4, UUID
from enum import Enum
import json
import logging

from .enums import AlertLevel, Priority, SecurityLevel


class ErrorCategory(str, Enum):
    """Catégories d'erreurs système"""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    BUSINESS_LOGIC = "business_logic"
    DATA_INTEGRITY = "data_integrity"
    EXTERNAL_SERVICE = "external_service"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CONFIGURATION = "configuration"
    SYSTEM = "system"
    NETWORK = "network"
    DATABASE = "database"
    CACHE = "cache"
    FILE_SYSTEM = "file_system"
    SERIALIZATION = "serialization"
    SECURITY = "security"
    PERFORMANCE = "performance"
    UNKNOWN = "unknown"


class ErrorSeverity(str, Enum):
    """Niveaux de sévérité des erreurs"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    FATAL = "fatal"


class ErrorContext:
    """Contexte détaillé d'une erreur"""
    
    def __init__(self,
                 user_id: Optional[str] = None,
                 tenant_id: Optional[str] = None,
                 request_id: Optional[str] = None,
                 session_id: Optional[str] = None,
                 operation: Optional[str] = None,
                 component: Optional[str] = None,
                 additional_data: Optional[Dict[str, Any]] = None):
        self.error_id = str(uuid4())
        self.timestamp = datetime.now(timezone.utc)
        self.user_id = user_id
        self.tenant_id = tenant_id
        self.request_id = request_id
        self.session_id = session_id
        self.operation = operation
        self.component = component
        self.additional_data = additional_data or {}
        self.stack_trace = self._capture_stack_trace()
        self.system_info = self._capture_system_info()
    
    def _capture_stack_trace(self) -> List[Dict[str, Any]]:
        """Capture la pile d'appels"""
        stack = []
        for frame_info in traceback.extract_tb(sys.exc_info()[2]):
            stack.append({
                'filename': frame_info.filename,
                'line_number': frame_info.lineno,
                'function_name': frame_info.name,
                'code': frame_info.line
            })
        return stack
    
    def _capture_system_info(self) -> Dict[str, Any]:
        """Capture les informations système"""
        return {
            'python_version': sys.version,
            'platform': sys.platform,
            'thread_id': id(sys.current_frame())
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit le contexte en dictionnaire"""
        return {
            'error_id': self.error_id,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'tenant_id': self.tenant_id,
            'request_id': self.request_id,
            'session_id': self.session_id,
            'operation': self.operation,
            'component': self.component,
            'additional_data': self.additional_data,
            'stack_trace': self.stack_trace,
            'system_info': self.system_info
        }
    
    def add_data(self, key: str, value: Any):
        """Ajoute des données au contexte"""
        self.additional_data[key] = value
    
    def get_data(self, key: str, default: Any = None) -> Any:
        """Récupère une donnée du contexte"""
        return self.additional_data.get(key, default)


class BaseSpotifyException(Exception):
    """Exception de base pour toutes les exceptions personnalisées"""
    
    def __init__(self,
                 message: str,
                 error_code: Optional[str] = None,
                 category: ErrorCategory = ErrorCategory.UNKNOWN,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Optional[ErrorContext] = None,
                 original_exception: Optional[Exception] = None,
                 user_message: Optional[str] = None,
                 recoverable: bool = True,
                 retry_after: Optional[int] = None):
        
        super().__init__(message)
        
        self.message = message
        self.error_code = error_code or self._generate_error_code()
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext()
        self.original_exception = original_exception
        self.user_message = user_message or self._generate_user_message()
        self.recoverable = recoverable
        self.retry_after = retry_after
        self.logged = False
        
        # Enrichir le contexte avec des informations sur l'exception
        self.context.add_data('exception_class', self.__class__.__name__)
        self.context.add_data('error_code', self.error_code)
        self.context.add_data('category', self.category.value)
        self.context.add_data('severity', self.severity.value)
        
        if original_exception:
            self.context.add_data('original_exception', {
                'type': type(original_exception).__name__,
                'message': str(original_exception)
            })
    
    def _generate_error_code(self) -> str:
        """Génère un code d'erreur unique"""
        return f"{self.__class__.__name__.upper()}_{str(uuid4())[:8]}"
    
    def _generate_user_message(self) -> str:
        """Génère un message utilisateur par défaut"""
        if self.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]:
            return "Une erreur critique s'est produite. Veuillez contacter le support."
        elif self.severity == ErrorSeverity.HIGH:
            return "Une erreur importante s'est produite. Veuillez réessayer ou contacter le support."
        else:
            return "Une erreur s'est produite. Veuillez réessayer."
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'exception en dictionnaire pour logging/API"""
        return {
            'error_code': self.error_code,
            'message': self.message,
            'user_message': self.user_message,
            'category': self.category.value,
            'severity': self.severity.value,
            'recoverable': self.recoverable,
            'retry_after': self.retry_after,
            'context': self.context.to_dict(),
            'class': self.__class__.__name__
        }
    
    def log_error(self, logger: Optional[logging.Logger] = None):
        """Log l'erreur avec le niveau approprié"""
        if self.logged:
            return
        
        if logger is None:
            logger = logging.getLogger(__name__)
        
        log_data = self.to_dict()
        
        if self.severity == ErrorSeverity.FATAL:
            logger.critical("Fatal error occurred", extra=log_data)
        elif self.severity == ErrorSeverity.CRITICAL:
            logger.critical("Critical error occurred", extra=log_data)
        elif self.severity == ErrorSeverity.HIGH:
            logger.error("High severity error occurred", extra=log_data)
        elif self.severity == ErrorSeverity.MEDIUM:
            logger.warning("Medium severity error occurred", extra=log_data)
        else:
            logger.info("Low severity error occurred", extra=log_data)
        
        self.logged = True
    
    def add_context(self, key: str, value: Any):
        """Ajoute du contexte à l'exception"""
        self.context.add_data(key, value)
        return self
    
    def with_user_message(self, message: str):
        """Définit un message utilisateur personnalisé"""
        self.user_message = message
        return self
    
    def as_recoverable(self, recoverable: bool = True, retry_after: Optional[int] = None):
        """Marque l'exception comme récupérable ou non"""
        self.recoverable = recoverable
        self.retry_after = retry_after
        return self


# Exceptions de validation
class ValidationError(BaseSpotifyException):
    """Erreur de validation des données"""
    
    def __init__(self, message: str, field_name: Optional[str] = None, 
                 field_value: Optional[Any] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        
        if field_name:
            self.add_context('field_name', field_name)
        if field_value is not None:
            self.add_context('field_value', field_value)


class SchemaValidationError(ValidationError):
    """Erreur de validation de schéma"""
    
    def __init__(self, schema_errors: List[Dict[str, Any]], **kwargs):
        message = f"Schema validation failed with {len(schema_errors)} errors"
        super().__init__(message=message, **kwargs)
        self.add_context('schema_errors', schema_errors)


class BusinessRuleViolationError(BaseSpotifyException):
    """Violation de règle métier"""
    
    def __init__(self, rule_name: str, **kwargs):
        message = f"Business rule violation: {rule_name}"
        super().__init__(
            message=message,
            category=ErrorCategory.BUSINESS_LOGIC,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        self.add_context('rule_name', rule_name)


# Exceptions d'authentification et autorisation
class AuthenticationError(BaseSpotifyException):
    """Erreur d'authentification"""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            user_message="Échec de l'authentification. Veuillez vérifier vos identifiants.",
            **kwargs
        )


class AuthorizationError(BaseSpotifyException):
    """Erreur d'autorisation"""
    
    def __init__(self, message: str = "Access denied", required_permission: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.HIGH,
            user_message="Accès refusé. Vous n'avez pas les permissions nécessaires.",
            **kwargs
        )
        
        if required_permission:
            self.add_context('required_permission', required_permission)


class TokenExpiredError(AuthenticationError):
    """Token d'authentification expiré"""
    
    def __init__(self, **kwargs):
        super().__init__(
            message="Authentication token has expired",
            user_message="Votre session a expiré. Veuillez vous reconnecter.",
            **kwargs
        )


# Exceptions de données
class DataIntegrityError(BaseSpotifyException):
    """Erreur d'intégrité des données"""
    
    def __init__(self, message: str, entity_type: Optional[str] = None, entity_id: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.DATA_INTEGRITY,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        
        if entity_type:
            self.add_context('entity_type', entity_type)
        if entity_id:
            self.add_context('entity_id', entity_id)


class ResourceNotFoundError(BaseSpotifyException):
    """Ressource introuvable"""
    
    def __init__(self, resource_type: str, resource_id: Optional[str] = None, **kwargs):
        message = f"{resource_type} not found"
        if resource_id:
            message += f" (ID: {resource_id})"
        
        super().__init__(
            message=message,
            category=ErrorCategory.DATA_INTEGRITY,
            severity=ErrorSeverity.MEDIUM,
            user_message="La ressource demandée est introuvable.",
            **kwargs
        )
        
        self.add_context('resource_type', resource_type)
        if resource_id:
            self.add_context('resource_id', resource_id)


class DuplicateResourceError(BaseSpotifyException):
    """Ressource en double"""
    
    def __init__(self, resource_type: str, conflicting_field: Optional[str] = None, **kwargs):
        message = f"Duplicate {resource_type}"
        if conflicting_field:
            message += f" (field: {conflicting_field})"
        
        super().__init__(
            message=message,
            category=ErrorCategory.DATA_INTEGRITY,
            severity=ErrorSeverity.MEDIUM,
            user_message="Cette ressource existe déjà.",
            **kwargs
        )
        
        self.add_context('resource_type', resource_type)
        if conflicting_field:
            self.add_context('conflicting_field', conflicting_field)


# Exceptions de services externes
class ExternalServiceError(BaseSpotifyException):
    """Erreur de service externe"""
    
    def __init__(self, service_name: str, status_code: Optional[int] = None, **kwargs):
        message = f"External service error: {service_name}"
        if status_code:
            message += f" (status: {status_code})"
        
        super().__init__(
            message=message,
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.HIGH,
            user_message="Un service externe est temporairement indisponible.",
            retry_after=60,
            **kwargs
        )
        
        self.add_context('service_name', service_name)
        if status_code:
            self.add_context('status_code', status_code)


class APIRateLimitError(ExternalServiceError):
    """Limite de taux API atteinte"""
    
    def __init__(self, service_name: str, reset_time: Optional[datetime] = None, **kwargs):
        super().__init__(
            service_name=service_name,
            message=f"API rate limit exceeded for {service_name}",
            user_message="Trop de requêtes. Veuillez réessayer plus tard.",
            **kwargs
        )
        
        if reset_time:
            self.add_context('reset_time', reset_time.isoformat())
            self.retry_after = int((reset_time - datetime.now(timezone.utc)).total_seconds())


class SpotifyAPIError(ExternalServiceError):
    """Erreur spécifique à l'API Spotify"""
    
    def __init__(self, error_code: str, **kwargs):
        super().__init__(
            service_name="Spotify API",
            message=f"Spotify API error: {error_code}",
            **kwargs
        )
        self.add_context('spotify_error_code', error_code)


# Exceptions de ressources
class ResourceExhaustionError(BaseSpotifyException):
    """Épuisement de ressources"""
    
    def __init__(self, resource_type: str, **kwargs):
        super().__init__(
            message=f"Resource exhaustion: {resource_type}",
            category=ErrorCategory.RESOURCE_EXHAUSTION,
            severity=ErrorSeverity.CRITICAL,
            user_message="Ressources système insuffisantes. Veuillez réessayer plus tard.",
            retry_after=300,
            **kwargs
        )
        self.add_context('resource_type', resource_type)


class MemoryExhaustionError(ResourceExhaustionError):
    """Épuisement de mémoire"""
    
    def __init__(self, **kwargs):
        super().__init__(resource_type="memory", **kwargs)


class StorageExhaustionError(ResourceExhaustionError):
    """Épuisement d'espace de stockage"""
    
    def __init__(self, **kwargs):
        super().__init__(resource_type="storage", **kwargs)


# Exceptions de configuration
class ConfigurationError(BaseSpotifyException):
    """Erreur de configuration"""
    
    def __init__(self, config_key: str, **kwargs):
        super().__init__(
            message=f"Configuration error: {config_key}",
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.CRITICAL,
            recoverable=False,
            **kwargs
        )
        self.add_context('config_key', config_key)


class MissingConfigurationError(ConfigurationError):
    """Configuration manquante"""
    
    def __init__(self, config_key: str, **kwargs):
        super().__init__(
            config_key=config_key,
            message=f"Missing required configuration: {config_key}",
            **kwargs
        )


# Exceptions de sécurité
class SecurityError(BaseSpotifyException):
    """Erreur de sécurité"""
    
    def __init__(self, security_violation: str, **kwargs):
        super().__init__(
            message=f"Security violation: {security_violation}",
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.CRITICAL,
            user_message="Violation de sécurité détectée.",
            recoverable=False,
            **kwargs
        )
        self.add_context('security_violation', security_violation)


class InjectionAttemptError(SecurityError):
    """Tentative d'injection détectée"""
    
    def __init__(self, injection_type: str, **kwargs):
        super().__init__(
            security_violation=f"{injection_type} injection attempt",
            **kwargs
        )
        self.add_context('injection_type', injection_type)


# Gestionnaire d'exceptions global
class ExceptionHandler:
    """Gestionnaire centralisé des exceptions"""
    
    def __init__(self):
        self.error_callbacks: Dict[Type[Exception], List[Callable]] = {}
        self.global_callbacks: List[Callable] = []
    
    def register_callback(self, exception_type: Type[Exception], callback: Callable):
        """Enregistre un callback pour un type d'exception"""
        if exception_type not in self.error_callbacks:
            self.error_callbacks[exception_type] = []
        self.error_callbacks[exception_type].append(callback)
    
    def register_global_callback(self, callback: Callable):
        """Enregistre un callback global pour toutes les exceptions"""
        self.global_callbacks.append(callback)
    
    def handle_exception(self, exception: Exception, context: Optional[ErrorContext] = None):
        """Gère une exception avec les callbacks appropriés"""
        # Convertir en exception personnalisée si nécessaire
        if not isinstance(exception, BaseSpotifyException):
            exception = self._wrap_exception(exception, context)
        
        # Log l'exception
        exception.log_error()
        
        # Exécuter les callbacks spécifiques
        exception_type = type(exception)
        callbacks = self.error_callbacks.get(exception_type, [])
        
        for callback in callbacks:
            try:
                callback(exception)
            except Exception as callback_error:
                logging.error(f"Error in exception callback: {callback_error}")
        
        # Exécuter les callbacks globaux
        for callback in self.global_callbacks:
            try:
                callback(exception)
            except Exception as callback_error:
                logging.error(f"Error in global exception callback: {callback_error}")
    
    def _wrap_exception(self, exception: Exception, context: Optional[ErrorContext] = None) -> BaseSpotifyException:
        """Encapsule une exception standard dans notre hiérarchie"""
        return BaseSpotifyException(
            message=str(exception),
            original_exception=exception,
            context=context,
            category=self._categorize_exception(exception)
        )
    
    def _categorize_exception(self, exception: Exception) -> ErrorCategory:
        """Catégorise automatiquement une exception"""
        exception_name = type(exception).__name__.lower()
        
        if 'validation' in exception_name or 'value' in exception_name:
            return ErrorCategory.VALIDATION
        elif 'auth' in exception_name:
            return ErrorCategory.AUTHENTICATION
        elif 'permission' in exception_name or 'access' in exception_name:
            return ErrorCategory.AUTHORIZATION
        elif 'connection' in exception_name or 'network' in exception_name:
            return ErrorCategory.NETWORK
        elif 'database' in exception_name or 'sql' in exception_name:
            return ErrorCategory.DATABASE
        elif 'file' in exception_name or 'io' in exception_name:
            return ErrorCategory.FILE_SYSTEM
        elif 'memory' in exception_name:
            return ErrorCategory.RESOURCE_EXHAUSTION
        else:
            return ErrorCategory.SYSTEM


# Instance globale du gestionnaire d'exceptions
global_exception_handler = ExceptionHandler()


__all__ = [
    'ErrorCategory', 'ErrorSeverity', 'ErrorContext', 'BaseSpotifyException',
    'ValidationError', 'SchemaValidationError', 'BusinessRuleViolationError',
    'AuthenticationError', 'AuthorizationError', 'TokenExpiredError',
    'DataIntegrityError', 'ResourceNotFoundError', 'DuplicateResourceError',
    'ExternalServiceError', 'APIRateLimitError', 'SpotifyAPIError',
    'ResourceExhaustionError', 'MemoryExhaustionError', 'StorageExhaustionError',
    'ConfigurationError', 'MissingConfigurationError',
    'SecurityError', 'InjectionAttemptError',
    'ExceptionHandler', 'global_exception_handler'
]
