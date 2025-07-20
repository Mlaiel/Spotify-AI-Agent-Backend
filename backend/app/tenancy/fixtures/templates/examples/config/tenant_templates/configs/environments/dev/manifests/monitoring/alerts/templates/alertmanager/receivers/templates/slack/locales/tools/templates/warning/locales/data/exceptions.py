"""
Exceptions Personnalisées - Spotify AI Agent Localization
========================================================

Définitions des exceptions spécifiques au module de localisation
pour une gestion d'erreurs granulaire et un debugging efficace.

Fonctionnalités:
- Hiérarchie d'exceptions structurée
- Messages d'erreur localisés
- Contexte d'erreur enrichi
- Support de la traçabilité d'erreurs

Author: Fahed Mlaiel
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from enum import Enum

from . import LocaleType


class ErrorSeverity(Enum):
    """Niveaux de sévérité des erreurs"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Catégories d'erreurs"""
    VALIDATION = "validation"
    LOCALIZATION = "localization"
    FORMATTING = "formatting"
    CACHE = "cache"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    SECURITY = "security"
    PERFORMANCE = "performance"


class LocalizationBaseException(Exception):
    """Exception de base pour le module de localisation"""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.LOCALIZATION,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        locale: Optional[LocaleType] = None,
        original_exception: Optional[Exception] = None
    ):
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.locale = locale
        self.original_exception = original_exception
        self.timestamp = datetime.now(timezone.utc)
        
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'exception en dictionnaire pour logging/serialization"""
        return {
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "context": self.context,
            "locale": self.locale.value if self.locale else None,
            "timestamp": self.timestamp.isoformat(),
            "original_exception": str(self.original_exception) if self.original_exception else None
        }


class LocaleValidationError(LocalizationBaseException):
    """Erreur de validation de locale"""
    
    def __init__(
        self,
        message: str,
        invalid_locale: str,
        supported_locales: Optional[List[str]] = None,
        **kwargs
    ):
        self.invalid_locale = invalid_locale
        self.supported_locales = supported_locales or []
        
        context = kwargs.get('context', {})
        context.update({
            'invalid_locale': invalid_locale,
            'supported_locales': self.supported_locales
        })
        
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.HIGH,
            context=context,
            **kwargs
        )


class FormatError(LocalizationBaseException):
    """Erreur de formatage de données"""
    
    def __init__(
        self,
        message: str,
        data_type: str,
        value: Any,
        expected_format: Optional[str] = None,
        **kwargs
    ):
        self.data_type = data_type
        self.value = value
        self.expected_format = expected_format
        
        context = kwargs.get('context', {})
        context.update({
            'data_type': data_type,
            'value': str(value),
            'expected_format': expected_format
        })
        
        super().__init__(
            message,
            category=ErrorCategory.FORMATTING,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            **kwargs
        )


class NumberFormatError(FormatError):
    """Erreur de formatage de nombre"""
    
    def __init__(
        self,
        message: str,
        value: Any,
        locale: Optional[LocaleType] = None,
        **kwargs
    ):
        super().__init__(
            message,
            data_type="number",
            value=value,
            locale=locale,
            **kwargs
        )


class CurrencyConversionError(LocalizationBaseException):
    """Erreur de conversion de devise"""
    
    def __init__(
        self,
        message: str,
        from_currency: str,
        to_currency: str,
        amount: Optional[float] = None,
        provider: Optional[str] = None,
        **kwargs
    ):
        self.from_currency = from_currency
        self.to_currency = to_currency
        self.amount = amount
        self.provider = provider
        
        context = kwargs.get('context', {})
        context.update({
            'from_currency': from_currency,
            'to_currency': to_currency,
            'amount': amount,
            'provider': provider
        })
        
        super().__init__(
            message,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.HIGH,
            context=context,
            **kwargs
        )


class CacheError(LocalizationBaseException):
    """Erreur de cache"""
    
    def __init__(
        self,
        message: str,
        operation: str,
        cache_key: Optional[str] = None,
        cache_level: Optional[str] = None,
        **kwargs
    ):
        self.operation = operation
        self.cache_key = cache_key
        self.cache_level = cache_level
        
        context = kwargs.get('context', {})
        context.update({
            'operation': operation,
            'cache_key': cache_key,
            'cache_level': cache_level
        })
        
        super().__init__(
            message,
            category=ErrorCategory.CACHE,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            **kwargs
        )


class TemplateError(LocalizationBaseException):
    """Erreur de template d'alerte"""
    
    def __init__(
        self,
        message: str,
        template_id: str,
        template_content: Optional[str] = None,
        missing_parameters: Optional[List[str]] = None,
        **kwargs
    ):
        self.template_id = template_id
        self.template_content = template_content
        self.missing_parameters = missing_parameters or []
        
        context = kwargs.get('context', {})
        context.update({
            'template_id': template_id,
            'template_content': template_content,
            'missing_parameters': self.missing_parameters
        })
        
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.HIGH,
            context=context,
            **kwargs
        )


class AlertLocalizationError(LocalizationBaseException):
    """Erreur de localisation d'alerte"""
    
    def __init__(
        self,
        message: str,
        alert_type: str,
        tenant_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.alert_type = alert_type
        self.tenant_id = tenant_id
        self.parameters = parameters or {}
        
        context = kwargs.get('context', {})
        context.update({
            'alert_type': alert_type,
            'tenant_id': tenant_id,
            'parameters': self.parameters
        })
        
        super().__init__(
            message,
            category=ErrorCategory.LOCALIZATION,
            severity=ErrorSeverity.HIGH,
            context=context,
            **kwargs
        )


class ConfigurationError(LocalizationBaseException):
    """Erreur de configuration"""
    
    def __init__(
        self,
        message: str,
        config_key: str,
        config_value: Optional[Any] = None,
        config_source: Optional[str] = None,
        **kwargs
    ):
        self.config_key = config_key
        self.config_value = config_value
        self.config_source = config_source
        
        context = kwargs.get('context', {})
        context.update({
            'config_key': config_key,
            'config_value': str(config_value) if config_value is not None else None,
            'config_source': config_source
        })
        
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            context=context,
            **kwargs
        )


class SecurityValidationError(LocalizationBaseException):
    """Erreur de validation de sécurité"""
    
    def __init__(
        self,
        message: str,
        security_check: str,
        input_data: Optional[str] = None,
        threat_level: Optional[str] = None,
        **kwargs
    ):
        self.security_check = security_check
        self.input_data = input_data
        self.threat_level = threat_level
        
        context = kwargs.get('context', {})
        context.update({
            'security_check': security_check,
            'input_data': input_data[:100] + "..." if input_data and len(input_data) > 100 else input_data,
            'threat_level': threat_level
        })
        
        super().__init__(
            message,
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.CRITICAL,
            context=context,
            **kwargs
        )


class PerformanceError(LocalizationBaseException):
    """Erreur de performance"""
    
    def __init__(
        self,
        message: str,
        operation: str,
        duration_ms: Optional[float] = None,
        threshold_ms: Optional[float] = None,
        **kwargs
    ):
        self.operation = operation
        self.duration_ms = duration_ms
        self.threshold_ms = threshold_ms
        
        context = kwargs.get('context', {})
        context.update({
            'operation': operation,
            'duration_ms': duration_ms,
            'threshold_ms': threshold_ms
        })
        
        super().__init__(
            message,
            category=ErrorCategory.PERFORMANCE,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            **kwargs
        )


class NetworkError(LocalizationBaseException):
    """Erreur réseau"""
    
    def __init__(
        self,
        message: str,
        endpoint: str,
        status_code: Optional[int] = None,
        response_time_ms: Optional[float] = None,
        **kwargs
    ):
        self.endpoint = endpoint
        self.status_code = status_code
        self.response_time_ms = response_time_ms
        
        context = kwargs.get('context', {})
        context.update({
            'endpoint': endpoint,
            'status_code': status_code,
            'response_time_ms': response_time_ms
        })
        
        super().__init__(
            message,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.HIGH,
            context=context,
            **kwargs
        )


def handle_exception_with_context(
    exception: Exception,
    operation: str,
    context: Optional[Dict[str, Any]] = None,
    locale: Optional[LocaleType] = None
) -> LocalizationBaseException:
    """Convertit une exception générique en exception de localisation avec contexte"""
    
    if isinstance(exception, LocalizationBaseException):
        return exception
    
    # Détermine la catégorie d'erreur basée sur le type d'exception
    category = ErrorCategory.LOCALIZATION
    severity = ErrorSeverity.MEDIUM
    
    if isinstance(exception, (ValueError, TypeError)):
        category = ErrorCategory.VALIDATION
        severity = ErrorSeverity.HIGH
    elif isinstance(exception, (ConnectionError, TimeoutError)):
        category = ErrorCategory.NETWORK
        severity = ErrorSeverity.HIGH
    elif isinstance(exception, PermissionError):
        category = ErrorCategory.SECURITY
        severity = ErrorSeverity.CRITICAL
    elif isinstance(exception, MemoryError):
        category = ErrorCategory.PERFORMANCE
        severity = ErrorSeverity.CRITICAL
    
    enhanced_context = context or {}
    enhanced_context.update({
        'operation': operation,
        'exception_type': type(exception).__name__
    })
    
    return LocalizationBaseException(
        message=f"Error in {operation}: {str(exception)}",
        category=category,
        severity=severity,
        context=enhanced_context,
        locale=locale,
        original_exception=exception
    )


# Décorateur pour la gestion automatique des exceptions
def handle_localization_exceptions(operation_name: str):
    """Décorateur pour la gestion automatique des exceptions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except LocalizationBaseException:
                # Laisse passer les exceptions de localisation
                raise
            except Exception as e:
                # Convertit les autres exceptions
                raise handle_exception_with_context(e, operation_name)
        
        # Version async
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except LocalizationBaseException:
                raise
            except Exception as e:
                raise handle_exception_with_context(e, operation_name)
        
        # Retourne la version appropriée
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


__all__ = [
    "ErrorSeverity",
    "ErrorCategory",
    "LocalizationBaseException",
    "LocaleValidationError",
    "FormatError",
    "NumberFormatError",
    "CurrencyConversionError",
    "CacheError",
    "TemplateError",
    "AlertLocalizationError",
    "ConfigurationError",
    "SecurityValidationError",
    "PerformanceError",
    "NetworkError",
    "handle_exception_with_context",
    "handle_localization_exceptions"
]
