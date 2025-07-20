"""
Exceptions Spécialisées pour le Système de Cache
===============================================

Hiérarchie complète d'exceptions pour une gestion d'erreurs précise
et un debugging avancé du système de cache multi-niveaux.

Chaque exception contient des informations contextuelles détaillées
pour faciliter le monitoring, l'alerting et la résolution d'incidents.

Auteurs: Équipe Spotify AI Agent - Direction technique Fahed Mlaiel
"""

from typing import Any, Dict, Optional, List
from datetime import datetime


class CacheException(Exception):
    """Exception de base pour toutes les erreurs de cache"""
    
    def __init__(self, message: str, error_code: str = None, 
                 context: Dict[str, Any] = None, cause: Exception = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "CACHE_ERROR"
        self.context = context or {}
        self.cause = cause
        self.timestamp = datetime.now()
        self.severity = "ERROR"
    
    def to_dict(self) -> Dict[str, Any]:
        """Sérialise l'exception pour logging et monitoring"""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "cause": str(self.cause) if self.cause else None
        }
    
    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"


class CacheBackendError(CacheException):
    """Erreur au niveau du backend de cache"""
    
    def __init__(self, message: str, backend_type: str = None, 
                 operation: str = None, **kwargs):
        super().__init__(message, "CACHE_BACKEND_ERROR", **kwargs)
        self.backend_type = backend_type
        self.operation = operation
        self.severity = "CRITICAL"
        
        if backend_type:
            self.context["backend_type"] = backend_type
        if operation:
            self.context["operation"] = operation


class CacheMissError(CacheException):
    """Erreur de cache miss critique (quand un hit était attendu)"""
    
    def __init__(self, key: str, tenant_id: str = None, 
                 expected_level: str = None, **kwargs):
        message = f"Critical cache miss for key: {key}"
        super().__init__(message, "CACHE_MISS_CRITICAL", **kwargs)
        self.key = key
        self.tenant_id = tenant_id
        self.expected_level = expected_level
        self.severity = "WARNING"
        
        self.context.update({
            "key": key,
            "tenant_id": tenant_id,
            "expected_level": expected_level
        })


class CacheTimeoutError(CacheException):
    """Timeout lors d'opérations de cache"""
    
    def __init__(self, operation: str, timeout_duration: float, 
                 backend_type: str = None, **kwargs):
        message = f"Cache operation '{operation}' timed out after {timeout_duration}s"
        super().__init__(message, "CACHE_TIMEOUT", **kwargs)
        self.operation = operation
        self.timeout_duration = timeout_duration
        self.backend_type = backend_type
        self.severity = "ERROR"
        
        self.context.update({
            "operation": operation,
            "timeout_duration": timeout_duration,
            "backend_type": backend_type
        })


class CacheSecurityError(CacheException):
    """Erreur de sécurité et permissions dans le cache"""
    
    def __init__(self, message: str, tenant_id: str = None, 
                 operation: str = None, resource: str = None, **kwargs):
        super().__init__(message, "CACHE_SECURITY_ERROR", **kwargs)
        self.tenant_id = tenant_id
        self.operation = operation
        self.resource = resource
        self.severity = "CRITICAL"
        
        self.context.update({
            "tenant_id": tenant_id,
            "operation": operation,
            "resource": resource,
            "security_violation": True
        })


class CacheQuotaExceededError(CacheException):
    """Dépassement de quota pour un tenant"""
    
    def __init__(self, tenant_id: str, current_usage: int = None, 
                 quota_limit: int = None, resource_type: str = "memory", **kwargs):
        message = f"Cache quota exceeded for tenant {tenant_id}"
        super().__init__(message, "CACHE_QUOTA_EXCEEDED", **kwargs)
        self.tenant_id = tenant_id
        self.current_usage = current_usage
        self.quota_limit = quota_limit
        self.resource_type = resource_type
        self.severity = "WARNING"
        
        self.context.update({
            "tenant_id": tenant_id,
            "current_usage": current_usage,
            "quota_limit": quota_limit,
            "resource_type": resource_type,
            "quota_exceeded": True
        })


class CacheCorruptionError(CacheException):
    """Corruption détectée dans les données de cache"""
    
    def __init__(self, key: str, corruption_type: str = "data_integrity", 
                 details: str = None, **kwargs):
        message = f"Cache corruption detected for key {key}: {corruption_type}"
        super().__init__(message, "CACHE_CORRUPTION", **kwargs)
        self.key = key
        self.corruption_type = corruption_type
        self.details = details
        self.severity = "CRITICAL"
        
        self.context.update({
            "key": key,
            "corruption_type": corruption_type,
            "details": details,
            "data_corruption": True
        })


class CacheConfigurationError(CacheException):
    """Erreur de configuration du système de cache"""
    
    def __init__(self, component: str, config_key: str = None, 
                 expected_type: str = None, actual_value: Any = None, **kwargs):
        message = f"Invalid cache configuration for {component}"
        super().__init__(message, "CACHE_CONFIG_ERROR", **kwargs)
        self.component = component
        self.config_key = config_key
        self.expected_type = expected_type
        self.actual_value = actual_value
        self.severity = "CRITICAL"
        
        self.context.update({
            "component": component,
            "config_key": config_key,
            "expected_type": expected_type,
            "actual_value": str(actual_value) if actual_value else None
        })


class CacheSerializationError(CacheException):
    """Erreur de sérialisation/désérialisation"""
    
    def __init__(self, operation: str, serializer_type: str = None, 
                 data_type: str = None, **kwargs):
        message = f"Cache serialization error during {operation}"
        super().__init__(message, "CACHE_SERIALIZATION_ERROR", **kwargs)
        self.operation = operation
        self.serializer_type = serializer_type
        self.data_type = data_type
        self.severity = "ERROR"
        
        self.context.update({
            "operation": operation,
            "serializer_type": serializer_type,
            "data_type": data_type
        })


class CacheCompressionError(CacheException):
    """Erreur de compression/décompression"""
    
    def __init__(self, operation: str, compression_algo: str = None, 
                 original_size: int = None, **kwargs):
        message = f"Cache compression error during {operation}"
        super().__init__(message, "CACHE_COMPRESSION_ERROR", **kwargs)
        self.operation = operation
        self.compression_algo = compression_algo
        self.original_size = original_size
        self.severity = "ERROR"
        
        self.context.update({
            "operation": operation,
            "compression_algo": compression_algo,
            "original_size": original_size
        })


class CacheNetworkError(CacheException):
    """Erreur réseau pour cache distribué"""
    
    def __init__(self, operation: str, node_address: str = None, 
                 network_error_type: str = None, retry_count: int = 0, **kwargs):
        message = f"Cache network error during {operation}"
        super().__init__(message, "CACHE_NETWORK_ERROR", **kwargs)
        self.operation = operation
        self.node_address = node_address
        self.network_error_type = network_error_type
        self.retry_count = retry_count
        self.severity = "ERROR"
        
        self.context.update({
            "operation": operation,
            "node_address": node_address,
            "network_error_type": network_error_type,
            "retry_count": retry_count
        })


class CacheEvictionError(CacheException):
    """Erreur lors de l'éviction de données"""
    
    def __init__(self, eviction_policy: str, keys_affected: List[str] = None, 
                 reason: str = None, **kwargs):
        message = f"Cache eviction error with policy {eviction_policy}"
        super().__init__(message, "CACHE_EVICTION_ERROR", **kwargs)
        self.eviction_policy = eviction_policy
        self.keys_affected = keys_affected or []
        self.reason = reason
        self.severity = "WARNING"
        
        self.context.update({
            "eviction_policy": eviction_policy,
            "keys_affected_count": len(self.keys_affected),
            "reason": reason
        })


class CacheCircuitBreakerError(CacheException):
    """Erreur de circuit breaker"""
    
    def __init__(self, circuit_name: str, state: str, failure_count: int = 0, 
                 last_failure_time: datetime = None, **kwargs):
        message = f"Circuit breaker '{circuit_name}' is {state}"
        super().__init__(message, "CACHE_CIRCUIT_BREAKER", **kwargs)
        self.circuit_name = circuit_name
        self.circuit_state = state
        self.failure_count = failure_count
        self.last_failure_time = last_failure_time
        self.severity = "WARNING" if state == "HALF_OPEN" else "ERROR"
        
        self.context.update({
            "circuit_name": circuit_name,
            "circuit_state": state,
            "failure_count": failure_count,
            "last_failure_time": last_failure_time.isoformat() if last_failure_time else None
        })


class CacheReplicationError(CacheException):
    """Erreur de réplication entre niveaux de cache"""
    
    def __init__(self, source_level: str, target_level: str, 
                 operation: str, **kwargs):
        message = f"Cache replication error from {source_level} to {target_level}"
        super().__init__(message, "CACHE_REPLICATION_ERROR", **kwargs)
        self.source_level = source_level
        self.target_level = target_level
        self.operation = operation
        self.severity = "WARNING"
        
        self.context.update({
            "source_level": source_level,
            "target_level": target_level,
            "operation": operation
        })


class CacheValidationError(CacheException):
    """Erreur de validation des données de cache"""
    
    def __init__(self, key: str, validation_type: str, 
                 expected_checksum: str = None, actual_checksum: str = None, **kwargs):
        message = f"Cache validation failed for key {key}: {validation_type}"
        super().__init__(message, "CACHE_VALIDATION_ERROR", **kwargs)
        self.key = key
        self.validation_type = validation_type
        self.expected_checksum = expected_checksum
        self.actual_checksum = actual_checksum
        self.severity = "ERROR"
        
        self.context.update({
            "key": key,
            "validation_type": validation_type,
            "expected_checksum": expected_checksum,
            "actual_checksum": actual_checksum
        })


class CacheMLError(CacheException):
    """Erreur dans les composants ML du cache"""
    
    def __init__(self, ml_component: str, model_name: str = None, 
                 prediction_error: str = None, **kwargs):
        message = f"Cache ML error in {ml_component}"
        super().__init__(message, "CACHE_ML_ERROR", **kwargs)
        self.ml_component = ml_component
        self.model_name = model_name
        self.prediction_error = prediction_error
        self.severity = "WARNING"
        
        self.context.update({
            "ml_component": ml_component,
            "model_name": model_name,
            "prediction_error": prediction_error
        })


# Utilitaires pour la gestion d'exceptions

class CacheExceptionHandler:
    """Gestionnaire centralisé des exceptions de cache"""
    
    def __init__(self, logger, alerting_system=None):
        self.logger = logger
        self.alerting_system = alerting_system
        self.exception_counts = {}
        self.last_exceptions = {}
    
    def handle_exception(self, exception: CacheException, 
                        context: Dict[str, Any] = None):
        """Traite une exception de cache"""
        # Logging structuré
        log_data = exception.to_dict()
        if context:
            log_data["additional_context"] = context
        
        if exception.severity == "CRITICAL":
            self.logger.critical("Cache critical error", extra=log_data)
        elif exception.severity == "ERROR":
            self.logger.error("Cache error", extra=log_data)
        elif exception.severity == "WARNING":
            self.logger.warning("Cache warning", extra=log_data)
        
        # Comptage des exceptions
        exception_type = exception.__class__.__name__
        self.exception_counts[exception_type] = self.exception_counts.get(exception_type, 0) + 1
        self.last_exceptions[exception_type] = exception
        
        # Alerting si configuré
        if self.alerting_system and exception.severity in ["CRITICAL", "ERROR"]:
            self._send_alert(exception)
    
    def _send_alert(self, exception: CacheException):
        """Envoie une alerte pour l'exception"""
        try:
            alert_data = {
                "alert_type": "cache_exception",
                "severity": exception.severity,
                "exception_data": exception.to_dict(),
                "requires_immediate_attention": exception.severity == "CRITICAL"
            }
            self.alerting_system.send_alert(alert_data)
        except Exception as e:
            self.logger.error(f"Failed to send cache exception alert: {e}")
    
    def get_exception_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des exceptions"""
        return {
            "total_exceptions": sum(self.exception_counts.values()),
            "exception_counts": self.exception_counts.copy(),
            "last_exceptions": {
                exc_type: exc.to_dict() 
                for exc_type, exc in self.last_exceptions.items()
            }
        }


# Décorateurs pour la gestion automatique d'exceptions

def cache_exception_handler(exception_handler: CacheExceptionHandler):
    """Décorateur pour gestion automatique des exceptions de cache"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except CacheException as e:
                exception_handler.handle_exception(e, {
                    "function": func.__name__,
                    "args": str(args),
                    "kwargs": str(kwargs)
                })
                raise
            except Exception as e:
                # Convertit les exceptions génériques en CacheException
                cache_exc = CacheException(
                    f"Unexpected error in {func.__name__}: {str(e)}",
                    "CACHE_UNEXPECTED_ERROR",
                    {"function": func.__name__, "original_error": str(e)},
                    e
                )
                exception_handler.handle_exception(cache_exc)
                raise cache_exc
        return wrapper
    return decorator


def retry_on_cache_error(max_retries: int = 3, delay: float = 1.0, 
                        backoff_factor: float = 2.0):
    """Décorateur pour retry automatique sur erreurs de cache"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except (CacheTimeoutError, CacheNetworkError, CacheBackendError) as e:
                    last_exception = e
                    if attempt < max_retries:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff_factor
                        continue
                    else:
                        raise
                except CacheException:
                    # N'effectue pas de retry pour les autres types d'erreurs
                    raise
            
            # Ne devrait jamais arriver
            raise last_exception
        return wrapper
    return decorator
