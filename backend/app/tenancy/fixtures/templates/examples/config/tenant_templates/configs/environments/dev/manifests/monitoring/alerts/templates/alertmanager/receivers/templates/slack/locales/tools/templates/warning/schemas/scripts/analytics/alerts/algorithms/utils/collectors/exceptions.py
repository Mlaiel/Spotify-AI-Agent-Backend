"""
Spotify AI Agent - Advanced Exceptions Module
============================================

Système d'exceptions ultra-avancé pour les collecteurs de données
avec hiérarchie complète, contexte enrichi, et intégration avec
les systèmes de monitoring et d'alerting.

Classes d'exceptions:
- CollectorException: Exception de base pour tous les collecteurs
- DataCollectionError: Erreurs de collecte de données
- ValidationError: Erreurs de validation de données
- ConnectionError: Erreurs de connexion aux services
- TimeoutError: Erreurs de timeout
- AuthenticationError: Erreurs d'authentification
- AuthorizationError: Erreurs d'autorisation
- ConfigurationError: Erreurs de configuration
- StorageError: Erreurs de stockage
- SerializationError: Erreurs de sérialisation
- CompressionError: Erreurs de compression
- EncryptionError: Erreurs de chiffrement
- CircuitBreakerError: Erreurs de circuit breaker
- RateLimitError: Erreurs de rate limiting
- RetryExhaustedError: Erreurs d'épuisement des tentatives

Fonctionnalités avancées:
- Contexte enrichi avec métadonnées
- Stack traces structurés
- Intégration avec observabilité
- Categorisation automatique
- Suggestions de résolution
- Escalation automatique
"""

from typing import Dict, Any, List, Optional, Union, Type
from datetime import datetime, timezone
import traceback
import sys
import json
import structlog
from enum import Enum
from dataclasses import dataclass, field


class ErrorSeverity(Enum):
    """Niveaux de sévérité des erreurs."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"


class ErrorCategory(Enum):
    """Catégories d'erreurs."""
    CONFIGURATION = "configuration"
    CONNECTION = "connection"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    DATA_COLLECTION = "data_collection"
    DATA_PROCESSING = "data_processing"
    STORAGE = "storage"
    NETWORK = "network"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SYSTEM = "system"
    APPLICATION = "application"
    EXTERNAL_SERVICE = "external_service"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Contexte enrichi pour les erreurs."""
    
    # Identification
    error_id: str = field(default_factory=lambda: f"err_{int(datetime.now().timestamp() * 1000)}")
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Classification
    category: ErrorCategory = ErrorCategory.UNKNOWN
    severity: ErrorSeverity = ErrorSeverity.ERROR
    
    # Localisation
    collector_name: Optional[str] = None
    tenant_id: Optional[str] = None
    environment: str = "unknown"
    service_name: Optional[str] = None
    hostname: Optional[str] = None
    
    # Contexte technique
    function_name: Optional[str] = None
    line_number: Optional[int] = None
    file_path: Optional[str] = None
    stack_trace: Optional[str] = None
    
    # Données contextuelles
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Métadonnées
    tags: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, Union[int, float]] = field(default_factory=dict)
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    # Résolution
    suggested_actions: List[str] = field(default_factory=list)
    documentation_url: Optional[str] = None
    escalation_required: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire pour sérialisation."""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "category": self.category.value,
            "severity": self.severity.value,
            "collector_name": self.collector_name,
            "tenant_id": self.tenant_id,
            "environment": self.environment,
            "service_name": self.service_name,
            "hostname": self.hostname,
            "function_name": self.function_name,
            "line_number": self.line_number,
            "file_path": self.file_path,
            "stack_trace": self.stack_trace,
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "tags": self.tags,
            "metrics": self.metrics,
            "additional_data": self.additional_data,
            "suggested_actions": self.suggested_actions,
            "documentation_url": self.documentation_url,
            "escalation_required": self.escalation_required
        }


class CollectorException(Exception):
    """
    Exception de base pour tous les collecteurs.
    
    Fournit un contexte enrichi, une categorisation automatique,
    et une intégration avec les systèmes d'observabilité.
    """
    
    def __init__(
        self,
        message: str,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        **kwargs
    ):
        super().__init__(message)
        
        self.message = message
        self.cause = cause
        self.context = context or ErrorContext()
        
        # Enrichissement automatique du contexte
        self._enrich_context(**kwargs)
        
        # Capture de la stack trace
        self._capture_stack_trace()
        
        # Logging structuré
        self._log_error()
    
    def _enrich_context(self, **kwargs) -> None:
        """Enrichit automatiquement le contexte de l'erreur."""
        
        # Mise à jour des champs depuis kwargs
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
            else:
                self.context.additional_data[key] = value
        
        # Détection automatique des informations de contexte
        frame = sys._getframe(2)  # Frame de l'appelant
        
        if frame:
            self.context.function_name = frame.f_code.co_name
            self.context.line_number = frame.f_lineno
            self.context.file_path = frame.f_code.co_filename
        
        # Enrichissement depuis l'exception cause
        if self.cause:
            self.context.additional_data["caused_by"] = {
                "type": type(self.cause).__name__,
                "message": str(self.cause),
                "args": self.cause.args if hasattr(self.cause, 'args') else []
            }
    
    def _capture_stack_trace(self) -> None:
        """Capture la stack trace de manière structurée."""
        try:
            tb_lines = traceback.format_exception(
                type(self.cause) if self.cause else type(self),
                self.cause if self.cause else self,
                self.cause.__traceback__ if self.cause else self.__traceback__
            )
            self.context.stack_trace = "".join(tb_lines)
        except Exception:
            self.context.stack_trace = "Stack trace capture failed"
    
    def _log_error(self) -> None:
        """Log l'erreur avec le contexte structuré."""
        logger = structlog.get_logger(__name__)
        
        log_data = {
            "error_type": type(self).__name__,
            "error_message": self.message,
            "error_context": self.context.to_dict()
        }
        
        # Sélection du niveau de log selon la sévérité
        if self.context.severity == ErrorSeverity.FATAL:
            logger.critical("Erreur fatale détectée", **log_data)
        elif self.context.severity == ErrorSeverity.CRITICAL:
            logger.critical("Erreur critique détectée", **log_data)
        elif self.context.severity == ErrorSeverity.ERROR:
            logger.error("Erreur détectée", **log_data)
        elif self.context.severity == ErrorSeverity.WARNING:
            logger.warning("Avertissement détecté", **log_data)
        else:
            logger.info("Information d'erreur", **log_data)
    
    def add_suggestion(self, action: str) -> None:
        """Ajoute une suggestion de résolution."""
        self.context.suggested_actions.append(action)
    
    def set_documentation(self, url: str) -> None:
        """Définit l'URL de documentation pour cette erreur."""
        self.context.documentation_url = url
    
    def require_escalation(self) -> None:
        """Marque cette erreur comme nécessitant une escalation."""
        self.context.escalation_required = True
        if self.context.severity.value in ["debug", "info", "warning"]:
            self.context.severity = ErrorSeverity.ERROR
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion complète en dictionnaire."""
        return {
            "exception_type": type(self).__name__,
            "message": self.message,
            "context": self.context.to_dict(),
            "has_cause": self.cause is not None
        }
    
    def __str__(self) -> str:
        """Représentation string enrichie."""
        return f"{type(self).__name__}: {self.message} [ID: {self.context.error_id}]"
    
    def __repr__(self) -> str:
        """Représentation détaillée."""
        return (
            f"{type(self).__name__}("
            f"message='{self.message}', "
            f"category={self.context.category.value}, "
            f"severity={self.context.severity.value}, "
            f"error_id='{self.context.error_id}'"
            f")"
        )


class DataCollectionError(CollectorException):
    """Erreur lors de la collecte de données."""
    
    def __init__(self, message: str, **kwargs):
        context = kwargs.get('context', ErrorContext())
        context.category = ErrorCategory.DATA_COLLECTION
        context.severity = kwargs.get('severity', ErrorSeverity.ERROR)
        
        # Suggestions automatiques
        context.suggested_actions.extend([
            "Vérifier la connectivité aux sources de données",
            "Valider les permissions d'accès",
            "Contrôler la configuration du collecteur",
            "Examiner les logs des services externes"
        ])
        
        super().__init__(message, context=context, **kwargs)


class ValidationError(CollectorException):
    """Erreur de validation de données."""
    
    def __init__(self, message: str, invalid_data: Optional[Any] = None, **kwargs):
        context = kwargs.get('context', ErrorContext())
        context.category = ErrorCategory.VALIDATION
        context.severity = kwargs.get('severity', ErrorSeverity.WARNING)
        
        if invalid_data is not None:
            context.additional_data["invalid_data"] = str(invalid_data)[:1000]  # Limiter la taille
        
        context.suggested_actions.extend([
            "Vérifier le format des données d'entrée",
            "Valider le schéma des données",
            "Contrôler les types de données",
            "Examiner les règles de validation"
        ])
        
        super().__init__(message, context=context, **kwargs)


class ConnectionError(CollectorException):
    """Erreur de connexion aux services externes."""
    
    def __init__(self, message: str, service_name: Optional[str] = None, **kwargs):
        context = kwargs.get('context', ErrorContext())
        context.category = ErrorCategory.CONNECTION
        context.severity = kwargs.get('severity', ErrorSeverity.ERROR)
        
        if service_name:
            context.service_name = service_name
            context.tags["failed_service"] = service_name
        
        context.suggested_actions.extend([
            "Vérifier la connectivité réseau",
            "Contrôler la disponibilité du service",
            "Examiner les paramètres de connexion",
            "Vérifier les firewalls et proxy",
            "Tester la résolution DNS"
        ])
        
        super().__init__(message, context=context, **kwargs)


class TimeoutError(CollectorException):
    """Erreur de timeout lors d'opérations."""
    
    def __init__(self, message: str, timeout_duration: Optional[float] = None, **kwargs):
        context = kwargs.get('context', ErrorContext())
        context.category = ErrorCategory.TIMEOUT
        context.severity = kwargs.get('severity', ErrorSeverity.WARNING)
        
        if timeout_duration:
            context.metrics["timeout_duration"] = timeout_duration
        
        context.suggested_actions.extend([
            "Augmenter les valeurs de timeout",
            "Optimiser les requêtes lentes",
            "Vérifier la charge du système",
            "Examiner les goulots d'étranglement réseau"
        ])
        
        super().__init__(message, context=context, **kwargs)


class AuthenticationError(CollectorException):
    """Erreur d'authentification."""
    
    def __init__(self, message: str, auth_method: Optional[str] = None, **kwargs):
        context = kwargs.get('context', ErrorContext())
        context.category = ErrorCategory.AUTHENTICATION
        context.severity = kwargs.get('severity', ErrorSeverity.ERROR)
        
        if auth_method:
            context.tags["auth_method"] = auth_method
        
        context.suggested_actions.extend([
            "Vérifier les credentials d'authentification",
            "Contrôler l'expiration des tokens",
            "Examiner la configuration d'authentification",
            "Vérifier les permissions de service"
        ])
        
        super().__init__(message, context=context, **kwargs)


class AuthorizationError(CollectorException):
    """Erreur d'autorisation."""
    
    def __init__(self, message: str, required_permission: Optional[str] = None, **kwargs):
        context = kwargs.get('context', ErrorContext())
        context.category = ErrorCategory.AUTHORIZATION
        context.severity = kwargs.get('severity', ErrorSeverity.ERROR)
        
        if required_permission:
            context.tags["required_permission"] = required_permission
        
        context.suggested_actions.extend([
            "Vérifier les permissions utilisateur",
            "Contrôler les rôles attribués",
            "Examiner les politiques d'accès",
            "Contacter l'administrateur système"
        ])
        
        super().__init__(message, context=context, **kwargs)


class ConfigurationError(CollectorException):
    """Erreur de configuration."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        context = kwargs.get('context', ErrorContext())
        context.category = ErrorCategory.CONFIGURATION
        context.severity = kwargs.get('severity', ErrorSeverity.ERROR)
        
        if config_key:
            context.tags["config_key"] = config_key
        
        context.suggested_actions.extend([
            "Vérifier la syntaxe de configuration",
            "Valider les valeurs de configuration",
            "Contrôler les fichiers de configuration",
            "Examiner la documentation de configuration"
        ])
        
        super().__init__(message, context=context, **kwargs)


class StorageError(CollectorException):
    """Erreur de stockage de données."""
    
    def __init__(self, message: str, storage_backend: Optional[str] = None, **kwargs):
        context = kwargs.get('context', ErrorContext())
        context.category = ErrorCategory.STORAGE
        context.severity = kwargs.get('severity', ErrorSeverity.ERROR)
        
        if storage_backend:
            context.tags["storage_backend"] = storage_backend
        
        context.suggested_actions.extend([
            "Vérifier l'espace disque disponible",
            "Contrôler les permissions d'écriture",
            "Examiner la santé du système de stockage",
            "Vérifier la connectivité au stockage"
        ])
        
        super().__init__(message, context=context, **kwargs)


class SerializationError(CollectorException):
    """Erreur de sérialisation/désérialisation."""
    
    def __init__(self, message: str, serialization_format: Optional[str] = None, **kwargs):
        context = kwargs.get('context', ErrorContext())
        context.category = ErrorCategory.DATA_PROCESSING
        context.severity = kwargs.get('severity', ErrorSeverity.WARNING)
        
        if serialization_format:
            context.tags["format"] = serialization_format
        
        context.suggested_actions.extend([
            "Vérifier le format des données",
            "Contrôler la compatibilité des versions",
            "Examiner l'encodage des caractères",
            "Valider la structure des données"
        ])
        
        super().__init__(message, context=context, **kwargs)


class CompressionError(CollectorException):
    """Erreur de compression/décompression."""
    
    def __init__(self, message: str, compression_type: Optional[str] = None, **kwargs):
        context = kwargs.get('context', ErrorContext())
        context.category = ErrorCategory.DATA_PROCESSING
        context.severity = kwargs.get('severity', ErrorSeverity.WARNING)
        
        if compression_type:
            context.tags["compression_type"] = compression_type
        
        context.suggested_actions.extend([
            "Vérifier l'algorithme de compression",
            "Contrôler l'intégrité des données",
            "Examiner la taille des données",
            "Tester avec différents algorithmes"
        ])
        
        super().__init__(message, context=context, **kwargs)


class EncryptionError(CollectorException):
    """Erreur de chiffrement/déchiffrement."""
    
    def __init__(self, message: str, encryption_algorithm: Optional[str] = None, **kwargs):
        context = kwargs.get('context', ErrorContext())
        context.category = ErrorCategory.DATA_PROCESSING
        context.severity = kwargs.get('severity', ErrorSeverity.ERROR)
        
        if encryption_algorithm:
            context.tags["encryption_algorithm"] = encryption_algorithm
        
        context.suggested_actions.extend([
            "Vérifier les clés de chiffrement",
            "Contrôler l'algorithme de chiffrement",
            "Examiner la configuration de sécurité",
            "Valider les certificats"
        ])
        
        super().__init__(message, context=context, **kwargs)


class CircuitBreakerError(CollectorException):
    """Erreur liée au circuit breaker."""
    
    def __init__(self, message: str, circuit_state: Optional[str] = None, **kwargs):
        context = kwargs.get('context', ErrorContext())
        context.category = ErrorCategory.SYSTEM
        context.severity = kwargs.get('severity', ErrorSeverity.WARNING)
        
        if circuit_state:
            context.tags["circuit_state"] = circuit_state
        
        context.suggested_actions.extend([
            "Attendre la récupération automatique",
            "Examiner les erreurs sous-jacentes",
            "Vérifier la santé des services",
            "Ajuster les seuils du circuit breaker"
        ])
        
        super().__init__(message, context=context, **kwargs)


class RateLimitError(CollectorException):
    """Erreur de limitation de débit."""
    
    def __init__(self, message: str, current_rate: Optional[float] = None, limit: Optional[float] = None, **kwargs):
        context = kwargs.get('context', ErrorContext())
        context.category = ErrorCategory.RESOURCE_EXHAUSTION
        context.severity = kwargs.get('severity', ErrorSeverity.WARNING)
        
        if current_rate:
            context.metrics["current_rate"] = current_rate
        if limit:
            context.metrics["rate_limit"] = limit
        
        context.suggested_actions.extend([
            "Réduire la fréquence des requêtes",
            "Implémenter un backoff exponentiel",
            "Examiner les quotas de l'API",
            "Optimiser les appels groupés"
        ])
        
        super().__init__(message, context=context, **kwargs)


class RetryExhaustedError(CollectorException):
    """Erreur d'épuisement des tentatives de retry."""
    
    def __init__(self, message: str, attempts: Optional[int] = None, **kwargs):
        context = kwargs.get('context', ErrorContext())
        context.category = ErrorCategory.SYSTEM
        context.severity = kwargs.get('severity', ErrorSeverity.ERROR)
        
        if attempts:
            context.metrics["failed_attempts"] = attempts
        
        context.suggested_actions.extend([
            "Examiner les causes des échecs",
            "Augmenter le nombre de tentatives",
            "Modifier la stratégie de retry",
            "Vérifier les conditions préalables"
        ])
        
        super().__init__(message, context=context, **kwargs)


class ResourceExhaustionError(CollectorException):
    """Erreur d'épuisement des ressources."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, **kwargs):
        context = kwargs.get('context', ErrorContext())
        context.category = ErrorCategory.RESOURCE_EXHAUSTION
        context.severity = kwargs.get('severity', ErrorSeverity.CRITICAL)
        
        if resource_type:
            context.tags["resource_type"] = resource_type
        
        context.suggested_actions.extend([
            "Libérer les ressources inutilisées",
            "Augmenter les limites de ressources",
            "Optimiser l'utilisation des ressources",
            "Examiner les fuites de mémoire"
        ])
        
        super().__init__(message, context=context, **kwargs)


class ExternalServiceError(CollectorException):
    """Erreur liée à un service externe."""
    
    def __init__(self, message: str, service_name: Optional[str] = None, status_code: Optional[int] = None, **kwargs):
        context = kwargs.get('context', ErrorContext())
        context.category = ErrorCategory.EXTERNAL_SERVICE
        context.severity = kwargs.get('severity', ErrorSeverity.ERROR)
        
        if service_name:
            context.service_name = service_name
        if status_code:
            context.metrics["status_code"] = status_code
        
        context.suggested_actions.extend([
            "Vérifier le statut du service externe",
            "Contrôler la documentation de l'API",
            "Examiner les limites de l'API",
            "Implémenter un fallback"
        ])
        
        super().__init__(message, context=context, **kwargs)


# Factory functions pour la création d'exceptions typées
def create_data_collection_error(
    message: str,
    collector_name: Optional[str] = None,
    tenant_id: Optional[str] = None,
    **kwargs
) -> DataCollectionError:
    """Crée une erreur de collecte de données avec contexte."""
    context = ErrorContext(
        collector_name=collector_name,
        tenant_id=tenant_id,
        **kwargs
    )
    return DataCollectionError(message, context=context)


def create_connection_error(
    message: str,
    service_name: str,
    host: Optional[str] = None,
    port: Optional[int] = None,
    **kwargs
) -> ConnectionError:
    """Crée une erreur de connexion avec contexte."""
    context = ErrorContext(service_name=service_name, **kwargs)
    if host:
        context.tags["host"] = host
    if port:
        context.tags["port"] = str(port)
    
    return ConnectionError(message, service_name=service_name, context=context)


def create_timeout_error(
    message: str,
    operation: str,
    timeout_duration: float,
    **kwargs
) -> TimeoutError:
    """Crée une erreur de timeout avec contexte."""
    context = ErrorContext(**kwargs)
    context.tags["operation"] = operation
    
    return TimeoutError(message, timeout_duration=timeout_duration, context=context)


def create_validation_error(
    message: str,
    field_name: Optional[str] = None,
    invalid_value: Optional[Any] = None,
    **kwargs
) -> ValidationError:
    """Crée une erreur de validation avec contexte."""
    context = ErrorContext(**kwargs)
    if field_name:
        context.tags["field_name"] = field_name
    
    return ValidationError(message, invalid_data=invalid_value, context=context)


# Gestionnaire d'exceptions global
class ExceptionManager:
    """
    Gestionnaire centralisé des exceptions avec fonctionnalités avancées.
    """
    
    def __init__(self):
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.error_history: deque = deque(maxlen=1000)
        self.escalation_handlers: List[Callable] = []
        self.logger = structlog.get_logger(__name__)
    
    def register_escalation_handler(self, handler: Callable[[CollectorException], None]) -> None:
        """Enregistre un gestionnaire d'escalation."""
        self.escalation_handlers.append(handler)
    
    def handle_exception(self, exception: CollectorException) -> None:
        """Gère une exception avec escalation si nécessaire."""
        
        # Comptage des erreurs
        error_type = type(exception).__name__
        self.error_counts[error_type] += 1
        
        # Ajout à l'historique
        self.error_history.append({
            "timestamp": exception.context.timestamp,
            "type": error_type,
            "message": exception.message,
            "severity": exception.context.severity.value,
            "category": exception.context.category.value,
            "error_id": exception.context.error_id
        })
        
        # Escalation si nécessaire
        if exception.context.escalation_required or exception.context.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]:
            for handler in self.escalation_handlers:
                try:
                    handler(exception)
                except Exception as e:
                    self.logger.error("Erreur lors de l'escalation", error=str(e))
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques d'erreurs."""
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_counts_by_type": dict(self.error_counts),
            "recent_errors": list(self.error_history)[-10:],
            "error_rate": len(self.error_history) / max(1, len(self.error_history))
        }
    
    def clear_statistics(self) -> None:
        """Remet à zéro les statistiques."""
        self.error_counts.clear()
        self.error_history.clear()


# Instance globale du gestionnaire d'exceptions
exception_manager = ExceptionManager()


# Décorateurs pour la gestion d'exceptions
def handle_collector_exceptions(
    reraise: bool = True,
    log_level: str = "error",
    fallback_value: Any = None
):
    """
    Décorateur pour la gestion automatique des exceptions de collecteur.
    
    Args:
        reraise: Si True, relance l'exception après traitement
        log_level: Niveau de log pour l'exception
        fallback_value: Valeur de retour en cas d'exception
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except CollectorException as e:
                exception_manager.handle_exception(e)
                if reraise:
                    raise
                return fallback_value
            except Exception as e:
                # Conversion en CollectorException
                collector_error = CollectorException(
                    f"Erreur inattendue dans {func.__name__}: {str(e)}",
                    cause=e
                )
                exception_manager.handle_exception(collector_error)
                if reraise:
                    raise collector_error
                return fallback_value
        
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except CollectorException as e:
                exception_manager.handle_exception(e)
                if reraise:
                    raise
                return fallback_value
            except Exception as e:
                collector_error = CollectorException(
                    f"Erreur inattendue dans {func.__name__}: {str(e)}",
                    cause=e
                )
                exception_manager.handle_exception(collector_error)
                if reraise:
                    raise collector_error
                return fallback_value
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Fonctions utilitaires
def format_exception_for_display(exception: CollectorException) -> str:
    """Formate une exception pour l'affichage utilisateur."""
    return f"""
Erreur détectée:
  Type: {type(exception).__name__}
  Message: {exception.message}
  Catégorie: {exception.context.category.value}
  Sévérité: {exception.context.severity.value}
  ID: {exception.context.error_id}
  Timestamp: {exception.context.timestamp.isoformat()}

Actions suggérées:
{chr(10).join(f"  - {action}" for action in exception.context.suggested_actions)}

{f"Documentation: {exception.context.documentation_url}" if exception.context.documentation_url else ""}
    """.strip()


def export_error_context_to_json(exception: CollectorException) -> str:
    """Exporte le contexte d'erreur en JSON."""
    return json.dumps(exception.to_dict(), indent=2, default=str)
