# 🎵 ML Analytics Exceptions
# =========================
# 
# Système d'exceptions personnalisées pour ML Analytics
# Gestion d'erreurs enterprise avec détails et logging
#
# 🎖️ Expert: Développeur Backend Senior

"""
🚨 ML Analytics Exception System
================================

Custom exception hierarchy for comprehensive error handling:
- Base ML Analytics exceptions
- Model-specific errors
- Pipeline and processing errors
- Configuration and validation errors
- Resource and performance errors
"""

import logging
import traceback
import sys
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from enum import Enum


class ErrorSeverity(Enum):
    """Niveaux de sévérité des erreurs"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Catégories d'erreurs"""
    MODEL = "model"
    PIPELINE = "pipeline"
    DATA = "data"
    CONFIGURATION = "configuration"
    RESOURCE = "resource"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class MLAnalyticsError(Exception):
    """Exception de base pour ML Analytics"""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        
        self.message = message
        self.error_code = error_code or self._generate_error_code()
        self.severity = severity
        self.category = category
        self.details = details or {}
        self.original_exception = original_exception
        self.timestamp = datetime.utcnow()
        self.traceback_info = self._capture_traceback()
        
        # Logging automatique
        self._log_error()
    
    def _generate_error_code(self) -> str:
        """Génération d'un code d'erreur unique"""
        timestamp = int(self.timestamp.timestamp())
        hash_value = hash(self.message) & 0x7FFFFFFF
        return f"MLA-{timestamp}-{hash_value:08X}"
    
    def _capture_traceback(self) -> str:
        """Capture de la stack trace"""
        return traceback.format_exc()
    
    def _log_error(self):
        """Logging automatique de l'erreur"""
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        log_data = {
            'error_code': self.error_code,
            'severity': self.severity.value,
            'category': self.category.value,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }
        
        if self.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"Erreur critique: {self.message}", extra=log_data)
        elif self.severity == ErrorSeverity.HIGH:
            logger.error(f"Erreur importante: {self.message}", extra=log_data)
        elif self.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"Erreur modérée: {self.message}", extra=log_data)
        else:
            logger.info(f"Erreur mineure: {self.message}", extra=log_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire"""
        return {
            'error_code': self.error_code,
            'message': self.message,
            'severity': self.severity.value,
            'category': self.category.value,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'original_exception': str(self.original_exception) if self.original_exception else None,
            'traceback': self.traceback_info
        }
    
    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code='{self.error_code}', message='{self.message}')"


class ModelError(MLAnalyticsError):
    """Erreurs liées aux modèles ML"""
    
    def __init__(self, message: str, model_id: Optional[str] = None, **kwargs):
        details = kwargs.get('details', {})
        if model_id:
            details['model_id'] = model_id
        
        super().__init__(
            message,
            category=ErrorCategory.MODEL,
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class ModelNotFoundError(ModelError):
    """Modèle non trouvé"""
    
    def __init__(self, model_id: str, **kwargs):
        super().__init__(
            f"Modèle '{model_id}' non trouvé",
            model_id=model_id,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class ModelLoadError(ModelError):
    """Erreur de chargement de modèle"""
    
    def __init__(self, model_id: str, reason: str, **kwargs):
        super().__init__(
            f"Impossible de charger le modèle '{model_id}': {reason}",
            model_id=model_id,
            severity=ErrorSeverity.HIGH,
            details={'reason': reason, **kwargs.get('details', {})},
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class ModelTrainingError(ModelError):
    """Erreur d'entraînement de modèle"""
    
    def __init__(self, model_id: str, epoch: Optional[int] = None, **kwargs):
        message = f"Erreur lors de l'entraînement du modèle '{model_id}'"
        if epoch is not None:
            message += f" à l'époque {epoch}"
        
        details = kwargs.get('details', {})
        if epoch is not None:
            details['epoch'] = epoch
        
        super().__init__(
            message,
            model_id=model_id,
            severity=ErrorSeverity.HIGH,
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class InferenceError(ModelError):
    """Erreur d'inférence"""
    
    def __init__(self, message: str, model_id: Optional[str] = None, **kwargs):
        super().__init__(
            f"Erreur d'inférence: {message}",
            model_id=model_id,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )


class TrainingError(ModelError):
    """Erreur générale d'entraînement"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            f"Erreur d'entraînement: {message}",
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class PipelineError(MLAnalyticsError):
    """Erreurs liées aux pipelines ML"""
    
    def __init__(self, message: str, pipeline_id: Optional[str] = None, step: Optional[str] = None, **kwargs):
        details = kwargs.get('details', {})
        if pipeline_id:
            details['pipeline_id'] = pipeline_id
        if step:
            details['step'] = step
        
        super().__init__(
            message,
            category=ErrorCategory.PIPELINE,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class PipelineStepError(PipelineError):
    """Erreur dans une étape de pipeline"""
    
    def __init__(self, pipeline_id: str, step: str, reason: str, **kwargs):
        super().__init__(
            f"Erreur dans l'étape '{step}' du pipeline '{pipeline_id}': {reason}",
            pipeline_id=pipeline_id,
            step=step,
            details={'reason': reason, **kwargs.get('details', {})},
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class PipelineTimeoutError(PipelineError):
    """Timeout de pipeline"""
    
    def __init__(self, pipeline_id: str, timeout_seconds: float, **kwargs):
        super().__init__(
            f"Timeout du pipeline '{pipeline_id}' après {timeout_seconds} secondes",
            pipeline_id=pipeline_id,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.TIMEOUT,
            details={'timeout_seconds': timeout_seconds, **kwargs.get('details', {})},
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class DataError(MLAnalyticsError):
    """Erreurs liées aux données"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.DATA,
            **kwargs
        )


class DataValidationError(DataError):
    """Erreur de validation des données"""
    
    def __init__(self, field: str, value: Any, reason: str, **kwargs):
        super().__init__(
            f"Validation échouée pour le champ '{field}' (valeur: {value}): {reason}",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.VALIDATION,
            details={'field': field, 'value': str(value), 'reason': reason, **kwargs.get('details', {})},
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class DataFormatError(DataError):
    """Erreur de format de données"""
    
    def __init__(self, expected_format: str, actual_format: str, **kwargs):
        super().__init__(
            f"Format de données incorrect. Attendu: {expected_format}, Reçu: {actual_format}",
            severity=ErrorSeverity.MEDIUM,
            details={'expected_format': expected_format, 'actual_format': actual_format, **kwargs.get('details', {})},
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class DataCorruptionError(DataError):
    """Erreur de corruption de données"""
    
    def __init__(self, data_source: str, **kwargs):
        super().__init__(
            f"Données corrompues détectées dans la source: {data_source}",
            severity=ErrorSeverity.HIGH,
            details={'data_source': data_source, **kwargs.get('details', {})},
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class ConfigurationError(MLAnalyticsError):
    """Erreurs de configuration"""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        details = kwargs.get('details', {})
        if config_key:
            details['config_key'] = config_key
        
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class InvalidConfigurationError(ConfigurationError):
    """Configuration invalide"""
    
    def __init__(self, config_key: str, value: Any, reason: str, **kwargs):
        super().__init__(
            f"Configuration invalide pour '{config_key}' (valeur: {value}): {reason}",
            config_key=config_key,
            details={'value': str(value), 'reason': reason, **kwargs.get('details', {})},
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class MissingConfigurationError(ConfigurationError):
    """Configuration manquante"""
    
    def __init__(self, config_key: str, **kwargs):
        super().__init__(
            f"Configuration manquante: '{config_key}'",
            config_key=config_key,
            **kwargs
        )


class ResourceError(MLAnalyticsError):
    """Erreurs liées aux ressources"""
    
    def __init__(self, message: str, resource_type: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.RESOURCE,
            details={'resource_type': resource_type, **kwargs.get('details', {})},
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class OutOfMemoryError(ResourceError):
    """Erreur de mémoire insuffisante"""
    
    def __init__(self, required_mb: Optional[float] = None, available_mb: Optional[float] = None, **kwargs):
        message = "Mémoire insuffisante"
        details = kwargs.get('details', {})
        
        if required_mb is not None:
            message += f" (requis: {required_mb:.1f} MB"
            details['required_mb'] = required_mb
            
            if available_mb is not None:
                message += f", disponible: {available_mb:.1f} MB"
                details['available_mb'] = available_mb
            message += ")"
        
        super().__init__(
            message,
            resource_type="memory",
            severity=ErrorSeverity.HIGH,
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class GPUError(ResourceError):
    """Erreur GPU"""
    
    def __init__(self, message: str, gpu_id: Optional[int] = None, **kwargs):
        details = kwargs.get('details', {})
        if gpu_id is not None:
            details['gpu_id'] = gpu_id
            message = f"Erreur GPU {gpu_id}: {message}"
        else:
            message = f"Erreur GPU: {message}"
        
        super().__init__(
            message,
            resource_type="gpu",
            severity=ErrorSeverity.HIGH,
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class DiskSpaceError(ResourceError):
    """Erreur d'espace disque"""
    
    def __init__(self, required_gb: float, available_gb: float, path: str, **kwargs):
        super().__init__(
            f"Espace disque insuffisant sur '{path}' (requis: {required_gb:.1f} GB, disponible: {available_gb:.1f} GB)",
            resource_type="disk",
            severity=ErrorSeverity.HIGH,
            details={
                'required_gb': required_gb,
                'available_gb': available_gb,
                'path': path,
                **kwargs.get('details', {})
            },
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class NetworkError(MLAnalyticsError):
    """Erreurs réseau"""
    
    def __init__(self, message: str, endpoint: Optional[str] = None, **kwargs):
        details = kwargs.get('details', {})
        if endpoint:
            details['endpoint'] = endpoint
        
        super().__init__(
            message,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class APIConnectionError(NetworkError):
    """Erreur de connexion API"""
    
    def __init__(self, endpoint: str, status_code: Optional[int] = None, **kwargs):
        message = f"Impossible de se connecter à l'API: {endpoint}"
        details = kwargs.get('details', {})
        
        if status_code is not None:
            message += f" (code: {status_code})"
            details['status_code'] = status_code
        
        super().__init__(
            message,
            endpoint=endpoint,
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class DatabaseConnectionError(NetworkError):
    """Erreur de connexion base de données"""
    
    def __init__(self, database_url: str, **kwargs):
        super().__init__(
            f"Impossible de se connecter à la base de données: {database_url}",
            severity=ErrorSeverity.HIGH,
            details={'database_url': database_url, **kwargs.get('details', {})},
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class AuthenticationError(MLAnalyticsError):
    """Erreurs d'authentification"""
    
    def __init__(self, message: str, user_id: Optional[str] = None, **kwargs):
        details = kwargs.get('details', {})
        if user_id:
            details['user_id'] = user_id
        
        super().__init__(
            message,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            details=details,
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class InvalidTokenError(AuthenticationError):
    """Token invalide"""
    
    def __init__(self, token_type: str = "access", **kwargs):
        super().__init__(
            f"Token {token_type} invalide ou expiré",
            details={'token_type': token_type, **kwargs.get('details', {})},
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class PermissionDeniedError(AuthenticationError):
    """Permission refusée"""
    
    def __init__(self, resource: str, action: str, user_id: Optional[str] = None, **kwargs):
        message = f"Permission refusée pour l'action '{action}' sur la ressource '{resource}'"
        
        super().__init__(
            message,
            user_id=user_id,
            details={'resource': resource, 'action': action, **kwargs.get('details', {})},
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


class TimeoutError(MLAnalyticsError):
    """Erreurs de timeout"""
    
    def __init__(self, operation: str, timeout_seconds: float, **kwargs):
        super().__init__(
            f"Timeout lors de l'opération '{operation}' après {timeout_seconds} secondes",
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            details={'operation': operation, 'timeout_seconds': timeout_seconds, **kwargs.get('details', {})},
            **{k: v for k, v in kwargs.items() if k != 'details'}
        )


# Utilitaires pour la gestion d'erreurs
class ErrorHandler:
    """Gestionnaire d'erreurs centralisé"""
    
    def __init__(self):
        self.error_stats = {
            'total_errors': 0,
            'errors_by_category': {},
            'errors_by_severity': {},
            'recent_errors': []
        }
        self.max_recent_errors = 100
    
    def handle_error(self, error: MLAnalyticsError):
        """Traitement centralisé d'une erreur"""
        self._update_stats(error)
        self._store_recent_error(error)
        
        # Actions spécifiques selon la sévérité
        if error.severity == ErrorSeverity.CRITICAL:
            self._handle_critical_error(error)
        elif error.severity == ErrorSeverity.HIGH:
            self._handle_high_severity_error(error)
    
    def _update_stats(self, error: MLAnalyticsError):
        """Mise à jour des statistiques d'erreurs"""
        self.error_stats['total_errors'] += 1
        
        category = error.category.value
        self.error_stats['errors_by_category'][category] = \
            self.error_stats['errors_by_category'].get(category, 0) + 1
        
        severity = error.severity.value
        self.error_stats['errors_by_severity'][severity] = \
            self.error_stats['errors_by_severity'].get(severity, 0) + 1
    
    def _store_recent_error(self, error: MLAnalyticsError):
        """Stockage des erreurs récentes"""
        self.error_stats['recent_errors'].append(error.to_dict())
        
        # Limitation du nombre d'erreurs récentes
        if len(self.error_stats['recent_errors']) > self.max_recent_errors:
            self.error_stats['recent_errors'] = \
                self.error_stats['recent_errors'][-self.max_recent_errors:]
    
    def _handle_critical_error(self, error: MLAnalyticsError):
        """Traitement des erreurs critiques"""
        # Notifications d'urgence, arrêt de services, etc.
        logging.critical(f"ERREUR CRITIQUE DÉTECTÉE: {error}")
    
    def _handle_high_severity_error(self, error: MLAnalyticsError):
        """Traitement des erreurs de haute sévérité"""
        # Alertes, notifications, retry automatique, etc.
        logging.error(f"Erreur de haute sévérité: {error}")
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Récupération des statistiques d'erreurs"""
        return dict(self.error_stats)
    
    def clear_stats(self):
        """Réinitialisation des statistiques"""
        self.error_stats = {
            'total_errors': 0,
            'errors_by_category': {},
            'errors_by_severity': {},
            'recent_errors': []
        }


# Instance globale du gestionnaire d'erreurs
error_handler = ErrorHandler()


def handle_ml_error(func):
    """Décorateur pour la gestion automatique d'erreurs ML"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except MLAnalyticsError as e:
            error_handler.handle_error(e)
            raise
        except Exception as e:
            # Conversion des exceptions standards en MLAnalyticsError
            ml_error = MLAnalyticsError(
                message=str(e),
                severity=ErrorSeverity.HIGH,
                original_exception=e
            )
            error_handler.handle_error(ml_error)
            raise ml_error
    
    return wrapper


# Exports publics
__all__ = [
    # Base exceptions
    'MLAnalyticsError',
    'ErrorSeverity',
    'ErrorCategory',
    
    # Model exceptions
    'ModelError',
    'ModelNotFoundError',
    'ModelLoadError',
    'ModelTrainingError',
    'InferenceError',
    'TrainingError',
    
    # Pipeline exceptions
    'PipelineError',
    'PipelineStepError',
    'PipelineTimeoutError',
    
    # Data exceptions
    'DataError',
    'DataValidationError',
    'DataFormatError',
    'DataCorruptionError',
    
    # Configuration exceptions
    'ConfigurationError',
    'InvalidConfigurationError',
    'MissingConfigurationError',
    
    # Resource exceptions
    'ResourceError',
    'OutOfMemoryError',
    'GPUError',
    'DiskSpaceError',
    
    # Network exceptions
    'NetworkError',
    'APIConnectionError',
    'DatabaseConnectionError',
    
    # Authentication exceptions
    'AuthenticationError',
    'InvalidTokenError',
    'PermissionDeniedError',
    
    # Timeout exceptions
    'TimeoutError',
    
    # Utilities
    'ErrorHandler',
    'error_handler',
    'handle_ml_error'
]
