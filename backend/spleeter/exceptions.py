"""
🎵 Spotify AI Agent - Spleeter Exceptions
=========================================

Classes d'exceptions personnalisées pour le module Spleeter.
Gestion d'erreurs avancée avec contexte détaillé et logging.

🎖️ Développé par l'équipe d'experts enterprise
"""

import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import traceback
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class SpleeterBaseException(Exception):
    """
    Exception de base pour le module Spleeter
    
    Features:
    - Contexte détaillé d'erreur
    - Logging automatique
    - Code d'erreur structuré
    - Données de debugging
    """
    
    def __init__(self,
                 message: str,
                 error_code: str = "SPLEETER_ERROR",
                 context: Optional[Dict[str, Any]] = None,
                 original_exception: Optional[Exception] = None,
                 severity: str = "ERROR"):
        
        super().__init__(message)
        
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.original_exception = original_exception
        self.severity = severity.upper()
        self.timestamp = datetime.now()
        
        # Capture de la stack trace
        self.stack_trace = traceback.format_stack()
        
        # Logging automatique
        self._log_error()
    
    def _log_error(self):
        """Log l'erreur selon sa sévérité"""
        log_data = {
            'error_code': self.error_code,
            'message': self.message,
            'context': self.context,
            'timestamp': self.timestamp.isoformat()
        }
        
        if self.original_exception:
            log_data['original_error'] = str(self.original_exception)
            log_data['original_type'] = type(self.original_exception).__name__
        
        log_message = f"[{self.error_code}] {self.message}"
        
        if self.severity == "CRITICAL":
            logger.critical(log_message, extra=log_data)
        elif self.severity == "ERROR":
            logger.error(log_message, extra=log_data)
        elif self.severity == "WARNING":
            logger.warning(log_message, extra=log_data)
        else:
            logger.info(log_message, extra=log_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit l'exception en dictionnaire pour serialization
        
        Returns:
            Dictionnaire représentant l'erreur
        """
        result = {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'severity': self.severity,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context
        }
        
        if self.original_exception:
            result['original_exception'] = {
                'type': type(self.original_exception).__name__,
                'message': str(self.original_exception)
            }
        
        return result
    
    def __str__(self) -> str:
        """Représentation string enrichie"""
        parts = [f"[{self.error_code}] {self.message}"]
        
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"Context: {context_str}")
        
        if self.original_exception:
            parts.append(f"Caused by: {type(self.original_exception).__name__}: {self.original_exception}")
        
        return " | ".join(parts)


class AudioProcessingError(SpleeterBaseException):
    """
    Erreurs liées au traitement audio
    
    Cas d'usage:
    - Échec de chargement audio
    - Format non supporté
    - Corruption de données
    - Problèmes de conversion
    """
    
    def __init__(self,
                 message: str,
                 file_path: Optional[Union[str, Path]] = None,
                 audio_format: Optional[str] = None,
                 sample_rate: Optional[int] = None,
                 channels: Optional[int] = None,
                 **kwargs):
        
        context = kwargs.get('context', {})
        
        if file_path:
            context['file_path'] = str(file_path)
        if audio_format:
            context['audio_format'] = audio_format
        if sample_rate:
            context['sample_rate'] = sample_rate
        if channels:
            context['channels'] = channels
        
        super().__init__(
            message=message,
            error_code=kwargs.get('error_code', 'AUDIO_PROCESSING_ERROR'),
            context=context,
            **{k: v for k, v in kwargs.items() if k not in ['context', 'error_code']}
        )


class ModelError(SpleeterBaseException):
    """
    Erreurs liées aux modèles Spleeter
    
    Cas d'usage:
    - Modèle non trouvé
    - Échec de téléchargement
    - Version incompatible
    - Corruption de modèle
    """
    
    def __init__(self,
                 message: str,
                 model_name: Optional[str] = None,
                 model_path: Optional[Union[str, Path]] = None,
                 version: Optional[str] = None,
                 **kwargs):
        
        context = kwargs.get('context', {})
        
        if model_name:
            context['model_name'] = model_name
        if model_path:
            context['model_path'] = str(model_path)
        if version:
            context['version'] = version
        
        super().__init__(
            message=message,
            error_code=kwargs.get('error_code', 'MODEL_ERROR'),
            context=context,
            **{k: v for k, v in kwargs.items() if k not in ['context', 'error_code']}
        )


class ModelNotFoundError(ModelError):
    """Modèle spécifique non trouvé"""
    
    def __init__(self, model_name: str, available_models: Optional[List[str]] = None, **kwargs):
        message = f"Modèle '{model_name}' non trouvé"
        
        if available_models:
            message += f". Modèles disponibles: {', '.join(available_models)}"
        
        super().__init__(
            message=message,
            model_name=model_name,
            error_code="MODEL_NOT_FOUND",
            context={'available_models': available_models} if available_models else None,
            **kwargs
        )


class ModelDownloadError(ModelError):
    """Erreur de téléchargement de modèle"""
    
    def __init__(self,
                 model_name: str,
                 download_url: Optional[str] = None,
                 http_status: Optional[int] = None,
                 **kwargs):
        
        message = f"Échec téléchargement modèle '{model_name}'"
        
        context = kwargs.get('context', {})
        if download_url:
            context['download_url'] = download_url
        if http_status:
            context['http_status'] = http_status
            message += f" (HTTP {http_status})"
        
        super().__init__(
            message=message,
            model_name=model_name,
            error_code="MODEL_DOWNLOAD_ERROR",
            context=context,
            **kwargs
        )


class ValidationError(SpleeterBaseException):
    """
    Erreurs de validation
    
    Cas d'usage:
    - Paramètres invalides
    - Fichiers corrompus
    - Données incohérentes
    - Violations de contraintes
    """
    
    def __init__(self,
                 message: str,
                 field_name: Optional[str] = None,
                 field_value: Optional[Any] = None,
                 expected_type: Optional[type] = None,
                 validation_rule: Optional[str] = None,
                 **kwargs):
        
        context = kwargs.get('context', {})
        
        if field_name:
            context['field_name'] = field_name
        if field_value is not None:
            context['field_value'] = str(field_value)
        if expected_type:
            context['expected_type'] = expected_type.__name__
        if validation_rule:
            context['validation_rule'] = validation_rule
        
        super().__init__(
            message=message,
            error_code=kwargs.get('error_code', 'VALIDATION_ERROR'),
            context=context,
            severity=kwargs.get('severity', 'WARNING'),
            **{k: v for k, v in kwargs.items() if k not in ['context', 'error_code', 'severity']}
        )


class CacheError(SpleeterBaseException):
    """
    Erreurs liées au système de cache
    
    Cas d'usage:
    - Échec d'écriture cache
    - Cache corrompu
    - Problèmes de permissions
    - Espace disque insuffisant
    """
    
    def __init__(self,
                 message: str,
                 cache_type: Optional[str] = None,
                 cache_key: Optional[str] = None,
                 cache_path: Optional[Union[str, Path]] = None,
                 operation: Optional[str] = None,
                 **kwargs):
        
        context = kwargs.get('context', {})
        
        if cache_type:
            context['cache_type'] = cache_type
        if cache_key:
            context['cache_key'] = cache_key
        if cache_path:
            context['cache_path'] = str(cache_path)
        if operation:
            context['operation'] = operation
        
        super().__init__(
            message=message,
            error_code=kwargs.get('error_code', 'CACHE_ERROR'),
            context=context,
            **{k: v for k, v in kwargs.items() if k not in ['context', 'error_code']}
        )


class ConfigurationError(SpleeterBaseException):
    """
    Erreurs de configuration
    
    Cas d'usage:
    - Configuration manquante
    - Valeurs invalides
    - Fichier de config corrompu
    - Paramètres incompatibles
    """
    
    def __init__(self,
                 message: str,
                 config_key: Optional[str] = None,
                 config_value: Optional[Any] = None,
                 config_file: Optional[Union[str, Path]] = None,
                 **kwargs):
        
        context = kwargs.get('context', {})
        
        if config_key:
            context['config_key'] = config_key
        if config_value is not None:
            context['config_value'] = str(config_value)
        if config_file:
            context['config_file'] = str(config_file)
        
        super().__init__(
            message=message,
            error_code=kwargs.get('error_code', 'CONFIGURATION_ERROR'),
            context=context,
            **{k: v for k, v in kwargs.items() if k not in ['context', 'error_code']}
        )


class ResourceError(SpleeterBaseException):
    """
    Erreurs liées aux ressources système
    
    Cas d'usage:
    - Mémoire insuffisante
    - GPU non disponible
    - Espace disque insuffisant
    - Limites système atteintes
    """
    
    def __init__(self,
                 message: str,
                 resource_type: Optional[str] = None,
                 current_usage: Optional[float] = None,
                 limit: Optional[float] = None,
                 unit: Optional[str] = None,
                 **kwargs):
        
        context = kwargs.get('context', {})
        
        if resource_type:
            context['resource_type'] = resource_type
        if current_usage is not None:
            context['current_usage'] = current_usage
        if limit is not None:
            context['limit'] = limit
        if unit:
            context['unit'] = unit
        
        super().__init__(
            message=message,
            error_code=kwargs.get('error_code', 'RESOURCE_ERROR'),
            context=context,
            severity=kwargs.get('severity', 'CRITICAL'),
            **{k: v for k, v in kwargs.items() if k not in ['context', 'error_code', 'severity']}
        )


class GPUError(ResourceError):
    """Erreurs spécifiques au GPU"""
    
    def __init__(self,
                 message: str,
                 gpu_id: Optional[int] = None,
                 gpu_memory_used: Optional[float] = None,
                 gpu_memory_total: Optional[float] = None,
                 **kwargs):
        
        context = kwargs.get('context', {})
        
        if gpu_id is not None:
            context['gpu_id'] = gpu_id
        if gpu_memory_used is not None:
            context['gpu_memory_used'] = gpu_memory_used
        if gpu_memory_total is not None:
            context['gpu_memory_total'] = gpu_memory_total
        
        super().__init__(
            message=message,
            resource_type="GPU",
            error_code=kwargs.get('error_code', 'GPU_ERROR'),
            context=context,
            **{k: v for k, v in kwargs.items() if k not in ['context', 'error_code']}
        )


class ProcessingTimeout(SpleeterBaseException):
    """
    Erreur de timeout de traitement
    
    Cas d'usage:
    - Traitement trop long
    - Modèle bloqué
    - Ressources indisponibles
    """
    
    def __init__(self,
                 message: str,
                 timeout_seconds: Optional[float] = None,
                 operation: Optional[str] = None,
                 **kwargs):
        
        context = kwargs.get('context', {})
        
        if timeout_seconds is not None:
            context['timeout_seconds'] = timeout_seconds
        if operation:
            context['operation'] = operation
        
        super().__init__(
            message=message,
            error_code=kwargs.get('error_code', 'PROCESSING_TIMEOUT'),
            context=context,
            severity=kwargs.get('severity', 'WARNING'),
            **{k: v for k, v in kwargs.items() if k not in ['context', 'error_code', 'severity']}
        )


class MonitoringError(SpleeterBaseException):
    """
    Erreurs du système de monitoring
    
    Cas d'usage:
    - Échec collecte métriques
    - Problèmes d'export
    - Alertes non fonctionnelles
    """
    
    def __init__(self,
                 message: str,
                 metric_name: Optional[str] = None,
                 collector_type: Optional[str] = None,
                 **kwargs):
        
        context = kwargs.get('context', {})
        
        if metric_name:
            context['metric_name'] = metric_name
        if collector_type:
            context['collector_type'] = collector_type
        
        super().__init__(
            message=message,
            error_code=kwargs.get('error_code', 'MONITORING_ERROR'),
            context=context,
            severity=kwargs.get('severity', 'WARNING'),
            **{k: v for k, v in kwargs.items() if k not in ['context', 'error_code', 'severity']}
        )


class SecurityError(SpleeterBaseException):
    """
    Erreurs de sécurité
    
    Cas d'usage:
    - Tentative d'accès non autorisé
    - Validation de sécurité échouée
    - Fichiers suspects
    - Path traversal
    """
    
    def __init__(self,
                 message: str,
                 security_check: Optional[str] = None,
                 attempted_path: Optional[Union[str, Path]] = None,
                 risk_level: str = "HIGH",
                 **kwargs):
        
        context = kwargs.get('context', {})
        
        if security_check:
            context['security_check'] = security_check
        if attempted_path:
            context['attempted_path'] = str(attempted_path)
        context['risk_level'] = risk_level
        
        super().__init__(
            message=message,
            error_code=kwargs.get('error_code', 'SECURITY_ERROR'),
            context=context,
            severity="CRITICAL",
            **{k: v for k, v in kwargs.items() if k not in ['context', 'error_code']}
        )


class DependencyError(SpleeterBaseException):
    """
    Erreurs de dépendances
    
    Cas d'usage:
    - Librairie manquante
    - Version incompatible
    - Import échoué
    """
    
    def __init__(self,
                 message: str,
                 dependency_name: Optional[str] = None,
                 required_version: Optional[str] = None,
                 current_version: Optional[str] = None,
                 installation_hint: Optional[str] = None,
                 **kwargs):
        
        context = kwargs.get('context', {})
        
        if dependency_name:
            context['dependency_name'] = dependency_name
        if required_version:
            context['required_version'] = required_version
        if current_version:
            context['current_version'] = current_version
        if installation_hint:
            context['installation_hint'] = installation_hint
        
        super().__init__(
            message=message,
            error_code=kwargs.get('error_code', 'DEPENDENCY_ERROR'),
            context=context,
            **{k: v for k, v in kwargs.items() if k not in ['context', 'error_code']}
        )


class BatchProcessingError(SpleeterBaseException):
    """
    Erreurs de traitement par batch
    
    Cas d'usage:
    - Échec partiel de batch
    - Queue saturée
    - Worker défaillant
    """
    
    def __init__(self,
                 message: str,
                 batch_id: Optional[str] = None,
                 failed_items: Optional[List[str]] = None,
                 total_items: Optional[int] = None,
                 success_count: Optional[int] = None,
                 **kwargs):
        
        context = kwargs.get('context', {})
        
        if batch_id:
            context['batch_id'] = batch_id
        if failed_items:
            context['failed_items'] = failed_items
            context['failed_count'] = len(failed_items)
        if total_items is not None:
            context['total_items'] = total_items
        if success_count is not None:
            context['success_count'] = success_count
        
        super().__init__(
            message=message,
            error_code=kwargs.get('error_code', 'BATCH_PROCESSING_ERROR'),
            context=context,
            **{k: v for k, v in kwargs.items() if k not in ['context', 'error_code']}
        )


# Utilitaires pour la gestion d'erreurs


def handle_exception(func):
    """
    Décorateur pour la gestion automatique d'exceptions
    
    Args:
        func: Fonction à décorer
        
    Returns:
        Fonction décorée
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SpleeterBaseException:
            # Re-lancer les exceptions Spleeter
            raise
        except Exception as e:
            # Encapsuler les autres exceptions
            raise SpleeterBaseException(
                message=f"Erreur inattendue dans {func.__name__}: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                original_exception=e,
                context={
                    'function': func.__name__,
                    'args': str(args),
                    'kwargs': str(kwargs)
                }
            )
    
    return wrapper


def validate_required_dependency(
    dependency_name: str,
    import_name: Optional[str] = None,
    min_version: Optional[str] = None,
    installation_hint: Optional[str] = None
):
    """
    Valide qu'une dépendance requise est disponible
    
    Args:
        dependency_name: Nom de la dépendance
        import_name: Nom d'import (si différent)
        min_version: Version minimale requise
        installation_hint: Conseil d'installation
        
    Raises:
        DependencyError: Si la dépendance n'est pas disponible
    """
    import_name = import_name or dependency_name
    
    try:
        module = __import__(import_name)
        
        # Vérification de version si requise
        if min_version and hasattr(module, '__version__'):
            current_version = module.__version__
            # Comparaison simple de version (pour une vraie app, utiliser packaging.version)
            if current_version < min_version:
                raise DependencyError(
                    f"Version {dependency_name} trop ancienne",
                    dependency_name=dependency_name,
                    required_version=f">={min_version}",
                    current_version=current_version,
                    installation_hint=installation_hint
                )
        
    except ImportError as e:
        hint = installation_hint or f"pip install {dependency_name}"
        raise DependencyError(
            f"Dépendance '{dependency_name}' non trouvée",
            dependency_name=dependency_name,
            installation_hint=hint,
            original_exception=e
        )


def create_error_context(
    operation: str,
    file_path: Optional[Union[str, Path]] = None,
    model_name: Optional[str] = None,
    additional_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Crée un contexte d'erreur standardisé
    
    Args:
        operation: Nom de l'opération
        file_path: Chemin de fichier impliqué
        model_name: Nom du modèle utilisé
        additional_data: Données additionnelles
        
    Returns:
        Dictionnaire de contexte
    """
    context = {
        'operation': operation,
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'platform': sys.platform
    }
    
    if file_path:
        context['file_path'] = str(file_path)
        
        # Informations sur le fichier si possible
        try:
            path_obj = Path(file_path)
            if path_obj.exists():
                context['file_size'] = path_obj.stat().st_size
                context['file_exists'] = True
            else:
                context['file_exists'] = False
        except Exception:
            pass
    
    if model_name:
        context['model_name'] = model_name
    
    if additional_data:
        context.update(additional_data)
    
    return context


def log_exception_details(
    exception: Exception,
    operation: str,
    logger_instance: Optional[logging.Logger] = None
):
    """
    Log les détails complets d'une exception
    
    Args:
        exception: Exception à logger
        operation: Opération en cours
        logger_instance: Logger à utiliser
    """
    log = logger_instance or logger
    
    if isinstance(exception, SpleeterBaseException):
        # Exception Spleeter - déjà loggée automatiquement
        return
    
    # Exception standard - log détaillé
    exc_info = sys.exc_info()
    log.error(
        f"Exception dans {operation}: {type(exception).__name__}: {exception}",
        exc_info=exc_info,
        extra={
            'operation': operation,
            'exception_type': type(exception).__name__,
            'exception_message': str(exception),
            'stack_trace': traceback.format_exception(*exc_info)
        }
    )


def safe_operation(
    operation_name: str,
    operation_func: callable,
    *args,
    fallback_value: Any = None,
    reraise: bool = True,
    log_errors: bool = True,
    **kwargs
) -> Any:
    """
    Exécute une opération avec gestion d'erreur sécurisée
    
    Args:
        operation_name: Nom de l'opération
        operation_func: Fonction à exécuter
        *args: Arguments positionnels
        fallback_value: Valeur de fallback en cas d'erreur
        reraise: Re-lancer l'exception
        log_errors: Logger les erreurs
        **kwargs: Arguments nommés
        
    Returns:
        Résultat de l'opération ou fallback_value
        
    Raises:
        Exception: Si reraise=True
    """
    try:
        return operation_func(*args, **kwargs)
    
    except Exception as e:
        if log_errors:
            log_exception_details(e, operation_name)
        
        if reraise:
            # Encapsuler si ce n'est pas déjà une exception Spleeter
            if not isinstance(e, SpleeterBaseException):
                raise SpleeterBaseException(
                    message=f"Échec opération '{operation_name}': {str(e)}",
                    error_code="OPERATION_FAILED",
                    original_exception=e,
                    context={'operation': operation_name}
                ) from e
            raise
        
        return fallback_value


# Classe pour collecter et analyser les erreurs en batch
class ErrorCollector:
    """
    Collecteur d'erreurs pour analyse et reporting
    
    Features:
    - Accumulation d'erreurs
    - Analyse de patterns
    - Génération de rapports
    - Statistiques d'erreurs
    """
    
    def __init__(self):
        self.errors: List[SpleeterBaseException] = []
        self.error_counts: Dict[str, int] = {}
        self.error_patterns: Dict[str, List[Dict[str, Any]]] = {}
    
    def add_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """
        Ajoute une erreur à la collection
        
        Args:
            error: Exception à ajouter
            context: Contexte additionnel
        """
        # Convertir en SpleeterBaseException si nécessaire
        if not isinstance(error, SpleeterBaseException):
            spleeter_error = SpleeterBaseException(
                message=str(error),
                error_code="COLLECTED_ERROR",
                original_exception=error,
                context=context
            )
        else:
            spleeter_error = error
            if context:
                spleeter_error.context.update(context)
        
        self.errors.append(spleeter_error)
        
        # Comptage par code d'erreur
        error_code = spleeter_error.error_code
        self.error_counts[error_code] = self.error_counts.get(error_code, 0) + 1
        
        # Analyse de patterns
        if error_code not in self.error_patterns:
            self.error_patterns[error_code] = []
        
        self.error_patterns[error_code].append({
            'message': spleeter_error.message,
            'context': spleeter_error.context,
            'timestamp': spleeter_error.timestamp.isoformat()
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Retourne un résumé des erreurs collectées
        
        Returns:
            Dictionnaire de résumé
        """
        return {
            'total_errors': len(self.errors),
            'error_counts': self.error_counts.copy(),
            'most_common_error': max(self.error_counts.items(), key=lambda x: x[1])[0] if self.error_counts else None,
            'error_rate_by_type': {
                error_type: (count / len(self.errors)) * 100
                for error_type, count in self.error_counts.items()
            } if self.errors else {},
            'collection_period': {
                'start': self.errors[0].timestamp.isoformat() if self.errors else None,
                'end': self.errors[-1].timestamp.isoformat() if self.errors else None
            }
        }
    
    def clear(self):
        """Vide la collection d'erreurs"""
        self.errors.clear()
        self.error_counts.clear()
        self.error_patterns.clear()
    
    def get_errors_by_type(self, error_code: str) -> List[SpleeterBaseException]:
        """
        Retourne les erreurs d'un type spécifique
        
        Args:
            error_code: Code d'erreur à filtrer
            
        Returns:
            Liste des erreurs correspondantes
        """
        return [error for error in self.errors if error.error_code == error_code]
