"""
Exceptions personnalisées pour le module ML - Spotify AI Agent
Gestion d'erreurs spécialisées pour Machine Learning et Intelligence Artificielle
"""

from typing import Any, Dict, List, Optional, Union
import traceback
from datetime import datetime, timezone

from ..base import BaseException, ErrorCode, ErrorSeverity


class MLException(BaseException):
    """Exception de base pour toutes les erreurs ML"""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        user_message: Optional[str] = None,
        suggested_action: Optional[str] = None,
        model_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=error_code or "ML_ERROR",
            details=details,
            severity=severity,
            user_message=user_message,
            suggested_action=suggested_action,
            **kwargs
        )
        self.model_id = model_id
        self.experiment_id = experiment_id


# Exceptions de données
class DataException(MLException):
    """Exceptions liées aux données"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="DATA_ERROR",
            **kwargs
        )


class DataValidationError(DataException):
    """Erreur de validation des données"""
    
    def __init__(
        self,
        message: str = "Erreur de validation des données",
        validation_errors: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        details['validation_errors'] = validation_errors or []
        
        super().__init__(
            message=message,
            error_code="DATA_VALIDATION_ERROR",
            details=details,
            user_message="Les données fournies ne respectent pas le format attendu",
            suggested_action="Vérifiez le format et la structure de vos données",
            **kwargs
        )


class DataQualityError(DataException):
    """Erreur de qualité des données"""
    
    def __init__(
        self,
        message: str = "Problème de qualité des données détecté",
        quality_issues: Optional[List[str]] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        details['quality_issues'] = quality_issues or []
        
        super().__init__(
            message=message,
            error_code="DATA_QUALITY_ERROR",
            details=details,
            user_message="La qualité des données est insuffisante pour l'entraînement",
            suggested_action="Nettoyez vos données et supprimez les valeurs aberrantes",
            **kwargs
        )


class DataSchemaError(DataException):
    """Erreur de schéma de données"""
    
    def __init__(
        self,
        message: str = "Schéma de données incompatible",
        expected_schema: Optional[Dict[str, Any]] = None,
        actual_schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        details.update({
            'expected_schema': expected_schema,
            'actual_schema': actual_schema
        })
        
        super().__init__(
            message=message,
            error_code="DATA_SCHEMA_ERROR",
            details=details,
            user_message="Le schéma des données ne correspond pas à celui attendu",
            suggested_action="Adaptez vos données au schéma requis ou mettez à jour le modèle",
            **kwargs
        )


class DataDriftError(DataException):
    """Erreur de dérive des données"""
    
    def __init__(
        self,
        message: str = "Dérive de données détectée",
        drift_score: Optional[float] = None,
        threshold: Optional[float] = None,
        affected_features: Optional[List[str]] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        details.update({
            'drift_score': drift_score,
            'threshold': threshold,
            'affected_features': affected_features or []
        })
        
        super().__init__(
            message=message,
            error_code="DATA_DRIFT_ERROR",
            details=details,
            severity=ErrorSeverity.WARNING,
            user_message="Une dérive des données a été détectée",
            suggested_action="Réévaluez le modèle avec les nouvelles données",
            **kwargs
        )


# Exceptions de modèles
class ModelException(MLException):
    """Exceptions liées aux modèles ML"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="MODEL_ERROR",
            **kwargs
        )


class ModelNotFoundError(ModelException):
    """Modèle non trouvé"""
    
    def __init__(
        self,
        model_id: str,
        message: Optional[str] = None,
        **kwargs
    ):
        message = message or f"Modèle '{model_id}' non trouvé"
        
        super().__init__(
            message=message,
            error_code="MODEL_NOT_FOUND",
            model_id=model_id,
            user_message=f"Le modèle {model_id} n'existe pas",
            suggested_action="Vérifiez l'ID du modèle ou créez un nouveau modèle",
            **kwargs
        )


class ModelLoadError(ModelException):
    """Erreur de chargement de modèle"""
    
    def __init__(
        self,
        message: str = "Impossible de charger le modèle",
        model_path: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        details['model_path'] = model_path
        
        super().__init__(
            message=message,
            error_code="MODEL_LOAD_ERROR",
            details=details,
            user_message="Le modèle n'a pas pu être chargé",
            suggested_action="Vérifiez le chemin du modèle et ses dépendances",
            **kwargs
        )


class ModelSaveError(ModelException):
    """Erreur de sauvegarde de modèle"""
    
    def __init__(
        self,
        message: str = "Impossible de sauvegarder le modèle",
        save_path: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        details['save_path'] = save_path
        
        super().__init__(
            message=message,
            error_code="MODEL_SAVE_ERROR",
            details=details,
            user_message="Le modèle n'a pas pu être sauvegardé",
            suggested_action="Vérifiez les permissions et l'espace disque disponible",
            **kwargs
        )


class ModelVersionError(ModelException):
    """Erreur de version de modèle"""
    
    def __init__(
        self,
        message: str = "Problème de version de modèle",
        current_version: Optional[str] = None,
        required_version: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        details.update({
            'current_version': current_version,
            'required_version': required_version
        })
        
        super().__init__(
            message=message,
            error_code="MODEL_VERSION_ERROR",
            details=details,
            user_message="Incompatibilité de version de modèle",
            suggested_action="Mettez à jour le modèle ou utilisez une version compatible",
            **kwargs
        )


class ModelPerformanceError(ModelException):
    """Erreur de performance de modèle"""
    
    def __init__(
        self,
        message: str = "Performance du modèle insuffisante",
        metric_name: Optional[str] = None,
        current_value: Optional[float] = None,
        threshold_value: Optional[float] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        details.update({
            'metric_name': metric_name,
            'current_value': current_value,
            'threshold_value': threshold_value
        })
        
        super().__init__(
            message=message,
            error_code="MODEL_PERFORMANCE_ERROR",
            details=details,
            severity=ErrorSeverity.WARNING,
            user_message="Les performances du modèle sont en dessous du seuil acceptable",
            suggested_action="Réévaluez ou réentraînez le modèle",
            **kwargs
        )


# Exceptions d'entraînement
class TrainingException(MLException):
    """Exceptions liées à l'entraînement"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="TRAINING_ERROR",
            **kwargs
        )


class TrainingDataError(TrainingException):
    """Erreur de données d'entraînement"""
    
    def __init__(
        self,
        message: str = "Problème avec les données d'entraînement",
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code="TRAINING_DATA_ERROR",
            user_message="Les données d'entraînement sont inadéquates",
            suggested_action="Vérifiez la qualité et la quantité des données d'entraînement",
            **kwargs
        )


class TrainingConvergenceError(TrainingException):
    """Erreur de convergence d'entraînement"""
    
    def __init__(
        self,
        message: str = "Le modèle n'a pas convergé",
        epoch: Optional[int] = None,
        loss_value: Optional[float] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        details.update({
            'epoch': epoch,
            'loss_value': loss_value
        })
        
        super().__init__(
            message=message,
            error_code="TRAINING_CONVERGENCE_ERROR",
            details=details,
            user_message="L'entraînement n'a pas convergé",
            suggested_action="Ajustez le taux d'apprentissage ou augmentez le nombre d'époques",
            **kwargs
        )


class TrainingResourceError(TrainingException):
    """Erreur de ressources d'entraînement"""
    
    def __init__(
        self,
        message: str = "Ressources insuffisantes pour l'entraînement",
        resource_type: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        details['resource_type'] = resource_type
        
        super().__init__(
            message=message,
            error_code="TRAINING_RESOURCE_ERROR",
            details=details,
            user_message="Ressources insuffisantes pour l'entraînement",
            suggested_action="Réduisez la taille du batch ou utilisez plus de ressources",
            **kwargs
        )


# Exceptions d'inférence
class InferenceException(MLException):
    """Exceptions liées à l'inférence"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="INFERENCE_ERROR",
            **kwargs
        )


class InferenceTimeoutError(InferenceException):
    """Timeout d'inférence"""
    
    def __init__(
        self,
        message: str = "Timeout lors de l'inférence",
        timeout_seconds: Optional[float] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        details['timeout_seconds'] = timeout_seconds
        
        super().__init__(
            message=message,
            error_code="INFERENCE_TIMEOUT_ERROR",
            details=details,
            user_message="L'inférence a pris trop de temps",
            suggested_action="Optimisez le modèle ou augmentez le timeout",
            **kwargs
        )


class InferenceInputError(InferenceException):
    """Erreur d'entrée d'inférence"""
    
    def __init__(
        self,
        message: str = "Format d'entrée invalide pour l'inférence",
        expected_format: Optional[str] = None,
        received_format: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        details.update({
            'expected_format': expected_format,
            'received_format': received_format
        })
        
        super().__init__(
            message=message,
            error_code="INFERENCE_INPUT_ERROR",
            details=details,
            user_message="Le format des données d'entrée est incorrect",
            suggested_action="Vérifiez le format des données d'entrée",
            **kwargs
        )


# Exceptions de pipeline
class PipelineException(MLException):
    """Exceptions liées aux pipelines ML"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="PIPELINE_ERROR",
            **kwargs
        )


class PipelineStepError(PipelineException):
    """Erreur d'étape de pipeline"""
    
    def __init__(
        self,
        message: str = "Erreur dans une étape du pipeline",
        step_name: Optional[str] = None,
        step_index: Optional[int] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        details.update({
            'step_name': step_name,
            'step_index': step_index
        })
        
        super().__init__(
            message=message,
            error_code="PIPELINE_STEP_ERROR",
            details=details,
            user_message=f"Erreur dans l'étape '{step_name}' du pipeline",
            suggested_action="Vérifiez la configuration de cette étape",
            **kwargs
        )


class PipelineConfigurationError(PipelineException):
    """Erreur de configuration de pipeline"""
    
    def __init__(
        self,
        message: str = "Configuration de pipeline invalide",
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code="PIPELINE_CONFIGURATION_ERROR",
            user_message="La configuration du pipeline est invalide",
            suggested_action="Vérifiez les paramètres de configuration du pipeline",
            **kwargs
        )


# Exceptions d'hyperparamètres
class HyperparameterException(MLException):
    """Exceptions liées aux hyperparamètres"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="HYPERPARAMETER_ERROR",
            **kwargs
        )


class HyperparameterValidationError(HyperparameterException):
    """Erreur de validation d'hyperparamètres"""
    
    def __init__(
        self,
        message: str = "Hyperparamètres invalides",
        invalid_params: Optional[List[str]] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        details['invalid_params'] = invalid_params or []
        
        super().__init__(
            message=message,
            error_code="HYPERPARAMETER_VALIDATION_ERROR",
            details=details,
            user_message="Certains hyperparamètres sont invalides",
            suggested_action="Vérifiez les valeurs des hyperparamètres",
            **kwargs
        )


class HyperparameterOptimizationError(HyperparameterException):
    """Erreur d'optimisation d'hyperparamètres"""
    
    def __init__(
        self,
        message: str = "Échec de l'optimisation des hyperparamètres",
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code="HYPERPARAMETER_OPTIMIZATION_ERROR",
            user_message="L'optimisation des hyperparamètres a échoué",
            suggested_action="Ajustez les paramètres d'optimisation ou l'espace de recherche",
            **kwargs
        )


# Exceptions de registre de modèles
class ModelRegistryException(MLException):
    """Exceptions liées au registre de modèles"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="MODEL_REGISTRY_ERROR",
            **kwargs
        )


class ModelRegistrationError(ModelRegistryException):
    """Erreur d'enregistrement de modèle"""
    
    def __init__(
        self,
        message: str = "Impossible d'enregistrer le modèle",
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code="MODEL_REGISTRATION_ERROR",
            user_message="Le modèle n'a pas pu être enregistré",
            suggested_action="Vérifiez les permissions et la connectivité au registre",
            **kwargs
        )


# Exceptions de déploiement
class DeploymentException(MLException):
    """Exceptions liées au déploiement"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="DEPLOYMENT_ERROR",
            **kwargs
        )


class DeploymentConfigurationError(DeploymentException):
    """Erreur de configuration de déploiement"""
    
    def __init__(
        self,
        message: str = "Configuration de déploiement invalide",
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code="DEPLOYMENT_CONFIGURATION_ERROR",
            user_message="La configuration de déploiement est invalide",
            suggested_action="Vérifiez les paramètres de déploiement",
            **kwargs
        )


class DeploymentResourceError(DeploymentException):
    """Erreur de ressources de déploiement"""
    
    def __init__(
        self,
        message: str = "Ressources insuffisantes pour le déploiement",
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code="DEPLOYMENT_RESOURCE_ERROR",
            user_message="Ressources insuffisantes pour le déploiement",
            suggested_action="Augmentez les ressources allouées ou optimisez le modèle",
            **kwargs
        )


# Exceptions de sécurité ML
class MLSecurityException(MLException):
    """Exceptions liées à la sécurité ML"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="ML_SECURITY_ERROR",
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )


class ModelPoisoningError(MLSecurityException):
    """Erreur d'empoisonnement de modèle"""
    
    def __init__(
        self,
        message: str = "Tentative d'empoisonnement de modèle détectée",
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code="MODEL_POISONING_ERROR",
            user_message="Une tentative d'attaque sur le modèle a été détectée",
            suggested_action="Suspendez immédiatement l'entraînement et vérifiez les données",
            **kwargs
        )


class AdversarialAttackError(MLSecurityException):
    """Erreur d'attaque adversariale"""
    
    def __init__(
        self,
        message: str = "Attaque adversariale détectée",
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code="ADVERSARIAL_ATTACK_ERROR",
            user_message="Une attaque adversariale a été détectée",
            suggested_action="Analysez les entrées suspectes et renforcez les défenses",
            **kwargs
        )


# Utilitaires d'exception
def create_ml_error_report(
    exception: MLException,
    include_traceback: bool = True,
    include_context: bool = True
) -> Dict[str, Any]:
    """Crée un rapport d'erreur ML détaillé"""
    
    report = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'error_type': type(exception).__name__,
        'error_code': exception.error_code,
        'message': str(exception),
        'severity': exception.severity.value if exception.severity else 'unknown',
        'user_message': exception.user_message,
        'suggested_action': exception.suggested_action,
        'model_id': getattr(exception, 'model_id', None),
        'experiment_id': getattr(exception, 'experiment_id', None),
        'details': exception.details or {}
    }
    
    if include_traceback:
        report['traceback'] = traceback.format_exc()
    
    if include_context and hasattr(exception, '__context__') and exception.__context__:
        report['context'] = str(exception.__context__)
    
    return report


__all__ = [
    # Exception de base
    'MLException',
    
    # Exceptions de données
    'DataException', 'DataValidationError', 'DataQualityError',
    'DataSchemaError', 'DataDriftError',
    
    # Exceptions de modèles
    'ModelException', 'ModelNotFoundError', 'ModelLoadError',
    'ModelSaveError', 'ModelVersionError', 'ModelPerformanceError',
    
    # Exceptions d'entraînement
    'TrainingException', 'TrainingDataError', 'TrainingConvergenceError',
    'TrainingResourceError',
    
    # Exceptions d'inférence
    'InferenceException', 'InferenceTimeoutError', 'InferenceInputError',
    
    # Exceptions de pipeline
    'PipelineException', 'PipelineStepError', 'PipelineConfigurationError',
    
    # Exceptions d'hyperparamètres
    'HyperparameterException', 'HyperparameterValidationError',
    'HyperparameterOptimizationError',
    
    # Exceptions de registre
    'ModelRegistryException', 'ModelRegistrationError',
    
    # Exceptions de déploiement
    'DeploymentException', 'DeploymentConfigurationError',
    'DeploymentResourceError',
    
    # Exceptions de sécurité
    'MLSecurityException', 'ModelPoisoningError', 'AdversarialAttackError',
    
    # Utilitaires
    'create_ml_error_report'
]
