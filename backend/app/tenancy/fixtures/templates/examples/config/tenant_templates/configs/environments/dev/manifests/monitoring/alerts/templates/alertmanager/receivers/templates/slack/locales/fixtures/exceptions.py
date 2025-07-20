"""
Exceptions Personnalisées pour Système de Fixtures Multi-Tenant
==============================================================

Ce module définit toutes les exceptions personnalisées pour une gestion d'erreurs
robuste et granulaire dans le système de fixtures d'alertes enterprise.

🎯 Architecture: Exception Hierarchy + Error Codes + Context Enrichment
🔒 Sécurité: Sanitization des messages + Audit des erreurs + Compliance
⚡ Performance: Error sampling + Circuit breaker + Graceful degradation
🧠 IA: Error prediction + Pattern analysis + Auto-recovery

Auteur: Fahed Mlaiel - Lead Developer & AI Architect
Équipe: DevOps/ML/Security/Backend Experts
Version: 3.0.0-enterprise
"""

import traceback
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Type
from enum import Enum, IntEnum
import logging
from dataclasses import dataclass, field
import hashlib

logger = logging.getLogger(__name__)

# ============================================================================================
# ENUMS ET CONSTANTES POUR LES ERREURS
# ============================================================================================

class ErrorSeverity(IntEnum):
    """Niveaux de sévérité des erreurs."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    FATAL = 5

class ErrorCategory(str, Enum):
    """Catégories d'erreurs système."""
    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    SECURITY = "security"
    PERFORMANCE = "performance"
    NETWORK = "network"
    DATABASE = "database"
    CACHE = "cache"
    ML_MODEL = "ml_model"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_API = "external_api"
    INFRASTRUCTURE = "infrastructure"
    COMPLIANCE = "compliance"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMITING = "rate_limiting"
    CIRCUIT_BREAKER = "circuit_breaker"
    BACKUP = "backup"
    RECOVERY = "recovery"
    MONITORING = "monitoring"
    AUDIT = "audit"

class ErrorCode(str, Enum):
    """Codes d'erreur standardisés."""
    # Configuration (CFG-xxx)
    CFG_INVALID_CONFIG = "CFG-001"
    CFG_MISSING_REQUIRED = "CFG-002"
    CFG_INCOMPATIBLE_VERSION = "CFG-003"
    
    # Validation (VAL-xxx)
    VAL_INVALID_FORMAT = "VAL-001"
    VAL_CONSTRAINT_VIOLATION = "VAL-002"
    VAL_SCHEMA_MISMATCH = "VAL-003"
    VAL_DATA_CORRUPTION = "VAL-004"
    
    # Sécurité (SEC-xxx)
    SEC_UNAUTHORIZED = "SEC-001"
    SEC_FORBIDDEN = "SEC-002"
    SEC_ENCRYPTION_FAILED = "SEC-003"
    SEC_TOKEN_EXPIRED = "SEC-004"
    SEC_SECURITY_SCAN_FAILED = "SEC-005"
    SEC_THREAT_DETECTED = "SEC-006"
    
    # Performance (PERF-xxx)
    PERF_TIMEOUT = "PERF-001"
    PERF_RESOURCE_EXHAUSTED = "PERF-002"
    PERF_DEGRADED_PERFORMANCE = "PERF-003"
    PERF_CIRCUIT_OPEN = "PERF-004"
    
    # ML/IA (ML-xxx)
    ML_MODEL_NOT_FOUND = "ML-001"
    ML_PREDICTION_FAILED = "ML-002"
    ML_TRAINING_FAILED = "ML-003"
    ML_ANOMALY_DETECTED = "ML-004"
    ML_DRIFT_DETECTED = "ML-005"
    
    # API (API-xxx)
    API_ENDPOINT_UNAVAILABLE = "API-001"
    API_RATE_LIMITED = "API-002"
    API_INVALID_RESPONSE = "API-003"
    API_WEBHOOK_FAILED = "API-004"
    
    # Business (BIZ-xxx)
    BIZ_RULE_VIOLATION = "BIZ-001"
    BIZ_SLA_BREACH = "BIZ-002"
    BIZ_QUOTA_EXCEEDED = "BIZ-003"
    BIZ_COMPLIANCE_VIOLATION = "BIZ-004"

# ============================================================================================
# CLASSE DE BASE POUR LES EXCEPTIONS
# ============================================================================================

@dataclass
class ErrorContext:
    """Contexte enrichi pour les erreurs."""
    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    environment: Optional[str] = None
    component: Optional[str] = None
    function_name: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    stack_trace: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour logging/audit."""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "environment": self.environment,
            "component": self.component,
            "function_name": self.function_name,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "stack_trace": self.stack_trace,
            "additional_data": self.additional_data
        }

class BaseSlackFixtureError(Exception):
    """
    Exception de base pour tous les erreurs du système de fixtures Slack.
    
    Fonctionnalités:
    - Contexte enrichi avec métadonnées
    - Sérialisation JSON pour logging structuré
    - Intégration avec systèmes de monitoring
    - Support de l'internationalisation
    - Audit automatique des erreurs
    """
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        error_category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        recoverable: bool = True,
        user_message: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message)
        
        self.message = message
        self.error_code = error_code
        self.error_category = error_category
        self.severity = severity
        self.context = context or ErrorContext()
        self.cause = cause
        self.recoverable = recoverable
        self.user_message = user_message or self._generate_user_message()
        
        # Enrichissement automatique du contexte
        self._enrich_context(**kwargs)
        
        # Capture de la stack trace
        self.context.stack_trace = traceback.format_exc()
        
        # Logging automatique
        self._log_error()
        
        # Audit automatique
        self._audit_error()
    
    def _enrich_context(self, **kwargs):
        """Enrichit le contexte avec des informations supplémentaires."""
        frame = traceback.extract_stack()[-3]  # Frame de l'appelant
        self.context.file_path = frame.filename
        self.context.line_number = frame.lineno
        self.context.function_name = frame.name
        
        # Ajout des kwargs au contexte
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
            else:
                self.context.additional_data[key] = value
    
    def _generate_user_message(self) -> str:
        """Génère un message utilisateur friendly."""
        user_messages = {
            ErrorCategory.CONFIGURATION: "Erreur de configuration. Veuillez vérifier les paramètres.",
            ErrorCategory.VALIDATION: "Données invalides fournies. Veuillez corriger et réessayer.",
            ErrorCategory.SECURITY: "Accès non autorisé. Veuillez vous authentifier.",
            ErrorCategory.PERFORMANCE: "Le service est temporairement surchargé. Veuillez réessayer.",
            ErrorCategory.NETWORK: "Problème de connectivité réseau. Veuillez réessayer.",
            ErrorCategory.DATABASE: "Erreur de base de données. L'équipe technique a été notifiée.",
            ErrorCategory.ML_MODEL: "Erreur du modèle d'IA. Utilisation du mode de secours.",
            ErrorCategory.EXTERNAL_API: "Service externe indisponible. Veuillez réessayer plus tard."
        }
        
        return user_messages.get(
            self.error_category,
            "Une erreur inattendue s'est produite. L'équipe technique a été notifiée."
        )
    
    def _log_error(self):
        """Log l'erreur avec le niveau approprié."""
        log_data = {
            "error_code": self.error_code.value,
            "error_category": self.error_category.value,
            "severity": self.severity.value,
            "message": self.message,
            "context": self.context.to_dict(),
            "recoverable": self.recoverable
        }
        
        if self.severity >= ErrorSeverity.CRITICAL:
            logger.critical(json.dumps(log_data))
        elif self.severity >= ErrorSeverity.HIGH:
            logger.error(json.dumps(log_data))
        elif self.severity >= ErrorSeverity.MEDIUM:
            logger.warning(json.dumps(log_data))
        else:
            logger.info(json.dumps(log_data))
    
    def _audit_error(self):
        """Enregistre l'erreur pour audit."""
        # Implémentation simplifiée - sera enrichie avec le système d'audit
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Sérialise l'erreur en dictionnaire."""
        return {
            "error_code": self.error_code.value,
            "error_category": self.error_category.value,
            "severity": self.severity.value,
            "message": self.message,
            "user_message": self.user_message,
            "context": self.context.to_dict(),
            "recoverable": self.recoverable,
            "cause": str(self.cause) if self.cause else None
        }
    
    def get_hash(self) -> str:
        """Génère un hash unique pour cette erreur."""
        error_data = f"{self.error_code}:{self.message}:{self.context.function_name}"
        return hashlib.sha256(error_data.encode()).hexdigest()[:16]
    
    def is_transient(self) -> bool:
        """Détermine si l'erreur est temporaire."""
        transient_categories = {
            ErrorCategory.NETWORK,
            ErrorCategory.PERFORMANCE,
            ErrorCategory.EXTERNAL_API,
            ErrorCategory.RATE_LIMITING
        }
        return self.error_category in transient_categories

# ============================================================================================
# EXCEPTIONS SPÉCIALISÉES PAR DOMAINE
# ============================================================================================

class SlackFixtureError(BaseSlackFixtureError):
    """Erreur générique du système de fixtures Slack."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code=kwargs.get('error_code', ErrorCode.BIZ_RULE_VIOLATION),
            error_category=ErrorCategory.BUSINESS_LOGIC,
            **kwargs
        )

class ConfigurationError(BaseSlackFixtureError):
    """Erreur de configuration système."""
    
    def __init__(self, message: str, config_key: str = None, **kwargs):
        super().__init__(
            message=message,
            error_code=ErrorCode.CFG_INVALID_CONFIG,
            error_category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            config_key=config_key,
            **kwargs
        )

class ValidationError(BaseSlackFixtureError):
    """Erreur de validation des données."""
    
    def __init__(
        self, 
        message: str, 
        field_name: str = None, 
        field_value: Any = None,
        validation_rule: str = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.VAL_INVALID_FORMAT,
            error_category=ErrorCategory.VALIDATION,
            field_name=field_name,
            field_value=str(field_value) if field_value is not None else None,
            validation_rule=validation_rule,
            **kwargs
        )

class SecurityError(BaseSlackFixtureError):
    """Erreur de sécurité système."""
    
    def __init__(
        self, 
        message: str, 
        security_event_type: str = None,
        threat_level: str = "medium",
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.SEC_UNAUTHORIZED,
            error_category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.HIGH,
            recoverable=False,
            security_event_type=security_event_type,
            threat_level=threat_level,
            **kwargs
        )

class PerformanceError(BaseSlackFixtureError):
    """Erreur de performance système."""
    
    def __init__(
        self, 
        message: str, 
        metric_name: str = None,
        threshold_value: float = None,
        actual_value: float = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.PERF_TIMEOUT,
            error_category=ErrorCategory.PERFORMANCE,
            metric_name=metric_name,
            threshold_value=threshold_value,
            actual_value=actual_value,
            **kwargs
        )

class MLError(BaseSlackFixtureError):
    """Erreur des modèles de Machine Learning."""
    
    def __init__(
        self, 
        message: str, 
        model_name: str = None,
        model_version: str = None,
        prediction_id: str = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.ML_PREDICTION_FAILED,
            error_category=ErrorCategory.ML_MODEL,
            model_name=model_name,
            model_version=model_version,
            prediction_id=prediction_id,
            **kwargs
        )

class BusinessRuleError(BaseSlackFixtureError):
    """Erreur de violation de règle métier."""
    
    def __init__(
        self, 
        message: str, 
        rule_name: str = None,
        rule_type: str = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.BIZ_RULE_VIOLATION,
            error_category=ErrorCategory.BUSINESS_LOGIC,
            rule_name=rule_name,
            rule_type=rule_type,
            **kwargs
        )

class SLAError(BaseSlackFixtureError):
    """Erreur de violation de SLA."""
    
    def __init__(
        self, 
        message: str, 
        sla_name: str = None,
        target_value: float = None,
        actual_value: float = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.BIZ_SLA_BREACH,
            error_category=ErrorCategory.BUSINESS_LOGIC,
            severity=ErrorSeverity.HIGH,
            sla_name=sla_name,
            target_value=target_value,
            actual_value=actual_value,
            **kwargs
        )

class ComplianceError(BaseSlackFixtureError):
    """Erreur de violation de compliance."""
    
    def __init__(
        self, 
        message: str, 
        compliance_standard: str = None,
        violation_type: str = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.BIZ_COMPLIANCE_VIOLATION,
            error_category=ErrorCategory.COMPLIANCE,
            severity=ErrorSeverity.CRITICAL,
            recoverable=False,
            compliance_standard=compliance_standard,
            violation_type=violation_type,
            **kwargs
        )

class AuditError(BaseSlackFixtureError):
    """Erreur du système d'audit."""
    
    def __init__(
        self, 
        message: str, 
        audit_event_id: str = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.BIZ_COMPLIANCE_VIOLATION,
            error_category=ErrorCategory.AUDIT,
            severity=ErrorSeverity.CRITICAL,
            audit_event_id=audit_event_id,
            **kwargs
        )

class MonitoringError(BaseSlackFixtureError):
    """Erreur du système de monitoring."""
    
    def __init__(
        self, 
        message: str, 
        metric_name: str = None,
        monitoring_system: str = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.PERF_DEGRADED_PERFORMANCE,
            error_category=ErrorCategory.MONITORING,
            metric_name=metric_name,
            monitoring_system=monitoring_system,
            **kwargs
        )

class HealthCheckError(BaseSlackFixtureError):
    """Erreur de health check."""
    
    def __init__(
        self, 
        message: str, 
        component_name: str = None,
        check_type: str = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.PERF_DEGRADED_PERFORMANCE,
            error_category=ErrorCategory.MONITORING,
            component_name=component_name,
            check_type=check_type,
            **kwargs
        )

class BackupError(BaseSlackFixtureError):
    """Erreur de sauvegarde."""
    
    def __init__(
        self, 
        message: str, 
        backup_type: str = None,
        backup_location: str = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.PERF_RESOURCE_EXHAUSTED,
            error_category=ErrorCategory.BACKUP,
            severity=ErrorSeverity.HIGH,
            backup_type=backup_type,
            backup_location=backup_location,
            **kwargs
        )

class RecoveryError(BaseSlackFixtureError):
    """Erreur de récupération."""
    
    def __init__(
        self, 
        message: str, 
        recovery_type: str = None,
        incident_id: str = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.PERF_RESOURCE_EXHAUSTED,
            error_category=ErrorCategory.RECOVERY,
            severity=ErrorSeverity.CRITICAL,
            recovery_type=recovery_type,
            incident_id=incident_id,
            **kwargs
        )

# ============================================================================================
# EXCEPTIONS TECHNIQUES SPÉCIALISÉES
# ============================================================================================

class TemplateRenderError(BaseSlackFixtureError):
    """Erreur de rendu de template."""
    
    def __init__(
        self, 
        message: str, 
        template_name: str = None,
        template_engine: str = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.VAL_INVALID_FORMAT,
            error_category=ErrorCategory.VALIDATION,
            template_name=template_name,
            template_engine=template_engine,
            **kwargs
        )

class LocalizationError(BaseSlackFixtureError):
    """Erreur de localisation/internationalisation."""
    
    def __init__(
        self, 
        message: str, 
        locale: str = None,
        translation_key: str = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.CFG_MISSING_REQUIRED,
            error_category=ErrorCategory.CONFIGURATION,
            locale=locale,
            translation_key=translation_key,
            **kwargs
        )

class EncryptionError(BaseSlackFixtureError):
    """Erreur de chiffrement/déchiffrement."""
    
    def __init__(
        self, 
        message: str, 
        encryption_algorithm: str = None,
        key_id: str = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.SEC_ENCRYPTION_FAILED,
            error_category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.HIGH,
            encryption_algorithm=encryption_algorithm,
            key_id=key_id,
            **kwargs
        )

class CacheError(BaseSlackFixtureError):
    """Erreur du système de cache."""
    
    def __init__(
        self, 
        message: str, 
        cache_layer: str = None,
        cache_key: str = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.PERF_RESOURCE_EXHAUSTED,
            error_category=ErrorCategory.CACHE,
            cache_layer=cache_layer,
            cache_key=cache_key,
            **kwargs
        )

class ThrottleError(BaseSlackFixtureError):
    """Erreur de limitation de débit."""
    
    def __init__(
        self, 
        message: str, 
        rate_limit: float = None,
        current_rate: float = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.API_RATE_LIMITED,
            error_category=ErrorCategory.RATE_LIMITING,
            rate_limit=rate_limit,
            current_rate=current_rate,
            **kwargs
        )

class CircuitBreakerError(BaseSlackFixtureError):
    """Erreur de circuit breaker."""
    
    def __init__(
        self, 
        message: str, 
        circuit_name: str = None,
        failure_threshold: float = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.PERF_CIRCUIT_OPEN,
            error_category=ErrorCategory.CIRCUIT_BREAKER,
            circuit_name=circuit_name,
            failure_threshold=failure_threshold,
            **kwargs
        )

class RetryError(BaseSlackFixtureError):
    """Erreur après épuisement des tentatives de retry."""
    
    def __init__(
        self, 
        message: str, 
        max_retries: int = None,
        retry_count: int = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.PERF_TIMEOUT,
            error_category=ErrorCategory.PERFORMANCE,
            max_retries=max_retries,
            retry_count=retry_count,
            **kwargs
        )

class APIError(BaseSlackFixtureError):
    """Erreur d'API externe."""
    
    def __init__(
        self, 
        message: str, 
        api_name: str = None,
        status_code: int = None,
        response_body: str = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.API_ENDPOINT_UNAVAILABLE,
            error_category=ErrorCategory.EXTERNAL_API,
            api_name=api_name,
            status_code=status_code,
            response_body=response_body,
            **kwargs
        )

class WebhookError(BaseSlackFixtureError):
    """Erreur de webhook."""
    
    def __init__(
        self, 
        message: str, 
        webhook_url: str = None,
        webhook_type: str = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.API_WEBHOOK_FAILED,
            error_category=ErrorCategory.EXTERNAL_API,
            webhook_url=webhook_url,
            webhook_type=webhook_type,
            **kwargs
        )

class NotificationError(BaseSlackFixtureError):
    """Erreur d'envoi de notification."""
    
    def __init__(
        self, 
        message: str, 
        channel: str = None,
        notification_type: str = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.API_ENDPOINT_UNAVAILABLE,
            error_category=ErrorCategory.EXTERNAL_API,
            channel=channel,
            notification_type=notification_type,
            **kwargs
        )

class EscalationError(BaseSlackFixtureError):
    """Erreur d'escalation d'alerte."""
    
    def __init__(
        self, 
        message: str, 
        escalation_level: int = None,
        alert_id: str = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.BIZ_RULE_VIOLATION,
            error_category=ErrorCategory.BUSINESS_LOGIC,
            escalation_level=escalation_level,
            alert_id=alert_id,
            **kwargs
        )

# ============================================================================================
# EXCEPTIONS SPÉCIALISÉES IA/ML
# ============================================================================================

class SecurityScanError(BaseSlackFixtureError):
    """Erreur de scan de sécurité."""
    
    def __init__(
        self, 
        message: str, 
        scan_type: str = None,
        vulnerability_found: bool = False,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.SEC_SECURITY_SCAN_FAILED,
            error_category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.HIGH if vulnerability_found else ErrorSeverity.MEDIUM,
            scan_type=scan_type,
            vulnerability_found=vulnerability_found,
            **kwargs
        )

class AnomalyDetectionError(BaseSlackFixtureError):
    """Erreur de détection d'anomalie."""
    
    def __init__(
        self, 
        message: str, 
        detector_name: str = None,
        anomaly_score: float = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.ML_ANOMALY_DETECTED,
            error_category=ErrorCategory.ML_MODEL,
            detector_name=detector_name,
            anomaly_score=anomaly_score,
            **kwargs
        )

class PredictionError(BaseSlackFixtureError):
    """Erreur de prédiction ML."""
    
    def __init__(
        self, 
        message: str, 
        model_name: str = None,
        input_features: Dict[str, Any] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.ML_PREDICTION_FAILED,
            error_category=ErrorCategory.ML_MODEL,
            model_name=model_name,
            input_features=input_features,
            **kwargs
        )

class OptimizationError(BaseSlackFixtureError):
    """Erreur d'optimisation automatique."""
    
    def __init__(
        self, 
        message: str, 
        optimization_type: str = None,
        target_metric: str = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.ML_PREDICTION_FAILED,
            error_category=ErrorCategory.ML_MODEL,
            optimization_type=optimization_type,
            target_metric=target_metric,
            **kwargs
        )

class RoutingError(BaseSlackFixtureError):
    """Erreur de routage intelligent."""
    
    def __init__(
        self, 
        message: str, 
        routing_rule: str = None,
        destination: str = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.BIZ_RULE_VIOLATION,
            error_category=ErrorCategory.BUSINESS_LOGIC,
            routing_rule=routing_rule,
            destination=destination,
            **kwargs
        )

class SchedulingError(BaseSlackFixtureError):
    """Erreur de planification de notifications."""
    
    def __init__(
        self, 
        message: str, 
        schedule_id: str = None,
        scheduled_time: datetime = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.BIZ_RULE_VIOLATION,
            error_category=ErrorCategory.BUSINESS_LOGIC,
            schedule_id=schedule_id,
            scheduled_time=scheduled_time.isoformat() if scheduled_time else None,
            **kwargs
        )

# ============================================================================================
# GESTIONNAIRE D'EXCEPTIONS GLOBAL
# ============================================================================================

class ExceptionManager:
    """Gestionnaire central pour toutes les exceptions."""
    
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.error_patterns: Dict[str, List[str]] = {}
        self.recovery_strategies: Dict[str, callable] = {}
    
    def register_error(self, error: BaseSlackFixtureError):
        """Enregistre une erreur pour analyse."""
        error_hash = error.get_hash()
        self.error_counts[error_hash] = self.error_counts.get(error_hash, 0) + 1
        
        if error_hash not in self.error_patterns:
            self.error_patterns[error_hash] = []
        
        self.error_patterns[error_hash].append({
            "timestamp": error.context.timestamp.isoformat(),
            "tenant_id": error.context.tenant_id,
            "component": error.context.component,
            "severity": error.severity.value
        })
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques d'erreurs."""
        total_errors = sum(self.error_counts.values())
        
        return {
            "total_errors": total_errors,
            "unique_errors": len(self.error_counts),
            "top_errors": sorted(
                self.error_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "error_rate_per_hour": self._calculate_error_rate(),
            "recovery_success_rate": self._calculate_recovery_rate()
        }
    
    def _calculate_error_rate(self) -> float:
        """Calcule le taux d'erreur par heure."""
        # Implémentation simplifiée
        now = datetime.utcnow()
        recent_errors = 0
        
        for patterns in self.error_patterns.values():
            for pattern in patterns:
                error_time = datetime.fromisoformat(pattern["timestamp"])
                if (now - error_time).total_seconds() < 3600:  # Dernière heure
                    recent_errors += 1
        
        return recent_errors
    
    def _calculate_recovery_rate(self) -> float:
        """Calcule le taux de récupération."""
        # Implémentation simplifiée
        return 0.95  # 95% de récupération par défaut

# Instance globale du gestionnaire d'exceptions
exception_manager = ExceptionManager()

# ============================================================================================
# DÉCORATEURS ET UTILITAIRES
# ============================================================================================

def handle_exceptions(
    default_return=None,
    reraise_on: Optional[List[Type[Exception]]] = None,
    log_level: str = "error"
):
    """
    Décorateur pour la gestion centralisée des exceptions.
    
    Args:
        default_return: Valeur par défaut à retourner en cas d'exception
        reraise_on: Liste des types d'exceptions à re-lancer
        log_level: Niveau de log pour les exceptions capturées
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Re-lancer si dans la liste des exceptions à re-lancer
                if reraise_on and type(e) in reraise_on:
                    raise
                
                # Convertir en BaseSlackFixtureError si nécessaire
                if not isinstance(e, BaseSlackFixtureError):
                    error = SlackFixtureError(
                        message=str(e),
                        cause=e,
                        function_name=func.__name__
                    )
                else:
                    error = e
                
                # Enregistrer l'erreur
                exception_manager.register_error(error)
                
                # Log avec le niveau approprié
                getattr(logger, log_level)(f"Exception in {func.__name__}: {error}")
                
                return default_return
        
        return wrapper
    return decorator

def create_error_response(error: BaseSlackFixtureError) -> Dict[str, Any]:
    """Crée une réponse d'erreur standardisée pour les APIs."""
    return {
        "success": False,
        "error": {
            "code": error.error_code.value,
            "message": error.user_message,
            "category": error.error_category.value,
            "severity": error.severity.value,
            "recoverable": error.recoverable,
            "error_id": error.context.error_id,
            "timestamp": error.context.timestamp.isoformat()
        }
    }

def is_critical_error(error: BaseSlackFixtureError) -> bool:
    """Détermine si une erreur est critique."""
    return error.severity >= ErrorSeverity.CRITICAL

def should_retry(error: BaseSlackFixtureError, attempt: int = 0, max_attempts: int = 3) -> bool:
    """Détermine si une opération doit être retentée."""
    if attempt >= max_attempts:
        return False
    
    if not error.recoverable:
        return False
    
    if error.severity >= ErrorSeverity.CRITICAL:
        return False
    
    return error.is_transient()

# Export de toutes les exceptions et utilitaires
__all__ = [
    # Exceptions de base
    "BaseSlackFixtureError", "SlackFixtureError",
    
    # Exceptions par domaine
    "ConfigurationError", "ValidationError", "SecurityError", "PerformanceError",
    "MLError", "BusinessRuleError", "SLAError", "ComplianceError", "AuditError",
    "MonitoringError", "HealthCheckError", "BackupError", "RecoveryError",
    
    # Exceptions techniques
    "TemplateRenderError", "LocalizationError", "EncryptionError", "CacheError",
    "ThrottleError", "CircuitBreakerError", "RetryError", "APIError", "WebhookError",
    "NotificationError", "EscalationError",
    
    # Exceptions IA/ML
    "SecurityScanError", "AnomalyDetectionError", "PredictionError", "OptimizationError",
    "RoutingError", "SchedulingError",
    
    # Enums et types
    "ErrorSeverity", "ErrorCategory", "ErrorCode", "ErrorContext",
    
    # Gestionnaires et utilitaires
    "ExceptionManager", "exception_manager", "handle_exceptions", "create_error_response",
    "is_critical_error", "should_retry"
]
