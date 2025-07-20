"""
Security Exceptions Module
=========================

Ce module définit les exceptions personnalisées pour le système de sécurité
multi-tenant du Spotify AI Agent.

Auteur: Fahed Mlaiel
"""

from typing import Optional, Dict, Any, List
from enum import Enum


class SecurityErrorCode(Enum):
    """Codes d'erreur de sécurité"""
    # Authentification
    AUTHENTICATION_FAILED = "AUTH_001"
    INVALID_CREDENTIALS = "AUTH_002"
    ACCOUNT_LOCKED = "AUTH_003"
    SESSION_EXPIRED = "AUTH_004"
    TOKEN_INVALID = "AUTH_005"
    TOKEN_EXPIRED = "AUTH_006"
    
    # Autorisation
    AUTHORIZATION_FAILED = "AUTHZ_001"
    INSUFFICIENT_PERMISSIONS = "AUTHZ_002"
    RESOURCE_ACCESS_DENIED = "AUTHZ_003"
    TENANT_ACCESS_DENIED = "AUTHZ_004"
    ROLE_NOT_FOUND = "AUTHZ_005"
    
    # Validation
    VALIDATION_FAILED = "VAL_001"
    INVALID_INPUT = "VAL_002"
    SCHEMA_VIOLATION = "VAL_003"
    CONSTRAINT_VIOLATION = "VAL_004"
    
    # Chiffrement
    ENCRYPTION_FAILED = "ENC_001"
    DECRYPTION_FAILED = "ENC_002"
    KEY_NOT_FOUND = "ENC_003"
    KEY_EXPIRED = "ENC_004"
    INVALID_ALGORITHM = "ENC_005"
    
    # Rate Limiting
    RATE_LIMIT_EXCEEDED = "RATE_001"
    QUOTA_EXCEEDED = "RATE_002"
    
    # Sécurité réseau
    IP_BLOCKED = "NET_001"
    SUSPICIOUS_ACTIVITY = "NET_002"
    GEOLOCATION_BLOCKED = "NET_003"
    
    # Compliance
    GDPR_VIOLATION = "COMP_001"
    DATA_RETENTION_VIOLATION = "COMP_002"
    AUDIT_REQUIRED = "COMP_003"
    
    # Configuration
    INVALID_CONFIGURATION = "CONF_001"
    MISSING_CONFIGURATION = "CONF_002"
    
    # Monitoring
    MONITORING_FAILED = "MON_001"
    ALERT_FAILED = "MON_002"
    
    # General
    SECURITY_POLICY_VIOLATION = "SEC_001"
    THREAT_DETECTED = "SEC_002"
    INTRUSION_DETECTED = "SEC_003"
    MALICIOUS_REQUEST = "SEC_004"


class SecurityExceptionSeverity(Enum):
    """Niveaux de sévérité des exceptions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BaseSecurityException(Exception):
    """
    Exception de base pour toutes les erreurs de sécurité
    """
    
    def __init__(
        self,
        message: str,
        error_code: SecurityErrorCode,
        severity: SecurityExceptionSeverity = SecurityExceptionSeverity.MEDIUM,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        log_event: bool = True,
        alert_required: bool = False
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.resource = resource
        self.details = details or {}
        self.log_event = log_event
        self.alert_required = alert_required
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'exception en dictionnaire"""
        return {
            "error_code": self.error_code.value,
            "message": self.message,
            "severity": self.severity.value,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "resource": self.resource,
            "details": self.details,
            "exception_type": self.__class__.__name__
        }
    
    def __str__(self) -> str:
        return f"[{self.error_code.value}] {self.message}"


# =============================================================================
# Exceptions d'Authentification
# =============================================================================

class AuthenticationException(BaseSecurityException):
    """Exception générale d'authentification"""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.AUTHENTICATION_FAILED,
            severity=SecurityExceptionSeverity.HIGH,
            alert_required=True,
            **kwargs
        )


class InvalidCredentialsException(AuthenticationException):
    """Identifiants invalides"""
    
    def __init__(self, message: str = "Invalid credentials provided", **kwargs):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.INVALID_CREDENTIALS,
            **kwargs
        )


class AccountLockedException(AuthenticationException):
    """Compte verrouillé"""
    
    def __init__(self, message: str = "Account is locked", lock_until: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.ACCOUNT_LOCKED,
            severity=SecurityExceptionSeverity.CRITICAL,
            details={"lock_until": lock_until} if lock_until else {},
            **kwargs
        )


class SessionExpiredException(AuthenticationException):
    """Session expirée"""
    
    def __init__(self, message: str = "Session has expired", **kwargs):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.SESSION_EXPIRED,
            severity=SecurityExceptionSeverity.MEDIUM,
            **kwargs
        )


class InvalidTokenException(AuthenticationException):
    """Token invalide"""
    
    def __init__(self, message: str = "Invalid or malformed token", token_type: str = "unknown", **kwargs):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.TOKEN_INVALID,
            details={"token_type": token_type},
            **kwargs
        )


class TokenExpiredException(AuthenticationException):
    """Token expiré"""
    
    def __init__(self, message: str = "Token has expired", expired_at: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.TOKEN_EXPIRED,
            details={"expired_at": expired_at} if expired_at else {},
            **kwargs
        )


# =============================================================================
# Exceptions d'Autorisation
# =============================================================================

class AuthorizationException(BaseSecurityException):
    """Exception générale d'autorisation"""
    
    def __init__(self, message: str = "Authorization failed", **kwargs):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.AUTHORIZATION_FAILED,
            severity=SecurityExceptionSeverity.HIGH,
            alert_required=True,
            **kwargs
        )


class InsufficientPermissionsException(AuthorizationException):
    """Permissions insuffisantes"""
    
    def __init__(
        self,
        message: str = "Insufficient permissions for this operation",
        required_permissions: Optional[List[str]] = None,
        user_permissions: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.INSUFFICIENT_PERMISSIONS,
            details={
                "required_permissions": required_permissions or [],
                "user_permissions": user_permissions or []
            },
            **kwargs
        )


class ResourceAccessDeniedException(AuthorizationException):
    """Accès à la ressource refusé"""
    
    def __init__(self, message: str = "Access to resource denied", resource_id: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.RESOURCE_ACCESS_DENIED,
            details={"resource_id": resource_id} if resource_id else {},
            **kwargs
        )


class TenantAccessDeniedException(AuthorizationException):
    """Accès au tenant refusé"""
    
    def __init__(self, message: str = "Access to tenant denied", attempted_tenant: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.TENANT_ACCESS_DENIED,
            details={"attempted_tenant": attempted_tenant} if attempted_tenant else {},
            **kwargs
        )


class RoleNotFoundException(AuthorizationException):
    """Rôle non trouvé"""
    
    def __init__(self, message: str = "Role not found", role_name: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.ROLE_NOT_FOUND,
            severity=SecurityExceptionSeverity.MEDIUM,
            details={"role_name": role_name} if role_name else {},
            **kwargs
        )


# =============================================================================
# Exceptions de Validation
# =============================================================================

class ValidationException(BaseSecurityException):
    """Exception de validation"""
    
    def __init__(self, message: str = "Validation failed", validation_errors: Optional[List[str]] = None, **kwargs):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.VALIDATION_FAILED,
            severity=SecurityExceptionSeverity.MEDIUM,
            details={"validation_errors": validation_errors or []},
            **kwargs
        )


class InvalidInputException(ValidationException):
    """Entrée invalide"""
    
    def __init__(self, message: str = "Invalid input provided", field_name: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.INVALID_INPUT,
            details={"field_name": field_name} if field_name else {},
            **kwargs
        )


class SchemaViolationException(ValidationException):
    """Violation de schéma"""
    
    def __init__(self, message: str = "Schema validation failed", schema_errors: Optional[List[str]] = None, **kwargs):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.SCHEMA_VIOLATION,
            details={"schema_errors": schema_errors or []},
            **kwargs
        )


class ConstraintViolationException(ValidationException):
    """Violation de contrainte"""
    
    def __init__(self, message: str = "Constraint violation", constraint_name: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.CONSTRAINT_VIOLATION,
            details={"constraint_name": constraint_name} if constraint_name else {},
            **kwargs
        )


# =============================================================================
# Exceptions de Chiffrement
# =============================================================================

class EncryptionException(BaseSecurityException):
    """Exception de chiffrement"""
    
    def __init__(self, message: str = "Encryption operation failed", **kwargs):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.ENCRYPTION_FAILED,
            severity=SecurityExceptionSeverity.HIGH,
            alert_required=True,
            **kwargs
        )


class DecryptionException(BaseSecurityException):
    """Exception de déchiffrement"""
    
    def __init__(self, message: str = "Decryption operation failed", **kwargs):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.DECRYPTION_FAILED,
            severity=SecurityExceptionSeverity.HIGH,
            alert_required=True,
            **kwargs
        )


class KeyNotFoundException(EncryptionException):
    """Clé de chiffrement non trouvée"""
    
    def __init__(self, message: str = "Encryption key not found", key_id: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.KEY_NOT_FOUND,
            details={"key_id": key_id} if key_id else {},
            **kwargs
        )


class KeyExpiredException(EncryptionException):
    """Clé de chiffrement expirée"""
    
    def __init__(self, message: str = "Encryption key has expired", key_id: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.KEY_EXPIRED,
            details={"key_id": key_id} if key_id else {},
            **kwargs
        )


class InvalidAlgorithmException(EncryptionException):
    """Algorithme de chiffrement invalide"""
    
    def __init__(self, message: str = "Invalid encryption algorithm", algorithm: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.INVALID_ALGORITHM,
            details={"algorithm": algorithm} if algorithm else {},
            **kwargs
        )


# =============================================================================
# Exceptions de Rate Limiting
# =============================================================================

class RateLimitExceededException(BaseSecurityException):
    """Limite de taux dépassée"""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        limit: Optional[int] = None,
        window_seconds: Optional[int] = None,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.RATE_LIMIT_EXCEEDED,
            severity=SecurityExceptionSeverity.MEDIUM,
            details={
                "limit": limit,
                "window_seconds": window_seconds,
                "retry_after": retry_after
            },
            **kwargs
        )


class QuotaExceededException(BaseSecurityException):
    """Quota dépassé"""
    
    def __init__(
        self,
        message: str = "Quota exceeded",
        quota_type: Optional[str] = None,
        current_usage: Optional[int] = None,
        quota_limit: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.QUOTA_EXCEEDED,
            severity=SecurityExceptionSeverity.HIGH,
            details={
                "quota_type": quota_type,
                "current_usage": current_usage,
                "quota_limit": quota_limit
            },
            **kwargs
        )


# =============================================================================
# Exceptions de Sécurité Réseau
# =============================================================================

class IPBlockedException(BaseSecurityException):
    """IP bloquée"""
    
    def __init__(self, message: str = "IP address is blocked", ip_address: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.IP_BLOCKED,
            severity=SecurityExceptionSeverity.HIGH,
            alert_required=True,
            details={"ip_address": ip_address} if ip_address else {},
            **kwargs
        )


class SuspiciousActivityException(BaseSecurityException):
    """Activité suspecte détectée"""
    
    def __init__(
        self,
        message: str = "Suspicious activity detected",
        activity_type: Optional[str] = None,
        risk_score: Optional[float] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.SUSPICIOUS_ACTIVITY,
            severity=SecurityExceptionSeverity.HIGH,
            alert_required=True,
            details={
                "activity_type": activity_type,
                "risk_score": risk_score
            },
            **kwargs
        )


class GeolocationBlockedException(BaseSecurityException):
    """Géolocalisation bloquée"""
    
    def __init__(
        self,
        message: str = "Access from this location is blocked",
        country_code: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.GEOLOCATION_BLOCKED,
            severity=SecurityExceptionSeverity.HIGH,
            details={"country_code": country_code} if country_code else {},
            **kwargs
        )


# =============================================================================
# Exceptions de Compliance
# =============================================================================

class GDPRViolationException(BaseSecurityException):
    """Violation GDPR"""
    
    def __init__(
        self,
        message: str = "GDPR compliance violation",
        violation_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.GDPR_VIOLATION,
            severity=SecurityExceptionSeverity.CRITICAL,
            alert_required=True,
            details={"violation_type": violation_type} if violation_type else {},
            **kwargs
        )


class DataRetentionViolationException(BaseSecurityException):
    """Violation de rétention des données"""
    
    def __init__(
        self,
        message: str = "Data retention policy violation",
        retention_period: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.DATA_RETENTION_VIOLATION,
            severity=SecurityExceptionSeverity.HIGH,
            alert_required=True,
            details={"retention_period": retention_period} if retention_period else {},
            **kwargs
        )


class AuditRequiredException(BaseSecurityException):
    """Audit requis"""
    
    def __init__(self, message: str = "Audit trail is required for this operation", **kwargs):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.AUDIT_REQUIRED,
            severity=SecurityExceptionSeverity.MEDIUM,
            **kwargs
        )


# =============================================================================
# Exceptions de Configuration
# =============================================================================

class InvalidConfigurationException(BaseSecurityException):
    """Configuration invalide"""
    
    def __init__(
        self,
        message: str = "Invalid security configuration",
        config_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.INVALID_CONFIGURATION,
            severity=SecurityExceptionSeverity.HIGH,
            details={"config_key": config_key} if config_key else {},
            **kwargs
        )


class MissingConfigurationException(BaseSecurityException):
    """Configuration manquante"""
    
    def __init__(
        self,
        message: str = "Required security configuration is missing",
        missing_config: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.MISSING_CONFIGURATION,
            severity=SecurityExceptionSeverity.HIGH,
            details={"missing_config": missing_config} if missing_config else {},
            **kwargs
        )


# =============================================================================
# Exceptions de Monitoring
# =============================================================================

class MonitoringFailedException(BaseSecurityException):
    """Échec du monitoring"""
    
    def __init__(self, message: str = "Security monitoring failed", monitor_type: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.MONITORING_FAILED,
            severity=SecurityExceptionSeverity.HIGH,
            details={"monitor_type": monitor_type} if monitor_type else {},
            **kwargs
        )


class AlertFailedException(BaseSecurityException):
    """Échec d'alerte"""
    
    def __init__(self, message: str = "Security alert failed", alert_type: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.ALERT_FAILED,
            severity=SecurityExceptionSeverity.HIGH,
            details={"alert_type": alert_type} if alert_type else {},
            **kwargs
        )


# =============================================================================
# Exceptions Générales de Sécurité
# =============================================================================

class SecurityPolicyViolationException(BaseSecurityException):
    """Violation de politique de sécurité"""
    
    def __init__(
        self,
        message: str = "Security policy violation",
        policy_name: Optional[str] = None,
        violation_details: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.SECURITY_POLICY_VIOLATION,
            severity=SecurityExceptionSeverity.HIGH,
            alert_required=True,
            details={
                "policy_name": policy_name,
                "violation_details": violation_details
            },
            **kwargs
        )


class ThreatDetectedException(BaseSecurityException):
    """Menace détectée"""
    
    def __init__(
        self,
        message: str = "Security threat detected",
        threat_type: Optional[str] = None,
        threat_score: Optional[float] = None,
        threat_indicators: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.THREAT_DETECTED,
            severity=SecurityExceptionSeverity.CRITICAL,
            alert_required=True,
            details={
                "threat_type": threat_type,
                "threat_score": threat_score,
                "threat_indicators": threat_indicators or []
            },
            **kwargs
        )


class IntrusionDetectedException(BaseSecurityException):
    """Intrusion détectée"""
    
    def __init__(
        self,
        message: str = "Intrusion attempt detected",
        intrusion_type: Optional[str] = None,
        source_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.INTRUSION_DETECTED,
            severity=SecurityExceptionSeverity.CRITICAL,
            alert_required=True,
            details={
                "intrusion_type": intrusion_type,
                "source_info": source_info or {}
            },
            **kwargs
        )


class MaliciousRequestException(BaseSecurityException):
    """Requête malveillante"""
    
    def __init__(
        self,
        message: str = "Malicious request detected",
        request_signature: Optional[str] = None,
        attack_pattern: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=SecurityErrorCode.MALICIOUS_REQUEST,
            severity=SecurityExceptionSeverity.HIGH,
            alert_required=True,
            details={
                "request_signature": request_signature,
                "attack_pattern": attack_pattern
            },
            **kwargs
        )


# =============================================================================
# Utilitaires pour les Exceptions
# =============================================================================

def get_exception_by_code(error_code: SecurityErrorCode) -> type:
    """Retourne la classe d'exception correspondant à un code d'erreur"""
    exception_mapping = {
        # Authentification
        SecurityErrorCode.AUTHENTICATION_FAILED: AuthenticationException,
        SecurityErrorCode.INVALID_CREDENTIALS: InvalidCredentialsException,
        SecurityErrorCode.ACCOUNT_LOCKED: AccountLockedException,
        SecurityErrorCode.SESSION_EXPIRED: SessionExpiredException,
        SecurityErrorCode.TOKEN_INVALID: InvalidTokenException,
        SecurityErrorCode.TOKEN_EXPIRED: TokenExpiredException,
        
        # Autorisation
        SecurityErrorCode.AUTHORIZATION_FAILED: AuthorizationException,
        SecurityErrorCode.INSUFFICIENT_PERMISSIONS: InsufficientPermissionsException,
        SecurityErrorCode.RESOURCE_ACCESS_DENIED: ResourceAccessDeniedException,
        SecurityErrorCode.TENANT_ACCESS_DENIED: TenantAccessDeniedException,
        SecurityErrorCode.ROLE_NOT_FOUND: RoleNotFoundException,
        
        # Validation
        SecurityErrorCode.VALIDATION_FAILED: ValidationException,
        SecurityErrorCode.INVALID_INPUT: InvalidInputException,
        SecurityErrorCode.SCHEMA_VIOLATION: SchemaViolationException,
        SecurityErrorCode.CONSTRAINT_VIOLATION: ConstraintViolationException,
        
        # Chiffrement
        SecurityErrorCode.ENCRYPTION_FAILED: EncryptionException,
        SecurityErrorCode.DECRYPTION_FAILED: DecryptionException,
        SecurityErrorCode.KEY_NOT_FOUND: KeyNotFoundException,
        SecurityErrorCode.KEY_EXPIRED: KeyExpiredException,
        SecurityErrorCode.INVALID_ALGORITHM: InvalidAlgorithmException,
        
        # Rate Limiting
        SecurityErrorCode.RATE_LIMIT_EXCEEDED: RateLimitExceededException,
        SecurityErrorCode.QUOTA_EXCEEDED: QuotaExceededException,
        
        # Sécurité réseau
        SecurityErrorCode.IP_BLOCKED: IPBlockedException,
        SecurityErrorCode.SUSPICIOUS_ACTIVITY: SuspiciousActivityException,
        SecurityErrorCode.GEOLOCATION_BLOCKED: GeolocationBlockedException,
        
        # Compliance
        SecurityErrorCode.GDPR_VIOLATION: GDPRViolationException,
        SecurityErrorCode.DATA_RETENTION_VIOLATION: DataRetentionViolationException,
        SecurityErrorCode.AUDIT_REQUIRED: AuditRequiredException,
        
        # Configuration
        SecurityErrorCode.INVALID_CONFIGURATION: InvalidConfigurationException,
        SecurityErrorCode.MISSING_CONFIGURATION: MissingConfigurationException,
        
        # Monitoring
        SecurityErrorCode.MONITORING_FAILED: MonitoringFailedException,
        SecurityErrorCode.ALERT_FAILED: AlertFailedException,
        
        # Général
        SecurityErrorCode.SECURITY_POLICY_VIOLATION: SecurityPolicyViolationException,
        SecurityErrorCode.THREAT_DETECTED: ThreatDetectedException,
        SecurityErrorCode.INTRUSION_DETECTED: IntrusionDetectedException,
        SecurityErrorCode.MALICIOUS_REQUEST: MaliciousRequestException,
    }
    
    return exception_mapping.get(error_code, BaseSecurityException)


def create_security_exception(
    error_code: SecurityErrorCode,
    message: Optional[str] = None,
    **kwargs
) -> BaseSecurityException:
    """Crée une exception de sécurité basée sur le code d'erreur"""
    exception_class = get_exception_by_code(error_code)
    
    if message:
        return exception_class(message=message, **kwargs)
    else:
        return exception_class(**kwargs)


def is_critical_exception(exception: BaseSecurityException) -> bool:
    """Vérifie si une exception est critique"""
    return exception.severity == SecurityExceptionSeverity.CRITICAL


def requires_alert(exception: BaseSecurityException) -> bool:
    """Vérifie si une exception nécessite une alerte"""
    return exception.alert_required


def should_log_exception(exception: BaseSecurityException) -> bool:
    """Vérifie si une exception doit être loggée"""
    return exception.log_event
