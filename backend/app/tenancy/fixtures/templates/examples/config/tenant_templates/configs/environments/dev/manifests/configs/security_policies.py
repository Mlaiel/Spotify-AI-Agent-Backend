"""
Security Policies and Compliance Framework
=========================================

Définit les politiques de sécurité et le framework de conformité pour
l'application Spotify AI Agent. Inclut les configurations de sécurité,
les politiques d'accès, et les exigences de conformité.

Author: Fahed Mlaiel
Team: Spotify AI Agent Development Team
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import re

class SecurityLevel(Enum):
    """Niveaux de sécurité."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ComplianceStandard(Enum):
    """Standards de conformité."""
    GDPR = "gdpr"
    CCPA = "ccpa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    HIPAA = "hipaa"
    ISO27001 = "iso27001"

class AccessLevel(Enum):
    """Niveaux d'accès."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

@dataclass
class PasswordPolicy:
    """Politique de mots de passe."""
    min_length: int = 12
    max_length: int = 128
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_digits: bool = True
    require_special_chars: bool = True
    special_chars: str = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    max_consecutive_chars: int = 3
    history_check: int = 5  # Nombre de mots de passe précédents à vérifier
    expire_days: int = 90
    warning_days: int = 15
    complexity_score_min: int = 80

@dataclass
class SessionPolicy:
    """Politique de session."""
    max_session_duration: timedelta = field(default_factory=lambda: timedelta(hours=8))
    idle_timeout: timedelta = field(default_factory=lambda: timedelta(minutes=30))
    max_concurrent_sessions: int = 5
    require_re_auth_for_sensitive: bool = True
    session_rotation_interval: timedelta = field(default_factory=lambda: timedelta(hours=2))
    secure_cookies: bool = True
    httponly_cookies: bool = True
    samesite_cookies: str = "Strict"

@dataclass
class RateLimitPolicy:
    """Politique de limitation de débit."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10
    window_size: int = 60  # secondes
    enable_whitelist: bool = True
    enable_blacklist: bool = True
    throttle_after_attempts: int = 5

@dataclass
class EncryptionPolicy:
    """Politique de chiffrement."""
    algorithm: str = "AES-256-GCM"
    key_rotation_days: int = 90
    use_hsm: bool = False  # Hardware Security Module
    encryption_at_rest: bool = True
    encryption_in_transit: bool = True
    key_derivation_function: str = "PBKDF2"
    salt_length: int = 32
    iterations: int = 100000

@dataclass
class AuditPolicy:
    """Politique d'audit."""
    enable_audit_logging: bool = True
    log_all_api_calls: bool = True
    log_authentication_events: bool = True
    log_authorization_events: bool = True
    log_data_access: bool = True
    log_data_modification: bool = True
    log_admin_actions: bool = True
    retention_days: int = 365
    export_format: str = "json"
    real_time_alerts: bool = True

class SecurityPolicyManager:
    """Gestionnaire des politiques de sécurité."""
    
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self.policies = self._initialize_policies()
        self.compliance_requirements = self._initialize_compliance()
    
    def _initialize_policies(self) -> Dict[str, Any]:
        """Initialise les politiques de sécurité."""
        if self.environment == "production":
            return self._get_production_policies()
        elif self.environment == "staging":
            return self._get_staging_policies()
        else:
            return self._get_development_policies()
    
    def _get_development_policies(self) -> Dict[str, Any]:
        """Politiques pour l'environnement de développement."""
        return {
            "password_policy": PasswordPolicy(
                min_length=8,
                require_special_chars=False,
                expire_days=180,
                history_check=3,
                complexity_score_min=60
            ),
            "session_policy": SessionPolicy(
                max_session_duration=timedelta(hours=12),
                idle_timeout=timedelta(hours=2),
                max_concurrent_sessions=10,
                require_re_auth_for_sensitive=False,
                secure_cookies=False
            ),
            "rate_limit_policy": RateLimitPolicy(
                requests_per_minute=120,
                requests_per_hour=5000,
                requests_per_day=50000,
                throttle_after_attempts=10
            ),
            "encryption_policy": EncryptionPolicy(
                key_rotation_days=180,
                use_hsm=False,
                iterations=50000
            ),
            "audit_policy": AuditPolicy(
                log_all_api_calls=False,
                retention_days=90,
                real_time_alerts=False
            )
        }
    
    def _get_staging_policies(self) -> Dict[str, Any]:
        """Politiques pour l'environnement de staging."""
        return {
            "password_policy": PasswordPolicy(
                min_length=10,
                require_special_chars=True,
                expire_days=120,
                history_check=4,
                complexity_score_min=70
            ),
            "session_policy": SessionPolicy(
                max_session_duration=timedelta(hours=10),
                idle_timeout=timedelta(minutes=45),
                max_concurrent_sessions=7,
                require_re_auth_for_sensitive=True,
                secure_cookies=True
            ),
            "rate_limit_policy": RateLimitPolicy(
                requests_per_minute=80,
                requests_per_hour=2000,
                requests_per_day=20000,
                throttle_after_attempts=7
            ),
            "encryption_policy": EncryptionPolicy(
                key_rotation_days=120,
                use_hsm=False,
                iterations=75000
            ),
            "audit_policy": AuditPolicy(
                log_all_api_calls=True,
                retention_days=180,
                real_time_alerts=True
            )
        }
    
    def _get_production_policies(self) -> Dict[str, Any]:
        """Politiques pour l'environnement de production."""
        return {
            "password_policy": PasswordPolicy(
                min_length=12,
                require_special_chars=True,
                expire_days=90,
                history_check=5,
                complexity_score_min=80,
                max_consecutive_chars=2
            ),
            "session_policy": SessionPolicy(
                max_session_duration=timedelta(hours=8),
                idle_timeout=timedelta(minutes=30),
                max_concurrent_sessions=5,
                require_re_auth_for_sensitive=True,
                session_rotation_interval=timedelta(hours=1),
                secure_cookies=True,
                samesite_cookies="Strict"
            ),
            "rate_limit_policy": RateLimitPolicy(
                requests_per_minute=60,
                requests_per_hour=1000,
                requests_per_day=10000,
                throttle_after_attempts=5,
                burst_limit=5
            ),
            "encryption_policy": EncryptionPolicy(
                key_rotation_days=90,
                use_hsm=True,
                iterations=100000,
                encryption_at_rest=True,
                encryption_in_transit=True
            ),
            "audit_policy": AuditPolicy(
                enable_audit_logging=True,
                log_all_api_calls=True,
                log_authentication_events=True,
                log_authorization_events=True,
                log_data_access=True,
                log_data_modification=True,
                log_admin_actions=True,
                retention_days=365,
                real_time_alerts=True
            )
        }
    
    def _initialize_compliance(self) -> Dict[ComplianceStandard, Dict[str, Any]]:
        """Initialise les exigences de conformité."""
        return {
            ComplianceStandard.GDPR: {
                "data_retention_max_days": 365,
                "consent_required": True,
                "right_to_deletion": True,
                "data_portability": True,
                "privacy_by_design": True,
                "dpo_required": True,
                "breach_notification_hours": 72
            },
            ComplianceStandard.CCPA: {
                "data_disclosure_required": True,
                "opt_out_right": True,
                "non_discrimination": True,
                "data_deletion_right": True,
                "third_party_sharing_disclosure": True
            },
            ComplianceStandard.SOX: {
                "financial_data_protection": True,
                "audit_trail_required": True,
                "management_certification": True,
                "internal_controls": True
            },
            ComplianceStandard.PCI_DSS: {
                "card_data_encryption": True,
                "network_security": True,
                "access_control": True,
                "monitoring_required": True,
                "security_testing": True,
                "security_policy_maintenance": True
            },
            ComplianceStandard.ISO27001: {
                "isms_required": True,
                "risk_assessment": True,
                "security_controls": True,
                "continuous_improvement": True,
                "documentation_required": True
            }
        }
    
    def get_policy(self, policy_type: str) -> Optional[Any]:
        """Récupère une politique de sécurité."""
        return self.policies.get(policy_type)
    
    def validate_password(self, password: str) -> Dict[str, Any]:
        """Valide un mot de passe selon la politique."""
        policy = self.get_policy("password_policy")
        if not policy:
            return {"valid": False, "errors": ["No password policy found"]}
        
        errors = []
        score = 0
        
        # Vérifications de base
        if len(password) < policy.min_length:
            errors.append(f"Password must be at least {policy.min_length} characters long")
        elif len(password) > policy.max_length:
            errors.append(f"Password must not exceed {policy.max_length} characters")
        else:
            score += 20
        
        # Vérifications de complexité
        if policy.require_uppercase and not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        else:
            score += 15
        
        if policy.require_lowercase and not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        else:
            score += 15
        
        if policy.require_digits and not re.search(r'\d', password):
            errors.append("Password must contain at least one digit")
        else:
            score += 15
        
        if policy.require_special_chars and not re.search(f'[{re.escape(policy.special_chars)}]', password):
            errors.append(f"Password must contain at least one special character: {policy.special_chars}")
        else:
            score += 15
        
        # Vérification des caractères consécutifs
        consecutive_count = 1
        max_consecutive = 1
        for i in range(1, len(password)):
            if password[i] == password[i-1]:
                consecutive_count += 1
                max_consecutive = max(max_consecutive, consecutive_count)
            else:
                consecutive_count = 1
        
        if max_consecutive > policy.max_consecutive_chars:
            errors.append(f"Password must not have more than {policy.max_consecutive_chars} consecutive identical characters")
        else:
            score += 10
        
        # Calcul du score de complexité
        entropy_score = min(10, len(set(password))) * 2  # Diversité des caractères
        score += entropy_score
        
        return {
            "valid": len(errors) == 0 and score >= policy.complexity_score_min,
            "errors": errors,
            "score": score,
            "min_score_required": policy.complexity_score_min
        }
    
    def check_session_validity(self, session_start: datetime, 
                             last_activity: datetime) -> Dict[str, Any]:
        """Vérifie la validité d'une session."""
        policy = self.get_policy("session_policy")
        if not policy:
            return {"valid": False, "reason": "No session policy found"}
        
        now = datetime.now()
        session_duration = now - session_start
        idle_duration = now - last_activity
        
        if session_duration > policy.max_session_duration:
            return {
                "valid": False,
                "reason": "Session duration exceeded",
                "max_duration": str(policy.max_session_duration)
            }
        
        if idle_duration > policy.idle_timeout:
            return {
                "valid": False,
                "reason": "Session idle timeout exceeded",
                "idle_timeout": str(policy.idle_timeout)
            }
        
        return {"valid": True}
    
    def should_rotate_session(self, session_start: datetime) -> bool:
        """Vérifie si une session doit être renouvelée."""
        policy = self.get_policy("session_policy")
        if not policy:
            return False
        
        session_duration = datetime.now() - session_start
        return session_duration >= policy.session_rotation_interval
    
    def check_rate_limit(self, request_count: int, 
                        window_start: datetime) -> Dict[str, Any]:
        """Vérifie les limites de débit."""
        policy = self.get_policy("rate_limit_policy")
        if not policy:
            return {"allowed": True}
        
        window_duration = datetime.now() - window_start
        
        # Vérification par minute
        if window_duration <= timedelta(minutes=1):
            if request_count > policy.requests_per_minute:
                return {
                    "allowed": False,
                    "reason": "Rate limit exceeded (per minute)",
                    "limit": policy.requests_per_minute,
                    "current": request_count
                }
        
        # Vérification par heure
        if window_duration <= timedelta(hours=1):
            if request_count > policy.requests_per_hour:
                return {
                    "allowed": False,
                    "reason": "Rate limit exceeded (per hour)",
                    "limit": policy.requests_per_hour,
                    "current": request_count
                }
        
        return {"allowed": True}
    
    def get_encryption_config(self) -> Dict[str, Any]:
        """Retourne la configuration de chiffrement."""
        policy = self.get_policy("encryption_policy")
        if not policy:
            return {}
        
        return {
            "algorithm": policy.algorithm,
            "key_derivation_function": policy.key_derivation_function,
            "salt_length": policy.salt_length,
            "iterations": policy.iterations,
            "encryption_at_rest": policy.encryption_at_rest,
            "encryption_in_transit": policy.encryption_in_transit
        }
    
    def get_audit_config(self) -> Dict[str, Any]:
        """Retourne la configuration d'audit."""
        policy = self.get_policy("audit_policy")
        if not policy:
            return {}
        
        return {
            "enable_audit_logging": policy.enable_audit_logging,
            "log_all_api_calls": policy.log_all_api_calls,
            "log_authentication_events": policy.log_authentication_events,
            "log_authorization_events": policy.log_authorization_events,
            "log_data_access": policy.log_data_access,
            "log_data_modification": policy.log_data_modification,
            "log_admin_actions": policy.log_admin_actions,
            "retention_days": policy.retention_days,
            "export_format": policy.export_format,
            "real_time_alerts": policy.real_time_alerts
        }
    
    def get_compliance_requirements(self, 
                                  standard: ComplianceStandard) -> Dict[str, Any]:
        """Retourne les exigences d'un standard de conformité."""
        return self.compliance_requirements.get(standard, {})
    
    def export_security_config(self) -> Dict[str, Any]:
        """Exporte la configuration de sécurité complète."""
        config = {}
        
        # Configuration des mots de passe
        password_policy = self.get_policy("password_policy")
        if password_policy:
            config.update({
                "PASSWORD_MIN_LENGTH": str(password_policy.min_length),
                "PASSWORD_MAX_LENGTH": str(password_policy.max_length),
                "PASSWORD_REQUIRE_UPPERCASE": str(password_policy.require_uppercase).lower(),
                "PASSWORD_REQUIRE_LOWERCASE": str(password_policy.require_lowercase).lower(),
                "PASSWORD_REQUIRE_DIGITS": str(password_policy.require_digits).lower(),
                "PASSWORD_REQUIRE_SPECIAL": str(password_policy.require_special_chars).lower(),
                "PASSWORD_SPECIAL_CHARS": password_policy.special_chars,
                "PASSWORD_EXPIRE_DAYS": str(password_policy.expire_days),
                "PASSWORD_HISTORY_CHECK": str(password_policy.history_check)
            })
        
        # Configuration des sessions
        session_policy = self.get_policy("session_policy")
        if session_policy:
            config.update({
                "SESSION_MAX_DURATION_HOURS": str(int(session_policy.max_session_duration.total_seconds() // 3600)),
                "SESSION_IDLE_TIMEOUT_MINUTES": str(int(session_policy.idle_timeout.total_seconds() // 60)),
                "SESSION_MAX_CONCURRENT": str(session_policy.max_concurrent_sessions),
                "SESSION_REQUIRE_REAUTH": str(session_policy.require_re_auth_for_sensitive).lower(),
                "SESSION_SECURE_COOKIES": str(session_policy.secure_cookies).lower(),
                "SESSION_HTTPONLY_COOKIES": str(session_policy.httponly_cookies).lower(),
                "SESSION_SAMESITE": session_policy.samesite_cookies
            })
        
        # Configuration du rate limiting
        rate_limit_policy = self.get_policy("rate_limit_policy")
        if rate_limit_policy:
            config.update({
                "RATE_LIMIT_PER_MINUTE": str(rate_limit_policy.requests_per_minute),
                "RATE_LIMIT_PER_HOUR": str(rate_limit_policy.requests_per_hour),
                "RATE_LIMIT_PER_DAY": str(rate_limit_policy.requests_per_day),
                "RATE_LIMIT_BURST": str(rate_limit_policy.burst_limit),
                "RATE_LIMIT_WINDOW": str(rate_limit_policy.window_size)
            })
        
        # Configuration du chiffrement
        encryption_policy = self.get_policy("encryption_policy")
        if encryption_policy:
            config.update({
                "ENCRYPTION_ALGORITHM": encryption_policy.algorithm,
                "ENCRYPTION_KEY_ROTATION_DAYS": str(encryption_policy.key_rotation_days),
                "ENCRYPTION_USE_HSM": str(encryption_policy.use_hsm).lower(),
                "ENCRYPTION_AT_REST": str(encryption_policy.encryption_at_rest).lower(),
                "ENCRYPTION_IN_TRANSIT": str(encryption_policy.encryption_in_transit).lower(),
                "ENCRYPTION_KDF": encryption_policy.key_derivation_function,
                "ENCRYPTION_ITERATIONS": str(encryption_policy.iterations)
            })
        
        # Configuration d'audit
        audit_policy = self.get_policy("audit_policy")
        if audit_policy:
            config.update({
                "AUDIT_ENABLED": str(audit_policy.enable_audit_logging).lower(),
                "AUDIT_LOG_API_CALLS": str(audit_policy.log_all_api_calls).lower(),
                "AUDIT_LOG_AUTH": str(audit_policy.log_authentication_events).lower(),
                "AUDIT_LOG_AUTHZ": str(audit_policy.log_authorization_events).lower(),
                "AUDIT_LOG_DATA_ACCESS": str(audit_policy.log_data_access).lower(),
                "AUDIT_LOG_DATA_MODIFY": str(audit_policy.log_data_modification).lower(),
                "AUDIT_LOG_ADMIN": str(audit_policy.log_admin_actions).lower(),
                "AUDIT_RETENTION_DAYS": str(audit_policy.retention_days),
                "AUDIT_REAL_TIME_ALERTS": str(audit_policy.real_time_alerts).lower()
            })
        
        return config

# Exportation des classes
__all__ = [
    'SecurityLevel',
    'ComplianceStandard',
    'AccessLevel',
    'PasswordPolicy',
    'SessionPolicy', 
    'RateLimitPolicy',
    'EncryptionPolicy',
    'AuditPolicy',
    'SecurityPolicyManager'
]
