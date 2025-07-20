"""
Règles de sécurité et validation - Spotify AI Agent
Validation avancée pour la sécurité et la conformité
"""

import hashlib
import re
import secrets
import string
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union
from ipaddress import AddressValueError, IPv4Address, IPv6Address, ip_address

from cryptography.fernet import Fernet
from pydantic import SecretStr, validator
from passlib.context import CryptContext
from passlib.hash import pbkdf2_sha256

from ..base.enums import SecurityLevel, UserRole, SECURITY_PATTERNS


class SecurityValidationRules:
    """Règles de validation pour la sécurité"""
    
    # Configuration de hachage des mots de passe
    pwd_context = CryptContext(
        schemes=["pbkdf2_sha256", "bcrypt"],
        default="pbkdf2_sha256",
        pbkdf2_sha256__default_rounds=150000
    )
    
    @classmethod
    def validate_password_strength(cls, password: str) -> str:
        """Valide la force d'un mot de passe"""
        if not password:
            raise ValueError("Password cannot be empty")
        
        if len(password) < 12:
            raise ValueError("Password must be at least 12 characters long")
        
        if len(password) > 128:
            raise ValueError("Password cannot exceed 128 characters")
        
        # Vérification des critères de complexité
        checks = {
            'lowercase': bool(re.search(r'[a-z]', password)),
            'uppercase': bool(re.search(r'[A-Z]', password)),
            'digit': bool(re.search(r'\d', password)),
            'special': bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password)),
        }
        
        failed_checks = [check for check, passed in checks.items() if not passed]
        
        if len(failed_checks) > 1:
            raise ValueError(
                f"Password must contain at least 3 of the following: "
                f"lowercase letters, uppercase letters, digits, special characters. "
                f"Missing: {', '.join(failed_checks)}"
            )
        
        # Vérification des patterns interdits
        forbidden_patterns = [
            r'(.)\1{3,}',  # 4+ caractères répétés
            r'(012|123|234|345|456|567|678|789|890)',  # Séquences numériques
            r'(abc|bcd|cde|def|efg|fgh|ghi|hij|ijk|jkl|klm|lmn|mno|nop|opq|pqr|qrs|rst|stu|tuv|uvw|vwx|wxy|xyz)',  # Séquences alphabétiques
            r'(qwer|asdf|zxcv|uiop|hjkl|bnm)',  # Patterns clavier
        ]
        
        for pattern in forbidden_patterns:
            if re.search(pattern, password.lower()):
                raise ValueError("Password contains predictable patterns")
        
        # Vérification des mots de passe communs
        common_passwords = {
            'password', '123456', 'qwerty', 'admin', 'letmein',
            'welcome', 'monkey', 'dragon', 'pass', 'master'
        }
        
        if password.lower() in common_passwords:
            raise ValueError("Password is too common")
        
        return password
    
    @classmethod
    def validate_api_key(cls, api_key: str) -> str:
        """Valide un format de clé API"""
        if not api_key or not api_key.strip():
            raise ValueError("API key cannot be empty")
        
        api_key = api_key.strip()
        
        # Format attendu: prefix_base64encoded_checksum
        if not re.match(SECURITY_PATTERNS["API_KEY"], api_key):
            raise ValueError("Invalid API key format")
        
        # Vérification de la longueur
        if len(api_key) < 32:
            raise ValueError("API key too short (minimum 32 characters)")
        
        if len(api_key) > 255:
            raise ValueError("API key too long (maximum 255 characters)")
        
        return api_key
    
    @classmethod
    def validate_jwt_token(cls, token: str) -> str:
        """Valide le format basique d'un token JWT"""
        if not token or not token.strip():
            raise ValueError("JWT token cannot be empty")
        
        token = token.strip()
        
        # Format JWT: header.payload.signature
        parts = token.split('.')
        if len(parts) != 3:
            raise ValueError("Invalid JWT format (must have 3 parts)")
        
        # Vérification que chaque partie est en base64
        for i, part in enumerate(parts):
            if not part:
                raise ValueError(f"JWT part {i+1} cannot be empty")
            
            if not re.match(r'^[A-Za-z0-9_-]+$', part):
                raise ValueError(f"JWT part {i+1} contains invalid characters")
        
        return token
    
    @classmethod
    def validate_encryption_key(cls, key: str, key_type: str = "fernet") -> str:
        """Valide une clé de chiffrement"""
        if not key:
            raise ValueError("Encryption key cannot be empty")
        
        if key_type.lower() == "fernet":
            try:
                # Tentative de création d'un objet Fernet pour validation
                Fernet(key.encode())
            except Exception:
                raise ValueError("Invalid Fernet encryption key")
        
        elif key_type.lower() == "aes":
            # Clé AES en base64
            if not re.match(r'^[A-Za-z0-9+/]*={0,2}$', key):
                raise ValueError("Invalid AES key format (must be base64)")
            
            # Vérification de la longueur décodée
            try:
                import base64
                decoded = base64.b64decode(key)
                if len(decoded) not in [16, 24, 32]:  # 128, 192, 256 bits
                    raise ValueError("AES key must be 128, 192, or 256 bits")
            except Exception:
                raise ValueError("Invalid AES key encoding")
        
        else:
            raise ValueError(f"Unsupported encryption key type: {key_type}")
        
        return key
    
    @classmethod
    def validate_ip_address(cls, ip: str, allow_private: bool = False) -> str:
        """Valide une adresse IP"""
        if not ip or not ip.strip():
            raise ValueError("IP address cannot be empty")
        
        ip = ip.strip()
        
        try:
            ip_obj = ip_address(ip)
        except AddressValueError:
            raise ValueError(f"Invalid IP address format: {ip}")
        
        # Vérification des adresses privées si non autorisées
        if not allow_private:
            if ip_obj.is_private:
                raise ValueError("Private IP addresses are not allowed")
            
            if ip_obj.is_loopback:
                raise ValueError("Loopback IP addresses are not allowed")
            
            if ip_obj.is_link_local:
                raise ValueError("Link-local IP addresses are not allowed")
        
        # Vérification des adresses réservées
        if ip_obj.is_reserved:
            raise ValueError("Reserved IP addresses are not allowed")
        
        if ip_obj.is_multicast:
            raise ValueError("Multicast IP addresses are not allowed")
        
        return str(ip_obj)
    
    @classmethod
    def validate_security_headers(cls, headers: Dict[str, str]) -> Dict[str, str]:
        """Valide les en-têtes de sécurité"""
        if not headers:
            return headers
        
        validated_headers = {}
        
        # En-têtes de sécurité recommandés
        security_headers = {
            'x-content-type-options': ['nosniff'],
            'x-frame-options': ['DENY', 'SAMEORIGIN'],
            'x-xss-protection': ['1; mode=block', '0'],
            'strict-transport-security': [r'max-age=\d+.*'],
            'content-security-policy': [r'.+'],
            'referrer-policy': [
                'no-referrer', 'no-referrer-when-downgrade',
                'origin', 'origin-when-cross-origin',
                'same-origin', 'strict-origin',
                'strict-origin-when-cross-origin', 'unsafe-url'
            ]
        }
        
        for header, value in headers.items():
            header_lower = header.lower().strip()
            value_clean = value.strip()
            
            if header_lower in security_headers:
                valid_values = security_headers[header_lower]
                
                # Vérification regex pour certains headers
                if header_lower in ['strict-transport-security', 'content-security-policy']:
                    if not any(re.match(pattern, value_clean) for pattern in valid_values):
                        raise ValueError(f"Invalid value for header '{header}': {value}")
                else:
                    if value_clean not in valid_values:
                        raise ValueError(
                            f"Invalid value for header '{header}': {value}. "
                            f"Allowed values: {', '.join(valid_values)}"
                        )
            
            validated_headers[header] = value_clean
        
        return validated_headers
    
    @classmethod
    def validate_permission_scope(cls, scope: str, user_role: UserRole) -> str:
        """Valide les permissions selon le rôle utilisateur"""
        if not scope or not scope.strip():
            raise ValueError("Permission scope cannot be empty")
        
        scope = scope.strip().lower()
        
        # Définition des scopes par rôle
        role_permissions = {
            UserRole.SUPER_ADMIN: {
                'system:*', 'tenant:*', 'user:*', 'alert:*',
                'notification:*', 'ml:*', 'analytics:*'
            },
            UserRole.TENANT_ADMIN: {
                'tenant:read', 'tenant:write', 'user:read', 'user:write',
                'alert:read', 'alert:write', 'notification:read', 'notification:write'
            },
            UserRole.OPERATOR: {
                'alert:read', 'alert:write', 'notification:read',
                'tenant:read', 'analytics:read'
            },
            UserRole.VIEWER: {
                'alert:read', 'notification:read', 'tenant:read', 'analytics:read'
            },
            UserRole.API_USER: {
                'alert:read', 'notification:write', 'analytics:read'
            }
        }
        
        allowed_scopes = role_permissions.get(user_role, set())
        
        # Vérification des wildcards
        scope_allowed = False
        
        for allowed_scope in allowed_scopes:
            if allowed_scope.endswith('*'):
                # Wildcard: vérifier le préfixe
                prefix = allowed_scope[:-1]
                if scope.startswith(prefix):
                    scope_allowed = True
                    break
            elif scope == allowed_scope:
                scope_allowed = True
                break
        
        if not scope_allowed:
            raise ValueError(f"Permission scope '{scope}' not allowed for role '{user_role.value}'")
        
        return scope
    
    @classmethod
    def validate_rate_limit_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Valide une configuration de rate limiting"""
        if not config:
            raise ValueError("Rate limit configuration cannot be empty")
        
        validated_config = {}
        
        # Limite de requêtes
        requests = config.get('requests')
        if requests is not None:
            if not isinstance(requests, int) or requests <= 0:
                raise ValueError("Requests limit must be a positive integer")
            if requests > 100000:
                raise ValueError("Requests limit cannot exceed 100,000")
            validated_config['requests'] = requests
        
        # Période en secondes
        period = config.get('period')
        if period is not None:
            if not isinstance(period, int) or period <= 0:
                raise ValueError("Period must be a positive integer")
            if period > 86400:  # 24 heures max
                raise ValueError("Period cannot exceed 24 hours (86400 seconds)")
            validated_config['period'] = period
        
        # Burst allowance
        burst = config.get('burst')
        if burst is not None:
            if not isinstance(burst, int) or burst < 0:
                raise ValueError("Burst must be a non-negative integer")
            if burst > validated_config.get('requests', 1000) * 2:
                raise ValueError("Burst cannot exceed twice the requests limit")
            validated_config['burst'] = burst
        
        # Stratégie de limitation
        strategy = config.get('strategy')
        if strategy:
            valid_strategies = {'sliding_window', 'fixed_window', 'token_bucket', 'leaky_bucket'}
            if strategy not in valid_strategies:
                raise ValueError(f"Invalid rate limit strategy: {strategy}")
            validated_config['strategy'] = strategy
        
        return validated_config
    
    @classmethod
    def generate_secure_token(cls, length: int = 32) -> str:
        """Génère un token sécurisé"""
        if length < 16:
            raise ValueError("Token length must be at least 16 characters")
        if length > 256:
            raise ValueError("Token length cannot exceed 256 characters")
        
        # Alphabet sécurisé (évite les caractères ambigus)
        alphabet = string.ascii_letters + string.digits + "-_"
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    @classmethod
    def hash_password(cls, password: str) -> str:
        """Hache un mot de passe de manière sécurisée"""
        # Validation préalable
        cls.validate_password_strength(password)
        
        # Hachage avec salt automatique
        return cls.pwd_context.hash(password)
    
    @classmethod
    def verify_password(cls, password: str, hashed: str) -> bool:
        """Vérifie un mot de passe contre son hash"""
        try:
            return cls.pwd_context.verify(password, hashed)
        except Exception:
            return False


class ComplianceValidators:
    """Validateurs pour la conformité réglementaire"""
    
    @classmethod
    def validate_gdpr_compliance(cls, data_fields: Dict[str, Any]) -> Dict[str, Any]:
        """Valide la conformité GDPR des champs de données"""
        if not data_fields:
            return data_fields
        
        validated_fields = {}
        
        # Champs sensibles selon GDPR
        sensitive_fields = {
            'email', 'phone', 'address', 'location', 'ip_address',
            'device_id', 'user_agent', 'biometric_data', 'health_data'
        }
        
        # Champs nécessitant consentement explicite
        consent_required_fields = {
            'marketing_consent', 'analytics_consent', 'personalization_consent'
        }
        
        for field, value in data_fields.items():
            field_lower = field.lower()
            
            # Vérification des champs sensibles
            if any(sensitive in field_lower for sensitive in sensitive_fields):
                if not data_fields.get(f'{field}_consent', False):
                    raise ValueError(f"Field '{field}' requires explicit consent")
                
                # Pseudonymisation pour certains champs
                if field_lower in ['email', 'phone']:
                    if not data_fields.get(f'{field}_pseudonymized', False):
                        # Hash pour pseudonymisation
                        hashed_value = hashlib.sha256(str(value).encode()).hexdigest()[:16]
                        validated_fields[f'{field}_pseudonymized'] = hashed_value
            
            validated_fields[field] = value
        
        # Vérification des consentements requis
        for consent_field in consent_required_fields:
            if consent_field in data_fields:
                consent_value = data_fields[consent_field]
                if not isinstance(consent_value, bool):
                    raise ValueError(f"Consent field '{consent_field}' must be boolean")
                
                # Horodatage du consentement
                if consent_value:
                    timestamp_field = f'{consent_field}_timestamp'
                    if timestamp_field not in data_fields:
                        validated_fields[timestamp_field] = datetime.utcnow().isoformat()
        
        return validated_fields
    
    @classmethod
    def validate_data_retention(cls, retention_config: Dict[str, Any]) -> Dict[str, Any]:
        """Valide une configuration de rétention des données"""
        if not retention_config:
            raise ValueError("Data retention configuration cannot be empty")
        
        validated_config = {}
        
        # Période de rétention
        retention_days = retention_config.get('retention_days')
        if retention_days is not None:
            if not isinstance(retention_days, int) or retention_days < 0:
                raise ValueError("Retention days must be a non-negative integer")
            
            # Limites réglementaires
            if retention_days > 2555:  # ~7 ans maximum
                raise ValueError("Retention period cannot exceed 7 years")
            
            validated_config['retention_days'] = retention_days
        
        # Catégories de données
        data_categories = retention_config.get('data_categories')
        if data_categories:
            valid_categories = {
                'personal_data', 'anonymous_data', 'operational_data',
                'security_logs', 'audit_logs', 'performance_metrics'
            }
            
            if not isinstance(data_categories, list):
                raise ValueError("Data categories must be a list")
            
            for category in data_categories:
                if category not in valid_categories:
                    raise ValueError(f"Invalid data category: {category}")
            
            validated_config['data_categories'] = data_categories
        
        # Politique de suppression
        deletion_policy = retention_config.get('deletion_policy')
        if deletion_policy:
            valid_policies = {'hard_delete', 'soft_delete', 'anonymize', 'archive'}
            if deletion_policy not in valid_policies:
                raise ValueError(f"Invalid deletion policy: {deletion_policy}")
            validated_config['deletion_policy'] = deletion_policy
        
        return validated_config


# Décorateurs de validation sécurisée
def validate_password_field():
    """Décorateur pour valider un champ mot de passe"""
    return validator('password', allow_reuse=True)(
        SecurityValidationRules.validate_password_strength
    )

def validate_api_key_field():
    """Décorateur pour valider un champ clé API"""
    return validator('api_key', allow_reuse=True)(
        SecurityValidationRules.validate_api_key
    )

def validate_ip_address_field(allow_private: bool = False):
    """Décorateur pour valider un champ adresse IP"""
    return validator('ip_address', allow_reuse=True)(
        lambda v: SecurityValidationRules.validate_ip_address(v, allow_private)
    )
