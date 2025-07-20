# 🔐 Authentication Module - Enterprise Security
# ===============================================
# 
# Module d'authentification enterprise avec OAuth2,
# MFA, biométrie et analyse de risque.
#
# 🎖️ Expert: Lead Dev + Architecte IA + Spécialiste Sécurité Backend
#
# Développé par l'équipe d'experts enterprise
# ===============================================

"""
🔐 Enterprise Authentication Module
===================================

Advanced authentication system providing:
- Multi-factor authentication (TOTP, SMS, Email, Push, Biometric)
- OAuth2 and OpenID Connect with PKCE
- Risk-based authentication with ML analysis
- Session management and device trust
- Password policies and passwordless authentication
- Social authentication providers
- WebAuthn/FIDO2 support
- Token management and rotation
"""

from .authenticator import (
    AuthenticationManager,
    MultiFactorAuthenticator,
    BiometricAuthenticator,
    RiskBasedAuthenticator,
    AuthenticationResult,
    MFAMethod,
    BiometricType,
    RiskLevel,
    AuthenticationContext
)

from .oauth2_provider import (
    OAuth2Provider,
    OpenIDConnectProvider,
    SSOManager,
    OAuth2Config,
    ClientConfig,
    AuthorizationRequest,
    TokenResponse,
    UserInfo
)

from .session_manager import (
    SecureSessionManager,
    DeviceManager,
    SessionInfo,
    DeviceInfo,
    SessionStatus,
    DeviceStatus,
    SessionRiskLevel,
    SessionStore
)

from .password_manager import (
    AdvancedPasswordManager,
    PasswordlessAuthManager,
    SocialAuthManager,
    PasswordPolicy,
    PasswordAnalysis,
    PasswordStrength,
    MagicLink,
    WebAuthnCredential,
    AuthenticationMethod,
    APIKeyScope
)

from .token_manager import (
    AdvancedTokenManager,
    APIKeyManager,
    TokenAnalytics,
    TokenCleanupService,
    TokenInfo,
    APIKey,
    TokenType,
    TokenStatus,
    JWTConfig,
    TokenMetadata
)

# Imports manquants pour les types
from typing import Dict, Any, Optional

__all__ = [
    # Authenticator
    "AuthenticationManager",
    "MultiFactorAuthenticator", 
    "BiometricAuthenticator",
    "RiskBasedAuthenticator",
    "AuthenticationResult",
    "MFAMethod",
    "BiometricType",
    "RiskLevel",
    "AuthenticationContext",
    
    # OAuth2 Provider
    "OAuth2Provider",
    "OpenIDConnectProvider",
    "SSOManager",
    "OAuth2Config",
    "ClientConfig",
    "AuthorizationRequest",
    "TokenResponse",
    "UserInfo",
    
    # Session Manager
    "SecureSessionManager",
    "DeviceManager",
    "SessionInfo",
    "DeviceInfo",
    "SessionStatus",
    "DeviceStatus",
    "SessionRiskLevel",
    "SessionStore",
    
    # Password Manager
    "AdvancedPasswordManager",
    "PasswordlessAuthManager",
    "SocialAuthManager",
    "PasswordPolicy",
    "PasswordAnalysis",
    "PasswordStrength",
    "MagicLink",
    "WebAuthnCredential",
    "AuthenticationMethod",
    "APIKeyScope",
    
    # Token Manager
    "AdvancedTokenManager",
    "APIKeyManager",
    "TokenAnalytics",
    "TokenCleanupService",
    "TokenInfo",
    "APIKey",
    "TokenType",
    "TokenStatus",
    "JWTConfig",
    "TokenMetadata"
]


# Configuration par défaut pour l'authentification enterprise
DEFAULT_AUTH_CONFIG = {
    # Politiques de mots de passe
    "password_policy": {
        "min_length": 12,
        "require_uppercase": True,
        "require_lowercase": True,
        "require_digits": True,
        "require_special_chars": True,
        "max_age_days": 90,
        "history_count": 24
    },
    
    # Configuration MFA
    "mfa": {
        "enabled": True,
        "required_for_admin": True,
        "backup_codes_count": 10,
        "totp_issuer": "Spotify AI Agent"
    },
    
    # Configuration OAuth2
    "oauth2": {
        "access_token_expire_minutes": 15,
        "refresh_token_expire_days": 30,
        "authorization_code_expire_minutes": 10
    },
    
    # Configuration des sessions
    "sessions": {
        "default_timeout_minutes": 60,
        "max_sessions_per_user": 5,
        "require_https": True
    },
    
    # Configuration de sécurité
    "security": {
        "rate_limit_attempts": 5,
        "lockout_duration_minutes": 15,
        "enable_device_tracking": True,
        "enable_geolocation": True
    }
}


def get_auth_config() -> Dict[str, Any]:
    """Récupère la configuration d'authentification"""
    return DEFAULT_AUTH_CONFIG.copy()


def create_authentication_manager(
    redis_client: Any,
    config: Optional[Dict[str, Any]] = None
) -> AuthenticationManager:
    """Factory pour créer un gestionnaire d'authentification"""
    auth_config = config or get_auth_config()
    
    return AuthenticationManager(
        redis_client=redis_client,
        config=auth_config
    )


def create_oauth2_provider(
    redis_client: Any,
    config: Optional[OAuth2Config] = None
) -> OAuth2Provider:
    """Factory pour créer un provider OAuth2"""
    if not config:
        config = OAuth2Config(
            authorization_endpoint="/oauth2/authorize",
            token_endpoint="/oauth2/token",
            userinfo_endpoint="/oauth2/userinfo",
            jwks_endpoint="/oauth2/jwks",
            issuer="https://localhost"
        )
    
    return OAuth2Provider(
        redis_client=redis_client,
        config=config
    )


def create_session_manager(
    redis_client: Any
) -> SecureSessionManager:
    """Factory pour créer un gestionnaire de sessions"""
    return SecureSessionManager(redis_client)


def create_password_manager(
    redis_client: Any,
    policy: Optional[PasswordPolicy] = None
) -> AdvancedPasswordManager:
    """Factory pour créer un gestionnaire de mots de passe"""
    if not policy:
        policy = PasswordPolicy.get_enterprise_policy()
    
    return AdvancedPasswordManager(
        redis_client=redis_client,
        password_policy=policy
    )


def create_token_manager(
    redis_client: Any,
    jwt_config: Optional[JWTConfig] = None
) -> AdvancedTokenManager:
    """Factory pour créer un gestionnaire de tokens"""
    if not jwt_config:
        jwt_config = JWTConfig(
            algorithm="RS256",
            issuer="spotify-ai-agent",
            audience="spotify-ai-agent-api"
        )
    
    return AdvancedTokenManager(
        redis_client=redis_client,
        jwt_config=jwt_config
    )

from .token_manager import (
    JWTTokenManager,
    RefreshTokenManager,
    APIKeyManager
)

__all__ = [
    'AuthenticationManager',
    'MultiFactorAuthenticator',
    'BiometricAuthenticator',
    'RiskBasedAuthenticator',
    'OAuth2Provider',
    'OpenIDConnectProvider',
    'SSOManager',
    'SecureSessionManager',
    'SessionStore',
    'DeviceManager',
    'PasswordManager',
    'PasswordPolicyValidator',
    'PasswordlessAuthenticator',
    'JWTTokenManager',
    'RefreshTokenManager',
    'APIKeyManager'
]
