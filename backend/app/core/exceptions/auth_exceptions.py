"""
Exceptions Authentification & Sécurité (JWT, OAuth, MFA, permissions)
"""
from .base_exceptions import SecurityException

class AuthenticationError(SecurityException):
    def __init__(self, message="Erreur d'authentification", code=401, details=None, locale="fr"):
        super().__init__(message, code, details, locale)

class AuthorizationError(SecurityException):
    def __init__(self, message="Non autorisé", code=403, details=None, locale="fr"):
        super().__init__(message, code, details, locale)

class RateLimitExceededError(SecurityException):
    def __init__(self, message="Trop de requêtes", code=429, details=None, locale="fr"):
        super().__init__(message, code, details, locale)

class SecurityViolationError(SecurityException):
    def __init__(self, message="Violation de sécurité", code=403, details=None, locale="fr"):
        super().__init__(message, code, details, locale)

# Compatibilité avec les anciennes exceptions
class AuthException(AuthenticationError):
    pass
class InvalidTokenException(AuthenticationError):
    pass
class PermissionDeniedException(AuthorizationError):
    pass
class MFARequiredException(AuthenticationError):
    pass
class OAuthException(AuthenticationError):
    pass

__all__ = [
    "AuthenticationError",
    "AuthorizationError",
    "RateLimitExceededError",
    "SecurityViolationError",
    "AuthException",
    "InvalidTokenException",
    "PermissionDeniedException",
    "MFARequiredException",
    "OAuthException"
]
