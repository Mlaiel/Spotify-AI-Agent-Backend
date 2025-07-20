"""
Exceptions API (HTTP, validation, throttling, payload, intégration FastAPI/Django)
"""
from .base_exceptions import CoreBaseException

class APIException(CoreBaseException):
    def __init__(self, message="Erreur API", code=500, details=None, locale="fr"):
        super().__init__(message, code, details, locale)

class BadRequestException(APIException):
    def __init__(self, message="Requête invalide", code=400, details=None, locale="fr"):
        super().__init__(message, code, details, locale)

class UnauthorizedException(APIException):
    def __init__(self, message="Non autorisé", code=401, details=None, locale="fr"):
        super().__init__(message, code, details, locale)

class ForbiddenException(APIException):
    def __init__(self, message="Accès interdit", code=403, details=None, locale="fr"):
        super().__init__(message, code, details, locale)

class NotFoundAPIException(APIException):
    def __init__(self, message="Ressource API non trouvée", code=404, details=None, locale="fr"):
        super().__init__(message, code, details, locale)

class RateLimitException(APIException):
    def __init__(self, message="Trop de requêtes", code=429, details=None, locale="fr"):
        super().__init__(message, code, details, locale)

class PayloadTooLargeException(APIException):
    def __init__(self, message="Payload trop volumineux", code=413, details=None, locale="fr"):
        super().__init__(message, code, details, locale)

class ServiceUnavailableError(APIException):
    def __init__(self, message="Service indisponible", code=503, details=None, locale="fr"):
        super().__init__(message, code, details, locale)

__all__ = [
    "APIException",
    "BadRequestException",
    "UnauthorizedException",
    "ForbiddenException",
    "NotFoundAPIException",
    "RateLimitException",
    "PayloadTooLargeException",
    "ServiceUnavailableError"
]
