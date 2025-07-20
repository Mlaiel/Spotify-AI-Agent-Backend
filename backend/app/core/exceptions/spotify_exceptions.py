"""
Exceptions Spotify (API, quotas, droits, intégration, business)
"""
from .base_exceptions import CoreBaseException

class SpotifyAPIException(CoreBaseException):
    def __init__(self, message="Erreur API Spotify", code=502, details=None, locale="fr"):
        super().__init__(message, code, details, locale)

class SpotifyQuotaException(SpotifyAPIException):
    def __init__(self, message="Quota Spotify dépassé", code=429, details=None, locale="fr"):
        super().__init__(message, code, details, locale)

class SpotifyPermissionException(SpotifyAPIException):
    def __init__(self, message="Permission Spotify refusée", code=403, details=None, locale="fr"):
        super().__init__(message, code, details, locale)

class SpotifyIntegrationException(SpotifyAPIException):
    def __init__(self, message="Erreur d'intégration Spotify", code=500, details=None, locale="fr"):
        super().__init__(message, code, details, locale)

class SpotifyBusinessException(SpotifyAPIException):
    def __init__(self, message="Erreur métier Spotify", code=400, details=None, locale="fr"):
        super().__init__(message, code, details, locale)

__all__ = [
    "SpotifyAPIException",
    "SpotifyQuotaException",
    "SpotifyPermissionException",
    "SpotifyIntegrationException",
    "SpotifyBusinessException"
]
