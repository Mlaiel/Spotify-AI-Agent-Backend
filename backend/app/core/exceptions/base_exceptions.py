"""
BaseException hiérarchique pour tout le backend
- Logging, code d’erreur, i18n, audit, sécurité
- Prêt pour FastAPI/Django, microservices
"""
import logging
from typing import Optional

class CoreBaseException(Exception):
    def __init__(self, message: str, code: int = 500, details: Optional[dict] = None, locale: str = "fr"):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
        self.locale = locale
        self.log()

    def log(self):
        logging.error(f"[{self.code}] {self.message} | details={self.details} | locale={self.locale}")

    def to_dict(self):
        return {
            "error": self.message,
            "code": self.code,
            "details": self.details,
            "locale": self.locale
        }

class BusinessException(CoreBaseException):
    pass

class SecurityException(CoreBaseException):
    pass

class NotFoundException(CoreBaseException):
    def __init__(self, message="Ressource non trouvée", code=404, details=None, locale="fr"):
        super().__init__(message, code, details, locale)

class ValidationException(CoreBaseException):
    def __init__(self, message="Erreur de validation", code=422, details=None, locale="fr"):
        super().__init__(message, code, details, locale)

class I18NError(CoreBaseException):
    """Exception pour les erreurs d'internationalisation avancée."""
    pass

class ConfigurationError(CoreBaseException):
    """Exception pour les erreurs de configuration système/infra/paramétrage."""
    pass

class LoggingError(CoreBaseException):
    """Exception pour les erreurs de logging avancé."""
    pass

__all__ = [
    "CoreBaseException",
    "BusinessException",
    "SecurityException",
    "NotFoundException",
    "ValidationException",
    "I18NError",
    "ConfigurationError",
    "LoggingError"
]
