from .base_exceptions import *
from .api_exceptions import *
from .auth_exceptions import *
from .database_exceptions import *
from .ai_exceptions import *
from .spotify_exceptions import *

# Exceptions pour le middleware de logging avancé
class LoggingError(Exception):
    """Exception pour les erreurs de logging"""
    pass

class MonitoringError(Exception):
    """Exception pour les erreurs de monitoring"""
    pass

class SecurityViolationError(Exception):
    """Exception pour les violations de sécurité"""
    pass

class ValidationError(Exception):
    """Exception pour les erreurs de validation"""
    pass

__all__ = [
    *base_exceptions.__all__,
    *api_exceptions.__all__,
    *auth_exceptions.__all__,
    *database_exceptions.__all__,
    *ai_exceptions.__all__,
    *spotify_exceptions.__all__,
    "LoggingError",
    "MonitoringError", 
    "SecurityViolationError",
    "ValidationError"
]
