"""
Exceptions base de données (SQL, NoSQL, transaction, intégrité, timeouts, audit)
"""
from .base_exceptions import CoreBaseException

class DatabaseException(CoreBaseException):
    def __init__(self, message="Erreur base de données", code=500, details=None, locale="fr"):
        super().__init__(message, code, details, locale)

class TransactionException(DatabaseException):
    def __init__(self, message="Erreur transactionnelle", code=500, details=None, locale="fr"):
        super().__init__(message, code, details, locale)

class IntegrityException(DatabaseException):
    def __init__(self, message="Violation d'intégrité", code=409, details=None, locale="fr"):
        super().__init__(message, code, details, locale)

class TimeoutException(DatabaseException):
    def __init__(self, message="Timeout base de données", code=504, details=None, locale="fr"):
        super().__init__(message, code, details, locale)

class NotFoundDBException(DatabaseException):
    def __init__(self, message="Entrée non trouvée en base", code=404, details=None, locale="fr"):
        super().__init__(message, code, details, locale)

__all__ = [
    "DatabaseException",
    "TransactionException",
    "IntegrityException",
    "TimeoutException",
    "NotFoundDBException"
]
