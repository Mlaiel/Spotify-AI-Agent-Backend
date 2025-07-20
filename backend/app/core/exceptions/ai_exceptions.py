"""
Exceptions IA (modèles, prompts, pipeline, quota, explainability, monitoring)
"""
from .base_exceptions import CoreBaseException

class AIException(CoreBaseException):
    def __init__(self, message="Erreur IA", code=500, details=None, locale="fr"):
        super().__init__(message, code, details, locale)

class ModelLoadException(AIException):
    def __init__(self, message="Chargement modèle IA échoué", code=500, details=None, locale="fr"):
        super().__init__(message, code, details, locale)

class PromptException(AIException):
    def __init__(self, message="Erreur de prompt IA", code=422, details=None, locale="fr"):
        super().__init__(message, code, details, locale)

class PipelineException(AIException):
    def __init__(self, message="Erreur pipeline IA", code=500, details=None, locale="fr"):
        super().__init__(message, code, details, locale)

class AIQuotaException(AIException):
    def __init__(self, message="Quota IA dépassé", code=429, details=None, locale="fr"):
        super().__init__(message, code, details, locale)

class ExplainabilityException(AIException):
    def __init__(self, message="Erreur d'explicabilité IA", code=500, details=None, locale="fr"):
        super().__init__(message, code, details, locale)

__all__ = [
    "AIException",
    "ModelLoadException",
    "PromptException",
    "PipelineException",
    "AIQuotaException",
    "ExplainabilityException"
]
