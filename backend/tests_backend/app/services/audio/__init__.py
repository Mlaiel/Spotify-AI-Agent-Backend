from unittest.mock import Mock

"""
Initialisation du module de tests audio avancés.

Ce module centralise tous les tests unitaires et d’intégration pour le backend audio Spotify AI Agent.
Il expose des helpers pour automatisation, fixtures, et intégration CI/CD.
"""

# Import direct des modules de test pour usage programmatique
try:
    from .test_audio_utils import *
except ImportError:
    pass  # Skip if dependencies missing

try:
    from .test_audio_analyzer import *
except ImportError:
    pass  # Skip if dependencies missing

try:
    from .test_spleeter_client import *
except ImportError:
    pass  # Skip if dependencies missing
