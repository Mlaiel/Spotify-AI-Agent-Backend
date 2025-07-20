from unittest.mock import Mock

"""
Initialisation du module de tests avancés pour spleeter_microservice.

Ce module centralise tous les tests unitaires et d’intégration pour le microservice Spleeter du backend Spotify AI Agent.
Il expose des helpers pour automatisation, fixtures, et intégration CI/CD.
Usage : import * pour exécuter tous les tests ou automatiser la validation.
"""

# Import direct des modules de test pour usage programmatique et CI/CD
from .test_config import *
from .test_utils import *
from .test_security import *
from .test_monitoring import *
from .test_health import *
