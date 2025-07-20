from unittest.mock import Mock

"""
Initialisation du sous-module de ressources audio/fixtures et scripts métiers pour tests avancés.

Ce module permet l’import et l’automatisation de tous les scripts d’audit, validation, conversion, génération de métadonnées, et la gestion des fixtures audio pour le microservice Spleeter.
Usage : import * pour automatiser la conformité, l’audit, ou l’intégration CI/CD.
"""

# Import direct des scripts métiers pour usage programmatique et CI/CD
from backend.services.spleeter_microservice.tests.generate_metadata import main as generate_metadata
from backend.services.spleeter_microservice.tests.validate_fixtures import main as validate_fixtures
from backend.services.spleeter_microservice.tests.audit_files import main as audit_files
from backend.services.spleeter_microservice.tests.convert_to_wav import main as convert_to_wav
