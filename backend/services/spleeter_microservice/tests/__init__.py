"""
Initialisation du module tests :
- Permet l'import Python du dossier pour automatisation, audit, ou intégration CI/CD.
- N'expose aucune logique de test, seulement des helpers et scripts d'audit/validation/metadata.
"""

# Import direct des scripts pour usage programmatique
from .generate_metadata import main as generate_metadata
from .validate_fixtures import main as validate_fixtures
from .audit_files import main as audit_files
from .convert_to_wav import main as convert_to_wav
