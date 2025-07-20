from unittest.mock import Mock
# Initialisation des tests avancés de configuration
"""
Ce module fournit des fixtures et utilitaires avancés pour les tests industriels de configuration.
"""

import os
import stat
import pytest

@pytest.fixture
def config_dir():
    """Retourne le chemin absolu du dossier de configuration à tester."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../config'))

def check_file_permissions(path, expected_mode=0o600):
    """Vérifie que le fichier a les droits attendus (par défaut : 600)."""
    st = os.stat(path)
    return stat.S_IMODE(st.st_mode) == expected_mode
