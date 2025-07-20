from unittest.mock import Mock
# Initialisation des tests industriels avancés Docker
"""
Ce module fournit des fixtures et utilitaires avancés pour les tests industriels des configurations Docker.
"""

import os
import stat
import pytest

@pytest.fixture
def docker_configs_dir():
    """Retourne le chemin absolu du dossier de configs Docker à tester."""
    return os.path.abspath(os.path.dirname(__file__))

def check_file_permissions(path, expected_mode=0o644):
    """Vérifie que le fichier a les droits attendus (par défaut : 644)."""
    st = os.stat(path)
    return stat.S_IMODE(st.st_mode) == expected_mode
