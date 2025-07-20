from unittest.mock import Mock
# Initialisation des tests industriels avancés de scripts de base de données
"""
Ce module fournit des fixtures et utilitaires avancés pour les tests industriels des scripts de base de données.
Database helpers and fixtures for Spotify AI Agent project.
"""

import os
import stat
import pytest

@pytest.fixture
def db_scripts_dir():
    """Retourne le chemin absolu du dossier de scripts DB à tester."""
    return os.path.abspath(os.path.dirname(__file__))

def check_file_permissions(path, expected_mode=0o755):
    """Vérifie que le fichier a les droits attendus (par défaut : 755 pour les scripts)."""
    st = os.stat(path)
    return stat.S_IMODE(st.st_mode) == expected_mode
