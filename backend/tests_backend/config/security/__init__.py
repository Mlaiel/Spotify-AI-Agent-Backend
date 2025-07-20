from unittest.mock import Mock
# Initialisierungsmodul für Sicherheitstests
"""
Dieses Modul stellt fortgeschrittene Fixtures und Hilfsfunktionen für die Sicherheits- und Geheimnis-Konfigurationstests bereit.
"""

import os
import stat
import yaml
import pytest

@pytest.fixture
def secrets_file_path():
    """Pfad zur secrets.encrypted Datei für Tests."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../config/security/secrets.encrypted'))

@pytest.fixture
def secrets_template_path():
    """Pfad zur secrets.template.yaml Datei für Tests."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../config/security/secrets.template.yaml'))

def check_file_permissions(path, expected_mode=0o600):
    """Überprüft, ob die Datei die erwarteten Berechtigungen hat (Standard: 600)."""
    st = os.stat(path)
    return stat.S_IMODE(st.st_mode) == expected_mode

def load_yaml(path):
    """Lädt und gibt den Inhalt einer YAML-Datei zurück."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)
