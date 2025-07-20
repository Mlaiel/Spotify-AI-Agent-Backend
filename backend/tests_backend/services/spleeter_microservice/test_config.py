from unittest.mock import Mock
"""
Tests avancés pour config.py (gestion des variables d'environnement, cohérence des settings).
"""


import sys
import os
import importlib
# Ajout du dossier racine backend au PYTHONPATH pour import absolu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../..', 'backend')))
from backend.services.spleeter_microservice import config

def test_default_settings():
    assert hasattr(config.settings, 'API_KEY')
    assert hasattr(config.settings, 'ENV')
    assert hasattr(config.settings, 'MAX_FILE_SIZE_MB')
    assert 'wav' in config.settings.ALLOWED_EXTENSIONS

def test_env_override(monkeypatch):
    monkeypatch.setenv('SPLEETER_API_KEY', 'testkey')
    import importlib
    import config as config_reload
    importlib.reload(config_reload)
    assert config.settings.API_KEY == 'testkey'
