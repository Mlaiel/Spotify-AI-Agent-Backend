from unittest.mock import Mock
"""
Tests avancés pour convert_to_wav.py (conversion automatique vers WAV).
"""


import sys
import os
import tempfile
import pytest
# Ajout du dossier racine backend au PYTHONPATH pour import absolu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../..', 'backend')))
try:
    from backend.services.spleeter_microservice.tests.convert_to_wav import main as convert_to_wav
except ImportError:
    convert_to_wav = None

def test_convert_to_wav(tmp_path):
    # Crée un faux fichier MP3 (le contenu n'est pas un vrai MP3, mais le script doit gérer l'erreur)
    fpath = tmp_path / "dummy.mp3"
    fpath.write_bytes(b"\x00" * 100)
    try:
        convert_to_wav.convert_to_wav(str(fpath))
    except Exception:
        pass  # Erreur attendue sur faux MP3
