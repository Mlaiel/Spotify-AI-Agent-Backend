from unittest.mock import Mock

"""
Tests avancés pour generate_metadata.py (extraction métadonnées, hash SHA256).
"""

import sys
import os
import json
import shutil
import pytest
# Ajout du dossier racine backend au PYTHONPATH pour import absolu
from backend.services.spleeter_microservice.tests.generate_metadata import main as generate_metadata

def test_generate_metadata(tmp_path):
    # Copie un fichier audio dummy dans tmp_path
    src = os.path.join(os.path.dirname(__file__), "../../../../../../services/spleeter_microservice/tests/test_audio.wav")
    dst = tmp_path / "test_audio.wav"
    shutil.copy(src, dst)
    # Lance le script sur le dossier temporaire
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        generate_metadata.main()
        assert os.path.exists("fixtures.json")
        with open("fixtures.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "test_audio.wav" in data
        assert "sha256" in data["test_audio.wav"]
    finally:
        os.chdir(cwd)
