from unittest.mock import Mock
"""
Tests avancés pour validate_fixtures.py (validation cohérence fixtures.json).
"""


import os
import sys
import os
import json
import tempfile
import shutil
import pytest
# Ajout du dossier racine backend au PYTHONPATH pour import absolu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../..', 'backend')))
try:
    from backend.services.spleeter_microservice.tests.validate_fixtures import main as validate_fixtures
except ImportError:
    validate_fixtures = None

def test_validate_fixtures(tmp_path):
    fpath = tmp_path / "test_audio.wav"
    with open(fpath, "wb") as f:
        f.write(b"\x00" * 100)
    fixtures = {"test_audio.wav": {"type": "wav", "sha256": "dummy", "license": "CC0"}}
    with open(tmp_path / "fixtures.json", "w", encoding="utf-8") as f:
        json.dump(fixtures, f)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        validate_fixtures.main()
    finally:
        os.chdir(cwd)
