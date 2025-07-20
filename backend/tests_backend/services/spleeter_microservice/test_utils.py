from unittest.mock import Mock
"""
Tests avancés pour utils.py (validation, gestion fichiers temporaires, sécurité).
"""


import sys
import os
import tempfile
import pytest
# Ajout du dossier racine backend au PYTHONPATH pour import absolu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../..', 'backend')))
from backend.services.spleeter_microservice import utils
from fastapi import UploadFile

class DummyFile:
    def __init__(self, name, content):
        self.filename = name
        self.file = tempfile.SpooledTemporaryFile()
        self.file.write(content)
        self.file.seek(0)

def test_validate_audio_file_valid():
    f = DummyFile('test.wav', b'\x00' * 100)
    assert utils.validate_audio_file(f) is True

def test_validate_audio_file_invalid_ext():
    f = DummyFile('test.txt', b'\x00' * 100)
    with pytest.raises(Exception):
        utils.validate_audio_file(f)

def test_save_and_cleanup_temp_file():
    f = DummyFile('test.wav', b'\x00' * 100)
    path = utils.save_temp_file(f)
    assert os.path.exists(path)
    utils.cleanup_temp_file(path)
    assert not os.path.exists(path)
