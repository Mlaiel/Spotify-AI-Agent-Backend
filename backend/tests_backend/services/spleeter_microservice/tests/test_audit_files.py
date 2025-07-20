from unittest.mock import Mock
"""
Tests avancés pour audit_files.py (audit ajout/suppression de fichiers).
"""


import sys
import os
import json
import tempfile
import pytest
# Ajout du dossier racine backend au PYTHONPATH pour import absolu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../..', 'backend')))
from backend.services.spleeter_microservice.tests.audit_files import main as audit_files


def test_audit_files(tmp_path):
    f1 = tmp_path / "a.wav"
    f2 = tmp_path / "b.wav"
    f1.write_bytes(b"\x00" * 10)
    f2.write_bytes(b"\x00" * 10)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        audit_files.main()
        assert os.path.exists("audit_log.json")
        with open("audit_log.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "a.wav" in data["files"] and "b.wav" in data["files"]
    finally:
        os.chdir(cwd)
