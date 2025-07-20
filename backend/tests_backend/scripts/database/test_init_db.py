from unittest.mock import Mock
import os
import pytest
import importlib.util

INIT_DB_PY = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../scripts/database/init_db.py'))

def test_init_db_script_exists():
    assert os.path.isfile(INIT_DB_PY), f"Fichier introuvable: {INIT_DB_PY}"

def test_init_db_script_importable():
    spec = importlib.util.spec_from_file_location("init_db", INIT_DB_PY)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        pytest.fail(f"Erreur d'import/init_db.py: {e}")

def test_init_db_no_secrets_in_code():
    with open(INIT_DB_PY, 'r') as f:
        content = f.read()
    forbidden = ['password', 'secret', 'AWS_ACCESS_KEY', 'AWS_SECRET', 'token']
    for word in forbidden:
        assert word not in content, f"Secret détecté dans le code: {word}"
