from unittest.mock import Mock
import os
import pytest
import importlib.util

SEED_DATA_PY = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../scripts/database/seed_data.py'))

def test_seed_data_script_exists():
    assert os.path.isfile(SEED_DATA_PY), f"Fichier introuvable: {SEED_DATA_PY}"

def test_seed_data_script_importable():
    spec = importlib.util.spec_from_file_location("seed_data", SEED_DATA_PY)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        pytest.fail(f"Erreur d'import/seed_data.py: {e}")

def test_seed_data_no_secrets_in_code():
    with open(SEED_DATA_PY, 'r') as f:
        content = f.read()
    forbidden = ['password', 'secret', 'AWS_ACCESS_KEY', 'AWS_SECRET', 'token']
    for word in forbidden:
        assert word not in content, f"Secret détecté dans le code: {word}"
