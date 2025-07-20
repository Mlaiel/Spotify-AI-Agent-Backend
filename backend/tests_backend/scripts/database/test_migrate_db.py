from unittest.mock import Mock
import os
import pytest
import importlib.util

MIGRATE_DB_PY = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../scripts/database/migrate_db.py'))

def test_migrate_db_script_exists():
    assert os.path.isfile(MIGRATE_DB_PY), f"Fichier introuvable: {MIGRATE_DB_PY}"

def test_migrate_db_script_importable():
    spec = importlib.util.spec_from_file_location("migrate_db", MIGRATE_DB_PY)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        pytest.fail(f"Erreur d'import/migrate_db.py: {e}")

def test_migrate_db_no_secrets_in_code():
    with open(MIGRATE_DB_PY, 'r') as f:
        content = f.read()
    forbidden = ['password', 'secret', 'AWS_ACCESS_KEY', 'AWS_SECRET', 'token']
    for word in forbidden:
        assert word not in content, f"Secret détecté dans le code: {word}"
