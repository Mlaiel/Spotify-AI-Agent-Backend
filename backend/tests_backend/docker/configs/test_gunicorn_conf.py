from unittest.mock import Mock
import os
import importlib.util
import pytest
from pathlib import Path

CONF_PATH = Path(__file__).parent.parent.parent.parent / 'docker' / 'configs' / 'gunicorn_conf.py'

def test_gunicorn_conf_exists():
    assert CONF_PATH.exists(), f"Le fichier gunicorn_conf.py n'existe pas: {CONF_PATH}"

def test_gunicorn_conf_syntax():
    with open(CONF_PATH, 'r') as f:
        code = f.read()
    try:
        compile(code, str(CONF_PATH), 'exec')
    except SyntaxError as e:
        pytest.fail(f"Erreur de syntaxe dans gunicorn_conf.py: {e}")

def test_gunicorn_conf_advanced_settings():
    spec = importlib.util.spec_from_file_location("gunicorn_conf", CONF_PATH)
    gunicorn_conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gunicorn_conf)
    assert hasattr(gunicorn_conf, 'workers') and gunicorn_conf.workers >= 2
    assert hasattr(gunicorn_conf, 'timeout') and gunicorn_conf.timeout >= 60
    assert hasattr(gunicorn_conf, 'bind') and gunicorn_conf.bind.startswith('0.0.0.0:')
    assert hasattr(gunicorn_conf, 'worker_class') and 'uvicorn' in gunicorn_conf.worker_class
    assert hasattr(gunicorn_conf, 'secure_scheme_headers')
import os
import importlib.util
import pytest
from pathlib import Path

CONF_PATH = Path(__file__).parent.parent.parent.parent / 'docker' / 'configs' / 'gunicorn.conf.py'

def test_gunicorn_conf_exists():
    assert CONF_PATH.exists(), f"Le fichier gunicorn.conf.py n'existe pas: {CONF_PATH}"

def test_gunicorn_conf_syntax():
    with open(CONF_PATH, 'r') as f:
        code = f.read()
    compile(code, str(CONF_PATH), 'exec')

def test_gunicorn_conf_advanced_settings():
    spec = importlib.util.spec_from_file_location("gunicorn_conf", CONF_PATH)
    gunicorn_conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gunicorn_conf)
    assert hasattr(gunicorn_conf, 'workers')
    assert gunicorn_conf.workers >= 2
    assert gunicorn_conf.worker_class == "uvicorn.workers.UvicornWorker"
    assert gunicorn_conf.timeout >= 60
    assert gunicorn_conf.preload_app is True
    assert hasattr(gunicorn_conf, 'when_ready')
    assert hasattr(gunicorn_conf, 'on_exit')
# Renommé depuis test_gunicorn.conf.py

