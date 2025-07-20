from unittest.mock import Mock
import os
import yaml
import pytest
from pathlib import Path

TEMPLATE_PATH = Path(__file__).parent.parent.parent.parent / 'config' / 'security' / 'secrets.template.yaml'

def test_secrets_template_exists():
    assert TEMPLATE_PATH.exists(), f"Le template de secrets n'existe pas: {TEMPLATE_PATH}"

def test_secrets_template_is_yaml():
    with open(TEMPLATE_PATH, 'r') as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            pytest.fail(f"Le fichier n'est pas un YAML valide: {e}")
    assert isinstance(data, dict), "Le template doit être un dictionnaire YAML."

def test_secrets_template_fields():
    with open(TEMPLATE_PATH, 'r') as f:
        data = yaml.safe_load(f)
    assert 'stringData' in data, "Champ stringData manquant dans le template."
    required = [
        'SECRET_KEY', 'POSTGRES_PASSWORD', 'REDIS_PASSWORD', 'SENTRY_DSN',
        'SPOTIFY_CLIENT_SECRET', 'EMAIL_HOST_PASSWORD'
    ]
    for field in required:
        assert field in data['stringData'], f"Secret obligatoire manquant: {field}"

def test_secrets_template_no_real_secrets():
    with open(TEMPLATE_PATH, 'r') as f:
        data = yaml.safe_load(f)
    for k, v in data.get('stringData', {}).items():
        assert v == '<REPLACE_ME>', f"Le champ {k} ne doit pas contenir de secret réel."
import os
import yaml
import pytest
from pathlib import Path

TEMPLATE_PATH = Path(__file__).parent.parent.parent.parent / 'config' / 'security' / 'secrets.template.yaml'

def test_secrets_template_exists():
    assert TEMPLATE_PATH.exists(), f"Le template de secrets n'existe pas: {TEMPLATE_PATH}"

def test_secrets_template_yaml_valid():
    with open(TEMPLATE_PATH, 'r') as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), "Le template doit être un dictionnaire YAML."
    assert data.get('apiVersion') == 'v1', "apiVersion doit être 'v1'"
    assert data.get('kind') == 'Secret', "kind doit être 'Secret'"
    assert 'stringData' in data, "Champ stringData manquant"

def test_secrets_template_required_fields():
    with open(TEMPLATE_PATH, 'r') as f:
        data = yaml.safe_load(f)
    required = [
        'SECRET_KEY', 'POSTGRES_PASSWORD', 'REDIS_PASSWORD', 'SENTRY_DSN',
        'SPOTIFY_CLIENT_SECRET', 'EMAIL_HOST_PASSWORD'
    ]
    for field in required:
        assert field in data['stringData'], f"Champ secret obligatoire manquant: {field}"
# Renommé depuis test_secrets.template.py

