from unittest.mock import Mock
import os
import yaml
import pytest
from pathlib import Path

COMPOSE_PATH = Path(__file__).parent.parent.parent / 'docker' / 'docker-compose.prod.yml'

def test_docker_compose_prod_exists():
    assert COMPOSE_PATH.exists(), f"Le fichier docker-compose.prod.yml n'existe pas: {COMPOSE_PATH}"

def test_docker_compose_prod_is_yaml():
    with open(COMPOSE_PATH, 'r') as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            pytest.fail(f"Le fichier n'est pas un YAML valide: {e}")
    assert isinstance(data, dict), "Le fichier doit être un dictionnaire YAML."

def test_docker_compose_prod_services():
    with open(COMPOSE_PATH, 'r') as f:
        data = yaml.safe_load(f)
    for service in ['backend', 'celery-worker', 'celery-beat', 'postgres', 'redis', 'nginx']:
        assert service in data.get('services', {}), f"Service manquant dans le compose: {service}"

def test_docker_compose_prod_networks():
    with open(COMPOSE_PATH, 'r') as f:
        data = yaml.safe_load(f)
    assert 'networks' in data, "Section networks manquante dans le compose."
import os
import yaml
import pytest
from pathlib import Path

COMPOSE_PATH = Path(__file__).parent.parent.parent / 'docker' / 'docker-compose.prod.yml'

def test_compose_prod_exists():
    assert COMPOSE_PATH.exists(), f"Le fichier docker-compose.prod.yml n'existe pas: {COMPOSE_PATH}"

def test_compose_prod_yaml_valid():
    with open(COMPOSE_PATH, 'r') as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), "Le fichier doit être un YAML valide."
    assert 'services' in data, "Section 'services' manquante."

def test_compose_prod_critical_services():
    with open(COMPOSE_PATH, 'r') as f:
        data = yaml.safe_load(f)
    for service in ['backend', 'celery-worker', 'celery-beat', 'postgres', 'redis', 'nginx']:
        assert service in data['services'], f"Service critique manquant: {service}"

def test_compose_prod_networks_and_volumes():
    with open(COMPOSE_PATH, 'r') as f:
        data = yaml.safe_load(f)
    assert 'networks' in data, "Section 'networks' manquante."
    assert 'volumes' in data or any('volumes' in s for s in data['services'].values()), "Volumes manquants."
# Renommé depuis test_docker_compose.prod.py

