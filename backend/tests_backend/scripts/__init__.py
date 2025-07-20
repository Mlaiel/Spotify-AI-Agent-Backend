from unittest.mock import Mock

"""
Initialisation du package de tests industriels avancés pour tous les scripts backend (déploiement, développement, maintenance, base de données).

Ce module prépare l’environnement de test global, charge les fixtures partagées et permet l’extension des tests automatisés pour tous les domaines métiers.
"""

import pytest
import logging

@pytest.fixture(scope="session", autouse=True)
def setup_global_logging():
    """Initialise un logger global pour tous les tests de scripts backend."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Initialisation du logger global pour tous les tests de scripts backend.")

@pytest.fixture(scope="session")
def global_env_vars(monkeypatch):
    """Charge des variables d'environnement globales pour tous les tests de scripts."""
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("TEST_MODE", "industrial")
    yield
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("TEST_MODE", raising=False)
