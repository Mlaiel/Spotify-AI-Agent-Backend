from unittest.mock import Mock
"""
Initialisation du package de tests industriels avancés pour le développement backend.

Ce module prépare l’environnement de test, charge les fixtures globales et permet l’extension des tests automatisés pour la génération de données, la documentation API, la qualité du code et la sécurité.
"""

import pytest
import logging

@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    """Initialise un logger global pour tous les tests de développement."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Initialisation du logger global pour les tests de développement.")

@pytest.fixture(scope="session")
def dev_env_vars(monkeypatch):
    """Charge des variables d'environnement de développement pour tous les tests."""
    monkeypatch.setenv("ENV", "development")
    monkeypatch.setenv("DEBUG", "1")
    yield
    monkeypatch.delenv("ENV", raising=False)
    monkeypatch.delenv("DEBUG", raising=False)
