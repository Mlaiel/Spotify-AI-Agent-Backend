from unittest.mock import Mock
"""
Initialisation du package de tests industriels avancés pour la maintenance backend.

Ce module prépare l’environnement de test, charge les fixtures globales et permet l’extension des tests automatisés pour le nettoyage, l’optimisation, le tuning et la gestion du cache.
"""

import pytest
import logging

@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    """Initialise un logger global pour tous les tests de maintenance."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Initialisation du logger global pour les tests de maintenance.")

@pytest.fixture(scope="session")
def maintenance_env_vars(monkeypatch):
    """Charge des variables d'environnement de maintenance pour tous les tests."""
    monkeypatch.setenv("ENV", "maintenance")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    yield
    monkeypatch.delenv("ENV", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
