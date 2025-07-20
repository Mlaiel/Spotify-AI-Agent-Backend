"""
🧪 Tests du Module Frameworks - Spotify AI Agent
==============================================

Suite de tests complète pour tous les frameworks enterprise:
- Framework Orchestrator (Core)
- Hybrid Backend (Django + FastAPI)
- ML/AI Frameworks
- Security Framework
- Monitoring Framework
- Microservices Framework

Tests développés par l'équipe d'experts avec couverture complète.
"""

import pytest
import asyncio
from typing import Dict, Any
import logging

# Configuration des tests
pytest_plugins = [
    "pytest_asyncio",
    "pytest_mock",
    "pytest_cov"
]

# Logger pour les tests
logger = logging.getLogger("tests.frameworks")

def pytest_configure(config):
    """Configuration globale des tests frameworks."""
    config.addinivalue_line(
        "markers", "core: Tests du framework orchestrator"
    )
    config.addinivalue_line(
        "markers", "hybrid: Tests du backend hybride Django/FastAPI"
    )
    config.addinivalue_line(
        "markers", "ml: Tests des frameworks ML/AI"
    )
    config.addinivalue_line(
        "markers", "security: Tests du framework de sécurité"
    )
    config.addinivalue_line(
        "markers", "monitoring: Tests du framework de monitoring"
    )
    config.addinivalue_line(
        "markers", "microservices: Tests du framework microservices"
    )
    config.addinivalue_line(
        "markers", "integration: Tests d'intégration cross-framework"
    )
    config.addinivalue_line(
        "markers", "performance: Tests de performance et charge"
    )

@pytest.fixture(scope="session")
def event_loop():
    """Event loop partagé pour tous les tests async."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def clean_frameworks():
    """Fixture pour nettoyer l'état des frameworks entre les tests."""
    # Setup avant test
    yield
    
    # Cleanup après test
    try:
        from backend.app.frameworks import framework_orchestrator
        if framework_orchestrator._instance:
            await framework_orchestrator.shutdown_all()
            framework_orchestrator._instance = None
    except Exception as e:
        logger.warning(f"Cleanup warning: {e}")

# Constantes pour les tests
TEST_CONFIG = {
    "test_database_url": "sqlite:///test_frameworks.db",
    "test_redis_url": "redis://localhost:6379/15",
    "test_jwt_secret": "test-secret-key-frameworks",
    "test_metrics_port": 9091,
    "test_service_port": 8001
}

__all__ = [
    "pytest_configure",
    "clean_frameworks", 
    "TEST_CONFIG",
    "logger"
]
