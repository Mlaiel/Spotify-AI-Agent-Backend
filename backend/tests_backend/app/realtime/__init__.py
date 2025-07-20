# 🎵 Spotify AI Agent - Real-Time Module Tests
# ===============================================
# 
# Suite complète de tests pour le module realtime
# avec couverture de code, tests d'intégration et de performance.
#
# 🎖️ Expert: Test Engineer + QA Specialist + Performance Engineer
#
# 👨‍💻 Développé par: Fahed Mlaiel
# ===============================================

"""
🧪 Real-Time Module Test Suite
==============================

Comprehensive test suite for the real-time communication system:
- Unit tests for all components
- Integration tests for end-to-end workflows
- Performance tests for scalability
- Security tests for authentication and authorization
- Load tests for connection pooling
- Chaos engineering tests for fault tolerance
- WebSocket protocol compliance tests
- Real-time analytics validation
"""

import os
import sys
import asyncio
import pytest
from pathlib import Path

# Ajouter le backend au path pour les imports
backend_path = Path(__file__).parent.parent.parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))

# Configuration des tests
REDIS_TEST_URL = os.getenv("REDIS_TEST_URL", "redis://localhost:6379/1")
WEBSOCKET_TEST_PORT = int(os.getenv("WEBSOCKET_TEST_PORT", "8765"))
TEST_TIMEOUT = int(os.getenv("TEST_TIMEOUT", "30"))

# Fixtures partagées
@pytest.fixture(scope="session")
def event_loop():
    """Event loop pour les tests async"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def redis_client():
    """Client Redis pour les tests"""
    import aioredis
    client = await aioredis.from_url(REDIS_TEST_URL)
    yield client
    await client.flushdb()  # Nettoyer après chaque test
    await client.close()

# Configuration pytest
def pytest_configure(config):
    """Configuration pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )

# Utilitaires de test
class TestUtils:
    """Utilitaires pour les tests"""
    
    @staticmethod
    def generate_test_user_id() -> str:
        """Génère un ID utilisateur de test"""
        import uuid
        return f"test_user_{uuid.uuid4()}"
    
    @staticmethod
    def generate_test_track_id() -> str:
        """Génère un ID de track de test"""
        import uuid
        return f"test_track_{uuid.uuid4()}"
    
    @staticmethod
    async def wait_for_condition(condition, timeout=5, interval=0.1):
        """Attend qu'une condition soit vraie"""
        import asyncio
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < timeout:
            if await condition() if asyncio.iscoroutinefunction(condition) else condition():
                return True
            await asyncio.sleep(interval)
        return False

# Export des utilitaires
__all__ = [
    "TestUtils",
    "REDIS_TEST_URL",
    "WEBSOCKET_TEST_PORT",
    "TEST_TIMEOUT"
]
