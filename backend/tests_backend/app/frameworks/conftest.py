"""
🧪 Configuration et Utilitaires de Tests - Frameworks Test Suite
===============================================================

Configuration complète pour l'exécution des tests avec:
- Fixtures globales
- Mocks partagés
- Configuration pytest
- Scripts de test automatisés
- Reporting et couverture

Développé par: Toute l'équipe d'experts
"""

import pytest
import asyncio
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import shutil

# Configuration du logging pour les tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Configuration pytest
pytest_plugins = [
    "pytest_asyncio",
    "pytest_mock", 
    "pytest_cov",
    "pytest_benchmark"
]

def pytest_configure(config):
    """Configuration globale pytest pour les frameworks."""
    # Markers personnalisés
    markers = [
        "core: Tests du framework orchestrator",
        "hybrid: Tests backend hybride Django/FastAPI", 
        "ml: Tests frameworks ML/AI",
        "security: Tests framework sécurité",
        "monitoring: Tests framework monitoring",
        "microservices: Tests framework microservices",
        "integration: Tests d'intégration cross-framework",
        "performance: Tests de performance et charge",
        "slow: Tests lents (> 1 seconde)",
        "external: Tests nécessitant services externes"
    ]
    
    for marker in markers:
        config.addinivalue_line("markers", marker)
        
    # Configuration couverture
    config.option.cov_source = ["backend/app/frameworks"]
    config.option.cov_report = ["term-missing", "html"]
    config.option.cov_fail_under = 85  # Minimum 85% de couverture

def pytest_collection_modifyitems(config, items):
    """Modifier la collection de tests."""
    # Marquer tests lents automatiquement
    for item in items:
        if "test_concurrent" in item.name or "test_performance" in item.name:
            item.add_marker(pytest.mark.slow)
            
        if "integration" in item.name or "end_to_end" in item.name:
            item.add_marker(pytest.mark.integration)

@pytest.fixture(scope="session")
def event_loop():
    """Event loop global pour tous les tests async."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_database_url():
    """URL base de données de test."""
    return "sqlite:///test_frameworks.db"

@pytest.fixture(scope="session")  
def test_redis_url():
    """URL Redis de test."""
    return "redis://localhost:6379/15"

@pytest.fixture(scope="session")
def test_config():
    """Configuration globale des tests."""
    return {
        "test_database_url": "sqlite:///test_frameworks.db",
        "test_redis_url": "redis://localhost:6379/15", 
        "test_jwt_secret": "test-secret-key-for-frameworks-tests",
        "test_metrics_port": 9091,
        "test_service_port": 8001,
        "test_consul_host": "localhost",
        "test_consul_port": 8500,
        "test_rabbitmq_url": "amqp://guest:guest@localhost:5672/",
        "test_jaeger_host": "localhost",
        "test_jaeger_port": 6831
    }

@pytest.fixture
def temp_directory():
    """Répertoire temporaire pour les tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_external_services():
    """Mock de tous les services externes."""
    from unittest.mock import Mock, AsyncMock, patch
    
    mocks = {}
    
    # Redis mock
    redis_mock = AsyncMock()
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    redis_mock.incr.return_value = 1
    redis_mock.expire.return_value = True
    redis_mock.delete.return_value = True
    mocks['redis'] = redis_mock
    
    # Consul mock
    consul_mock = Mock()
    consul_mock.agent.service.register.return_value = True
    consul_mock.agent.service.deregister.return_value = True
    consul_mock.health.service.return_value = (None, [])
    mocks['consul'] = consul_mock
    
    # RabbitMQ mock
    rabbitmq_mock = AsyncMock()
    rabbitmq_mock.channel.return_value = AsyncMock()
    mocks['rabbitmq'] = rabbitmq_mock
    
    # HTTP client mock
    http_mock = AsyncMock()
    http_mock.get.return_value = Mock(status_code=200, json=lambda: {"status": "ok"})
    http_mock.post.return_value = Mock(status_code=201, json=lambda: {"id": "12345"})
    mocks['http'] = http_mock
    
    with patch('redis.asyncio.from_url', return_value=redis_mock), \
         patch('consul.Consul', return_value=consul_mock), \
         patch('aio_pika.connect_robust', return_value=rabbitmq_mock), \
         patch('httpx.AsyncClient', return_value=http_mock):
        yield mocks

@pytest.fixture
async def clean_frameworks():
    """Nettoyer l'état des frameworks entre tests."""
    # Setup avant test
    yield
    
    # Cleanup après test
    try:
        from backend.app.frameworks import framework_orchestrator
        if hasattr(framework_orchestrator, '_instance') and framework_orchestrator._instance:
            await framework_orchestrator.shutdown_all()
            framework_orchestrator._instance = None
    except Exception as e:
        logging.warning(f"Cleanup warning: {e}")

# Constantes pour les tests
TEST_USER_DATA = {
    "user_id": "test_user_12345",
    "username": "test_user",
    "email": "test@example.com",
    "preferences": ["rock", "electronic", "indie"]
}

TEST_AUDIO_FEATURES = {
    "danceability": 0.7,
    "energy": 0.8, 
    "valence": 0.6,
    "acousticness": 0.2,
    "instrumentalness": 0.1,
    "speechiness": 0.05,
    "tempo": 120.0,
    "duration_ms": 210000
}

TEST_SERVICE_CONFIGS = {
    "user_service": {
        "name": "user-service",
        "host": "localhost", 
        "port": 8001,
        "health_path": "/health"
    },
    "playlist_service": {
        "name": "playlist-service",
        "host": "localhost",
        "port": 8002, 
        "health_path": "/health"
    },
    "recommendation_service": {
        "name": "recommendation-service",
        "host": "localhost",
        "port": 8003,
        "health_path": "/api/health"
    }
}

# Utilitaires de test
class TestDataGenerator:
    """Générateur de données de test."""
    
    @staticmethod
    def generate_user_data(count: int = 1) -> List[Dict[str, Any]]:
        """Générer données utilisateur."""
        return [
            {
                "user_id": f"user_{i:06d}",
                "username": f"testuser_{i}",
                "email": f"user_{i}@test.com",
                "created_at": f"2024-01-{(i % 28) + 1:02d}T10:00:00Z"
            }
            for i in range(count)
        ]
    
    @staticmethod 
    def generate_audio_features(count: int = 1) -> List[Dict[str, float]]:
        """Générer features audio."""
        import random
        return [
            {
                "danceability": random.uniform(0.0, 1.0),
                "energy": random.uniform(0.0, 1.0),
                "valence": random.uniform(0.0, 1.0),
                "acousticness": random.uniform(0.0, 1.0),
                "instrumentalness": random.uniform(0.0, 1.0),
                "speechiness": random.uniform(0.0, 1.0),
                "tempo": random.uniform(60.0, 200.0),
                "duration_ms": random.randint(30000, 600000)
            }
            for _ in range(count)
        ]
    
    @staticmethod
    def generate_playlist_data(count: int = 1) -> List[Dict[str, Any]]:
        """Générer données playlist."""
        import random
        genres = ["rock", "pop", "electronic", "jazz", "classical", "hip-hop"]
        
        return [
            {
                "playlist_id": f"playlist_{i:06d}",
                "name": f"Test Playlist {i}",
                "description": f"Generated test playlist {i}",
                "genre": random.choice(genres),
                "track_count": random.randint(10, 50),
                "duration_ms": random.randint(600000, 3600000),
                "is_public": random.choice([True, False])
            }
            for i in range(count)
        ]

class TestAssertions:
    """Assertions personnalisées pour les tests."""
    
    @staticmethod
    def assert_framework_healthy(health_status):
        """Vérifier qu'un framework est sain."""
        assert health_status is not None
        assert health_status.status.name in ["RUNNING", "INITIALIZING"]
        assert health_status.message is not None
        assert isinstance(health_status.details, dict)
    
    @staticmethod
    def assert_metrics_valid(metrics: Dict[str, Any]):
        """Vérifier validité des métriques."""
        assert isinstance(metrics, dict)
        for key, value in metrics.items():
            assert isinstance(key, str)
            assert value is not None
            if isinstance(value, (int, float)):
                assert value >= 0  # Métriques positives
    
    @staticmethod
    def assert_prediction_result_valid(result):
        """Vérifier validité résultat prédiction."""
        assert result is not None
        assert hasattr(result, 'predictions')
        assert hasattr(result, 'confidence') 
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.predictions, dict)
    
    @staticmethod
    def assert_service_config_valid(config):
        """Vérifier validité configuration service."""
        assert config.name is not None
        assert config.host is not None
        assert 1 <= config.port <= 65535
        assert config.service_type is not None

class PerformanceTracker:
    """Tracker de performance pour les tests."""
    
    def __init__(self):
        self.measurements = {}
    
    def start_measurement(self, operation: str):
        """Démarrer mesure performance."""
        import time
        self.measurements[operation] = {"start": time.time()}
    
    def end_measurement(self, operation: str) -> float:
        """Terminer mesure et retourner durée."""
        import time
        if operation in self.measurements:
            end_time = time.time()
            duration = end_time - self.measurements[operation]["start"]
            self.measurements[operation]["duration"] = duration
            return duration
        return 0.0
    
    def get_summary(self) -> Dict[str, float]:
        """Obtenir résumé performances."""
        return {
            op: data.get("duration", 0.0) 
            for op, data in self.measurements.items()
        }

# Configuration spécifique environnements
class TestEnvironment:
    """Gestion environnement de test."""
    
    @staticmethod
    def is_ci_environment() -> bool:
        """Vérifier si on est en CI."""
        return os.getenv("CI", "false").lower() == "true"
    
    @staticmethod
    def get_test_mode() -> str:
        """Obtenir mode de test."""
        return os.getenv("TEST_MODE", "unit")  # unit, integration, performance
    
    @staticmethod
    def should_skip_external_tests() -> bool:
        """Vérifier si on doit ignorer tests externes."""
        return os.getenv("SKIP_EXTERNAL_TESTS", "false").lower() == "true"
    
    @staticmethod
    def get_parallel_workers() -> int:
        """Obtenir nombre de workers parallèles."""
        default_workers = 1 if TestEnvironment.is_ci_environment() else 4
        return int(os.getenv("TEST_WORKERS", default_workers))

# Hooks pytest personnalisés
def pytest_runtest_setup(item):
    """Setup avant chaque test."""
    # Skip tests externes si configuré
    if TestEnvironment.should_skip_external_tests():
        if "external" in [marker.name for marker in item.iter_markers()]:
            pytest.skip("Skipping external tests")
    
    # Skip tests lents en mode rapide
    if os.getenv("FAST_TESTS", "false").lower() == "true":
        if "slow" in [marker.name for marker in item.iter_markers()]:
            pytest.skip("Skipping slow tests in fast mode")

def pytest_runtest_teardown(item):
    """Teardown après chaque test."""
    # Cleanup automatique si nécessaire
    pass

# Export des utilitaires
__all__ = [
    "TEST_USER_DATA",
    "TEST_AUDIO_FEATURES", 
    "TEST_SERVICE_CONFIGS",
    "TestDataGenerator",
    "TestAssertions",
    "PerformanceTracker",
    "TestEnvironment",
    "clean_frameworks",
    "mock_external_services"
]
