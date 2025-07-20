"""
🎵 Configuration Pytest Ultra-Avancée pour API Core Tests
==========================================================

Configuration enterprise-grade pour tous les tests du module API Core
avec fixtures partagées, marks de test, et configuration avancée.

Développé par Fahed Mlaiel - Enterprise Testing Configuration Expert
"""

import pytest
import asyncio
import warnings
import os
import sys
from typing import Dict, Any, Generator
from unittest.mock import Mock, patch

# Ajouter le chemin de l'application au Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../"))

# Configuration des marks de test
def pytest_configure(config):
    """Configuration des marks de test personnalisés"""
    config.addinivalue_line(
        "markers", "unit: Tests unitaires pour composants individuels"
    )
    config.addinivalue_line(
        "markers", "integration: Tests d'intégration entre composants"
    )
    config.addinivalue_line(
        "markers", "performance: Tests de performance et benchmarks"
    )
    config.addinivalue_line(
        "markers", "security: Tests de sécurité et vulnérabilités"
    )
    config.addinivalue_line(
        "markers", "e2e: Tests end-to-end complets"
    )
    config.addinivalue_line(
        "markers", "slow: Tests qui prennent du temps à s'exécuter"
    )
    config.addinivalue_line(
        "markers", "network: Tests qui nécessitent une connexion réseau"
    )
    config.addinivalue_line(
        "markers", "database: Tests qui nécessitent une base de données"
    )
    config.addinivalue_line(
        "markers", "configuration: Tests de configuration et environnement"
    )


# Configuration des warnings
def pytest_sessionstart(session):
    """Configuration au démarrage de la session de test"""
    # Filtrer les warnings non critiques
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    warnings.filterwarnings("ignore", module="urllib3")


# =============================================================================
# FIXTURES GLOBALES POUR TOUS LES TESTS
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Event loop partagé pour les tests async"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_config():
    """Configuration de test globale"""
    return {
        "app": {
            "name": "Test Spotify AI Agent",
            "version": "1.0.0-test",
            "debug": True,
            "environment": "test",
            "testing": True
        },
        "database": {
            "url": "sqlite:///test.db",
            "echo": False,
            "pool_size": 1,
            "max_overflow": 0
        },
        "redis": {
            "url": "redis://localhost:6379/15",  # DB 15 pour les tests
            "timeout": 5,
            "max_connections": 5
        },
        "monitoring": {
            "enabled": True,
            "metrics": {
                "enabled": True,
                "port": 9091  # Port différent pour les tests
            },
            "health": {
                "enabled": True,
                "path": "/health"
            },
            "alerts": {
                "enabled": False  # Désactivé pour les tests
            }
        },
        "security": {
            "cors_enabled": True,
            "rate_limit": {
                "enabled": False  # Désactivé pour les tests
            }
        },
        "logging": {
            "level": "WARNING",  # Moins verbose pour les tests
            "format": "simple"
        }
    }


@pytest.fixture(scope="function")
def clean_environment():
    """Environnement propre pour chaque test"""
    # Sauvegarder l'état initial
    original_env = os.environ.copy()
    
    # Variables d'environnement pour les tests
    test_env = {
        "ENVIRONMENT": "test",
        "TESTING": "true",
        "LOG_LEVEL": "WARNING",
        "DATABASE_URL": "sqlite:///test.db",
        "REDIS_URL": "redis://localhost:6379/15"
    }
    
    # Appliquer les variables de test
    os.environ.update(test_env)
    
    yield
    
    # Restaurer l'environnement original
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(scope="function")
def mock_external_services():
    """Mock des services externes pour les tests"""
    mocks = {}
    
    # Mock Spotify API
    mocks['spotify_api'] = Mock()
    mocks['spotify_api'].get_user_profile.return_value = {
        "id": "test_user",
        "display_name": "Test User"
    }
    
    # Mock Database
    mocks['database'] = Mock()
    mocks['database'].execute.return_value = Mock()
    
    # Mock Redis
    mocks['redis'] = Mock()
    mocks['redis'].get.return_value = None
    mocks['redis'].set.return_value = True
    
    # Mock External APIs
    mocks['external_api'] = Mock()
    mocks['external_api'].request.return_value = {"status": "success"}
    
    return mocks


@pytest.fixture(scope="function")
def patch_singletons():
    """Patch des singletons pour éviter les effets de bord"""
    patches = []
    
    # Patch ComponentFactory
    factory_patch = patch('app.api.core.factory.ComponentFactory._instance', None)
    patches.append(factory_patch)
    factory_patch.start()
    
    # Patch DependencyContainer
    container_patch = patch('app.api.core.factory.DependencyContainer._instance', None)
    patches.append(container_patch)
    container_patch.start()
    
    # Patch APIMetrics
    metrics_patch = patch('app.api.core.monitoring.APIMetrics._instance', None)
    patches.append(metrics_patch)
    metrics_patch.start()
    
    # Patch HealthChecker
    health_patch = patch('app.api.core.monitoring.HealthChecker._instance', None)
    patches.append(health_patch)
    health_patch.start()
    
    yield
    
    # Arrêter tous les patches
    for patch_obj in patches:
        patch_obj.stop()


@pytest.fixture(scope="function")
def clear_caches():
    """Nettoyer les caches entre les tests"""
    # Nettoyer les caches de modules si ils existent
    modules_to_clear = [
        'app.api.core.config',
        'app.api.core.context',
        'app.api.core.factory',
        'app.api.core.monitoring'
    ]
    
    for module_name in modules_to_clear:
        if module_name in sys.modules:
            module = sys.modules[module_name]
            # Nettoyer les caches spécifiques du module
            if hasattr(module, '_cache'):
                module._cache.clear()
            if hasattr(module, '_instances'):
                module._instances.clear()
    
    yield
    
    # Nettoyage final si nécessaire


@pytest.fixture(scope="function")
def time_limit():
    """Limite de temps pour les tests"""
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Test timeout exceeded")
    
    # Définir un timeout de 30 secondes par test
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)
    
    yield
    
    # Annuler le timeout
    signal.alarm(0)


# =============================================================================
# FIXTURES SPÉCIALISÉES
# =============================================================================

@pytest.fixture
def sample_user_context():
    """Contexte utilisateur pour les tests"""
    from app.api.core.context import UserContext
    
    return UserContext(
        user_id="test_user_123",
        username="testuser",
        roles=["user", "beta_tester"],
        permissions=["read", "write"],
        metadata={
            "test_user": True,
            "created_for_test": True
        }
    )


@pytest.fixture
def sample_request_data():
    """Données de requête type pour les tests"""
    return {
        "method": "GET",
        "path": "/api/v1/test",
        "headers": {
            "User-Agent": "TestClient/1.0",
            "Accept": "application/json",
            "Content-Type": "application/json"
        },
        "query_params": {},
        "body": None,
        "client_ip": "127.0.0.1"
    }


@pytest.fixture
def performance_thresholds():
    """Seuils de performance pour les tests"""
    return {
        "response_time_ms": 100,     # Temps de réponse max
        "memory_usage_mb": 50,       # Utilisation mémoire max
        "cpu_usage_percent": 50,     # Utilisation CPU max
        "database_queries": 5,       # Nombre max de requêtes DB
        "cache_hit_ratio": 0.8,      # Ratio de cache hits min
        "concurrent_requests": 100   # Requêtes concurrentes max
    }


@pytest.fixture
def security_test_data():
    """Données pour les tests de sécurité"""
    return {
        "xss_payloads": [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "{{constructor.constructor('alert(\"xss\")')()}}"
        ],
        "sql_injection_payloads": [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "1; DELETE FROM users WHERE 1=1; --",
            "' UNION SELECT password FROM users; --"
        ],
        "command_injection_payloads": [
            "; cat /etc/passwd",
            "| ls -la",
            "&& whoami",
            "$(id)"
        ],
        "path_traversal_payloads": [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
            "....//....//....//etc/passwd"
        ]
    }


# =============================================================================
# FIXTURES DE PERFORMANCE
# =============================================================================

@pytest.fixture
def memory_profiler():
    """Profiler de mémoire pour les tests de performance"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    class MemoryProfiler:
        def __init__(self, initial_mem):
            self.initial_memory = initial_mem
            self.process = process
        
        def get_current_usage(self):
            return self.process.memory_info().rss
        
        def get_increase(self):
            return self.get_current_usage() - self.initial_memory
        
        def get_increase_mb(self):
            return self.get_increase() / (1024 * 1024)
    
    yield MemoryProfiler(initial_memory)


@pytest.fixture
def cpu_profiler():
    """Profiler de CPU pour les tests de performance"""
    import psutil
    import time
    
    process = psutil.Process()
    
    class CPUProfiler:
        def __init__(self):
            self.process = process
            self.start_time = None
            self.start_cpu_time = None
        
        def start(self):
            self.start_time = time.time()
            self.start_cpu_time = self.process.cpu_percent()
        
        def get_usage(self):
            if self.start_time is None:
                self.start()
                time.sleep(0.1)  # Petit délai pour la mesure
            
            return self.process.cpu_percent()
    
    yield CPUProfiler()


# =============================================================================
# HOOKS PYTEST
# =============================================================================

@pytest.fixture(autouse=True)
def setup_test_logging():
    """Configuration du logging pour les tests"""
    import logging
    
    # Réduire le niveau de logging pour les tests
    logging.getLogger().setLevel(logging.WARNING)
    
    # Désactiver les loggers verbeux
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    yield


def pytest_runtest_setup(item):
    """Hook exécuté avant chaque test"""
    # Marquer le début du test
    if hasattr(item, 'obj') and hasattr(item.obj, '__doc__'):
        test_doc = item.obj.__doc__
        if test_doc:
            print(f"\n🧪 Exécution: {item.name}")


def pytest_runtest_teardown(item, nextitem):
    """Hook exécuté après chaque test"""
    # Nettoyage automatique après chaque test
    import gc
    gc.collect()


def pytest_collection_modifyitems(config, items):
    """Modifier la collection de tests"""
    # Ajouter des marks automatiques basés sur les noms
    for item in items:
        # Mark performance tests
        if "performance" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.performance)
        
        # Mark slow tests
        if "slow" in item.name or "load" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in item.name:
            item.add_marker(pytest.mark.integration)
        
        # Mark security tests
        if "security" in item.name or "xss" in item.name or "injection" in item.name:
            item.add_marker(pytest.mark.security)


# =============================================================================
# CONFIGURATION POUR TESTS PARALLÈLES
# =============================================================================

@pytest.fixture(scope="session")
def worker_id(request):
    """ID du worker pour les tests parallèles"""
    if hasattr(request.config, 'workerinput'):
        return request.config.workerinput['workerid']
    return 'master'


@pytest.fixture(scope="session")
def test_database_url(worker_id):
    """URL de base de données spécifique au worker"""
    if worker_id == 'master':
        return "sqlite:///test.db"
    else:
        return f"sqlite:///test_{worker_id}.db"


# =============================================================================
# UTILITAIRES DE TEST
# =============================================================================

@pytest.fixture
def assert_performance():
    """Assertions de performance"""
    def _assert_performance(actual_time, expected_time, message=""):
        """Assert que le temps actual est inférieur au temps attendu"""
        assert actual_time <= expected_time, (
            f"Performance test failed: {actual_time}ms > {expected_time}ms. {message}"
        )
    
    return _assert_performance


@pytest.fixture
def assert_memory():
    """Assertions de mémoire"""
    def _assert_memory(actual_mb, expected_mb, message=""):
        """Assert que l'utilisation mémoire est acceptable"""
        assert actual_mb <= expected_mb, (
            f"Memory test failed: {actual_mb}MB > {expected_mb}MB. {message}"
        )
    
    return _assert_memory


# =============================================================================
# CONFIGURATION FINALE
# =============================================================================

# Options pytest par défaut - Moved to top-level conftest.py
# pytest_plugins should be defined in the top-level conftest.py
# to avoid deprecation warnings in pytest

# Configuration pour pytest-asyncio
pytestmark = pytest.mark.asyncio
