"""
🧪 Tests Hybrid Backend - Django + FastAPI Integration
=====================================================

Tests complets du backend hybride avec:
- Intégration Django/FastAPI
- Middleware partagé
- Session management
- Database optimization
- CORS et sécurité

Développé par: Senior Backend Developer
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any
import json
import httpx
from fastapi.testclient import TestClient

from backend.app.frameworks.hybrid_backend import (
    DjangoFramework,
    FastAPIFramework,
    HybridBackend,
    HybridConfig,
    SharedMiddleware,
    DatabaseManager
)
from backend.app.frameworks import TEST_CONFIG, clean_frameworks, logger


@pytest.fixture
def hybrid_config():
    """Configuration test pour le backend hybride."""
    return HybridConfig(
        database_url=TEST_CONFIG["test_database_url"],
        fastapi_title="Test Spotify AI Agent",
        fastapi_version="1.0.0-test",
        enable_cors=True,
        cors_origins=["http://localhost:3000"],
        enable_compression=True,
        database_pool_size=5,
        redis_url=TEST_CONFIG["test_redis_url"],
        jwt_secret_key=TEST_CONFIG["test_jwt_secret"]
    )


@pytest.fixture
def mock_django_settings():
    """Mock des settings Django pour les tests."""
    with patch('django.conf.settings') as mock_settings:
        mock_settings.DATABASES = {
            'default': {
                'ENGINE': 'django.db.backends.sqlite3',
                'NAME': 'test_db.sqlite3'
            }
        }
        mock_settings.INSTALLED_APPS = [
            'django.contrib.auth',
            'django.contrib.contenttypes',
            'django.contrib.sessions',
            'django.contrib.admin'
        ]
        mock_settings.SECRET_KEY = 'test-secret-key'
        mock_settings.DEBUG = True
        yield mock_settings


@pytest.mark.hybrid
class TestHybridConfig:
    """Tests de la configuration hybride."""
    
    def test_hybrid_config_creation(self):
        """Test création configuration hybride."""
        config = HybridConfig(
            database_url="sqlite:///test.db",
            fastapi_title="Test API"
        )
        
        assert config.database_url == "sqlite:///test.db"
        assert config.fastapi_title == "Test API"
        assert config.fastapi_version == "1.0.0"
        assert config.enable_cors is True
        assert config.database_pool_size == 20
        
    def test_hybrid_config_validation(self):
        """Test validation configuration hybride."""
        # Configuration valide
        config = HybridConfig(database_url="postgresql://user:pass@localhost/db")
        assert config.database_url.startswith("postgresql://")
        
        # Configuration avec URL invalide
        with pytest.raises(ValueError):
            HybridConfig(database_url="invalid-url")
            
    def test_hybrid_config_django_settings_generation(self):
        """Test génération settings Django."""
        config = HybridConfig(
            database_url="postgresql://user:pass@localhost:5432/testdb",
            debug=True
        )
        
        django_settings = config.get_django_settings()
        
        assert django_settings["DEBUG"] is True
        assert "django.contrib.admin" in django_settings["INSTALLED_APPS"]
        assert django_settings["DATABASES"]["default"]["ENGINE"] == "django.db.backends.postgresql"
        assert django_settings["DATABASES"]["default"]["NAME"] == "testdb"
        assert django_settings["DATABASES"]["default"]["HOST"] == "localhost"
        assert django_settings["DATABASES"]["default"]["PORT"] == 5432


@pytest.mark.hybrid
class TestSharedMiddleware:
    """Tests du middleware partagé."""
    
    def test_shared_middleware_creation(self):
        """Test création middleware partagé."""
        middleware = SharedMiddleware()
        assert middleware.request_count == 0
        assert len(middleware.active_sessions) == 0
        
    @pytest.mark.asyncio
    async def test_cors_middleware(self):
        """Test middleware CORS."""
        middleware = SharedMiddleware()
        
        # Mock request/response
        request = Mock()
        request.headers = {"origin": "http://localhost:3000"}
        response = Mock()
        response.headers = {}
        
        # Traitement CORS
        processed_response = await middleware.process_cors(request, response)
        
        assert "Access-Control-Allow-Origin" in processed_response.headers
        assert processed_response.headers["Access-Control-Allow-Origin"] == "http://localhost:3000"
        
    @pytest.mark.asyncio
    async def test_compression_middleware(self):
        """Test middleware compression."""
        middleware = SharedMiddleware()
        
        # Mock response avec contenu
        response = Mock()
        response.headers = {}
        response.body = b"a" * 1000  # Contenu > seuil compression
        
        compressed_response = await middleware.process_compression(response)
        
        assert "Content-Encoding" in compressed_response.headers
        assert compressed_response.headers["Content-Encoding"] == "gzip"
        assert len(compressed_response.body) < len(response.body)
        
    @pytest.mark.asyncio
    async def test_session_management(self):
        """Test gestion des sessions."""
        middleware = SharedMiddleware()
        
        # Créer session
        session_data = {"user_id": 123, "username": "testuser"}
        session_id = await middleware.create_session(session_data)
        
        assert session_id is not None
        assert len(middleware.active_sessions) == 1
        
        # Récupérer session
        retrieved_session = await middleware.get_session(session_id)
        assert retrieved_session["user_id"] == 123
        assert retrieved_session["username"] == "testuser"
        
        # Supprimer session
        await middleware.delete_session(session_id)
        assert len(middleware.active_sessions) == 0
        
    @pytest.mark.asyncio
    async def test_request_logging(self):
        """Test logging des requêtes."""
        middleware = SharedMiddleware()
        
        request = Mock()
        request.method = "GET"
        request.url = "http://localhost:8000/api/test"
        request.headers = {"user-agent": "test-client"}
        
        start_time = await middleware.start_request_logging(request)
        assert middleware.request_count == 1
        
        response = Mock()
        response.status_code = 200
        
        await middleware.end_request_logging(request, response, start_time)
        # Vérifier que le logging a été effectué


@pytest.mark.hybrid
class TestDatabaseManager:
    """Tests du gestionnaire de base de données."""
    
    @pytest.mark.asyncio
    async def test_database_manager_creation(self):
        """Test création gestionnaire BDD."""
        manager = DatabaseManager(TEST_CONFIG["test_database_url"])
        assert manager.database_url == TEST_CONFIG["test_database_url"]
        assert manager.pool is None
        
    @pytest.mark.asyncio
    async def test_database_connection_pool(self):
        """Test pool de connexions."""
        manager = DatabaseManager(TEST_CONFIG["test_database_url"])
        
        await manager.create_pool(pool_size=5)
        assert manager.pool is not None
        
        # Test connexion
        async with manager.get_connection() as conn:
            assert conn is not None
            
        await manager.close_pool()
        
    @pytest.mark.asyncio
    async def test_database_health_check(self):
        """Test health check BDD."""
        manager = DatabaseManager(TEST_CONFIG["test_database_url"])
        await manager.create_pool()
        
        is_healthy = await manager.health_check()
        assert is_healthy is True
        
        await manager.close_pool()
        
    @pytest.mark.asyncio
    async def test_database_migration_check(self):
        """Test vérification migrations."""
        manager = DatabaseManager(TEST_CONFIG["test_database_url"])
        
        # Mock du système de migration Django
        with patch('django.core.management.call_command') as mock_command:
            await manager.check_migrations()
            mock_command.assert_called()


@pytest.mark.hybrid
class TestDjangoFramework:
    """Tests du framework Django."""
    
    @pytest.mark.asyncio
    async def test_django_framework_initialization(self, hybrid_config, mock_django_settings):
        """Test initialisation framework Django."""
        django_framework = DjangoFramework(hybrid_config)
        
        with patch('django.setup'), patch('django.core.management.call_command'):
            result = await django_framework.initialize()
            assert result is True
            assert django_framework.status.name == "RUNNING"
            
    @pytest.mark.asyncio
    async def test_django_admin_interface(self, hybrid_config, mock_django_settings):
        """Test interface admin Django."""
        django_framework = DjangoFramework(hybrid_config)
        
        with patch('django.setup'), patch('django.core.management.call_command'):
            await django_framework.initialize()
            
            admin_urls = django_framework.get_admin_urls()
            assert "/admin/" in str(admin_urls)
            
    @pytest.mark.asyncio
    async def test_django_health_check(self, hybrid_config):
        """Test health check Django."""
        django_framework = DjangoFramework(hybrid_config)
        
        with patch.object(django_framework.db_manager, 'health_check', return_value=True):
            health = await django_framework.health_check()
            assert health.status.name == "RUNNING"
            assert "Django framework" in health.message


@pytest.mark.hybrid
class TestFastAPIFramework:
    """Tests du framework FastAPI."""
    
    @pytest.mark.asyncio
    async def test_fastapi_framework_initialization(self, hybrid_config):
        """Test initialisation framework FastAPI."""
        fastapi_framework = FastAPIFramework(hybrid_config)
        
        result = await fastapi_framework.initialize()
        assert result is True
        assert fastapi_framework.status.name == "RUNNING"
        assert fastapi_framework.app is not None
        
    def test_fastapi_app_configuration(self, hybrid_config):
        """Test configuration app FastAPI."""
        fastapi_framework = FastAPIFramework(hybrid_config)
        asyncio.run(fastapi_framework.initialize())
        
        app = fastapi_framework.app
        assert app.title == hybrid_config.fastapi_title
        assert app.version == hybrid_config.fastapi_version
        
    def test_fastapi_middleware_setup(self, hybrid_config):
        """Test setup middleware FastAPI."""
        fastapi_framework = FastAPIFramework(hybrid_config)
        asyncio.run(fastapi_framework.initialize())
        
        # Vérifier que les middlewares sont configurés
        middlewares = [middleware.cls.__name__ for middleware in fastapi_framework.app.middleware]
        assert "CORSMiddleware" in middlewares
        assert "GZipMiddleware" in middlewares
        
    @pytest.mark.asyncio
    async def test_fastapi_health_endpoints(self, hybrid_config):
        """Test endpoints health FastAPI."""
        fastapi_framework = FastAPIFramework(hybrid_config)
        await fastapi_framework.initialize()
        
        client = TestClient(fastapi_framework.app)
        
        # Test endpoint health
        response = client.get("/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data["status"] == "healthy"
        assert "timestamp" in health_data
        
    @pytest.mark.asyncio
    async def test_fastapi_metrics_endpoints(self, hybrid_config):
        """Test endpoints métriques FastAPI."""
        fastapi_framework = FastAPIFramework(hybrid_config)
        await fastapi_framework.initialize()
        
        client = TestClient(fastapi_framework.app)
        
        # Test endpoint métriques
        response = client.get("/metrics")
        assert response.status_code == 200
        
        # Vérifier format Prometheus
        assert "# HELP" in response.text
        assert "# TYPE" in response.text


@pytest.mark.hybrid
class TestHybridBackend:
    """Tests du backend hybride complet."""
    
    @pytest.mark.asyncio
    async def test_hybrid_backend_initialization(self, hybrid_config, clean_frameworks):
        """Test initialisation backend hybride."""
        hybrid_backend = HybridBackend(hybrid_config)
        
        with patch('django.setup'), patch('django.core.management.call_command'):
            result = await hybrid_backend.initialize()
            assert result is True
            assert hybrid_backend.status.name == "RUNNING"
            
    @pytest.mark.asyncio
    async def test_hybrid_backend_django_fastapi_integration(self, hybrid_config, clean_frameworks):
        """Test intégration Django/FastAPI."""
        hybrid_backend = HybridBackend(hybrid_config)
        
        with patch('django.setup'), patch('django.core.management.call_command'):
            await hybrid_backend.initialize()
            
            # Vérifier que les deux apps sont disponibles
            django_app = hybrid_backend.get_django_app()
            fastapi_app = hybrid_backend.get_fastapi_app()
            
            assert django_app is not None
            assert fastapi_app is not None
            
    @pytest.mark.asyncio
    async def test_hybrid_backend_shared_middleware(self, hybrid_config, clean_frameworks):
        """Test middleware partagé."""
        hybrid_backend = HybridBackend(hybrid_config)
        
        with patch('django.setup'), patch('django.core.management.call_command'):
            await hybrid_backend.initialize()
            
            middleware = hybrid_backend.get_shared_middleware()
            assert middleware is not None
            assert isinstance(middleware, SharedMiddleware)
            
    @pytest.mark.asyncio
    async def test_hybrid_backend_health_check(self, hybrid_config, clean_frameworks):
        """Test health check backend hybride."""
        hybrid_backend = HybridBackend(hybrid_config)
        
        with patch('django.setup'), patch('django.core.management.call_command'):
            await hybrid_backend.initialize()
            
            health = await hybrid_backend.health_check()
            assert health.status.name == "RUNNING"
            assert "Hybrid backend" in health.message
            assert "django" in health.details
            assert "fastapi" in health.details
            
    @pytest.mark.asyncio
    async def test_hybrid_backend_graceful_shutdown(self, hybrid_config, clean_frameworks):
        """Test arrêt graceful."""
        hybrid_backend = HybridBackend(hybrid_config)
        
        with patch('django.setup'), patch('django.core.management.call_command'):
            await hybrid_backend.initialize()
            result = await hybrid_backend.shutdown()
            
            assert result is True
            assert hybrid_backend.status.name == "STOPPED"


@pytest.mark.hybrid
@pytest.mark.integration
class TestHybridBackendIntegration:
    """Tests d'intégration backend hybride."""
    
    @pytest.mark.asyncio
    async def test_full_request_lifecycle(self, hybrid_config, clean_frameworks):
        """Test cycle de vie complet d'une requête."""
        hybrid_backend = HybridBackend(hybrid_config)
        
        with patch('django.setup'), patch('django.core.management.call_command'):
            await hybrid_backend.initialize()
            
            fastapi_app = hybrid_backend.get_fastapi_app()
            client = TestClient(fastapi_app)
            
            # Test requête API
            response = client.get("/health")
            assert response.status_code == 200
            
            # Vérifier métriques après requête
            middleware = hybrid_backend.get_shared_middleware()
            assert middleware.request_count > 0
            
    @pytest.mark.asyncio
    async def test_database_integration(self, hybrid_config, clean_frameworks):
        """Test intégration base de données."""
        hybrid_backend = HybridBackend(hybrid_config)
        
        with patch('django.setup'), patch('django.core.management.call_command'):
            await hybrid_backend.initialize()
            
            db_manager = hybrid_backend.get_database_manager()
            is_healthy = await db_manager.health_check()
            assert is_healthy is True
            
    @pytest.mark.asyncio
    async def test_session_sharing_between_frameworks(self, hybrid_config, clean_frameworks):
        """Test partage de sessions entre frameworks."""
        hybrid_backend = HybridBackend(hybrid_config)
        
        with patch('django.setup'), patch('django.core.management.call_command'):
            await hybrid_backend.initialize()
            
            middleware = hybrid_backend.get_shared_middleware()
            
            # Créer session via middleware
            session_data = {"user_id": 123, "framework": "test"}
            session_id = await middleware.create_session(session_data)
            
            # Vérifier que la session est accessible
            retrieved_session = await middleware.get_session(session_id)
            assert retrieved_session["user_id"] == 123
            assert retrieved_session["framework"] == "test"


@pytest.mark.hybrid
@pytest.mark.performance
class TestHybridBackendPerformance:
    """Tests de performance backend hybride."""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_handling(self, hybrid_config, clean_frameworks):
        """Test gestion requêtes concurrentes."""
        hybrid_backend = HybridBackend(hybrid_config)
        
        with patch('django.setup'), patch('django.core.management.call_command'):
            await hybrid_backend.initialize()
            
            fastapi_app = hybrid_backend.get_fastapi_app()
            
            # Simuler requêtes concurrentes
            async def make_request():
                client = TestClient(fastapi_app)
                response = client.get("/health")
                return response.status_code
                
            # Exécuter 10 requêtes concurrentes
            tasks = [make_request() for _ in range(10)]
            results = await asyncio.gather(*tasks)
            
            # Toutes les requêtes doivent réussir
            assert all(status == 200 for status in results)
            
    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self, hybrid_config, clean_frameworks):
        """Test monitoring utilisation mémoire."""
        hybrid_backend = HybridBackend(hybrid_config)
        
        with patch('django.setup'), patch('django.core.management.call_command'):
            await hybrid_backend.initialize()
            
            # Collecter métriques mémoire
            metrics = hybrid_backend.get_metrics()
            
            assert "memory_usage" in metrics
            assert "database_connections" in metrics
            assert metrics["memory_usage"] > 0
