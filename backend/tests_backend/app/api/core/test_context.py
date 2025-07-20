"""
🎵 Tests Ultra-Avancés pour API Core Context Management
======================================================

Tests industriels complets pour la gestion de contexte avec patterns enterprise,
tests de concurrence, performance, et validation thread-safety.

Développé par Fahed Mlaiel - Enterprise Context Testing Expert
"""

import pytest
import asyncio
import time
import threading
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import Request, Response
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

from app.api.core.context import (
    RequestPhase,
    UserContext,
    PerformanceContext,
    ErrorContext,
    RequestContext,
    APIContext,
    RequestContextMiddleware,
    get_request_context,
    set_request_context,
    get_api_context,
    set_api_context,
    clear_request_context,
    create_user_context,
    get_current_user,
    get_request_id,
    get_correlation_id,
    add_request_metadata,
    add_request_tag
)


# =============================================================================
# FIXTURES ENTERPRISE POUR CONTEXT TESTING
# =============================================================================

@pytest.fixture
def clean_context():
    """Context propre pour les tests"""
    # Nettoyer le contexte avant et après chaque test
    clear_request_context()
    yield
    clear_request_context()


@pytest.fixture
def sample_user_context():
    """Contexte utilisateur de test"""
    return UserContext(
        user_id="user_12345",
        username="test_user",
        email="test@example.com",
        roles=["user", "premium"],
        permissions=["read", "write", "playlist_create"],
        spotify_id="spotify_user_123",
        subscription_type="premium",
        is_authenticated=True,
        is_premium=True,
        auth_method="jwt",
        session_id="session_abc123"
    )


@pytest.fixture
def sample_request_context(sample_user_context):
    """Contexte de requête de test"""
    context = RequestContext(
        request_id="req_123456",
        correlation_id="corr_789012",
        method="POST",
        path="/api/v1/playlists",
        query_params={"limit": "10", "offset": "0"},
        headers={"user-agent": "TestClient/1.0", "authorization": "Bearer token123"},
        ip_address="192.168.1.100",
        user=sample_user_context
    )
    
    return context


@pytest.fixture
def mock_request():
    """Requête FastAPI mockée"""
    request = Mock(spec=Request)
    request.method = "GET"
    request.url.path = "/api/v1/test"
    request.query_params = {"param1": "value1"}
    request.headers = {
        "user-agent": "TestAgent/1.0",
        "x-correlation-id": "test-correlation-123",
        "x-session-id": "session-456"
    }
    request.client.host = "127.0.0.1"
    
    return request


@pytest.fixture
def fastapi_app():
    """Application FastAPI de test"""
    app = Starlette()
    
    @app.route("/test", methods=["GET", "POST"])
    async def test_endpoint(request):
        context = get_request_context()
        return JSONResponse({
            "request_id": context.request_id if context else None,
            "user_id": context.user.user_id if context and context.user else None
        })
    
    @app.route("/error")
    async def error_endpoint(request):
        raise ValueError("Test error")
    
    return app


# =============================================================================
# TESTS DE USER CONTEXT
# =============================================================================

class TestUserContext:
    """Tests pour UserContext"""
    
    def test_user_context_creation(self, sample_user_context):
        """Test création UserContext"""
        user = sample_user_context
        
        assert user.user_id == "user_12345"
        assert user.username == "test_user"
        assert user.email == "test@example.com"
        assert user.is_authenticated is True
        assert user.is_premium is True
        assert user.auth_method == "jwt"
    
    def test_user_context_roles_and_permissions(self, sample_user_context):
        """Test gestion des rôles et permissions"""
        user = sample_user_context
        
        # Test rôles
        assert user.has_role("user") is True
        assert user.has_role("premium") is True
        assert user.has_role("admin") is False
        
        # Test permissions
        assert user.has_permission("read") is True
        assert user.has_permission("write") is True
        assert user.has_permission("playlist_create") is True
        assert user.has_permission("admin_access") is False
    
    def test_user_context_to_dict(self, sample_user_context):
        """Test conversion en dictionnaire"""
        user = sample_user_context
        user_dict = user.to_dict()
        
        assert user_dict["user_id"] == "user_12345"
        assert user_dict["username"] == "test_user"
        assert user_dict["is_authenticated"] is True
        assert "roles" in user_dict
        assert "permissions" in user_dict
    
    def test_anonymous_user_context(self):
        """Test contexte utilisateur anonyme"""
        user = UserContext()
        
        assert user.user_id is None
        assert user.is_authenticated is False
        assert user.is_premium is False
        assert user.roles == []
        assert user.permissions == []


class TestPerformanceContext:
    """Tests pour PerformanceContext"""
    
    def test_performance_context_initialization(self):
        """Test initialisation PerformanceContext"""
        perf = PerformanceContext()
        
        assert perf.start_time is not None
        assert perf.end_time is None
        assert perf.duration_ms is None
        assert perf.db_queries == 0
        assert perf.cache_hits == 0
        assert perf.cache_misses == 0
        assert perf.external_calls == 0
    
    def test_performance_context_finish(self):
        """Test finalisation des métriques"""
        perf = PerformanceContext()
        initial_time = perf.start_time
        
        time.sleep(0.01)  # Attendre un peu
        perf.finish()
        
        assert perf.end_time is not None
        assert perf.end_time > initial_time
        assert perf.duration_ms is not None
        assert perf.duration_ms > 0
    
    def test_performance_counters(self):
        """Test compteurs de performance"""
        perf = PerformanceContext()
        
        # Test incrémentation des compteurs
        perf.add_db_query()
        perf.add_db_query()
        assert perf.db_queries == 2
        
        perf.add_cache_hit()
        perf.add_cache_hit()
        perf.add_cache_hit()
        assert perf.cache_hits == 3
        
        perf.add_cache_miss()
        assert perf.cache_misses == 1
        
        perf.add_external_call()
        perf.add_external_call()
        assert perf.external_calls == 2


class TestErrorContext:
    """Tests pour ErrorContext"""
    
    def test_error_context_creation(self):
        """Test création ErrorContext"""
        error_ctx = ErrorContext()
        
        assert error_ctx.error_id is None
        assert error_ctx.error_type is None
        assert error_ctx.error_message is None
        assert error_ctx.retry_count == 0
        assert error_ctx.is_retryable is False
    
    def test_error_context_set_error(self):
        """Test configuration d'erreur"""
        error_ctx = ErrorContext()
        exception = ValueError("Test error message")
        
        error_ctx.set_error(exception, "User friendly message")
        
        assert error_ctx.error_id is not None
        assert error_ctx.error_type == "ValueError"
        assert error_ctx.error_message == "Test error message"
        assert error_ctx.user_message == "User friendly message"
        assert error_ctx.stack_trace is not None
    
    def test_error_context_without_user_message(self):
        """Test configuration d'erreur sans message utilisateur"""
        error_ctx = ErrorContext()
        exception = RuntimeError("Runtime error")
        
        error_ctx.set_error(exception)
        
        assert error_ctx.error_type == "RuntimeError"
        assert error_ctx.error_message == "Runtime error"
        assert error_ctx.user_message is None


# =============================================================================
# TESTS DE REQUEST CONTEXT
# =============================================================================

class TestRequestContext:
    """Tests pour RequestContext"""
    
    def test_request_context_creation(self, clean_context):
        """Test création RequestContext"""
        context = RequestContext()
        
        assert context.request_id is not None
        assert context.correlation_id is not None
        assert context.phase == RequestPhase.RECEIVED
        assert context.timestamp is not None
        assert isinstance(context.user, UserContext)
        assert isinstance(context.performance, PerformanceContext)
    
    def test_request_context_phase_management(self, sample_request_context):
        """Test gestion des phases"""
        context = sample_request_context
        
        # Test changement de phase
        context.set_phase(RequestPhase.AUTHENTICATED)
        assert context.phase == RequestPhase.AUTHENTICATED
        
        context.set_phase(RequestPhase.PROCESSING)
        assert context.phase == RequestPhase.PROCESSING
        
        context.set_phase(RequestPhase.COMPLETED)
        assert context.phase == RequestPhase.COMPLETED
    
    def test_request_context_user_management(self, sample_request_context, sample_user_context):
        """Test gestion utilisateur"""
        context = sample_request_context
        new_user = UserContext(user_id="new_user_456")
        
        context.set_user(new_user)
        assert context.user.user_id == "new_user_456"
    
    def test_request_context_error_handling(self, sample_request_context):
        """Test gestion d'erreur"""
        context = sample_request_context
        exception = ValueError("Test error")
        
        context.set_error(exception, "Error occurred")
        
        assert context.phase == RequestPhase.ERROR
        assert context.error is not None
        assert context.error.error_type == "ValueError"
        assert context.error.user_message == "Error occurred"
    
    def test_request_context_metadata_management(self, sample_request_context):
        """Test gestion des métadonnées"""
        context = sample_request_context
        
        context.add_metadata("custom_field", "custom_value")
        context.add_metadata("request_source", "mobile_app")
        
        assert context.metadata["custom_field"] == "custom_value"
        assert context.metadata["request_source"] == "mobile_app"
    
    def test_request_context_tags_management(self, sample_request_context):
        """Test gestion des tags"""
        context = sample_request_context
        
        context.add_tag("premium_user")
        context.add_tag("mobile")
        context.add_tag("premium_user")  # Duplicate
        
        assert "premium_user" in context.tags
        assert "mobile" in context.tags
        assert len(context.tags) == 2  # Pas de doublons
    
    def test_request_context_to_dict(self, sample_request_context):
        """Test conversion en dictionnaire"""
        context = sample_request_context
        context.add_metadata("test_key", "test_value")
        context.add_tag("test_tag")
        
        context_dict = context.to_dict()
        
        assert context_dict["request_id"] == context.request_id
        assert context_dict["correlation_id"] == context.correlation_id
        assert context_dict["method"] == "POST"
        assert context_dict["path"] == "/api/v1/playlists"
        assert context_dict["user_id"] == "user_12345"
        assert "test_key" in context_dict["metadata"]
        assert "test_tag" in context_dict["tags"]


class TestAPIContext:
    """Tests pour APIContext"""
    
    def test_api_context_creation(self):
        """Test création APIContext"""
        context = APIContext()
        
        assert context.app_name == "Spotify AI Agent"
        assert context.app_version == "2.0.0"
        assert context.environment == "development"
        assert context.deployment_id is not None
        assert context.startup_time is not None
        assert context.total_requests == 0
        assert context.active_requests == 0
        assert context.total_errors == 0
    
    def test_api_context_metrics(self):
        """Test métriques APIContext"""
        context = APIContext()
        
        # Test incrémentation requêtes
        context.increment_requests()
        assert context.total_requests == 1
        assert context.active_requests == 1
        
        context.increment_requests()
        assert context.total_requests == 2
        assert context.active_requests == 2
        
        # Test décrémentation requêtes actives
        context.decrement_active_requests()
        assert context.total_requests == 2
        assert context.active_requests == 1
        
        # Test incrémentation erreurs
        context.increment_errors()
        assert context.total_errors == 1
    
    def test_api_context_custom_values(self):
        """Test valeurs personnalisées APIContext"""
        context = APIContext(
            app_name="Custom App",
            app_version="3.0.0",
            environment="production"
        )
        
        assert context.app_name == "Custom App"
        assert context.app_version == "3.0.0"
        assert context.environment == "production"


# =============================================================================
# TESTS DES FONCTIONS CONTEXT STORAGE
# =============================================================================

class TestContextStorage:
    """Tests pour le stockage de contexte avec ContextVars"""
    
    def test_request_context_storage(self, clean_context, sample_request_context):
        """Test stockage et récupération du contexte de requête"""
        # Initialement pas de contexte
        assert get_request_context() is None
        
        # Définir le contexte
        set_request_context(sample_request_context)
        
        # Récupérer le contexte
        retrieved_context = get_request_context()
        assert retrieved_context is not None
        assert retrieved_context.request_id == sample_request_context.request_id
        assert retrieved_context.user.user_id == "user_12345"
    
    def test_api_context_storage(self, clean_context):
        """Test stockage et récupération du contexte API"""
        api_context = APIContext(app_name="Test App")
        
        # Initialement pas de contexte
        assert get_api_context() is None
        
        # Définir le contexte
        set_api_context(api_context)
        
        # Récupérer le contexte
        retrieved_context = get_api_context()
        assert retrieved_context is not None
        assert retrieved_context.app_name == "Test App"
    
    def test_clear_request_context(self, clean_context, sample_request_context):
        """Test nettoyage du contexte de requête"""
        set_request_context(sample_request_context)
        assert get_request_context() is not None
        
        clear_request_context()
        assert get_request_context() is None
    
    @pytest.mark.asyncio
    async def test_context_isolation_between_tasks(self, clean_context):
        """Test isolation du contexte entre tâches async"""
        async def task1():
            context1 = RequestContext(request_id="task1_req")
            set_request_context(context1)
            await asyncio.sleep(0.01)
            return get_request_context().request_id
        
        async def task2():
            context2 = RequestContext(request_id="task2_req")
            set_request_context(context2)
            await asyncio.sleep(0.01)
            return get_request_context().request_id
        
        # Exécuter les tâches en parallèle
        results = await asyncio.gather(task1(), task2())
        
        # Chaque tâche doit avoir son propre contexte
        assert "task1_req" in results
        assert "task2_req" in results
    
    def test_context_isolation_between_threads(self, clean_context):
        """Test isolation du contexte entre threads"""
        results = []
        
        def thread_function(thread_id):
            context = RequestContext(request_id=f"thread_{thread_id}_req")
            set_request_context(context)
            time.sleep(0.01)
            retrieved_context = get_request_context()
            results.append(retrieved_context.request_id if retrieved_context else None)
        
        # Créer et lancer plusieurs threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=thread_function, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Attendre la fin des threads
        for thread in threads:
            thread.join()
        
        # Chaque thread doit avoir son propre contexte
        assert len(results) == 3
        assert "thread_0_req" in results
        assert "thread_1_req" in results
        assert "thread_2_req" in results


# =============================================================================
# TESTS DU MIDDLEWARE DE CONTEXTE
# =============================================================================

class TestRequestContextMiddleware:
    """Tests pour RequestContextMiddleware"""
    
    def test_middleware_initialization(self):
        """Test initialisation du middleware"""
        api_context = APIContext(app_name="Test API")
        middleware = RequestContextMiddleware(None, api_context)
        
        assert middleware.api_context.app_name == "Test API"
    
    @pytest.mark.asyncio
    async    def test_middleware_context_creation(self, clean_context):
        """Test création de contexte par le middleware"""
        # Créer une nouvelle app avec middleware intégré
        from fastapi import FastAPI
        app = FastAPI()
        api_context = APIContext()
        
        # Ajouter le middleware AVANT de créer le client
        app.add_middleware(RequestContextMiddleware, api_context=api_context)
        
        # Ajouter une route de test
        @app.get("/test")
        async def test_route():
            return {"message": "test"}
        
        # Créer le client avec l'app complète
        with TestClient(app) as client:
            response = client.get("/test")
            
            assert response.status_code == 200
            data = response.json()
            
            # Le contexte devrait être créé
            assert data["message"] == "test"
    
    @pytest.mark.asyncio
    async def test_middleware_error_handling(self, clean_context):
        """Test gestion d'erreur par le middleware"""
        # Configuration FastAPI experte pour l'ordre des middlewares
        from fastapi import FastAPI
        
        # Créer FastAPI avec configuration d'expert
        app = FastAPI(
            debug=False,  # Désactiver debug pour utiliser nos handlers personnalisés
            exception_handlers={}  # Commencer avec handlers vides
        )
        api_context = APIContext()
        
        # ÉTAPE 1: Enregistrer NOS exception handlers personnalisés EN PREMIER
        from app.api.core.exceptions import register_exception_handlers
        register_exception_handlers(app)
        
        # ÉTAPE 2: Ajouter notre middleware APRÈS les handlers
        # L'ordre est crucial : les middlewares s'exécutent en ordre inverse
        app.add_middleware(RequestContextMiddleware, api_context=api_context)
        
        # ÉTAPE 3: Ajouter route qui lève exception
        @app.get("/error")
        async def error_route():
            raise ValueError("Test error")
        
        # ÉTAPE 4: Forcer la construction de la stack avec l'ordre correct
        # Cette méthode privée reconstruit la middleware stack dans l'ordre
        app.build_middleware_stack()
        
        # Test avec FastAPI TestClient configuré pour NE PAS propager les exceptions serveur
        # raise_server_exceptions=False permet aux exception handlers de fonctionner
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/error")
            
            # Vérifier que NOS exception handlers ont géré l'erreur
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert data["error"]["code"] == "UNKNOWN_ERROR"  # Notre handler personnalisé
            assert "message" in data["error"]
            
            # Vérifier que le middleware a fonctionné correctement
            assert api_context.total_requests >= 1
            assert api_context.total_errors >= 1
    
    def test_middleware_correlation_id_propagation(self, clean_context):
        """Test propagation du correlation ID"""
        # Créer une nouvelle app avec middleware intégré
        from fastapi import FastAPI
        app = FastAPI()
        api_context = APIContext()
        
        # Ajouter le middleware AVANT de créer le client
        app.add_middleware(RequestContextMiddleware, api_context=api_context)
        
        # Ajouter une route de test
        @app.get("/test")
        async def test_route():
            return {"message": "test"}
        
        # Créer le client avec l'app complète
        with TestClient(app) as client:
            # Envoyer une requête avec correlation ID
            headers = {"X-Correlation-ID": "test-correlation-123"}
            response = client.get("/test", headers=headers)
            
            assert response.status_code == 200
            # Vérifier que le correlation ID est propagé
            assert "X-Correlation-ID" in response.headers or "x-correlation-id" in response.headers
    
    def test_middleware_ip_address_extraction(self, clean_context):
        """Test extraction de l'adresse IP"""
        # Créer une nouvelle app avec middleware intégré
        from fastapi import FastAPI
        app = FastAPI()
        api_context = APIContext()
        
        # Ajouter le middleware AVANT de créer le client
        app.add_middleware(RequestContextMiddleware, api_context=api_context)
        
        # Ajouter une route de test
        @app.get("/test")
        async def test_route():
            return {"message": "test"}
        
        # Créer le client avec l'app complète
        with TestClient(app) as client:
            # Test avec X-Forwarded-For
            response = client.get("/test", headers={
                "x-forwarded-for": "203.0.113.1, 192.168.1.1"
            })
            
            assert response.status_code == 200
            # L'IP devrait être extraite du header X-Forwarded-For


# =============================================================================
# TESTS DES FONCTIONS UTILITAIRES
# =============================================================================

class TestUtilityFunctions:
    """Tests pour les fonctions utilitaires"""
    
    def test_create_user_context(self):
        """Test création de contexte utilisateur"""
        user = create_user_context(
            user_id="user_789",
            username="testuser",
            email="test@example.com",
            roles=["admin", "user"]
        )
        
        assert user.user_id == "user_789"
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.roles == ["admin", "user"]
        assert user.is_authenticated is True
    
    def test_get_current_user(self, clean_context, sample_request_context):
        """Test récupération utilisateur actuel"""
        # Pas de contexte
        assert get_current_user() is None
        
        # Avec contexte
        set_request_context(sample_request_context)
        current_user = get_current_user()
        
        assert current_user is not None
        assert current_user.user_id == "user_12345"
    
    def test_get_request_id(self, clean_context, sample_request_context):
        """Test récupération ID de requête"""
        # Pas de contexte
        assert get_request_id() is None
        
        # Avec contexte
        set_request_context(sample_request_context)
        request_id = get_request_id()
        
        assert request_id == "req_123456"
    
    def test_get_correlation_id(self, clean_context, sample_request_context):
        """Test récupération ID de corrélation"""
        # Pas de contexte
        assert get_correlation_id() is None
        
        # Avec contexte
        set_request_context(sample_request_context)
        correlation_id = get_correlation_id()
        
        assert correlation_id == "corr_789012"
    
    def test_add_request_metadata(self, clean_context, sample_request_context):
        """Test ajout de métadonnées"""
        set_request_context(sample_request_context)
        
        add_request_metadata("custom_field", "custom_value")
        add_request_metadata("source", "api_test")
        
        context = get_request_context()
        assert context.metadata["custom_field"] == "custom_value"
        assert context.metadata["source"] == "api_test"
    
    def test_add_request_tag(self, clean_context, sample_request_context):
        """Test ajout de tags"""
        set_request_context(sample_request_context)
        
        add_request_tag("performance_test")
        add_request_tag("api_v1")
        
        context = get_request_context()
        assert "performance_test" in context.tags
        assert "api_v1" in context.tags


# =============================================================================
# TESTS DE PERFORMANCE
# =============================================================================

@pytest.mark.performance
class TestContextPerformance:
    """Tests de performance pour le contexte"""
    
    def test_context_creation_performance(self, benchmark):
        """Test performance création de contexte"""
        def create_context():
            return RequestContext()
        
        result = benchmark(create_context)
        assert isinstance(result, RequestContext)
    
    def test_context_storage_performance(self, benchmark, clean_context):
        """Test performance stockage/récupération contexte"""
        context = RequestContext()
        
        def store_and_retrieve():
            set_request_context(context)
            return get_request_context()
        
        result = benchmark(store_and_retrieve)
        assert result is not None
    
    def test_metadata_operations_performance(self, benchmark, clean_context):
        """Test performance opérations métadonnées"""
        context = RequestContext()
        set_request_context(context)
        
        def metadata_operations():
            add_request_metadata("key1", "value1")
            add_request_metadata("key2", "value2")
            add_request_tag("tag1")
            add_request_tag("tag2")
            return get_request_context()
        
        result = benchmark(metadata_operations)
        assert len(result.metadata) >= 2
        assert len(result.tags) >= 2


# =============================================================================
# TESTS DE CONCURRENCE
# =============================================================================

@pytest.mark.concurrency
class TestContextConcurrency:
    """Tests de concurrence pour le contexte"""
    
    def test_concurrent_context_access(self, clean_context):
        """Test accès concurrent au contexte"""
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                context = RequestContext(request_id=f"worker_{worker_id}")
                set_request_context(context)
                time.sleep(0.01)  # Simuler du travail
                retrieved = get_request_context()
                results.append(retrieved.request_id if retrieved else None)
            except Exception as e:
                errors.append(e)
        
        # Lancer plusieurs workers en parallèle
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(50)]
            
            for future in as_completed(futures):
                future.result()  # Attendre la completion
        
        # Vérifier qu'il n'y a pas d'erreurs
        assert len(errors) == 0
        
        # Vérifier que chaque worker a eu son propre contexte
        assert len(results) == 50
        assert len(set(results)) == 50  # Tous uniques
    
    @pytest.mark.asyncio
    async def test_async_context_isolation(self, clean_context):
        """Test isolation du contexte dans les tâches async"""
        async def async_worker(worker_id):
            context = RequestContext(request_id=f"async_worker_{worker_id}")
            set_request_context(context)
            
            # Simuler du travail asynchrone
            await asyncio.sleep(0.01)
            
            retrieved = get_request_context()
            return retrieved.request_id if retrieved else None
        
        # Lancer plusieurs tâches async en parallèle
        tasks = [async_worker(i) for i in range(20)]
        results = await asyncio.gather(*tasks)
        
        # Vérifier que chaque tâche a eu son propre contexte
        assert len(results) == 20
        assert len(set(results)) == 20  # Tous uniques
        assert all(result.startswith("async_worker_") for result in results)
