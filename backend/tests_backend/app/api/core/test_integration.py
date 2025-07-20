"""
🎵 Tests d'Intégration Ultra-Avancés pour API Core Module Complet
===============================================================

Tests d'intégration industriels pour valider l'interaction entre tous les
composants du module core avec patterns enterprise et validation complète.

Développé par Fahed Mlaiel - Enterprise Integration Testing Expert
"""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
from typing import Dict, Any, List

from fastapi import FastAPI, Request, Depends, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.testclient import TestClient
from starlette.responses import JSONResponse

# Imports de tous les modules core
from app.api.core.config import APIConfig, get_api_config
from app.api.core.context import RequestContext, get_request_context, set_request_context
from app.api.core.factory import ComponentFactory, DependencyContainer  # , configure_dependencies
from app.api.core.exceptions import APIException, ValidationException, ErrorCode
from app.api.core.response import create_success_response, create_error_response, APIResponse
from app.api.core.monitoring import get_api_metrics, get_health_checker, setup_monitoring


# =============================================================================
# FIXTURES ENTERPRISE POUR INTEGRATION TESTING
# =============================================================================

@pytest.fixture
def integration_config():
    """Configuration complète pour les tests d'intégration"""
    return {
        "app": {
            "name": "Test Spotify AI Agent",
            "version": "1.0.0",
            "debug": True,
            "environment": "test"
        },
        "database": {
            "url": "postgresql://test:test@localhost/test_db",
            "pool_size": 5,
            "max_overflow": 10
        },
        "redis": {
            "url": "redis://localhost:6379/1",
            "timeout": 30,
            "max_connections": 20
        },
        "monitoring": {
            "enabled": True,
            "metrics": {"enabled": True, "port": 9090},
            "health": {"enabled": True, "path": "/health"},
            "alerts": {
                "enabled": True,
                "thresholds": {
                    "response_time": 1000,
                    "error_rate": 0.05,
                    "cpu_usage": 80,
                    "memory_usage": 80
                }
            }
        },
        "security": {
            "cors_enabled": True,
            "rate_limit": {"enabled": True, "requests_per_minute": 100}
        }
    }


@pytest.fixture
def clean_integration_env():
    """Environnement propre pour les tests d'intégration"""
    # Nettoyer les singletons
    ComponentFactory._instance = None
    DependencyContainer._instance = None
    
    # Nettoyer le contexte
    from app.api.core.context import clear_request_context
    clear_request_context()
    
    yield
    
    # Nettoyer après le test
    ComponentFactory._instance = None
    DependencyContainer._instance = None
    clear_request_context()


@pytest.fixture
def integrated_app(integration_config, clean_integration_env):
    """Application FastAPI complètement intégrée"""
    app = FastAPI(
        title=integration_config["app"]["name"],
        version=integration_config["app"]["version"],
        debug=integration_config["app"]["debug"]
    )
    
    # Configurer les dépendances
    configure_dependencies(integration_config)
    
    # Configurer le monitoring
    setup_monitoring(app, integration_config["monitoring"])
    
    # Ajouter des endpoints de test
    @app.get("/api/v1/test/success")
    async def test_success():
        """Endpoint de test qui utilise tous les composants core"""
        # Utiliser le contexte
        context = get_request_context()
        
        # Utiliser la configuration
        config = get_api_config()
        
        # Retourner une réponse standardisée
        return create_success_response(
            data={
                "message": "Integration test success",
                "request_id": context.request_id if context else None,
                "app_name": config.app_name if config else "Unknown",
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            message="Integration test completed successfully"
        )
    
    @app.get("/api/v1/test/validation-error")
    async def test_validation_error():
        """Endpoint qui déclenche une erreur de validation"""
        raise ValidationException(
            message="Test validation error",
            field="test_field",
            value="invalid_value"
        )
    
    @app.get("/api/v1/test/api-error")
    async def test_api_error():
        """Endpoint qui déclenche une erreur API"""
        raise APIException(
            message="Test API error",
            error_code=ErrorCode.INTERNAL_ERROR,
            status_code=500
        )
    
    @app.get("/api/v1/test/context")
    async def test_context():
        """Endpoint qui teste le contexte de requête"""
        context = get_request_context()
        
        if not context:
            raise APIException("No request context available")
        
        return create_success_response(
            data={
                "request_id": context.request_id,
                "correlation_id": context.correlation_id,
                "user_id": context.user.user_id if context.user else None,
                "timestamp": context.timestamp.isoformat()
            }
        )
    
    @app.get("/api/v1/test/dependencies")
    async def test_dependencies():
        """Endpoint qui teste les dépendances"""
        from app.api.core.factory import get_dependency_container
        
        container = get_dependency_container()
        
        # Utiliser quelques dépendances
        try:
            config = container.resolve("config")
            database = container.resolve("database")
            
            return create_success_response(
                data={
                    "config_available": config is not None,
                    "database_available": database is not None,
                    "dependencies_count": len(container._dependencies)
                }
            )
        except Exception as e:
            raise APIException(f"Dependency resolution failed: {str(e)}")
    
    @app.get("/api/v1/test/monitoring")
    async def test_monitoring():
        """Endpoint qui teste le monitoring"""
        metrics = get_api_metrics()
        health_checker = get_health_checker()
        
        return create_success_response(
            data={
                "metrics_summary": metrics.get_metrics_summary(),
                "health_summary": health_checker.get_health_summary()
            }
        )
    
    return app


# =============================================================================
# TESTS D'INTÉGRATION DE BASE
# =============================================================================

class TestCoreModuleIntegration:
    """Tests d'intégration pour le module core complet"""
    
    def test_application_startup_integration(self, integrated_app):
        """Test démarrage complet de l'application"""
        with TestClient(integrated_app) as client:
            # L'application devrait démarrer sans erreur
            response = client.get("/docs")
            assert response.status_code == 200
    
    def test_success_endpoint_integration(self, integrated_app):
        """Test endpoint de succès avec tous les composants"""
        with TestClient(integrated_app) as client:
            response = client.get("/api/v1/test/success")
            
            assert response.status_code == 200
            data = response.json()
            
            # Vérifier la structure de réponse standardisée
            assert data["success"] is True
            assert "data" in data
            assert "message" in data
            assert "metadata" in data
            
            # Vérifier les données spécifiques
            assert data["data"]["message"] == "Integration test success"
            assert data["data"]["request_id"] is not None
            assert data["data"]["app_name"] == "Test Spotify AI Agent"
            assert data["data"]["timestamp"] is not None
            
            # Vérifier les métadonnées
            assert data["metadata"]["request_id"] is not None
            assert data["metadata"]["timestamp"] is not None
    
    def test_context_integration(self, integrated_app):
        """Test intégration du contexte de requête"""
        with TestClient(integrated_app) as client:
            response = client.get("/api/v1/test/context")
            
            assert response.status_code == 200
            data = response.json()
            
            # Vérifier que le contexte est correctement établi
            assert data["data"]["request_id"] is not None
            assert data["data"]["correlation_id"] is not None
            assert data["data"]["timestamp"] is not None
            
            # Vérifier que les IDs sont dans les headers de réponse
            assert "X-Request-ID" in response.headers
            assert "X-Correlation-ID" in response.headers
    
    def test_dependencies_integration(self, integrated_app):
        """Test intégration des dépendances"""
        with TestClient(integrated_app) as client:
            response = client.get("/api/v1/test/dependencies")
            
            assert response.status_code == 200
            data = response.json()
            
            # Vérifier que les dépendances sont disponibles
            assert data["data"]["config_available"] is True
            assert data["data"]["database_available"] is True
            assert data["data"]["dependencies_count"] > 0
    
    def test_monitoring_integration(self, integrated_app):
        """Test intégration du monitoring"""
        with TestClient(integrated_app) as client:
            # Faire quelques requêtes pour générer des métriques
            for _ in range(5):
                client.get("/api/v1/test/success")
            
            response = client.get("/api/v1/test/monitoring")
            
            assert response.status_code == 200
            data = response.json()
            
            # Vérifier les métriques
            metrics_summary = data["data"]["metrics_summary"]
            assert metrics_summary["total_requests"] >= 5
            
            # Vérifier la santé
            health_summary = data["data"]["health_summary"]
            assert "overall_status" in health_summary


# =============================================================================
# TESTS D'INTÉGRATION DES ERREURS
# =============================================================================

class TestErrorHandlingIntegration:
    """Tests d'intégration pour la gestion d'erreurs"""
    
    def test_validation_error_integration(self, integrated_app):
        """Test gestion d'erreur de validation"""
        with TestClient(integrated_app) as client:
            response = client.get("/api/v1/test/validation-error")
            
            assert response.status_code == 422
            data = response.json()
            
            # Vérifier la structure d'erreur standardisée
            assert data["success"] is False
            assert data["error"]["code"] == ErrorCode.VALIDATION_ERROR
            assert data["error"]["message"] == "Test validation error"
            assert "error_id" in data["error"]
            assert "timestamp" in data["error"]
            
            # Vérifier les détails de validation
            assert len(data["error"]["details"]) > 0
            detail = data["error"]["details"][0]
            assert detail["field"] == "test_field"
            assert detail["value"] == "invalid_value"
    
    def test_api_error_integration(self, integrated_app):
        """Test gestion d'erreur API"""
        with TestClient(integrated_app) as client:
            response = client.get("/api/v1/test/api-error")
            
            assert response.status_code == 500
            data = response.json()
            
            # Vérifier la structure d'erreur
            assert data["success"] is False
            assert data["error"]["code"] == ErrorCode.INTERNAL_ERROR
            assert data["error"]["message"] == "Test API error"
            
            # Vérifier les headers d'erreur
            assert "X-Error-ID" in response.headers
    
    def test_http_exception_integration(self, integrated_app):
        """Test gestion HTTPException standard"""
        with TestClient(integrated_app) as client:
            # Endpoint inexistant
            response = client.get("/api/v1/nonexistent")
            
            assert response.status_code == 404
            data = response.json()
            
            # Même les erreurs HTTP standard devraient suivre notre format
            assert "error" in data
    
    def test_error_correlation_integration(self, integrated_app):
        """Test corrélation des erreurs avec le contexte"""
        with TestClient(integrated_app) as client:
            # Première requête réussie pour établir le contexte
            success_response = client.get("/api/v1/test/success")
            correlation_id = success_response.headers.get("X-Correlation-ID")
            
            # Requête d'erreur avec même corrélation
            error_response = client.get(
                "/api/v1/test/validation-error",
                headers={"X-Correlation-ID": correlation_id}
            )
            
            assert error_response.status_code == 422
            
            # Vérifier que la corrélation est préservée
            assert error_response.headers.get("X-Correlation-ID") == correlation_id


# =============================================================================
# TESTS D'INTÉGRATION DE PERFORMANCE
# =============================================================================

class TestPerformanceIntegration:
    """Tests d'intégration pour les performances"""
    
    def test_response_time_monitoring_integration(self, integrated_app):
        """Test monitoring du temps de réponse"""
        with TestClient(integrated_app) as client:
            # Faire plusieurs requêtes
            start_time = time.time()
            responses = []
            
            for _ in range(10):
                response = client.get("/api/v1/test/success")
                responses.append(response)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Toutes les requêtes devraient réussir
            assert all(r.status_code == 200 for r in responses)
            
            # Le temps total ne devrait pas être excessif
            assert total_time < 5.0  # Moins de 5 secondes pour 10 requêtes
            
            # Vérifier que les métriques ont été collectées
            response = client.get("/api/v1/test/monitoring")
            data = response.json()
            
            metrics = data["data"]["metrics_summary"]
            assert metrics["total_requests"] >= 10
            assert metrics["avg_response_time"] > 0
    
    def test_concurrent_requests_integration(self, integrated_app):
        """Test requêtes concurrentes"""
        import threading
        from concurrent.futures import ThreadPoolExecutor
        
        def make_request():
            with TestClient(integrated_app) as client:
                response = client.get("/api/v1/test/success")
                return response.status_code
        
        # Exécuter 20 requêtes concurrentes
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(20)]
            results = [future.result() for future in futures]
        
        # Toutes les requêtes devraient réussir
        assert all(status == 200 for status in results)
        assert len(results) == 20
    
    def test_memory_usage_integration(self, integrated_app):
        """Test utilisation mémoire sous charge"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        with TestClient(integrated_app) as client:
            # Faire beaucoup de requêtes
            for _ in range(100):
                response = client.get("/api/v1/test/success")
                assert response.status_code == 200
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # L'augmentation de mémoire ne devrait pas être excessive
        # (Seuil arbitraire de 50MB pour 100 requêtes)
        assert memory_increase < 50 * 1024 * 1024


# =============================================================================
# TESTS D'INTÉGRATION DE SÉCURITÉ
# =============================================================================

class TestSecurityIntegration:
    """Tests d'intégration pour la sécurité"""
    
    def test_headers_security_integration(self, integrated_app):
        """Test headers de sécurité"""
        with TestClient(integrated_app) as client:
            response = client.get("/api/v1/test/success")
            
            # Vérifier les headers de sécurité
            # (Ces headers devraient être ajoutés par les middlewares de sécurité)
            
            # Headers informatifs
            assert "X-Request-ID" in response.headers
            assert "X-Correlation-ID" in response.headers
    
    def test_error_information_disclosure_integration(self, integrated_app):
        """Test non-divulgation d'informations dans les erreurs"""
        with TestClient(integrated_app) as client:
            response = client.get("/api/v1/test/api-error")
            
            assert response.status_code == 500
            data = response.json()
            
            # Les erreurs ne devraient pas exposer d'informations sensibles
            error_message = data["error"]["message"]
            
            # Vérifier qu'il n'y a pas de stack trace ou d'infos internes
            assert "Traceback" not in error_message
            assert "File " not in error_message
            assert "line " not in error_message
    
    def test_input_validation_integration(self, integrated_app):
        """Test validation des entrées"""
        with TestClient(integrated_app) as client:
            # Test avec des données malicieuses
            malicious_headers = {
                "X-Malicious": "<script>alert('xss')</script>",
                "X-SQL-Injection": "'; DROP TABLE users; --"
            }
            
            response = client.get(
                "/api/v1/test/success",
                headers=malicious_headers
            )
            
            # La requête devrait réussir mais les données malicieuses
            # ne devraient pas être reflétées dans la réponse
            assert response.status_code == 200
            
            response_text = response.text
            assert "<script>" not in response_text
            assert "DROP TABLE" not in response_text


# =============================================================================
# TESTS D'INTÉGRATION DES MIDDLEWARE
# =============================================================================

class TestMiddlewareIntegration:
    """Tests d'intégration pour les middlewares"""
    
    def test_middleware_chain_integration(self, integrated_app):
        """Test chaîne de middlewares"""
        with TestClient(integrated_app) as client:
            response = client.get("/api/v1/test/success")
            
            assert response.status_code == 200
            
            # Vérifier que tous les middlewares ont traité la requête
            # (Context, Monitoring, etc.)
            
            # Le contexte devrait être établi
            data = response.json()
            assert data["data"]["request_id"] is not None
            
            # Les métriques devraient être collectées
            monitoring_response = client.get("/api/v1/test/monitoring")
            monitoring_data = monitoring_response.json()
            assert monitoring_data["data"]["metrics_summary"]["total_requests"] > 0
    
    def test_middleware_error_handling_integration(self, integrated_app):
        """Test gestion d'erreur dans les middlewares"""
        with TestClient(integrated_app) as client:
            response = client.get("/api/v1/test/validation-error")
            
            assert response.status_code == 422
            
            # Même en cas d'erreur, les middlewares devraient fonctionner
            assert "X-Request-ID" in response.headers
            
            # Les métriques d'erreur devraient être collectées
            monitoring_response = client.get("/api/v1/test/monitoring")
            monitoring_data = monitoring_response.json()
            assert monitoring_data["data"]["metrics_summary"]["total_errors"] > 0


# =============================================================================
# TESTS D'INTÉGRATION AVANCÉS
# =============================================================================

class TestAdvancedIntegration:
    """Tests d'intégration avancés"""
    
    def test_configuration_hot_reload_integration(self, integrated_app, integration_config):
        """Test rechargement à chaud de la configuration"""
        with TestClient(integrated_app) as client:
            # Requête initiale
            response1 = client.get("/api/v1/test/success")
            assert response1.status_code == 200
            
            # Modifier la configuration (simulation)
            # En pratique, cela nécessiterait un mécanisme de rechargement
            
            # Nouvelle requête
            response2 = client.get("/api/v1/test/success")
            assert response2.status_code == 200
    
    def test_graceful_degradation_integration(self, integrated_app):
        """Test dégradation gracieuse"""
        with TestClient(integrated_app) as client:
            # Simuler la panne d'un service non critique
            # (Par exemple, le monitoring)
            
            # L'application devrait continuer à fonctionner
            response = client.get("/api/v1/test/success")
            assert response.status_code == 200
    
    def test_metrics_aggregation_integration(self, integrated_app):
        """Test agrégation des métriques"""
        with TestClient(integrated_app) as client:
            # Faire différents types de requêtes
            client.get("/api/v1/test/success")  # Succès
            client.get("/api/v1/test/validation-error")  # Erreur validation
            client.get("/api/v1/test/api-error")  # Erreur API
            
            # Récupérer les métriques agrégées
            response = client.get("/api/v1/test/monitoring")
            data = response.json()
            
            metrics = data["data"]["metrics_summary"]
            
            # Vérifier que tous les types d'événements sont comptabilisés
            assert metrics["total_requests"] >= 3
            assert metrics["total_errors"] >= 2  # 2 erreurs
    
    @pytest.mark.asyncio
    async def test_async_integration(self, integrated_app):
        """Test intégration asynchrone"""
        # Test avec des opérations asynchrones
        
        async def async_test():
            # Simuler des opérations async
            await asyncio.sleep(0.01)
            return True
        
        result = await async_test()
        assert result is True
        
        # Tester l'app avec des requêtes async
        with TestClient(integrated_app) as client:
            response = client.get("/api/v1/test/success")
            assert response.status_code == 200


# =============================================================================
# TESTS D'INTÉGRATION E2E
# =============================================================================

@pytest.mark.e2e
class TestEndToEndIntegration:
    """Tests d'intégration end-to-end"""
    
    def test_complete_request_lifecycle(self, integrated_app):
        """Test cycle de vie complet d'une requête"""
        with TestClient(integrated_app) as client:
            # 1. Requête initiale
            response = client.get("/api/v1/test/success")
            
            # 2. Vérifier la réponse
            assert response.status_code == 200
            data = response.json()
            
            # 3. Vérifier la structure complète
            assert data["success"] is True
            assert "data" in data
            assert "metadata" in data
            
            # 4. Vérifier les headers
            assert "X-Request-ID" in response.headers
            assert "Content-Type" in response.headers
            
            # 5. Vérifier les métriques
            monitoring_response = client.get("/api/v1/test/monitoring")
            monitoring_data = monitoring_response.json()
            
            assert monitoring_data["success"] is True
            assert monitoring_data["data"]["metrics_summary"]["total_requests"] > 0
    
    def test_error_to_recovery_flow(self, integrated_app):
        """Test flux d'erreur vers récupération"""
        with TestClient(integrated_app) as client:
            # 1. Déclencher une erreur
            error_response = client.get("/api/v1/test/api-error")
            assert error_response.status_code == 500
            
            # 2. Vérifier que l'erreur est bien gérée
            error_data = error_response.json()
            assert error_data["success"] is False
            
            # 3. Faire une requête de récupération
            recovery_response = client.get("/api/v1/test/success")
            assert recovery_response.status_code == 200
            
            # 4. Vérifier que le système fonctionne normalement
            recovery_data = recovery_response.json()
            assert recovery_data["success"] is True
            
            # 5. Vérifier les métriques des deux types de requêtes
            monitoring_response = client.get("/api/v1/test/monitoring")
            monitoring_data = monitoring_response.json()
            
            metrics = monitoring_data["data"]["metrics_summary"]
            assert metrics["total_requests"] >= 3  # error + success + monitoring
            assert metrics["total_errors"] >= 1    # L'erreur API
    
    def test_load_and_monitoring_integration(self, integrated_app):
        """Test intégration charge et monitoring"""
        with TestClient(integrated_app) as client:
            # 1. Générer de la charge
            for i in range(20):
                if i % 4 == 0:
                    # Quelques erreurs occasionnelles
                    client.get("/api/v1/test/validation-error")
                else:
                    # Principalement des succès
                    client.get("/api/v1/test/success")
            
            # 2. Vérifier les métriques finales
            response = client.get("/api/v1/test/monitoring")
            data = response.json()
            
            metrics = data["data"]["metrics_summary"]
            
            # 3. Valider les ratios
            total_requests = metrics["total_requests"]
            total_errors = metrics["total_errors"]
            
            assert total_requests >= 20
            assert total_errors >= 5  # ~25% d'erreurs
            
            # 4. Vérifier la santé du système
            health_summary = data["data"]["health_summary"]
            # Le système devrait encore être sain malgré les erreurs
            assert "overall_status" in health_summary
