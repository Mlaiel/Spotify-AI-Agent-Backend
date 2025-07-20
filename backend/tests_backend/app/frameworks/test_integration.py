"""
🧪 Tests d'Intégration Frameworks - Enterprise Integration Tests
=============================================================

Tests d'intégration cross-framework avec:
- Orchestrateur + tous les frameworks
- Workflows end-to-end
- Performance intégrée
- Resilience testing
- Business scenarios

Développé par: Toute l'équipe d'experts
"""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List
from datetime import datetime, timedelta

from backend.app.frameworks import (
    setup_all_frameworks,
    framework_orchestrator,
    FrameworkOrchestrator,
    FrameworkStatus
)
from backend.app.frameworks.core import BaseFramework, HealthStatus
from backend.app.frameworks.hybrid_backend import HybridBackend, HybridConfig
from backend.app.frameworks.ml_frameworks import MLModelManager, ModelConfig, ModelType
from backend.app.frameworks.security import SecurityFramework, SecurityConfig
from backend.app.frameworks.monitoring import MonitoringFramework, MonitoringConfig
from backend.app.frameworks.microservices import MicroservicesFramework, ServiceConfig, ServiceType
from backend.app.frameworks import TEST_CONFIG, clean_frameworks, logger


@pytest.fixture
def integration_config():
    """Configuration complète pour tests d'intégration."""
    return {
        "hybrid": HybridConfig(
            database_url=TEST_CONFIG["test_database_url"],
            fastapi_title="Integration Test API",
            enable_cors=True,
            redis_url=TEST_CONFIG["test_redis_url"]
        ),
        "security": SecurityConfig(
            jwt_secret_key=TEST_CONFIG["test_jwt_secret"],
            redis_url=TEST_CONFIG["test_redis_url"],
            rate_limit_requests=1000,
            enable_audit_logging=True
        ),
        "monitoring": MonitoringConfig(
            enable_prometheus=True,
            prometheus_port=TEST_CONFIG["test_metrics_port"],
            enable_tracing=True,
            enable_alerting=True
        ),
        "ml": {
            "recommendation_model": ModelConfig(
                name="spotify_recommendation",
                model_type=ModelType.RECOMMENDATION,
                version="1.0.0"
            )
        }
    }


@pytest.fixture
def mock_all_external_dependencies():
    """Mock toutes les dépendances externes."""
    with patch('redis.asyncio.from_url') as mock_redis, \
         patch('django.setup'), \
         patch('django.core.management.call_command'), \
         patch('consul.Consul'), \
         patch('aio_pika.connect_robust'), \
         patch('prometheus_client.start_http_server'):
        
        # Configuration Redis mock
        redis_client = AsyncMock()
        redis_client.get.return_value = None
        redis_client.set.return_value = True
        redis_client.incr.return_value = 1
        mock_redis.return_value = redis_client
        
        yield {
            'redis': redis_client,
            'django': True,
            'consul': True,
            'rabbitmq': True,
            'prometheus': True
        }


@pytest.mark.integration
class TestFrameworkOrchestrationIntegration:
    """Tests d'intégration orchestration frameworks."""
    
    @pytest.mark.asyncio
    async def test_setup_all_frameworks_success(self, integration_config, mock_all_external_dependencies, clean_frameworks):
        """Test setup complet de tous les frameworks."""
        # Mock configuration
        with patch('backend.app.frameworks.hybrid_backend.HybridConfig') as mock_hybrid_config, \
             patch('backend.app.frameworks.security.SecurityConfig') as mock_security_config, \
             patch('backend.app.frameworks.monitoring.MonitoringConfig') as mock_monitoring_config:
            
            mock_hybrid_config.return_value = integration_config["hybrid"]
            mock_security_config.return_value = integration_config["security"]
            mock_monitoring_config.return_value = integration_config["monitoring"]
            
            # Setup tous les frameworks
            result = await setup_all_frameworks()
            
        assert result["status"] == "success"
        assert len(result["frameworks"]) >= 5  # Au moins 5 frameworks
        assert "hybrid_backend" in result["frameworks"]
        assert "security" in result["frameworks"]
        assert "monitoring" in result["frameworks"]
        assert "ml_frameworks" in result["frameworks"]
        assert "microservices" in result["frameworks"]
        
    @pytest.mark.asyncio
    async def test_framework_dependencies_resolution(self, mock_all_external_dependencies, clean_frameworks):
        """Test résolution dépendances entre frameworks."""
        orchestrator = FrameworkOrchestrator()
        
        # Créer frameworks avec dépendances
        class MockSecurityFramework(BaseFramework):
            def __init__(self):
                super().__init__()
                self.dependencies = []  # Pas de dépendance
                
            async def initialize(self):
                self.status = FrameworkStatus.RUNNING
                return True
                
            async def shutdown(self):
                self.status = FrameworkStatus.STOPPED
                return True
                
            async def health_check(self):
                return HealthStatus(FrameworkStatus.RUNNING, "Security OK", {})
        
        class MockMLFramework(BaseFramework):
            def __init__(self):
                super().__init__()
                self.dependencies = ["security"]  # Dépend de security
                
            async def initialize(self):
                self.status = FrameworkStatus.RUNNING
                return True
                
            async def shutdown(self):
                self.status = FrameworkStatus.STOPPED
                return True
                
            async def health_check(self):
                return HealthStatus(FrameworkStatus.RUNNING, "ML OK", {})
        
        # Enregistrer frameworks
        orchestrator.register_framework("security", MockSecurityFramework())
        orchestrator.register_framework("ml", MockMLFramework())
        
        # Initialiser - doit respecter l'ordre des dépendances
        result = await orchestrator.initialize_all()
        
        assert result["status"] == "success"
        assert "security" in result["frameworks"]
        assert "ml" in result["frameworks"]
        
    @pytest.mark.asyncio
    async def test_framework_health_monitoring_integration(self, mock_all_external_dependencies, clean_frameworks):
        """Test monitoring santé intégré."""
        orchestrator = FrameworkOrchestrator()
        
        # Framework sain
        class HealthyFramework(BaseFramework):
            async def initialize(self):
                self.status = FrameworkStatus.RUNNING
                return True
                
            async def shutdown(self):
                self.status = FrameworkStatus.STOPPED
                return True
                
            async def health_check(self):
                return HealthStatus(
                    FrameworkStatus.RUNNING, 
                    "Healthy framework",
                    {"uptime": 120, "memory_usage": "50MB"}
                )
        
        # Framework avec problème
        class UnhealthyFramework(BaseFramework):
            async def initialize(self):
                self.status = FrameworkStatus.RUNNING
                return True
                
            async def shutdown(self):
                self.status = FrameworkStatus.STOPPED
                return True
                
            async def health_check(self):
                return HealthStatus(
                    FrameworkStatus.ERROR, 
                    "Connection failed",
                    {"error": "Database timeout", "last_error": datetime.utcnow().isoformat()}
                )
        
        orchestrator.register_framework("healthy", HealthyFramework())
        orchestrator.register_framework("unhealthy", UnhealthyFramework())
        
        await orchestrator.initialize_all()
        
        # Vérifier santé globale
        health_status = await orchestrator.get_health_status()
        
        assert "healthy" in health_status
        assert "unhealthy" in health_status
        assert health_status["healthy"].status == FrameworkStatus.RUNNING
        assert health_status["unhealthy"].status == FrameworkStatus.ERROR
        
        # Métriques globales
        metrics = orchestrator.get_metrics()
        assert len(metrics) == 2


@pytest.mark.integration
class TestEndToEndWorkflows:
    """Tests workflows end-to-end complets."""
    
    @pytest.mark.asyncio
    async def test_user_authentication_workflow(self, integration_config, mock_all_external_dependencies, clean_frameworks):
        """Test workflow d'authentification utilisateur complet."""
        # Initialiser frameworks nécessaires
        security_framework = SecurityFramework(integration_config["security"])
        monitoring_framework = MonitoringFramework(integration_config["monitoring"])
        
        await security_framework.initialize()
        await monitoring_framework.initialize()
        
        # 1. Tentative de login
        with patch.object(security_framework.crypto_manager, 'verify_password', return_value=True):
            login_result = await security_framework.login_user(
                username="testuser",
                password="correct_password",
                ip_address="192.168.1.100",
                user_agent="TestClient/1.0"
            )
            
        assert login_result["success"] is True
        access_token = login_result["access_token"]
        
        # 2. Enregistrer métrique de login dans monitoring
        await monitoring_framework.record_http_request(
            "POST", "/auth/login", 200, 0.25
        )
        
        # 3. Utiliser token pour authentification
        with patch.object(security_framework.jwt_manager, 'validate_token') as mock_validate:
            mock_validate.return_value = {"sub": "user_123", "username": "testuser"}
            
            authenticated_user = await security_framework.authenticate_user(access_token)
            
        assert authenticated_user["user_id"] == "user_123"
        
        # 4. Vérifier événements d'audit
        audit_events = security_framework.audit_logger.events
        assert len(audit_events) > 0
        assert any(event.event_type == "login_success" for event in audit_events)
        
        # 5. Vérifier métriques monitoring
        metrics = monitoring_framework.get_metrics()
        assert "http_requests_total" in metrics
        
    @pytest.mark.asyncio
    async def test_ai_recommendation_workflow(self, integration_config, mock_all_external_dependencies, clean_frameworks):
        """Test workflow recommandation IA complet."""
        # Initialiser frameworks
        ml_manager = MLModelManager()
        monitoring_framework = MonitoringFramework(integration_config["monitoring"])
        microservices_framework = MicroservicesFramework()
        
        await ml_manager.initialize()
        await monitoring_framework.initialize()
        await microservices_framework.initialize()
        
        # 1. Créer et enregistrer modèle de recommandation
        from backend.app.frameworks.ml_frameworks import SpotifyRecommendationModel
        
        model_config = integration_config["ml"]["recommendation_model"]
        recommendation_model = SpotifyRecommendationModel(model_config)
        
        await ml_manager.register_model(model_config, recommendation_model)
        
        # 2. Mock entraînement modèle
        with patch.object(recommendation_model, '_train_collaborative_model', return_value=0.87), \
             patch.object(recommendation_model, '_train_content_model', return_value=0.82):
            
            train_data = {
                'user_features': [[0.1, 0.2, 0.3] for _ in range(100)],
                'item_features': [[0.4, 0.5] for _ in range(200)],
                'interactions': [[1, 0, 1] for _ in range(100)]
            }
            
            metrics = await ml_manager.train_model(model_config.name, train_data, train_data)
            assert metrics.accuracy > 0.8
            
        # 3. Démarrer trace pour prédiction
        trace_span = await monitoring_framework.start_trace(
            "spotify_recommendation_prediction",
            tags={"user_id": "user_123", "model_version": "1.0.0"}
        )
        
        # 4. Faire prédiction
        with patch.object(recommendation_model, '_collaborative_predict', return_value=[0.9, 0.8, 0.7]), \
             patch.object(recommendation_model, '_content_predict', return_value=[0.85, 0.75, 0.65]):
            
            prediction_input = {
                'user_id': 'user_123',
                'candidate_items': ['track_1', 'track_2', 'track_3'],
                'user_features': [0.1, 0.2, 0.3],
                'item_features': [[0.4, 0.5], [0.6, 0.7], [0.8, 0.9]]
            }
            
            prediction_result = await ml_manager.predict(model_config.name, prediction_input)
            
        assert len(prediction_result.recommendations) == 3
        assert prediction_result.confidence > 0.7
        
        # 5. Enregistrer métriques IA
        await monitoring_framework.record_ai_prediction(
            model_config.name,
            "recommendation",
            0.18,  # Durée
            prediction_result.confidence
        )
        
        # 6. Finaliser trace
        finished_span = await monitoring_framework.finish_trace(trace_span.span_id)
        assert finished_span.duration > 0
        
        # 7. Publier événement via microservices
        await microservices_framework.publish_event(
            "recommendation_generated",
            {
                "user_id": "user_123",
                "recommendations": prediction_result.recommendations,
                "model_version": model_config.version
            }
        )
        
        # Vérifications finales
        ml_metrics = monitoring_framework.get_metrics()
        assert "ai_predictions_total" in ml_metrics
        
    @pytest.mark.asyncio
    async def test_microservices_communication_workflow(self, mock_all_external_dependencies, clean_frameworks):
        """Test workflow communication microservices."""
        # Initialiser frameworks
        microservices_framework = MicroservicesFramework()
        security_framework = SecurityFramework(SecurityConfig(
            jwt_secret_key=TEST_CONFIG["test_jwt_secret"],
            redis_url=TEST_CONFIG["test_redis_url"]
        ))
        monitoring_framework = MonitoringFramework(MonitoringConfig(
            prometheus_port=TEST_CONFIG["test_metrics_port"]
        ))
        
        await microservices_framework.initialize()
        await security_framework.initialize()
        await monitoring_framework.initialize()
        
        # 1. Enregistrer services
        user_service = ServiceConfig("user-service", ServiceType.WEB_API, "localhost", 8001)
        playlist_service = ServiceConfig("playlist-service", ServiceType.WEB_API, "localhost", 8002)
        
        with patch.object(microservices_framework.service_registry, 'register_service') as mock_register:
            mock_register.side_effect = ["user-service-id", "playlist-service-id"]
            
            user_service_id = await microservices_framework.register_service(user_service)
            playlist_service_id = await microservices_framework.register_service(playlist_service)
            
        # 2. Appel inter-services avec authentification
        jwt_token = security_framework.jwt_manager.create_access_token("user_123")
        
        with patch('httpx.AsyncClient.request') as mock_request:
            # Mock réponse user service
            mock_user_response = Mock()
            mock_user_response.status_code = 200
            mock_user_response.json.return_value = {
                "user_id": "user_123",
                "preferences": ["rock", "electronic"]
            }
            
            # Mock réponse playlist service
            mock_playlist_response = Mock()
            mock_playlist_response.status_code = 201
            mock_playlist_response.json.return_value = {
                "playlist_id": "playlist_456",
                "status": "created"
            }
            
            mock_request.side_effect = [mock_user_response, mock_playlist_response]
            
            # Appel 1: Récupérer préférences utilisateur
            user_data = await microservices_framework.call_service(
                "user-service",
                "/api/users/user_123/preferences",
                "GET",
                headers={"Authorization": f"Bearer {jwt_token}"}
            )
            
            # Appel 2: Créer playlist basée sur préférences
            playlist_data = await microservices_framework.call_service(
                "playlist-service",
                "/api/playlists",
                "POST",
                json={
                    "user_id": "user_123",
                    "preferences": user_data.json()["preferences"],
                    "name": "My Rock & Electronic Mix"
                },
                headers={"Authorization": f"Bearer {jwt_token}"}
            )
            
        # 3. Enregistrer métriques
        await monitoring_framework.record_http_request("GET", "/api/users/user_123/preferences", 200, 0.15)
        await monitoring_framework.record_http_request("POST", "/api/playlists", 201, 0.25)
        
        # 4. Publier événement
        await microservices_framework.publish_event(
            "playlist_created",
            {
                "user_id": "user_123",
                "playlist_id": playlist_data.json()["playlist_id"],
                "preferences": user_data.json()["preferences"]
            }
        )
        
        # Vérifications
        assert user_data.json()["user_id"] == "user_123"
        assert playlist_data.json()["playlist_id"] == "playlist_456"
        
        metrics = monitoring_framework.get_metrics()
        assert "http_requests_total" in metrics


@pytest.mark.integration
class TestResilienceAndFailover:
    """Tests de résilience et failover."""
    
    @pytest.mark.asyncio
    async def test_framework_failure_isolation(self, mock_all_external_dependencies, clean_frameworks):
        """Test isolation des échecs de frameworks."""
        orchestrator = FrameworkOrchestrator()
        
        # Framework qui fonctionne
        class StableFramework(BaseFramework):
            async def initialize(self):
                self.status = FrameworkStatus.RUNNING
                return True
                
            async def shutdown(self):
                self.status = FrameworkStatus.STOPPED
                return True
                
            async def health_check(self):
                return HealthStatus(FrameworkStatus.RUNNING, "Stable", {})
        
        # Framework qui échoue
        class FailingFramework(BaseFramework):
            async def initialize(self):
                raise Exception("Simulated initialization failure")
                
            async def shutdown(self):
                self.status = FrameworkStatus.STOPPED
                return True
                
            async def health_check(self):
                return HealthStatus(FrameworkStatus.ERROR, "Failed", {})
        
        orchestrator.register_framework("stable", StableFramework())
        orchestrator.register_framework("failing", FailingFramework())
        
        # Initialisation - un framework échoue, l'autre continue
        result = await orchestrator.initialize_all()
        
        assert result["status"] == "partial_success"
        assert "stable" in result["frameworks"]
        assert "failing" in result["failed"]
        assert result["frameworks"]["stable"] == "initialized"
        
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, mock_all_external_dependencies, clean_frameworks):
        """Test intégration circuit breaker."""
        microservices_framework = MicroservicesFramework()
        await microservices_framework.initialize()
        
        # Service instable
        unstable_service = ServiceConfig("unstable-service", ServiceType.WEB_API, "localhost", 8999)
        
        with patch.object(microservices_framework.service_registry, 'register_service', return_value="unstable-id"):
            await microservices_framework.register_service(unstable_service)
            
        # Configuration circuit breaker
        from backend.app.frameworks.microservices import CircuitBreakerConfig
        circuit_config = CircuitBreakerConfig(
            failure_threshold=3,
            timeout_duration=10,
            max_requests=5
        )
        
        await microservices_framework.service_mesh.configure_circuit_breaker(
            "unstable-service", 
            circuit_config
        )
        
        # Simuler échecs successifs
        with patch('httpx.AsyncClient.request') as mock_request:
            # 3 premiers appels échouent
            mock_request.side_effect = [
                Exception("Connection timeout"),
                Exception("Service unavailable"),
                Exception("Internal error"),
                # Circuit breaker devrait s'ouvrir ici
            ]
            
            # Tenter appels
            failures = 0
            for i in range(5):
                try:
                    await microservices_framework.call_service(
                        "unstable-service",
                        "/api/test",
                        "GET"
                    )
                except Exception:
                    failures += 1
                    
            # Au moins 3 échecs avant ouverture circuit breaker
            assert failures >= 3
            
    @pytest.mark.asyncio
    async def test_graceful_degradation(self, mock_all_external_dependencies, clean_frameworks):
        """Test dégradation gracieuse des services."""
        # Initialiser frameworks avec un qui échoue partiellement
        orchestrator = FrameworkOrchestrator()
        
        class DegradedFramework(BaseFramework):
            def __init__(self):
                super().__init__()
                self.degraded_mode = False
                
            async def initialize(self):
                # Simuler échec d'une fonctionnalité non critique
                try:
                    # Fonctionnalité critique réussit
                    self.status = FrameworkStatus.RUNNING
                    return True
                except Exception:
                    # Mode dégradé
                    self.degraded_mode = True
                    self.status = FrameworkStatus.RUNNING  # Toujours opérationnel
                    return True
                    
            async def shutdown(self):
                self.status = FrameworkStatus.STOPPED
                return True
                
            async def health_check(self):
                if self.degraded_mode:
                    return HealthStatus(
                        FrameworkStatus.RUNNING,
                        "Running in degraded mode",
                        {"degraded_features": ["advanced_analytics"]}
                    )
                return HealthStatus(FrameworkStatus.RUNNING, "Fully operational", {})
        
        orchestrator.register_framework("degraded", DegradedFramework())
        
        result = await orchestrator.initialize_all()
        assert result["status"] == "success"
        
        # Health check révèle mode dégradé
        health = await orchestrator.get_health_status()
        assert "degraded_features" in health["degraded"].details


@pytest.mark.integration
@pytest.mark.performance
class TestIntegratedPerformance:
    """Tests de performance intégrée."""
    
    @pytest.mark.asyncio
    async def test_concurrent_framework_operations(self, mock_all_external_dependencies, clean_frameworks):
        """Test opérations concurrentes sur tous les frameworks."""
        # Initialiser tous les frameworks
        orchestrator = FrameworkOrchestrator()
        
        # Mock frameworks légers pour tests de performance
        class FastFramework(BaseFramework):
            def __init__(self, name):
                super().__init__()
                self.name = name
                
            async def initialize(self):
                await asyncio.sleep(0.01)  # Simulation travail
                self.status = FrameworkStatus.RUNNING
                return True
                
            async def shutdown(self):
                self.status = FrameworkStatus.STOPPED
                return True
                
            async def health_check(self):
                await asyncio.sleep(0.005)  # Health check rapide
                return HealthStatus(FrameworkStatus.RUNNING, f"{self.name} OK", {})
        
        # Enregistrer plusieurs frameworks
        for i in range(10):
            orchestrator.register_framework(f"framework_{i}", FastFramework(f"framework_{i}"))
            
        # Test initialisation concurrente
        start_time = time.time()
        result = await orchestrator.initialize_all()
        init_duration = time.time() - start_time
        
        assert result["status"] == "success"
        assert len(result["frameworks"]) == 10
        assert init_duration < 2.0  # Initialisation en moins de 2 secondes
        
        # Test health checks concurrents
        start_time = time.time()
        health_status = await orchestrator.get_health_status()
        health_duration = time.time() - start_time
        
        assert len(health_status) == 10
        assert health_duration < 1.0  # Health checks en moins de 1 seconde
        
        # Test arrêt concurrent
        start_time = time.time()
        shutdown_result = await orchestrator.shutdown_all()
        shutdown_duration = time.time() - start_time
        
        assert shutdown_result["status"] == "success"
        assert shutdown_duration < 1.0  # Arrêt en moins de 1 seconde
        
    @pytest.mark.asyncio
    async def test_high_load_integration_scenario(self, mock_all_external_dependencies, clean_frameworks):
        """Test scénario intégration haute charge."""
        # Initialiser frameworks nécessaires
        monitoring_framework = MonitoringFramework(MonitoringConfig(
            prometheus_port=TEST_CONFIG["test_metrics_port"]
        ))
        security_framework = SecurityFramework(SecurityConfig(
            jwt_secret_key=TEST_CONFIG["test_jwt_secret"],
            redis_url=TEST_CONFIG["test_redis_url"],
            rate_limit_requests=10000  # Limite élevée pour test
        ))
        
        await monitoring_framework.initialize()
        await security_framework.initialize()
        
        # Simuler charge élevée
        async def simulate_user_request():
            # 1. Authentification
            token = security_framework.jwt_manager.create_access_token("load_test_user")
            
            # 2. Vérification rate limit
            allowed = await security_framework.check_rate_limit("load_test_user")
            
            if allowed:
                # 3. Enregistrer métrique
                await monitoring_framework.record_http_request("GET", "/api/test", 200, 0.01)
                return True
            else:
                await monitoring_framework.record_http_request("GET", "/api/test", 429, 0.001)
                return False
                
        # Lancer 1000 requêtes concurrentes
        start_time = time.time()
        tasks = [simulate_user_request() for _ in range(1000)]
        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time
        
        successful_requests = sum(1 for r in results if r)
        
        # Vérifications performance
        assert duration < 5.0  # Moins de 5 secondes pour 1000 requêtes
        assert successful_requests > 950  # Au moins 95% de réussite
        
        # Vérifier métriques collectées
        metrics = monitoring_framework.get_metrics()
        total_requests = metrics.get("http_requests_total", {})
        assert sum(total_requests.values()) == 1000


@pytest.mark.integration
class TestBusinessScenarios:
    """Tests de scénarios business complets."""
    
    @pytest.mark.asyncio
    async def test_spotify_playlist_creation_scenario(self, integration_config, mock_all_external_dependencies, clean_frameworks):
        """Test scénario complet création playlist Spotify."""
        # Initialiser stack complet
        security_framework = SecurityFramework(integration_config["security"])
        ml_manager = MLModelManager()
        monitoring_framework = MonitoringFramework(integration_config["monitoring"])
        microservices_framework = MicroservicesFramework()
        
        await security_framework.initialize()
        await ml_manager.initialize()
        await monitoring_framework.initialize()
        await microservices_framework.initialize()
        
        # Scénario: Utilisateur crée une playlist basée sur ses goûts
        
        # 1. Authentification utilisateur
        with patch.object(security_framework.crypto_manager, 'verify_password', return_value=True):
            login_result = await security_framework.login_user(
                username="music_lover",
                password="password123",
                ip_address="192.168.1.50"
            )
            
        user_token = login_result["access_token"]
        
        # 2. Analyse des préférences via ML
        from backend.app.frameworks.ml_frameworks import SpotifyRecommendationModel, ModelConfig, ModelType
        
        model_config = ModelConfig(
            name="user_preference_analyzer",
            model_type=ModelType.RECOMMENDATION
        )
        
        preference_model = SpotifyRecommendationModel(model_config)
        await ml_manager.register_model(model_config, preference_model)
        
        # Mock analyse préférences
        with patch.object(preference_model, '_collaborative_predict', return_value=[0.95, 0.89, 0.87, 0.85, 0.82]), \
             patch.object(preference_model, '_content_predict', return_value=[0.92, 0.88, 0.84, 0.81, 0.79]):
            
            user_preferences = await ml_manager.predict(
                "user_preference_analyzer",
                {
                    "user_id": "music_lover",
                    "listening_history": ["rock", "electronic", "indie"],
                    "candidate_items": ["track_1", "track_2", "track_3", "track_4", "track_5"]
                }
            )
            
        # 3. Enregistrement métriques ML
        await monitoring_framework.record_ai_prediction(
            "user_preference_analyzer",
            "preference_analysis",
            0.25,
            user_preferences.confidence
        )
        
        # 4. Appel service playlist
        with patch('httpx.AsyncClient.request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 201
            mock_response.json.return_value = {
                "playlist_id": "spotify_playlist_789",
                "name": "AI Generated Mix",
                "tracks": user_preferences.recommendations[:10],
                "created_at": datetime.utcnow().isoformat()
            }
            mock_request.return_value = mock_response
            
            playlist_response = await microservices_framework.call_service(
                "spotify-playlist-service",
                "/api/playlists",
                "POST",
                json={
                    "user_id": "music_lover",
                    "recommendations": user_preferences.recommendations,
                    "playlist_name": "My AI Mix"
                },
                headers={"Authorization": f"Bearer {user_token}"}
            )
            
        # 5. Publier événement de création
        await microservices_framework.publish_event(
            "playlist_created",
            {
                "user_id": "music_lover",
                "playlist_id": playlist_response.json()["playlist_id"],
                "ai_generated": True,
                "confidence_score": user_preferences.confidence
            }
        )
        
        # 6. Audit de sécurité
        audit_events = security_framework.audit_logger.events
        login_events = [e for e in audit_events if e.event_type == "login_success"]
        assert len(login_events) > 0
        
        # Vérifications finales
        assert playlist_response.json()["playlist_id"] == "spotify_playlist_789"
        assert len(user_preferences.recommendations) >= 5
        assert user_preferences.confidence > 0.8
        
        # Métriques business
        metrics = monitoring_framework.get_metrics()
        assert "ai_predictions_total" in metrics
        assert "http_requests_total" in metrics
        
    @pytest.mark.asyncio
    async def test_spotify_discovery_and_analytics_scenario(self, mock_all_external_dependencies, clean_frameworks):
        """Test scénario découverte musicale et analytics."""
        # Configuration légère pour test rapide
        monitoring_config = MonitoringConfig(
            prometheus_port=TEST_CONFIG["test_metrics_port"],
            enable_alerting=False  # Désactiver alertes pour test
        )
        
        monitoring_framework = MonitoringFramework(monitoring_config)
        ml_manager = MLModelManager()
        
        await monitoring_framework.initialize()
        await ml_manager.initialize()
        
        # Scénario: Analyser patterns d'écoute et générer insights
        
        # 1. Collecte données d'écoute (simulée)
        listening_data = [
            {"user_id": f"user_{i}", "track_id": f"track_{i%20}", "duration": 180 + (i%60)}
            for i in range(100)
        ]
        
        # 2. Analyse patterns ML
        from backend.app.frameworks.ml_frameworks import AudioAnalysisModel, ModelConfig, ModelType
        
        audio_model_config = ModelConfig(
            name="audio_pattern_analyzer",
            model_type=ModelType.CLASSIFICATION
        )
        
        audio_model = AudioAnalysisModel(audio_model_config)
        await ml_manager.register_model(audio_model_config, audio_model)
        
        # 3. Traitement batch des données
        start_time = time.time()
        
        analysis_results = []
        for data in listening_data[:10]:  # Échantillon pour test
            # Mock analyse audio
            with patch.object(audio_model, 'predict') as mock_predict:
                mock_predict.return_value = type('PredictionResult', (), {
                    'predictions': {'genre': 'electronic', 'emotion': 'energetic'},
                    'confidence': 0.88
                })()
                
                result = await ml_manager.predict(
                    "audio_pattern_analyzer",
                    {"track_id": data["track_id"], "user_context": data}
                )
                analysis_results.append(result)
                
            # Enregistrer métrique
            await monitoring_framework.record_ai_prediction(
                "audio_pattern_analyzer",
                "pattern_analysis",
                0.05,
                result.confidence
            )
            
        processing_time = time.time() - start_time
        
        # 4. Génération insights
        genres_detected = [r.predictions['genre'] for r in analysis_results]
        emotions_detected = [r.predictions['emotion'] for r in analysis_results]
        
        insights = {
            "dominant_genre": max(set(genres_detected), key=genres_detected.count),
            "emotional_trend": max(set(emotions_detected), key=emotions_detected.count),
            "processing_time": processing_time,
            "confidence_avg": sum(r.confidence for r in analysis_results) / len(analysis_results)
        }
        
        # Vérifications
        assert len(analysis_results) == 10
        assert processing_time < 2.0  # Traitement rapide
        assert insights["confidence_avg"] > 0.8
        assert insights["dominant_genre"] in ["electronic", "rock", "pop"]  # Genres valides
        
        # Métriques finales
        final_metrics = monitoring_framework.get_metrics()
        ai_predictions = final_metrics.get("ai_predictions_total", {})
        assert sum(ai_predictions.values()) == 10  # 10 prédictions enregistrées
