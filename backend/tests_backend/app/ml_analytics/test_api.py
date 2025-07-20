"""
Tests pour l'API REST ML Analytics
==================================

Tests complets pour les endpoints REST avec couverture de:
- Authentification et autorisation
- Validation des données et sérialisation
- Endpoints de recommandation et analyse audio
- Gestion d'erreurs et codes de réponse
- Performance et rate limiting
- Sécurité et conformité
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import Dict, Any, List
import numpy as np

# FastAPI testing
from fastapi.testclient import TestClient
from fastapi import status
import httpx

from ml_analytics.api import app, get_current_user, verify_api_key
from ml_analytics.exceptions import MLAnalyticsError, AuthenticationError
from ml_analytics.models import SpotifyRecommendationModel
from ml_analytics.audio import AudioAnalysisModel


# Configuration de test pour l'API
TEST_API_CONFIG = {
    "testing": True,
    "jwt_secret": "test-secret-key",
    "rate_limit": 1000,
    "cors_origins": ["http://localhost:3000"],
    "api_version": "v1"
}


class TestAPIAuthentication:
    """Tests pour l'authentification de l'API."""
    
    @pytest.fixture
    def client(self):
        """Client de test FastAPI."""
        with patch('ml_analytics.api.get_ml_analytics_config') as mock_config:
            mock_config.return_value = TEST_API_CONFIG
            return TestClient(app)
    
    @pytest.fixture
    def valid_token(self):
        """Token JWT valide pour les tests."""
        from jose import jwt
        payload = {
            "user_id": "test_user_123",
            "permissions": ["recommendations", "audio_analysis"],
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        return jwt.encode(payload, TEST_API_CONFIG["jwt_secret"], algorithm="HS256")
    
    @pytest.fixture
    def expired_token(self):
        """Token JWT expiré pour les tests."""
        from jose import jwt
        payload = {
            "user_id": "test_user_123",
            "permissions": ["recommendations"],
            "exp": datetime.utcnow() - timedelta(hours=1)
        }
        return jwt.encode(payload, TEST_API_CONFIG["jwt_secret"], algorithm="HS256")
    
    def test_health_endpoint_no_auth(self, client):
        """Test l'endpoint de santé sans authentification."""
        response = client.get("/ml-analytics/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
    
    def test_protected_endpoint_no_token(self, client):
        """Test d'accès à un endpoint protégé sans token."""
        response = client.post("/ml-analytics/recommendations")
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "detail" in response.json()
    
    def test_protected_endpoint_invalid_token(self, client):
        """Test d'accès avec un token invalide."""
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.post("/ml-analytics/recommendations", headers=headers)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_protected_endpoint_expired_token(self, client, expired_token):
        """Test d'accès avec un token expiré."""
        headers = {"Authorization": f"Bearer {expired_token}"}
        response = client.post("/ml-analytics/recommendations", headers=headers)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_protected_endpoint_valid_token(self, client, valid_token):
        """Test d'accès avec un token valide."""
        headers = {"Authorization": f"Bearer {valid_token}"}
        
        with patch('ml_analytics.api.recommendation_service') as mock_service:
            mock_service.generate_recommendations = AsyncMock(return_value=[])
            
            response = client.post(
                "/ml-analytics/recommendations",
                headers=headers,
                json={"user_id": "test_user", "num_recommendations": 10}
            )
            
            # Devrait passer l'authentification mais peut échouer sur la logique métier
            assert response.status_code in [status.HTTP_200_OK, status.HTTP_422_UNPROCESSABLE_ENTITY]
    
    def test_permissions_validation(self, client):
        """Test la validation des permissions."""
        from jose import jwt
        
        # Token avec permissions limitées
        limited_payload = {
            "user_id": "limited_user",
            "permissions": ["audio_analysis"],  # Pas de "recommendations"
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        limited_token = jwt.encode(limited_payload, TEST_API_CONFIG["jwt_secret"], algorithm="HS256")
        
        headers = {"Authorization": f"Bearer {limited_token}"}
        response = client.post(
            "/ml-analytics/recommendations",
            headers=headers,
            json={"user_id": "test_user"}
        )
        
        # Devrait être refusé pour manque de permissions
        assert response.status_code == status.HTTP_403_FORBIDDEN


class TestRecommendationEndpoints:
    """Tests pour les endpoints de recommandation."""
    
    @pytest.fixture
    def authenticated_client(self, client, valid_token):
        """Client authentifié."""
        client.headers = {"Authorization": f"Bearer {valid_token}"}
        return client
    
    @pytest.mark.asyncio
    async def test_generate_recommendations_success(self, authenticated_client):
        """Test la génération de recommandations réussie."""
        request_data = {
            "user_id": "test_user_123",
            "num_recommendations": 10,
            "context": {
                "time_of_day": "evening",
                "mood": "relaxed"
            },
            "filters": {
                "genres": ["rock", "pop"],
                "min_popularity": 50
            }
        }
        
        mock_recommendations = [
            {
                "track_id": "spotify:track:test1",
                "track_name": "Test Song 1",
                "artist_name": "Test Artist 1",
                "score": 0.95,
                "confidence": 0.88,
                "explanation": "Based on your listening history"
            },
            {
                "track_id": "spotify:track:test2", 
                "track_name": "Test Song 2",
                "artist_name": "Test Artist 2",
                "score": 0.87,
                "confidence": 0.82,
                "explanation": "Similar to tracks you liked"
            }
        ]
        
        with patch('ml_analytics.api.recommendation_service.generate_recommendations') as mock_gen:
            mock_gen.return_value = mock_recommendations
            
            response = authenticated_client.post(
                "/ml-analytics/recommendations",
                json=request_data
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert "recommendations" in data
            assert "metadata" in data
            assert len(data["recommendations"]) == 2
            assert data["recommendations"][0]["score"] == 0.95
    
    def test_generate_recommendations_validation_error(self, authenticated_client):
        """Test la validation des données de recommandation."""
        # Données invalides
        invalid_data = {
            "user_id": "",  # Vide
            "num_recommendations": -1,  # Négatif
            "context": "invalid_context"  # Mauvais type
        }
        
        response = authenticated_client.post(
            "/ml-analytics/recommendations",
            json=invalid_data
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "detail" in response.json()
    
    @pytest.mark.asyncio
    async def test_get_user_recommendations_history(self, authenticated_client):
        """Test la récupération de l'historique des recommandations."""
        user_id = "test_user_123"
        
        mock_history = [
            {
                "timestamp": "2024-01-15T10:30:00Z",
                "request_id": "req_123",
                "num_recommendations": 10,
                "context": {"mood": "happy"},
                "recommendations_count": 10
            }
        ]
        
        with patch('ml_analytics.api.recommendation_service.get_user_history') as mock_history_fn:
            mock_history_fn.return_value = mock_history
            
            response = authenticated_client.get(f"/ml-analytics/users/{user_id}/recommendations/history")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "history" in data
            assert len(data["history"]) == 1
    
    @pytest.mark.asyncio
    async def test_recommendation_feedback(self, authenticated_client):
        """Test l'enregistrement de feedback sur les recommandations."""
        feedback_data = {
            "recommendation_id": "rec_123",
            "user_id": "test_user_123",
            "feedback_type": "like",
            "track_id": "spotify:track:test1",
            "rating": 5,
            "comment": "Great recommendation!"
        }
        
        with patch('ml_analytics.api.recommendation_service.record_feedback') as mock_feedback:
            mock_feedback.return_value = {"status": "recorded", "feedback_id": "fb_456"}
            
            response = authenticated_client.post(
                "/ml-analytics/recommendations/feedback",
                json=feedback_data
            )
            
            assert response.status_code == status.HTTP_201_CREATED
            data = response.json()
            assert data["status"] == "recorded"
            assert "feedback_id" in data
    
    @pytest.mark.asyncio
    async def test_contextual_recommendations(self, authenticated_client):
        """Test les recommandations contextuelles."""
        context_data = {
            "user_id": "test_user_123",
            "context": {
                "location": {"latitude": 48.8566, "longitude": 2.3522},  # Paris
                "weather": "sunny",
                "activity": "running",
                "time_of_day": "morning",
                "device": "mobile"
            },
            "num_recommendations": 15
        }
        
        mock_contextual_recs = [
            {
                "track_id": "spotify:track:energetic1",
                "track_name": "Running Song",
                "score": 0.92,
                "context_relevance": 0.95
            }
        ]
        
        with patch('ml_analytics.api.recommendation_service.generate_contextual_recommendations') as mock_contextual:
            mock_contextual.return_value = mock_contextual_recs
            
            response = authenticated_client.post(
                "/ml-analytics/recommendations/contextual",
                json=context_data
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "recommendations" in data
            assert data["recommendations"][0]["context_relevance"] == 0.95


class TestAudioAnalysisEndpoints:
    """Tests pour les endpoints d'analyse audio."""
    
    @pytest.fixture
    def authenticated_client(self, client, valid_token):
        """Client authentifié."""
        client.headers = {"Authorization": f"Bearer {valid_token}"}
        return client
    
    @pytest.mark.asyncio
    async def test_analyze_audio_file_upload(self, authenticated_client):
        """Test l'analyse d'un fichier audio uploadé."""
        # Simuler un fichier audio
        audio_content = b"fake_audio_data" * 1000  # Données audio factices
        
        mock_analysis = {
            "duration": 180.5,
            "sample_rate": 22050,
            "tempo": 120.5,
            "key": "C",
            "genre_prediction": {
                "rock": 0.7,
                "pop": 0.2,
                "jazz": 0.1
            },
            "mood_analysis": {
                "valence": 0.8,
                "arousal": 0.6,
                "mood_label": "happy"
            },
            "quality_score": 0.92,
            "mfcc_features": np.random.rand(13, 100).tolist()
        }
        
        with patch('ml_analytics.api.audio_service.analyze_audio_file') as mock_analyze:
            mock_analyze.return_value = mock_analysis
            
            files = {"audio_file": ("test.mp3", audio_content, "audio/mpeg")}
            response = authenticated_client.post(
                "/ml-analytics/audio/analyze",
                files=files
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert "analysis" in data
            assert data["analysis"]["duration"] == 180.5
            assert data["analysis"]["tempo"] == 120.5
            assert "genre_prediction" in data["analysis"]
    
    def test_analyze_audio_invalid_file(self, authenticated_client):
        """Test l'analyse avec un fichier invalide."""
        # Fichier non-audio
        invalid_content = b"this is not audio data"
        
        files = {"audio_file": ("test.txt", invalid_content, "text/plain")}
        response = authenticated_client.post(
            "/ml-analytics/audio/analyze",
            files=files
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "unsupported file type" in response.json()["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_analyze_audio_url(self, authenticated_client):
        """Test l'analyse d'un fichier audio via URL."""
        url_data = {
            "audio_url": "https://example.com/audio.mp3",
            "analysis_config": {
                "extract_mfcc": True,
                "classify_genre": True,
                "analyze_mood": True,
                "assess_quality": False
            }
        }
        
        mock_analysis = {
            "source": "url",
            "url": url_data["audio_url"],
            "analysis": {
                "duration": 240.0,
                "genre_prediction": {"pop": 0.8, "rock": 0.2},
                "mood_analysis": {"valence": 0.7, "arousal": 0.8}
            }
        }
        
        with patch('ml_analytics.api.audio_service.analyze_audio_from_url') as mock_analyze_url:
            mock_analyze_url.return_value = mock_analysis
            
            response = authenticated_client.post(
                "/ml-analytics/audio/analyze-url",
                json=url_data
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["source"] == "url"
            assert "analysis" in data
    
    @pytest.mark.asyncio
    async def test_batch_audio_analysis(self, authenticated_client):
        """Test l'analyse par lot de fichiers audio."""
        batch_data = {
            "audio_urls": [
                "https://example.com/song1.mp3",
                "https://example.com/song2.mp3",
                "https://example.com/song3.mp3"
            ],
            "analysis_config": {
                "extract_mfcc": True,
                "classify_genre": True
            }
        }
        
        mock_batch_results = [
            {
                "url": "https://example.com/song1.mp3",
                "status": "success",
                "analysis": {"genre_prediction": {"rock": 0.9}}
            },
            {
                "url": "https://example.com/song2.mp3",
                "status": "success", 
                "analysis": {"genre_prediction": {"pop": 0.8}}
            },
            {
                "url": "https://example.com/song3.mp3",
                "status": "error",
                "error": "Failed to download audio"
            }
        ]
        
        with patch('ml_analytics.api.audio_service.analyze_batch') as mock_batch:
            mock_batch.return_value = mock_batch_results
            
            response = authenticated_client.post(
                "/ml-analytics/audio/analyze-batch",
                json=batch_data
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "results" in data
            assert len(data["results"]) == 3
            assert data["results"][0]["status"] == "success"
            assert data["results"][2]["status"] == "error"
    
    @pytest.mark.asyncio
    async def test_audio_similarity_comparison(self, authenticated_client):
        """Test la comparaison de similarité audio."""
        comparison_data = {
            "reference_track": "spotify:track:ref123",
            "comparison_tracks": [
                "spotify:track:comp1",
                "spotify:track:comp2"
            ],
            "similarity_metrics": ["mfcc", "chroma", "spectral"]
        }
        
        mock_similarities = [
            {
                "track_id": "spotify:track:comp1",
                "similarity_score": 0.85,
                "metric_scores": {
                    "mfcc": 0.82,
                    "chroma": 0.88,
                    "spectral": 0.85
                }
            },
            {
                "track_id": "spotify:track:comp2",
                "similarity_score": 0.62,
                "metric_scores": {
                    "mfcc": 0.60,
                    "chroma": 0.65,
                    "spectral": 0.61
                }
            }
        ]
        
        with patch('ml_analytics.api.audio_service.compare_audio_similarity') as mock_similarity:
            mock_similarity.return_value = mock_similarities
            
            response = authenticated_client.post(
                "/ml-analytics/audio/similarity",
                json=comparison_data
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "similarities" in data
            assert len(data["similarities"]) == 2
            assert data["similarities"][0]["similarity_score"] == 0.85


class TestModelManagementEndpoints:
    """Tests pour les endpoints de gestion des modèles."""
    
    @pytest.fixture
    def authenticated_client(self, client, valid_token):
        """Client authentifié avec permissions admin."""
        from jose import jwt
        admin_payload = {
            "user_id": "admin_user",
            "permissions": ["admin", "model_management"],
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        admin_token = jwt.encode(admin_payload, TEST_API_CONFIG["jwt_secret"], algorithm="HS256")
        client.headers = {"Authorization": f"Bearer {admin_token}"}
        return client
    
    @pytest.mark.asyncio
    async def test_list_models(self, authenticated_client):
        """Test la liste des modèles disponibles."""
        mock_models = [
            {
                "model_id": "recommendation_v1",
                "model_type": "hybrid_recommendation",
                "version": "1.0.0",
                "status": "active",
                "accuracy": 0.85,
                "last_updated": "2024-01-15T10:30:00Z"
            },
            {
                "model_id": "audio_classifier_v2",
                "model_type": "genre_classification",
                "version": "2.1.0",
                "status": "active",
                "accuracy": 0.92,
                "last_updated": "2024-01-10T14:20:00Z"
            }
        ]
        
        with patch('ml_analytics.api.model_service.list_models') as mock_list:
            mock_list.return_value = mock_models
            
            response = authenticated_client.get("/ml-analytics/models")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "models" in data
            assert len(data["models"]) == 2
            assert data["models"][0]["model_id"] == "recommendation_v1"
    
    @pytest.mark.asyncio
    async def test_get_model_details(self, authenticated_client):
        """Test les détails d'un modèle spécifique."""
        model_id = "recommendation_v1"
        
        mock_details = {
            "model_id": model_id,
            "model_type": "hybrid_recommendation",
            "version": "1.0.0",
            "status": "active",
            "config": {
                "embedding_dim": 128,
                "num_factors": 100,
                "learning_rate": 0.001
            },
            "metrics": {
                "accuracy": 0.85,
                "precision": 0.83,
                "recall": 0.87,
                "f1_score": 0.85
            },
            "training_info": {
                "training_data_size": 1000000,
                "training_duration": "2h 30m",
                "last_trained": "2024-01-15T10:30:00Z"
            }
        }
        
        with patch('ml_analytics.api.model_service.get_model_details') as mock_details_fn:
            mock_details_fn.return_value = mock_details
            
            response = authenticated_client.get(f"/ml-analytics/models/{model_id}")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["model_id"] == model_id
            assert "config" in data
            assert "metrics" in data
    
    @pytest.mark.asyncio
    async def test_train_model(self, authenticated_client):
        """Test l'entraînement d'un modèle."""
        model_id = "recommendation_v1"
        training_config = {
            "training_data_path": "/data/training_set.csv",
            "validation_split": 0.2,
            "epochs": 50,
            "batch_size": 128,
            "learning_rate": 0.001,
            "early_stopping": True
        }
        
        mock_training_result = {
            "status": "started",
            "job_id": "training_job_456",
            "estimated_duration": "3h",
            "config": training_config
        }
        
        with patch('ml_analytics.api.model_service.start_training') as mock_train:
            mock_train.return_value = mock_training_result
            
            response = authenticated_client.post(
                f"/ml-analytics/models/{model_id}/train",
                json=training_config
            )
            
            assert response.status_code == status.HTTP_202_ACCEPTED
            data = response.json()
            assert data["status"] == "started"
            assert "job_id" in data
    
    @pytest.mark.asyncio
    async def test_model_deployment(self, authenticated_client):
        """Test le déploiement d'un modèle."""
        model_id = "recommendation_v2"
        deployment_config = {
            "environment": "production",
            "replicas": 3,
            "resources": {
                "cpu": "2",
                "memory": "4Gi"
            },
            "auto_scaling": True,
            "health_checks": True
        }
        
        mock_deployment = {
            "status": "deploying",
            "deployment_id": "deploy_789",
            "estimated_time": "10m",
            "endpoints": ["https://api.example.com/ml-analytics/models/recommendation_v2"]
        }
        
        with patch('ml_analytics.api.model_service.deploy_model') as mock_deploy:
            mock_deploy.return_value = mock_deployment
            
            response = authenticated_client.post(
                f"/ml-analytics/models/{model_id}/deploy",
                json=deployment_config
            )
            
            assert response.status_code == status.HTTP_202_ACCEPTED
            data = response.json()
            assert data["status"] == "deploying"
            assert "deployment_id" in data


class TestMonitoringEndpoints:
    """Tests pour les endpoints de monitoring."""
    
    @pytest.fixture
    def authenticated_client(self, client, valid_token):
        """Client authentifié."""
        client.headers = {"Authorization": f"Bearer {valid_token}"}
        return client
    
    @pytest.mark.asyncio
    async def test_system_health(self, authenticated_client):
        """Test l'endpoint de santé système."""
        mock_health = {
            "healthy": True,
            "timestamp": "2024-01-15T12:00:00Z",
            "components": {
                "database": {"status": "healthy", "response_time": 0.012},
                "cache": {"status": "healthy", "hit_ratio": 0.89},
                "models": {"status": "healthy", "loaded": 5, "failed": 0}
            },
            "version": "2.0.0",
            "uptime": "5d 12h 30m"
        }
        
        with patch('ml_analytics.api.monitoring_service.get_system_health') as mock_health_fn:
            mock_health_fn.return_value = mock_health
            
            response = authenticated_client.get("/ml-analytics/monitoring/health")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["healthy"] is True
            assert "components" in data
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, authenticated_client):
        """Test les métriques de performance."""
        mock_metrics = {
            "timestamp": "2024-01-15T12:00:00Z",
            "requests_per_second": 150.5,
            "average_response_time": 0.234,
            "error_rate": 0.001,
            "p95_response_time": 0.456,
            "p99_response_time": 0.789,
            "memory_usage": 0.65,
            "cpu_usage": 0.45,
            "cache_metrics": {
                "hit_ratio": 0.89,
                "miss_ratio": 0.11,
                "evictions": 15
            },
            "model_metrics": {
                "prediction_latency": 0.123,
                "model_accuracy": 0.85,
                "drift_score": 0.05
            }
        }
        
        with patch('ml_analytics.api.monitoring_service.get_performance_metrics') as mock_metrics_fn:
            mock_metrics_fn.return_value = mock_metrics
            
            response = authenticated_client.get("/ml-analytics/monitoring/metrics")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "requests_per_second" in data
            assert "cache_metrics" in data
            assert "model_metrics" in data
    
    @pytest.mark.asyncio
    async def test_active_alerts(self, authenticated_client):
        """Test les alertes actives."""
        mock_alerts = [
            {
                "alert_id": "alert_123",
                "severity": "warning",
                "message": "High memory usage detected",
                "component": "model_service",
                "threshold": 0.85,
                "current_value": 0.91,
                "triggered_at": "2024-01-15T11:45:00Z",
                "status": "active"
            },
            {
                "alert_id": "alert_124",
                "severity": "info",
                "message": "Model accuracy below threshold",
                "component": "recommendation_model",
                "threshold": 0.80,
                "current_value": 0.78,
                "triggered_at": "2024-01-15T11:30:00Z",
                "status": "acknowledged"
            }
        ]
        
        with patch('ml_analytics.api.monitoring_service.get_active_alerts') as mock_alerts_fn:
            mock_alerts_fn.return_value = mock_alerts
            
            response = authenticated_client.get("/ml-analytics/monitoring/alerts")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "alerts" in data
            assert len(data["alerts"]) == 2
            assert data["alerts"][0]["severity"] == "warning"


class TestAPIErrorHandling:
    """Tests pour la gestion d'erreurs de l'API."""
    
    @pytest.fixture
    def authenticated_client(self, client, valid_token):
        """Client authentifié."""
        client.headers = {"Authorization": f"Bearer {valid_token}"}
        return client
    
    @pytest.mark.asyncio
    async def test_internal_server_error(self, authenticated_client):
        """Test la gestion d'erreur serveur interne."""
        with patch('ml_analytics.api.recommendation_service.generate_recommendations') as mock_gen:
            mock_gen.side_effect = Exception("Internal processing error")
            
            response = authenticated_client.post(
                "/ml-analytics/recommendations",
                json={"user_id": "test_user", "num_recommendations": 10}
            )
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            data = response.json()
            assert "detail" in data
            assert "internal" in data["detail"].lower()
    
    def test_validation_error_response(self, authenticated_client):
        """Test la réponse d'erreur de validation."""
        # Données manquantes
        response = authenticated_client.post(
            "/ml-analytics/recommendations",
            json={}  # Données vides
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "detail" in data
        assert isinstance(data["detail"], list)  # Liste d'erreurs de validation
    
    @pytest.mark.asyncio
    async def test_rate_limit_error(self, authenticated_client):
        """Test la limitation de taux."""
        # Simuler dépassement de limite
        with patch('ml_analytics.api.check_rate_limit') as mock_rate_limit:
            mock_rate_limit.return_value = False
            
            response = authenticated_client.post(
                "/ml-analytics/recommendations",
                json={"user_id": "test_user", "num_recommendations": 10}
            )
            
            assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
            assert "rate limit" in response.json()["detail"].lower()
    
    def test_not_found_error(self, authenticated_client):
        """Test l'erreur de ressource non trouvée."""
        response = authenticated_client.get("/ml-analytics/models/nonexistent_model")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"].lower()


@pytest.mark.performance
class TestAPIPerformance:
    """Tests de performance pour l'API."""
    
    @pytest.fixture
    def authenticated_client(self, client, valid_token):
        """Client authentifié."""
        client.headers = {"Authorization": f"Bearer {valid_token}"}
        return client
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, authenticated_client):
        """Test les requêtes concurrentes."""
        import asyncio
        
        async def make_request():
            with patch('ml_analytics.api.recommendation_service.generate_recommendations') as mock_gen:
                mock_gen.return_value = [{"track_id": "test", "score": 0.8}]
                
                return authenticated_client.post(
                    "/ml-analytics/recommendations",
                    json={"user_id": "test_user", "num_recommendations": 5}
                )
        
        # Faire 50 requêtes concurrentes
        start_time = datetime.now()
        
        responses = await asyncio.gather(*[make_request() for _ in range(50)])
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Vérifier que toutes les requêtes ont réussi
        assert all(r.status_code == status.HTTP_200_OK for r in responses)
        
        # Performance: moins de 10 secondes pour 50 requêtes
        assert duration < 10.0
    
    @pytest.mark.asyncio
    async def test_large_payload_handling(self, authenticated_client):
        """Test la gestion de gros payloads."""
        # Créer un gros payload de recommandation
        large_request = {
            "user_id": "test_user",
            "num_recommendations": 100,
            "context": {
                "listening_history": [
                    {"track_id": f"track_{i}", "rating": 0.8}
                    for i in range(1000)  # 1000 pistes dans l'historique
                ]
            }
        }
        
        with patch('ml_analytics.api.recommendation_service.generate_recommendations') as mock_gen:
            mock_gen.return_value = [
                {"track_id": f"rec_{i}", "score": 0.8 - i * 0.01}
                for i in range(100)
            ]
            
            start_time = datetime.now()
            
            response = authenticated_client.post(
                "/ml-analytics/recommendations",
                json=large_request
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            assert response.status_code == status.HTTP_200_OK
            assert len(response.json()["recommendations"]) == 100
            
            # Performance: moins de 5 secondes pour un gros payload
            assert duration < 5.0


@pytest.mark.integration
class TestAPIIntegration:
    """Tests d'intégration complets pour l'API."""
    
    @pytest.fixture
    def authenticated_client(self, client, valid_token):
        """Client authentifié."""
        client.headers = {"Authorization": f"Bearer {valid_token}"}
        return client
    
    @pytest.mark.asyncio
    async def test_full_recommendation_workflow(self, authenticated_client):
        """Test complet du workflow de recommandation."""
        # 1. Obtenir des recommandations
        with patch('ml_analytics.api.recommendation_service.generate_recommendations') as mock_gen:
            mock_gen.return_value = [
                {"track_id": "track_1", "score": 0.9, "explanation": "Great match"}
            ]
            
            rec_response = authenticated_client.post(
                "/ml-analytics/recommendations",
                json={"user_id": "test_user", "num_recommendations": 5}
            )
            
            assert rec_response.status_code == status.HTTP_200_OK
            recommendations = rec_response.json()["recommendations"]
        
        # 2. Enregistrer du feedback
        with patch('ml_analytics.api.recommendation_service.record_feedback') as mock_feedback:
            mock_feedback.return_value = {"status": "recorded"}
            
            feedback_response = authenticated_client.post(
                "/ml-analytics/recommendations/feedback",
                json={
                    "recommendation_id": "rec_123",
                    "user_id": "test_user",
                    "track_id": recommendations[0]["track_id"],
                    "feedback_type": "like"
                }
            )
            
            assert feedback_response.status_code == status.HTTP_201_CREATED
        
        # 3. Vérifier l'historique
        with patch('ml_analytics.api.recommendation_service.get_user_history') as mock_history:
            mock_history.return_value = [{"timestamp": "2024-01-15T12:00:00Z"}]
            
            history_response = authenticated_client.get(
                "/ml-analytics/users/test_user/recommendations/history"
            )
            
            assert history_response.status_code == status.HTTP_200_OK
            assert len(history_response.json()["history"]) >= 0
    
    @pytest.mark.asyncio
    async def test_cross_service_integration(self, authenticated_client):
        """Test l'intégration entre services (recommandation + audio)."""
        # 1. Analyser un fichier audio
        with patch('ml_analytics.api.audio_service.analyze_audio_file') as mock_audio:
            mock_audio.return_value = {
                "genre_prediction": {"rock": 0.8},
                "mood_analysis": {"valence": 0.7, "arousal": 0.8}
            }
            
            audio_response = authenticated_client.post(
                "/ml-analytics/audio/analyze",
                files={"audio_file": ("test.mp3", b"fake_audio", "audio/mpeg")}
            )
            
            assert audio_response.status_code == status.HTTP_200_OK
            audio_analysis = audio_response.json()["analysis"]
        
        # 2. Utiliser l'analyse pour des recommandations contextuelles
        with patch('ml_analytics.api.recommendation_service.generate_contextual_recommendations') as mock_contextual:
            mock_contextual.return_value = [
                {"track_id": "similar_track", "score": 0.85}
            ]
            
            contextual_response = authenticated_client.post(
                "/ml-analytics/recommendations/contextual",
                json={
                    "user_id": "test_user",
                    "context": {
                        "audio_analysis": audio_analysis,
                        "preference_match": True
                    }
                }
            )
            
            assert contextual_response.status_code == status.HTTP_200_OK
            contextual_recs = contextual_response.json()["recommendations"]
            assert len(contextual_recs) > 0
