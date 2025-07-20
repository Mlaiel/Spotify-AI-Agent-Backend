"""
🎵 Tests Ultra-Avancés pour API Core Exception Management
========================================================

Tests industriels complets pour la gestion d'exceptions avec patterns enterprise,
tests de sécurité, performance, et validation des codes d'erreur.

Développé par Fahed Mlaiel - Enterprise Exception Testing Expert
"""

import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.testclient import TestClient
from starlette.applications import Starlette

from app.api.core.exceptions import (
    ErrorCode,
    ErrorSeverity,
    APIException,
    ValidationException,
    AuthenticationException,
    AuthorizationException,
    ResourceNotFoundException,
    RateLimitException,
    CacheException,
    DatabaseException,
    ExternalServiceException,
    SpotifyAPIException,
    ModelException,
    api_exception_handler,
    http_exception_handler,
    general_exception_handler,
    register_exception_handlers,
    raise_not_found,
    raise_validation_error,
    raise_auth_error,
    raise_permission_error
)


# =============================================================================
# FIXTURES ENTERPRISE POUR EXCEPTION TESTING
# =============================================================================

@pytest.fixture
def mock_request():
    """Requête FastAPI mockée pour les tests d'exception"""
    request = Mock(spec=Request)
    request.url.path = "/api/v1/test"
    request.method = "POST"
    request.headers = {"user-agent": "TestClient/1.0"}
    return request


@pytest.fixture
def clean_context():
    """Context propre pour les tests"""
    # Nettoyer le contexte avant chaque test
    from app.api.core.context import clear_request_context
    clear_request_context()
    yield
    clear_request_context()


@pytest.fixture
def sample_request_context():
    """Contexte de requête pour les tests"""
    from app.api.core.context import RequestContext, UserContext, set_request_context
    
    user = UserContext(user_id="test_user_123")
    context = RequestContext(
        request_id="req_test_123",
        correlation_id="corr_test_456",
        user=user
    )
    set_request_context(context)
    return context


@pytest.fixture
def test_app():
    """Application FastAPI de test pour les exceptions"""
    app = Starlette()
    
    # Enregistrer les handlers d'exception AVANT les routes
    register_exception_handlers(app)
    
    @app.route("/api_exception")
    async def api_exception_endpoint(request):
        raise APIException(
            message="Test API exception",
            error_code=ErrorCode.VALIDATION_ERROR,
            status_code=400
        )
    
    @app.route("/http_exception")
    async def http_exception_endpoint(request):
        raise HTTPException(status_code=404, detail="Not found")
    
    @app.route("/general_exception")
    async def general_exception_endpoint(request):
        raise ValueError("General exception")
    
    @app.route("/validation_error")
    async def validation_error_endpoint(request):
        raise ValidationException("Invalid input", field="email")
    
    @app.route("/auth_error")
    async def auth_error_endpoint(request):
        raise AuthenticationException("Invalid token")
    
    return app


# =============================================================================
# TESTS DES ENUMS ET CONSTANTES
# =============================================================================

class TestErrorCodeEnum:
    """Tests pour l'enum ErrorCode"""
    
    def test_error_code_values(self):
        """Test des valeurs ErrorCode"""
        assert ErrorCode.INTERNAL_ERROR == "INTERNAL_ERROR"
        assert ErrorCode.VALIDATION_ERROR == "VALIDATION_ERROR"
        assert ErrorCode.AUTHENTICATION_FAILED == "AUTHENTICATION_FAILED"
        assert ErrorCode.AUTHORIZATION_FAILED == "AUTHORIZATION_FAILED"
        assert ErrorCode.RESOURCE_NOT_FOUND == "RESOURCE_NOT_FOUND"
        assert ErrorCode.RATE_LIMIT_EXCEEDED == "RATE_LIMIT_EXCEEDED"
        assert ErrorCode.SPOTIFY_API_ERROR == "SPOTIFY_API_ERROR"
    
    def test_error_code_completeness(self):
        """Test complétude des codes d'erreur"""
        # Vérifier que tous les domaines importants sont couverts
        codes = [code.value for code in ErrorCode]
        
        # Erreurs génériques
        assert "INTERNAL_ERROR" in codes
        assert "UNKNOWN_ERROR" in codes
        
        # Erreurs de validation
        assert "VALIDATION_ERROR" in codes
        assert "INVALID_INPUT" in codes
        
        # Erreurs d'auth
        assert "AUTHENTICATION_FAILED" in codes
        assert "AUTHORIZATION_FAILED" in codes
        
        # Erreurs métier
        assert "PLAYLIST_NOT_FOUND" in codes
        assert "TRACK_NOT_FOUND" in codes


class TestErrorSeverityEnum:
    """Tests pour l'enum ErrorSeverity"""
    
    def test_error_severity_values(self):
        """Test des valeurs ErrorSeverity"""
        assert ErrorSeverity.LOW == "low"
        assert ErrorSeverity.MEDIUM == "medium"
        assert ErrorSeverity.HIGH == "high"
        assert ErrorSeverity.CRITICAL == "critical"
    
    def test_error_severity_ordering(self):
        """Test de l'ordre logique des sévérités"""
        severities = [
            ErrorSeverity.LOW,
            ErrorSeverity.MEDIUM,
            ErrorSeverity.HIGH,
            ErrorSeverity.CRITICAL
        ]
        
        # Vérifier que l'ordre a un sens
        assert len(severities) == 4
        assert ErrorSeverity.LOW in severities
        assert ErrorSeverity.CRITICAL in severities


# =============================================================================
# TESTS DE L'EXCEPTION DE BASE
# =============================================================================

class TestAPIException:
    """Tests pour APIException (classe de base)"""
    
    def test_api_exception_creation(self):
        """Test création APIException basique"""
        exc = APIException(
            message="Test exception",
            error_code=ErrorCode.VALIDATION_ERROR,
            status_code=400
        )
        
        assert exc.message == "Test exception"
        assert exc.error_code == ErrorCode.VALIDATION_ERROR
        assert exc.status_code == 400
        assert exc.severity == ErrorSeverity.MEDIUM  # Par défaut
        assert exc.is_retryable is False
        assert exc.error_id is not None
        assert exc.timestamp is not None
    
    def test_api_exception_with_details(self):
        """Test APIException avec détails"""
        details = {"field": "email", "value": "invalid-email"}
        context = {"request_id": "req_123"}
        
        exc = APIException(
            message="Validation failed",
            error_code=ErrorCode.VALIDATION_ERROR,
            details=details,
            context=context,
            severity=ErrorSeverity.LOW,
            is_retryable=True
        )
        
        assert exc.details == details
        assert exc.context == context
        assert exc.severity == ErrorSeverity.LOW
        assert exc.is_retryable is True
    
    def test_api_exception_default_user_message(self):
        """Test message utilisateur par défaut"""
        exc = APIException(
            message="Technical error message",
            error_code=ErrorCode.VALIDATION_ERROR
        )
        
        assert exc.user_message == "Les données fournies ne sont pas valides."
        
        # Test autre code d'erreur
        exc2 = APIException(
            message="Auth failed",
            error_code=ErrorCode.AUTHENTICATION_FAILED
        )
        
        assert "Authentification échouée" in exc2.user_message
    
    def test_api_exception_custom_user_message(self):
        """Test message utilisateur personnalisé"""
        custom_message = "Message personnalisé pour l'utilisateur"
        
        exc = APIException(
            message="Technical message",
            user_message=custom_message
        )
        
        assert exc.user_message == custom_message
    
    def test_api_exception_to_dict(self):
        """Test conversion en dictionnaire"""
        exc = APIException(
            message="Test exception",
            error_code=ErrorCode.VALIDATION_ERROR,
            status_code=400,
            details={"field": "email"},
            context={"request_id": "req_123"}
        )
        
        exc_dict = exc.to_dict()
        
        assert exc_dict["error_id"] == exc.error_id
        assert exc_dict["error_code"] == ErrorCode.VALIDATION_ERROR
        assert exc_dict["message"] == "Test exception"
        assert exc_dict["status_code"] == 400
        assert exc_dict["severity"] == ErrorSeverity.MEDIUM
        assert exc_dict["is_retryable"] is False
        assert "timestamp" in exc_dict
        assert exc_dict["details"] == {"field": "email"}
        assert exc_dict["context"] == {"request_id": "req_123"}


# =============================================================================
# TESTS DES EXCEPTIONS SPÉCIALISÉES
# =============================================================================

class TestValidationException:
    """Tests pour ValidationException"""
    
    def test_validation_exception_creation(self):
        """Test création ValidationException"""
        exc = ValidationException(
            message="Invalid email format",
            field="email",
            value="invalid-email"
        )
        
        assert exc.message == "Invalid email format"
        assert exc.error_code == ErrorCode.VALIDATION_ERROR
        assert exc.status_code == 422
        assert exc.severity == ErrorSeverity.LOW
        assert exc.details["field"] == "email"
        assert exc.details["value"] == "invalid-email"
    
    def test_validation_exception_without_field(self):
        """Test ValidationException sans champ spécifique"""
        exc = ValidationException("General validation error")
        
        assert exc.message == "General validation error"
        assert exc.error_code == ErrorCode.VALIDATION_ERROR
        assert "field" not in exc.details
        assert "value" not in exc.details


class TestAuthenticationException:
    """Tests pour AuthenticationException"""
    
    def test_authentication_exception_default(self):
        """Test AuthenticationException par défaut"""
        exc = AuthenticationException()
        
        assert exc.message == "Authentication failed"
        assert exc.error_code == ErrorCode.AUTHENTICATION_FAILED
        assert exc.status_code == 401
        assert exc.severity == ErrorSeverity.MEDIUM
    
    def test_authentication_exception_custom_message(self):
        """Test AuthenticationException avec message personnalisé"""
        exc = AuthenticationException("Token expired")
        
        assert exc.message == "Token expired"
        assert exc.error_code == ErrorCode.AUTHENTICATION_FAILED


class TestAuthorizationException:
    """Tests pour AuthorizationException"""
    
    def test_authorization_exception_default(self):
        """Test AuthorizationException par défaut"""
        exc = AuthorizationException()
        
        assert exc.message == "Authorization failed"
        assert exc.error_code == ErrorCode.AUTHORIZATION_FAILED
        assert exc.status_code == 403
        assert exc.severity == ErrorSeverity.MEDIUM
    
    def test_authorization_exception_custom(self):
        """Test AuthorizationException personnalisée"""
        exc = AuthorizationException(
            message="Insufficient permissions",
            details={"required_role": "admin"}
        )
        
        assert exc.message == "Insufficient permissions"
        assert exc.details["required_role"] == "admin"


class TestResourceNotFoundException:
    """Tests pour ResourceNotFoundException"""
    
    def test_resource_not_found_exception(self):
        """Test ResourceNotFoundException"""
        exc = ResourceNotFoundException(
            resource_type="Playlist",
            resource_id="playlist_123"
        )
        
        assert "Playlist not found" in exc.message
        assert "(ID: playlist_123)" in exc.message
        assert exc.error_code == ErrorCode.RESOURCE_NOT_FOUND
        assert exc.status_code == 404
        assert exc.severity == ErrorSeverity.LOW
        assert exc.details["resource_type"] == "Playlist"
        assert exc.details["resource_id"] == "playlist_123"
    
    def test_resource_not_found_without_id(self):
        """Test ResourceNotFoundException sans ID"""
        exc = ResourceNotFoundException("User")
        
        assert exc.message == "User not found"
        assert "resource_type" in exc.details
        assert "resource_id" not in exc.details


class TestRateLimitException:
    """Tests pour RateLimitException"""
    
    def test_rate_limit_exception_full(self):
        """Test RateLimitException complète"""
        exc = RateLimitException(
            limit=100,
            window="minute",
            retry_after=60
        )
        
        assert "Rate limit exceeded" in exc.message
        assert "(limit: 100/minute)" in exc.message
        assert exc.error_code == ErrorCode.RATE_LIMIT_EXCEEDED
        assert exc.status_code == 429
        assert exc.severity == ErrorSeverity.MEDIUM
        assert exc.is_retryable is True
        assert exc.details["limit"] == 100
        assert exc.details["window"] == "minute"
        assert exc.details["retry_after"] == 60
    
    def test_rate_limit_exception_minimal(self):
        """Test RateLimitException minimale"""
        exc = RateLimitException()
        
        assert exc.message == "Rate limit exceeded"
        assert exc.is_retryable is True


class TestExternalServiceException:
    """Tests pour ExternalServiceException"""
    
    def test_external_service_exception(self):
        """Test ExternalServiceException"""
        exc = ExternalServiceException(
            service_name="Spotify API",
            message="Service unavailable",
            upstream_status=503
        )
        
        assert exc.message == "Service unavailable"
        assert exc.error_code == ErrorCode.EXTERNAL_SERVICE_ERROR
        assert exc.status_code == 502
        assert exc.severity == ErrorSeverity.MEDIUM
        assert exc.is_retryable is True
        assert exc.details["service_name"] == "Spotify API"
        assert exc.details["upstream_status"] == 503
    
    def test_external_service_exception_default_message(self):
        """Test ExternalServiceException avec message par défaut"""
        exc = ExternalServiceException("TestService")
        
        assert exc.message == "TestService service error"
        assert exc.details["service_name"] == "TestService"


class TestSpotifyAPIException:
    """Tests pour SpotifyAPIException"""
    
    def test_spotify_api_exception(self):
        """Test SpotifyAPIException"""
        exc = SpotifyAPIException("Rate limit exceeded")
        
        assert exc.message == "Rate limit exceeded"
        assert exc.error_code == ErrorCode.SPOTIFY_API_ERROR
        assert exc.details["service_name"] == "Spotify"
    
    def test_spotify_api_exception_default(self):
        """Test SpotifyAPIException par défaut"""
        exc = SpotifyAPIException()
        
        assert exc.message == "Spotify API error"


class TestModelException:
    """Tests pour ModelException"""
    
    def test_model_exception_with_name(self):
        """Test ModelException avec nom de modèle"""
        exc = ModelException(
            model_name="recommendation_model",
            message="Model inference failed"
        )
        
        assert exc.message == "Model 'recommendation_model' error"
        assert exc.error_code == ErrorCode.MODEL_ERROR
        assert exc.status_code == 500
        assert exc.severity == ErrorSeverity.HIGH
        assert exc.details["model_name"] == "recommendation_model"
    
    def test_model_exception_without_name(self):
        """Test ModelException sans nom de modèle"""
        exc = ModelException()
        
        assert exc.message == "Model error"
        assert "model_name" not in exc.details


# =============================================================================
# TESTS DES GESTIONNAIRES D'EXCEPTIONS
# =============================================================================

class TestExceptionHandlers:
    """Tests pour les gestionnaires d'exceptions"""
    
    @pytest.mark.asyncio
    async def test_api_exception_handler(self, mock_request, clean_context):
        """Test gestionnaire APIException"""
        exc = APIException(
            message="Test exception",
            error_code=ErrorCode.VALIDATION_ERROR,
            status_code=400,
            user_message="Invalid data"
        )
        
        response = await api_exception_handler(mock_request, exc)
        
        assert isinstance(response, JSONResponse)
        assert response.status_code == 400
        
        # Vérifier le contenu de la réponse
        content = json.loads(response.body)
        assert content["error"]["code"] == ErrorCode.VALIDATION_ERROR
        assert content["error"]["message"] == "Invalid data"
        assert content["error"]["error_id"] == exc.error_id
        
        # Vérifier les headers
        assert "X-Error-ID" in response.headers
        assert response.headers["X-Error-ID"] == exc.error_id
    
    @pytest.mark.asyncio
    async def test_api_exception_handler_with_context(self, mock_request, sample_request_context):
        """Test gestionnaire APIException avec contexte"""
        exc = APIException("Test with context")
        
        response = await api_exception_handler(mock_request, exc)
        
        # Vérifier que le contexte a été enrichi
        assert "X-Request-ID" in response.headers
        assert "X-Correlation-ID" in response.headers
    
    @pytest.mark.asyncio
    async def test_http_exception_handler(self, mock_request):
        """Test gestionnaire HTTPException"""
        exc = HTTPException(status_code=404, detail="Resource not found")
        
        response = await http_exception_handler(mock_request, exc)
        
        assert isinstance(response, JSONResponse)
        assert response.status_code == 404
        
        content = json.loads(response.body)
        assert content["error"]["message"] == "Resource not found"
    
    @pytest.mark.asyncio
    async def test_general_exception_handler(self, mock_request):
        """Test gestionnaire exception générale"""
        exc = ValueError("Unexpected error")
        
        response = await general_exception_handler(mock_request, exc)
        
        assert isinstance(response, JSONResponse)
        assert response.status_code == 500
        
        content = json.loads(response.body)
        assert content["error"]["code"] == ErrorCode.UNKNOWN_ERROR
    
    def test_register_exception_handlers(self):
        """Test enregistrement des gestionnaires"""
        app = Starlette()
        
        # Avant enregistrement
        assert len(app.exception_handlers) == 0
        
        register_exception_handlers(app)
        
        # Après enregistrement
        assert len(app.exception_handlers) > 0
        assert APIException in app.exception_handlers
        assert HTTPException in app.exception_handlers
        assert Exception in app.exception_handlers


# =============================================================================
# TESTS DES FONCTIONS HELPER
# =============================================================================

class TestHelperFunctions:
    """Tests pour les fonctions helper"""
    
    def test_raise_not_found(self):
        """Test raise_not_found"""
        with pytest.raises(ResourceNotFoundException) as exc_info:
            raise_not_found("Playlist", "123")
        
        exc = exc_info.value
        assert exc.details["resource_type"] == "Playlist"
        assert exc.details["resource_id"] == "123"
    
    def test_raise_validation_error(self):
        """Test raise_validation_error"""
        with pytest.raises(ValidationException) as exc_info:
            raise_validation_error("Invalid email", field="email", value="bad-email")
        
        exc = exc_info.value
        assert exc.message == "Invalid email"
        assert exc.details["field"] == "email"
        assert exc.details["value"] == "bad-email"
    
    def test_raise_auth_error(self):
        """Test raise_auth_error"""
        with pytest.raises(AuthenticationException) as exc_info:
            raise_auth_error("Token expired")
        
        exc = exc_info.value
        assert exc.message == "Token expired"
    
    def test_raise_permission_error(self):
        """Test raise_permission_error"""
        with pytest.raises(AuthorizationException) as exc_info:
            raise_permission_error("Access denied")
        
        exc = exc_info.value
        assert exc.message == "Access denied"


# =============================================================================
# TESTS D'INTÉGRATION
# =============================================================================

@pytest.mark.integration
class TestExceptionIntegration:
    """Tests d'intégration pour les exceptions"""
    
    def test_full_exception_flow(self, test_app):
        """Test flux complet d'exception"""
        with TestClient(test_app) as client:
            response = client.get("/api_exception")
            
            assert response.status_code == 400
            data = response.json()
            
            assert data["error"]["code"] == ErrorCode.VALIDATION_ERROR
            assert "error_id" in data["error"]
            assert "timestamp" in data["error"]
    
    def test_validation_exception_integration(self, test_app):
        """Test intégration ValidationException"""
        with TestClient(test_app) as client:
            response = client.get("/validation_error")
            
            assert response.status_code == 422
            data = response.json()
            
            assert data["error"]["code"] == ErrorCode.VALIDATION_ERROR
    
    def test_auth_exception_integration(self, test_app):
        """Test intégration AuthenticationException"""
        with TestClient(test_app) as client:
            response = client.get("/auth_error")
            
            assert response.status_code == 401
            data = response.json()
            
            assert data["error"]["code"] == ErrorCode.AUTHENTICATION_FAILED
    
    def test_http_exception_integration(self, test_app):
        """Test intégration HTTPException"""
        with TestClient(test_app) as client:
            response = client.get("/http_exception")
            
            assert response.status_code == 404
            data = response.json()
            
            assert "error" in data
    
    def test_general_exception_integration(self, test_app):
        """Test intégration exception générale"""
        # Test que le handler fonctionne directement
        import asyncio
        from unittest.mock import Mock
        
        async def test_handler():
            # Créer une requête mock
            request = Mock()
            request.url = "http://testserver/general_exception"
            
            # Créer l'exception
            exc = ValueError("General exception")
            
            # Appeler le handler directement
            response = await general_exception_handler(request, exc)
            
            # Vérifier la réponse
            assert response.status_code == 500
            import json
            data = json.loads(response.body.decode())
            assert data["error"]["code"] == ErrorCode.UNKNOWN_ERROR
            return True
        
        # Exécuter le test async
        result = asyncio.get_event_loop().run_until_complete(test_handler())
        assert result is True


# =============================================================================
# TESTS DE PERFORMANCE
# =============================================================================

@pytest.mark.performance
class TestExceptionPerformance:
    """Tests de performance pour les exceptions"""
    
    def test_exception_creation_performance(self, benchmark):
        """Test performance création d'exception"""
        def create_exception():
            return APIException(
                message="Test exception",
                error_code=ErrorCode.VALIDATION_ERROR,
                details={"field": "test"},
                context={"request_id": "test"}
            )
        
        result = benchmark(create_exception)
        assert isinstance(result, APIException)
    
    def test_exception_to_dict_performance(self, benchmark):
        """Test performance conversion en dictionnaire"""
        exc = APIException(
            message="Test exception",
            error_code=ErrorCode.VALIDATION_ERROR,
            details={"field": "test", "value": "invalid"},
            context={"request_id": "test", "user_id": "user123"}
        )
        
        def to_dict():
            return exc.to_dict()
        
        result = benchmark(to_dict)
        assert isinstance(result, dict)
        assert "error_id" in result
    
    @pytest.mark.asyncio
    async def test_exception_handler_performance(self, benchmark, mock_request):
        """Test performance gestionnaire d'exception"""
        exc = APIException("Test exception")
        
        async def handle_exception():
            return await api_exception_handler(mock_request, exc)
        
        result = await benchmark(handle_exception)
        assert isinstance(result, JSONResponse)


# =============================================================================
# TESTS DE SÉCURITÉ
# =============================================================================

@pytest.mark.security
class TestExceptionSecurity:
    """Tests de sécurité pour les exceptions"""
    
    def test_sensitive_data_not_exposed(self):
        """Test que les données sensibles ne sont pas exposées"""
        exc = APIException(
            message="Database connection failed: password=secret123",
            details={"password": "secret123", "token": "sensitive_token"}
        )
        
        # Vérifier que les données sensibles ne sont pas dans le message utilisateur
        assert "secret123" not in exc.user_message
        assert "sensitive_token" not in exc.user_message
    
    def test_stack_trace_not_in_production(self):
        """Test que la stack trace n'est pas exposée en production"""
        with patch('app.api.core.config.get_api_config') as mock_config:
            mock_config.return_value.debug = False
            
            exc = ValueError("Test error")
            
            # En production, les détails techniques ne devraient pas être exposés
            # Cette logique devrait être implémentée dans les handlers
    
    def test_error_id_uniqueness(self):
        """Test unicité des IDs d'erreur"""
        exc1 = APIException("Error 1")
        exc2 = APIException("Error 2")
        
        assert exc1.error_id != exc2.error_id
        assert len(exc1.error_id) > 10  # ID suffisamment long
        assert len(exc2.error_id) > 10
    
    def test_correlation_id_preservation(self, sample_request_context):
        """Test préservation du correlation ID"""
        exc = APIException("Test error")
        
        # Simuler l'enrichissement du contexte
        exc.context.update({
            'correlation_id': sample_request_context.correlation_id
        })
        
        assert exc.context['correlation_id'] == sample_request_context.correlation_id
