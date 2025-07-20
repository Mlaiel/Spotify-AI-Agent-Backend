"""
🎵 Tests Ultra-Avancés pour API Core Response Management
=======================================================

Tests industriels complets pour la gestion des réponses API avec patterns enterprise,
tests de sécurité, performance, et validation des formats de réponse.

Développé par Fahed Mlaiel - Enterprise Response Testing Expert
"""

import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from decimal import Decimal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.testclient import TestClient
from pydantic import BaseModel

from app.api.core.response import (
    APIResponse,
    SuccessResponse,
    ErrorResponse,
    # PaginatedResponse,  # Not implemented yet
    ResponseMetadata,
    # PaginationMetadata,  # Not implemented yet
    # ErrorDetail,  # Not implemented yet
    # create_success_response,  # Not implemented yet
    # create_error_response,  # Not implemented yet
    # create_paginated_response,  # Not implemented yet
    # format_response_data,  # Not implemented yet
    enrich_response_metadata,
    validate_response_data,
    ResponseBuilder,
    ResponseStatus,
    ResponseMimeType
)


# =============================================================================
# FIXTURES ENTERPRISE POUR RESPONSE TESTING
# =============================================================================

@pytest.fixture
def sample_data():
    """Données de test simples"""
    return {
        "id": "123",
        "name": "Test Item",
        "description": "Test description",
        "value": 42.5,
        "active": True,
        "tags": ["test", "sample"]
    }


@pytest.fixture
def sample_list_data():
    """Liste de données de test"""
    return [
        {"id": "1", "name": "Item 1", "value": 10},
        {"id": "2", "name": "Item 2", "value": 20},
        {"id": "3", "name": "Item 3", "value": 30}
    ]


@pytest.fixture
def sample_pagination():
    """Métadonnées de pagination"""
    return PaginationMetadata(
        page=1,
        per_page=10,
        total=100,
        total_pages=10
    )


@pytest.fixture
def sample_metadata():
    """Métadonnées de réponse"""
    return ResponseMetadata(
        request_id="req_123",
        correlation_id="corr_456",
        timestamp=datetime.now(timezone.utc),
        version="1.0",
        environment="test"
    )


@pytest.fixture
def sample_error_detail():
    """Détail d'erreur"""
    return ErrorDetail(
        code="VALIDATION_ERROR",
        message="Validation failed",
        field="email",
        value="invalid-email"
    )


@pytest.fixture
def mock_request():
    """Requête mockée"""
    request = Mock(spec=Request)
    request.url.path = "/api/v1/test"
    request.method = "GET"
    request.headers = {"user-agent": "TestClient/1.0"}
    return request


@pytest.fixture
def test_app():
    """Application FastAPI de test"""
    app = FastAPI()
    
    @app.get("/success")
    async def success_endpoint():
        return create_success_response(
            data={"message": "Success"},
            message="Operation successful"
        )
    
    @app.get("/error")
    async def error_endpoint():
        return create_error_response(
            message="Test error",
            error_code="TEST_ERROR",
            status_code=400
        )
    
    @app.get("/paginated")
    async def paginated_endpoint():
        data = [{"id": i, "name": f"Item {i}"} for i in range(1, 6)]
        pagination = PaginationMetadata(page=1, per_page=5, total=50, total_pages=10)
        return create_paginated_response(
            data=data,
            pagination=pagination
        )
    
    return app


# =============================================================================
# TESTS DES MODÈLES DE BASE
# =============================================================================

class TestResponseMetadata:
    """Tests pour ResponseMetadata"""
    
    def test_response_metadata_creation(self):
        """Test création ResponseMetadata"""
        metadata = ResponseMetadata(
            request_id="req_123",
            correlation_id="corr_456",
            timestamp=datetime.now(timezone.utc),
            version="1.0"
        )
        
        assert metadata.request_id == "req_123"
        assert metadata.correlation_id == "corr_456"
        assert metadata.version == "1.0"
        assert metadata.timestamp is not None
    
    def test_response_metadata_defaults(self):
        """Test valeurs par défaut"""
        metadata = ResponseMetadata()
        
        assert metadata.request_id is not None
        assert metadata.correlation_id is not None
        assert metadata.timestamp is not None
        assert metadata.version == "1.0"
        assert metadata.environment == "development"
    
    def test_response_metadata_serialization(self):
        """Test sérialisation"""
        metadata = ResponseMetadata(
            request_id="req_123",
            version="2.0"
        )
        
        metadata_dict = metadata.model_dump()
        
        assert metadata_dict["request_id"] == "req_123"
        assert metadata_dict["version"] == "2.0"
        assert "timestamp" in metadata_dict
    
    def test_response_metadata_with_execution_time(self):
        """Test métadonnées avec temps d'exécution"""
        metadata = ResponseMetadata(
            execution_time_ms=150.5,
            server_time=datetime.now(timezone.utc)
        )
        
        assert metadata.execution_time_ms == 150.5
        assert metadata.server_time is not None


class TestPaginationMetadata:
    """Tests pour PaginationMetadata"""
    
    def test_pagination_metadata_creation(self):
        """Test création PaginationMetadata"""
        pagination = PaginationMetadata(
            page=2,
            per_page=20,
            total=150,
            total_pages=8
        )
        
        assert pagination.page == 2
        assert pagination.per_page == 20
        assert pagination.total == 150
        assert pagination.total_pages == 8
    
    def test_pagination_metadata_calculated_fields(self):
        """Test champs calculés"""
        pagination = PaginationMetadata(
            page=3,
            per_page=10,
            total=55
        )
        
        # total_pages devrait être calculé automatiquement
        assert pagination.total_pages == 6  # ceil(55/10)
        
        # Propriétés calculées
        assert pagination.has_next is True
        assert pagination.has_previous is True
        assert pagination.offset == 20  # (page-1) * per_page
    
    def test_pagination_metadata_edge_cases(self):
        """Test cas limites"""
        # Première page
        pagination_first = PaginationMetadata(page=1, per_page=10, total=25)
        assert pagination_first.has_previous is False
        assert pagination_first.has_next is True
        
        # Dernière page
        pagination_last = PaginationMetadata(page=3, per_page=10, total=25)
        assert pagination_last.has_previous is True
        assert pagination_last.has_next is False
        
        # Page unique
        pagination_single = PaginationMetadata(page=1, per_page=10, total=5)
        assert pagination_single.has_previous is False
        assert pagination_single.has_next is False
    
    def test_pagination_metadata_validation(self):
        """Test validation des données de pagination"""
        # Page invalide
        with pytest.raises(ValueError):
            PaginationMetadata(page=0, per_page=10, total=50)
        
        # Per_page invalide
        with pytest.raises(ValueError):
            PaginationMetadata(page=1, per_page=0, total=50)
        
        # Total négatif
        with pytest.raises(ValueError):
            PaginationMetadata(page=1, per_page=10, total=-1)


class TestErrorDetail:
    """Tests pour ErrorDetail"""
    
    def test_error_detail_creation(self):
        """Test création ErrorDetail"""
        error = ErrorDetail(
            code="VALIDATION_ERROR",
            message="Field is required",
            field="email",
            value=None
        )
        
        assert error.code == "VALIDATION_ERROR"
        assert error.message == "Field is required"
        assert error.field == "email"
        assert error.value is None
    
    def test_error_detail_minimal(self):
        """Test ErrorDetail minimal"""
        error = ErrorDetail(
            code="GENERAL_ERROR",
            message="Something went wrong"
        )
        
        assert error.code == "GENERAL_ERROR"
        assert error.message == "Something went wrong"
        assert error.field is None
        assert error.value is None
    
    def test_error_detail_with_context(self):
        """Test ErrorDetail avec contexte"""
        error = ErrorDetail(
            code="DATABASE_ERROR",
            message="Connection failed",
            context={
                "host": "localhost",
                "port": 5432,
                "database": "testdb"
            }
        )
        
        assert error.context["host"] == "localhost"
        assert error.context["port"] == 5432


# =============================================================================
# TESTS DES CLASSES DE RÉPONSE
# =============================================================================

class TestAPIResponse:
    """Tests pour APIResponse (classe de base)"""
    
    def test_api_response_creation(self, sample_data, sample_metadata):
        """Test création APIResponse"""
        response = APIResponse(
            success=True,
            data=sample_data,
            message="Success",
            metadata=sample_metadata
        )
        
        assert response.success is True
        assert response.data == sample_data
        assert response.message == "Success"
        assert response.metadata == sample_metadata
    
    def test_api_response_defaults(self):
        """Test valeurs par défaut"""
        response = APIResponse()
        
        assert response.success is True
        assert response.data is None
        assert response.message is None
        assert response.metadata is not None
        assert isinstance(response.metadata, ResponseMetadata)
    
    def test_api_response_serialization(self, sample_data):
        """Test sérialisation"""
        response = APIResponse(
            success=True,
            data=sample_data,
            message="Test message"
        )
        
        response_dict = response.model_dump()
        
        assert response_dict["success"] is True
        assert response_dict["data"] == sample_data
        assert response_dict["message"] == "Test message"
        assert "metadata" in response_dict
    
    def test_api_response_json_serialization(self, sample_data):
        """Test sérialisation JSON"""
        response = APIResponse(
            success=True,
            data=sample_data
        )
        
        json_str = response.model_dump_json()
        parsed = json.loads(json_str)
        
        assert parsed["success"] is True
        assert parsed["data"]["id"] == "123"


class TestSuccessResponse:
    """Tests pour SuccessResponse"""
    
    def test_success_response_creation(self, sample_data):
        """Test création SuccessResponse"""
        response = SuccessResponse(
            data=sample_data,
            message="Operation successful"
        )
        
        assert response.success is True
        assert response.data == sample_data
        assert response.message == "Operation successful"
    
    def test_success_response_without_data(self):
        """Test SuccessResponse sans données"""
        response = SuccessResponse(
            message="Operation completed"
        )
        
        assert response.success is True
        assert response.data is None
        assert response.message == "Operation completed"
    
    def test_success_response_with_status_code(self, sample_data):
        """Test SuccessResponse avec status code"""
        response = SuccessResponse(
            data=sample_data,
            status_code=201
        )
        
        assert response.success is True
        assert response.status_code == 201


class TestErrorResponse:
    """Tests pour ErrorResponse"""
    
    def test_error_response_creation(self, sample_error_detail):
        """Test création ErrorResponse"""
        response = ErrorResponse(
            message="An error occurred",
            error_code="GENERAL_ERROR",
            error_details=[sample_error_detail]
        )
        
        assert response.success is False
        assert response.message == "An error occurred"
        assert response.error_code == "GENERAL_ERROR"
        assert len(response.error_details) == 1
        assert response.error_details[0] == sample_error_detail
    
    def test_error_response_minimal(self):
        """Test ErrorResponse minimal"""
        response = ErrorResponse(
            message="Error occurred"
        )
        
        assert response.success is False
        assert response.message == "Error occurred"
        assert response.error_code is None
        assert response.error_details == []
    
    def test_error_response_with_status_code(self):
        """Test ErrorResponse avec status code"""
        response = ErrorResponse(
            message="Validation failed",
            error_code="VALIDATION_ERROR",
            status_code=422
        )
        
        assert response.success is False
        assert response.status_code == 422
    
    def test_error_response_multiple_details(self):
        """Test ErrorResponse avec plusieurs détails"""
        details = [
            ErrorDetail(code="FIELD_REQUIRED", message="Email required", field="email"),
            ErrorDetail(code="FIELD_INVALID", message="Invalid format", field="phone")
        ]
        
        response = ErrorResponse(
            message="Validation failed",
            error_details=details
        )
        
        assert len(response.error_details) == 2
        assert response.error_details[0].field == "email"
        assert response.error_details[1].field == "phone"


class TestPaginatedResponse:
    """Tests pour PaginatedResponse"""
    
    def test_paginated_response_creation(self, sample_list_data, sample_pagination):
        """Test création PaginatedResponse"""
        response = PaginatedResponse(
            data=sample_list_data,
            pagination=sample_pagination,
            message="Data retrieved"
        )
        
        assert response.success is True
        assert response.data == sample_list_data
        assert response.pagination == sample_pagination
        assert response.message == "Data retrieved"
    
    def test_paginated_response_empty_data(self, sample_pagination):
        """Test PaginatedResponse avec données vides"""
        response = PaginatedResponse(
            data=[],
            pagination=sample_pagination
        )
        
        assert response.success is True
        assert response.data == []
        assert response.pagination == sample_pagination
    
    def test_paginated_response_serialization(self, sample_list_data, sample_pagination):
        """Test sérialisation PaginatedResponse"""
        response = PaginatedResponse(
            data=sample_list_data,
            pagination=sample_pagination
        )
        
        response_dict = response.model_dump()
        
        assert "data" in response_dict
        assert "pagination" in response_dict
        assert response_dict["pagination"]["page"] == 1
        assert response_dict["pagination"]["total"] == 100


# =============================================================================
# TESTS DES FONCTIONS DE CRÉATION
# =============================================================================

class TestResponseCreationFunctions:
    """Tests pour les fonctions de création de réponse"""
    
    def test_create_success_response(self, sample_data):
        """Test create_success_response"""
        response = create_success_response(
            data=sample_data,
            message="Success",
            status_code=200
        )
        
        assert isinstance(response, JSONResponse)
        assert response.status_code == 200
        
        # Vérifier le contenu
        content = json.loads(response.body)
        assert content["success"] is True
        assert content["data"]["id"] == "123"
        assert content["message"] == "Success"
    
    def test_create_success_response_no_data(self):
        """Test create_success_response sans données"""
        response = create_success_response(
            message="Operation completed"
        )
        
        assert isinstance(response, JSONResponse)
        content = json.loads(response.body)
        assert content["success"] is True
        assert content["data"] is None
    
    def test_create_error_response(self):
        """Test create_error_response"""
        response = create_error_response(
            message="Error occurred",
            error_code="TEST_ERROR",
            status_code=400
        )
        
        assert isinstance(response, JSONResponse)
        assert response.status_code == 400
        
        content = json.loads(response.body)
        assert content["success"] is False
        assert content["message"] == "Error occurred"
        assert content["error_code"] == "TEST_ERROR"
    
    def test_create_error_response_with_details(self):
        """Test create_error_response avec détails"""
        details = [
            ErrorDetail(code="FIELD_REQUIRED", message="Email required", field="email")
        ]
        
        response = create_error_response(
            message="Validation failed",
            error_details=details,
            status_code=422
        )
        
        content = json.loads(response.body)
        assert len(content["error_details"]) == 1
        assert content["error_details"][0]["field"] == "email"
    
    def test_create_paginated_response(self, sample_list_data, sample_pagination):
        """Test create_paginated_response"""
        response = create_paginated_response(
            data=sample_list_data,
            pagination=sample_pagination,
            message="Data retrieved"
        )
        
        assert isinstance(response, JSONResponse)
        content = json.loads(response.body)
        
        assert content["success"] is True
        assert len(content["data"]) == 3
        assert content["pagination"]["page"] == 1
        assert content["pagination"]["total"] == 100
    
    def test_create_paginated_response_empty(self):
        """Test create_paginated_response avec données vides"""
        pagination = PaginationMetadata(page=1, per_page=10, total=0, total_pages=0)
        
        response = create_paginated_response(
            data=[],
            pagination=pagination
        )
        
        content = json.loads(response.body)
        assert content["data"] == []
        assert content["pagination"]["total"] == 0


# =============================================================================
# TESTS DES FONCTIONS UTILITAIRES
# =============================================================================

class TestResponseUtilities:
    """Tests pour les fonctions utilitaires"""
    
    def test_format_response_data(self, sample_data):
        """Test format_response_data"""
        formatted = format_response_data(sample_data)
        
        # Les données devraient être formatées pour JSON
        assert isinstance(formatted, dict)
        assert formatted["id"] == "123"
        assert formatted["value"] == 42.5
    
    def test_format_response_data_with_datetime(self):
        """Test formatage avec datetime"""
        data = {
            "id": "123",
            "created_at": datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        }
        
        formatted = format_response_data(data)
        
        # Les datetimes devraient être formatées
        assert isinstance(formatted["created_at"], str)
        assert isinstance(formatted["updated_at"], str)
    
    def test_format_response_data_with_decimal(self):
        """Test formatage avec Decimal"""
        data = {
            "id": "123",
            "price": Decimal("19.99"),
            "tax": Decimal("2.50")
        }
        
        formatted = format_response_data(data)
        
        # Les Decimals devraient être convertis en float
        assert isinstance(formatted["price"], float)
        assert formatted["price"] == 19.99
    
    def test_format_response_data_nested(self):
        """Test formatage avec structures imbriquées"""
        data = {
            "user": {
                "id": "123",
                "profile": {
                    "name": "Test User",
                    "created_at": datetime.now(timezone.utc)
                }
            },
            "items": [
                {"id": "1", "price": Decimal("10.50")},
                {"id": "2", "price": Decimal("15.75")}
            ]
        }
        
        formatted = format_response_data(data)
        
        # Vérifier le formatage récursif
        assert isinstance(formatted["user"]["profile"]["created_at"], str)
        assert isinstance(formatted["items"][0]["price"], float)
        assert formatted["items"][0]["price"] == 10.50
    
    def test_enrich_response_metadata(self, mock_request):
        """Test enrich_response_metadata"""
        metadata = ResponseMetadata()
        
        enriched = enrich_response_metadata(metadata, mock_request)
        
        # Les métadonnées devraient être enrichies avec les infos de la requête
        assert enriched.request_path == "/api/v1/test"
        assert enriched.request_method == "GET"
        assert "TestClient/1.0" in enriched.user_agent
    
    def test_validate_response_data(self, sample_data):
        """Test validate_response_data"""
        # Données valides
        assert validate_response_data(sample_data) is True
        
        # Données avec types non sérialisables
        invalid_data = {
            "id": "123",
            "func": lambda x: x,  # Non sérialisable
            "obj": object()  # Non sérialisable
        }
        
        assert validate_response_data(invalid_data) is False
    
    def test_validate_response_data_edge_cases(self):
        """Test validation cas limites"""
        # None
        assert validate_response_data(None) is True
        
        # Liste vide
        assert validate_response_data([]) is True
        
        # Dict vide
        assert validate_response_data({}) is True
        
        # Types de base
        assert validate_response_data("string") is True
        assert validate_response_data(123) is True
        assert validate_response_data(45.67) is True
        assert validate_response_data(True) is True


# =============================================================================
# TESTS DU RESPONSEBUILDER
# =============================================================================

class TestResponseBuilder:
    """Tests pour ResponseBuilder (Builder pattern)"""
    
    def test_response_builder_success(self, sample_data):
        """Test ResponseBuilder pour succès"""
        builder = ResponseBuilder()
        
        response = (builder
                   .success()
                   .data(sample_data)
                   .message("Operation successful")
                   .status_code(200)
                   .build())
        
        assert isinstance(response, JSONResponse)
        assert response.status_code == 200
        
        content = json.loads(response.body)
        assert content["success"] is True
        assert content["data"]["id"] == "123"
    
    def test_response_builder_error(self):
        """Test ResponseBuilder pour erreur"""
        builder = ResponseBuilder()
        
        response = (builder
                   .error()
                   .message("Error occurred")
                   .error_code("TEST_ERROR")
                   .status_code(400)
                   .build())
        
        assert response.status_code == 400
        content = json.loads(response.body)
        assert content["success"] is False
    
    def test_response_builder_paginated(self, sample_list_data, sample_pagination):
        """Test ResponseBuilder pour réponse paginée"""
        builder = ResponseBuilder()
        
        response = (builder
                   .success()
                   .data(sample_list_data)
                   .pagination(sample_pagination)
                   .build())
        
        content = json.loads(response.body)
        assert "pagination" in content
        assert content["pagination"]["page"] == 1
    
    def test_response_builder_with_metadata(self, sample_data, sample_metadata):
        """Test ResponseBuilder avec métadonnées"""
        builder = ResponseBuilder()
        
        response = (builder
                   .success()
                   .data(sample_data)
                   .metadata(sample_metadata)
                   .build())
        
        content = json.loads(response.body)
        assert content["metadata"]["request_id"] == "req_123"
    
    def test_response_builder_chain_validation(self):
        """Test validation du chaînage"""
        builder = ResponseBuilder()
        
        # Tenter de construire sans définir le type
        with pytest.raises(ValueError, match="Response type not set"):
            builder.build()
        
        # Tenter d'utiliser error() après success()
        builder.success()
        with pytest.raises(ValueError, match="Response type already set"):
            builder.error()


# =============================================================================
# TESTS D'INTÉGRATION
# =============================================================================

@pytest.mark.integration
class TestResponseIntegration:
    """Tests d'intégration pour les réponses"""
    
    def test_success_endpoint_integration(self, test_app):
        """Test endpoint de succès"""
        with TestClient(test_app) as client:
            response = client.get("/success")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["data"]["message"] == "Success"
            assert data["message"] == "Operation successful"
            assert "metadata" in data
    
    def test_error_endpoint_integration(self, test_app):
        """Test endpoint d'erreur"""
        with TestClient(test_app) as client:
            response = client.get("/error")
            
            assert response.status_code == 400
            data = response.json()
            
            assert data["success"] is False
            assert data["message"] == "Test error"
            assert data["error_code"] == "TEST_ERROR"
    
    def test_paginated_endpoint_integration(self, test_app):
        """Test endpoint paginé"""
        with TestClient(test_app) as client:
            response = client.get("/paginated")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert len(data["data"]) == 5
            assert "pagination" in data
            assert data["pagination"]["page"] == 1
            assert data["pagination"]["total"] == 50
    
    def test_response_headers_integration(self, test_app):
        """Test headers de réponse"""
        with TestClient(test_app) as client:
            response = client.get("/success")
            
            # Vérifier les headers standards
            assert response.headers["content-type"] == "application/json"
            
            # Vérifier les headers personnalisés si présents
            if "X-Request-ID" in response.headers:
                assert len(response.headers["X-Request-ID"]) > 0


# =============================================================================
# TESTS DE PERFORMANCE
# =============================================================================

@pytest.mark.performance
class TestResponsePerformance:
    """Tests de performance pour les réponses"""
    
    def test_response_creation_performance(self, benchmark, sample_data):
        """Test performance création de réponse"""
        def create_response():
            return create_success_response(
                data=sample_data,
                message="Test message"
            )
        
        result = benchmark(create_response)
        assert isinstance(result, JSONResponse)
    
    def test_response_serialization_performance(self, benchmark, sample_data):
        """Test performance sérialisation"""
        response = SuccessResponse(data=sample_data)
        
        def serialize_response():
            return response.model_dump_json()
        
        result = benchmark(serialize_response)
        assert isinstance(result, str)
    
    def test_large_data_response_performance(self, benchmark):
        """Test performance avec grandes données"""
        # Créer un grand dataset
        large_data = [
            {"id": i, "name": f"Item {i}", "value": i * 1.5}
            for i in range(1000)
        ]
        
        def create_large_response():
            return create_success_response(data=large_data)
        
        result = benchmark(create_large_response)
        assert isinstance(result, JSONResponse)
    
    def test_nested_data_performance(self, benchmark):
        """Test performance avec données imbriquées"""
        nested_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "items": [{"id": i, "data": f"value_{i}"} for i in range(100)]
                    }
                }
            }
        }
        
        def format_nested_data():
            return format_response_data(nested_data)
        
        result = benchmark(format_nested_data)
        assert isinstance(result, dict)


# =============================================================================
# TESTS DE SÉCURITÉ
# =============================================================================

@pytest.mark.security
class TestResponseSecurity:
    """Tests de sécurité pour les réponses"""
    
    def test_sensitive_data_filtering(self):
        """Test filtrage des données sensibles"""
        sensitive_data = {
            "id": "123",
            "username": "testuser",
            "password": "secret123",
            "api_key": "sk_test_123456",
            "token": "bearer_token_secret"
        }
        
        # Les données sensibles devraient être filtrées
        formatted = format_response_data(sensitive_data, filter_sensitive=True)
        
        assert "password" not in formatted
        assert "api_key" not in formatted
        assert "token" not in formatted
        assert formatted["username"] == "testuser"  # Champ sûr
    
    def test_xss_prevention_in_responses(self):
        """Test prévention XSS dans les réponses"""
        malicious_data = {
            "name": "<script>alert('xss')</script>",
            "description": "javascript:alert('xss')",
            "comment": "<img src=x onerror=alert('xss')>"
        }
        
        response = create_success_response(data=malicious_data)
        content = json.loads(response.body)
        
        # Le contenu malicieux devrait être échappé ou nettoyé
        # (Cette logique devrait être implémentée dans format_response_data)
    
    def test_response_size_limits(self):
        """Test limites de taille de réponse"""
        # Créer des données très volumineuses
        huge_data = {
            "items": ["x" * 10000 for _ in range(1000)]  # ~10MB
        }
        
        # La réponse ne devrait pas dépasser certaines limites
        # (Cette logique devrait être implémentée avec des limits)
    
    def test_error_information_disclosure(self):
        """Test divulgation d'informations dans les erreurs"""
        # En production, les erreurs ne devraient pas exposer de détails internes
        with patch('app.api.core.config.get_api_config') as mock_config:
            mock_config.return_value.debug = False
            
            response = create_error_response(
                message="Database connection failed: host=secret_host",
                error_code="DATABASE_ERROR"
            )
            
            content = json.loads(response.body)
            
            # Les détails internes ne devraient pas être exposés en production
            assert "secret_host" not in content["message"]


# =============================================================================
# TESTS DE VALIDATION
# =============================================================================

@pytest.mark.validation
class TestResponseValidation:
    """Tests de validation pour les réponses"""
    
    def test_response_schema_validation(self, sample_data):
        """Test validation du schéma de réponse"""
        response = SuccessResponse(data=sample_data)
        
        # La réponse devrait être valide selon le schéma Pydantic
        assert response.model_validate(response.model_dump())
    
    def test_pagination_validation(self):
        """Test validation de la pagination"""
        # Pagination valide
        valid_pagination = PaginationMetadata(page=1, per_page=10, total=50)
        response = PaginatedResponse(data=[], pagination=valid_pagination)
        
        assert response.pagination.total_pages == 5
        
        # Pagination invalide devrait lever une erreur
        with pytest.raises(ValueError):
            PaginationMetadata(page=0, per_page=10, total=50)
    
    def test_error_detail_validation(self):
        """Test validation des détails d'erreur"""
        # Détail valide
        valid_detail = ErrorDetail(
            code="VALIDATION_ERROR",
            message="Field is required"
        )
        
        response = ErrorResponse(
            message="Validation failed",
            error_details=[valid_detail]
        )
        
        assert len(response.error_details) == 1
        assert response.error_details[0].code == "VALIDATION_ERROR"
    
    def test_metadata_validation(self):
        """Test validation des métadonnées"""
        # Métadonnées valides
        metadata = ResponseMetadata(
            request_id="req_123",
            version="1.0"
        )
        
        response = APIResponse(metadata=metadata)
        assert response.metadata.request_id == "req_123"
        
        # Les métadonnées par défaut devraient être valides
        default_response = APIResponse()
        assert default_response.metadata is not None
        assert default_response.metadata.version is not None
