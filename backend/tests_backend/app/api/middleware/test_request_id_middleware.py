"""
Tests Ultra-Avancés pour Request ID Middleware Enterprise
=======================================================

Tests industriels complets pour gestion d'identifiants de requête avec
traçabilité distribuée, corrélation multi-services, et patterns enterprise.

Développé par l'équipe Test Engineering Expert sous la direction de Fahed Mlaiel.
Architecture: Enterprise Request Tracing Framework avec UUID optimisé.
"""

import pytest
import asyncio
import uuid
import time
import json
import hashlib
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
import re
from concurrent.futures import ThreadPoolExecutor
import threading


# =============================================================================
# TESTS FONCTIONNELS ENTERPRISE REQUEST ID
# =============================================================================

def test_request_id_generation():
    """Test de génération d'identifiants de requête."""
    # Test de génération UUID4 standard
    request_ids = set()
    
    for _ in range(1000):
        request_id = str(uuid.uuid4())
        
        # Vérifications de format
        assert isinstance(request_id, str)
        assert len(request_id) == 36  # Format UUID standard
        assert request_id.count('-') == 4  # 4 tirets
        
        # Pattern UUID4
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
        assert re.match(uuid_pattern, request_id), f"Invalid UUID4 format: {request_id}"
        
        # Unicité
        assert request_id not in request_ids, f"Duplicate UUID generated: {request_id}"
        request_ids.add(request_id)
    
    # Vérifications statistiques d'unicité
    assert len(request_ids) == 1000  # Tous uniques

def test_request_id_custom_formats():
    """Test de formats personnalisés d'identifiants."""
    # Format avec préfixe temporel
    def generate_timestamped_id():
        timestamp = int(time.time() * 1000)  # Millisecondes
        random_part = uuid.uuid4().hex[:8]
        return f"req_{timestamp}_{random_part}"
    
    # Format avec hash de contenu
    def generate_content_based_id(content: str):
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        timestamp = int(time.time() * 1000)
        return f"req_{timestamp}_{content_hash}"
    
    # Format hiérarchique
    def generate_hierarchical_id(service: str, component: str):
        base_id = uuid.uuid4().hex[:12]
        return f"{service}.{component}.{base_id}"
    
    # Test des formats personnalisés
    timestamped_ids = [generate_timestamped_id() for _ in range(100)]
    content_ids = [generate_content_based_id(f"content_{i}") for i in range(100)]
    hierarchical_ids = [generate_hierarchical_id("api", "auth") for _ in range(100)]
    
    # Vérifications format timestamped
    for req_id in timestamped_ids:
        assert req_id.startswith("req_")
        parts = req_id.split("_")
        assert len(parts) == 3
        assert parts[1].isdigit()  # Timestamp
        assert len(parts[2]) == 8  # Random part
    
    # Vérifications format content-based
    for req_id in content_ids:
        assert req_id.startswith("req_")
        parts = req_id.split("_")
        assert len(parts) == 3
        assert len(parts[2]) == 16  # Hash part
    
    # Vérifications format hiérarchique
    for req_id in hierarchical_ids:
        parts = req_id.split(".")
        assert len(parts) == 3
        assert parts[0] == "api"
        assert parts[1] == "auth"
        assert len(parts[2]) == 12
    
    # Unicité dans chaque format
    assert len(set(timestamped_ids)) == len(timestamped_ids)
    assert len(set(content_ids)) == len(content_ids)
    assert len(set(hierarchical_ids)) == len(hierarchical_ids)

def test_request_id_propagation():
    """Test de propagation d'identifiants entre services."""
    # Simulation d'une requête multi-services
    class ServiceRequest:
        def __init__(self, request_id: str, service_name: str, parent_id: Optional[str] = None):
            self.request_id = request_id
            self.service_name = service_name
            self.parent_id = parent_id
            self.timestamp = datetime.now()
            self.children: List['ServiceRequest'] = []
        
        def create_child_request(self, child_service: str) -> 'ServiceRequest':
            """Crée une requête enfant pour un service downstream."""
            child_id = f"{self.request_id}.{uuid.uuid4().hex[:8]}"
            child_request = ServiceRequest(child_id, child_service, self.request_id)
            self.children.append(child_request)
            return child_request
        
        def get_trace_chain(self) -> List[str]:
            """Retourne la chaîne de traçage complète."""
            chain = [self.request_id]
            for child in self.children:
                chain.extend(child.get_trace_chain())
            return chain
    
    # Simulation d'un flow de requêtes
    root_request = ServiceRequest("req_001", "api-gateway")
    
    # Première branche: API Gateway -> Auth Service -> User Service
    auth_request = root_request.create_child_request("auth-service")
    user_request = auth_request.create_child_request("user-service")
    
    # Deuxième branche: API Gateway -> Product Service -> Inventory Service
    product_request = root_request.create_child_request("product-service")
    inventory_request = product_request.create_child_request("inventory-service")
    
    # Troisième branche: Product Service -> Price Service
    price_request = product_request.create_child_request("price-service")
    
    # Vérifications de structure
    assert len(root_request.children) == 2  # auth et product
    assert len(auth_request.children) == 1  # user
    assert len(product_request.children) == 2  # inventory et price
    
    # Vérifications de chaîne de traçage
    full_trace = root_request.get_trace_chain()
    expected_services = 6  # root + auth + user + product + inventory + price
    assert len(full_trace) == expected_services
    
    # Vérifications des relations parent-enfant
    assert auth_request.parent_id == root_request.request_id
    assert user_request.parent_id == auth_request.request_id
    assert product_request.parent_id == root_request.request_id
    assert inventory_request.parent_id == product_request.request_id
    assert price_request.parent_id == product_request.request_id
    
    # Vérifications de format des IDs enfants
    for child in root_request.children:
        assert child.request_id.startswith(root_request.request_id + ".")
        
        for grandchild in child.children:
            assert grandchild.request_id.startswith(child.request_id + ".")

def test_request_id_correlation():
    """Test de corrélation d'identifiants entre systèmes."""
    # Simulation de corrélation multi-systèmes
    class CorrelationManager:
        def __init__(self):
            self.correlation_map: Dict[str, Set[str]] = {}
            self.request_metadata: Dict[str, Dict[str, Any]] = {}
        
        def add_correlation(self, request_id: str, external_id: str, system: str):
            """Ajoute une corrélation avec un système externe."""
            if request_id not in self.correlation_map:
                self.correlation_map[request_id] = set()
            
            correlation_key = f"{system}:{external_id}"
            self.correlation_map[request_id].add(correlation_key)
            
            # Métadonnées
            if request_id not in self.request_metadata:
                self.request_metadata[request_id] = {}
            
            self.request_metadata[request_id][f"{system}_id"] = external_id
            self.request_metadata[request_id][f"{system}_timestamp"] = datetime.now()
        
        def get_correlations(self, request_id: str) -> Set[str]:
            """Récupère toutes les corrélations pour un request ID."""
            return self.correlation_map.get(request_id, set())
        
        def find_by_external_id(self, external_id: str, system: str) -> List[str]:
            """Trouve les request IDs corrélés à un ID externe."""
            correlation_key = f"{system}:{external_id}"
            results = []
            
            for req_id, correlations in self.correlation_map.items():
                if correlation_key in correlations:
                    results.append(req_id)
            
            return results
    
    # Test de corrélation
    manager = CorrelationManager()
    
    # Requête principale
    main_request_id = str(uuid.uuid4())
    
    # Corrélations avec différents systèmes
    correlations_data = [
        ("payment_gateway", "pay_123456789"),
        ("notification_service", "notif_abc123"),
        ("audit_system", "audit_xyz789"),
        ("analytics", "track_456def"),
        ("external_api", "ext_789ghi")
    ]
    
    for system, external_id in correlations_data:
        manager.add_correlation(main_request_id, external_id, system)
    
    # Vérifications
    correlations = manager.get_correlations(main_request_id)
    assert len(correlations) == len(correlations_data)
    
    # Vérifier chaque corrélation
    for system, external_id in correlations_data:
        expected_key = f"{system}:{external_id}"
        assert expected_key in correlations
        
        # Test de recherche inverse
        found_requests = manager.find_by_external_id(external_id, system)
        assert main_request_id in found_requests
    
    # Test de métadonnées
    metadata = manager.request_metadata[main_request_id]
    for system, external_id in correlations_data:
        assert f"{system}_id" in metadata
        assert metadata[f"{system}_id"] == external_id
        assert f"{system}_timestamp" in metadata
    
    # Test de corrélations multiples
    second_request_id = str(uuid.uuid4())
    manager.add_correlation(second_request_id, "pay_123456789", "payment_gateway")  # Même payment ID
    
    # Les deux requêtes devraient être liées par le payment ID
    payment_requests = manager.find_by_external_id("pay_123456789", "payment_gateway")
    assert len(payment_requests) == 2
    assert main_request_id in payment_requests
    assert second_request_id in payment_requests

def test_request_id_lifecycle_management():
    """Test de gestion du cycle de vie des identifiants."""
    class RequestLifecycleManager:
        def __init__(self):
            self.active_requests: Dict[str, Dict[str, Any]] = {}
            self.completed_requests: Dict[str, Dict[str, Any]] = {}
            self.failed_requests: Dict[str, Dict[str, Any]] = {}
        
        def start_request(self, request_id: str, endpoint: str, method: str):
            """Démarre le suivi d'une requête."""
            self.active_requests[request_id] = {
                'endpoint': endpoint,
                'method': method,
                'start_time': datetime.now(),
                'status': 'active',
                'operations': [],
                'resources_used': []
            }
        
        def add_operation(self, request_id: str, operation: str, duration_ms: float):
            """Ajoute une opération à la requête."""
            if request_id in self.active_requests:
                self.active_requests[request_id]['operations'].append({
                    'operation': operation,
                    'duration_ms': duration_ms,
                    'timestamp': datetime.now()
                })
        
        def complete_request(self, request_id: str, status_code: int, response_size: int):
            """Marque une requête comme terminée."""
            if request_id in self.active_requests:
                request_data = self.active_requests.pop(request_id)
                request_data.update({
                    'status': 'completed',
                    'end_time': datetime.now(),
                    'status_code': status_code,
                    'response_size': response_size,
                    'total_duration_ms': (datetime.now() - request_data['start_time']).total_seconds() * 1000
                })
                self.completed_requests[request_id] = request_data
        
        def fail_request(self, request_id: str, error_type: str, error_message: str):
            """Marque une requête comme échouée."""
            if request_id in self.active_requests:
                request_data = self.active_requests.pop(request_id)
                request_data.update({
                    'status': 'failed',
                    'end_time': datetime.now(),
                    'error_type': error_type,
                    'error_message': error_message,
                    'total_duration_ms': (datetime.now() - request_data['start_time']).total_seconds() * 1000
                })
                self.failed_requests[request_id] = request_data
        
        def get_active_count(self) -> int:
            """Retourne le nombre de requêtes actives."""
            return len(self.active_requests)
        
        def cleanup_old_requests(self, max_age_hours: int = 24):
            """Nettoie les anciennes requêtes terminées."""
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            # Nettoyer les requêtes terminées
            old_completed = [
                req_id for req_id, data in self.completed_requests.items()
                if data['end_time'] < cutoff_time
            ]
            
            # Nettoyer les requêtes échouées
            old_failed = [
                req_id for req_id, data in self.failed_requests.items()
                if data['end_time'] < cutoff_time
            ]
            
            for req_id in old_completed:
                del self.completed_requests[req_id]
            
            for req_id in old_failed:
                del self.failed_requests[req_id]
            
            return len(old_completed) + len(old_failed)
    
    # Test du cycle de vie
    manager = RequestLifecycleManager()
    
    # Créer plusieurs requêtes
    request_ids = [str(uuid.uuid4()) for _ in range(5)]
    
    # Démarrer les requêtes
    for i, req_id in enumerate(request_ids):
        manager.start_request(req_id, f"/api/endpoint{i}", "GET")
    
    assert manager.get_active_count() == 5
    
    # Ajouter des opérations
    for req_id in request_ids[:3]:
        manager.add_operation(req_id, "database_query", 50.5)
        manager.add_operation(req_id, "cache_lookup", 5.2)
        manager.add_operation(req_id, "response_serialization", 12.8)
    
    # Terminer quelques requêtes avec succès
    for req_id in request_ids[:2]:
        manager.complete_request(req_id, 200, 1024)
    
    # Faire échouer une requête
    manager.fail_request(request_ids[2], "ValidationError", "Invalid input parameter")
    
    # Vérifications
    assert manager.get_active_count() == 2  # 2 encore actives
    assert len(manager.completed_requests) == 2
    assert len(manager.failed_requests) == 1
    
    # Vérifier les données des requêtes terminées
    for req_id in request_ids[:2]:
        completed_data = manager.completed_requests[req_id]
        assert completed_data['status'] == 'completed'
        assert completed_data['status_code'] == 200
        assert completed_data['response_size'] == 1024
        assert len(completed_data['operations']) == 3
        assert completed_data['total_duration_ms'] > 0
    
    # Vérifier les données de la requête échouée
    failed_data = manager.failed_requests[request_ids[2]]
    assert failed_data['status'] == 'failed'
    assert failed_data['error_type'] == 'ValidationError'
    assert 'Invalid input' in failed_data['error_message']
    
    # Test de nettoyage (simuler des anciennes requêtes)
    # Modifier manuellement les timestamps pour simuler des anciennes requêtes
    old_time = datetime.now() - timedelta(hours=25)
    for req_id in request_ids[:2]:
        manager.completed_requests[req_id]['end_time'] = old_time
    
    cleaned_count = manager.cleanup_old_requests(max_age_hours=24)
    assert cleaned_count == 2
    assert len(manager.completed_requests) == 0


# Tests automatiques pour rétrocompatibilité
def test_requestidformat_class():
    # Test des valeurs Enum RequestIdFormat
    try:
        from backend.app.api.middleware import request_id_middleware
        RequestIdFormat = getattr(request_id_middleware, 'RequestIdFormat')
        
        # Test des valeurs enum disponibles
        values = list(RequestIdFormat)
        assert len(values) > 0, "L'enum doit avoir au moins une valeur"
        
        # Test instanciation avec première valeur
        if values:
            instance = RequestIdFormat(values[0].value)
            assert instance == values[0]
    except Exception as exc:
        pytest.fail('Erreur lors du test RequestIdFormat : {}'.format(exc))

def test_requestcontext_class():
    # Test avec request_id requis
    try:
        from backend.app.api.middleware import request_id_middleware
        RequestContext = getattr(request_id_middleware, 'RequestContext')
        
        obj = RequestContext(request_id="test-123")
        assert obj is not None
        assert obj.request_id == "test-123"
    except Exception as exc:
        pytest.fail('Erreur lors du test RequestContext : {}'.format(exc))

def test_requestmetrics_class():
    # Instanciation réelle
    try:
        from backend.app.api.middleware import request_id_middleware
        obj = getattr(request_id_middleware, 'RequestMetrics')(
            request_id="test-123",
            endpoint="/test",
            method="GET",
            duration=0.1,
            status_code=200
        )
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_requestidgenerator_class():
    # Instanciation réelle
    try:
        from backend.app.api.middleware import request_id_middleware
        obj = getattr(request_id_middleware, 'RequestIdGenerator')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_requesttracker_class():
    # Instanciation réelle
    try:
        from backend.app.api.middleware import request_id_middleware
        obj = getattr(request_id_middleware, 'RequestTracker')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_requestcorrelation_class():
    # Instanciation réelle
    try:
        from backend.app.api.middleware import request_id_middleware
        obj = getattr(request_id_middleware, 'RequestCorrelation')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_requestidmetrics_class():
    # Instanciation réelle
    try:
        from backend.app.api.middleware import request_id_middleware
        from unittest.mock import patch
        # Mock the Prometheus registry to avoid duplicates
        with patch('prometheus_client.registry.REGISTRY.register'):
            obj = getattr(request_id_middleware, 'RequestIdMetrics')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_advancedrequestidmiddleware_class():
    # Instanciation réelle
    try:
        from backend.app.api.middleware import request_id_middleware
        from fastapi import FastAPI
        mock_app = FastAPI()
        obj = getattr(request_id_middleware, 'AdvancedRequestIdMiddleware')(
            mock_app, enable_metrics=False
        )
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_create_request_id_middleware():
    # Appel réel de la fonction
    result = None
    try:
        from backend.app.api.middleware import request_id_middleware
        from fastapi import FastAPI
        mock_app = FastAPI()
        result = getattr(request_id_middleware, 'create_request_id_middleware')(
            mock_app, enable_metrics=False
        )
    except Exception as exc:
        pytest.fail('Erreur lors de l\'appel réel : {}'.format(exc))
    assert result is not None

def test_request_context_decorator():
    # Appel réel de la fonction
    result = None
    try:
        from backend.app.api.middleware import request_id_middleware
        result = getattr(request_id_middleware, 'request_context_decorator')()
    except Exception as exc:
        pytest.fail('Erreur lors de l\'appel réel : {}'.format(exc))
    assert result is not None

def test_get_current_request_id():
    # Appel réel de la fonction
    result = None
    try:
        from backend.app.api.middleware import request_id_middleware
        result = getattr(request_id_middleware, 'get_current_request_id')()
    except Exception as exc:
        pytest.fail('Erreur lors de l\'appel réel : {}'.format(exc))
    # Note: get_current_request_id() returns None when no request context exists, which is expected in tests
    assert result is None or isinstance(result, str)

