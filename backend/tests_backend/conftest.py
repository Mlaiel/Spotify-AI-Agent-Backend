"""
Enterprise Test Configuration for Spotify AI Agent
===============================================

Configuration centrale pour tous les tests enterprise avec fixtures avancées,
mocks intelligents, et infrastructure de test complète.

Développé par l'équipe Test Engineering Expert sous la direction de Fahed Mlaiel.
"""

# Configuration pytest plugins
pytest_plugins = [
    "pytest_asyncio",
    "pytest_benchmark", 
    "pytest_mock",
    "pytest_cov"
]

import asyncio
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import tempfile
import os
import json
import sys

# Ajouter le répertoire racine au PATH pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Nettoyer les registres de métriques pour chaque test
@pytest.fixture(autouse=True)
def clear_prometheus_registry():
    """Nettoie automatiquement le registre Prometheus entre les tests."""
    try:
        from prometheus_client import REGISTRY
        collectors = list(REGISTRY._collector_to_names.keys())
        for collector in collectors:
            try:
                REGISTRY.unregister(collector)
            except KeyError:
                pass  # Le collector n'était pas enregistré
    except ImportError:
        pass  # Prometheus non disponible
    
    # Nettoyer notre gestionnaire de métriques
    try:
        from app.utils.metrics_manager import clear_metrics
        clear_metrics()
    except ImportError:
        pass
    
    yield  # Exécuter le test
    
    # Nettoyage après le test
    try:
        from prometheus_client import REGISTRY
        collectors = list(REGISTRY._collector_to_names.keys())
        for collector in collectors:
            try:
                REGISTRY.unregister(collector)
            except KeyError:
                pass
    except ImportError:
        pass
    
    try:
        from app.utils.metrics_manager import clear_metrics
        clear_metrics()
    except ImportError:
        pass

@pytest.fixture
def test_registry():
    """Fournit un registre de métriques propre pour les tests."""
    try:
        from prometheus_client import CollectorRegistry
        return CollectorRegistry()
    except ImportError:
        return None

# =============================================================================
# CORE FIXTURES - Infrastructure de base
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Event loop asyncio pour tests asynchrones."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def temp_dir():
    """Répertoire temporaire pour les tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def mock_datetime():
    """Mock datetime pour tests déterministes."""
    with patch('datetime.datetime') as mock_dt:
        mock_dt.utcnow.return_value = datetime(2024, 1, 15, 10, 30, 0)
        mock_dt.now.return_value = datetime(2024, 1, 15, 10, 30, 0)
        yield mock_dt

# =============================================================================
# DATABASE FIXTURES - Mocks base de données enterprise
# =============================================================================

@pytest.fixture(scope="session")
def db_engine():
    """Mock SQLAlchemy engine avec méthodes avancées."""
    engine = MagicMock(name="db_engine_mock")
    engine.execute.return_value = MagicMock()
    engine.begin.return_value.__enter__ = MagicMock()
    engine.begin.return_value.__exit__ = MagicMock()
    return engine

@pytest.fixture(scope="session")
def pg_engine():
    """Mock PostgreSQL engine avec fonctionnalités spécialisées."""
    engine = MagicMock(name="pg_engine_mock")
    engine.execute.return_value.fetchall.return_value = []
    engine.execute.return_value.fetchone.return_value = None
    return engine

@pytest.fixture
def redis_mock():
    """Mock Redis avec méthodes courantes."""
    redis_mock = MagicMock()
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    redis_mock.delete.return_value = 1
    redis_mock.exists.return_value = False
    redis_mock.expire.return_value = True
    redis_mock.pipeline.return_value = redis_mock
    redis_mock.execute.return_value = []
    return redis_mock

@pytest.fixture
def mongodb_mock():
    """Mock MongoDB avec collections."""
    mongo_mock = MagicMock()
    collection_mock = MagicMock()
    collection_mock.find.return_value = []
    collection_mock.insert_one.return_value.inserted_id = "507f1f77bcf86cd799439011"
    collection_mock.update_one.return_value.modified_count = 1
    collection_mock.delete_one.return_value.deleted_count = 1
    mongo_mock.__getitem__.return_value = collection_mock
    return mongo_mock

# =============================================================================
# ML/AI FIXTURES - Machine Learning et IA
# =============================================================================

@pytest.fixture
def ml_model_mock():
    """Mock modèle ML avec prédictions."""
    model = MagicMock()
    model.predict.return_value = np.array([0.8, 0.6, 0.9, 0.3])
    model.predict_proba.return_value = np.array([[0.2, 0.8], [0.4, 0.6]])
    model.score.return_value = 0.85
    model.feature_importances_ = np.array([0.3, 0.2, 0.4, 0.1])
    return model

@pytest.fixture
def tensorflow_mock():
    """Mock TensorFlow/Keras."""
    tf_mock = MagicMock()
    model_mock = MagicMock()
    model_mock.predict.return_value = np.array([[0.8, 0.2]])
    model_mock.evaluate.return_value = [0.1, 0.95]  # loss, accuracy
    tf_mock.keras.models.load_model.return_value = model_mock
    return tf_mock

@pytest.fixture
def pytorch_mock():
    """Mock PyTorch."""
    torch_mock = MagicMock()
    tensor_mock = MagicMock()
    tensor_mock.numpy.return_value = np.array([0.8, 0.6, 0.9])
    torch_mock.tensor.return_value = tensor_mock
    torch_mock.load.return_value = MagicMock()
    return torch_mock

# =============================================================================
# AUDIO/MUSIC FIXTURES - Données audio et musicales
# =============================================================================

@pytest.fixture
def sample_audio_data():
    """Données audio échantillon pour tests."""
    return {
        'sample_rate': 44100,
        'duration': 180.5,  # 3 minutes
        'audio_array': np.random.random(44100 * 180),
        'format': 'wav',
        'channels': 2,
        'bit_depth': 16
    }

@pytest.fixture
def sample_music_features():
    """Features musicales échantillon."""
    return {
        'mfcc': np.random.random((13, 100)),
        'chroma': np.random.random((12, 100)),
        'spectral_centroid': np.random.random(100),
        'spectral_bandwidth': np.random.random(100),
        'tempo': 120.5,
        'energy': 0.75,
        'valence': 0.68,
        'danceability': 0.82
    }

@pytest.fixture
def sample_user_profile():
    """Profil utilisateur échantillon."""
    return {
        'user_id': 'user_12345',
        'age': 25,
        'country': 'FR',
        'subscription_tier': 'premium',
        'listening_history_length': 15000,
        'favorite_genres': ['electronic', 'indie_pop', 'jazz'],
        'engagement_score': 0.87,
        'last_activity': datetime.utcnow() - timedelta(hours=2)
    }

# =============================================================================
# BUSINESS FIXTURES - Données métier
# =============================================================================

@pytest.fixture
def sample_recommendations():
    """Recommandations échantillon."""
    return [
        {
            'track_id': f'track_{i}',
            'title': f'Song {i}',
            'artist': f'Artist {i}',
            'score': 0.9 - (i * 0.1),
            'explanation': f'Recommended because reason {i}'
        }
        for i in range(1, 6)
    ]

@pytest.fixture
def sample_business_metrics():
    """Métriques business échantillon."""
    return {
        'dau': 856000,
        'session_duration_avg': 32.4,
        'conversion_rate': 0.35,
        'churn_rate': 0.046,
        'revenue_per_user': 7.95,
        'engagement_score': 0.84
    }

# =============================================================================
# SECURITY FIXTURES - Sécurité et cryptographie
# =============================================================================

@pytest.fixture
def crypto_keys():
    """Clés cryptographiques pour tests."""
    return {
        'aes_key': b'0123456789ABCDEF0123456789ABCDEF',
        'rsa_private_key': 'mock_rsa_private_key',
        'rsa_public_key': 'mock_rsa_public_key',
        'ecc_private_key': 'mock_ecc_private_key',
        'ecc_public_key': 'mock_ecc_public_key'
    }

@pytest.fixture
def sample_encrypted_data():
    """Données chiffrées échantillon."""
    return {
        'ciphertext': b'encrypted_data_mock',
        'iv': b'initialization_vector',
        'auth_tag': b'authentication_tag',
        'algorithm': 'aes-256-gcm'
    }

# =============================================================================
# COMPLIANCE FIXTURES - Conformité et audit
# =============================================================================

@pytest.fixture
def gdpr_request_sample():
    """Demande GDPR échantillon."""
    return {
        'request_id': str(uuid.uuid4()),
        'request_type': 'access',
        'data_subject_id': 'user_12345',
        'timestamp': datetime.utcnow(),
        'verification_status': 'verified',
        'scope': 'all_personal_data'
    }

@pytest.fixture
def compliance_audit_data():
    """Données audit compliance."""
    return {
        'gdpr_score': 0.96,
        'ccpa_score': 0.92,
        'iso27001_score': 0.94,
        'data_subject_requests': 187,
        'privacy_incidents': 0,
        'audit_date': datetime.utcnow()
    }

# =============================================================================
# MONITORING FIXTURES - Observabilité et monitoring
# =============================================================================

@pytest.fixture
def prometheus_mock():
    """Mock Prometheus client."""
    prom_mock = MagicMock()
    counter_mock = MagicMock()
    histogram_mock = MagicMock()
    gauge_mock = MagicMock()
    
    prom_mock.Counter.return_value = counter_mock
    prom_mock.Histogram.return_value = histogram_mock
    prom_mock.Gauge.return_value = gauge_mock
    
    return prom_mock

@pytest.fixture
def sample_metrics():
    """Métriques échantillon."""
    return {
        'api_requests_total': 15420,
        'api_latency_p95': 245.7,
        'error_rate': 0.003,
        'active_users': 1247,
        'cpu_usage': 0.67,
        'memory_usage': 0.82
    }

# =============================================================================
# STREAMING FIXTURES - Streaming et CDN
# =============================================================================

@pytest.fixture
def streaming_session_mock():
    """Session streaming mock."""
    session = MagicMock()
    session.id = str(uuid.uuid4())
    session.user_id = 'user_12345'
    session.bitrate = 256
    session.quality = 'high'
    session.latency_ms = 45
    session.buffer_health = 0.85
    return session

@pytest.fixture
def cdn_response_mock():
    """Réponse CDN mock."""
    return {
        'edge_server': 'paris-edge-01.cdn.com',
        'latency_ms': 23,
        'cache_status': 'hit',
        'content_length': 5242880,  # 5MB
        'compression': 'gzip'
    }

# =============================================================================
# PERFORMANCE FIXTURES - Tests de performance
# =============================================================================

@pytest.fixture
def performance_benchmarks():
    """Benchmarks performance cibles."""
    return {
        'api_latency_target_ms': 100,
        'recommendation_generation_ms': 50,
        'ml_inference_ms': 25,
        'database_query_ms': 10,
        'cache_access_ms': 1,
        'throughput_rps': 1000
    }

# =============================================================================
# ELASTICSEARCH FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def es_mock():
    """Mock Elasticsearch avec recherches."""
    es = MagicMock(name="es_mock")
    es.search.return_value = {
        'hits': {
            'total': {'value': 10},
            'hits': [
                {'_source': {'title': 'Test Song', 'artist': 'Test Artist'}}
            ]
        }
    }
    es.index.return_value = {'_id': 'test_id', 'result': 'created'}
    return es
