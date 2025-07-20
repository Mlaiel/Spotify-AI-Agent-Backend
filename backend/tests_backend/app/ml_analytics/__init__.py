# ====================================================================
# ML Analytics Tests Module - Enterprise Test Suite
# ====================================================================
# 
# Tests complets pour le module ML Analytics
# Architecture de tests enterprise avec couverture complète
#
# 🎖️ Expert Team Implementation:
# ✅ Lead Dev + Architecte IA
# ✅ Développeur Backend Senior (Python/FastAPI/Django)  
# ✅ Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
# ✅ DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
# ✅ Spécialiste Sécurité Backend
# ✅ Architecte Microservices
#
# 👨‍💻 Developed by: Fahed Mlaiel
# ====================================================================

"""
ML Analytics Test Suite - Enterprise Testing Framework

Ce module fournit une suite de tests complète pour le système ML Analytics,
incluant des tests unitaires, d'intégration, de performance et de sécurité.

Modules de test disponibles:
- test_core: Tests du moteur ML principal
- test_models: Tests des modèles de recommandation
- test_audio: Tests d'analyse audio
- test_api: Tests des endpoints REST
- test_monitoring: Tests du système de monitoring
- test_utils: Tests des utilitaires
- test_config: Tests de configuration
- test_exceptions: Tests de gestion d'erreurs
- test_scripts: Tests des scripts d'automatisation
- test_security: Tests de sécurité
- test_performance: Tests de performance
- test_integration: Tests d'intégration
"""

import pytest
import asyncio
import logging
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from datetime import datetime, timedelta
import json
import tempfile
import os
from pathlib import Path

# Import des modules ML Analytics à tester
try:
    from ml_analytics.core import MLAnalyticsEngine
    from ml_analytics.models import SpotifyRecommendationModel
    from ml_analytics.audio import AudioAnalysisModel
    from ml_analytics.config import MLAnalyticsConfig
    from ml_analytics.exceptions import MLAnalyticsError
    from ml_analytics.utils import AdvancedCache, DataProcessor, PerformanceMonitor
    from ml_analytics.monitoring import AlertManager, HealthMonitor
    from ml_analytics.api import app as ml_app
    from ml_analytics import initialize_ml_analytics, get_module_info
except ImportError as e:
    logging.warning(f"Certains modules ML Analytics ne sont pas disponibles: {e}")

# Configuration des tests
TESTING_CONFIG = {
    "environment": "testing",
    "database": {
        "url": "sqlite:///:memory:",
        "pool_size": 1,
        "max_overflow": 0
    },
    "redis": {
        "url": "redis://localhost:6379/15",  # Base de test dédiée
        "decode_responses": True
    },
    "ml_models": {
        "path": "/tmp/test_models",
        "cache_size": 100,
        "batch_size": 32
    },
    "monitoring": {
        "enabled": False,  # Désactivé pendant les tests
        "alert_threshold": 0.9
    },
    "security": {
        "jwt_secret": "test-secret-key",
        "rate_limit": 10000  # Limite élevée pour les tests
    }
}

# Fixtures partagées
@pytest.fixture(scope="session")
def event_loop():
    """Event loop pour les tests asynchrones."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def ml_engine():
    """Instance du moteur ML pour les tests."""
    engine = MLAnalyticsEngine()
    await engine.initialize(config=TESTING_CONFIG)
    yield engine
    await engine.cleanup()

@pytest.fixture
def sample_audio_data():
    """Données audio de test."""
    # Générer un signal audio de test
    duration = 5.0  # 5 secondes
    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Signal composite avec plusieurs fréquences
    signal = (
        0.5 * np.sin(2 * np.pi * 440 * t) +  # La4
        0.3 * np.sin(2 * np.pi * 880 * t) +  # La5
        0.2 * np.sin(2 * np.pi * 1320 * t)   # Mi6
    )
    
    # Ajouter un peu de bruit
    noise = 0.1 * np.random.normal(0, 1, signal.shape)
    signal += noise
    
    return {
        "audio_data": signal.astype(np.float32),
        "sample_rate": sample_rate,
        "duration": duration,
        "channels": 1
    }

@pytest.fixture
def sample_user_data():
    """Données utilisateur de test."""
    return {
        "user_id": "test_user_12345",
        "preferences": {
            "genres": ["rock", "pop", "jazz"],
            "artists": ["The Beatles", "Queen", "Miles Davis"],
            "audio_features": {
                "energy": 0.7,
                "valence": 0.6,
                "danceability": 0.8
            }
        },
        "listening_history": [
            {
                "track_id": "spotify:track:test1",
                "artist_name": "Test Artist 1",
                "track_name": "Test Song 1",
                "genre": "rock",
                "play_count": 15,
                "last_played": "2024-01-15T10:30:00Z"
            },
            {
                "track_id": "spotify:track:test2",
                "artist_name": "Test Artist 2",
                "track_name": "Test Song 2",
                "genre": "pop",
                "play_count": 8,
                "last_played": "2024-01-14T16:45:00Z"
            }
        ]
    }

@pytest.fixture
def sample_track_data():
    """Données de pistes de test."""
    return [
        {
            "track_id": "spotify:track:test1",
            "track_name": "Test Song 1",
            "artist_name": "Test Artist 1",
            "album_name": "Test Album 1",
            "genre": "rock",
            "year": 2023,
            "audio_features": {
                "danceability": 0.735,
                "energy": 0.578,
                "key": 5,
                "loudness": -11.84,
                "mode": 0,
                "speechiness": 0.0461,
                "acousticness": 0.514,
                "instrumentalness": 0.0902,
                "liveness": 0.159,
                "valence": 0.636,
                "tempo": 98.002
            },
            "popularity": 85
        },
        {
            "track_id": "spotify:track:test2",
            "track_name": "Test Song 2",
            "artist_name": "Test Artist 2",
            "album_name": "Test Album 2",
            "genre": "pop",
            "year": 2024,
            "audio_features": {
                "danceability": 0.825,
                "energy": 0.693,
                "key": 2,
                "loudness": -8.32,
                "mode": 1,
                "speechiness": 0.0356,
                "acousticness": 0.145,
                "instrumentalness": 0.000012,
                "liveness": 0.0876,
                "valence": 0.789,
                "tempo": 125.456
            },
            "popularity": 92
        }
    ]

@pytest.fixture
def temp_model_directory():
    """Répertoire temporaire pour les modèles de test."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def mock_database():
    """Base de données mockée pour les tests."""
    with patch('ml_analytics.config.create_async_engine') as mock_engine:
        mock_conn = AsyncMock()
        mock_engine.return_value.connect.return_value.__aenter__.return_value = mock_conn
        yield mock_conn

@pytest.fixture
def mock_redis():
    """Redis mocké pour les tests."""
    with patch('ml_analytics.utils.redis.from_url') as mock_redis:
        mock_client = AsyncMock()
        mock_redis.return_value = mock_client
        yield mock_client

# Utilitaires de test
class TestDataGenerator:
    """Générateur de données de test réalistes."""
    
    @staticmethod
    def generate_user_interaction_matrix(num_users: int = 100, num_tracks: int = 1000, sparsity: float = 0.95):
        """Génère une matrice d'interactions utilisateur-piste."""
        matrix = np.random.rand(num_users, num_tracks)
        mask = np.random.rand(num_users, num_tracks) < sparsity
        matrix[mask] = 0
        return matrix
    
    @staticmethod
    def generate_audio_features_dataset(num_tracks: int = 1000):
        """Génère un dataset de caractéristiques audio."""
        features = {
            'danceability': np.random.beta(2, 2, num_tracks),
            'energy': np.random.beta(2, 2, num_tracks),
            'valence': np.random.beta(2, 2, num_tracks),
            'acousticness': np.random.beta(1, 3, num_tracks),
            'instrumentalness': np.random.beta(1, 9, num_tracks),
            'liveness': np.random.beta(1, 4, num_tracks),
            'speechiness': np.random.beta(1, 9, num_tracks),
            'tempo': np.random.normal(120, 30, num_tracks),
            'loudness': np.random.normal(-10, 5, num_tracks)
        }
        return pd.DataFrame(features)
    
    @staticmethod
    def generate_mock_spotify_response(num_tracks: int = 20):
        """Génère une réponse Spotify API mockée."""
        tracks = []
        for i in range(num_tracks):
            track = {
                'id': f'test_track_{i}',
                'name': f'Test Song {i}',
                'artists': [{'name': f'Test Artist {i}'}],
                'album': {'name': f'Test Album {i}'},
                'popularity': np.random.randint(0, 100),
                'audio_features': {
                    'danceability': np.random.rand(),
                    'energy': np.random.rand(),
                    'valence': np.random.rand(),
                    'tempo': np.random.uniform(60, 200)
                }
            }
            tracks.append(track)
        return {'tracks': tracks}

class AsyncTestCase:
    """Classe de base pour les tests asynchrones."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup pour chaque méthode de test."""
        self.start_time = datetime.now()
    
    def teardown_method(self):
        """Cleanup après chaque méthode de test."""
        end_time = datetime.now()
        duration = end_time - self.start_time
        logging.info(f"Test duration: {duration.total_seconds():.2f}s")

# Décorateurs de test utiles
def requires_gpu(func):
    """Décorateur pour les tests nécessitant un GPU."""
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="GPU not available"
    )(func)

def requires_internet(func):
    """Décorateur pour les tests nécessitant une connexion internet."""
    import socket
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        internet_available = True
    except OSError:
        internet_available = False
    
    return pytest.mark.skipif(
        not internet_available,
        reason="Internet connection not available"
    )(func)

def slow_test(func):
    """Décorateur pour les tests lents."""
    return pytest.mark.slow(func)

def integration_test(func):
    """Décorateur pour les tests d'intégration."""
    return pytest.mark.integration(func)

# Configuration des markers pytest
pytest_markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
    "security: marks tests as security tests",
    "performance: marks tests as performance tests",
    "gpu: marks tests that require GPU",
    "internet: marks tests that require internet connection"
]

# Informations du module de test
__version__ = "2.0.0"
__author__ = "Fahed Mlaiel & Expert Team"
__description__ = "Enterprise ML Analytics Test Suite"
__test_modules__ = [
    "test_core",
    "test_models", 
    "test_audio",
    "test_api",
    "test_monitoring",
    "test_utils",
    "test_config",
    "test_exceptions",
    "test_scripts",
    "test_security",
    "test_performance",
    "test_integration"
]

def get_test_info():
    """Retourne les informations du module de test."""
    return {
        "name": "ML Analytics Test Suite",
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "test_modules": __test_modules__,
        "total_tests": "200+",
        "coverage_target": "95%",
        "framework": "pytest + asyncio"
    }

# Logging configuration pour les tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/ml_analytics_tests.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("ML Analytics Test Suite initialized successfully")
