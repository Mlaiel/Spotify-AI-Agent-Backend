"""
🎵 Configuration des Tests Spleeter - Fixtures et Utilitaires
=============================================================

Configuration centralisée pour tous les tests du module Spleeter.
Fixtures pytest, mocks, données de test et utilitaires.

🎖️ Développé par l'équipe d'experts enterprise
"""

import os
import pytest
import asyncio
import tempfile
import shutil
import numpy as np
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, Optional, List
import json
import sqlite3
from datetime import datetime, timedelta

# Import des modules Spleeter à tester
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "spleeter"))

from spleeter import (
    SpleeterEngine, 
    SpleeterConfig, 
    ModelManager, 
    CacheManager,
    AudioProcessor,
    BatchProcessor
)
from spleeter.monitoring import MetricsCollector, PerformanceTimer
from spleeter.utils import AudioUtils, ValidationUtils, PerformanceOptimizer
from spleeter.exceptions import (
    SpleeterBaseException,
    AudioProcessingError,
    ModelError,
    ValidationError,
    CacheError
)


# ===== CONFIGURATION PYTEST =====

@pytest.fixture(scope="session")
def event_loop():
    """Event loop pour les tests async"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


# ===== FIXTURES DONNÉES DE TEST =====

@pytest.fixture(scope="session")
def test_data_dir():
    """Répertoire temporaire pour les données de test"""
    temp_dir = tempfile.mkdtemp(prefix="spleeter_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_audio_file(test_data_dir):
    """Fichier audio de test (WAV synthétique)"""
    # Génération d'un signal audio synthétique
    sample_rate = 44100
    duration = 2.0  # 2 secondes
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Signal stéréo avec deux fréquences différentes
    left_channel = np.sin(2 * np.pi * 440 * t) * 0.5  # La (440 Hz)
    right_channel = np.sin(2 * np.pi * 880 * t) * 0.3  # La octave (880 Hz)
    
    audio_data = np.column_stack([left_channel, right_channel])
    
    # Sauvegarde en fichier WAV
    audio_file = test_data_dir / "test_audio.wav"
    
    # Mock de soundfile pour éviter la dépendance
    with patch('soundfile.write') as mock_write:
        mock_write.return_value = None
        
        # Créer un fichier factice
        audio_file.write_bytes(b"RIFF" + b"0" * 44 + audio_data.astype(np.float32).tobytes())
    
    return audio_file


@pytest.fixture
def sample_audio_files(test_data_dir):
    """Plusieurs fichiers audio pour tests batch"""
    files = []
    for i in range(3):
        file_path = test_data_dir / f"test_audio_{i}.wav"
        # Fichier factice
        file_path.write_bytes(b"RIFF" + b"0" * 44 + np.random.random(8192).astype(np.float32).tobytes())
        files.append(file_path)
    return files


@pytest.fixture
def sample_metadata():
    """Métadonnées audio de test"""
    return {
        'filename': 'test_audio.wav',
        'duration': 2.0,
        'sample_rate': 44100,
        'channels': 2,
        'bit_depth': 16,
        'format': 'wav',
        'codec': 'pcm',
        'file_size': 176444,
        'title': 'Test Song',
        'artist': 'Test Artist',
        'album': 'Test Album'
    }


# ===== FIXTURES CONFIGURATION =====

@pytest.fixture
def test_config():
    """Configuration Spleeter pour les tests"""
    return SpleeterConfig(
        # Performance
        enable_gpu=False,  # Pas de GPU en tests
        batch_size=2,
        worker_threads=2,
        memory_limit_mb=1024,
        
        # Cache
        cache_enabled=True,
        cache_size_mb=128,
        redis_url=None,  # Pas de Redis en tests
        
        # Audio
        default_sample_rate=44100,
        enable_preprocessing=True,
        normalize_loudness=False,  # Éviter les deps complexes
        
        # Monitoring
        enable_monitoring=False,  # Simplifié pour tests
        metrics_export_interval=5
    )


@pytest.fixture
def test_config_minimal():
    """Configuration minimale pour tests rapides"""
    return SpleeterConfig(
        enable_gpu=False,
        batch_size=1,
        worker_threads=1,
        cache_enabled=False,
        enable_monitoring=False,
        enable_preprocessing=False
    )


# ===== FIXTURES MOCKS =====

@pytest.fixture
def mock_tensorflow():
    """Mock TensorFlow pour éviter les dépendances lourdes"""
    with patch.dict('sys.modules', {
        'tensorflow': MagicMock(),
        'tensorflow.keras': MagicMock(),
        'tensorflow.config': MagicMock()
    }):
        tf_mock = MagicMock()
        tf_mock.config.list_physical_devices.return_value = []
        tf_mock.keras.models.load_model.return_value = MagicMock()
        yield tf_mock


@pytest.fixture
def mock_librosa():
    """Mock librosa pour les tests"""
    with patch.dict('sys.modules', {
        'librosa': MagicMock(),
        'librosa.feature': MagicMock(),
        'librosa.core': MagicMock()
    }):
        librosa_mock = MagicMock()
        
        # Mock des fonctions principales
        librosa_mock.load.return_value = (np.random.random(44100), 44100)
        librosa_mock.feature.mfcc.return_value = np.random.random((12, 100))
        librosa_mock.feature.spectral_centroid.return_value = np.random.random((1, 100))
        librosa_mock.feature.rms.return_value = np.random.random((1, 100))
        librosa_mock.feature.zero_crossing_rate.return_value = np.random.random((1, 100))
        
        yield librosa_mock


@pytest.fixture
def mock_soundfile():
    """Mock soundfile pour les tests"""
    with patch.dict('sys.modules', {'soundfile': MagicMock()}):
        sf_mock = MagicMock()
        
        # Mock info
        info_mock = MagicMock()
        info_mock.duration = 2.0
        info_mock.samplerate = 44100
        info_mock.channels = 2
        sf_mock.info.return_value = info_mock
        
        # Mock read/write
        sf_mock.read.return_value = (np.random.random((88200, 2)), 44100)
        sf_mock.write.return_value = None
        
        yield sf_mock


@pytest.fixture
def mock_redis():
    """Mock Redis pour les tests de cache"""
    redis_mock = AsyncMock()
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    redis_mock.delete.return_value = 1
    redis_mock.exists.return_value = False
    redis_mock.ttl.return_value = -1
    redis_mock.ping.return_value = True
    
    with patch('aioredis.from_url', return_value=redis_mock):
        yield redis_mock


# ===== FIXTURES SPLEETER ENGINE =====

@pytest.fixture
async def spleeter_engine(test_config, mock_tensorflow, mock_librosa, mock_soundfile):
    """Instance SpleeterEngine pour les tests"""
    engine = SpleeterEngine(config=test_config)
    
    # Mock des méthodes nécessitant des dépendances lourdes
    engine._load_model = AsyncMock(return_value=MagicMock())
    engine._process_audio = AsyncMock(return_value={
        'vocals': np.random.random((88200, 2)),
        'accompaniment': np.random.random((88200, 2))
    })
    
    await engine.initialize()
    yield engine
    await engine.cleanup()


@pytest.fixture
async def model_manager(test_data_dir, mock_tensorflow):
    """Instance ModelManager pour les tests"""
    manager = ModelManager(models_dir=test_data_dir)
    
    # Mock des téléchargements
    manager._download_file = AsyncMock(return_value=True)
    manager._validate_model_file = Mock(return_value=True)
    
    return manager


@pytest.fixture
async def cache_manager(test_data_dir):
    """Instance CacheManager pour les tests"""
    cache_dir = test_data_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    manager = CacheManager(
        memory_size_mb=64,
        disk_cache_dir=cache_dir,
        redis_url=None  # Pas de Redis en tests
    )
    
    await manager.initialize()
    yield manager
    await manager.cleanup()


@pytest.fixture
def audio_processor(test_config, mock_librosa, mock_soundfile):
    """Instance AudioProcessor pour les tests"""
    processor = AudioProcessor(config=test_config)
    
    # Mock des méthodes de traitement audio
    processor._apply_filter = Mock(side_effect=lambda x, *args, **kwargs: x)
    processor._normalize_audio = Mock(side_effect=lambda x, *args, **kwargs: x)
    
    return processor


@pytest.fixture
def batch_processor(test_config, mock_librosa, mock_soundfile):
    """Instance BatchProcessor pour les tests"""
    processor = BatchProcessor(
        config=test_config,
        max_workers=2,
        queue_size=10
    )
    return processor


# ===== FIXTURES MONITORING =====

@pytest.fixture
def metrics_collector():
    """Instance MetricsCollector pour les tests"""
    collector = MetricsCollector(
        buffer_size=100,
        retention_hours=1,
        enable_system_metrics=False  # Éviter les métriques système en tests
    )
    yield collector
    collector.stop_system_monitoring()


@pytest.fixture
def performance_timer(metrics_collector):
    """Instance PerformanceTimer pour les tests"""
    return PerformanceTimer("test_operation", collector=metrics_collector)


# ===== FIXTURES UTILITAIRES =====

@pytest.fixture
def temp_output_dir(test_data_dir):
    """Répertoire temporaire pour les sorties"""
    output_dir = test_data_dir / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def mock_file_system(test_data_dir):
    """Mock du système de fichiers pour tests isolation"""
    original_cwd = os.getcwd()
    os.chdir(test_data_dir)
    yield test_data_dir
    os.chdir(original_cwd)


# ===== FIXTURES DONNÉES COMPLEXES =====

@pytest.fixture
def complex_audio_scenario():
    """Scénario audio complexe pour tests avancés"""
    return {
        'sample_rates': [22050, 44100, 48000],
        'channels': [1, 2],
        'durations': [1.0, 5.0, 30.0],
        'formats': ['wav', 'mp3', 'flac'],
        'bit_depths': [16, 24, 32]
    }


@pytest.fixture
def batch_processing_scenario(test_data_dir):
    """Scénario de traitement par lots"""
    scenario = {
        'small_batch': [],
        'medium_batch': [],
        'large_batch': []
    }
    
    # Petits fichiers
    for i in range(3):
        file_path = test_data_dir / f"small_{i}.wav"
        file_path.write_bytes(b"RIFF" + b"0" * 44 + np.random.random(1024).astype(np.float32).tobytes())
        scenario['small_batch'].append(file_path)
    
    # Fichiers moyens
    for i in range(10):
        file_path = test_data_dir / f"medium_{i}.wav"
        file_path.write_bytes(b"RIFF" + b"0" * 44 + np.random.random(8192).astype(np.float32).tobytes())
        scenario['medium_batch'].append(file_path)
    
    # Gros fichiers (simulés)
    for i in range(2):
        file_path = test_data_dir / f"large_{i}.wav"
        file_path.write_bytes(b"RIFF" + b"0" * 44 + np.random.random(32768).astype(np.float32).tobytes())
        scenario['large_batch'].append(file_path)
    
    return scenario


# ===== FIXTURES ERREURS ET EXCEPTIONS =====

@pytest.fixture
def error_scenarios():
    """Scénarios d'erreur pour tests de robustesse"""
    return {
        'file_not_found': '/nonexistent/file.wav',
        'invalid_format': 'invalid_file.txt',
        'corrupted_audio': b'INVALID_AUDIO_DATA',
        'insufficient_permissions': '/root/restricted_file.wav',
        'network_timeout': 'http://timeout.example.com/model.zip',
        'disk_full': '/dev/full/output.wav',
        'memory_limit': 'very_large_file.wav'
    }


# ===== HELPERS ET UTILITAIRES =====

class TestHelper:
    """Classe d'aide pour les tests"""
    
    @staticmethod
    def create_fake_model_file(path: Path, model_type: str = "2stems"):
        """Crée un fichier de modèle factice"""
        model_data = {
            'model_type': model_type,
            'version': '1.0.0',
            'sample_rate': 44100,
            'created_at': datetime.now().isoformat()
        }
        
        # Créer structure répertoire modèle
        model_dir = path / model_type
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Fichier config
        (model_dir / "config.json").write_text(json.dumps(model_data))
        
        # Fichier modèle factice
        (model_dir / "model.h5").write_bytes(b"FAKE_MODEL_DATA" + b"0" * 1000)
        
        return model_dir
    
    @staticmethod
    def create_test_cache_db(cache_dir: Path):
        """Crée une base de données de cache de test"""
        db_path = cache_dir / "cache.db"
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                key TEXT PRIMARY KEY,
                data BLOB,
                created_at TIMESTAMP,
                expires_at TIMESTAMP,
                access_count INTEGER DEFAULT 0
            )
        """)
        
        # Quelques entrées de test
        test_entries = [
            ('test_key_1', b'test_data_1', datetime.now(), datetime.now() + timedelta(hours=1)),
            ('test_key_2', b'test_data_2', datetime.now(), datetime.now() + timedelta(hours=2)),
            ('expired_key', b'expired_data', datetime.now() - timedelta(hours=2), datetime.now() - timedelta(hours=1))
        ]
        
        for key, data, created, expires in test_entries:
            cursor.execute(
                "INSERT INTO cache_entries (key, data, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (key, data, created, expires)
            )
        
        conn.commit()
        conn.close()
        
        return db_path
    
    @staticmethod
    def assert_audio_data_valid(audio_data: np.ndarray, expected_shape: tuple = None):
        """Valide que les données audio sont correctes"""
        assert isinstance(audio_data, np.ndarray), "Audio data should be numpy array"
        assert audio_data.dtype in [np.float32, np.float64], f"Invalid audio dtype: {audio_data.dtype}"
        assert -1.0 <= audio_data.max() <= 1.0, "Audio values should be normalized"
        assert -1.0 <= audio_data.min() <= 1.0, "Audio values should be normalized"
        
        if expected_shape:
            assert audio_data.shape == expected_shape, f"Expected shape {expected_shape}, got {audio_data.shape}"
    
    @staticmethod
    def create_performance_baseline():
        """Crée une baseline de performance pour les tests"""
        return {
            'separation_time_2stems': 2.0,  # seconds per audio second
            'separation_time_4stems': 3.5,
            'separation_time_5stems': 4.5,
            'cache_hit_rate_min': 80.0,  # %
            'memory_usage_max': 2048,  # MB
            'cpu_usage_max': 90.0,  # %
        }


@pytest.fixture
def test_helper():
    """Instance de TestHelper"""
    return TestHelper()


# ===== MARKERS PYTEST =====

def pytest_configure(config):
    """Configuration des markers pytest"""
    config.addinivalue_line("markers", "unit: Tests unitaires")
    config.addinivalue_line("markers", "integration: Tests d'intégration")
    config.addinivalue_line("markers", "performance: Tests de performance")
    config.addinivalue_line("markers", "security: Tests de sécurité")
    config.addinivalue_line("markers", "slow: Tests lents")
    config.addinivalue_line("markers", "gpu: Tests nécessitant GPU")
    config.addinivalue_line("markers", "network: Tests nécessitant réseau")


# ===== CONFIGURATION LOGGING TESTS =====

import logging

@pytest.fixture(autouse=True)
def configure_test_logging():
    """Configure le logging pour les tests"""
    logging.getLogger('spleeter').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('librosa').setLevel(logging.ERROR)
