"""
🎵 Spotify AI Agent - Tests Utilitaires Spleeter
==============================================

Utilitaires et helpers pour faciliter les tests
du module Spleeter avec mocks et fixtures.

🎖️ Développé par l'équipe d'experts enterprise
"""

import pytest
import asyncio
import tempfile
import numpy as np
from pathlib import Path
import json
import sqlite3
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Optional
import time
import uuid

from spleeter import SpleeterEngine, SpleeterConfig
from spleeter.utils import AudioMetadata


class MockAudioData:
    """Générateur de données audio de test"""
    
    @staticmethod
    def generate_sine_wave(frequency: float = 440.0, 
                          duration: float = 2.0, 
                          sample_rate: int = 44100) -> np.ndarray:
        """Génère une onde sinusoïdale"""
        t = np.linspace(0, duration, int(sample_rate * duration))
        return np.sin(2 * np.pi * frequency * t)
    
    @staticmethod
    def generate_stereo_audio(duration: float = 2.0, 
                            sample_rate: int = 44100) -> np.ndarray:
        """Génère de l'audio stéréo de test"""
        left = MockAudioData.generate_sine_wave(440.0, duration, sample_rate)
        right = MockAudioData.generate_sine_wave(880.0, duration, sample_rate)
        return np.column_stack([left, right])
    
    @staticmethod
    def generate_complex_audio(duration: float = 5.0, 
                             sample_rate: int = 44100) -> np.ndarray:
        """Génère un signal audio complexe (multi-fréquences)"""
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Mélange de plusieurs fréquences
        signal = (np.sin(2 * np.pi * 440 * t) +  # La
                 0.7 * np.sin(2 * np.pi * 880 * t) +  # La octave
                 0.5 * np.sin(2 * np.pi * 1320 * t) +  # Mi
                 0.3 * np.sin(2 * np.pi * 220 * t))    # La grave
        
        # Ajouter du bruit réaliste
        noise = np.random.normal(0, 0.02, len(signal))
        
        # Enveloppe pour éviter les clicks
        envelope = np.ones_like(signal)
        fade_samples = int(0.1 * sample_rate)  # 100ms fade
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        
        return (signal + noise) * envelope
    
    @staticmethod
    def create_audio_metadata(filename: str = "test.wav",
                            duration: float = 2.0,
                            sample_rate: int = 44100) -> AudioMetadata:
        """Crée des métadonnées audio de test"""
        return AudioMetadata(
            filename=filename,
            duration=duration,
            sample_rate=sample_rate,
            channels=2,
            bit_depth=16,
            format="wav",
            codec="pcm",
            file_size=int(duration * sample_rate * 2 * 2),  # stéréo 16-bit
            title="Test Song",
            artist="Test Artist",
            album="Test Album",
            genre="Test",
            year=2023,
            rms_level=0.1,
            peak_level=0.8,
            dynamic_range=15.0,
            spectral_centroid=2000.0,
            zero_crossing_rate=0.05
        )


class MockTensorFlowModel:
    """Mock pour les modèles TensorFlow"""
    
    def __init__(self, model_name: str = "2stems"):
        self.model_name = model_name
        self.stems_count = int(model_name[0]) if model_name[0].isdigit() else 2
    
    def predict(self, x):
        """Simule la prédiction TensorFlow"""
        batch_size, time_frames, freq_bins, channels = x.shape
        
        # Générer des masques aléatoires réalistes pour chaque stem
        masks = []
        for i in range(self.stems_count):
            # Masque avec des valeurs entre 0 et 1
            mask = np.random.beta(2, 2, (batch_size, time_frames, freq_bins, 1))
            masks.append(mask)
        
        return masks
    
    def __call__(self, x):
        return self.predict(x)


class MockRedisConnection:
    """Mock pour la connexion Redis"""
    
    def __init__(self):
        self._storage = {}
        self._ttl = {}
    
    async def get(self, key: str) -> Optional[bytes]:
        """Récupère une valeur"""
        if key in self._storage:
            return self._storage[key]
        return None
    
    async def set(self, key: str, value: bytes, ex: Optional[int] = None) -> bool:
        """Stocke une valeur"""
        self._storage[key] = value
        if ex:
            self._ttl[key] = time.time() + ex
        return True
    
    async def delete(self, key: str) -> int:
        """Supprime une clé"""
        if key in self._storage:
            del self._storage[key]
            if key in self._ttl:
                del self._ttl[key]
            return 1
        return 0
    
    async def exists(self, key: str) -> int:
        """Vérifie l'existence d'une clé"""
        return 1 if key in self._storage else 0
    
    async def flushall(self) -> bool:
        """Vide tout le cache"""
        self._storage.clear()
        self._ttl.clear()
        return True
    
    def close(self):
        """Ferme la connexion"""
        pass


class MockSQLiteCache:
    """Mock pour le cache SQLite"""
    
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._connection = None
    
    def connect(self):
        """Établit la connexion"""
        self._connection = sqlite3.connect(self.db_path)
        self._create_tables()
    
    def _create_tables(self):
        """Crée les tables nécessaires"""
        if self._connection:
            cursor = self._connection.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    created_at INTEGER,
                    ttl INTEGER,
                    size INTEGER
                )
            ''')
            self._connection.commit()
    
    def get(self, key: str) -> Optional[bytes]:
        """Récupère une entrée"""
        if not self._connection:
            return None
        
        cursor = self._connection.cursor()
        cursor.execute(
            'SELECT value FROM cache_entries WHERE key = ? AND (ttl = 0 OR created_at + ttl > ?)',
            (key, int(time.time()))
        )
        result = cursor.fetchone()
        return result[0] if result else None
    
    def set(self, key: str, value: bytes, ttl: int = 0) -> bool:
        """Stocke une entrée"""
        if not self._connection:
            return False
        
        cursor = self._connection.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO cache_entries (key, value, created_at, ttl, size)
            VALUES (?, ?, ?, ?, ?)
        ''', (key, value, int(time.time()), ttl, len(value)))
        self._connection.commit()
        return True
    
    def delete(self, key: str) -> bool:
        """Supprime une entrée"""
        if not self._connection:
            return False
        
        cursor = self._connection.cursor()
        cursor.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
        self._connection.commit()
        return cursor.rowcount > 0
    
    def close(self):
        """Ferme la connexion"""
        if self._connection:
            self._connection.close()
            self._connection = None


class TestFixtures:
    """Fixtures réutilisables pour les tests"""
    
    @staticmethod
    @pytest.fixture
    async def temp_directory():
        """Répertoire temporaire pour les tests"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)
    
    @staticmethod
    @pytest.fixture
    def sample_config(temp_directory):
        """Configuration de test standard"""
        return SpleeterConfig(
            models_dir=str(temp_directory / "models"),
            cache_dir=str(temp_directory / "cache"),
            enable_gpu=False,
            batch_size=2,
            worker_threads=2,
            cache_enabled=True,
            enable_monitoring=False,  # Désactivé pour les tests
            default_sample_rate=44100
        )
    
    @staticmethod
    @pytest.fixture
    async def mock_spleeter_engine(sample_config):
        """Engine Spleeter mocké"""
        engine = SpleeterEngine(config=sample_config)
        
        # Mock TensorFlow
        with patch('tensorflow.keras.models.load_model') as mock_load:
            mock_model = MockTensorFlowModel()
            mock_load.return_value = mock_model
            
            await engine.initialize()
            yield engine
            await engine.cleanup()
    
    @staticmethod
    @pytest.fixture
    def sample_audio_files(temp_directory):
        """Fichiers audio de test"""
        files = []
        
        for i, (name, duration) in enumerate([
            ("short", 1.0),
            ("medium", 5.0),
            ("long", 10.0)
        ]):
            audio_file = temp_directory / f"{name}_test_{i}.wav"
            
            # Créer un fichier avec des métadonnées
            audio_data = MockAudioData.generate_complex_audio(duration)
            
            # Mock l'écriture du fichier
            with patch('soundfile.write'):
                audio_file.write_bytes(b'MOCK_AUDIO_DATA')
            
            files.append({
                'path': str(audio_file),
                'name': name,
                'duration': duration,
                'data': audio_data
            })
        
        return files


class MockPatches:
    """Collection de patches pour les tests"""
    
    @staticmethod
    def patch_audio_loading():
        """Patch pour le chargement audio"""
        def mock_librosa_load(path, *args, **kwargs):
            # Simuler différents types de fichiers basés sur le nom
            if "short" in str(path):
                duration = 1.0
            elif "medium" in str(path):
                duration = 5.0
            elif "long" in str(path):
                duration = 10.0
            else:
                duration = 2.0
            
            sample_rate = kwargs.get('sr', 44100)
            audio_data = MockAudioData.generate_complex_audio(duration, sample_rate)
            
            return audio_data, sample_rate
        
        return patch('librosa.load', side_effect=mock_librosa_load)
    
    @staticmethod
    def patch_audio_writing():
        """Patch pour l'écriture audio"""
        def mock_soundfile_write(file, data, samplerate, *args, **kwargs):
            # Simuler l'écriture réussie
            return len(data)
        
        return patch('soundfile.write', side_effect=mock_soundfile_write)
    
    @staticmethod
    def patch_audio_analysis():
        """Patch pour l'analyse audio"""
        patches = [
            patch('librosa.stft', return_value=np.random.random((513, 1024)) + 1j * np.random.random((513, 1024))),
            patch('librosa.istft', return_value=np.random.random(88200)),
            patch('librosa.feature.rms', return_value=np.array([[0.1, 0.15, 0.12]])),
            patch('librosa.feature.spectral_centroid', return_value=np.array([[2000, 2200, 1800]])),
            patch('librosa.feature.zero_crossing_rate', return_value=np.array([[0.05, 0.06, 0.04]]))
        ]
        return patches
    
    @staticmethod
    def patch_tensorflow():
        """Patch pour TensorFlow"""
        return patch('tensorflow.keras.models.load_model', return_value=MockTensorFlowModel())
    
    @staticmethod
    def patch_redis():
        """Patch pour Redis"""
        mock_redis = MockRedisConnection()
        return patch('aioredis.from_url', return_value=mock_redis)
    
    @staticmethod
    def patch_file_operations():
        """Patch pour les opérations de fichiers"""
        return [
            patch('pathlib.Path.exists', return_value=True),
            patch('pathlib.Path.is_file', return_value=True),
            patch('pathlib.Path.stat'),
            patch('os.access', return_value=True)
        ]


class AssertionHelpers:
    """Helpers pour les assertions de tests"""
    
    @staticmethod
    def assert_audio_result(result, expected_stems: int = 2):
        """Vérifie un résultat de séparation audio"""
        assert result is not None
        assert hasattr(result, 'success')
        assert result.success is True
        assert hasattr(result, 'output_files')
        assert len(result.output_files) == expected_stems
        assert hasattr(result, 'processing_time')
        assert result.processing_time > 0
    
    @staticmethod
    def assert_batch_results(results, expected_count: int, min_success_rate: float = 0.8):
        """Vérifie des résultats de traitement par lots"""
        assert len(results) == expected_count
        
        successful = sum(1 for r in results if getattr(r, 'success', False))
        success_rate = successful / len(results)
        
        assert success_rate >= min_success_rate, f"Success rate too low: {success_rate:.2f}"
    
    @staticmethod
    def assert_cache_stats(stats, min_hit_rate: float = 0.0):
        """Vérifie les statistiques de cache"""
        assert 'memory_cache' in stats
        assert 'hit_rate' in stats['memory_cache']
        assert stats['memory_cache']['hit_rate'] >= min_hit_rate
        assert stats['memory_cache']['size'] >= 0
    
    @staticmethod
    def assert_processing_stats(stats):
        """Vérifie les statistiques de traitement"""
        required_fields = [
            'total_files', 'successful_files', 'failed_files',
            'success_rate', 'total_processing_time'
        ]
        
        for field in required_fields:
            assert field in stats, f"Missing field: {field}"
        
        assert stats['total_files'] >= 0
        assert stats['successful_files'] >= 0
        assert stats['failed_files'] >= 0
        assert 0 <= stats['success_rate'] <= 100
        assert stats['total_processing_time'] >= 0
    
    @staticmethod
    def assert_monitoring_data(monitoring_data):
        """Vérifie les données de monitoring"""
        assert 'processing_stats' in monitoring_data
        assert 'system_health' in monitoring_data
        assert 'recent_metrics' in monitoring_data
        
        # Vérifier la santé système
        health = monitoring_data['system_health']
        assert 'cpu_usage' in health
        assert 'memory_usage' in health
        assert 'health_score' in health
        assert 0 <= health['health_score'] <= 100


class PerformanceHelpers:
    """Helpers pour les tests de performance"""
    
    @staticmethod
    def measure_execution_time(func, *args, **kwargs):
        """Mesure le temps d'exécution d'une fonction"""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        return result, end_time - start_time
    
    @staticmethod
    async def measure_async_execution_time(coro):
        """Mesure le temps d'exécution d'une coroutine"""
        start_time = time.perf_counter()
        result = await coro
        end_time = time.perf_counter()
        
        return result, end_time - start_time
    
    @staticmethod
    def get_memory_usage():
        """Obtient l'utilisation mémoire actuelle"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0
    
    @staticmethod
    def benchmark_function(func, iterations: int = 10, *args, **kwargs):
        """Benchmark une fonction sur plusieurs itérations"""
        times = []
        
        for _ in range(iterations):
            _, execution_time = PerformanceHelpers.measure_execution_time(func, *args, **kwargs)
            times.append(execution_time)
        
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'times': times
        }


class TestDataGenerator:
    """Générateur de données de test complexes"""
    
    @staticmethod
    def generate_test_scenarios():
        """Génère des scénarios de test variés"""
        return [
            {
                'name': 'small_mono',
                'duration': 2.0,
                'channels': 1,
                'sample_rate': 22050,
                'model': 'spleeter:2stems-16kHz'
            },
            {
                'name': 'standard_stereo',
                'duration': 30.0,
                'channels': 2,
                'sample_rate': 44100,
                'model': 'spleeter:2stems-16kHz'
            },
            {
                'name': 'high_quality',
                'duration': 60.0,
                'channels': 2,
                'sample_rate': 48000,
                'model': 'spleeter:4stems-16kHz'
            },
            {
                'name': 'long_track',
                'duration': 300.0,
                'channels': 2,
                'sample_rate': 44100,
                'model': 'spleeter:5stems-16kHz'
            }
        ]
    
    @staticmethod
    def generate_batch_test_data(count: int = 10):
        """Génère des données pour tests de traitement par lots"""
        scenarios = TestDataGenerator.generate_test_scenarios()
        
        test_files = []
        for i in range(count):
            scenario = scenarios[i % len(scenarios)]
            
            test_files.append({
                'id': f"batch_test_{i:03d}",
                'filename': f"test_audio_{i:03d}.wav",
                'scenario': scenario,
                'expected_success': True if i < count * 0.9 else False  # 90% succès
            })
        
        return test_files
    
    @staticmethod
    def generate_error_scenarios():
        """Génère des scénarios d'erreur pour les tests"""
        return [
            {
                'name': 'file_not_found',
                'file_path': '/nonexistent/file.wav',
                'expected_exception': 'FileNotFoundError'
            },
            {
                'name': 'invalid_format',
                'file_path': 'test.txt',
                'expected_exception': 'ValidationError'
            },
            {
                'name': 'corrupted_audio',
                'file_path': 'corrupted.wav',
                'expected_exception': 'AudioProcessingError'
            },
            {
                'name': 'invalid_model',
                'model_name': 'invalid:model-name',
                'expected_exception': 'ModelError'
            }
        ]


class AsyncTestHelpers:
    """Helpers pour les tests asynchrones"""
    
    @staticmethod
    async def run_with_timeout(coro, timeout: float = 10.0):
        """Exécute une coroutine avec timeout"""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            pytest.fail(f"Operation timed out after {timeout} seconds")
    
    @staticmethod
    async def wait_for_condition(condition_func, timeout: float = 5.0, interval: float = 0.1):
        """Attend qu'une condition soit vraie"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if condition_func():
                return True
            await asyncio.sleep(interval)
        
        return False
    
    @staticmethod
    async def run_concurrent_tasks(tasks: List, max_concurrent: int = 5):
        """Exécute des tâches avec limitation de concurrence"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_task(task):
            async with semaphore:
                return await task
        
        return await asyncio.gather(*[limited_task(task) for task in tasks])


# Configuration globale pour les tests
TEST_CONFIG = {
    'default_timeout': 30.0,
    'slow_test_timeout': 120.0,
    'performance_test_timeout': 300.0,
    'max_memory_usage_mb': 1000,
    'min_success_rate': 0.85,
    'cache_test_entries': 100,
    'batch_test_size': 20,
    'concurrent_test_size': 10
}


def pytest_configure(config):
    """Configuration personnalisée pour pytest"""
    # Ajouter des marqueurs personnalisés
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (may take several seconds)"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "stress: marks tests as stress tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modifie la collection de tests"""
    # Ajouter le marqueur slow automatiquement pour certains tests
    for item in items:
        if "performance" in item.nodeid or "stress" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
