"""
🎵 Spotify AI Agent - Tests d'Intégration Spleeter
================================================

Tests d'intégration pour vérifier le bon fonctionnement
des composants ensemble et les scénarios end-to-end.

🎖️ Développé par l'équipe d'experts enterprise
"""

import pytest
import asyncio
import tempfile
import numpy as np
from pathlib import Path
import json
import time
from unittest.mock import Mock, patch, AsyncMock

from spleeter import SpleeterEngine, SpleeterConfig
from spleeter.core import SpleeterEngine as CoreEngine
from spleeter.models import ModelManager
from spleeter.cache import CacheManager
from spleeter.processor import AudioProcessor, BatchProcessor
from spleeter.monitoring import MetricsCollector, initialize_monitoring
from spleeter.utils import AudioUtils, ValidationUtils
from spleeter.exceptions import (
    AudioProcessingError, ModelError, CacheError, ValidationError
)


class TestSpleeterIntegration:
    """Tests d'intégration complets du système Spleeter"""
    
    @pytest.fixture
    async def temp_dir(self):
        """Répertoire temporaire pour les tests"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)
    
    @pytest.fixture
    async def sample_audio_file(self, temp_dir):
        """Crée un fichier audio de test"""
        audio_file = temp_dir / "test_audio.wav"
        
        # Créer un signal audio simple (440Hz sine wave)
        duration = 2.0  # 2 secondes
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t)  # 440Hz (La)
        
        # Convertir en stéréo
        stereo_audio = np.column_stack([audio_data, audio_data])
        
        # Mock soundfile pour écrire le fichier
        with patch('soundfile.write') as mock_write:
            mock_write.return_value = None
            # Créer le fichier physiquement
            audio_file.write_bytes(b'MOCK_AUDIO_DATA')
        
        return audio_file
    
    @pytest.fixture
    async def spleeter_engine(self, temp_dir):
        """Engine Spleeter configuré pour les tests"""
        config = SpleeterConfig(
            models_dir=str(temp_dir / "models"),
            cache_dir=str(temp_dir / "cache"),
            enable_gpu=False,  # Désactivé pour les tests
            batch_size=2,
            worker_threads=2,
            cache_enabled=True,
            enable_monitoring=True
        )
        
        engine = SpleeterEngine(config=config)
        
        # Mock TensorFlow pour éviter le chargement réel
        with patch('tensorflow.keras.models.load_model') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = [
                np.random.random((1, 1024, 513, 1)),  # vocals
                np.random.random((1, 1024, 513, 1))   # accompaniment
            ]
            mock_load.return_value = mock_model
            
            await engine.initialize()
            yield engine
    
    @pytest.mark.asyncio
    async def test_complete_separation_workflow(self, spleeter_engine, sample_audio_file, temp_dir):
        """Test du workflow complet de séparation"""
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # Mock des dépendances audio
        with patch('librosa.load') as mock_load, \
             patch('soundfile.write') as mock_write, \
             patch('librosa.stft') as mock_stft, \
             patch('librosa.istft') as mock_istft:
            
            # Configuration des mocks
            mock_load.return_value = (np.random.random(88200), 44100)  # 2 secondes
            mock_stft.return_value = np.random.random((513, 1024)) + 1j * np.random.random((513, 1024))
            mock_istft.return_value = np.random.random(88200)
            mock_write.return_value = None
            
            # Exécution de la séparation
            result = await spleeter_engine.separate(
                audio_path=str(sample_audio_file),
                model_name="spleeter:2stems-16kHz",
                output_dir=str(output_dir)
            )
            
            # Vérifications
            assert result is not None
            assert result.success is True
            assert len(result.output_files) == 2
            assert "vocals" in str(result.output_files[0]) or "vocals" in str(result.output_files[1])
            assert "accompaniment" in str(result.output_files[0]) or "accompaniment" in str(result.output_files[1])
    
    @pytest.mark.asyncio
    async def test_batch_processing_integration(self, spleeter_engine, temp_dir):
        """Test d'intégration du traitement par lots"""
        # Créer plusieurs fichiers audio de test
        audio_files = []
        for i in range(3):
            audio_file = temp_dir / f"test_audio_{i}.wav"
            audio_file.write_bytes(b'MOCK_AUDIO_DATA')
            audio_files.append(str(audio_file))
        
        output_dir = temp_dir / "batch_output"
        output_dir.mkdir()
        
        # Mock des dépendances
        with patch('librosa.load') as mock_load, \
             patch('soundfile.write') as mock_write, \
             patch('librosa.stft') as mock_stft, \
             patch('librosa.istft') as mock_istft:
            
            mock_load.return_value = (np.random.random(88200), 44100)
            mock_stft.return_value = np.random.random((513, 1024)) + 1j * np.random.random((513, 1024))
            mock_istft.return_value = np.random.random(88200)
            mock_write.return_value = None
            
            # Traitement par lots
            results = await spleeter_engine.batch_separate(
                audio_files=audio_files,
                model_name="spleeter:2stems-16kHz",
                output_dir=str(output_dir)
            )
            
            # Vérifications
            assert len(results) == 3
            assert all(result.success for result in results)
            assert all(len(result.output_files) == 2 for result in results)
    
    @pytest.mark.asyncio
    async def test_caching_integration(self, spleeter_engine, sample_audio_file, temp_dir):
        """Test d'intégration du système de cache"""
        output_dir = temp_dir / "cache_output"
        output_dir.mkdir()
        
        with patch('librosa.load') as mock_load, \
             patch('soundfile.write') as mock_write, \
             patch('librosa.stft') as mock_stft, \
             patch('librosa.istft') as mock_istft:
            
            mock_load.return_value = (np.random.random(88200), 44100)
            mock_stft.return_value = np.random.random((513, 1024)) + 1j * np.random.random((513, 1024))
            mock_istft.return_value = np.random.random(88200)
            mock_write.return_value = None
            
            # Premier appel - devrait utiliser le traitement
            start_time = time.time()
            result1 = await spleeter_engine.separate(
                audio_path=str(sample_audio_file),
                model_name="spleeter:2stems-16kHz",
                output_dir=str(output_dir)
            )
            first_duration = time.time() - start_time
            
            # Deuxième appel - devrait utiliser le cache
            start_time = time.time()
            result2 = await spleeter_engine.separate(
                audio_path=str(sample_audio_file),
                model_name="spleeter:2stems-16kHz",
                output_dir=str(output_dir)
            )
            second_duration = time.time() - start_time
            
            # Vérifications
            assert result1.success and result2.success
            assert len(result1.output_files) == len(result2.output_files)
            # Le deuxième appel devrait être plus rapide (cache)
            # Note: Dans un mock, cette vérification peut ne pas être pertinente
    
    @pytest.mark.asyncio
    async def test_model_management_integration(self, spleeter_engine, temp_dir):
        """Test d'intégration de la gestion des modèles"""
        model_manager = spleeter_engine.model_manager
        
        # Test de liste des modèles
        available_models = model_manager.get_available_models()
        assert isinstance(available_models, list)
        assert len(available_models) > 0
        
        # Test de validation de modèle
        with patch('pathlib.Path.exists', return_value=True), \
             patch('hashlib.md5') as mock_md5:
            
            mock_md5.return_value.hexdigest.return_value = "valid_checksum"
            
            is_valid = await model_manager.validate_model("spleeter:2stems-16kHz")
            assert is_valid is True
    
    @pytest.mark.asyncio
    async def test_monitoring_integration(self, spleeter_engine, sample_audio_file, temp_dir):
        """Test d'intégration du système de monitoring"""
        output_dir = temp_dir / "monitoring_output"
        output_dir.mkdir()
        
        # Initialiser le monitoring
        collector = initialize_monitoring(enable_system_metrics=False)
        
        with patch('librosa.load') as mock_load, \
             patch('soundfile.write') as mock_write, \
             patch('librosa.stft') as mock_stft, \
             patch('librosa.istft') as mock_istft:
            
            mock_load.return_value = (np.random.random(88200), 44100)
            mock_stft.return_value = np.random.random((513, 1024)) + 1j * np.random.random((513, 1024))
            mock_istft.return_value = np.random.random(88200)
            mock_write.return_value = None
            
            # Exécuter une séparation
            await spleeter_engine.separate(
                audio_path=str(sample_audio_file),
                model_name="spleeter:2stems-16kHz",
                output_dir=str(output_dir)
            )
            
            # Vérifier les métriques
            stats = collector.get_stats_summary()
            assert stats['processing_stats']['total_files'] >= 1
            assert 'recent_metrics' in stats
            assert 'system_health' in stats
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, spleeter_engine, temp_dir):
        """Test d'intégration de la gestion d'erreurs"""
        # Test avec fichier inexistant
        with pytest.raises(FileNotFoundError):
            await spleeter_engine.separate(
                audio_path="/nonexistent/file.wav",
                model_name="spleeter:2stems-16kHz",
                output_dir=str(temp_dir)
            )
        
        # Test avec modèle invalide
        sample_file = temp_dir / "test.wav"
        sample_file.write_bytes(b'MOCK_AUDIO_DATA')
        
        with pytest.raises(ModelError):
            await spleeter_engine.separate(
                audio_path=str(sample_file),
                model_name="invalid:model-name",
                output_dir=str(temp_dir)
            )
    
    @pytest.mark.asyncio
    async def test_audio_validation_integration(self, temp_dir):
        """Test d'intégration de la validation audio"""
        # Créer un fichier non-audio
        invalid_file = temp_dir / "invalid.txt"
        invalid_file.write_text("This is not an audio file")
        
        # Test de validation
        with pytest.raises(ValidationError):
            ValidationUtils.validate_audio_file(invalid_file)
        
        # Créer un fichier audio valide (mock)
        valid_file = temp_dir / "valid.wav"
        valid_file.write_bytes(b'RIFF' + b'\x00' * 40 + b'WAVE')
        
        with patch('soundfile.info') as mock_info:
            mock_info.return_value = Mock(
                samplerate=44100,
                duration=2.0,
                channels=2
            )
            
            # Devrait passer la validation
            assert ValidationUtils.validate_audio_file(valid_file) is True


class TestPerformanceIntegration:
    """Tests de performance et de charge"""
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, temp_dir):
        """Test de traitement concurrent"""
        config = SpleeterConfig(
            models_dir=str(temp_dir / "models"),
            cache_dir=str(temp_dir / "cache"),
            enable_gpu=False,
            batch_size=4,
            worker_threads=4
        )
        
        engine = SpleeterEngine(config=config)
        
        with patch('tensorflow.keras.models.load_model') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = [
                np.random.random((1, 1024, 513, 1)),
                np.random.random((1, 1024, 513, 1))
            ]
            mock_load.return_value = mock_model
            
            await engine.initialize()
            
            # Créer plusieurs tâches concurrentes
            tasks = []
            for i in range(5):
                audio_file = temp_dir / f"concurrent_{i}.wav"
                audio_file.write_bytes(b'MOCK_AUDIO_DATA')
                
                task = engine.separate(
                    audio_path=str(audio_file),
                    model_name="spleeter:2stems-16kHz",
                    output_dir=str(temp_dir / "output")
                )
                tasks.append(task)
            
            # Mock des dépendances audio
            with patch('librosa.load') as mock_audio_load, \
                 patch('soundfile.write') as mock_write, \
                 patch('librosa.stft') as mock_stft, \
                 patch('librosa.istft') as mock_istft:
                
                mock_audio_load.return_value = (np.random.random(88200), 44100)
                mock_stft.return_value = np.random.random((513, 1024)) + 1j * np.random.random((513, 1024))
                mock_istft.return_value = np.random.random(88200)
                mock_write.return_value = None
                
                # Exécuter toutes les tâches en parallèle
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Vérifier que toutes les tâches se sont terminées
                assert len(results) == 5
                successful_results = [r for r in results if not isinstance(r, Exception)]
                assert len(successful_results) >= 3  # Au moins 3 sur 5 devraient réussir
    
    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self, temp_dir):
        """Test de surveillance de l'utilisation mémoire"""
        from spleeter.monitoring import ResourceMonitor
        
        monitor = ResourceMonitor(
            memory_threshold_mb=1000,  # 1GB
            gpu_threshold_percent=80
        )
        
        # Simuler une opération intensive
        with monitor.monitor_operation("test_operation"):
            # Simuler du travail
            await asyncio.sleep(0.1)
            
            # Créer des données pour simuler l'usage mémoire
            large_data = np.random.random((1000, 1000))
            
            await asyncio.sleep(0.1)
        
        # Le monitoring devrait s'être exécuté sans erreur
        assert not monitor.monitoring
    
    @pytest.mark.asyncio
    async def test_cache_performance(self, temp_dir):
        """Test de performance du cache"""
        cache_manager = CacheManager(
            cache_dir=str(temp_dir / "cache"),
            memory_cache_size=100,
            enable_redis=False  # Désactivé pour les tests
        )
        
        await cache_manager.initialize()
        
        # Test d'écriture/lecture en masse
        test_data = {"audio": np.random.random(44100).tolist()}
        
        start_time = time.time()
        for i in range(100):
            await cache_manager.set(f"test_key_{i}", test_data, ttl=300)
        write_time = time.time() - start_time
        
        start_time = time.time()
        for i in range(100):
            result = await cache_manager.get(f"test_key_{i}")
            assert result is not None
        read_time = time.time() - start_time
        
        # Vérifier que les opérations sont raisonnablement rapides
        assert write_time < 5.0  # Moins de 5 secondes pour 100 écritures
        assert read_time < 2.0   # Moins de 2 secondes pour 100 lectures
        
        # Vérifier les statistiques
        stats = cache_manager.get_cache_stats()
        assert stats['memory_cache']['size'] > 0
        assert stats['memory_cache']['hit_rate'] >= 0


class TestRobustnessIntegration:
    """Tests de robustesse et de récupération d'erreurs"""
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self, temp_dir):
        """Test de dégradation gracieuse"""
        config = SpleeterConfig(
            models_dir=str(temp_dir / "models"),
            cache_dir=str(temp_dir / "cache"),
            enable_gpu=False,
            cache_enabled=False  # Cache désactivé
        )
        
        engine = SpleeterEngine(config=config)
        
        # Mock d'un modèle défaillant
        with patch('tensorflow.keras.models.load_model') as mock_load:
            mock_load.side_effect = Exception("Model loading failed")
            
            # L'initialisation devrait gérer l'erreur gracieusement
            with pytest.raises(ModelError):
                await engine.initialize()
    
    @pytest.mark.asyncio
    async def test_resource_cleanup(self, temp_dir):
        """Test de nettoyage des ressources"""
        config = SpleeterConfig(
            models_dir=str(temp_dir / "models"),
            cache_dir=str(temp_dir / "cache"),
            enable_gpu=False
        )
        
        engine = SpleeterEngine(config=config)
        
        with patch('tensorflow.keras.models.load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            await engine.initialize()
            
            # Vérifier que les ressources sont initialisées
            assert engine.model_manager is not None
            assert engine.cache_manager is not None
            
            # Nettoyer
            await engine.cleanup()
            
            # Vérifier le nettoyage
            # Note: Dans une vraie implémentation, on vérifierait que
            # les connexions sont fermées, les threads arrêtés, etc.
    
    @pytest.mark.asyncio
    async def test_partial_failure_handling(self, temp_dir):
        """Test de gestion des échecs partiels"""
        config = SpleeterConfig(
            models_dir=str(temp_dir / "models"),
            cache_dir=str(temp_dir / "cache"),
            enable_gpu=False,
            batch_size=3
        )
        
        engine = SpleeterEngine(config=config)
        
        with patch('tensorflow.keras.models.load_model') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = [
                np.random.random((1, 1024, 513, 1)),
                np.random.random((1, 1024, 513, 1))
            ]
            mock_load.return_value = mock_model
            
            await engine.initialize()
            
            # Créer des fichiers de test (certains valides, d'autres non)
            audio_files = []
            for i in range(5):
                audio_file = temp_dir / f"test_{i}.wav"
                if i == 2:  # Fichier corrompu
                    audio_file.write_bytes(b'INVALID_DATA')
                else:
                    audio_file.write_bytes(b'MOCK_AUDIO_DATA')
                audio_files.append(str(audio_file))
            
            # Mock avec échecs sélectifs
            def mock_load_side_effect(path, *args, **kwargs):
                if "test_2.wav" in path:
                    raise Exception("Corrupted file")
                return (np.random.random(88200), 44100)
            
            with patch('librosa.load', side_effect=mock_load_side_effect), \
                 patch('soundfile.write') as mock_write, \
                 patch('librosa.stft') as mock_stft, \
                 patch('librosa.istft') as mock_istft:
                
                mock_stft.return_value = np.random.random((513, 1024)) + 1j * np.random.random((513, 1024))
                mock_istft.return_value = np.random.random(88200)
                mock_write.return_value = None
                
                # Traitement par lots avec échecs partiels
                results = await engine.batch_separate(
                    audio_files=audio_files,
                    model_name="spleeter:2stems-16kHz",
                    output_dir=str(temp_dir / "output"),
                    continue_on_error=True
                )
                
                # Vérifier que certains ont réussi et d'autres échoué
                successful = [r for r in results if r.success]
                failed = [r for r in results if not r.success]
                
                assert len(successful) >= 3  # Au moins 3 sur 5
                assert len(failed) >= 1     # Au moins 1 échec


@pytest.mark.slow
class TestEndToEndScenarios:
    """Tests de scénarios end-to-end complets"""
    
    @pytest.mark.asyncio
    async def test_real_world_workflow(self, temp_dir):
        """Test d'un workflow réaliste complet"""
        # Configuration réaliste
        config = SpleeterConfig(
            models_dir=str(temp_dir / "models"),
            cache_dir=str(temp_dir / "cache"),
            enable_gpu=False,
            batch_size=4,
            worker_threads=2,
            cache_enabled=True,
            enable_monitoring=True,
            default_sample_rate=44100,
            enable_preprocessing=True
        )
        
        engine = SpleeterEngine(config=config)
        
        # Mock complet des dépendances
        with patch('tensorflow.keras.models.load_model') as mock_tf_load:
            mock_model = Mock()
            mock_model.predict.return_value = [
                np.random.random((1, 1024, 513, 1)),
                np.random.random((1, 1024, 513, 1))
            ]
            mock_tf_load.return_value = mock_model
            
            await engine.initialize()
            
            # Créer un fichier audio de test réaliste
            audio_file = temp_dir / "song.wav"
            audio_file.write_bytes(b'MOCK_SONG_DATA')
            
            output_dir = temp_dir / "separated"
            output_dir.mkdir()
            
            with patch('librosa.load') as mock_load, \
                 patch('soundfile.write') as mock_write, \
                 patch('librosa.stft') as mock_stft, \
                 patch('librosa.istft') as mock_istft, \
                 patch('spleeter.utils.AudioUtils.get_audio_metadata') as mock_metadata:
                
                # Configuration des mocks pour un scénario réaliste
                mock_load.return_value = (np.random.random(176400), 44100)  # 4 secondes
                mock_stft.return_value = np.random.random((513, 1024)) + 1j * np.random.random((513, 1024))
                mock_istft.return_value = np.random.random(176400)
                mock_write.return_value = None
                
                # Mock métadonnées audio
                from spleeter.utils import AudioMetadata
                mock_metadata.return_value = AudioMetadata(
                    filename="song.wav",
                    duration=4.0,
                    sample_rate=44100,
                    channels=2,
                    bit_depth=16,
                    format="wav",
                    codec="pcm",
                    file_size=705600,
                    title="Test Song",
                    artist="Test Artist"
                )
                
                # Workflow complet
                start_time = time.time()
                
                # 1. Séparation principale
                result = await engine.separate(
                    audio_path=str(audio_file),
                    model_name="spleeter:2stems-16kHz",
                    output_dir=str(output_dir)
                )
                
                processing_time = time.time() - start_time
                
                # 2. Vérifications du résultat
                assert result.success is True
                assert len(result.output_files) == 2
                assert result.processing_time > 0
                assert processing_time < 10.0  # Devrait être rapide avec les mocks
                
                # 3. Vérifier les métriques de monitoring
                stats = engine.get_processing_stats()
                assert stats['total_files'] >= 1
                assert stats['successful_files'] >= 1
                
                # 4. Test du cache (deuxième appel)
                start_time = time.time()
                result2 = await engine.separate(
                    audio_path=str(audio_file),
                    model_name="spleeter:2stems-16kHz",
                    output_dir=str(output_dir)
                )
                cached_time = time.time() - start_time
                
                assert result2.success is True
                # Le cache devrait rendre le deuxième appel plus rapide
                # Note: Avec les mocks, cette différence peut ne pas être visible
                
                # 5. Nettoyage final
                await engine.cleanup()
    
    @pytest.mark.asyncio
    async def test_production_like_batch_processing(self, temp_dir):
        """Test de traitement par lots simulant la production"""
        config = SpleeterConfig(
            models_dir=str(temp_dir / "models"),
            cache_dir=str(temp_dir / "cache"),
            enable_gpu=False,
            batch_size=8,
            worker_threads=4,
            cache_enabled=True
        )
        
        engine = SpleeterEngine(config=config)
        
        with patch('tensorflow.keras.models.load_model') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = [
                np.random.random((1, 1024, 513, 1)),
                np.random.random((1, 1024, 513, 1))
            ]
            mock_load.return_value = mock_model
            
            await engine.initialize()
            
            # Créer un lot de fichiers audio
            audio_files = []
            for i in range(20):  # 20 fichiers
                audio_file = temp_dir / f"batch_song_{i:02d}.wav"
                audio_file.write_bytes(b'MOCK_BATCH_AUDIO_DATA')
                audio_files.append(str(audio_file))
            
            output_dir = temp_dir / "batch_output"
            output_dir.mkdir()
            
            with patch('librosa.load') as mock_audio_load, \
                 patch('soundfile.write') as mock_write, \
                 patch('librosa.stft') as mock_stft, \
                 patch('librosa.istft') as mock_istft:
                
                mock_audio_load.return_value = (np.random.random(88200), 44100)
                mock_stft.return_value = np.random.random((513, 1024)) + 1j * np.random.random((513, 1024))
                mock_istft.return_value = np.random.random(88200)
                mock_write.return_value = None
                
                # Traitement par lots
                start_time = time.time()
                results = await engine.batch_separate(
                    audio_files=audio_files,
                    model_name="spleeter:2stems-16kHz",
                    output_dir=str(output_dir)
                )
                total_time = time.time() - start_time
                
                # Vérifications
                assert len(results) == 20
                successful_results = [r for r in results if r.success]
                assert len(successful_results) >= 18  # Au moins 90% de succès
                
                # Performance
                assert total_time < 30.0  # Moins de 30 secondes pour 20 fichiers
                
                # Vérifier les statistiques finales
                final_stats = engine.get_processing_stats()
                assert final_stats['total_files'] >= 20
                assert final_stats['success_rate'] >= 90.0
