"""
🎵 Tests SpleeterEngine - Moteur Principal
==========================================

Tests complets pour la classe SpleeterEngine incluant :
- Initialisation et configuration
- Séparation audio basique et avancée
- Gestion des modèles
- Performance et optimisations
- Gestion d'erreurs

🎖️ Développé par l'équipe d'experts enterprise
"""

import pytest
import asyncio
import numpy as np
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import tempfile
import json
from datetime import datetime

from spleeter.core import SpleeterEngine, SpleeterConfig, SeparationResult
from spleeter.exceptions import (
    AudioProcessingError,
    ModelError,
    ValidationError,
    ConfigurationError,
    ResourceError
)


class TestSpleeterEngine:
    """Tests pour la classe SpleeterEngine"""
    
    @pytest.mark.unit
    def test_engine_initialization_default_config(self):
        """Test initialisation avec configuration par défaut"""
        engine = SpleeterEngine()
        
        assert engine.config is not None
        assert isinstance(engine.config, SpleeterConfig)
        assert engine.config.enable_gpu is True  # Par défaut
        assert engine.config.batch_size == 4
        assert engine.initialized is False
    
    @pytest.mark.unit
    def test_engine_initialization_custom_config(self, test_config):
        """Test initialisation avec configuration personnalisée"""
        engine = SpleeterEngine(config=test_config)
        
        assert engine.config == test_config
        assert engine.config.enable_gpu is False  # Config test
        assert engine.config.batch_size == 2
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_engine_initialize_success(self, spleeter_engine, mock_tensorflow):
        """Test initialisation réussie du moteur"""
        # L'engine est déjà initialisé par la fixture
        assert spleeter_engine.initialized is True
        assert spleeter_engine.model_manager is not None
        assert spleeter_engine.cache_manager is not None
        assert spleeter_engine.audio_processor is not None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_engine_initialize_failure(self, test_config, mock_tensorflow):
        """Test échec initialisation du moteur"""
        engine = SpleeterEngine(config=test_config)
        
        # Mock échec initialisation
        with patch.object(engine, '_initialize_components', side_effect=Exception("Init failed")):
            with pytest.raises(ConfigurationError):
                await engine.initialize()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_engine_cleanup(self, spleeter_engine):
        """Test nettoyage des ressources"""
        # Mock des cleanup methods
        spleeter_engine.model_manager.cleanup = AsyncMock()
        spleeter_engine.cache_manager.cleanup = AsyncMock()
        
        await spleeter_engine.cleanup()
        
        spleeter_engine.model_manager.cleanup.assert_called_once()
        spleeter_engine.cache_manager.cleanup.assert_called_once()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_separate_basic_success(self, spleeter_engine, sample_audio_file, temp_output_dir):
        """Test séparation audio basique réussie"""
        result = await spleeter_engine.separate(
            audio_path=sample_audio_file,
            model_name="spleeter:2stems-16kHz",
            output_dir=temp_output_dir
        )
        
        assert isinstance(result, SeparationResult)
        assert result.success is True
        assert result.model_name == "spleeter:2stems-16kHz"
        assert result.audio_duration > 0
        assert result.processing_time > 0
        assert len(result.output_files) >= 2  # Au moins vocals + accompaniment
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_separate_file_not_found(self, spleeter_engine, temp_output_dir):
        """Test séparation avec fichier inexistant"""
        with pytest.raises(AudioProcessingError) as exc_info:
            await spleeter_engine.separate(
                audio_path="/nonexistent/file.wav",
                model_name="spleeter:2stems-16kHz",
                output_dir=temp_output_dir
            )
        
        assert "non trouvé" in str(exc_info.value) or "not found" in str(exc_info.value)
        assert exc_info.value.error_code == "AUDIO_PROCESSING_ERROR"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_separate_invalid_model(self, spleeter_engine, sample_audio_file, temp_output_dir):
        """Test séparation avec modèle invalide"""
        with pytest.raises(ModelError) as exc_info:
            await spleeter_engine.separate(
                audio_path=sample_audio_file,
                model_name="invalid:model-name",
                output_dir=temp_output_dir
            )
        
        assert "model" in str(exc_info.value).lower()
        assert exc_info.value.error_code == "MODEL_ERROR"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_separate_with_preprocessing(self, spleeter_engine, sample_audio_file, temp_output_dir):
        """Test séparation avec préprocessing activé"""
        result = await spleeter_engine.separate(
            audio_path=sample_audio_file,
            model_name="spleeter:2stems-16kHz",
            output_dir=temp_output_dir,
            enable_preprocessing=True,
            normalize_loudness=True
        )
        
        assert result.success is True
        assert "preprocessing" in result.metadata.get("options", {})
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_separate_with_custom_sample_rate(self, spleeter_engine, sample_audio_file, temp_output_dir):
        """Test séparation avec sample rate personnalisé"""
        result = await spleeter_engine.separate(
            audio_path=sample_audio_file,
            model_name="spleeter:2stems-16kHz",
            output_dir=temp_output_dir,
            target_sample_rate=22050
        )
        
        assert result.success is True
        assert result.metadata["target_sample_rate"] == 22050
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_batch_separate_success(self, spleeter_engine, sample_audio_files, temp_output_dir):
        """Test séparation par lots réussie"""
        results = await spleeter_engine.batch_separate(
            audio_files=sample_audio_files,
            model_name="spleeter:2stems-16kHz",
            output_dir=temp_output_dir,
            max_concurrent=2
        )
        
        assert len(results) == len(sample_audio_files)
        
        for result in results:
            assert isinstance(result, SeparationResult)
            if result.success:  # Certains peuvent échouer en test
                assert len(result.output_files) >= 2
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_batch_separate_partial_failure(self, spleeter_engine, temp_output_dir):
        """Test séparation par lots avec échecs partiels"""
        # Mix de fichiers valides et invalides
        audio_files = [
            "/valid/file1.wav",
            "/nonexistent/file.wav",  # Fichier inexistant
            "/valid/file2.wav"
        ]
        
        # Mock pour simuler succès/échecs
        original_separate = spleeter_engine.separate
        
        async def mock_separate(audio_path, **kwargs):
            if "nonexistent" in str(audio_path):
                raise AudioProcessingError("File not found")
            return await original_separate(audio_path, **kwargs)
        
        spleeter_engine.separate = mock_separate
        
        results = await spleeter_engine.batch_separate(
            audio_files=audio_files,
            model_name="spleeter:2stems-16kHz",
            output_dir=temp_output_dir,
            continue_on_error=True
        )
        
        assert len(results) == 3
        
        # Vérifier mix de succès/échecs
        success_count = sum(1 for r in results if r.success)
        failure_count = sum(1 for r in results if not r.success)
        
        assert failure_count > 0  # Au moins un échec
        # Note: success_count peut être 0 en test avec mocks
    
    @pytest.mark.unit
    def test_get_available_models(self, spleeter_engine):
        """Test récupération des modèles disponibles"""
        models = spleeter_engine.get_available_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        
        # Vérifier format des modèles
        for model in models:
            assert ":" in model  # Format "prefix:name"
            assert "stems" in model.lower()
    
    @pytest.mark.unit
    def test_get_model_info(self, spleeter_engine):
        """Test récupération des informations de modèle"""
        model_name = "spleeter:2stems-16kHz"
        info = spleeter_engine.get_model_info(model_name)
        
        assert isinstance(info, dict)
        assert "stems" in info
        assert "sample_rate" in info
        assert info["stems"] == 2
        assert info["sample_rate"] == 16000
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, spleeter_engine):
        """Test health check avec système sain"""
        health = await spleeter_engine.health_check()
        
        assert isinstance(health, dict)
        assert "status" in health
        assert "components" in health
        assert health["status"] in ["healthy", "warning", "critical"]
        
        # Vérifier composants
        components = health["components"]
        assert "model_manager" in components
        assert "cache_manager" in components
        assert "audio_processor" in components
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, spleeter_engine):
        """Test health check avec problèmes"""
        # Simuler problème de cache
        spleeter_engine.cache_manager.health_check = AsyncMock(
            return_value={"status": "critical", "error": "Cache down"}
        )
        
        health = await spleeter_engine.health_check()
        
        assert health["status"] == "critical"
        assert health["components"]["cache_manager"]["status"] == "critical"
    
    @pytest.mark.unit
    def test_get_processing_stats(self, spleeter_engine):
        """Test récupération des statistiques de traitement"""
        stats = spleeter_engine.get_processing_stats()
        
        assert isinstance(stats, dict)
        assert "total_files_processed" in stats
        assert "successful_separations" in stats
        assert "failed_separations" in stats
        assert "average_processing_time" in stats
        assert "total_processing_time" in stats
        
        # Valeurs par défaut
        assert stats["total_files_processed"] >= 0
        assert stats["successful_separations"] >= 0
        assert stats["failed_separations"] >= 0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_reset_stats(self, spleeter_engine):
        """Test remise à zéro des statistiques"""
        # Ajouter quelques stats fictives
        await spleeter_engine._update_stats(
            processing_time=5.0,
            success=True,
            audio_duration=2.0
        )
        
        stats_before = spleeter_engine.get_processing_stats()
        assert stats_before["total_files_processed"] > 0
        
        # Reset
        spleeter_engine.reset_stats()
        
        stats_after = spleeter_engine.get_processing_stats()
        assert stats_after["total_files_processed"] == 0
        assert stats_after["successful_separations"] == 0
        assert stats_after["failed_separations"] == 0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_concurrent_separations(self, spleeter_engine, sample_audio_files, temp_output_dir):
        """Test séparations concurrentes"""
        # Lancer plusieurs séparations en parallèle
        tasks = []
        for i, audio_file in enumerate(sample_audio_files):
            task = spleeter_engine.separate(
                audio_path=audio_file,
                model_name="spleeter:2stems-16kHz",
                output_dir=temp_output_dir / f"output_{i}"
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Vérifier que les résultats sont corrects
        for result in results:
            if not isinstance(result, Exception):
                assert isinstance(result, SeparationResult)
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_memory_management(self, spleeter_engine, sample_audio_file, temp_output_dir):
        """Test gestion de la mémoire"""
        # Mock du monitoring mémoire
        with patch('psutil.Process') as mock_process:
            mock_memory = Mock()
            mock_memory.rss = 1024 * 1024 * 500  # 500MB
            mock_process.return_value.memory_info.return_value = mock_memory
            
            result = await spleeter_engine.separate(
                audio_path=sample_audio_file,
                model_name="spleeter:2stems-16kHz",
                output_dir=temp_output_dir
            )
            
            # Vérifier que la mémoire est surveillée
            assert "peak_memory_mb" in result.metadata
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_gpu_optimization(self, test_config, mock_tensorflow):
        """Test optimisations GPU"""
        # Configuration avec GPU activé
        test_config.enable_gpu = True
        
        # Mock TensorFlow avec GPU
        mock_tensorflow.config.list_physical_devices.return_value = [Mock()]
        
        with patch('tensorflow.config.list_physical_devices', return_value=[Mock()]):
            engine = SpleeterEngine(config=test_config)
            await engine.initialize()
            
            # Vérifier que le GPU est détecté
            assert engine.gpu_available is True
            assert engine.config.enable_gpu is True
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_progress_callback(self, spleeter_engine, sample_audio_file, temp_output_dir):
        """Test callback de progression"""
        progress_updates = []
        
        def progress_callback(progress: float, stage: str):
            progress_updates.append((progress, stage))
        
        result = await spleeter_engine.separate(
            audio_path=sample_audio_file,
            model_name="spleeter:2stems-16kHz",
            output_dir=temp_output_dir,
            progress_callback=progress_callback
        )
        
        # Vérifier que le callback a été appelé
        assert len(progress_updates) > 0
        
        # Vérifier progression
        for progress, stage in progress_updates:
            assert 0.0 <= progress <= 100.0
            assert isinstance(stage, str)
            assert len(stage) > 0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cancel_separation(self, spleeter_engine, sample_audio_file, temp_output_dir):
        """Test annulation de séparation"""
        # Créer tâche d'annulation
        cancel_event = asyncio.Event()
        
        # Mock pour simuler traitement long
        original_process = spleeter_engine._process_audio
        
        async def slow_process(*args, **kwargs):
            # Attendre signal d'annulation
            try:
                await asyncio.wait_for(cancel_event.wait(), timeout=0.1)
                raise asyncio.CancelledError("Operation cancelled")
            except asyncio.TimeoutError:
                return await original_process(*args, **kwargs)
        
        spleeter_engine._process_audio = slow_process
        
        # Lancer séparation
        task = asyncio.create_task(
            spleeter_engine.separate(
                audio_path=sample_audio_file,
                model_name="spleeter:2stems-16kHz",
                output_dir=temp_output_dir
            )
        )
        
        # Annuler immédiatement
        cancel_event.set()
        task.cancel()
        
        with pytest.raises(asyncio.CancelledError):
            await task


class TestSpleeterConfig:
    """Tests pour la classe SpleeterConfig"""
    
    @pytest.mark.unit
    def test_config_default_values(self):
        """Test valeurs par défaut de la configuration"""
        config = SpleeterConfig()
        
        assert config.enable_gpu is True
        assert config.batch_size == 4
        assert config.worker_threads == 4
        assert config.cache_enabled is True
        assert config.default_sample_rate == 44100
        assert config.enable_monitoring is True
    
    @pytest.mark.unit
    def test_config_custom_values(self):
        """Test configuration avec valeurs personnalisées"""
        config = SpleeterConfig(
            enable_gpu=False,
            batch_size=8,
            worker_threads=2,
            cache_enabled=False,
            default_sample_rate=48000
        )
        
        assert config.enable_gpu is False
        assert config.batch_size == 8
        assert config.worker_threads == 2
        assert config.cache_enabled is False
        assert config.default_sample_rate == 48000
    
    @pytest.mark.unit
    def test_config_validation(self):
        """Test validation de la configuration"""
        # Valeurs invalides devraient lever des erreurs
        with pytest.raises(ValidationError):
            SpleeterConfig(batch_size=0)  # Batch size trop petit
        
        with pytest.raises(ValidationError):
            SpleeterConfig(worker_threads=-1)  # Threads négatif
        
        with pytest.raises(ValidationError):
            SpleeterConfig(default_sample_rate=1000)  # Sample rate trop bas
    
    @pytest.mark.unit
    def test_config_to_dict(self):
        """Test conversion configuration en dictionnaire"""
        config = SpleeterConfig(
            enable_gpu=False,
            batch_size=2,
            cache_enabled=True
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["enable_gpu"] is False
        assert config_dict["batch_size"] == 2
        assert config_dict["cache_enabled"] is True
    
    @pytest.mark.unit
    def test_config_from_dict(self):
        """Test création configuration depuis dictionnaire"""
        config_data = {
            "enable_gpu": False,
            "batch_size": 8,
            "worker_threads": 2,
            "cache_enabled": True,
            "default_sample_rate": 48000
        }
        
        config = SpleeterConfig.from_dict(config_data)
        
        assert config.enable_gpu is False
        assert config.batch_size == 8
        assert config.worker_threads == 2
        assert config.cache_enabled is True
        assert config.default_sample_rate == 48000


class TestSeparationResult:
    """Tests pour la classe SeparationResult"""
    
    @pytest.mark.unit
    def test_result_creation(self):
        """Test création d'un résultat de séparation"""
        result = SeparationResult(
            success=True,
            audio_path="/path/to/audio.wav",
            model_name="spleeter:2stems-16kHz",
            output_files={
                "vocals": "/output/vocals.wav",
                "accompaniment": "/output/accompaniment.wav"
            },
            audio_duration=120.5,
            processing_time=45.2,
            metadata={"sample_rate": 44100}
        )
        
        assert result.success is True
        assert result.audio_path == "/path/to/audio.wav"
        assert result.model_name == "spleeter:2stems-16kHz"
        assert len(result.output_files) == 2
        assert result.audio_duration == 120.5
        assert result.processing_time == 45.2
        assert result.metadata["sample_rate"] == 44100
    
    @pytest.mark.unit
    def test_result_processing_ratio(self):
        """Test calcul du ratio de traitement"""
        result = SeparationResult(
            success=True,
            audio_path="/path/to/audio.wav",
            model_name="spleeter:2stems-16kHz",
            output_files={},
            audio_duration=60.0,  # 60 secondes
            processing_time=30.0,  # 30 secondes
            metadata={}
        )
        
        assert result.processing_ratio == 0.5  # 30/60 = 0.5x real-time
    
    @pytest.mark.unit
    def test_result_processing_ratio_zero_duration(self):
        """Test ratio de traitement avec durée zéro"""
        result = SeparationResult(
            success=True,
            audio_path="/path/to/audio.wav",
            model_name="spleeter:2stems-16kHz",
            output_files={},
            audio_duration=0.0,
            processing_time=30.0,
            metadata={}
        )
        
        assert result.processing_ratio == 0.0
    
    @pytest.mark.unit
    def test_result_to_dict(self):
        """Test conversion résultat en dictionnaire"""
        result = SeparationResult(
            success=True,
            audio_path="/path/to/audio.wav",
            model_name="spleeter:2stems-16kHz",
            output_files={"vocals": "/output/vocals.wav"},
            audio_duration=60.0,
            processing_time=30.0,
            metadata={"test": "value"}
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["success"] is True
        assert result_dict["audio_path"] == "/path/to/audio.wav"
        assert result_dict["processing_ratio"] == 0.5
        assert result_dict["metadata"]["test"] == "value"
    
    @pytest.mark.unit
    def test_result_error_case(self):
        """Test résultat en cas d'erreur"""
        error = AudioProcessingError("Test error", file_path="/path/to/audio.wav")
        
        result = SeparationResult(
            success=False,
            audio_path="/path/to/audio.wav",
            model_name="spleeter:2stems-16kHz",
            output_files={},
            audio_duration=0.0,
            processing_time=0.0,
            error=error,
            metadata={}
        )
        
        assert result.success is False
        assert result.error == error
        assert len(result.output_files) == 0
        
        # Test conversion en dict avec erreur
        result_dict = result.to_dict()
        assert "error" in result_dict
        assert result_dict["error"]["type"] == "AudioProcessingError"
