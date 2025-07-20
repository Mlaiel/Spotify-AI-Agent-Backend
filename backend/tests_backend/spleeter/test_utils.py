"""
Tests pour le module utils.py du système Spleeter
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import os

from spleeter.utils import (
    AudioMetadata, AudioUtils, ValidationUtils, PerformanceOptimizer
)
from spleeter.exceptions import ValidationError, AudioProcessingError


class TestAudioMetadata:
    """Tests pour la classe AudioMetadata"""
    
    def test_audio_metadata_creation(self):
        """Test de création des métadonnées audio"""
        metadata = AudioMetadata(
            filename="test.wav",
            duration=120.5,
            sample_rate=44100,
            channels=2,
            bit_depth=16,
            format="wav",
            codec="pcm",
            file_size=10000000,
            bitrate=1411200,
            title="Test Song",
            artist="Test Artist"
        )
        
        assert metadata.filename == "test.wav"
        assert metadata.duration == 120.5
        assert metadata.sample_rate == 44100
        assert metadata.channels == 2
        assert metadata.bit_depth == 16
        assert metadata.format == "wav"
        assert metadata.codec == "pcm"
        assert metadata.file_size == 10000000
        assert metadata.bitrate == 1411200
        assert metadata.title == "Test Song"
        assert metadata.artist == "Test Artist"
    
    def test_audio_metadata_to_dict(self):
        """Test de conversion en dictionnaire"""
        metadata = AudioMetadata(
            filename="test.wav",
            duration=60.0,
            sample_rate=44100,
            channels=2,
            bit_depth=16,
            format="wav",
            codec="pcm",
            file_size=5000000,
            title="Test"
        )
        
        metadata_dict = metadata.to_dict()
        
        assert metadata_dict["filename"] == "test.wav"
        assert metadata_dict["duration"] == 60.0
        assert metadata_dict["sample_rate"] == 44100
        assert metadata_dict["title"] == "Test"
        # Les champs None ne devraient pas être inclus
        assert "artist" not in metadata_dict


class TestAudioUtils:
    """Tests pour la classe AudioUtils"""
    
    def test_is_audio_file_valid_extensions(self):
        """Test de détection de fichiers audio valides"""
        valid_files = [
            "song.wav", "music.mp3", "audio.flac", 
            "track.ogg", "sound.m4a", "file.aac"
        ]
        
        for file_path in valid_files:
            assert AudioUtils.is_audio_file(file_path) is True
    
    def test_is_audio_file_invalid_extensions(self):
        """Test de détection de fichiers non-audio"""
        invalid_files = [
            "document.txt", "image.jpg", "video.mp4", 
            "archive.zip", "code.py"
        ]
        
        for file_path in invalid_files:
            assert AudioUtils.is_audio_file(file_path) is False
    
    def test_is_audio_file_case_insensitive(self):
        """Test de détection insensible à la casse"""
        case_variants = [
            "song.WAV", "music.MP3", "audio.FLAC", 
            "track.OGG", "sound.M4A"
        ]
        
        for file_path in case_variants:
            assert AudioUtils.is_audio_file(file_path) is True
    
    def test_get_format_info_known_formats(self):
        """Test d'information de format pour formats connus"""
        format_tests = [
            ("test.wav", {"codec": "pcm", "lossless": True}),
            ("test.mp3", {"codec": "mp3", "lossless": False}),
            ("test.flac", {"codec": "flac", "lossless": True}),
            ("test.ogg", {"codec": "vorbis", "lossless": False})
        ]
        
        for file_path, expected in format_tests:
            info = AudioUtils.get_format_info(file_path)
            assert info["codec"] == expected["codec"]
            assert info["lossless"] == expected["lossless"]
    
    def test_get_format_info_unknown_format(self):
        """Test d'information pour format inconnu"""
        info = AudioUtils.get_format_info("test.xyz")
        assert info["codec"] == "unknown"
        assert info["lossless"] is False
    
    @pytest.mark.asyncio
    async def test_get_audio_metadata_file_not_found(self):
        """Test de métadonnées pour fichier inexistant"""
        with pytest.raises(FileNotFoundError):
            await AudioUtils.get_audio_metadata("nonexistent.wav")
    
    @pytest.mark.asyncio
    async def test_get_audio_metadata_invalid_format(self):
        """Test de métadonnées pour format invalide"""
        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp_file:
            with pytest.raises(AudioProcessingError):
                await AudioUtils.get_audio_metadata(tmp_file.name)
    
    @pytest.mark.asyncio
    @patch('spleeter.utils.AudioUtils._extract_with_librosa')
    @patch('spleeter.utils.AudioUtils._extract_with_mutagen')
    async def test_get_audio_metadata_fallback(self, mock_mutagen, mock_librosa):
        """Test de fallback pour extraction métadonnées"""
        # Simuler échec mutagen, succès librosa
        mock_mutagen.return_value = None
        mock_librosa.return_value = AudioMetadata(
            filename="test.wav",
            duration=30.0,
            sample_rate=44100,
            channels=2,
            bit_depth=16,
            format="wav",
            codec="pcm",
            file_size=1000000
        )
        
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
            metadata = await AudioUtils.get_audio_metadata(tmp_file.name)
            assert metadata.filename == "test.wav"
            assert metadata.duration == 30.0
    
    def test_calculate_audio_hash_consistency(self):
        """Test de cohérence du hash audio"""
        with patch('librosa.load') as mock_load:
            # Données audio consistantes
            audio_data = np.random.rand(44100)
            mock_load.return_value = (audio_data, 44100)
            
            with patch('librosa.feature.mfcc') as mock_mfcc, \
                 patch('librosa.feature.spectral_centroid') as mock_centroid:
                
                mock_mfcc.return_value = np.random.rand(12, 100)
                mock_centroid.return_value = np.random.rand(1, 100)
                
                # Le même fichier devrait donner le même hash
                hash1 = AudioUtils.calculate_audio_hash("test.wav")
                hash2 = AudioUtils.calculate_audio_hash("test.wav")
                
                assert hash1 == hash2
                assert len(hash1) == 32  # MD5 hash length
    
    def test_detect_silence_basic(self):
        """Test de détection de silence basique"""
        # Créer un signal avec silence au milieu
        sample_rate = 44100
        duration = 3.0
        samples = int(duration * sample_rate)
        
        # Signal fort au début et à la fin, silence au milieu
        signal = np.zeros(samples)
        signal[:sample_rate] = 0.5  # Première seconde forte
        signal[-sample_rate:] = 0.5  # Dernière seconde forte
        # Le milieu reste silencieux
        
        silence_segments = AudioUtils.detect_silence(
            signal, sample_rate, threshold_db=-40.0, min_duration=0.5
        )
        
        # Devrait détecter au moins un segment de silence
        assert len(silence_segments) >= 1
        
        # Le segment devrait être dans la partie centrale
        start, end = silence_segments[0]
        assert start > 0.5  # Après la première seconde
        assert end < 2.5   # Avant la dernière seconde
    
    def test_detect_silence_no_silence(self):
        """Test de détection sans silence"""
        # Signal constant fort
        sample_rate = 44100
        signal = np.full(sample_rate, 0.5)  # 1 seconde de signal fort
        
        silence_segments = AudioUtils.detect_silence(
            signal, sample_rate, threshold_db=-40.0
        )
        
        # Ne devrait pas détecter de silence
        assert len(silence_segments) == 0
    
    def test_detect_silence_stereo(self):
        """Test de détection de silence en stéréo"""
        sample_rate = 44100
        samples = sample_rate * 2  # 2 secondes
        
        # Signal stéréo avec silence au milieu
        stereo_signal = np.zeros((samples, 2))
        stereo_signal[:sample_rate//2, :] = 0.5  # Début fort
        stereo_signal[-sample_rate//2:, :] = 0.5  # Fin forte
        
        silence_segments = AudioUtils.detect_silence(
            stereo_signal, sample_rate, threshold_db=-40.0, min_duration=0.5
        )
        
        assert len(silence_segments) >= 1
    
    def test_trim_silence_basic(self):
        """Test de suppression de silence"""
        sample_rate = 44100
        
        # Signal avec silence au début et à la fin
        silence_duration = sample_rate // 4  # 0.25 seconde
        signal_duration = sample_rate // 2   # 0.5 seconde
        
        full_signal = np.concatenate([
            np.zeros(silence_duration),      # Silence début
            np.full(signal_duration, 0.5),  # Signal
            np.zeros(silence_duration)       # Silence fin
        ])
        
        trimmed = AudioUtils.trim_silence(full_signal, sample_rate)
        
        # Le signal trimé devrait être plus court
        assert len(trimmed) < len(full_signal)
        # Devrait être approximativement la durée du signal central
        assert abs(len(trimmed) - signal_duration) < sample_rate // 10
    
    def test_trim_silence_all_silence(self):
        """Test de suppression avec que du silence"""
        sample_rate = 44100
        silence_signal = np.zeros(sample_rate)  # 1 seconde de silence
        
        trimmed = AudioUtils.trim_silence(silence_signal, sample_rate)
        
        # Devrait garder au moins 1 seconde
        assert len(trimmed) == sample_rate
    
    def test_normalize_loudness_mock(self):
        """Test de normalisation LUFS (avec mock)"""
        audio_data = np.random.rand(44100) * 0.1  # Signal faible
        
        with patch('pyloudnorm.Meter') as mock_meter_class:
            mock_meter = Mock()
            mock_meter.integrated_loudness.return_value = -30.0  # Signal faible
            mock_meter_class.return_value = mock_meter
            
            normalized = AudioUtils.normalize_loudness(audio_data, target_lufs=-23.0)
            
            # Le signal normalisé devrait être plus fort
            assert np.max(np.abs(normalized)) > np.max(np.abs(audio_data))
    
    def test_normalize_loudness_fallback(self):
        """Test de fallback de normalisation"""
        audio_data = np.random.rand(44100) * 0.1
        
        # Sans pyloudnorm, devrait utiliser normalisation peak
        with patch('pyloudnorm.Meter', side_effect=ImportError):
            normalized = AudioUtils.normalize_loudness(audio_data)
            
            # Devrait être normalisé à ~0.7 peak
            assert np.max(np.abs(normalized)) <= 0.75
            assert np.max(np.abs(normalized)) >= 0.65


class TestValidationUtils:
    """Tests pour la classe ValidationUtils"""
    
    def test_validate_audio_file_not_found(self):
        """Test de validation fichier inexistant"""
        with pytest.raises(ValidationError) as exc_info:
            ValidationUtils.validate_audio_file("nonexistent.wav")
        
        assert "non trouvé" in str(exc_info.value)
    
    def test_validate_audio_file_empty(self):
        """Test de validation fichier vide"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            with pytest.raises(ValidationError) as exc_info:
                ValidationUtils.validate_audio_file(tmp_path)
            
            assert "vide" in str(exc_info.value)
        finally:
            os.unlink(tmp_path)
    
    def test_validate_audio_file_too_large(self):
        """Test de validation fichier trop volumineux"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            # Écrire un fichier plus grand que la limite
            large_data = b"0" * (ValidationUtils.MAX_FILE_SIZE + 1000)
            tmp_file.write(large_data)
            tmp_path = tmp_file.name
        
        try:
            with pytest.raises(ValidationError) as exc_info:
                ValidationUtils.validate_audio_file(tmp_path)
            
            assert "volumineux" in str(exc_info.value)
        finally:
            os.unlink(tmp_path)
    
    def test_validate_audio_file_wrong_format(self):
        """Test de validation format incorrect"""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
            tmp_file.write(b"This is not audio")
            tmp_path = tmp_file.name
        
        try:
            with pytest.raises(ValidationError) as exc_info:
                ValidationUtils.validate_audio_file(tmp_path)
            
            assert "non supporté" in str(exc_info.value)
        finally:
            os.unlink(tmp_path)
    
    def test_validate_sample_rate_valid(self):
        """Test de validation sample rate valide"""
        valid_rates = [8000, 16000, 22050, 44100, 48000, 96000]
        
        for rate in valid_rates:
            assert ValidationUtils.validate_sample_rate(rate) is True
    
    def test_validate_sample_rate_invalid_type(self):
        """Test de validation sample rate type invalide"""
        with pytest.raises(ValidationError):
            ValidationUtils.validate_sample_rate("44100")
        
        with pytest.raises(ValidationError):
            ValidationUtils.validate_sample_rate(44100.5)
    
    def test_validate_sample_rate_out_of_range(self):
        """Test de validation sample rate hors limites"""
        with pytest.raises(ValidationError):
            ValidationUtils.validate_sample_rate(4000)  # Trop bas
        
        with pytest.raises(ValidationError):
            ValidationUtils.validate_sample_rate(300000)  # Trop élevé
    
    def test_validate_model_name_valid(self):
        """Test de validation nom de modèle valide"""
        valid_names = [
            "spleeter:2stems-16kHz",
            "spleeter:4stems-44kHz",
            "custom:my-model",
            "provider:model-v1"
        ]
        
        for name in valid_names:
            assert ValidationUtils.validate_model_name(name) is True
    
    def test_validate_model_name_invalid_type(self):
        """Test de validation nom de modèle type invalide"""
        with pytest.raises(ValidationError):
            ValidationUtils.validate_model_name(123)
        
        with pytest.raises(ValidationError):
            ValidationUtils.validate_model_name(None)
    
    def test_validate_model_name_empty(self):
        """Test de validation nom de modèle vide"""
        with pytest.raises(ValidationError):
            ValidationUtils.validate_model_name("")
        
        with pytest.raises(ValidationError):
            ValidationUtils.validate_model_name("   ")
    
    def test_validate_model_name_dangerous_chars(self):
        """Test de validation caractères dangereux"""
        dangerous_names = [
            "model/../other",
            "model<script>",
            "model|command",
            "model*wildcard"
        ]
        
        for name in dangerous_names:
            with pytest.raises(ValidationError):
                ValidationUtils.validate_model_name(name)
    
    def test_validate_output_directory_existing(self):
        """Test de validation répertoire existant"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            assert ValidationUtils.validate_output_directory(tmp_dir) is True
    
    def test_validate_output_directory_non_existing_valid_parent(self):
        """Test de validation répertoire non-existant avec parent valide"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            new_dir = Path(tmp_dir) / "new_subdir"
            assert ValidationUtils.validate_output_directory(new_dir) is True
    
    def test_validate_output_directory_file_instead_of_dir(self):
        """Test de validation fichier au lieu de répertoire"""
        with tempfile.NamedTemporaryFile() as tmp_file:
            with pytest.raises(ValidationError) as exc_info:
                ValidationUtils.validate_output_directory(tmp_file.name)
            
            assert "répertoire" in str(exc_info.value)
    
    def test_validate_batch_size_valid(self):
        """Test de validation taille batch valide"""
        valid_sizes = [1, 4, 8, 16, 32]
        
        for size in valid_sizes:
            assert ValidationUtils.validate_batch_size(size) is True
    
    def test_validate_batch_size_invalid(self):
        """Test de validation taille batch invalide"""
        with pytest.raises(ValidationError):
            ValidationUtils.validate_batch_size(0)
        
        with pytest.raises(ValidationError):
            ValidationUtils.validate_batch_size(-5)
        
        with pytest.raises(ValidationError):
            ValidationUtils.validate_batch_size(4.5)
    
    def test_sanitize_filename_basic(self):
        """Test de nettoyage nom de fichier basique"""
        dangerous_name = "file/with\\dangerous:chars*.txt"
        sanitized = ValidationUtils.sanitize_filename(dangerous_name)
        
        # Les caractères dangereux devraient être remplacés
        assert "/" not in sanitized
        assert "\\" not in sanitized
        assert ":" not in sanitized
        assert "*" not in sanitized
        assert "_" in sanitized  # Remplacé par underscore
    
    def test_sanitize_filename_long_name(self):
        """Test de nettoyage nom trop long"""
        long_name = "a" * 300 + ".txt"
        sanitized = ValidationUtils.sanitize_filename(long_name)
        
        # Devrait être limité à 255 caractères
        assert len(sanitized) <= 255
        assert sanitized.endswith(".txt")  # Extension préservée
    
    def test_sanitize_filename_reserved_windows(self):
        """Test de nettoyage noms réservés Windows"""
        reserved_names = ["CON.txt", "PRN.wav", "AUX.mp3", "NUL.flac"]
        
        for name in reserved_names:
            sanitized = ValidationUtils.sanitize_filename(name)
            assert sanitized.startswith("file_")
    
    def test_validate_disk_space_sufficient(self):
        """Test de validation espace disque suffisant"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Demander très peu d'espace (devrait toujours passer)
            assert ValidationUtils.validate_disk_space(1024, tmp_dir) is True
    
    @patch('shutil.disk_usage')
    def test_validate_disk_space_insufficient(self, mock_disk_usage):
        """Test de validation espace disque insuffisant"""
        # Simuler très peu d'espace libre
        mock_disk_usage.return_value = (1000000, 800000, 1000)  # total, used, free
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(ValidationError) as exc_info:
                ValidationUtils.validate_disk_space(10000, tmp_dir)
            
            assert "insuffisant" in str(exc_info.value)


class TestPerformanceOptimizer:
    """Tests pour la classe PerformanceOptimizer"""
    
    @patch('os.cpu_count')
    @patch('psutil.virtual_memory')
    def test_detect_optimal_config_basic(self, mock_memory, mock_cpu_count):
        """Test de détection configuration optimale"""
        # Simuler système avec 8 CPU et 16GB RAM
        mock_cpu_count.return_value = 8
        mock_memory.return_value = Mock(total=16 * 1024 * 1024 * 1024)
        
        with patch.object(PerformanceOptimizer, '_detect_gpu', return_value=False):
            config = PerformanceOptimizer.detect_optimal_config()
        
        assert config['cpu_count'] == 8
        assert config['memory_gb'] == 16
        assert config['recommended_workers'] == 4  # cpu_count // 2
        assert config['recommended_batch_size'] == 8  # 16GB RAM
        assert config['enable_gpu'] is False
    
    @patch('os.cpu_count')
    @patch('psutil.virtual_memory')
    def test_detect_optimal_config_with_gpu(self, mock_memory, mock_cpu_count):
        """Test de configuration avec GPU"""
        mock_cpu_count.return_value = 4
        mock_memory.return_value = Mock(total=8 * 1024 * 1024 * 1024)
        
        with patch.object(PerformanceOptimizer, '_detect_gpu', return_value=True):
            config = PerformanceOptimizer.detect_optimal_config()
        
        assert config['gpu_available'] is True
        assert config['enable_gpu'] is True
        assert config['recommended_batch_size'] == 8  # 4 * 2 (GPU multiplier)
    
    @patch('os.cpu_count')
    @patch('psutil.virtual_memory')
    def test_detect_optimal_config_low_resources(self, mock_memory, mock_cpu_count):
        """Test de configuration ressources limitées"""
        mock_cpu_count.return_value = 2
        mock_memory.return_value = Mock(total=4 * 1024 * 1024 * 1024)
        
        with patch.object(PerformanceOptimizer, '_detect_gpu', return_value=False):
            config = PerformanceOptimizer.detect_optimal_config()
        
        assert config['recommended_workers'] == 4  # Min workers
        assert config['recommended_batch_size'] == 2  # Low memory
    
    def test_estimate_processing_time_2stems(self):
        """Test d'estimation temps de traitement 2 stems"""
        duration = 180.0  # 3 minutes
        
        # Sans GPU
        time_cpu = PerformanceOptimizer.estimate_processing_time(
            duration, "2stems", use_gpu=False
        )
        
        # Avec GPU
        time_gpu = PerformanceOptimizer.estimate_processing_time(
            duration, "2stems", use_gpu=True
        )
        
        # GPU devrait être plus rapide
        assert time_gpu < time_cpu
        # Devrait inclure l'overhead
        assert time_cpu > duration  # Plus long que l'audio original
    
    def test_estimate_processing_time_complexity(self):
        """Test d'estimation selon complexité modèle"""
        duration = 60.0
        
        time_2stems = PerformanceOptimizer.estimate_processing_time(
            duration, "2stems", use_gpu=False
        )
        time_4stems = PerformanceOptimizer.estimate_processing_time(
            duration, "4stems", use_gpu=False
        )
        time_5stems = PerformanceOptimizer.estimate_processing_time(
            duration, "5stems", use_gpu=False
        )
        
        # Plus de stems = plus de temps
        assert time_2stems < time_4stems < time_5stems
    
    def test_get_memory_requirements_basic(self):
        """Test de calcul besoins mémoire"""
        requirements = PerformanceOptimizer.get_memory_requirements(
            audio_duration=120.0,  # 2 minutes
            sample_rate=44100,
            model_complexity="2stems"
        )
        
        assert "audio_mb" in requirements
        assert "model_mb" in requirements
        assert "processing_mb" in requirements
        assert "output_mb" in requirements
        assert "total_mb" in requirements
        assert "recommended_system_mb" in requirements
        
        # Vérifications logiques
        assert requirements["total_mb"] > requirements["model_mb"]
        assert requirements["recommended_system_mb"] > requirements["total_mb"]
    
    def test_get_memory_requirements_complexity_scaling(self):
        """Test d'évolution besoins selon complexité"""
        duration = 60.0
        
        req_2stems = PerformanceOptimizer.get_memory_requirements(
            duration, model_complexity="2stems"
        )
        req_4stems = PerformanceOptimizer.get_memory_requirements(
            duration, model_complexity="4stems"
        )
        req_5stems = PerformanceOptimizer.get_memory_requirements(
            duration, model_complexity="5stems"
        )
        
        # Plus de stems = plus de mémoire
        assert req_2stems["total_mb"] < req_4stems["total_mb"] < req_5stems["total_mb"]
        assert req_2stems["model_mb"] < req_4stems["model_mb"] < req_5stems["model_mb"]
    
    def test_get_memory_requirements_duration_scaling(self):
        """Test d'évolution besoins selon durée"""
        short_req = PerformanceOptimizer.get_memory_requirements(
            audio_duration=30.0, model_complexity="2stems"
        )
        long_req = PerformanceOptimizer.get_memory_requirements(
            audio_duration=300.0, model_complexity="2stems"
        )
        
        # Plus de durée = plus de mémoire (pour audio et processing)
        assert short_req["audio_mb"] < long_req["audio_mb"]
        assert short_req["processing_mb"] < long_req["processing_mb"]
        assert short_req["output_mb"] < long_req["output_mb"]
        
        # Mais modèle reste constant
        assert short_req["model_mb"] == long_req["model_mb"]
