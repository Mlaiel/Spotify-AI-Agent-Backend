"""
Tests pour le module processor.py du système Spleeter
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
import tempfile
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

from spleeter.processor import AudioProcessor, BatchProcessor, ProcessingJob
from spleeter.exceptions import AudioProcessingError, ValidationError


class TestAudioProcessor:
    """Tests pour la classe AudioProcessor"""
    
    @pytest.fixture
    def processor(self):
        """Fixture pour créer un processeur audio"""
        return AudioProcessor()
    
    @pytest.fixture
    def sample_audio(self):
        """Fixture pour créer des données audio de test"""
        duration = 2.0  # 2 secondes
        sample_rate = 44100
        samples = int(duration * sample_rate)
        
        # Génération d'un signal sinusoïdal stéréo
        t = np.linspace(0, duration, samples)
        freq = 440  # La 440Hz
        
        left_channel = 0.5 * np.sin(2 * np.pi * freq * t)
        right_channel = 0.3 * np.sin(2 * np.pi * freq * 1.5 * t)
        
        audio_data = np.column_stack([left_channel, right_channel])
        return audio_data, sample_rate
    
    @pytest.mark.asyncio
    async def test_load_audio_file_success(self, processor):
        """Test du chargement réussi d'un fichier audio"""
        with patch('librosa.load') as mock_load:
            mock_load.return_value = (np.random.rand(44100), 44100)
            
            audio_data, sr = await processor.load_audio_file("test.wav")
            
            assert audio_data.shape[0] == 44100
            assert sr == 44100
            mock_load.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_audio_file_not_found(self, processor):
        """Test du chargement d'un fichier inexistant"""
        with patch('librosa.load', side_effect=FileNotFoundError("File not found")):
            with pytest.raises(AudioProcessingError) as exc_info:
                await processor.load_audio_file("nonexistent.wav")
            
            assert "Impossible de charger" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_load_audio_file_invalid_format(self, processor):
        """Test du chargement d'un format invalide"""
        with patch('librosa.load', side_effect=Exception("Format not supported")):
            with pytest.raises(AudioProcessingError):
                await processor.load_audio_file("invalid.xyz")
    
    @pytest.mark.asyncio
    async def test_save_audio_file_success(self, processor, sample_audio):
        """Test de la sauvegarde réussie d'un fichier audio"""
        audio_data, sample_rate = sample_audio
        
        with patch('soundfile.write') as mock_write:
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
                await processor.save_audio_file(
                    audio_data, sample_rate, tmp_file.name
                )
                
                mock_write.assert_called_once()
                args, kwargs = mock_write.call_args
                assert args[0] == tmp_file.name
                np.testing.assert_array_equal(args[1], audio_data)
                assert args[2] == sample_rate
    
    @pytest.mark.asyncio
    async def test_save_audio_file_permission_error(self, processor, sample_audio):
        """Test de sauvegarde avec erreur de permissions"""
        audio_data, sample_rate = sample_audio
        
        with patch('soundfile.write', side_effect=PermissionError("Access denied")):
            with pytest.raises(AudioProcessingError) as exc_info:
                await processor.save_audio_file(
                    audio_data, sample_rate, "/readonly/test.wav"
                )
            
            assert "Impossible de sauvegarder" in str(exc_info.value)
    
    def test_normalize_audio(self, processor, sample_audio):
        """Test de la normalisation audio"""
        audio_data, _ = sample_audio
        
        # Test normalisation peak
        normalized = processor.normalize_audio(audio_data, method="peak")
        
        # Vérifier que le pic maximum est proche de 1.0
        assert np.max(np.abs(normalized)) <= 1.0
        assert np.max(np.abs(normalized)) > 0.9  # Devrait être normalisé
    
    def test_normalize_audio_rms(self, processor, sample_audio):
        """Test de la normalisation RMS"""
        audio_data, _ = sample_audio
        
        normalized = processor.normalize_audio(audio_data, method="rms", target_level=-20)
        
        # Vérifier que l'audio a été modifié
        assert not np.array_equal(audio_data, normalized)
        assert normalized.shape == audio_data.shape
    
    def test_normalize_audio_invalid_method(self, processor, sample_audio):
        """Test de normalisation avec méthode invalide"""
        audio_data, _ = sample_audio
        
        with pytest.raises(ValueError):
            processor.normalize_audio(audio_data, method="invalid")
    
    def test_apply_filter_lowpass(self, processor, sample_audio):
        """Test du filtre passe-bas"""
        audio_data, sample_rate = sample_audio
        
        filtered = processor.apply_filter(
            audio_data, sample_rate, filter_type="lowpass", cutoff=5000
        )
        
        assert filtered.shape == audio_data.shape
        # Le signal filtré devrait être différent de l'original
        assert not np.array_equal(audio_data, filtered)
    
    def test_apply_filter_highpass(self, processor, sample_audio):
        """Test du filtre passe-haut"""
        audio_data, sample_rate = sample_audio
        
        filtered = processor.apply_filter(
            audio_data, sample_rate, filter_type="highpass", cutoff=100
        )
        
        assert filtered.shape == audio_data.shape
        assert not np.array_equal(audio_data, filtered)
    
    def test_apply_filter_bandpass(self, processor, sample_audio):
        """Test du filtre passe-bande"""
        audio_data, sample_rate = sample_audio
        
        filtered = processor.apply_filter(
            audio_data, sample_rate, 
            filter_type="bandpass", 
            cutoff=[200, 4000]
        )
        
        assert filtered.shape == audio_data.shape
        assert not np.array_equal(audio_data, filtered)
    
    def test_apply_filter_invalid_type(self, processor, sample_audio):
        """Test de filtre avec type invalide"""
        audio_data, sample_rate = sample_audio
        
        with pytest.raises(ValueError):
            processor.apply_filter(
                audio_data, sample_rate, filter_type="invalid", cutoff=1000
            )
    
    def test_resample_audio_upsample(self, processor, sample_audio):
        """Test du rééchantillonnage vers une fréquence plus élevée"""
        audio_data, original_sr = sample_audio
        target_sr = 48000
        
        resampled = processor.resample_audio(audio_data, original_sr, target_sr)
        
        # Vérifier que la longueur a changé proportionnellement
        expected_length = int(len(audio_data) * target_sr / original_sr)
        assert abs(len(resampled) - expected_length) <= 1
        assert resampled.shape[1] == audio_data.shape[1]  # Même nombre de canaux
    
    def test_resample_audio_downsample(self, processor, sample_audio):
        """Test du rééchantillonnage vers une fréquence plus basse"""
        audio_data, original_sr = sample_audio
        target_sr = 22050
        
        resampled = processor.resample_audio(audio_data, original_sr, target_sr)
        
        expected_length = int(len(audio_data) * target_sr / original_sr)
        assert abs(len(resampled) - expected_length) <= 1
        assert resampled.shape[1] == audio_data.shape[1]
    
    def test_resample_audio_same_rate(self, processor, sample_audio):
        """Test du rééchantillonnage avec la même fréquence"""
        audio_data, sample_rate = sample_audio
        
        resampled = processor.resample_audio(audio_data, sample_rate, sample_rate)
        
        # Devrait retourner les données originales
        np.testing.assert_array_almost_equal(audio_data, resampled, decimal=5)
    
    def test_convert_to_mono(self, processor, sample_audio):
        """Test de conversion en mono"""
        audio_data, _ = sample_audio
        
        mono = processor.convert_to_mono(audio_data)
        
        assert mono.ndim == 1
        assert len(mono) == len(audio_data)
    
    def test_convert_to_mono_already_mono(self, processor):
        """Test de conversion en mono d'un signal déjà mono"""
        mono_audio = np.random.rand(44100)
        
        result = processor.convert_to_mono(mono_audio)
        
        np.testing.assert_array_equal(mono_audio, result)
    
    def test_ensure_stereo_from_mono(self, processor):
        """Test de conversion mono vers stéréo"""
        mono_audio = np.random.rand(44100)
        
        stereo = processor.ensure_stereo(mono_audio)
        
        assert stereo.shape == (44100, 2)
        # Les deux canaux devraient être identiques
        np.testing.assert_array_equal(stereo[:, 0], stereo[:, 1])
    
    def test_ensure_stereo_already_stereo(self, processor, sample_audio):
        """Test de conversion d'un signal déjà stéréo"""
        audio_data, _ = sample_audio
        
        result = processor.ensure_stereo(audio_data)
        
        np.testing.assert_array_equal(audio_data, result)
    
    def test_get_audio_info(self, processor, sample_audio):
        """Test de récupération des informations audio"""
        audio_data, sample_rate = sample_audio
        
        info = processor.get_audio_info(audio_data, sample_rate)
        
        assert info["duration"] == pytest.approx(2.0, abs=0.1)
        assert info["sample_rate"] == sample_rate
        assert info["channels"] == 2
        assert info["samples"] == len(audio_data)
        assert "rms_level" in info
        assert "peak_level" in info
    
    def test_get_audio_info_mono(self, processor):
        """Test d'informations audio pour signal mono"""
        mono_audio = np.random.rand(22050)  # 0.5 seconde à 44100Hz
        sample_rate = 44100
        
        info = processor.get_audio_info(mono_audio, sample_rate)
        
        assert info["duration"] == pytest.approx(0.5, abs=0.01)
        assert info["channels"] == 1


class TestBatchProcessor:
    """Tests pour la classe BatchProcessor"""
    
    @pytest.fixture
    def batch_processor(self):
        """Fixture pour créer un processeur par lots"""
        return BatchProcessor(max_workers=2, queue_size=10)
    
    @pytest.fixture
    def mock_processing_function(self):
        """Fixture pour une fonction de traitement mock"""
        async def mock_process(job):
            # Simule un traitement
            await asyncio.sleep(0.1)
            return f"processed_{job.input_data}"
        
        return mock_process
    
    @pytest.mark.asyncio
    async def test_batch_processor_initialization(self, batch_processor):
        """Test de l'initialisation du processeur par lots"""
        assert batch_processor.max_workers == 2
        assert batch_processor.queue_size == 10
        assert not batch_processor.is_running
        assert batch_processor.job_queue.maxsize == 10
    
    @pytest.mark.asyncio
    async def test_start_stop_processor(self, batch_processor):
        """Test du démarrage et arrêt du processeur"""
        async def dummy_process(job):
            return "result"
        
        await batch_processor.start(dummy_process)
        assert batch_processor.is_running
        
        await batch_processor.stop()
        assert not batch_processor.is_running
    
    @pytest.mark.asyncio
    async def test_add_job_success(self, batch_processor, mock_processing_function):
        """Test d'ajout de job réussi"""
        await batch_processor.start(mock_processing_function)
        
        job = ProcessingJob(
            job_id="test_job_1",
            input_data="test_input",
            priority=1
        )
        
        job_id = await batch_processor.add_job(job)
        assert job_id == "test_job_1"
        
        await batch_processor.stop()
    
    @pytest.mark.asyncio
    async def test_add_job_queue_full(self, batch_processor, mock_processing_function):
        """Test d'ajout de job avec queue pleine"""
        # Créer un processeur avec une petite queue
        small_processor = BatchProcessor(max_workers=1, queue_size=1)
        await small_processor.start(mock_processing_function)
        
        # Ajouter un job pour remplir la queue
        job1 = ProcessingJob(job_id="job1", input_data="data1")
        await small_processor.add_job(job1)
        
        # Essayer d'ajouter un deuxième job (devrait lever une exception)
        job2 = ProcessingJob(job_id="job2", input_data="data2")
        
        with pytest.raises(Exception):  # Queue pleine
            await small_processor.add_job(job2, timeout=0.1)
        
        await small_processor.stop()
    
    @pytest.mark.asyncio
    async def test_get_job_status_success(self, batch_processor, mock_processing_function):
        """Test de récupération du statut d'un job"""
        await batch_processor.start(mock_processing_function)
        
        job = ProcessingJob(job_id="status_test", input_data="test")
        await batch_processor.add_job(job)
        
        # Attendre que le job soit traité
        await asyncio.sleep(0.2)
        
        status = batch_processor.get_job_status("status_test")
        assert status is not None
        assert status["job_id"] == "status_test"
        
        await batch_processor.stop()
    
    @pytest.mark.asyncio
    async def test_get_job_status_not_found(self, batch_processor):
        """Test de récupération de statut pour job inexistant"""
        status = batch_processor.get_job_status("nonexistent")
        assert status is None
    
    @pytest.mark.asyncio
    async def test_get_job_result_success(self, batch_processor, mock_processing_function):
        """Test de récupération du résultat d'un job"""
        await batch_processor.start(mock_processing_function)
        
        job = ProcessingJob(job_id="result_test", input_data="test_data")
        await batch_processor.add_job(job)
        
        # Attendre que le job soit traité
        result = await batch_processor.get_job_result("result_test", timeout=1.0)
        
        assert result == "processed_test_data"
        
        await batch_processor.stop()
    
    @pytest.mark.asyncio
    async def test_get_job_result_timeout(self, batch_processor):
        """Test de timeout lors de la récupération de résultat"""
        async def slow_process(job):
            await asyncio.sleep(2.0)  # Plus long que le timeout
            return "result"
        
        await batch_processor.start(slow_process)
        
        job = ProcessingJob(job_id="timeout_test", input_data="data")
        await batch_processor.add_job(job)
        
        with pytest.raises(asyncio.TimeoutError):
            await batch_processor.get_job_result("timeout_test", timeout=0.5)
        
        await batch_processor.stop()
    
    @pytest.mark.asyncio
    async def test_get_statistics(self, batch_processor, mock_processing_function):
        """Test de récupération des statistiques"""
        await batch_processor.start(mock_processing_function)
        
        # Ajouter quelques jobs
        for i in range(3):
            job = ProcessingJob(job_id=f"stats_job_{i}", input_data=f"data_{i}")
            await batch_processor.add_job(job)
        
        # Attendre que les jobs soient traités
        await asyncio.sleep(0.5)
        
        stats = batch_processor.get_statistics()
        
        assert "total_jobs" in stats
        assert "completed_jobs" in stats
        assert "failed_jobs" in stats
        assert "pending_jobs" in stats
        assert "average_processing_time" in stats
        
        assert stats["total_jobs"] >= 3
        
        await batch_processor.stop()
    
    @pytest.mark.asyncio
    async def test_clear_completed_jobs(self, batch_processor, mock_processing_function):
        """Test de nettoyage des jobs terminés"""
        await batch_processor.start(mock_processing_function)
        
        job = ProcessingJob(job_id="clear_test", input_data="data")
        await batch_processor.add_job(job)
        
        # Attendre que le job soit traité
        await asyncio.sleep(0.2)
        
        # Vérifier que le job existe
        assert batch_processor.get_job_status("clear_test") is not None
        
        # Nettoyer les jobs terminés
        cleared_count = batch_processor.clear_completed_jobs()
        
        assert cleared_count >= 1
        # Le job ne devrait plus exister
        assert batch_processor.get_job_status("clear_test") is None
        
        await batch_processor.stop()
    
    @pytest.mark.asyncio
    async def test_job_priority_ordering(self, batch_processor):
        """Test de l'ordre de priorité des jobs"""
        async def slow_process(job):
            await asyncio.sleep(0.1)
            return f"processed_{job.input_data}_priority_{job.priority}"
        
        # Créer un processeur avec un seul worker pour contrôler l'ordre
        single_processor = BatchProcessor(max_workers=1, queue_size=10)
        await single_processor.start(slow_process)
        
        # Ajouter des jobs avec différentes priorités
        jobs = [
            ProcessingJob(job_id="low", input_data="low", priority=1),
            ProcessingJob(job_id="high", input_data="high", priority=3),
            ProcessingJob(job_id="medium", input_data="medium", priority=2),
        ]
        
        for job in jobs:
            await single_processor.add_job(job)
        
        # Récupérer les résultats
        results = []
        for job_id in ["high", "medium", "low"]:  # Ordre attendu par priorité
            result = await single_processor.get_job_result(job_id, timeout=2.0)
            results.append(result)
        
        # Vérifier que les jobs ont été traités dans l'ordre de priorité
        assert "priority_3" in results[0]  # High priority first
        assert "priority_2" in results[1]  # Medium priority second
        assert "priority_1" in results[2]  # Low priority last
        
        await single_processor.stop()
    
    @pytest.mark.asyncio
    async def test_processor_error_handling(self, batch_processor):
        """Test de la gestion d'erreurs du processeur"""
        async def failing_process(job):
            if job.input_data == "fail":
                raise ValueError("Simulated processing error")
            return f"processed_{job.input_data}"
        
        await batch_processor.start(failing_process)
        
        # Ajouter un job qui va échouer
        fail_job = ProcessingJob(job_id="fail_job", input_data="fail")
        await batch_processor.add_job(fail_job)
        
        # Ajouter un job qui va réussir
        success_job = ProcessingJob(job_id="success_job", input_data="success")
        await batch_processor.add_job(success_job)
        
        # Attendre que les jobs soient traités
        await asyncio.sleep(0.3)
        
        # Vérifier les statuts
        fail_status = batch_processor.get_job_status("fail_job")
        success_status = batch_processor.get_job_status("success_job")
        
        assert fail_status["status"] == "failed"
        assert "error" in fail_status
        assert success_status["status"] == "completed"
        
        await batch_processor.stop()


class TestProcessingJob:
    """Tests pour la classe ProcessingJob"""
    
    def test_job_creation(self):
        """Test de création d'un job"""
        job = ProcessingJob(
            job_id="test_job",
            input_data="test_input",
            priority=2,
            metadata={"source": "test"}
        )
        
        assert job.job_id == "test_job"
        assert job.input_data == "test_input"
        assert job.priority == 2
        assert job.metadata["source"] == "test"
        assert job.status == "pending"
        assert job.created_at is not None
    
    def test_job_comparison(self):
        """Test de comparaison des jobs pour le tri par priorité"""
        job1 = ProcessingJob(job_id="job1", input_data="data1", priority=1)
        job2 = ProcessingJob(job_id="job2", input_data="data2", priority=2)
        job3 = ProcessingJob(job_id="job3", input_data="data3", priority=2)
        
        # Les jobs avec priorité plus élevée devraient être "plus petits" (traités en premier)
        assert job2 < job1
        assert not (job1 < job2)
        
        # Jobs avec même priorité : comparaison par timestamp
        assert job2 < job3  # job2 créé avant job3
    
    def test_job_to_dict(self):
        """Test de conversion en dictionnaire"""
        job = ProcessingJob(
            job_id="dict_test",
            input_data="test_data",
            priority=1,
            metadata={"key": "value"}
        )
        
        job_dict = job.to_dict()
        
        assert job_dict["job_id"] == "dict_test"
        assert job_dict["input_data"] == "test_data"
        assert job_dict["priority"] == 1
        assert job_dict["status"] == "pending"
        assert job_dict["metadata"]["key"] == "value"
        assert "created_at" in job_dict
    
    def test_job_update_status(self):
        """Test de mise à jour du statut"""
        job = ProcessingJob(job_id="status_test", input_data="data")
        
        assert job.status == "pending"
        
        job.status = "processing"
        assert job.status == "processing"
        
        job.status = "completed"
        assert job.status == "completed"
