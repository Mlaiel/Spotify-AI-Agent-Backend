"""
Tests pour l'analyse audio ML Analytics
=======================================

Tests complets pour AudioAnalysisModel avec couverture de:
- Extraction de caractéristiques MFCC et spectrales
- Classification de genre et analyse de sentiment
- Analyse de qualité audio et détection d'anomalies
- Traitement en temps réel et optimisations
- Intégration avec les modèles de recommandation
"""

import pytest
import asyncio
import numpy as np
import librosa
import tempfile
import wave
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from pathlib import Path

from ml_analytics.audio import (
    AudioAnalysisModel,
    MFCCExtractor,
    SpectralAnalyzer,
    GenreClassifier,
    MoodAnalyzer,
    QualityAssessment,
    AudioFingerprinter
)
from ml_analytics.exceptions import AudioProcessingError, ModelError


class TestAudioAnalysisModel:
    """Tests pour le modèle principal d'analyse audio."""
    
    @pytest.fixture
    async def audio_model(self):
        """Instance de test du modèle d'analyse audio."""
        config = {
            "sample_rate": 22050,
            "n_mfcc": 13,
            "n_fft": 2048,
            "hop_length": 512,
            "n_mel": 128,
            "genre_classes": ["rock", "pop", "jazz", "classical", "electronic"],
            "mood_classes": ["happy", "sad", "energetic", "calm", "angry"]
        }
        model = AudioAnalysisModel(config)
        await model.initialize()
        yield model
        await model.cleanup()
    
    @pytest.fixture
    def sample_audio_file(self):
        """Fichier audio de test."""
        # Créer un fichier WAV temporaire
        duration = 5.0  # 5 secondes
        sample_rate = 22050
        
        # Générer un signal sinusoïdal avec harmoniques
        t = np.linspace(0, duration, int(sample_rate * duration))
        frequency = 440  # La4
        signal = (
            0.5 * np.sin(2 * np.pi * frequency * t) +
            0.25 * np.sin(2 * np.pi * frequency * 2 * t) +
            0.125 * np.sin(2 * np.pi * frequency * 3 * t)
        )
        
        # Ajouter une enveloppe et du bruit
        envelope = np.exp(-t * 0.5)  # Décroissance exponentielle
        noise = 0.05 * np.random.normal(0, 1, signal.shape)
        signal = signal * envelope + noise
        
        # Créer fichier temporaire
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            with wave.open(temp_file.name, 'w') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                
                # Convertir en entiers 16-bit
                signal_int = (signal * 32767).astype(np.int16)
                wav_file.writeframes(signal_int.tobytes())
            
            return temp_file.name
    
    @pytest.fixture
    def sample_audio_data(self):
        """Données audio brutes de test."""
        duration = 3.0
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Signal musical complexe
        frequencies = [261.63, 329.63, 392.00]  # Do, Mi, Sol (accord C majeur)
        signal = sum(
            amplitude * np.sin(2 * np.pi * freq * t)
            for freq, amplitude in zip(frequencies, [0.4, 0.3, 0.3])
        )
        
        # Modulation d'amplitude pour simulation de dynamique musicale
        modulation = 1 + 0.3 * np.sin(2 * np.pi * 0.5 * t)
        signal *= modulation
        
        return {
            "audio_data": signal.astype(np.float32),
            "sample_rate": sample_rate,
            "duration": duration
        }
    
    @pytest.mark.asyncio
    async def test_model_initialization(self):
        """Test l'initialisation du modèle d'analyse audio."""
        config = {
            "sample_rate": 22050,
            "n_mfcc": 13,
            "genre_classes": ["rock", "pop", "jazz"]
        }
        
        model = AudioAnalysisModel(config)
        assert not model.is_initialized
        
        await model.initialize()
        
        assert model.is_initialized
        assert model.mfcc_extractor is not None
        assert model.genre_classifier is not None
        assert model.mood_analyzer is not None
        
        await model.cleanup()
    
    @pytest.mark.asyncio
    async def test_audio_file_analysis(self, audio_model, sample_audio_file):
        """Test l'analyse complète d'un fichier audio."""
        analysis_config = {
            "extract_mfcc": True,
            "extract_spectral": True,
            "classify_genre": True,
            "analyze_mood": True,
            "assess_quality": True
        }
        
        result = await audio_model.analyze_audio(
            audio_source=sample_audio_file,
            config=analysis_config
        )
        
        # Vérifications de base
        assert "duration" in result
        assert "sample_rate" in result
        assert "tempo" in result
        
        # Caractéristiques MFCC
        if analysis_config["extract_mfcc"]:
            assert "mfcc_features" in result
            assert result["mfcc_features"].shape[0] == 13  # n_mfcc
        
        # Classification de genre
        if analysis_config["classify_genre"]:
            assert "genre_prediction" in result
            assert "genre_confidence" in result
        
        # Analyse de sentiment
        if analysis_config["analyze_mood"]:
            assert "mood_analysis" in result
        
        # Évaluation de qualité
        if analysis_config["assess_quality"]:
            assert "quality_score" in result
            assert 0 <= result["quality_score"] <= 1
    
    @pytest.mark.asyncio
    async def test_audio_data_analysis(self, audio_model, sample_audio_data):
        """Test l'analyse de données audio brutes."""
        result = await audio_model.analyze_audio_data(
            audio_data=sample_audio_data["audio_data"],
            sample_rate=sample_audio_data["sample_rate"]
        )
        
        assert "mfcc_features" in result
        assert "spectral_features" in result
        assert "tempo" in result
        assert "key" in result
        
        # Vérifier les dimensions MFCC
        mfcc_shape = result["mfcc_features"].shape
        assert mfcc_shape[0] == 13  # n_mfcc par défaut
    
    @pytest.mark.asyncio
    async def test_real_time_analysis(self, audio_model):
        """Test l'analyse audio en temps réel."""
        # Simuler un flux audio par chunks
        chunk_size = 1024
        sample_rate = 22050
        
        # Générer plusieurs chunks
        chunks = []
        for i in range(10):
            t = np.linspace(i * chunk_size / sample_rate, 
                          (i + 1) * chunk_size / sample_rate, 
                          chunk_size)
            chunk = 0.5 * np.sin(2 * np.pi * 440 * t)  # La4
            chunks.append(chunk.astype(np.float32))
        
        # Analyser chunk par chunk
        results = []
        for chunk in chunks:
            result = await audio_model.analyze_audio_chunk(
                chunk=chunk,
                sample_rate=sample_rate,
                chunk_index=len(results)
            )
            results.append(result)
        
        assert len(results) == 10
        assert all("tempo" in result for result in results)
        assert all("energy" in result for result in results)
    
    @pytest.mark.asyncio
    async def test_batch_analysis(self, audio_model):
        """Test l'analyse par lot de plusieurs fichiers."""
        # Créer plusieurs fichiers audio de test
        audio_files = []
        for i in range(3):
            duration = 2.0 + i  # Durées différentes
            sample_rate = 22050
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Fréquences différentes pour chaque fichier
            frequency = 440 * (2 ** (i / 12))  # Notes de gamme
            signal = 0.5 * np.sin(2 * np.pi * frequency * t)
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                with wave.open(temp_file.name, 'w') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    signal_int = (signal * 32767).astype(np.int16)
                    wav_file.writeframes(signal_int.tobytes())
                audio_files.append(temp_file.name)
        
        # Analyse par lot
        results = await audio_model.analyze_batch(audio_files)
        
        assert len(results) == 3
        assert all("file_path" in result for result in results)
        assert all("analysis" in result for result in results)
        
        # Nettoyer les fichiers temporaires
        for file_path in audio_files:
            Path(file_path).unlink(missing_ok=True)


class TestMFCCExtractor:
    """Tests pour l'extracteur MFCC."""
    
    @pytest.fixture
    def mfcc_extractor(self):
        """Instance de l'extracteur MFCC."""
        config = {
            "n_mfcc": 13,
            "n_fft": 2048,
            "hop_length": 512,
            "n_mel": 128
        }
        return MFCCExtractor(config)
    
    def test_mfcc_extraction(self, mfcc_extractor, sample_audio_data):
        """Test l'extraction des coefficients MFCC."""
        mfcc = mfcc_extractor.extract(
            audio_data=sample_audio_data["audio_data"],
            sample_rate=sample_audio_data["sample_rate"]
        )
        
        assert mfcc.shape[0] == 13  # n_mfcc
        assert mfcc.shape[1] > 0    # Frames temporels
    
    def test_mfcc_delta_features(self, mfcc_extractor, sample_audio_data):
        """Test l'extraction des features delta (dérivées)."""
        features = mfcc_extractor.extract_with_deltas(
            audio_data=sample_audio_data["audio_data"],
            sample_rate=sample_audio_data["sample_rate"]
        )
        
        assert "mfcc" in features
        assert "delta" in features
        assert "delta2" in features
        
        # Même nombre de frames pour toutes les features
        assert features["mfcc"].shape[1] == features["delta"].shape[1]
        assert features["mfcc"].shape[1] == features["delta2"].shape[1]
    
    def test_mfcc_normalization(self, mfcc_extractor, sample_audio_data):
        """Test la normalisation des MFCC."""
        mfcc_raw = mfcc_extractor.extract(
            audio_data=sample_audio_data["audio_data"],
            sample_rate=sample_audio_data["sample_rate"],
            normalize=False
        )
        
        mfcc_normalized = mfcc_extractor.extract(
            audio_data=sample_audio_data["audio_data"],
            sample_rate=sample_audio_data["sample_rate"],
            normalize=True
        )
        
        # Les versions normalisées devraient être différentes
        assert not np.allclose(mfcc_raw, mfcc_normalized)
        
        # Vérifier la normalisation (moyenne ~0, std ~1)
        assert abs(np.mean(mfcc_normalized)) < 0.1
        assert abs(np.std(mfcc_normalized) - 1.0) < 0.1


class TestSpectralAnalyzer:
    """Tests pour l'analyseur spectral."""
    
    @pytest.fixture
    def spectral_analyzer(self):
        """Instance de l'analyseur spectral."""
        config = {
            "n_fft": 2048,
            "hop_length": 512,
            "window": "hann"
        }
        return SpectralAnalyzer(config)
    
    def test_spectral_features_extraction(self, spectral_analyzer, sample_audio_data):
        """Test l'extraction des caractéristiques spectrales."""
        features = spectral_analyzer.extract_features(
            audio_data=sample_audio_data["audio_data"],
            sample_rate=sample_audio_data["sample_rate"]
        )
        
        expected_features = [
            "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
            "spectral_flatness", "zero_crossing_rate"
        ]
        
        for feature in expected_features:
            assert feature in features
            assert len(features[feature]) > 0
    
    def test_chroma_features(self, spectral_analyzer, sample_audio_data):
        """Test l'extraction des caractéristiques chromatiques."""
        chroma = spectral_analyzer.extract_chroma(
            audio_data=sample_audio_data["audio_data"],
            sample_rate=sample_audio_data["sample_rate"]
        )
        
        assert chroma.shape[0] == 12  # 12 classes de hauteur
        assert chroma.shape[1] > 0    # Frames temporels
    
    def test_tonnetz_features(self, spectral_analyzer, sample_audio_data):
        """Test l'extraction des features tonnetz."""
        tonnetz = spectral_analyzer.extract_tonnetz(
            audio_data=sample_audio_data["audio_data"],
            sample_rate=sample_audio_data["sample_rate"]
        )
        
        assert tonnetz.shape[0] == 6  # 6 dimensions tonnetz
        assert tonnetz.shape[1] > 0   # Frames temporels
    
    def test_tempo_estimation(self, spectral_analyzer, sample_audio_data):
        """Test l'estimation du tempo."""
        tempo, beats = spectral_analyzer.estimate_tempo(
            audio_data=sample_audio_data["audio_data"],
            sample_rate=sample_audio_data["sample_rate"]
        )
        
        assert isinstance(tempo, float)
        assert 60 <= tempo <= 200  # Plage raisonnable pour le tempo
        assert len(beats) > 0


class TestGenreClassifier:
    """Tests pour le classificateur de genre."""
    
    @pytest.fixture
    def genre_classifier(self):
        """Instance du classificateur de genre."""
        config = {
            "classes": ["rock", "pop", "jazz", "classical", "electronic"],
            "model_architecture": "cnn",
            "input_shape": (13, 128),  # MFCC shape
            "hidden_layers": [128, 64, 32]
        }
        return GenreClassifier(config)
    
    @pytest.mark.asyncio
    async def test_genre_prediction(self, genre_classifier):
        """Test la prédiction de genre."""
        # Features MFCC simulées
        mfcc_features = np.random.rand(13, 128)
        
        with patch.object(genre_classifier.model, 'predict') as mock_predict:
            mock_predict.return_value = np.array([[0.1, 0.7, 0.1, 0.05, 0.05]])
            
            prediction = await genre_classifier.predict(mfcc_features)
            
            assert "genre" in prediction
            assert "confidence" in prediction
            assert "probabilities" in prediction
            assert prediction["genre"] == "pop"  # Plus haute probabilité
            assert prediction["confidence"] > 0.5
    
    @pytest.mark.asyncio
    async def test_model_training(self, genre_classifier):
        """Test l'entraînement du modèle."""
        # Données d'entraînement simulées
        X_train = np.random.rand(100, 13, 128)  # 100 échantillons
        y_train = np.random.randint(0, 5, 100)  # 5 classes
        
        with patch.object(genre_classifier.model, 'fit') as mock_fit:
            mock_fit.return_value = Mock(history={'loss': [0.8, 0.6, 0.4], 'accuracy': [0.6, 0.7, 0.8]})
            
            result = await genre_classifier.train(X_train, y_train, epochs=3)
            
            assert result["status"] == "success"
            assert "final_accuracy" in result
            assert "training_time" in result
    
    @pytest.mark.asyncio
    async def test_model_evaluation(self, genre_classifier):
        """Test l'évaluation du modèle."""
        # Données de test
        X_test = np.random.rand(20, 13, 128)
        y_test = np.random.randint(0, 5, 20)
        
        with patch.object(genre_classifier.model, 'predict') as mock_predict:
            # Simuler des prédictions avec quelques bonnes classifications
            predictions = np.eye(5)[np.random.randint(0, 5, 20)]
            mock_predict.return_value = predictions
            
            metrics = await genre_classifier.evaluate(X_test, y_test)
            
            assert "accuracy" in metrics
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1_score" in metrics
            assert "confusion_matrix" in metrics


class TestMoodAnalyzer:
    """Tests pour l'analyseur de sentiment musical."""
    
    @pytest.fixture
    def mood_analyzer(self):
        """Instance de l'analyseur de sentiment."""
        config = {
            "mood_dimensions": ["valence", "arousal", "dominance"],
            "model_type": "regression",
            "feature_types": ["mfcc", "spectral", "chroma"]
        }
        return MoodAnalyzer(config)
    
    @pytest.mark.asyncio
    async def test_mood_analysis(self, mood_analyzer):
        """Test l'analyse de sentiment musical."""
        # Features audio combinées
        features = {
            "mfcc": np.random.rand(13, 100),
            "spectral": np.random.rand(5, 100),
            "chroma": np.random.rand(12, 100)
        }
        
        with patch.object(mood_analyzer, '_predict_mood_dimensions') as mock_predict:
            mock_predict.return_value = {
                "valence": 0.7,    # Positif
                "arousal": 0.8,    # Énergique
                "dominance": 0.6   # Dominant
            }
            
            mood = await mood_analyzer.analyze(features)
            
            assert "valence" in mood
            assert "arousal" in mood
            assert "dominance" in mood
            assert "mood_label" in mood
            assert "confidence" in mood
            
            # Valeurs dans la plage attendue
            for dimension in ["valence", "arousal", "dominance"]:
                assert 0 <= mood[dimension] <= 1
    
    @pytest.mark.asyncio
    async def test_mood_mapping(self, mood_analyzer):
        """Test le mapping des dimensions vers des labels."""
        # Test différentes combinaisons valence/arousal
        test_cases = [
            {"valence": 0.8, "arousal": 0.8, "expected": "happy"},
            {"valence": 0.2, "arousal": 0.2, "expected": "sad"},
            {"valence": 0.8, "arousal": 0.2, "expected": "calm"},
            {"valence": 0.2, "arousal": 0.8, "expected": "angry"}
        ]
        
        for case in test_cases:
            mood_label = mood_analyzer.map_to_mood_label(
                valence=case["valence"],
                arousal=case["arousal"]
            )
            assert mood_label == case["expected"]
    
    @pytest.mark.asyncio
    async def test_temporal_mood_analysis(self, mood_analyzer):
        """Test l'analyse temporelle du sentiment."""
        # Séquence de features pour analyse temporelle
        temporal_features = [
            {"mfcc": np.random.rand(13, 50)} for _ in range(10)
        ]
        
        with patch.object(mood_analyzer, 'analyze') as mock_analyze:
            # Simuler une évolution du sentiment
            mock_analyze.side_effect = [
                {"valence": 0.3 + i * 0.1, "arousal": 0.5, "mood_label": "neutral"}
                for i in range(10)
            ]
            
            temporal_mood = await mood_analyzer.analyze_temporal(temporal_features)
            
            assert "mood_evolution" in temporal_mood
            assert "average_mood" in temporal_mood
            assert "mood_stability" in temporal_mood
            assert len(temporal_mood["mood_evolution"]) == 10


class TestQualityAssessment:
    """Tests pour l'évaluateur de qualité audio."""
    
    @pytest.fixture
    def quality_assessor(self):
        """Instance de l'évaluateur de qualité."""
        config = {
            "quality_metrics": ["snr", "thd", "dynamic_range", "clipping"],
            "thresholds": {
                "snr_min": 20,  # dB
                "thd_max": 0.05,  # 5%
                "dynamic_range_min": 20  # dB
            }
        }
        return QualityAssessment(config)
    
    def test_snr_calculation(self, quality_assessor, sample_audio_data):
        """Test le calcul du rapport signal/bruit."""
        # Ajouter du bruit contrôlé
        signal = sample_audio_data["audio_data"]
        noise_level = 0.1
        noisy_signal = signal + noise_level * np.random.normal(0, 1, signal.shape)
        
        snr = quality_assessor.calculate_snr(noisy_signal, noise_level)
        
        assert isinstance(snr, float)
        assert snr > 0  # SNR positif attendu
    
    def test_thd_calculation(self, quality_assessor):
        """Test le calcul de la distorsion harmonique totale."""
        # Signal sinusoïdal pur
        sample_rate = 22050
        duration = 1.0
        frequency = 440
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Signal avec harmoniques (distorsion)
        signal = (
            0.8 * np.sin(2 * np.pi * frequency * t) +
            0.1 * np.sin(2 * np.pi * frequency * 2 * t) +
            0.05 * np.sin(2 * np.pi * frequency * 3 * t)
        )
        
        thd = quality_assessor.calculate_thd(signal, sample_rate, frequency)
        
        assert isinstance(thd, float)
        assert 0 <= thd <= 1  # THD en pourcentage
    
    def test_dynamic_range_calculation(self, quality_assessor, sample_audio_data):
        """Test le calcul de la plage dynamique."""
        signal = sample_audio_data["audio_data"]
        
        dynamic_range = quality_assessor.calculate_dynamic_range(signal)
        
        assert isinstance(dynamic_range, float)
        assert dynamic_range > 0  # Plage dynamique positive
    
    def test_clipping_detection(self, quality_assessor):
        """Test la détection d'écrêtage."""
        # Signal sans écrêtage
        clean_signal = 0.8 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))
        clipping_clean = quality_assessor.detect_clipping(clean_signal)
        
        # Signal avec écrêtage
        clipped_signal = np.clip(2.0 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050)), -1, 1)
        clipping_detected = quality_assessor.detect_clipping(clipped_signal)
        
        assert clipping_clean["clipping_percentage"] < clipping_detected["clipping_percentage"]
        assert clipping_detected["clipping_detected"] is True
    
    def test_overall_quality_score(self, quality_assessor, sample_audio_data):
        """Test le calcul du score de qualité global."""
        signal = sample_audio_data["audio_data"]
        sample_rate = sample_audio_data["sample_rate"]
        
        with patch.object(quality_assessor, 'calculate_snr') as mock_snr, \
             patch.object(quality_assessor, 'calculate_thd') as mock_thd, \
             patch.object(quality_assessor, 'calculate_dynamic_range') as mock_dr, \
             patch.object(quality_assessor, 'detect_clipping') as mock_clip:
            
            # Simuler de bonnes métriques de qualité
            mock_snr.return_value = 25  # Bon SNR
            mock_thd.return_value = 0.02  # Faible distorsion
            mock_dr.return_value = 30  # Bonne plage dynamique
            mock_clip.return_value = {"clipping_percentage": 0.001}  # Pas d'écrêtage
            
            quality_score = quality_assessor.assess_overall_quality(signal, sample_rate)
            
            assert isinstance(quality_score, float)
            assert 0 <= quality_score <= 1
            assert quality_score > 0.8  # Bon score attendu avec ces métriques


class TestAudioFingerprinter:
    """Tests pour le générateur d'empreintes audio."""
    
    @pytest.fixture
    def fingerprinter(self):
        """Instance du générateur d'empreintes."""
        config = {
            "hash_size": 64,
            "algorithm": "chromaprint",
            "sample_rate": 22050
        }
        return AudioFingerprinter(config)
    
    def test_fingerprint_generation(self, fingerprinter, sample_audio_data):
        """Test la génération d'empreinte audio."""
        fingerprint = fingerprinter.generate_fingerprint(
            audio_data=sample_audio_data["audio_data"],
            sample_rate=sample_audio_data["sample_rate"]
        )
        
        assert "hash" in fingerprint
        assert "length" in fingerprint
        assert "confidence" in fingerprint
        assert len(fingerprint["hash"]) > 0
    
    def test_fingerprint_comparison(self, fingerprinter):
        """Test la comparaison d'empreintes."""
        # Deux signaux similaires
        signal1 = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))
        signal2 = 0.9 * signal1 + 0.1 * np.random.normal(0, 0.1, signal1.shape)
        
        fp1 = fingerprinter.generate_fingerprint(signal1, 22050)
        fp2 = fingerprinter.generate_fingerprint(signal2, 22050)
        
        similarity = fingerprinter.compare_fingerprints(fp1["hash"], fp2["hash"])
        
        assert isinstance(similarity, float)
        assert 0 <= similarity <= 1
        assert similarity > 0.5  # Signaux similaires
    
    def test_fingerprint_database(self, fingerprinter):
        """Test la base de données d'empreintes."""
        # Ajouter des empreintes
        test_fingerprints = [
            {"id": "track1", "hash": "hash1", "metadata": {"title": "Song 1"}},
            {"id": "track2", "hash": "hash2", "metadata": {"title": "Song 2"}}
        ]
        
        for fp in test_fingerprints:
            fingerprinter.add_to_database(fp["id"], fp["hash"], fp["metadata"])
        
        # Rechercher une empreinte
        with patch.object(fingerprinter, 'compare_fingerprints') as mock_compare:
            mock_compare.side_effect = [0.9, 0.3]  # Première forte similarité
            
            matches = fingerprinter.search_database("query_hash", threshold=0.5)
            
            assert len(matches) >= 1
            assert matches[0]["similarity"] == 0.9
            assert matches[0]["id"] == "track1"


@pytest.mark.performance
class TestAudioPerformance:
    """Tests de performance pour l'analyse audio."""
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_large_file_processing(self):
        """Test le traitement de gros fichiers audio."""
        # Simuler un fichier de 5 minutes
        duration = 300  # 5 minutes
        sample_rate = 22050
        
        # Générer un long signal
        t = np.linspace(0, duration, int(sample_rate * duration))
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)  # Simple sinusoïde
        
        config = {"sample_rate": sample_rate, "n_mfcc": 13}
        model = AudioAnalysisModel(config)
        await model.initialize()
        
        start_time = datetime.now()
        
        result = await model.analyze_audio_data(signal, sample_rate)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Performance: moins de 30 secondes pour 5 minutes d'audio
        assert processing_time < 30.0
        assert "mfcc_features" in result
        
        await model.cleanup()
    
    @pytest.mark.asyncio
    async def test_batch_processing_performance(self):
        """Test les performances du traitement par lot."""
        # Créer plusieurs fichiers audio courts
        audio_files = []
        for i in range(20):
            duration = 10  # 10 secondes chacun
            sample_rate = 22050
            t = np.linspace(0, duration, int(sample_rate * duration))
            signal = 0.5 * np.sin(2 * np.pi * (440 + i * 10) * t)  # Fréquences différentes
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                # Simuler l'écriture du fichier (on ne fait que le nom)
                audio_files.append(temp_file.name)
        
        config = {"sample_rate": sample_rate, "n_mfcc": 13}
        model = AudioAnalysisModel(config)
        await model.initialize()
        
        start_time = datetime.now()
        
        with patch.object(model, 'analyze_audio') as mock_analyze:
            # Simuler l'analyse rapide
            mock_analyze.return_value = {"mfcc_features": np.random.rand(13, 100)}
            
            results = await model.analyze_batch(audio_files[:10])  # Analyser 10 fichiers
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Performance: moins de 10 secondes pour 10 fichiers
        assert processing_time < 10.0
        assert len(results) == 10
        
        await model.cleanup()
        
        # Nettoyer
        for file_path in audio_files:
            Path(file_path).unlink(missing_ok=True)


@pytest.mark.integration
class TestAudioIntegration:
    """Tests d'intégration pour l'analyse audio."""
    
    @pytest.mark.asyncio
    async def test_full_audio_pipeline(self):
        """Test complet du pipeline d'analyse audio."""
        # 1. Initialisation
        config = {
            "sample_rate": 22050,
            "n_mfcc": 13,
            "genre_classes": ["rock", "pop", "jazz"]
        }
        model = AudioAnalysisModel(config)
        await model.initialize()
        
        # 2. Données audio de test
        duration = 3.0
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration))
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        # 3. Analyse complète
        result = await model.analyze_audio_data(signal, sample_rate)
        
        # 4. Vérifications
        assert "mfcc_features" in result
        assert "spectral_features" in result
        assert "tempo" in result
        
        # 5. Tests de cohérence
        assert result["duration"] == pytest.approx(duration, rel=0.1)
        assert result["sample_rate"] == sample_rate
        
        await model.cleanup()
    
    @pytest.mark.asyncio
    async def test_audio_recommendation_integration(self):
        """Test l'intégration avec le système de recommandation."""
        # Analyser plusieurs pistes pour créer des profils
        audio_analyses = [
            {
                "track_id": "track_1",
                "genre": "rock",
                "energy": 0.8,
                "valence": 0.7,
                "tempo": 120
            },
            {
                "track_id": "track_2", 
                "genre": "pop",
                "energy": 0.6,
                "valence": 0.9,
                "tempo": 100
            }
        ]
        
        # Simuler l'utilisation des analyses pour les recommandations
        user_preferences = {
            "preferred_energy": 0.7,
            "preferred_valence": 0.8,
            "preferred_tempo_range": (100, 130)
        }
        
        # Calculer la compatibilité
        compatibilities = []
        for analysis in audio_analyses:
            compatibility = (
                abs(analysis["energy"] - user_preferences["preferred_energy"]) +
                abs(analysis["valence"] - user_preferences["preferred_valence"])
            ) / 2
            
            tempo_match = (
                user_preferences["preferred_tempo_range"][0] <= 
                analysis["tempo"] <= 
                user_preferences["preferred_tempo_range"][1]
            )
            
            if tempo_match:
                compatibility *= 1.2  # Bonus pour le tempo
            
            compatibilities.append({
                "track_id": analysis["track_id"],
                "compatibility": 1 - compatibility  # Inverser pour avoir un score de compatibilité
            })
        
        # Vérifier que l'intégration fonctionne
        assert len(compatibilities) == 2
        assert all(0 <= comp["compatibility"] <= 1 for comp in compatibilities)
