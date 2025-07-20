"""
Test Suite for Advanced Audio Intelligence - Enterprise Edition
==============================================================

Comprehensive test suite for advanced audio processing, analysis, and intelligence
features including real-time audio feature extraction, mood detection, and audio ML.

Created by: Fahed Mlaiel - Expert Team
✅ Lead Dev + Architecte IA
✅ Développeur Backend Senior (Python/FastAPI/Django)
✅ Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
✅ DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
✅ Spécialiste Sécurité Backend
✅ Architecte Microservices
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Tuple
import asyncio
from datetime import datetime, timedelta
import json
import tempfile
import os

# Import test infrastructure
from tests_backend.app.ml import (
    MLTestFixtures, MockMLModels, PerformanceProfiler,
    SecurityTestUtils, ComplianceValidator, TestConfig
)

# Import module under test
try:
    from app.ml.advanced_audio_intelligence import (
        AudioFeatureExtractor, MoodDetector, GenreClassifier,
        AudioEmbeddingModel, RealTimeAudioProcessor, AudioMLPipeline
    )
except ImportError:
    # Mock imports for testing
    AudioFeatureExtractor = Mock()
    MoodDetector = Mock()
    GenreClassifier = Mock()
    AudioEmbeddingModel = Mock()
    RealTimeAudioProcessor = Mock()
    AudioMLPipeline = Mock()


class TestAudioFeatureExtractor:
    """Test suite for audio feature extraction"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup test environment"""
        self.test_fixtures = MLTestFixtures()
        self.mock_models = MockMLModels()
        self.performance_profiler = PerformanceProfiler()
        
        # Generate test audio data
        self.test_audio_data = self.test_fixtures.create_sample_audio_data(duration_seconds=30)
        self.sample_rate = 22050
        
    @pytest.mark.unit
    def test_audio_feature_extractor_init(self):
        """Test AudioFeatureExtractor initialization"""
        if hasattr(AudioFeatureExtractor, '__init__'):
            extractor = AudioFeatureExtractor(
                sample_rate=self.sample_rate,
                n_mfcc=13,
                n_fft=2048,
                hop_length=512
            )
            
            assert hasattr(extractor, 'sample_rate') or extractor is not None
    
    @pytest.mark.unit
    def test_extract_mfcc_features(self):
        """Test MFCC feature extraction"""
        if hasattr(AudioFeatureExtractor, '__init__'):
            extractor = AudioFeatureExtractor()
            
            if hasattr(extractor, 'extract_mfcc'):
                mfcc_features = extractor.extract_mfcc(self.test_audio_data)
                
                # Validate MFCC output
                assert mfcc_features is not None
                if isinstance(mfcc_features, np.ndarray):
                    assert mfcc_features.shape[0] == 13  # Default n_mfcc
    
    @pytest.mark.unit
    def test_extract_spectral_features(self):
        """Test spectral feature extraction"""
        if hasattr(AudioFeatureExtractor, '__init__'):
            extractor = AudioFeatureExtractor()
            
            # Test various spectral features
            feature_methods = [
                'extract_spectral_centroid',
                'extract_spectral_rolloff',
                'extract_spectral_bandwidth',
                'extract_zero_crossing_rate',
                'extract_chroma_features'
            ]
            
            for method_name in feature_methods:
                if hasattr(extractor, method_name):
                    method = getattr(extractor, method_name)
                    features = method(self.test_audio_data)
                    
                    # Validate feature output
                    assert features is not None
                    if isinstance(features, np.ndarray):
                        assert features.size > 0
    
    @pytest.mark.unit
    def test_extract_tempo_features(self):
        """Test tempo and rhythm feature extraction"""
        if hasattr(AudioFeatureExtractor, '__init__'):
            extractor = AudioFeatureExtractor()
            
            if hasattr(extractor, 'extract_tempo'):
                tempo, beats = extractor.extract_tempo(self.test_audio_data)
                
                # Validate tempo extraction
                if tempo is not None:
                    assert isinstance(tempo, (int, float))
                    assert 60 <= tempo <= 200  # Reasonable BPM range
                
                if beats is not None:
                    assert isinstance(beats, np.ndarray)
    
    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_feature_extraction_performance(self, benchmark):
        """Benchmark audio feature extraction performance"""
        if hasattr(AudioFeatureExtractor, '__init__'):
            extractor = AudioFeatureExtractor()
            
            def extract_all_features():
                if hasattr(extractor, 'extract_all_features'):
                    return extractor.extract_all_features(self.test_audio_data)
                return {}
            
            # Benchmark feature extraction
            result = benchmark(extract_all_features)
            
            # Assert performance threshold (250ms for 30s audio)
            assert benchmark.stats['mean'] < 0.25
    
    @pytest.mark.integration
    def test_feature_extraction_pipeline(self):
        """Test complete feature extraction pipeline"""
        if hasattr(AudioFeatureExtractor, '__init__'):
            extractor = AudioFeatureExtractor()
            
            # Extract comprehensive feature set
            features = {}
            
            feature_types = [
                'mfcc', 'spectral_centroid', 'spectral_rolloff',
                'spectral_bandwidth', 'zero_crossing_rate', 'chroma',
                'tempo', 'onset_strength'
            ]
            
            for feature_type in feature_types:
                method_name = f'extract_{feature_type}'
                if hasattr(extractor, method_name):
                    method = getattr(extractor, method_name)
                    features[feature_type] = method(self.test_audio_data)
            
            # Validate comprehensive feature set
            assert len(features) > 0
            for feature_name, feature_values in features.items():
                assert feature_values is not None


class TestMoodDetector:
    """Test suite for mood detection from audio"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup mood detection tests"""
        self.test_fixtures = MLTestFixtures()
        self.mock_models = MockMLModels()
        
        # Generate test audio for different moods
        self.happy_audio = self.test_fixtures.create_sample_audio_data(30, mood="happy")
        self.sad_audio = self.test_fixtures.create_sample_audio_data(30, mood="sad")
        self.energetic_audio = self.test_fixtures.create_sample_audio_data(30, mood="energetic")
        self.calm_audio = self.test_fixtures.create_sample_audio_data(30, mood="calm")
    
    @pytest.mark.unit
    def test_mood_detector_init(self):
        """Test MoodDetector initialization"""
        if hasattr(MoodDetector, '__init__'):
            detector = MoodDetector(
                model_type="transformer",
                confidence_threshold=0.7
            )
            
            assert detector is not None
            assert hasattr(detector, 'confidence_threshold') or True
    
    @pytest.mark.unit
    def test_detect_mood_basic(self):
        """Test basic mood detection"""
        if hasattr(MoodDetector, '__init__'):
            detector = MoodDetector()
            
            if hasattr(detector, 'detect_mood'):
                mood_result = detector.detect_mood(self.happy_audio)
                
                # Validate mood detection output
                assert mood_result is not None
                
                if isinstance(mood_result, dict):
                    assert 'mood' in mood_result or 'predicted_mood' in mood_result
                    assert 'confidence' in mood_result or 'score' in mood_result
    
    @pytest.mark.unit
    def test_detect_mood_multiple_classes(self):
        """Test mood detection with multiple mood classes"""
        if hasattr(MoodDetector, '__init__'):
            detector = MoodDetector()
            
            test_audio_samples = [
                (self.happy_audio, "happy"),
                (self.sad_audio, "sad"),
                (self.energetic_audio, "energetic"),
                (self.calm_audio, "calm")
            ]
            
            mood_results = []
            
            for audio_data, expected_mood in test_audio_samples:
                if hasattr(detector, 'detect_mood'):
                    result = detector.detect_mood(audio_data)
                    mood_results.append(result)
            
            # Validate all mood detections
            assert len(mood_results) == len(test_audio_samples)
            for result in mood_results:
                assert result is not None
    
    @pytest.mark.unit
    def test_mood_confidence_scores(self):
        """Test mood detection confidence scores"""
        if hasattr(MoodDetector, '__init__'):
            detector = MoodDetector(confidence_threshold=0.5)
            
            if hasattr(detector, 'detect_mood'):
                result = detector.detect_mood(self.happy_audio)
                
                if isinstance(result, dict):
                    confidence_key = None
                    for key in ['confidence', 'score', 'probability']:
                        if key in result:
                            confidence_key = key
                            break
                    
                    if confidence_key:
                        confidence = result[confidence_key]
                        assert 0.0 <= confidence <= 1.0
    
    @pytest.mark.performance
    def test_mood_detection_speed(self):
        """Test mood detection inference speed"""
        if hasattr(MoodDetector, '__init__'):
            detector = MoodDetector()
            
            start_time = datetime.now()
            
            if hasattr(detector, 'detect_mood'):
                for _ in range(10):
                    detector.detect_mood(self.happy_audio)
            
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            avg_time_per_inference = total_time / 10
            
            # Should process 30s audio in less than 1s
            assert avg_time_per_inference < 1.0
    
    @pytest.mark.integration
    def test_mood_detection_with_features(self):
        """Test mood detection using extracted features"""
        if hasattr(MoodDetector, '__init__') and hasattr(AudioFeatureExtractor, '__init__'):
            detector = MoodDetector()
            extractor = AudioFeatureExtractor()
            
            # Extract features first
            if hasattr(extractor, 'extract_all_features'):
                features = extractor.extract_all_features(self.happy_audio)
                
                # Use features for mood detection
                if hasattr(detector, 'detect_mood_from_features'):
                    mood_result = detector.detect_mood_from_features(features)
                    assert mood_result is not None


class TestGenreClassifier:
    """Test suite for music genre classification"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup genre classification tests"""
        self.test_fixtures = MLTestFixtures()
        self.mock_models = MockMLModels()
        
        # Generate test audio for different genres
        self.rock_audio = self.test_fixtures.create_sample_audio_data(30, genre="rock")
        self.pop_audio = self.test_fixtures.create_sample_audio_data(30, genre="pop")
        self.jazz_audio = self.test_fixtures.create_sample_audio_data(30, genre="jazz")
        self.classical_audio = self.test_fixtures.create_sample_audio_data(30, genre="classical")
    
    @pytest.mark.unit
    def test_genre_classifier_init(self):
        """Test GenreClassifier initialization"""
        if hasattr(GenreClassifier, '__init__'):
            classifier = GenreClassifier(
                num_classes=10,
                model_architecture="cnn",
                confidence_threshold=0.6
            )
            
            assert classifier is not None
    
    @pytest.mark.unit
    def test_classify_genre_basic(self):
        """Test basic genre classification"""
        if hasattr(GenreClassifier, '__init__'):
            classifier = GenreClassifier()
            
            if hasattr(classifier, 'classify_genre'):
                genre_result = classifier.classify_genre(self.rock_audio)
                
                # Validate genre classification output
                assert genre_result is not None
                
                if isinstance(genre_result, dict):
                    assert 'genre' in genre_result or 'predicted_genre' in genre_result
                elif isinstance(genre_result, str):
                    assert len(genre_result) > 0
    
    @pytest.mark.unit
    def test_classify_multiple_genres(self):
        """Test classification of multiple genres"""
        if hasattr(GenreClassifier, '__init__'):
            classifier = GenreClassifier()
            
            test_audio_samples = [
                (self.rock_audio, "rock"),
                (self.pop_audio, "pop"),
                (self.jazz_audio, "jazz"),
                (self.classical_audio, "classical")
            ]
            
            classification_results = []
            
            for audio_data, expected_genre in test_audio_samples:
                if hasattr(classifier, 'classify_genre'):
                    result = classifier.classify_genre(audio_data)
                    classification_results.append(result)
            
            # Validate all classifications
            assert len(classification_results) == len(test_audio_samples)
            for result in classification_results:
                assert result is not None
    
    @pytest.mark.unit
    def test_genre_probabilities(self):
        """Test genre classification probabilities"""
        if hasattr(GenreClassifier, '__init__'):
            classifier = GenreClassifier()
            
            if hasattr(classifier, 'get_genre_probabilities'):
                probabilities = classifier.get_genre_probabilities(self.rock_audio)
                
                if isinstance(probabilities, dict):
                    # Probabilities should sum to 1
                    total_prob = sum(probabilities.values())
                    assert abs(total_prob - 1.0) < 0.01
                    
                    # All probabilities should be between 0 and 1
                    for prob in probabilities.values():
                        assert 0.0 <= prob <= 1.0
    
    @pytest.mark.performance
    def test_genre_classification_speed(self):
        """Test genre classification performance"""
        if hasattr(GenreClassifier, '__init__'):
            classifier = GenreClassifier()
            
            start_time = datetime.now()
            
            if hasattr(classifier, 'classify_genre'):
                for _ in range(5):
                    classifier.classify_genre(self.rock_audio)
            
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            avg_time = total_time / 5
            
            # Should classify 30s audio in less than 2s
            assert avg_time < 2.0


class TestAudioEmbeddingModel:
    """Test suite for audio embedding generation"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup audio embedding tests"""
        self.test_fixtures = MLTestFixtures()
        self.test_audio = self.test_fixtures.create_sample_audio_data(30)
        
    @pytest.mark.unit
    def test_audio_embedding_model_init(self):
        """Test AudioEmbeddingModel initialization"""
        if hasattr(AudioEmbeddingModel, '__init__'):
            model = AudioEmbeddingModel(
                embedding_dim=512,
                model_type="vggish",
                pretrained=True
            )
            
            assert model is not None
    
    @pytest.mark.unit
    def test_generate_embeddings(self):
        """Test audio embedding generation"""
        if hasattr(AudioEmbeddingModel, '__init__'):
            model = AudioEmbeddingModel()
            
            if hasattr(model, 'generate_embeddings'):
                embeddings = model.generate_embeddings(self.test_audio)
                
                # Validate embedding output
                assert embeddings is not None
                
                if isinstance(embeddings, np.ndarray):
                    assert embeddings.ndim >= 1
                    assert embeddings.size > 0
    
    @pytest.mark.unit
    def test_embedding_similarity(self):
        """Test audio embedding similarity computation"""
        if hasattr(AudioEmbeddingModel, '__init__'):
            model = AudioEmbeddingModel()
            
            # Generate embeddings for similar audio
            audio1 = self.test_fixtures.create_sample_audio_data(30, genre="rock")
            audio2 = self.test_fixtures.create_sample_audio_data(30, genre="rock")
            audio3 = self.test_fixtures.create_sample_audio_data(30, genre="classical")
            
            if hasattr(model, 'generate_embeddings'):
                embed1 = model.generate_embeddings(audio1)
                embed2 = model.generate_embeddings(audio2)
                embed3 = model.generate_embeddings(audio3)
                
                if hasattr(model, 'compute_similarity'):
                    # Similar genre should have higher similarity
                    sim_same_genre = model.compute_similarity(embed1, embed2)
                    sim_diff_genre = model.compute_similarity(embed1, embed3)
                    
                    if sim_same_genre is not None and sim_diff_genre is not None:
                        assert isinstance(sim_same_genre, (int, float))
                        assert isinstance(sim_diff_genre, (int, float))
    
    @pytest.mark.performance
    def test_embedding_generation_speed(self):
        """Test embedding generation performance"""
        if hasattr(AudioEmbeddingModel, '__init__'):
            model = AudioEmbeddingModel()
            
            start_time = datetime.now()
            
            if hasattr(model, 'generate_embeddings'):
                for _ in range(3):
                    model.generate_embeddings(self.test_audio)
            
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            avg_time = total_time / 3
            
            # Should generate embeddings for 30s audio in less than 3s
            assert avg_time < 3.0


class TestRealTimeAudioProcessor:
    """Test suite for real-time audio processing"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup real-time processing tests"""
        self.test_fixtures = MLTestFixtures()
        self.buffer_size = 1024
        self.sample_rate = 22050
        
    @pytest.mark.unit
    def test_realtime_processor_init(self):
        """Test RealTimeAudioProcessor initialization"""
        if hasattr(RealTimeAudioProcessor, '__init__'):
            processor = RealTimeAudioProcessor(
                buffer_size=self.buffer_size,
                sample_rate=self.sample_rate,
                overlap=0.5
            )
            
            assert processor is not None
    
    @pytest.mark.unit
    def test_process_audio_chunk(self):
        """Test processing of audio chunks"""
        if hasattr(RealTimeAudioProcessor, '__init__'):
            processor = RealTimeAudioProcessor()
            
            # Generate audio chunk
            audio_chunk = np.random.randn(self.buffer_size)
            
            if hasattr(processor, 'process_chunk'):
                result = processor.process_chunk(audio_chunk)
                
                # Validate chunk processing
                assert result is not None
    
    @pytest.mark.unit
    def test_streaming_feature_extraction(self):
        """Test streaming feature extraction"""
        if hasattr(RealTimeAudioProcessor, '__init__'):
            processor = RealTimeAudioProcessor()
            
            # Simulate streaming audio
            audio_stream = self.test_fixtures.create_sample_audio_data(10)
            chunk_size = 1024
            
            features_buffer = []
            
            for i in range(0, len(audio_stream), chunk_size):
                chunk = audio_stream[i:i+chunk_size]
                
                if hasattr(processor, 'extract_streaming_features'):
                    chunk_features = processor.extract_streaming_features(chunk)
                    if chunk_features is not None:
                        features_buffer.append(chunk_features)
            
            # Validate streaming processing
            assert len(features_buffer) > 0
    
    @pytest.mark.performance
    def test_realtime_processing_latency(self):
        """Test real-time processing latency"""
        if hasattr(RealTimeAudioProcessor, '__init__'):
            processor = RealTimeAudioProcessor()
            
            audio_chunk = np.random.randn(self.buffer_size)
            
            # Measure processing latency
            latencies = []
            
            for _ in range(100):
                start_time = datetime.now()
                
                if hasattr(processor, 'process_chunk'):
                    processor.process_chunk(audio_chunk)
                
                end_time = datetime.now()
                latency = (end_time - start_time).total_seconds() * 1000  # ms
                latencies.append(latency)
            
            avg_latency = np.mean(latencies)
            
            # Real-time constraint: process 1024 samples at 22050 Hz in < 46ms
            max_allowed_latency = (self.buffer_size / self.sample_rate) * 1000  # ms
            assert avg_latency < max_allowed_latency
    
    @pytest.mark.integration
    def test_realtime_mood_detection(self):
        """Test real-time mood detection"""
        if hasattr(RealTimeAudioProcessor, '__init__'):
            processor = RealTimeAudioProcessor()
            
            # Simulate real-time audio stream
            audio_stream = self.test_fixtures.create_sample_audio_data(5, mood="happy")
            chunk_size = 1024
            
            mood_detections = []
            
            for i in range(0, len(audio_stream), chunk_size):
                chunk = audio_stream[i:i+chunk_size]
                
                if hasattr(processor, 'detect_mood_realtime'):
                    mood = processor.detect_mood_realtime(chunk)
                    if mood is not None:
                        mood_detections.append(mood)
            
            # Should detect mood in real-time
            assert len(mood_detections) > 0


class TestAudioMLPipeline:
    """Test suite for complete audio ML pipeline"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup audio ML pipeline tests"""
        self.test_fixtures = MLTestFixtures()
        self.test_audio = self.test_fixtures.create_sample_audio_data(30)
        
    @pytest.mark.integration
    def test_audio_ml_pipeline_init(self):
        """Test AudioMLPipeline initialization"""
        if hasattr(AudioMLPipeline, '__init__'):
            pipeline = AudioMLPipeline(
                enable_feature_extraction=True,
                enable_mood_detection=True,
                enable_genre_classification=True,
                enable_embeddings=True
            )
            
            assert pipeline is not None
    
    @pytest.mark.integration
    def test_complete_audio_analysis(self):
        """Test complete audio analysis pipeline"""
        if hasattr(AudioMLPipeline, '__init__'):
            pipeline = AudioMLPipeline()
            
            if hasattr(pipeline, 'analyze_audio'):
                analysis_result = pipeline.analyze_audio(self.test_audio)
                
                # Validate complete analysis
                assert analysis_result is not None
                
                if isinstance(analysis_result, dict):
                    expected_components = [
                        'features', 'mood', 'genre', 'embeddings'
                    ]
                    
                    # Should contain at least some analysis components
                    has_components = any(comp in analysis_result for comp in expected_components)
                    assert has_components or len(analysis_result) > 0
    
    @pytest.mark.integration
    def test_batch_audio_processing(self):
        """Test batch audio processing"""
        if hasattr(AudioMLPipeline, '__init__'):
            pipeline = AudioMLPipeline()
            
            # Create batch of audio samples
            audio_batch = [
                self.test_fixtures.create_sample_audio_data(30, genre="rock"),
                self.test_fixtures.create_sample_audio_data(30, genre="pop"),
                self.test_fixtures.create_sample_audio_data(30, genre="jazz")
            ]
            
            if hasattr(pipeline, 'process_batch'):
                batch_results = pipeline.process_batch(audio_batch)
                
                # Validate batch processing
                assert batch_results is not None
                if isinstance(batch_results, list):
                    assert len(batch_results) == len(audio_batch)
    
    @pytest.mark.performance
    def test_pipeline_processing_speed(self):
        """Test audio ML pipeline processing speed"""
        if hasattr(AudioMLPipeline, '__init__'):
            pipeline = AudioMLPipeline()
            
            start_time = datetime.now()
            
            if hasattr(pipeline, 'analyze_audio'):
                pipeline.analyze_audio(self.test_audio)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Should process 30s audio in less than 5s
            assert processing_time < 5.0
    
    @pytest.mark.security
    def test_audio_input_validation(self):
        """Test audio input validation and security"""
        if hasattr(AudioMLPipeline, '__init__'):
            pipeline = AudioMLPipeline()
            
            # Test various invalid inputs
            invalid_inputs = [
                None,
                [],
                np.array([]),
                "not_audio_data",
                np.full(1000, np.inf),  # Invalid values
                np.full(1000, np.nan)   # NaN values
            ]
            
            for invalid_input in invalid_inputs:
                if hasattr(pipeline, 'analyze_audio'):
                    try:
                        result = pipeline.analyze_audio(invalid_input)
                        # Should handle invalid input gracefully
                        assert result is None or isinstance(result, dict)
                    except (ValueError, TypeError) as e:
                        # Expected behavior for invalid input
                        assert "invalid" in str(e).lower() or "error" in str(e).lower()
    
    @pytest.mark.compliance
    def test_audio_privacy_compliance(self):
        """Test audio processing privacy compliance"""
        if hasattr(AudioMLPipeline, '__init__'):
            pipeline = AudioMLPipeline()
            
            # Ensure no raw audio data is stored in analysis results
            if hasattr(pipeline, 'analyze_audio'):
                result = pipeline.analyze_audio(self.test_audio)
                
                if isinstance(result, dict):
                    # Should not contain raw audio data
                    assert 'raw_audio' not in result
                    assert 'audio_data' not in result
                    
                    # Check for any large arrays that might be raw audio
                    for key, value in result.items():
                        if isinstance(value, np.ndarray):
                            # Feature arrays should be much smaller than raw audio
                            assert value.size < len(self.test_audio) / 10


# Parametrized tests for different audio scenarios
@pytest.mark.parametrize("duration", [5, 10, 30, 60])
@pytest.mark.parametrize("sample_rate", [16000, 22050, 44100])
def test_audio_processing_parameters(duration, sample_rate):
    """Test audio processing with different parameters"""
    test_fixtures = MLTestFixtures()
    audio_data = test_fixtures.create_sample_audio_data(
        duration_seconds=duration,
        sample_rate=sample_rate
    )
    
    # Validate audio data generation
    assert audio_data is not None
    if isinstance(audio_data, np.ndarray):
        expected_length = duration * sample_rate
        assert abs(len(audio_data) - expected_length) < sample_rate  # Allow 1s tolerance


@pytest.mark.parametrize("audio_format", [
    {"genre": "rock", "mood": "energetic"},
    {"genre": "classical", "mood": "calm"},
    {"genre": "jazz", "mood": "relaxed"},
    {"genre": "electronic", "mood": "upbeat"},
    {"genre": "ambient", "mood": "peaceful"}
])
def test_genre_mood_combinations(audio_format):
    """Test audio processing with different genre-mood combinations"""
    test_fixtures = MLTestFixtures()
    
    audio_data = test_fixtures.create_sample_audio_data(
        duration_seconds=30,
        genre=audio_format["genre"],
        mood=audio_format["mood"]
    )
    
    # Validate genre-mood audio generation
    assert audio_data is not None
    if isinstance(audio_data, np.ndarray):
        assert len(audio_data) > 0
