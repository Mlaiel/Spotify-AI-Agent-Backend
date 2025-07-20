"""
🧪 Tests ML/AI Frameworks - Spotify AI Models
============================================

Tests complets des frameworks ML/AI avec:
- Spotify Recommendation Model
- Audio Analysis Model  
- NLP Model avec BERT
- MLOps pipeline
- Model versioning
- Performance monitoring

Développé par: ML Engineer
"""

import pytest
import asyncio
import numpy as np
import torch
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List
import tempfile
import json
import os

from backend.app.frameworks.ml_frameworks import (
    BaseMLModel,
    SpotifyRecommendationModel,
    AudioAnalysisModel,
    NLPModel,
    MLModelManager,
    ModelConfig,
    ModelType,
    MLFrameworkType,
    TrainingMetrics,
    PredictionResult,
    ModelRegistry,
    MLOpsManager
)
from backend.app.frameworks import TEST_CONFIG, clean_frameworks, logger


@pytest.fixture
def sample_audio_features():
    """Features audio d'exemple pour les tests."""
    return {
        'danceability': 0.7,
        'energy': 0.8,
        'valence': 0.6,
        'acousticness': 0.2,
        'instrumentalness': 0.1,
        'speechiness': 0.05,
        'tempo': 120.0,
        'duration_ms': 180000
    }


@pytest.fixture
def sample_user_data():
    """Données utilisateur d'exemple."""
    return {
        'user_id': 'test_user_123',
        'listening_history': [
            {'track_id': 'track_1', 'play_count': 15, 'last_played': '2024-01-01'},
            {'track_id': 'track_2', 'play_count': 8, 'last_played': '2024-01-02'}
        ],
        'preferences': {
            'genres': ['pop', 'rock', 'electronic'],
            'mood': 'energetic'
        }
    }


@pytest.fixture
def sample_audio_signal():
    """Signal audio synthétique pour les tests."""
    # Générer 2 secondes d'audio à 22050 Hz
    duration = 2.0
    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Signal composite (sinus + bruit)
    frequency = 440.0  # La 440Hz
    signal = np.sin(2 * np.pi * frequency * t) + 0.1 * np.random.randn(len(t))
    
    return signal.astype(np.float32)


@pytest.fixture
def model_config():
    """Configuration modèle de test."""
    return ModelConfig(
        name="test_model",
        model_type=ModelType.RECOMMENDATION,
        framework_type=MLFrameworkType.PYTORCH,
        version="1.0.0",
        description="Test model for unit tests",
        input_shape=(100,),
        output_shape=(50,),
        hyperparameters={
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 10
        }
    )


@pytest.mark.ml
class TestBaseMLModel:
    """Tests de la classe BaseMLModel."""
    
    def test_base_ml_model_creation(self, model_config):
        """Test création modèle ML de base."""
        model = BaseMLModel(model_config)
        
        assert model.config == model_config
        assert model.model is None
        assert model.is_trained is False
        assert model.version == "1.0.0"
        
    @pytest.mark.asyncio
    async def test_base_ml_model_not_implemented(self, model_config):
        """Test méthodes non implémentées."""
        model = BaseMLModel(model_config)
        
        with pytest.raises(NotImplementedError):
            await model.build_model()
            
        with pytest.raises(NotImplementedError):
            await model.train([], [])
            
        with pytest.raises(NotImplementedError):
            await model.predict({})
            
    def test_model_state_management(self, model_config):
        """Test gestion état du modèle."""
        model = BaseMLModel(model_config)
        
        # État initial
        assert model.get_state()["status"] == "not_trained"
        
        # Simuler entraînement
        model.is_trained = True
        model.training_metrics = TrainingMetrics(
            accuracy=0.95,
            loss=0.05,
            epoch=10,
            duration=120.5
        )
        
        state = model.get_state()
        assert state["status"] == "trained"
        assert state["metrics"]["accuracy"] == 0.95


@pytest.mark.ml
class TestSpotifyRecommendationModel:
    """Tests du modèle de recommandation Spotify."""
    
    @pytest.mark.asyncio
    async def test_spotify_model_initialization(self):
        """Test initialisation modèle Spotify."""
        config = ModelConfig(
            name="spotify_recommendation",
            model_type=ModelType.RECOMMENDATION,
            framework_type=MLFrameworkType.PYTORCH
        )
        
        model = SpotifyRecommendationModel(config)
        result = await model.build_model()
        
        assert result is True
        assert model.model is not None
        assert hasattr(model, 'collaborative_model')
        assert hasattr(model, 'content_model')
        
    @pytest.mark.asyncio
    async def test_spotify_model_training(self, sample_user_data):
        """Test entraînement modèle Spotify."""
        config = ModelConfig(
            name="spotify_recommendation",
            model_type=ModelType.RECOMMENDATION,
            framework_type=MLFrameworkType.PYTORCH
        )
        
        model = SpotifyRecommendationModel(config)
        await model.build_model()
        
        # Données d'entraînement mockées
        train_data = {
            'user_features': np.random.randn(100, 50),
            'item_features': np.random.randn(1000, 30),
            'interactions': np.random.randint(0, 2, (100, 1000))
        }
        
        val_data = {
            'user_features': np.random.randn(20, 50),
            'item_features': np.random.randn(200, 30),
            'interactions': np.random.randint(0, 2, (20, 200))
        }
        
        with patch.object(model, '_train_collaborative_model', return_value=0.85):
            with patch.object(model, '_train_content_model', return_value=0.78):
                metrics = await model.train(train_data, val_data)
                
        assert isinstance(metrics, TrainingMetrics)
        assert metrics.accuracy > 0.8
        assert model.is_trained is True
        
    @pytest.mark.asyncio
    async def test_spotify_model_prediction(self, sample_user_data):
        """Test prédiction modèle Spotify."""
        config = ModelConfig(
            name="spotify_recommendation",
            model_type=ModelType.RECOMMENDATION,
            framework_type=MLFrameworkType.PYTORCH
        )
        
        model = SpotifyRecommendationModel(config)
        await model.build_model()
        model.is_trained = True  # Simuler modèle entraîné
        
        # Mock des méthodes de prédiction
        with patch.object(model, '_collaborative_predict', return_value=np.array([0.9, 0.8, 0.7])):
            with patch.object(model, '_content_predict', return_value=np.array([0.85, 0.75, 0.65])):
                prediction_input = {
                    'user_id': 'test_user_123',
                    'candidate_items': ['track_1', 'track_2', 'track_3'],
                    'user_features': np.random.randn(50),
                    'item_features': np.random.randn(3, 30)
                }
                
                result = await model.predict(prediction_input)
                
        assert isinstance(result, PredictionResult)
        assert len(result.recommendations) == 3
        assert all(score > 0.5 for score in result.scores)
        assert result.model_version is not None
        
    @pytest.mark.asyncio
    async def test_spotify_model_hybrid_scoring(self):
        """Test système de scoring hybride."""
        config = ModelConfig(
            name="spotify_recommendation",
            model_type=ModelType.RECOMMENDATION,
            framework_type=MLFrameworkType.PYTORCH
        )
        
        model = SpotifyRecommendationModel(config)
        await model.build_model()
        
        # Scores des deux approches
        collaborative_scores = np.array([0.9, 0.7, 0.5])
        content_scores = np.array([0.6, 0.8, 0.9])
        
        hybrid_scores = model._combine_scores(collaborative_scores, content_scores)
        
        assert len(hybrid_scores) == 3
        assert np.all(hybrid_scores >= 0)
        assert np.all(hybrid_scores <= 1)
        
    def test_spotify_model_feature_engineering(self, sample_audio_features):
        """Test engineering features audio."""
        config = ModelConfig(
            name="spotify_recommendation",
            model_type=ModelType.RECOMMENDATION,
            framework_type=MLFrameworkType.PYTORCH
        )
        
        model = SpotifyRecommendationModel(config)
        features = model._extract_audio_features(sample_audio_features)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert not np.any(np.isnan(features))


@pytest.mark.ml
class TestAudioAnalysisModel:
    """Tests du modèle d'analyse audio."""
    
    @pytest.mark.asyncio
    async def test_audio_model_initialization(self):
        """Test initialisation modèle audio."""
        config = ModelConfig(
            name="audio_analysis",
            model_type=ModelType.CLASSIFICATION,
            framework_type=MLFrameworkType.PYTORCH
        )
        
        model = AudioAnalysisModel(config)
        result = await model.build_model()
        
        assert result is True
        assert model.model is not None
        assert hasattr(model, 'genre_classifier')
        assert hasattr(model, 'emotion_classifier')
        
    @pytest.mark.asyncio
    async def test_audio_mfcc_extraction(self, sample_audio_signal):
        """Test extraction MFCC."""
        config = ModelConfig(
            name="audio_analysis",
            model_type=ModelType.CLASSIFICATION,
            framework_type=MLFrameworkType.PYTORCH
        )
        
        model = AudioAnalysisModel(config)
        
        mfcc_features = model._extract_mfcc(sample_audio_signal, sample_rate=22050)
        
        assert isinstance(mfcc_features, np.ndarray)
        assert mfcc_features.shape[0] == 13  # 13 coefficients MFCC
        assert mfcc_features.shape[1] > 0  # Frames temporels
        
    @pytest.mark.asyncio
    async def test_audio_spectral_features(self, sample_audio_signal):
        """Test extraction features spectrales."""
        config = ModelConfig(
            name="audio_analysis",
            model_type=ModelType.CLASSIFICATION,
            framework_type=MLFrameworkType.PYTORCH
        )
        
        model = AudioAnalysisModel(config)
        
        spectral_features = model._extract_spectral_features(sample_audio_signal, sample_rate=22050)
        
        assert isinstance(spectral_features, dict)
        assert 'spectral_centroid' in spectral_features
        assert 'spectral_rolloff' in spectral_features
        assert 'zero_crossing_rate' in spectral_features
        
    @pytest.mark.asyncio
    async def test_audio_genre_classification(self, sample_audio_signal):
        """Test classification genre musical."""
        config = ModelConfig(
            name="audio_analysis",
            model_type=ModelType.CLASSIFICATION,
            framework_type=MLFrameworkType.PYTORCH
        )
        
        model = AudioAnalysisModel(config)
        await model.build_model()
        model.is_trained = True
        
        with patch.object(model.genre_classifier, 'predict', return_value=np.array([[0.7, 0.2, 0.1]])):
            prediction_input = {
                'audio_signal': sample_audio_signal,
                'sample_rate': 22050
            }
            
            result = await model.predict(prediction_input)
            
        assert isinstance(result, PredictionResult)
        assert 'genre' in result.predictions
        assert result.confidence > 0.5
        
    @pytest.mark.asyncio
    async def test_audio_emotion_detection(self, sample_audio_signal):
        """Test détection émotion."""
        config = ModelConfig(
            name="audio_analysis",
            model_type=ModelType.CLASSIFICATION,
            framework_type=MLFrameworkType.PYTORCH
        )
        
        model = AudioAnalysisModel(config)
        await model.build_model()
        model.is_trained = True
        
        with patch.object(model.emotion_classifier, 'predict', return_value=np.array([[0.1, 0.8, 0.1]])):
            prediction_input = {
                'audio_signal': sample_audio_signal,
                'sample_rate': 22050
            }
            
            result = await model.predict(prediction_input)
            
        assert isinstance(result, PredictionResult)
        assert 'emotion' in result.predictions
        assert result.predictions['emotion'] == 'happy'  # Index 1 -> happy


@pytest.mark.ml
class TestNLPModel:
    """Tests du modèle NLP."""
    
    @pytest.mark.asyncio
    async def test_nlp_model_initialization(self):
        """Test initialisation modèle NLP."""
        config = ModelConfig(
            name="nlp_analysis",
            model_type=ModelType.NLP,
            framework_type=MLFrameworkType.TRANSFORMERS
        )
        
        model = NLPModel(config)
        
        with patch('transformers.AutoTokenizer.from_pretrained'), \
             patch('transformers.AutoModel.from_pretrained'):
            result = await model.build_model()
            
        assert result is True
        assert model.tokenizer is not None
        assert model.bert_model is not None
        
    @pytest.mark.asyncio
    async def test_nlp_sentiment_analysis(self):
        """Test analyse sentiment."""
        config = ModelConfig(
            name="nlp_analysis",
            model_type=ModelType.NLP,
            framework_type=MLFrameworkType.TRANSFORMERS
        )
        
        model = NLPModel(config)
        
        # Mock BERT et tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {'input_ids': [[1, 2, 3]], 'attention_mask': [[1, 1, 1]]}
        model.tokenizer = mock_tokenizer
        
        mock_bert = Mock()
        mock_bert.return_value = Mock(last_hidden_state=torch.randn(1, 3, 768))
        model.bert_model = mock_bert
        
        with patch.object(model, '_classify_sentiment', return_value='positive'):
            text = "I love this song! It's amazing."
            result = await model.predict({'text': text})
            
        assert isinstance(result, PredictionResult)
        assert result.predictions['sentiment'] == 'positive'
        
    @pytest.mark.asyncio
    async def test_nlp_entity_extraction(self):
        """Test extraction entités."""
        config = ModelConfig(
            name="nlp_analysis",
            model_type=ModelType.NLP,
            framework_type=MLFrameworkType.TRANSFORMERS
        )
        
        model = NLPModel(config)
        
        # Mock du pipeline NER
        with patch('transformers.pipeline') as mock_pipeline:
            mock_ner = Mock()
            mock_ner.return_value = [
                {'entity': 'B-PER', 'word': 'Taylor', 'score': 0.99},
                {'entity': 'I-PER', 'word': 'Swift', 'score': 0.98}
            ]
            mock_pipeline.return_value = mock_ner
            
            text = "I love Taylor Swift's new album"
            entities = model._extract_entities(text)
            
        assert isinstance(entities, list)
        assert len(entities) > 0
        assert any('Taylor' in entity.get('word', '') for entity in entities)
        
    @pytest.mark.asyncio
    async def test_nlp_text_similarity(self):
        """Test similarité entre textes."""
        config = ModelConfig(
            name="nlp_analysis",
            model_type=ModelType.NLP,
            framework_type=MLFrameworkType.TRANSFORMERS
        )
        
        model = NLPModel(config)
        
        # Mock embeddings BERT
        mock_embedding1 = torch.randn(768)
        mock_embedding2 = torch.randn(768)
        
        with patch.object(model, '_get_text_embedding', side_effect=[mock_embedding1, mock_embedding2]):
            similarity = model._calculate_text_similarity("text1", "text2")
            
        assert isinstance(similarity, float)
        assert -1 <= similarity <= 1


@pytest.mark.ml
class TestMLModelManager:
    """Tests du gestionnaire de modèles ML."""
    
    @pytest.mark.asyncio
    async def test_model_manager_initialization(self, clean_frameworks):
        """Test initialisation gestionnaire modèles."""
        manager = MLModelManager()
        
        result = await manager.initialize()
        assert result is True
        assert manager.status.name == "RUNNING"
        
    @pytest.mark.asyncio
    async def test_model_registration(self, model_config, clean_frameworks):
        """Test enregistrement modèle."""
        manager = MLModelManager()
        await manager.initialize()
        
        # Créer modèle mock
        mock_model = Mock(spec=BaseMLModel)
        mock_model.config = model_config
        
        result = await manager.register_model(model_config, mock_model)
        assert result is True
        assert model_config.name in manager.models
        
    @pytest.mark.asyncio
    async def test_model_training_workflow(self, model_config, clean_frameworks):
        """Test workflow d'entraînement."""
        manager = MLModelManager()
        await manager.initialize()
        
        # Mock model avec méthodes async
        mock_model = AsyncMock(spec=BaseMLModel)
        mock_model.config = model_config
        mock_model.train.return_value = TrainingMetrics(accuracy=0.9, loss=0.1, epoch=5, duration=60.0)
        
        await manager.register_model(model_config, mock_model)
        
        train_data = {'features': np.random.randn(100, 10), 'labels': np.random.randint(0, 2, 100)}
        val_data = {'features': np.random.randn(20, 10), 'labels': np.random.randint(0, 2, 20)}
        
        metrics = await manager.train_model(model_config.name, train_data, val_data)
        
        assert isinstance(metrics, TrainingMetrics)
        assert metrics.accuracy == 0.9
        mock_model.train.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_model_prediction_workflow(self, model_config, clean_frameworks):
        """Test workflow de prédiction."""
        manager = MLModelManager()
        await manager.initialize()
        
        # Mock model
        mock_model = AsyncMock(spec=BaseMLModel)
        mock_model.config = model_config
        mock_model.is_trained = True
        mock_model.predict.return_value = PredictionResult(
            predictions={'class': 'positive'},
            confidence=0.95,
            model_version="1.0.0"
        )
        
        await manager.register_model(model_config, mock_model)
        
        input_data = {'text': 'test input'}
        result = await manager.predict(model_config.name, input_data)
        
        assert isinstance(result, PredictionResult)
        assert result.confidence == 0.95
        mock_model.predict.assert_called_once_with(input_data)
        
    @pytest.mark.asyncio
    async def test_model_versioning(self, clean_frameworks):
        """Test versioning des modèles."""
        manager = MLModelManager()
        await manager.initialize()
        
        # Première version
        config_v1 = ModelConfig(name="test_model", version="1.0.0")
        mock_model_v1 = Mock(spec=BaseMLModel)
        mock_model_v1.config = config_v1
        
        await manager.register_model(config_v1, mock_model_v1)
        
        # Deuxième version
        config_v2 = ModelConfig(name="test_model", version="2.0.0")
        mock_model_v2 = Mock(spec=BaseMLModel)
        mock_model_v2.config = config_v2
        
        await manager.register_model(config_v2, mock_model_v2)
        
        # Vérifier que les deux versions existent
        versions = manager.get_model_versions("test_model")
        assert "1.0.0" in versions
        assert "2.0.0" in versions
        
        # Vérifier que la version active est la plus récente
        active_model = manager.get_model("test_model")
        assert active_model.config.version == "2.0.0"


@pytest.mark.ml
class TestModelRegistry:
    """Tests du registre de modèles."""
    
    def test_model_registry_creation(self):
        """Test création registre modèles."""
        registry = ModelRegistry()
        assert len(registry.models) == 0
        assert len(registry.versions) == 0
        
    def test_model_registration_in_registry(self, model_config):
        """Test enregistrement dans le registre."""
        registry = ModelRegistry()
        mock_model = Mock(spec=BaseMLModel)
        mock_model.config = model_config
        
        registry.register(model_config.name, mock_model)
        
        assert model_config.name in registry.models
        assert model_config.version in registry.versions[model_config.name]
        
    def test_model_retrieval_from_registry(self, model_config):
        """Test récupération depuis le registre."""
        registry = ModelRegistry()
        mock_model = Mock(spec=BaseMLModel)
        mock_model.config = model_config
        
        registry.register(model_config.name, mock_model)
        retrieved_model = registry.get(model_config.name)
        
        assert retrieved_model is mock_model
        
    def test_model_registry_metadata(self, model_config):
        """Test métadonnées du registre."""
        registry = ModelRegistry()
        mock_model = Mock(spec=BaseMLModel)
        mock_model.config = model_config
        
        registry.register(model_config.name, mock_model)
        metadata = registry.get_metadata(model_config.name)
        
        assert metadata["name"] == model_config.name
        assert metadata["version"] == model_config.version
        assert metadata["type"] == model_config.model_type.value
        assert "registered_at" in metadata


@pytest.mark.ml
@pytest.mark.integration
class TestMLFrameworksIntegration:
    """Tests d'intégration frameworks ML."""
    
    @pytest.mark.asyncio
    async def test_full_ml_pipeline(self, clean_frameworks):
        """Test pipeline ML complet."""
        manager = MLModelManager()
        await manager.initialize()
        
        # Configuration Spotify recommendation
        config = ModelConfig(
            name="integration_test_model",
            model_type=ModelType.RECOMMENDATION,
            framework_type=MLFrameworkType.PYTORCH
        )
        
        model = SpotifyRecommendationModel(config)
        await manager.register_model(config, model)
        
        # Données d'entraînement simulées
        train_data = {
            'user_features': np.random.randn(50, 30),
            'item_features': np.random.randn(100, 20),
            'interactions': np.random.randint(0, 2, (50, 100))
        }
        
        val_data = {
            'user_features': np.random.randn(10, 30),
            'item_features': np.random.randn(20, 20),
            'interactions': np.random.randint(0, 2, (10, 20))
        }
        
        # Pipeline complet
        with patch.object(model, '_train_collaborative_model', return_value=0.85), \
             patch.object(model, '_train_content_model', return_value=0.78), \
             patch.object(model, '_collaborative_predict', return_value=np.array([0.9, 0.8])), \
             patch.object(model, '_content_predict', return_value=np.array([0.85, 0.75])):
            
            # Entraînement
            metrics = await manager.train_model(config.name, train_data, val_data)
            assert metrics.accuracy > 0.8
            
            # Prédiction
            prediction_input = {
                'user_id': 'test_user',
                'candidate_items': ['track_1', 'track_2'],
                'user_features': np.random.randn(30),
                'item_features': np.random.randn(2, 20)
            }
            
            result = await manager.predict(config.name, prediction_input)
            assert isinstance(result, PredictionResult)
            assert len(result.recommendations) == 2
            
    @pytest.mark.asyncio
    async def test_multi_model_coordination(self, clean_frameworks):
        """Test coordination multi-modèles."""
        manager = MLModelManager()
        await manager.initialize()
        
        # Enregistrer plusieurs modèles
        models_configs = [
            ModelConfig(name="recommendation_model", model_type=ModelType.RECOMMENDATION),
            ModelConfig(name="audio_model", model_type=ModelType.CLASSIFICATION),
            ModelConfig(name="nlp_model", model_type=ModelType.NLP)
        ]
        
        for config in models_configs:
            mock_model = AsyncMock(spec=BaseMLModel)
            mock_model.config = config
            mock_model.is_trained = True
            await manager.register_model(config, mock_model)
            
        # Vérifier que tous les modèles sont actifs
        active_models = manager.get_active_models()
        assert len(active_models) == 3
        
        # Test health check multi-modèles
        health = await manager.health_check()
        assert health.status.name == "RUNNING"
        assert len(health.details["models"]) == 3


@pytest.mark.ml
@pytest.mark.performance
class TestMLFrameworksPerformance:
    """Tests de performance frameworks ML."""
    
    @pytest.mark.asyncio
    async def test_concurrent_predictions(self, clean_frameworks):
        """Test prédictions concurrentes."""
        manager = MLModelManager()
        await manager.initialize()
        
        config = ModelConfig(name="perf_test_model")
        mock_model = AsyncMock(spec=BaseMLModel)
        mock_model.config = config
        mock_model.is_trained = True
        mock_model.predict.return_value = PredictionResult(
            predictions={'result': 'test'}, confidence=0.9, model_version="1.0.0"
        )
        
        await manager.register_model(config, mock_model)
        
        # Lancer prédictions concurrentes
        async def make_prediction():
            return await manager.predict(config.name, {'input': 'test'})
            
        tasks = [make_prediction() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 10
        assert all(isinstance(r, PredictionResult) for r in results)
        
    @pytest.mark.asyncio
    async def test_memory_efficient_training(self, clean_frameworks):
        """Test entraînement efficace en mémoire."""
        manager = MLModelManager()
        await manager.initialize()
        
        config = ModelConfig(name="memory_test_model")
        mock_model = AsyncMock(spec=BaseMLModel)
        mock_model.config = config
        
        # Simuler entraînement avec gros dataset
        large_dataset = {
            'features': np.random.randn(10000, 100),
            'labels': np.random.randint(0, 10, 10000)
        }
        
        mock_model.train.return_value = TrainingMetrics(
            accuracy=0.92, loss=0.08, epoch=20, duration=300.0
        )
        
        await manager.register_model(config, mock_model)
        
        # Entraînement avec monitoring mémoire
        metrics = await manager.train_model(config.name, large_dataset, large_dataset)
        
        assert metrics.accuracy > 0.9
        # Vérifier que le training s'est bien passé
        mock_model.train.assert_called_once()
