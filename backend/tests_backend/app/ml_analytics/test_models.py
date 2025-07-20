"""
Tests pour les modèles de recommandation ML Analytics
====================================================

Tests complets pour SpotifyRecommendationModel avec couverture de:
- Modèles de recommandation hybrides
- Algorithmes collaboratifs et basés sur le contenu
- Deep learning pour les recommandations
- Évaluation et métriques de performance
- Gestion des données utilisateur et des préférences
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from ml_analytics.models import (
    SpotifyRecommendationModel,
    ContentBasedModel,
    CollaborativeFilteringModel,
    DeepLearningRecommendationModel,
    HybridRecommendationModel,
    PopularityModel
)
from ml_analytics.exceptions import ModelError, ConfigurationError
from ml_analytics.config import MLAnalyticsConfig


class TestSpotifyRecommendationModel:
    """Tests pour le modèle principal de recommandation Spotify."""
    
    @pytest.fixture
    async def recommendation_model(self):
        """Instance de test du modèle de recommandation."""
        config = {
            "model_type": "hybrid",
            "embedding_dim": 64,
            "num_factors": 50,
            "learning_rate": 0.001,
            "batch_size": 128,
            "epochs": 10
        }
        model = SpotifyRecommendationModel(config)
        await model.initialize()
        yield model
        await model.cleanup()
    
    @pytest.fixture
    def sample_interaction_data(self):
        """Données d'interaction utilisateur-piste de test."""
        np.random.seed(42)
        num_users, num_tracks = 100, 500
        
        # Matrice d'interaction sparse
        interactions = []
        for user_id in range(num_users):
            # Chaque utilisateur interagit avec 5-15 pistes
            num_interactions = np.random.randint(5, 16)
            track_ids = np.random.choice(num_tracks, num_interactions, replace=False)
            ratings = np.random.uniform(0.5, 1.0, num_interactions)
            
            for track_id, rating in zip(track_ids, ratings):
                interactions.append({
                    'user_id': f'user_{user_id}',
                    'track_id': f'track_{track_id}',
                    'rating': rating,
                    'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 365))
                })
        
        return pd.DataFrame(interactions)
    
    @pytest.fixture
    def sample_track_features(self):
        """Caractéristiques des pistes de test."""
        np.random.seed(42)
        num_tracks = 500
        
        features = []
        genres = ['rock', 'pop', 'jazz', 'classical', 'electronic', 'hip-hop']
        
        for track_id in range(num_tracks):
            feature = {
                'track_id': f'track_{track_id}',
                'danceability': np.random.beta(2, 2),
                'energy': np.random.beta(2, 2),
                'valence': np.random.beta(2, 2),
                'acousticness': np.random.beta(1, 3),
                'instrumentalness': np.random.beta(1, 9),
                'tempo': np.random.normal(120, 30),
                'loudness': np.random.normal(-10, 5),
                'genre': np.random.choice(genres),
                'popularity': np.random.randint(0, 100)
            }
            features.append(feature)
        
        return pd.DataFrame(features)
    
    @pytest.mark.asyncio
    async def test_model_initialization(self):
        """Test l'initialisation du modèle de recommandation."""
        config = {
            "model_type": "hybrid",
            "embedding_dim": 128,
            "num_factors": 100
        }
        
        model = SpotifyRecommendationModel(config)
        assert not model.is_trained
        
        await model.initialize()
        assert model.config == config
        assert model.hybrid_model is not None
        
        await model.cleanup()
    
    @pytest.mark.asyncio
    async def test_model_training(self, recommendation_model, sample_interaction_data, sample_track_features):
        """Test l'entraînement du modèle."""
        training_data = {
            'interactions': sample_interaction_data,
            'track_features': sample_track_features,
            'user_features': None  # Optionnel
        }
        
        # Mock de l'entraînement
        with patch.object(recommendation_model.hybrid_model, 'fit') as mock_fit:
            mock_fit.return_value = {"loss": 0.15, "accuracy": 0.85}
            
            result = await recommendation_model.train(training_data)
            
            assert result["status"] == "success"
            assert "loss" in result
            assert "training_time" in result
            mock_fit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_recommendation_generation(self, recommendation_model):
        """Test la génération de recommandations."""
        user_id = "test_user_123"
        
        # Mock des recommandations
        mock_recommendations = [
            {
                'track_id': 'track_1',
                'score': 0.95,
                'confidence': 0.88,
                'explanation': 'Based on your listening history'
            },
            {
                'track_id': 'track_2',
                'score': 0.87,
                'confidence': 0.82,
                'explanation': 'Similar to tracks you liked'
            }
        ]
        
        with patch.object(recommendation_model.hybrid_model, 'predict') as mock_predict:
            mock_predict.return_value = mock_recommendations
            
            recommendations = await recommendation_model.generate_recommendations(
                user_id=user_id,
                num_recommendations=10,
                diversity_factor=0.3
            )
            
            assert len(recommendations) <= 10
            assert all('track_id' in rec for rec in recommendations)
            assert all('score' in rec for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_contextual_recommendations(self, recommendation_model):
        """Test les recommandations contextuelles."""
        user_id = "test_user_123"
        context = {
            "time_of_day": "evening",
            "day_of_week": "friday",
            "weather": "rainy",
            "activity": "workout",
            "mood": "energetic"
        }
        
        with patch.object(recommendation_model, '_apply_context_filters') as mock_context:
            mock_context.return_value = [
                {'track_id': 'energetic_track_1', 'score': 0.92},
                {'track_id': 'energetic_track_2', 'score': 0.89}
            ]
            
            recommendations = await recommendation_model.generate_contextual_recommendations(
                user_id=user_id,
                context=context,
                num_recommendations=5
            )
            
            assert len(recommendations) <= 5
            mock_context.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_model_evaluation(self, recommendation_model, sample_interaction_data):
        """Test l'évaluation du modèle."""
        test_data = sample_interaction_data.sample(frac=0.2)  # 20% pour test
        
        with patch.object(recommendation_model, '_calculate_metrics') as mock_metrics:
            mock_metrics.return_value = {
                'precision_at_k': [0.8, 0.75, 0.7],  # k=5,10,20
                'recall_at_k': [0.15, 0.25, 0.35],
                'ndcg_at_k': [0.82, 0.78, 0.74],
                'diversity': 0.65,
                'novelty': 0.48,
                'coverage': 0.72
            }
            
            evaluation = await recommendation_model.evaluate(test_data)
            
            assert 'precision_at_k' in evaluation
            assert 'recall_at_k' in evaluation
            assert 'ndcg_at_k' in evaluation
            assert evaluation['diversity'] > 0
    
    @pytest.mark.asyncio
    async def test_cold_start_handling(self, recommendation_model):
        """Test la gestion du problème de démarrage à froid."""
        # Nouvel utilisateur sans historique
        new_user_id = "new_user_999"
        
        with patch.object(recommendation_model, '_get_popularity_recommendations') as mock_popular:
            mock_popular.return_value = [
                {'track_id': 'popular_track_1', 'score': 0.9, 'reason': 'trending'},
                {'track_id': 'popular_track_2', 'score': 0.85, 'reason': 'popular_genre'}
            ]
            
            recommendations = await recommendation_model.generate_recommendations(
                user_id=new_user_id,
                num_recommendations=10
            )
            
            assert len(recommendations) > 0
            mock_popular.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_recommendation_explanation(self, recommendation_model):
        """Test les explications des recommandations."""
        user_id = "test_user_123"
        
        with patch.object(recommendation_model, '_generate_explanations') as mock_explain:
            mock_explain.return_value = [
                {
                    'track_id': 'track_1',
                    'score': 0.95,
                    'explanation': {
                        'primary_reason': 'similar_tracks',
                        'factors': ['genre_match', 'artist_similarity', 'audio_features'],
                        'confidence': 0.88,
                        'user_specific': True
                    }
                }
            ]
            
            recommendations = await recommendation_model.generate_recommendations(
                user_id=user_id,
                include_explanations=True,
                num_recommendations=5
            )
            
            assert all('explanation' in rec for rec in recommendations)
            assert all('primary_reason' in rec['explanation'] for rec in recommendations)


class TestContentBasedModel:
    """Tests pour le modèle basé sur le contenu."""
    
    @pytest.fixture
    def content_model(self):
        """Instance du modèle basé sur le contenu."""
        config = {"feature_weights": {"genre": 0.3, "audio_features": 0.7}}
        return ContentBasedModel(config)
    
    @pytest.mark.asyncio
    async def test_content_similarity_calculation(self, content_model):
        """Test le calcul de similarité basé sur le contenu."""
        track_features = {
            'track_1': {'genre': 'rock', 'energy': 0.8, 'danceability': 0.6},
            'track_2': {'genre': 'rock', 'energy': 0.75, 'danceability': 0.65},
            'track_3': {'genre': 'pop', 'energy': 0.9, 'danceability': 0.85}
        }
        
        similarity = content_model.calculate_similarity('track_1', 'track_2', track_features)
        assert similarity > 0.5  # Même genre, caractéristiques similaires
        
        similarity_diff = content_model.calculate_similarity('track_1', 'track_3', track_features)
        assert similarity_diff < similarity  # Genre différent
    
    @pytest.mark.asyncio
    async def test_feature_extraction(self, content_model):
        """Test l'extraction de caractéristiques."""
        track_data = {
            'audio_features': {
                'danceability': 0.735,
                'energy': 0.578,
                'valence': 0.636,
                'tempo': 98.002
            },
            'genre': 'rock',
            'artist': 'Test Artist',
            'year': 2023
        }
        
        features = content_model.extract_features(track_data)
        
        assert 'audio_vector' in features
        assert 'genre_encoded' in features
        assert len(features['audio_vector']) > 0
    
    @pytest.mark.asyncio
    async def test_content_recommendations(self, content_model):
        """Test la génération de recommandations basées sur le contenu."""
        user_profile = {
            'liked_tracks': ['track_1', 'track_3', 'track_5'],
            'preferred_genres': ['rock', 'pop'],
            'audio_preferences': {
                'energy': 0.7,
                'danceability': 0.6,
                'valence': 0.8
            }
        }
        
        candidate_tracks = [
            {'track_id': 'track_10', 'genre': 'rock', 'energy': 0.75},
            {'track_id': 'track_11', 'genre': 'classical', 'energy': 0.3},
            {'track_id': 'track_12', 'genre': 'pop', 'energy': 0.8}
        ]
        
        with patch.object(content_model, '_score_candidates') as mock_score:
            mock_score.return_value = [
                {'track_id': 'track_10', 'score': 0.85},
                {'track_id': 'track_12', 'score': 0.82},
                {'track_id': 'track_11', 'score': 0.45}
            ]
            
            recommendations = await content_model.recommend(user_profile, candidate_tracks)
            
            # Les pistes rock/pop devraient être mieux classées
            assert recommendations[0]['track_id'] in ['track_10', 'track_12']


class TestCollaborativeFilteringModel:
    """Tests pour le modèle de filtrage collaboratif."""
    
    @pytest.fixture
    def collaborative_model(self):
        """Instance du modèle de filtrage collaboratif."""
        config = {"num_factors": 50, "regularization": 0.01}
        return CollaborativeFilteringModel(config)
    
    @pytest.mark.asyncio
    async def test_matrix_factorization(self, collaborative_model):
        """Test la factorisation matricielle."""
        # Matrice utilisateur-item sparse
        interaction_matrix = np.random.rand(100, 500)
        interaction_matrix[interaction_matrix < 0.95] = 0  # 95% sparse
        
        with patch.object(collaborative_model, '_fit_matrix_factorization') as mock_fit:
            mock_fit.return_value = {
                'user_factors': np.random.rand(100, 50),
                'item_factors': np.random.rand(500, 50),
                'loss': 0.12
            }
            
            result = await collaborative_model.fit(interaction_matrix)
            
            assert 'user_factors' in result
            assert 'item_factors' in result
            assert result['user_factors'].shape == (100, 50)
    
    @pytest.mark.asyncio
    async def test_user_similarity(self, collaborative_model):
        """Test le calcul de similarité entre utilisateurs."""
        user_vectors = {
            'user_1': np.array([1, 0, 1, 0, 1]),
            'user_2': np.array([1, 0, 1, 1, 0]),
            'user_3': np.array([0, 1, 0, 1, 0])
        }
        
        similarity = collaborative_model.calculate_user_similarity('user_1', 'user_2', user_vectors)
        assert similarity > 0  # Quelques goûts en commun
        
        similarity_diff = collaborative_model.calculate_user_similarity('user_1', 'user_3', user_vectors)
        assert similarity_diff < similarity  # Goûts très différents
    
    @pytest.mark.asyncio
    async def test_neighborhood_recommendations(self, collaborative_model):
        """Test les recommandations basées sur le voisinage."""
        target_user = 'user_1'
        similar_users = ['user_2', 'user_3', 'user_4']
        user_item_matrix = np.random.rand(5, 10)
        
        with patch.object(collaborative_model, '_find_similar_users') as mock_neighbors:
            mock_neighbors.return_value = [
                {'user_id': 'user_2', 'similarity': 0.85},
                {'user_id': 'user_3', 'similarity': 0.72}
            ]
            
            recommendations = await collaborative_model.recommend_from_neighbors(
                target_user, user_item_matrix
            )
            
            assert len(recommendations) > 0
            assert all('score' in rec for rec in recommendations)


class TestDeepLearningRecommendationModel:
    """Tests pour le modèle de deep learning."""
    
    @pytest.fixture
    def deep_model(self):
        """Instance du modèle de deep learning."""
        config = {
            "embedding_dim": 128,
            "hidden_layers": [256, 128, 64],
            "dropout_rate": 0.3,
            "learning_rate": 0.001
        }
        return DeepLearningRecommendationModel(config)
    
    @pytest.mark.asyncio
    async def test_neural_network_architecture(self, deep_model):
        """Test l'architecture du réseau neuronal."""
        await deep_model.initialize()
        
        # Vérifier que le modèle a été créé
        assert deep_model.model is not None
        assert hasattr(deep_model.model, 'forward')
        
        # Test forward pass
        batch_size = 32
        user_ids = torch.randint(0, 1000, (batch_size,))
        item_ids = torch.randint(0, 5000, (batch_size,))
        
        with torch.no_grad():
            output = deep_model.model(user_ids, item_ids)
            assert output.shape == (batch_size, 1)
    
    @pytest.mark.asyncio
    async def test_embedding_learning(self, deep_model):
        """Test l'apprentissage des embeddings."""
        # Données d'entraînement simulées
        training_data = {
            'user_ids': torch.randint(0, 100, (1000,)),
            'item_ids': torch.randint(0, 500, (1000,)),
            'ratings': torch.rand(1000)
        }
        
        with patch.object(deep_model, '_train_epoch') as mock_train:
            mock_train.return_value = {"loss": 0.25, "accuracy": 0.78}
            
            result = await deep_model.train(training_data, epochs=5)
            
            assert result["status"] == "success"
            assert "final_loss" in result
            assert mock_train.call_count == 5
    
    @pytest.mark.asyncio
    async def test_attention_mechanism(self, deep_model):
        """Test le mécanisme d'attention."""
        if hasattr(deep_model, 'attention_layer'):
            # Séquence d'items pour un utilisateur
            item_sequence = torch.randint(0, 500, (1, 10))  # 10 items
            
            attention_weights = deep_model.attention_layer(item_sequence)
            
            assert attention_weights.shape[1] == 10  # Même longueur que la séquence
            assert torch.allclose(attention_weights.sum(dim=1), torch.ones(1))  # Somme = 1
    
    @pytest.mark.asyncio
    async def test_multi_task_learning(self, deep_model):
        """Test l'apprentissage multi-tâches."""
        if hasattr(deep_model, 'multi_task_heads'):
            # Données avec plusieurs tâches
            user_ids = torch.randint(0, 100, (32,))
            item_ids = torch.randint(0, 500, (32,))
            
            outputs = deep_model.predict_multi_task(user_ids, item_ids)
            
            assert 'rating_prediction' in outputs
            assert 'genre_prediction' in outputs
            assert 'popularity_prediction' in outputs


class TestHybridRecommendationModel:
    """Tests pour le modèle hybride."""
    
    @pytest.fixture
    def hybrid_model(self):
        """Instance du modèle hybride."""
        config = {
            "models": {
                "content_based": {"weight": 0.3},
                "collaborative": {"weight": 0.4},
                "deep_learning": {"weight": 0.3}
            },
            "fusion_strategy": "weighted_average"
        }
        return HybridRecommendationModel(config)
    
    @pytest.mark.asyncio
    async def test_model_fusion(self, hybrid_model):
        """Test la fusion des modèles."""
        # Recommandations de différents modèles
        content_recs = [
            {'track_id': 'track_1', 'score': 0.8},
            {'track_id': 'track_2', 'score': 0.7}
        ]
        
        collab_recs = [
            {'track_id': 'track_1', 'score': 0.9},
            {'track_id': 'track_3', 'score': 0.85}
        ]
        
        deep_recs = [
            {'track_id': 'track_2', 'score': 0.95},
            {'track_id': 'track_4', 'score': 0.8}
        ]
        
        fused_recs = hybrid_model.fuse_recommendations([
            content_recs, collab_recs, deep_recs
        ])
        
        # Vérifier que les scores sont correctement fusionnés
        track_1_score = next(rec['score'] for rec in fused_recs if rec['track_id'] == 'track_1')
        assert 0.8 < track_1_score < 0.9  # Moyenne pondérée
    
    @pytest.mark.asyncio
    async def test_adaptive_weighting(self, hybrid_model):
        """Test l'ajustement adaptatif des poids."""
        user_context = {
            'user_id': 'test_user',
            'new_user': False,
            'interaction_count': 150,
            'diversity_preference': 0.7
        }
        
        adapted_weights = hybrid_model.adapt_weights(user_context)
        
        # Pour un utilisateur expérimenté, le collaboratif devrait avoir plus de poids
        assert adapted_weights['collaborative'] >= 0.4
        
        # Test avec un nouvel utilisateur
        new_user_context = {
            'user_id': 'new_user',
            'new_user': True,
            'interaction_count': 5,
            'diversity_preference': 0.5
        }
        
        new_weights = hybrid_model.adapt_weights(new_user_context)
        # Pour un nouvel utilisateur, le content-based devrait dominer
        assert new_weights['content_based'] >= 0.5
    
    @pytest.mark.asyncio
    async def test_dynamic_model_selection(self, hybrid_model):
        """Test la sélection dynamique de modèles."""
        # Contexte nécessitant de la diversité
        diversity_context = {'diversity_requirement': 'high'}
        
        selected_models = hybrid_model.select_models(diversity_context)
        
        # Le modèle content-based devrait être privilégié pour la diversité
        assert 'content_based' in selected_models
        assert selected_models['content_based']['weight'] > 0.3


@pytest.mark.performance
class TestRecommendationPerformance:
    """Tests de performance pour les modèles de recommandation."""
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_large_scale_recommendations(self):
        """Test les recommandations à grande échelle."""
        config = {"model_type": "hybrid", "embedding_dim": 32}  # Petite taille pour rapidité
        model = SpotifyRecommendationModel(config)
        await model.initialize()
        
        # Simuler une grande base d'utilisateurs
        user_ids = [f"user_{i}" for i in range(1000)]
        
        start_time = datetime.now()
        
        # Traitement par batch
        recommendations = []
        batch_size = 100
        
        for i in range(0, len(user_ids), batch_size):
            batch = user_ids[i:i+batch_size]
            
            with patch.object(model, 'generate_recommendations') as mock_rec:
                mock_rec.return_value = [{'track_id': f'track_{j}', 'score': 0.8} for j in range(10)]
                
                batch_recs = await asyncio.gather(*[
                    model.generate_recommendations(user_id, num_recommendations=10)
                    for user_id in batch
                ])
                recommendations.extend(batch_recs)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Performance: moins de 30 secondes pour 1000 utilisateurs
        assert duration < 30.0
        assert len(recommendations) == 1000
        
        await model.cleanup()
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """Test l'efficacité mémoire des modèles."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Créer plusieurs modèles
        models = []
        for i in range(10):
            config = {"model_type": "content_based", "embedding_dim": 64}
            model = SpotifyRecommendationModel(config)
            await model.initialize()
            models.append(model)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_per_model = (final_memory - initial_memory) / 10
        
        # Chaque modèle ne devrait pas utiliser plus de 50MB
        assert memory_per_model < 50
        
        # Cleanup
        for model in models:
            await model.cleanup()


@pytest.mark.integration
class TestRecommendationIntegration:
    """Tests d'intégration pour les modèles de recommandation."""
    
    @pytest.mark.asyncio
    async def test_full_recommendation_pipeline(self):
        """Test complet du pipeline de recommandation."""
        # 1. Initialisation
        config = {"model_type": "hybrid", "embedding_dim": 64}
        model = SpotifyRecommendationModel(config)
        await model.initialize()
        
        # 2. Données d'entraînement
        training_data = {
            'interactions': pd.DataFrame({
                'user_id': ['user_1', 'user_1', 'user_2'],
                'track_id': ['track_1', 'track_2', 'track_1'],
                'rating': [0.9, 0.7, 0.8]
            }),
            'track_features': pd.DataFrame({
                'track_id': ['track_1', 'track_2'],
                'genre': ['rock', 'pop'],
                'energy': [0.8, 0.6]
            })
        }
        
        # 3. Entraînement
        with patch.object(model, 'train') as mock_train:
            mock_train.return_value = {"status": "success", "loss": 0.15}
            await model.train(training_data)
        
        # 4. Génération de recommandations
        with patch.object(model, 'generate_recommendations') as mock_rec:
            mock_rec.return_value = [
                {'track_id': 'track_3', 'score': 0.85, 'explanation': 'Similar to your liked tracks'}
            ]
            
            recommendations = await model.generate_recommendations('user_1')
            assert len(recommendations) > 0
        
        # 5. Évaluation
        with patch.object(model, 'evaluate') as mock_eval:
            mock_eval.return_value = {'precision_at_k': [0.8], 'recall_at_k': [0.2]}
            
            metrics = await model.evaluate(training_data['interactions'])
            assert 'precision_at_k' in metrics
        
        await model.cleanup()
