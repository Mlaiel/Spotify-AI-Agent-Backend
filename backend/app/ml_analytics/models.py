# 🎵 Advanced Music Recommendation Models
# =======================================
# 
# Modèles IA avancés pour recommandations musicales Spotify
# Architecture hybride avec deep learning et collaborative filtering
#
# 🎖️ Expert: Ingénieur Machine Learning

"""
🎵 Advanced Music Recommendation Models
=======================================

Enterprise-grade music recommendation system providing:
- Hybrid recommendation algorithms
- Deep learning content analysis
- Collaborative filtering
- Audio feature extraction
- Real-time personalization
- Cold start problem handling
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import librosa
import librosa.display
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import pickle
import json
import asyncio
from datetime import datetime, timedelta
import redis
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .core import IMLModel, ModelType
from .config import ModelConfig
from .utils import ensure_async, validate_input_data, normalize_features
from .exceptions import ModelError, InferenceError, TrainingError


@dataclass
class TrackFeatures:
    """Caractéristiques d'une piste musicale"""
    track_id: str
    audio_features: np.ndarray  # MFCC, spectral features, etc.
    metadata_features: np.ndarray  # genre, tempo, danceability, etc.
    text_features: np.ndarray  # lyrics embedding
    popularity_score: float
    release_date: datetime
    artist_features: np.ndarray
    album_features: np.ndarray


@dataclass
class UserProfile:
    """Profil utilisateur pour personnalisation"""
    user_id: str
    listening_history: List[str]  # track_ids
    explicit_preferences: Dict[str, float]  # genre -> score
    implicit_preferences: np.ndarray  # learned features
    demographics: Dict[str, Any]
    interaction_patterns: Dict[str, float]
    last_updated: datetime


@dataclass
class RecommendationContext:
    """Contexte pour les recommandations"""
    time_of_day: str
    day_of_week: str
    season: str
    mood: Optional[str] = None
    activity: Optional[str] = None
    location: Optional[str] = None
    weather: Optional[str] = None
    social_context: Optional[str] = None


class SpotifyRecommendationModel(IMLModel):
    """Modèle hybride de recommandation musicale Spotify"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_id = config.model_id
        self.logger = logging.getLogger(f"{__name__}.SpotifyRecommendationModel")
        
        # Composants du modèle
        self.content_model: Optional[ContentBasedModel] = None
        self.collaborative_model: Optional[CollaborativeFilteringModel] = None
        self.deep_model: Optional[DeepRecommendationModel] = None
        self.popularity_model: Optional[PopularityModel] = None
        
        # Cache et données
        self.track_features: Dict[str, TrackFeatures] = {}
        self.user_profiles: Dict[str, UserProfile] = {}
        self.item_embeddings: Optional[np.ndarray] = None
        self.user_embeddings: Optional[np.ndarray] = None
        
        # État du modèle
        self.is_trained = False
        self.training_metrics = {}
        self.last_training = None
        
        # Redis pour cache
        self.redis_client = None
        
    async def initialize(self) -> bool:
        """Initialisation du modèle"""
        try:
            self.logger.info("Initialisation du modèle de recommandation Spotify...")
            
            # Initialisation des sous-modèles
            self.content_model = ContentBasedModel()
            self.collaborative_model = CollaborativeFilteringModel()
            self.deep_model = DeepRecommendationModel()
            self.popularity_model = PopularityModel()
            
            # Connexion Redis pour cache
            if hasattr(self.config, 'redis_config'):
                import redis
                self.redis_client = redis.Redis(
                    host=self.config.redis_config.host,
                    port=self.config.redis_config.port,
                    db=self.config.redis_config.database,
                    decode_responses=True
                )
            
            # Chargement des modèles pré-entraînés si disponibles
            await self._load_pretrained_models()
            
            self.logger.info("Modèle de recommandation initialisé avec succès")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation: {e}")
            return False
    
    async def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Génération de recommandations personnalisées"""
        try:
            user_id = input_data.get('user_id')
            num_recommendations = input_data.get('num_recommendations', 20)
            context = input_data.get('context', {})
            exclude_tracks = input_data.get('exclude_tracks', [])
            
            if not user_id:
                raise ValueError("user_id requis pour les recommandations")
            
            # Récupération du profil utilisateur
            user_profile = await self._get_user_profile(user_id)
            recommendation_context = RecommendationContext(**context)
            
            # Génération des recommandations hybrides
            recommendations = await self._generate_hybrid_recommendations(
                user_profile, recommendation_context, num_recommendations, exclude_tracks
            )
            
            # Post-processing et diversification
            final_recommendations = await self._diversify_recommendations(
                recommendations, user_profile, num_recommendations
            )
            
            # Métriques et logging
            await self._log_recommendation_event(user_id, final_recommendations, context)
            
            return {
                'user_id': user_id,
                'recommendations': final_recommendations,
                'context': context,
                'generated_at': datetime.utcnow().isoformat(),
                'model_version': self.config.version,
                'confidence_scores': [rec['confidence'] for rec in final_recommendations]
            }
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la prédiction: {e}")
            raise InferenceError(f"Erreur de recommandation: {e}")
    
    async def train(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Entraînement du modèle hybride"""
        try:
            self.logger.info("Début de l'entraînement du modèle de recommandation...")
            
            # Préparation des données
            user_item_matrix = training_data.get('user_item_matrix')
            track_features = training_data.get('track_features')
            user_profiles = training_data.get('user_profiles')
            interactions = training_data.get('interactions')
            
            if not all([user_item_matrix is not None, track_features is not None]):
                raise ValueError("Données d'entraînement incomplètes")
            
            # Entraînement des sous-modèles
            training_results = {}
            
            # 1. Modèle basé sur le contenu
            if self.content_model:
                content_metrics = await self.content_model.train(track_features)
                training_results['content_model'] = content_metrics
            
            # 2. Filtrage collaboratif
            if self.collaborative_model:
                collab_metrics = await self.collaborative_model.train(user_item_matrix)
                training_results['collaborative_model'] = collab_metrics
            
            # 3. Modèle deep learning
            if self.deep_model:
                deep_metrics = await self.deep_model.train({
                    'user_item_matrix': user_item_matrix,
                    'track_features': track_features,
                    'interactions': interactions
                })
                training_results['deep_model'] = deep_metrics
            
            # 4. Modèle de popularité
            if self.popularity_model:
                pop_metrics = await self.popularity_model.train(interactions)
                training_results['popularity_model'] = pop_metrics
            
            # Calcul des poids pour l'ensemble
            ensemble_weights = await self._calculate_ensemble_weights(training_results)
            
            # Sauvegarde du modèle
            await self._save_model_state()
            
            self.is_trained = True
            self.last_training = datetime.utcnow()
            self.training_metrics = training_results
            
            self.logger.info("Entraînement terminé avec succès")
            
            return {
                'training_completed': True,
                'training_time': datetime.utcnow().isoformat(),
                'model_metrics': training_results,
                'ensemble_weights': ensemble_weights,
                'model_version': self.config.version
            }
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'entraînement: {e}")
            raise TrainingError(f"Erreur d'entraînement: {e}")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Métadonnées du modèle"""
        return {
            'model_id': self.model_id,
            'model_type': ModelType.RECOMMENDATION.value,
            'version': self.config.version,
            'is_trained': self.is_trained,
            'last_training': self.last_training.isoformat() if self.last_training else None,
            'training_metrics': self.training_metrics,
            'num_tracks': len(self.track_features),
            'num_users': len(self.user_profiles),
            'components': {
                'content_based': self.content_model is not None,
                'collaborative_filtering': self.collaborative_model is not None,
                'deep_learning': self.deep_model is not None,
                'popularity_based': self.popularity_model is not None
            }
        }
    
    def is_ready(self) -> bool:
        """Vérification de l'état de préparation"""
        return (
            self.content_model is not None and
            self.collaborative_model is not None and
            self.deep_model is not None and
            self.popularity_model is not None
        )
    
    async def _get_user_profile(self, user_id: str) -> UserProfile:
        """Récupération ou création du profil utilisateur"""
        if user_id in self.user_profiles:
            return self.user_profiles[user_id]
        
        # Tentative de récupération depuis le cache Redis
        if self.redis_client:
            cached_profile = await self._get_cached_user_profile(user_id)
            if cached_profile:
                self.user_profiles[user_id] = cached_profile
                return cached_profile
        
        # Création d'un nouveau profil
        new_profile = UserProfile(
            user_id=user_id,
            listening_history=[],
            explicit_preferences={},
            implicit_preferences=np.zeros(128),  # dimension des features
            demographics={},
            interaction_patterns={},
            last_updated=datetime.utcnow()
        )
        
        self.user_profiles[user_id] = new_profile
        return new_profile
    
    async def _generate_hybrid_recommendations(
        self,
        user_profile: UserProfile,
        context: RecommendationContext,
        num_recommendations: int,
        exclude_tracks: List[str]
    ) -> List[Dict[str, Any]]:
        """Génération de recommandations hybrides"""
        
        recommendations = []
        
        # 1. Recommandations basées sur le contenu
        if self.content_model:
            content_recs = await self.content_model.recommend(
                user_profile, num_recommendations
            )
            recommendations.extend([
                {**rec, 'source': 'content', 'weight': 0.3}
                for rec in content_recs
            ])
        
        # 2. Filtrage collaboratif
        if self.collaborative_model:
            collab_recs = await self.collaborative_model.recommend(
                user_profile, num_recommendations
            )
            recommendations.extend([
                {**rec, 'source': 'collaborative', 'weight': 0.4}
                for rec in collab_recs
            ])
        
        # 3. Deep learning
        if self.deep_model:
            deep_recs = await self.deep_model.recommend(
                user_profile, context, num_recommendations
            )
            recommendations.extend([
                {**rec, 'source': 'deep', 'weight': 0.25}
                for rec in deep_recs
            ])
        
        # 4. Popularité
        if self.popularity_model:
            pop_recs = await self.popularity_model.recommend(
                context, num_recommendations // 4
            )
            recommendations.extend([
                {**rec, 'source': 'popularity', 'weight': 0.05}
                for rec in pop_recs
            ])
        
        # Agrégation et scoring
        aggregated_recs = await self._aggregate_recommendations(
            recommendations, exclude_tracks
        )
        
        return aggregated_recs[:num_recommendations]
    
    async def _aggregate_recommendations(
        self,
        recommendations: List[Dict[str, Any]],
        exclude_tracks: List[str]
    ) -> List[Dict[str, Any]]:
        """Agrégation des recommandations de différentes sources"""
        
        # Groupement par track_id
        track_scores = {}
        for rec in recommendations:
            track_id = rec['track_id']
            
            if track_id in exclude_tracks:
                continue
            
            if track_id not in track_scores:
                track_scores[track_id] = {
                    'track_id': track_id,
                    'total_score': 0.0,
                    'sources': [],
                    'metadata': rec.get('metadata', {})
                }
            
            # Calcul du score pondéré
            weighted_score = rec['score'] * rec['weight']
            track_scores[track_id]['total_score'] += weighted_score
            track_scores[track_id]['sources'].append(rec['source'])
        
        # Tri par score
        sorted_recommendations = sorted(
            track_scores.values(),
            key=lambda x: x['total_score'],
            reverse=True
        )
        
        # Ajout de métriques de confiance
        for i, rec in enumerate(sorted_recommendations):
            rec['rank'] = i + 1
            rec['confidence'] = min(rec['total_score'], 1.0)
            rec['source_diversity'] = len(set(rec['sources']))
        
        return sorted_recommendations
    
    async def _diversify_recommendations(
        self,
        recommendations: List[Dict[str, Any]],
        user_profile: UserProfile,
        num_recommendations: int
    ) -> List[Dict[str, Any]]:
        """Diversification des recommandations"""
        
        if len(recommendations) <= num_recommendations:
            return recommendations
        
        diversified = []
        seen_artists = set()
        seen_genres = set()
        
        # Première passe : sélection avec diversité
        for rec in recommendations:
            if len(diversified) >= num_recommendations:
                break
            
            metadata = rec.get('metadata', {})
            artist = metadata.get('artist_id')
            genre = metadata.get('genre')
            
            # Critères de diversité
            artist_diversity = artist not in seen_artists
            genre_diversity = genre not in seen_genres or len(seen_genres) < 3
            
            if artist_diversity and genre_diversity:
                diversified.append(rec)
                if artist:
                    seen_artists.add(artist)
                if genre:
                    seen_genres.add(genre)
        
        # Compléter si nécessaire
        remaining_slots = num_recommendations - len(diversified)
        if remaining_slots > 0:
            for rec in recommendations:
                if len(diversified) >= num_recommendations:
                    break
                if rec not in diversified:
                    diversified.append(rec)
        
        return diversified[:num_recommendations]
    
    async def _calculate_ensemble_weights(
        self, training_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calcul des poids pour l'ensemble de modèles"""
        
        # Analyse des performances de chaque modèle
        weights = {
            'content': 0.3,
            'collaborative': 0.4,
            'deep': 0.25,
            'popularity': 0.05
        }
        
        # Ajustement basé sur les métriques d'entraînement
        for model_name, metrics in training_results.items():
            if 'accuracy' in metrics:
                accuracy = metrics['accuracy']
                model_key = model_name.replace('_model', '')
                if model_key in weights:
                    weights[model_key] *= (1 + accuracy * 0.2)
        
        # Normalisation
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    async def _load_pretrained_models(self):
        """Chargement des modèles pré-entraînés"""
        # Implémentation du chargement des modèles sauvegardés
        pass
    
    async def _save_model_state(self):
        """Sauvegarde de l'état du modèle"""
        # Implémentation de la sauvegarde
        pass
    
    async def _get_cached_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Récupération du profil utilisateur depuis le cache"""
        # Implémentation du cache Redis
        return None
    
    async def _log_recommendation_event(
        self, user_id: str, recommendations: List[Dict], context: Dict
    ):
        """Logging des événements de recommandation"""
        self.logger.info(
            f"Recommandations générées pour l'utilisateur {user_id}",
            extra={
                'user_id': user_id,
                'num_recommendations': len(recommendations),
                'context': context,
                'top_recommendation': recommendations[0]['track_id'] if recommendations else None
            }
        )


class ContentBasedModel:
    """Modèle de recommandation basé sur le contenu"""
    
    def __init__(self):
        self.track_features = {}
        self.feature_vectors = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        self.scaler = StandardScaler()
        
    async def train(self, track_features: Dict[str, TrackFeatures]) -> Dict[str, Any]:
        """Entraînement du modèle basé sur le contenu"""
        
        self.track_features = track_features
        
        # Extraction des caractéristiques
        feature_matrix = []
        track_ids = []
        
        for track_id, features in track_features.items():
            # Combinaison des features audio et metadata
            combined_features = np.concatenate([
                features.audio_features,
                features.metadata_features,
                features.text_features,
                features.artist_features,
                features.album_features
            ])
            
            feature_matrix.append(combined_features)
            track_ids.append(track_id)
        
        # Normalisation
        self.feature_vectors = self.scaler.fit_transform(feature_matrix)
        
        return {
            'model_type': 'content_based',
            'num_tracks': len(track_ids),
            'feature_dimensions': self.feature_vectors.shape[1],
            'training_completed': True
        }
    
    async def recommend(
        self, user_profile: UserProfile, num_recommendations: int
    ) -> List[Dict[str, Any]]:
        """Génération de recommandations basées sur le contenu"""
        
        if self.feature_vectors is None:
            return []
        
        # Calcul du profil utilisateur basé sur l'historique
        user_vector = await self._calculate_user_content_profile(user_profile)
        
        # Calcul de similarité
        similarities = cosine_similarity([user_vector], self.feature_vectors)[0]
        
        # Tri et sélection
        track_similarities = [
            {
                'track_id': track_id,
                'score': sim,
                'metadata': self.track_features.get(track_id, {})
            }
            for track_id, sim in zip(self.track_features.keys(), similarities)
            if track_id not in user_profile.listening_history
        ]
        
        # Tri par similarité
        track_similarities.sort(key=lambda x: x['score'], reverse=True)
        
        return track_similarities[:num_recommendations]
    
    async def _calculate_user_content_profile(
        self, user_profile: UserProfile
    ) -> np.ndarray:
        """Calcul du profil utilisateur basé sur le contenu"""
        
        if not user_profile.listening_history:
            # Profil par défaut
            return np.zeros(self.feature_vectors.shape[1])
        
        # Moyenne pondérée des tracks écoutées
        user_features = []
        for track_id in user_profile.listening_history[-50:]:  # 50 dernières
            if track_id in self.track_features:
                track_idx = list(self.track_features.keys()).index(track_id)
                user_features.append(self.feature_vectors[track_idx])
        
        if user_features:
            return np.mean(user_features, axis=0)
        else:
            return np.zeros(self.feature_vectors.shape[1])


class CollaborativeFilteringModel:
    """Modèle de filtrage collaboratif avec NMF"""
    
    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self.nmf_model = NMF(n_components=n_components, random_state=42)
        self.user_factors = None
        self.item_factors = None
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        
    async def train(self, user_item_matrix: np.ndarray) -> Dict[str, Any]:
        """Entraînement du modèle collaboratif"""
        
        # Factorisation matricielle
        self.user_factors = self.nmf_model.fit_transform(user_item_matrix)
        self.item_factors = self.nmf_model.components_.T
        
        # Calcul des métriques
        reconstruction_error = self.nmf_model.reconstruction_err_
        
        return {
            'model_type': 'collaborative_filtering',
            'n_components': self.n_components,
            'reconstruction_error': reconstruction_error,
            'user_factors_shape': self.user_factors.shape,
            'item_factors_shape': self.item_factors.shape
        }
    
    async def recommend(
        self, user_profile: UserProfile, num_recommendations: int
    ) -> List[Dict[str, Any]]:
        """Génération de recommandations collaboratives"""
        
        if self.user_factors is None or self.item_factors is None:
            return []
        
        # Recherche de l'utilisateur ou utilisateurs similaires
        user_idx = self.user_to_idx.get(user_profile.user_id)
        
        if user_idx is None:
            # Utilisateur nouveau - recommandations populaires
            return await self._recommend_for_new_user(num_recommendations)
        
        # Calcul des scores pour tous les items
        user_vector = self.user_factors[user_idx]
        item_scores = np.dot(user_vector, self.item_factors.T)
        
        # Tri et sélection
        item_indices = np.argsort(item_scores)[::-1]
        
        recommendations = []
        for idx in item_indices[:num_recommendations]:
            item_id = self.idx_to_item.get(idx, f"item_{idx}")
            recommendations.append({
                'track_id': item_id,
                'score': item_scores[idx],
                'metadata': {}
            })
        
        return recommendations
    
    async def _recommend_for_new_user(self, num_recommendations: int) -> List[Dict]:
        """Recommandations pour un nouvel utilisateur"""
        # Recommandations basées sur la popularité globale
        popular_items = np.mean(self.item_factors, axis=1)
        top_indices = np.argsort(popular_items)[::-1]
        
        recommendations = []
        for idx in top_indices[:num_recommendations]:
            item_id = self.idx_to_item.get(idx, f"item_{idx}")
            recommendations.append({
                'track_id': item_id,
                'score': popular_items[idx],
                'metadata': {}
            })
        
        return recommendations


class DeepRecommendationModel(nn.Module, IMLModel):
    """Modèle de recommandation basé sur le deep learning"""
    
    def __init__(self, num_users: int = 10000, num_items: int = 50000, embedding_dim: int = 128):
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Réseau neuronal
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # État d'entraînement
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        
        # Concaténation des embeddings
        concat_embeds = torch.cat([user_embeds, item_embeds], dim=1)
        
        # Prédiction
        output = self.fc_layers(concat_embeds)
        return output.squeeze()
    
    async def train(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Entraînement du modèle deep learning"""
        
        interactions = training_data.get('interactions', [])
        epochs = 10
        batch_size = 1024
        learning_rate = 0.001
        
        # Préparation des données
        dataset = RecommendationDataset(interactions)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimiseur et loss
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        # Entraînement
        train_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_user_ids, batch_item_ids, batch_ratings in dataloader:
                batch_user_ids = batch_user_ids.to(self.device)
                batch_item_ids = batch_item_ids.to(self.device)
                batch_ratings = batch_ratings.float().to(self.device)
                
                # Forward pass
                predictions = self.forward(batch_user_ids, batch_item_ids)
                loss = criterion(predictions, batch_ratings)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)
            
            if epoch % 2 == 0:
                logging.info(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        return {
            'model_type': 'deep_learning',
            'epochs': epochs,
            'final_loss': train_losses[-1],
            'training_losses': train_losses,
            'num_parameters': sum(p.numel() for p in self.parameters())
        }
    
    async def predict(self, input_data: Dict[str, Any]) -> Any:
        """Prédiction avec le modèle deep learning"""
        user_id = input_data.get('user_id')
        item_ids = input_data.get('item_ids', [])
        
        if not user_id or not item_ids:
            return []
        
        self.eval()
        with torch.no_grad():
            user_tensor = torch.tensor([user_id] * len(item_ids)).to(self.device)
            item_tensor = torch.tensor(item_ids).to(self.device)
            
            predictions = self.forward(user_tensor, item_tensor)
            scores = predictions.cpu().numpy()
        
        return [
            {'item_id': item_id, 'score': float(score)}
            for item_id, score in zip(item_ids, scores)
        ]
    
    async def recommend(
        self,
        user_profile: UserProfile,
        context: RecommendationContext,
        num_recommendations: int
    ) -> List[Dict[str, Any]]:
        """Génération de recommandations avec deep learning"""
        
        # Conversion user_id en index numérique (à implémenter)
        user_idx = hash(user_profile.user_id) % self.num_users
        
        # Génération de candidats (sample des items)
        candidate_items = np.random.choice(self.num_items, size=1000, replace=False)
        
        # Prédiction des scores
        predictions = await self.predict({
            'user_id': user_idx,
            'item_ids': candidate_items.tolist()
        })
        
        # Tri et sélection
        predictions.sort(key=lambda x: x['score'], reverse=True)
        
        recommendations = []
        for pred in predictions[:num_recommendations]:
            recommendations.append({
                'track_id': f"track_{pred['item_id']}",
                'score': pred['score'],
                'metadata': {}
            })
        
        return recommendations
    
    def get_metadata(self) -> Dict[str, Any]:
        """Métadonnées du modèle"""
        return {
            'model_type': 'deep_recommendation',
            'num_users': self.num_users,
            'num_items': self.num_items,
            'embedding_dim': self.embedding_dim,
            'device': str(self.device),
            'num_parameters': sum(p.numel() for p in self.parameters())
        }
    
    def is_ready(self) -> bool:
        """Vérification de l'état de préparation"""
        return True


class PopularityModel:
    """Modèle de recommandation basé sur la popularité"""
    
    def __init__(self):
        self.item_popularity = {}
        self.trending_items = {}
        self.seasonal_trends = {}
        
    async def train(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Entraînement du modèle de popularité"""
        
        # Calcul de la popularité globale
        item_counts = {}
        recent_interactions = []
        
        for interaction in interactions:
            item_id = interaction.get('item_id')
            timestamp = interaction.get('timestamp')
            
            if item_id:
                item_counts[item_id] = item_counts.get(item_id, 0) + 1
                
                # Interactions récentes (30 derniers jours)
                if timestamp and (datetime.utcnow() - timestamp).days <= 30:
                    recent_interactions.append(interaction)
        
        # Normalisation de la popularité
        max_count = max(item_counts.values()) if item_counts else 1
        self.item_popularity = {
            item_id: count / max_count
            for item_id, count in item_counts.items()
        }
        
        # Calcul des tendances
        await self._calculate_trending_items(recent_interactions)
        
        return {
            'model_type': 'popularity',
            'total_items': len(self.item_popularity),
            'trending_items': len(self.trending_items),
            'avg_popularity': np.mean(list(self.item_popularity.values()))
        }
    
    async def recommend(
        self, context: RecommendationContext, num_recommendations: int
    ) -> List[Dict[str, Any]]:
        """Recommandations basées sur la popularité"""
        
        # Sélection selon le contexte
        if context.time_of_day == "evening" and self.trending_items:
            # Items tendance le soir
            items = self.trending_items
        else:
            # Popularité globale
            items = self.item_popularity
        
        # Tri par popularité
        sorted_items = sorted(
            items.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        recommendations = []
        for item_id, score in sorted_items[:num_recommendations]:
            recommendations.append({
                'track_id': item_id,
                'score': score,
                'metadata': {'source': 'popularity'}
            })
        
        return recommendations
    
    async def _calculate_trending_items(self, recent_interactions: List[Dict]):
        """Calcul des items tendance"""
        
        # Groupement par jour
        daily_counts = {}
        for interaction in recent_interactions:
            item_id = interaction.get('item_id')
            date = interaction.get('timestamp', datetime.utcnow()).date()
            
            if item_id:
                if date not in daily_counts:
                    daily_counts[date] = {}
                daily_counts[date][item_id] = daily_counts[date].get(item_id, 0) + 1
        
        # Calcul de la tendance (croissance récente)
        item_trends = {}
        for item_id in self.item_popularity.keys():
            recent_counts = []
            for date in sorted(daily_counts.keys())[-7:]:  # 7 derniers jours
                count = daily_counts[date].get(item_id, 0)
                recent_counts.append(count)
            
            if len(recent_counts) >= 3:
                # Tendance basée sur la croissance
                trend_score = (recent_counts[-1] - recent_counts[0]) / max(recent_counts[0], 1)
                item_trends[item_id] = trend_score
        
        # Sélection des items tendance
        self.trending_items = {
            item_id: score for item_id, score in item_trends.items()
            if score > 0.1  # Seuil de croissance
        }


class RecommendationDataset(Dataset):
    """Dataset pour l'entraînement du modèle deep learning"""
    
    def __init__(self, interactions: List[Dict[str, Any]]):
        self.user_ids = []
        self.item_ids = []
        self.ratings = []
        
        for interaction in interactions:
            user_id = interaction.get('user_id')
            item_id = interaction.get('item_id')
            rating = interaction.get('rating', 1.0)
            
            if user_id is not None and item_id is not None:
                # Conversion en indices numériques
                self.user_ids.append(hash(str(user_id)) % 10000)
                self.item_ids.append(hash(str(item_id)) % 50000)
                self.ratings.append(rating)
    
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.user_ids[idx], dtype=torch.long),
            torch.tensor(self.item_ids[idx], dtype=torch.long),
            torch.tensor(self.ratings[idx], dtype=torch.float)
        )


# Exports publics
__all__ = [
    'SpotifyRecommendationModel',
    'ContentBasedModel',
    'CollaborativeFilteringModel', 
    'DeepRecommendationModel',
    'PopularityModel',
    'TrackFeatures',
    'UserProfile',
    'RecommendationContext',
    'RecommendationDataset'
]
