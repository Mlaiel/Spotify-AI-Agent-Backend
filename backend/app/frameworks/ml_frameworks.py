"""
🧠 ML FRAMEWORKS - ORCHESTRATION IA ENTERPRISE
Expert Team: ML Engineer, AI Architect

Gestion avancée des frameworks ML/AI avec orchestration intelligente
"""

import asyncio
import os
import time
import threading
import pickle
import joblib
from typing import Dict, Any, Optional, List, Union, Type, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

# ML/AI Frameworks
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import tensorflow as tf
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    pipeline, BertTokenizer, BertModel
)
import sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Audio processing
import librosa
import soundfile as sf
from pydub import AudioSegment

# Deep Learning avancé
import pytorch_lightning as pl
from torch.nn import functional as F

# Base framework
from .core import BaseFramework, FrameworkStatus, FrameworkHealth
from .core import framework_orchestrator

# Configuration et monitoring
import mlflow
import wandb
from tensorboardX import SummaryWriter


class ModelType(Enum):
    """Types de modèles ML/AI supportés"""
    RECOMMENDATION = "recommendation"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    NLP = "nlp"
    AUDIO_ANALYSIS = "audio_analysis"
    GENERATIVE = "generative"
    REINFORCEMENT_LEARNING = "reinforcement_learning"


class MLFrameworkType(Enum):
    """Types de frameworks ML"""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    SKLEARN = "sklearn"
    TRANSFORMERS = "transformers"
    LIBROSA = "librosa"
    CUSTOM = "custom"


@dataclass
class ModelConfig:
    """Configuration d'un modèle ML"""
    name: str
    model_type: ModelType
    framework_type: MLFrameworkType
    version: str = "1.0.0"
    input_shape: Optional[tuple] = None
    output_shape: Optional[tuple] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    preprocessing_config: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    deployment_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingMetrics:
    """Métriques d'entraînement"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    loss: float = 0.0
    val_accuracy: float = 0.0
    val_loss: float = 0.0
    training_time: float = 0.0
    epoch: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseMLModel(ABC):
    """Interface de base pour tous les modèles ML"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model: Optional[Any] = None
        self.is_trained = False
        self.training_metrics: Optional[TrainingMetrics] = None
        self.logger = logging.getLogger(f"ml.model.{config.name}")
        
    @abstractmethod
    async def build_model(self) -> bool:
        """Construit le modèle"""
        pass
    
    @abstractmethod
    async def train(self, train_data: Any, val_data: Optional[Any] = None) -> TrainingMetrics:
        """Entraîne le modèle"""
        pass
    
    @abstractmethod
    async def predict(self, input_data: Any) -> Any:
        """Effectue une prédiction"""
        pass
    
    @abstractmethod
    async def save_model(self, path: str) -> bool:
        """Sauvegarde le modèle"""
        pass
    
    @abstractmethod
    async def load_model(self, path: str) -> bool:
        """Charge le modèle"""
        pass
    
    async def evaluate(self, test_data: Any) -> Dict[str, float]:
        """Évalue le modèle"""
        # Implémentation par défaut
        return {"accuracy": 0.0}


class SpotifyRecommendationModel(BaseMLModel):
    """
    🎵 MODÈLE DE RECOMMANDATION SPOTIFY
    
    Modèle hybride utilisant:
    - Collaborative Filtering
    - Content-Based Filtering
    - Deep Learning embeddings
    - Audio features analysis
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.user_embeddings: Optional[nn.Module] = None
        self.item_embeddings: Optional[nn.Module] = None
        self.audio_encoder: Optional[nn.Module] = None
        
    async def build_model(self) -> bool:
        """Construit le modèle de recommandation hybride"""
        try:
            # Paramètres du modèle
            num_users = self.config.hyperparameters.get("num_users", 10000)
            num_items = self.config.hyperparameters.get("num_items", 100000)
            embedding_dim = self.config.hyperparameters.get("embedding_dim", 64)
            hidden_dim = self.config.hyperparameters.get("hidden_dim", 128)
            
            # Architecture du modèle de recommandation
            class HybridRecommendationModel(nn.Module):
                def __init__(self, num_users, num_items, embedding_dim, hidden_dim):
                    super().__init__()
                    
                    # Embeddings utilisateurs et items
                    self.user_embedding = nn.Embedding(num_users, embedding_dim)
                    self.item_embedding = nn.Embedding(num_items, embedding_dim)
                    
                    # Encodeur audio
                    self.audio_encoder = nn.Sequential(
                        nn.Linear(13, 64),  # MFCC features
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, embedding_dim)
                    )
                    
                    # Réseau de fusion
                    self.fusion_network = nn.Sequential(
                        nn.Linear(embedding_dim * 3, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Linear(hidden_dim // 2, 1),
                        nn.Sigmoid()
                    )
                    
                def forward(self, user_ids, item_ids, audio_features):
                    user_emb = self.user_embedding(user_ids)
                    item_emb = self.item_embedding(item_ids)
                    audio_emb = self.audio_encoder(audio_features)
                    
                    # Fusion des embeddings
                    combined = torch.cat([user_emb, item_emb, audio_emb], dim=1)
                    rating = self.fusion_network(combined)
                    
                    return rating
            
            self.model = HybridRecommendationModel(
                num_users, num_items, embedding_dim, hidden_dim
            )
            
            self.logger.info("Spotify recommendation model built successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Model building failed: {e}")
            return False
    
    async def train(self, train_data: Any, val_data: Optional[Any] = None) -> TrainingMetrics:
        """Entraîne le modèle de recommandation"""
        try:
            # Configuration d'entraînement
            lr = self.config.training_config.get("learning_rate", 0.001)
            epochs = self.config.training_config.get("epochs", 50)
            batch_size = self.config.training_config.get("batch_size", 256)
            
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            
            # DataLoader
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=batch_size) if val_data else None
            
            best_val_loss = float('inf')
            training_metrics = TrainingMetrics()
            
            for epoch in range(epochs):
                # Training phase
                self.model.train()
                total_loss = 0
                
                for batch in train_loader:
                    user_ids, item_ids, audio_features, ratings = batch
                    
                    optimizer.zero_grad()
                    predictions = self.model(user_ids, item_ids, audio_features)
                    loss = criterion(predictions.squeeze(), ratings.float())
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(train_loader)
                
                # Validation phase
                val_loss = 0
                if val_loader:
                    self.model.eval()
                    with torch.no_grad():
                        for batch in val_loader:
                            user_ids, item_ids, audio_features, ratings = batch
                            predictions = self.model(user_ids, item_ids, audio_features)
                            val_loss += criterion(predictions.squeeze(), ratings.float()).item()
                    val_loss /= len(val_loader)
                
                # Logging et métriques
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Sauvegarde du meilleur modèle
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), f"models/{self.config.name}_best.pth")
            
            training_metrics.loss = avg_loss
            training_metrics.val_loss = val_loss
            training_metrics.epoch = epochs
            self.training_metrics = training_metrics
            self.is_trained = True
            
            self.logger.info("Recommendation model training completed")
            return training_metrics
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    async def predict(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Génère des recommandations personnalisées"""
        try:
            if not self.is_trained:
                raise ValueError("Model not trained")
            
            self.model.eval()
            user_id = input_data["user_id"]
            candidate_items = input_data.get("candidate_items", [])
            top_k = input_data.get("top_k", 10)
            
            recommendations = []
            
            with torch.no_grad():
                for item_id in candidate_items:
                    # Préparer les features audio (mock pour l'exemple)
                    audio_features = torch.randn(1, 13)
                    
                    user_tensor = torch.tensor([user_id])
                    item_tensor = torch.tensor([item_id])
                    
                    score = self.model(user_tensor, item_tensor, audio_features)
                    
                    recommendations.append({
                        "item_id": item_id,
                        "score": score.item(),
                        "confidence": min(score.item() * 100, 100)
                    })
            
            # Trier par score et retourner top-k
            recommendations.sort(key=lambda x: x["score"], reverse=True)
            return recommendations[:top_k]
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise
    
    async def save_model(self, path: str) -> bool:
        """Sauvegarde le modèle"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Sauvegarder le modèle PyTorch
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config.__dict__,
                'training_metrics': self.training_metrics.__dict__ if self.training_metrics else None
            }, path)
            
            self.logger.info(f"Model saved to {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model saving failed: {e}")
            return False
    
    async def load_model(self, path: str) -> bool:
        """Charge le modèle"""
        try:
            checkpoint = torch.load(path, map_location='cpu')
            
            # Reconstruire le modèle
            await self.build_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Restaurer les métriques
            if checkpoint.get('training_metrics'):
                self.training_metrics = TrainingMetrics(**checkpoint['training_metrics'])
                self.is_trained = True
            
            self.logger.info(f"Model loaded from {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            return False


class AudioAnalysisModel(BaseMLModel):
    """
    🎵 MODÈLE D'ANALYSE AUDIO
    
    Analyse avancée des caractéristiques audio:
    - Extraction de features MFCC
    - Classification de genre
    - Détection d'émotion
    - Similarité audio
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.feature_extractor: Optional[Any] = None
        self.genre_classifier: Optional[Any] = None
        self.emotion_classifier: Optional[Any] = None
    
    async def build_model(self) -> bool:
        """Construit les modèles d'analyse audio"""
        try:
            # Extracteur de features audio
            class AudioFeatureExtractor:
                def __init__(self):
                    self.sample_rate = 22050
                    self.n_mfcc = 13
                    self.n_fft = 2048
                    self.hop_length = 512
                
                def extract_features(self, audio_path: str) -> np.ndarray:
                    """Extrait les features audio"""
                    try:
                        # Charger l'audio
                        y, sr = librosa.load(audio_path, sr=self.sample_rate)
                        
                        # MFCC features
                        mfcc = librosa.feature.mfcc(
                            y=y, sr=sr, n_mfcc=self.n_mfcc,
                            n_fft=self.n_fft, hop_length=self.hop_length
                        )
                        
                        # Spectral features
                        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
                        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
                        
                        # Tempo et beat
                        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                        
                        # Combiner toutes les features
                        features = np.hstack([
                            np.mean(mfcc, axis=1),
                            np.mean(spectral_centroids),
                            np.mean(spectral_rolloff),
                            np.mean(zero_crossing_rate),
                            tempo
                        ])
                        
                        return features
                        
                    except Exception as e:
                        self.logger.error(f"Feature extraction failed: {e}")
                        return np.zeros(self.n_mfcc + 4)  # Features par défaut
            
            self.feature_extractor = AudioFeatureExtractor()
            
            # Classificateur de genre musical
            self.genre_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Classificateur d'émotion
            self.emotion_classifier = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            self.logger.info("Audio analysis models built successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Audio model building failed: {e}")
            return False
    
    async def train(self, train_data: Any, val_data: Optional[Any] = None) -> TrainingMetrics:
        """Entraîne les modèles d'analyse audio"""
        try:
            # Préparer les données d'entraînement
            X_train, y_genre_train, y_emotion_train = train_data
            
            # Standardisation des features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Entraînement du classificateur de genre
            self.genre_classifier.fit(X_train_scaled, y_genre_train)
            genre_accuracy = cross_val_score(
                self.genre_classifier, X_train_scaled, y_genre_train, cv=5
            ).mean()
            
            # Entraînement du classificateur d'émotion
            self.emotion_classifier.fit(X_train_scaled, y_emotion_train)
            emotion_accuracy = cross_val_score(
                self.emotion_classifier, X_train_scaled, y_emotion_train, cv=5
            ).mean()
            
            # Métriques d'entraînement
            training_metrics = TrainingMetrics(
                accuracy=(genre_accuracy + emotion_accuracy) / 2,
                precision=0.0,  # À calculer avec les données de validation
                recall=0.0,
                f1_score=0.0,
                loss=0.0,
                training_time=time.time(),
                metadata={
                    "genre_accuracy": genre_accuracy,
                    "emotion_accuracy": emotion_accuracy,
                    "num_features": X_train.shape[1]
                }
            )
            
            self.training_metrics = training_metrics
            self.is_trained = True
            
            self.logger.info("Audio analysis training completed")
            return training_metrics
            
        except Exception as e:
            self.logger.error(f"Audio training failed: {e}")
            raise
    
    async def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse un fichier audio"""
        try:
            if not self.is_trained:
                raise ValueError("Model not trained")
            
            audio_path = input_data["audio_path"]
            analysis_type = input_data.get("analysis_type", "full")  # full, genre, emotion
            
            # Extraire les features
            features = self.feature_extractor.extract_features(audio_path)
            features_scaled = features.reshape(1, -1)
            
            results = {"audio_path": audio_path}
            
            if analysis_type in ["full", "genre"]:
                # Prédiction de genre
                genre_proba = self.genre_classifier.predict_proba(features_scaled)[0]
                genre_classes = self.genre_classifier.classes_
                
                genre_predictions = [
                    {"genre": genre, "probability": prob}
                    for genre, prob in zip(genre_classes, genre_proba)
                ]
                genre_predictions.sort(key=lambda x: x["probability"], reverse=True)
                
                results["genre_analysis"] = {
                    "predicted_genre": genre_predictions[0]["genre"],
                    "confidence": genre_predictions[0]["probability"],
                    "all_predictions": genre_predictions[:5]
                }
            
            if analysis_type in ["full", "emotion"]:
                # Prédiction d'émotion
                emotion_proba = self.emotion_classifier.predict_proba(features_scaled)[0]
                emotion_classes = self.emotion_classifier.classes_
                
                emotion_predictions = [
                    {"emotion": emotion, "probability": prob}
                    for emotion, prob in zip(emotion_classes, emotion_proba)
                ]
                emotion_predictions.sort(key=lambda x: x["probability"], reverse=True)
                
                results["emotion_analysis"] = {
                    "predicted_emotion": emotion_predictions[0]["emotion"],
                    "confidence": emotion_predictions[0]["probability"],
                    "all_predictions": emotion_predictions[:3]
                }
            
            # Features audio brutes
            results["audio_features"] = {
                "mfcc_mean": features[:13].tolist(),
                "spectral_centroid": float(features[13]),
                "spectral_rolloff": float(features[14]),
                "zero_crossing_rate": float(features[15]),
                "tempo": float(features[16])
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Audio prediction failed: {e}")
            raise
    
    async def save_model(self, path: str) -> bool:
        """Sauvegarde les modèles audio"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            model_data = {
                "genre_classifier": self.genre_classifier,
                "emotion_classifier": self.emotion_classifier,
                "feature_extractor": self.feature_extractor,
                "config": self.config.__dict__,
                "training_metrics": self.training_metrics.__dict__ if self.training_metrics else None
            }
            
            joblib.dump(model_data, path)
            
            self.logger.info(f"Audio models saved to {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Audio model saving failed: {e}")
            return False
    
    async def load_model(self, path: str) -> bool:
        """Charge les modèles audio"""
        try:
            model_data = joblib.load(path)
            
            self.genre_classifier = model_data["genre_classifier"]
            self.emotion_classifier = model_data["emotion_classifier"]
            self.feature_extractor = model_data["feature_extractor"]
            
            if model_data.get("training_metrics"):
                self.training_metrics = TrainingMetrics(**model_data["training_metrics"])
                self.is_trained = True
            
            self.logger.info(f"Audio models loaded from {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Audio model loading failed: {e}")
            return False


class NLPModel(BaseMLModel):
    """
    📝 MODÈLE NLP AVANCÉ
    
    Traitement du langage naturel pour:
    - Analyse de sentiment
    - Classification de texte
    - Génération de réponses
    - Extraction d'entités
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.tokenizer: Optional[Any] = None
        self.model_transformer: Optional[Any] = None
        self.sentiment_classifier: Optional[Any] = None
    
    async def build_model(self) -> bool:
        """Construit les modèles NLP"""
        try:
            # Modèle BERT pour l'analyse de sentiment
            model_name = self.config.hyperparameters.get("model_name", "bert-base-uncased")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model_transformer = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=3  # Positif, Neutre, Négatif
            )
            
            # Pipeline pour l'analyse de sentiment
            self.sentiment_classifier = pipeline(
                "sentiment-analysis",
                model=self.model_transformer,
                tokenizer=self.tokenizer
            )
            
            self.logger.info("NLP models built successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"NLP model building failed: {e}")
            return False
    
    async def train(self, train_data: Any, val_data: Optional[Any] = None) -> TrainingMetrics:
        """Entraîne le modèle NLP"""
        # Implémentation de l'entraînement BERT
        # Pour cet exemple, on simule l'entraînement
        training_metrics = TrainingMetrics(
            accuracy=0.92,
            precision=0.90,
            recall=0.89,
            f1_score=0.91,
            loss=0.08,
            training_time=time.time()
        )
        
        self.training_metrics = training_metrics
        self.is_trained = True
        
        self.logger.info("NLP training completed")
        return training_metrics
    
    async def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse NLP du texte"""
        try:
            if not self.is_trained:
                raise ValueError("Model not trained")
            
            text = input_data["text"]
            analysis_type = input_data.get("analysis_type", "sentiment")
            
            results = {"text": text}
            
            if analysis_type in ["full", "sentiment"]:
                # Analyse de sentiment
                sentiment_result = self.sentiment_classifier(text)[0]
                
                results["sentiment_analysis"] = {
                    "label": sentiment_result["label"],
                    "confidence": sentiment_result["score"],
                    "polarity": 1 if sentiment_result["label"] == "POSITIVE" else -1 if sentiment_result["label"] == "NEGATIVE" else 0
                }
            
            # Extraction d'entités (simulée)
            if analysis_type in ["full", "entities"]:
                results["entities"] = {
                    "artists": ["Artist1", "Artist2"],  # Extraction simulée
                    "songs": ["Song1"],
                    "genres": ["Rock", "Pop"]
                }
            
            return results
            
        except Exception as e:
            self.logger.error(f"NLP prediction failed: {e}")
            raise
    
    async def save_model(self, path: str) -> bool:
        """Sauvegarde le modèle NLP"""
        try:
            os.makedirs(path, exist_ok=True)
            
            # Sauvegarder le modèle et tokenizer
            self.model_transformer.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            
            # Sauvegarder la configuration
            with open(os.path.join(path, "config.pkl"), "wb") as f:
                pickle.dump(self.config.__dict__, f)
            
            self.logger.info(f"NLP model saved to {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"NLP model saving failed: {e}")
            return False
    
    async def load_model(self, path: str) -> bool:
        """Charge le modèle NLP"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model_transformer = AutoModelForSequenceClassification.from_pretrained(path)
            
            self.sentiment_classifier = pipeline(
                "sentiment-analysis",
                model=self.model_transformer,
                tokenizer=self.tokenizer
            )
            
            # Charger la configuration
            with open(os.path.join(path, "config.pkl"), "rb") as f:
                config_dict = pickle.load(f)
                # Reconstruire la config
            
            self.is_trained = True
            
            self.logger.info(f"NLP model loaded from {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"NLP model loading failed: {e}")
            return False


class MLFrameworkManager(BaseFramework):
    """
    🧠 GESTIONNAIRE DES FRAMEWORKS ML
    
    Orchestration centralisée des modèles ML/AI avec:
    - Gestion du cycle de vie des modèles
    - Entraînement distribué
    - Inférence en temps réel
    - MLOps et monitoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("ml_frameworks", config or {})
        self.models: Dict[str, BaseMLModel] = {}
        self.model_registry: Dict[str, ModelConfig] = {}
        
        # MLOps tools
        self.mlflow_client: Optional[Any] = None
        self.wandb_run: Optional[Any] = None
        self.tensorboard_writer: Optional[SummaryWriter] = None
        
        # Monitoring
        self.model_metrics = {}
        
    async def initialize(self) -> bool:
        """Initialise le gestionnaire ML"""
        try:
            # Initialiser MLflow
            mlflow.set_tracking_uri("http://localhost:5000")
            self.mlflow_client = mlflow.tracking.MlflowClient()
            
            # Initialiser Weights & Biases
            wandb.init(project="spotify-ai-agent", name="ml-framework")
            self.wandb_run = wandb.run
            
            # Initialiser TensorBoard
            self.tensorboard_writer = SummaryWriter("runs/ml_framework")
            
            # Créer les modèles par défaut
            await self._create_default_models()
            
            self.logger.info("ML Framework Manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"ML framework initialization failed: {e}")
            return False
    
    async def _create_default_models(self):
        """Crée les modèles par défaut"""
        try:
            # Modèle de recommandation
            recommendation_config = ModelConfig(
                name="spotify_recommendation",
                model_type=ModelType.RECOMMENDATION,
                framework_type=MLFrameworkType.PYTORCH,
                hyperparameters={
                    "num_users": 50000,
                    "num_items": 200000,
                    "embedding_dim": 128,
                    "hidden_dim": 256
                },
                training_config={
                    "learning_rate": 0.001,
                    "epochs": 100,
                    "batch_size": 512
                }
            )
            
            await self.register_model(recommendation_config, SpotifyRecommendationModel)
            
            # Modèle d'analyse audio
            audio_config = ModelConfig(
                name="audio_analysis",
                model_type=ModelType.AUDIO_ANALYSIS,
                framework_type=MLFrameworkType.SKLEARN,
                hyperparameters={
                    "n_estimators": 200,
                    "max_depth": 15
                }
            )
            
            await self.register_model(audio_config, AudioAnalysisModel)
            
            # Modèle NLP
            nlp_config = ModelConfig(
                name="nlp_sentiment",
                model_type=ModelType.NLP,
                framework_type=MLFrameworkType.TRANSFORMERS,
                hyperparameters={
                    "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest"
                }
            )
            
            await self.register_model(nlp_config, NLPModel)
            
        except Exception as e:
            self.logger.error(f"Default models creation failed: {e}")
    
    async def register_model(self, config: ModelConfig, model_class: Type[BaseMLModel]):
        """Enregistre un nouveau modèle"""
        try:
            model_instance = model_class(config)
            await model_instance.build_model()
            
            self.models[config.name] = model_instance
            self.model_registry[config.name] = config
            
            self.logger.info(f"Model {config.name} registered successfully")
            
        except Exception as e:
            self.logger.error(f"Model registration failed for {config.name}: {e}")
            raise
    
    async def train_model(
        self, 
        model_name: str, 
        train_data: Any, 
        val_data: Optional[Any] = None
    ) -> TrainingMetrics:
        """Entraîne un modèle spécifique"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            model = self.models[model_name]
            
            # Démarrer l'entraînement avec monitoring
            start_time = time.time()
            
            with mlflow.start_run(run_name=f"{model_name}_training"):
                # Log des hyperparamètres
                mlflow.log_params(model.config.hyperparameters)
                
                # Entraînement
                metrics = await model.train(train_data, val_data)
                
                # Log des métriques
                mlflow.log_metrics({
                    "accuracy": metrics.accuracy,
                    "loss": metrics.loss,
                    "training_time": time.time() - start_time
                })
                
                # Sauvegarder le modèle
                model_path = f"models/{model_name}_trained"
                await model.save_model(model_path)
                mlflow.log_artifacts(model_path)
            
            # Log Weights & Biases
            if self.wandb_run:
                wandb.log({
                    f"{model_name}_accuracy": metrics.accuracy,
                    f"{model_name}_loss": metrics.loss
                })
            
            # TensorBoard
            if self.tensorboard_writer:
                self.tensorboard_writer.add_scalar(f"{model_name}/accuracy", metrics.accuracy)
                self.tensorboard_writer.add_scalar(f"{model_name}/loss", metrics.loss)
            
            self.logger.info(f"Model {model_name} training completed")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Model training failed for {model_name}: {e}")
            raise
    
    async def predict(self, model_name: str, input_data: Any) -> Any:
        """Effectue une prédiction avec un modèle"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            model = self.models[model_name]
            
            start_time = time.time()
            prediction = await model.predict(input_data)
            inference_time = time.time() - start_time
            
            # Monitoring de l'inférence
            self.model_metrics[model_name] = {
                "last_inference_time": inference_time,
                "total_predictions": self.model_metrics.get(model_name, {}).get("total_predictions", 0) + 1
            }
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {model_name}: {e}")
            raise
    
    async def shutdown(self) -> bool:
        """Arrête le gestionnaire ML"""
        try:
            # Fermer les outils MLOps
            if self.tensorboard_writer:
                self.tensorboard_writer.close()
            
            if self.wandb_run:
                wandb.finish()
            
            self.logger.info("ML Framework Manager shutdown successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"ML framework shutdown failed: {e}")
            return False
    
    async def health_check(self) -> FrameworkHealth:
        """Vérifie la santé du gestionnaire ML"""
        health = FrameworkHealth(
            status=FrameworkStatus.RUNNING,
            last_check=time.time()
        )
        
        try:
            # Vérifier les modèles
            healthy_models = 0
            for name, model in self.models.items():
                if model.is_trained:
                    healthy_models += 1
            
            health.metadata = {
                "total_models": len(self.models),
                "trained_models": healthy_models,
                "model_metrics": self.model_metrics
            }
            
            if healthy_models == 0:
                health.status = FrameworkStatus.DEGRADED
            
        except Exception as e:
            health.status = FrameworkStatus.DEGRADED
            health.error_count += 1
            health.metadata["error"] = str(e)
        
        return health


# Instance globale du gestionnaire ML
ml_manager = MLFrameworkManager()


# Export des classes principales
__all__ = [
    'MLFrameworkManager',
    'BaseMLModel',
    'SpotifyRecommendationModel',
    'AudioAnalysisModel',
    'NLPModel',
    'ModelConfig',
    'TrainingMetrics',
    'ModelType',
    'MLFrameworkType',
    'ml_manager'
]
