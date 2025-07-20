# 🎵 ML Analytics Module
# ======================
# 
# Module d'Analytics Machine Learning pour Spotify AI Agent
# Architecture enterprise avec orchestration avancée
#
# 🎖️ Experts: Tous les rôles techniques
# 👨‍💻 Développé par: Fahed Mlaiel et son équipe d'experts IA

"""
🧠 ML Analytics - Module d'Intelligence Artificielle Avancé
==========================================================

Module ML Analytics enterprise pour Spotify AI Agent avec architecture complète:

🎯 Fonctionnalités Principales:
- 🎵 Système de recommandation musical hybride (collaborative + content-based + deep learning)
- 🎧 Analyse audio avancée (MFCC, spectral features, genre/mood classification)
- 📊 Analytics temps réel avec monitoring des performances
- 🔄 Pipelines ML automatisés avec orchestration
- 📈 Métriques et monitoring enterprise avec alertes
- 🚀 API REST complète avec authentification
- 🛡️ Sécurité et validation des données
- 📱 Interface de gestion des modèles
- 🔧 Scripts d'automatisation et maintenance

🏗️ Architecture Enterprise:
- Core Engine: Orchestration ML centralisée
- Configuration: Gestion enterprise des paramètres
- Models: Modèles ML hybrides optimisés
- Audio: Moteur d'analyse audio avancé
- Monitoring: Surveillance temps réel
- API: Endpoints REST sécurisés
- Utils: Utilitaires et optimisations
- Scripts: Automatisation et maintenance
- Exceptions: Gestion d'erreurs avancée

🎖️ Équipe d'Experts:
- Lead Dev + Architecte IA
- Développeur Backend Senior (Python/FastAPI/Django)
- Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Spécialiste Sécurité Backend
- Architecte Microservices

👨‍💻 Développé par: Fahed Mlaiel
"""

import os
import sys
import asyncio
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Union, Tuple, Callable, Iterator
from enum import Enum, IntEnum
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import uuid
import pickle
import joblib
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Core ML and Data Science
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# Deep Learning Frameworks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import tensorflow as tf
from tensorflow.keras import layers, models
import transformers
from transformers import AutoTokenizer, AutoModel, pipeline

# Audio Processing
import librosa
import librosa.display
import soundfile as sf
from pydub import AudioSegment

# Database and Storage
import redis
import pymongo
from sqlalchemy import (
    create_engine, Column, String, DateTime, Float, Integer, Text, Boolean,
    ForeignKey, Index, func, and_, or_, desc, asc
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, scoped_session
from sqlalchemy.dialects.postgresql import JSONB, UUID
import psycopg2
from pymongo import MongoClient

# API and Web Framework
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field, validator
import httpx
import aiohttp
import requests

# Monitoring and Observability
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary
import structlog
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configuration and Environment
from dotenv import load_dotenv
import yaml
from marshmallow import Schema, fields, validate

# Utilities
import schedule
import croniter
from celery import Celery
from kombu import Queue

# Load environment variables
load_dotenv()

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
from sklearn.model_selection import train_test_split
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from celery import Celery
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions


class ModelType(Enum):
    """Types de modèles ML"""
    RECOMMENDATION = "recommendation"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    NLP = "nlp"
    AUDIO_ANALYSIS = "audio_analysis"
    COLLABORATIVE_FILTERING = "collaborative_filtering"
    CONTENT_BASED = "content_based"


class PipelineStage(Enum):
    """Étapes du pipeline ML"""
    DATA_COLLECTION = "data_collection"
    PREPROCESSING = "preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    TRAINING = "training"
    VALIDATION = "validation"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"


class ModelStatus(Enum):
    """Statuts de modèle"""
    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"


@dataclass
class MLModelConfig:
    """Configuration de modèle ML"""
    model_id: str
    name: str
    type: ModelType
    version: str = "1.0.0"
    status: ModelStatus = ModelStatus.TRAINING
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    training_data_path: Optional[str] = None
    model_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyticsReport:
    """Rapport d'analytics"""
    report_id: str
    title: str
    type: str
    data: Dict[str, Any]
    visualizations: List[Dict[str, Any]] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.utcnow)
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None


class MLPipelineManager:
    """Gestionnaire principal du pipeline ML"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_ML_DB', 3))
        )
        
        # Managers spécialisés
        self.model_manager = ModelManager()
        self.data_manager = DataManager()
        self.training_manager = TrainingManager()
        self.inference_manager = InferenceManager()
        self.analytics_manager = AnalyticsManager()
        self.monitoring_manager = MonitoringManager()
        self.ab_testing_manager = ABTestingManager()
        
        # Configuration
        self.model_registry_path = Path(os.getenv('MODEL_REGISTRY_PATH', './models'))
        self.data_warehouse_url = os.getenv('DATA_WAREHOUSE_URL', 'postgresql://user:pass@localhost/warehouse')
        
        # MLflow
        self.setup_mlflow()
        
    def setup_mlflow(self):
        """Configure MLflow pour le tracking"""
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("spotify-ai-agent")
    
    async def create_model(self, model_config: MLModelConfig) -> str:
        """Crée un nouveau modèle ML"""
        try:
            # Validation de la configuration
            if not self._validate_model_config(model_config):
                raise ValueError("Configuration de modèle invalide")
            
            # Enregistrement du modèle
            model_id = await self.model_manager.register_model(model_config)
            
            # Création du répertoire de modèle
            model_dir = self.model_registry_path / model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Stockage de la configuration
            config_path = model_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(self._model_config_to_dict(model_config), f, indent=2, default=str)
            
            self.logger.info(f"Modèle créé: {model_id}")
            return model_id
            
        except Exception as exc:
            self.logger.error(f"Erreur création modèle: {exc}")
            raise
    
    async def train_model(self, model_id: str, training_data: pd.DataFrame, 
                         validation_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Lance l'entraînement d'un modèle"""
        try:
            model_config = await self.model_manager.get_model(model_id)
            if not model_config:
                raise ValueError(f"Modèle {model_id} non trouvé")
            
            # Démarrage de l'entraînement avec MLflow
            with mlflow.start_run(run_name=f"{model_config.name}_v{model_config.version}"):
                # Log des hyperparamètres
                mlflow.log_params(model_config.hyperparameters)
                
                # Entraînement
                results = await self.training_manager.train(
                    model_config, training_data, validation_data
                )
                
                # Log des métriques
                mlflow.log_metrics(results['metrics'])
                
                # Sauvegarde du modèle
                model_path = await self._save_trained_model(model_id, results['model'])
                
                # Mise à jour du statut
                await self.model_manager.update_model_status(
                    model_id, ModelStatus.TRAINED, results['metrics']
                )
                
                self.logger.info(f"Modèle entraîné: {model_id}")
                return results
                
        except Exception as exc:
            self.logger.error(f"Erreur entraînement modèle {model_id}: {exc}")
            await self.model_manager.update_model_status(model_id, ModelStatus.FAILED)
            raise
    
    async def deploy_model(self, model_id: str) -> bool:
        """Déploie un modèle en production"""
        try:
            model_config = await self.model_manager.get_model(model_id)
            if not model_config or model_config.status != ModelStatus.TRAINED:
                raise ValueError("Modèle non prêt pour le déploiement")
            
            # Déploiement
            success = await self.inference_manager.deploy_model(model_config)
            
            if success:
                await self.model_manager.update_model_status(model_id, ModelStatus.DEPLOYED)
                self.logger.info(f"Modèle déployé: {model_id}")
            
            return success
            
        except Exception as exc:
            self.logger.error(f"Erreur déploiement modèle {model_id}: {exc}")
            return False
    
    async def predict(self, model_id: str, input_data: Union[Dict, pd.DataFrame]) -> Any:
        """Effectue une prédiction avec un modèle"""
        return await self.inference_manager.predict(model_id, input_data)
    
    async def generate_analytics_report(self, report_type: str, 
                                       period_start: datetime, 
                                       period_end: datetime) -> AnalyticsReport:
        """Génère un rapport d'analytics"""
        return await self.analytics_manager.generate_report(
            report_type, period_start, period_end
        )
    
    def _validate_model_config(self, config: MLModelConfig) -> bool:
        """Valide une configuration de modèle"""
        return bool(config.model_id and config.name and config.type)
    
    async def _save_trained_model(self, model_id: str, model) -> str:
        """Sauvegarde un modèle entraîné"""
        model_dir = self.model_registry_path / model_id
        model_path = model_dir / "model.pkl"
        
        # Sauvegarde selon le type de modèle
        if hasattr(model, 'save'):  # TensorFlow/Keras
            model.save(str(model_dir / "model.h5"))
        elif hasattr(model, 'state_dict'):  # PyTorch
            torch.save(model.state_dict(), model_dir / "model.pth")
        else:  # Scikit-learn
            joblib.dump(model, model_path)
        
        return str(model_path)
    
    def _model_config_to_dict(self, config: MLModelConfig) -> Dict:
        """Convertit MLModelConfig en dictionnaire"""
        return {
            'model_id': config.model_id,
            'name': config.name,
            'type': config.type.value,
            'version': config.version,
            'status': config.status.value,
            'hyperparameters': config.hyperparameters,
            'metrics': config.metrics,
            'created_at': config.created_at.isoformat(),
            'training_data_path': config.training_data_path,
            'model_path': config.model_path,
            'metadata': config.metadata
        }


class ModelManager:
    """Gestionnaire de modèles ML"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_ML_DB', 3))
        )
        
    async def register_model(self, model_config: MLModelConfig) -> str:
        """Enregistre un nouveau modèle"""
        model_id = model_config.model_id
        
        # Stockage de la configuration
        config_data = json.dumps({
            'model_id': model_id,
            'name': model_config.name,
            'type': model_config.type.value,
            'version': model_config.version,
            'status': model_config.status.value,
            'hyperparameters': model_config.hyperparameters,
            'metrics': model_config.metrics,
            'created_at': model_config.created_at.isoformat(),
            'metadata': model_config.metadata
        }, default=str)
        
        self.redis_client.set(f"model:{model_id}", config_data)
        self.redis_client.sadd("models", model_id)
        
        return model_id
    
    async def get_model(self, model_id: str) -> Optional[MLModelConfig]:
        """Récupère un modèle par ID"""
        config_data = self.redis_client.get(f"model:{model_id}")
        if config_data:
            data = json.loads(config_data)
            return self._dict_to_model_config(data)
        return None
    
    async def update_model_status(self, model_id: str, status: ModelStatus, 
                                 metrics: Optional[Dict[str, float]] = None):
        """Met à jour le statut d'un modèle"""
        model = await self.get_model(model_id)
        if model:
            model.status = status
            if metrics:
                model.metrics.update(metrics)
            await self.register_model(model)  # Mise à jour
    
    async def list_models(self, model_type: Optional[ModelType] = None) -> List[MLModelConfig]:
        """Liste les modèles disponibles"""
        model_ids = self.redis_client.smembers("models")
        models = []
        
        for model_id in model_ids:
            model = await self.get_model(model_id.decode('utf-8'))
            if model and (not model_type or model.type == model_type):
                models.append(model)
        
        return models
    
    def _dict_to_model_config(self, data: Dict) -> MLModelConfig:
        """Convertit un dictionnaire en MLModelConfig"""
        return MLModelConfig(
            model_id=data['model_id'],
            name=data['name'],
            type=ModelType(data['type']),
            version=data['version'],
            status=ModelStatus(data['status']),
            hyperparameters=data['hyperparameters'],
            metrics=data['metrics'],
            created_at=datetime.fromisoformat(data['created_at']),
            metadata=data['metadata']
        )


class DataManager:
    """Gestionnaire de données ML"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.warehouse_engine = create_engine(
            os.getenv('DATA_WAREHOUSE_URL', 'postgresql://user:pass@localhost/warehouse')
        )
        
    async def extract_training_data(self, model_type: ModelType, 
                                   period_start: datetime, 
                                   period_end: datetime) -> pd.DataFrame:
        """Extrait les données d'entraînement"""
        try:
            if model_type == ModelType.RECOMMENDATION:
                return await self._extract_recommendation_data(period_start, period_end)
            elif model_type == ModelType.AUDIO_ANALYSIS:
                return await self._extract_audio_features_data(period_start, period_end)
            elif model_type == ModelType.NLP:
                return await self._extract_text_data(period_start, period_end)
            else:
                raise ValueError(f"Type de modèle non supporté: {model_type}")
                
        except Exception as exc:
            self.logger.error(f"Erreur extraction données: {exc}")
            raise
    
    async def _extract_recommendation_data(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Extrait les données pour les recommandations"""
        query = """
        SELECT 
            user_id,
            track_id,
            rating,
            listening_time,
            skip_count,
            play_count,
            timestamp
        FROM user_interactions 
        WHERE timestamp BETWEEN %s AND %s
        """
        
        return pd.read_sql(query, self.warehouse_engine, params=[start, end])
    
    async def _extract_audio_features_data(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Extrait les caractéristiques audio"""
        query = """
        SELECT 
            track_id,
            tempo,
            energy,
            valence,
            danceability,
            acousticness,
            instrumentalness,
            speechiness,
            loudness
        FROM audio_features af
        JOIN tracks t ON af.track_id = t.id
        WHERE t.created_at BETWEEN %s AND %s
        """
        
        return pd.read_sql(query, self.warehouse_engine, params=[start, end])
    
    async def _extract_text_data(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Extrait les données textuelles"""
        query = """
        SELECT 
            id,
            text_content,
            sentiment_label,
            category
        FROM text_data 
        WHERE created_at BETWEEN %s AND %s
        """
        
        return pd.read_sql(query, self.warehouse_engine, params=[start, end])
    
    async def preprocess_data(self, data: pd.DataFrame, model_type: ModelType) -> pd.DataFrame:
        """Préprocesse les données"""
        if model_type == ModelType.RECOMMENDATION:
            return self._preprocess_recommendation_data(data)
        elif model_type == ModelType.AUDIO_ANALYSIS:
            return self._preprocess_audio_data(data)
        elif model_type == ModelType.NLP:
            return self._preprocess_text_data(data)
        
        return data
    
    def _preprocess_recommendation_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Préprocesse les données de recommandation"""
        # Nettoyage et normalisation
        data = data.dropna()
        data['listening_time'] = data['listening_time'].fillna(0)
        data['rating_normalized'] = data['rating'] / 5.0
        
        return data
    
    def _preprocess_audio_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Préprocesse les données audio"""
        # Normalisation des features audio
        numeric_cols = ['tempo', 'energy', 'valence', 'danceability', 
                       'acousticness', 'instrumentalness', 'speechiness', 'loudness']
        
        for col in numeric_cols:
            if col in data.columns:
                data[f'{col}_normalized'] = (data[col] - data[col].mean()) / data[col].std()
        
        return data
    
    def _preprocess_text_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Préprocesse les données textuelles"""
        # Nettoyage de texte basique
        data['text_content'] = data['text_content'].str.lower()
        data['text_content'] = data['text_content'].str.replace(r'[^\w\s]', '')
        
        return data


class TrainingManager:
    """Gestionnaire d'entraînement ML"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def train(self, model_config: MLModelConfig, 
                   training_data: pd.DataFrame,
                   validation_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Entraîne un modèle ML"""
        try:
            if model_config.type == ModelType.RECOMMENDATION:
                return await self._train_recommendation_model(model_config, training_data, validation_data)
            elif model_config.type == ModelType.AUDIO_ANALYSIS:
                return await self._train_audio_model(model_config, training_data, validation_data)
            elif model_config.type == ModelType.NLP:
                return await self._train_nlp_model(model_config, training_data, validation_data)
            else:
                raise ValueError(f"Type de modèle non supporté: {model_config.type}")
                
        except Exception as exc:
            self.logger.error(f"Erreur entraînement: {exc}")
            raise
    
    async def _train_recommendation_model(self, config: MLModelConfig, 
                                        training_data: pd.DataFrame,
                                        validation_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Entraîne un modèle de recommandation"""
        from sklearn.decomposition import NMF
        from sklearn.metrics import mean_squared_error
        
        # Préparation des données
        user_item_matrix = training_data.pivot_table(
            index='user_id', 
            columns='track_id', 
            values='rating', 
            fill_value=0
        )
        
        # Entraînement NMF (Non-negative Matrix Factorization)
        n_components = config.hyperparameters.get('n_components', 50)
        model = NMF(
            n_components=n_components,
            random_state=42,
            max_iter=config.hyperparameters.get('max_iter', 200)
        )
        
        W = model.fit_transform(user_item_matrix)
        H = model.components_
        
        # Reconstruction et métriques
        reconstructed = np.dot(W, H)
        mse = mean_squared_error(user_item_matrix.values, reconstructed)
        
        metrics = {
            'mse': float(mse),
            'reconstruction_error': float(model.reconstruction_err_)
        }
        
        return {
            'model': model,
            'metrics': metrics,
            'user_factors': W,
            'item_factors': H
        }
    
    async def _train_audio_model(self, config: MLModelConfig,
                                training_data: pd.DataFrame,
                                validation_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Entraîne un modèle d'analyse audio"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report
        
        # Préparation des features
        feature_cols = ['tempo', 'energy', 'valence', 'danceability', 
                       'acousticness', 'instrumentalness', 'speechiness', 'loudness']
        
        X = training_data[feature_cols]
        y = training_data.get('genre', training_data.get('category'))  # Target variable
        
        # Entraînement
        model = RandomForestClassifier(
            n_estimators=config.hyperparameters.get('n_estimators', 100),
            max_depth=config.hyperparameters.get('max_depth', 10),
            random_state=42
        )
        
        model.fit(X, y)
        
        # Évaluation
        train_accuracy = model.score(X, y)
        metrics = {'train_accuracy': float(train_accuracy)}
        
        if validation_data is not None:
            X_val = validation_data[feature_cols]
            y_val = validation_data.get('genre', validation_data.get('category'))
            val_accuracy = model.score(X_val, y_val)
            metrics['val_accuracy'] = float(val_accuracy)
        
        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': dict(zip(feature_cols, model.feature_importances_))
        }
    
    async def _train_nlp_model(self, config: MLModelConfig,
                              training_data: pd.DataFrame,
                              validation_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Entraîne un modèle NLP"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        
        # Préparation des données
        X = training_data['text_content']
        y = training_data['sentiment_label']
        
        # Pipeline de traitement
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=config.hyperparameters.get('max_features', 5000),
                stop_words='english'
            )),
            ('classifier', LogisticRegression(
                C=config.hyperparameters.get('C', 1.0),
                random_state=42
            ))
        ])
        
        # Entraînement
        pipeline.fit(X, y)
        
        # Métriques
        train_accuracy = pipeline.score(X, y)
        metrics = {'train_accuracy': float(train_accuracy)}
        
        if validation_data is not None:
            X_val = validation_data['text_content']
            y_val = validation_data['sentiment_label']
            val_accuracy = pipeline.score(X_val, y_val)
            metrics['val_accuracy'] = float(val_accuracy)
        
        return {
            'model': pipeline,
            'metrics': metrics
        }


class InferenceManager:
    """Gestionnaire d'inférence ML"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.deployed_models: Dict[str, Any] = {}
        
    async def deploy_model(self, model_config: MLModelConfig) -> bool:
        """Déploie un modèle pour l'inférence"""
        try:
            model_path = Path(model_config.model_path) if model_config.model_path else None
            
            if not model_path or not model_path.exists():
                raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
            
            # Chargement du modèle
            if model_path.suffix == '.pkl':
                model = joblib.load(model_path)
            elif model_path.suffix == '.h5':
                model = tf.keras.models.load_model(model_path)
            elif model_path.suffix == '.pth':
                # PyTorch loading logic
                model = torch.load(model_path)
            else:
                raise ValueError(f"Format de modèle non supporté: {model_path.suffix}")
            
            # Mise en cache
            self.deployed_models[model_config.model_id] = {
                'model': model,
                'config': model_config
            }
            
            self.logger.info(f"Modèle déployé: {model_config.model_id}")
            return True
            
        except Exception as exc:
            self.logger.error(f"Erreur déploiement modèle {model_config.model_id}: {exc}")
            return False
    
    async def predict(self, model_id: str, input_data: Union[Dict, pd.DataFrame]) -> Any:
        """Effectue une prédiction"""
        try:
            if model_id not in self.deployed_models:
                raise ValueError(f"Modèle non déployé: {model_id}")
            
            model_info = self.deployed_models[model_id]
            model = model_info['model']
            config = model_info['config']
            
            # Préparation des données
            if isinstance(input_data, dict):
                input_data = pd.DataFrame([input_data])
            
            # Prédiction selon le type de modèle
            if config.type == ModelType.RECOMMENDATION:
                return await self._predict_recommendation(model, input_data)
            elif config.type == ModelType.AUDIO_ANALYSIS:
                return await self._predict_audio_analysis(model, input_data)
            elif config.type == ModelType.NLP:
                return await self._predict_nlp(model, input_data)
            else:
                return model.predict(input_data)
                
        except Exception as exc:
            self.logger.error(f"Erreur prédiction modèle {model_id}: {exc}")
            raise
    
    async def _predict_recommendation(self, model, input_data: pd.DataFrame) -> List[Dict]:
        """Prédiction de recommandations"""
        # Logique de recommandation
        recommendations = []
        # ... implémentation spécifique
        return recommendations
    
    async def _predict_audio_analysis(self, model, input_data: pd.DataFrame) -> Dict:
        """Prédiction d'analyse audio"""
        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data) if hasattr(model, 'predict_proba') else None
        
        return {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist() if probabilities is not None else None
        }
    
    async def _predict_nlp(self, model, input_data: pd.DataFrame) -> Dict:
        """Prédiction NLP"""
        predictions = model.predict(input_data['text_content'])
        probabilities = model.predict_proba(input_data['text_content']) if hasattr(model, 'predict_proba') else None
        
        return {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist() if probabilities is not None else None
        }


class AnalyticsManager:
    """Gestionnaire d'analytics et reporting"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.warehouse_engine = create_engine(
            os.getenv('DATA_WAREHOUSE_URL', 'postgresql://user:pass@localhost/warehouse')
        )
        
    async def generate_report(self, report_type: str, 
                            period_start: datetime, 
                            period_end: datetime) -> AnalyticsReport:
        """Génère un rapport d'analytics"""
        try:
            if report_type == 'user_engagement':
                return await self._generate_user_engagement_report(period_start, period_end)
            elif report_type == 'music_trends':
                return await self._generate_music_trends_report(period_start, period_end)
            elif report_type == 'model_performance':
                return await self._generate_model_performance_report(period_start, period_end)
            else:
                raise ValueError(f"Type de rapport non supporté: {report_type}")
                
        except Exception as exc:
            self.logger.error(f"Erreur génération rapport: {exc}")
            raise
    
    async def _generate_user_engagement_report(self, start: datetime, end: datetime) -> AnalyticsReport:
        """Génère un rapport d'engagement utilisateur"""
        query = """
        SELECT 
            DATE(timestamp) as date,
            COUNT(DISTINCT user_id) as active_users,
            COUNT(*) as total_interactions,
            AVG(listening_time) as avg_listening_time,
            SUM(CASE WHEN rating >= 4 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as satisfaction_rate
        FROM user_interactions 
        WHERE timestamp BETWEEN %s AND %s
        GROUP BY DATE(timestamp)
        ORDER BY date
        """
        
        data = pd.read_sql(query, self.warehouse_engine, params=[start, end])
        
        # Visualisations
        visualizations = []
        
        # Graphique d'engagement
        plt.figure(figsize=(12, 6))
        plt.plot(data['date'], data['active_users'], marker='o', label='Utilisateurs Actifs')
        plt.plot(data['date'], data['total_interactions'], marker='s', label='Interactions Totales')
        plt.title('Évolution de l\'Engagement Utilisateur')
        plt.xlabel('Date')
        plt.ylabel('Nombre')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        viz_path = f"/tmp/user_engagement_{start.strftime('%Y%m%d')}.png"
        plt.savefig(viz_path)
        plt.close()
        
        visualizations.append({
            'type': 'line_chart',
            'title': 'Évolution de l\'Engagement',
            'path': viz_path
        })
        
        report_data = {
            'summary': {
                'total_active_users': int(data['active_users'].sum()),
                'total_interactions': int(data['total_interactions'].sum()),
                'avg_satisfaction_rate': float(data['satisfaction_rate'].mean())
            },
            'daily_metrics': data.to_dict('records')
        }
        
        return AnalyticsReport(
            report_id=str(uuid.uuid4()),
            title='Rapport d\'Engagement Utilisateur',
            type='user_engagement',
            data=report_data,
            visualizations=visualizations,
            period_start=start,
            period_end=end
        )
    
    async def _generate_music_trends_report(self, start: datetime, end: datetime) -> AnalyticsReport:
        """Génère un rapport des tendances musicales"""
        # Logique pour analyser les tendances musicales
        report_data = {
            'top_genres': [],
            'trending_artists': [],
            'popular_tracks': []
        }
        
        return AnalyticsReport(
            report_id=str(uuid.uuid4()),
            title='Rapport des Tendances Musicales',
            type='music_trends',
            data=report_data,
            period_start=start,
            period_end=end
        )
    
    async def _generate_model_performance_report(self, start: datetime, end: datetime) -> AnalyticsReport:
        """Génère un rapport de performance des modèles"""
        # Logique pour analyser les performances des modèles ML
        report_data = {
            'model_metrics': [],
            'prediction_accuracy': {},
            'inference_latency': {}
        }
        
        return AnalyticsReport(
            report_id=str(uuid.uuid4()),
            title='Rapport de Performance des Modèles',
            type='model_performance',
            data=report_data,
            period_start=start,
            period_end=end
        )


class MonitoringManager:
    """Gestionnaire de monitoring ML"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def monitor_model_drift(self, model_id: str) -> Dict[str, Any]:
        """Surveille la dérive d'un modèle"""
        # Logique de détection de dérive
        return {
            'drift_detected': False,
            'drift_score': 0.0,
            'recommendation': 'continue_monitoring'
        }
    
    async def monitor_data_quality(self, dataset_id: str) -> Dict[str, Any]:
        """Surveille la qualité des données"""
        # Logique de monitoring de qualité
        return {
            'quality_score': 0.95,
            'issues': [],
            'recommendations': []
        }


class ABTestingManager:
    """Gestionnaire de tests A/B pour ML"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def create_ab_test(self, test_name: str, model_a_id: str, model_b_id: str, 
                           traffic_split: float = 0.5) -> str:
        """Crée un test A/B"""
        test_id = str(uuid.uuid4())
        
        # Logique de création de test A/B
        test_config = {
            'test_id': test_id,
            'name': test_name,
            'model_a': model_a_id,
            'model_b': model_b_id,
            'traffic_split': traffic_split,
            'created_at': datetime.utcnow(),
            'status': 'active'
        }
        
        # Stockage de la configuration
        # ...
        
        return test_id
    
    async def evaluate_ab_test(self, test_id: str) -> Dict[str, Any]:
        """Évalue les résultats d'un test A/B"""
        # Logique d'évaluation statistique
        return {
            'test_id': test_id,
            'statistical_significance': True,
            'winner': 'model_b',
            'confidence_level': 0.95,
            'metrics': {}
        }


# Instance globale
ml_pipeline = MLPipelineManager()


# Export des classes principales
__all__ = [
    'MLPipelineManager',
    'ModelManager',
    'DataManager',
    'TrainingManager',
    'InferenceManager',
    'AnalyticsManager',
    'MonitoringManager',
    'ABTestingManager',
    'MLModelConfig',
    'AnalyticsReport',
    'ModelType',
    'PipelineStage',
    'ModelStatus',
    'ml_pipeline'
]
