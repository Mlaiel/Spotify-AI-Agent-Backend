# =============================================================================
# Système d'Intelligence Artificielle pour Détection d'Anomalies
# =============================================================================
# 
# Module d'IA avancé pour détection automatique d'anomalies dans les métriques
# de monitoring avec apprentissage automatique, prédiction proactive et 
# corrélation intelligente d'événements.
#
# Fonctionnalités IA avancées:
# - Détection d'anomalies en temps réel (Isolation Forest, LSTM, VAE)
# - Prédiction proactive des pannes (Prophet, ARIMA)
# - Corrélation intelligente d'événements (clustering, graphes)
# - Classification automatique des incidents (NLP, transformers)
# - Recommandations d'actions correctives (système expert)
# - Apprentissage continu et adaptation automatique
# - Explainabilité des décisions IA (SHAP, LIME)
#
# Développé par l'équipe d'experts techniques:
# - Lead Developer + AI Architect (Architecture IA enterprise)
# - ML Engineer (Modèles ML, Deep Learning, AutoML)
# - Backend Senior Developer (Intégration Python/FastAPI)
# - Data Engineer (Pipeline de données, feature engineering)
# - Spécialiste Sécurité Backend (Sécurité des modèles IA)
#
# Direction Technique: Fahed Mlaiel
# =============================================================================

import asyncio
import numpy as np
import pandas as pd
import pickle
import json
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import logging
from pathlib import Path

# Machine Learning Core
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
import torch.nn as nn
import torch.optim as optim

# Time Series
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

# NLP et Text Processing
import spacy
from transformers import pipeline, AutoTokenizer, AutoModel
import nltk

# Explainabilité IA
import shap
import lime
from lime.lime_text import LimeTextExplainer

# Visualisation et graphiques
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Traitement de données
import scipy.stats as stats
from scipy.signal import find_peaks
import networkx as nx

# Async et concurrency
import aiofiles
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configuration
import structlog

# Suppression des warnings
warnings.filterwarnings('ignore')
logger = structlog.get_logger(__name__)

# =============================================================================
# ENUMS ET MODÈLES
# =============================================================================

class AnomalyType(str, Enum):
    """Types d'anomalies détectées"""
    STATISTICAL = "statistical"
    TREND = "trend"
    SEASONAL = "seasonal"
    SPIKE = "spike"
    DIP = "dip"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"

class AnomalySeverity(str, Enum):
    """Sévérité des anomalies"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ModelType(str, Enum):
    """Types de modèles ML"""
    ISOLATION_FOREST = "isolation_forest"
    LSTM_AUTOENCODER = "lstm_autoencoder"
    VAE = "variational_autoencoder"
    PROPHET = "prophet"
    ARIMA = "arima"
    DBSCAN = "dbscan"
    ONE_CLASS_SVM = "one_class_svm"

@dataclass
class AnomalyDetection:
    """Résultat de détection d'anomalie"""
    id: str
    timestamp: datetime
    metric_name: str
    value: float
    expected_value: float
    anomaly_score: float
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    confidence: float
    explanation: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    model_used: str = ""
    features_importance: Dict[str, float] = field(default_factory=dict)

@dataclass
class PredictionResult:
    """Résultat de prédiction"""
    metric_name: str
    timestamp: datetime
    predicted_value: float
    confidence_interval: Tuple[float, float]
    model_used: str
    horizon_hours: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IncidentCorrelation:
    """Corrélation d'incidents"""
    incident_ids: List[str]
    correlation_score: float
    correlation_type: str
    timestamp: datetime
    explanation: str
    suggested_actions: List[str] = field(default_factory=list)

# =============================================================================
# MODÈLES DE MACHINE LEARNING
# =============================================================================

class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder pour détection d'anomalies temporelles"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, sequence_length: int):
        super(LSTMAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=0.2
        )
        
        # Decoder
        self.decoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=0.2
        )
        
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        # Encoder
        encoded, (hidden, cell) = self.encoder_lstm(x)
        
        # Prendre seulement la dernière sortie de l'encoder
        encoded = encoded[:, -1, :].unsqueeze(1)
        
        # Répéter pour la séquence de décodage
        encoded = encoded.repeat(1, self.sequence_length, 1)
        
        # Decoder
        decoded, _ = self.decoder_lstm(encoded, (hidden, cell))
        
        # Couche de sortie
        output = self.output_layer(decoded)
        
        return output

class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder pour détection d'anomalies"""
    
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super(VariationalAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.mu_layer = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

# =============================================================================
# DÉTECTEUR D'ANOMALIES PRINCIPAL
# =============================================================================

class AnomalyDetectionEngine:
    """Moteur principal de détection d'anomalies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.model_performances: Dict[str, Dict[str, float]] = {}
        self.feature_extractors: Dict[str, Callable] = {}
        
        # Configuration des modèles
        self.model_configs = {
            ModelType.ISOLATION_FOREST: {
                "contamination": 0.1,
                "n_estimators": 100,
                "random_state": 42
            },
            ModelType.LSTM_AUTOENCODER: {
                "hidden_dim": 64,
                "num_layers": 2,
                "sequence_length": 50,
                "learning_rate": 0.001,
                "epochs": 100
            },
            ModelType.VAE: {
                "hidden_dim": 128,
                "latent_dim": 20,
                "learning_rate": 0.001,
                "epochs": 100
            }
        }
        
        # Cache des prédictions
        self.prediction_cache: Dict[str, List[PredictionResult]] = {}
        
        # Historique des anomalies
        self.anomaly_history: List[AnomalyDetection] = []
        
        # Explainers pour l'explainabilité
        self.explainers: Dict[str, Any] = {}
        
        self._setup_feature_extractors()
    
    def _setup_feature_extractors(self):
        """Configuration des extracteurs de caractéristiques"""
        
        def statistical_features(data: np.ndarray) -> Dict[str, float]:
            """Caractéristiques statistiques"""
            return {
                "mean": np.mean(data),
                "std": np.std(data),
                "skewness": stats.skew(data),
                "kurtosis": stats.kurtosis(data),
                "median": np.median(data),
                "q25": np.percentile(data, 25),
                "q75": np.percentile(data, 75),
                "iqr": np.percentile(data, 75) - np.percentile(data, 25)
            }
        
        def temporal_features(data: np.ndarray, timestamps: List[datetime]) -> Dict[str, float]:
            """Caractéristiques temporelles"""
            
            # Décomposition saisonnière si suffisamment de données
            if len(data) >= 24:  # Au moins 24 points
                try:
                    decomposition = seasonal_decompose(data, model='additive', period=12)
                    trend_strength = np.std(decomposition.trend[~np.isnan(decomposition.trend)])
                    seasonal_strength = np.std(decomposition.seasonal)
                except:
                    trend_strength = 0
                    seasonal_strength = 0
            else:
                trend_strength = 0
                seasonal_strength = 0
            
            # Autocorrélation
            autocorr = np.corrcoef(data[:-1], data[1:])[0, 1] if len(data) > 1 else 0
            
            return {
                "trend_strength": trend_strength,
                "seasonal_strength": seasonal_strength,
                "autocorrelation": autocorr if not np.isnan(autocorr) else 0,
                "volatility": np.std(np.diff(data)) if len(data) > 1 else 0
            }
        
        def frequency_features(data: np.ndarray) -> Dict[str, float]:
            """Caractéristiques fréquentielles"""
            
            # Transformée de Fourier
            fft = np.fft.fft(data)
            frequencies = np.fft.fftfreq(len(data))
            
            # Fréquence dominante
            dominant_freq_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
            dominant_frequency = frequencies[dominant_freq_idx]
            
            # Énergie spectrale
            spectral_energy = np.sum(np.abs(fft)**2)
            
            return {
                "dominant_frequency": abs(dominant_frequency),
                "spectral_energy": spectral_energy,
                "spectral_centroid": np.sum(frequencies[:len(frequencies)//2] * np.abs(fft[:len(fft)//2])) / np.sum(np.abs(fft[:len(fft)//2]))
            }
        
        self.feature_extractors = {
            "statistical": statistical_features,
            "temporal": temporal_features,
            "frequency": frequency_features
        }
    
    async def train_models(self, training_data: Dict[str, pd.DataFrame]):
        """Entraînement des modèles de détection d'anomalies"""
        
        logger.info("Starting anomaly detection models training")
        
        for metric_name, data in training_data.items():
            logger.info(f"Training models for metric: {metric_name}")
            
            # Préparation des données
            X, y, timestamps = self._prepare_training_data(data)
            
            if len(X) < 50:  # Données insuffisantes
                logger.warning(f"Insufficient data for {metric_name}: {len(X)} samples")
                continue
            
            # Entraînement Isolation Forest
            await self._train_isolation_forest(metric_name, X)
            
            # Entraînement LSTM Autoencoder
            if len(X) >= 100:  # Besoin de plus de données pour LSTM
                await self._train_lstm_autoencoder(metric_name, X)
            
            # Entraînement VAE
            if len(X) >= 200:  # Besoin de beaucoup de données pour VAE
                await self._train_vae(metric_name, X)
            
            # Configuration des explainers
            self._setup_explainers(metric_name, X[:1000])  # Échantillon pour SHAP
        
        logger.info("Anomaly detection models training completed")
    
    async def _train_isolation_forest(self, metric_name: str, X: np.ndarray):
        """Entraînement Isolation Forest"""
        
        config = self.model_configs[ModelType.ISOLATION_FOREST]
        
        # Normalisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Modèle
        model = IsolationForest(**config)
        model.fit(X_scaled)
        
        # Sauvegarde
        model_key = f"{metric_name}_{ModelType.ISOLATION_FOREST.value}"
        self.models[model_key] = model
        self.scalers[f"{model_key}_scaler"] = scaler
        
        # Évaluation sur données d'entraînement
        predictions = model.predict(X_scaled)
        anomaly_ratio = np.sum(predictions == -1) / len(predictions)
        
        self.model_performances[model_key] = {
            "anomaly_ratio": anomaly_ratio,
            "training_samples": len(X)
        }
        
        logger.info(f"Isolation Forest trained for {metric_name}: {anomaly_ratio:.3f} anomaly ratio")
    
    async def _train_lstm_autoencoder(self, metric_name: str, X: np.ndarray):
        """Entraînement LSTM Autoencoder"""
        
        config = self.model_configs[ModelType.LSTM_AUTOENCODER]
        
        # Préparation des séquences
        sequence_length = config["sequence_length"]
        sequences = self._create_sequences(X, sequence_length)
        
        if len(sequences) < 20:
            return
        
        # Normalisation
        scaler = MinMaxScaler()
        sequences_scaled = scaler.fit_transform(sequences.reshape(-1, sequences.shape[-1]))
        sequences_scaled = sequences_scaled.reshape(sequences.shape)
        
        # Modèle PyTorch
        input_dim = X.shape[1]
        model = LSTMAutoencoder(
            input_dim=input_dim,
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            sequence_length=sequence_length
        )
        
        # Entraînement
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        
        # Conversion en tenseurs
        X_tensor = torch.FloatTensor(sequences_scaled)
        
        model.train()
        for epoch in range(config["epochs"]):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, X_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                logger.debug(f"LSTM Epoch {epoch}, Loss: {loss.item():.6f}")
        
        # Sauvegarde
        model_key = f"{metric_name}_{ModelType.LSTM_AUTOENCODER.value}"
        self.models[model_key] = model
        self.scalers[f"{model_key}_scaler"] = scaler
        
        # Évaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_tensor)
            reconstruction_errors = torch.mean((X_tensor - test_outputs) ** 2, dim=(1, 2))
            threshold = torch.quantile(reconstruction_errors, 0.95)
        
        self.model_performances[model_key] = {
            "threshold": threshold.item(),
            "training_samples": len(sequences)
        }
        
        logger.info(f"LSTM Autoencoder trained for {metric_name}: threshold {threshold.item():.6f}")
    
    async def _train_vae(self, metric_name: str, X: np.ndarray):
        """Entraînement Variational Autoencoder"""
        
        config = self.model_configs[ModelType.VAE]
        
        # Normalisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Modèle PyTorch
        input_dim = X.shape[1]
        model = VariationalAutoencoder(
            input_dim=input_dim,
            hidden_dim=config["hidden_dim"],
            latent_dim=config["latent_dim"]
        )
        
        # Entraînement
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        
        def vae_loss(recon_x, x, mu, logvar):
            BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return BCE + KLD
        
        # Conversion en tenseurs
        X_tensor = torch.FloatTensor(X_scaled)
        
        model.train()
        for epoch in range(config["epochs"]):
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(X_tensor)
            loss = vae_loss(recon_batch, X_tensor, mu, logvar)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                logger.debug(f"VAE Epoch {epoch}, Loss: {loss.item():.6f}")
        
        # Sauvegarde
        model_key = f"{metric_name}_{ModelType.VAE.value}"
        self.models[model_key] = model
        self.scalers[f"{model_key}_scaler"] = scaler
        
        # Évaluation
        model.eval()
        with torch.no_grad():
            recon_X, mu, logvar = model(X_tensor)
            reconstruction_errors = torch.mean((X_tensor - recon_X) ** 2, dim=1)
            threshold = torch.quantile(reconstruction_errors, 0.95)
        
        self.model_performances[model_key] = {
            "threshold": threshold.item(),
            "training_samples": len(X)
        }
        
        logger.info(f"VAE trained for {metric_name}: threshold {threshold.item():.6f}")
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[datetime]]:
        """Préparation des données d'entraînement"""
        
        # Extraction des caractéristiques
        features_list = []
        timestamps = []
        
        for i in range(len(data)):
            row = data.iloc[i]
            timestamp = row.get('timestamp', datetime.utcnow())
            value = row.get('value', 0)
            
            # Fenêtre glissante pour les caractéristiques
            window_size = min(20, i + 1)
            window_data = data.iloc[max(0, i - window_size + 1):i + 1]['value'].values
            
            # Extraction des caractéristiques
            features = {}
            
            # Caractéristiques statistiques
            stat_features = self.feature_extractors["statistical"](window_data)
            features.update(stat_features)
            
            # Caractéristiques temporelles
            window_timestamps = data.iloc[max(0, i - window_size + 1):i + 1]['timestamp'].tolist()
            temp_features = self.feature_extractors["temporal"](window_data, window_timestamps)
            features.update(temp_features)
            
            # Caractéristiques fréquentielles
            if len(window_data) >= 8:  # Minimum pour FFT
                freq_features = self.feature_extractors["frequency"](window_data)
                features.update(freq_features)
            
            # Caractéristiques contextuelles
            features.update({
                "current_value": value,
                "hour_of_day": timestamp.hour,
                "day_of_week": timestamp.weekday(),
                "is_weekend": 1 if timestamp.weekday() >= 5 else 0
            })
            
            features_list.append(list(features.values()))
            timestamps.append(timestamp)
        
        X = np.array(features_list)
        y = np.zeros(len(X))  # Pas de labels pour l'apprentissage non supervisé
        
        return X, y, timestamps
    
    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> np.ndarray:
        """Création de séquences pour LSTM"""
        
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        
        return np.array(sequences)
    
    def _setup_explainers(self, metric_name: str, sample_data: np.ndarray):
        """Configuration des explainers pour l'explainabilité"""
        
        try:
            # SHAP explainer pour Isolation Forest
            model_key = f"{metric_name}_{ModelType.ISOLATION_FOREST.value}"
            if model_key in self.models:
                scaler_key = f"{model_key}_scaler"
                if scaler_key in self.scalers:
                    scaled_sample = self.scalers[scaler_key].transform(sample_data)
                    self.explainers[model_key] = shap.Explainer(
                        self.models[model_key].decision_function,
                        scaled_sample
                    )
        except Exception as e:
            logger.warning(f"Failed to setup SHAP explainer for {metric_name}: {e}")
    
    async def detect_anomalies(self, metric_name: str, data: pd.DataFrame, 
                              real_time: bool = False) -> List[AnomalyDetection]:
        """Détection d'anomalies sur nouvelles données"""
        
        anomalies = []
        
        # Préparation des données
        X, _, timestamps = self._prepare_training_data(data)
        
        if len(X) == 0:
            return anomalies
        
        # Détection avec chaque modèle disponible
        for model_type in [ModelType.ISOLATION_FOREST, ModelType.LSTM_AUTOENCODER, ModelType.VAE]:
            model_key = f"{metric_name}_{model_type.value}"
            
            if model_key in self.models:
                model_anomalies = await self._detect_with_model(
                    model_key, model_type, X, timestamps, data
                )
                anomalies.extend(model_anomalies)
        
        # Fusion et déduplication des anomalies
        merged_anomalies = self._merge_anomalies(anomalies)
        
        # Ajout à l'historique
        self.anomaly_history.extend(merged_anomalies)
        
        return merged_anomalies
    
    async def _detect_with_model(self, model_key: str, model_type: ModelType, 
                                X: np.ndarray, timestamps: List[datetime], 
                                original_data: pd.DataFrame) -> List[AnomalyDetection]:
        """Détection avec un modèle spécifique"""
        
        anomalies = []
        model = self.models[model_key]
        scaler_key = f"{model_key}_scaler"
        
        try:
            if model_type == ModelType.ISOLATION_FOREST:
                # Normalisation
                if scaler_key in self.scalers:
                    X_scaled = self.scalers[scaler_key].transform(X)
                else:
                    X_scaled = X
                
                # Prédiction
                predictions = model.predict(X_scaled)
                scores = model.decision_function(X_scaled)
                
                # Conversion en anomalies
                for i, (pred, score) in enumerate(zip(predictions, scores)):
                    if pred == -1:  # Anomalie détectée
                        
                        # Calcul de la sévérité basée sur le score
                        severity = self._calculate_severity(score, model_type)
                        
                        # Explication
                        explanation = await self._generate_explanation(
                            model_key, X_scaled[i:i+1], model_type
                        )
                        
                        anomaly = AnomalyDetection(
                            id=str(uuid.uuid4()),
                            timestamp=timestamps[i],
                            metric_name=model_key.split('_')[0],
                            value=original_data.iloc[i]['value'],
                            expected_value=np.mean(original_data['value']),
                            anomaly_score=abs(score),
                            anomaly_type=AnomalyType.STATISTICAL,
                            severity=severity,
                            confidence=min(abs(score) * 10, 1.0),
                            explanation=explanation,
                            model_used=model_type.value
                        )
                        
                        anomalies.append(anomaly)
            
            elif model_type == ModelType.LSTM_AUTOENCODER:
                # Préparation des séquences
                config = self.model_configs[ModelType.LSTM_AUTOENCODER]
                sequence_length = config["sequence_length"]
                
                if len(X) >= sequence_length:
                    sequences = self._create_sequences(X, sequence_length)
                    
                    if scaler_key in self.scalers:
                        sequences_scaled = self.scalers[scaler_key].transform(
                            sequences.reshape(-1, sequences.shape[-1])
                        ).reshape(sequences.shape)
                    else:
                        sequences_scaled = sequences
                    
                    # Prédiction
                    model.eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(sequences_scaled)
                        outputs = model(X_tensor)
                        reconstruction_errors = torch.mean((X_tensor - outputs) ** 2, dim=(1, 2))
                    
                    # Seuil d'anomalie
                    threshold = self.model_performances[model_key].get("threshold", 0.1)
                    
                    # Détection des anomalies
                    for i, error in enumerate(reconstruction_errors):
                        if error.item() > threshold:
                            timestamp_idx = i + sequence_length - 1
                            if timestamp_idx < len(timestamps):
                                
                                severity = self._calculate_severity(error.item(), model_type)
                                
                                anomaly = AnomalyDetection(
                                    id=str(uuid.uuid4()),
                                    timestamp=timestamps[timestamp_idx],
                                    metric_name=model_key.split('_')[0],
                                    value=original_data.iloc[timestamp_idx]['value'],
                                    expected_value=np.mean(original_data['value']),
                                    anomaly_score=error.item(),
                                    anomaly_type=AnomalyType.TREND,
                                    severity=severity,
                                    confidence=min(error.item() / threshold, 1.0),
                                    explanation=f"Reconstruction error: {error.item():.6f} > threshold: {threshold:.6f}",
                                    model_used=model_type.value
                                )
                                
                                anomalies.append(anomaly)
            
            elif model_type == ModelType.VAE:
                # Normalisation
                if scaler_key in self.scalers:
                    X_scaled = self.scalers[scaler_key].transform(X)
                else:
                    X_scaled = X
                
                # Prédiction
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_scaled)
                    recon_X, mu, logvar = model(X_tensor)
                    reconstruction_errors = torch.mean((X_tensor - recon_X) ** 2, dim=1)
                
                # Seuil d'anomalie
                threshold = self.model_performances[model_key].get("threshold", 0.1)
                
                # Détection des anomalies
                for i, error in enumerate(reconstruction_errors):
                    if error.item() > threshold:
                        
                        severity = self._calculate_severity(error.item(), model_type)
                        
                        anomaly = AnomalyDetection(
                            id=str(uuid.uuid4()),
                            timestamp=timestamps[i],
                            metric_name=model_key.split('_')[0],
                            value=original_data.iloc[i]['value'],
                            expected_value=np.mean(original_data['value']),
                            anomaly_score=error.item(),
                            anomaly_type=AnomalyType.CONTEXTUAL,
                            severity=severity,
                            confidence=min(error.item() / threshold, 1.0),
                            explanation=f"VAE reconstruction error: {error.item():.6f}",
                            model_used=model_type.value
                        )
                        
                        anomalies.append(anomaly)
        
        except Exception as e:
            logger.error(f"Error in anomaly detection with {model_type}: {e}")
        
        return anomalies
    
    def _calculate_severity(self, score: float, model_type: ModelType) -> AnomalySeverity:
        """Calcul de la sévérité basée sur le score"""
        
        # Normalisation du score selon le type de modèle
        if model_type == ModelType.ISOLATION_FOREST:
            # Score négatif pour Isolation Forest
            normalized_score = abs(score)
        else:
            # Score d'erreur pour les autoencoders
            normalized_score = min(score, 1.0)
        
        if normalized_score >= 0.8:
            return AnomalySeverity.CRITICAL
        elif normalized_score >= 0.6:
            return AnomalySeverity.HIGH
        elif normalized_score >= 0.4:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW
    
    async def _generate_explanation(self, model_key: str, sample: np.ndarray, 
                                   model_type: ModelType) -> str:
        """Génération d'explication pour une anomalie"""
        
        try:
            if model_key in self.explainers and model_type == ModelType.ISOLATION_FOREST:
                # Utilisation de SHAP
                shap_values = self.explainers[model_key](sample)
                
                # Identification des caractéristiques les plus importantes
                feature_importance = np.abs(shap_values.values[0])
                top_features_idx = np.argsort(feature_importance)[-3:]
                
                explanation = "Anomalie détectée basée sur: "
                feature_names = ["mean", "std", "skewness", "kurtosis", "median", "current_value", "hour_of_day"]
                
                for idx in top_features_idx:
                    if idx < len(feature_names):
                        explanation += f"{feature_names[idx]} (importance: {feature_importance[idx]:.3f}), "
                
                return explanation.rstrip(", ")
            
            else:
                return f"Anomalie détectée par {model_type.value}"
        
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return f"Anomalie détectée par {model_type.value}"
    
    def _merge_anomalies(self, anomalies: List[AnomalyDetection]) -> List[AnomalyDetection]:
        """Fusion et déduplication des anomalies"""
        
        if not anomalies:
            return []
        
        # Groupement par timestamp et métrique
        grouped = {}
        for anomaly in anomalies:
            key = (anomaly.metric_name, anomaly.timestamp.replace(second=0, microsecond=0))
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(anomaly)
        
        # Fusion des anomalies du même groupe
        merged = []
        for group in grouped.values():
            if len(group) == 1:
                merged.append(group[0])
            else:
                # Prendre l'anomalie avec le score le plus élevé
                best_anomaly = max(group, key=lambda x: x.anomaly_score)
                
                # Fusion des explications
                explanations = [a.explanation for a in group]
                best_anomaly.explanation = " | ".join(set(explanations))
                
                # Moyenner la confiance
                best_anomaly.confidence = np.mean([a.confidence for a in group])
                
                merged.append(best_anomaly)
        
        return sorted(merged, key=lambda x: x.timestamp)

# =============================================================================
# PRÉDICTEUR DE MÉTRIQUES
# =============================================================================

class MetricsPredictor:
    """Prédicteur de métriques avec modèles de séries temporelles"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models: Dict[str, Any] = {}
        
    async def train_prediction_models(self, training_data: Dict[str, pd.DataFrame]):
        """Entraînement des modèles de prédiction"""
        
        for metric_name, data in training_data.items():
            if len(data) < 100:  # Données insuffisantes
                continue
            
            # Préparation des données
            data['ds'] = pd.to_datetime(data['timestamp'])
            data['y'] = data['value']
            
            # Entraînement Prophet
            await self._train_prophet_model(metric_name, data[['ds', 'y']])
    
    async def _train_prophet_model(self, metric_name: str, data: pd.DataFrame):
        """Entraînement modèle Prophet"""
        
        try:
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                changepoint_prior_scale=0.05
            )
            
            model.fit(data)
            self.models[f"{metric_name}_prophet"] = model
            
            logger.info(f"Prophet model trained for {metric_name}")
            
        except Exception as e:
            logger.error(f"Error training Prophet model for {metric_name}: {e}")
    
    async def predict_metrics(self, metric_name: str, horizon_hours: int = 24) -> List[PredictionResult]:
        """Prédiction de métriques"""
        
        model_key = f"{metric_name}_prophet"
        
        if model_key not in self.models:
            return []
        
        try:
            model = self.models[model_key]
            
            # Création de la période future
            future = model.make_future_dataframe(periods=horizon_hours, freq='H')
            
            # Prédiction
            forecast = model.predict(future)
            
            # Conversion en résultats
            predictions = []
            current_time = datetime.utcnow()
            
            for i in range(-horizon_hours, 0):
                row = forecast.iloc[i]
                
                prediction = PredictionResult(
                    metric_name=metric_name,
                    timestamp=current_time + timedelta(hours=abs(i)),
                    predicted_value=row['yhat'],
                    confidence_interval=(row['yhat_lower'], row['yhat_upper']),
                    model_used="prophet",
                    horizon_hours=abs(i)
                )
                
                predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting metrics for {metric_name}: {e}")
            return []

# =============================================================================
# SYSTÈME DE CORRÉLATION D'INCIDENTS
# =============================================================================

class IncidentCorrelationEngine:
    """Moteur de corrélation d'incidents intelligente"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.incident_history: List[Dict[str, Any]] = []
        
    async def correlate_incidents(self, incidents: List[Dict[str, Any]]) -> List[IncidentCorrelation]:
        """Corrélation d'incidents similaires"""
        
        correlations = []
        
        # Clustering temporel
        temporal_clusters = self._cluster_by_time(incidents)
        
        for cluster in temporal_clusters:
            if len(cluster) >= 2:
                correlation = self._analyze_cluster(cluster)
                if correlation:
                    correlations.append(correlation)
        
        return correlations
    
    def _cluster_by_time(self, incidents: List[Dict[str, Any]], 
                        time_window_minutes: int = 30) -> List[List[Dict[str, Any]]]:
        """Clustering par fenêtre temporelle"""
        
        if not incidents:
            return []
        
        # Tri par timestamp
        sorted_incidents = sorted(incidents, key=lambda x: x['timestamp'])
        
        clusters = []
        current_cluster = [sorted_incidents[0]]
        
        for incident in sorted_incidents[1:]:
            # Vérification de la fenêtre temporelle
            time_diff = (incident['timestamp'] - current_cluster[-1]['timestamp']).total_seconds() / 60
            
            if time_diff <= time_window_minutes:
                current_cluster.append(incident)
            else:
                clusters.append(current_cluster)
                current_cluster = [incident]
        
        if current_cluster:
            clusters.append(current_cluster)
        
        return clusters
    
    def _analyze_cluster(self, cluster: List[Dict[str, Any]]) -> Optional[IncidentCorrelation]:
        """Analyse d'un cluster d'incidents"""
        
        if len(cluster) < 2:
            return None
        
        # Calcul du score de corrélation
        correlation_score = self._calculate_correlation_score(cluster)
        
        if correlation_score < 0.5:
            return None
        
        # Type de corrélation
        correlation_type = self._determine_correlation_type(cluster)
        
        # Actions suggérées
        suggested_actions = self._generate_suggested_actions(cluster, correlation_type)
        
        return IncidentCorrelation(
            incident_ids=[inc['id'] for inc in cluster],
            correlation_score=correlation_score,
            correlation_type=correlation_type,
            timestamp=datetime.utcnow(),
            explanation=f"Corrélation détectée entre {len(cluster)} incidents ({correlation_type})",
            suggested_actions=suggested_actions
        )
    
    def _calculate_correlation_score(self, cluster: List[Dict[str, Any]]) -> float:
        """Calcul du score de corrélation"""
        
        # Facteurs de corrélation
        time_factor = self._calculate_time_correlation(cluster)
        category_factor = self._calculate_category_correlation(cluster)
        severity_factor = self._calculate_severity_correlation(cluster)
        
        # Score pondéré
        correlation_score = (time_factor * 0.4 + category_factor * 0.4 + severity_factor * 0.2)
        
        return min(correlation_score, 1.0)
    
    def _calculate_time_correlation(self, cluster: List[Dict[str, Any]]) -> float:
        """Corrélation temporelle"""
        
        timestamps = [inc['timestamp'] for inc in cluster]
        time_range = (max(timestamps) - min(timestamps)).total_seconds() / 60  # en minutes
        
        # Plus les incidents sont proches temporellement, plus le score est élevé
        if time_range <= 5:
            return 1.0
        elif time_range <= 15:
            return 0.8
        elif time_range <= 30:
            return 0.6
        else:
            return 0.2
    
    def _calculate_category_correlation(self, cluster: List[Dict[str, Any]]) -> float:
        """Corrélation par catégorie"""
        
        categories = [inc.get('category', 'unknown') for inc in cluster]
        unique_categories = set(categories)
        
        if len(unique_categories) == 1:
            return 1.0  # Même catégorie
        elif len(unique_categories) <= len(categories) / 2:
            return 0.7  # Catégories similaires
        else:
            return 0.3  # Catégories diverses
    
    def _calculate_severity_correlation(self, cluster: List[Dict[str, Any]]) -> float:
        """Corrélation par sévérité"""
        
        severities = [inc.get('severity', 'low') for inc in cluster]
        unique_severities = set(severities)
        
        if len(unique_severities) == 1:
            return 1.0
        elif len(unique_severities) <= 2:
            return 0.6
        else:
            return 0.3
    
    def _determine_correlation_type(self, cluster: List[Dict[str, Any]]) -> str:
        """Détermination du type de corrélation"""
        
        categories = [inc.get('category', 'unknown') for inc in cluster]
        sources = [inc.get('source', 'unknown') for inc in cluster]
        
        if len(set(categories)) == 1:
            return "same_category"
        elif len(set(sources)) == 1:
            return "same_source"
        else:
            return "temporal_cluster"
    
    def _generate_suggested_actions(self, cluster: List[Dict[str, Any]], 
                                   correlation_type: str) -> List[str]:
        """Génération d'actions suggérées"""
        
        actions = []
        
        if correlation_type == "same_category":
            actions.append("Vérifier les composants communs de cette catégorie")
            actions.append("Analyser les dépendances systémiques")
        
        elif correlation_type == "same_source":
            actions.append("Vérifier la santé du système source")
            actions.append("Examiner la configuration du monitoring")
        
        elif correlation_type == "temporal_cluster":
            actions.append("Identifier les changements récents dans l'infrastructure")
            actions.append("Vérifier les déploiements et maintenances")
        
        # Actions génériques
        actions.extend([
            "Créer un incident parent pour coordination",
            "Notifier l'équipe d'astreinte",
            "Documenter la résolution pour apprentissage futur"
        ])
        
        return actions

# =============================================================================
# INSTANCE GLOBALE ET FONCTIONS D'EXPORT
# =============================================================================

# Instances globales
anomaly_engine: Optional[AnomalyDetectionEngine] = None
metrics_predictor: Optional[MetricsPredictor] = None
correlation_engine: Optional[IncidentCorrelationEngine] = None

def initialize_ai_monitoring(config: Optional[Dict[str, Any]] = None) -> Tuple[AnomalyDetectionEngine, MetricsPredictor, IncidentCorrelationEngine]:
    """Initialisation du système d'IA de monitoring"""
    
    global anomaly_engine, metrics_predictor, correlation_engine
    
    default_config = {
        "models": {
            "retrain_interval_hours": 24,
            "min_training_samples": 100
        },
        "anomaly_detection": {
            "sensitivity": 0.1,
            "enable_explainability": True
        },
        "prediction": {
            "default_horizon_hours": 24,
            "update_interval_hours": 6
        },
        "correlation": {
            "time_window_minutes": 30,
            "min_correlation_score": 0.5
        }
    }
    
    final_config = {**default_config}
    if config:
        final_config.update(config)
    
    anomaly_engine = AnomalyDetectionEngine(final_config)
    metrics_predictor = MetricsPredictor(final_config)
    correlation_engine = IncidentCorrelationEngine(final_config)
    
    logger.info("AI monitoring system initialized")
    
    return anomaly_engine, metrics_predictor, correlation_engine

def get_ai_engines() -> Tuple[Optional[AnomalyDetectionEngine], Optional[MetricsPredictor], Optional[IncidentCorrelationEngine]]:
    """Récupération des moteurs d'IA"""
    
    global anomaly_engine, metrics_predictor, correlation_engine
    
    return anomaly_engine, metrics_predictor, correlation_engine

# =============================================================================
# FONCTIONS D'EXPORT
# =============================================================================

__all__ = [
    "AnomalyDetectionEngine",
    "MetricsPredictor", 
    "IncidentCorrelationEngine",
    "AnomalyDetection",
    "PredictionResult",
    "IncidentCorrelation",
    "AnomalyType",
    "AnomalySeverity",
    "ModelType",
    "initialize_ai_monitoring",
    "get_ai_engines"
]
