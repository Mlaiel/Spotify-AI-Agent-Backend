"""
Détecteurs d'Anomalies Basés sur Machine Learning - Module Avancé
================================================================

Auteur: Fahed Mlaiel
Rôles: Lead Dev + Architecte IA, Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)

Ce module implémente des détecteurs d'anomalies sophistiqués utilisant
des algorithmes de machine learning pour la détection en temps réel.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from typing import Dict, List, Optional, Tuple, Any
import logging
import asyncio
from datetime import datetime, timedelta
import json
import redis
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AnomalyResult:
    """Résultat de détection d'anomalie"""
    is_anomaly: bool
    confidence_score: float
    anomaly_type: str
    timestamp: datetime
    features: Dict[str, Any]
    recommendation: str
    severity: str
    context: Dict[str, Any]

class AutoEncoderDetector(nn.Module):
    """Détecteur d'anomalies basé sur AutoEncoder"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super().__init__()
        self.input_dim = input_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        for i in range(len(hidden_dims) - 1, -1, -1):
            if i == 0:
                decoder_layers.extend([
                    nn.Linear(prev_dim, input_dim),
                    nn.Sigmoid()
                ])
            else:
                decoder_layers.extend([
                    nn.Linear(prev_dim, hidden_dims[i-1]),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ])
                prev_dim = hidden_dims[i-1]
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_reconstruction_error(self, x):
        """Calcule l'erreur de reconstruction"""
        with torch.no_grad():
            reconstructed = self.forward(x)
            mse = torch.mean((x - reconstructed) ** 2, dim=1)
            return mse.numpy()

class LSTMAnomalyDetector(nn.Module):
    """Détecteur d'anomalies LSTM pour séries temporelles"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, sequence_length: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        output = self.fc(self.dropout(lstm_out[:, -1, :]))
        return output

class MLAnomalyDetector:
    """Détecteur d'anomalies ML avancé avec ensemble de modèles"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.thresholds = {}
        self.feature_importance = {}
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        
        # Initialisation des modèles
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialise les modèles de détection"""
        
        # Isolation Forest pour détection d'outliers
        self.models['isolation_forest'] = IsolationForest(
            contamination=self.config.get('contamination', 0.1),
            random_state=42,
            n_estimators=100
        )
        
        # One-Class SVM pour données non-linéaires
        self.models['one_class_svm'] = OneClassSVM(
            nu=self.config.get('nu', 0.1),
            gamma='scale'
        )
        
        # DBSCAN pour clustering d'anomalies
        self.models['dbscan'] = DBSCAN(
            eps=self.config.get('eps', 0.5),
            min_samples=self.config.get('min_samples', 5)
        )
        
        # AutoEncoder pour anomalies complexes
        if self.config.get('use_autoencoder', True):
            input_dim = self.config.get('input_dim', 10)
            hidden_dims = self.config.get('hidden_dims', [8, 4, 2])
            self.models['autoencoder'] = AutoEncoderDetector(input_dim, hidden_dims)
        
        # LSTM pour séries temporelles
        if self.config.get('use_lstm', True):
            self.models['lstm'] = LSTMAnomalyDetector(
                input_size=self.config.get('input_size', 1),
                hidden_size=self.config.get('hidden_size', 50),
                num_layers=self.config.get('num_layers', 2),
                sequence_length=self.config.get('sequence_length', 10)
            )
        
        # Scalers pour normalisation
        self.scalers['standard'] = StandardScaler()
        self.scalers['minmax'] = MinMaxScaler()
        
        logger.info("Modèles ML initialisés avec succès")
    
    async def detect_anomalies(self, data: np.ndarray, feature_names: List[str] = None) -> List[AnomalyResult]:
        """Détecte les anomalies dans les données"""
        results = []
        
        try:
            # Préprocessing des données
            scaled_data = self._preprocess_data(data)
            
            # Détection avec chaque modèle
            for model_name, model in self.models.items():
                if model_name in ['autoencoder', 'lstm']:
                    continue  # Traités séparément
                
                anomaly_scores = await self._detect_with_model(model_name, scaled_data)
                
                for i, score in enumerate(anomaly_scores):
                    if self._is_anomaly(model_name, score):
                        result = AnomalyResult(
                            is_anomaly=True,
                            confidence_score=abs(score),
                            anomaly_type=model_name,
                            timestamp=datetime.now(),
                            features=self._extract_features(data[i], feature_names),
                            recommendation=self._generate_recommendation(model_name, score),
                            severity=self._calculate_severity(score),
                            context={'model': model_name, 'threshold': self.thresholds.get(model_name, 0)}
                        )
                        results.append(result)
            
            # Détection avec AutoEncoder si disponible
            if 'autoencoder' in self.models:
                autoencoder_results = await self._detect_with_autoencoder(data)
                results.extend(autoencoder_results)
            
            # Consensus et filtrage
            final_results = self._apply_ensemble_consensus(results)
            
            # Cache des résultats
            await self._cache_results(final_results)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Erreur lors de la détection d'anomalies: {e}")
            return []
    
    def _preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """Préprocesse les données"""
        # Normalisation standard
        if not hasattr(self.scalers['standard'], 'mean_'):
            return self.scalers['standard'].fit_transform(data)
        return self.scalers['standard'].transform(data)
    
    async def _detect_with_model(self, model_name: str, data: np.ndarray) -> np.ndarray:
        """Détecte avec un modèle spécifique"""
        model = self.models[model_name]
        
        if model_name == 'isolation_forest':
            if not hasattr(model, 'estimators_'):
                model.fit(data)
            scores = model.decision_function(data)
            
        elif model_name == 'one_class_svm':
            if not hasattr(model, 'support_'):
                model.fit(data)
            scores = model.decision_function(data)
            
        elif model_name == 'dbscan':
            labels = model.fit_predict(data)
            scores = np.array([1.0 if label == -1 else 0.0 for label in labels])
        
        else:
            scores = np.zeros(len(data))
        
        return scores
    
    async def _detect_with_autoencoder(self, data: np.ndarray) -> List[AnomalyResult]:
        """Détection avec AutoEncoder"""
        results = []
        
        try:
            autoencoder = self.models['autoencoder']
            tensor_data = torch.FloatTensor(data)
            
            reconstruction_errors = autoencoder.get_reconstruction_error(tensor_data)
            threshold = np.percentile(reconstruction_errors, 95)
            
            for i, error in enumerate(reconstruction_errors):
                if error > threshold:
                    result = AnomalyResult(
                        is_anomaly=True,
                        confidence_score=float(error),
                        anomaly_type='autoencoder',
                        timestamp=datetime.now(),
                        features=self._extract_features(data[i]),
                        recommendation=f"Erreur de reconstruction élevée: {error:.4f}",
                        severity=self._calculate_severity(error, threshold),
                        context={'reconstruction_error': float(error), 'threshold': threshold}
                    )
                    results.append(result)
        
        except Exception as e:
            logger.error(f"Erreur AutoEncoder: {e}")
        
        return results
    
    def _is_anomaly(self, model_name: str, score: float) -> bool:
        """Détermine si un score indique une anomalie"""
        threshold = self.thresholds.get(model_name, -0.1)
        
        if model_name in ['isolation_forest', 'one_class_svm']:
            return score < threshold
        elif model_name == 'dbscan':
            return score == 1.0
        
        return False
    
    def _extract_features(self, data_point: np.ndarray, feature_names: List[str] = None) -> Dict[str, Any]:
        """Extrait les caractéristiques d'un point de données"""
        features = {}
        
        if feature_names:
            for i, name in enumerate(feature_names):
                if i < len(data_point):
                    features[name] = float(data_point[i])
        else:
            for i, value in enumerate(data_point):
                features[f'feature_{i}'] = float(value)
        
        return features
    
    def _generate_recommendation(self, model_name: str, score: float) -> str:
        """Génère une recommandation basée sur le type d'anomalie"""
        recommendations = {
            'isolation_forest': "Vérifiez les métriques système et les patterns de données",
            'one_class_svm': "Analysez les corrélations multi-dimensionnelles",
            'dbscan': "Examinez les groupes d'utilisateurs ou de contenus similaires",
            'autoencoder': "Inspectez les patterns complexes dans les données"
        }
        
        base_rec = recommendations.get(model_name, "Analysez les données en détail")
        severity_level = abs(score)
        
        if severity_level > 1.0:
            return f"URGENT: {base_rec} - Score élevé détecté"
        elif severity_level > 0.5:
            return f"ATTENTION: {base_rec} - Anomalie modérée"
        else:
            return f"INFO: {base_rec} - Anomalie mineure"
    
    def _calculate_severity(self, score: float, threshold: float = None) -> str:
        """Calcule la sévérité de l'anomalie"""
        abs_score = abs(score)
        
        if threshold:
            ratio = abs_score / threshold
            if ratio > 3:
                return "critical"
            elif ratio > 2:
                return "high"
            elif ratio > 1.5:
                return "medium"
            else:
                return "low"
        
        if abs_score > 2.0:
            return "critical"
        elif abs_score > 1.0:
            return "high"
        elif abs_score > 0.5:
            return "medium"
        else:
            return "low"
    
    def _apply_ensemble_consensus(self, results: List[AnomalyResult]) -> List[AnomalyResult]:
        """Applique un consensus d'ensemble pour filtrer les faux positifs"""
        if len(results) <= 1:
            return results
        
        # Grouper par timestamp/data point
        grouped = {}
        for result in results:
            key = result.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)
        
        consensus_results = []
        for timestamp, group in grouped.items():
            if len(group) >= 2:  # Consensus de 2+ modèles
                # Prendre le résultat avec le score de confiance le plus élevé
                best_result = max(group, key=lambda r: r.confidence_score)
                best_result.context['consensus_count'] = len(group)
                consensus_results.append(best_result)
        
        return consensus_results
    
    async def _cache_results(self, results: List[AnomalyResult]):
        """Cache les résultats dans Redis"""
        try:
            for result in results:
                cache_key = f"anomaly:{result.timestamp.isoformat()}:{result.anomaly_type}"
                cache_data = {
                    'is_anomaly': result.is_anomaly,
                    'confidence_score': result.confidence_score,
                    'anomaly_type': result.anomaly_type,
                    'severity': result.severity,
                    'recommendation': result.recommendation
                }
                
                await asyncio.get_event_loop().run_in_executor(
                    None, 
                    self.redis_client.setex,
                    cache_key, 
                    3600,  # 1 heure
                    json.dumps(cache_data)
                )
        
        except Exception as e:
            logger.warning(f"Erreur mise en cache: {e}")
    
    def update_thresholds(self, model_name: str, threshold: float):
        """Met à jour les seuils de détection"""
        self.thresholds[model_name] = threshold
        logger.info(f"Seuil mis à jour pour {model_name}: {threshold}")
    
    def get_feature_importance(self, model_name: str) -> Dict[str, float]:
        """Retourne l'importance des features pour un modèle"""
        return self.feature_importance.get(model_name, {})

# Factory pour créer des détecteurs
class DetectorFactory:
    """Factory pour créer des détecteurs ML"""
    
    @staticmethod
    def create_music_anomaly_detector() -> MLAnomalyDetector:
        """Créé un détecteur spécialisé pour les données musicales"""
        config = {
            'contamination': 0.05,
            'nu': 0.05,
            'eps': 0.3,
            'min_samples': 3,
            'use_autoencoder': True,
            'use_lstm': True,
            'input_dim': 15,  # Features audio
            'hidden_dims': [12, 8, 4],
            'input_size': 1,
            'hidden_size': 64,
            'num_layers': 3,
            'sequence_length': 30
        }
        return MLAnomalyDetector(config)
    
    @staticmethod
    def create_user_behavior_detector() -> MLAnomalyDetector:
        """Créé un détecteur pour le comportement utilisateur"""
        config = {
            'contamination': 0.1,
            'nu': 0.1,
            'eps': 0.5,
            'min_samples': 5,
            'use_autoencoder': True,
            'use_lstm': False,
            'input_dim': 20,  # Features comportementales
            'hidden_dims': [16, 10, 6]
        }
        return MLAnomalyDetector(config)
    
    @staticmethod
    def create_performance_detector() -> MLAnomalyDetector:
        """Créé un détecteur pour les performances système"""
        config = {
            'contamination': 0.15,
            'nu': 0.15,
            'eps': 0.4,
            'min_samples': 4,
            'use_autoencoder': True,
            'use_lstm': True,
            'input_dim': 10,  # Métriques système
            'hidden_dims': [8, 5, 3],
            'sequence_length': 20
        }
        return MLAnomalyDetector(config)
