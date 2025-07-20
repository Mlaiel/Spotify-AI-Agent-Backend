"""
Algorithmes avancés de détection d'anomalies pour le monitoring intelligent.

Ce module implémente plusieurs approches de détection d'anomalies :
- Isolation Forest pour la détection d'outliers
- One-Class SVM pour la classification binaire
- LSTM Autoencodeurs pour les séries temporelles
- Ensemble methods pour combiner les approches

Optimisé pour la production avec support temps réel et haute performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras import layers, Model
import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pickle
import json
import redis
from prometheus_client import Counter, Histogram, Gauge

# Métriques Prometheus pour monitoring
ANOMALY_DETECTION_COUNTER = Counter('anomaly_detections_total', 'Total anomalies detected', ['algorithm', 'severity'])
DETECTION_LATENCY = Histogram('anomaly_detection_duration_seconds', 'Time spent detecting anomalies', ['algorithm'])
ACTIVE_ANOMALIES = Gauge('active_anomalies_count', 'Current number of active anomalies', ['type'])

logger = logging.getLogger(__name__)

@dataclass
class AnomalyResult:
    """Résultat de détection d'anomalie."""
    timestamp: datetime
    metric_name: str
    value: float
    anomaly_score: float
    is_anomaly: bool
    confidence: float
    algorithm_used: str
    severity: str = field(default="medium")
    context: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None

class BaseAnomalyDetector(ABC):
    """Classe de base pour tous les détecteurs d'anomalies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """Entraîne le modèle sur les données."""
        pass
    
    @abstractmethod
    def predict(self, data: np.ndarray) -> List[AnomalyResult]:
        """Prédit les anomalies dans les données."""
        pass
    
    def preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """Prétraite les données avant détection."""
        return self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
    
    def save_model(self, path: str) -> None:
        """Sauvegarde le modèle entraîné."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'is_trained': self.is_trained
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
            
    def load_model(self, path: str) -> None:
        """Charge un modèle pré-entraîné."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.config = model_data['config']
            self.is_trained = model_data['is_trained']

class IsolationForestDetector(BaseAnomalyDetector):
    """Détecteur d'anomalies basé sur Isolation Forest."""
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            'contamination': 0.1,
            'n_estimators': 100,
            'max_samples': 'auto',
            'random_state': 42,
            'n_jobs': -1
        }
        config = config or default_config
        super().__init__(config)
        
    def fit(self, data: np.ndarray) -> None:
        """Entraîne l'Isolation Forest."""
        with DETECTION_LATENCY.labels(algorithm='isolation_forest').time():
            try:
                processed_data = self.preprocess_data(data)
                
                self.model = IsolationForest(
                    contamination=self.config['contamination'],
                    n_estimators=self.config['n_estimators'],
                    max_samples=self.config['max_samples'],
                    random_state=self.config['random_state'],
                    n_jobs=self.config['n_jobs']
                )
                
                self.model.fit(processed_data.reshape(-1, 1))
                self.is_trained = True
                
                logger.info(f"Isolation Forest trained on {len(data)} samples")
                
            except Exception as e:
                logger.error(f"Error training Isolation Forest: {e}")
                raise
    
    def predict(self, data: np.ndarray, metric_names: List[str] = None) -> List[AnomalyResult]:
        """Détecte les anomalies avec Isolation Forest."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        results = []
        
        with DETECTION_LATENCY.labels(algorithm='isolation_forest').time():
            try:
                processed_data = self.scaler.transform(data.reshape(-1, 1))
                
                # Prédictions : -1 pour anomalie, 1 pour normal
                predictions = self.model.predict(processed_data)
                anomaly_scores = self.model.decision_function(processed_data)
                
                for i, (pred, score) in enumerate(zip(predictions, anomaly_scores)):
                    is_anomaly = pred == -1
                    confidence = abs(score)
                    
                    # Détermination de la sévérité basée sur le score
                    if confidence > 0.5:
                        severity = "critical"
                    elif confidence > 0.3:
                        severity = "high"
                    elif confidence > 0.1:
                        severity = "medium"
                    else:
                        severity = "low"
                    
                    result = AnomalyResult(
                        timestamp=datetime.now(),
                        metric_name=metric_names[i] if metric_names else f"metric_{i}",
                        value=data[i],
                        anomaly_score=float(score),
                        is_anomaly=is_anomaly,
                        confidence=confidence,
                        algorithm_used="isolation_forest",
                        severity=severity,
                        context={"model_params": self.config}
                    )
                    
                    results.append(result)
                    
                    if is_anomaly:
                        ANOMALY_DETECTION_COUNTER.labels(
                            algorithm='isolation_forest', 
                            severity=severity
                        ).inc()
                
                logger.info(f"Processed {len(data)} points, found {sum(r.is_anomaly for r in results)} anomalies")
                
            except Exception as e:
                logger.error(f"Error in Isolation Forest prediction: {e}")
                raise
                
        return results

class OneClassSVMDetector(BaseAnomalyDetector):
    """Détecteur d'anomalies basé sur One-Class SVM."""
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            'kernel': 'rbf',
            'gamma': 'scale',
            'nu': 0.1,
            'cache_size': 200
        }
        config = config or default_config
        super().__init__(config)
        
    def fit(self, data: np.ndarray) -> None:
        """Entraîne le One-Class SVM."""
        with DETECTION_LATENCY.labels(algorithm='one_class_svm').time():
            try:
                processed_data = self.preprocess_data(data)
                
                self.model = OneClassSVM(
                    kernel=self.config['kernel'],
                    gamma=self.config['gamma'],
                    nu=self.config['nu'],
                    cache_size=self.config['cache_size']
                )
                
                self.model.fit(processed_data.reshape(-1, 1))
                self.is_trained = True
                
                logger.info(f"One-Class SVM trained on {len(data)} samples")
                
            except Exception as e:
                logger.error(f"Error training One-Class SVM: {e}")
                raise
    
    def predict(self, data: np.ndarray, metric_names: List[str] = None) -> List[AnomalyResult]:
        """Détecte les anomalies avec One-Class SVM."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        results = []
        
        with DETECTION_LATENCY.labels(algorithm='one_class_svm').time():
            try:
                processed_data = self.scaler.transform(data.reshape(-1, 1))
                
                predictions = self.model.predict(processed_data)
                decision_scores = self.model.decision_function(processed_data)
                
                for i, (pred, score) in enumerate(zip(predictions, decision_scores)):
                    is_anomaly = pred == -1
                    confidence = abs(score)
                    
                    # Sévérité basée sur la distance à la frontière de décision
                    if confidence > 1.0:
                        severity = "critical"
                    elif confidence > 0.5:
                        severity = "high"
                    elif confidence > 0.2:
                        severity = "medium"
                    else:
                        severity = "low"
                    
                    result = AnomalyResult(
                        timestamp=datetime.now(),
                        metric_name=metric_names[i] if metric_names else f"metric_{i}",
                        value=data[i],
                        anomaly_score=float(score),
                        is_anomaly=is_anomaly,
                        confidence=confidence,
                        algorithm_used="one_class_svm",
                        severity=severity,
                        context={"model_params": self.config}
                    )
                    
                    results.append(result)
                    
                    if is_anomaly:
                        ANOMALY_DETECTION_COUNTER.labels(
                            algorithm='one_class_svm', 
                            severity=severity
                        ).inc()
                
            except Exception as e:
                logger.error(f"Error in One-Class SVM prediction: {e}")
                raise
                
        return results

class LSTMAutoencoder:
    """Autoencodeur LSTM pour la détection d'anomalies dans les séries temporelles."""
    
    def __init__(self, sequence_length: int = 10, encoding_dim: int = 32):
        self.sequence_length = sequence_length
        self.encoding_dim = encoding_dim
        self.model = None
        self.scaler = MinMaxScaler()
        
    def build_model(self, input_dim: int) -> Model:
        """Construit l'architecture de l'autoencodeur LSTM."""
        # Encodeur
        input_layer = layers.Input(shape=(self.sequence_length, input_dim))
        encoded = layers.LSTM(self.encoding_dim, return_sequences=False)(input_layer)
        encoded = layers.RepeatVector(self.sequence_length)(encoded)
        
        # Décodeur
        decoded = layers.LSTM(self.encoding_dim, return_sequences=True)(encoded)
        decoded = layers.TimeDistributed(layers.Dense(input_dim))(decoded)
        
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    
    def prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prépare les séquences pour l'entraînement LSTM."""
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        sequences = []
        for i in range(len(scaled_data) - self.sequence_length + 1):
            sequences.append(scaled_data[i:i + self.sequence_length])
            
        sequences = np.array(sequences)
        return sequences, sequences  # Autoencoder utilise même entrée/sortie

class LSTMAnomAlyDetector(BaseAnomalyDetector):
    """Détecteur d'anomalies basé sur autoencodeur LSTM."""
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            'sequence_length': 10,
            'encoding_dim': 32,
            'epochs': 50,
            'batch_size': 32,
            'threshold_percentile': 95
        }
        config = config or default_config
        super().__init__(config)
        self.autoencoder = LSTMAutoencoder(
            sequence_length=config['sequence_length'],
            encoding_dim=config['encoding_dim']
        )
        self.threshold = None
        
    def fit(self, data: np.ndarray) -> None:
        """Entraîne l'autoencodeur LSTM."""
        with DETECTION_LATENCY.labels(algorithm='lstm_autoencoder').time():
            try:
                sequences, targets = self.autoencoder.prepare_sequences(data)
                
                self.autoencoder.model = self.autoencoder.build_model(input_dim=1)
                
                # Entraînement
                history = self.autoencoder.model.fit(
                    sequences, targets,
                    epochs=self.config['epochs'],
                    batch_size=self.config['batch_size'],
                    validation_split=0.2,
                    verbose=0
                )
                
                # Calcul du seuil d'anomalie
                train_predictions = self.autoencoder.model.predict(sequences)
                mse = np.mean(np.power(sequences - train_predictions, 2), axis=(1, 2))
                self.threshold = np.percentile(mse, self.config['threshold_percentile'])
                
                self.is_trained = True
                logger.info(f"LSTM Autoencoder trained with threshold: {self.threshold}")
                
            except Exception as e:
                logger.error(f"Error training LSTM Autoencoder: {e}")
                raise
    
    def predict(self, data: np.ndarray, metric_names: List[str] = None) -> List[AnomalyResult]:
        """Détecte les anomalies avec l'autoencodeur LSTM."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        results = []
        
        with DETECTION_LATENCY.labels(algorithm='lstm_autoencoder').time():
            try:
                sequences, _ = self.autoencoder.prepare_sequences(data)
                
                if len(sequences) == 0:
                    return results
                
                predictions = self.autoencoder.model.predict(sequences)
                mse_scores = np.mean(np.power(sequences - predictions, 2), axis=(1, 2))
                
                for i, mse in enumerate(mse_scores):
                    is_anomaly = mse > self.threshold
                    confidence = min(mse / self.threshold, 3.0)  # Cap à 3x le seuil
                    
                    # Sévérité basée sur l'erreur de reconstruction
                    if mse > self.threshold * 2.5:
                        severity = "critical"
                    elif mse > self.threshold * 1.5:
                        severity = "high"
                    elif mse > self.threshold:
                        severity = "medium"
                    else:
                        severity = "low"
                    
                    # Index original dans les données
                    original_idx = i + self.config['sequence_length'] - 1
                    
                    result = AnomalyResult(
                        timestamp=datetime.now(),
                        metric_name=metric_names[original_idx] if metric_names else f"metric_{original_idx}",
                        value=data[original_idx],
                        anomaly_score=float(mse),
                        is_anomaly=is_anomaly,
                        confidence=confidence,
                        algorithm_used="lstm_autoencoder",
                        severity=severity,
                        context={
                            "threshold": self.threshold,
                            "reconstruction_error": float(mse),
                            "sequence_length": self.config['sequence_length']
                        }
                    )
                    
                    results.append(result)
                    
                    if is_anomaly:
                        ANOMALY_DETECTION_COUNTER.labels(
                            algorithm='lstm_autoencoder', 
                            severity=severity
                        ).inc()
                
            except Exception as e:
                logger.error(f"Error in LSTM Autoencoder prediction: {e}")
                raise
                
        return results

class EnsembleAnomalyDetector:
    """Détecteur d'ensemble combinant plusieurs algorithmes."""
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            'voting_strategy': 'weighted',  # 'majority', 'weighted', 'unanimous'
            'weights': {
                'isolation_forest': 0.4,
                'one_class_svm': 0.3,
                'lstm_autoencoder': 0.3
            },
            'confidence_threshold': 0.6
        }
        self.config = config or default_config
        
        # Initialisation des détecteurs
        self.detectors = {
            'isolation_forest': IsolationForestDetector(),
            'one_class_svm': OneClassSVMDetector(),
            'lstm_autoencoder': LSTMAnomAlyDetector()
        }
        
        self.is_trained = False
        
    def fit(self, data: np.ndarray) -> None:
        """Entraîne tous les détecteurs de l'ensemble."""
        try:
            logger.info("Training ensemble detectors...")
            
            for name, detector in self.detectors.items():
                logger.info(f"Training {name}...")
                detector.fit(data)
                
            self.is_trained = True
            logger.info("Ensemble training completed")
            
        except Exception as e:
            logger.error(f"Error training ensemble: {e}")
            raise
    
    def predict(self, data: np.ndarray, metric_names: List[str] = None) -> List[AnomalyResult]:
        """Prédit les anomalies en combinant les résultats des détecteurs."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before prediction")
            
        try:
            # Collecte des prédictions de tous les détecteurs
            all_predictions = {}
            
            for name, detector in self.detectors.items():
                try:
                    predictions = detector.predict(data, metric_names)
                    all_predictions[name] = predictions
                except Exception as e:
                    logger.warning(f"Detector {name} failed: {e}")
                    continue
            
            if not all_predictions:
                return []
            
            # Combinaison des résultats
            ensemble_results = self._combine_predictions(all_predictions, data, metric_names)
            
            # Mise à jour des métriques
            anomaly_count = sum(1 for r in ensemble_results if r.is_anomaly)
            ACTIVE_ANOMALIES.labels(type='ensemble').set(anomaly_count)
            
            return ensemble_results
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            raise
    
    def _combine_predictions(self, predictions: Dict[str, List[AnomalyResult]], 
                           data: np.ndarray, metric_names: List[str]) -> List[AnomalyResult]:
        """Combine les prédictions selon la stratégie configurée."""
        
        if self.config['voting_strategy'] == 'majority':
            return self._majority_voting(predictions, data, metric_names)
        elif self.config['voting_strategy'] == 'weighted':
            return self._weighted_voting(predictions, data, metric_names)
        elif self.config['voting_strategy'] == 'unanimous':
            return self._unanimous_voting(predictions, data, metric_names)
        else:
            raise ValueError(f"Unknown voting strategy: {self.config['voting_strategy']}")
    
    def _weighted_voting(self, predictions: Dict[str, List[AnomalyResult]], 
                        data: np.ndarray, metric_names: List[str]) -> List[AnomalyResult]:
        """Vote pondéré basé sur les poids configurés."""
        results = []
        weights = self.config['weights']
        
        # Déterminer la longueur minimale pour éviter les erreurs d'index
        min_length = min(len(preds) for preds in predictions.values())
        
        for i in range(min_length):
            weighted_score = 0.0
            weighted_confidence = 0.0
            anomaly_votes = 0
            total_weight = 0
            
            detector_results = {}
            
            for detector_name, detector_predictions in predictions.items():
                if i < len(detector_predictions):
                    pred = detector_predictions[i]
                    weight = weights.get(detector_name, 0.0)
                    
                    weighted_score += pred.anomaly_score * weight
                    weighted_confidence += pred.confidence * weight
                    total_weight += weight
                    
                    if pred.is_anomaly:
                        anomaly_votes += weight
                        
                    detector_results[detector_name] = pred
            
            if total_weight > 0:
                avg_score = weighted_score / total_weight
                avg_confidence = weighted_confidence / total_weight
                is_anomaly = (anomaly_votes / total_weight) >= self.config['confidence_threshold']
                
                # Détermination de la sévérité basée sur le consensus
                if anomaly_votes / total_weight >= 0.8:
                    severity = "critical"
                elif anomaly_votes / total_weight >= 0.6:
                    severity = "high"
                elif anomaly_votes / total_weight >= 0.4:
                    severity = "medium"
                else:
                    severity = "low"
                
                result = AnomalyResult(
                    timestamp=datetime.now(),
                    metric_name=metric_names[i] if metric_names else f"metric_{i}",
                    value=data[i] if i < len(data) else 0.0,
                    anomaly_score=avg_score,
                    is_anomaly=is_anomaly,
                    confidence=avg_confidence,
                    algorithm_used="ensemble_weighted",
                    severity=severity,
                    context={
                        "voting_strategy": "weighted",
                        "anomaly_votes_ratio": anomaly_votes / total_weight,
                        "detector_results": {k: v.is_anomaly for k, v in detector_results.items()}
                    }
                )
                
                results.append(result)
                
                if is_anomaly:
                    ANOMALY_DETECTION_COUNTER.labels(
                        algorithm='ensemble_weighted', 
                        severity=severity
                    ).inc()
        
        return results
    
    def _majority_voting(self, predictions: Dict[str, List[AnomalyResult]], 
                        data: np.ndarray, metric_names: List[str]) -> List[AnomalyResult]:
        """Vote majoritaire simple."""
        results = []
        min_length = min(len(preds) for preds in predictions.values())
        
        for i in range(min_length):
            anomaly_count = 0
            total_detectors = 0
            avg_score = 0.0
            avg_confidence = 0.0
            
            for detector_predictions in predictions.values():
                if i < len(detector_predictions):
                    pred = detector_predictions[i]
                    if pred.is_anomaly:
                        anomaly_count += 1
                    avg_score += pred.anomaly_score
                    avg_confidence += pred.confidence
                    total_detectors += 1
            
            if total_detectors > 0:
                is_anomaly = anomaly_count > (total_detectors // 2)
                avg_score /= total_detectors
                avg_confidence /= total_detectors
                
                result = AnomalyResult(
                    timestamp=datetime.now(),
                    metric_name=metric_names[i] if metric_names else f"metric_{i}",
                    value=data[i] if i < len(data) else 0.0,
                    anomaly_score=avg_score,
                    is_anomaly=is_anomaly,
                    confidence=avg_confidence,
                    algorithm_used="ensemble_majority",
                    severity="medium" if is_anomaly else "low",
                    context={
                        "voting_strategy": "majority",
                        "anomaly_votes": anomaly_count,
                        "total_votes": total_detectors
                    }
                )
                
                results.append(result)
        
        return results
    
    async def predict_async(self, data: np.ndarray, metric_names: List[str] = None) -> List[AnomalyResult]:
        """Version asynchrone de la prédiction pour traitement concurrent."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before prediction")
        
        # Exécution asynchrone des détecteurs
        tasks = []
        for name, detector in self.detectors.items():
            task = asyncio.create_task(
                asyncio.to_thread(detector.predict, data, metric_names)
            )
            tasks.append((name, task))
        
        # Collecte des résultats
        all_predictions = {}
        for name, task in tasks:
            try:
                predictions = await task
                all_predictions[name] = predictions
            except Exception as e:
                logger.warning(f"Async detector {name} failed: {e}")
                continue
        
        return self._combine_predictions(all_predictions, data, metric_names)

# Fonctions utilitaires pour l'ensemble

def create_ensemble_config(strategy: str = 'weighted') -> Dict[str, Any]:
    """Crée une configuration optimisée pour l'ensemble."""
    configs = {
        'conservative': {
            'voting_strategy': 'unanimous',
            'confidence_threshold': 0.8,
            'weights': {'isolation_forest': 0.5, 'one_class_svm': 0.3, 'lstm_autoencoder': 0.2}
        },
        'balanced': {
            'voting_strategy': 'weighted',
            'confidence_threshold': 0.6,
            'weights': {'isolation_forest': 0.4, 'one_class_svm': 0.3, 'lstm_autoencoder': 0.3}
        },
        'aggressive': {
            'voting_strategy': 'majority',
            'confidence_threshold': 0.4,
            'weights': {'isolation_forest': 0.3, 'one_class_svm': 0.3, 'lstm_autoencoder': 0.4}
        }
    }
    
    return configs.get(strategy, configs['balanced'])

def evaluate_detector_performance(detector: BaseAnomalyDetector, 
                                test_data: np.ndarray, 
                                true_labels: np.ndarray) -> Dict[str, float]:
    """Évalue les performances d'un détecteur."""
    predictions = detector.predict(test_data)
    pred_labels = [1 if r.is_anomaly else 0 for r in predictions]
    
    return {
        'precision': precision_score(true_labels, pred_labels),
        'recall': recall_score(true_labels, pred_labels),
        'f1_score': f1_score(true_labels, pred_labels),
        'anomaly_rate': sum(pred_labels) / len(pred_labels)
    }
