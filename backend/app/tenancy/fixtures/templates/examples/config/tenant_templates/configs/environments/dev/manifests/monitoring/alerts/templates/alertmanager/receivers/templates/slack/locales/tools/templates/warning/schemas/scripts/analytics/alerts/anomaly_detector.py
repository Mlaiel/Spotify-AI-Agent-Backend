"""
Détecteur d'anomalies avancé pour Spotify AI Agent
Implémentation d'algorithmes ML de pointe pour la détection d'anomalies en temps réel
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from prometheus_client import Counter, Histogram, Gauge
import aioredis

logger = logging.getLogger(__name__)

class AnomalyType(Enum):
    """Types d'anomalies détectées"""
    STATISTICAL = "statistical"
    BEHAVIORAL = "behavioral"
    TEMPORAL = "temporal"
    MULTIDIMENSIONAL = "multidimensional"
    CONTEXTUAL = "contextual"

class DetectionMethod(Enum):
    """Méthodes de détection disponibles"""
    ISOLATION_FOREST = "isolation_forest"
    STATISTICAL_OUTLIER = "statistical_outlier"
    LSTM_AUTOENCODER = "lstm_autoencoder"
    DBSCAN_CLUSTERING = "dbscan_clustering"
    ELLIPTIC_ENVELOPE = "elliptic_envelope"
    ENSEMBLE = "ensemble"

@dataclass
class AnomalyResult:
    """Résultat de détection d'anomalie"""
    timestamp: datetime
    anomaly_score: float
    anomaly_type: AnomalyType
    detection_method: DetectionMethod
    confidence: float
    features_contribution: Dict[str, float]
    explanation: str
    severity_level: str

@dataclass
class TimeSeriesPoint:
    """Point de données temporelles"""
    timestamp: datetime
    value: float
    features: Dict[str, float]
    metadata: Dict[str, Any]

class AnomalyDetector:
    """
    Détecteur d'anomalies multi-algorithmes
    
    Combine plusieurs techniques ML pour une détection robuste:
    - Isolation Forest pour détection globale
    - LSTM Autoencoder pour patterns temporels
    - Méthodes statistiques pour outliers
    - Clustering pour anomalies comportementales
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client: Optional[aioredis.Redis] = None
        
        # Modèles ML
        self.isolation_forest = None
        self.lstm_autoencoder = None
        self.dbscan = None
        self.elliptic_envelope = None
        
        # Scalers
        self.standard_scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        
        # État et historique
        self.is_trained = False
        self.training_data = []
        self.baseline_stats = {}
        
        # Métriques Prometheus
        self.anomalies_detected = Counter(
            'anomalies_detected_total',
            'Total des anomalies détectées',
            ['type', 'method', 'severity']
        )
        self.detection_latency = Histogram(
            'anomaly_detection_duration_seconds',
            'Latence de détection d\'anomalies'
        )
        self.anomaly_score_distribution = Histogram(
            'anomaly_score_distribution',
            'Distribution des scores d\'anomalie',
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
    async def initialize(self):
        """Initialisation du détecteur"""
        try:
            # Connexion Redis
            self.redis_client = await aioredis.from_url(
                self.config.get('redis_url', 'redis://localhost:6379'),
                encoding='utf-8',
                decode_responses=True
            )
            
            # Chargement des modèles pré-entraînés
            await self._load_pretrained_models()
            
            # Initialisation des algorithmes
            await self._initialize_algorithms()
            
            logger.info("Détecteur d'anomalies initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur initialisation détecteur: {e}")
            raise
    
    async def _initialize_algorithms(self):
        """Initialisation des algorithmes de détection"""
        try:
            # Isolation Forest
            self.isolation_forest = IsolationForest(
                contamination=self.config.get('contamination_rate', 0.1),
                n_estimators=self.config.get('n_estimators', 100),
                max_samples='auto',
                random_state=42,
                n_jobs=-1
            )
            
            # DBSCAN pour clustering
            self.dbscan = DBSCAN(
                eps=self.config.get('dbscan_eps', 0.5),
                min_samples=self.config.get('dbscan_min_samples', 5)
            )
            
            # Elliptic Envelope pour détection robuste
            self.elliptic_envelope = EllipticEnvelope(
                contamination=self.config.get('contamination_rate', 0.1),
                random_state=42
            )
            
            # Construction du modèle LSTM Autoencoder
            await self._build_lstm_autoencoder()
            
        except Exception as e:
            logger.error(f"Erreur initialisation algorithmes: {e}")
            raise
    
    async def _build_lstm_autoencoder(self):
        """Construction du modèle LSTM Autoencoder"""
        try:
            sequence_length = self.config.get('sequence_length', 50)
            n_features = self.config.get('n_features', 10)
            
            # Architecture encoder-decoder
            input_layer = layers.Input(shape=(sequence_length, n_features))
            
            # Encoder
            encoded = layers.LSTM(64, return_sequences=True)(input_layer)
            encoded = layers.Dropout(0.2)(encoded)
            encoded = layers.LSTM(32, return_sequences=False)(encoded)
            encoded = layers.Dropout(0.2)(encoded)
            
            # Bottleneck
            bottleneck = layers.Dense(16, activation='relu')(encoded)
            
            # Decoder
            decoded = layers.RepeatVector(sequence_length)(bottleneck)
            decoded = layers.LSTM(32, return_sequences=True)(decoded)
            decoded = layers.Dropout(0.2)(decoded)
            decoded = layers.LSTM(64, return_sequences=True)(decoded)
            decoded = layers.TimeDistributed(layers.Dense(n_features))(decoded)
            
            # Modèle complet
            self.lstm_autoencoder = models.Model(input_layer, decoded)
            self.lstm_autoencoder.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            logger.info("Modèle LSTM Autoencoder construit")
            
        except Exception as e:
            logger.error(f"Erreur construction LSTM: {e}")
    
    async def detect_anomaly(
        self, 
        data_point: Union[TimeSeriesPoint, List[float], Dict[str, float]],
        method: DetectionMethod = DetectionMethod.ENSEMBLE
    ) -> AnomalyResult:
        """
        Détection d'anomalie sur un point de données
        
        Args:
            data_point: Point de données à analyser
            method: Méthode de détection à utiliser
            
        Returns:
            AnomalyResult: Résultat de la détection
        """
        start_time = datetime.now()
        
        try:
            # Normalisation du point de données
            normalized_data = await self._normalize_data_point(data_point)
            
            if method == DetectionMethod.ENSEMBLE:
                result = await self._ensemble_detection(normalized_data)
            elif method == DetectionMethod.ISOLATION_FOREST:
                result = await self._isolation_forest_detection(normalized_data)
            elif method == DetectionMethod.LSTM_AUTOENCODER:
                result = await self._lstm_detection(normalized_data)
            elif method == DetectionMethod.STATISTICAL_OUTLIER:
                result = await self._statistical_detection(normalized_data)
            elif method == DetectionMethod.DBSCAN_CLUSTERING:
                result = await self._clustering_detection(normalized_data)
            elif method == DetectionMethod.ELLIPTIC_ENVELOPE:
                result = await self._elliptic_detection(normalized_data)
            else:
                raise ValueError(f"Méthode non supportée: {method}")
            
            # Mise à jour des métriques
            self.anomalies_detected.labels(
                type=result.anomaly_type.value,
                method=result.detection_method.value,
                severity=result.severity_level
            ).inc()
            
            duration = (datetime.now() - start_time).total_seconds()
            self.detection_latency.observe(duration)
            self.anomaly_score_distribution.observe(result.anomaly_score)
            
            # Cache du résultat
            await self._cache_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur détection anomalie: {e}")
            # Retour d'un résultat par défaut en cas d'erreur
            return AnomalyResult(
                timestamp=datetime.now(),
                anomaly_score=0.5,
                anomaly_type=AnomalyType.STATISTICAL,
                detection_method=method,
                confidence=0.0,
                features_contribution={},
                explanation="Erreur lors de la détection",
                severity_level="UNKNOWN"
            )
    
    async def _ensemble_detection(self, data_point: np.ndarray) -> AnomalyResult:
        """Détection par ensemble de méthodes"""
        try:
            results = []
            weights = self.config.get('ensemble_weights', {
                'isolation_forest': 0.3,
                'lstm': 0.25,
                'statistical': 0.2,
                'clustering': 0.15,
                'elliptic': 0.1
            })
            
            # Exécution parallèle des différentes méthodes
            tasks = [
                self._isolation_forest_detection(data_point),
                self._lstm_detection(data_point),
                self._statistical_detection(data_point),
                self._clustering_detection(data_point),
                self._elliptic_detection(data_point)
            ]
            
            method_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Agrégation des scores
            weighted_score = 0.0
            total_weight = 0.0
            confidence_sum = 0.0
            explanations = []
            
            for i, (method_name, weight) in enumerate(weights.items()):
                if i < len(method_results) and not isinstance(method_results[i], Exception):
                    result = method_results[i]
                    weighted_score += result.anomaly_score * weight
                    confidence_sum += result.confidence * weight
                    total_weight += weight
                    explanations.append(f"{method_name}: {result.explanation}")
            
            # Normalisation
            if total_weight > 0:
                final_score = weighted_score / total_weight
                final_confidence = confidence_sum / total_weight
            else:
                final_score = 0.5
                final_confidence = 0.0
            
            # Détermination du type d'anomalie dominant
            anomaly_type = await self._determine_dominant_anomaly_type(method_results)
            
            # Calcul de la contribution des features
            features_contribution = await self._calculate_features_contribution(
                data_point, final_score
            )
            
            return AnomalyResult(
                timestamp=datetime.now(),
                anomaly_score=final_score,
                anomaly_type=anomaly_type,
                detection_method=DetectionMethod.ENSEMBLE,
                confidence=final_confidence,
                features_contribution=features_contribution,
                explanation="; ".join(explanations),
                severity_level=self._score_to_severity(final_score)
            )
            
        except Exception as e:
            logger.error(f"Erreur détection ensemble: {e}")
            raise
    
    async def _isolation_forest_detection(self, data_point: np.ndarray) -> AnomalyResult:
        """Détection par Isolation Forest"""
        try:
            if not self.is_trained:
                await self._train_models()
            
            # Reshape pour sklearn
            if data_point.ndim == 1:
                data_point = data_point.reshape(1, -1)
            
            # Prédiction
            anomaly_score = self.isolation_forest.decision_function(data_point)[0]
            is_outlier = self.isolation_forest.predict(data_point)[0] == -1
            
            # Normalisation du score (Isolation Forest retourne des valeurs négatives)
            normalized_score = max(0.0, min(1.0, (0.5 - anomaly_score) * 2))
            
            if is_outlier:
                normalized_score = max(0.7, normalized_score)
            
            explanation = f"Isolation Forest score: {anomaly_score:.3f}, Outlier: {is_outlier}"
            
            return AnomalyResult(
                timestamp=datetime.now(),
                anomaly_score=normalized_score,
                anomaly_type=AnomalyType.MULTIDIMENSIONAL,
                detection_method=DetectionMethod.ISOLATION_FOREST,
                confidence=0.8 if is_outlier else 0.6,
                features_contribution={},
                explanation=explanation,
                severity_level=self._score_to_severity(normalized_score)
            )
            
        except Exception as e:
            logger.error(f"Erreur Isolation Forest: {e}")
            return self._default_result(DetectionMethod.ISOLATION_FOREST)
    
    async def _lstm_detection(self, data_point: np.ndarray) -> AnomalyResult:
        """Détection par LSTM Autoencoder"""
        try:
            if self.lstm_autoencoder is None:
                await self._build_lstm_autoencoder()
            
            # Préparation séquence temporelle
            sequence = await self._prepare_sequence_for_lstm(data_point)
            
            if sequence is None:
                return self._default_result(DetectionMethod.LSTM_AUTOENCODER)
            
            # Prédiction et calcul d'erreur de reconstruction
            reconstructed = self.lstm_autoencoder.predict(sequence, verbose=0)
            reconstruction_error = np.mean(np.square(sequence - reconstructed))
            
            # Normalisation basée sur l'historique
            threshold = await self._get_lstm_threshold()
            anomaly_score = min(1.0, reconstruction_error / threshold)
            
            explanation = f"LSTM reconstruction error: {reconstruction_error:.4f}, threshold: {threshold:.4f}"
            
            return AnomalyResult(
                timestamp=datetime.now(),
                anomaly_score=anomaly_score,
                anomaly_type=AnomalyType.TEMPORAL,
                detection_method=DetectionMethod.LSTM_AUTOENCODER,
                confidence=0.7,
                features_contribution={},
                explanation=explanation,
                severity_level=self._score_to_severity(anomaly_score)
            )
            
        except Exception as e:
            logger.error(f"Erreur LSTM détection: {e}")
            return self._default_result(DetectionMethod.LSTM_AUTOENCODER)
    
    async def _statistical_detection(self, data_point: np.ndarray) -> AnomalyResult:
        """Détection statistique (Z-score modifié)"""
        try:
            # Calcul des statistiques de référence
            baseline = await self._get_baseline_statistics()
            
            if not baseline:
                return self._default_result(DetectionMethod.STATISTICAL_OUTLIER)
            
            # Calcul Z-score modifié (MAD)
            median = baseline.get('median', 0)
            mad = baseline.get('mad', 1)  # Median Absolute Deviation
            
            modified_z_scores = []
            for value in data_point:
                if mad != 0:
                    z_score = 0.6745 * (value - median) / mad
                    modified_z_scores.append(abs(z_score))
                else:
                    modified_z_scores.append(0)
            
            max_z_score = max(modified_z_scores) if modified_z_scores else 0
            
            # Seuil pour anomalie (généralement 3.5 pour Z-score modifié)
            threshold = self.config.get('statistical_threshold', 3.5)
            anomaly_score = min(1.0, max_z_score / threshold)
            
            explanation = f"Max Modified Z-score: {max_z_score:.3f}, threshold: {threshold}"
            
            return AnomalyResult(
                timestamp=datetime.now(),
                anomaly_score=anomaly_score,
                anomaly_type=AnomalyType.STATISTICAL,
                detection_method=DetectionMethod.STATISTICAL_OUTLIER,
                confidence=0.9 if max_z_score > threshold else 0.7,
                features_contribution={f"feature_{i}": score for i, score in enumerate(modified_z_scores)},
                explanation=explanation,
                severity_level=self._score_to_severity(anomaly_score)
            )
            
        except Exception as e:
            logger.error(f"Erreur détection statistique: {e}")
            return self._default_result(DetectionMethod.STATISTICAL_OUTLIER)
    
    async def _clustering_detection(self, data_point: np.ndarray) -> AnomalyResult:
        """Détection par clustering DBSCAN"""
        try:
            # Récupération des données récentes pour clustering
            recent_data = await self._get_recent_data_for_clustering()
            
            if len(recent_data) < 10:
                return self._default_result(DetectionMethod.DBSCAN_CLUSTERING)
            
            # Ajout du point actuel
            all_data = np.vstack([recent_data, data_point.reshape(1, -1)])
            
            # Clustering
            cluster_labels = self.dbscan.fit_predict(all_data)
            
            # Le dernier point est notre point d'intérêt
            point_cluster = cluster_labels[-1]
            
            # Point est anomalie si cluster = -1 (noise)
            is_anomaly = point_cluster == -1
            
            # Calcul score basé sur distance au cluster le plus proche
            if is_anomaly:
                # Distance minimale aux points des clusters
                clustered_points = all_data[cluster_labels != -1]
                if len(clustered_points) > 0:
                    distances = np.linalg.norm(
                        clustered_points - data_point.reshape(1, -1), axis=1
                    )
                    min_distance = np.min(distances)
                    anomaly_score = min(1.0, min_distance / np.std(distances))
                else:
                    anomaly_score = 0.8
            else:
                # Distance moyenne aux autres points du même cluster
                same_cluster_points = all_data[cluster_labels == point_cluster]
                if len(same_cluster_points) > 1:
                    distances = np.linalg.norm(
                        same_cluster_points - data_point.reshape(1, -1), axis=1
                    )
                    avg_distance = np.mean(distances)
                    cluster_std = np.std(distances)
                    anomaly_score = min(0.5, avg_distance / (cluster_std + 1e-6))
                else:
                    anomaly_score = 0.3
            
            explanation = f"DBSCAN cluster: {point_cluster}, is_anomaly: {is_anomaly}"
            
            return AnomalyResult(
                timestamp=datetime.now(),
                anomaly_score=anomaly_score,
                anomaly_type=AnomalyType.BEHAVIORAL,
                detection_method=DetectionMethod.DBSCAN_CLUSTERING,
                confidence=0.8 if is_anomaly else 0.6,
                features_contribution={},
                explanation=explanation,
                severity_level=self._score_to_severity(anomaly_score)
            )
            
        except Exception as e:
            logger.error(f"Erreur clustering DBSCAN: {e}")
            return self._default_result(DetectionMethod.DBSCAN_CLUSTERING)
    
    async def _elliptic_detection(self, data_point: np.ndarray) -> AnomalyResult:
        """Détection par Elliptic Envelope"""
        try:
            if not self.is_trained:
                await self._train_models()
            
            # Reshape pour sklearn
            if data_point.ndim == 1:
                data_point = data_point.reshape(1, -1)
            
            # Prédiction
            is_outlier = self.elliptic_envelope.predict(data_point)[0] == -1
            
            # Score de Mahalanobis
            mahalanobis_dist = self.elliptic_envelope.mahalanobis(data_point)[0]
            
            # Normalisation du score
            threshold = self.config.get('elliptic_threshold', 2.0)
            anomaly_score = min(1.0, mahalanobis_dist / threshold)
            
            if is_outlier:
                anomaly_score = max(0.7, anomaly_score)
            
            explanation = f"Mahalanobis distance: {mahalanobis_dist:.3f}, outlier: {is_outlier}"
            
            return AnomalyResult(
                timestamp=datetime.now(),
                anomaly_score=anomaly_score,
                anomaly_type=AnomalyType.CONTEXTUAL,
                detection_method=DetectionMethod.ELLIPTIC_ENVELOPE,
                confidence=0.75,
                features_contribution={},
                explanation=explanation,
                severity_level=self._score_to_severity(anomaly_score)
            )
            
        except Exception as e:
            logger.error(f"Erreur Elliptic Envelope: {e}")
            return self._default_result(DetectionMethod.ELLIPTIC_ENVELOPE)
    
    # Méthodes utilitaires
    
    async def _normalize_data_point(
        self, 
        data_point: Union[TimeSeriesPoint, List[float], Dict[str, float]]
    ) -> np.ndarray:
        """Normalisation du point de données"""
        try:
            if isinstance(data_point, TimeSeriesPoint):
                features = list(data_point.features.values())
                features.append(data_point.value)
                return np.array(features)
            elif isinstance(data_point, dict):
                return np.array(list(data_point.values()))
            elif isinstance(data_point, (list, np.ndarray)):
                return np.array(data_point)
            else:
                raise ValueError(f"Type de données non supporté: {type(data_point)}")
                
        except Exception as e:
            logger.error(f"Erreur normalisation données: {e}")
            return np.array([0.0])
    
    def _score_to_severity(self, score: float) -> str:
        """Conversion score en niveau de gravité"""
        if score >= 0.9:
            return "CRITICAL"
        elif score >= 0.7:
            return "HIGH"
        elif score >= 0.5:
            return "MEDIUM"
        elif score >= 0.3:
            return "LOW"
        else:
            return "INFO"
    
    def _default_result(self, method: DetectionMethod) -> AnomalyResult:
        """Résultat par défaut en cas d'erreur"""
        return AnomalyResult(
            timestamp=datetime.now(),
            anomaly_score=0.5,
            anomaly_type=AnomalyType.STATISTICAL,
            detection_method=method,
            confidence=0.0,
            features_contribution={},
            explanation="Détection échouée, score par défaut",
            severity_level="UNKNOWN"
        )
    
    async def _determine_dominant_anomaly_type(
        self, 
        method_results: List[AnomalyResult]
    ) -> AnomalyType:
        """Détermination du type d'anomalie dominant"""
        type_scores = {}
        
        for result in method_results:
            if isinstance(result, AnomalyResult):
                anomaly_type = result.anomaly_type
                if anomaly_type not in type_scores:
                    type_scores[anomaly_type] = []
                type_scores[anomaly_type].append(result.anomaly_score * result.confidence)
        
        # Type avec le score pondéré le plus élevé
        if type_scores:
            dominant_type = max(type_scores.keys(), 
                              key=lambda t: sum(type_scores[t]) / len(type_scores[t]))
            return dominant_type
        
        return AnomalyType.STATISTICAL
    
    async def _calculate_features_contribution(
        self, 
        data_point: np.ndarray, 
        anomaly_score: float
    ) -> Dict[str, float]:
        """Calcul de la contribution des features à l'anomalie"""
        try:
            if len(data_point) == 0:
                return {}
            
            # Analyse de sensibilité simple
            contributions = {}
            baseline = await self._get_baseline_statistics()
            
            if baseline and 'feature_means' in baseline:
                means = baseline['feature_means']
                stds = baseline['feature_stds']
                
                for i, value in enumerate(data_point):
                    if i < len(means) and stds[i] > 0:
                        z_score = abs((value - means[i]) / stds[i])
                        contribution = min(1.0, z_score / 3.0)  # Normalisation
                        contributions[f"feature_{i}"] = contribution
            
            return contributions
            
        except Exception as e:
            logger.error(f"Erreur calcul contributions: {e}")
            return {}
    
    async def _train_models(self):
        """Entraînement des modèles ML"""
        try:
            # Chargement des données d'entraînement
            training_data = await self._load_training_data()
            
            if len(training_data) < 100:
                logger.warning("Données d'entraînement insuffisantes")
                return
            
            # Préparation des données
            X = np.array(training_data)
            X_scaled = self.standard_scaler.fit_transform(X)
            
            # Entraînement Isolation Forest
            self.isolation_forest.fit(X_scaled)
            
            # Entraînement Elliptic Envelope
            self.elliptic_envelope.fit(X_scaled)
            
            # Calcul des statistiques de base
            await self._calculate_baseline_statistics(X)
            
            self.is_trained = True
            logger.info("Modèles ML entraînés avec succès")
            
        except Exception as e:
            logger.error(f"Erreur entraînement modèles: {e}")
    
    async def _load_training_data(self) -> List[List[float]]:
        """Chargement des données d'entraînement"""
        try:
            # Simulation de données historiques
            # En production, charger depuis la base de données
            np.random.seed(42)
            normal_data = np.random.normal(0, 1, (1000, 5))
            return normal_data.tolist()
            
        except Exception as e:
            logger.error(f"Erreur chargement données: {e}")
            return []
    
    async def _calculate_baseline_statistics(self, data: np.ndarray):
        """Calcul des statistiques de référence"""
        try:
            self.baseline_stats = {
                'mean': np.mean(data, axis=0).tolist(),
                'std': np.std(data, axis=0).tolist(),
                'median': np.median(data, axis=0).tolist(),
                'mad': np.median(np.abs(data - np.median(data, axis=0)), axis=0).tolist(),
                'feature_means': np.mean(data, axis=0).tolist(),
                'feature_stds': np.std(data, axis=0).tolist()
            }
            
            # Cache des statistiques
            await self.redis_client.setex(
                'baseline_statistics',
                3600,  # 1 heure
                str(self.baseline_stats)
            )
            
        except Exception as e:
            logger.error(f"Erreur calcul statistiques: {e}")
    
    async def _get_baseline_statistics(self) -> Dict[str, Any]:
        """Récupération des statistiques de référence"""
        try:
            # Tentative depuis le cache
            cached_stats = await self.redis_client.get('baseline_statistics')
            if cached_stats:
                return eval(cached_stats)  # En production, utiliser JSON
            
            return self.baseline_stats
            
        except Exception as e:
            logger.error(f"Erreur récupération statistiques: {e}")
            return {}
    
    async def _prepare_sequence_for_lstm(self, data_point: np.ndarray) -> Optional[np.ndarray]:
        """Préparation d'une séquence pour LSTM"""
        try:
            # Récupération de l'historique récent
            sequence_length = self.config.get('sequence_length', 50)
            
            # Simulation d'une séquence (en production, charger depuis la DB)
            sequence = np.random.normal(0, 1, (1, sequence_length, len(data_point)))
            
            # Ajout du point actuel à la fin
            sequence[0, -1, :] = data_point
            
            return sequence
            
        except Exception as e:
            logger.error(f"Erreur préparation séquence LSTM: {e}")
            return None
    
    async def _get_lstm_threshold(self) -> float:
        """Récupération du seuil LSTM"""
        try:
            cached_threshold = await self.redis_client.get('lstm_threshold')
            if cached_threshold:
                return float(cached_threshold)
            
            # Seuil par défaut
            return self.config.get('lstm_threshold', 0.1)
            
        except Exception as e:
            logger.error(f"Erreur récupération seuil LSTM: {e}")
            return 0.1
    
    async def _get_recent_data_for_clustering(self) -> np.ndarray:
        """Récupération de données récentes pour clustering"""
        try:
            # Simulation de données récentes
            # En production, requête base de données
            recent_data = np.random.normal(0, 1, (100, 5))
            return recent_data
            
        except Exception as e:
            logger.error(f"Erreur récupération données clustering: {e}")
            return np.array([])
    
    async def _cache_result(self, result: AnomalyResult):
        """Cache du résultat de détection"""
        try:
            cache_key = f"anomaly_result:{result.timestamp.isoformat()}"
            result_data = {
                'anomaly_score': result.anomaly_score,
                'anomaly_type': result.anomaly_type.value,
                'detection_method': result.detection_method.value,
                'confidence': result.confidence,
                'severity_level': result.severity_level
            }
            
            await self.redis_client.setex(
                cache_key,
                300,  # 5 minutes
                str(result_data)
            )
            
        except Exception as e:
            logger.error(f"Erreur cache résultat: {e}")
    
    async def _load_pretrained_models(self):
        """Chargement des modèles pré-entraînés"""
        try:
            # Tentative de chargement depuis le stockage
            # En production, charger depuis S3/GCS
            logger.info("Chargement des modèles pré-entraînés...")
            
            # Si pas de modèles pré-entraînés, on va les entraîner
            await self._train_models()
            
        except Exception as e:
            logger.error(f"Erreur chargement modèles pré-entraînés: {e}")
    
    async def batch_detect_anomalies(
        self, 
        data_points: List[Union[TimeSeriesPoint, List[float], Dict[str, float]]],
        method: DetectionMethod = DetectionMethod.ENSEMBLE
    ) -> List[AnomalyResult]:
        """Détection d'anomalies en lot"""
        try:
            tasks = [
                self.detect_anomaly(data_point, method) 
                for data_point in data_points
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filtrage des exceptions
            valid_results = [
                result for result in results 
                if isinstance(result, AnomalyResult)
            ]
            
            return valid_results
            
        except Exception as e:
            logger.error(f"Erreur détection en lot: {e}")
            return []
    
    async def get_detection_statistics(self) -> Dict[str, Any]:
        """Statistiques de détection"""
        try:
            stats = {
                'total_detections': sum(self.anomalies_detected._value.values()),
                'is_trained': self.is_trained,
                'baseline_available': bool(self.baseline_stats),
                'models_initialized': {
                    'isolation_forest': self.isolation_forest is not None,
                    'lstm_autoencoder': self.lstm_autoencoder is not None,
                    'dbscan': self.dbscan is not None,
                    'elliptic_envelope': self.elliptic_envelope is not None
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Erreur récupération statistiques: {e}")
            return {}
    
    async def close(self):
        """Fermeture propre"""
        try:
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("Détecteur d'anomalies fermé proprement")
            
        except Exception as e:
            logger.error(f"Erreur fermeture détecteur: {e}")


# Factory pour création d'instance
async def create_anomaly_detector(config: Dict[str, Any]) -> AnomalyDetector:
    """Factory pour créer et initialiser le détecteur d'anomalies"""
    detector = AnomalyDetector(config)
    await detector.initialize()
    return detector
