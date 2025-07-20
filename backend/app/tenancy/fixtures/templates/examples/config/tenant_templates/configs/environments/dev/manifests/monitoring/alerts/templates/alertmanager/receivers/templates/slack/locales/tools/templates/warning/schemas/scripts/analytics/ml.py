"""
Machine Learning Module - Module ML pour Analytics
=================================================

Ce module fournit des capacités de machine learning avancées pour
l'analytics, incluant la détection d'anomalies, l'analyse prédictive,
les recommandations et l'analyse comportementale.

Classes:
- BaseMLModel: Modèle ML de base
- AnomalyDetector: Détection d'anomalies
- PredictiveAnalytics: Analyses prédictives
- RecommendationEngine: Moteur de recommandations
- BehaviorAnalyzer: Analyseur comportemental
- ModelManager: Gestionnaire de modèles
"""

import asyncio
import pickle
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import json

# ML Libraries
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, mean_squared_error, accuracy_score
from sklearn.decomposition import PCA
import tensorflow as tf
import torch
import torch.nn as nn
from scipy import stats
from scipy.signal import find_peaks

from ..config import AnalyticsConfig
from ..models import Metric, Event, Alert
from ..utils import Logger, Timer, measure_time


@dataclass
class MLPrediction:
    """Résultat de prédiction ML."""
    model_name: str
    prediction: Any
    confidence: float
    timestamp: datetime
    input_features: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'input_features': self.input_features,
            'metadata': self.metadata
        }


@dataclass
class ModelMetrics:
    """Métriques d'un modèle ML."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    mse: float = 0.0
    mae: float = 0.0
    training_time: float = 0.0
    prediction_time: float = 0.0
    model_size_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'mse': self.mse,
            'mae': self.mae,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time,
            'model_size_mb': self.model_size_mb
        }


class BaseMLModel(ABC):
    """Modèle ML de base."""
    
    def __init__(self, config: AnalyticsConfig, model_name: str):
        self.config = config
        self.model_name = model_name
        self.logger = Logger(f"ML.{model_name}")
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.is_trained = False
        self.metrics = ModelMetrics()
        self.feature_names = []
        
        # Chemins de sauvegarde
        self.model_path = Path(config.ml.models_base_path) / f"{model_name}.pkl"
        self.scaler_path = Path(config.ml.models_base_path) / f"{model_name}_scaler.pkl"
        self.metadata_path = Path(config.ml.models_base_path) / f"{model_name}_metadata.json"
    
    @abstractmethod
    async def train(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> ModelMetrics:
        """Entraîne le modèle."""
        pass
    
    @abstractmethod
    async def predict(self, data: pd.DataFrame) -> MLPrediction:
        """Fait une prédiction."""
        pass
    
    async def save_model(self):
        """Sauvegarde le modèle."""
        try:
            # Créer le répertoire si nécessaire
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarder le modèle
            if self.model is not None:
                joblib.dump(self.model, self.model_path)
            
            # Sauvegarder le scaler
            if self.scaler is not None:
                joblib.dump(self.scaler, self.scaler_path)
            
            # Sauvegarder les métadonnées
            metadata = {
                'model_name': self.model_name,
                'is_trained': self.is_trained,
                'feature_names': self.feature_names,
                'metrics': self.metrics.to_dict(),
                'created_at': datetime.utcnow().isoformat(),
                'model_type': self.__class__.__name__
            }
            
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Modèle {self.model_name} sauvegardé")
            
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde modèle: {e}")
            raise
    
    async def load_model(self):
        """Charge le modèle."""
        try:
            # Charger le modèle
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
            
            # Charger le scaler
            if self.scaler_path.exists():
                self.scaler = joblib.load(self.scaler_path)
            
            # Charger les métadonnées
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.is_trained = metadata.get('is_trained', False)
                self.feature_names = metadata.get('feature_names', [])
                
                # Restaurer les métriques
                metrics_data = metadata.get('metrics', {})
                self.metrics = ModelMetrics(**metrics_data)
            
            self.logger.info(f"Modèle {self.model_name} chargé")
            
        except Exception as e:
            self.logger.error(f"Erreur chargement modèle: {e}")
            raise
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prépare les features pour l'entraînement/prédiction."""
        # Sélectionner les colonnes numériques
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        features = data[numeric_cols].fillna(0)
        
        # Normalisation
        if self.scaler is None:
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features)
        else:
            features_scaled = self.scaler.transform(features)
        
        self.feature_names = list(numeric_cols)
        return features_scaled


class AnomalyDetector(BaseMLModel):
    """Détecteur d'anomalies basé sur Isolation Forest."""
    
    def __init__(self, config: AnalyticsConfig):
        super().__init__(config, "anomaly_detector")
        self.contamination = config.ml.anomaly_threshold
        self.window_size = config.ml.anomaly_window_size
        self.min_samples = config.ml.anomaly_min_samples
    
    async def train(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> ModelMetrics:
        """Entraîne le détecteur d'anomalies."""
        start_time = time.time()
        
        try:
            if len(data) < self.min_samples:
                raise ValueError(f"Données insuffisantes: {len(data)} < {self.min_samples}")
            
            # Préparer les features
            features = self._prepare_features(data)
            
            # Créer et entraîner le modèle
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
            
            self.model.fit(features)
            self.is_trained = True
            
            # Évaluer le modèle
            scores = self.model.decision_function(features)
            anomalies = self.model.predict(features)
            
            # Calculer les métriques
            training_time = time.time() - start_time
            self.metrics.training_time = training_time
            
            # Métrique simple: pourcentage d'anomalies détectées
            anomaly_rate = np.sum(anomalies == -1) / len(anomalies)
            self.metrics.accuracy = 1.0 - abs(anomaly_rate - self.contamination)
            
            self.logger.info(
                f"Détecteur d'anomalies entraîné: {anomaly_rate:.3f} anomalies détectées"
            )
            
            return self.metrics
            
        except Exception as e:
            self.logger.error(f"Erreur entraînement détecteur anomalies: {e}")
            raise
    
    async def predict(self, data: pd.DataFrame) -> MLPrediction:
        """Détecte les anomalies."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Modèle non entraîné")
        
        start_time = time.time()
        
        try:
            # Préparer les features
            features = self._prepare_features(data)
            
            # Prédiction
            predictions = self.model.predict(features)
            scores = self.model.decision_function(features)
            
            # Normaliser les scores en confiance (0-1)
            confidence_scores = (scores - scores.min()) / (scores.max() - scores.min())
            
            # Résultats
            anomalies = []
            for i, (pred, score, conf) in enumerate(zip(predictions, scores, confidence_scores)):
                anomalies.append({
                    'index': i,
                    'is_anomaly': pred == -1,
                    'anomaly_score': float(score),
                    'confidence': float(conf)
                })
            
            prediction_time = time.time() - start_time
            self.metrics.prediction_time = prediction_time
            
            return MLPrediction(
                model_name=self.model_name,
                prediction=anomalies,
                confidence=float(np.mean(confidence_scores)),
                timestamp=datetime.utcnow(),
                input_features=self.feature_names,
                metadata={
                    'total_samples': len(data),
                    'anomalies_detected': np.sum(predictions == -1),
                    'anomaly_rate': float(np.sum(predictions == -1) / len(predictions))
                }
            )
            
        except Exception as e:
            self.logger.error(f"Erreur prédiction anomalies: {e}")
            raise


class PredictiveAnalytics(BaseMLModel):
    """Analyses prédictives avec Random Forest."""
    
    def __init__(self, config: AnalyticsConfig):
        super().__init__(config, "predictive_analytics")
        self.forecast_horizon = 24  # heures
        self.confidence_threshold = config.ml.prediction_confidence_threshold
    
    async def train(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> ModelMetrics:
        """Entraîne le modèle prédictif."""
        start_time = time.time()
        
        try:
            if target is None:
                raise ValueError("Target requis pour l'analyse prédictive")
            
            # Préparer les features
            features = self._prepare_features(data)
            
            # Déterminer le type de problème
            if target.dtype in ['object', 'category'] or len(target.unique()) < 10:
                # Classification
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    max_depth=10
                )
                problem_type = "classification"
            else:
                # Régression
                self.model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    max_depth=10
                )
                problem_type = "regression"
            
            # Division train/test
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42
            )
            
            # Entraînement
            self.model.fit(X_train, y_train)
            self.is_trained = True
            
            # Évaluation
            y_pred = self.model.predict(X_test)
            
            if problem_type == "classification":
                self.metrics.accuracy = accuracy_score(y_test, y_pred)
                # Report détaillé (simplifié)
                report = classification_report(y_test, y_pred, output_dict=True)
                self.metrics.precision = report['weighted avg']['precision']
                self.metrics.recall = report['weighted avg']['recall']
                self.metrics.f1_score = report['weighted avg']['f1-score']
            else:
                self.metrics.mse = mean_squared_error(y_test, y_pred)
                self.metrics.mae = np.mean(np.abs(y_test - y_pred))
            
            training_time = time.time() - start_time
            self.metrics.training_time = training_time
            
            self.logger.info(f"Modèle prédictif entraîné ({problem_type})")
            
            return self.metrics
            
        except Exception as e:
            self.logger.error(f"Erreur entraînement modèle prédictif: {e}")
            raise
    
    async def predict(self, data: pd.DataFrame) -> MLPrediction:
        """Fait des prédictions."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Modèle non entraîné")
        
        start_time = time.time()
        
        try:
            # Préparer les features
            features = self._prepare_features(data)
            
            # Prédictions
            predictions = self.model.predict(features)
            
            # Confiance (basée sur la variance des arbres pour Random Forest)
            if hasattr(self.model, 'estimators_'):
                # Pour Random Forest, calculer la variance des prédictions
                tree_predictions = np.array([
                    tree.predict(features) for tree in self.model.estimators_
                ])
                prediction_variance = np.var(tree_predictions, axis=0)
                confidence = 1.0 / (1.0 + prediction_variance)  # Confiance inversée à la variance
            else:
                confidence = np.ones(len(predictions)) * 0.8  # Confiance par défaut
            
            prediction_time = time.time() - start_time
            self.metrics.prediction_time = prediction_time
            
            return MLPrediction(
                model_name=self.model_name,
                prediction=predictions.tolist(),
                confidence=float(np.mean(confidence)),
                timestamp=datetime.utcnow(),
                input_features=self.feature_names,
                metadata={
                    'prediction_count': len(predictions),
                    'confidence_scores': confidence.tolist(),
                    'prediction_variance': float(np.mean(prediction_variance)) if 'prediction_variance' in locals() else None
                }
            )
            
        except Exception as e:
            self.logger.error(f"Erreur prédiction: {e}")
            raise
    
    async def forecast_timeseries(
        self,
        timeseries_data: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        value_col: str = 'value'
    ) -> MLPrediction:
        """Prévision de séries temporelles."""
        try:
            # Préparer les données de série temporelle
            ts_data = timeseries_data.copy()
            ts_data[timestamp_col] = pd.to_datetime(ts_data[timestamp_col])
            ts_data = ts_data.sort_values(timestamp_col)
            
            # Créer des features temporelles
            ts_data['hour'] = ts_data[timestamp_col].dt.hour
            ts_data['day_of_week'] = ts_data[timestamp_col].dt.dayofweek
            ts_data['day_of_month'] = ts_data[timestamp_col].dt.day
            ts_data['month'] = ts_data[timestamp_col].dt.month
            
            # Features de lag
            for lag in [1, 2, 3, 6, 12, 24]:
                ts_data[f'lag_{lag}'] = ts_data[value_col].shift(lag)
            
            # Features de rolling
            for window in [3, 6, 12, 24]:
                ts_data[f'rolling_mean_{window}'] = ts_data[value_col].rolling(window).mean()
                ts_data[f'rolling_std_{window}'] = ts_data[value_col].rolling(window).std()
            
            # Supprimer les NaN
            ts_data = ts_data.dropna()
            
            if len(ts_data) < 48:  # Minimum 48 points pour une prévision horaire
                raise ValueError("Données insuffisantes pour la prévision")
            
            # Préparer les features et target
            feature_cols = [col for col in ts_data.columns 
                          if col not in [timestamp_col, value_col]]
            X = ts_data[feature_cols]
            y = ts_data[value_col]
            
            # Entraîner si nécessaire
            if not self.is_trained:
                await self.train(X, y)
            
            # Prévision pour les prochaines heures
            last_point = ts_data.iloc[-1:].copy()
            forecasts = []
            
            for i in range(self.forecast_horizon):
                # Prédire le prochain point
                pred = await self.predict(last_point[feature_cols])
                forecast_value = pred.prediction[0]
                
                # Mise à jour pour la prochaine prédiction
                # (simplifiée - dans la réalité, il faudrait mettre à jour tous les lags)
                last_point[value_col] = forecast_value
                
                forecasts.append({
                    'horizon': i + 1,
                    'forecast': forecast_value,
                    'confidence': pred.confidence
                })
            
            return MLPrediction(
                model_name=f"{self.model_name}_forecast",
                prediction=forecasts,
                confidence=float(np.mean([f['confidence'] for f in forecasts])),
                timestamp=datetime.utcnow(),
                input_features=feature_cols,
                metadata={
                    'forecast_horizon': self.forecast_horizon,
                    'data_points_used': len(ts_data),
                    'forecast_type': 'timeseries'
                }
            )
            
        except Exception as e:
            self.logger.error(f"Erreur prévision série temporelle: {e}")
            raise


class BehaviorAnalyzer(BaseMLModel):
    """Analyseur comportemental utilisant le clustering."""
    
    def __init__(self, config: AnalyticsConfig):
        super().__init__(config, "behavior_analyzer")
        self.n_clusters = 5
        self.clustering_model = None
    
    async def train(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> ModelMetrics:
        """Entraîne l'analyseur comportemental."""
        start_time = time.time()
        
        try:
            # Préparer les features
            features = self._prepare_features(data)
            
            # Réduction de dimensionnalité si nécessaire
            if features.shape[1] > 10:
                self.pca = PCA(n_components=10)
                features = self.pca.fit_transform(features)
            
            # Clustering K-means
            self.clustering_model = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                init='k-means++',
                n_init=10
            )
            
            clusters = self.clustering_model.fit_predict(features)
            
            # DBSCAN pour la détection d'outliers
            self.dbscan_model = DBSCAN(eps=0.5, min_samples=5)
            outliers = self.dbscan_model.fit_predict(features)
            
            self.is_trained = True
            
            # Métriques de clustering
            inertia = self.clustering_model.inertia_
            silhouette_avg = self._calculate_silhouette_score(features, clusters)
            
            training_time = time.time() - start_time
            self.metrics.training_time = training_time
            self.metrics.accuracy = silhouette_avg  # Utiliser silhouette comme métrique
            
            self.logger.info(
                f"Analyseur comportemental entraîné: "
                f"{self.n_clusters} clusters, silhouette={silhouette_avg:.3f}"
            )
            
            return self.metrics
            
        except Exception as e:
            self.logger.error(f"Erreur entraînement analyseur comportemental: {e}")
            raise
    
    async def predict(self, data: pd.DataFrame) -> MLPrediction:
        """Analyse le comportement."""
        if not self.is_trained or self.clustering_model is None:
            raise RuntimeError("Modèle non entraîné")
        
        start_time = time.time()
        
        try:
            # Préparer les features
            features = self._prepare_features(data)
            
            # Réduction de dimensionnalité si nécessaire
            if hasattr(self, 'pca'):
                features = self.pca.transform(features)
            
            # Prédiction des clusters
            clusters = self.clustering_model.predict(features)
            distances = self.clustering_model.transform(features)
            
            # Distance au centroïde le plus proche (confiance)
            min_distances = np.min(distances, axis=1)
            confidence = 1.0 / (1.0 + min_distances)  # Confiance inversée à la distance
            
            # Détection d'outliers
            outliers = self.dbscan_model.fit_predict(features)
            
            # Analyse des comportements
            behavior_analysis = []
            for i, (cluster, outlier, conf) in enumerate(zip(clusters, outliers, confidence)):
                behavior_analysis.append({
                    'index': i,
                    'cluster': int(cluster),
                    'is_outlier': outlier == -1,
                    'confidence': float(conf),
                    'behavior_type': self._interpret_cluster(cluster),
                    'anomaly_level': self._calculate_anomaly_level(min_distances[i])
                })
            
            prediction_time = time.time() - start_time
            self.metrics.prediction_time = prediction_time
            
            return MLPrediction(
                model_name=self.model_name,
                prediction=behavior_analysis,
                confidence=float(np.mean(confidence)),
                timestamp=datetime.utcnow(),
                input_features=self.feature_names,
                metadata={
                    'n_clusters': self.n_clusters,
                    'outliers_detected': np.sum(outliers == -1),
                    'cluster_distribution': {
                        str(i): int(np.sum(clusters == i)) 
                        for i in range(self.n_clusters)
                    }
                }
            )
            
        except Exception as e:
            self.logger.error(f"Erreur analyse comportementale: {e}")
            raise
    
    def _calculate_silhouette_score(self, features: np.ndarray, clusters: np.ndarray) -> float:
        """Calcule le score de silhouette."""
        try:
            from sklearn.metrics import silhouette_score
            return silhouette_score(features, clusters)
        except:
            return 0.5  # Score par défaut
    
    def _interpret_cluster(self, cluster: int) -> str:
        """Interprète un cluster en type de comportement."""
        behavior_types = [
            "heavy_user", "moderate_user", "light_user", 
            "power_user", "irregular_user"
        ]
        return behavior_types[cluster % len(behavior_types)]
    
    def _calculate_anomaly_level(self, distance: float) -> str:
        """Calcule le niveau d'anomalie basé sur la distance."""
        if distance < 0.5:
            return "normal"
        elif distance < 1.0:
            return "unusual"
        elif distance < 2.0:
            return "anomalous"
        else:
            return "highly_anomalous"


class ModelManager:
    """Gestionnaire de modèles ML."""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.logger = Logger(__name__)
        self.models: Dict[str, BaseMLModel] = {}
        
        # Initialiser les modèles
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialise tous les modèles."""
        self.models = {
            'anomaly_detector': AnomalyDetector(self.config),
            'predictive_analytics': PredictiveAnalytics(self.config),
            'behavior_analyzer': BehaviorAnalyzer(self.config)
        }
    
    async def load_all_models(self):
        """Charge tous les modèles sauvegardés."""
        for name, model in self.models.items():
            try:
                await model.load_model()
                self.logger.info(f"Modèle {name} chargé")
            except Exception as e:
                self.logger.warning(f"Impossible de charger le modèle {name}: {e}")
    
    async def save_all_models(self):
        """Sauvegarde tous les modèles."""
        for name, model in self.models.items():
            try:
                if model.is_trained:
                    await model.save_model()
                    self.logger.info(f"Modèle {name} sauvegardé")
            except Exception as e:
                self.logger.error(f"Erreur sauvegarde modèle {name}: {e}")
    
    def get_model(self, model_name: str) -> Optional[BaseMLModel]:
        """Récupère un modèle par nom."""
        return self.models.get(model_name)
    
    def get_all_model_stats(self) -> Dict[str, Dict[str, Any]]:
        """Récupère les stats de tous les modèles."""
        stats = {}
        for name, model in self.models.items():
            stats[name] = {
                'is_trained': model.is_trained,
                'metrics': model.metrics.to_dict(),
                'feature_count': len(model.feature_names)
            }
        return stats
    
    async def retrain_model(
        self, 
        model_name: str, 
        data: pd.DataFrame, 
        target: Optional[pd.Series] = None
    ) -> ModelMetrics:
        """Réentraîne un modèle."""
        model = self.get_model(model_name)
        if not model:
            raise ValueError(f"Modèle {model_name} non trouvé")
        
        metrics = await model.train(data, target)
        await model.save_model()
        
        self.logger.info(f"Modèle {model_name} réentraîné")
        return metrics


# Fonctions utilitaires
async def create_model_manager(config: AnalyticsConfig) -> ModelManager:
    """Crée et initialise un gestionnaire de modèles."""
    manager = ModelManager(config)
    await manager.load_all_models()
    return manager


def prepare_metric_features(metrics: List[Metric]) -> pd.DataFrame:
    """Prépare les features à partir de métriques."""
    data = []
    for metric in metrics:
        row = {
            'value': metric.value,
            'hour': metric.timestamp.hour,
            'day_of_week': metric.timestamp.weekday(),
            'minute': metric.timestamp.minute,
            'tenant_id_hash': hash(metric.tenant_id) % 1000,
            'metric_name_hash': hash(metric.name) % 1000,
            'tag_count': len(metric.tags)
        }
        
        # Ajouter des features de tags
        for i, (key, value) in enumerate(metric.tags.items()):
            if i < 5:  # Limiter à 5 tags
                row[f'tag_{i}_hash'] = hash(f"{key}={value}") % 1000
        
        data.append(row)
    
    return pd.DataFrame(data).fillna(0)


def prepare_event_features(events: List[Event]) -> pd.DataFrame:
    """Prépare les features à partir d'événements."""
    data = []
    for event in events:
        row = {
            'hour': event.timestamp.hour,
            'day_of_week': event.timestamp.weekday(),
            'minute': event.timestamp.minute,
            'event_type_hash': hash(event.event_type) % 1000,
            'source_hash': hash(event.source) % 1000,
            'tenant_id_hash': hash(event.tenant_id) % 1000,
            'data_size': len(str(event.data)),
            'context_size': len(str(event.context)),
            'tag_count': len(event.tags)
        }
        
        data.append(row)
    
    return pd.DataFrame(data).fillna(0)
