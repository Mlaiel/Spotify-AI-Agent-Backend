"""
Audio Classification Engine - Enterprise Edition
===============================================

Moteur de classification audio ML/AI pour Spotify AI Agent.
Classification genre, mood, instruments, et analyse sémantique avancée.
"""

import asyncio
import logging
import numpy as np
import joblib
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import librosa
from concurrent.futures import ThreadPoolExecutor
import json
import pickle
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# === Configuration et types ===
class ClassificationType(Enum):
    GENRE = "genre"
    MOOD = "mood"
    INSTRUMENT = "instrument"
    ENERGY_LEVEL = "energy_level"
    VOCAL_PRESENCE = "vocal_presence"
    TEMPO_CLASS = "tempo_class"
    KEY_SIGNATURE = "key_signature"

class ModelType(Enum):
    RANDOM_FOREST = "random_forest"
    SVM = "svm"
    NEURAL_NETWORK = "neural_network"
    DEEP_CNN = "deep_cnn"
    ENSEMBLE = "ensemble"

@dataclass
class ClassificationConfig:
    """Configuration pour la classification."""
    model_type: ModelType = ModelType.ENSEMBLE
    classification_type: ClassificationType = ClassificationType.GENRE
    feature_scaling: bool = True
    cross_validation: bool = True
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    save_model: bool = True
    model_path: Optional[str] = None

@dataclass
class ClassificationResult:
    """Résultat de classification."""
    predicted_class: str
    confidence: float
    probability_distribution: Dict[str, float]
    feature_importance: Optional[Dict[str, float]] = None
    processing_time: float = 0.0
    model_version: str = "1.0"

@dataclass
class ModelPerformance:
    """Métriques de performance du modèle."""
    accuracy: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1_score: Dict[str, float]
    confusion_matrix: List[List[int]]
    cross_val_scores: List[float]
    training_time: float

# === Classificateurs spécialisés ===
class GenreClassifier:
    """Classificateur de genre musical."""
    
    GENRE_LABELS = [
        'rock', 'pop', 'jazz', 'classical', 'hip-hop', 'electronic',
        'country', 'blues', 'reggae', 'folk', 'latin', 'funk',
        'punk', 'metal', 'indie', 'soul', 'r&b', 'alternative'
    ]
    
    def __init__(self, config: ClassificationConfig = None):
        self.config = config or ClassificationConfig(classification_type=ClassificationType.GENRE)
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        
    async def train(
        self,
        features_df: pd.DataFrame,
        labels: List[str]
    ) -> ModelPerformance:
        """Entraîne le modèle de classification de genre."""
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Préparation des données
            X = features_df.values
            y = self.label_encoder.fit_transform(labels)
            
            # Scaling des features
            if self.config.feature_scaling:
                X = self.scaler.fit_transform(X)
            
            # Division train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_size, 
                random_state=self.config.random_state, stratify=y
            )
            
            # Entraînement des modèles
            await self._train_models(X_train, y_train)
            
            # Évaluation
            performance = await self._evaluate_models(X_test, y_test, X, y)
            
            training_time = asyncio.get_event_loop().time() - start_time
            performance.training_time = training_time
            
            # Sauvegarde
            if self.config.save_model:
                await self._save_models()
            
            logger.info(f"Genre classifier trained in {training_time:.2f}s")
            logger.info(f"Best accuracy: {performance.accuracy:.3f}")
            
            return performance
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    async def _train_models(self, X_train: np.ndarray, y_train: np.ndarray):
        """Entraîne différents modèles."""
        
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=4)
        
        def train_rf():
            rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.config.random_state,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)
            return rf
        
        def train_svm():
            svm = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=self.config.random_state
            )
            svm.fit(X_train, y_train)
            return svm
        
        def train_mlp():
            mlp = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                learning_rate='adaptive',
                max_iter=500,
                random_state=self.config.random_state
            )
            mlp.fit(X_train, y_train)
            return mlp
        
        # Entraînement parallèle
        if self.config.model_type == ModelType.ENSEMBLE:
            tasks = [
                loop.run_in_executor(executor, train_rf),
                loop.run_in_executor(executor, train_svm),
                loop.run_in_executor(executor, train_mlp)
            ]
            
            rf_model, svm_model, mlp_model = await asyncio.gather(*tasks)
            
            self.models = {
                'random_forest': rf_model,
                'svm': svm_model,
                'neural_network': mlp_model
            }
        else:
            # Entraînement d'un seul modèle
            if self.config.model_type == ModelType.RANDOM_FOREST:
                self.models['random_forest'] = await loop.run_in_executor(executor, train_rf)
            elif self.config.model_type == ModelType.SVM:
                self.models['svm'] = await loop.run_in_executor(executor, train_svm)
            elif self.config.model_type == ModelType.NEURAL_NETWORK:
                self.models['neural_network'] = await loop.run_in_executor(executor, train_mlp)
        
        executor.shutdown(wait=True)
    
    async def _evaluate_models(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        X_full: np.ndarray,
        y_full: np.ndarray
    ) -> ModelPerformance:
        """Évalue les performances des modèles."""
        
        best_model = None
        best_score = 0.0
        
        # Test de chaque modèle
        for model_name, model in self.models.items():
            score = accuracy_score(y_test, model.predict(X_test))
            logger.info(f"{model_name} accuracy: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_model = model
        
        # Métriques détaillées sur le meilleur modèle
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Cross-validation
        cv_scores = []
        if self.config.cross_validation:
            cv_scores = cross_val_score(
                best_model, X_full, y_full, 
                cv=self.config.cv_folds, scoring='accuracy'
            ).tolist()
        
        return ModelPerformance(
            accuracy=best_score,
            precision={str(k): v['precision'] for k, v in report.items() if isinstance(v, dict)},
            recall={str(k): v['recall'] for k, v in report.items() if isinstance(v, dict)},
            f1_score={str(k): v['f1-score'] for k, v in report.items() if isinstance(v, dict)},
            confusion_matrix=confusion_matrix(y_test, y_pred).tolist(),
            cross_val_scores=cv_scores,
            training_time=0.0
        )
    
    async def predict(self, features: np.ndarray) -> ClassificationResult:
        """Prédit le genre d'un échantillon audio."""
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Préparation des features
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            if self.config.feature_scaling:
                features = self.scaler.transform(features)
            
            if self.config.model_type == ModelType.ENSEMBLE:
                # Prédiction par ensemble
                predictions = {}
                probabilities = {}
                
                for model_name, model in self.models.items():
                    pred = model.predict(features)[0]
                    predictions[model_name] = pred
                    
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(features)[0]
                        probabilities[model_name] = prob
                
                # Vote majoritaire
                predicted_class_idx = max(set(predictions.values()), key=list(predictions.values()).count)
                
                # Moyenne des probabilités
                if probabilities:
                    avg_proba = np.mean(list(probabilities.values()), axis=0)
                    confidence = float(np.max(avg_proba))
                    
                    prob_dist = {
                        self.label_encoder.inverse_transform([i])[0]: float(prob)
                        for i, prob in enumerate(avg_proba)
                    }
                else:
                    confidence = 0.8
                    prob_dist = {}
            
            else:
                # Prédiction simple
                model = list(self.models.values())[0]
                predicted_class_idx = model.predict(features)[0]
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features)[0]
                    confidence = float(np.max(proba))
                    
                    prob_dist = {
                        self.label_encoder.inverse_transform([i])[0]: float(prob)
                        for i, prob in enumerate(proba)
                    }
                else:
                    confidence = 0.8
                    prob_dist = {}
            
            # Classe prédite
            predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return ClassificationResult(
                predicted_class=predicted_class,
                confidence=confidence,
                probability_distribution=prob_dist,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return ClassificationResult(
                predicted_class="unknown",
                confidence=0.0,
                probability_distribution={}
            )
    
    async def _save_models(self):
        """Sauvegarde les modèles entraînés."""
        
        model_dir = Path(self.config.model_path or "models/genre_classifier")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarde des modèles
        for model_name, model in self.models.items():
            model_path = model_dir / f"{model_name}.joblib"
            joblib.dump(model, model_path)
        
        # Sauvegarde du scaler et label encoder
        joblib.dump(self.scaler, model_dir / "scaler.joblib")
        joblib.dump(self.label_encoder, model_dir / "label_encoder.joblib")
        
        # Métadonnées
        metadata = {
            'model_type': self.config.model_type.value,
            'classification_type': self.config.classification_type.value,
            'feature_scaling': self.config.feature_scaling,
            'timestamp': datetime.now().isoformat(),
            'genre_labels': self.GENRE_LABELS
        }
        
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Models saved to {model_dir}")

class MoodClassifier:
    """Classificateur d'humeur musicale."""
    
    MOOD_LABELS = [
        'happy', 'sad', 'energetic', 'calm', 'aggressive', 'romantic',
        'melancholic', 'uplifting', 'dark', 'bright', 'nostalgic', 'dreamy'
    ]
    
    def __init__(self, config: ClassificationConfig = None):
        self.config = config or ClassificationConfig(classification_type=ClassificationType.MOOD)
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    async def train(self, features_df: pd.DataFrame, labels: List[str]) -> ModelPerformance:
        """Entraîne le classificateur d'humeur."""
        
        # Architecture CNN pour l'analyse d'humeur
        X = features_df.values
        y = self.label_encoder.fit_transform(labels)
        
        if self.config.feature_scaling:
            X = self.scaler.fit_transform(X)
        
        # Reshape pour CNN si nécessaire
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, 
            random_state=self.config.random_state
        )
        
        # Construction du modèle CNN
        model = models.Sequential([
            layers.Conv1D(64, 3, activation='relu', input_shape=(X.shape[1], X.shape[2])),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            
            layers.Conv1D(128, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            
            layers.Conv1D(256, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalMaxPooling1D(),
            
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(len(self.MOOD_LABELS), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001
        )
        
        # Entraînement
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=100,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        self.model = model
        
        # Évaluation
        _, accuracy = model.evaluate(X_test, y_test, verbose=0)
        y_pred = np.argmax(model.predict(X_test), axis=1)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return ModelPerformance(
            accuracy=accuracy,
            precision={str(k): v['precision'] for k, v in report.items() if isinstance(v, dict)},
            recall={str(k): v['recall'] for k, v in report.items() if isinstance(v, dict)},
            f1_score={str(k): v['f1-score'] for k, v in report.items() if isinstance(v, dict)},
            confusion_matrix=confusion_matrix(y_test, y_pred).tolist(),
            cross_val_scores=[],
            training_time=0.0
        )
    
    async def predict(self, features: np.ndarray) -> ClassificationResult:
        """Prédit l'humeur d'un échantillon audio."""
        
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Préparation
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        if self.config.feature_scaling:
            features = self.scaler.transform(features)
        
        if features.ndim == 2:
            features = features.reshape(features.shape[0], features.shape[1], 1)
        
        # Prédiction
        predictions = self.model.predict(features)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        
        prob_dist = {
            self.label_encoder.inverse_transform([i])[0]: float(prob)
            for i, prob in enumerate(predictions[0])
        }
        
        return ClassificationResult(
            predicted_class=predicted_class,
            confidence=confidence,
            probability_distribution=prob_dist
        )

class InstrumentClassifier:
    """Classificateur d'instruments de musique."""
    
    INSTRUMENT_LABELS = [
        'guitar', 'piano', 'drums', 'bass', 'violin', 'saxophone',
        'trumpet', 'vocals', 'synthesizer', 'flute', 'cello', 'harmonica'
    ]
    
    def __init__(self, config: ClassificationConfig = None):
        self.config = config or ClassificationConfig(classification_type=ClassificationType.INSTRUMENT)
        self.models = {}
        self.scaler = StandardScaler()
    
    async def train_multi_label(
        self,
        features_df: pd.DataFrame,
        instrument_labels: pd.DataFrame
    ) -> Dict[str, ModelPerformance]:
        """Entraîne des classificateurs binaires pour chaque instrument."""
        
        performances = {}
        X = features_df.values
        
        if self.config.feature_scaling:
            X = self.scaler.fit_transform(X)
        
        # Entraînement d'un classificateur par instrument
        for instrument in self.INSTRUMENT_LABELS:
            if instrument in instrument_labels.columns:
                y = instrument_labels[instrument].values
                
                # Modèle spécialisé pour cet instrument
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.config.random_state,
                    n_jobs=-1
                )
                
                # Entraînement avec validation croisée
                scores = cross_val_score(model, X, y, cv=5, scoring='f1')
                model.fit(X, y)
                
                self.models[instrument] = model
                
                performances[instrument] = ModelPerformance(
                    accuracy=float(np.mean(scores)),
                    precision={},
                    recall={},
                    f1_score={},
                    confusion_matrix=[],
                    cross_val_scores=scores.tolist(),
                    training_time=0.0
                )
        
        return performances
    
    async def predict_instruments(self, features: np.ndarray) -> Dict[str, ClassificationResult]:
        """Prédit la présence de chaque instrument."""
        
        if not self.models:
            raise ValueError("Models not trained yet")
        
        # Préparation
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        if self.config.feature_scaling:
            features = self.scaler.transform(features)
        
        results = {}
        
        for instrument, model in self.models.items():
            # Prédiction binaire
            prediction = model.predict(features)[0]
            
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(features)[0]
                confidence = float(prob[1] if prediction == 1 else prob[0])
                prob_dist = {'absent': float(prob[0]), 'present': float(prob[1])}
            else:
                confidence = 0.8
                prob_dist = {}
            
            results[instrument] = ClassificationResult(
                predicted_class='present' if prediction == 1 else 'absent',
                confidence=confidence,
                probability_distribution=prob_dist
            )
        
        return results

# === Classificateur principal unifié ===
class AudioClassificationEngine:
    """
    Moteur de classification audio unifié.
    """
    
    def __init__(self):
        self.genre_classifier = None
        self.mood_classifier = None
        self.instrument_classifier = None
        self.loaded_models = {}
    
    async def load_pretrained_models(self, models_dir: str):
        """Charge des modèles pré-entraînés."""
        
        models_path = Path(models_dir)
        
        # Chargement du classificateur de genre
        genre_path = models_path / "genre_classifier"
        if genre_path.exists():
            self.genre_classifier = GenreClassifier()
            await self._load_genre_models(genre_path)
        
        # Chargement du classificateur d'humeur
        mood_path = models_path / "mood_classifier"
        if mood_path.exists():
            self.mood_classifier = MoodClassifier()
            await self._load_mood_models(mood_path)
        
        # Chargement du classificateur d'instruments
        instrument_path = models_path / "instrument_classifier"
        if instrument_path.exists():
            self.instrument_classifier = InstrumentClassifier()
            await self._load_instrument_models(instrument_path)
        
        logger.info("Pretrained models loaded successfully")
    
    async def _load_genre_models(self, model_path: Path):
        """Charge les modèles de genre."""
        try:
            # Chargement des modèles
            for model_file in model_path.glob("*.joblib"):
                if model_file.stem not in ['scaler', 'label_encoder']:
                    model = joblib.load(model_file)
                    self.genre_classifier.models[model_file.stem] = model
            
            # Chargement du scaler et label encoder
            self.genre_classifier.scaler = joblib.load(model_path / "scaler.joblib")
            self.genre_classifier.label_encoder = joblib.load(model_path / "label_encoder.joblib")
            
        except Exception as e:
            logger.error(f"Failed to load genre models: {e}")
    
    async def _load_mood_models(self, model_path: Path):
        """Charge les modèles d'humeur."""
        try:
            # Chargement du modèle CNN
            model_file = model_path / "mood_cnn_model.h5"
            if model_file.exists():
                self.mood_classifier.model = tf.keras.models.load_model(str(model_file))
            
            # Chargement du scaler
            scaler_file = model_path / "scaler.joblib"
            if scaler_file.exists():
                self.mood_classifier.scaler = joblib.load(scaler_file)
            
        except Exception as e:
            logger.error(f"Failed to load mood models: {e}")
    
    async def _load_instrument_models(self, model_path: Path):
        """Charge les modèles d'instruments."""
        try:
            for model_file in model_path.glob("*.joblib"):
                if model_file.stem != 'scaler':
                    model = joblib.load(model_file)
                    self.instrument_classifier.models[model_file.stem] = model
            
            # Chargement du scaler
            scaler_file = model_path / "scaler.joblib"
            if scaler_file.exists():
                self.instrument_classifier.scaler = joblib.load(scaler_file)
                
        except Exception as e:
            logger.error(f"Failed to load instrument models: {e}")
    
    async def classify_audio(
        self,
        features: np.ndarray,
        classification_types: List[ClassificationType] = None
    ) -> Dict[str, Any]:
        """
        Classification complète d'un échantillon audio.
        
        Args:
            features: Vecteur de features extractées
            classification_types: Types de classification à effectuer
            
        Returns:
            Dictionnaire avec tous les résultats de classification
        """
        
        if classification_types is None:
            classification_types = [
                ClassificationType.GENRE,
                ClassificationType.MOOD,
                ClassificationType.INSTRUMENT
            ]
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'classifications': {}
        }
        
        # Classification de genre
        if (ClassificationType.GENRE in classification_types 
            and self.genre_classifier is not None):
            try:
                genre_result = await self.genre_classifier.predict(features)
                results['classifications']['genre'] = asdict(genre_result)
            except Exception as e:
                logger.error(f"Genre classification failed: {e}")
                results['classifications']['genre'] = {'error': str(e)}
        
        # Classification d'humeur
        if (ClassificationType.MOOD in classification_types 
            and self.mood_classifier is not None):
            try:
                mood_result = await self.mood_classifier.predict(features)
                results['classifications']['mood'] = asdict(mood_result)
            except Exception as e:
                logger.error(f"Mood classification failed: {e}")
                results['classifications']['mood'] = {'error': str(e)}
        
        # Classification d'instruments
        if (ClassificationType.INSTRUMENT in classification_types 
            and self.instrument_classifier is not None):
            try:
                instrument_results = await self.instrument_classifier.predict_instruments(features)
                results['classifications']['instruments'] = {
                    inst: asdict(res) for inst, res in instrument_results.items()
                }
            except Exception as e:
                logger.error(f"Instrument classification failed: {e}")
                results['classifications']['instruments'] = {'error': str(e)}
        
        return results
    
    def get_available_classifiers(self) -> List[str]:
        """Retourne la liste des classificateurs disponibles."""
        available = []
        
        if self.genre_classifier is not None:
            available.append('genre')
        if self.mood_classifier is not None:
            available.append('mood')
        if self.instrument_classifier is not None:
            available.append('instrument')
        
        return available

# === Factory functions ===
def create_genre_classifier(
    model_type: ModelType = ModelType.ENSEMBLE
) -> GenreClassifier:
    """Factory pour créer un classificateur de genre."""
    config = ClassificationConfig(
        model_type=model_type,
        classification_type=ClassificationType.GENRE
    )
    return GenreClassifier(config)

def create_mood_classifier() -> MoodClassifier:
    """Factory pour créer un classificateur d'humeur."""
    config = ClassificationConfig(
        model_type=ModelType.DEEP_CNN,
        classification_type=ClassificationType.MOOD
    )
    return MoodClassifier(config)

def create_instrument_classifier() -> InstrumentClassifier:
    """Factory pour créer un classificateur d'instruments."""
    config = ClassificationConfig(
        model_type=ModelType.RANDOM_FOREST,
        classification_type=ClassificationType.INSTRUMENT
    )
    return InstrumentClassifier(config)

def create_classification_engine() -> AudioClassificationEngine:
    """Factory pour créer le moteur de classification complet."""
    return AudioClassificationEngine()
