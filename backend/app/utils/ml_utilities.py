"""
Enterprise ML Utilities
=======================
Advanced machine learning utilities for Spotify AI Agent streaming platform.

Expert Team Implementation:
- Lead Developer + AI Architect: AutoML pipelines and model optimization
- Machine Learning Engineer: TensorFlow/PyTorch model management and training
- Senior Backend Developer: Async ML serving and high-performance inference
- DBA & Data Engineer: Feature stores and ML data pipelines
- Security Specialist: Model security and privacy-preserving ML
- Microservices Architect: Distributed ML serving and model versioning
"""

import asyncio
import logging
import json
import pickle
import joblib
import hashlib
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple, Callable, Iterator
from abc import ABC, abstractmethod
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from pathlib import Path
import tempfile
import shutil

# ML framework imports
try:
    import sklearn
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    sklearn = None

# Deep learning imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

# MLOps imports
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

logger = logging.getLogger(__name__)

# === ML Types and Enums ===
class ModelType(Enum):
    """Supported model types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    RECOMMENDATION = "recommendation"
    ANOMALY_DETECTION = "anomaly_detection"
    NLP = "nlp"
    AUDIO = "audio"
    COMPUTER_VISION = "computer_vision"

class ModelFramework(Enum):
    """Supported ML frameworks."""
    SKLEARN = "sklearn"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"

class ModelStatus(Enum):
    """Model lifecycle status."""
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATING = "validating"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"

@dataclass
class ModelMetrics:
    """Model performance metrics."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_roc: float = 0.0
    loss: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
@dataclass
class ModelInfo:
    """Model information and metadata."""
    model_id: str
    name: str
    version: str
    model_type: ModelType
    framework: ModelFramework
    status: ModelStatus
    created_at: datetime
    updated_at: datetime
    metrics: ModelMetrics
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    features: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    description: str = ""

@dataclass
class TrainingResult:
    """Training operation result."""
    success: bool
    model_info: Optional[ModelInfo] = None
    metrics: Optional[ModelMetrics] = None
    training_history: Dict[str, List[float]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    training_time_seconds: float = 0.0

@dataclass
class PredictionResult:
    """Prediction operation result."""
    success: bool
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    confidence_scores: Optional[np.ndarray] = None
    model_id: str = ""
    prediction_time_ms: float = 0.0
    errors: List[str] = field(default_factory=list)

# === Feature Engineering ===
class FeatureExtractor:
    """Advanced feature extraction for ML models."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.feature_transformers = {}
        self.feature_history = deque(maxlen=1000)
        
    async def extract_features(self, data: Any, feature_type: str = "auto") -> Dict[str, Any]:
        """Extract features from various data types."""
        start_time = datetime.now()
        
        try:
            if feature_type == "audio" or self._is_audio_data(data):
                features = await self._extract_audio_features(data)
            elif feature_type == "text" or isinstance(data, str):
                features = await self._extract_text_features(data)
            elif feature_type == "user_behavior" or self._is_user_behavior(data):
                features = await self._extract_user_features(data)
            elif feature_type == "playlist" or self._is_playlist_data(data):
                features = await self._extract_playlist_features(data)
            else:
                features = await self._extract_general_features(data)
            
            # Add metadata
            features['_extraction_time'] = (datetime.now() - start_time).total_seconds()
            features['_feature_count'] = len([k for k in features.keys() if not k.startswith('_')])
            features['_extraction_timestamp'] = datetime.now().isoformat()
            
            self.feature_history.append({
                'timestamp': datetime.now(),
                'feature_count': features['_feature_count'],
                'extraction_time': features['_extraction_time']
            })
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return {'error': str(e)}
    
    async def _extract_audio_features(self, data: Any) -> Dict[str, Any]:
        """Extract ML-ready features from audio data."""
        features = {}
        
        # Assume data is already processed audio features from AudioProcessor
        if isinstance(data, dict) and 'mfcc' in data:
            # Statistical features from MFCC
            mfcc = np.array(data['mfcc'])
            features.update({
                'mfcc_mean': np.mean(mfcc, axis=1).tolist(),
                'mfcc_std': np.std(mfcc, axis=1).tolist(),
                'mfcc_skew': self._calculate_skewness(mfcc).tolist(),
                'mfcc_kurtosis': self._calculate_kurtosis(mfcc).tolist()
            })
            
            # Temporal features
            if 'tempo' in data:
                features['tempo'] = data['tempo']
                features['tempo_stability'] = self._calculate_tempo_stability(data.get('beat_frames', []))
            
            # Spectral features
            if 'spectral_centroid' in data:
                sc = np.array(data['spectral_centroid'])
                features.update({
                    'spectral_centroid_mean': float(np.mean(sc)),
                    'spectral_centroid_std': float(np.std(sc)),
                    'spectral_centroid_range': float(np.max(sc) - np.min(sc))
                })
            
            # Harmonic features
            if 'chroma' in data:
                chroma = np.array(data['chroma'])
                features['chroma_vector'] = np.mean(chroma, axis=1).tolist()
                features['key_strength'] = float(np.max(np.mean(chroma, axis=1)))
            
            # Energy features
            if 'rms_energy' in data:
                rms = np.array(data['rms_energy'])
                features.update({
                    'energy_mean': float(np.mean(rms)),
                    'energy_std': float(np.std(rms)),
                    'energy_dynamics': float(np.std(rms) / np.mean(rms)) if np.mean(rms) > 0 else 0.0
                })
        
        return features
    
    async def _extract_text_features(self, text: str) -> Dict[str, Any]:
        """Extract features from text data (song titles, lyrics, etc.)."""
        features = {}
        
        # Basic text statistics
        features.update({
            'text_length': len(text),
            'word_count': len(text.split()),
            'char_count': len(text),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
            'sentence_count': len(text.split('.')),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'digit_ratio': sum(1 for c in text if c.isdigit()) / len(text) if text else 0,
            'punctuation_ratio': sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text) if text else 0
        })
        
        # Language pattern features
        common_words = ['the', 'and', 'you', 'love', 'me', 'I', 'to', 'a', 'of', 'in']
        text_lower = text.lower()
        features['common_words_count'] = sum(1 for word in common_words if word in text_lower)
        
        # Music-specific patterns
        music_keywords = ['love', 'heart', 'night', 'time', 'dance', 'music', 'song', 'beat']
        features['music_keywords_count'] = sum(1 for keyword in music_keywords if keyword in text_lower)
        
        return features
    
    async def _extract_user_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract user behavior features."""
        features = {}
        
        # Listening patterns
        if 'listening_history' in data:
            history = data['listening_history']
            features.update({
                'total_listening_time': sum(item.get('duration', 0) for item in history),
                'unique_artists': len(set(item.get('artist', '') for item in history)),
                'unique_genres': len(set(item.get('genre', '') for item in history)),
                'avg_song_duration': np.mean([item.get('duration', 0) for item in history]),
                'listening_frequency': len(history),
                'skip_rate': sum(1 for item in history if item.get('skipped', False)) / len(history) if history else 0
            })
        
        # Engagement features
        if 'interactions' in data:
            interactions = data['interactions']
            features.update({
                'likes_count': sum(1 for i in interactions if i.get('type') == 'like'),
                'shares_count': sum(1 for i in interactions if i.get('type') == 'share'),
                'playlist_additions': sum(1 for i in interactions if i.get('type') == 'add_to_playlist'),
                'search_frequency': sum(1 for i in interactions if i.get('type') == 'search')
            })
        
        # Temporal features
        if 'timestamps' in data:
            timestamps = [datetime.fromisoformat(ts) for ts in data['timestamps']]
            features.update({
                'peak_listening_hour': self._find_peak_listening_hour(timestamps),
                'weekend_listening_ratio': self._calculate_weekend_ratio(timestamps),
                'listening_consistency': self._calculate_listening_consistency(timestamps)
            })
        
        return features
    
    async def _extract_playlist_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract playlist analysis features."""
        features = {}
        
        if 'tracks' in data:
            tracks = data['tracks']
            
            # Diversity features
            artists = [track.get('artist', '') for track in tracks]
            genres = [track.get('genre', '') for track in tracks]
            
            features.update({
                'track_count': len(tracks),
                'artist_diversity': len(set(artists)) / len(tracks) if tracks else 0,
                'genre_diversity': len(set(genres)) / len(tracks) if tracks else 0,
                'avg_track_popularity': np.mean([track.get('popularity', 0) for track in tracks]),
                'total_duration': sum(track.get('duration', 0) for track in tracks),
                'avg_track_duration': np.mean([track.get('duration', 0) for track in tracks])
            })
            
            # Audio feature aggregations
            if all('audio_features' in track for track in tracks):
                audio_features = [track['audio_features'] for track in tracks]
                
                # Aggregate tempo, energy, etc.
                features.update({
                    'avg_tempo': np.mean([af.get('tempo', 120) for af in audio_features]),
                    'tempo_variance': np.var([af.get('tempo', 120) for af in audio_features]),
                    'avg_energy': np.mean([af.get('energy', 0.5) for af in audio_features]),
                    'energy_variance': np.var([af.get('energy', 0.5) for af in audio_features])
                })
        
        return features
    
    async def _extract_general_features(self, data: Any) -> Dict[str, Any]:
        """Extract general features from structured data."""
        features = {}
        
        if isinstance(data, dict):
            # Count different data types
            features.update({
                'dict_size': len(data),
                'string_fields': sum(1 for v in data.values() if isinstance(v, str)),
                'numeric_fields': sum(1 for v in data.values() if isinstance(v, (int, float))),
                'list_fields': sum(1 for v in data.values() if isinstance(v, list)),
                'null_fields': sum(1 for v in data.values() if v is None)
            })
            
            # Extract numeric values
            numeric_values = [v for v in data.values() if isinstance(v, (int, float))]
            if numeric_values:
                features.update({
                    'numeric_mean': float(np.mean(numeric_values)),
                    'numeric_std': float(np.std(numeric_values)),
                    'numeric_min': float(np.min(numeric_values)),
                    'numeric_max': float(np.max(numeric_values))
                })
        
        elif isinstance(data, list):
            features.update({
                'list_length': len(data),
                'unique_items': len(set(str(item) for item in data)),
                'avg_item_length': np.mean([len(str(item)) for item in data])
            })
        
        return features
    
    def _is_audio_data(self, data: Any) -> bool:
        """Check if data contains audio features."""
        return isinstance(data, dict) and any(
            key in data for key in ['mfcc', 'tempo', 'spectral_centroid', 'chroma']
        )
    
    def _is_user_behavior(self, data: Any) -> bool:
        """Check if data contains user behavior information."""
        return isinstance(data, dict) and any(
            key in data for key in ['listening_history', 'interactions', 'user_id']
        )
    
    def _is_playlist_data(self, data: Any) -> bool:
        """Check if data contains playlist information."""
        return isinstance(data, dict) and 'tracks' in data
    
    def _calculate_skewness(self, data: np.ndarray) -> np.ndarray:
        """Calculate skewness for each feature."""
        from scipy import stats
        return np.array([stats.skew(row) for row in data])
    
    def _calculate_kurtosis(self, data: np.ndarray) -> np.ndarray:
        """Calculate kurtosis for each feature."""
        from scipy import stats
        return np.array([stats.kurtosis(row) for row in data])
    
    def _calculate_tempo_stability(self, beat_frames: List[float]) -> float:
        """Calculate tempo stability from beat frames."""
        if len(beat_frames) < 2:
            return 1.0
        
        intervals = np.diff(beat_frames)
        return 1.0 / (1.0 + np.std(intervals)) if len(intervals) > 0 else 1.0
    
    def _find_peak_listening_hour(self, timestamps: List[datetime]) -> int:
        """Find peak listening hour."""
        hours = [ts.hour for ts in timestamps]
        if not hours:
            return 12
        
        hour_counts = defaultdict(int)
        for hour in hours:
            hour_counts[hour] += 1
        
        return max(hour_counts.items(), key=lambda x: x[1])[0]
    
    def _calculate_weekend_ratio(self, timestamps: List[datetime]) -> float:
        """Calculate weekend listening ratio."""
        if not timestamps:
            return 0.0
        
        weekend_count = sum(1 for ts in timestamps if ts.weekday() >= 5)
        return weekend_count / len(timestamps)
    
    def _calculate_listening_consistency(self, timestamps: List[datetime]) -> float:
        """Calculate listening consistency score."""
        if len(timestamps) < 2:
            return 1.0
        
        # Calculate time gaps between listening sessions
        sorted_timestamps = sorted(timestamps)
        gaps = [(sorted_timestamps[i+1] - sorted_timestamps[i]).total_seconds() 
                for i in range(len(sorted_timestamps)-1)]
        
        # Consistency is inversely related to gap variance
        if len(gaps) == 0:
            return 1.0
        
        gap_std = np.std(gaps)
        gap_mean = np.mean(gaps)
        
        return 1.0 / (1.0 + gap_std / gap_mean) if gap_mean > 0 else 1.0

# === Model Manager ===
class ModelManager:
    """Enterprise model lifecycle management."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.models = {}
        self.model_store_path = Path(config.get('model_store_path', './models'))
        self.model_store_path.mkdir(exist_ok=True)
        self.model_registry = {}
        self.performance_history = defaultdict(list)
        
    async def register_model(self, model: Any, model_info: ModelInfo) -> bool:
        """Register a new model."""
        try:
            # Save model to disk
            model_path = self.model_store_path / f"{model_info.model_id}_{model_info.version}"
            model_path.mkdir(exist_ok=True)
            
            # Save model based on framework
            if model_info.framework == ModelFramework.SKLEARN:
                joblib.dump(model, model_path / "model.pkl")
            elif model_info.framework == ModelFramework.PYTORCH and TORCH_AVAILABLE:
                torch.save(model.state_dict(), model_path / "model.pth")
            elif model_info.framework == ModelFramework.TENSORFLOW and TF_AVAILABLE:
                model.save(str(model_path / "model.h5"))
            else:
                # Fallback to pickle
                with open(model_path / "model.pkl", 'wb') as f:
                    pickle.dump(model, f)
            
            # Save model info
            with open(model_path / "info.json", 'w') as f:
                json.dump({
                    'model_id': model_info.model_id,
                    'name': model_info.name,
                    'version': model_info.version,
                    'model_type': model_info.model_type.value,
                    'framework': model_info.framework.value,
                    'status': model_info.status.value,
                    'created_at': model_info.created_at.isoformat(),
                    'updated_at': model_info.updated_at.isoformat(),
                    'hyperparameters': model_info.hyperparameters,
                    'features': model_info.features,
                    'tags': model_info.tags,
                    'description': model_info.description,
                    'metrics': {
                        'accuracy': model_info.metrics.accuracy,
                        'precision': model_info.metrics.precision,
                        'recall': model_info.metrics.recall,
                        'f1_score': model_info.metrics.f1_score,
                        'custom_metrics': model_info.metrics.custom_metrics
                    }
                }, indent=2)
            
            # Update registry
            self.model_registry[model_info.model_id] = model_info
            self.models[model_info.model_id] = model
            
            logger.info(f"Registered model {model_info.model_id} v{model_info.version}")
            return True
            
        except Exception as e:
            logger.error(f"Model registration error: {e}")
            return False
    
    async def load_model(self, model_id: str, version: str = None) -> Optional[Any]:
        """Load model from storage."""
        try:
            # Check if already loaded
            if model_id in self.models:
                return self.models[model_id]
            
            # Find model info
            if model_id not in self.model_registry:
                await self._scan_model_store()
            
            if model_id not in self.model_registry:
                logger.error(f"Model {model_id} not found in registry")
                return None
            
            model_info = self.model_registry[model_id]
            model_version = version or model_info.version
            
            # Load model from disk
            model_path = self.model_store_path / f"{model_id}_{model_version}"
            
            if model_info.framework == ModelFramework.SKLEARN:
                model = joblib.load(model_path / "model.pkl")
            elif model_info.framework == ModelFramework.PYTORCH and TORCH_AVAILABLE:
                # Would need model architecture to load PyTorch model
                with open(model_path / "model.pkl", 'rb') as f:
                    model = pickle.load(f)
            elif model_info.framework == ModelFramework.TENSORFLOW and TF_AVAILABLE:
                model = tf.keras.models.load_model(str(model_path / "model.h5"))
            else:
                with open(model_path / "model.pkl", 'rb') as f:
                    model = pickle.load(f)
            
            self.models[model_id] = model
            logger.info(f"Loaded model {model_id} v{model_version}")
            return model
            
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            return None
    
    async def predict(self, model_id: str, features: np.ndarray, **kwargs) -> PredictionResult:
        """Make predictions using registered model."""
        start_time = datetime.now()
        
        try:
            # Load model if not in memory
            model = await self.load_model(model_id)
            if model is None:
                return PredictionResult(
                    success=False,
                    errors=[f"Model {model_id} not found or failed to load"]
                )
            
            # Make predictions
            if hasattr(model, 'predict'):
                predictions = model.predict(features)
            else:
                return PredictionResult(
                    success=False,
                    errors=[f"Model {model_id} does not support prediction"]
                )
            
            # Get probabilities if available
            probabilities = None
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(features)
                except:
                    pass
            
            # Calculate confidence scores
            confidence_scores = None
            if probabilities is not None:
                confidence_scores = np.max(probabilities, axis=1)
            
            prediction_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return PredictionResult(
                success=True,
                predictions=predictions,
                probabilities=probabilities,
                confidence_scores=confidence_scores,
                model_id=model_id,
                prediction_time_ms=prediction_time
            )
            
        except Exception as e:
            logger.error(f"Prediction error for model {model_id}: {e}")
            return PredictionResult(
                success=False,
                model_id=model_id,
                errors=[str(e)],
                prediction_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
    
    async def _scan_model_store(self):
        """Scan model store and update registry."""
        try:
            for model_dir in self.model_store_path.iterdir():
                if model_dir.is_dir():
                    info_file = model_dir / "info.json"
                    if info_file.exists():
                        with open(info_file, 'r') as f:
                            info_data = json.load(f)
                        
                        model_info = ModelInfo(
                            model_id=info_data['model_id'],
                            name=info_data['name'],
                            version=info_data['version'],
                            model_type=ModelType(info_data['model_type']),
                            framework=ModelFramework(info_data['framework']),
                            status=ModelStatus(info_data['status']),
                            created_at=datetime.fromisoformat(info_data['created_at']),
                            updated_at=datetime.fromisoformat(info_data['updated_at']),
                            metrics=ModelMetrics(**info_data['metrics']),
                            hyperparameters=info_data.get('hyperparameters', {}),
                            features=info_data.get('features', []),
                            tags=info_data.get('tags', []),
                            description=info_data.get('description', '')
                        )
                        
                        self.model_registry[model_info.model_id] = model_info
                        
        except Exception as e:
            logger.error(f"Model store scan error: {e}")
    
    async def list_models(self) -> List[ModelInfo]:
        """List all registered models."""
        await self._scan_model_store()
        return list(self.model_registry.values())
    
    async def delete_model(self, model_id: str) -> bool:
        """Delete model from storage and registry."""
        try:
            if model_id in self.model_registry:
                model_info = self.model_registry[model_id]
                model_path = self.model_store_path / f"{model_id}_{model_info.version}"
                
                if model_path.exists():
                    shutil.rmtree(model_path)
                
                del self.model_registry[model_id]
                
                if model_id in self.models:
                    del self.models[model_id]
                
                logger.info(f"Deleted model {model_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Model deletion error: {e}")
            return False

# === AutoML Helper ===
class AutoMLHelper:
    """Automated machine learning for music recommendation and analysis."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.feature_extractor = FeatureExtractor(config)
        self.model_manager = ModelManager(config)
        
    async def auto_train_classifier(self, 
                                   training_data: List[Dict[str, Any]], 
                                   target_column: str,
                                   **kwargs) -> TrainingResult:
        """Automatically train classification model."""
        start_time = datetime.now()
        
        if not SKLEARN_AVAILABLE:
            return TrainingResult(
                success=False,
                errors=["Scikit-learn not available for AutoML"]
            )
        
        try:
            # Extract features
            logger.info("Extracting features for AutoML training...")
            features_list = []
            targets = []
            
            for item in training_data:
                features = await self.feature_extractor.extract_features(item)
                if 'error' not in features:
                    # Remove metadata fields
                    clean_features = {k: v for k, v in features.items() if not k.startswith('_')}
                    features_list.append(clean_features)
                    targets.append(item[target_column])
            
            if not features_list:
                return TrainingResult(
                    success=False,
                    errors=["No valid features extracted from training data"]
                )
            
            # Convert to DataFrame for easier handling
            df_features = pd.DataFrame(features_list)
            df_features = df_features.fillna(0)  # Handle missing values
            
            # Prepare data
            X = df_features.values
            y = np.array(targets)
            
            # Encode labels if categorical
            label_encoder = None
            if y.dtype == 'object':
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Try multiple algorithms
            models = {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingClassifier(random_state=42),
                'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
            }
            
            best_model = None
            best_score = 0
            best_name = ""
            training_history = {}
            
            for name, model in models.items():
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                training_history[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
                
                if accuracy > best_score:
                    best_score = accuracy
                    best_model = Pipeline([
                        ('scaler', scaler),
                        ('classifier', model)
                    ])
                    best_name = name
            
            # Create model info
            model_id = f"automl_{target_column}_{int(datetime.now().timestamp())}"
            model_info = ModelInfo(
                model_id=model_id,
                name=f"AutoML {target_column} Classifier",
                version="1.0",
                model_type=ModelType.CLASSIFICATION,
                framework=ModelFramework.SKLEARN,
                status=ModelStatus.TRAINED,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                metrics=ModelMetrics(
                    accuracy=training_history[best_name]['accuracy'],
                    precision=training_history[best_name]['precision'],
                    recall=training_history[best_name]['recall'],
                    f1_score=training_history[best_name]['f1_score']
                ),
                features=list(df_features.columns),
                tags=['automl', 'classification', target_column],
                description=f"Automatically trained {best_name} classifier for {target_column}"
            )
            
            # Register model
            await self.model_manager.register_model(best_model, model_info)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"AutoML training completed. Best model: {best_name} (accuracy: {best_score:.3f})")
            
            return TrainingResult(
                success=True,
                model_info=model_info,
                metrics=model_info.metrics,
                training_history=training_history,
                training_time_seconds=training_time
            )
            
        except Exception as e:
            logger.error(f"AutoML training error: {e}")
            return TrainingResult(
                success=False,
                errors=[str(e)],
                training_time_seconds=(datetime.now() - start_time).total_seconds()
            )
    
    async def auto_recommend_songs(self, 
                                  user_data: Dict[str, Any], 
                                  candidate_songs: List[Dict[str, Any]],
                                  top_k: int = 10) -> List[Dict[str, Any]]:
        """Auto-generate song recommendations."""
        try:
            # Extract user features
            user_features = await self.feature_extractor.extract_features(
                user_data, feature_type="user_behavior"
            )
            
            # Extract song features
            song_features_list = []
            for song in candidate_songs:
                features = await self.feature_extractor.extract_features(song, feature_type="audio")
                if 'error' not in features:
                    song_features_list.append({
                        'song': song,
                        'features': features
                    })
            
            # Simple recommendation based on feature similarity
            recommendations = []
            
            for song_data in song_features_list:
                similarity_score = await self._calculate_similarity(
                    user_features, song_data['features']
                )
                
                recommendations.append({
                    'song': song_data['song'],
                    'score': similarity_score,
                    'features': song_data['features']
                })
            
            # Sort by similarity score and return top K
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            return recommendations[:top_k]
            
        except Exception as e:
            logger.error(f"Auto-recommendation error: {e}")
            return []
    
    async def _calculate_similarity(self, user_features: Dict[str, Any], song_features: Dict[str, Any]) -> float:
        """Calculate similarity between user preferences and song features."""
        # Simplified similarity calculation
        # In production, use more sophisticated methods like cosine similarity
        
        # Extract numeric features
        user_numeric = {k: v for k, v in user_features.items() 
                       if isinstance(v, (int, float)) and not k.startswith('_')}
        song_numeric = {k: v for k, v in song_features.items() 
                       if isinstance(v, (int, float)) and not k.startswith('_')}
        
        if not user_numeric or not song_numeric:
            return 0.0
        
        # Calculate normalized distance
        common_features = set(user_numeric.keys()) & set(song_numeric.keys())
        if not common_features:
            return 0.0
        
        distances = []
        for feature in common_features:
            user_val = user_numeric[feature]
            song_val = song_numeric[feature]
            
            # Normalize to 0-1 range (simplified)
            max_val = max(abs(user_val), abs(song_val), 1.0)
            normalized_distance = abs(user_val - song_val) / max_val
            distances.append(normalized_distance)
        
        # Return similarity (1 - average distance)
        avg_distance = np.mean(distances)
        return max(0.0, 1.0 - avg_distance)

# === Data Pipeline ===
class DataPipeline:
    """ML data preprocessing and feature engineering pipeline."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.steps = []
        self.feature_extractor = FeatureExtractor(config)
        
    def add_step(self, step_func: Callable, **kwargs):
        """Add processing step to pipeline."""
        self.steps.append({'func': step_func, 'kwargs': kwargs})
    
    async def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process data through pipeline."""
        result = data
        
        for step in self.steps:
            try:
                result = await step['func'](result, **step['kwargs'])
            except Exception as e:
                logger.error(f"Pipeline step error: {e}")
                continue
        
        return result
    
    async def extract_features_batch(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract features from batch of data."""
        results = []
        
        for item in data:
            features = await self.feature_extractor.extract_features(item)
            if 'error' not in features:
                results.append({**item, 'extracted_features': features})
            else:
                results.append(item)
        
        return results

# === Factory Functions ===
def create_feature_extractor(config: Dict[str, Any] = None) -> FeatureExtractor:
    """Create feature extractor instance."""
    return FeatureExtractor(config)

def create_model_manager(config: Dict[str, Any] = None) -> ModelManager:
    """Create model manager instance."""
    return ModelManager(config)

def create_automl_helper(config: Dict[str, Any] = None) -> AutoMLHelper:
    """Create AutoML helper instance."""
    return AutoMLHelper(config)

def create_data_pipeline(config: Dict[str, Any] = None) -> DataPipeline:
    """Create data pipeline instance."""
    return DataPipeline(config)

# === Export Classes ===
__all__ = [
    'FeatureExtractor', 'ModelManager', 'AutoMLHelper', 'DataPipeline',
    'ModelType', 'ModelFramework', 'ModelStatus', 'ModelMetrics', 
    'ModelInfo', 'TrainingResult', 'PredictionResult',
    'create_feature_extractor', 'create_model_manager', 
    'create_automl_helper', 'create_data_pipeline'
]
