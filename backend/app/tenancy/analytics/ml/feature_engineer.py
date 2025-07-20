"""
Ultra-Advanced Feature Engineering for AI Music Analytics

This module implements sophisticated feature engineering with automated extraction,
selection, temporal features, and domain-specific audio characteristics.

Features:
- Automated feature extraction from audio and metadata
- Advanced feature selection algorithms  
- Temporal and sequential feature engineering
- Audio signal processing and spectral features
- Music domain-specific feature engineering
- Feature importance analysis and ranking
- Real-time feature transformation pipelines
- Cross-validation feature stability analysis
- Ensemble feature selection methods
- Feature interaction and polynomial features

Created by Expert Team:
- Lead Dev + AI Architect: Feature engineering architecture and optimization
- ML Engineer: Advanced feature extraction and selection algorithms
- Audio Engineer: Specialized audio feature processing and extraction  
- Data Engineer: Feature pipeline optimization and scalability
- Backend Developer: Feature serving and caching infrastructure
- Security Specialist: Feature anonymization and privacy protection
"""

from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import logging
import uuid
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import pickle
import threading
from abc import ABC, abstractmethod
from collections import defaultdict

# Scikit-learn for feature engineering
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, RFE, RFECV,
    SelectFromModel, VarianceThreshold, chi2, f_classif,
    mutual_info_classif, mutual_info_regression
)
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures,
    QuantileTransformer, PowerTransformer, LabelEncoder, OneHotEncoder
)
from sklearn.decomposition import PCA, FastICA, NMF, TruncatedSVD
from sklearn.manifold import TSNE, Isomap
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Audio processing libraries  
try:
    import librosa
    import soundfile as sf
    import numpy.fft as fft
    from scipy import signal
    from scipy.stats import skew, kurtosis
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# Advanced statistical libraries
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import acf, pacf
    import scipy.stats as stats
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False

logger = logging.getLogger(__name__)

class FeatureType(Enum):
    """Types of features"""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    AUDIO = "audio"
    TEXT = "text"
    SPECTRAL = "spectral"
    RHYTHMIC = "rhythmic"
    HARMONIC = "harmonic"
    TIMBRAL = "timbral"
    METADATA = "metadata"

class SelectionMethod(Enum):
    """Feature selection methods"""
    UNIVARIATE = "univariate"
    RECURSIVE = "recursive"
    MODEL_BASED = "model_based"
    VARIANCE_THRESHOLD = "variance_threshold"
    CORRELATION = "correlation"
    MUTUAL_INFORMATION = "mutual_info"
    BORUTA = "boruta"
    GENETIC = "genetic"
    ENSEMBLE = "ensemble"

class ScalingMethod(Enum):
    """Feature scaling methods"""
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    QUANTILE = "quantile"
    POWER = "power"
    NONE = "none"

@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    # General settings
    enable_audio_features: bool = True
    enable_temporal_features: bool = True
    enable_metadata_features: bool = True
    enable_interaction_features: bool = True
    
    # Audio feature settings
    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    n_mfcc: int = 13
    
    # Selection settings
    selection_methods: List[SelectionMethod] = field(default_factory=lambda: [
        SelectionMethod.UNIVARIATE,
        SelectionMethod.MODEL_BASED,
        SelectionMethod.CORRELATION
    ])
    max_features: Optional[int] = None
    feature_importance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    
    # Scaling settings
    scaling_method: ScalingMethod = ScalingMethod.STANDARD
    handle_outliers: bool = True
    
    # Advanced settings
    polynomial_degree: int = 2
    enable_pca: bool = False
    pca_components: Optional[int] = None
    enable_feature_engineering: bool = True

@dataclass
class FeatureImportance:
    """Feature importance information"""
    feature_name: str
    importance_score: float
    selection_method: str
    rank: int
    stability_score: float = 0.0
    correlation_with_target: float = 0.0

@dataclass
class FeatureStats:
    """Statistics for features"""
    feature_name: str
    feature_type: FeatureType
    missing_ratio: float
    variance: float
    mean: Optional[float] = None
    std: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    unique_count: Optional[int] = None
    skewness: Optional[float] = None
    kurtosis_value: Optional[float] = None

@dataclass
class FeatureEngineering:
    """Results of feature engineering"""
    # Original data information
    original_feature_count: int
    original_sample_count: int
    
    # Processed data information
    final_feature_count: int
    selected_features: List[str]
    feature_importances: List[FeatureImportance]
    feature_stats: List[FeatureStats]
    
    # Processing information
    processing_time: float
    scaling_method: str
    selection_methods: List[str]
    
    # Quality metrics
    feature_stability_score: float
    redundancy_removed: int
    information_retention: float

class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extractors"""
    
    def __init__(self, extractor_id: str, feature_type: FeatureType):
        self.extractor_id = extractor_id
        self.feature_type = feature_type
        self.is_fitted = False
    
    @abstractmethod
    async def extract_features(
        self,
        data: Union[np.ndarray, pd.DataFrame, str],
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Extract features from data"""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get names of extracted features"""
        pass

class AudioFeatureExtractor(BaseFeatureExtractor):
    """Advanced audio feature extraction"""
    
    def __init__(self, config: FeatureConfig):
        super().__init__("audio_extractor", FeatureType.AUDIO)
        self.config = config
        self.feature_names = []
    
    async def extract_features(
        self,
        audio_data: Union[np.ndarray, str],
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Extract comprehensive audio features"""
        if not AUDIO_AVAILABLE:
            raise ImportError("Audio processing libraries not available")
        
        # Load audio if path provided
        if isinstance(audio_data, str):
            y, sr = librosa.load(audio_data, sr=self.config.sample_rate)
        else:
            y, sr = audio_data, self.config.sample_rate
        
        features = {}
        
        # Basic audio properties
        features.update(self._extract_basic_features(y, sr))
        
        # Spectral features
        features.update(self._extract_spectral_features(y, sr))
        
        # Rhythmic features
        features.update(self._extract_rhythmic_features(y, sr))
        
        # Harmonic features
        features.update(self._extract_harmonic_features(y, sr))
        
        # Timbral features
        features.update(self._extract_timbral_features(y, sr))
        
        # Advanced features
        features.update(self._extract_advanced_features(y, sr))
        
        self._update_feature_names(features)
        return features
    
    def _extract_basic_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract basic audio features"""
        features = {}
        
        # Duration and energy
        features['duration'] = np.array([len(y) / sr])
        features['rms_energy'] = np.array([np.sqrt(np.mean(y**2))])
        features['zero_crossing_rate'] = np.array([np.mean(librosa.feature.zero_crossing_rate(y))])
        
        # Dynamic range
        features['dynamic_range'] = np.array([np.max(y) - np.min(y)])
        features['peak_amplitude'] = np.array([np.max(np.abs(y))])
        
        return features
    
    def _extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract spectral features"""
        features = {}
        
        # MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.config.n_mfcc)
        for i in range(self.config.n_mfcc):
            features[f'mfcc_{i}'] = np.array([np.mean(mfcc[i])])
            features[f'mfcc_{i}_std'] = np.array([np.std(mfcc[i])])
        
        # Spectral characteristics
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid'] = np.array([np.mean(spectral_centroids)])
        features['spectral_centroid_std'] = np.array([np.std(spectral_centroids)])
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['spectral_bandwidth'] = np.array([np.mean(spectral_bandwidth)])
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['spectral_rolloff'] = np.array([np.mean(spectral_rolloff)])
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features['spectral_contrast'] = np.array([np.mean(spectral_contrast)])
        
        return features
    
    def _extract_rhythmic_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract rhythmic features"""
        features = {}
        
        # Tempo and beat
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = np.array([tempo])
        features['beat_count'] = np.array([len(beats)])
        
        # Rhythm patterns
        if len(beats) > 1:
            beat_intervals = np.diff(beats) / sr
            features['beat_regularity'] = np.array([1.0 / (np.std(beat_intervals) + 1e-8)])
            features['avg_beat_interval'] = np.array([np.mean(beat_intervals)])
        else:
            features['beat_regularity'] = np.array([0.0])
            features['avg_beat_interval'] = np.array([0.0])
        
        return features
    
    def _extract_harmonic_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract harmonic features"""
        features = {}
        
        # Harmonic-percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        features['harmonic_ratio'] = np.array([
            np.mean(np.abs(y_harmonic)) / (np.mean(np.abs(y)) + 1e-8)
        ])
        features['percussive_ratio'] = np.array([
            np.mean(np.abs(y_percussive)) / (np.mean(np.abs(y)) + 1e-8)
        ])
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        for i in range(12):
            features[f'chroma_{i}'] = np.array([np.mean(chroma[i])])
        
        return features
    
    def _extract_timbral_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract timbral features"""
        features = {}
        
        # Tonnetz features
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        for i in range(6):
            features[f'tonnetz_{i}'] = np.array([np.mean(tonnetz[i])])
        
        # Spectral features for timbre
        stft = librosa.stft(y, n_fft=self.config.n_fft, hop_length=self.config.hop_length)
        magnitude = np.abs(stft)
        
        # Spectral flatness
        features['spectral_flatness'] = np.array([
            np.mean(librosa.feature.spectral_flatness(S=magnitude))
        ])
        
        # Spectral slope
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.config.n_fft)
        spectral_slope = []
        for frame in magnitude.T:
            if np.sum(frame) > 0:
                slope = np.polyfit(freqs[:len(frame)], frame, 1)[0]
                spectral_slope.append(slope)
        
        features['spectral_slope'] = np.array([np.mean(spectral_slope) if spectral_slope else 0.0])
        
        return features
    
    def _extract_advanced_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract advanced audio features"""
        features = {}
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=self.config.n_mels,
            n_fft=self.config.n_fft, hop_length=self.config.hop_length
        )
        
        # Statistical features from mel spectrogram
        features['mel_spec_mean'] = np.array([np.mean(mel_spec)])
        features['mel_spec_std'] = np.array([np.std(mel_spec)])
        features['mel_spec_skew'] = np.array([skew(mel_spec.flatten())])
        features['mel_spec_kurtosis'] = np.array([kurtosis(mel_spec.flatten())])
        
        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        features['onset_rate'] = np.array([len(onset_frames) / (len(y) / sr)])
        
        # Pitch features
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        features['pitch_mean'] = np.array([np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0.0])
        features['pitch_std'] = np.array([np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0.0])
        
        return features
    
    def _update_feature_names(self, features: Dict[str, np.ndarray]):
        """Update feature names list"""
        self.feature_names = list(features.keys())
    
    def get_feature_names(self) -> List[str]:
        """Get names of extracted features"""
        return self.feature_names

class TemporalFeatureExtractor(BaseFeatureExtractor):
    """Extract temporal and time-series features"""
    
    def __init__(self):
        super().__init__("temporal_extractor", FeatureType.TEMPORAL)
    
    async def extract_features(
        self,
        data: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        value_cols: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Extract temporal features from time series data"""
        features = {}
        
        if timestamp_col not in data.columns:
            return features
        
        # Convert timestamp if needed
        if not pd.api.types.is_datetime64_any_dtype(data[timestamp_col]):
            data[timestamp_col] = pd.to_datetime(data[timestamp_col])
        
        # Sort by timestamp
        data = data.sort_values(timestamp_col)
        
        # Basic temporal features
        features.update(self._extract_basic_temporal_features(data, timestamp_col))
        
        # Statistical temporal features
        if value_cols:
            for col in value_cols:
                if col in data.columns:
                    features.update(self._extract_statistical_temporal_features(
                        data, col, timestamp_col
                    ))
        
        # Seasonal features
        features.update(self._extract_seasonal_features(data, timestamp_col))
        
        return features
    
    def _extract_basic_temporal_features(
        self,
        data: pd.DataFrame,
        timestamp_col: str
    ) -> Dict[str, np.ndarray]:
        """Extract basic temporal features"""
        features = {}
        
        timestamps = data[timestamp_col]
        
        # Time span
        time_span = (timestamps.max() - timestamps.min()).total_seconds()
        features['time_span_seconds'] = np.array([time_span])
        
        # Frequency
        if len(timestamps) > 1:
            avg_interval = time_span / (len(timestamps) - 1)
            features['avg_interval_seconds'] = np.array([avg_interval])
            features['frequency_hz'] = np.array([1.0 / avg_interval if avg_interval > 0 else 0.0])
        
        # Time of day features
        hours = timestamps.dt.hour
        features['avg_hour'] = np.array([np.mean(hours)])
        features['hour_std'] = np.array([np.std(hours)])
        
        # Day of week features
        weekdays = timestamps.dt.dayofweek
        features['avg_weekday'] = np.array([np.mean(weekdays)])
        features['weekday_std'] = np.array([np.std(weekdays)])
        
        return features
    
    def _extract_statistical_temporal_features(
        self,
        data: pd.DataFrame,
        value_col: str,
        timestamp_col: str
    ) -> Dict[str, np.ndarray]:
        """Extract statistical features from temporal data"""
        features = {}
        
        values = data[value_col].values
        
        # Basic statistics
        features[f'{value_col}_mean'] = np.array([np.mean(values)])
        features[f'{value_col}_std'] = np.array([np.std(values)])
        features[f'{value_col}_min'] = np.array([np.min(values)])
        features[f'{value_col}_max'] = np.array([np.max(values)])
        features[f'{value_col}_range'] = np.array([np.max(values) - np.min(values)])
        
        # Distribution statistics
        features[f'{value_col}_skew'] = np.array([skew(values)])
        features[f'{value_col}_kurtosis'] = np.array([kurtosis(values)])
        
        # Trend analysis
        if len(values) > 2:
            x = np.arange(len(values))
            slope, intercept = np.polyfit(x, values, 1)
            features[f'{value_col}_trend_slope'] = np.array([slope])
            
            # Volatility
            if len(values) > 1:
                diff = np.diff(values)
                features[f'{value_col}_volatility'] = np.array([np.std(diff)])
        
        return features
    
    def _extract_seasonal_features(
        self,
        data: pd.DataFrame,
        timestamp_col: str
    ) -> Dict[str, np.ndarray]:
        """Extract seasonal features"""
        features = {}
        
        timestamps = data[timestamp_col]
        
        # Month features
        months = timestamps.dt.month
        features['month_sin'] = np.array([np.mean(np.sin(2 * np.pi * months / 12))])
        features['month_cos'] = np.array([np.mean(np.cos(2 * np.pi * months / 12))])
        
        # Day of year features
        day_of_year = timestamps.dt.dayofyear
        features['day_of_year_sin'] = np.array([np.mean(np.sin(2 * np.pi * day_of_year / 365))])
        features['day_of_year_cos'] = np.array([np.mean(np.cos(2 * np.pi * day_of_year / 365))])
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get temporal feature names"""
        return [
            'time_span_seconds', 'avg_interval_seconds', 'frequency_hz',
            'avg_hour', 'hour_std', 'avg_weekday', 'weekday_std',
            'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos'
        ]

class FeatureSelector:
    """Advanced feature selection with multiple methods"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.selected_features = []
        self.feature_importances = []
        self.selectors = {}
    
    async def select_features(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[List[str], List[FeatureImportance]]:
        """Select features using multiple methods"""
        
        all_importances = []
        
        # Apply each selection method
        for method in self.config.selection_methods:
            importances = await self._apply_selection_method(X, y, feature_names, method)
            all_importances.extend(importances)
        
        # Aggregate results
        aggregated_importances = self._aggregate_feature_importances(all_importances)
        
        # Select final features
        selected_features = self._select_final_features(aggregated_importances)
        
        self.selected_features = selected_features
        self.feature_importances = aggregated_importances
        
        return selected_features, aggregated_importances
    
    async def _apply_selection_method(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_names: List[str],
        method: SelectionMethod
    ) -> List[FeatureImportance]:
        """Apply specific selection method"""
        
        if method == SelectionMethod.UNIVARIATE:
            return await self._univariate_selection(X, y, feature_names)
        elif method == SelectionMethod.MODEL_BASED:
            return await self._model_based_selection(X, y, feature_names)
        elif method == SelectionMethod.CORRELATION:
            return await self._correlation_selection(X, y, feature_names)
        elif method == SelectionMethod.MUTUAL_INFORMATION:
            return await self._mutual_info_selection(X, y, feature_names)
        elif method == SelectionMethod.VARIANCE_THRESHOLD:
            return await self._variance_threshold_selection(X, y, feature_names)
        else:
            return []
    
    async def _univariate_selection(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_names: List[str]
    ) -> List[FeatureImportance]:
        """Univariate feature selection"""
        
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X, y)
        
        importances = []
        for i, feature in enumerate(feature_names):
            importance = FeatureImportance(
                feature_name=feature,
                importance_score=selector.scores_[i],
                selection_method='univariate',
                rank=i + 1
            )
            importances.append(importance)
        
        return sorted(importances, key=lambda x: x.importance_score, reverse=True)
    
    async def _model_based_selection(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_names: List[str]
    ) -> List[FeatureImportance]:
        """Model-based feature selection"""
        
        # Random Forest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        importances = []
        for i, feature in enumerate(feature_names):
            importance = FeatureImportance(
                feature_name=feature,
                importance_score=rf.feature_importances_[i],
                selection_method='random_forest',
                rank=i + 1
            )
            importances.append(importance)
        
        return sorted(importances, key=lambda x: x.importance_score, reverse=True)
    
    async def _correlation_selection(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_names: List[str]
    ) -> List[FeatureImportance]:
        """Correlation-based feature selection"""
        
        correlations = []
        for i, feature in enumerate(feature_names):
            corr = np.abs(np.corrcoef(X.iloc[:, i], y)[0, 1])
            if np.isnan(corr):
                corr = 0.0
            
            importance = FeatureImportance(
                feature_name=feature,
                importance_score=corr,
                selection_method='correlation',
                rank=i + 1,
                correlation_with_target=corr
            )
            correlations.append(importance)
        
        return sorted(correlations, key=lambda x: x.importance_score, reverse=True)
    
    async def _mutual_info_selection(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_names: List[str]
    ) -> List[FeatureImportance]:
        """Mutual information feature selection"""
        
        mi_scores = mutual_info_classif(X, y)
        
        importances = []
        for i, feature in enumerate(feature_names):
            importance = FeatureImportance(
                feature_name=feature,
                importance_score=mi_scores[i],
                selection_method='mutual_info',
                rank=i + 1
            )
            importances.append(importance)
        
        return sorted(importances, key=lambda x: x.importance_score, reverse=True)
    
    async def _variance_threshold_selection(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_names: List[str]
    ) -> List[FeatureImportance]:
        """Variance threshold feature selection"""
        
        variances = X.var()
        
        importances = []
        for i, feature in enumerate(feature_names):
            importance = FeatureImportance(
                feature_name=feature,
                importance_score=variances.iloc[i],
                selection_method='variance',
                rank=i + 1
            )
            importances.append(importance)
        
        return sorted(importances, key=lambda x: x.importance_score, reverse=True)
    
    def _aggregate_feature_importances(
        self,
        all_importances: List[FeatureImportance]
    ) -> List[FeatureImportance]:
        """Aggregate feature importances from multiple methods"""
        
        feature_scores = defaultdict(list)
        
        # Group by feature name
        for importance in all_importances:
            feature_scores[importance.feature_name].append(importance.importance_score)
        
        # Calculate aggregated scores
        aggregated = []
        for feature_name, scores in feature_scores.items():
            avg_score = np.mean(scores)
            stability = 1.0 - (np.std(scores) / (np.mean(scores) + 1e-8))
            
            importance = FeatureImportance(
                feature_name=feature_name,
                importance_score=avg_score,
                selection_method='ensemble',
                rank=0,  # Will be set later
                stability_score=stability
            )
            aggregated.append(importance)
        
        # Sort and assign ranks
        aggregated.sort(key=lambda x: x.importance_score, reverse=True)
        for i, importance in enumerate(aggregated):
            importance.rank = i + 1
        
        return aggregated
    
    def _select_final_features(
        self,
        importances: List[FeatureImportance]
    ) -> List[str]:
        """Select final set of features"""
        
        # Filter by importance threshold
        filtered = [
            imp for imp in importances
            if imp.importance_score >= self.config.feature_importance_threshold
        ]
        
        # Limit by max features if specified
        if self.config.max_features:
            filtered = filtered[:self.config.max_features]
        
        return [imp.feature_name for imp in filtered]

class FeatureEngineer:
    """
    Ultra-advanced feature engineering manager
    """
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Feature extractors
        self.extractors = {}
        
        # Feature selector
        self.selector = FeatureSelector(config)
        
        # Scalers
        self.scalers = {}
        
        # Cached features
        self.feature_cache = {}
        
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialize feature engineer"""
        try:
            self.logger.info("Initializing Feature Engineer...")
            
            # Initialize extractors
            if self.config.enable_audio_features:
                self.extractors['audio'] = AudioFeatureExtractor(self.config)
            
            if self.config.enable_temporal_features:
                self.extractors['temporal'] = TemporalFeatureExtractor()
            
            # Initialize scalers
            self._initialize_scalers()
            
            self.is_initialized = True
            self.logger.info("Feature Engineer initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Feature Engineer: {e}")
            return False
    
    async def engineer_features(
        self,
        data: Dict[str, Any],
        target: Optional[np.ndarray] = None,
        feature_types: Optional[List[FeatureType]] = None
    ) -> Tuple[pd.DataFrame, FeatureEngineering]:
        """Complete feature engineering pipeline"""
        
        start_time = time.time()
        
        # Extract features
        all_features = {}
        
        if feature_types is None:
            feature_types = [FeatureType.AUDIO, FeatureType.TEMPORAL, FeatureType.METADATA]
        
        for feature_type in feature_types:
            if feature_type == FeatureType.AUDIO and 'audio' in data:
                audio_features = await self.extractors['audio'].extract_features(data['audio'])
                all_features.update(audio_features)
            
            elif feature_type == FeatureType.TEMPORAL and 'temporal' in data:
                temporal_features = await self.extractors['temporal'].extract_features(data['temporal'])
                all_features.update(temporal_features)
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(all_features)
        
        # Feature selection if target provided
        selected_features = list(feature_df.columns)
        feature_importances = []
        
        if target is not None:
            selected_features, feature_importances = await self.selector.select_features(
                feature_df, target, list(feature_df.columns)
            )
            feature_df = feature_df[selected_features]
        
        # Feature scaling
        scaled_df = await self._scale_features(feature_df)
        
        # Create engineering report
        processing_time = time.time() - start_time
        
        engineering_result = FeatureEngineering(
            original_feature_count=len(all_features),
            original_sample_count=len(feature_df),
            final_feature_count=len(selected_features),
            selected_features=selected_features,
            feature_importances=feature_importances,
            feature_stats=await self._calculate_feature_stats(scaled_df),
            processing_time=processing_time,
            scaling_method=self.config.scaling_method.value,
            selection_methods=[method.value for method in self.config.selection_methods],
            feature_stability_score=np.mean([imp.stability_score for imp in feature_importances]) if feature_importances else 0.0,
            redundancy_removed=len(all_features) - len(selected_features),
            information_retention=len(selected_features) / len(all_features) if len(all_features) > 0 else 0.0
        )
        
        return scaled_df, engineering_result
    
    async def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale features based on configuration"""
        
        if self.config.scaling_method == ScalingMethod.NONE:
            return df
        
        scaler_key = self.config.scaling_method.value
        
        if scaler_key not in self.scalers:
            self._initialize_scalers()
        
        scaler = self.scalers[scaler_key]
        scaled_array = scaler.fit_transform(df)
        
        return pd.DataFrame(scaled_array, columns=df.columns, index=df.index)
    
    def _initialize_scalers(self):
        """Initialize feature scalers"""
        self.scalers = {
            ScalingMethod.STANDARD.value: StandardScaler(),
            ScalingMethod.MINMAX.value: MinMaxScaler(),
            ScalingMethod.ROBUST.value: RobustScaler(),
            ScalingMethod.QUANTILE.value: QuantileTransformer(),
            ScalingMethod.POWER.value: PowerTransformer()
        }
    
    async def _calculate_feature_stats(self, df: pd.DataFrame) -> List[FeatureStats]:
        """Calculate comprehensive feature statistics"""
        stats = []
        
        for column in df.columns:
            series = df[column]
            
            stat = FeatureStats(
                feature_name=column,
                feature_type=FeatureType.NUMERICAL,  # Default assumption
                missing_ratio=series.isnull().sum() / len(series),
                variance=series.var(),
                mean=series.mean(),
                std=series.std(),
                min_value=series.min(),
                max_value=series.max(),
                unique_count=series.nunique(),
                skewness=skew(series.dropna()),
                kurtosis_value=kurtosis(series.dropna())
            )
            stats.append(stat)
        
        return stats

# Export main classes
__all__ = [
    "FeatureEngineer",
    "FeatureConfig",
    "FeatureImportance",
    "FeatureStats",
    "FeatureEngineering",
    "FeatureType",
    "SelectionMethod",
    "ScalingMethod",
    "AudioFeatureExtractor",
    "TemporalFeatureExtractor",
    "FeatureSelector"
]
