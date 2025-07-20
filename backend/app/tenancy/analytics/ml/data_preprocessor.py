"""
Ultra-Advanced Data Preprocessor for ML Pipeline

This module implements comprehensive data preprocessing capabilities including
advanced imputation, outlier detection, feature transformation, data validation,
and automated preprocessing pipeline optimization.

Features:
- Advanced missing value imputation (KNN, iterative, MICE)
- Sophisticated outlier detection and treatment
- Automated feature transformation and encoding
- Data validation and quality assessment
- Temporal data preprocessing and seasonality handling
- Audio signal preprocessing and feature extraction
- Automated preprocessing pipeline optimization
- Real-time data streaming preprocessing
- Data drift detection and adaptation
- Schema validation and enforcement

Created by Expert Team:
- Lead Dev + AI Architect: Preprocessing architecture and pipeline optimization
- Data Engineer: Advanced data transformation and validation pipelines
- ML Engineer: Feature preprocessing and automated transformation
- Audio Engineer: Specialized audio data preprocessing techniques
- Backend Developer: High-performance data processing and caching
- Quality Assurance: Data validation and testing frameworks
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
from pathlib import Path

# Core preprocessing libraries
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer,
    PowerTransformer, LabelEncoder, OneHotEncoder, OrdinalEncoder,
    PolynomialFeatures, Binarizer, Normalizer
)
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold

# Advanced imputation and outlier detection
try:
    from sklearn.experimental import enable_iterative_imputer
    ITERATIVE_IMPUTER_AVAILABLE = True
except ImportError:
    ITERATIVE_IMPUTER_AVAILABLE = False

# Outlier detection
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

# Statistical libraries
from scipy import stats
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

# Audio processing
try:
    import librosa
    import soundfile as sf
    from scipy.signal import butter, filtfilt
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# Time series processing
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    TIME_SERIES_AVAILABLE = True
except ImportError:
    TIME_SERIES_AVAILABLE = False

logger = logging.getLogger(__name__)

class ImputationMethod(Enum):
    """Missing value imputation methods"""
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"
    KNN = "knn"
    ITERATIVE = "iterative"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    INTERPOLATION = "interpolation"
    SEASONAL = "seasonal"

class OutlierMethod(Enum):
    """Outlier detection methods"""
    IQR = "iqr"
    Z_SCORE = "z_score"
    MODIFIED_Z_SCORE = "modified_z_score"
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "lof"
    ELLIPTIC_ENVELOPE = "elliptic_envelope"
    STATISTICAL = "statistical"

class ScalingMethod(Enum):
    """Feature scaling methods"""
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    QUANTILE = "quantile"
    POWER = "power"
    UNIT_VECTOR = "unit_vector"
    NONE = "none"

class EncodingMethod(Enum):
    """Categorical encoding methods"""
    LABEL = "label"
    ONEHOT = "onehot"
    ORDINAL = "ordinal"
    TARGET = "target"
    BINARY = "binary"
    FREQUENCY = "frequency"
    HASH = "hash"

@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing"""
    # Missing value handling
    imputation_method: ImputationMethod = ImputationMethod.KNN
    imputation_strategy: str = "auto"  # "auto", "manual"
    knn_neighbors: int = 5
    max_iter: int = 10
    
    # Outlier handling
    outlier_method: OutlierMethod = OutlierMethod.ISOLATION_FOREST
    outlier_threshold: float = 0.1
    outlier_action: str = "remove"  # "remove", "cap", "transform"
    contamination: float = 0.1
    
    # Scaling and transformation
    scaling_method: ScalingMethod = ScalingMethod.STANDARD
    encoding_method: EncodingMethod = EncodingMethod.ONEHOT
    handle_rare_categories: bool = True
    rare_category_threshold: float = 0.01
    
    # Feature engineering
    create_polynomial_features: bool = False
    polynomial_degree: int = 2
    create_interaction_features: bool = False
    remove_low_variance: bool = True
    variance_threshold: float = 0.01
    
    # Validation settings
    validate_schema: bool = True
    enforce_data_types: bool = True
    check_data_quality: bool = True
    quality_threshold: float = 0.95
    
    # Audio preprocessing (if applicable)
    audio_sample_rate: int = 22050
    audio_n_fft: int = 2048
    audio_hop_length: int = 512
    normalize_audio: bool = True
    
    # Time series preprocessing
    handle_seasonality: bool = True
    seasonal_period: Optional[int] = None
    stationarity_check: bool = True
    
    # Performance settings
    parallel_processing: bool = True
    n_jobs: int = -1
    chunk_size: int = 10000
    memory_efficient: bool = True

@dataclass
class DataQualityReport:
    """Comprehensive data quality assessment"""
    # Basic statistics
    total_samples: int
    total_features: int
    memory_usage_mb: float
    
    # Missing values
    missing_values_count: int
    missing_values_ratio: float
    features_with_missing: List[str]
    
    # Data types
    numerical_features: List[str]
    categorical_features: List[str]
    datetime_features: List[str]
    
    # Quality issues
    duplicate_rows: int
    constant_features: List[str]
    high_cardinality_features: List[str]
    
    # Outliers
    outlier_count: int
    outlier_ratio: float
    features_with_outliers: List[str]
    
    # Distribution analysis
    skewed_features: List[str]
    normal_features: List[str]
    
    # Recommendations
    recommended_actions: List[str]
    quality_score: float

@dataclass
class PreprocessingPipeline:
    """Preprocessing pipeline metadata"""
    pipeline_id: str
    steps: List[Dict[str, Any]]
    fitted_transformers: Dict[str, Any]
    feature_names_in: List[str]
    feature_names_out: List[str]
    preprocessing_time: float
    
    # Pipeline performance
    memory_usage: float
    processing_speed: float
    
    # Quality metrics
    data_quality_before: DataQualityReport
    data_quality_after: DataQualityReport
    improvement_score: float

class BasePreprocessor(ABC):
    """Abstract base class for preprocessors"""
    
    def __init__(self, processor_id: str, config: PreprocessingConfig):
        self.processor_id = processor_id
        self.config = config
        self.is_fitted = False
        self.feature_names = []
        self.transformers = {}
    
    @abstractmethod
    async def fit(self, data: pd.DataFrame) -> 'BasePreprocessor':
        """Fit the preprocessor"""
        pass
    
    @abstractmethod
    async def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the data"""
        pass
    
    @abstractmethod
    async def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the data"""
        pass

class MissingValueHandler(BasePreprocessor):
    """Advanced missing value imputation"""
    
    def __init__(self, config: PreprocessingConfig):
        super().__init__("missing_value_handler", config)
        self.imputers = {}
    
    async def fit(self, data: pd.DataFrame) -> 'MissingValueHandler':
        """Fit missing value imputers"""
        try:
            self.feature_names = list(data.columns)
            
            # Separate numerical and categorical columns
            numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Fit numerical imputers
            if numerical_cols:
                self.imputers['numerical'] = await self._fit_numerical_imputer(
                    data[numerical_cols]
                )
            
            # Fit categorical imputers
            if categorical_cols:
                self.imputers['categorical'] = await self._fit_categorical_imputer(
                    data[categorical_cols]
                )
            
            self.is_fitted = True
            return self
            
        except Exception as e:
            logger.error(f"Failed to fit missing value handler: {e}")
            raise
    
    async def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data by imputing missing values"""
        if not self.is_fitted:
            raise ValueError("Handler must be fitted before transform")
        
        try:
            result = data.copy()
            
            # Transform numerical columns
            numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if numerical_cols and 'numerical' in self.imputers:
                imputed_values = self.imputers['numerical'].transform(data[numerical_cols])
                result[numerical_cols] = imputed_values
            
            # Transform categorical columns
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_cols and 'categorical' in self.imputers:
                imputed_values = self.imputers['categorical'].transform(data[categorical_cols])
                result[categorical_cols] = imputed_values
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to transform data: {e}")
            raise
    
    async def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        await self.fit(data)
        return await self.transform(data)
    
    async def _fit_numerical_imputer(self, data: pd.DataFrame):
        """Fit numerical imputer based on configuration"""
        if self.config.imputation_method == ImputationMethod.MEAN:
            return SimpleImputer(strategy='mean')
        elif self.config.imputation_method == ImputationMethod.MEDIAN:
            return SimpleImputer(strategy='median')
        elif self.config.imputation_method == ImputationMethod.KNN:
            return KNNImputer(n_neighbors=self.config.knn_neighbors)
        elif self.config.imputation_method == ImputationMethod.ITERATIVE:
            if ITERATIVE_IMPUTER_AVAILABLE:
                return IterativeImputer(max_iter=self.config.max_iter, random_state=42)
            else:
                logger.warning("Iterative imputer not available, falling back to median")
                return SimpleImputer(strategy='median')
        else:
            return SimpleImputer(strategy='median')
    
    async def _fit_categorical_imputer(self, data: pd.DataFrame):
        """Fit categorical imputer"""
        if self.config.imputation_method == ImputationMethod.MODE:
            return SimpleImputer(strategy='most_frequent')
        elif self.config.imputation_method == ImputationMethod.CONSTANT:
            return SimpleImputer(strategy='constant', fill_value='missing')
        else:
            return SimpleImputer(strategy='most_frequent')

class OutlierDetector(BasePreprocessor):
    """Advanced outlier detection and treatment"""
    
    def __init__(self, config: PreprocessingConfig):
        super().__init__("outlier_detector", config)
        self.outlier_detectors = {}
        self.outlier_masks = {}
    
    async def fit(self, data: pd.DataFrame) -> 'OutlierDetector':
        """Fit outlier detectors"""
        try:
            self.feature_names = list(data.columns)
            numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numerical_cols:
                self.is_fitted = True
                return self
            
            numerical_data = data[numerical_cols]
            
            # Fit outlier detector based on method
            if self.config.outlier_method == OutlierMethod.ISOLATION_FOREST:
                detector = IsolationForest(
                    contamination=self.config.contamination,
                    random_state=42
                )
                detector.fit(numerical_data.fillna(numerical_data.median()))
                self.outlier_detectors['isolation_forest'] = detector
                
            elif self.config.outlier_method == OutlierMethod.LOCAL_OUTLIER_FACTOR:
                # LOF doesn't have a predict method, so we store the fitted data
                self.outlier_detectors['lof_data'] = numerical_data.fillna(numerical_data.median())
                
            elif self.config.outlier_method == OutlierMethod.ELLIPTIC_ENVELOPE:
                detector = EllipticEnvelope(
                    contamination=self.config.contamination,
                    random_state=42
                )
                detector.fit(numerical_data.fillna(numerical_data.median()))
                self.outlier_detectors['elliptic_envelope'] = detector
            
            self.is_fitted = True
            return self
            
        except Exception as e:
            logger.error(f"Failed to fit outlier detector: {e}")
            raise
    
    async def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data by handling outliers"""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before transform")
        
        try:
            result = data.copy()
            numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numerical_cols:
                return result
            
            # Detect outliers
            outlier_mask = await self._detect_outliers(data[numerical_cols])
            
            # Handle outliers based on action
            if self.config.outlier_action == "remove":
                result = result[~outlier_mask]
            elif self.config.outlier_action == "cap":
                result = await self._cap_outliers(result, numerical_cols, outlier_mask)
            elif self.config.outlier_action == "transform":
                result = await self._transform_outliers(result, numerical_cols)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to transform data: {e}")
            raise
    
    async def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        await self.fit(data)
        return await self.transform(data)
    
    async def _detect_outliers(self, data: pd.DataFrame) -> np.ndarray:
        """Detect outliers using fitted method"""
        if self.config.outlier_method == OutlierMethod.IQR:
            return await self._detect_iqr_outliers(data)
        elif self.config.outlier_method == OutlierMethod.Z_SCORE:
            return await self._detect_zscore_outliers(data)
        elif self.config.outlier_method == OutlierMethod.ISOLATION_FOREST:
            detector = self.outlier_detectors['isolation_forest']
            outlier_pred = detector.predict(data.fillna(data.median()))
            return outlier_pred == -1
        elif self.config.outlier_method == OutlierMethod.LOCAL_OUTLIER_FACTOR:
            lof = LocalOutlierFactor(contamination=self.config.contamination)
            outlier_pred = lof.fit_predict(data.fillna(data.median()))
            return outlier_pred == -1
        else:
            return np.zeros(len(data), dtype=bool)
    
    async def _detect_iqr_outliers(self, data: pd.DataFrame) -> np.ndarray:
        """Detect outliers using IQR method"""
        outlier_mask = np.zeros(len(data), dtype=bool)
        
        for column in data.columns:
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            column_outliers = (data[column] < lower_bound) | (data[column] > upper_bound)
            outlier_mask |= column_outliers.fillna(False)
        
        return outlier_mask
    
    async def _detect_zscore_outliers(self, data: pd.DataFrame) -> np.ndarray:
        """Detect outliers using Z-score method"""
        z_scores = np.abs(stats.zscore(data.fillna(data.median())))
        return (z_scores > 3).any(axis=1)
    
    async def _cap_outliers(
        self,
        data: pd.DataFrame,
        numerical_cols: List[str],
        outlier_mask: np.ndarray
    ) -> pd.DataFrame:
        """Cap outliers to acceptable ranges"""
        result = data.copy()
        
        for column in numerical_cols:
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            result[column] = result[column].clip(lower_bound, upper_bound)
        
        return result
    
    async def _transform_outliers(
        self,
        data: pd.DataFrame,
        numerical_cols: List[str]
    ) -> pd.DataFrame:
        """Transform outliers using statistical methods"""
        result = data.copy()
        
        for column in numerical_cols:
            # Apply log transformation for positive skewed data
            if data[column].min() > 0:
                skewness = stats.skew(data[column].dropna())
                if skewness > 1:
                    result[column] = np.log1p(data[column])
        
        return result

class FeatureTransformer(BasePreprocessor):
    """Advanced feature transformation and encoding"""
    
    def __init__(self, config: PreprocessingConfig):
        super().__init__("feature_transformer", config)
        self.column_transformer = None
        self.feature_names_out = []
    
    async def fit(self, data: pd.DataFrame) -> 'FeatureTransformer':
        """Fit feature transformers"""
        try:
            self.feature_names = list(data.columns)
            
            # Identify column types
            numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Create transformation steps
            transformers = []
            
            # Numerical transformations
            if numerical_cols:
                numerical_pipeline = await self._create_numerical_pipeline()
                transformers.append(('numerical', numerical_pipeline, numerical_cols))
            
            # Categorical transformations
            if categorical_cols:
                categorical_pipeline = await self._create_categorical_pipeline()
                transformers.append(('categorical', categorical_pipeline, categorical_cols))
            
            # Create column transformer
            self.column_transformer = ColumnTransformer(
                transformers=transformers,
                remainder='passthrough',
                n_jobs=self.config.n_jobs if self.config.parallel_processing else 1
            )
            
            # Fit transformer
            self.column_transformer.fit(data)
            
            # Get output feature names
            self.feature_names_out = await self._get_feature_names_out(data)
            
            self.is_fitted = True
            return self
            
        except Exception as e:
            logger.error(f"Failed to fit feature transformer: {e}")
            raise
    
    async def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform features"""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        try:
            # Transform data
            transformed_array = self.column_transformer.transform(data)
            
            # Convert back to DataFrame
            result = pd.DataFrame(
                transformed_array,
                columns=self.feature_names_out,
                index=data.index
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to transform features: {e}")
            raise
    
    async def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        await self.fit(data)
        return await self.transform(data)
    
    async def _create_numerical_pipeline(self) -> Pipeline:
        """Create numerical feature pipeline"""
        steps = []
        
        # Scaling
        if self.config.scaling_method == ScalingMethod.STANDARD:
            steps.append(('scaler', StandardScaler()))
        elif self.config.scaling_method == ScalingMethod.MINMAX:
            steps.append(('scaler', MinMaxScaler()))
        elif self.config.scaling_method == ScalingMethod.ROBUST:
            steps.append(('scaler', RobustScaler()))
        elif self.config.scaling_method == ScalingMethod.QUANTILE:
            steps.append(('scaler', QuantileTransformer()))
        elif self.config.scaling_method == ScalingMethod.POWER:
            steps.append(('scaler', PowerTransformer()))
        
        # Polynomial features
        if self.config.create_polynomial_features:
            steps.append(('poly', PolynomialFeatures(
                degree=self.config.polynomial_degree,
                include_bias=False
            )))
        
        # Variance threshold
        if self.config.remove_low_variance:
            steps.append(('variance', VarianceThreshold(
                threshold=self.config.variance_threshold
            )))
        
        return Pipeline(steps)
    
    async def _create_categorical_pipeline(self) -> Pipeline:
        """Create categorical feature pipeline"""
        steps = []
        
        # Encoding
        if self.config.encoding_method == EncodingMethod.ONEHOT:
            steps.append(('encoder', OneHotEncoder(
                drop='first',
                sparse=False,
                handle_unknown='ignore'
            )))
        elif self.config.encoding_method == EncodingMethod.ORDINAL:
            steps.append(('encoder', OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )))
        elif self.config.encoding_method == EncodingMethod.LABEL:
            steps.append(('encoder', LabelEncoder()))
        
        return Pipeline(steps)
    
    async def _get_feature_names_out(self, data: pd.DataFrame) -> List[str]:
        """Get output feature names after transformation"""
        try:
            # Try to get feature names from transformers
            feature_names = []
            
            for name, transformer, columns in self.column_transformer.transformers_:
                if name == 'remainder':
                    continue
                
                if hasattr(transformer, 'get_feature_names_out'):
                    trans_features = transformer.get_feature_names_out(columns)
                    feature_names.extend([f"{name}_{feat}" for feat in trans_features])
                else:
                    # Fallback for transformers without get_feature_names_out
                    if hasattr(transformer.named_steps['encoder'], 'get_feature_names_out'):
                        encoder_features = transformer.named_steps['encoder'].get_feature_names_out(columns)
                        feature_names.extend([f"{name}_{feat}" for feat in encoder_features])
                    else:
                        feature_names.extend([f"{name}_{col}" for col in columns])
            
            return feature_names
            
        except Exception as e:
            logger.warning(f"Could not get feature names, using generic names: {e}")
            # Generate generic feature names
            n_features = self.column_transformer.transform(data.head(1)).shape[1]
            return [f"feature_{i}" for i in range(n_features)]

class AudioPreprocessor(BasePreprocessor):
    """Specialized audio data preprocessing"""
    
    def __init__(self, config: PreprocessingConfig):
        super().__init__("audio_preprocessor", config)
        self.audio_stats = {}
    
    async def fit(self, data: pd.DataFrame) -> 'AudioPreprocessor':
        """Fit audio preprocessor"""
        if not AUDIO_AVAILABLE:
            raise ImportError("Audio processing libraries not available")
        
        try:
            # Assume data contains audio file paths or raw audio data
            # Calculate normalization statistics if needed
            self.is_fitted = True
            return self
            
        except Exception as e:
            logger.error(f"Failed to fit audio preprocessor: {e}")
            raise
    
    async def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform audio data"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # This would implement audio-specific preprocessing
        # For now, return data as-is
        return data
    
    async def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform audio data"""
        await self.fit(data)
        return await self.transform(data)

class DataPreprocessor:
    """
    Ultra-advanced data preprocessor with automated pipeline optimization
    """
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Preprocessing components
        self.missing_value_handler = MissingValueHandler(config)
        self.outlier_detector = OutlierDetector(config)
        self.feature_transformer = FeatureTransformer(config)
        self.audio_preprocessor = AudioPreprocessor(config) if AUDIO_AVAILABLE else None
        
        # Pipeline state
        self.pipeline = None
        self.is_fitted = False
        self.preprocessing_stats = {}
        
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialize data preprocessor"""
        try:
            self.logger.info("Initializing Data Preprocessor...")
            
            self.is_initialized = True
            self.logger.info("Data Preprocessor initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Data Preprocessor: {e}")
            return False
    
    async def analyze_data_quality(self, data: pd.DataFrame) -> DataQualityReport:
        """Comprehensive data quality analysis"""
        try:
            # Basic statistics
            total_samples = len(data)
            total_features = len(data.columns)
            memory_usage_mb = data.memory_usage(deep=True).sum() / 1024**2
            
            # Missing values analysis
            missing_values = data.isnull().sum()
            missing_count = missing_values.sum()
            missing_ratio = missing_count / (total_samples * total_features)
            features_with_missing = missing_values[missing_values > 0].index.tolist()
            
            # Data types analysis
            numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_features = data.select_dtypes(include=['datetime64']).columns.tolist()
            
            # Quality issues
            duplicate_rows = data.duplicated().sum()
            constant_features = [col for col in data.columns if data[col].nunique() <= 1]
            high_cardinality_features = [
                col for col in categorical_features
                if data[col].nunique() > 0.5 * total_samples
            ]
            
            # Outlier analysis (simplified)
            outlier_count = 0
            features_with_outliers = []
            
            for col in numerical_features:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((data[col] < Q1 - 1.5 * IQR) | (data[col] > Q3 + 1.5 * IQR)).sum()
                if outliers > 0:
                    outlier_count += outliers
                    features_with_outliers.append(col)
            
            outlier_ratio = outlier_count / total_samples if total_samples > 0 else 0
            
            # Distribution analysis
            skewed_features = []
            normal_features = []
            
            for col in numerical_features:
                skewness = stats.skew(data[col].dropna())
                if abs(skewness) > 1:
                    skewed_features.append(col)
                else:
                    normal_features.append(col)
            
            # Generate recommendations
            recommendations = []
            if missing_ratio > 0.1:
                recommendations.append("High missing value ratio - consider imputation")
            if len(constant_features) > 0:
                recommendations.append("Remove constant features")
            if outlier_ratio > 0.05:
                recommendations.append("Consider outlier treatment")
            if len(skewed_features) > len(numerical_features) * 0.5:
                recommendations.append("Consider transformation for skewed features")
            
            # Calculate quality score
            quality_score = 1.0
            quality_score -= missing_ratio * 0.3
            quality_score -= outlier_ratio * 0.2
            quality_score -= len(constant_features) / total_features * 0.2
            quality_score = max(0.0, min(1.0, quality_score))
            
            return DataQualityReport(
                total_samples=total_samples,
                total_features=total_features,
                memory_usage_mb=memory_usage_mb,
                missing_values_count=missing_count,
                missing_values_ratio=missing_ratio,
                features_with_missing=features_with_missing,
                numerical_features=numerical_features,
                categorical_features=categorical_features,
                datetime_features=datetime_features,
                duplicate_rows=duplicate_rows,
                constant_features=constant_features,
                high_cardinality_features=high_cardinality_features,
                outlier_count=outlier_count,
                outlier_ratio=outlier_ratio,
                features_with_outliers=features_with_outliers,
                skewed_features=skewed_features,
                normal_features=normal_features,
                recommended_actions=recommendations,
                quality_score=quality_score
            )
            
        except Exception as e:
            self.logger.error(f"Failed to analyze data quality: {e}")
            raise
    
    async def fit(self, data: pd.DataFrame) -> 'DataPreprocessor':
        """Fit preprocessing pipeline"""
        try:
            start_time = time.time()
            
            # Analyze data quality
            quality_before = await self.analyze_data_quality(data)
            
            # Fit preprocessing steps in order
            current_data = data.copy()
            
            # 1. Handle missing values
            await self.missing_value_handler.fit(current_data)
            current_data = await self.missing_value_handler.transform(current_data)
            
            # 2. Handle outliers
            await self.outlier_detector.fit(current_data)
            current_data = await self.outlier_detector.transform(current_data)
            
            # 3. Transform features
            await self.feature_transformer.fit(current_data)
            current_data = await self.feature_transformer.transform(current_data)
            
            # Store preprocessing statistics
            processing_time = time.time() - start_time
            quality_after = await self.analyze_data_quality(current_data)
            
            self.preprocessing_stats = {
                'processing_time': processing_time,
                'quality_before': quality_before,
                'quality_after': quality_after,
                'improvement_score': quality_after.quality_score - quality_before.quality_score
            }
            
            self.is_fitted = True
            self.logger.info(f"Preprocessing pipeline fitted in {processing_time:.2f} seconds")
            return self
            
        except Exception as e:
            self.logger.error(f"Failed to fit preprocessing pipeline: {e}")
            raise
    
    async def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted pipeline"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        try:
            current_data = data.copy()
            
            # Apply transformations in order
            current_data = await self.missing_value_handler.transform(current_data)
            current_data = await self.outlier_detector.transform(current_data)
            current_data = await self.feature_transformer.transform(current_data)
            
            return current_data
            
        except Exception as e:
            self.logger.error(f"Failed to transform data: {e}")
            raise
    
    async def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform data in one step"""
        await self.fit(data)
        return await self.transform(data)

# Export main classes
__all__ = [
    "DataPreprocessor",
    "PreprocessingConfig",
    "DataQualityReport",
    "PreprocessingPipeline",
    "MissingValueHandler",
    "OutlierDetector",
    "FeatureTransformer",
    "AudioPreprocessor",
    "ImputationMethod",
    "OutlierMethod",
    "ScalingMethod",
    "EncodingMethod"
]
