"""
Utilitaires ML - Spotify AI Agent
Fonctions utilitaires pour Machine Learning et Intelligence Artificielle
"""

import os
import json
import pickle
import joblib
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from uuid import uuid4
import time
import asyncio
from functools import wraps, partial

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

from .types import (
    ModelType, FrameworkType, ModelStatus, TrainingStatus,
    InferenceMode, DataType, FeatureType, ModelFormat,
    MetricDict, HyperparameterDict, DataArray, FeatureVector
)
from .exceptions import (
    MLException, DataValidationError, ModelLoadError,
    TrainingException, InferenceException
)


# Configuration et logging
logger = logging.getLogger(__name__)

# Constantes
DEFAULT_RANDOM_SEED = 42
MAX_RETRY_ATTEMPTS = 3
DEFAULT_TIMEOUT_SECONDS = 300
SUPPORTED_AUDIO_FORMATS = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']


class Timer:
    """Utilitaire pour mesurer le temps d'exécution"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        logger.debug(f"Début de {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        logger.debug(f"Fin de {self.name} - Durée: {self.duration:.4f}s")
    
    def get_duration(self) -> Optional[float]:
        """Retourne la durée en secondes"""
        return self.duration


def timing_decorator(func: Callable) -> Callable:
    """Décorateur pour mesurer le temps d'exécution d'une fonction"""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        with Timer(f"{func.__name__}"):
            return func(*args, **kwargs)
    
    return wrapper


def async_timing_decorator(func: Callable) -> Callable:
    """Décorateur pour mesurer le temps d'exécution d'une fonction async"""
    
    @wraps(func)
    async def wrapper(*args, **kwargs):
        with Timer(f"{func.__name__}"):
            return await func(*args, **kwargs)
    
    return wrapper


def retry_on_failure(
    max_attempts: int = MAX_RETRY_ATTEMPTS,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple = (Exception,)
) -> Callable:
    """Décorateur pour réessayer une fonction en cas d'échec"""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            current_delay = delay
            
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.error(f"Échec définitif après {max_attempts} tentatives: {e}")
                        raise
                    
                    logger.warning(f"Tentative {attempt} échouée: {e}. Nouvelle tentative dans {current_delay}s")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
        return wrapper
    return decorator


# Utilitaires de données
def validate_data_types(data: Any, expected_type: DataType) -> bool:
    """Valide que les données correspondent au type attendu"""
    
    try:
        if expected_type == DataType.TABULAR:
            return isinstance(data, (pd.DataFrame, np.ndarray, list))
        elif expected_type == DataType.TEXT:
            return isinstance(data, (str, list))
        elif expected_type == DataType.IMAGE:
            return isinstance(data, (np.ndarray, str, Path))
        elif expected_type == DataType.AUDIO:
            return isinstance(data, (np.ndarray, str, Path))
        elif expected_type == DataType.TIME_SERIES:
            return isinstance(data, (pd.DataFrame, np.ndarray, list))
        else:
            return True
    except Exception as e:
        logger.error(f"Erreur lors de la validation du type de données: {e}")
        return False


def calculate_data_hash(data: Any) -> str:
    """Calcule un hash pour identifier de manière unique un dataset"""
    
    try:
        if isinstance(data, pd.DataFrame):
            # Pour les DataFrames, on utilise les valeurs et les colonnes
            content = str(data.values.tobytes()) + str(list(data.columns))
        elif isinstance(data, np.ndarray):
            content = data.tobytes()
        elif isinstance(data, (list, tuple)):
            content = str(data)
        elif isinstance(data, str):
            content = data
        else:
            content = str(data)
        
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul du hash: {e}")
        return str(uuid4())[:16]


def split_data(
    X: DataArray,
    y: Optional[DataArray] = None,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = DEFAULT_RANDOM_SEED,
    stratify: bool = False
) -> Union[Tuple[DataArray, DataArray], Tuple[DataArray, DataArray, DataArray, DataArray, DataArray, DataArray]]:
    """Divise les données en ensembles d'entraînement, validation et test"""
    
    try:
        if y is None:
            # Données non supervisées
            if validation_size > 0:
                X_temp, X_test = train_test_split(
                    X, test_size=test_size, random_state=random_state
                )
                val_size_adj = validation_size / (1 - test_size)
                X_train, X_val = train_test_split(
                    X_temp, test_size=val_size_adj, random_state=random_state
                )
                return X_train, X_val, X_test
            else:
                X_train, X_test = train_test_split(
                    X, test_size=test_size, random_state=random_state
                )
                return X_train, X_test
        else:
            # Données supervisées
            stratify_param = y if stratify else None
            
            if validation_size > 0:
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state,
                    stratify=stratify_param
                )
                val_size_adj = validation_size / (1 - test_size)
                stratify_param_temp = y_temp if stratify else None
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=val_size_adj, random_state=random_state,
                    stratify=stratify_param_temp
                )
                return X_train, X_val, X_test, y_train, y_val, y_test
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state,
                    stratify=stratify_param
                )
                return X_train, X_test, y_train, y_test
    
    except Exception as e:
        raise DataValidationError(f"Erreur lors de la division des données: {e}")


def detect_data_drift(
    reference_data: DataArray,
    current_data: DataArray,
    threshold: float = 0.1,
    method: str = "ks_test"
) -> Dict[str, Any]:
    """Détecte la dérive des données entre deux datasets"""
    
    try:
        from scipy import stats
        
        results = {
            'drift_detected': False,
            'drift_score': 0.0,
            'threshold': threshold,
            'method': method,
            'details': {}
        }
        
        if isinstance(reference_data, pd.DataFrame) and isinstance(current_data, pd.DataFrame):
            # Analyse colonne par colonne
            drift_scores = []
            for col in reference_data.columns:
                if col in current_data.columns:
                    if method == "ks_test":
                        statistic, p_value = stats.ks_2samp(
                            reference_data[col].dropna(),
                            current_data[col].dropna()
                        )
                        drift_score = 1 - p_value
                    else:
                        # Méthode simple basée sur la différence des moyennes
                        ref_mean = reference_data[col].mean()
                        cur_mean = current_data[col].mean()
                        drift_score = abs(ref_mean - cur_mean) / (abs(ref_mean) + 1e-8)
                    
                    drift_scores.append(drift_score)
                    results['details'][col] = drift_score
            
            results['drift_score'] = np.mean(drift_scores) if drift_scores else 0.0
        else:
            # Analyse globale pour les arrays
            if method == "ks_test":
                statistic, p_value = stats.ks_2samp(
                    reference_data.flatten(),
                    current_data.flatten()
                )
                results['drift_score'] = 1 - p_value
            else:
                ref_mean = np.mean(reference_data)
                cur_mean = np.mean(current_data)
                results['drift_score'] = abs(ref_mean - cur_mean) / (abs(ref_mean) + 1e-8)
        
        results['drift_detected'] = results['drift_score'] > threshold
        
        return results
    
    except Exception as e:
        logger.error(f"Erreur lors de la détection de dérive: {e}")
        return {
            'drift_detected': False,
            'drift_score': 0.0,
            'threshold': threshold,
            'method': method,
            'error': str(e)
        }


# Utilitaires de modèles
def save_model(
    model: Any,
    file_path: Union[str, Path],
    model_format: ModelFormat = ModelFormat.PICKLE,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """Sauvegarde un modèle dans le format spécifié"""
    
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if model_format == ModelFormat.PICKLE:
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
        elif model_format == ModelFormat.JOBLIB:
            joblib.dump(model, file_path)
        elif model_format == ModelFormat.ONNX:
            # Nécessite onnx et onnxmltools
            model.save(str(file_path))
        else:
            raise ValueError(f"Format de modèle non supporté: {model_format}")
        
        # Sauvegarde des métadonnées si fournies
        if metadata:
            metadata_path = file_path.with_suffix('.metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Modèle sauvegardé: {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du modèle: {e}")
        raise ModelLoadError(f"Impossible de sauvegarder le modèle: {e}")


def load_model(
    file_path: Union[str, Path],
    model_format: ModelFormat = ModelFormat.PICKLE
) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """Charge un modèle depuis un fichier"""
    
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier de modèle non trouvé: {file_path}")
        
        if model_format == ModelFormat.PICKLE:
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
        elif model_format == ModelFormat.JOBLIB:
            model = joblib.load(file_path)
        else:
            raise ValueError(f"Format de modèle non supporté: {model_format}")
        
        # Chargement des métadonnées si disponibles
        metadata = None
        metadata_path = file_path.with_suffix('.metadata.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        logger.info(f"Modèle chargé: {file_path}")
        return model, metadata
    
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}")
        raise ModelLoadError(f"Impossible de charger le modèle: {e}")


def calculate_model_metrics(
    y_true: DataArray,
    y_pred: DataArray,
    model_type: ModelType,
    average: str = 'weighted'
) -> MetricDict:
    """Calcule les métriques appropriées selon le type de modèle"""
    
    try:
        metrics = {}
        
        if model_type in [ModelType.CLASSIFICATION, ModelType.MULTICLASS_CLASSIFICATION]:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
            metrics['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0)
            
            # AUC pour classification binaire ou avec probabilités
            try:
                if len(np.unique(y_true)) == 2:
                    metrics['auc_roc'] = roc_auc_score(y_true, y_pred)
            except:
                pass
        
        elif model_type == ModelType.REGRESSION:
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2_score'] = r2_score(y_true, y_pred)
            
            # MAPE (Mean Absolute Percentage Error)
            try:
                mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
                metrics['mape'] = mape
            except:
                pass
        
        return metrics
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul des métriques: {e}")
        return {}


def generate_model_id(
    model_name: str,
    version: str = "1.0.0",
    timestamp: Optional[datetime] = None
) -> str:
    """Génère un ID unique pour un modèle"""
    
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    base_id = f"{model_name}_v{version}_{timestamp_str}"
    
    # Ajouter un hash court pour éviter les collisions
    hash_suffix = hashlib.md5(base_id.encode()).hexdigest()[:8]
    
    return f"{base_id}_{hash_suffix}"


def validate_hyperparameters(
    hyperparams: HyperparameterDict,
    param_space: Dict[str, Dict[str, Any]]
) -> Tuple[bool, List[str]]:
    """Valide les hyperparamètres contre un espace de paramètres défini"""
    
    errors = []
    
    try:
        for param_name, param_value in hyperparams.items():
            if param_name not in param_space:
                errors.append(f"Paramètre inconnu: {param_name}")
                continue
            
            param_config = param_space[param_name]
            param_type = param_config.get('type')
            
            # Validation du type
            if param_type == 'int' and not isinstance(param_value, int):
                errors.append(f"{param_name} doit être un entier")
            elif param_type == 'float' and not isinstance(param_value, (int, float)):
                errors.append(f"{param_name} doit être un nombre")
            elif param_type == 'str' and not isinstance(param_value, str):
                errors.append(f"{param_name} doit être une chaîne")
            elif param_type == 'bool' and not isinstance(param_value, bool):
                errors.append(f"{param_name} doit être un booléen")
            
            # Validation des plages
            if 'min' in param_config and param_value < param_config['min']:
                errors.append(f"{param_name} doit être >= {param_config['min']}")
            if 'max' in param_config and param_value > param_config['max']:
                errors.append(f"{param_name} doit être <= {param_config['max']}")
            
            # Validation des choix
            if 'choices' in param_config and param_value not in param_config['choices']:
                errors.append(f"{param_name} doit être parmi {param_config['choices']}")
        
        return len(errors) == 0, errors
    
    except Exception as e:
        logger.error(f"Erreur lors de la validation des hyperparamètres: {e}")
        return False, [f"Erreur de validation: {e}"]


# Utilitaires de preprocessing
def auto_preprocess_data(
    data: pd.DataFrame,
    target_column: Optional[str] = None,
    handle_missing: str = 'drop',
    scale_features: bool = True,
    encode_categorical: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Preprocessing automatique des données tabulaires"""
    
    preprocessing_info = {
        'missing_values_handled': False,
        'features_scaled': False,
        'categorical_encoded': False,
        'scalers': {},
        'encoders': {},
        'dropped_columns': []
    }
    
    try:
        df = data.copy()
        
        # Gestion des valeurs manquantes
        if handle_missing == 'drop':
            initial_shape = df.shape
            df = df.dropna()
            if df.shape != initial_shape:
                preprocessing_info['missing_values_handled'] = True
                logger.info(f"Suppression des lignes avec valeurs manquantes: {initial_shape} -> {df.shape}")
        
        elif handle_missing == 'fill':
            for column in df.columns:
                if df[column].dtype in ['int64', 'float64']:
                    df[column].fillna(df[column].median(), inplace=True)
                else:
                    df[column].fillna(df[column].mode()[0] if not df[column].mode().empty else 'unknown', inplace=True)
            preprocessing_info['missing_values_handled'] = True
        
        # Encodage des variables catégorielles
        if encode_categorical:
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns
            if target_column and target_column in categorical_columns:
                categorical_columns = categorical_columns.drop(target_column)
            
            for column in categorical_columns:
                encoder = LabelEncoder()
                df[column] = encoder.fit_transform(df[column].astype(str))
                preprocessing_info['encoders'][column] = encoder
            
            if len(categorical_columns) > 0:
                preprocessing_info['categorical_encoded'] = True
        
        # Mise à l'échelle des features numériques
        if scale_features:
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
            if target_column and target_column in numeric_columns:
                numeric_columns = numeric_columns.drop(target_column)
            
            if len(numeric_columns) > 0:
                scaler = StandardScaler()
                df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
                preprocessing_info['scalers']['standard'] = scaler
                preprocessing_info['features_scaled'] = True
        
        return df, preprocessing_info
    
    except Exception as e:
        logger.error(f"Erreur lors du preprocessing: {e}")
        raise DataValidationError(f"Échec du preprocessing automatique: {e}")


# Utilitaires de monitoring
class PerformanceMonitor:
    """Moniteur de performance pour les modèles ML"""
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.metrics_history = []
        self.start_time = datetime.now(timezone.utc)
    
    def log_prediction(
        self,
        prediction_time: float,
        input_size: int,
        confidence: Optional[float] = None
    ):
        """Enregistre une prédiction"""
        self.metrics_history.append({
            'timestamp': datetime.now(timezone.utc),
            'prediction_time': prediction_time,
            'input_size': input_size,
            'confidence': confidence
        })
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Calcule les statistiques de performance"""
        if not self.metrics_history:
            return {}
        
        times = [m['prediction_time'] for m in self.metrics_history]
        sizes = [m['input_size'] for m in self.metrics_history]
        confidences = [m['confidence'] for m in self.metrics_history if m['confidence'] is not None]
        
        return {
            'model_id': self.model_id,
            'total_predictions': len(self.metrics_history),
            'avg_prediction_time': np.mean(times),
            'median_prediction_time': np.median(times),
            'p95_prediction_time': np.percentile(times, 95),
            'avg_input_size': np.mean(sizes),
            'avg_confidence': np.mean(confidences) if confidences else None,
            'uptime_hours': (datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600
        }


# Utilitaires d'export
def export_model_summary(
    model: Any,
    model_id: str,
    metrics: MetricDict,
    hyperparams: HyperparameterDict,
    output_path: Union[str, Path]
) -> bool:
    """Exporte un résumé complet du modèle"""
    
    try:
        summary = {
            'model_id': model_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'model_type': str(type(model).__name__),
            'metrics': metrics,
            'hyperparameters': hyperparams,
            'model_info': {}
        }
        
        # Informations spécifiques au modèle
        if hasattr(model, 'get_params'):
            summary['model_info']['parameters'] = model.get_params()
        
        if hasattr(model, 'feature_importances_'):
            summary['model_info']['feature_importances'] = model.feature_importances_.tolist()
        
        if hasattr(model, 'classes_'):
            summary['model_info']['classes'] = model.classes_.tolist()
        
        # Sauvegarde
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Résumé du modèle exporté: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Erreur lors de l'export du résumé: {e}")
        return False


__all__ = [
    # Classes utilitaires
    'Timer', 'PerformanceMonitor',
    
    # Décorateurs
    'timing_decorator', 'async_timing_decorator', 'retry_on_failure',
    
    # Utilitaires de données
    'validate_data_types', 'calculate_data_hash', 'split_data',
    'detect_data_drift', 'auto_preprocess_data',
    
    # Utilitaires de modèles
    'save_model', 'load_model', 'calculate_model_metrics',
    'generate_model_id', 'validate_hyperparameters',
    
    # Utilitaires d'export
    'export_model_summary',
    
    # Constantes
    'DEFAULT_RANDOM_SEED', 'MAX_RETRY_ATTEMPTS', 'DEFAULT_TIMEOUT_SECONDS',
    'SUPPORTED_AUDIO_FORMATS', 'SUPPORTED_IMAGE_FORMATS'
]
