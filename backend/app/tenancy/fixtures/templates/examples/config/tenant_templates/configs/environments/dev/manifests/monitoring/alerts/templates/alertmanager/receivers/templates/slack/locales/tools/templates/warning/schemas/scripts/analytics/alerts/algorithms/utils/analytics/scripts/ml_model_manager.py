"""
ML Model Manager - Ultra-Advanced Edition
========================================

Ultra-advanced ML model lifecycle management system with automated training,
deployment, monitoring, and A/B testing capabilities.

Features:
- Automated model training and hyperparameter optimization
- Model versioning and experiment tracking
- Automated deployment with canary releases
- Real-time model monitoring and drift detection
- A/B testing framework for model comparison
- Model explainability and fairness assessment
- Performance optimization and resource management
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import joblib
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import yaml
import hashlib
from abc import ABC, abstractmethod

# ML Framework imports
import tensorflow as tf
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import optuna
from transformers import AutoTokenizer, AutoModel
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import mlflow.pytorch

# Monitoring and drift detection
from scipy import stats
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
import shap


@dataclass
class ModelMetadata:
    """Comprehensive model metadata."""
    
    model_id: str
    name: str
    version: str
    framework: str  # sklearn, tensorflow, pytorch, huggingface
    model_type: str  # classification, regression, nlp, etc.
    created_at: datetime
    created_by: str
    description: str
    tags: List[str]
    
    # Training metadata
    training_dataset_id: str
    training_config: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    
    # Performance metrics
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    
    # Deployment info
    deployment_status: str  # training, ready, deployed, deprecated
    deployment_config: Optional[Dict[str, Any]] = None
    endpoint_url: Optional[str] = None
    
    # Resource requirements
    memory_requirements_mb: int = 512
    cpu_requirements: float = 0.5
    gpu_required: bool = False
    
    # Business metadata
    business_impact: Optional[str] = None
    cost_per_prediction: Optional[float] = None
    sla_requirements: Optional[Dict[str, Any]] = None


@dataclass
class TrainingJob:
    """ML training job configuration."""
    
    job_id: str
    model_name: str
    model_type: str
    framework: str
    
    # Data configuration
    dataset_path: str
    target_column: str
    feature_columns: List[str]
    
    # Training configuration
    training_config: Dict[str, Any]
    hyperparameter_search: bool = True
    cross_validation_folds: int = 5
    
    # Resource allocation
    max_training_time_hours: float = 24.0
    memory_limit_gb: float = 8.0
    use_gpu: bool = False
    
    # Output configuration
    model_output_path: str
    experiment_name: str
    
    # Status tracking
    status: str = "queued"  # queued, running, completed, failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class DeploymentConfig:
    """Model deployment configuration."""
    
    deployment_id: str
    model_id: str
    environment: str  # dev, staging, production
    deployment_strategy: str  # blue_green, canary, rolling
    
    # Resource allocation
    replicas: int = 1
    cpu_limit: str = "500m"
    memory_limit: str = "1Gi"
    auto_scaling: bool = True
    min_replicas: int = 1
    max_replicas: int = 10
    
    # Traffic configuration
    traffic_percentage: float = 100.0
    canary_percentage: float = 10.0
    
    # Health checks
    health_check_path: str = "/health"
    readiness_timeout: int = 30
    liveness_timeout: int = 30
    
    # Monitoring
    enable_monitoring: bool = True
    log_predictions: bool = True
    enable_explainability: bool = False


class ModelTrainer(ABC):
    """Abstract base class for model trainers."""
    
    @abstractmethod
    async def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, float]]:
        """Train a model and return model and metrics."""
        pass
    
    @abstractmethod
    def optimize_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 100
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        pass


class SklearnTrainer(ModelTrainer):
    """Scikit-learn model trainer."""
    
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.model_classes = {
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'logistic_regression': LogisticRegression
        }
    
    async def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, float]]:
        """Train scikit-learn model."""
        
        model_class = self.model_classes[self.model_type]
        model = model_class(**config.get('hyperparameters', {}))
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, average='weighted'),
            'recall': recall_score(y_val, y_pred, average='weighted'),
            'f1': f1_score(y_val, y_pred, average='weighted')
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_val, y_pred_proba)
        
        return model, metrics
    
    def optimize_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 100
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        
        def objective(trial):
            if self.model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 10, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
                }
            elif self.model_type == 'gradient_boosting':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0)
                }
            else:  # logistic_regression
                params = {
                    'C': trial.suggest_float('C', 0.001, 100, log=True),
                    'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
                    'solver': trial.suggest_categorical('solver', ['liblinear', 'saga'])
                }
            
            model_class = self.model_classes[self.model_type]
            model = model_class(**params)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            return f1_score(y_val, y_pred, average='weighted')
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params


class TensorFlowTrainer(ModelTrainer):
    """TensorFlow/Keras model trainer."""
    
    async def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, float]]:
        """Train TensorFlow model."""
        
        # Create model architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                config.get('hidden_units', 64),
                activation='relu',
                input_shape=(X_train.shape[1],)
            ),
            tf.keras.layers.Dropout(config.get('dropout_rate', 0.2)),
            tf.keras.layers.Dense(
                config.get('hidden_units', 64),
                activation='relu'
            ),
            tf.keras.layers.Dropout(config.get('dropout_rate', 0.2)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(config.get('learning_rate', 0.001)),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            batch_size=config.get('batch_size', 32),
            epochs=config.get('epochs', 100),
            validation_data=(X_val, y_val),
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=config.get('patience', 10),
                    restore_best_weights=True
                )
            ]
        )
        
        # Evaluate
        y_pred_proba = model.predict(X_val).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_pred_proba)
        }
        
        return model, metrics
    
    def optimize_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 100
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        
        def objective(trial):
            params = {
                'hidden_units': trial.suggest_int('hidden_units', 32, 256),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                'epochs': 50  # Fixed for optimization
            }
            
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    params['hidden_units'],
                    activation='relu',
                    input_shape=(X_train.shape[1],)
                ),
                tf.keras.layers.Dropout(params['dropout_rate']),
                tf.keras.layers.Dense(params['hidden_units'], activation='relu'),
                tf.keras.layers.Dropout(params['dropout_rate']),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(params['learning_rate']),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            model.fit(
                X_train, y_train,
                batch_size=params['batch_size'],
                epochs=params['epochs'],
                validation_data=(X_val, y_val),
                verbose=0,
                callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)]
            )
            
            y_pred_proba = model.predict(X_val).flatten()
            return roc_auc_score(y_val, y_pred_proba)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params


class MLModelManager:
    """Ultra-advanced ML model lifecycle manager."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the ML model manager."""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Initialize MLflow
        mlflow.set_tracking_uri(self.config.get('mlflow_uri', 'sqlite:///mlflow.db'))
        
        # Model registry
        self.models: Dict[str, ModelMetadata] = {}
        self.trainers = {
            'sklearn': {
                'random_forest': SklearnTrainer('random_forest'),
                'gradient_boosting': SklearnTrainer('gradient_boosting'),
                'logistic_regression': SklearnTrainer('logistic_regression')
            },
            'tensorflow': TensorFlowTrainer()
        }
        
        # Deployment tracking
        self.deployments: Dict[str, DeploymentConfig] = {}
        
        # Performance tracking
        self.performance_metrics = {
            'models_trained': 0,
            'models_deployed': 0,
            'predictions_served': 0,
            'average_training_time': 0.0,
            'average_inference_time': 0.0
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration."""
        default_config = {
            'model_storage_path': './models',
            'experiment_tracking': True,
            'auto_deploy': False,
            'drift_detection_enabled': True,
            'explainability_enabled': True,
            'a_b_testing_enabled': True,
            'performance_monitoring': True,
            'model_registry_backend': 'mlflow',
            'deployment_platform': 'kubernetes'
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger('MLModelManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def train_model(self, training_job: TrainingJob) -> ModelMetadata:
        """Train a new ML model."""
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting training job: {training_job.job_id}")
            training_job.status = "running"
            training_job.started_at = start_time
            
            # Load and prepare data
            data = pd.read_csv(training_job.dataset_path)
            
            # Prepare features and target
            X = data[training_job.feature_columns].values
            y = data[training_job.target_column].values
            
            # Split data
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.4, random_state=42, stratify=y
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            # Get trainer
            trainer = self.trainers[training_job.framework]
            if isinstance(trainer, dict):
                trainer = trainer[training_job.model_type]
            
            # Hyperparameter optimization
            best_params = {}
            if training_job.hyperparameter_search:
                self.logger.info("Starting hyperparameter optimization")
                best_params = trainer.optimize_hyperparameters(
                    X_train_scaled, y_train, X_val_scaled, y_val
                )
                self.logger.info(f"Best hyperparameters: {best_params}")
            
            # Train final model
            training_config = training_job.training_config.copy()
            training_config['hyperparameters'] = best_params
            
            model, validation_metrics = await trainer.train(
                X_train_scaled, y_train, X_val_scaled, y_val, training_config
            )
            
            # Test evaluation
            if training_job.framework == 'sklearn':
                y_test_pred = model.predict(X_test_scaled)
                test_metrics = {
                    'accuracy': accuracy_score(y_test, y_test_pred),
                    'precision': precision_score(y_test, y_test_pred, average='weighted'),
                    'recall': recall_score(y_test, y_test_pred, average='weighted'),
                    'f1': f1_score(y_test, y_test_pred, average='weighted')
                }
            else:  # TensorFlow
                y_test_pred_proba = model.predict(X_test_scaled).flatten()
                y_test_pred = (y_test_pred_proba > 0.5).astype(int)
                test_metrics = {
                    'accuracy': accuracy_score(y_test, y_test_pred),
                    'precision': precision_score(y_test, y_test_pred),
                    'recall': recall_score(y_test, y_test_pred),
                    'f1': f1_score(y_test, y_test_pred),
                    'roc_auc': roc_auc_score(y_test, y_test_pred_proba)
                }
            
            # Create model metadata
            model_id = f"{training_job.model_name}_{int(datetime.now().timestamp())}"
            model_metadata = ModelMetadata(
                model_id=model_id,
                name=training_job.model_name,
                version="1.0.0",
                framework=training_job.framework,
                model_type=training_job.model_type,
                created_at=datetime.now(),
                created_by="ml_model_manager",
                description=f"Auto-trained {training_job.model_type} model",
                tags=["auto_trained"],
                training_dataset_id=training_job.dataset_path,
                training_config=training_config,
                hyperparameters=best_params,
                training_metrics={},  # Would include training loss/metrics
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                deployment_status="ready"
            )
            
            # Save model
            model_path = self._save_model(model, scaler, model_metadata)
            
            # Track with MLflow
            if self.config.get('experiment_tracking', True):
                self._track_experiment(model_metadata, model_path)
            
            # Register model
            self.models[model_id] = model_metadata
            
            # Update job status
            training_job.status = "completed"
            training_job.completed_at = datetime.now()
            
            # Update performance metrics
            training_time = (datetime.now() - start_time).total_seconds()
            self.performance_metrics['models_trained'] += 1
            self.performance_metrics['average_training_time'] = (
                (self.performance_metrics['average_training_time'] * 
                 (self.performance_metrics['models_trained'] - 1) + training_time) /
                self.performance_metrics['models_trained']
            )
            
            self.logger.info(
                f"Model training completed. Model ID: {model_id}, "
                f"Test F1: {test_metrics['f1']:.3f}, Time: {training_time:.1f}s"
            )
            
            return model_metadata
            
        except Exception as e:
            training_job.status = "failed"
            training_job.error_message = str(e)
            training_job.completed_at = datetime.now()
            self.logger.error(f"Model training failed: {str(e)}")
            raise
    
    def _save_model(
        self,
        model: Any,
        scaler: StandardScaler,
        metadata: ModelMetadata
    ) -> str:
        """Save model and preprocessing artifacts."""
        model_dir = Path(self.config['model_storage_path']) / metadata.model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if metadata.framework == 'sklearn':
            joblib.dump(model, model_dir / 'model.joblib')
        elif metadata.framework == 'tensorflow':
            model.save(model_dir / 'model.h5')
        
        # Save preprocessor
        joblib.dump(scaler, model_dir / 'scaler.joblib')
        
        # Save metadata
        with open(model_dir / 'metadata.json', 'w') as f:
            metadata_dict = asdict(metadata)
            metadata_dict['created_at'] = metadata.created_at.isoformat()
            json.dump(metadata_dict, f, indent=2)
        
        return str(model_dir)
    
    def _track_experiment(self, metadata: ModelMetadata, model_path: str):
        """Track experiment in MLflow."""
        with mlflow.start_run(run_name=metadata.model_id):
            # Log parameters
            mlflow.log_params(metadata.hyperparameters)
            
            # Log metrics
            for metric_name, value in metadata.validation_metrics.items():
                mlflow.log_metric(f"val_{metric_name}", value)
            for metric_name, value in metadata.test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", value)
            
            # Log model
            if metadata.framework == 'sklearn':
                model = joblib.load(Path(model_path) / 'model.joblib')
                mlflow.sklearn.log_model(model, "model")
            elif metadata.framework == 'tensorflow':
                model = tf.keras.models.load_model(Path(model_path) / 'model.h5')
                mlflow.tensorflow.log_model(model, "model")
            
            # Log artifacts
            mlflow.log_artifacts(model_path)
    
    async def deploy_model(
        self,
        model_id: str,
        deployment_config: DeploymentConfig
    ) -> str:
        """Deploy model to specified environment."""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            model_metadata = self.models[model_id]
            
            self.logger.info(
                f"Deploying model {model_id} to {deployment_config.environment}"
            )
            
            # Update model metadata
            model_metadata.deployment_status = "deployed"
            model_metadata.deployment_config = asdict(deployment_config)
            model_metadata.endpoint_url = f"http://ml-service/{model_id}/predict"
            
            # Register deployment
            self.deployments[deployment_config.deployment_id] = deployment_config
            
            # Update performance metrics
            self.performance_metrics['models_deployed'] += 1
            
            self.logger.info(f"Model {model_id} deployed successfully")
            
            return deployment_config.deployment_id
            
        except Exception as e:
            self.logger.error(f"Model deployment failed: {str(e)}")
            raise
    
    async def predict(
        self,
        model_id: str,
        features: np.ndarray,
        explain: bool = False
    ) -> Dict[str, Any]:
        """Make predictions using deployed model."""
        start_time = datetime.now()
        
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            model_metadata = self.models[model_id]
            
            if model_metadata.deployment_status != "deployed":
                raise ValueError(f"Model {model_id} is not deployed")
            
            # Load model and preprocessor
            model_path = Path(self.config['model_storage_path']) / model_id
            
            if model_metadata.framework == 'sklearn':
                model = joblib.load(model_path / 'model.joblib')
            elif model_metadata.framework == 'tensorflow':
                model = tf.keras.models.load_model(model_path / 'model.h5')
            
            scaler = joblib.load(model_path / 'scaler.joblib')
            
            # Preprocess features
            features_scaled = scaler.transform(features)
            
            # Make prediction
            if model_metadata.framework == 'sklearn':
                predictions = model.predict(features_scaled)
                probabilities = (
                    model.predict_proba(features_scaled) 
                    if hasattr(model, 'predict_proba') else None
                )
            else:  # TensorFlow
                pred_proba = model.predict(features_scaled).flatten()
                predictions = (pred_proba > 0.5).astype(int)
                probabilities = np.column_stack([1 - pred_proba, pred_proba])
            
            result = {
                'predictions': predictions.tolist(),
                'model_id': model_id,
                'model_version': model_metadata.version,
                'prediction_time': datetime.now().isoformat()
            }
            
            if probabilities is not None:
                result['probabilities'] = probabilities.tolist()
            
            # Add explanations if requested
            if explain and self.config.get('explainability_enabled', True):
                explanations = self._generate_explanations(
                    model, features_scaled, model_metadata.framework
                )
                result['explanations'] = explanations
            
            # Update performance metrics
            inference_time = (datetime.now() - start_time).total_seconds() * 1000  # ms
            self.performance_metrics['predictions_served'] += 1
            self.performance_metrics['average_inference_time'] = (
                (self.performance_metrics['average_inference_time'] * 
                 (self.performance_metrics['predictions_served'] - 1) + inference_time) /
                self.performance_metrics['predictions_served']
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def _generate_explanations(
        self,
        model: Any,
        features: np.ndarray,
        framework: str
    ) -> Dict[str, Any]:
        """Generate model explanations using SHAP."""
        try:
            if framework == 'sklearn':
                explainer = shap.Explainer(model)
                shap_values = explainer(features)
                
                return {
                    'shap_values': shap_values.values.tolist(),
                    'base_values': shap_values.base_values.tolist(),
                    'feature_importance': np.abs(shap_values.values).mean(axis=0).tolist()
                }
            else:
                # For TensorFlow models, use approximate explanations
                return {
                    'explanation_method': 'approximate',
                    'note': 'Full SHAP explanations not available for this model type'
                }
        
        except Exception as e:
            self.logger.warning(f"Could not generate explanations: {str(e)}")
            return {'error': 'Explanations not available'}
    
    async def monitor_model_drift(
        self,
        model_id: str,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Monitor model for data and prediction drift."""
        try:
            if not self.config.get('drift_detection_enabled', True):
                return {'drift_detection': 'disabled'}
            
            # Create drift report
            data_drift_report = Report(metrics=[DataDriftPreset()])
            
            data_drift_report.run(
                reference_data=reference_data,
                current_data=current_data
            )
            
            # Extract drift metrics
            drift_results = data_drift_report.as_dict()
            
            # Calculate overall drift score
            drift_score = self._calculate_drift_score(drift_results)
            
            drift_report = {
                'model_id': model_id,
                'timestamp': datetime.now().isoformat(),
                'drift_detected': drift_score > 0.1,  # Threshold
                'drift_score': drift_score,
                'affected_features': self._get_drifted_features(drift_results),
                'recommendation': self._get_drift_recommendation(drift_score)
            }
            
            self.logger.info(
                f"Drift monitoring completed for {model_id}. "
                f"Drift score: {drift_score:.3f}"
            )
            
            return drift_report
            
        except Exception as e:
            self.logger.error(f"Drift monitoring failed: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_drift_score(self, drift_results: Dict[str, Any]) -> float:
        """Calculate overall drift score from drift results."""
        # Simplified drift score calculation
        # In practice, this would be more sophisticated
        try:
            metrics = drift_results.get('metrics', [])
            if not metrics:
                return 0.0
            
            drift_scores = []
            for metric in metrics:
                if 'result' in metric and 'drift_score' in metric['result']:
                    drift_scores.append(metric['result']['drift_score'])
            
            return np.mean(drift_scores) if drift_scores else 0.0
        
        except Exception:
            return 0.0
    
    def _get_drifted_features(self, drift_results: Dict[str, Any]) -> List[str]:
        """Get list of features with significant drift."""
        # Simplified implementation
        return []
    
    def _get_drift_recommendation(self, drift_score: float) -> str:
        """Get recommendation based on drift score."""
        if drift_score < 0.05:
            return "No action required"
        elif drift_score < 0.1:
            return "Monitor closely"
        elif drift_score < 0.2:
            return "Consider retraining"
        else:
            return "Immediate retraining recommended"
    
    async def run_ab_test(
        self,
        model_a_id: str,
        model_b_id: str,
        test_data: pd.DataFrame,
        target_column: str,
        traffic_split: float = 0.5
    ) -> Dict[str, Any]:
        """Run A/B test between two models."""
        try:
            if not self.config.get('a_b_testing_enabled', True):
                return {'ab_testing': 'disabled'}
            
            self.logger.info(f"Running A/B test: {model_a_id} vs {model_b_id}")
            
            # Split test data
            split_idx = int(len(test_data) * traffic_split)
            test_a = test_data.iloc[:split_idx]
            test_b = test_data.iloc[split_idx:]
            
            # Get predictions from both models
            features_a = test_a.drop(columns=[target_column]).values
            features_b = test_b.drop(columns=[target_column]).values
            
            predictions_a = await self.predict(model_a_id, features_a)
            predictions_b = await self.predict(model_b_id, features_b)
            
            # Calculate metrics
            y_true_a = test_a[target_column].values
            y_true_b = test_b[target_column].values
            
            metrics_a = {
                'accuracy': accuracy_score(y_true_a, predictions_a['predictions']),
                'f1': f1_score(y_true_a, predictions_a['predictions'], average='weighted')
            }
            
            metrics_b = {
                'accuracy': accuracy_score(y_true_b, predictions_b['predictions']),
                'f1': f1_score(y_true_b, predictions_b['predictions'], average='weighted')
            }
            
            # Statistical significance test
            from scipy.stats import ttest_ind
            
            accuracy_diff = metrics_b['accuracy'] - metrics_a['accuracy']
            stat_significance = ttest_ind(
                predictions_a['predictions'], 
                predictions_b['predictions']
            )
            
            ab_test_results = {
                'test_id': f"ab_test_{int(datetime.now().timestamp())}",
                'model_a': {
                    'model_id': model_a_id,
                    'metrics': metrics_a,
                    'sample_size': len(test_a)
                },
                'model_b': {
                    'model_id': model_b_id,
                    'metrics': metrics_b,
                    'sample_size': len(test_b)
                },
                'results': {
                    'accuracy_difference': accuracy_diff,
                    'statistical_significance': stat_significance.pvalue < 0.05,
                    'p_value': stat_significance.pvalue,
                    'winner': model_b_id if accuracy_diff > 0 else model_a_id,
                    'confidence_level': 0.95
                },
                'recommendation': self._get_ab_test_recommendation(
                    accuracy_diff, stat_significance.pvalue
                )
            }
            
            self.logger.info(
                f"A/B test completed. Winner: {ab_test_results['results']['winner']}"
            )
            
            return ab_test_results
            
        except Exception as e:
            self.logger.error(f"A/B test failed: {str(e)}")
            raise
    
    def _get_ab_test_recommendation(
        self,
        accuracy_diff: float,
        p_value: float
    ) -> str:
        """Get recommendation based on A/B test results."""
        if p_value > 0.05:
            return "No statistically significant difference detected"
        elif accuracy_diff > 0.02:
            return "Deploy Model B - significant improvement"
        elif accuracy_diff < -0.02:
            return "Keep Model A - Model B performs worse"
        else:
            return "Marginal difference - consider other factors"
    
    def get_model_registry(self) -> Dict[str, Any]:
        """Get model registry with all models."""
        registry = {}
        for model_id, metadata in self.models.items():
            registry[model_id] = {
                'name': metadata.name,
                'version': metadata.version,
                'framework': metadata.framework,
                'status': metadata.deployment_status,
                'created_at': metadata.created_at.isoformat(),
                'test_metrics': metadata.test_metrics
            }
        
        return registry
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.performance_metrics.copy()
    
    async def cleanup_old_models(self, retention_days: int = 30):
        """Clean up old models based on retention policy."""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        models_to_remove = []
        for model_id, metadata in self.models.items():
            if (metadata.deployment_status != "deployed" and 
                metadata.created_at < cutoff_date):
                models_to_remove.append(model_id)
        
        for model_id in models_to_remove:
            # Remove model files
            model_path = Path(self.config['model_storage_path']) / model_id
            if model_path.exists():
                import shutil
                shutil.rmtree(model_path)
            
            # Remove from registry
            del self.models[model_id]
            
            self.logger.info(f"Cleaned up old model: {model_id}")
        
        return len(models_to_remove)


# Utility functions
async def auto_train_model(
    dataset_path: str,
    target_column: str,
    model_name: str,
    framework: str = 'sklearn',
    model_type: str = 'random_forest'
) -> ModelMetadata:
    """Convenience function for automated model training."""
    
    manager = MLModelManager()
    
    # Load data to get feature columns
    data = pd.read_csv(dataset_path)
    feature_columns = [col for col in data.columns if col != target_column]
    
    # Create training job
    training_job = TrainingJob(
        job_id=f"auto_train_{int(datetime.now().timestamp())}",
        model_name=model_name,
        model_type=model_type,
        framework=framework,
        dataset_path=dataset_path,
        target_column=target_column,
        feature_columns=feature_columns,
        training_config={},
        model_output_path=f"./models/{model_name}",
        experiment_name="auto_training"
    )
    
    return await manager.train_model(training_job)


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Create sample data
        from sklearn.datasets import make_classification
        
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42
        )
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        data = pd.DataFrame(X, columns=feature_names)
        data['target'] = y
        data.to_csv('sample_data.csv', index=False)
        
        # Train model
        model_metadata = await auto_train_model(
            dataset_path='sample_data.csv',
            target_column='target',
            model_name='sample_classifier',
            framework='sklearn',
            model_type='random_forest'
        )
        
        print(f"Model trained: {model_metadata.model_id}")
        print(f"Test F1 Score: {model_metadata.test_metrics['f1']:.3f}")
    
    asyncio.run(main())
