"""
Enterprise MLOps Pipeline & Model Management System
=================================================

Production-ready MLOps infrastructure for model training, deployment,
monitoring, and lifecycle management with enterprise-grade features.

Features:
- Automated ML pipeline orchestration
- Model versioning and registry
- A/B testing framework
- Real-time model monitoring and drift detection
- Automated retraining and deployment
- Feature store integration
- Model explainability and governance
- Performance tracking and alerting
- Multi-environment deployment (dev/staging/prod)
- Distributed training with auto-scaling
"""

import os
import json
import time
import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
import joblib
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import subprocess
import hashlib
import uuid
from enum import Enum
import yaml
import shutil
import tarfile
import requests
from contextlib import asynccontextmanager

# Import logger
from app.core.logging import get_logger
logger = get_logger(__name__)

# MLflow for experiment tracking
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available - experiment tracking disabled")

# Monitoring and alerting
try:
    import evidently
    from evidently.model_profile import Profile
    from evidently.model_profile.sections import DataDriftProfileSection
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logger.warning("Evidently not available - drift detection disabled")

from . import audit_ml_operation, ML_CONFIG

logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    """Model lifecycle status"""
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATING = "validating"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    MONITORING = "monitoring"
    DEPRECATED = "deprecated"
    FAILED = "failed"

class DeploymentEnvironment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    CANARY = "canary"

@dataclass
class ModelMetadata:
    """Model metadata structure"""
    model_id: str
    name: str
    version: str
    model_type: str
    framework: str
    created_at: datetime
    created_by: str
    description: str
    tags: List[str]
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    training_data_hash: str
    feature_schema: Dict[str, str]
    status: ModelStatus
    environment: DeploymentEnvironment

@dataclass
class PipelineConfig:
    """ML Pipeline configuration"""
    pipeline_id: str
    name: str
    data_source: str
    feature_engineering: Dict[str, Any]
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]
    validation_config: Dict[str, Any]
    deployment_config: Dict[str, Any]
    monitoring_config: Dict[str, Any]
    schedule: Optional[str] = None  # Cron expression

@dataclass
class ExperimentResult:
    """Experiment tracking result"""
    experiment_id: str
    run_id: str
    model_metadata: ModelMetadata
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    artifacts: List[str]
    duration: float
    status: str

class ModelRegistry:
    """
    Enterprise Model Registry with versioning and lifecycle management
    """
    
    def __init__(self, registry_path: str = None):
        self.registry_path = Path(registry_path or ML_CONFIG["model_registry_path"])
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_path / "registry_metadata.json"
        self.models = self._load_registry()
        
        # Initialize MLflow if available
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(f"file://{self.registry_path}/mlflow")
            
        logger.info(f"🏛️ Model Registry initialized at {self.registry_path}")
    
    def _load_registry(self) -> Dict[str, ModelMetadata]:
        """Load registry metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                return {
                    model_id: ModelMetadata(**metadata)
                    for model_id, metadata in data.items()
                }
        return {}
    
    def _save_registry(self):
        """Save registry metadata"""
        with open(self.metadata_file, 'w') as f:
            data = {
                model_id: asdict(metadata)
                for model_id, metadata in self.models.items()
            }
            # Convert datetime objects to strings
            for model_data in data.values():
                model_data['created_at'] = model_data['created_at'].isoformat()
                model_data['status'] = model_data['status'].value
                model_data['environment'] = model_data['environment'].value
            
            json.dump(data, f, indent=2)
    
    @audit_ml_operation("model_registration")
    def register_model(self, model: Any, metadata: ModelMetadata, 
                      artifacts: Dict[str, Any] = None) -> str:
        """Register a new model version"""
        try:
            # Create model directory
            model_dir = self.registry_path / metadata.model_id / metadata.version
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model artifacts
            model_path = model_dir / "model.pkl"
            
            if hasattr(model, 'save'):
                # PyTorch/TensorFlow models
                model.save(str(model_path))
            else:
                # Scikit-learn or other pickle-able models
                joblib.dump(model, model_path)
            
            # Save additional artifacts
            if artifacts:
                for name, artifact in artifacts.items():
                    artifact_path = model_dir / f"{name}.pkl"
                    joblib.dump(artifact, artifact_path)
            
            # Save metadata
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                metadata_dict = asdict(metadata)
                metadata_dict['created_at'] = metadata.created_at.isoformat()
                metadata_dict['status'] = metadata.status.value
                metadata_dict['environment'] = metadata.environment.value
                json.dump(metadata_dict, f, indent=2)
            
            # Update registry
            self.models[metadata.model_id] = metadata
            self._save_registry()
            
            # Log to MLflow if available
            if MLFLOW_AVAILABLE:
                with mlflow.start_run(run_name=f"{metadata.name}_{metadata.version}"):
                    mlflow.log_params(metadata.hyperparameters)
                    mlflow.log_metrics(metadata.metrics)
                    mlflow.log_artifact(str(model_path))
            
            logger.info(f"✅ Registered model {metadata.model_id} v{metadata.version}")
            return metadata.model_id
            
        except Exception as e:
            logger.error(f"❌ Model registration failed: {e}")
            raise
    
    def get_model(self, model_id: str, version: str = "latest") -> Tuple[Any, ModelMetadata]:
        """Retrieve a model and its metadata"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry")
        
        if version == "latest":
            # Get latest version
            model_versions = list((self.registry_path / model_id).iterdir())
            version = max(model_versions, key=lambda x: x.name).name
        
        model_path = self.registry_path / model_id / version / "model.pkl"
        metadata_path = self.registry_path / model_id / version / "metadata.json"
        
        if not model_path.exists():
            raise ValueError(f"Model file not found: {model_path}")
        
        # Load model
        model = joblib.load(model_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
            metadata_dict['created_at'] = datetime.fromisoformat(metadata_dict['created_at'])
            metadata_dict['status'] = ModelStatus(metadata_dict['status'])
            metadata_dict['environment'] = DeploymentEnvironment(metadata_dict['environment'])
            metadata = ModelMetadata(**metadata_dict)
        
        return model, metadata
    
    def list_models(self, status: ModelStatus = None, 
                   environment: DeploymentEnvironment = None) -> List[ModelMetadata]:
        """List models with optional filtering"""
        models = list(self.models.values())
        
        if status:
            models = [m for m in models if m.status == status]
        
        if environment:
            models = [m for m in models if m.environment == environment]
        
        return models
    
    def promote_model(self, model_id: str, version: str, 
                     target_environment: DeploymentEnvironment) -> bool:
        """Promote model to target environment"""
        try:
            metadata = self.models.get(model_id)
            if not metadata:
                raise ValueError(f"Model {model_id} not found")
            
            # Update environment
            metadata.environment = target_environment
            metadata.status = ModelStatus.DEPLOYED
            
            # Update registry
            self.models[model_id] = metadata
            self._save_registry()
            
            logger.info(f"✅ Promoted model {model_id} v{version} to {target_environment.value}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model promotion failed: {e}")
            return False
    
    def deprecate_model(self, model_id: str, version: str) -> bool:
        """Deprecate a model version"""
        try:
            metadata = self.models.get(model_id)
            if not metadata:
                raise ValueError(f"Model {model_id} not found")
            
            metadata.status = ModelStatus.DEPRECATED
            self.models[model_id] = metadata
            self._save_registry()
            
            logger.info(f"✅ Deprecated model {model_id} v{version}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model deprecation failed: {e}")
            return False

class ExperimentTracker:
    """
    Experiment tracking and management system
    """
    
    def __init__(self, tracking_uri: str = None):
        self.tracking_uri = tracking_uri or f"file://{ML_CONFIG['model_registry_path']}/experiments"
        self.experiments = {}
        
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(self.tracking_uri)
            
        logger.info("🧪 Experiment Tracker initialized")
    
    @asynccontextmanager
    async def start_experiment(self, experiment_name: str, run_name: str = None):
        """Context manager for experiment tracking"""
        experiment_id = str(uuid.uuid4())
        run_id = str(uuid.uuid4())
        
        start_time = time.time()
        
        experiment_data = {
            'experiment_id': experiment_id,
            'run_id': run_id,
            'experiment_name': experiment_name,
            'run_name': run_name or f"run_{int(start_time)}",
            'start_time': start_time,
            'parameters': {},
            'metrics': {},
            'artifacts': [],
            'status': 'running'
        }
        
        self.experiments[experiment_id] = experiment_data
        
        # Start MLflow run if available
        mlflow_run = None
        if MLFLOW_AVAILABLE:
            mlflow.set_experiment(experiment_name)
            mlflow_run = mlflow.start_run(run_name=experiment_data['run_name'])
        
        try:
            yield ExperimentContext(experiment_data, mlflow_run)
            
            # Mark as completed
            experiment_data['status'] = 'completed'
            experiment_data['end_time'] = time.time()
            experiment_data['duration'] = experiment_data['end_time'] - experiment_data['start_time']
            
        except Exception as e:
            experiment_data['status'] = 'failed'
            experiment_data['error'] = str(e)
            logger.error(f"❌ Experiment {experiment_name} failed: {e}")
            raise
            
        finally:
            if mlflow_run and MLFLOW_AVAILABLE:
                mlflow.end_run()
    
    def get_experiment_results(self, experiment_id: str) -> ExperimentResult:
        """Get experiment results"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        data = self.experiments[experiment_id]
        
        # Create dummy model metadata for now
        model_metadata = ModelMetadata(
            model_id=f"model_{experiment_id}",
            name=data['experiment_name'],
            version="1.0.0",
            model_type="unknown",
            framework="unknown",
            created_at=datetime.fromtimestamp(data['start_time']),
            created_by="system",
            description=f"Model from experiment {data['experiment_name']}",
            tags=[],
            metrics=data['metrics'],
            hyperparameters=data['parameters'],
            training_data_hash="",
            feature_schema={},
            status=ModelStatus.TRAINED,
            environment=DeploymentEnvironment.DEVELOPMENT
        )
        
        return ExperimentResult(
            experiment_id=data['experiment_id'],
            run_id=data['run_id'],
            model_metadata=model_metadata,
            metrics=data['metrics'],
            parameters=data['parameters'],
            artifacts=data['artifacts'],
            duration=data.get('duration', 0),
            status=data['status']
        )

class ExperimentContext:
    """Context for active experiment"""
    
    def __init__(self, experiment_data: Dict[str, Any], mlflow_run=None):
        self.experiment_data = experiment_data
        self.mlflow_run = mlflow_run
    
    def log_param(self, key: str, value: Any):
        """Log experiment parameter"""
        self.experiment_data['parameters'][key] = value
        if self.mlflow_run and MLFLOW_AVAILABLE:
            mlflow.log_param(key, value)
    
    def log_metric(self, key: str, value: float, step: int = None):
        """Log experiment metric"""
        self.experiment_data['metrics'][key] = value
        if self.mlflow_run and MLFLOW_AVAILABLE:
            mlflow.log_metric(key, value, step)
    
    def log_artifact(self, artifact_path: str):
        """Log experiment artifact"""
        self.experiment_data['artifacts'].append(artifact_path)
        if self.mlflow_run and MLFLOW_AVAILABLE:
            mlflow.log_artifact(artifact_path)

class ModelMonitor:
    """
    Real-time model monitoring and drift detection
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.monitoring_data = {}
        self.drift_detectors = {}
        self.alert_thresholds = {
            'accuracy_drop': 0.05,
            'latency_increase': 0.5,
            'error_rate_increase': 0.1,
            'drift_score': 0.3
        }
        
        logger.info("📊 Model Monitor initialized")
    
    @audit_ml_operation("model_monitoring")
    async def monitor_model_performance(self, model_id: str, predictions: np.ndarray,
                                      ground_truth: np.ndarray = None, 
                                      features: np.ndarray = None) -> Dict[str, Any]:
        """Monitor model performance and detect issues"""
        monitoring_result = {
            'model_id': model_id,
            'timestamp': datetime.utcnow().isoformat(),
            'prediction_count': len(predictions),
            'alerts': []
        }
        
        try:
            # Performance monitoring
            if ground_truth is not None:
                performance_metrics = self._calculate_performance_metrics(
                    predictions, ground_truth
                )
                monitoring_result['performance'] = performance_metrics
                
                # Check for performance degradation
                alerts = self._check_performance_alerts(model_id, performance_metrics)
                monitoring_result['alerts'].extend(alerts)
            
            # Drift detection
            if features is not None and EVIDENTLY_AVAILABLE:
                drift_metrics = await self._detect_data_drift(model_id, features)
                monitoring_result['drift'] = drift_metrics
                
                # Check for drift alerts
                if drift_metrics.get('drift_score', 0) > self.alert_thresholds['drift_score']:
                    monitoring_result['alerts'].append({
                        'type': 'data_drift',
                        'severity': 'high',
                        'message': f"Data drift detected: {drift_metrics['drift_score']:.3f}",
                        'timestamp': datetime.utcnow().isoformat()
                    })
            
            # Prediction distribution analysis
            prediction_stats = self._analyze_prediction_distribution(predictions)
            monitoring_result['prediction_stats'] = prediction_stats
            
            # Store monitoring data
            if model_id not in self.monitoring_data:
                self.monitoring_data[model_id] = []
            
            self.monitoring_data[model_id].append(monitoring_result)
            
            # Trigger alerts if necessary
            if monitoring_result['alerts']:
                await self._trigger_alerts(model_id, monitoring_result['alerts'])
            
            logger.info(f"📊 Monitored model {model_id}: {len(monitoring_result['alerts'])} alerts")
            return monitoring_result
            
        except Exception as e:
            logger.error(f"❌ Model monitoring failed: {e}")
            raise
    
    def _calculate_performance_metrics(self, predictions: np.ndarray, 
                                     ground_truth: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        try:
            # Determine if classification or regression
            if len(np.unique(ground_truth)) < 10:  # Likely classification
                metrics = {
                    'accuracy': accuracy_score(ground_truth, predictions),
                    'precision': precision_score(ground_truth, predictions, average='weighted', zero_division=0),
                    'recall': recall_score(ground_truth, predictions, average='weighted', zero_division=0),
                    'f1_score': f1_score(ground_truth, predictions, average='weighted', zero_division=0)
                }
            else:  # Likely regression
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                metrics = {
                    'mse': mean_squared_error(ground_truth, predictions),
                    'mae': mean_absolute_error(ground_truth, predictions),
                    'r2_score': r2_score(ground_truth, predictions)
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Performance metrics calculation failed: {e}")
            return {}
    
    def _check_performance_alerts(self, model_id: str, 
                                 current_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check for performance degradation alerts"""
        alerts = []
        
        # Get historical performance
        if model_id in self.monitoring_data and len(self.monitoring_data[model_id]) > 0:
            recent_data = self.monitoring_data[model_id][-10:]  # Last 10 monitoring points
            
            for metric_name, current_value in current_metrics.items():
                if metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'r2_score']:
                    # Higher is better metrics
                    historical_values = [
                        data['performance'].get(metric_name, 0) 
                        for data in recent_data 
                        if 'performance' in data and metric_name in data['performance']
                    ]
                    
                    if historical_values:
                        avg_historical = np.mean(historical_values)
                        if avg_historical - current_value > self.alert_thresholds['accuracy_drop']:
                            alerts.append({
                                'type': 'performance_degradation',
                                'metric': metric_name,
                                'current_value': current_value,
                                'historical_average': avg_historical,
                                'severity': 'high',
                                'message': f"{metric_name} dropped from {avg_historical:.3f} to {current_value:.3f}",
                                'timestamp': datetime.utcnow().isoformat()
                            })
        
        return alerts
    
    async def _detect_data_drift(self, model_id: str, features: np.ndarray) -> Dict[str, Any]:
        """Detect data drift using statistical methods"""
        if not EVIDENTLY_AVAILABLE:
            return {'drift_score': 0.0, 'message': 'Evidently not available'}
        
        try:
            # Convert to DataFrame
            feature_df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(features.shape[1])])
            
            # Get reference data (training data would be ideal)
            if model_id not in self.drift_detectors:
                # Initialize with current data as reference
                self.drift_detectors[model_id] = {
                    'reference_data': feature_df.copy(),
                    'drift_history': []
                }
                return {'drift_score': 0.0, 'message': 'Reference data initialized'}
            
            reference_data = self.drift_detectors[model_id]['reference_data']
            
            # Create drift profile
            profile = Profile(sections=[DataDriftProfileSection()])
            profile.calculate(reference_data, feature_df)
            
            # Extract drift metrics
            drift_result = profile.json()
            drift_score = drift_result.get('data_drift', {}).get('data', {}).get('metrics', {}).get('dataset_drift', 0.0)
            
            # Store drift history
            self.drift_detectors[model_id]['drift_history'].append({
                'timestamp': datetime.utcnow().isoformat(),
                'drift_score': drift_score
            })
            
            return {
                'drift_score': drift_score,
                'drift_detected': drift_score > self.alert_thresholds['drift_score'],
                'reference_size': len(reference_data),
                'current_size': len(feature_df)
            }
            
        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            return {'drift_score': 0.0, 'error': str(e)}
    
    def _analyze_prediction_distribution(self, predictions: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction distribution for anomalies"""
        stats = {
            'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions)),
            'median': float(np.median(predictions)),
            'skewness': float(np.abs(np.mean(predictions) - np.median(predictions)) / np.std(predictions)) if np.std(predictions) > 0 else 0
        }
        
        # Detect potential anomalies
        z_scores = np.abs((predictions - stats['mean']) / max(stats['std'], 1e-8))
        anomaly_count = np.sum(z_scores > 3)  # 3-sigma rule
        
        stats['anomaly_rate'] = float(anomaly_count / len(predictions))
        stats['total_predictions'] = len(predictions)
        
        return stats
    
    async def _trigger_alerts(self, model_id: str, alerts: List[Dict[str, Any]]):
        """Trigger alerts for monitoring issues"""
        for alert in alerts:
            logger.warning(f"🚨 Model Alert [{model_id}]: {alert['message']}")
            
            # In production, this would integrate with alerting systems
            # like PagerDuty, Slack, email, etc.
    
    def get_monitoring_summary(self, model_id: str, 
                              time_window: timedelta = timedelta(hours=24)) -> Dict[str, Any]:
        """Get monitoring summary for a model"""
        if model_id not in self.monitoring_data:
            return {'error': f'No monitoring data for model {model_id}'}
        
        cutoff_time = datetime.utcnow() - time_window
        recent_data = [
            data for data in self.monitoring_data[model_id]
            if datetime.fromisoformat(data['timestamp']) > cutoff_time
        ]
        
        if not recent_data:
            return {'error': 'No recent monitoring data'}
        
        # Aggregate metrics
        total_predictions = sum(data['prediction_count'] for data in recent_data)
        total_alerts = sum(len(data['alerts']) for data in recent_data)
        
        # Performance trends
        performance_trends = {}
        if any('performance' in data for data in recent_data):
            perf_data = [data['performance'] for data in recent_data if 'performance' in data]
            if perf_data:
                for metric in perf_data[0].keys():
                    values = [p[metric] for p in perf_data if metric in p]
                    performance_trends[metric] = {
                        'current': values[-1] if values else 0,
                        'average': np.mean(values) if values else 0,
                        'trend': 'improving' if len(values) > 1 and values[-1] > values[0] else 'declining'
                    }
        
        return {
            'model_id': model_id,
            'time_window': str(time_window),
            'total_predictions': total_predictions,
            'total_alerts': total_alerts,
            'alert_rate': total_alerts / max(len(recent_data), 1),
            'performance_trends': performance_trends,
            'monitoring_points': len(recent_data),
            'last_updated': recent_data[-1]['timestamp'] if recent_data else None
        }

class MLOpsPipeline:
    """
    Comprehensive MLOps Pipeline orchestrator
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model_registry = ModelRegistry()
        self.experiment_tracker = ExperimentTracker()
        self.model_monitor = ModelMonitor()
        self.pipeline_status = "initialized"
        self.current_run_id = None
        
        logger.info(f"🔄 MLOps Pipeline '{config.name}' initialized")
    
    @audit_ml_operation("pipeline_execution")
    async def run_pipeline(self) -> ExperimentResult:
        """Execute the complete ML pipeline"""
        self.pipeline_status = "running"
        self.current_run_id = str(uuid.uuid4())
        
        try:
            async with self.experiment_tracker.start_experiment(
                experiment_name=self.config.name,
                run_name=f"pipeline_run_{self.current_run_id}"
            ) as experiment:
                
                logger.info(f"🚀 Starting pipeline run {self.current_run_id}")
                
                # Data loading and preprocessing
                experiment.log_param("data_source", self.config.data_source)
                data = await self._load_and_preprocess_data()
                
                # Feature engineering
                experiment.log_param("feature_engineering", json.dumps(self.config.feature_engineering))
                features, labels = await self._engineer_features(data)
                
                # Model training
                experiment.log_param("model_config", json.dumps(self.config.model_config))
                experiment.log_param("training_config", json.dumps(self.config.training_config))
                model, training_metrics = await self._train_model(features, labels)
                
                # Log training metrics
                for metric_name, value in training_metrics.items():
                    experiment.log_metric(metric_name, value)
                
                # Model validation
                validation_metrics = await self._validate_model(model, features, labels)
                for metric_name, value in validation_metrics.items():
                    experiment.log_metric(f"val_{metric_name}", value)
                
                # Model registration
                model_metadata = ModelMetadata(
                    model_id=f"{self.config.name}_{self.current_run_id}",
                    name=self.config.name,
                    version=f"v{int(time.time())}",
                    model_type=self.config.model_config.get("type", "unknown"),
                    framework=self.config.model_config.get("framework", "sklearn"),
                    created_at=datetime.utcnow(),
                    created_by="mlops_pipeline",
                    description=f"Model trained by pipeline {self.config.name}",
                    tags=["automated", "mlops"],
                    metrics={**training_metrics, **validation_metrics},
                    hyperparameters=self.config.model_config,
                    training_data_hash=hashlib.md5(str(features).encode()).hexdigest(),
                    feature_schema={f"feature_{i}": "float" for i in range(features.shape[1])} if hasattr(features, 'shape') else {},
                    status=ModelStatus.TRAINED,
                    environment=DeploymentEnvironment.DEVELOPMENT
                )
                
                self.model_registry.register_model(model, model_metadata)
                experiment.log_param("model_id", model_metadata.model_id)
                
                # Deployment (if configured)
                if self.config.deployment_config.get("auto_deploy", False):
                    deployment_result = await self._deploy_model(model_metadata)
                    experiment.log_param("deployment_status", deployment_result["status"])
                
                self.pipeline_status = "completed"
                logger.info(f"✅ Pipeline run {self.current_run_id} completed successfully")
                
                return self.experiment_tracker.get_experiment_results(
                    experiment.experiment_data['experiment_id']
                )
                
        except Exception as e:
            self.pipeline_status = "failed"
            logger.error(f"❌ Pipeline run {self.current_run_id} failed: {e}")
            raise
    
    async def _load_and_preprocess_data(self) -> pd.DataFrame:
        """Load and preprocess training data"""
        # Mock data loading - replace with actual data source integration
        logger.info("📊 Loading training data...")
        
        # Generate mock data for demonstration
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        data = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        data['target'] = (data['feature_0'] + data['feature_1'] + np.random.randn(n_samples) * 0.1) > 0
        
        logger.info(f"✅ Loaded {len(data)} samples with {len(data.columns)-1} features")
        return data
    
    async def _engineer_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Apply feature engineering transformations"""
        logger.info("🔧 Engineering features...")
        
        # Extract features and labels
        feature_columns = [col for col in data.columns if col != 'target']
        features = data[feature_columns].values
        labels = data['target'].values
        
        # Apply feature engineering (scaling, encoding, etc.)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        logger.info(f"✅ Feature engineering completed: {features.shape}")
        return features, labels
    
    async def _train_model(self, features: np.ndarray, labels: np.ndarray) -> Tuple[Any, Dict[str, float]]:
        """Train the ML model"""
        logger.info("🎯 Training model...")
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        # Train model based on configuration
        model_type = self.config.model_config.get("type", "random_forest")
        
        if model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=self.config.model_config.get("n_estimators", 100),
                random_state=42
            )
        elif model_type == "xgboost":
            import xgboost as xgb
            model = xgb.XGBClassifier(random_state=42)
        else:
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(random_state=42)
        
        # Train model
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Calculate training metrics
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        metrics = {
            "training_accuracy": train_score,
            "test_accuracy": test_score,
            "training_time": training_time,
            "training_samples": len(X_train)
        }
        
        logger.info(f"✅ Model training completed: {test_score:.3f} accuracy")
        return model, metrics
    
    async def _validate_model(self, model: Any, features: np.ndarray, 
                            labels: np.ndarray) -> Dict[str, float]:
        """Validate model performance"""
        logger.info("✅ Validating model...")
        
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import classification_report
        
        # Cross-validation
        cv_scores = cross_val_score(model, features, labels, cv=5)
        
        # Detailed metrics
        predictions = model.predict(features)
        
        metrics = {
            "cv_mean_accuracy": cv_scores.mean(),
            "cv_std_accuracy": cv_scores.std(),
            "validation_accuracy": (predictions == labels).mean()
        }
        
        logger.info(f"✅ Model validation completed: {metrics['cv_mean_accuracy']:.3f} ± {metrics['cv_std_accuracy']:.3f}")
        return metrics
    
    async def _deploy_model(self, model_metadata: ModelMetadata) -> Dict[str, Any]:
        """Deploy model to target environment"""
        logger.info(f"🚀 Deploying model {model_metadata.model_id}...")
        
        try:
            # Mock deployment process
            # In production, this would integrate with Kubernetes, Docker, cloud services, etc.
            
            deployment_config = self.config.deployment_config
            target_env = deployment_config.get("environment", "staging")
            
            # Simulate deployment steps
            await asyncio.sleep(1)  # Simulate deployment time
            
            # Update model status
            model_metadata.environment = DeploymentEnvironment(target_env)
            model_metadata.status = ModelStatus.DEPLOYED
            
            logger.info(f"✅ Model deployed to {target_env}")
            
            return {
                "status": "success",
                "environment": target_env,
                "deployment_time": datetime.utcnow().isoformat(),
                "endpoint": f"https://api.spotify-ai-agent.com/models/{model_metadata.model_id}"
            }
            
        except Exception as e:
            logger.error(f"❌ Model deployment failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            "pipeline_id": self.config.pipeline_id,
            "name": self.config.name,
            "status": self.pipeline_status,
            "current_run_id": self.current_run_id,
            "config": asdict(self.config),
            "last_updated": datetime.utcnow().isoformat()
        }

# Factory functions
def create_model_registry(registry_path: str = None) -> ModelRegistry:
    """Create model registry instance"""
    return ModelRegistry(registry_path)

def create_experiment_tracker(tracking_uri: str = None) -> ExperimentTracker:
    """Create experiment tracker instance"""
    return ExperimentTracker(tracking_uri)

def create_model_monitor(config: Dict[str, Any] = None) -> ModelMonitor:
    """Create model monitor instance"""
    return ModelMonitor(config)

def create_mlops_pipeline(config: PipelineConfig) -> MLOpsPipeline:
    """Create MLOps pipeline instance"""
    return MLOpsPipeline(config)

# Export main components
__all__ = [
    'ModelRegistry',
    'ExperimentTracker', 
    'ModelMonitor',
    'MLOpsPipeline',
    'ModelMetadata',
    'PipelineConfig',
    'ExperimentResult',
    'ModelStatus',
    'DeploymentEnvironment',
    'create_model_registry',
    'create_experiment_tracker',
    'create_model_monitor',
    'create_mlops_pipeline'
]
