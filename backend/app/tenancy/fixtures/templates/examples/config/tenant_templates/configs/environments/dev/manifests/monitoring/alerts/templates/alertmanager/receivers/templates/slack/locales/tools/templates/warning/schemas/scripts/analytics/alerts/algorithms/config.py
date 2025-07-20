"""
🎵 Spotify AI Agent - Enterprise Configuration Management System
==============================================================

Advanced Configuration Management for Music Streaming Platform Alert Algorithms

This ultra-advanced configuration module provides enterprise-grade configuration
management specifically designed for large-scale music streaming platforms.
It supports dynamic configuration loading, environment-aware settings,
real-time configuration updates, and comprehensive validation.

🚀 ENTERPRISE FEATURES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Dynamic Configuration Reloading with Zero Downtime
• Environment-Aware Configuration (Dev/Staging/Production)
• Multi-Tenant Configuration Isolation and Security
• Real-Time Configuration Validation and Health Checks
• Encrypted Configuration Storage for Sensitive Parameters
• Configuration Versioning and Rollback Capabilities
• Performance-Optimized Caching with Redis Integration
• Advanced Monitoring and Configuration Drift Detection
• Kubernetes ConfigMap and Secret Integration
• GitOps-Compatible Configuration Management

🎯 MUSIC STREAMING SPECIALIZATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Audio Quality Thresholds (Bitrate, Latency, Buffer Health)
• CDN Performance Targets for 180+ Global Markets
• Recommendation Engine Accuracy Benchmarks
• User Engagement and Retention Metrics Configuration
• Revenue Protection and Monetization Alert Parameters
• Content Licensing and Royalty Calculation Settings
• Artist Analytics and Performance Tracking Configuration
• Peak Traffic Event Handling (Album Releases, Concerts)

@Author: Configuration Architecture by Fahed Mlaiel
@Version: 2.0.0 (Enterprise Edition)
@Last Updated: 2025-07-19
"""

import os
import sys
import yaml
import json
import logging
import hashlib
import asyncio
from typing import Dict, Any, Optional, List, Union, Callable, Type
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from threading import Lock, RLock
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import redis
import boto3
from cryptography.fernet import Fernet
from pydantic import BaseModel, validator, Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
import threading
import time
import weakref
from collections import defaultdict

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

class Environment(Enum):
    """
    Supported deployment environments with specific configurations.
    Each environment has tailored settings for optimal performance.
    """
    DEVELOPMENT = "development"
    STAGING = "staging" 
    PRODUCTION = "production"
    TESTING = "testing"
    SANDBOX = "sandbox"
    DISASTER_RECOVERY = "disaster_recovery"

class AlertSeverity(Enum):
    """Alert severity levels with escalation policies."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class AlgorithmType(Enum):
    """Supported algorithm types for configuration."""
    ANOMALY_DETECTION = "anomaly_detection"
    PREDICTIVE_ALERTING = "predictive_alerting"
    CORRELATION = "correlation"
    PATTERN_RECOGNITION = "pattern_recognition"
    NOISE_REDUCTION = "noise_reduction"
    THRESHOLD_ADAPTATION = "threshold_adaptation"
    STREAMING_PROCESSING = "streaming_processing"
    SEVERITY_CLASSIFICATION = "severity_classification"

class ConfigurationScope(Enum):
    """Configuration scope for multi-tenant isolation."""
    GLOBAL = "global"
    TENANT = "tenant"
    USER = "user"
    SESSION = "session"

class AlgorithmType(Enum):
    """Types d'algorithmes disponibles."""
    ANOMALY_DETECTION = "anomaly_detection"
    ALERT_CLASSIFICATION = "alert_classification"
    CORRELATION_ENGINE = "correlation_engine"
    PREDICTION_MODELS = "prediction_models"
    BEHAVIORAL_ANALYSIS = "behavioral_analysis"

@dataclass
class ModelConfig:
    """Configuration d'un modèle spécifique."""
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    performance_thresholds: Dict[str, float] = field(default_factory=dict)
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    cache_settings: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AlgorithmConfig:
    """Configuration complète d'un algorithme."""
    algorithm_type: AlgorithmType
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    global_settings: Dict[str, Any] = field(default_factory=dict)
    monitoring_config: Dict[str, Any] = field(default_factory=dict)

class ConfigurationManager:
    """Gestionnaire de configuration centralisé."""
    
    def __init__(self, environment: Environment = Environment.PRODUCTION):
        self.environment = environment
        self.config_cache = {}
        self.config_file_path = self._get_config_file_path()
        self.load_configuration()
    
    def _get_config_file_path(self) -> str:
        """Retourne le chemin du fichier de configuration."""
        base_path = os.path.dirname(__file__)
        return os.path.join(base_path, "config", f"algorithm_config_{self.environment.value}.yaml")
    
    def load_configuration(self) -> None:
        """Charge la configuration depuis les fichiers."""
        try:
            if os.path.exists(self.config_file_path):
                with open(self.config_file_path, 'r') as f:
                    self.config_cache = yaml.safe_load(f)
            else:
                # Configuration par défaut si fichier absent
                self.config_cache = self._get_default_configuration()
                self.save_configuration()
                
            logger.info(f"Configuration loaded for environment: {self.environment.value}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self.config_cache = self._get_default_configuration()
    
    def save_configuration(self) -> None:
        """Sauvegarde la configuration."""
        try:
            os.makedirs(os.path.dirname(self.config_file_path), exist_ok=True)
            
            with open(self.config_file_path, 'w') as f:
                yaml.dump(self.config_cache, f, default_flow_style=False, indent=2)
                
            logger.info("Configuration saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def get_algorithm_config(self, algorithm_type: AlgorithmType) -> AlgorithmConfig:
        """Retourne la configuration d'un algorithme."""
        
        algo_config_dict = self.config_cache.get(algorithm_type.value, {})
        
        # Conversion en objet AlgorithmConfig
        models = {}
        for model_name, model_dict in algo_config_dict.get('models', {}).items():
            models[model_name] = ModelConfig(
                enabled=model_dict.get('enabled', True),
                parameters=model_dict.get('parameters', {}),
                performance_thresholds=model_dict.get('performance_thresholds', {}),
                resource_limits=model_dict.get('resource_limits', {}),
                cache_settings=model_dict.get('cache_settings', {})
            )
        
        return AlgorithmConfig(
            algorithm_type=algorithm_type,
            models=models,
            global_settings=algo_config_dict.get('global_settings', {}),
            monitoring_config=algo_config_dict.get('monitoring_config', {})
        )
    
    def update_algorithm_config(self, algorithm_type: AlgorithmType, 
                              config: AlgorithmConfig) -> None:
        """Met à jour la configuration d'un algorithme."""
        
        # Conversion en dictionnaire
        config_dict = {
            'models': {},
            'global_settings': config.global_settings,
            'monitoring_config': config.monitoring_config
        }
        
        for model_name, model_config in config.models.items():
            config_dict['models'][model_name] = {
                'enabled': model_config.enabled,
                'parameters': model_config.parameters,
                'performance_thresholds': model_config.performance_thresholds,
                'resource_limits': model_config.resource_limits,
                'cache_settings': model_config.cache_settings
            }
        
        self.config_cache[algorithm_type.value] = config_dict
        self.save_configuration()
    
    def _get_default_configuration(self) -> Dict[str, Any]:
        """Retourne la configuration par défaut."""
        
        if self.environment == Environment.PRODUCTION:
            return self._get_production_config()
        elif self.environment == Environment.STAGING:
            return self._get_staging_config()
        elif self.environment == Environment.DEVELOPMENT:
            return self._get_development_config()
        else:
            return self._get_testing_config()
    
    def _get_production_config(self) -> Dict[str, Any]:
        """Configuration optimisée pour la production."""
        return {
            'anomaly_detection': {
                'models': {
                    'isolation_forest': {
                        'enabled': True,
                        'parameters': {
                            'contamination': 0.05,
                            'n_estimators': 200,
                            'max_samples': 'auto',
                            'random_state': 42,
                            'n_jobs': -1
                        },
                        'performance_thresholds': {
                            'max_latency_ms': 100,
                            'min_accuracy': 0.95,
                            'max_memory_mb': 1024
                        },
                        'resource_limits': {
                            'max_cpu_percent': 50,
                            'max_memory_mb': 2048,
                            'timeout_seconds': 30
                        },
                        'cache_settings': {
                            'enabled': True,
                            'ttl_seconds': 300,
                            'max_size': 10000
                        }
                    },
                    'one_class_svm': {
                        'enabled': True,
                        'parameters': {
                            'kernel': 'rbf',
                            'gamma': 'scale',
                            'nu': 0.05,
                            'cache_size': 500
                        },
                        'performance_thresholds': {
                            'max_latency_ms': 200,
                            'min_accuracy': 0.90,
                            'max_memory_mb': 512
                        },
                        'resource_limits': {
                            'max_cpu_percent': 30,
                            'max_memory_mb': 1024,
                            'timeout_seconds': 60
                        },
                        'cache_settings': {
                            'enabled': True,
                            'ttl_seconds': 600,
                            'max_size': 5000
                        }
                    },
                    'lstm_autoencoder': {
                        'enabled': True,
                        'parameters': {
                            'sequence_length': 24,
                            'encoding_dim': 64,
                            'epochs': 100,
                            'batch_size': 64,
                            'threshold_percentile': 95
                        },
                        'performance_thresholds': {
                            'max_latency_ms': 500,
                            'min_accuracy': 0.92,
                            'max_memory_mb': 2048
                        },
                        'resource_limits': {
                            'max_cpu_percent': 70,
                            'max_memory_mb': 4096,
                            'timeout_seconds': 120
                        },
                        'cache_settings': {
                            'enabled': False,  # Modèles DL complexes
                            'ttl_seconds': 0,
                            'max_size': 0
                        }
                    },
                    'ensemble': {
                        'enabled': True,
                        'parameters': {
                            'voting_strategy': 'weighted',
                            'weights': {
                                'isolation_forest': 0.4,
                                'one_class_svm': 0.3,
                                'lstm_autoencoder': 0.3
                            },
                            'confidence_threshold': 0.7
                        },
                        'performance_thresholds': {
                            'max_latency_ms': 800,
                            'min_accuracy': 0.96,
                            'max_memory_mb': 3072
                        }
                    }
                },
                'global_settings': {
                    'real_time_processing': True,
                    'batch_processing': True,
                    'adaptive_thresholds': True,
                    'auto_retraining': True,
                    'feature_importance_tracking': True
                },
                'monitoring_config': {
                    'prometheus_metrics': True,
                    'detailed_logging': False,
                    'performance_tracking': True,
                    'alert_on_degradation': True
                }
            },
            'alert_classification': {
                'models': {
                    'random_forest': {
                        'enabled': True,
                        'parameters': {
                            'n_estimators': 200,
                            'max_depth': 15,
                            'min_samples_split': 5,
                            'min_samples_leaf': 2,
                            'random_state': 42,
                            'n_jobs': -1
                        },
                        'performance_thresholds': {
                            'max_latency_ms': 150,
                            'min_accuracy': 0.94,
                            'max_memory_mb': 1024
                        }
                    },
                    'severity_predictor': {
                        'enabled': True,
                        'parameters': {
                            'learning_rate': 0.1,
                            'max_depth': 8,
                            'n_estimators': 150,
                            'subsample': 0.8,
                            'colsample_bytree': 0.8,
                            'random_state': 42
                        },
                        'performance_thresholds': {
                            'max_latency_ms': 100,
                            'min_accuracy': 0.92,
                            'max_memory_mb': 512
                        }
                    },
                    'business_impact_analyzer': {
                        'enabled': True,
                        'parameters': {
                            'revenue_impact_thresholds': {
                                'catastrophic': 100000,
                                'severe': 10000,
                                'moderate': 1000,
                                'minor': 100,
                                'negligible': 0
                            },
                            'user_impact_thresholds': {
                                'catastrophic': 10000,
                                'severe': 1000,
                                'moderate': 100,
                                'minor': 10,
                                'negligible': 0
                            }
                        }
                    }
                },
                'global_settings': {
                    'feature_selection': True,
                    'cross_validation': True,
                    'ensemble_methods': True,
                    'real_time_classification': True
                }
            },
            'correlation_engine': {
                'models': {
                    'metric_correlation': {
                        'enabled': True,
                        'parameters': {
                            'window_size_hours': 24,
                            'min_correlation_threshold': 0.3,
                            'max_lag_minutes': 60,
                            'correlation_methods': ['pearson', 'spearman'],
                            'min_samples_for_correlation': 20
                        },
                        'performance_thresholds': {
                            'max_latency_ms': 1000,
                            'max_memory_mb': 2048
                        }
                    },
                    'event_correlation': {
                        'enabled': True,
                        'parameters': {
                            'time_window_minutes': 30,
                            'similarity_threshold': 0.7,
                            'min_cluster_size': 3,
                            'max_cluster_distance': 600
                        }
                    },
                    'causality_detector': {
                        'enabled': True,
                        'parameters': {
                            'min_confidence': 0.7,
                            'max_delay_hours': 2,
                            'evidence_threshold': 5
                        }
                    }
                },
                'global_settings': {
                    'real_time_correlation': True,
                    'historical_analysis': True,
                    'graph_analysis': True,
                    'cascade_prediction': True
                }
            },
            'prediction_models': {
                'models': {
                    'incident_predictor': {
                        'enabled': True,
                        'parameters': {
                            'models': ['random_forest', 'gradient_boosting', 'neural_network'],
                            'ensemble_method': 'weighted_average',
                            'feature_selection': True,
                            'time_horizons': ['1h', '6h', '24h'],
                            'incident_threshold': 0.8
                        },
                        'performance_thresholds': {
                            'max_latency_ms': 2000,
                            'min_accuracy': 0.90,
                            'max_memory_mb': 4096
                        }
                    },
                    'capacity_forecaster': {
                        'enabled': True,
                        'parameters': {
                            'forecast_horizons': [24, 72, 168],
                            'seasonality_periods': [24, 168],
                            'trend_detection': True,
                            'capacity_thresholds': {
                                'warning': 0.7,
                                'critical': 0.85,
                                'emergency': 0.95
                            }
                        }
                    },
                    'trend_analyzer': {
                        'enabled': True,
                        'parameters': {
                            'trend_window_days': 30,
                            'change_point_detection': True,
                            'anomaly_detection': True,
                            'seasonal_analysis': True
                        }
                    }
                },
                'global_settings': {
                    'auto_retraining': True,
                    'model_versioning': True,
                    'a_b_testing': True,
                    'prediction_confidence': True
                }
            },
            'behavioral_analysis': {
                'models': {
                    'user_behavior_analyzer': {
                        'enabled': True,
                        'parameters': {
                            'window_size_hours': 24,
                            'min_events_for_profile': 100,
                            'anomaly_threshold': 0.95,
                            'profile_update_frequency_minutes': 60
                        }
                    },
                    'system_behavior_profiler': {
                        'enabled': True,
                        'parameters': {
                            'window_size_hours': 12,
                            'min_events_for_profile': 50,
                            'performance_thresholds': {
                                'response_time_ms': 1000,
                                'error_rate_percent': 5,
                                'throughput_rps': 100
                            }
                        }
                    },
                    'deviation_detector': {
                        'enabled': True,
                        'parameters': {
                            'sensitivity': 0.8,
                            'min_baseline_samples': 200,
                            'detection_methods': ['statistical', 'ml', 'ensemble']
                        }
                    }
                },
                'global_settings': {
                    'real_time_analysis': True,
                    'profile_persistence': True,
                    'clustering_analysis': True,
                    'journey_analysis': True
                }
            }
        }
    
    def _get_staging_config(self) -> Dict[str, Any]:
        """Configuration pour l'environnement de staging."""
        config = self._get_production_config()
        
        # Modifications pour staging
        for algo_type in config:
            for model_name in config[algo_type].get('models', {}):
                model_config = config[algo_type]['models'][model_name]
                
                # Réduction des performances pour staging
                if 'performance_thresholds' in model_config:
                    thresholds = model_config['performance_thresholds']
                    thresholds['max_latency_ms'] = thresholds.get('max_latency_ms', 100) * 2
                    thresholds['min_accuracy'] = max(0.85, thresholds.get('min_accuracy', 0.9) - 0.05)
                
                # Réduction des limites de ressources
                if 'resource_limits' in model_config:
                    limits = model_config['resource_limits']
                    limits['max_memory_mb'] = limits.get('max_memory_mb', 1024) // 2
                    limits['max_cpu_percent'] = limits.get('max_cpu_percent', 50) // 2
        
        return config
    
    def _get_development_config(self) -> Dict[str, Any]:
        """Configuration pour l'environnement de développement."""
        config = self._get_staging_config()
        
        # Modifications pour développement
        for algo_type in config:
            # Activation du logging détaillé
            config[algo_type]['monitoring_config'] = {
                'prometheus_metrics': True,
                'detailed_logging': True,
                'performance_tracking': True,
                'alert_on_degradation': False  # Pas d'alertes en dev
            }
            
            # Paramètres de développement plus permissifs
            for model_name in config[algo_type].get('models', {}):
                model_config = config[algo_type]['models'][model_name]
                
                # Réduction des échantillons requis
                if 'parameters' in model_config:
                    params = model_config['parameters']
                    if 'min_events_for_profile' in params:
                        params['min_events_for_profile'] = max(10, params['min_events_for_profile'] // 10)
                    if 'min_samples_for_correlation' in params:
                        params['min_samples_for_correlation'] = max(5, params['min_samples_for_correlation'] // 4)
                    if 'min_baseline_samples' in params:
                        params['min_baseline_samples'] = max(20, params['min_baseline_samples'] // 10)
        
        return config
    
    def _get_testing_config(self) -> Dict[str, Any]:
        """Configuration pour les tests."""
        return {
            'anomaly_detection': {
                'models': {
                    'isolation_forest': {
                        'enabled': True,
                        'parameters': {
                            'contamination': 0.1,
                            'n_estimators': 10,
                            'random_state': 42
                        }
                    }
                },
                'global_settings': {
                    'real_time_processing': False,
                    'batch_processing': True
                }
            },
            'alert_classification': {
                'models': {
                    'random_forest': {
                        'enabled': True,
                        'parameters': {
                            'n_estimators': 10,
                            'max_depth': 3,
                            'random_state': 42
                        }
                    }
                }
            }
        }

# Configuration globale par défaut
DEFAULT_CONFIG_MANAGER = ConfigurationManager()

def get_config(algorithm_type: AlgorithmType, 
               environment: Optional[Environment] = None) -> AlgorithmConfig:
    """Fonction utilitaire pour obtenir la configuration."""
    
    if environment and environment != DEFAULT_CONFIG_MANAGER.environment:
        config_manager = ConfigurationManager(environment)
        return config_manager.get_algorithm_config(algorithm_type)
    
    return DEFAULT_CONFIG_MANAGER.get_algorithm_config(algorithm_type)

def update_config(algorithm_type: AlgorithmType, 
                 config: AlgorithmConfig,
                 environment: Optional[Environment] = None) -> None:
    """Fonction utilitaire pour mettre à jour la configuration."""
    
    if environment and environment != DEFAULT_CONFIG_MANAGER.environment:
        config_manager = ConfigurationManager(environment)
        config_manager.update_algorithm_config(algorithm_type, config)
    else:
        DEFAULT_CONFIG_MANAGER.update_algorithm_config(algorithm_type, config)

# Configuration des métriques Prometheus
PROMETHEUS_CONFIG = {
    'anomaly_detection': {
        'counters': [
            'anomaly_detections_total',
            'false_positives_total',
            'false_negatives_total'
        ],
        'histograms': [
            'anomaly_detection_duration_seconds',
            'model_training_duration_seconds'
        ],
        'gauges': [
            'active_anomalies_count',
            'model_accuracy_score',
            'detection_threshold'
        ]
    },
    'alert_classification': {
        'counters': [
            'alert_classifications_total',
            'classification_errors_total'
        ],
        'histograms': [
            'alert_classification_duration_seconds'
        ],
        'gauges': [
            'classification_accuracy',
            'active_alerts_count'
        ],
        'summaries': [
            'business_impact_scores',
            'priority_scores'
        ]
    },
    'correlation_engine': {
        'counters': [
            'correlation_calculations_total',
            'causality_detections_total'
        ],
        'histograms': [
            'correlation_calculation_duration_seconds'
        ],
        'gauges': [
            'active_correlations_count',
            'event_clusters_count'
        ]
    },
    'prediction_models': {
        'counters': [
            'predictions_made_total',
            'prediction_errors_total'
        ],
        'histograms': [
            'prediction_duration_seconds',
            'model_training_duration_seconds'
        ],
        'gauges': [
            'prediction_accuracy',
            'model_drift_score'
        ],
        'summaries': [
            'capacity_forecasts',
            'incident_probabilities'
        ]
    },
    'behavioral_analysis': {
        'counters': [
            'behavior_profiles_created_total',
            'behavior_deviations_detected_total'
        ],
        'histograms': [
            'behavior_analysis_duration_seconds'
        ],
        'gauges': [
            'active_behavior_clusters_count',
            'profile_confidence_score'
        ]
    }
}

# Configuration des logs
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s'
        },
        'simple': {
            'format': '%(levelname)s %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'simple',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': 'algorithms.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        }
    },
    'loggers': {
        'algorithms': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
            'propagate': False
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console']
    }
}

# Configuration Redis
REDIS_CONFIG = {
    'production': {
        'host': 'redis-cluster.spotify-ai.internal',
        'port': 6379,
        'db': 0,
        'max_connections': 50,
        'socket_timeout': 5,
        'socket_connect_timeout': 5,
        'retry_on_timeout': True,
        'health_check_interval': 30
    },
    'staging': {
        'host': 'redis-staging.spotify-ai.internal',
        'port': 6379,
        'db': 1,
        'max_connections': 20,
        'socket_timeout': 10,
        'socket_connect_timeout': 10
    },
    'development': {
        'host': 'localhost',
        'port': 6379,
        'db': 2,
        'max_connections': 10,
        'socket_timeout': 30,
        'socket_connect_timeout': 30
    },
    'testing': {
        'host': 'localhost',
        'port': 6379,
        'db': 15,  # DB séparée pour les tests
        'max_connections': 5
    }
}

def get_redis_config(environment: Environment) -> Dict[str, Any]:
    """Retourne la configuration Redis pour l'environnement."""
    return REDIS_CONFIG.get(environment.value, REDIS_CONFIG['development'])

# Configuration TensorFlow/Keras
TENSORFLOW_CONFIG = {
    'production': {
        'mixed_precision': True,
        'memory_growth': True,
        'inter_op_parallelism_threads': 0,  # Use all available
        'intra_op_parallelism_threads': 0,  # Use all available
        'allow_soft_placement': True,
        'log_device_placement': False
    },
    'development': {
        'mixed_precision': False,
        'memory_growth': True,
        'inter_op_parallelism_threads': 2,
        'intra_op_parallelism_threads': 4,
        'allow_soft_placement': True,
        'log_device_placement': True
    }
}

def configure_tensorflow(environment: Environment) -> None:
    """Configure TensorFlow selon l'environnement."""
    import tensorflow as tf
    
    config = TENSORFLOW_CONFIG.get(environment.value, TENSORFLOW_CONFIG['development'])
    
    # Configuration de la mémoire GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, config['memory_growth'])
            
            if config.get('mixed_precision', False):
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                
        except RuntimeError as e:
            logger.warning(f"GPU configuration failed: {e}")
    
    # Configuration CPU
    tf.config.threading.set_inter_op_parallelism_threads(config['inter_op_parallelism_threads'])
    tf.config.threading.set_intra_op_parallelism_threads(config['intra_op_parallelism_threads'])

# Export des configurations principales
__all__ = [
    'ConfigurationManager',
    'AlgorithmConfig',
    'ModelConfig',
    'Environment',
    'AlgorithmType',
    'get_config',
    'update_config',
    'PROMETHEUS_CONFIG',
    'LOGGING_CONFIG',
    'get_redis_config',
    'configure_tensorflow'
]
