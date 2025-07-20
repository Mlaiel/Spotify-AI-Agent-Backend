"""
Configuration principale pour le système d'analytics d'alertes
Configuration complète avec validation et optimisation automatique
"""

import os
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseSettings, Field, validator, root_validator
from datetime import timedelta

class LogLevel(str, Enum):
    """Niveaux de log"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class EnvironmentType(str, Enum):
    """Types d'environnement"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"

class ScalingStrategy(str, Enum):
    """Stratégies de scaling"""
    MANUAL = "manual"
    AUTO_CPU = "auto_cpu"
    AUTO_MEMORY = "auto_memory"
    AUTO_THROUGHPUT = "auto_throughput"
    PREDICTIVE = "predictive"

class MLModelType(str, Enum):
    """Types de modèles ML"""
    ISOLATION_FOREST = "isolation_forest"
    LSTM_AUTOENCODER = "lstm_autoencoder"
    ENSEMBLE = "ensemble"
    STATISTICAL = "statistical"
    TRANSFORMER = "transformer"

class MLConfig(BaseSettings):
    """Configuration des modèles ML"""
    
    # Modèles activés
    enabled_models: List[MLModelType] = Field(
        default=[MLModelType.ISOLATION_FOREST, MLModelType.STATISTICAL],
        description="Modèles ML activés"
    )
    
    # Détection d'anomalies
    anomaly_detection_enabled: bool = Field(True, description="Détection d'anomalies activée")
    anomaly_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Seuil d'anomalie")
    anomaly_window_size: int = Field(100, ge=10, le=10000, description="Taille fenêtre anomalie")
    
    # Corrélation
    correlation_enabled: bool = Field(True, description="Analyse de corrélation activée")
    correlation_window_minutes: int = Field(30, ge=1, le=1440, description="Fenêtre corrélation (min)")
    min_correlation_confidence: float = Field(0.7, ge=0.0, le=1.0, description="Confiance min corrélation")
    
    # Prédiction
    prediction_enabled: bool = Field(True, description="Prédiction activée")
    prediction_horizon_hours: int = Field(4, ge=1, le=48, description="Horizon prédiction (h)")
    model_retrain_hours: int = Field(24, ge=1, le=168, description="Réentraînement modèle (h)")
    
    # Performance
    batch_size: int = Field(100, ge=1, le=10000, description="Taille de batch ML")
    max_concurrent_predictions: int = Field(50, ge=1, le=1000, description="Prédictions simultanées max")
    model_cache_size: int = Field(1000, ge=10, le=100000, description="Taille cache modèles")
    
    # Isolation Forest spécifique
    isolation_forest_contamination: float = Field(0.1, ge=0.01, le=0.5, description="Taux contamination IF")
    isolation_forest_n_estimators: int = Field(100, ge=10, le=1000, description="Nb estimateurs IF")
    
    # LSTM spécifique
    lstm_sequence_length: int = Field(50, ge=10, le=500, description="Longueur séquence LSTM")
    lstm_hidden_units: int = Field(64, ge=16, le=512, description="Unités cachées LSTM")
    lstm_epochs: int = Field(50, ge=10, le=200, description="Époques LSTM")
    
    class Config:
        env_prefix = "ML_"
        case_sensitive = False

class StreamingConfig(BaseSettings):
    """Configuration du streaming"""
    
    # Activation
    streaming_enabled: bool = Field(True, description="Streaming activé")
    processing_mode: str = Field("real_time", description="Mode de traitement")
    
    # Fenêtres temporelles
    window_size_seconds: int = Field(60, ge=1, le=3600, description="Taille fenêtre (s)")
    slide_interval_seconds: int = Field(10, ge=1, le=300, description="Intervalle slide (s)")
    
    # Performance
    buffer_size: int = Field(10000, ge=100, le=1000000, description="Taille buffer")
    batch_size: int = Field(100, ge=1, le=10000, description="Taille batch")
    max_parallel_processing: int = Field(10, ge=1, le=100, description="Traitement parallèle max")
    
    # Back-pressure
    enable_backpressure: bool = Field(True, description="Back-pressure activé")
    backpressure_threshold: float = Field(0.8, ge=0.1, le=1.0, description="Seuil back-pressure")
    
    # Kafka
    kafka_enabled: bool = Field(False, description="Kafka activé")
    kafka_bootstrap_servers: List[str] = Field(
        default=["localhost:9092"], 
        description="Serveurs Kafka"
    )
    kafka_input_topic: str = Field("alert-events", description="Topic input Kafka")
    kafka_output_topic: str = Field("alert-analytics", description="Topic output Kafka")
    kafka_consumer_group: str = Field("alert-analytics-group", description="Groupe consommateur")
    
    # Métriques
    metrics_enabled: bool = Field(True, description="Métriques activées")
    metrics_interval_seconds: int = Field(30, ge=5, le=300, description="Intervalle métriques (s)")
    
    class Config:
        env_prefix = "STREAMING_"
        case_sensitive = False

class AnalyticsConfig(BaseSettings):
    """Configuration principale du système d'analytics"""
    
    # Environnement
    environment: EnvironmentType = Field(
        default=EnvironmentType.DEVELOPMENT,
        description="Type d'environnement"
    )
    debug: bool = Field(False, description="Mode debug")
    log_level: LogLevel = Field(LogLevel.INFO, description="Niveau de log")
    
    # Composants
    ml_config: MLConfig = Field(default_factory=MLConfig, description="Configuration ML")
    streaming_config: StreamingConfig = Field(default_factory=StreamingConfig, description="Configuration streaming")
    
    # Base de données
    database_url: str = Field(
        default="postgresql://user:pass@localhost:5432/spotify_ai_agent",
        description="URL base de données"
    )
    database_pool_min_size: int = Field(5, ge=1, le=50, description="Taille min pool DB")
    database_pool_max_size: int = Field(20, ge=5, le=200, description="Taille max pool DB")
    
    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="URL Redis"
    )
    redis_max_connections: int = Field(20, ge=1, le=100, description="Connexions max Redis")
    cache_ttl_seconds: int = Field(3600, ge=60, le=86400, description="TTL cache (s)")
    
    # Performance
    scaling_strategy: ScalingStrategy = Field(
        default=ScalingStrategy.AUTO_THROUGHPUT,
        description="Stratégie de scaling"
    )
    max_concurrent_analyses: int = Field(100, ge=10, le=10000, description="Analyses simultanées max")
    analysis_timeout_seconds: int = Field(300, ge=30, le=3600, description="Timeout analyse (s)")
    
    # Seuils et limites
    critical_alert_threshold: float = Field(0.9, ge=0.5, le=1.0, description="Seuil alerte critique")
    high_alert_threshold: float = Field(0.7, ge=0.3, le=0.9, description="Seuil alerte haute")
    medium_alert_threshold: float = Field(0.5, ge=0.1, le=0.7, description="Seuil alerte moyenne")
    
    # Rétention
    alert_retention_days: int = Field(90, ge=7, le=365, description="Rétention alertes (jours)")
    analytics_retention_days: int = Field(30, ge=7, le=180, description="Rétention analytics (jours)")
    model_retention_days: int = Field(180, ge=30, le=365, description="Rétention modèles (jours)")
    
    # Sécurité
    enable_authentication: bool = Field(True, description="Authentification activée")
    api_key_required: bool = Field(True, description="Clé API requise")
    rate_limit_per_minute: int = Field(1000, ge=10, le=100000, description="Limite taux (req/min)")
    
    # Monitoring
    health_check_interval_seconds: int = Field(60, ge=10, le=300, description="Intervalle health check (s)")
    enable_prometheus_metrics: bool = Field(True, description="Métriques Prometheus activées")
    prometheus_port: int = Field(8000, ge=1024, le=65535, description="Port Prometheus")
    
    # Alerting
    enable_self_monitoring: bool = Field(True, description="Auto-monitoring activé")
    self_alert_cooldown_minutes: int = Field(30, ge=5, le=1440, description="Cooldown auto-alerte (min)")
    
    # Services critiques
    critical_services: List[str] = Field(
        default=["user-api", "payment-service", "recommendation-engine"],
        description="Services critiques"
    )
    
    # Fenêtres d'analyse
    short_term_window_hours: int = Field(1, ge=1, le=24, description="Fenêtre court terme (h)")
    medium_term_window_hours: int = Field(24, ge=4, le=168, description="Fenêtre moyen terme (h)")
    long_term_window_hours: int = Field(168, ge=24, le=720, description="Fenêtre long terme (h)")
    
    @validator('database_url')
    def validate_database_url(cls, v):
        """Validation URL database"""
        if not v.startswith(('postgresql://', 'postgres://')):
            raise ValueError('URL database doit être PostgreSQL')
        return v
    
    @validator('redis_url')
    def validate_redis_url(cls, v):
        """Validation URL Redis"""
        if not v.startswith('redis://'):
            raise ValueError('URL Redis doit commencer par redis://')
        return v
    
    @root_validator
    def validate_thresholds(cls, values):
        """Validation cohérence des seuils"""
        critical = values.get('critical_alert_threshold', 0.9)
        high = values.get('high_alert_threshold', 0.7)
        medium = values.get('medium_alert_threshold', 0.5)
        
        if not (critical > high > medium):
            raise ValueError('Les seuils doivent être dans l\'ordre croissant')
        
        return values
    
    @root_validator
    def validate_production_settings(cls, values):
        """Validation settings production"""
        environment = values.get('environment')
        debug = values.get('debug')
        
        if environment == EnvironmentType.PRODUCTION and debug:
            raise ValueError('Debug mode interdit en production')
        
        return values
    
    def get_window_timedelta(self, window_type: str) -> timedelta:
        """Récupération d'une fenêtre temporelle"""
        if window_type == 'short':
            return timedelta(hours=self.short_term_window_hours)
        elif window_type == 'medium':
            return timedelta(hours=self.medium_term_window_hours)
        elif window_type == 'long':
            return timedelta(hours=self.long_term_window_hours)
        else:
            raise ValueError(f"Type de fenêtre invalide: {window_type}")
    
    def is_service_critical(self, service_name: str) -> bool:
        """Vérification si un service est critique"""
        return service_name.lower() in [s.lower() for s in self.critical_services]
    
    def get_severity_threshold(self, severity: str) -> float:
        """Récupération du seuil pour une gravité"""
        severity_lower = severity.lower()
        if severity_lower == 'critical':
            return self.critical_alert_threshold
        elif severity_lower == 'high':
            return self.high_alert_threshold
        elif severity_lower == 'medium':
            return self.medium_alert_threshold
        else:
            return 0.1  # Seuil par défaut pour low/info
    
    def should_auto_scale(self) -> bool:
        """Vérification si l'auto-scaling est activé"""
        return self.scaling_strategy != ScalingStrategy.MANUAL
    
    def get_cache_key_prefix(self) -> str:
        """Préfixe pour les clés de cache"""
        return f"spotify_ai_agent:analytics:{self.environment.value}"
    
    def get_database_config(self) -> Dict[str, Any]:
        """Configuration database pour asyncpg"""
        return {
            'dsn': self.database_url,
            'min_size': self.database_pool_min_size,
            'max_size': self.database_pool_max_size,
            'command_timeout': 30,
            'server_settings': {
                'application_name': 'spotify_ai_agent_analytics',
                'jit': 'off'
            }
        }
    
    def get_redis_config(self) -> Dict[str, Any]:
        """Configuration Redis pour aioredis"""
        return {
            'url': self.redis_url,
            'max_connections': self.redis_max_connections,
            'retry_on_timeout': True,
            'health_check_interval': 30,
            'socket_keepalive': True,
            'socket_keepalive_options': {
                'TCP_KEEPIDLE': 1,
                'TCP_KEEPINTVL': 3,
                'TCP_KEEPCNT': 5
            }
        }
    
    def export_for_ml_models(self) -> Dict[str, Any]:
        """Export configuration pour modèles ML"""
        return {
            'batch_size': self.ml_config.batch_size,
            'anomaly_threshold': self.ml_config.anomaly_threshold,
            'prediction_horizon': self.ml_config.prediction_horizon_hours,
            'retrain_interval': self.ml_config.model_retrain_hours,
            'isolation_forest': {
                'contamination': self.ml_config.isolation_forest_contamination,
                'n_estimators': self.ml_config.isolation_forest_n_estimators
            },
            'lstm': {
                'sequence_length': self.ml_config.lstm_sequence_length,
                'hidden_units': self.ml_config.lstm_hidden_units,
                'epochs': self.ml_config.lstm_epochs
            }
        }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        schema_extra = {
            "example": {
                "environment": "production",
                "debug": False,
                "log_level": "INFO",
                "database_url": "postgresql://user:pass@db:5432/spotify_ai_agent",
                "redis_url": "redis://redis:6379/0",
                "ml_config": {
                    "anomaly_detection_enabled": True,
                    "correlation_enabled": True,
                    "prediction_enabled": True
                },
                "streaming_config": {
                    "streaming_enabled": True,
                    "processing_mode": "real_time",
                    "kafka_enabled": True
                }
            }
        }

# Configuration globale singleton
_config_instance: Optional[AnalyticsConfig] = None

def get_analytics_config() -> AnalyticsConfig:
    """Récupération de la configuration globale"""
    global _config_instance
    if _config_instance is None:
        _config_instance = AnalyticsConfig()
    return _config_instance

def reload_config() -> AnalyticsConfig:
    """Rechargement de la configuration"""
    global _config_instance
    _config_instance = AnalyticsConfig()
    return _config_instance

# Configuration par environnement
def get_development_config() -> AnalyticsConfig:
    """Configuration développement"""
    return AnalyticsConfig(
        environment=EnvironmentType.DEVELOPMENT,
        debug=True,
        log_level=LogLevel.DEBUG,
        ml_config=MLConfig(
            batch_size=10,
            max_concurrent_predictions=5
        ),
        streaming_config=StreamingConfig(
            buffer_size=1000,
            batch_size=10
        )
    )

def get_production_config() -> AnalyticsConfig:
    """Configuration production"""
    return AnalyticsConfig(
        environment=EnvironmentType.PRODUCTION,
        debug=False,
        log_level=LogLevel.INFO,
        ml_config=MLConfig(
            batch_size=1000,
            max_concurrent_predictions=100
        ),
        streaming_config=StreamingConfig(
            buffer_size=100000,
            batch_size=1000,
            kafka_enabled=True
        ),
        scaling_strategy=ScalingStrategy.PREDICTIVE
    )
