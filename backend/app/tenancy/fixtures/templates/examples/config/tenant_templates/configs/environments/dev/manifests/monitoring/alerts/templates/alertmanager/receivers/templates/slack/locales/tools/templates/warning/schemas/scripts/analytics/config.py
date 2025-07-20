"""
Configuration Module - Module de Configuration Analytics
======================================================

Ce module gère toute la configuration du système d'analytics,
incluant les paramètres de bases de données, cache, ML, sécurité
et monitoring.

Classes:
- AnalyticsConfig: Configuration principale
- DatabaseConfig: Configuration bases de données
- CacheConfig: Configuration cache
- MLConfig: Configuration Machine Learning
- SecurityConfig: Configuration sécurité
- MonitoringConfig: Configuration monitoring
"""

import os
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

import yaml
from pydantic import BaseSettings, validator, Field


class Environment(str, Enum):
    """Environnements d'exécution."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Niveaux de log."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class DatabaseConfig:
    """Configuration des bases de données."""
    
    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "analytics"
    postgres_user: str = "analytics_user"
    postgres_password: str = "secure_password"
    postgres_ssl_mode: str = "require"
    postgres_pool_size: int = 20
    postgres_max_overflow: int = 30
    postgres_pool_timeout: int = 30
    
    # InfluxDB (Time Series)
    influx_host: str = "localhost"
    influx_port: int = 8086
    influx_org: str = "spotify-ai"
    influx_bucket: str = "analytics"
    influx_token: str = "secure_influx_token"
    influx_ssl: bool = True
    influx_timeout: int = 30
    
    # Elasticsearch
    elastic_hosts: List[str] = field(default_factory=lambda: ["localhost:9200"])
    elastic_username: str = "elastic"
    elastic_password: str = "secure_elastic_password"
    elastic_index_prefix: str = "analytics"
    elastic_ssl: bool = True
    elastic_verify_certs: bool = True
    elastic_timeout: int = 30
    
    # MongoDB (Document Store)
    mongo_host: str = "localhost"
    mongo_port: int = 27017
    mongo_db: str = "analytics"
    mongo_username: str = "analytics_user"
    mongo_password: str = "secure_mongo_password"
    mongo_auth_source: str = "admin"
    mongo_ssl: bool = True
    mongo_replica_set: Optional[str] = None
    
    @property
    def postgres_url(self) -> str:
        """URL de connexion PostgreSQL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            f"?sslmode={self.postgres_ssl_mode}"
        )
    
    @property
    def mongo_url(self) -> str:
        """URL de connexion MongoDB."""
        url = f"mongodb://{self.mongo_username}:{self.mongo_password}@{self.mongo_host}:{self.mongo_port}/{self.mongo_db}"
        if self.mongo_ssl:
            url += "?ssl=true"
        if self.mongo_replica_set:
            url += f"&replicaSet={self.mongo_replica_set}"
        return url


@dataclass
class CacheConfig:
    """Configuration du cache."""
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = "secure_redis_password"
    redis_ssl: bool = True
    redis_ssl_cert_reqs: str = "required"
    redis_socket_timeout: int = 5
    redis_socket_connect_timeout: int = 5
    redis_health_check_interval: int = 30
    
    # Pool settings
    redis_max_connections: int = 100
    redis_retry_on_timeout: bool = True
    redis_decode_responses: bool = True
    
    # Cluster settings
    redis_cluster_enabled: bool = False
    redis_cluster_nodes: List[str] = field(default_factory=list)
    
    # TTL settings
    default_ttl: int = 3600  # 1 heure
    metrics_ttl: int = 86400  # 24 heures
    alerts_ttl: int = 1800  # 30 minutes
    sessions_ttl: int = 7200  # 2 heures
    
    @property
    def redis_url(self) -> str:
        """URL de connexion Redis."""
        protocol = "rediss" if self.redis_ssl else "redis"
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"{protocol}://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"


@dataclass
class MLConfig:
    """Configuration Machine Learning."""
    
    # Models paths
    models_base_path: str = "/models"
    models_cache_path: str = "/cache/models"
    models_artifacts_path: str = "/artifacts"
    
    # TensorFlow/Keras
    tf_enable_gpu: bool = True
    tf_memory_growth: bool = True
    tf_log_level: str = "ERROR"
    tf_inter_op_parallelism: int = 0
    tf_intra_op_parallelism: int = 0
    
    # PyTorch
    torch_device: str = "auto"  # auto, cpu, cuda
    torch_num_threads: int = 4
    torch_deterministic: bool = False
    
    # Scikit-learn
    sklearn_n_jobs: int = -1
    sklearn_random_state: int = 42
    
    # Model serving
    model_server_host: str = "localhost"
    model_server_port: int = 8501
    model_server_timeout: int = 30
    batch_prediction_size: int = 32
    
    # Training
    default_epochs: int = 100
    default_batch_size: int = 32
    default_learning_rate: float = 0.001
    early_stopping_patience: int = 10
    model_checkpoint_frequency: int = 5
    
    # Anomaly Detection
    anomaly_threshold: float = 0.8
    anomaly_window_size: int = 100
    anomaly_min_samples: int = 50
    
    # Prediction
    prediction_confidence_threshold: float = 0.7
    prediction_cache_ttl: int = 300  # 5 minutes
    
    @property
    def torch_device_resolved(self) -> str:
        """Résout le device PyTorch."""
        if self.torch_device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.torch_device


@dataclass
class SecurityConfig:
    """Configuration sécurité."""
    
    # JWT
    jwt_secret_key: str = "ultra_secure_jwt_secret_key_change_in_production"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7
    
    # API Keys
    api_key_length: int = 32
    api_key_ttl_days: int = 90
    api_key_max_requests_per_hour: int = 1000
    
    # Encryption
    encryption_key: str = "32_byte_encryption_key_change_me"
    encryption_algorithm: str = "AES"
    encryption_mode: str = "GCM"
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 100
    rate_limit_burst: int = 10
    rate_limit_redis_key_prefix: str = "rate_limit:"
    
    # CORS
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    cors_headers: List[str] = field(default_factory=lambda: ["*"])
    cors_credentials: bool = True
    
    # Security Headers
    security_headers_enabled: bool = True
    hsts_max_age: int = 31536000  # 1 an
    content_security_policy: str = "default-src 'self'"
    
    # Audit
    audit_enabled: bool = True
    audit_retention_days: int = 90
    audit_sensitive_fields: List[str] = field(default_factory=lambda: ["password", "token", "key"])


@dataclass
class MonitoringConfig:
    """Configuration monitoring."""
    
    # Prometheus
    prometheus_enabled: bool = True
    prometheus_host: str = "localhost"
    prometheus_port: int = 9090
    prometheus_metrics_port: int = 8000
    prometheus_push_gateway: Optional[str] = None
    
    # Grafana
    grafana_enabled: bool = True
    grafana_host: str = "localhost"
    grafana_port: int = 3000
    grafana_api_key: Optional[str] = None
    
    # Jaeger (Tracing)
    jaeger_enabled: bool = True
    jaeger_host: str = "localhost"
    jaeger_port: int = 14268
    jaeger_service_name: str = "analytics-service"
    jaeger_sampler_type: str = "const"
    jaeger_sampler_param: float = 1.0
    
    # Health Checks
    health_check_enabled: bool = True
    health_check_interval: int = 30
    health_check_timeout: int = 10
    health_check_retries: int = 3
    
    # Alerting
    alerting_enabled: bool = True
    alerting_channels: List[str] = field(default_factory=lambda: ["slack", "email"])
    slack_webhook_url: Optional[str] = None
    email_smtp_host: str = "localhost"
    email_smtp_port: int = 587
    email_username: Optional[str] = None
    email_password: Optional[str] = None
    email_from: str = "analytics@spotify-ai.com"
    
    # Logging
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "json"
    log_file: Optional[str] = None
    log_max_size: str = "100MB"
    log_backup_count: int = 5
    log_compression: bool = True


class AnalyticsConfig(BaseSettings):
    """Configuration principale du système d'analytics."""
    
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    testing: bool = False
    
    # Application
    app_name: str = "Spotify AI Analytics"
    app_version: str = "2.0.0"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_workers: int = 4
    app_reload: bool = False
    
    # Components configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Features flags
    enable_realtime_analytics: bool = True
    enable_ml_predictions: bool = True
    enable_anomaly_detection: bool = True
    enable_auto_scaling: bool = True
    enable_data_retention: bool = True
    
    # Performance
    max_concurrent_requests: int = 1000
    request_timeout: int = 30
    worker_timeout: int = 60
    graceful_shutdown_timeout: int = 30
    
    # Data Retention
    metrics_retention_days: int = 30
    events_retention_days: int = 90
    alerts_retention_days: int = 365
    logs_retention_days: int = 7
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        case_sensitive = False
    
    @validator("environment", pre=True)
    def validate_environment(cls, v):
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    @validator("debug")
    def validate_debug(cls, v, values):
        if values.get("environment") == Environment.PRODUCTION and v:
            raise ValueError("Debug mode cannot be enabled in production")
        return v
    
    @property
    def is_production(self) -> bool:
        """Vérifie si on est en production."""
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        """Vérifie si on est en développement."""
        return self.environment == Environment.DEVELOPMENT
    
    @property
    def redis_url(self) -> str:
        """URL Redis depuis cache config."""
        return self.cache.redis_url
    
    @property
    def postgres_url(self) -> str:
        """URL PostgreSQL depuis database config."""
        return self.database.postgres_url
    
    def load_from_file(self, config_path: Union[str, Path]):
        """Charge la configuration depuis un fichier."""
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Fichier de configuration non trouvé: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix.lower() in ['.yml', '.yaml']:
                config_data = yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                config_data = json.load(f)
            else:
                raise ValueError(f"Format de fichier non supporté: {path.suffix}")
        
        # Mettre à jour la configuration
        for key, value in config_data.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save_to_file(self, config_path: Union[str, Path], format: str = "yaml"):
        """Sauvegarde la configuration dans un fichier."""
        path = Path(config_path)
        config_dict = self.dict()
        
        with open(path, 'w', encoding='utf-8') as f:
            if format.lower() in ['yml', 'yaml']:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            elif format.lower() == 'json':
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Format non supporté: {format}")
    
    def get_database_config(self, database_type: str) -> Dict[str, Any]:
        """Retourne la configuration pour un type de base de données."""
        configs = {
            "postgres": {
                "url": self.database.postgres_url,
                "pool_size": self.database.postgres_pool_size,
                "max_overflow": self.database.postgres_max_overflow,
                "timeout": self.database.postgres_pool_timeout
            },
            "influx": {
                "host": self.database.influx_host,
                "port": self.database.influx_port,
                "org": self.database.influx_org,
                "bucket": self.database.influx_bucket,
                "token": self.database.influx_token,
                "ssl": self.database.influx_ssl,
                "timeout": self.database.influx_timeout
            },
            "elasticsearch": {
                "hosts": self.database.elastic_hosts,
                "username": self.database.elastic_username,
                "password": self.database.elastic_password,
                "index_prefix": self.database.elastic_index_prefix,
                "ssl": self.database.elastic_ssl,
                "verify_certs": self.database.elastic_verify_certs,
                "timeout": self.database.elastic_timeout
            },
            "mongodb": {
                "url": self.database.mongo_url,
                "database": self.database.mongo_db,
                "ssl": self.database.mongo_ssl
            }
        }
        
        if database_type not in configs:
            raise ValueError(f"Type de base de données non supporté: {database_type}")
        
        return configs[database_type]
    
    def validate_config(self) -> List[str]:
        """Valide la configuration et retourne les erreurs."""
        errors = []
        
        # Validation production
        if self.is_production:
            if self.debug:
                errors.append("Debug mode activé en production")
            
            if self.security.jwt_secret_key == "ultra_secure_jwt_secret_key_change_in_production":
                errors.append("Clé JWT par défaut utilisée en production")
            
            if not self.cache.redis_ssl:
                errors.append("SSL Redis désactivé en production")
            
            if not self.database.postgres_ssl_mode or self.database.postgres_ssl_mode == "disable":
                errors.append("SSL PostgreSQL désactivé en production")
        
        # Validation ports
        if self.app_port == self.monitoring.prometheus_metrics_port:
            errors.append("Port application identique au port métriques Prometheus")
        
        # Validation ML
        if self.enable_ml_predictions and not self.ml.models_base_path:
            errors.append("Chemin des modèles ML non configuré")
        
        return errors


# Configuration globale par défaut
_default_config: Optional[AnalyticsConfig] = None


def get_config() -> AnalyticsConfig:
    """Retourne l'instance de configuration globale."""
    global _default_config
    
    if _default_config is None:
        _default_config = AnalyticsConfig()
    
    return _default_config


def set_config(config: AnalyticsConfig):
    """Définit la configuration globale."""
    global _default_config
    _default_config = config


def load_config_from_env() -> AnalyticsConfig:
    """Charge la configuration depuis les variables d'environnement."""
    return AnalyticsConfig()


def create_development_config() -> AnalyticsConfig:
    """Crée une configuration pour le développement."""
    config = AnalyticsConfig()
    config.environment = Environment.DEVELOPMENT
    config.debug = True
    config.app_reload = True
    config.monitoring.log_level = LogLevel.DEBUG
    config.cache.redis_ssl = False
    config.database.postgres_ssl_mode = "prefer"
    config.security.rate_limit_requests_per_minute = 1000
    return config


def create_production_config() -> AnalyticsConfig:
    """Crée une configuration pour la production."""
    config = AnalyticsConfig()
    config.environment = Environment.PRODUCTION
    config.debug = False
    config.app_reload = False
    config.monitoring.log_level = LogLevel.INFO
    config.security.rate_limit_requests_per_minute = 100
    
    # Sécurité renforcée
    config.cache.redis_ssl = True
    config.database.postgres_ssl_mode = "require"
    config.database.influx_ssl = True
    config.database.elastic_ssl = True
    config.database.mongo_ssl = True
    
    return config


# Configurations prédéfinies
CONFIGS = {
    "development": create_development_config,
    "production": create_production_config,
    "testing": lambda: AnalyticsConfig(environment=Environment.TESTING, testing=True)
}
