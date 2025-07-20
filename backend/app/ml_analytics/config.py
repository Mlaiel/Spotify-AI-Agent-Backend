# 🎵 ML Analytics Configuration
# ============================
# 
# Configuration avancée et gestion d'environnement
# Settings enterprise avec validation et sécurité
#
# 🎖️ Expert: DBA & Data Engineer

"""
🔧 ML Analytics Configuration System
====================================

Enterprise configuration management providing:
- Environment-specific settings
- Validation and type safety
- Security and secrets management
- Dynamic configuration updates
- Performance optimization settings
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Type
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from urllib.parse import urlparse
import secrets
import hashlib
from datetime import timedelta

# Validation
from pydantic import BaseSettings, Field, validator, SecretStr
from marshmallow import Schema, fields, validate, ValidationError


class Environment(Enum):
    """Environnements de déploiement"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class DatabaseType(Enum):
    """Types de bases de données supportées"""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"
    REDIS = "redis"
    SQLITE = "sqlite"


class MLFramework(Enum):
    """Frameworks ML supportés"""
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    SCIKIT_LEARN = "scikit_learn"
    HUGGINGFACE = "huggingface"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"


@dataclass
class DatabaseConfig:
    """Configuration base de données"""
    type: DatabaseType
    host: str
    port: int
    database: str
    username: str
    password: str
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    ssl_mode: str = "prefer"
    connection_timeout: int = 30
    query_timeout: int = 300
    
    @property
    def connection_url(self) -> str:
        """URL de connexion"""
        if self.type == DatabaseType.POSTGRESQL:
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.type == DatabaseType.MYSQL:
            return f"mysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.type == DatabaseType.MONGODB:
            return f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.type == DatabaseType.REDIS:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.type == DatabaseType.SQLITE:
            return f"sqlite:///{self.database}"
        else:
            raise ValueError(f"Type de base de données non supporté: {self.type}")
    
    def validate(self) -> bool:
        """Validation de la configuration"""
        if not self.host or not self.database:
            return False
        if self.port <= 0 or self.port > 65535:
            return False
        if self.pool_size <= 0 or self.max_overflow < 0:
            return False
        return True


@dataclass
class RedisConfig:
    """Configuration Redis pour cache et queues"""
    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: Optional[str] = None
    ssl: bool = False
    connection_pool_size: int = 50
    socket_timeout: int = 30
    socket_connect_timeout: int = 30
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    
    @property
    def connection_url(self) -> str:
        """URL de connexion Redis"""
        scheme = "rediss" if self.ssl else "redis"
        auth = f":{self.password}@" if self.password else ""
        return f"{scheme}://{auth}{self.host}:{self.port}/{self.database}"


@dataclass
class MongoConfig:
    """Configuration MongoDB pour documents"""
    host: str = "localhost"
    port: int = 27017
    database: str = "ml_analytics"
    username: Optional[str] = None
    password: Optional[str] = None
    replica_set: Optional[str] = None
    auth_source: str = "admin"
    ssl: bool = False
    connection_timeout: int = 30000
    server_selection_timeout: int = 30000
    max_pool_size: int = 100
    
    @property
    def connection_url(self) -> str:
        """URL de connexion MongoDB"""
        auth = f"{self.username}:{self.password}@" if self.username and self.password else ""
        replica = f"?replicaSet={self.replica_set}" if self.replica_set else ""
        return f"mongodb://{auth}{self.host}:{self.port}/{self.database}{replica}"


@dataclass
class ModelConfig:
    """Configuration d'un modèle ML"""
    model_id: str
    model_type: str
    framework: MLFramework
    version: str = "1.0.0"
    model_path: Optional[str] = None
    config_path: Optional[str] = None
    batch_size: int = 32
    max_sequence_length: int = 512
    device: str = "auto"  # auto, cpu, cuda, mps
    precision: str = "float32"  # float16, float32, float64
    optimization_level: str = "O1"  # O0, O1, O2, O3
    cache_predictions: bool = True
    warmup_requests: int = 5
    
    # Paramètres spécifiques par framework
    tensorflow_config: Optional[Dict[str, Any]] = None
    pytorch_config: Optional[Dict[str, Any]] = None
    huggingface_config: Optional[Dict[str, Any]] = None
    
    def validate(self) -> bool:
        """Validation de la configuration du modèle"""
        if not self.model_id or not self.model_type:
            return False
        if self.batch_size <= 0 or self.max_sequence_length <= 0:
            return False
        if self.device not in ["auto", "cpu", "cuda", "mps"]:
            return False
        if self.precision not in ["float16", "float32", "float64"]:
            return False
        return True


@dataclass
class PipelineConfig:
    """Configuration d'un pipeline ML"""
    pipeline_id: str
    description: str
    steps: List[Dict[str, Any]]
    parallel_execution: bool = False
    max_retries: int = 3
    timeout_seconds: int = 3600
    checkpoint_enabled: bool = True
    checkpoint_interval: int = 300
    monitoring_enabled: bool = True
    
    # Configuration des ressources
    max_memory_gb: Optional[float] = None
    max_cpu_cores: Optional[int] = None
    gpu_required: bool = False
    priority: int = 5  # 1-10, 10 = highest priority
    
    def validate(self) -> bool:
        """Validation de la configuration du pipeline"""
        if not self.pipeline_id or not self.steps:
            return False
        if self.max_retries < 0 or self.timeout_seconds <= 0:
            return False
        if self.priority < 1 or self.priority > 10:
            return False
        return True


@dataclass
class SecurityConfig:
    """Configuration sécurité"""
    enable_authentication: bool = True
    enable_authorization: bool = True
    secret_key: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    api_key_length: int = 32
    token_expiry_hours: int = 24
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    
    # Chiffrement
    encryption_algorithm: str = "AES-256-GCM"
    hash_algorithm: str = "SHA-256"
    salt_length: int = 16
    
    # TLS/SSL
    tls_enabled: bool = True
    cert_path: Optional[str] = None
    key_path: Optional[str] = None
    ca_path: Optional[str] = None
    
    # Audit
    audit_enabled: bool = True
    audit_retention_days: int = 90
    sensitive_data_masking: bool = True
    
    def generate_api_key(self) -> str:
        """Génération d'une clé API sécurisée"""
        return secrets.token_urlsafe(self.api_key_length)
    
    def hash_password(self, password: str) -> str:
        """Hashage sécurisé d'un mot de passe"""
        salt = secrets.token_bytes(self.salt_length)
        hashed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
        return salt.hex() + hashed.hex()
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Vérification d'un mot de passe"""
        try:
            salt = bytes.fromhex(hashed[:self.salt_length * 2])
            stored_hash = hashed[self.salt_length * 2:]
            new_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
            return new_hash.hex() == stored_hash
        except (ValueError, TypeError):
            return False


@dataclass
class MonitoringConfig:
    """Configuration monitoring et observabilité"""
    enabled: bool = True
    metrics_endpoint: str = "/metrics"
    health_endpoint: str = "/health"
    
    # Prometheus
    prometheus_enabled: bool = True
    prometheus_port: int = 8000
    prometheus_path: str = "/metrics"
    
    # Jaeger tracing
    jaeger_enabled: bool = True
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    jaeger_service_name: str = "ml-analytics"
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "structured"  # structured, text
    log_file: Optional[str] = None
    log_rotation: bool = True
    log_max_size_mb: int = 100
    log_backup_count: int = 5
    
    # Alerting
    alerting_enabled: bool = True
    alert_webhook_url: Optional[str] = None
    alert_email: Optional[str] = None
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "error_rate": 0.05,  # 5%
        "latency_p99": 1000,  # 1 seconde
        "memory_usage": 0.8,  # 80%
        "cpu_usage": 0.8,     # 80%
    })


@dataclass
class PerformanceConfig:
    """Configuration performance"""
    # Threading et async
    max_workers: int = 4
    async_pool_size: int = 100
    queue_max_size: int = 1000
    
    # Cache
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    cache_max_size_mb: int = 1024
    
    # Optimisations ML
    model_compilation: bool = True
    mixed_precision: bool = False
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 2
    prefetch_factor: int = 2
    
    # Batch processing
    default_batch_size: int = 32
    max_batch_size: int = 256
    auto_batch_sizing: bool = True
    
    # Memory management
    memory_limit_gb: Optional[float] = None
    memory_cleanup_threshold: float = 0.8
    garbage_collection_interval: int = 300


class MLAnalyticsConfig(BaseSettings):
    """Configuration principale ML Analytics"""
    
    # Environnement
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    testing: bool = False
    
    # Application
    app_name: str = "ML Analytics"
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    
    # Serveur
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    
    # Bases de données
    database: DatabaseConfig = field(default_factory=lambda: DatabaseConfig(
        type=DatabaseType.POSTGRESQL,
        host="localhost",
        port=5432,
        database="ml_analytics",
        username="postgres",
        password="password"
    ))
    
    redis: RedisConfig = field(default_factory=RedisConfig)
    mongodb: MongoConfig = field(default_factory=MongoConfig)
    
    # Sécurité
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Monitoring
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Performance
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Modèles
    models: List[ModelConfig] = field(default_factory=list)
    
    # Pipelines
    pipelines: List[PipelineConfig] = field(default_factory=list)
    
    # Répertoires
    data_dir: str = "./data"
    models_dir: str = "./models"
    logs_dir: str = "./logs"
    cache_dir: str = "./cache"
    checkpoints_dir: str = "./checkpoints"
    
    class Config:
        """Configuration Pydantic"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @validator('environment', pre=True)
    def validate_environment(cls, v):
        """Validation de l'environnement"""
        if isinstance(v, str):
            try:
                return Environment(v.lower())
            except ValueError:
                return Environment.DEVELOPMENT
        return v
    
    @validator('port')
    def validate_port(cls, v):
        """Validation du port"""
        if not (1 <= v <= 65535):
            raise ValueError("Le port doit être entre 1 et 65535")
        return v
    
    @validator('workers')
    def validate_workers(cls, v):
        """Validation du nombre de workers"""
        if v < 1:
            raise ValueError("Le nombre de workers doit être >= 1")
        return v
    
    def validate_all(self) -> List[str]:
        """Validation complète de la configuration"""
        errors = []
        
        # Validation de la base de données
        if not self.database.validate():
            errors.append("Configuration base de données invalide")
        
        # Validation des modèles
        for model in self.models:
            if not model.validate():
                errors.append(f"Configuration modèle {model.model_id} invalide")
        
        # Validation des pipelines
        for pipeline in self.pipelines:
            if not pipeline.validate():
                errors.append(f"Configuration pipeline {pipeline.pipeline_id} invalide")
        
        # Validation des répertoires
        for dir_name in ['data_dir', 'models_dir', 'logs_dir', 'cache_dir', 'checkpoints_dir']:
            dir_path = getattr(self, dir_name)
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path, exist_ok=True)
                except Exception as e:
                    errors.append(f"Impossible de créer le répertoire {dir_path}: {e}")
        
        return errors
    
    def create_directories(self) -> bool:
        """Création des répertoires nécessaires"""
        try:
            directories = [
                self.data_dir, self.models_dir, self.logs_dir,
                self.cache_dir, self.checkpoints_dir
            ]
            
            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)
            
            return True
        except Exception as e:
            logging.error(f"Erreur lors de la création des répertoires: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire"""
        return asdict(self)
    
    def save_to_file(self, file_path: str, format: str = "yaml") -> bool:
        """Sauvegarde de la configuration dans un fichier"""
        try:
            config_dict = self.to_dict()
            
            if format.lower() == "yaml":
                with open(file_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif format.lower() == "json":
                with open(file_path, 'w') as f:
                    json.dump(config_dict, f, indent=2, default=str)
            else:
                raise ValueError(f"Format non supporté: {format}")
            
            return True
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde: {e}")
            return False
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'MLAnalyticsConfig':
        """Chargement de la configuration depuis un fichier"""
        try:
            with open(file_path, 'r') as f:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    config_dict = yaml.safe_load(f)
                elif file_path.endswith('.json'):
                    config_dict = json.load(f)
                else:
                    raise ValueError(f"Format de fichier non supporté: {file_path}")
            
            return cls(**config_dict)
        except Exception as e:
            logging.error(f"Erreur lors du chargement: {e}")
            return cls()
    
    def get_database_url(self, db_name: Optional[str] = None) -> str:
        """URL de connexion à la base de données"""
        if db_name:
            # Créer une copie avec un nom de base différent
            db_config = DatabaseConfig(
                type=self.database.type,
                host=self.database.host,
                port=self.database.port,
                database=db_name,
                username=self.database.username,
                password=self.database.password
            )
            return db_config.connection_url
        return self.database.connection_url
    
    def get_redis_url(self, db_index: Optional[int] = None) -> str:
        """URL de connexion Redis"""
        if db_index is not None:
            redis_config = RedisConfig(
                host=self.redis.host,
                port=self.redis.port,
                database=db_index,
                password=self.redis.password,
                ssl=self.redis.ssl
            )
            return redis_config.connection_url
        return self.redis.connection_url
    
    def get_mongodb_url(self, db_name: Optional[str] = None) -> str:
        """URL de connexion MongoDB"""
        if db_name:
            mongo_config = MongoConfig(
                host=self.mongodb.host,
                port=self.mongodb.port,
                database=db_name,
                username=self.mongodb.username,
                password=self.mongodb.password
            )
            return mongo_config.connection_url
        return self.mongodb.connection_url


# Configuration par défaut pour le développement
DEFAULT_DEV_CONFIG = MLAnalyticsConfig(
    environment=Environment.DEVELOPMENT,
    debug=True,
    reload=True,
    database=DatabaseConfig(
        type=DatabaseType.SQLITE,
        host="localhost",
        port=0,
        database="./data/ml_analytics_dev.db",
        username="",
        password=""
    )
)

# Configuration par défaut pour la production
DEFAULT_PROD_CONFIG = MLAnalyticsConfig(
    environment=Environment.PRODUCTION,
    debug=False,
    reload=False,
    workers=4,
    database=DatabaseConfig(
        type=DatabaseType.POSTGRESQL,
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME", "ml_analytics"),
        username=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password"),
        pool_size=20,
        max_overflow=40
    )
)


def load_config(
    config_file: Optional[str] = None,
    environment: Optional[str] = None
) -> MLAnalyticsConfig:
    """Chargement de la configuration"""
    
    # Déterminer l'environnement
    env = environment or os.getenv("ENVIRONMENT", "development")
    
    try:
        env_enum = Environment(env.lower())
    except ValueError:
        env_enum = Environment.DEVELOPMENT
    
    # Charger la configuration de base
    if config_file and os.path.exists(config_file):
        config = MLAnalyticsConfig.load_from_file(config_file)
    else:
        if env_enum == Environment.PRODUCTION:
            config = DEFAULT_PROD_CONFIG
        else:
            config = DEFAULT_DEV_CONFIG
    
    # Override avec les variables d'environnement
    config.environment = env_enum
    
    # Créer les répertoires nécessaires
    config.create_directories()
    
    # Validation
    errors = config.validate_all()
    if errors:
        logging.warning(f"Erreurs de configuration: {errors}")
    
    return config


# Instance globale de configuration
_config_instance: Optional[MLAnalyticsConfig] = None


def get_config() -> MLAnalyticsConfig:
    """Récupération de l'instance globale de configuration"""
    global _config_instance
    
    if _config_instance is None:
        _config_instance = load_config()
    
    return _config_instance


def set_config(config: MLAnalyticsConfig):
    """Définition de l'instance globale de configuration"""
    global _config_instance
    _config_instance = config


# Exports publics
__all__ = [
    'MLAnalyticsConfig',
    'DatabaseConfig',
    'RedisConfig', 
    'MongoConfig',
    'ModelConfig',
    'PipelineConfig',
    'SecurityConfig',
    'MonitoringConfig',
    'PerformanceConfig',
    'Environment',
    'DatabaseType',
    'MLFramework',
    'load_config',
    'get_config',
    'set_config',
    'DEFAULT_DEV_CONFIG',
    'DEFAULT_PROD_CONFIG'
]
