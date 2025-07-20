"""
Configuration Validation Schemas
===============================

Comprehensive validation schemas for all configuration types in the
Spotify AI Agent multi-tenant system. Provides strict type checking,
business rule validation, and security compliance verification.

Author: Fahed Mlaiel
Team: Spotify AI Agent Development Team
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum
import re
from datetime import datetime

class LogLevel(str, Enum):
    """Niveaux de logging autorisés."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class CacheStrategy(str, Enum):
    """Stratégies de cache disponibles."""
    LRU = "LRU"
    LFU = "LFU"
    FIFO = "FIFO"
    LIFO = "LIFO"
    RANDOM = "RANDOM"

class DatabaseType(str, Enum):
    """Types de base de données supportés."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"
    REDIS = "redis"
    ELASTICSEARCH = "elasticsearch"

class AuthProvider(str, Enum):
    """Fournisseurs d'authentification OAuth."""
    GOOGLE = "google"
    SPOTIFY = "spotify"
    GITHUB = "github"
    FACEBOOK = "facebook"
    TWITTER = "twitter"

class ApplicationConfigSchema(BaseModel):
    """Schema de validation pour la configuration application."""
    
    # Core Settings
    DEBUG: bool = Field(default=True, description="Mode debug")
    LOG_LEVEL: LogLevel = Field(default=LogLevel.DEBUG)
    ENVIRONMENT: str = Field(regex=r"^(development|staging|production|local)$")
    API_VERSION: str = Field(regex=r"^v\d+(\.\d+)?$")
    APPLICATION_NAME: str = Field(min_length=3, max_length=100)
    APPLICATION_VERSION: str = Field(regex=r"^\d+\.\d+\.\d+$")
    
    # Server Configuration
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(gt=0, le=65535, default=8000)
    MAX_WORKERS: int = Field(gt=0, le=100, default=4)
    WORKER_TIMEOUT: int = Field(gt=0, le=300, default=30)
    KEEP_ALIVE: int = Field(gt=0, le=10, default=2)
    
    # Performance & Scaling
    AUTO_SCALING_ENABLED: bool = Field(default=True)
    MIN_REPLICAS: int = Field(gt=0, le=100, default=2)
    MAX_REPLICAS: int = Field(gt=0, le=1000, default=10)
    CPU_THRESHOLD: int = Field(gt=0, le=100, default=70)
    MEMORY_THRESHOLD: int = Field(gt=0, le=100, default=80)
    
    # Caching
    CACHE_ENABLED: bool = Field(default=True)
    CACHE_TTL: int = Field(gt=0, le=86400, default=3600)
    CACHE_MAX_SIZE: int = Field(gt=0, default=1000)
    CACHE_STRATEGY: CacheStrategy = Field(default=CacheStrategy.LRU)
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = Field(default=True)
    RATE_LIMIT_REQUESTS: int = Field(gt=0, le=10000, default=100)
    RATE_LIMIT_WINDOW: int = Field(gt=0, le=3600, default=60)
    RATE_LIMIT_STORAGE: str = Field(regex=r"^(memory|redis)$", default="redis")
    
    # File Upload
    MAX_FILE_SIZE: int = Field(gt=0, le=1073741824, default=52428800)  # Max 1GB
    ALLOWED_FILE_TYPES: str = Field(default="mp3,wav,flac,m4a,ogg")
    UPLOAD_PATH: str = Field(default="/tmp/uploads")
    
    # Business Logic
    PAGINATION_SIZE: int = Field(gt=0, le=1000, default=50)
    MAX_SEARCH_RESULTS: int = Field(gt=0, le=10000, default=1000)
    SEARCH_TIMEOUT: int = Field(gt=0, le=60, default=5)
    
    @validator('ALLOWED_FILE_TYPES')
    def validate_file_types(cls, v):
        """Valide les types de fichiers autorisés."""
        allowed_extensions = ['mp3', 'wav', 'flac', 'm4a', 'ogg', 'aac', 'wma']
        types = [t.strip().lower() for t in v.split(',')]
        for file_type in types:
            if file_type not in allowed_extensions:
                raise ValueError(f"Type de fichier non autorisé: {file_type}")
        return v
    
    @root_validator
    def validate_scaling_config(cls, values):
        """Valide la configuration de scaling."""
        min_replicas = values.get('MIN_REPLICAS', 2)
        max_replicas = values.get('MAX_REPLICAS', 10)
        if min_replicas >= max_replicas:
            raise ValueError("MIN_REPLICAS doit être inférieur à MAX_REPLICAS")
        return values

class DatabaseConfigSchema(BaseModel):
    """Schema de validation pour la configuration database."""
    
    # PostgreSQL Primary
    DB_HOST: str = Field(min_length=3, max_length=255)
    DB_PORT: int = Field(gt=0, le=65535, default=5432)
    DB_NAME: str = Field(min_length=1, max_length=63)
    DB_USER: str = Field(min_length=1, max_length=63)
    DB_POOL_SIZE: int = Field(gt=0, le=1000, default=20)
    DB_MAX_OVERFLOW: int = Field(gt=0, le=1000, default=30)
    DB_POOL_TIMEOUT: int = Field(gt=0, le=300, default=30)
    DB_POOL_RECYCLE: int = Field(gt=0, le=86400, default=3600)
    DB_ECHO: bool = Field(default=False)
    
    # PostgreSQL Read Replica
    DB_READ_HOST: Optional[str] = Field(min_length=3, max_length=255)
    DB_READ_PORT: Optional[int] = Field(gt=0, le=65535, default=5432)
    DB_READ_ENABLED: bool = Field(default=False)
    
    # Redis Configuration
    REDIS_HOST: str = Field(min_length=3, max_length=255)
    REDIS_PORT: int = Field(gt=0, le=65535, default=6379)
    REDIS_DB: int = Field(ge=0, le=15, default=0)
    REDIS_MAX_CONNECTIONS: int = Field(gt=0, le=1000, default=100)
    REDIS_RETRY_ON_TIMEOUT: bool = Field(default=True)
    REDIS_HEALTH_CHECK_INTERVAL: int = Field(gt=0, le=300, default=30)
    
    # MongoDB Configuration
    MONGO_HOST: str = Field(min_length=3, max_length=255)
    MONGO_PORT: int = Field(gt=0, le=65535, default=27017)
    MONGO_DATABASE: str = Field(min_length=1, max_length=63)
    MONGO_REPLICA_SET: Optional[str] = Field(max_length=100)
    MONGO_AUTH_SOURCE: str = Field(default="admin")
    
    # ElasticSearch
    ELASTICSEARCH_HOST: str = Field(min_length=3, max_length=255)
    ELASTICSEARCH_PORT: int = Field(gt=0, le=65535, default=9200)
    ELASTICSEARCH_INDEX_PREFIX: str = Field(min_length=1, max_length=50)
    
    @validator('DB_NAME', 'MONGO_DATABASE')
    def validate_database_name(cls, v):
        """Valide le nom de la base de données."""
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', v):
            raise ValueError("Le nom de la base de données doit commencer par une lettre et contenir uniquement des lettres, chiffres et underscores")
        return v
    
    @root_validator
    def validate_pool_config(cls, values):
        """Valide la configuration du pool de connexions."""
        pool_size = values.get('DB_POOL_SIZE', 20)
        max_overflow = values.get('DB_MAX_OVERFLOW', 30)
        if pool_size + max_overflow > 1000:
            raise ValueError("La taille totale du pool (pool_size + max_overflow) ne peut pas dépasser 1000")
        return values

class SecurityConfigSchema(BaseModel):
    """Schema de validation pour la configuration sécurité."""
    
    # JWT Configuration
    JWT_ALGORITHM: str = Field(regex=r"^(HS256|HS384|HS512|RS256|RS384|RS512)$", default="HS256")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(gt=0, le=1440, default=30)  # Max 24h
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(gt=0, le=365, default=7)  # Max 1 year
    JWT_ISSUER: str = Field(min_length=3, max_length=100)
    JWT_AUDIENCE: str = Field(min_length=3, max_length=100)
    
    # OAuth Configuration
    OAUTH_ENABLED: bool = Field(default=True)
    OAUTH_PROVIDERS: str = Field(default="google,spotify,github")
    OAUTH_REDIRECT_URI: str = Field(regex=r'^https?://.+$')
    
    # API Security
    API_KEY_ENABLED: bool = Field(default=True)
    API_KEY_HEADER: str = Field(default="X-API-Key")
    API_RATE_LIMIT_ENABLED: bool = Field(default=True)
    API_RATE_LIMIT_REQUESTS: int = Field(gt=0, le=100000, default=1000)
    API_RATE_LIMIT_WINDOW: int = Field(gt=0, le=86400, default=3600)
    
    # HTTPS Configuration
    HTTPS_ONLY: bool = Field(default=False)
    SECURE_COOKIES: bool = Field(default=False)
    CSRF_PROTECTION: bool = Field(default=True)
    CSRF_TOKEN_HEADER: str = Field(default="X-CSRF-Token")
    
    # Session Management
    SESSION_TIMEOUT: int = Field(gt=0, le=86400, default=3600)  # Max 24h
    SESSION_REFRESH_ENABLED: bool = Field(default=True)
    MAX_SESSIONS_PER_USER: int = Field(gt=0, le=100, default=5)
    
    # Password Policy
    PASSWORD_MIN_LENGTH: int = Field(gt=0, le=128, default=8)
    PASSWORD_REQUIRE_UPPERCASE: bool = Field(default=True)
    PASSWORD_REQUIRE_LOWERCASE: bool = Field(default=True)
    PASSWORD_REQUIRE_DIGITS: bool = Field(default=True)
    PASSWORD_REQUIRE_SPECIAL: bool = Field(default=True)
    
    # Account Security
    MAX_LOGIN_ATTEMPTS: int = Field(gt=0, le=50, default=5)
    ACCOUNT_LOCKOUT_DURATION: int = Field(gt=0, le=86400, default=900)  # Max 24h
    TWO_FACTOR_ENABLED: bool = Field(default=False)
    
    @validator('OAUTH_PROVIDERS')
    def validate_oauth_providers(cls, v):
        """Valide les fournisseurs OAuth."""
        providers = [p.strip().lower() for p in v.split(',')]
        valid_providers = [p.value for p in AuthProvider]
        for provider in providers:
            if provider not in valid_providers:
                raise ValueError(f"Fournisseur OAuth non supporté: {provider}")
        return v
    
    @root_validator
    def validate_password_policy(cls, values):
        """Valide la politique de mots de passe."""
        min_length = values.get('PASSWORD_MIN_LENGTH', 8)
        if min_length < 8:
            raise ValueError("La longueur minimale du mot de passe doit être au moins 8 caractères")
        return values

class MLConfigSchema(BaseModel):
    """Schema de validation pour la configuration ML."""
    
    # Model Configuration
    ML_MODEL_PATH: str = Field(default="/app/models")
    ML_MODEL_VERSION: str = Field(regex=r"^\d+\.\d+\.\d+$", default="1.0.0")
    ML_MODEL_FORMAT: str = Field(regex=r"^(pytorch|tensorflow|onnx|sklearn)$", default="pytorch")
    ML_BATCH_SIZE: int = Field(gt=0, le=1024, default=32)
    ML_MAX_SEQUENCE_LENGTH: int = Field(gt=0, le=8192, default=512)
    
    # Training Configuration
    ML_TRAINING_ENABLED: bool = Field(default=True)
    ML_TRAINING_EPOCHS: int = Field(gt=0, le=1000, default=10)
    ML_LEARNING_RATE: float = Field(gt=0, le=1, default=0.001)
    ML_VALIDATION_SPLIT: float = Field(gt=0, lt=1, default=0.2)
    ML_EARLY_STOPPING: bool = Field(default=True)
    
    # Inference Configuration
    ML_INFERENCE_TIMEOUT: int = Field(gt=0, le=300, default=30)
    ML_INFERENCE_BATCH_SIZE: int = Field(gt=0, le=512, default=16)
    ML_GPU_ENABLED: bool = Field(default=False)
    ML_CPU_THREADS: int = Field(gt=0, le=64, default=4)
    
    # Audio Processing
    AUDIO_SAMPLE_RATE: int = Field(gt=0, le=192000, default=44100)
    AUDIO_CHUNK_SIZE: int = Field(gt=0, le=8192, default=1024)
    AUDIO_FORMAT: str = Field(regex=r"^(wav|mp3|flac|aac)$", default="wav")
    AUDIO_CHANNELS: int = Field(gt=0, le=8, default=2)
    
    # Spleeter Configuration
    SPLEETER_MODEL: str = Field(regex=r"^spleeter:(2|4|5)stems-(16|44.1)kHz$", default="spleeter:2stems-16kHz")
    SPLEETER_CACHE_ENABLED: bool = Field(default=True)
    SPLEETER_MAX_DURATION: int = Field(gt=0, le=3600, default=600)  # Max 1 hour
    
    @validator('ML_LEARNING_RATE')
    def validate_learning_rate(cls, v):
        """Valide le taux d'apprentissage."""
        if v <= 0 or v > 1:
            raise ValueError("Le taux d'apprentissage doit être entre 0 et 1")
        return v
    
    @validator('AUDIO_SAMPLE_RATE')
    def validate_sample_rate(cls, v):
        """Valide le taux d'échantillonnage audio."""
        valid_rates = [8000, 16000, 22050, 44100, 48000, 96000, 192000]
        if v not in valid_rates:
            raise ValueError(f"Taux d'échantillonnage non supporté: {v}. Valeurs autorisées: {valid_rates}")
        return v

class MonitoringConfigSchema(BaseModel):
    """Schema de validation pour la configuration monitoring."""
    
    # Prometheus Configuration
    PROMETHEUS_ENABLED: bool = Field(default=True)
    PROMETHEUS_PORT: int = Field(gt=0, le=65535, default=9090)
    PROMETHEUS_METRICS_PATH: str = Field(regex=r"^/.*", default="/metrics")
    PROMETHEUS_SCRAPE_INTERVAL: str = Field(regex=r"^\d+[smh]$", default="15s")
    
    # Grafana Configuration
    GRAFANA_ENABLED: bool = Field(default=True)
    GRAFANA_PORT: int = Field(gt=0, le=65535, default=3000)
    GRAFANA_ADMIN_USER: str = Field(min_length=3, max_length=50, default="admin")
    
    # Jaeger Tracing
    JAEGER_ENABLED: bool = Field(default=True)
    JAEGER_AGENT_HOST: str = Field(min_length=3, max_length=255)
    JAEGER_AGENT_PORT: int = Field(gt=0, le=65535, default=6831)
    JAEGER_SAMPLER_TYPE: str = Field(regex=r"^(const|probabilistic|rate_limiting)$", default="const")
    JAEGER_SAMPLER_PARAM: float = Field(ge=0, le=1, default=1)
    
    # Logging Configuration
    LOG_FORMAT: str = Field(regex=r"^(json|text)$", default="json")
    LOG_LEVEL: LogLevel = Field(default=LogLevel.DEBUG)
    LOG_ROTATION: str = Field(regex=r"^(daily|weekly|size)$", default="daily")
    LOG_MAX_SIZE: str = Field(regex=r"^\d+[KMGT]?B$", default="100MB")
    LOG_BACKUP_COUNT: int = Field(gt=0, le=365, default=7)
    
    # Health Checks
    HEALTH_CHECK_ENABLED: bool = Field(default=True)
    HEALTH_CHECK_PATH: str = Field(regex=r"^/.*", default="/health")
    HEALTH_CHECK_INTERVAL: str = Field(regex=r"^\d+[smh]$", default="30s")
    READINESS_CHECK_PATH: str = Field(regex=r"^/.*", default="/ready")
    LIVENESS_CHECK_PATH: str = Field(regex=r"^/.*", default="/live")
    
    # Resource Monitoring
    RESOURCE_MONITORING_ENABLED: bool = Field(default=True)
    CPU_ALERT_THRESHOLD: int = Field(gt=0, le=100, default=80)
    MEMORY_ALERT_THRESHOLD: int = Field(gt=0, le=100, default=85)
    DISK_ALERT_THRESHOLD: int = Field(gt=0, le=100, default=90)
    
    @validator('PROMETHEUS_SCRAPE_INTERVAL', 'HEALTH_CHECK_INTERVAL')
    def validate_time_interval(cls, v):
        """Valide les intervalles de temps."""
        if not re.match(r'^\d+[smh]$', v):
            raise ValueError("L'intervalle doit être au format '15s', '5m', ou '1h'")
        return v

class ConfigurationValidator:
    """Validateur principal pour toutes les configurations."""
    
    def __init__(self):
        self.schemas = {
            'application': ApplicationConfigSchema,
            'database': DatabaseConfigSchema,
            'security': SecurityConfigSchema,
            'ml': MLConfigSchema,
            'monitoring': MonitoringConfigSchema
        }
    
    def validate_config(self, config_type: str, config_data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Valide une configuration selon son type."""
        if config_type not in self.schemas:
            return False, [f"Type de configuration non supporté: {config_type}"]
        
        try:
            schema_class = self.schemas[config_type]
            schema_class(**config_data)
            return True, []
        except Exception as e:
            return False, [str(e)]
    
    def validate_all_configs(self, all_configs: Dict[str, Dict[str, Any]]) -> tuple[bool, Dict[str, List[str]]]:
        """Valide toutes les configurations."""
        results = {}
        all_valid = True
        
        for config_type, config_data in all_configs.items():
            is_valid, errors = self.validate_config(config_type, config_data)
            if not is_valid:
                all_valid = False
            results[config_type] = errors
        
        return all_valid, results

# Exportation des classes
__all__ = [
    'ApplicationConfigSchema',
    'DatabaseConfigSchema', 
    'SecurityConfigSchema',
    'MLConfigSchema',
    'MonitoringConfigSchema',
    'ConfigurationValidator',
    'LogLevel',
    'CacheStrategy',
    'DatabaseType',
    'AuthProvider'
]
