"""
🎵 Spotify AI Agent - API Core Configuration
===========================================

Configuration centralisée pour l'API avec patterns enterprise,
validation des settings, et gestion des environnements.

Architecture:
- Configuration multi-environnements (dev, test, prod)
- Validation Pydantic des settings
- Settings injectables et testables
- Configuration des middlewares
- Settings de sécurité enterprise
- Configuration cache et database
- Monitoring et observabilité

Développé par Fahed Mlaiel - Enterprise Configuration Expert
"""

import os
import secrets
from typing import Any, Dict, List, Optional, Union, ClassVar
from enum import Enum
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    """Environnements disponibles"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging" 
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Niveaux de log"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SecurityConfig(BaseSettings):
    """Configuration de sécurité enterprise"""
    
    # JWT Configuration
    secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # API Security
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    cors_credentials: bool = True
    cors_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    cors_headers: List[str] = ["*"]
    
    # Rate Limiting
    rate_limit_per_minute: int = 100
    rate_limit_burst: int = 20
    
    # Security Headers
    security_headers_enabled: bool = True
    hsts_max_age: int = 31536000
    content_security_policy: str = "default-src 'self'"
    
    # API Keys
    api_key_header: str = "X-API-Key"
    api_key_validation_enabled: bool = True
    
    model_config = {
        "env_prefix": "SECURITY_",
        "case_sensitive": False
    }


class CacheConfig(BaseSettings):
    """Configuration du système de cache"""
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_ssl: bool = False
    redis_pool_size: int = 10
    redis_timeout: int = 5
    
    # Memcached Configuration
    memcached_servers: List[str] = ["localhost:11211"]
    memcached_timeout: int = 5
    
    # Cache Behavior
    default_ttl: int = 3600  # 1 hour
    max_key_length: int = 250
    compression_enabled: bool = True
    compression_threshold: int = 1024
    
    # Cache Levels
    l1_enabled: bool = True  # Memory cache
    l1_max_size: int = 1000
    l1_ttl: int = 300
    
    l2_enabled: bool = True  # Redis cache
    l2_ttl: int = 3600
    
    l3_enabled: bool = False  # Memcached cache
    l3_ttl: int = 7200
    
    model_config = {
        "env_prefix": "CACHE_",
        "case_sensitive": False
    }


class DatabaseConfig(BaseSettings):
    """Configuration des bases de données"""
    
    # PostgreSQL Primary
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "spotify_user"
    postgres_password: str = "spotify_pass"
    postgres_db: str = "spotify_ai_agent"
    postgres_schema: str = "public"
    postgres_ssl_mode: str = "prefer"
    
    # Connection Pool
    postgres_pool_size: int = 20
    postgres_max_overflow: int = 10
    postgres_pool_timeout: int = 30
    postgres_pool_recycle: int = 3600
    
    # MongoDB Configuration
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db: str = "spotify_ai_agent"
    mongodb_collection_prefix: str = "ai_"
    
    # Elasticsearch Configuration
    elasticsearch_hosts: List[str] = ["http://localhost:9200"]
    elasticsearch_timeout: int = 10
    elasticsearch_max_retries: int = 3
    
    @property
    def postgres_url(self) -> str:
        """URL de connexion PostgreSQL"""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    @property
    def postgres_async_url(self) -> str:
        """URL de connexion PostgreSQL async"""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    model_config = {
        "env_prefix": "DB_",
        "case_sensitive": False
    }


class RedisConfig(BaseSettings):
    """Configuration Redis détaillée"""
    
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    ssl_cert_reqs: Optional[str] = None
    ssl_ca_certs: Optional[str] = None
    
    # Connection Pool
    max_connections: int = 50
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    
    # Sentinel Configuration
    sentinel_enabled: bool = False
    sentinel_hosts: List[str] = []
    sentinel_service: str = "mymaster"
    
    # Cluster Configuration  
    cluster_enabled: bool = False
    cluster_nodes: List[str] = []
    
    @property
    def url(self) -> str:
        """URL de connexion Redis"""
        auth = f":{self.password}@" if self.password else ""
        protocol = "rediss" if self.ssl else "redis"
        return f"{protocol}://{auth}{self.host}:{self.port}/{self.db}"
    
    model_config = {
        "env_prefix": "REDIS_",
        "case_sensitive": False
    }


class MonitoringConfig(BaseSettings):
    """Configuration du monitoring et observabilité"""
    
    # Métriques
    metrics_enabled: bool = True
    metrics_port: int = 8080
    metrics_path: str = "/metrics"
    
    # Health Checks
    health_checks_enabled: bool = True
    health_check_interval: int = 30
    health_check_timeout: int = 5
    
    # Logging
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "json"
    log_file: Optional[str] = None
    
    # Tracing
    tracing_enabled: bool = False
    jaeger_endpoint: Optional[str] = None
    
    # Alerting
    alerting_enabled: bool = False
    slack_webhook: Optional[str] = None
    email_alerts: List[str] = []
    
    model_config = {
        "env_prefix": "MONITORING_",
        "case_sensitive": False
    }


class APIConfig(BaseSettings):
    """Configuration principale de l'API"""
    
    # Application
    app_name: str = "Spotify AI Agent API"
    app_version: str = "2.0.0"
    app_description: str = "Enterprise AI Agent for Spotify with advanced ML capabilities"
    
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    testing: bool = False
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    
    # API Configuration
    api_v1_prefix: str = "/api/v1"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"
    
    # Request/Response
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    request_timeout: int = 30
    
    # Middleware
    middleware_enabled: Dict[str, bool] = {
        "cors": True,
        "gzip": True,
        "security": True,
        "rate_limit": True,
        "cache": True,
        "monitoring": True,
        "auth": True
    }
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        if v not in Environment:
            raise ValueError(f"Invalid environment: {v}")
        return v
    
    @model_validator(mode='after')
    def validate_config(self):
        env = self.environment
        
        # Production validations
        if env == Environment.PRODUCTION:
            if self.debug:
                raise ValueError("Debug mode cannot be enabled in production")
            if self.reload:
                raise ValueError("Reload cannot be enabled in production")
        
        return self
    
    model_config = {
        "env_prefix": "API_",
        "case_sensitive": False,
        "validate_assignment": True
    }


class APISettings(BaseSettings):
    """Settings globaux de l'API avec configuration composée"""
    
    # Core Configuration
    api: APIConfig = Field(default_factory=APIConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    # Feature Flags
    features: Dict[str, bool] = {
        "ml_recommendations": True,
        "audio_analysis": True,
        "social_features": True,
        "analytics": True,
        "real_time_sync": True,
        "advanced_search": True,
        "ai_playlists": True
    }
    
    # External Services
    spotify_client_id: Optional[str] = None
    spotify_client_secret: Optional[str] = None
    openai_api_key: Optional[str] = None
    huggingface_token: Optional[str] = None
    
    @field_validator("spotify_client_id")
    @classmethod
    def validate_spotify_client_id(cls, v, info):
        api_config = info.data.get("api")
        if api_config and hasattr(api_config, 'environment') and api_config.environment == Environment.PRODUCTION and not v:
            raise ValueError("Spotify client ID is required in production")
        return v
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "validate_assignment": True,
        "extra": "ignore"
    }


# =============================================================================
# INSTANCES GLOBALES ET FACTORIES
# =============================================================================

_settings: Optional[APISettings] = None


def get_settings() -> APISettings:
    """Retourne l'instance globale des settings (Singleton)"""
    global _settings
    if _settings is None:
        _settings = APISettings()
    return _settings


def get_api_config() -> APIConfig:
    """Retourne la configuration API"""
    return get_settings().api


def get_security_config() -> SecurityConfig:
    """Retourne la configuration de sécurité"""
    return get_settings().security


def get_cache_config() -> CacheConfig:
    """Retourne la configuration cache"""
    return get_settings().cache


def get_database_config() -> DatabaseConfig:
    """Retourne la configuration database"""
    return get_settings().database


def get_redis_config() -> RedisConfig:
    """Retourne la configuration Redis"""
    return get_settings().redis


def get_monitoring_config() -> MonitoringConfig:
    """Retourne la configuration monitoring"""
    return get_settings().monitoring


def create_development_config() -> APISettings:
    """Crée une configuration pour développement"""
    return APISettings(
        api=APIConfig(
            environment=Environment.DEVELOPMENT,
            debug=True,
            reload=True,
            workers=1
        ),
        security=SecurityConfig(
            cors_origins=["http://localhost:3000", "http://localhost:8000"]
        ),
        cache=CacheConfig(
            l1_enabled=True,
            l2_enabled=False,
            l3_enabled=False
        )
    )


def create_production_config() -> APISettings:
    """Crée une configuration pour production"""
    return APISettings(
        api=APIConfig(
            environment=Environment.PRODUCTION,
            debug=False,
            reload=False,
            workers=4
        ),
        security=SecurityConfig(
            cors_origins=[],  # À configurer selon le domaine
            rate_limit_per_minute=60
        ),
        cache=CacheConfig(
            l1_enabled=True,
            l2_enabled=True,
            l3_enabled=True
        ),
        spotify_client_id="prod_spotify_client_id",
        spotify_client_secret="prod_spotify_secret"
    )


def create_testing_config() -> APISettings:
    """Crée une configuration pour les tests"""
    return APISettings(
        api=APIConfig(
            environment=Environment.TESTING,
            debug=True,
            testing=True
        ),
        database=DatabaseConfig(
            postgres_db="spotify_ai_agent_test"
        ),
        cache=CacheConfig(
            redis_db=1,  # Base différente pour les tests
            default_ttl=10  # TTL court pour les tests
        )
    )


# Instance globale pour compatibilité
api_config = get_api_config()

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "Environment",
    "LogLevel",
    "SecurityConfig",
    "CacheConfig", 
    "DatabaseConfig",
    "RedisConfig",
    "MonitoringConfig",
    "APIConfig",
    "APISettings",
    "get_settings",
    "get_api_config",
    "get_security_config",
    "get_cache_config",
    "get_database_config",
    "get_redis_config",
    "get_monitoring_config",
    "create_development_config",
    "create_production_config",
    "create_testing_config",
    "api_config"
]
