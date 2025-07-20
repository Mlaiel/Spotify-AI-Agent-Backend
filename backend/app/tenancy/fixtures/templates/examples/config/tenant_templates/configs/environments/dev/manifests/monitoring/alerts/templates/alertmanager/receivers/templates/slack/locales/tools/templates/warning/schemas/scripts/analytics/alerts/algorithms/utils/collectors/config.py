"""
Spotify AI Agent - Advanced Configuration Module
==============================================

Configuration ultra-avancée pour le système de collecteurs multi-tenant.
Supporte tous les profils de tenants, environnements, et patterns enterprise
avec validation, sérialisation et hot-reloading.

Classes de configuration:
- GlobalConfig: Configuration globale du système
- TenantConfig: Configuration spécifique par tenant
- EnvironmentConfig: Configuration par environnement
- SecurityConfig: Configuration de sécurité avancée
- PerformanceConfig: Configuration de performance
- AlertingConfig: Configuration d'alerting
- StorageConfig: Configuration de stockage
- MonitoringConfig: Configuration de monitoring

Fonctionnalités:
- Validation automatique avec Pydantic
- Hot-reloading de configuration
- Profils de configuration prédéfinis
- Inheritance et override de configuration
- Encryption des secrets
- Audit trail des changements
"""

from typing import Dict, Any, List, Optional, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import os
import json
import yaml
from pathlib import Path
import logging
import structlog
from pydantic import BaseModel, validator, Field
from cryptography.fernet import Fernet


class TenantProfile(Enum):
    """Profils de tenants supportés."""
    STARTER = "starter"
    STANDARD = "standard" 
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class Environment(Enum):
    """Environnements supportés."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    DR = "disaster_recovery"


class StorageBackend(Enum):
    """Backends de stockage supportés."""
    REDIS = "redis"
    POSTGRESQL = "postgresql"
    TIMESCALEDB = "timescaledb"
    INFLUXDB = "influxdb"
    ELASTICSEARCH = "elasticsearch"
    KAFKA = "kafka"
    S3 = "s3"
    GCS = "gcs"


class AlertChannel(Enum):
    """Canaux d'alerte supportés."""
    SLACK = "slack"
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    OPSGENIE = "opsgenie"
    TEAMS = "teams"


class CompressionLevel(Enum):
    """Niveaux de compression."""
    NONE = 0
    LOW = 1
    MEDIUM = 5
    HIGH = 9


@dataclass
class CircuitBreakerConfig:
    """Configuration du circuit breaker."""
    enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout_seconds: int = 60
    half_open_max_calls: int = 3
    expected_exceptions: List[str] = field(default_factory=lambda: ["ConnectionError", "TimeoutError"])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout_seconds": self.recovery_timeout_seconds,
            "half_open_max_calls": self.half_open_max_calls,
            "expected_exceptions": self.expected_exceptions
        }


@dataclass
class RateLimitConfig:
    """Configuration du rate limiting."""
    enabled: bool = True
    max_calls_per_minute: int = 1000
    burst_limit: int = 100
    algorithm: str = "token_bucket"  # token_bucket, sliding_window, fixed_window
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_calls_per_minute": self.max_calls_per_minute,
            "burst_limit": self.burst_limit,
            "algorithm": self.algorithm
        }


@dataclass
class RetryConfig:
    """Configuration des tentatives de retry."""
    enabled: bool = True
    max_attempts: int = 3
    backoff_factor: float = 2.0
    max_backoff_seconds: int = 300
    jitter: bool = True
    retryable_exceptions: List[str] = field(default_factory=lambda: ["ConnectionError", "TimeoutError"])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_attempts": self.max_attempts,
            "backoff_factor": self.backoff_factor,
            "max_backoff_seconds": self.max_backoff_seconds,
            "jitter": self.jitter,
            "retryable_exceptions": self.retryable_exceptions
        }


class SecurityConfig(BaseModel):
    """Configuration de sécurité avancée."""
    
    # Encryption
    enable_encryption: bool = True
    encryption_algorithm: str = "AES-256-GCM"
    key_rotation_days: int = 90
    encryption_key: Optional[str] = None
    
    # Authentication & Authorization
    enable_authentication: bool = True
    auth_provider: str = "oauth2"  # oauth2, jwt, api_key, mtls
    api_key: Optional[str] = None
    oauth_client_id: Optional[str] = None
    oauth_client_secret: Optional[str] = None
    jwt_secret: Optional[str] = None
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    
    # TLS/SSL
    enable_tls: bool = True
    tls_version: str = "1.3"
    cert_path: Optional[str] = None
    key_path: Optional[str] = None
    ca_path: Optional[str] = None
    verify_certificates: bool = True
    
    # Access Control
    enable_rbac: bool = True
    default_role: str = "reader"
    admin_roles: List[str] = field(default_factory=lambda: ["admin", "super_admin"])
    
    # Compliance
    gdpr_mode: bool = True
    sox_compliance: bool = False
    pci_dss_compliance: bool = False
    data_residency_regions: List[str] = field(default_factory=lambda: ["EU", "US"])
    
    # Audit
    enable_audit_log: bool = True
    audit_retention_days: int = 365
    sensitive_fields: List[str] = field(default_factory=lambda: ["password", "token", "key"])
    
    class Config:
        extra = "forbid"
        validate_assignment = True
    
    @validator('encryption_key')
    def validate_encryption_key(cls, v):
        if v and len(v) < 32:
            raise ValueError('Encryption key must be at least 32 characters')
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        return self.dict()


class PerformanceConfig(BaseModel):
    """Configuration de performance avancée."""
    
    # Threading & Concurrency
    max_workers: int = 10
    thread_pool_size: int = 4
    async_pool_size: int = 100
    
    # Memory Management
    max_memory_mb: int = 1024
    gc_threshold: int = 1000
    enable_memory_profiling: bool = False
    
    # Caching
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    cache_max_size: int = 10000
    cache_eviction_policy: str = "LRU"
    
    # Batching
    enable_batching: bool = True
    batch_size: int = 1000
    batch_timeout_seconds: int = 5
    max_batch_memory_mb: int = 100
    
    # Compression
    enable_compression: bool = True
    compression_algorithm: str = "lz4"
    compression_level: CompressionLevel = CompressionLevel.MEDIUM
    
    # Connection Pooling
    max_connections: int = 100
    connection_timeout_seconds: int = 30
    connection_retry_attempts: int = 3
    
    # Optimization Flags
    enable_cpu_optimization: bool = True
    enable_memory_optimization: bool = True
    enable_network_optimization: bool = True
    enable_disk_optimization: bool = True
    
    class Config:
        extra = "forbid"
        validate_assignment = True
        use_enum_values = True
    
    @validator('max_memory_mb')
    def validate_memory(cls, v):
        if v < 128:
            raise ValueError('Max memory must be at least 128MB')
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        return self.dict()


class AlertingConfig(BaseModel):
    """Configuration d'alerting avancée."""
    
    # Global Settings
    enabled: bool = True
    default_severity: str = "warning"
    escalation_timeout_minutes: int = 15
    
    # Channels
    enabled_channels: List[AlertChannel] = field(default_factory=lambda: [AlertChannel.SLACK, AlertChannel.EMAIL])
    
    # Slack Configuration
    slack_webhook_url: Optional[str] = None
    slack_channel: str = "#alerts"
    slack_username: str = "Spotify AI Agent"
    slack_emoji: str = ":warning:"
    
    # Email Configuration
    smtp_server: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_use_tls: bool = True
    email_from: str = "alerts@spotify-ai-agent.com"
    email_to: List[str] = field(default_factory=list)
    
    # PagerDuty Configuration
    pagerduty_api_key: Optional[str] = None
    pagerduty_service_key: Optional[str] = None
    
    # Webhook Configuration
    webhook_urls: List[str] = field(default_factory=list)
    webhook_timeout_seconds: int = 10
    webhook_retry_attempts: int = 3
    
    # Alert Rules
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    alert_cooldown_minutes: int = 5
    alert_aggregation_window_minutes: int = 1
    
    # Notification Preferences
    business_hours_only: bool = False
    timezone: str = "UTC"
    quiet_hours_start: str = "22:00"
    quiet_hours_end: str = "08:00"
    
    class Config:
        extra = "forbid"
        validate_assignment = True
        use_enum_values = True
    
    def to_dict(self) -> Dict[str, Any]:
        return self.dict()


class StorageConfig(BaseModel):
    """Configuration de stockage avancée."""
    
    # Primary Storage
    primary_backend: StorageBackend = StorageBackend.REDIS
    
    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_ssl: bool = False
    redis_cluster_enabled: bool = False
    redis_cluster_nodes: List[str] = field(default_factory=list)
    redis_max_connections: int = 100
    redis_connection_timeout: int = 10
    
    # PostgreSQL Configuration
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "spotify_ai_agent"
    postgres_username: str = "postgres"
    postgres_password: Optional[str] = None
    postgres_ssl_mode: str = "prefer"
    postgres_max_connections: int = 20
    postgres_connection_timeout: int = 30
    
    # TimescaleDB Configuration (extends PostgreSQL)
    timescaledb_enabled: bool = False
    timescaledb_chunk_time_interval: str = "1 day"
    timescaledb_compression_enabled: bool = True
    timescaledb_retention_policy: str = "30 days"
    
    # InfluxDB Configuration
    influxdb_url: str = "http://localhost:8086"
    influxdb_token: Optional[str] = None
    influxdb_org: str = "spotify-ai-agent"
    influxdb_bucket: str = "collectors"
    influxdb_timeout: int = 30
    
    # Elasticsearch Configuration
    elasticsearch_hosts: List[str] = field(default_factory=lambda: ["localhost:9200"])
    elasticsearch_username: Optional[str] = None
    elasticsearch_password: Optional[str] = None
    elasticsearch_index_prefix: str = "collectors"
    elasticsearch_timeout: int = 30
    elasticsearch_max_retries: int = 3
    
    # Kafka Configuration
    kafka_bootstrap_servers: List[str] = field(default_factory=lambda: ["localhost:9092"])
    kafka_topic_prefix: str = "collectors"
    kafka_producer_config: Dict[str, Any] = field(default_factory=dict)
    kafka_consumer_config: Dict[str, Any] = field(default_factory=dict)
    
    # S3 Configuration
    s3_bucket: Optional[str] = None
    s3_region: str = "us-east-1"
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    s3_endpoint_url: Optional[str] = None
    
    # Data Management
    data_retention_days: int = 30
    backup_enabled: bool = True
    backup_schedule: str = "0 2 * * *"  # Cron format
    compression_enabled: bool = True
    encryption_at_rest: bool = True
    
    # Partitioning
    enable_partitioning: bool = True
    partition_by: str = "tenant_id"
    partition_size_mb: int = 1024
    
    class Config:
        extra = "forbid"
        validate_assignment = True
        use_enum_values = True
    
    def to_dict(self) -> Dict[str, Any]:
        return self.dict()


class MonitoringConfig(BaseModel):
    """Configuration de monitoring et observabilité."""
    
    # Global Settings
    enabled: bool = True
    collection_interval_seconds: int = 60
    
    # Metrics
    enable_metrics: bool = True
    metrics_port: int = 9090
    metrics_path: str = "/metrics"
    metrics_retention_days: int = 15
    
    # Prometheus Configuration
    prometheus_enabled: bool = True
    prometheus_host: str = "localhost"
    prometheus_port: int = 9090
    prometheus_push_gateway: Optional[str] = None
    
    # Grafana Configuration
    grafana_enabled: bool = True
    grafana_host: str = "localhost"
    grafana_port: int = 3000
    grafana_api_key: Optional[str] = None
    
    # Tracing
    enable_tracing: bool = True
    trace_sample_rate: float = 0.1
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    
    # Logging
    enable_structured_logging: bool = True
    log_level: str = "INFO"
    log_format: str = "json"
    log_rotation_size_mb: int = 100
    log_retention_days: int = 30
    
    # Health Checks
    enable_health_checks: bool = True
    health_check_interval_seconds: int = 30
    health_check_timeout_seconds: int = 10
    health_check_unhealthy_threshold: int = 3
    
    # Custom Metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    class Config:
        extra = "forbid"
        validate_assignment = True
    
    def to_dict(self) -> Dict[str, Any]:
        return self.dict()


class TenantConfig(BaseModel):
    """Configuration spécifique par tenant."""
    
    # Basic Information
    tenant_id: str
    name: str
    profile: TenantProfile = TenantProfile.STANDARD
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Environment
    environment: Environment = Environment.PRODUCTION
    
    # Monitoring Configuration
    monitoring_level: str = "standard"  # minimal, standard, comprehensive
    real_time_enabled: bool = False
    
    # Compliance
    compliance_mode: str = "standard"  # relaxed, standard, strict
    data_residency: str = "EU"
    gdpr_enabled: bool = True
    
    # Resource Limits
    max_collections_per_minute: int = 1000
    max_data_points_per_collection: int = 10000
    max_storage_mb: int = 1024
    max_retention_days: int = 30
    
    # Feature Flags
    enabled_collectors: List[str] = field(default_factory=list)
    experimental_features: List[str] = field(default_factory=list)
    
    # Custom Configuration
    custom_config: Dict[str, Any] = field(default_factory=dict)
    
    # Sub-configurations
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    rate_limiting: RateLimitConfig = field(default_factory=RateLimitConfig)
    retry_policy: RetryConfig = field(default_factory=RetryConfig)
    
    class Config:
        extra = "forbid"
        validate_assignment = True
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @validator('tenant_id')
    def validate_tenant_id(cls, v):
        if not v or len(v) < 3:
            raise ValueError('Tenant ID must be at least 3 characters')
        return v.lower()
    
    def to_dict(self) -> Dict[str, Any]:
        return self.dict()


class EnvironmentConfig(BaseModel):
    """Configuration par environnement."""
    
    environment: Environment
    
    # Infrastructure
    cpu_limit: float = 2.0
    memory_limit_mb: int = 2048
    storage_limit_gb: int = 100
    
    # Networking
    max_connections: int = 1000
    connection_timeout_seconds: int = 30
    
    # Performance
    performance_profile: str = "balanced"  # minimal, balanced, high_performance
    auto_scaling_enabled: bool = True
    min_replicas: int = 1
    max_replicas: int = 10
    
    # Security
    security_level: str = "standard"  # minimal, standard, high, maximum
    
    # Logging & Monitoring
    debug_enabled: bool = False
    verbose_logging: bool = False
    
    class Config:
        extra = "forbid"
        validate_assignment = True
        use_enum_values = True
    
    def to_dict(self) -> Dict[str, Any]:
        return self.dict()


class GlobalConfig(BaseModel):
    """Configuration globale du système."""
    
    # System Information
    system_name: str = "Spotify AI Agent Collectors"
    version: str = "3.0.0"
    build_number: Optional[str] = None
    deployed_at: datetime = field(default_factory=datetime.utcnow)
    
    # Default Environment
    default_environment: Environment = Environment.PRODUCTION
    
    # Sub-configurations
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    alerting: AlertingConfig = field(default_factory=AlertingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Environment-specific configurations
    environments: Dict[str, EnvironmentConfig] = field(default_factory=dict)
    
    # Tenant configurations
    tenants: Dict[str, TenantConfig] = field(default_factory=dict)
    
    # Feature Flags
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    
    # Hot Reload
    hot_reload_enabled: bool = True
    config_file_path: Optional[str] = None
    config_check_interval_seconds: int = 60
    
    class Config:
        extra = "forbid"
        validate_assignment = True
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return self.dict()
    
    def get_tenant_config(self, tenant_id: str) -> Optional[TenantConfig]:
        """Récupère la configuration d'un tenant."""
        return self.tenants.get(tenant_id)
    
    def get_environment_config(self, environment: Union[str, Environment]) -> Optional[EnvironmentConfig]:
        """Récupère la configuration d'un environnement."""
        if isinstance(environment, Environment):
            environment = environment.value
        return self.environments.get(environment)
    
    def add_tenant(self, tenant_config: TenantConfig) -> None:
        """Ajoute un tenant à la configuration."""
        self.tenants[tenant_config.tenant_id] = tenant_config
    
    def remove_tenant(self, tenant_id: str) -> bool:
        """Supprime un tenant de la configuration."""
        if tenant_id in self.tenants:
            del self.tenants[tenant_id]
            return True
        return False


class ConfigurationManager:
    """
    Gestionnaire de configuration ultra-avancé.
    
    Fonctionnalités:
    - Chargement depuis fichiers YAML/JSON
    - Hot-reloading automatique
    - Validation et sérialisation
    - Encryption des secrets
    - Audit trail des changements
    - Cache intelligent
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config: Optional[GlobalConfig] = None
        self.config_cache: Dict[str, Any] = {}
        self.last_reload: Optional[datetime] = None
        self.encryption_key: Optional[bytes] = None
        self.logger = structlog.get_logger(__name__)
        
        # Initialisation
        if config_path:
            self.load_configuration()
    
    def load_configuration(self, config_path: Optional[str] = None) -> GlobalConfig:
        """Charge la configuration depuis un fichier."""
        if config_path:
            self.config_path = config_path
        
        if not self.config_path:
            raise ValueError("Chemin de configuration non spécifié")
        
        path = Path(self.config_path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier de configuration non trouvé: {self.config_path}")
        
        try:
            # Chargement du fichier
            with open(path, 'r', encoding='utf-8') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                elif path.suffix.lower() == '.json':
                    data = json.load(f)
                else:
                    raise ValueError(f"Format de fichier non supporté: {path.suffix}")
            
            # Décryption des secrets si nécessaire
            if self.encryption_key:
                data = self._decrypt_secrets(data)
            
            # Validation et création de la configuration
            self.config = GlobalConfig(**data)
            self.config.config_file_path = str(path)
            self.last_reload = datetime.utcnow()
            
            self.logger.info(
                "Configuration chargée avec succès",
                path=self.config_path,
                tenants_count=len(self.config.tenants),
                environments_count=len(self.config.environments)
            )
            
            return self.config
            
        except Exception as e:
            self.logger.error("Erreur lors du chargement de la configuration", error=str(e))
            raise
    
    def save_configuration(self, config_path: Optional[str] = None) -> None:
        """Sauvegarde la configuration dans un fichier."""
        if not self.config:
            raise ValueError("Aucune configuration à sauvegarder")
        
        if config_path:
            self.config_path = config_path
        
        if not self.config_path:
            raise ValueError("Chemin de configuration non spécifié")
        
        path = Path(self.config_path)
        
        try:
            # Conversion en dictionnaire
            data = self.config.to_dict()
            
            # Encryption des secrets si nécessaire
            if self.encryption_key:
                data = self._encrypt_secrets(data)
            
            # Sauvegarde
            with open(path, 'w', encoding='utf-8') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(data, f, default_flow_style=False, indent=2)
                elif path.suffix.lower() == '.json':
                    json.dump(data, f, indent=2, default=str)
                else:
                    raise ValueError(f"Format de fichier non supporté: {path.suffix}")
            
            self.logger.info("Configuration sauvegardée", path=self.config_path)
            
        except Exception as e:
            self.logger.error("Erreur lors de la sauvegarde", error=str(e))
            raise
    
    def get_tenant_config(self, tenant_id: str) -> Optional[TenantConfig]:
        """Récupère la configuration d'un tenant avec cache."""
        cache_key = f"tenant_{tenant_id}"
        
        if cache_key in self.config_cache:
            return self.config_cache[cache_key]
        
        if self.config:
            tenant_config = self.config.get_tenant_config(tenant_id)
            if tenant_config:
                self.config_cache[cache_key] = tenant_config
            return tenant_config
        
        return None
    
    def create_tenant_config(
        self,
        tenant_id: str,
        name: str,
        profile: TenantProfile = TenantProfile.STANDARD,
        **kwargs
    ) -> TenantConfig:
        """Crée une nouvelle configuration de tenant."""
        if not self.config:
            raise ValueError("Configuration globale non chargée")
        
        # Configuration par défaut selon le profil
        default_configs = self._get_default_tenant_config_by_profile(profile)
        default_configs.update(kwargs)
        
        tenant_config = TenantConfig(
            tenant_id=tenant_id,
            name=name,
            profile=profile,
            **default_configs
        )
        
        # Ajout à la configuration globale
        self.config.add_tenant(tenant_config)
        
        # Mise à jour du cache
        self.config_cache[f"tenant_{tenant_id}"] = tenant_config
        
        self.logger.info("Nouveau tenant créé", tenant_id=tenant_id, profile=profile.value)
        
        return tenant_config
    
    def _get_default_tenant_config_by_profile(self, profile: TenantProfile) -> Dict[str, Any]:
        """Retourne la configuration par défaut selon le profil."""
        
        configs = {
            TenantProfile.STARTER: {
                "monitoring_level": "minimal",
                "real_time_enabled": False,
                "max_collections_per_minute": 100,
                "max_data_points_per_collection": 1000,
                "max_storage_mb": 100,
                "max_retention_days": 7,
                "enabled_collectors": ["system_performance", "user_behavior"]
            },
            TenantProfile.STANDARD: {
                "monitoring_level": "standard",
                "real_time_enabled": False,
                "max_collections_per_minute": 1000,
                "max_data_points_per_collection": 10000,
                "max_storage_mb": 1024,
                "max_retention_days": 30,
                "enabled_collectors": [
                    "system_performance", "database_performance", 
                    "user_behavior", "streaming_quality"
                ]
            },
            TenantProfile.PREMIUM: {
                "monitoring_level": "comprehensive",
                "real_time_enabled": True,
                "max_collections_per_minute": 5000,
                "max_data_points_per_collection": 50000,
                "max_storage_mb": 5120,
                "max_retention_days": 90,
                "enabled_collectors": [
                    "system_performance", "database_performance", "api_performance",
                    "user_behavior", "streaming_quality", "business_metrics",
                    "ml_performance"
                ]
            },
            TenantProfile.ENTERPRISE: {
                "monitoring_level": "comprehensive",
                "real_time_enabled": True,
                "max_collections_per_minute": 20000,
                "max_data_points_per_collection": 100000,
                "max_storage_mb": 20480,
                "max_retention_days": 365,
                "compliance_mode": "strict",
                "enabled_collectors": [
                    "system_performance", "database_performance", "api_performance",
                    "security_events", "compliance", "audit_trail",
                    "user_behavior", "streaming_quality", "business_metrics",
                    "ml_performance", "infrastructure", "real_time"
                ],
                "experimental_features": ["advanced_analytics", "predictive_monitoring"]
            }
        }
        
        return configs.get(profile, configs[TenantProfile.STANDARD])
    
    def set_encryption_key(self, key: Union[str, bytes]) -> None:
        """Configure la clé de chiffrement pour les secrets."""
        if isinstance(key, str):
            key = key.encode('utf-8')
        
        self.encryption_key = key
        self.logger.info("Clé de chiffrement configurée")
    
    def _encrypt_secrets(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Chiffre les champs secrets dans la configuration."""
        if not self.encryption_key:
            return data
        
        fernet = Fernet(self.encryption_key)
        
        def encrypt_recursive(obj):
            if isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    if any(secret_field in key.lower() for secret_field in ['password', 'key', 'secret', 'token']):
                        if isinstance(value, str):
                            result[key] = fernet.encrypt(value.encode()).decode()
                        else:
                            result[key] = value
                    else:
                        result[key] = encrypt_recursive(value)
                return result
            elif isinstance(obj, list):
                return [encrypt_recursive(item) for item in obj]
            else:
                return obj
        
        return encrypt_recursive(data)
    
    def _decrypt_secrets(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Déchiffre les champs secrets dans la configuration."""
        if not self.encryption_key:
            return data
        
        fernet = Fernet(self.encryption_key)
        
        def decrypt_recursive(obj):
            if isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    if any(secret_field in key.lower() for secret_field in ['password', 'key', 'secret', 'token']):
                        if isinstance(value, str):
                            try:
                                result[key] = fernet.decrypt(value.encode()).decode()
                            except Exception:
                                result[key] = value  # Probablement pas chiffré
                        else:
                            result[key] = value
                    else:
                        result[key] = decrypt_recursive(value)
                return result
            elif isinstance(obj, list):
                return [decrypt_recursive(item) for item in obj]
            else:
                return obj
        
        return decrypt_recursive(data)
    
    def start_hot_reload(self) -> None:
        """Démarre le hot-reloading automatique."""
        if not self.config or not self.config.hot_reload_enabled:
            return
        
        async def hot_reload_loop():
            while self.config and self.config.hot_reload_enabled:
                try:
                    if self.config_path and Path(self.config_path).exists():
                        file_mtime = datetime.fromtimestamp(
                            Path(self.config_path).stat().st_mtime
                        )
                        
                        if self.last_reload and file_mtime > self.last_reload:
                            self.logger.info("Rechargement de la configuration détecté")
                            self.load_configuration()
                            self.config_cache.clear()  # Vider le cache
                    
                    await asyncio.sleep(self.config.config_check_interval_seconds)
                    
                except Exception as e:
                    self.logger.error("Erreur dans le hot-reload", error=str(e))
                    await asyncio.sleep(60)  # Attendre plus longtemps en cas d'erreur
        
        import asyncio
        asyncio.create_task(hot_reload_loop())
        self.logger.info("Hot-reloading activé")


# Instance globale du gestionnaire de configuration
config_manager = ConfigurationManager()


# Fonctions utilitaires
def load_config_from_file(config_path: str) -> GlobalConfig:
    """Charge la configuration depuis un fichier."""
    return config_manager.load_configuration(config_path)


def get_global_config() -> Optional[GlobalConfig]:
    """Récupère la configuration globale."""
    return config_manager.config


def get_tenant_config(tenant_id: str) -> Optional[TenantConfig]:
    """Récupère la configuration d'un tenant."""
    return config_manager.get_tenant_config(tenant_id)


def create_default_config() -> GlobalConfig:
    """Crée une configuration par défaut."""
    return GlobalConfig()


def create_tenant_config_for_profile(
    tenant_id: str,
    name: str,
    profile: TenantProfile
) -> TenantConfig:
    """Crée une configuration de tenant pour un profil donné."""
    return config_manager.create_tenant_config(tenant_id, name, profile)
