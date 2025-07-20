"""
Schémas de configuration système - Spotify AI Agent
Configuration avancée pour les environnements et déploiements
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator, root_validator

from ..base import BaseSchema, TimestampMixin, TenantMixin
from ..base.enums import Environment, SecurityLevel, LogLevel
from ..validation import ValidationRules, SecurityValidationRules


class DatabaseType(str, Enum):
    """Types de bases de données supportées"""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"
    REDIS = "redis"
    ELASTICSEARCH = "elasticsearch"
    INFLUXDB = "influxdb"


class CacheType(str, Enum):
    """Types de cache supportés"""
    REDIS = "redis"
    MEMCACHED = "memcached"
    IN_MEMORY = "in_memory"
    DISTRIBUTED = "distributed"


class MonitoringProvider(str, Enum):
    """Fournisseurs de monitoring"""
    PROMETHEUS = "prometheus"
    GRAFANA = "grafana"
    DATADOG = "datadog"
    NEW_RELIC = "new_relic"
    ELASTIC_APM = "elastic_apm"


class DatabaseConfig(BaseSchema):
    """Configuration de base de données"""
    
    type: DatabaseType = Field(..., description="Type de base de données")
    host: str = Field(..., description="Adresse du serveur")
    port: int = Field(..., ge=1, le=65535, description="Port de connexion")
    database: str = Field(..., description="Nom de la base de données")
    username: str = Field(..., description="Nom d'utilisateur")
    password: str = Field(..., description="Mot de passe (sera chiffré)")
    
    # Configuration de la pool de connexions
    min_connections: int = Field(default=5, ge=1, le=100)
    max_connections: int = Field(default=20, ge=1, le=1000)
    connection_timeout: int = Field(default=30, ge=1, le=300, description="Timeout en secondes")
    idle_timeout: int = Field(default=300, ge=60, le=3600, description="Timeout d'inactivité en secondes")
    
    # Configuration SSL
    ssl_enabled: bool = Field(default=True, description="Activer SSL/TLS")
    ssl_cert_path: Optional[str] = Field(default=None, description="Chemin vers le certificat SSL")
    ssl_key_path: Optional[str] = Field(default=None, description="Chemin vers la clé SSL")
    ssl_ca_path: Optional[str] = Field(default=None, description="Chemin vers le CA SSL")
    
    # Configuration spécifique au type
    engine_options: Dict[str, Any] = Field(default_factory=dict, description="Options moteur spécifiques")
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Database password must be at least 8 characters")
        return v
    
    @root_validator
    def validate_connection_pool(cls, values):
        min_conn = values.get('min_connections', 5)
        max_conn = values.get('max_connections', 20)
        
        if min_conn > max_conn:
            raise ValueError("min_connections cannot be greater than max_connections")
        
        return values


class CacheConfig(BaseSchema):
    """Configuration de cache"""
    
    type: CacheType = Field(..., description="Type de cache")
    host: str = Field(..., description="Adresse du serveur cache")
    port: int = Field(..., ge=1, le=65535, description="Port de connexion")
    
    # Configuration Redis/Memcached
    password: Optional[str] = Field(default=None, description="Mot de passe cache")
    database: int = Field(default=0, ge=0, le=15, description="Numéro de base (Redis)")
    
    # Configuration de performance
    max_connections: int = Field(default=100, ge=10, le=1000)
    connection_timeout: int = Field(default=5, ge=1, le=30)
    socket_timeout: int = Field(default=5, ge=1, le=30)
    
    # Configuration TTL par défaut
    default_ttl: int = Field(default=3600, ge=60, le=86400, description="TTL par défaut en secondes")
    max_ttl: int = Field(default=86400, ge=300, le=604800, description="TTL maximum en secondes")
    
    # Configuration de clustering
    cluster_enabled: bool = Field(default=False, description="Mode cluster")
    cluster_nodes: List[str] = Field(default_factory=list, description="Nœuds du cluster")
    
    # Configuration de compression
    compression_enabled: bool = Field(default=True, description="Activer la compression")
    compression_threshold: int = Field(default=1024, ge=100, le=10240, description="Seuil de compression en bytes")


class SecurityConfig(BaseSchema):
    """Configuration de sécurité"""
    
    level: SecurityLevel = Field(..., description="Niveau de sécurité")
    
    # Configuration JWT
    jwt_secret_key: str = Field(..., description="Clé secrète JWT")
    jwt_algorithm: str = Field(default="HS256", description="Algorithme JWT")
    jwt_expiration_minutes: int = Field(default=60, ge=5, le=1440)
    jwt_refresh_expiration_days: int = Field(default=7, ge=1, le=30)
    
    # Configuration de chiffrement
    encryption_key: str = Field(..., description="Clé de chiffrement principale")
    encryption_algorithm: str = Field(default="fernet", description="Algorithme de chiffrement")
    
    # Configuration des mots de passe
    password_min_length: int = Field(default=12, ge=8, le=128)
    password_require_uppercase: bool = Field(default=True)
    password_require_lowercase: bool = Field(default=True)
    password_require_digits: bool = Field(default=True)
    password_require_special: bool = Field(default=True)
    password_history_count: int = Field(default=5, ge=0, le=24)
    
    # Configuration des sessions
    session_timeout_minutes: int = Field(default=30, ge=5, le=480)
    max_concurrent_sessions: int = Field(default=3, ge=1, le=10)
    
    # Configuration rate limiting
    rate_limit_enabled: bool = Field(default=True)
    rate_limit_requests_per_minute: int = Field(default=100, ge=10, le=10000)
    rate_limit_burst_multiplier: float = Field(default=1.5, ge=1.0, le=5.0)
    
    # Configuration CORS
    cors_enabled: bool = Field(default=True)
    cors_allowed_origins: List[str] = Field(default_factory=list)
    cors_allowed_methods: List[str] = Field(default=["GET", "POST", "PUT", "DELETE"])
    cors_allowed_headers: List[str] = Field(default=["Content-Type", "Authorization"])
    
    # Configuration CSP
    csp_enabled: bool = Field(default=True)
    csp_directives: Dict[str, str] = Field(default_factory=dict)
    
    @validator('jwt_secret_key')
    def validate_jwt_secret(cls, v):
        if len(v) < 32:
            raise ValueError("JWT secret key must be at least 32 characters")
        return v
    
    @validator('encryption_key')
    def validate_encryption_key(cls, v):
        return SecurityValidationRules.validate_encryption_key(v, "fernet")


class LoggingConfig(BaseSchema):
    """Configuration de logging"""
    
    level: LogLevel = Field(..., description="Niveau de log global")
    
    # Configuration des handlers
    console_enabled: bool = Field(default=True, description="Logging console")
    file_enabled: bool = Field(default=True, description="Logging fichier")
    remote_enabled: bool = Field(default=False, description="Logging distant")
    
    # Configuration fichier
    log_file_path: str = Field(default="/var/log/spotify-ai-agent/app.log")
    log_file_max_size: str = Field(default="100MB", description="Taille max du fichier de log")
    log_file_backup_count: int = Field(default=5, ge=1, le=50)
    log_file_rotation: str = Field(default="daily", description="Rotation des logs")
    
    # Configuration remote logging
    remote_host: Optional[str] = Field(default=None, description="Serveur de logs distant")
    remote_port: Optional[int] = Field(default=None, ge=1, le=65535)
    remote_protocol: str = Field(default="tcp", description="Protocole pour logs distants")
    
    # Configuration par logger
    logger_levels: Dict[str, LogLevel] = Field(
        default_factory=dict,
        description="Niveaux spécifiques par logger"
    )
    
    # Configuration de formatage
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Format des logs"
    )
    date_format: str = Field(
        default="%Y-%m-%d %H:%M:%S",
        description="Format des dates"
    )
    
    # Configuration de filtrage
    sensitive_fields: List[str] = Field(
        default=["password", "token", "secret", "key"],
        description="Champs sensibles à masquer"
    )
    
    # Configuration de performance
    async_logging: bool = Field(default=True, description="Logging asynchrone")
    buffer_size: int = Field(default=1000, ge=100, le=10000)
    flush_interval: int = Field(default=5, ge=1, le=60, description="Intervalle de flush en secondes")


class MonitoringConfig(BaseSchema):
    """Configuration de monitoring"""
    
    enabled: bool = Field(default=True, description="Activer le monitoring")
    provider: MonitoringProvider = Field(..., description="Fournisseur de monitoring")
    
    # Configuration métriques
    metrics_enabled: bool = Field(default=True)
    metrics_port: int = Field(default=8090, ge=1024, le=65535)
    metrics_path: str = Field(default="/metrics")
    
    # Configuration health checks
    health_check_enabled: bool = Field(default=True)
    health_check_path: str = Field(default="/health")
    health_check_interval: int = Field(default=30, ge=5, le=300, description="Intervalle en secondes")
    
    # Configuration alerting
    alerting_enabled: bool = Field(default=True)
    alert_endpoints: List[str] = Field(default_factory=list)
    alert_thresholds: Dict[str, float] = Field(default_factory=dict)
    
    # Configuration tracing
    tracing_enabled: bool = Field(default=True)
    tracing_sample_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    tracing_endpoint: Optional[str] = Field(default=None)
    
    # Configuration des dashboards
    dashboard_enabled: bool = Field(default=True)
    dashboard_port: int = Field(default=3000, ge=1024, le=65535)
    dashboard_auth_enabled: bool = Field(default=True)
    
    # Configuration spécifique par provider
    provider_config: Dict[str, Any] = Field(default_factory=dict)


class PerformanceConfig(BaseSchema):
    """Configuration de performance"""
    
    # Configuration des workers
    worker_count: int = Field(default=4, ge=1, le=32, description="Nombre de workers")
    worker_timeout: int = Field(default=30, ge=5, le=300, description="Timeout worker en secondes")
    worker_memory_limit: str = Field(default="512MB", description="Limite mémoire par worker")
    
    # Configuration de threading
    thread_pool_size: int = Field(default=10, ge=1, le=100)
    max_concurrent_requests: int = Field(default=1000, ge=10, le=10000)
    
    # Configuration de cache
    cache_enabled: bool = Field(default=True)
    cache_default_ttl: int = Field(default=300, ge=60, le=3600)
    cache_max_size: int = Field(default=1000, ge=100, le=100000)
    
    # Configuration de compression
    compression_enabled: bool = Field(default=True)
    compression_level: int = Field(default=6, ge=1, le=9)
    compression_min_size: int = Field(default=1024, ge=100, le=10240)
    
    # Configuration de pagination
    default_page_size: int = Field(default=20, ge=1, le=1000)
    max_page_size: int = Field(default=1000, ge=10, le=10000)
    
    # Configuration des timeouts
    request_timeout: int = Field(default=30, ge=1, le=300)
    database_timeout: int = Field(default=30, ge=1, le=300)
    cache_timeout: int = Field(default=5, ge=1, le=30)
    external_api_timeout: int = Field(default=10, ge=1, le=120)
    
    # Configuration de retry
    retry_enabled: bool = Field(default=True)
    retry_max_attempts: int = Field(default=3, ge=1, le=10)
    retry_backoff_factor: float = Field(default=2.0, ge=1.0, le=10.0)


class EnvironmentConfig(BaseSchema, TimestampMixin, TenantMixin):
    """Configuration complète d'environnement"""
    
    environment: Environment = Field(..., description="Type d'environnement")
    version: str = Field(..., description="Version de la configuration")
    
    # Configurations des composants
    database: DatabaseConfig = Field(..., description="Configuration base de données")
    cache: CacheConfig = Field(..., description="Configuration cache")
    security: SecurityConfig = Field(..., description="Configuration sécurité")
    logging: LoggingConfig = Field(..., description="Configuration logging")
    monitoring: MonitoringConfig = Field(..., description="Configuration monitoring")
    performance: PerformanceConfig = Field(..., description="Configuration performance")
    
    # Configuration des features
    feature_flags: Dict[str, bool] = Field(
        default_factory=dict,
        description="Feature flags par fonctionnalité"
    )
    
    # Configuration des intégrations externes
    external_apis: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Configuration des APIs externes"
    )
    
    # Configuration des tâches
    scheduled_tasks: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Configuration des tâches planifiées"
    )
    
    # Métadonnées
    description: Optional[str] = Field(default=None, description="Description de la configuration")
    maintainer: Optional[str] = Field(default=None, description="Responsable de la configuration")
    
    # Validation et contraintes
    validation_enabled: bool = Field(default=True, description="Activer la validation")
    strict_mode: bool = Field(default=False, description="Mode strict")
    
    @validator('version')
    def validate_version(cls, v):
        # Validation du format de version sémantique
        if not ValidationRules.VERSION_PATTERN.match(v):
            raise ValueError("Version must follow semantic versioning (e.g., 1.2.3)")
        return v
    
    @root_validator
    def validate_environment_consistency(cls, values):
        env = values.get('environment')
        security = values.get('security')
        monitoring = values.get('monitoring')
        
        # Validation selon l'environnement
        if env == Environment.PRODUCTION:
            if security and security.level != SecurityLevel.HIGH:
                raise ValueError("Production environment requires HIGH security level")
            
            if monitoring and not monitoring.enabled:
                raise ValueError("Production environment requires monitoring enabled")
        
        elif env == Environment.DEVELOPMENT:
            # En dev, on peut être plus permissif
            pass
        
        return values


class ConfigurationTemplate(BaseSchema):
    """Template de configuration réutilisable"""
    
    name: str = Field(..., description="Nom du template")
    description: str = Field(..., description="Description du template")
    category: str = Field(..., description="Catégorie du template")
    version: str = Field(..., description="Version du template")
    
    # Template de base
    base_config: EnvironmentConfig = Field(..., description="Configuration de base")
    
    # Variables du template
    variables: Dict[str, Any] = Field(
        default_factory=dict,
        description="Variables configurables du template"
    )
    
    # Contraintes et validations
    constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Contraintes de validation"
    )
    
    # Métadonnées du template
    tags: List[str] = Field(default_factory=list, description="Tags du template")
    author: str = Field(..., description="Auteur du template")
    environments: List[Environment] = Field(..., description="Environnements supportés")
    
    # Date de création et mise à jour
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Statut du template
    active: bool = Field(default=True, description="Template actif")
    deprecated: bool = Field(default=False, description="Template déprécié")
    deprecation_date: Optional[datetime] = Field(default=None)
    
    @validator('name')
    def validate_template_name(cls, v):
        if not ValidationRules.TENANT_ID_PATTERN.match(v):
            raise ValueError("Template name must contain only letters, numbers, hyphens and underscores")
        return v
    
    def render(self, variables: Dict[str, Any]) -> EnvironmentConfig:
        """Génère une configuration à partir du template"""
        # Merge des variables par défaut avec celles fournies
        merged_vars = {**self.variables, **variables}
        
        # Ici on pourrait implémenter un moteur de template
        # Pour simplifier, on retourne la config de base
        return self.base_config


class DeploymentConfig(BaseSchema):
    """Configuration de déploiement"""
    
    name: str = Field(..., description="Nom du déploiement")
    environment: Environment = Field(..., description="Environnement cible")
    
    # Configuration de l'infrastructure
    infrastructure: Dict[str, Any] = Field(..., description="Configuration infrastructure")
    
    # Configuration des services
    services: Dict[str, Any] = Field(..., description="Configuration des services")
    
    # Configuration du réseau
    network: Dict[str, Any] = Field(default_factory=dict, description="Configuration réseau")
    
    # Configuration des volumes
    volumes: Dict[str, Any] = Field(default_factory=dict, description="Configuration des volumes")
    
    # Configuration des secrets
    secrets: Dict[str, str] = Field(default_factory=dict, description="Secrets du déploiement")
    
    # Stratégie de déploiement
    deployment_strategy: str = Field(default="rolling", description="Stratégie de déploiement")
    
    # Configuration du rollback
    rollback_enabled: bool = Field(default=True, description="Rollback automatique")
    rollback_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Seuil de succès pour rollback")
    
    # Health checks
    health_check_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Configuration des ressources
    resource_limits: Dict[str, str] = Field(default_factory=dict)
    resource_requests: Dict[str, str] = Field(default_factory=dict)
