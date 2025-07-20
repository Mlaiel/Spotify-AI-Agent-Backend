"""
Configuration Ultra-Avancée pour le Système de Locales
Configuration centralisée pour tous les composants du système
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path


class EnvironmentType(Enum):
    """Types d'environnement"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class CacheBackendType(Enum):
    """Types de backend de cache"""
    MEMORY = "memory"
    REDIS = "redis"
    MEMCACHED = "memcached"
    HYBRID = "hybrid"


class SecurityLevel(Enum):
    """Niveaux de sécurité"""
    BASIC = "basic"
    STANDARD = "standard"
    ENTERPRISE = "enterprise"
    GOVERNMENT = "government"


@dataclass
class DatabaseConfig:
    """Configuration de base de données"""
    url: str = "postgresql://postgres:password@localhost:5432/locales"
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    
    # Configuration avancée
    ssl_mode: str = "prefer"
    connect_timeout: int = 10
    command_timeout: int = 60
    server_settings: Dict[str, str] = field(default_factory=lambda: {
        "application_name": "spotify_ai_agent_locales",
        "timezone": "UTC",
        "statement_timeout": "60s",
        "idle_in_transaction_session_timeout": "300s"
    })


@dataclass
class RedisConfig:
    """Configuration Redis"""
    url: str = "redis://localhost:6379/0"
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    ssl_cert_reqs: Optional[str] = None
    ssl_ca_certs: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    
    # Configuration de pool
    max_connections: int = 50
    retry_on_timeout: bool = True
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    socket_keepalive: bool = True
    socket_keepalive_options: Dict[str, int] = field(default_factory=lambda: {
        1: 1,  # TCP_KEEPIDLE
        2: 3,  # TCP_KEEPINTVL
        3: 5   # TCP_KEEPCNT
    })
    
    # Configuration cluster
    cluster_enabled: bool = False
    cluster_nodes: List[Dict[str, Any]] = field(default_factory=list)
    read_from_replicas: bool = True
    skip_full_coverage_check: bool = False


@dataclass
class CacheConfig:
    """Configuration du cache"""
    backend: CacheBackendType = CacheBackendType.HYBRID
    ttl_default: int = 3600  # 1 heure
    ttl_short: int = 300     # 5 minutes
    ttl_long: int = 86400    # 24 heures
    ttl_permanent: int = 31536000  # 1 an
    
    # Configuration du cache mémoire
    memory_cache_size: int = 1000
    memory_cache_ttl: int = 300
    
    # Configuration Redis
    redis: RedisConfig = field(default_factory=RedisConfig)
    
    # Configuration de compression
    compression_enabled: bool = True
    compression_algorithm: str = "lz4"  # lz4, gzip, zstd
    compression_threshold: int = 1024  # bytes
    
    # Configuration d'éviction
    eviction_policy: str = "lru"  # lru, lfu, random
    max_memory_policy: str = "allkeys-lru"
    
    # Configuration de préchargement
    preload_enabled: bool = True
    preload_popular_keys: bool = True
    preload_patterns: List[str] = field(default_factory=lambda: [
        "locale:*:messages",
        "tenant:*:config",
        "user:*:preferences"
    ])


@dataclass
class SecurityConfig:
    """Configuration de sécurité"""
    level: SecurityLevel = SecurityLevel.ENTERPRISE
    
    # Configuration de chiffrement
    encryption_algorithm: str = "AES-256-GCM"
    key_derivation_function: str = "PBKDF2"
    key_derivation_iterations: int = 100000
    salt_length: int = 32
    
    # Configuration JWT
    jwt_secret_key: str = "your-super-secret-jwt-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 3600  # 1 heure
    jwt_refresh_expiration: int = 604800  # 7 jours
    
    # Configuration d'audit
    audit_enabled: bool = True
    audit_log_level: str = "INFO"
    audit_retention_days: int = 365
    audit_encrypt_logs: bool = True
    
    # Configuration RBAC
    rbac_enabled: bool = True
    default_permissions: List[str] = field(default_factory=lambda: [
        "locale:read",
        "tenant:read:own"
    ])
    admin_permissions: List[str] = field(default_factory=lambda: [
        "locale:*",
        "tenant:*",
        "system:*",
        "audit:*"
    ])
    
    # Configuration de rate limiting
    rate_limiting_enabled: bool = True
    rate_limit_per_minute: int = 1000
    rate_limit_per_hour: int = 10000
    rate_limit_per_day: int = 100000
    
    # Configuration de protection
    csrf_protection: bool = True
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: [
        "https://localhost:3000",
        "https://spotify-ai-agent.com"
    ])
    
    # Configuration de validation
    input_validation_strict: bool = True
    output_sanitization: bool = True
    sql_injection_protection: bool = True
    xss_protection: bool = True


@dataclass
class MonitoringConfig:
    """Configuration du monitoring"""
    enabled: bool = True
    
    # Configuration des métriques
    metrics_enabled: bool = True
    metrics_port: int = 9090
    metrics_path: str = "/metrics"
    metrics_retention: int = 86400  # 24 heures
    
    # Configuration des alertes
    alerts_enabled: bool = True
    alert_manager_url: str = "http://localhost:9093"
    webhook_url: Optional[str] = None
    
    # Configuration du tracing
    tracing_enabled: bool = True
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    sampling_rate: float = 0.1  # 10%
    
    # Configuration des logs
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: Optional[str] = None
    log_rotation: bool = True
    log_max_size: str = "100MB"
    log_retention: int = 30  # jours
    
    # Configuration de santé
    health_check_enabled: bool = True
    health_check_interval: int = 30  # secondes
    health_check_timeout: int = 5   # secondes
    
    # Seuils de performance
    performance_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "response_time_p95": 1000.0,  # ms
        "response_time_p99": 2000.0,  # ms
        "error_rate": 0.01,           # 1%
        "cache_hit_rate": 0.9,        # 90%
        "memory_usage": 0.8,          # 80%
        "cpu_usage": 0.7,             # 70%
        "disk_usage": 0.8             # 80%
    })


@dataclass
class LocaleConfig:
    """Configuration des locales"""
    default_locale: str = "en_US"
    fallback_locale: str = "en_US"
    supported_locales: List[str] = field(default_factory=lambda: [
        "en_US", "en_GB", "fr_FR", "es_ES", "de_DE", "it_IT",
        "pt_BR", "ja_JP", "ko_KR", "zh_CN", "zh_TW", "ru_RU"
    ])
    
    # Configuration de chargement
    auto_reload: bool = True
    reload_interval: int = 300  # 5 minutes
    lazy_loading: bool = True
    preload_locales: List[str] = field(default_factory=lambda: [
        "en_US", "fr_FR", "es_ES"
    ])
    
    # Configuration de validation
    strict_validation: bool = True
    validate_placeholders: bool = True
    validate_html: bool = True
    validate_unicode: bool = True
    
    # Configuration de formatage
    pluralization_enabled: bool = True
    gender_support: bool = True
    currency_formatting: bool = True
    date_formatting: bool = True
    number_formatting: bool = True
    
    # Configuration de traduction
    auto_translation: bool = False
    translation_service: str = "google"  # google, aws, azure, deepl
    translation_cache_ttl: int = 86400   # 24 heures
    quality_threshold: float = 0.8       # Seuil de qualité minimum


@dataclass
class PerformanceConfig:
    """Configuration de performance"""
    # Configuration des pools
    worker_processes: int = 4
    worker_threads: int = 8
    max_requests: int = 1000
    max_requests_jitter: int = 50
    
    # Configuration des timeouts
    request_timeout: int = 30
    keepalive_timeout: int = 5
    client_timeout: int = 60
    
    # Configuration du cache
    cache_size: int = 1000
    cache_ttl: int = 3600
    
    # Configuration de la compression
    compression_enabled: bool = True
    compression_level: int = 6
    compression_minimum_size: int = 1024
    
    # Configuration des connexions
    max_connections: int = 1000
    connection_pool_size: int = 20
    connection_timeout: int = 10
    
    # Configuration des ressources
    memory_limit: int = 1073741824  # 1GB
    cpu_limit: float = 2.0          # 2 CPU cores
    
    # Configuration d'optimisation
    enable_gzip: bool = True
    enable_brotli: bool = True
    enable_http2: bool = True
    enable_caching: bool = True


@dataclass
class AIConfig:
    """Configuration de l'IA"""
    enabled: bool = True
    
    # Configuration des modèles
    translation_model: str = "google/mt5-base"
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment"
    language_detection_model: str = "langdetect"
    
    # Configuration de l'optimisation prédictive
    predictive_caching: bool = True
    prediction_window: int = 3600    # 1 heure
    prediction_confidence: float = 0.8
    
    # Configuration de l'apprentissage
    continuous_learning: bool = True
    feedback_threshold: int = 100    # Nombre de feedbacks pour réentraîner
    model_update_interval: int = 86400  # 24 heures
    
    # Configuration des ressources
    gpu_enabled: bool = False
    max_batch_size: int = 32
    inference_timeout: int = 10
    
    # Configuration de la qualité
    quality_threshold: float = 0.85
    auto_correction: bool = True
    human_review_threshold: float = 0.7


@dataclass
class LocaleSystemConfig:
    """Configuration principale du système de locales"""
    environment: EnvironmentType = EnvironmentType.DEVELOPMENT
    debug: bool = False
    
    # Configurations des composants
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    locale: LocaleConfig = field(default_factory=LocaleConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    
    # Configuration des chemins
    base_path: Path = field(default_factory=lambda: Path("/app/locales"))
    data_path: Path = field(default_factory=lambda: Path("/app/data/locales"))
    cache_path: Path = field(default_factory=lambda: Path("/app/cache/locales"))
    log_path: Path = field(default_factory=lambda: Path("/app/logs"))
    backup_path: Path = field(default_factory=lambda: Path("/app/backups"))
    
    # Configuration des features flags
    feature_flags: Dict[str, bool] = field(default_factory=lambda: {
        "enable_ai_translation": True,
        "enable_predictive_caching": True,
        "enable_auto_scaling": True,
        "enable_blockchain_audit": False,
        "enable_quantum_encryption": False,
        "enable_edge_deployment": False,
        "enable_realtime_sync": True,
        "enable_multi_region": True,
        "enable_cdn_integration": True,
        "enable_mobile_optimization": True
    })
    
    # Configuration des limites
    limits: Dict[str, int] = field(default_factory=lambda: {
        "max_tenants": 10000,
        "max_locales_per_tenant": 100,
        "max_messages_per_locale": 100000,
        "max_file_size": 10485760,  # 10MB
        "max_concurrent_requests": 1000,
        "max_cache_size": 1073741824,  # 1GB
        "max_log_file_size": 104857600,  # 100MB
        "max_backup_retention": 30  # jours
    })
    
    @classmethod
    def from_environment(cls) -> "LocaleSystemConfig":
        """Crée la configuration à partir des variables d'environnement"""
        config = cls()
        
        # Configuration de l'environnement
        env_type = os.getenv("LOCALE_ENV", "development")
        config.environment = EnvironmentType(env_type)
        config.debug = os.getenv("LOCALE_DEBUG", "false").lower() == "true"
        
        # Configuration de la base de données
        if db_url := os.getenv("DATABASE_URL"):
            config.database.url = db_url
        
        # Configuration Redis
        if redis_url := os.getenv("REDIS_URL"):
            config.cache.redis.url = redis_url
        
        # Configuration de sécurité
        if jwt_secret := os.getenv("JWT_SECRET_KEY"):
            config.security.jwt_secret_key = jwt_secret
        
        # Configuration de monitoring
        if jaeger_endpoint := os.getenv("JAEGER_ENDPOINT"):
            config.monitoring.jaeger_endpoint = jaeger_endpoint
        
        # Ajuster la configuration selon l'environnement
        if config.environment == EnvironmentType.PRODUCTION:
            config._apply_production_settings()
        elif config.environment == EnvironmentType.DEVELOPMENT:
            config._apply_development_settings()
        
        return config
    
    def _apply_production_settings(self):
        """Applique les paramètres de production"""
        self.debug = False
        self.security.level = SecurityLevel.ENTERPRISE
        self.security.audit_enabled = True
        self.monitoring.tracing_enabled = True
        self.monitoring.sampling_rate = 0.01  # 1% en production
        self.performance.worker_processes = 8
        self.cache.ttl_default = 7200  # 2 heures
        self.ai.enabled = True
        self.ai.predictive_caching = True
    
    def _apply_development_settings(self):
        """Applique les paramètres de développement"""
        self.debug = True
        self.security.level = SecurityLevel.BASIC
        self.security.audit_enabled = False
        self.monitoring.tracing_enabled = False
        self.monitoring.sampling_rate = 1.0  # 100% en développement
        self.performance.worker_processes = 2
        self.cache.ttl_default = 300  # 5 minutes
        self.ai.enabled = False  # Désactivé par défaut en dev
    
    def validate(self) -> List[str]:
        """Valide la configuration"""
        errors = []
        
        # Validation des chemins
        required_paths = [
            self.base_path,
            self.data_path,
            self.cache_path,
            self.log_path,
            self.backup_path
        ]
        
        for path in required_paths:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create path {path}: {e}")
        
        # Validation des URLs
        if not self.database.url:
            errors.append("Database URL is required")
        
        if not self.cache.redis.url:
            errors.append("Redis URL is required")
        
        # Validation de sécurité
        if self.environment == EnvironmentType.PRODUCTION:
            if self.security.jwt_secret_key == "your-super-secret-jwt-key-change-in-production":
                errors.append("JWT secret key must be changed in production")
            
            if not self.security.audit_enabled:
                errors.append("Audit must be enabled in production")
        
        # Validation des limites
        if self.limits["max_tenants"] <= 0:
            errors.append("max_tenants must be positive")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit la configuration en dictionnaire"""
        from dataclasses import asdict
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LocaleSystemConfig":
        """Crée la configuration à partir d'un dictionnaire"""
        return cls(**data)


# Configuration globale par défaut
DEFAULT_CONFIG = LocaleSystemConfig.from_environment()

# Fonctions utilitaires
def get_config() -> LocaleSystemConfig:
    """Retourne la configuration actuelle"""
    return DEFAULT_CONFIG

def set_config(config: LocaleSystemConfig):
    """Définit la configuration globale"""
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = config

def load_config_from_file(file_path: str) -> LocaleSystemConfig:
    """Charge la configuration depuis un fichier"""
    import json
    import yaml
    
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")
    
    if path.suffix == ".json":
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif path.suffix in [".yaml", ".yml"]:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config file format: {path.suffix}")
    
    return LocaleSystemConfig.from_dict(data)

def save_config_to_file(config: LocaleSystemConfig, file_path: str):
    """Sauvegarde la configuration dans un fichier"""
    import json
    import yaml
    
    path = Path(file_path)
    data = config.to_dict()
    
    if path.suffix == ".json":
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    elif path.suffix in [".yaml", ".yml"]:
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    else:
        raise ValueError(f"Unsupported config file format: {path.suffix}")
