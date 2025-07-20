"""
Configuration centralisée pour le système de notifications
=========================================================

Configuration ultra-avancée avec validation, hot-reload,
et support multi-environnement.
"""

import os
import json
import yaml
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
from datetime import timedelta

from pydantic import BaseSettings, Field, validator, SecretStr
from pydantic.env_settings import SettingsSourceCallable


class Environment(str, Enum):
    """Environnements supportés"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Niveaux de log"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class DatabaseConfig:
    """Configuration de base de données"""
    url: str
    pool_size: int = 20
    max_overflow: int = 40
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    ssl_ca_path: Optional[str] = None


@dataclass
class RedisConfig:
    """Configuration Redis"""
    url: str = "redis://localhost:6379"
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    ssl_cert_reqs: Optional[str] = None
    ssl_ca_certs: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    connection_pool_max_connections: int = 100
    retry_on_timeout: bool = True
    decode_responses: bool = True


@dataclass
class SlackConfig:
    """Configuration Slack"""
    enabled: bool = True
    bot_token: Optional[SecretStr] = None
    user_token: Optional[SecretStr] = None
    signing_secret: Optional[SecretStr] = None
    bot_name: str = "Spotify AI Agent"
    default_channel: str = "#notifications"
    rate_limit_per_minute: int = 60
    retry_attempts: int = 3
    timeout: int = 30
    unfurl_links: bool = True
    unfurl_media: bool = True


@dataclass
class EmailConfig:
    """Configuration Email"""
    enabled: bool = True
    smtp_host: str = "localhost"
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[SecretStr] = None
    use_tls: bool = True
    use_ssl: bool = False
    from_email: str = "noreply@spotify-ai-agent.com"
    from_name: str = "Spotify AI Agent"
    reply_to: Optional[str] = None
    rate_limit_per_minute: int = 100
    timeout: int = 30
    max_attachment_size: int = 25 * 1024 * 1024  # 25MB


@dataclass
class SMSConfig:
    """Configuration SMS (Twilio)"""
    enabled: bool = False
    twilio_account_sid: Optional[SecretStr] = None
    twilio_auth_token: Optional[SecretStr] = None
    from_phone: Optional[str] = None
    rate_limit_per_minute: int = 10
    timeout: int = 30


@dataclass
class PushConfig:
    """Configuration Push (Firebase)"""
    enabled: bool = False
    firebase_credentials_path: Optional[str] = None
    rate_limit_per_minute: int = 1000
    timeout: int = 30


@dataclass
class WebhookConfig:
    """Configuration Webhook"""
    enabled: bool = True
    default_timeout: int = 30
    max_retries: int = 3
    rate_limit_per_minute: int = 100
    verify_ssl: bool = True


@dataclass
class QueueConfig:
    """Configuration Queue"""
    default_priority: int = 5
    max_retries: int = 3
    retry_backoff_base: float = 2.0
    retry_backoff_max: float = 300.0  # 5 minutes
    batch_size: int = 100
    processing_timeout: int = 300  # 5 minutes
    cleanup_interval: int = 3600  # 1 hour
    dead_letter_queue_enabled: bool = True


@dataclass
class SecurityConfig:
    """Configuration sécurité"""
    secret_key: SecretStr
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    api_key_header: str = "X-API-Key"
    rate_limiting_enabled: bool = True
    cors_origins: List[str] = field(default_factory=list)
    allowed_hosts: List[str] = field(default_factory=list)
    encryption_key: Optional[SecretStr] = None


@dataclass
class MonitoringConfig:
    """Configuration monitoring"""
    enabled: bool = True
    prometheus_enabled: bool = True
    prometheus_port: int = 8000
    jaeger_enabled: bool = False
    jaeger_endpoint: Optional[str] = None
    sentry_enabled: bool = False
    sentry_dsn: Optional[SecretStr] = None
    health_check_interval: int = 30
    metrics_retention_days: int = 90


@dataclass
class CacheConfig:
    """Configuration cache"""
    template_cache_ttl: int = 3600  # 1 hour
    metrics_cache_ttl: int = 300   # 5 minutes
    user_preferences_cache_ttl: int = 1800  # 30 minutes
    rate_limit_cache_ttl: int = 60  # 1 minute
    max_cache_size: int = 10000


@dataclass
class MLConfig:
    """Configuration Machine Learning"""
    enabled: bool = True
    anomaly_detection_enabled: bool = True
    anomaly_model_path: Optional[str] = None
    clustering_enabled: bool = True
    clustering_model_path: Optional[str] = None
    prediction_enabled: bool = False
    model_retrain_interval_hours: int = 24
    feature_store_enabled: bool = False


@dataclass
class PerformanceConfig:
    """Configuration performance"""
    worker_processes: int = 4
    max_concurrent_notifications: int = 1000
    connection_timeout: float = 30.0
    read_timeout: float = 30.0
    write_timeout: float = 30.0
    keepalive_timeout: float = 2.0
    graceful_timeout: float = 30.0
    preload_app: bool = True


class NotificationSettings(BaseSettings):
    """Configuration principale du système de notifications"""
    
    # Environnement
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    log_level: LogLevel = LogLevel.INFO
    
    # Base de données
    database_url: str = Field(..., env="DATABASE_URL")
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    # Redis
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    redis: RedisConfig = field(default_factory=RedisConfig)
    
    # Channels
    slack: SlackConfig = field(default_factory=SlackConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    sms: SMSConfig = field(default_factory=SMSConfig)
    push: PushConfig = field(default_factory=PushConfig)
    webhook: WebhookConfig = field(default_factory=WebhookConfig)
    
    # Queue
    queue: QueueConfig = field(default_factory=QueueConfig)
    
    # Sécurité
    security: SecurityConfig = field(default_factory=lambda: SecurityConfig(secret_key=SecretStr("change-me")))
    
    # Monitoring
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Cache
    cache: CacheConfig = field(default_factory=CacheConfig)
    
    # Machine Learning
    ml: MLConfig = field(default_factory=MLConfig)
    
    # Performance
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Directories
    template_dir: str = "templates"
    fallback_templates_dir: str = "templates/fallback"
    upload_dir: str = "uploads"
    logs_dir: str = "logs"
    
    # Features flags
    features: Dict[str, bool] = field(default_factory=lambda: {
        "ab_testing": True,
        "smart_batching": True,
        "ai_filtering": True,
        "sentiment_analysis": False,
        "auto_escalation": True,
        "predictive_delivery": False,
        "advanced_analytics": True,
        "real_time_metrics": True,
        "circuit_breaker": True,
        "distributed_tracing": True
    })
    
    # Rate limiting global
    global_rate_limits: Dict[str, int] = field(default_factory=lambda: {
        "per_tenant_per_minute": 1000,
        "per_tenant_per_hour": 10000,
        "per_tenant_per_day": 100000,
        "per_user_per_minute": 60,
        "per_user_per_hour": 600
    })
    
    # Alerting thresholds
    alert_thresholds: Dict[str, Union[int, float]] = field(default_factory=lambda: {
        "error_rate_warning": 0.05,     # 5%
        "error_rate_critical": 0.10,    # 10%
        "delivery_time_warning": 10000,  # 10s
        "delivery_time_critical": 30000, # 30s
        "queue_size_warning": 5000,
        "queue_size_critical": 10000,
        "memory_usage_warning": 0.80,   # 80%
        "memory_usage_critical": 0.90,  # 90%
        "cpu_usage_warning": 0.70,      # 70%
        "cpu_usage_critical": 0.85      # 85%
    })
    
    # Tenant limits
    tenant_limits: Dict[str, int] = field(default_factory=lambda: {
        "max_notifications_per_hour": 10000,
        "max_templates": 100,
        "max_rules": 50,
        "max_channels": 10,
        "max_attachment_size": 10 * 1024 * 1024,  # 10MB
        "max_recipients_per_notification": 1000
    })
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        @classmethod
        def customise_sources(
            cls,
            init_settings: SettingsSourceCallable,
            env_settings: SettingsSourceCallable,
            file_secret_settings: SettingsSourceCallable,
        ) -> tuple[SettingsSourceCallable, ...]:
            return (
                init_settings,
                yaml_config_settings,
                env_settings,
                file_secret_settings,
            )
    
    @validator('environment')
    def validate_environment(cls, v):
        """Valider l'environnement"""
        if v not in Environment:
            raise ValueError(f"Environment must be one of {list(Environment)}")
        return v
    
    @validator('database_url')
    def validate_database_url(cls, v):
        """Valider l'URL de base de données"""
        if not v.startswith(('postgresql://', 'postgresql+asyncpg://', 'sqlite:///')):
            raise ValueError("Database URL must be PostgreSQL or SQLite")
        return v
    
    @validator('redis_url')
    def validate_redis_url(cls, v):
        """Valider l'URL Redis"""
        if not v.startswith(('redis://', 'rediss://')):
            raise ValueError("Redis URL must start with redis:// or rediss://")
        return v
    
    def get_channel_config(self, channel_type: str) -> Optional[Dict[str, Any]]:
        """Obtenir la configuration d'un canal"""
        channel_configs = {
            'slack': self.slack,
            'email': self.email,
            'sms': self.sms,
            'push': self.push,
            'webhook': self.webhook
        }
        
        config = channel_configs.get(channel_type.lower())
        if config:
            return config.__dict__
        return None
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Vérifier si une fonctionnalité est activée"""
        return self.features.get(feature_name, False)
    
    def get_rate_limit(self, limit_type: str) -> Optional[int]:
        """Obtenir une limite de taux"""
        return self.global_rate_limits.get(limit_type)
    
    def get_alert_threshold(self, threshold_name: str) -> Optional[Union[int, float]]:
        """Obtenir un seuil d'alerte"""
        return self.alert_thresholds.get(threshold_name)
    
    def get_tenant_limit(self, limit_type: str) -> Optional[int]:
        """Obtenir une limite de tenant"""
        return self.tenant_limits.get(limit_type)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir en dictionnaire"""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if hasattr(field_value, '__dict__'):
                # Objet avec attributs
                result[field_name] = field_value.__dict__
            elif isinstance(field_value, SecretStr):
                # Secret
                result[field_name] = '***'
            else:
                result[field_name] = field_value
        return result


def yaml_config_settings(settings: BaseSettings) -> Dict[str, Any]:
    """Charger la configuration depuis un fichier YAML"""
    
    config_files = [
        "config/notification.yaml",
        "config/notification.yml",
        f"config/notification.{settings.environment.value}.yaml",
        f"config/notification.{settings.environment.value}.yml",
        "notification.yaml",
        "notification.yml"
    ]
    
    for config_file in config_files:
        config_path = Path(config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                    if config_data:
                        return config_data
            except Exception as e:
                logging.warning(f"Erreur lecture fichier config {config_file}: {e}")
    
    return {}


class ConfigManager:
    """Gestionnaire de configuration avec hot-reload"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self._settings: Optional[NotificationSettings] = None
        self._last_modified = 0
        self._watchers: List[Callable] = []
    
    def get_settings(self) -> NotificationSettings:
        """Obtenir la configuration actuelle"""
        if self._should_reload():
            self._reload_config()
        
        if self._settings is None:
            self._settings = NotificationSettings()
        
        return self._settings
    
    def _should_reload(self) -> bool:
        """Vérifier si la configuration doit être rechargée"""
        if not self.config_file:
            return False
        
        config_path = Path(self.config_file)
        if not config_path.exists():
            return False
        
        current_modified = config_path.stat().st_mtime
        if current_modified > self._last_modified:
            self._last_modified = current_modified
            return True
        
        return False
    
    def _reload_config(self):
        """Recharger la configuration"""
        try:
            old_settings = self._settings
            self._settings = NotificationSettings()
            
            # Notifier les watchers du changement
            for watcher in self._watchers:
                try:
                    watcher(old_settings, self._settings)
                except Exception as e:
                    logging.error(f"Erreur dans watcher de config: {e}")
        
        except Exception as e:
            logging.error(f"Erreur rechargement config: {e}")
    
    def add_config_watcher(self, callback: Callable):
        """Ajouter un watcher pour les changements de config"""
        self._watchers.append(callback)
    
    def remove_config_watcher(self, callback: Callable):
        """Supprimer un watcher"""
        if callback in self._watchers:
            self._watchers.remove(callback)


# Instance globale du gestionnaire de configuration
config_manager = ConfigManager()


def get_settings() -> NotificationSettings:
    """Obtenir la configuration actuelle"""
    return config_manager.get_settings()


def create_config_from_env() -> NotificationSettings:
    """Créer la configuration depuis les variables d'environnement"""
    
    # Configuration de base depuis l'environnement
    config = NotificationSettings()
    
    # Surcharger avec les variables d'environnement spécifiques
    
    # Database
    if os.getenv('DB_POOL_SIZE'):
        config.database.pool_size = int(os.getenv('DB_POOL_SIZE'))
    if os.getenv('DB_MAX_OVERFLOW'):
        config.database.max_overflow = int(os.getenv('DB_MAX_OVERFLOW'))
    
    # Redis
    if os.getenv('REDIS_DB'):
        config.redis.db = int(os.getenv('REDIS_DB'))
    if os.getenv('REDIS_PASSWORD'):
        config.redis.password = os.getenv('REDIS_PASSWORD')
    
    # Slack
    if os.getenv('SLACK_BOT_TOKEN'):
        config.slack.bot_token = SecretStr(os.getenv('SLACK_BOT_TOKEN'))
    if os.getenv('SLACK_SIGNING_SECRET'):
        config.slack.signing_secret = SecretStr(os.getenv('SLACK_SIGNING_SECRET'))
    
    # Email
    if os.getenv('SMTP_HOST'):
        config.email.smtp_host = os.getenv('SMTP_HOST')
    if os.getenv('SMTP_PORT'):
        config.email.smtp_port = int(os.getenv('SMTP_PORT'))
    if os.getenv('SMTP_USERNAME'):
        config.email.smtp_username = os.getenv('SMTP_USERNAME')
    if os.getenv('SMTP_PASSWORD'):
        config.email.smtp_password = SecretStr(os.getenv('SMTP_PASSWORD'))
    
    # SMS
    if os.getenv('TWILIO_ACCOUNT_SID'):
        config.sms.twilio_account_sid = SecretStr(os.getenv('TWILIO_ACCOUNT_SID'))
    if os.getenv('TWILIO_AUTH_TOKEN'):
        config.sms.twilio_auth_token = SecretStr(os.getenv('TWILIO_AUTH_TOKEN'))
    if os.getenv('TWILIO_FROM_PHONE'):
        config.sms.from_phone = os.getenv('TWILIO_FROM_PHONE')
    
    # Push
    if os.getenv('FIREBASE_CREDENTIALS_PATH'):
        config.push.firebase_credentials_path = os.getenv('FIREBASE_CREDENTIALS_PATH')
    
    # Security
    if os.getenv('SECRET_KEY'):
        config.security.secret_key = SecretStr(os.getenv('SECRET_KEY'))
    if os.getenv('ENCRYPTION_KEY'):
        config.security.encryption_key = SecretStr(os.getenv('ENCRYPTION_KEY'))
    
    # Monitoring
    if os.getenv('SENTRY_DSN'):
        config.monitoring.sentry_dsn = SecretStr(os.getenv('SENTRY_DSN'))
        config.monitoring.sentry_enabled = True
    
    return config


def validate_config(settings: NotificationSettings) -> List[str]:
    """Valider la configuration et retourner les erreurs"""
    errors = []
    
    # Vérifier les configurations requises
    if not settings.database_url:
        errors.append("DATABASE_URL est requis")
    
    if not settings.security.secret_key:
        errors.append("SECRET_KEY est requis")
    
    # Vérifier les configurations des canaux activés
    if settings.slack.enabled and not settings.slack.bot_token:
        errors.append("SLACK_BOT_TOKEN est requis si Slack est activé")
    
    if settings.email.enabled and not all([
        settings.email.smtp_host,
        settings.email.smtp_username,
        settings.email.smtp_password
    ]):
        errors.append("Configuration SMTP complète requise si Email est activé")
    
    if settings.sms.enabled and not all([
        settings.sms.twilio_account_sid,
        settings.sms.twilio_auth_token,
        settings.sms.from_phone
    ]):
        errors.append("Configuration Twilio complète requise si SMS est activé")
    
    if settings.push.enabled and not settings.push.firebase_credentials_path:
        errors.append("Firebase credentials requis si Push est activé")
    
    # Vérifier les seuils
    for threshold_name, threshold_value in settings.alert_thresholds.items():
        if isinstance(threshold_value, float) and not 0 <= threshold_value <= 1:
            if 'rate' in threshold_name:
                errors.append(f"Seuil {threshold_name} doit être entre 0 et 1")
    
    # Vérifier les limites
    for limit_name, limit_value in settings.tenant_limits.items():
        if limit_value <= 0:
            errors.append(f"Limite {limit_name} doit être positive")
    
    return errors


# Configuration par défaut pour les tests
TEST_SETTINGS = NotificationSettings(
    environment=Environment.TESTING,
    debug=True,
    database_url="sqlite:///:memory:",
    redis_url="redis://localhost:6379/15",  # Base dédiée aux tests
    security=SecurityConfig(secret_key=SecretStr("test-secret-key")),
    monitoring=MonitoringConfig(enabled=False),
    features={
        "ab_testing": False,
        "smart_batching": False,
        "ai_filtering": False,
        "sentiment_analysis": False,
        "auto_escalation": False,
        "predictive_delivery": False,
        "advanced_analytics": False,
        "real_time_metrics": False,
        "circuit_breaker": False,
        "distributed_tracing": False
    }
)
