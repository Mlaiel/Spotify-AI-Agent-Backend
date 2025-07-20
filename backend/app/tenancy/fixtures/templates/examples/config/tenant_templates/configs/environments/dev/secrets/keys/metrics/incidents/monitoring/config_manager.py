# =============================================================================
# Monitoring Configuration Manager - Enterprise
# =============================================================================
# 
# Gestionnaire de configuration centralisé pour le système de monitoring
# enterprise avec validation, chiffrement et gestion des environnements.
#
# Développé par l'équipe d'experts techniques:
# - Lead Developer + AI Architect (Architecture de configuration)
# - Backend Senior Developer (Python/FastAPI/Django)
# - Spécialiste Sécurité Backend (Chiffrement et validation)
# - DevOps Senior Engineer (Gestion des environnements)
#
# Direction Technique: Fahed Mlaiel
# =============================================================================

import os
import sys
import json
import yaml
import base64
import hashlib
import secrets
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import structlog
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import pydantic
from pydantic import BaseModel, validator, Field

logger = structlog.get_logger(__name__)

# =============================================================================
# MODÈLES DE CONFIGURATION
# =============================================================================

class Environment(Enum):
    """Environnements supportés"""
    DEVELOPMENT = "dev"
    STAGING = "staging"
    PRODUCTION = "prod"
    TEST = "test"

class ConfigurationLevel(Enum):
    """Niveaux de configuration"""
    SYSTEM = "system"
    TENANT = "tenant"
    USER = "user"
    TEMPORARY = "temporary"

@dataclass
class EncryptionConfig:
    """Configuration du chiffrement"""
    algorithm: str = "AES-256-GCM"
    key_rotation_days: int = 90
    encrypted_fields: List[str] = field(default_factory=lambda: [
        "database_password", "api_keys", "secrets", "tokens"
    ])

class PrometheusConfig(BaseModel):
    """Configuration Prometheus validée"""
    host: str = Field(default="localhost", min_length=1)
    port: int = Field(default=9090, ge=1, le=65535)
    scrape_interval: str = Field(default="15s", regex=r'^\d+[smh]$')
    evaluation_interval: str = Field(default="15s", regex=r'^\d+[smh]$')
    retention_time: str = Field(default="30d", regex=r'^\d+[dwmy]$')
    storage_path: str = Field(default="/prometheus/data")
    
    # Configuration avancée
    max_samples_per_send: int = Field(default=10000, ge=1000)
    remote_timeout: str = Field(default="30s", regex=r'^\d+[smh]$')
    external_labels: Dict[str, str] = Field(default_factory=dict)
    
    # Configuration de sécurité
    basic_auth_username: Optional[str] = None
    basic_auth_password: Optional[str] = None
    bearer_token: Optional[str] = None
    tls_config: Dict[str, Any] = Field(default_factory=dict)

class GrafanaConfig(BaseModel):
    """Configuration Grafana validée"""
    host: str = Field(default="localhost", min_length=1)
    port: int = Field(default=3000, ge=1, le=65535)
    protocol: str = Field(default="http", regex=r'^https?$')
    admin_user: str = Field(default="admin", min_length=1)
    admin_password: str = Field(min_length=8)
    api_key: Optional[str] = None
    org_id: int = Field(default=1, ge=1)
    
    # Configuration des datasources
    datasources: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Configuration des dashboards
    dashboards_path: str = Field(default="/var/lib/grafana/dashboards")
    provisioning_path: str = Field(default="/etc/grafana/provisioning")
    
    # Plugins
    plugins: List[str] = Field(default_factory=lambda: [
        "grafana-piechart-panel",
        "grafana-clock-panel",
        "grafana-worldmap-panel"
    ])
    
    # Configuration SMTP pour alerting
    smtp_host: Optional[str] = None
    smtp_port: Optional[int] = Field(None, ge=1, le=65535)
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None

class AlertingConfig(BaseModel):
    """Configuration d'alerting validée"""
    # Configuration SMTP
    smtp_host: str = Field(default="localhost")
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_from: str = Field(default="alerts@monitoring.local")
    smtp_tls: bool = Field(default=True)
    
    # Configuration Slack
    slack_webhook_url: Optional[str] = Field(None, regex=r'^https://hooks\.slack\.com/')
    slack_channel: str = Field(default="#alerts")
    slack_username: str = Field(default="MonitoringBot")
    slack_icon_emoji: str = Field(default=":warning:")
    
    # Configuration PagerDuty
    pagerduty_integration_key: Optional[str] = None
    pagerduty_severity_mapping: Dict[str, str] = Field(default_factory=lambda: {
        "critical": "critical",
        "warning": "warning",
        "info": "info"
    })
    
    # Configuration webhook générique
    webhook_url: Optional[str] = None
    webhook_headers: Dict[str, str] = Field(default_factory=dict)
    webhook_timeout: int = Field(default=30, ge=1, le=300)
    
    # Configuration Microsoft Teams
    teams_webhook_url: Optional[str] = None
    
    # Configuration des règles d'escalade
    escalation_rules: List[Dict[str, Any]] = Field(default_factory=list)

class SecurityConfig(BaseModel):
    """Configuration de sécurité validée"""
    encryption_enabled: bool = Field(default=True)
    encryption_key: Optional[str] = None
    jwt_secret: str = Field(min_length=32)
    jwt_expiration_hours: int = Field(default=24, ge=1, le=168)
    
    # Configuration RBAC
    rbac_enabled: bool = Field(default=True)
    default_role: str = Field(default="viewer")
    
    # Configuration de l'audit
    audit_logging: bool = Field(default=True)
    audit_retention_days: int = Field(default=90, ge=1)
    
    # Configuration rate limiting
    rate_limiting_enabled: bool = Field(default=True)
    requests_per_minute: int = Field(default=100, ge=1)
    
    # Configuration IP whitelist
    ip_whitelist: List[str] = Field(default_factory=list)
    
    # Configuration CORS
    cors_enabled: bool = Field(default=True)
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])

class MLConfig(BaseModel):
    """Configuration Machine Learning validée"""
    # Détection d'anomalies
    anomaly_detection_enabled: bool = Field(default=True)
    anomaly_model_type: str = Field(default="isolation_forest", regex=r'^(isolation_forest|one_class_svm|local_outlier_factor)$')
    anomaly_sensitivity: float = Field(default=0.1, ge=0.01, le=1.0)
    anomaly_training_window: str = Field(default="7d", regex=r'^\d+[dwmy]$')
    
    # Prédiction et scaling
    predictive_scaling_enabled: bool = Field(default=True)
    prediction_horizon: str = Field(default="1h", regex=r'^\d+[smh]$')
    prediction_confidence: float = Field(default=0.95, ge=0.5, le=0.99)
    
    # Reconnaissance de patterns
    pattern_recognition_enabled: bool = Field(default=True)
    min_pattern_length: int = Field(default=10, ge=5)
    pattern_similarity_threshold: float = Field(default=0.8, ge=0.5, le=1.0)
    
    # Configuration des modèles
    model_update_frequency: str = Field(default="1d", regex=r'^\d+[dwmy]$')
    model_validation_split: float = Field(default=0.2, ge=0.1, le=0.5)

class DatabaseConfig(BaseModel):
    """Configuration base de données validée"""
    # PostgreSQL pour métriques
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432, ge=1, le=65535)
    postgres_database: str = Field(default="monitoring")
    postgres_username: str = Field(default="monitoring_user")
    postgres_password: str = Field(min_length=8)
    postgres_ssl_mode: str = Field(default="prefer", regex=r'^(disable|allow|prefer|require|verify-ca|verify-full)$')
    
    # Redis pour cache et sessions
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379, ge=1, le=65535)
    redis_password: Optional[str] = None
    redis_database: int = Field(default=0, ge=0, le=15)
    
    # InfluxDB pour time series (optionnel)
    influxdb_enabled: bool = Field(default=False)
    influxdb_host: str = Field(default="localhost")
    influxdb_port: int = Field(default=8086, ge=1, le=65535)
    influxdb_database: str = Field(default="monitoring")
    influxdb_username: Optional[str] = None
    influxdb_password: Optional[str] = None

# =============================================================================
# GESTIONNAIRE DE CONFIGURATION PRINCIPAL
# =============================================================================

class ConfigurationManager:
    """
    Gestionnaire centralisé de configuration avec validation,
    chiffrement et gestion des environnements.
    """
    
    def __init__(self, environment: Environment = Environment.DEVELOPMENT):
        self.environment = environment
        self.config_dir = Path(__file__).parent / "configs"
        self.secrets_dir = self.config_dir / "secrets"
        
        # Configuration de chiffrement
        self._encryption_key: Optional[bytes] = None
        self._cipher_suite: Optional[Fernet] = None
        
        # Cache de configuration
        self._config_cache: Dict[str, Any] = {}
        self._config_timestamps: Dict[str, float] = {}
        
        # Validation automatique
        self._validators = {
            'prometheus': PrometheusConfig,
            'grafana': GrafanaConfig,
            'alerting': AlertingConfig,
            'security': SecurityConfig,
            'ml': MLConfig,
            'database': DatabaseConfig
        }
        
        self._ensure_directories()
        logger.info(f"ConfigurationManager initialisé pour {environment.value}")

    def _ensure_directories(self):
        """Création des répertoires de configuration"""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.secrets_dir.mkdir(parents=True, exist_ok=True)
        
        # Permissions sécurisées pour les secrets
        if os.name != 'nt':  # Unix/Linux
            os.chmod(self.secrets_dir, 0o700)

    def _initialize_encryption(self, password: Optional[str] = None) -> bytes:
        """Initialisation du système de chiffrement"""
        
        if password is None:
            password = os.getenv('MONITORING_ENCRYPTION_PASSWORD')
            if not password:
                # Génération d'un mot de passe automatique en développement
                if self.environment == Environment.DEVELOPMENT:
                    password = "dev-monitoring-key-2025"
                else:
                    raise ValueError("Mot de passe de chiffrement requis en production")
        
        # Dérivation de clé avec PBKDF2
        salt = b'monitoring-salt-2025'  # En production, utiliser un salt aléatoire
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        self._encryption_key = key
        self._cipher_suite = Fernet(key)
        
        return key

    def encrypt_value(self, value: str) -> str:
        """Chiffrement d'une valeur"""
        if not self._cipher_suite:
            self._initialize_encryption()
        
        encrypted = self._cipher_suite.encrypt(value.encode())
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt_value(self, encrypted_value: str) -> str:
        """Déchiffrement d'une valeur"""
        if not self._cipher_suite:
            self._initialize_encryption()
        
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode())
        decrypted = self._cipher_suite.decrypt(encrypted_bytes)
        return decrypted.decode()

    def load_config(self, component: str, use_cache: bool = True) -> Dict[str, Any]:
        """Chargement d'une configuration de composant"""
        
        # Vérification du cache
        cache_key = f"{component}_{self.environment.value}"
        if use_cache and cache_key in self._config_cache:
            # Vérification de la fraîcheur du cache (5 minutes)
            if time.time() - self._config_timestamps.get(cache_key, 0) < 300:
                return self._config_cache[cache_key]
        
        # Chargement du fichier de configuration
        config_file = self.config_dir / f"{component}_{self.environment.value}.yaml"
        default_config_file = self.config_dir / f"{component}_default.yaml"
        
        config = {}
        
        # Chargement de la configuration par défaut
        if default_config_file.exists():
            config.update(self._load_yaml_file(default_config_file))
        
        # Chargement de la configuration spécifique à l'environnement
        if config_file.exists():
            config.update(self._load_yaml_file(config_file))
        
        # Déchiffrement des valeurs sensibles
        config = self._decrypt_sensitive_fields(config)
        
        # Validation avec Pydantic
        if component in self._validators:
            try:
                validated_config = self._validators[component](**config)
                config = validated_config.dict()
            except pydantic.ValidationError as e:
                logger.error(f"Erreur validation configuration {component}: {e}")
                raise
        
        # Mise en cache
        self._config_cache[cache_key] = config
        self._config_timestamps[cache_key] = time.time()
        
        logger.debug(f"Configuration {component} chargée pour {self.environment.value}")
        return config

    def save_config(self, component: str, config: Dict[str, Any], 
                   encrypt_sensitive: bool = True):
        """Sauvegarde d'une configuration"""
        
        # Validation avant sauvegarde
        if component in self._validators:
            try:
                validated_config = self._validators[component](**config)
                config = validated_config.dict()
            except pydantic.ValidationError as e:
                logger.error(f"Erreur validation configuration {component}: {e}")
                raise
        
        # Chiffrement des champs sensibles
        if encrypt_sensitive:
            config = self._encrypt_sensitive_fields(config)
        
        # Sauvegarde
        config_file = self.config_dir / f"{component}_{self.environment.value}.yaml"
        self._save_yaml_file(config_file, config)
        
        # Invalidation du cache
        cache_key = f"{component}_{self.environment.value}"
        self._config_cache.pop(cache_key, None)
        self._config_timestamps.pop(cache_key, None)
        
        logger.info(f"Configuration {component} sauvegardée")

    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Chargement d'un fichier YAML"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Erreur lecture fichier {file_path}: {e}")
            return {}

    def _save_yaml_file(self, file_path: Path, data: Dict[str, Any]):
        """Sauvegarde d'un fichier YAML"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
        except Exception as e:
            logger.error(f"Erreur écriture fichier {file_path}: {e}")
            raise

    def _encrypt_sensitive_fields(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Chiffrement des champs sensibles"""
        
        sensitive_fields = [
            'password', 'secret', 'key', 'token', 'api_key',
            'webhook_url', 'smtp_password', 'jwt_secret'
        ]
        
        def encrypt_recursive(obj, path=""):
            if isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    if any(field in key.lower() for field in sensitive_fields):
                        if isinstance(value, str) and value:
                            result[key] = f"encrypted:{self.encrypt_value(value)}"
                        else:
                            result[key] = value
                    else:
                        result[key] = encrypt_recursive(value, current_path)
                return result
            elif isinstance(obj, list):
                return [encrypt_recursive(item, f"{path}[{i}]") for i, item in enumerate(obj)]
            else:
                return obj
        
        return encrypt_recursive(config)

    def _decrypt_sensitive_fields(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Déchiffrement des champs sensibles"""
        
        def decrypt_recursive(obj):
            if isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    if isinstance(value, str) and value.startswith("encrypted:"):
                        try:
                            result[key] = self.decrypt_value(value[10:])
                        except Exception as e:
                            logger.warning(f"Erreur déchiffrement {key}: {e}")
                            result[key] = ""
                    else:
                        result[key] = decrypt_recursive(value)
                return result
            elif isinstance(obj, list):
                return [decrypt_recursive(item) for item in obj]
            else:
                return obj
        
        return decrypt_recursive(config)

    def generate_default_configs(self):
        """Génération des configurations par défaut"""
        
        default_configs = {
            'prometheus': {
                'host': 'localhost',
                'port': 9090,
                'scrape_interval': '15s',
                'evaluation_interval': '15s',
                'retention_time': '30d',
                'storage_path': '/prometheus/data',
                'external_labels': {
                    'environment': self.environment.value,
                    'cluster': 'monitoring-cluster'
                }
            },
            'grafana': {
                'host': 'localhost',
                'port': 3000,
                'protocol': 'http',
                'admin_user': 'admin',
                'admin_password': 'monitoring123!',
                'org_id': 1,
                'plugins': [
                    'grafana-piechart-panel',
                    'grafana-clock-panel',
                    'grafana-worldmap-panel'
                ]
            },
            'alerting': {
                'smtp_host': 'localhost',
                'smtp_port': 587,
                'smtp_from': 'alerts@monitoring.local',
                'smtp_tls': True,
                'slack_channel': '#alerts',
                'slack_username': 'MonitoringBot',
                'webhook_timeout': 30
            },
            'security': {
                'encryption_enabled': True,
                'jwt_secret': secrets.token_urlsafe(32),
                'jwt_expiration_hours': 24,
                'rbac_enabled': True,
                'default_role': 'viewer',
                'audit_logging': True,
                'audit_retention_days': 90,
                'rate_limiting_enabled': True,
                'requests_per_minute': 100,
                'cors_enabled': True,
                'cors_origins': ['*']
            },
            'ml': {
                'anomaly_detection_enabled': True,
                'anomaly_model_type': 'isolation_forest',
                'anomaly_sensitivity': 0.1,
                'anomaly_training_window': '7d',
                'predictive_scaling_enabled': True,
                'prediction_horizon': '1h',
                'prediction_confidence': 0.95,
                'pattern_recognition_enabled': True,
                'min_pattern_length': 10
            },
            'database': {
                'postgres_host': 'localhost',
                'postgres_port': 5432,
                'postgres_database': 'monitoring',
                'postgres_username': 'monitoring_user',
                'postgres_password': 'monitoring_password_2025!',
                'postgres_ssl_mode': 'prefer',
                'redis_host': 'localhost',
                'redis_port': 6379,
                'redis_database': 0,
                'influxdb_enabled': False
            }
        }
        
        for component, config in default_configs.items():
            self.save_config(component, config)
        
        logger.info("Configurations par défaut générées")

    def validate_all_configs(self) -> Dict[str, bool]:
        """Validation de toutes les configurations"""
        
        results = {}
        
        for component in self._validators.keys():
            try:
                self.load_config(component, use_cache=False)
                results[component] = True
                logger.info(f"Configuration {component}: VALIDE")
            except Exception as e:
                results[component] = False
                logger.error(f"Configuration {component}: INVALIDE - {e}")
        
        return results

    def export_configuration(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Export de toute la configuration"""
        
        export_data = {
            'environment': self.environment.value,
            'export_timestamp': datetime.utcnow().isoformat(),
            'components': {}
        }
        
        for component in self._validators.keys():
            try:
                config = self.load_config(component, use_cache=False)
                
                if not include_secrets:
                    # Masquage des champs sensibles
                    config = self._mask_sensitive_fields(config)
                
                export_data['components'][component] = config
            except Exception as e:
                logger.error(f"Erreur export {component}: {e}")
                export_data['components'][component] = {'error': str(e)}
        
        return export_data

    def _mask_sensitive_fields(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Masquage des champs sensibles pour export"""
        
        sensitive_fields = [
            'password', 'secret', 'key', 'token', 'api_key',
            'webhook_url', 'smtp_password', 'jwt_secret'
        ]
        
        def mask_recursive(obj):
            if isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    if any(field in key.lower() for field in sensitive_fields):
                        result[key] = "***MASKED***" if value else ""
                    else:
                        result[key] = mask_recursive(value)
                return result
            elif isinstance(obj, list):
                return [mask_recursive(item) for item in obj]
            else:
                return obj
        
        return mask_recursive(config)

    def get_config_summary(self) -> Dict[str, Any]:
        """Résumé de la configuration"""
        
        summary = {
            'environment': self.environment.value,
            'components': {},
            'cache_status': {
                'cached_configs': len(self._config_cache),
                'cache_hit_ratio': 0.0
            }
        }
        
        for component in self._validators.keys():
            try:
                config = self.load_config(component)
                summary['components'][component] = {
                    'status': 'loaded',
                    'keys_count': len(config),
                    'has_sensitive_data': any(
                        field in str(config).lower() 
                        for field in ['password', 'secret', 'key', 'token']
                    )
                }
            except Exception as e:
                summary['components'][component] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return summary

# =============================================================================
# FACTORY ET UTILITAIRES
# =============================================================================

def create_config_manager(environment: str = "dev") -> ConfigurationManager:
    """Factory pour création d'un gestionnaire de configuration"""
    
    env_map = {
        'dev': Environment.DEVELOPMENT,
        'development': Environment.DEVELOPMENT,
        'staging': Environment.STAGING,
        'prod': Environment.PRODUCTION,
        'production': Environment.PRODUCTION,
        'test': Environment.TEST
    }
    
    env = env_map.get(environment.lower(), Environment.DEVELOPMENT)
    return ConfigurationManager(env)

def setup_monitoring_config(environment: str = "dev", 
                          generate_defaults: bool = True) -> ConfigurationManager:
    """Configuration complète du monitoring"""
    
    config_manager = create_config_manager(environment)
    
    if generate_defaults:
        config_manager.generate_default_configs()
    
    # Validation
    validation_results = config_manager.validate_all_configs()
    
    failed_validations = [comp for comp, valid in validation_results.items() if not valid]
    if failed_validations:
        logger.error(f"Échec validation: {', '.join(failed_validations)}")
        raise ValueError(f"Configuration invalide pour: {', '.join(failed_validations)}")
    
    logger.info(f"Configuration monitoring initialisée pour {environment}")
    return config_manager

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'ConfigurationManager',
    'Environment',
    'ConfigurationLevel',
    'PrometheusConfig',
    'GrafanaConfig',
    'AlertingConfig',
    'SecurityConfig',
    'MLConfig',
    'DatabaseConfig',
    'create_config_manager',
    'setup_monitoring_config'
]
