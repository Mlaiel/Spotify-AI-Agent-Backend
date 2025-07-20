"""
Configuration Management System
===============================

Système de configuration avancé pour le module core de tenancy.
Gestion des configurations multi-environnements avec validation,
chiffrement et hot-reload.

Auteur: Fahed Mlaiel
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import logging
from cryptography.fernet import Fernet
import aiofiles

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Configuration de base de données"""
    host: str = "localhost"
    port: int = 5432
    database: str = "spotify_ai_agent"
    username: str = "postgres"
    password: str = ""
    ssl_mode: str = "prefer"
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False
    tenant_schema_pattern: str = "tenant_{tenant_id}"
    isolation_level: str = "READ_COMMITTED"


@dataclass
class CacheConfig:
    """Configuration du cache"""
    backend: str = "redis"
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    database: int = 0
    ttl_default: int = 3600
    ttl_tenant_config: int = 1800
    ttl_templates: int = 7200
    compression: bool = True
    serializer: str = "pickle"
    key_prefix: str = "tenancy:"


@dataclass
class SecurityConfig:
    """Configuration de sécurité"""
    encryption_key: Optional[str] = None
    jwt_secret: Optional[str] = None
    jwt_expiry: int = 3600
    password_min_length: int = 8
    password_require_special: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    audit_log_enabled: bool = True
    audit_log_level: str = "INFO"
    tenant_isolation_strict: bool = True
    api_key_length: int = 32


@dataclass
class MonitoringConfig:
    """Configuration du monitoring"""
    enabled: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 30
    performance_tracking: bool = True
    memory_threshold: float = 80.0
    cpu_threshold: float = 70.0
    disk_threshold: float = 85.0
    network_monitoring: bool = True
    custom_metrics: bool = True
    export_prometheus: bool = True
    export_graphite: bool = False


@dataclass
class AlertsConfig:
    """Configuration des alertes"""
    enabled: bool = True
    webhook_url: Optional[str] = None
    slack_token: Optional[str] = None
    slack_channel: str = "#alerts"
    email_smtp_host: str = "localhost"
    email_smtp_port: int = 587
    email_from: str = "noreply@spotify-ai-agent.com"
    email_to: list = field(default_factory=list)
    alert_cooldown: int = 300
    severity_levels: list = field(default_factory=lambda: ["LOW", "MEDIUM", "HIGH", "CRITICAL"])
    notification_channels: list = field(default_factory=lambda: ["slack", "email", "webhook"])


@dataclass
class TemplateConfig:
    """Configuration des templates"""
    template_dir: str = "templates"
    cache_templates: bool = True
    auto_reload: bool = True
    jinja_extensions: list = field(default_factory=lambda: ["jinja2.ext.do", "jinja2.ext.loopcontrols"])
    custom_filters: bool = True
    template_timeout: int = 30
    max_template_size: int = 1048576  # 1MB
    allowed_extensions: list = field(default_factory=lambda: [".j2", ".jinja", ".template"])


@dataclass
class LocalizationConfig:
    """Configuration de la localisation"""
    default_language: str = "en"
    supported_languages: list = field(default_factory=lambda: ["en", "fr", "de", "es"])
    locale_dir: str = "locales"
    fallback_language: str = "en"
    auto_detect: bool = True
    cache_translations: bool = True
    pluralization_rules: bool = True


@dataclass
class TenantConfig:
    """Configuration des tenants"""
    max_tenants: int = 1000
    default_schema: str = "public"
    auto_create_schema: bool = True
    auto_migrate: bool = True
    backup_on_delete: bool = True
    encryption_at_rest: bool = True
    data_retention_days: int = 365
    audit_changes: bool = True
    resource_limits: Dict[str, Any] = field(default_factory=lambda: {
        "max_users": 10000,
        "max_storage_gb": 100,
        "max_api_calls_per_hour": 10000
    })


@dataclass
class AutomationConfig:
    """Configuration de l'automation"""
    enabled: bool = True
    scheduler_timezone: str = "UTC"
    max_concurrent_tasks: int = 10
    task_timeout: int = 3600
    retry_attempts: int = 3
    retry_delay: int = 60
    cleanup_interval: int = 86400  # 24 heures
    health_check_automation: bool = True
    auto_scaling: bool = False
    auto_backup: bool = True


@dataclass
class MetricsConfig:
    """Configuration des métriques"""
    enabled: bool = True
    collection_interval: int = 60
    retention_days: int = 30
    aggregation_levels: list = field(default_factory=lambda: ["minute", "hour", "day"])
    custom_metrics: bool = True
    performance_metrics: bool = True
    business_metrics: bool = True
    export_formats: list = field(default_factory=lambda: ["prometheus", "json", "csv"])


class CoreConfig:
    """
    Configuration principale du système core
    
    Gère la configuration complète avec validation, chiffrement
    et rechargement à chaud.
    """
    
    def __init__(self, config_data: Optional[Dict[str, Any]] = None):
        """
        Initialise la configuration
        
        Args:
            config_data: Données de configuration optionnelles
        """
        self.config_data = config_data or {}
        self.config_file: Optional[Path] = None
        self.last_modified: Optional[datetime] = None
        self.encryption_key: Optional[bytes] = None
        
        # Initialisation des configurations
        self._init_encryption()
        self._load_configuration()
        
        logger.info("Configuration initialisée")
    
    def _init_encryption(self):
        """Initialise le système de chiffrement"""
        key = os.getenv("TENANCY_ENCRYPTION_KEY")
        if key:
            try:
                self.encryption_key = key.encode('utf-8')
                # Validation de la clé
                Fernet(self.encryption_key)
            except Exception as e:
                logger.warning(f"Clé de chiffrement invalide: {e}")
                self.encryption_key = None
        
        if not self.encryption_key:
            # Génération d'une nouvelle clé
            self.encryption_key = Fernet.generate_key()
            logger.info("Nouvelle clé de chiffrement générée")
    
    def _load_configuration(self):
        """Charge la configuration depuis différentes sources"""
        # Configuration par défaut
        self._load_defaults()
        
        # Configuration depuis fichier
        config_file = os.getenv("TENANCY_CONFIG_FILE")
        if config_file:
            self._load_from_file(Path(config_file))
        
        # Configuration depuis variables d'environnement
        self._load_from_env()
        
        # Configuration depuis les données passées
        if self.config_data:
            self._merge_config(self.config_data)
        
        # Validation finale
        self._validate_configuration()
    
    def _load_defaults(self):
        """Charge la configuration par défaut"""
        self.database_config = DatabaseConfig()
        self.cache_config = CacheConfig()
        self.security_config = SecurityConfig()
        self.monitoring_config = MonitoringConfig()
        self.alerts_config = AlertsConfig()
        self.template_config = TemplateConfig()
        self.localization_config = LocalizationConfig()
        self.tenant_config = TenantConfig()
        self.automation_config = AutomationConfig()
        self.metrics_config = MetricsConfig()
    
    def _load_from_file(self, config_path: Path):
        """
        Charge la configuration depuis un fichier
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        if not config_path.exists():
            logger.warning(f"Fichier de configuration non trouvé: {config_path}")
            return
        
        try:
            self.config_file = config_path
            self.last_modified = datetime.fromtimestamp(config_path.stat().st_mtime)
            
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                logger.error(f"Format de fichier non supporté: {config_path.suffix}")
                return
            
            self._merge_config(data)
            logger.info(f"Configuration chargée depuis: {config_path}")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {e}")
    
    def _load_from_env(self):
        """Charge la configuration depuis les variables d'environnement"""
        env_mappings = {
            # Database
            "TENANCY_DB_HOST": ("database_config", "host"),
            "TENANCY_DB_PORT": ("database_config", "port"),
            "TENANCY_DB_NAME": ("database_config", "database"),
            "TENANCY_DB_USER": ("database_config", "username"),
            "TENANCY_DB_PASSWORD": ("database_config", "password"),
            
            # Cache
            "TENANCY_CACHE_HOST": ("cache_config", "host"),
            "TENANCY_CACHE_PORT": ("cache_config", "port"),
            "TENANCY_CACHE_PASSWORD": ("cache_config", "password"),
            
            # Security
            "TENANCY_JWT_SECRET": ("security_config", "jwt_secret"),
            "TENANCY_ENCRYPTION_KEY": ("security_config", "encryption_key"),
            
            # Monitoring
            "TENANCY_MONITORING_PORT": ("monitoring_config", "metrics_port"),
            
            # Alerts
            "TENANCY_SLACK_TOKEN": ("alerts_config", "slack_token"),
            "TENANCY_SLACK_CHANNEL": ("alerts_config", "slack_channel"),
            "TENANCY_WEBHOOK_URL": ("alerts_config", "webhook_url"),
        }
        
        for env_var, (config_section, config_key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                config_obj = getattr(self, config_section)
                
                # Conversion de type si nécessaire
                field_type = type(getattr(config_obj, config_key))
                if field_type == int:
                    value = int(value)
                elif field_type == bool:
                    value = value.lower() in ('true', '1', 'yes', 'on')
                elif field_type == float:
                    value = float(value)
                
                setattr(config_obj, config_key, value)
    
    def _merge_config(self, data: Dict[str, Any]):
        """
        Fusionne les données de configuration
        
        Args:
            data: Données à fusionner
        """
        for section, section_data in data.items():
            if hasattr(self, f"{section}_config"):
                config_obj = getattr(self, f"{section}_config")
                for key, value in section_data.items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)
    
    def _validate_configuration(self):
        """Valide la configuration"""
        # Validation des configurations critiques
        
        # Base de données
        if not self.database_config.host:
            raise ValueError("Host de base de données manquant")
        
        # Sécurité
        if not self.security_config.jwt_secret:
            self.security_config.jwt_secret = os.urandom(32).hex()
            logger.warning("JWT secret généré automatiquement")
        
        # Cache
        if self.cache_config.ttl_default <= 0:
            raise ValueError("TTL par défaut du cache doit être positif")
        
        logger.info("Configuration validée avec succès")
    
    async def reload_configuration(self) -> bool:
        """
        Recharge la configuration depuis le fichier
        
        Returns:
            True si la configuration a été rechargée
        """
        if not self.config_file or not self.config_file.exists():
            return False
        
        try:
            current_modified = datetime.fromtimestamp(self.config_file.stat().st_mtime)
            
            if self.last_modified and current_modified <= self.last_modified:
                return False
            
            # Sauvegarde de la configuration actuelle
            backup_config = self._create_backup()
            
            try:
                # Rechargement
                self._load_from_file(self.config_file)
                logger.info("Configuration rechargée avec succès")
                return True
                
            except Exception as e:
                # Restauration de la sauvegarde
                self._restore_backup(backup_config)
                logger.error(f"Erreur lors du rechargement, configuration restaurée: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Erreur lors du rechargement de la configuration: {e}")
            return False
    
    def _create_backup(self) -> Dict[str, Any]:
        """Crée une sauvegarde de la configuration actuelle"""
        return {
            "database_config": self.database_config,
            "cache_config": self.cache_config,
            "security_config": self.security_config,
            "monitoring_config": self.monitoring_config,
            "alerts_config": self.alerts_config,
            "template_config": self.template_config,
            "localization_config": self.localization_config,
            "tenant_config": self.tenant_config,
            "automation_config": self.automation_config,
            "metrics_config": self.metrics_config,
        }
    
    def _restore_backup(self, backup: Dict[str, Any]):
        """Restaure une sauvegarde de configuration"""
        for key, value in backup.items():
            setattr(self, key, value)
    
    def encrypt_value(self, value: str) -> str:
        """
        Chiffre une valeur
        
        Args:
            value: Valeur à chiffrer
            
        Returns:
            Valeur chiffrée
        """
        if not self.encryption_key:
            return value
        
        try:
            fernet = Fernet(self.encryption_key)
            encrypted = fernet.encrypt(value.encode('utf-8'))
            return encrypted.decode('utf-8')
        except Exception as e:
            logger.error(f"Erreur de chiffrement: {e}")
            return value
    
    def decrypt_value(self, encrypted_value: str) -> str:
        """
        Déchiffre une valeur
        
        Args:
            encrypted_value: Valeur chiffrée
            
        Returns:
            Valeur déchiffrée
        """
        if not self.encryption_key:
            return encrypted_value
        
        try:
            fernet = Fernet(self.encryption_key)
            decrypted = fernet.decrypt(encrypted_value.encode('utf-8'))
            return decrypted.decode('utf-8')
        except Exception as e:
            logger.error(f"Erreur de déchiffrement: {e}")
            return encrypted_value
    
    async def save_to_file(self, file_path: Path, format: str = "yaml") -> bool:
        """
        Sauvegarde la configuration dans un fichier
        
        Args:
            file_path: Chemin du fichier
            format: Format de sauvegarde (yaml/json)
            
        Returns:
            True si sauvegardé avec succès
        """
        try:
            config_dict = {
                "database": self.database_config.__dict__,
                "cache": self.cache_config.__dict__,
                "security": self.security_config.__dict__,
                "monitoring": self.monitoring_config.__dict__,
                "alerts": self.alerts_config.__dict__,
                "template": self.template_config.__dict__,
                "localization": self.localization_config.__dict__,
                "tenant": self.tenant_config.__dict__,
                "automation": self.automation_config.__dict__,
                "metrics": self.metrics_config.__dict__,
            }
            
            # Masquage des données sensibles
            sensitive_keys = ["password", "secret", "key", "token"]
            self._mask_sensitive_data(config_dict, sensitive_keys)
            
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                if format.lower() == "yaml":
                    content = yaml.dump(config_dict, default_flow_style=False, indent=2)
                else:
                    content = json.dumps(config_dict, indent=2)
                
                await f.write(content)
            
            logger.info(f"Configuration sauvegardée dans: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {e}")
            return False
    
    def _mask_sensitive_data(self, data: Dict[str, Any], sensitive_keys: list):
        """Masque les données sensibles dans la configuration"""
        for key, value in data.items():
            if isinstance(value, dict):
                self._mask_sensitive_data(value, sensitive_keys)
            elif isinstance(key, str) and any(sensitive in key.lower() for sensitive in sensitive_keys):
                if value:
                    data[key] = "***MASKED***"
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Retourne un résumé de la configuration
        
        Returns:
            Résumé de configuration
        """
        return {
            "database": {
                "host": self.database_config.host,
                "port": self.database_config.port,
                "database": self.database_config.database,
                "pool_size": self.database_config.pool_size
            },
            "cache": {
                "backend": self.cache_config.backend,
                "host": self.cache_config.host,
                "port": self.cache_config.port
            },
            "security": {
                "encryption_enabled": bool(self.security_config.encryption_key),
                "jwt_configured": bool(self.security_config.jwt_secret),
                "rate_limit": f"{self.security_config.rate_limit_requests}/{self.security_config.rate_limit_window}s"
            },
            "monitoring": {
                "enabled": self.monitoring_config.enabled,
                "port": self.monitoring_config.metrics_port
            },
            "tenants": {
                "max_tenants": self.tenant_config.max_tenants,
                "auto_create_schema": self.tenant_config.auto_create_schema
            }
        }
