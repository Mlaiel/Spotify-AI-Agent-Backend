#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration avancée pour le système i18n Slack

Ce module centralise toute la configuration du système d'internationalisation
avec support pour:
- Configuration par environnement (dev/staging/prod)
- Variables d'environnement avec fallbacks intelligents
- Validation automatique de la configuration
- Configuration hot-reload pour les mises à jour à chaud
- Secrets management intégré
- Profils de performance adaptatifs

Auteur: Expert Team
Version: 2.0.0
"""

import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import yaml
import json

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Environnements supportés"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(Enum):
    """Niveaux de log"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class RedisConfig:
    """Configuration Redis"""
    host: str = "localhost"
    port: int = 6379
    db: int = 2
    password: Optional[str] = None
    ssl: bool = False
    ssl_cert_reqs: Optional[str] = None
    ssl_ca_certs: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    max_connections: int = 20
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    socket_keepalive: bool = True
    socket_keepalive_options: Dict[str, int] = field(default_factory=dict)
    connection_pool_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheConfig:
    """Configuration du cache"""
    enabled: bool = True
    default_ttl: int = 3600  # 1 heure
    max_local_size: int = 10000
    local_ttl: int = 300  # 5 minutes
    compression_enabled: bool = True
    compression_level: int = 6
    key_prefix: str = "i18n:slack:"
    cleanup_interval: int = 3600  # 1 heure
    
    # Stratégies TTL par type
    ttl_strategies: Dict[str, int] = field(default_factory=lambda: {
        "translations": 3600,      # 1h
        "detections": 1800,        # 30min
        "user_profiles": 86400,    # 24h
        "ai_enhanced": 7200,       # 2h
        "cultural_format": 14400   # 4h
    })


@dataclass
class AIConfig:
    """Configuration IA"""
    enabled: bool = True
    provider: str = "openai"  # openai, anthropic, local
    model: str = "gpt-4-turbo"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.3
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Seuils de qualité
    confidence_threshold: float = 0.85
    auto_improve: bool = True
    learning_rate: float = 0.1
    batch_size: int = 10
    
    # Limites de sécurité
    max_requests_per_minute: int = 60
    max_requests_per_day: int = 10000


@dataclass
class DetectionConfig:
    """Configuration détection de langue"""
    enabled: bool = True
    default_language: str = "en"
    supported_languages: List[str] = field(default_factory=lambda: [
        "en", "fr", "de", "es", "pt", "it", "ru", "zh", "ja", "ar", "he"
    ])
    
    # Méthodes de détection
    methods_enabled: Dict[str, bool] = field(default_factory=lambda: {
        "content_analysis": True,
        "user_preference": True,
        "geographic": True,
        "browser_header": True,
        "tenant_default": True,
        "ml_prediction": True
    })
    
    # Seuils de confiance
    confidence_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "very_high": 0.95,
        "high": 0.85,
        "medium": 0.70,
        "low": 0.50
    })
    
    # Cache des profils utilisateur
    user_profiles_ttl: int = 86400 * 30  # 30 jours
    max_detection_history: int = 100


@dataclass
class MonitoringConfig:
    """Configuration monitoring"""
    enabled: bool = True
    metrics_enabled: bool = True
    tracing_enabled: bool = True
    alerting_enabled: bool = True
    
    # Endpoints
    metrics_endpoint: str = "/metrics"
    health_endpoint: str = "/health"
    status_endpoint: str = "/status"
    
    # Seuils d'alerte
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "translation_latency_ms": 100.0,
        "cache_hit_ratio": 0.85,
        "error_rate": 0.01,
        "ai_response_time_ms": 5000.0,
        "memory_usage_mb": 1024.0
    })
    
    # Webhooks pour alertes
    alert_webhooks: List[str] = field(default_factory=list)
    
    # Rétention des métriques
    metrics_retention_days: int = 30
    logs_retention_days: int = 90


@dataclass
class SecurityConfig:
    """Configuration sécurité"""
    enabled: bool = True
    
    # Sanitisation
    input_sanitization: bool = True
    html_escape: bool = True
    script_blocking: bool = True
    
    # Validation
    max_key_length: int = 256
    max_value_length: int = 10000
    max_context_size: int = 100000
    
    # Rate limiting
    rate_limiting_enabled: bool = True
    requests_per_minute: int = 1000
    requests_per_hour: int = 10000
    burst_limit: int = 100
    
    # Chiffrement
    encryption_enabled: bool = False
    encryption_key: Optional[str] = None
    
    # Audit
    audit_enabled: bool = True
    sensitive_fields: List[str] = field(default_factory=lambda: [
        "user_id", "ip_address", "user_agent"
    ])


@dataclass
class ComplianceConfig:
    """Configuration conformité"""
    gdpr_enabled: bool = True
    data_retention_days: int = 90
    anonymization_enabled: bool = True
    consent_required: bool = False
    
    # Audit trail
    audit_logging: bool = True
    audit_events: List[str] = field(default_factory=lambda: [
        "translation_request",
        "language_detection",
        "ai_enhancement",
        "cache_access",
        "error_occurrence"
    ])
    
    # Géolocalisation
    geo_restrictions: Dict[str, List[str]] = field(default_factory=dict)
    data_residency_rules: Dict[str, str] = field(default_factory=dict)


@dataclass
class I18nConfig:
    """Configuration complète du système i18n"""
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    log_level: LogLevel = LogLevel.INFO
    
    # Chemins
    translations_path: str = "./i18n"
    logs_path: str = "./logs"
    cache_path: str = "./cache"
    
    # Configurations des composants
    redis: RedisConfig = field(default_factory=RedisConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)
    
    # Configuration culturelle
    cultural_formatting: bool = True
    rtl_support: bool = True
    emoji_support: bool = True
    timezone_aware: bool = True
    
    # Performance
    performance_mode: str = "balanced"  # fast, balanced, quality
    max_concurrent_requests: int = 100
    request_timeout: int = 30
    
    # Features flags
    features: Dict[str, bool] = field(default_factory=lambda: {
        "ai_enhancement": True,
        "cultural_formatting": True,
        "real_time_updates": True,
        "advanced_caching": True,
        "performance_monitoring": True,
        "auto_scaling": False,
        "multi_tenant": True
    })


class ConfigManager:
    """Gestionnaire de configuration ultra-avancé"""
    
    def __init__(self, config_path: Optional[str] = None, environment: Optional[str] = None):
        """
        Initialise le gestionnaire de configuration
        
        Args:
            config_path: Chemin vers le fichier de configuration
            environment: Environnement cible (dev/staging/prod)
        """
        self.config_path = Path(config_path) if config_path else None
        self.environment = Environment(environment) if environment else self._detect_environment()
        self._config: Optional[I18nConfig] = None
        self._watchers: List[callable] = []
        
        logger.info(f"Configuration manager initialisé pour l'environnement: {self.environment.value}")
    
    def _detect_environment(self) -> Environment:
        """Détecte automatiquement l'environnement"""
        env = os.getenv("ENVIRONMENT", os.getenv("ENV", "development")).lower()
        
        env_mapping = {
            "dev": Environment.DEVELOPMENT,
            "development": Environment.DEVELOPMENT,
            "staging": Environment.STAGING,
            "stage": Environment.STAGING,
            "prod": Environment.PRODUCTION,
            "production": Environment.PRODUCTION,
            "test": Environment.TESTING,
            "testing": Environment.TESTING
        }
        
        return env_mapping.get(env, Environment.DEVELOPMENT)
    
    def load_config(self) -> I18nConfig:
        """Charge la configuration complète avec toutes les sources"""
        if self._config:
            return self._config
        
        # Configuration de base
        config = I18nConfig()
        config.environment = self.environment
        
        # 1. Configuration par défaut selon l'environnement
        self._apply_environment_defaults(config)
        
        # 2. Chargement depuis fichier YAML si présent
        if self.config_path and self.config_path.exists():
            self._load_from_file(config)
        
        # 3. Variables d'environnement (priorité maximale)
        self._load_from_environment(config)
        
        # 4. Validation de la configuration
        self._validate_config(config)
        
        # 5. Application des optimisations selon l'environnement
        self._apply_optimizations(config)
        
        self._config = config
        logger.info("Configuration chargée avec succès")
        
        return config
    
    def _apply_environment_defaults(self, config: I18nConfig) -> None:
        """Applique les valeurs par défaut selon l'environnement"""
        if config.environment == Environment.DEVELOPMENT:
            config.debug = True
            config.log_level = LogLevel.DEBUG
            config.redis.host = "localhost"
            config.cache.enabled = True
            config.ai.enabled = False  # Pas d'IA en dev par défaut
            config.monitoring.enabled = False
            config.security.rate_limiting_enabled = False
            
        elif config.environment == Environment.STAGING:
            config.debug = False
            config.log_level = LogLevel.INFO
            config.cache.enabled = True
            config.ai.enabled = True
            config.monitoring.enabled = True
            config.security.rate_limiting_enabled = True
            config.compliance.gdpr_enabled = True
            
        elif config.environment == Environment.PRODUCTION:
            config.debug = False
            config.log_level = LogLevel.WARNING
            config.cache.enabled = True
            config.ai.enabled = True
            config.monitoring.enabled = True
            config.security.rate_limiting_enabled = True
            config.compliance.gdpr_enabled = True
            config.performance_mode = "fast"
            
            # Paramètres production optimisés
            config.redis.max_connections = 50
            config.cache.default_ttl = 7200  # 2h
            config.max_concurrent_requests = 500
            
        elif config.environment == Environment.TESTING:
            config.debug = True
            config.log_level = LogLevel.DEBUG
            config.cache.enabled = False  # Tests déterministes
            config.ai.enabled = False
            config.monitoring.enabled = False
            config.security.rate_limiting_enabled = False
    
    def _load_from_file(self, config: I18nConfig) -> None:
        """Charge la configuration depuis un fichier YAML"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
            
            # Application récursive de la configuration
            self._merge_config_dict(config, file_config)
            
            logger.info(f"Configuration chargée depuis: {self.config_path}")
            
        except Exception as e:
            logger.warning(f"Erreur chargement fichier config: {e}")
    
    def _load_from_environment(self, config: I18nConfig) -> None:
        """Charge la configuration depuis les variables d'environnement"""
        
        # Mapping des variables d'environnement
        env_mappings = {
            # Redis
            "REDIS_HOST": ("redis", "host"),
            "REDIS_PORT": ("redis", "port", int),
            "REDIS_DB": ("redis", "db", int),
            "REDIS_PASSWORD": ("redis", "password"),
            "REDIS_SSL": ("redis", "ssl", bool),
            "REDIS_MAX_CONNECTIONS": ("redis", "max_connections", int),
            
            # Cache
            "CACHE_ENABLED": ("cache", "enabled", bool),
            "CACHE_DEFAULT_TTL": ("cache", "default_ttl", int),
            "CACHE_MAX_LOCAL_SIZE": ("cache", "max_local_size", int),
            
            # IA
            "AI_ENABLED": ("ai", "enabled", bool),
            "AI_PROVIDER": ("ai", "provider"),
            "AI_MODEL": ("ai", "model"),
            "AI_API_KEY": ("ai", "api_key"),
            "AI_MAX_TOKENS": ("ai", "max_tokens", int),
            "AI_TEMPERATURE": ("ai", "temperature", float),
            
            # Détection
            "DETECTION_DEFAULT_LANGUAGE": ("detection", "default_language"),
            
            # Monitoring
            "MONITORING_ENABLED": ("monitoring", "enabled", bool),
            "METRICS_ENABLED": ("monitoring", "metrics_enabled", bool),
            
            # Sécurité
            "SECURITY_ENABLED": ("security", "enabled", bool),
            "RATE_LIMITING_ENABLED": ("security", "rate_limiting_enabled", bool),
            "REQUESTS_PER_MINUTE": ("security", "requests_per_minute", int),
            
            # Général
            "DEBUG": ("debug", None, bool),
            "LOG_LEVEL": ("log_level", None, LogLevel),
            "TRANSLATIONS_PATH": ("translations_path",),
            "PERFORMANCE_MODE": ("performance_mode",),
            "MAX_CONCURRENT_REQUESTS": ("max_concurrent_requests", int)
        }
        
        for env_var, mapping in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_config_value(config, mapping, value)
    
    def _set_config_value(self, config: I18nConfig, mapping: tuple, value: str) -> None:
        """Définit une valeur de configuration à partir du mapping"""
        try:
            # Conversion du type si spécifié
            if len(mapping) > 2:
                converter = mapping[2]
                if converter == bool:
                    value = value.lower() in ('true', '1', 'yes', 'on')
                elif converter == int:
                    value = int(value)
                elif converter == float:
                    value = float(value)
                elif converter == LogLevel:
                    value = LogLevel(value.upper())
                else:
                    value = converter(value)
            
            # Navigation dans la structure de config
            if len(mapping) == 1:
                setattr(config, mapping[0], value)
            elif len(mapping) >= 2:
                sub_config = getattr(config, mapping[0])
                setattr(sub_config, mapping[1], value)
                
        except Exception as e:
            logger.warning(f"Erreur conversion variable {mapping}: {e}")
    
    def _merge_config_dict(self, config: I18nConfig, config_dict: Dict[str, Any]) -> None:
        """Fusionne un dictionnaire de configuration avec l'objet config"""
        for key, value in config_dict.items():
            if hasattr(config, key):
                attr = getattr(config, key)
                if isinstance(value, dict) and hasattr(attr, '__dict__'):
                    # Fusion récursive pour les sous-objets
                    for sub_key, sub_value in value.items():
                        if hasattr(attr, sub_key):
                            setattr(attr, sub_key, sub_value)
                else:
                    setattr(config, key, value)
    
    def _validate_config(self, config: I18nConfig) -> None:
        """Valide la configuration et applique les corrections nécessaires"""
        
        # Validation Redis
        if config.redis.port < 1 or config.redis.port > 65535:
            logger.warning(f"Port Redis invalide: {config.redis.port}, utilisation de 6379")
            config.redis.port = 6379
        
        # Validation Cache
        if config.cache.default_ttl <= 0:
            logger.warning("TTL cache invalide, utilisation de 3600s")
            config.cache.default_ttl = 3600
        
        # Validation IA
        if config.ai.enabled and not config.ai.api_key:
            logger.warning("IA activée mais pas de clé API, désactivation")
            config.ai.enabled = False
        
        # Validation seuils monitoring
        for metric, threshold in config.monitoring.alert_thresholds.items():
            if threshold <= 0:
                logger.warning(f"Seuil invalide pour {metric}: {threshold}")
                config.monitoring.alert_thresholds[metric] = 100.0
        
        # Validation langues supportées
        if config.detection.default_language not in config.detection.supported_languages:
            logger.warning("Langue par défaut non supportée, ajout à la liste")
            config.detection.supported_languages.append(config.detection.default_language)
        
        # Validation chemins
        for path_attr in ['translations_path', 'logs_path', 'cache_path']:
            path = Path(getattr(config, path_attr))
            if not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Répertoire créé: {path}")
                except Exception as e:
                    logger.error(f"Impossible de créer {path}: {e}")
    
    def _apply_optimizations(self, config: I18nConfig) -> None:
        """Applique les optimisations selon le mode de performance"""
        
        if config.performance_mode == "fast":
            # Mode performance maximale
            config.cache.default_ttl = 7200  # TTL plus long
            config.cache.compression_enabled = False  # Pas de compression
            config.ai.timeout = 10  # Timeout IA réduit
            config.detection.max_detection_history = 50  # Moins d'historique
            
        elif config.performance_mode == "quality":
            # Mode qualité maximale
            config.cache.default_ttl = 1800  # TTL plus court
            config.cache.compression_enabled = True
            config.ai.timeout = 60  # Timeout IA plus long
            config.ai.retry_attempts = 5  # Plus de tentatives
            config.detection.max_detection_history = 200  # Plus d'historique
            
        # Mode "balanced" utilise les valeurs par défaut
    
    def get_config(self) -> I18nConfig:
        """Retourne la configuration actuelle"""
        if not self._config:
            return self.load_config()
        return self._config
    
    def reload_config(self) -> I18nConfig:
        """Recharge la configuration"""
        self._config = None
        config = self.load_config()
        
        # Notification des watchers
        for watcher in self._watchers:
            try:
                watcher(config)
            except Exception as e:
                logger.error(f"Erreur notification watcher: {e}")
        
        return config
    
    def add_config_watcher(self, callback: callable) -> None:
        """Ajoute un watcher pour les changements de configuration"""
        self._watchers.append(callback)
    
    def export_config(self, format: str = "yaml") -> str:
        """Exporte la configuration dans le format spécifié"""
        config = self.get_config()
        
        if format.lower() == "yaml":
            return yaml.dump(config.__dict__, default_flow_style=False, allow_unicode=True)
        elif format.lower() == "json":
            return json.dumps(config.__dict__, indent=2, default=str, ensure_ascii=False)
        else:
            raise ValueError(f"Format non supporté: {format}")
    
    def validate_environment(self) -> Dict[str, Any]:
        """Valide l'environnement et retourne un rapport"""
        config = self.get_config()
        report = {
            "environment": config.environment.value,
            "checks": [],
            "warnings": [],
            "errors": []
        }
        
        # Vérification Redis
        try:
            import redis
            r = redis.Redis(
                host=config.redis.host,
                port=config.redis.port,
                db=config.redis.db,
                password=config.redis.password
            )
            r.ping()
            report["checks"].append("✅ Connexion Redis OK")
        except Exception as e:
            report["errors"].append(f"❌ Connexion Redis échouée: {e}")
        
        # Vérification chemins
        for path_name, path_value in [
            ("translations", config.translations_path),
            ("logs", config.logs_path),
            ("cache", config.cache_path)
        ]:
            path = Path(path_value)
            if path.exists():
                report["checks"].append(f"✅ Répertoire {path_name} existe: {path}")
            else:
                report["warnings"].append(f"⚠️  Répertoire {path_name} manquant: {path}")
        
        # Vérification IA
        if config.ai.enabled:
            if config.ai.api_key:
                report["checks"].append("✅ Configuration IA présente")
            else:
                report["errors"].append("❌ IA activée mais clé API manquante")
        
        return report


# Instance globale pour faciliter l'utilisation
_config_manager: Optional[ConfigManager] = None

def get_config_manager(config_path: Optional[str] = None, environment: Optional[str] = None) -> ConfigManager:
    """Retourne l'instance globale du gestionnaire de configuration"""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager(config_path, environment)
    
    return _config_manager

def get_config() -> I18nConfig:
    """Raccourci pour obtenir la configuration actuelle"""
    return get_config_manager().get_config()

# Export des classes principales
__all__ = [
    "I18nConfig",
    "RedisConfig",
    "CacheConfig", 
    "AIConfig",
    "DetectionConfig",
    "MonitoringConfig",
    "SecurityConfig",
    "ComplianceConfig",
    "ConfigManager",
    "Environment",
    "LogLevel",
    "get_config_manager",
    "get_config"
]
