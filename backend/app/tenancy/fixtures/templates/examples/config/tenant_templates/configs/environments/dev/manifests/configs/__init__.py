"""
Configurations Module - Advanced Kubernetes ConfigMaps Management
================================================================

Module de gestion avancée des ConfigMaps pour le système multi-tenant 
Spotify AI Agent. Fournit une abstraction complète pour la configuration
des environnements de développement avec validation, transformation et 
gestion des dépendances.

Author: Fahed Mlaiel
Team: Spotify AI Agent Development Team
Version: 2.0.0
License: MIT

Roles Contributors:
- Lead Developer & AI Architect: Fahed Mlaiel
- Senior Backend Developer (Python/FastAPI/Django): Fahed Mlaiel  
- Machine Learning Engineer (TensorFlow/PyTorch/Hugging Face): Fahed Mlaiel
- Database Administrator & Data Engineer (PostgreSQL/Redis/MongoDB): Fahed Mlaiel
- Backend Security Specialist: Fahed Mlaiel
- Microservices Architect: Fahed Mlaiel
"""

import json
import yaml
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from ...__init__ import ManifestGenerator, DEFAULT_LABELS

__version__ = "2.0.0"
__author__ = "Fahed Mlaiel"
__maintainer__ = "Fahed Mlaiel Development Team"

class ConfigMapType(Enum):
    """Types de ConfigMaps disponibles."""
    APPLICATION = "application"
    DATABASE = "database"
    CACHE = "cache"
    SECURITY = "security"
    MONITORING = "monitoring"
    ML_MODEL = "ml-model"
    API_GATEWAY = "api-gateway"
    MICROSERVICE = "microservice"
    FEATURE_FLAGS = "feature-flags"
    LOGGING = "logging"

class EnvironmentTier(Enum):
    """Niveaux d'environnement."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    LOCAL = "local"

@dataclass
class ConfigMapSpec:
    """Spécification pour la création d'une ConfigMap."""
    name: str
    config_type: ConfigMapType
    data: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    namespace: str = "spotify-ai-agent-dev"
    immutable: bool = False
    binary_data: Dict[str, bytes] = field(default_factory=dict)

class ConfigurationValidator:
    """Validateur de configuration avancé."""
    
    @staticmethod
    def validate_database_config(config: Dict[str, str]) -> Tuple[bool, List[str]]:
        """Valide la configuration de base de données."""
        required_fields = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER"]
        errors = []
        
        for field in required_fields:
            if field not in config:
                errors.append(f"Champ requis manquant: {field}")
        
        if "DB_PORT" in config:
            try:
                port = int(config["DB_PORT"])
                if port < 1 or port > 65535:
                    errors.append("DB_PORT doit être entre 1 et 65535")
            except ValueError:
                errors.append("DB_PORT doit être un nombre entier")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_security_config(config: Dict[str, str]) -> Tuple[bool, List[str]]:
        """Valide la configuration de sécurité."""
        errors = []
        
        if "JWT_SECRET_KEY" in config and len(config["JWT_SECRET_KEY"]) < 32:
            errors.append("JWT_SECRET_KEY doit faire au moins 32 caractères")
        
        if "RATE_LIMIT_REQUESTS" in config:
            try:
                rate_limit = int(config["RATE_LIMIT_REQUESTS"])
                if rate_limit <= 0:
                    errors.append("RATE_LIMIT_REQUESTS doit être positif")
            except ValueError:
                errors.append("RATE_LIMIT_REQUESTS doit être un nombre entier")
        
        return len(errors) == 0, errors

class ConfigMapManager:
    """Gestionnaire avancé des ConfigMaps Kubernetes."""
    
    def __init__(self, 
                 namespace: str = "spotify-ai-agent-dev",
                 environment: EnvironmentTier = EnvironmentTier.DEVELOPMENT):
        self.namespace = namespace
        self.environment = environment
        self.manifest_generator = ManifestGenerator(namespace)
        self.validator = ConfigurationValidator()
        self._config_registry = {}
        
    def register_config(self, spec: ConfigMapSpec) -> None:
        """Enregistre une spécification de configuration."""
        self._config_registry[spec.name] = spec
    
    def create_application_config(self) -> Dict[str, Any]:
        """Crée la ConfigMap principale de l'application."""
        config_data = {
            # Core Application
            "DEBUG": "true",
            "LOG_LEVEL": "DEBUG", 
            "ENVIRONMENT": self.environment.value,
            "API_VERSION": "v2",
            "APPLICATION_NAME": "Spotify AI Agent",
            "APPLICATION_VERSION": "2.0.0",
            "BUILD_DATE": datetime.now().isoformat(),
            
            # Server Configuration
            "HOST": "0.0.0.0",
            "PORT": "8000",
            "MAX_WORKERS": "4",
            "WORKER_TIMEOUT": "30",
            "KEEP_ALIVE": "2",
            "WORKER_CLASS": "uvicorn.workers.UvicornWorker",
            
            # Performance & Scaling
            "AUTO_SCALING_ENABLED": "true",
            "MIN_REPLICAS": "2",
            "MAX_REPLICAS": "10",
            "CPU_THRESHOLD": "70",
            "MEMORY_THRESHOLD": "80",
            
            # Feature Flags
            "FEATURE_AI_ENABLED": "true",
            "FEATURE_ANALYTICS_ENABLED": "true", 
            "FEATURE_COLLABORATION_ENABLED": "true",
            "FEATURE_REAL_TIME_ENABLED": "true",
            "FEATURE_ML_TRAINING_ENABLED": "true",
            "FEATURE_BATCH_PROCESSING_ENABLED": "true",
            
            # Caching Strategy
            "CACHE_ENABLED": "true",
            "CACHE_TTL": "3600",
            "CACHE_MAX_SIZE": "1000",
            "CACHE_STRATEGY": "LRU",
            
            # Rate Limiting
            "RATE_LIMIT_ENABLED": "true",
            "RATE_LIMIT_REQUESTS": "100",
            "RATE_LIMIT_WINDOW": "60",
            "RATE_LIMIT_STORAGE": "redis",
            
            # CORS Configuration
            "CORS_ENABLED": "true",
            "ALLOWED_ORIGINS": "http://localhost:3000,http://localhost:8080",
            "ALLOWED_METHODS": "GET,POST,PUT,DELETE,OPTIONS,PATCH",
            "ALLOWED_HEADERS": "Content-Type,Authorization,X-Requested-With",
            
            # File Upload
            "MAX_FILE_SIZE": "52428800",  # 50MB
            "ALLOWED_FILE_TYPES": "mp3,wav,flac,m4a,ogg",
            "UPLOAD_PATH": "/tmp/uploads",
            
            # Monitoring & Observability
            "MONITORING_ENABLED": "true",
            "METRICS_ENABLED": "true",
            "TRACING_ENABLED": "true",
            "HEALTH_CHECK_INTERVAL": "30",
            "METRICS_PORT": "9090",
            
            # Business Logic
            "PAGINATION_SIZE": "50",
            "MAX_SEARCH_RESULTS": "1000",
            "SEARCH_TIMEOUT": "5",
            "BACKUP_ENABLED": "false",
            "AUDIT_LOG_ENABLED": "true"
        }
        
        return {
            **self.manifest_generator.generate_base_manifest(
                "ConfigMap",
                "spotify-ai-agent-config", 
                {
                    "app.kubernetes.io/component": "configuration",
                    "config.type": "application",
                    "config.version": "2.0.0"
                }
            ),
            "data": config_data
        }
    
    def create_database_config(self) -> Dict[str, Any]:
        """Crée la ConfigMap pour la configuration de base de données."""
        config_data = {
            # PostgreSQL Primary
            "DB_HOST": "postgres-primary.spotify-ai-agent-dev.svc.cluster.local",
            "DB_PORT": "5432",
            "DB_NAME": "spotify_ai_agent_dev",
            "DB_USER": "spotify_user",
            "DB_POOL_SIZE": "20",
            "DB_MAX_OVERFLOW": "30",
            "DB_POOL_TIMEOUT": "30",
            "DB_POOL_RECYCLE": "3600",
            "DB_ECHO": "true",
            
            # PostgreSQL Read Replica
            "DB_READ_HOST": "postgres-replica.spotify-ai-agent-dev.svc.cluster.local",
            "DB_READ_PORT": "5432",
            "DB_READ_ENABLED": "true",
            
            # Redis Configuration
            "REDIS_HOST": "redis.spotify-ai-agent-dev.svc.cluster.local",
            "REDIS_PORT": "6379",
            "REDIS_DB": "0",
            "REDIS_MAX_CONNECTIONS": "100",
            "REDIS_RETRY_ON_TIMEOUT": "true",
            "REDIS_HEALTH_CHECK_INTERVAL": "30",
            
            # MongoDB Configuration
            "MONGO_HOST": "mongodb.spotify-ai-agent-dev.svc.cluster.local",
            "MONGO_PORT": "27017",
            "MONGO_DATABASE": "spotify_ai_analytics",
            "MONGO_REPLICA_SET": "rs0",
            "MONGO_AUTH_SOURCE": "admin",
            
            # ElasticSearch
            "ELASTICSEARCH_HOST": "elasticsearch.spotify-ai-agent-dev.svc.cluster.local",
            "ELASTICSEARCH_PORT": "9200",
            "ELASTICSEARCH_INDEX_PREFIX": "spotify-ai-dev",
            
            # Connection Pooling
            "DB_CONNECTION_POOL_ENABLED": "true",
            "DB_CONNECTION_POOL_SIZE": "20",
            "DB_CONNECTION_POOL_MAX_OVERFLOW": "30"
        }
        
        is_valid, errors = self.validator.validate_database_config(config_data)
        if not is_valid:
            raise ValueError(f"Configuration de base de données invalide: {errors}")
        
        return {
            **self.manifest_generator.generate_base_manifest(
                "ConfigMap",
                "spotify-ai-agent-database-config",
                {
                    "app.kubernetes.io/component": "database",
                    "config.type": "database",
                    "config.version": "1.0.0"
                }
            ),
            "data": config_data
        }
    
    def create_security_config(self) -> Dict[str, Any]:
        """Crée la ConfigMap pour la configuration de sécurité."""
        config_data = {
            # JWT Configuration
            "JWT_ALGORITHM": "HS256",
            "JWT_ACCESS_TOKEN_EXPIRE_MINUTES": "30",
            "JWT_REFRESH_TOKEN_EXPIRE_DAYS": "7",
            "JWT_ISSUER": "spotify-ai-agent",
            "JWT_AUDIENCE": "spotify-ai-users",
            
            # OAuth Configuration
            "OAUTH_ENABLED": "true",
            "OAUTH_PROVIDERS": "google,spotify,github",
            "OAUTH_REDIRECT_URI": "http://localhost:8000/auth/callback",
            
            # API Security
            "API_KEY_ENABLED": "true",
            "API_KEY_HEADER": "X-API-Key",
            "API_RATE_LIMIT_ENABLED": "true",
            "API_RATE_LIMIT_REQUESTS": "1000",
            "API_RATE_LIMIT_WINDOW": "3600",
            
            # HTTPS Configuration
            "HTTPS_ONLY": "false",
            "SECURE_COOKIES": "false",
            "CSRF_PROTECTION": "true",
            "CSRF_TOKEN_HEADER": "X-CSRF-Token",
            
            # Session Management
            "SESSION_TIMEOUT": "3600",
            "SESSION_REFRESH_ENABLED": "true",
            "MAX_SESSIONS_PER_USER": "5",
            
            # Password Policy
            "PASSWORD_MIN_LENGTH": "8",
            "PASSWORD_REQUIRE_UPPERCASE": "true",
            "PASSWORD_REQUIRE_LOWERCASE": "true",
            "PASSWORD_REQUIRE_DIGITS": "true",
            "PASSWORD_REQUIRE_SPECIAL": "true",
            
            # Account Security
            "MAX_LOGIN_ATTEMPTS": "5",
            "ACCOUNT_LOCKOUT_DURATION": "900",  # 15 minutes
            "TWO_FACTOR_ENABLED": "false",
            
            # Audit & Compliance
            "AUDIT_LOG_ENABLED": "true",
            "AUDIT_LOG_LEVEL": "INFO",
            "COMPLIANCE_MODE": "GDPR",
            "DATA_RETENTION_DAYS": "365"
        }
        
        is_valid, errors = self.validator.validate_security_config(config_data)
        if not is_valid:
            raise ValueError(f"Configuration de sécurité invalide: {errors}")
        
        return {
            **self.manifest_generator.generate_base_manifest(
                "ConfigMap",
                "spotify-ai-agent-security-config",
                {
                    "app.kubernetes.io/component": "security",
                    "config.type": "security",
                    "config.version": "1.0.0"
                }
            ),
            "data": config_data
        }
    
    def create_ml_config(self) -> Dict[str, Any]:
        """Crée la ConfigMap pour la configuration ML."""
        config_data = {
            # Model Configuration
            "ML_MODEL_PATH": "/app/models",
            "ML_MODEL_VERSION": "1.0.0",
            "ML_MODEL_FORMAT": "pytorch",
            "ML_BATCH_SIZE": "32",
            "ML_MAX_SEQUENCE_LENGTH": "512",
            
            # Training Configuration
            "ML_TRAINING_ENABLED": "true",
            "ML_TRAINING_EPOCHS": "10",
            "ML_LEARNING_RATE": "0.001",
            "ML_VALIDATION_SPLIT": "0.2",
            "ML_EARLY_STOPPING": "true",
            
            # Inference Configuration
            "ML_INFERENCE_TIMEOUT": "30",
            "ML_INFERENCE_BATCH_SIZE": "16",
            "ML_GPU_ENABLED": "false",
            "ML_CPU_THREADS": "4",
            
            # Model Registry
            "ML_REGISTRY_ENABLED": "true",
            "ML_REGISTRY_URL": "http://mlflow.spotify-ai-agent-dev.svc.cluster.local:5000",
            "ML_EXPERIMENT_NAME": "spotify-ai-agent-dev",
            
            # Feature Store
            "FEATURE_STORE_ENABLED": "true",
            "FEATURE_STORE_CACHE_TTL": "3600",
            "FEATURE_EXTRACTION_TIMEOUT": "10",
            
            # Audio Processing
            "AUDIO_SAMPLE_RATE": "44100",
            "AUDIO_CHUNK_SIZE": "1024",
            "AUDIO_FORMAT": "wav",
            "AUDIO_CHANNELS": "2",
            
            # Spleeter Configuration
            "SPLEETER_MODEL": "spleeter:2stems-16kHz",
            "SPLEETER_CACHE_ENABLED": "true",
            "SPLEETER_MAX_DURATION": "600",  # 10 minutes
            
            # AI/ML Features
            "AI_RECOMMENDATION_ENABLED": "true",
            "AI_SENTIMENT_ANALYSIS_ENABLED": "true",
            "AI_PLAYLIST_GENERATION_ENABLED": "true",
            "AI_MUSIC_CLASSIFICATION_ENABLED": "true"
        }
        
        return {
            **self.manifest_generator.generate_base_manifest(
                "ConfigMap",
                "spotify-ai-agent-ml-config",
                {
                    "app.kubernetes.io/component": "machine-learning",
                    "config.type": "ml-model",
                    "config.version": "1.0.0"
                }
            ),
            "data": config_data
        }
    
    def create_monitoring_config(self) -> Dict[str, Any]:
        """Crée la ConfigMap pour la configuration de monitoring."""
        config_data = {
            # Prometheus Configuration
            "PROMETHEUS_ENABLED": "true",
            "PROMETHEUS_PORT": "9090",
            "PROMETHEUS_METRICS_PATH": "/metrics",
            "PROMETHEUS_SCRAPE_INTERVAL": "15s",
            
            # Grafana Configuration
            "GRAFANA_ENABLED": "true",
            "GRAFANA_PORT": "3000",
            "GRAFANA_ADMIN_USER": "admin",
            
            # Jaeger Tracing
            "JAEGER_ENABLED": "true",
            "JAEGER_AGENT_HOST": "jaeger-agent.spotify-ai-agent-dev.svc.cluster.local",
            "JAEGER_AGENT_PORT": "6831",
            "JAEGER_SAMPLER_TYPE": "const",
            "JAEGER_SAMPLER_PARAM": "1",
            
            # Logging Configuration
            "LOG_FORMAT": "json",
            "LOG_LEVEL": "DEBUG",
            "LOG_ROTATION": "daily",
            "LOG_MAX_SIZE": "100MB",
            "LOG_BACKUP_COUNT": "7",
            
            # Health Checks
            "HEALTH_CHECK_ENABLED": "true",
            "HEALTH_CHECK_PATH": "/health",
            "HEALTH_CHECK_INTERVAL": "30s",
            "READINESS_CHECK_PATH": "/ready",
            "LIVENESS_CHECK_PATH": "/live",
            
            # Alerting
            "ALERTING_ENABLED": "true",
            "ALERT_WEBHOOK_URL": "",
            "ALERT_CHANNELS": "slack,email",
            
            # Performance Monitoring
            "APM_ENABLED": "true",
            "APM_SERVICE_NAME": "spotify-ai-agent",
            "APM_ENVIRONMENT": "development",
            "APM_SAMPLE_RATE": "0.1",
            
            # Resource Monitoring
            "RESOURCE_MONITORING_ENABLED": "true",
            "CPU_ALERT_THRESHOLD": "80",
            "MEMORY_ALERT_THRESHOLD": "85",
            "DISK_ALERT_THRESHOLD": "90"
        }
        
        return {
            **self.manifest_generator.generate_base_manifest(
                "ConfigMap",
                "spotify-ai-agent-monitoring-config",
                {
                    "app.kubernetes.io/component": "monitoring",
                    "config.type": "monitoring",
                    "config.version": "1.0.0"
                }
            ),
            "data": config_data
        }
    
    def generate_all_configs(self) -> List[Dict[str, Any]]:
        """Génère toutes les ConfigMaps."""
        configs = [
            self.create_application_config(),
            self.create_database_config(),
            self.create_security_config(),
            self.create_ml_config(),
            self.create_monitoring_config()
        ]
        return configs
    
    def export_to_yaml(self, configs: List[Dict[str, Any]], output_path: str) -> None:
        """Exporte les configurations en YAML."""
        yaml_content = "---\n".join([yaml.dump(config, default_flow_style=False) for config in configs])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
    
    def validate_all_configs(self) -> Tuple[bool, List[str]]:
        """Valide toutes les configurations."""
        all_errors = []
        
        try:
            configs = self.generate_all_configs()
            # Validation logique métier ici
            return len(all_errors) == 0, all_errors
        except Exception as e:
            all_errors.append(f"Erreur lors de la génération: {str(e)}")
            return False, all_errors

# Utilitaires pour la gestion des configurations
class ConfigMapUtils:
    """Utilitaires pour la gestion des ConfigMaps."""
    
    @staticmethod
    def merge_configs(*configs: Dict[str, str]) -> Dict[str, str]:
        """Fusionne plusieurs configurations."""
        merged = {}
        for config in configs:
            merged.update(config)
        return merged
    
    @staticmethod
    def filter_by_prefix(config: Dict[str, str], prefix: str) -> Dict[str, str]:
        """Filtre les configurations par préfixe."""
        return {k: v for k, v in config.items() if k.startswith(prefix)}
    
    @staticmethod
    def transform_to_env_format(config: Dict[str, str]) -> str:
        """Transforme en format de variables d'environnement."""
        return "\n".join([f"export {k}='{v}'" for k, v in config.items()])

# Classes d'export
__all__ = [
    'ConfigMapManager',
    'ConfigMapType', 
    'EnvironmentTier',
    'ConfigMapSpec',
    'ConfigurationValidator',
    'ConfigMapUtils'
]
