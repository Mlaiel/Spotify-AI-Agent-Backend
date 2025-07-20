"""
Environment-Specific Configuration Profiles
==========================================

Pre-configured environment profiles for different deployment scenarios.
Provides optimized settings for development, staging, and production
environments with appropriate security, performance, and monitoring configurations.

Author: Fahed Mlaiel
Team: Spotify AI Agent Development Team
Version: 1.0.0
"""

from typing import Dict, Any
from enum import Enum
from dataclasses import dataclass, field

class EnvironmentType(Enum):
    """Types d'environnement disponibles."""
    LOCAL = "local"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class EnvironmentProfile:
    """Profil de configuration pour un environnement."""
    name: str
    description: str
    application_config: Dict[str, Any] = field(default_factory=dict)
    database_config: Dict[str, Any] = field(default_factory=dict)
    security_config: Dict[str, Any] = field(default_factory=dict)
    ml_config: Dict[str, Any] = field(default_factory=dict)
    monitoring_config: Dict[str, Any] = field(default_factory=dict)

class EnvironmentProfileManager:
    """Gestionnaire des profils d'environnement."""
    
    def __init__(self):
        self.profiles = self._initialize_profiles()
    
    def _initialize_profiles(self) -> Dict[str, EnvironmentProfile]:
        """Initialise tous les profils d'environnement."""
        return {
            EnvironmentType.LOCAL.value: self._create_local_profile(),
            EnvironmentType.DEVELOPMENT.value: self._create_development_profile(),
            EnvironmentType.STAGING.value: self._create_staging_profile(),
            EnvironmentType.PRODUCTION.value: self._create_production_profile()
        }
    
    def _create_local_profile(self) -> EnvironmentProfile:
        """Profil pour développement local."""
        return EnvironmentProfile(
            name="Local Development",
            description="Configuration optimisée pour le développement local sur machine développeur",
            application_config={
                "DEBUG": "true",
                "LOG_LEVEL": "DEBUG",
                "ENVIRONMENT": "local",
                "API_VERSION": "v2",
                "HOST": "127.0.0.1",
                "PORT": "8000",
                "MAX_WORKERS": "1",
                "WORKER_TIMEOUT": "30",
                "AUTO_SCALING_ENABLED": "false",
                "MIN_REPLICAS": "1",
                "MAX_REPLICAS": "1",
                "CACHE_ENABLED": "true",
                "CACHE_TTL": "1800",
                "RATE_LIMIT_ENABLED": "false",
                "MONITORING_ENABLED": "false",
                "METRICS_ENABLED": "false",
                "CORS_ENABLED": "true",
                "ALLOWED_ORIGINS": "http://localhost:3000,http://127.0.0.1:3000",
                "FEATURE_AI_ENABLED": "true",
                "FEATURE_ANALYTICS_ENABLED": "false",
                "FEATURE_COLLABORATION_ENABLED": "false"
            },
            database_config={
                "DB_HOST": "localhost",
                "DB_PORT": "5432",
                "DB_NAME": "spotify_ai_local",
                "DB_USER": "spotify_local",
                "DB_POOL_SIZE": "5",
                "DB_MAX_OVERFLOW": "10",
                "DB_ECHO": "true",
                "REDIS_HOST": "localhost",
                "REDIS_PORT": "6379",
                "REDIS_DB": "0",
                "REDIS_MAX_CONNECTIONS": "10",
                "MONGO_HOST": "localhost",
                "MONGO_PORT": "27017",
                "MONGO_DATABASE": "spotify_ai_local",
                "ELASTICSEARCH_HOST": "localhost",
                "ELASTICSEARCH_PORT": "9200"
            },
            security_config={
                "JWT_ACCESS_TOKEN_EXPIRE_MINUTES": "60",
                "JWT_REFRESH_TOKEN_EXPIRE_DAYS": "30",
                "OAUTH_ENABLED": "false",
                "API_KEY_ENABLED": "false",
                "RATE_LIMIT_ENABLED": "false",
                "HTTPS_ONLY": "false",
                "SECURE_COOKIES": "false",
                "CSRF_PROTECTION": "false",
                "PASSWORD_MIN_LENGTH": "6",
                "MAX_LOGIN_ATTEMPTS": "10",
                "TWO_FACTOR_ENABLED": "false"
            },
            ml_config={
                "ML_TRAINING_ENABLED": "true",
                "ML_BATCH_SIZE": "8",
                "ML_TRAINING_EPOCHS": "5",
                "ML_GPU_ENABLED": "false",
                "ML_CPU_THREADS": "2",
                "SPLEETER_CACHE_ENABLED": "true",
                "SPLEETER_MAX_DURATION": "300",
                "AUDIO_SAMPLE_RATE": "22050"
            },
            monitoring_config={
                "PROMETHEUS_ENABLED": "false",
                "GRAFANA_ENABLED": "false",
                "JAEGER_ENABLED": "false",
                "LOG_FORMAT": "text",
                "LOG_LEVEL": "DEBUG",
                "HEALTH_CHECK_ENABLED": "true",
                "RESOURCE_MONITORING_ENABLED": "false"
            }
        )
    
    def _create_development_profile(self) -> EnvironmentProfile:
        """Profil pour environnement de développement (conteneur/cloud)."""
        return EnvironmentProfile(
            name="Development Environment",
            description="Configuration pour environnement de développement en conteneur avec monitoring complet",
            application_config={
                "DEBUG": "true",
                "LOG_LEVEL": "DEBUG",
                "ENVIRONMENT": "development",
                "API_VERSION": "v2",
                "HOST": "0.0.0.0",
                "PORT": "8000",
                "MAX_WORKERS": "4",
                "WORKER_TIMEOUT": "30",
                "AUTO_SCALING_ENABLED": "true",
                "MIN_REPLICAS": "2",
                "MAX_REPLICAS": "10",
                "CPU_THRESHOLD": "70",
                "MEMORY_THRESHOLD": "80",
                "CACHE_ENABLED": "true",
                "CACHE_TTL": "3600",
                "CACHE_STRATEGY": "LRU",
                "RATE_LIMIT_ENABLED": "true",
                "RATE_LIMIT_REQUESTS": "100",
                "RATE_LIMIT_WINDOW": "60",
                "MONITORING_ENABLED": "true",
                "METRICS_ENABLED": "true",
                "CORS_ENABLED": "true",
                "ALLOWED_ORIGINS": "http://localhost:3000,http://localhost:8080",
                "MAX_FILE_SIZE": "52428800",
                "FEATURE_AI_ENABLED": "true",
                "FEATURE_ANALYTICS_ENABLED": "true",
                "FEATURE_COLLABORATION_ENABLED": "true"
            },
            database_config={
                "DB_HOST": "postgres-primary.spotify-ai-agent-dev.svc.cluster.local",
                "DB_PORT": "5432",
                "DB_NAME": "spotify_ai_agent_dev",
                "DB_USER": "spotify_user",
                "DB_POOL_SIZE": "20",
                "DB_MAX_OVERFLOW": "30",
                "DB_POOL_TIMEOUT": "30",
                "DB_ECHO": "true",
                "DB_READ_HOST": "postgres-replica.spotify-ai-agent-dev.svc.cluster.local",
                "DB_READ_ENABLED": "true",
                "REDIS_HOST": "redis.spotify-ai-agent-dev.svc.cluster.local",
                "REDIS_PORT": "6379",
                "REDIS_DB": "0",
                "REDIS_MAX_CONNECTIONS": "100",
                "MONGO_HOST": "mongodb.spotify-ai-agent-dev.svc.cluster.local",
                "MONGO_PORT": "27017",
                "MONGO_DATABASE": "spotify_ai_analytics",
                "ELASTICSEARCH_HOST": "elasticsearch.spotify-ai-agent-dev.svc.cluster.local",
                "ELASTICSEARCH_PORT": "9200"
            },
            security_config={
                "JWT_ACCESS_TOKEN_EXPIRE_MINUTES": "30",
                "JWT_REFRESH_TOKEN_EXPIRE_DAYS": "7",
                "OAUTH_ENABLED": "true",
                "OAUTH_PROVIDERS": "google,spotify,github",
                "API_KEY_ENABLED": "true",
                "RATE_LIMIT_ENABLED": "true",
                "API_RATE_LIMIT_REQUESTS": "1000",
                "HTTPS_ONLY": "false",
                "SECURE_COOKIES": "false",
                "CSRF_PROTECTION": "true",
                "PASSWORD_MIN_LENGTH": "8",
                "MAX_LOGIN_ATTEMPTS": "5",
                "ACCOUNT_LOCKOUT_DURATION": "900",
                "TWO_FACTOR_ENABLED": "false"
            },
            ml_config={
                "ML_TRAINING_ENABLED": "true",
                "ML_BATCH_SIZE": "32",
                "ML_TRAINING_EPOCHS": "10",
                "ML_LEARNING_RATE": "0.001",
                "ML_GPU_ENABLED": "false",
                "ML_CPU_THREADS": "4",
                "ML_INFERENCE_TIMEOUT": "30",
                "SPLEETER_CACHE_ENABLED": "true",
                "SPLEETER_MAX_DURATION": "600",
                "AUDIO_SAMPLE_RATE": "44100",
                "AI_RECOMMENDATION_ENABLED": "true"
            },
            monitoring_config={
                "PROMETHEUS_ENABLED": "true",
                "PROMETHEUS_PORT": "9090",
                "GRAFANA_ENABLED": "true",
                "GRAFANA_PORT": "3000",
                "JAEGER_ENABLED": "true",
                "JAEGER_SAMPLER_PARAM": "1",
                "LOG_FORMAT": "json",
                "LOG_LEVEL": "DEBUG",
                "LOG_ROTATION": "daily",
                "HEALTH_CHECK_ENABLED": "true",
                "RESOURCE_MONITORING_ENABLED": "true",
                "CPU_ALERT_THRESHOLD": "80",
                "MEMORY_ALERT_THRESHOLD": "85"
            }
        )
    
    def _create_staging_profile(self) -> EnvironmentProfile:
        """Profil pour environnement de staging."""
        return EnvironmentProfile(
            name="Staging Environment",
            description="Configuration pour environnement de staging, proche de la production avec monitoring avancé",
            application_config={
                "DEBUG": "false",
                "LOG_LEVEL": "INFO",
                "ENVIRONMENT": "staging",
                "API_VERSION": "v2",
                "HOST": "0.0.0.0",
                "PORT": "8000",
                "MAX_WORKERS": "8",
                "WORKER_TIMEOUT": "60",
                "AUTO_SCALING_ENABLED": "true",
                "MIN_REPLICAS": "3",
                "MAX_REPLICAS": "20",
                "CPU_THRESHOLD": "60",
                "MEMORY_THRESHOLD": "70",
                "CACHE_ENABLED": "true",
                "CACHE_TTL": "7200",
                "CACHE_STRATEGY": "LRU",
                "RATE_LIMIT_ENABLED": "true",
                "RATE_LIMIT_REQUESTS": "500",
                "RATE_LIMIT_WINDOW": "60",
                "MONITORING_ENABLED": "true",
                "METRICS_ENABLED": "true",
                "CORS_ENABLED": "true",
                "ALLOWED_ORIGINS": "https://staging.spotify-ai-agent.com",
                "MAX_FILE_SIZE": "104857600",  # 100MB
                "FEATURE_AI_ENABLED": "true",
                "FEATURE_ANALYTICS_ENABLED": "true",
                "FEATURE_COLLABORATION_ENABLED": "true"
            },
            database_config={
                "DB_HOST": "postgres-primary.spotify-ai-agent-staging.svc.cluster.local",
                "DB_PORT": "5432",
                "DB_NAME": "spotify_ai_agent_staging",
                "DB_USER": "spotify_user",
                "DB_POOL_SIZE": "50",
                "DB_MAX_OVERFLOW": "100",
                "DB_POOL_TIMEOUT": "60",
                "DB_ECHO": "false",
                "DB_READ_HOST": "postgres-replica.spotify-ai-agent-staging.svc.cluster.local",
                "DB_READ_ENABLED": "true",
                "REDIS_HOST": "redis.spotify-ai-agent-staging.svc.cluster.local",
                "REDIS_PORT": "6379",
                "REDIS_DB": "0",
                "REDIS_MAX_CONNECTIONS": "200",
                "MONGO_HOST": "mongodb.spotify-ai-agent-staging.svc.cluster.local",
                "MONGO_PORT": "27017",
                "MONGO_DATABASE": "spotify_ai_analytics",
                "ELASTICSEARCH_HOST": "elasticsearch.spotify-ai-agent-staging.svc.cluster.local",
                "ELASTICSEARCH_PORT": "9200"
            },
            security_config={
                "JWT_ACCESS_TOKEN_EXPIRE_MINUTES": "15",
                "JWT_REFRESH_TOKEN_EXPIRE_DAYS": "7",
                "OAUTH_ENABLED": "true",
                "OAUTH_PROVIDERS": "google,spotify,github",
                "API_KEY_ENABLED": "true",
                "RATE_LIMIT_ENABLED": "true",
                "API_RATE_LIMIT_REQUESTS": "5000",
                "HTTPS_ONLY": "true",
                "SECURE_COOKIES": "true",
                "CSRF_PROTECTION": "true",
                "PASSWORD_MIN_LENGTH": "10",
                "MAX_LOGIN_ATTEMPTS": "3",
                "ACCOUNT_LOCKOUT_DURATION": "1800",
                "TWO_FACTOR_ENABLED": "true"
            },
            ml_config={
                "ML_TRAINING_ENABLED": "true",
                "ML_BATCH_SIZE": "64",
                "ML_TRAINING_EPOCHS": "20",
                "ML_LEARNING_RATE": "0.0005",
                "ML_GPU_ENABLED": "true",
                "ML_CPU_THREADS": "8",
                "ML_INFERENCE_TIMEOUT": "20",
                "SPLEETER_CACHE_ENABLED": "true",
                "SPLEETER_MAX_DURATION": "900",
                "AUDIO_SAMPLE_RATE": "44100",
                "AI_RECOMMENDATION_ENABLED": "true"
            },
            monitoring_config={
                "PROMETHEUS_ENABLED": "true",
                "PROMETHEUS_PORT": "9090",
                "GRAFANA_ENABLED": "true",
                "GRAFANA_PORT": "3000",
                "JAEGER_ENABLED": "true",
                "JAEGER_SAMPLER_PARAM": "0.1",
                "LOG_FORMAT": "json",
                "LOG_LEVEL": "INFO",
                "LOG_ROTATION": "daily",
                "HEALTH_CHECK_ENABLED": "true",
                "RESOURCE_MONITORING_ENABLED": "true",
                "CPU_ALERT_THRESHOLD": "70",
                "MEMORY_ALERT_THRESHOLD": "75",
                "ALERTING_ENABLED": "true"
            }
        )
    
    def _create_production_profile(self) -> EnvironmentProfile:
        """Profil pour environnement de production."""
        return EnvironmentProfile(
            name="Production Environment",
            description="Configuration optimisée pour la production avec haute disponibilité et sécurité maximale",
            application_config={
                "DEBUG": "false",
                "LOG_LEVEL": "WARNING",
                "ENVIRONMENT": "production",
                "API_VERSION": "v2",
                "HOST": "0.0.0.0",
                "PORT": "8000",
                "MAX_WORKERS": "16",
                "WORKER_TIMEOUT": "120",
                "AUTO_SCALING_ENABLED": "true",
                "MIN_REPLICAS": "5",
                "MAX_REPLICAS": "100",
                "CPU_THRESHOLD": "50",
                "MEMORY_THRESHOLD": "60",
                "CACHE_ENABLED": "true",
                "CACHE_TTL": "14400",  # 4 hours
                "CACHE_STRATEGY": "LRU",
                "RATE_LIMIT_ENABLED": "true",
                "RATE_LIMIT_REQUESTS": "1000",
                "RATE_LIMIT_WINDOW": "60",
                "MONITORING_ENABLED": "true",
                "METRICS_ENABLED": "true",
                "CORS_ENABLED": "true",
                "ALLOWED_ORIGINS": "https://spotify-ai-agent.com,https://www.spotify-ai-agent.com",
                "MAX_FILE_SIZE": "209715200",  # 200MB
                "FEATURE_AI_ENABLED": "true",
                "FEATURE_ANALYTICS_ENABLED": "true",
                "FEATURE_COLLABORATION_ENABLED": "true",
                "BACKUP_ENABLED": "true"
            },
            database_config={
                "DB_HOST": "postgres-primary.spotify-ai-agent-prod.svc.cluster.local",
                "DB_PORT": "5432",
                "DB_NAME": "spotify_ai_agent_prod",
                "DB_USER": "spotify_user",
                "DB_POOL_SIZE": "100",
                "DB_MAX_OVERFLOW": "200",
                "DB_POOL_TIMEOUT": "120",
                "DB_ECHO": "false",
                "DB_READ_HOST": "postgres-replica.spotify-ai-agent-prod.svc.cluster.local",
                "DB_READ_ENABLED": "true",
                "REDIS_HOST": "redis.spotify-ai-agent-prod.svc.cluster.local",
                "REDIS_PORT": "6379",
                "REDIS_DB": "0",
                "REDIS_MAX_CONNECTIONS": "500",
                "MONGO_HOST": "mongodb.spotify-ai-agent-prod.svc.cluster.local",
                "MONGO_PORT": "27017",
                "MONGO_DATABASE": "spotify_ai_analytics",
                "ELASTICSEARCH_HOST": "elasticsearch.spotify-ai-agent-prod.svc.cluster.local",
                "ELASTICSEARCH_PORT": "9200"
            },
            security_config={
                "JWT_ACCESS_TOKEN_EXPIRE_MINUTES": "10",
                "JWT_REFRESH_TOKEN_EXPIRE_DAYS": "3",
                "OAUTH_ENABLED": "true",
                "OAUTH_PROVIDERS": "google,spotify,github",
                "API_KEY_ENABLED": "true",
                "RATE_LIMIT_ENABLED": "true",
                "API_RATE_LIMIT_REQUESTS": "10000",
                "HTTPS_ONLY": "true",
                "SECURE_COOKIES": "true",
                "CSRF_PROTECTION": "true",
                "PASSWORD_MIN_LENGTH": "12",
                "MAX_LOGIN_ATTEMPTS": "3",
                "ACCOUNT_LOCKOUT_DURATION": "3600",
                "TWO_FACTOR_ENABLED": "true",
                "AUDIT_LOG_ENABLED": "true"
            },
            ml_config={
                "ML_TRAINING_ENABLED": "false",  # Training in dedicated environment
                "ML_BATCH_SIZE": "128",
                "ML_GPU_ENABLED": "true",
                "ML_CPU_THREADS": "16",
                "ML_INFERENCE_TIMEOUT": "10",
                "SPLEETER_CACHE_ENABLED": "true",
                "SPLEETER_MAX_DURATION": "1200",
                "AUDIO_SAMPLE_RATE": "44100",
                "AI_RECOMMENDATION_ENABLED": "true",
                "ML_REGISTRY_ENABLED": "true"
            },
            monitoring_config={
                "PROMETHEUS_ENABLED": "true",
                "PROMETHEUS_PORT": "9090",
                "GRAFANA_ENABLED": "true",
                "GRAFANA_PORT": "3000",
                "JAEGER_ENABLED": "true",
                "JAEGER_SAMPLER_PARAM": "0.01",  # 1% sampling
                "LOG_FORMAT": "json",
                "LOG_LEVEL": "WARNING",
                "LOG_ROTATION": "daily",
                "LOG_BACKUP_COUNT": "30",
                "HEALTH_CHECK_ENABLED": "true",
                "RESOURCE_MONITORING_ENABLED": "true",
                "CPU_ALERT_THRESHOLD": "60",
                "MEMORY_ALERT_THRESHOLD": "65",
                "DISK_ALERT_THRESHOLD": "80",
                "ALERTING_ENABLED": "true",
                "APM_ENABLED": "true"
            }
        )
    
    def get_profile(self, environment: str) -> EnvironmentProfile:
        """Récupère un profil d'environnement."""
        if environment not in self.profiles:
            raise ValueError(f"Profil d'environnement non trouvé: {environment}")
        return self.profiles[environment]
    
    def list_profiles(self) -> Dict[str, str]:
        """Liste tous les profils disponibles."""
        return {
            name: profile.description 
            for name, profile in self.profiles.items()
        }
    
    def merge_with_overrides(self, 
                           environment: str, 
                           overrides: Dict[str, Dict[str, Any]]) -> EnvironmentProfile:
        """Fusionne un profil avec des surcharges personnalisées."""
        profile = self.get_profile(environment)
        
        # Créer une copie du profil
        merged_profile = EnvironmentProfile(
            name=f"{profile.name} (Custom)",
            description=f"{profile.description} avec surcharges personnalisées",
            application_config=profile.application_config.copy(),
            database_config=profile.database_config.copy(),
            security_config=profile.security_config.copy(),
            ml_config=profile.ml_config.copy(),
            monitoring_config=profile.monitoring_config.copy()
        )
        
        # Appliquer les surcharges
        for config_type, config_overrides in overrides.items():
            if hasattr(merged_profile, f"{config_type}_config"):
                config_dict = getattr(merged_profile, f"{config_type}_config")
                config_dict.update(config_overrides)
        
        return merged_profile
    
    def export_profile_to_configmaps(self, environment: str) -> Dict[str, Dict[str, Any]]:
        """Exporte un profil sous forme de ConfigMaps Kubernetes."""
        profile = self.get_profile(environment)
        
        return {
            "application": profile.application_config,
            "database": profile.database_config,
            "security": profile.security_config,
            "ml": profile.ml_config,
            "monitoring": profile.monitoring_config
        }

# Classe d'utilité pour la comparaison de profils
class ProfileComparator:
    """Compare les différences entre profils d'environnement."""
    
    @staticmethod
    def compare_profiles(profile1: EnvironmentProfile, 
                        profile2: EnvironmentProfile) -> Dict[str, Dict[str, Any]]:
        """Compare deux profils et retourne les différences."""
        differences = {}
        
        config_types = ['application_config', 'database_config', 'security_config', 
                       'ml_config', 'monitoring_config']
        
        for config_type in config_types:
            config1 = getattr(profile1, config_type)
            config2 = getattr(profile2, config_type)
            
            diff = {}
            all_keys = set(config1.keys()) | set(config2.keys())
            
            for key in all_keys:
                val1 = config1.get(key)
                val2 = config2.get(key)
                
                if val1 != val2:
                    diff[key] = {
                        profile1.name: val1,
                        profile2.name: val2
                    }
            
            if diff:
                differences[config_type] = diff
        
        return differences

# Exportation des classes
__all__ = [
    'EnvironmentType',
    'EnvironmentProfile', 
    'EnvironmentProfileManager',
    'ProfileComparator'
]
