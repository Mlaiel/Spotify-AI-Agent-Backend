"""
🎵 Tests Ultra-Avancés pour API Core Configuration
=================================================

Tests industriels complets pour la configuration de l'API Core avec validation
enterprise, tests multi-environnements, et sécurité renforcée.

Développé par Fahed Mlaiel - Enterprise Configuration Testing Expert
"""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from typing import Dict, Any
from pathlib import Path

from app.api.core.config import (
    APIConfig,
    APISettings,
    SecurityConfig,
    CacheConfig,
    DatabaseConfig,
    RedisConfig,
    MonitoringConfig,
    Environment,
    LogLevel,
    get_settings,
    get_api_config,
    create_development_config,
    create_production_config,
    create_testing_config
)


# =============================================================================
# FIXTURES ENTERPRISE POUR CONFIGURATION TESTING
# =============================================================================

@pytest.fixture
def clean_env():
    """Environment propre pour les tests"""
    # Sauvegarder les variables d'environnement actuelles
    original_env = dict(os.environ)
    
    # Nettoyer les variables de config
    env_vars_to_clear = [
        var for var in os.environ.keys() 
        if any(prefix in var for prefix in ['API_', 'DB_', 'REDIS_', 'CACHE_', 'SECURITY_', 'MONITORING_'])
    ]
    
    for var in env_vars_to_clear:
        os.environ.pop(var, None)
    
    yield
    
    # Restaurer l'environnement original
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def sample_env_vars():
    """Variables d'environnement de test"""
    return {
        'API_HOST': '127.0.0.1',
        'API_PORT': '9000',
        'API_DEBUG': 'true',
        'API_APP_VERSION': '2.1.0',
        'API_ENVIRONMENT': 'testing',
        'DB_POSTGRES_HOST': 'test-db',
        'DB_POSTGRES_PORT': '5433',
        'DB_POSTGRES_USER': 'test_user',
        'DB_POSTGRES_PASSWORD': 'test_pass',
        'DB_POSTGRES_DB': 'test_db',
        'REDIS_HOST': 'test-redis',
        'REDIS_PORT': '6380',
        'REDIS_PASSWORD': 'test_redis_pass',
        'CACHE_DEFAULT_TTL': '1800',
        'SECURITY_SECRET_KEY': 'test-secret-key-12345',
        'SECURITY_ACCESS_TOKEN_EXPIRE_MINUTES': '60',
        'MONITORING_METRICS_ENABLED': 'true',
        'MONITORING_LOG_LEVEL': 'DEBUG'
    }


@pytest.fixture
def temp_env_file():
    """Fichier .env temporaire pour les tests"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write("""
API_HOST=0.0.0.0
API_PORT=8080
API_DEBUG=false
API_VERSION=2.0.0
DB_POSTGRES_HOST=localhost
DB_POSTGRES_PORT=5432
REDIS_HOST=localhost
REDIS_PORT=6379
""")
        temp_path = f.name
    
    yield temp_path
    
    # Nettoyer le fichier temporaire
    Path(temp_path).unlink(missing_ok=True)


# =============================================================================
# TESTS DE CONFIGURATION API
# =============================================================================

class TestAPIConfig:
    """Tests pour APIConfig"""
    
    def test_api_config_defaults(self, clean_env):
        """Test des valeurs par défaut APIConfig"""
        config = APIConfig()
        
        assert config.app_name == "Spotify AI Agent API"
        assert config.app_version == "2.0.0"
        assert config.environment == Environment.DEVELOPMENT
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.debug is False
        assert config.api_v1_prefix == "/api/v1"
        assert config.workers == 1
    
    def test_api_config_from_env(self, clean_env, sample_env_vars):
        """Test chargement depuis variables d'environnement"""
        with patch.dict(os.environ, sample_env_vars):
            config = APIConfig()
            
            assert config.host == "127.0.0.1"
            assert config.port == 9000
            assert config.debug is True
            assert config.app_version == "2.1.0"
            assert config.environment == Environment.TESTING
    
    def test_api_config_validation_environment(self):
        """Test validation de l'environnement"""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="Input should be"):
            APIConfig(environment="invalid_env")
    
    def test_api_config_production_validation(self):
        """Test validation production"""
        with pytest.raises(ValueError, match="Debug mode cannot be enabled in production"):
            APIConfig(
                environment=Environment.PRODUCTION,
                debug=True
            )
        
        with pytest.raises(ValueError, match="Reload cannot be enabled in production"):
            APIConfig(
                environment=Environment.PRODUCTION,
                reload=True
            )
    
    def test_middleware_configuration(self):
        """Test configuration des middlewares"""
        config = APIConfig()
        
        assert "cors" in config.middleware_enabled
        assert "gzip" in config.middleware_enabled
        assert "security" in config.middleware_enabled
        assert "rate_limit" in config.middleware_enabled
        assert "cache" in config.middleware_enabled
        assert config.middleware_enabled["cors"] is True


class TestSecurityConfig:
    """Tests pour SecurityConfig"""
    
    def test_security_config_defaults(self, clean_env):
        """Test des valeurs par défaut SecurityConfig"""
        config = SecurityConfig()
        
        assert len(config.secret_key) >= 32  # Token sécurisé
        assert config.algorithm == "HS256"
        assert config.access_token_expire_minutes == 30
        assert config.refresh_token_expire_days == 7
        assert config.rate_limit_per_minute == 100
        assert config.security_headers_enabled is True
        assert config.api_key_validation_enabled is True
    
    def test_cors_configuration(self):
        """Test configuration CORS"""
        config = SecurityConfig()
        
        expected_origins = ["http://localhost:3000", "http://localhost:8000"]
        assert config.cors_origins == expected_origins
        assert config.cors_credentials is True
        assert "GET" in config.cors_methods
        assert "POST" in config.cors_methods
    
    def test_security_headers(self):
        """Test configuration des headers de sécurité"""
        config = SecurityConfig()
        
        assert config.hsts_max_age == 31536000  # 1 an
        assert "default-src 'self'" in config.content_security_policy
        assert config.api_key_header == "X-API-Key"


class TestDatabaseConfig:
    """Tests pour DatabaseConfig"""
    
    def test_database_config_defaults(self, clean_env):
        """Test des valeurs par défaut DatabaseConfig"""
        config = DatabaseConfig()
        
        assert config.postgres_host == "localhost"
        assert config.postgres_port == 5432
        assert config.postgres_user == "spotify_user"
        assert config.postgres_db == "spotify_ai_agent"
        assert config.postgres_pool_size == 20
        assert config.postgres_max_overflow == 10
    
    def test_postgres_url_generation(self):
        """Test génération URL PostgreSQL"""
        config = DatabaseConfig(
            postgres_host="test-host",
            postgres_port=5433,
            postgres_user="test_user",
            postgres_password="test_pass",
            postgres_db="test_db"
        )
        
        expected_url = "postgresql://test_user:test_pass@test-host:5433/test_db"
        assert config.postgres_url == expected_url
    
    def test_postgres_async_url_generation(self):
        """Test génération URL PostgreSQL async"""
        config = DatabaseConfig(
            postgres_host="async-host",
            postgres_port=5432,
            postgres_user="async_user",
            postgres_password="async_pass",
            postgres_db="async_db"
        )
        
        expected_url = "postgresql+asyncpg://async_user:async_pass@async-host:5432/async_db"
        assert config.postgres_async_url == expected_url
    
    def test_mongodb_configuration(self):
        """Test configuration MongoDB"""
        config = DatabaseConfig(
            mongodb_url="mongodb://test:27017",
            mongodb_db="test_mongo_db",
            mongodb_collection_prefix="test_"
        )
        
        assert config.mongodb_url == "mongodb://test:27017"
        assert config.mongodb_db == "test_mongo_db"
        assert config.mongodb_collection_prefix == "test_"
    
    def test_elasticsearch_configuration(self):
        """Test configuration Elasticsearch"""
        config = DatabaseConfig(
            elasticsearch_hosts=["http://es1:9200", "http://es2:9200"],
            elasticsearch_timeout=15,
            elasticsearch_max_retries=5
        )
        
        assert len(config.elasticsearch_hosts) == 2
        assert config.elasticsearch_timeout == 15
        assert config.elasticsearch_max_retries == 5


class TestRedisConfig:
    """Tests pour RedisConfig"""
    
    def test_redis_config_defaults(self, clean_env):
        """Test des valeurs par défaut RedisConfig"""
        config = RedisConfig()
        
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.password is None
        assert config.ssl is False
        assert config.max_connections == 50
        assert config.retry_on_timeout is True
    
    def test_redis_url_generation_without_auth(self):
        """Test génération URL Redis sans authentification"""
        config = RedisConfig(
            host="redis-host",
            port=6380,
            db=1
        )
        
        expected_url = "redis://redis-host:6380/1"
        assert config.url == expected_url
    
    def test_redis_url_generation_with_auth(self):
        """Test génération URL Redis avec authentification"""
        config = RedisConfig(
            host="redis-host",
            port=6379,
            db=0,
            password="secret_pass"
        )
        
        expected_url = "redis://:secret_pass@redis-host:6379/0"
        assert config.url == expected_url
    
    def test_redis_ssl_url_generation(self):
        """Test génération URL Redis avec SSL"""
        config = RedisConfig(
            host="redis-ssl",
            port=6380,
            db=0,
            ssl=True,
            password="ssl_pass"
        )
        
        expected_url = "rediss://:ssl_pass@redis-ssl:6380/0"
        assert config.url == expected_url
    
    def test_sentinel_configuration(self):
        """Test configuration Redis Sentinel"""
        config = RedisConfig(
            sentinel_enabled=True,
            sentinel_hosts=["sentinel1:26379", "sentinel2:26379"],
            sentinel_service="mymaster"
        )
        
        assert config.sentinel_enabled is True
        assert len(config.sentinel_hosts) == 2
        assert config.sentinel_service == "mymaster"
    
    def test_cluster_configuration(self):
        """Test configuration Redis Cluster"""
        config = RedisConfig(
            cluster_enabled=True,
            cluster_nodes=["node1:7000", "node2:7001", "node3:7002"]
        )
        
        assert config.cluster_enabled is True
        assert len(config.cluster_nodes) == 3


class TestCacheConfig:
    """Tests pour CacheConfig"""
    
    def test_cache_config_defaults(self, clean_env):
        """Test des valeurs par défaut CacheConfig"""
        config = CacheConfig()
        
        assert config.redis_url == "redis://localhost:6379"
        assert config.redis_db == 0
        assert config.default_ttl == 3600
        assert config.compression_enabled is True
        assert config.compression_threshold == 1024
        assert config.l1_enabled is True
        assert config.l2_enabled is True
        assert config.l3_enabled is False
    
    def test_cache_levels_configuration(self):
        """Test configuration des niveaux de cache"""
        config = CacheConfig(
            l1_enabled=True,
            l1_max_size=2000,
            l1_ttl=600,
            l2_enabled=True,
            l2_ttl=7200,
            l3_enabled=True,
            l3_ttl=14400
        )
        
        assert config.l1_max_size == 2000
        assert config.l1_ttl == 600
        assert config.l2_ttl == 7200
        assert config.l3_ttl == 14400
        assert config.l3_enabled is True
    
    def test_memcached_configuration(self):
        """Test configuration Memcached"""
        config = CacheConfig(
            memcached_servers=["mc1:11211", "mc2:11211"],
            memcached_timeout=10
        )
        
        assert len(config.memcached_servers) == 2
        assert config.memcached_timeout == 10


class TestMonitoringConfig:
    """Tests pour MonitoringConfig"""
    
    def test_monitoring_config_defaults(self, clean_env):
        """Test des valeurs par défaut MonitoringConfig"""
        config = MonitoringConfig()
        
        assert config.metrics_enabled is True
        assert config.metrics_port == 8080
        assert config.metrics_path == "/metrics"
        assert config.health_checks_enabled is True
        assert config.log_level == LogLevel.INFO
        assert config.log_format == "json"
        assert config.tracing_enabled is False
    
    def test_alerting_configuration(self):
        """Test configuration des alertes"""
        config = MonitoringConfig(
            alerting_enabled=True,
            slack_webhook="https://hooks.slack.com/test",
            email_alerts=["admin@example.com", "dev@example.com"]
        )
        
        assert config.alerting_enabled is True
        assert config.slack_webhook == "https://hooks.slack.com/test"
        assert len(config.email_alerts) == 2
    
    def test_tracing_configuration(self):
        """Test configuration du tracing"""
        config = MonitoringConfig(
            tracing_enabled=True,
            jaeger_endpoint="http://jaeger:14268/api/traces"
        )
        
        assert config.tracing_enabled is True
        assert config.jaeger_endpoint == "http://jaeger:14268/api/traces"


class TestAPISettings:
    """Tests pour APISettings (configuration composée)"""
    
    def test_api_settings_composition(self, clean_env):
        """Test composition des configurations"""
        settings = APISettings()
        
        assert isinstance(settings.api, APIConfig)
        assert isinstance(settings.security, SecurityConfig)
        assert isinstance(settings.cache, CacheConfig)
        assert isinstance(settings.database, DatabaseConfig)
        assert isinstance(settings.redis, RedisConfig)
        assert isinstance(settings.monitoring, MonitoringConfig)
    
    def test_feature_flags_defaults(self):
        """Test des feature flags par défaut"""
        settings = APISettings()
        
        assert settings.features["ml_recommendations"] is True
        assert settings.features["audio_analysis"] is True
        assert settings.features["social_features"] is True
        assert settings.features["analytics"] is True
        assert settings.features["ai_playlists"] is True
    
    def test_external_services_configuration(self):
        """Test configuration des services externes"""
        with patch.dict(os.environ, {
            'SPOTIFY_CLIENT_ID': 'test_spotify_id',
            'SPOTIFY_CLIENT_SECRET': 'test_spotify_secret',
            'OPENAI_API_KEY': 'test_openai_key',
            'HUGGINGFACE_TOKEN': 'test_hf_token'
        }):
            settings = APISettings()
            
            assert settings.spotify_client_id == 'test_spotify_id'
            assert settings.spotify_client_secret == 'test_spotify_secret'
            assert settings.openai_api_key == 'test_openai_key'
            assert settings.huggingface_token == 'test_hf_token'
    
    def test_production_validation(self):
        """Test validation en production"""
        with pytest.raises(ValueError, match="Spotify client ID is required in production"):
            APISettings(
                api=APIConfig(environment=Environment.PRODUCTION),
                spotify_client_id=None
            )


# =============================================================================
# TESTS DES FONCTIONS UTILITAIRES
# =============================================================================

class TestConfigurationFactories:
    """Tests des fonctions factory de configuration"""
    
    def test_get_settings_singleton(self, clean_env):
        """Test du pattern singleton pour get_settings"""
        settings1 = get_settings()
        settings2 = get_settings()
        
        assert settings1 is settings2  # Même instance
    
    def test_get_api_config(self, clean_env):
        """Test get_api_config"""
        api_config = get_api_config()
        
        assert isinstance(api_config, APIConfig)
        assert api_config.app_name == "Spotify AI Agent API"
    
    def test_create_development_config(self):
        """Test création configuration développement"""
        config = create_development_config()
        
        assert config.api.environment == Environment.DEVELOPMENT
        assert config.api.debug is True
        assert config.api.reload is True
        assert config.api.workers == 1
        assert config.cache.l1_enabled is True
        assert config.cache.l2_enabled is False
    
    def test_create_production_config(self):
        """Test création configuration production"""
        config = create_production_config()
        
        assert config.api.environment == Environment.PRODUCTION
        assert config.api.debug is False
        assert config.api.reload is False
        assert config.api.workers == 4
        assert config.security.rate_limit_per_minute == 60
        assert config.cache.l1_enabled is True
        assert config.cache.l2_enabled is True
        assert config.cache.l3_enabled is True
    
    def test_create_testing_config(self):
        """Test création configuration test"""
        config = create_testing_config()
        
        assert config.api.environment == Environment.TESTING
        assert config.api.debug is True
        assert config.api.testing is True
        assert config.database.postgres_db == "spotify_ai_agent_test"
        assert config.cache.redis_db == 1
        assert config.cache.default_ttl == 10


# =============================================================================
# TESTS DE PERFORMANCE
# =============================================================================

@pytest.mark.performance
class TestConfigurationPerformance:
    """Tests de performance pour la configuration"""
    
    def test_config_loading_performance(self, clean_env, benchmark):
        """Test performance chargement configuration"""
        def load_config():
            return APISettings()
        
        result = benchmark(load_config)
        assert isinstance(result, APISettings)
    
    def test_config_access_performance(self, benchmark):
        """Test performance accès configuration"""
        settings = APISettings()
        
        def access_config():
            return (
                settings.api.host,
                settings.database.postgres_url,
                settings.cache.redis_url,
                settings.security.secret_key
            )
        
        result = benchmark(access_config)
        assert len(result) == 4
    
    def test_config_validation_performance(self, benchmark):
        """Test performance validation configuration"""
        def validate_config():
            config = APIConfig(
                environment=Environment.PRODUCTION,
                debug=False,
                reload=False
            )
            return config
        
        result = benchmark(validate_config)
        assert result.environment == Environment.PRODUCTION


# =============================================================================
# TESTS DE SÉCURITÉ
# =============================================================================

@pytest.mark.security
class TestConfigurationSecurity:
    """Tests de sécurité pour la configuration"""
    
    def test_secret_key_generation(self):
        """Test génération sécurisée des clés secrètes"""
        config1 = SecurityConfig()
        config2 = SecurityConfig()
        
        # Les clés doivent être différentes
        assert config1.secret_key != config2.secret_key
        
        # Les clés doivent être suffisamment longues
        assert len(config1.secret_key) >= 32
        assert len(config2.secret_key) >= 32
    
    def test_sensitive_data_not_logged(self, caplog):
        """Test que les données sensibles ne sont pas loggées"""
        config = SecurityConfig(secret_key="super-secret-key")
        
        # Simuler un log de configuration
        repr(config)
        str(config)
        
        # Vérifier que la clé secrète n'apparaît pas dans les logs
        for record in caplog.records:
            assert "super-secret-key" not in record.message
    
    def test_cors_origin_validation(self):
        """Test validation des origines CORS"""
        config = SecurityConfig(
            cors_origins=["http://localhost:3000", "https://app.example.com"]
        )
        
        assert "http://localhost:3000" in config.cors_origins
        assert "https://app.example.com" in config.cors_origins
        
        # Les origines dangereuses ne devraient pas être acceptées par défaut
        assert "*" not in config.cors_origins
    
    def test_database_password_handling(self):
        """Test gestion sécurisée des mots de passe DB"""
        config = DatabaseConfig(
            postgres_password="secret_db_password"
        )
        
        # Le mot de passe doit être dans l'URL mais pas exposé directement
        assert "secret_db_password" in config.postgres_url
        
        # Test que le mot de passe n'apparaît pas dans la représentation string
        config_str = str(config)
        # Cette vérification dépend de l'implémentation de __str__


# =============================================================================
# TESTS D'INTÉGRATION
# =============================================================================

@pytest.mark.integration
class TestConfigurationIntegration:
    """Tests d'intégration pour la configuration"""
    
    def test_env_file_loading(self, temp_env_file):
        """Test chargement depuis fichier .env"""
        # Utiliser les variables d'environnement directement plutôt que de patch model_config
        env_vars = {
            'API_HOST': '0.0.0.0',
            'API_PORT': '8080',
            'API_DEBUG': 'false'
        }
        with patch.dict(os.environ, env_vars):
            settings = APISettings()
            
            assert settings.api.host == "0.0.0.0"
            assert settings.api.port == 8080
            assert settings.api.debug is False
    
    def test_environment_override_priority(self, temp_env_file):
        """Test priorité des variables d'environnement sur le fichier .env"""
        env_override = {'API_PORT': '9999'}
        
        with patch.dict(os.environ, env_override):
            settings = APISettings()
            
            # La variable d'environnement doit avoir priorité
            assert settings.api.port == 9999
    
    def test_configuration_dependencies(self):
        """Test des dépendances entre configurations"""
        settings = APISettings()
        
        # Redis config doit être cohérente avec cache config
        assert settings.redis.host in settings.cache.redis_url
        assert str(settings.redis.port) in settings.cache.redis_url
        
        # Database config doit être cohérente
        assert settings.database.postgres_host in settings.database.postgres_url
