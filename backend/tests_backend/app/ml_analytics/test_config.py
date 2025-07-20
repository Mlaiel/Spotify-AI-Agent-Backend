# 🧪 ML Analytics Configuration Tests
# ====================================
# 
# Tests ultra-avancés pour le système de configuration
# Enterprise configuration management testing
#
# 🎖️ Implementation par l'équipe d'experts:
# ✅ DBA & Data Engineer + Lead Dev + Architecte Backend
#
# 👨‍💻 Développé par: Fahed Mlaiel
# ====================================

"""
🔧 Configuration Management Test Suite
=======================================

Comprehensive testing for configuration system:
- Environment-specific configurations
- Validation and type safety
- Security and secrets management
- Dynamic configuration updates
- Performance optimization settings
"""

import pytest
import os
import tempfile
import json
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import timedelta
import asyncio
from typing import Dict, Any

# Import modules to test
from app.ml_analytics.config import (
    Environment, DatabaseType, CacheType,
    MLAnalyticsConfig, DatabaseConfig, CacheConfig,
    SecurityConfig, MonitoringConfig, MLModelConfig,
    ConfigManager, ConfigValidator, ConfigError,
    load_config_from_file, merge_configs,
    validate_environment_config
)


class TestEnvironmentEnum:
    """Tests pour l'énumération Environment"""
    
    def test_environment_values(self):
        """Test des valeurs d'environnement disponibles"""
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.TESTING.value == "testing"
        assert Environment.STAGING.value == "staging"
        assert Environment.PRODUCTION.value == "production"
    
    def test_environment_from_string(self):
        """Test de création d'environnement depuis string"""
        env = Environment("production")
        assert env == Environment.PRODUCTION
    
    def test_invalid_environment(self):
        """Test d'environnement invalide"""
        with pytest.raises(ValueError):
            Environment("invalid_env")


class TestDatabaseType:
    """Tests pour l'énumération DatabaseType"""
    
    def test_database_types(self):
        """Test des types de base de données supportés"""
        assert DatabaseType.POSTGRESQL.value == "postgresql"
        assert DatabaseType.MYSQL.value == "mysql"
        assert DatabaseType.SQLITE.value == "sqlite"
        assert DatabaseType.MONGODB.value == "mongodb"
    
    def test_database_type_from_url(self):
        """Test d'extraction du type depuis une URL"""
        postgres_url = "postgresql://user:pass@localhost:5432/db"
        assert postgres_url.startswith(DatabaseType.POSTGRESQL.value)


class TestDatabaseConfig:
    """Tests pour la configuration de base de données"""
    
    def test_database_config_creation(self):
        """Test de création de configuration DB"""
        config = DatabaseConfig(
            url="postgresql://user:pass@localhost:5432/spotify_ai",
            pool_size=20,
            max_overflow=10,
            pool_timeout=30,
            echo=False
        )
        
        assert config.url == "postgresql://user:pass@localhost:5432/spotify_ai"
        assert config.pool_size == 20
        assert config.max_overflow == 10
        assert config.pool_timeout == 30
        assert config.echo is False
    
    def test_database_config_defaults(self):
        """Test des valeurs par défaut"""
        config = DatabaseConfig(url="sqlite:///test.db")
        
        assert config.pool_size == 10
        assert config.max_overflow == 20
        assert config.pool_timeout == 30
        assert config.echo is False
    
    def test_database_config_validation(self):
        """Test de validation de configuration DB"""
        # URL invalide
        with pytest.raises(ValueError):
            DatabaseConfig(url="invalid_url")
        
        # Pool size négatif
        with pytest.raises(ValueError):
            DatabaseConfig(
                url="postgresql://localhost/db",
                pool_size=-1
            )
    
    def test_database_url_parsing(self):
        """Test de parsing d'URL de base de données"""
        config = DatabaseConfig(url="postgresql://user:pass@localhost:5432/spotify_ai")
        
        parsed = config.parse_url()
        assert parsed['scheme'] == 'postgresql'
        assert parsed['username'] == 'user'
        assert parsed['password'] == 'pass'
        assert parsed['host'] == 'localhost'
        assert parsed['port'] == 5432
        assert parsed['database'] == 'spotify_ai'


class TestCacheConfig:
    """Tests pour la configuration de cache"""
    
    def test_cache_config_redis(self):
        """Test de configuration Redis"""
        config = CacheConfig(
            type=CacheType.REDIS,
            url="redis://localhost:6379/0",
            ttl=3600,
            max_connections=100
        )
        
        assert config.type == CacheType.REDIS
        assert config.url == "redis://localhost:6379/0"
        assert config.ttl == 3600
        assert config.max_connections == 100
    
    def test_cache_config_memory(self):
        """Test de configuration mémoire"""
        config = CacheConfig(
            type=CacheType.MEMORY,
            max_size=1000,
            ttl=1800
        )
        
        assert config.type == CacheType.MEMORY
        assert config.max_size == 1000
        assert config.ttl == 1800
    
    def test_cache_config_validation(self):
        """Test de validation de configuration cache"""
        # TTL négatif
        with pytest.raises(ValueError):
            CacheConfig(
                type=CacheType.REDIS,
                url="redis://localhost:6379",
                ttl=-1
            )


class TestSecurityConfig:
    """Tests pour la configuration de sécurité"""
    
    def test_security_config_creation(self):
        """Test de création de configuration sécurité"""
        config = SecurityConfig(
            secret_key="super_secret_key_123",
            jwt_algorithm="HS256",
            jwt_expiration=timedelta(hours=24),
            password_min_length=8,
            enable_cors=True,
            allowed_origins=["http://localhost:3000"]
        )
        
        assert config.secret_key == "super_secret_key_123"
        assert config.jwt_algorithm == "HS256"
        assert config.jwt_expiration == timedelta(hours=24)
        assert config.password_min_length == 8
        assert config.enable_cors is True
        assert "http://localhost:3000" in config.allowed_origins
    
    def test_security_config_validation(self):
        """Test de validation de configuration sécurité"""
        # Secret key trop courte
        with pytest.raises(ValueError):
            SecurityConfig(secret_key="short")
        
        # Mot de passe trop court
        with pytest.raises(ValueError):
            SecurityConfig(
                secret_key="valid_secret_key_123",
                password_min_length=3
            )
    
    def test_jwt_token_generation(self):
        """Test de génération de token JWT"""
        config = SecurityConfig(secret_key="test_secret_key_123")
        
        payload = {"user_id": 123, "username": "testuser"}
        token = config.generate_jwt_token(payload)
        
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_jwt_token_validation(self):
        """Test de validation de token JWT"""
        config = SecurityConfig(secret_key="test_secret_key_123")
        
        payload = {"user_id": 123, "username": "testuser"}
        token = config.generate_jwt_token(payload)
        
        decoded = config.validate_jwt_token(token)
        assert decoded["user_id"] == 123
        assert decoded["username"] == "testuser"
    
    def test_password_hashing(self):
        """Test de hachage de mot de passe"""
        config = SecurityConfig(secret_key="test_secret_key_123")
        
        password = "test_password_123"
        hashed = config.hash_password(password)
        
        assert hashed != password
        assert config.verify_password(password, hashed)
        assert not config.verify_password("wrong_password", hashed)


class TestMonitoringConfig:
    """Tests pour la configuration de monitoring"""
    
    def test_monitoring_config_creation(self):
        """Test de création de configuration monitoring"""
        config = MonitoringConfig(
            enabled=True,
            metrics_port=8000,
            health_check_interval=30,
            alert_email="admin@example.com",
            log_level="INFO",
            retention_days=30
        )
        
        assert config.enabled is True
        assert config.metrics_port == 8000
        assert config.health_check_interval == 30
        assert config.alert_email == "admin@example.com"
        assert config.log_level == "INFO"
        assert config.retention_days == 30
    
    def test_monitoring_config_validation(self):
        """Test de validation de configuration monitoring"""
        # Port invalide
        with pytest.raises(ValueError):
            MonitoringConfig(metrics_port=70000)
        
        # Email invalide
        with pytest.raises(ValueError):
            MonitoringConfig(alert_email="invalid_email")
        
        # Niveau de log invalide
        with pytest.raises(ValueError):
            MonitoringConfig(log_level="INVALID")


class TestMLModelConfig:
    """Tests pour la configuration des modèles ML"""
    
    def test_ml_model_config_creation(self):
        """Test de création de configuration ML"""
        config = MLModelConfig(
            model_path="/models",
            max_model_size=1000000000,  # 1GB
            cache_predictions=True,
            batch_size=32,
            max_workers=4,
            timeout=300
        )
        
        assert config.model_path == "/models"
        assert config.max_model_size == 1000000000
        assert config.cache_predictions is True
        assert config.batch_size == 32
        assert config.max_workers == 4
        assert config.timeout == 300
    
    def test_ml_model_config_validation(self):
        """Test de validation de configuration ML"""
        # Batch size invalide
        with pytest.raises(ValueError):
            MLModelConfig(batch_size=0)
        
        # Max workers invalide
        with pytest.raises(ValueError):
            MLModelConfig(max_workers=-1)
        
        # Timeout invalide
        with pytest.raises(ValueError):
            MLModelConfig(timeout=-1)


class TestMLAnalyticsConfig:
    """Tests pour la configuration principale"""
    
    def test_main_config_creation(self):
        """Test de création de configuration principale"""
        config = MLAnalyticsConfig(
            environment=Environment.TESTING,
            debug=True,
            version="1.0.0"
        )
        
        assert config.environment == Environment.TESTING
        assert config.debug is True
        assert config.version == "1.0.0"
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.cache, CacheConfig)
        assert isinstance(config.security, SecurityConfig)
        assert isinstance(config.monitoring, MonitoringConfig)
        assert isinstance(config.ml_models, MLModelConfig)
    
    def test_production_config_validation(self):
        """Test de validation pour production"""
        config = MLAnalyticsConfig(
            environment=Environment.PRODUCTION,
            debug=False
        )
        
        # Debug doit être False en production
        assert config.debug is False
        
        # Secret key ne peut pas être par défaut en production
        with pytest.raises(ValueError):
            config.security.secret_key = "default_secret"
            config.validate_production_config()
    
    def test_config_to_dict(self):
        """Test de conversion en dictionnaire"""
        config = MLAnalyticsConfig(environment=Environment.TESTING)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["environment"] == "testing"
        assert "database" in config_dict
        assert "cache" in config_dict
        assert "security" in config_dict
    
    def test_config_from_dict(self):
        """Test de création depuis dictionnaire"""
        config_dict = {
            "environment": "testing",
            "debug": True,
            "version": "1.0.0",
            "database": {
                "url": "sqlite:///test.db"
            }
        }
        
        config = MLAnalyticsConfig.from_dict(config_dict)
        
        assert config.environment == Environment.TESTING
        assert config.debug is True
        assert config.version == "1.0.0"


class TestConfigManager:
    """Tests pour le gestionnaire de configuration"""
    
    def setup_method(self):
        """Configuration avant chaque test"""
        self.config_manager = ConfigManager()
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "config.yaml"
    
    def teardown_method(self):
        """Nettoyage après chaque test"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_manager_creation(self):
        """Test de création du gestionnaire"""
        assert isinstance(self.config_manager, ConfigManager)
        assert self.config_manager.config is None
    
    def test_load_config_from_file(self):
        """Test de chargement depuis fichier"""
        config_data = {
            "environment": "testing",
            "debug": True,
            "database": {
                "url": "sqlite:///test.db"
            }
        }
        
        # Écrire fichier YAML
        with open(self.config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config = self.config_manager.load_from_file(str(self.config_file))
        
        assert config.environment == Environment.TESTING
        assert config.debug is True
    
    def test_load_config_from_json(self):
        """Test de chargement depuis fichier JSON"""
        config_data = {
            "environment": "testing",
            "debug": True,
            "database": {
                "url": "sqlite:///test.db"
            }
        }
        
        json_file = Path(self.temp_dir) / "config.json"
        with open(json_file, 'w') as f:
            json.dump(config_data, f)
        
        config = self.config_manager.load_from_file(str(json_file))
        
        assert config.environment == Environment.TESTING
        assert config.debug is True
    
    def test_load_config_from_env(self):
        """Test de chargement depuis variables d'environnement"""
        with patch.dict(os.environ, {
            'ML_ANALYTICS_ENVIRONMENT': 'testing',
            'ML_ANALYTICS_DEBUG': 'true',
            'ML_ANALYTICS_DATABASE_URL': 'sqlite:///test.db'
        }):
            config = self.config_manager.load_from_env()
            
            assert config.environment == Environment.TESTING
            assert config.debug is True
    
    def test_config_validation(self):
        """Test de validation de configuration"""
        config = MLAnalyticsConfig(environment=Environment.TESTING)
        
        # Configuration valide
        assert self.config_manager.validate_config(config) is True
        
        # Configuration invalide
        config.database.url = "invalid_url"
        with pytest.raises(ConfigError):
            self.config_manager.validate_config(config)
    
    def test_config_hot_reload(self):
        """Test de rechargement à chaud"""
        config_data = {
            "environment": "testing",
            "debug": True
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Chargement initial
        self.config_manager.load_from_file(str(self.config_file))
        assert self.config_manager.config.debug is True
        
        # Modification du fichier
        config_data["debug"] = False
        with open(self.config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Rechargement
        self.config_manager.reload()
        assert self.config_manager.config.debug is False
    
    @pytest.mark.asyncio
    async def test_async_config_operations(self):
        """Test des opérations asynchrones"""
        config_data = {
            "environment": "testing",
            "debug": True
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Chargement asynchrone
        config = await self.config_manager.load_from_file_async(str(self.config_file))
        assert config.environment == Environment.TESTING
        
        # Validation asynchrone
        is_valid = await self.config_manager.validate_config_async(config)
        assert is_valid is True


class TestConfigValidator:
    """Tests pour le validateur de configuration"""
    
    def setup_method(self):
        """Configuration avant chaque test"""
        self.validator = ConfigValidator()
    
    def test_validate_database_config(self):
        """Test de validation de configuration DB"""
        # Configuration valide
        valid_config = DatabaseConfig(url="postgresql://localhost/db")
        result = self.validator.validate_database_config(valid_config)
        assert result.is_valid is True
        
        # Configuration invalide
        invalid_config = DatabaseConfig(url="invalid_url")
        result = self.validator.validate_database_config(invalid_config)
        assert result.is_valid is False
        assert len(result.errors) > 0
    
    def test_validate_security_config(self):
        """Test de validation de configuration sécurité"""
        # Configuration valide
        valid_config = SecurityConfig(secret_key="valid_secret_key_123")
        result = self.validator.validate_security_config(valid_config)
        assert result.is_valid is True
        
        # Configuration invalide
        invalid_config = SecurityConfig(secret_key="short")
        result = self.validator.validate_security_config(invalid_config)
        assert result.is_valid is False
    
    def test_validate_environment_specific(self):
        """Test de validation spécifique à l'environnement"""
        # Production
        prod_config = MLAnalyticsConfig(environment=Environment.PRODUCTION)
        result = self.validator.validate_environment_specific(prod_config)
        
        # Development
        dev_config = MLAnalyticsConfig(environment=Environment.DEVELOPMENT)
        result = self.validator.validate_environment_specific(dev_config)
        assert result.is_valid is True


class TestConfigUtilities:
    """Tests pour les utilitaires de configuration"""
    
    def test_merge_configs(self):
        """Test de fusion de configurations"""
        base_config = {
            "environment": "testing",
            "debug": True,
            "database": {
                "url": "sqlite:///base.db",
                "pool_size": 10
            }
        }
        
        override_config = {
            "debug": False,
            "database": {
                "pool_size": 20,
                "max_overflow": 30
            },
            "new_field": "new_value"
        }
        
        merged = merge_configs(base_config, override_config)
        
        assert merged["environment"] == "testing"
        assert merged["debug"] is False
        assert merged["database"]["url"] == "sqlite:///base.db"
        assert merged["database"]["pool_size"] == 20
        assert merged["database"]["max_overflow"] == 30
        assert merged["new_field"] == "new_value"
    
    def test_load_config_from_file(self):
        """Test de chargement de fichier de configuration"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                "environment": "testing",
                "debug": True
            }, f)
            temp_file = f.name
        
        try:
            config = load_config_from_file(temp_file)
            assert config["environment"] == "testing"
            assert config["debug"] is True
        finally:
            os.unlink(temp_file)
    
    def test_validate_environment_config(self):
        """Test de validation de configuration d'environnement"""
        # Configuration de test valide
        test_config = MLAnalyticsConfig(environment=Environment.TESTING)
        assert validate_environment_config(test_config) is True
        
        # Configuration de production avec debug activé (invalide)
        prod_config = MLAnalyticsConfig(
            environment=Environment.PRODUCTION,
            debug=True
        )
        assert validate_environment_config(prod_config) is False


class TestConfigPerformance:
    """Tests de performance pour la configuration"""
    
    @pytest.mark.performance
    def test_config_loading_performance(self):
        """Test de performance du chargement de configuration"""
        import time
        
        config_manager = ConfigManager()
        
        # Test de chargement répété
        start_time = time.time()
        
        for _ in range(100):
            config = MLAnalyticsConfig(environment=Environment.TESTING)
            config_manager.validate_config(config)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Devrait être très rapide (< 1 seconde pour 100 validations)
        assert duration < 1.0
    
    @pytest.mark.performance
    def test_config_validation_performance(self):
        """Test de performance de validation"""
        import time
        
        validator = ConfigValidator()
        config = MLAnalyticsConfig(environment=Environment.TESTING)
        
        start_time = time.time()
        
        for _ in range(50):
            validator.validate_config(config)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Validation rapide
        assert duration < 0.5


class TestConfigSecurity:
    """Tests de sécurité pour la configuration"""
    
    def test_secret_masking_in_logs(self):
        """Test du masquage des secrets dans les logs"""
        config = SecurityConfig(secret_key="super_secret_key_123")
        
        # Conversion en string ne doit pas exposer le secret
        config_str = str(config)
        assert "super_secret_key_123" not in config_str
        assert "***" in config_str or "hidden" in config_str.lower()
    
    def test_config_serialization_security(self):
        """Test de sécurité de sérialisation"""
        config = MLAnalyticsConfig(environment=Environment.TESTING)
        config.security.secret_key = "sensitive_secret_123"
        
        # Sérialisation sécurisée
        safe_dict = config.to_dict(include_secrets=False)
        assert config.security.secret_key not in str(safe_dict)
    
    def test_environment_variable_security(self):
        """Test de sécurité des variables d'environnement"""
        with patch.dict(os.environ, {
            'ML_ANALYTICS_SECRET_KEY': 'env_secret_key',
            'ML_ANALYTICS_DATABASE_PASSWORD': 'db_password'
        }):
            config_manager = ConfigManager()
            config = config_manager.load_from_env()
            
            # Les secrets ne doivent pas être loggés
            logged_config = config_manager.get_safe_config_for_logging()
            assert 'env_secret_key' not in str(logged_config)
            assert 'db_password' not in str(logged_config)


class TestConfigErrorHandling:
    """Tests de gestion d'erreur pour la configuration"""
    
    def test_invalid_config_file(self):
        """Test de fichier de configuration invalide"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_file = f.name
        
        try:
            config_manager = ConfigManager()
            with pytest.raises(ConfigError):
                config_manager.load_from_file(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_missing_required_fields(self):
        """Test de champs requis manquants"""
        incomplete_config = {
            "environment": "testing"
            # Manque database config
        }
        
        with pytest.raises(ConfigError):
            MLAnalyticsConfig.from_dict(incomplete_config)
    
    def test_invalid_environment_values(self):
        """Test de valeurs d'environnement invalides"""
        with pytest.raises(ConfigError):
            MLAnalyticsConfig(environment="invalid_environment")


# Fixtures pour les tests
@pytest.fixture
def sample_config():
    """Configuration de test"""
    return MLAnalyticsConfig(
        environment=Environment.TESTING,
        debug=True,
        version="1.0.0-test"
    )


@pytest.fixture
def config_manager():
    """Gestionnaire de configuration de test"""
    return ConfigManager()


@pytest.fixture
def temp_config_file():
    """Fichier de configuration temporaire"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config_data = {
            "environment": "testing",
            "debug": True,
            "database": {
                "url": "sqlite:///test.db"
            }
        }
        yaml.dump(config_data, f)
        temp_file = f.name
    
    yield temp_file
    
    # Nettoyage
    os.unlink(temp_file)


# Tests d'intégration
@pytest.mark.integration
class TestConfigIntegration:
    """Tests d'intégration pour la configuration"""
    
    @pytest.mark.asyncio
    async def test_full_config_lifecycle(self):
        """Test du cycle de vie complet de configuration"""
        config_manager = ConfigManager()
        
        # 1. Création de configuration
        config = MLAnalyticsConfig(environment=Environment.TESTING)
        
        # 2. Validation
        assert config_manager.validate_config(config) is True
        
        # 3. Sauvegarde
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_dict = config.to_dict()
            yaml.dump(config_dict, f)
            temp_file = f.name
        
        try:
            # 4. Rechargement
            reloaded_config = config_manager.load_from_file(temp_file)
            
            # 5. Vérification
            assert reloaded_config.environment == config.environment
            assert reloaded_config.debug == config.debug
        finally:
            os.unlink(temp_file)
    
    def test_config_with_real_services(self):
        """Test de configuration avec services réels"""
        # Configuration avec services réels (mocked)
        config = MLAnalyticsConfig(
            environment=Environment.TESTING,
            database=DatabaseConfig(url="postgresql://test:test@localhost:5432/test_db"),
            cache=CacheConfig(
                type=CacheType.REDIS,
                url="redis://localhost:6379/1"
            )
        )
        
        # Test de connectivité (mocked)
        with patch('app.ml_analytics.config.test_database_connection') as mock_db:
            with patch('app.ml_analytics.config.test_cache_connection') as mock_cache:
                mock_db.return_value = True
                mock_cache.return_value = True
                
                connectivity_test = config.test_connectivity()
                assert connectivity_test['database'] is True
                assert connectivity_test['cache'] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
