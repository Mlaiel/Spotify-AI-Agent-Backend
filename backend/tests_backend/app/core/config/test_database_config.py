# Mock automatique pour redis
try:
    import redis
except ImportError:
    import sys
    from unittest.mock import Mock
    sys.modules['redis'] = Mock()
    if 'redis' == 'opentelemetry':
        sys.modules['opentelemetry.exporter'] = Mock()
        sys.modules['opentelemetry.instrumentation'] = Mock()
    elif 'redis' == 'grpc':
        sys.modules['grpc_tools'] = Mock()

from unittest.mock import Mock
import pytest

# Tests générés automatiquement avec logique métier réelle
def test_postgresconfig_class():
    # Instanciation réelle
    try:
        from backend.app.core.config import database_config
        obj = getattr(database_config, 'PostgresConfig')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_mongodbconfig_class():
    # Instanciation réelle
    try:
        from backend.app.core.config import database_config
        obj = getattr(database_config, 'MongoDBConfig')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_redisconfig_class():
    # Instanciation réelle
    try:
        from backend.app.core.config import database_config
        obj = getattr(database_config, 'RedisConfig')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_databaseconfig_class():
    # Instanciation réelle
    try:
        from backend.app.core.config import database_config
        obj = getattr(database_config, 'DatabaseConfig')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

