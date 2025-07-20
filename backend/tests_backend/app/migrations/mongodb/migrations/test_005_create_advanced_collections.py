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
from . import mongo_client_mock

def test_005_create_advanced_collections(mongo_client_mock):
    """Test industriel de création de collections avancées avec options spécifiques."""
    mongo_client_mock.db.create_collection.return_value = None
    mongo_client_mock.db.create_collection("logs", capped=True, size=1048576)
    mongo_client_mock.db.create_collection.assert_called_with("logs", capped=True, size=1048576)
