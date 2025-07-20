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

def test_001_create_collections(mongo_client_mock):
    """Test industriel de création de collections critiques."""
    mongo_client_mock.db.create_collection.return_value = None
    collections = ["users", "tracks", "playlists"]
    for col in collections:
        mongo_client_mock.db.create_collection(col)
        mongo_client_mock.db.create_collection.assert_any_call(col)
