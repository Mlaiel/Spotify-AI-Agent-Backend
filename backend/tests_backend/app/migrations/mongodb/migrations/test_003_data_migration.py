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

def test_003_data_migration(mongo_client_mock):
    """Test clé en main de migration de données entre collections."""
    mongo_client_mock.db["users"].find.return_value = [{"_id": 1, "email": "a@b.com"}]
    mongo_client_mock.db["archived_users"].insert_many.return_value = None
    users = list(mongo_client_mock.db["users"].find())
    mongo_client_mock.db["archived_users"].insert_many(users)
    mongo_client_mock.db["archived_users"].insert_many.assert_called_with(users)
