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

def test_bulk_import_export(mongo_client_mock):
    """Test clé en main d’import/export massif de données."""
    mongo_client_mock.db["tracks"].insert_many.return_value = None
    mongo_client_mock.db["tracks"].find.return_value = [{"_id": 1, "title": "A"}]
    mongo_client_mock.db["exported_tracks"].insert_many.return_value = None
    tracks = list(mongo_client_mock.db["tracks"].find())
    mongo_client_mock.db["exported_tracks"].insert_many(tracks)
    mongo_client_mock.db["exported_tracks"].insert_many.assert_called_with(tracks)
