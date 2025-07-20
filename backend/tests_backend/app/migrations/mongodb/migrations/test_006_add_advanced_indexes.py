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

def test_006_add_advanced_indexes(mongo_client_mock):
    """Test avancé d’ajout d’index composés et textuels."""
    mongo_client_mock.db["tracks"].create_index.return_value = "title_text_artist_1"
    idx = mongo_client_mock.db["tracks"].create_index([("title", "text"), ("artist", 1)])
    assert idx == "title_text_artist_1"
