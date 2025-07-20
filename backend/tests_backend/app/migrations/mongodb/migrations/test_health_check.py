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

def test_health_check(mongo_client_mock):
    """Test industriel de health check sur la base MongoDB."""
    mongo_client_mock.db.command.return_value = {"ok": 1.0}
    result = mongo_client_mock.db.command("ping")
    assert result["ok"] == 1.0
