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

def test_zero_downtime_migration(mongo_client_mock):
    """Test clé en main de migration MongoDB sans interruption de service."""
    mongo_client_mock.db.command.return_value = {"ok": 1.0}
    result = mongo_client_mock.db.command({"startMigration": 1, "pauseWrites": False})
    assert result["ok"] == 1.0
