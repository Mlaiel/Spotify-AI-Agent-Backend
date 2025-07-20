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
def test_005_add_analytics_tables(pg_engine):
    """Test industriel d’ajout de tables analytics (analytics, audit, security)."""
    inspector = inspect(pg_engine)
    for table in ["analytics", "audit", "security"]:
        assert table in inspector.get_table_names()
