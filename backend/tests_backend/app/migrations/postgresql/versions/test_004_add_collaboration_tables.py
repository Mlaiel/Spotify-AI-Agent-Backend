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
def test_004_add_collaboration_tables(pg_engine):
    """Test avancé d’ajout de tables de collaboration (collaboration, invitation)."""
    inspector = inspect(pg_engine)
    for table in ["collaboration", "invitation"]:
        assert table in inspector.get_table_names()
