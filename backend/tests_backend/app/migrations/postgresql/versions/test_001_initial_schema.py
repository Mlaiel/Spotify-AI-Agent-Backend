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
def test_001_initial_schema(pg_engine):
    """Test industriel de création du schéma initial (users, tracks, playlists)."""
    inspector = inspect(pg_engine)
    for table in ["users", "tracks", "playlists"]:
        assert table in inspector.get_table_names()
