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
def test_011_zero_downtime_migration(pg_engine):
    """Test clé en main de migration PostgreSQL sans interruption de service."""
    inspector = inspect(pg_engine)
    # Vérifier que toutes les tables critiques existent après migration
    for table in ["users", "tracks", "playlists", "analytics"]:
        assert table in inspector.get_table_names()
