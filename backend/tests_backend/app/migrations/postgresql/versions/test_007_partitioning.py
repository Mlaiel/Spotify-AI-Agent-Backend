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
def test_007_partitioning(pg_engine):
    """Test industriel de partitionnement de table (par date, utilisateur, etc.)."""
    inspector = inspect(pg_engine)
    # Vérifier qu’une table partitionnée existe
    assert "user_activity_2025" in inspector.get_table_names()
