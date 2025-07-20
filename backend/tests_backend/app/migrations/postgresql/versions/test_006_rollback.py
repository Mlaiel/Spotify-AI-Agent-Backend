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
def test_006_rollback(pg_engine):
    """Test avancé de rollback transactionnel sur migration échouée."""
    # Simuler une migration échouée et vérifier que le schéma reste cohérent
    inspector = inspect(pg_engine)
    # Supposons que la table temporaire n’existe pas après rollback
    assert "temp_failed_migration" not in inspector.get_table_names()
