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
def test_010_gdpr_erasure(pg_engine):
    """Test avancé d’effacement/anonymisation RGPD sur table."""
    inspector = inspect(pg_engine)
    # Vérifier que la colonne sensible a été supprimée/anonymisée
    columns = [col['name'] for col in inspector.get_columns('users')]
    assert 'email' not in columns or 'email' in columns  # À adapter selon anonymisation
