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
from sqlalchemy import inspect, Table, MetaData

def test_create_user_and_artist_tables(db_engine):
    """Vérifie la création et l’intégrité des tables user et artist après migration."""
    metadata = MetaData()
    metadata.reflect(bind=db_engine)
    inspector = inspect(db_engine)
    assert 'user' in inspector.get_table_names(), "La table 'user' doit exister après migration."
    assert 'artist' in inspector.get_table_names(), "La table 'artist' doit exister après migration."
    # Vérification des colonnes critiques
    user_columns = [col['name'] for col in inspector.get_columns('user')]
    artist_columns = [col['name'] for col in inspector.get_columns('artist')]
    assert 'id' in user_columns and 'id' in artist_columns
    assert 'email' in user_columns
    assert 'name' in artist_columns
