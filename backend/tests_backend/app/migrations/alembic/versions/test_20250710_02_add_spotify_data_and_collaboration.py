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
from sqlalchemy import inspect

def test_add_spotify_data_and_collaboration(db_engine):
    """Vérifie l’ajout des champs Spotify et la gestion des collaborations après migration."""
    inspector = inspect(db_engine)
    # Vérification des nouvelles colonnes Spotify
    user_columns = [col['name'] for col in inspector.get_columns('user')]
    assert 'spotify_id' in user_columns, "La colonne 'spotify_id' doit exister dans 'user'."
    # Vérification de la table de collaboration
    assert 'collaboration' in inspector.get_table_names(), "La table 'collaboration' doit exister."
    collab_columns = [col['name'] for col in inspector.get_columns('collaboration')]
    assert 'user_id' in collab_columns and 'artist_id' in collab_columns
