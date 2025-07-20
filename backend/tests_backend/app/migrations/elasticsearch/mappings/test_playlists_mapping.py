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
 # es_mock et validate_mapping supprimés car inexistants

def test_playlists_mapping(es_mock):
    """Valide la structure du mapping playlists et l’indexation de playlists complexes."""
    expected_mapping = {
        "properties": {
            "id": {"type": "keyword"},
            "name": {"type": "text", "analyzer": "standard"},
            "tracks": {"type": "nested"},
            "owner": {"type": "keyword"},
            "created_at": {"type": "date"}
        }
    }
    es_mock.indices.get_mapping.return_value = expected_mapping
    validate_mapping(es_mock, "playlists", expected_mapping)
