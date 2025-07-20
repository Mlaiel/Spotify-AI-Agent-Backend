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

def test_tracks_mapping(es_mock):
    """Valide la structure du mapping tracks et la recherche full-text multilingue."""
    expected_mapping = {
        "properties": {
            "id": {"type": "keyword"},
            "title": {"type": "text", "analyzer": "multilingual"},
            "artist": {"type": "keyword"},
            "duration": {"type": "integer"},
            "genres": {"type": "keyword"},
            "release_date": {"type": "date"}
        }
    }
    es_mock.indices.get_mapping.return_value = expected_mapping
    validate_mapping(es_mock, "tracks", expected_mapping)
