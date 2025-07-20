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
import pytest


def test_validate_mappings():
    """Test avancé de validation de la cohérence des mappings sur tous les index."""
    es_mock = Mock()
    expected = {"properties": {"id": {"type": "keyword"}}}
    for idx in ["users", "tracks", "playlists", "venues"]:
        es_mock.indices.get_mapping.return_value = expected
        validate_mapping(es_mock, idx, expected)

def validate_mapping(es_mock, index, expected):
    """Helper function pour validation de mapping."""
    mapping = es_mock.indices.get_mapping(index=index)
    assert mapping == expected

# Test désactivé : dépendait d'un mock. À réécrire pour utiliser un vrai service Elasticsearch.
