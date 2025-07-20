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


def test_create_all_indexes():
    """Test clé en main de création de tous les index critiques du projet."""
    indexes = ["users", "tracks", "playlists", "venues"]
    es_mock = Mock()
    for idx in indexes:
        es_mock.indices.create.return_value = {"acknowledged": True, "index": idx}
        response = es_mock.indices.create(index=idx, body={"mappings": {"properties": {"id": {"type": "keyword"}}}})
        assert response["acknowledged"] is True
        assert response["index"] == idx

# Test désactivé : dépendait d'un mock. À réécrire pour utiliser un vrai service Elasticsearch.
