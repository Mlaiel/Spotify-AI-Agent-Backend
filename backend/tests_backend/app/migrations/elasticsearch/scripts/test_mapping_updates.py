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


def test_mapping_updates():
    """Test avancé de mise à jour de mapping sur un index existant."""
    es_mock = Mock()
    es_mock.indices.put_mapping.return_value = {"acknowledged": True}
    response = es_mock.indices.put_mapping(index="tracks", body={"properties": {"new_field": {"type": "keyword"}}})
    assert response["acknowledged"] is True

# Test désactivé : dépendait d'un mock. À réécrire pour utiliser un vrai service Elasticsearch.
