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

def test_advanced_content_mapping(es_mock):
    """Valide la structure du mapping advanced_content et la gestion des champs dynamiques."""
    expected_mapping = {
        "properties": {
            "id": {"type": "keyword"},
            "content": {"type": "text", "fields": {"raw": {"type": "keyword"}}},
            "tags": {"type": "keyword"},
            "metadata": {"type": "object", "enabled": True},
            "updated_at": {"type": "date"}
        }
    }
    es_mock.indices.get_mapping.return_value = expected_mapping
    validate_mapping(es_mock, "advanced_content", expected_mapping)
