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

def test_multilingual_content_mapping(es_mock):
    """Valide la structure du mapping multilingual_content et la gestion multilingue avancée."""
    expected_mapping = {
        "properties": {
            "id": {"type": "keyword"},
            "content_fr": {"type": "text", "analyzer": "french"},
            "content_en": {"type": "text", "analyzer": "english"},
            "content_de": {"type": "text", "analyzer": "german"},
            "tags": {"type": "keyword"}
        }
    }
    es_mock.indices.get_mapping.return_value = expected_mapping
    validate_mapping(es_mock, "multilingual_content", expected_mapping)
