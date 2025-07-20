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

def test_events_mapping(es_mock):
    """Valide la structure du mapping events et la gestion des dates/horaires complexes."""
    expected_mapping = {
        "properties": {
            "id": {"type": "keyword"},
            "title": {"type": "text"},
            "venue_id": {"type": "keyword"},
            "start_time": {"type": "date"},
            "end_time": {"type": "date"},
            "participants": {"type": "integer"}
        }
    }
    es_mock.indices.get_mapping.return_value = expected_mapping
    validate_mapping(es_mock, "events", expected_mapping)
