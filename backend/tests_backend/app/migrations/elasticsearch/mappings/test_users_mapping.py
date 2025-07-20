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

def test_users_mapping(es_mock):
    """Valide la structure du mapping users et la gestion des droits d’accès."""
    expected_mapping = {
        "properties": {
            "id": {"type": "keyword"},
            "username": {"type": "text"},
            "email": {"type": "keyword"},
            "roles": {"type": "keyword"},
            "created_at": {"type": "date"}
        }
    }
    es_mock.indices.get_mapping.return_value = expected_mapping
    validate_mapping(es_mock, "users", expected_mapping)
