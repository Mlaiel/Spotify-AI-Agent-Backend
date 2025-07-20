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

# Mock automatique pour elasticsearch
try:
    import elasticsearch
except ImportError:
    import sys
    from unittest.mock import Mock
    sys.modules['elasticsearch'] = Mock()
    if 'elasticsearch' == 'opentelemetry':
        sys.modules['opentelemetry.exporter'] = Mock()
        sys.modules['opentelemetry.instrumentation'] = Mock()
    elif 'elasticsearch' == 'grpc':
        sys.modules['grpc_tools'] = Mock()

import pytest

# Tests générés automatiquement avec logique métier réelle
def test_facetedsearchrequest_class():
    # Instanciation réelle
    try:
        from backend.app.api.v1.search import faceted_search
        obj = getattr(faceted_search, 'FacetedSearchRequest')(
            query="test", 
            facets=[], 
            index="test_index"
        )
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_facetedsearch_class():
    # Instanciation réelle
    try:
        from backend.app.api.v1.search import faceted_search
        from unittest.mock import Mock
        obj = getattr(faceted_search, 'FacetedSearch')(es_client=Mock())
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

