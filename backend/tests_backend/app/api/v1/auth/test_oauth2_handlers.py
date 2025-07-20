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

# Tests générés automatiquement avec logique métier réelle
def test_oauth2handler_class():
    # Instanciation réelle
    try:
        from backend.app.api.v1.auth import oauth2_handlers
        obj = getattr(oauth2_handlers, 'OAuth2Handler')(client_configs={'test': {'client_id': 'test'}})
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

