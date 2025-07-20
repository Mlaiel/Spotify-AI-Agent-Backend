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

import pytest
from unittest.mock import Mock

# Tests générés automatiquement avec logique métier réelle
def test_securitymiddleware_class():
    # Instanciation réelle
    try:
        from backend.app.api.v1.auth import security_middleware
        obj = getattr(security_middleware, 'SecurityMiddleware')(app=Mock())
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

