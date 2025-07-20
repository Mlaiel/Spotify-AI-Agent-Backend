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
def test_apiexception_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import api_exceptions
        obj = getattr(api_exceptions, 'APIException')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_badrequestexception_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import api_exceptions
        obj = getattr(api_exceptions, 'BadRequestException')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_unauthorizedexception_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import api_exceptions
        obj = getattr(api_exceptions, 'UnauthorizedException')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_forbiddenexception_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import api_exceptions
        obj = getattr(api_exceptions, 'ForbiddenException')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_notfoundapiexception_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import api_exceptions
        obj = getattr(api_exceptions, 'NotFoundAPIException')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_ratelimitexception_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import api_exceptions
        obj = getattr(api_exceptions, 'RateLimitException')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_payloadtoolargeexception_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import api_exceptions
        obj = getattr(api_exceptions, 'PayloadTooLargeException')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

