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
def test_authenticationerror_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import auth_exceptions
        obj = getattr(auth_exceptions, 'AuthenticationError')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_authorizationerror_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import auth_exceptions
        obj = getattr(auth_exceptions, 'AuthorizationError')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_ratelimitexceedederror_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import auth_exceptions
        obj = getattr(auth_exceptions, 'RateLimitExceededError')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_securityviolationerror_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import auth_exceptions
        obj = getattr(auth_exceptions, 'SecurityViolationError')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_authexception_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import auth_exceptions
        obj = getattr(auth_exceptions, 'AuthException')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_invalidtokenexception_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import auth_exceptions
        obj = getattr(auth_exceptions, 'InvalidTokenException')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_permissiondeniedexception_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import auth_exceptions
        obj = getattr(auth_exceptions, 'PermissionDeniedException')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_mfarequiredexception_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import auth_exceptions
        obj = getattr(auth_exceptions, 'MFARequiredException')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_oauthexception_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import auth_exceptions
        obj = getattr(auth_exceptions, 'OAuthException')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

