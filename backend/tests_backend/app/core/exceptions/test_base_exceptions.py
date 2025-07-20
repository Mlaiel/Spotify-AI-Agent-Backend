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
def test_corebaseexception_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import base_exceptions
        obj = getattr(base_exceptions, 'CoreBaseException')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_businessexception_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import base_exceptions
        obj = getattr(base_exceptions, 'BusinessException')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_securityexception_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import base_exceptions
        obj = getattr(base_exceptions, 'SecurityException')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_notfoundexception_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import base_exceptions
        obj = getattr(base_exceptions, 'NotFoundException')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_validationexception_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import base_exceptions
        obj = getattr(base_exceptions, 'ValidationException')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_i18nerror_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import base_exceptions
        obj = getattr(base_exceptions, 'I18NError')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_configurationerror_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import base_exceptions
        obj = getattr(base_exceptions, 'ConfigurationError')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_loggingerror_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import base_exceptions
        obj = getattr(base_exceptions, 'LoggingError')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

