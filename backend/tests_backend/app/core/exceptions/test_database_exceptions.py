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
def test_databaseexception_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import database_exceptions
        obj = getattr(database_exceptions, 'DatabaseException')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_transactionexception_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import database_exceptions
        obj = getattr(database_exceptions, 'TransactionException')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_integrityexception_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import database_exceptions
        obj = getattr(database_exceptions, 'IntegrityException')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_timeoutexception_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import database_exceptions
        obj = getattr(database_exceptions, 'TimeoutException')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_notfounddbexception_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import database_exceptions
        obj = getattr(database_exceptions, 'NotFoundDBException')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

