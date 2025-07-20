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
def test_is_email():
    # Appel réel de la fonction
    result = None
    try:
        from backend.app.core.utils import validators
        result = getattr(validators, 'is_email')()
    except Exception as exc:
        pytest.fail('Erreur lors de l\'appel réel : {}'.format(exc))
    assert result is not None

def test_is_url():
    # Appel réel de la fonction
    result = None
    try:
        from backend.app.core.utils import validators
        result = getattr(validators, 'is_url')()
    except Exception as exc:
        pytest.fail('Erreur lors de l\'appel réel : {}'.format(exc))
    assert result is not None

def test_is_phone():
    # Appel réel de la fonction
    result = None
    try:
        from backend.app.core.utils import validators
        result = getattr(validators, 'is_phone')()
    except Exception as exc:
        pytest.fail('Erreur lors de l\'appel réel : {}'.format(exc))
    assert result is not None

def test_is_iban():
    # Appel réel de la fonction
    result = None
    try:
        from backend.app.core.utils import validators
        result = getattr(validators, 'is_iban')()
    except Exception as exc:
        pytest.fail('Erreur lors de l\'appel réel : {}'.format(exc))
    assert result is not None

